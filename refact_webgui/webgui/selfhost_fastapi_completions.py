import time
import json
import aiohttp
import aiofiles
import termcolor
import os
import re
import litellm
import traceback

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import Response, StreamingResponse

from refact_utils.scripts import env
from refact_utils.finetune.utils import running_models_and_loras
from refact_webgui.webgui.selfhost_model_resolve import resolve_model_context_size
from refact_webgui.webgui.selfhost_model_resolve import static_resolve_model
from refact_webgui.webgui.selfhost_queue import Ticket
from refact_webgui.webgui.selfhost_webutils import log
from refact_webgui.webgui.selfhost_queue import InferenceQueue
from refact_webgui.webgui.selfhost_model_assigner import ModelAssigner
from refact_webgui.webgui.selfhost_login import RefactSession

from refact_webgui.webgui.selfhost_sampling_params import NlpCompletion
from refact_webgui.webgui.selfhost_sampling_params import ChatContext
from refact_webgui.webgui.selfhost_sampling_params import EmbeddingsStyleOpenAI
from refact_webgui.webgui.streamers import litellm_streamer
from refact_webgui.webgui.streamers import litellm_non_streamer
from refact_webgui.webgui.streamers import refact_lsp_streamer  # TODO: deprecated
from refact_webgui.webgui.streamers import completion_streamer
from refact_webgui.webgui.streamers import embeddings_streamer
from refact_webgui.webgui.streamers import chat_completion_streamer


from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Any

__all__ = ["BaseCompletionsRouter", "CompletionsRouter"]


def red_time(base_ts):
    return termcolor.colored("%0.1fms" % (1000*(time.time() - base_ts)), "red")


class BaseCompletionsRouter(APIRouter):

    def __init__(self,
                 inference_queue: InferenceQueue,
                 id2ticket: Dict[str, Ticket],
                 model_assigner: ModelAssigner,
                 timeout: int = 30,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # deprecated APIs
        self.add_api_route("/coding_assistant_caps.json", self._coding_assistant_caps, methods=["GET"])
        self.add_api_route("/v1/login", self._login, methods=["GET"])

        # API for LSP server
        self.add_api_route("/refact-caps", self._caps, methods=["GET"])
        self.add_api_route("/v1/completions", self._completions, methods=["POST"])
        self.add_api_route("/v1/embeddings", self._embeddings_style_openai, methods=["POST"])
        self.add_api_route("/v1/chat/completions", self._chat_completions, methods=["POST"])

        self.add_api_route("/v1/models", self._models, methods=["GET"])
        self.add_api_route("/tokenizer/{model_name}", self._tokenizer, methods=["GET"])

        self._inference_queue = inference_queue
        self._id2ticket = id2ticket
        self._model_assigner = model_assigner
        self._timeout = timeout

    @property
    def _caps_version(self) -> int:
        cfg_active_lora_mtime = int(os.path.getmtime(env.CONFIG_ACTIVE_LORA)) if os.path.isfile(env.CONFIG_ACTIVE_LORA) else 0
        return max(self._model_assigner.config_inference_mtime(), cfg_active_lora_mtime)

    async def _account_from_bearer(self, authorization: str) -> str:
        raise NotImplementedError()

    @staticmethod
    def _integrations_env_setup():
        inference = {}
        if os.path.exists(env.CONFIG_INFERENCE):
            inference = json.load(open(env.CONFIG_INFERENCE, 'r'))
        integrations = {}
        if os.path.exists(env.CONFIG_INTEGRATIONS):
            integrations = json.load(open(env.CONFIG_INTEGRATIONS, 'r'))

        def _integrations_env_setup(env_var_name: str, api_key_name: str, api_enable_name: str):
            os.environ[env_var_name] = integrations.get(api_key_name, "") if inference.get(api_enable_name, False) else ""

        _integrations_env_setup("OPENAI_API_KEY", "openai_api_key", "openai_api_enable")
        _integrations_env_setup("ANTHROPIC_API_KEY", "anthropic_api_key", "anthropic_api_enable")

    def _models_available_dict_rewrite(self, models_available: List[str]) -> Dict[str, Any]:
        rewrite_dict = {}
        for model in models_available:
            d = {}
            if n_ctx := resolve_model_context_size(model, self._model_assigner):
                d["n_ctx"] = n_ctx
            if "tools" in self._model_assigner.models_db_with_passthrough.get(model, {}).get("filter_caps", []):
                d["supports_tools"] = True

            rewrite_dict[model] = d
        return rewrite_dict

    def _caps_base_data(self) -> Dict[str, Any]:
        running = running_models_and_loras(self._model_assigner)
        models_available = self._inference_queue.models_available(force_read=True)
        code_completion_default_model, _ = self._inference_queue.completion_model()
        code_chat_default_model = ""
        embeddings_default_model = ""
        for model_name in models_available:
            if "chat" in self._model_assigner.models_db.get(model_name, {}).get("filter_caps", []) or model_name in litellm.model_list:
                if not code_chat_default_model:
                    code_chat_default_model = model_name
            if "embeddings" in self._model_assigner.models_db.get(model_name, {}).get("filter_caps", []):
                if not embeddings_default_model:
                    embeddings_default_model = model_name
        data = {
            "cloud_name": "Refact Self-Hosted",
            "endpoint_template": "/v1/completions",
            "endpoint_chat_passthrough": "/v1/chat/completions",
            "endpoint_style": "openai",
            "telemetry_basic_dest": "/stats/telemetry-basic",
            "telemetry_corrected_snippets_dest": "/stats/telemetry-snippets",
            "telemetry_basic_retrieve_my_own": "/stats/rh-stats",
            "running_models": [r for r in [*running['completion'], *running['chat']]],
            "code_completion_default_model": code_completion_default_model,
            "code_chat_default_model": code_chat_default_model,
            "models_dict_patch": self._models_available_dict_rewrite(models_available),

            "default_embeddings_model": embeddings_default_model,
            "endpoint_embeddings_template": "v1/embeddings",
            "endpoint_embeddings_style": "openai",
            "size_embeddings": 768,

            "tokenizer_path_template": "/tokenizer/$MODEL",
            "tokenizer_rewrite_path": {model: model.replace("/", "--") for model in models_available},
            "caps_version": self._caps_version,
        }

        return data

    async def _coding_assistant_caps(self):
        log(f"Your refact-lsp version is deprecated, finetune is unavailable. Please update your plugin.")
        return Response(content=json.dumps(self._caps_base_data(), indent=4), media_type="application/json")

    async def _caps(self, authorization: str = Header(None), user_agent: str = Header(None)):
        if isinstance(user_agent, str):
            m = re.match(r"^refact-lsp (\d+)\.(\d+)\.(\d+)$", user_agent)
            if m:
                major, minor, patch = map(int, m.groups())
                log("user version %d.%d.%d" % (major, minor, patch))
        data = self._caps_base_data()
        running = running_models_and_loras(self._model_assigner)

        def _select_default_lora_if_exists(model_name: str, running_models: List[str]):
            model_variants = [r for r in running_models if r.split(":")[0] == model_name and r != model_name]
            return model_variants[0] if model_variants else model_name

        data["code_completion_default_model"] = _select_default_lora_if_exists(
            data["code_completion_default_model"],
            running['completion'],
        )
        data["code_chat_default_model"] = _select_default_lora_if_exists(
            data["code_chat_default_model"],
            running['chat'],
        )

        return Response(content=json.dumps(data, indent=4), media_type="application/json")

    async def _local_tokenizer(self, model_path: str) -> str:
        model_dir = Path(env.DIR_WEIGHTS) / f"models--{model_path.replace('/', '--')}"
        tokenizer_paths = list(sorted(model_dir.rglob("tokenizer.json"), key=lambda p: p.stat().st_ctime))
        if not tokenizer_paths:
            raise HTTPException(404, detail=f"tokenizer.json for {model_path} does not exist")

        data = ""
        async with aiofiles.open(tokenizer_paths[-1], mode='r') as f:
            while True:
                if not (chunk := await f.read(1024 * 1024)):
                    break
                data += chunk

        return data

    async def _passthrough_tokenizer(self, model_path: str) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                tokenizer_url = f"https://huggingface.co/{model_path}/resolve/main/tokenizer.json"
                async with session.get(tokenizer_url) as resp:
                    return await resp.text()
        except:
            raise HTTPException(404, detail=f"can't load tokenizer.json for passthrough {model_path}")

    async def _tokenizer(self, model_name: str):
        model_name = model_name.replace("--", "/")
        if model_name in self._model_assigner.models_db:
            model_path = self._model_assigner.models_db[model_name]["model_path"]
            data = await self._local_tokenizer(model_path)
        elif model_name in self._model_assigner.passthrough_mini_db:
            model_path = self._model_assigner.passthrough_mini_db[model_name]["tokenizer_path"]
            data = await self._passthrough_tokenizer(model_path)
        else:
            raise HTTPException(404, detail=f"model '{model_name}' does not exists in db")
        return Response(content=data, media_type='application/json')

    async def _login(self, authorization: str = Header(None)) -> Dict:
        account = await self._account_from_bearer(authorization)
        return {
            "account": account,
            "retcode": "OK",
            "chat-v1-style": 1,
        }

    async def _resolve_model_lora(self, model_name: str) -> Tuple[str, Optional[Dict[str, str]]]:
        running = running_models_and_loras(self._model_assigner)
        if model_name not in {r for r in [*running['completion'], *running['chat']]}:
            return model_name, None

        model_name, run_id, checkpoint_id = (*model_name.split(":"), None, None)[:3]
        if run_id is None or checkpoint_id is None:
            return model_name, None

        return model_name, {
            "run_id": run_id,
            "checkpoint_id": checkpoint_id,
        }

    async def _completions(self, post: NlpCompletion, authorization: str = Header(None)):
        account = await self._account_from_bearer(authorization)

        ticket = Ticket("comp-")
        req = post.clamp()
        caps_version = self._caps_version  # use mtime as a version, if that changes the client will know to refresh caps

        model_name, lora_config = await self._resolve_model_lora(post.model)
        model_name, err_msg = static_resolve_model(model_name, self._inference_queue)

        if err_msg:
            log("%s model resolve \"%s\" -> error \"%s\" from %s" % (ticket.id(), post.model, err_msg, account))
            return Response(status_code=400, content=json.dumps({"detail": err_msg, "caps_version": caps_version}, indent=4), media_type="application/json")

        if lora_config:
            log(f'{ticket.id()} model resolve "{post.model}" -> "{model_name}" lora {lora_config} from {account}')
        else:
            log(f'{ticket.id()} model resolve "{post.model}" -> "{model_name}" from {account}')

        req.update({
            "object": "text_completion_req",
            "account": account,
            "prompt": post.prompt,
            "model": model_name,
            "stream": post.stream,
            "echo": post.echo,
            "lora_config": lora_config,
        })
        ticket.call.update(req)
        q = self._inference_queue.model_name_to_queue(ticket, model_name)
        self._id2ticket[ticket.id()] = ticket
        await q.put(ticket)
        seen = [""] * post.n
        return StreamingResponse(
            completion_streamer(ticket, post, self._timeout, seen, req["created"], caps_version=caps_version),
            media_type=("text/event-stream" if post.stream else "application/json"),
        )

    async def _generate_embeddings(self, account: str, inputs: Union[str, List[str]], model_name: str):
        if model_name not in self._inference_queue.models_available():
            log(f"model {model_name} is not running")
            raise HTTPException(status_code=400, detail=f"model {model_name} is not running")

        tickets, reqs = [], []
        inputs = inputs if isinstance(inputs, list) else [inputs]
        for inp in inputs:
            ticket = Ticket("embed-")
            req = {
                "inputs": inp,
            }
            req.update({
                "id": ticket.id(),
                "account": account,
                "object": "embeddings_req",
                "model": model_name,
                "stream": True,
                "created": time.time()
            })
            ticket.call.update(req)
            q = self._inference_queue.model_name_to_queue(ticket, model_name, no_checks=True)
            self._id2ticket[ticket.id()] = ticket
            await q.put(ticket)
            tickets.append(ticket)
            reqs.append(req)

        for idx, (ticket, req) in enumerate(zip(tickets, reqs)):
            async for resp in embeddings_streamer(ticket, 60, req["created"]):
                resp = json.loads(resp)
                embedding = []
                try:
                    embedding = resp[0]
                except IndexError:
                    pass
                yield {"embedding": embedding, "index": idx}

    async def _embeddings_style_openai(self, post: EmbeddingsStyleOpenAI, authorization: str = Header(None)):
        account = await self._account_from_bearer(authorization)
        data = [
            {
                "embedding": res["embedding"],
                "index": res["index"],
                "object": "embedding",
            }
            async for res in self._generate_embeddings(account, post.input, post.model)
        ]
        data.sort(key=lambda x: x["index"])

        return {
            "data": data,
            "model": post.model,
            "object": "list",
            "usage": {"prompt_tokens": -1, "total_tokens": -1}
        }

    async def _models(self, authorization: str = Header(None)):
        await self._account_from_bearer(authorization)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://127.0.0.1:8001/v1/caps") as resp:
                    lsp_server_caps = await resp.json()
        except aiohttp.ClientConnectorError as e:
            err_msg = f"LSP server is not ready yet: {e}"
            log(err_msg)
            raise HTTPException(status_code=401, detail=err_msg)
        completion_models = set()
        for model, caps in lsp_server_caps["code_completion_models"].items():
            completion_models.update({model, *caps["similar_models"]})
        chat_models = set()
        for model, caps in lsp_server_caps["code_chat_models"].items():
            chat_models.update({model, *caps["similar_models"]})
        data = [
            {
                "id": model, "root": model, "object": "model",
                "created": 0, "owned_by": "", "permission": [], "parent": None,
                "completion": model in completion_models, "chat": model in chat_models,
            }
            for model in lsp_server_caps["running_models"]
        ]
        return {
            "object": "list",
            "data": data,
        }

    async def _chat_completions_streamer(
            self,
            messages: List,
            post: ChatContext,
            account: str,
            model_dict: Dict,
            model_name: str,
            lora_config: Dict,
            caps_version: int,
    ):
        ticket = Ticket("chat-comp-")
        req = post.clamp()

        if "chat" not in model_dict.get("filter_caps", []):
            err_message = f'chat completion from "{account}" failed: "{post.model}" does not support chat'
            log(err_message)
            raise HTTPException(status_code=401, detail=err_message)

        if "model_path" not in model_dict:
            err_message = f'chat completion from "{account}" failed: "{post.model}" has no model_path'
            log(err_message)
            raise HTTPException(status_code=401, detail=err_message)

        from transformers import AutoTokenizer

        # preload map tokenizers
        tokenizer = AutoTokenizer.from_pretrained(
            model_dict["model_path"], cache_dir=env.DIR_WEIGHTS,
            local_files_only=True, trust_remote_code=True,
        )
        # NOTE: idk how to use tool_choice with transformers models
        prompt = tokenizer.apply_chat_template(messages, tools=post.tools, tokenize=False)

        req.update({
            "object": "text_completion_req",
            "account": account,
            "prompt": prompt,
            "model": model_name,
            "stream": post.stream,
            "echo": False,
            "lora_config": lora_config,
        })
        ticket.call.update(req)
        q = self._inference_queue.model_name_to_queue(ticket, model_name)
        self._id2ticket[ticket.id()] = ticket
        await q.put(ticket)

        return chat_completion_streamer(ticket, post, self._timeout, req["created"], caps_version=caps_version)

    async def _chat_completions(self, post: ChatContext, authorization: str = Header(None)):
        account = await self._account_from_bearer(authorization)

        caps_version = self._caps_version
        model_name, lora_config = await self._resolve_model_lora(post.model)
        model_name, err_msg = static_resolve_model(model_name, self._inference_queue)

        if err_msg:
            log(f'chat completions model resolve "{post.model}" -> error "{err_msg}" from {account}')
            return Response(status_code=400,
                            content=json.dumps({"detail": err_msg, "caps_version": caps_version}, indent=4),
                            media_type="application/json")

        if lora_config:
            log(f'chat completions model resolve "{post.model}" -> "{model_name}" lora {lora_config} from {account}')
        else:
            log(f'chat completions model resolve "{post.model}" -> "{model_name}" from {account}')

        messages = []
        for m in (i.dict() for i in post.messages):
            # drop tool_calls if empty, otherwise litellm tokenizing won't work
            if "tool_calls" in m and not m["tool_calls"]:
                del m["tool_calls"]
            messages.append(m)

        model_dict = self._model_assigner.models_db_with_passthrough.get(model_name, {})

        if model_dict.get('backend') == 'litellm':
            log(f"chat/completions: model resolve {post.model} -> {model_dict['resolve_as']}")
            response_streamer = litellm_streamer(model_dict, messages, post) if post.stream else litellm_non_streamer(model_dict, messages, post)
        else:
            # NOTE: always stream=True
            # response_streamer = refact_lsp_streamer(messages, post)
            response_streamer = await self._chat_completions_streamer(
                messages, post, account, model_dict, model_name, lora_config, caps_version)
            # return await completions_streaming()

        return StreamingResponse(response_streamer, media_type="text/event-stream")

    # async def _chat_completions(self, post: ChatContext, authorization: str = Header(None)):
    #     def compose_usage_dict(model_dict, prompt_tokens_n, generated_tokens_n) -> Dict[str, Any]:
    #         usage_dict = dict()
    #         usage_dict["pp1000t_prompt"] = model_dict.get("pp1000t_prompt", 0)
    #         usage_dict["pp1000t_generated"] = model_dict.get("pp1000t_generated", 0)
    #         usage_dict["metering_prompt_tokens_n"] = prompt_tokens_n
    #         usage_dict["metering_generated_tokens_n"] = generated_tokens_n
    #         return usage_dict
    #
    #     _account = await self._account_from_bearer(authorization)
    #     messages = []
    #     for m in (i.dict() for i in post.messages):
    #         # drop tool_calls if empty, otherwise litellm tokenizing won't work
    #         if "tool_calls" in m and not m["tool_calls"]:
    #             del m["tool_calls"]
    #         messages.append(m)
    #
    #     prefix, postfix = "data: ", "\n\n"
    #     model_dict = self._model_assigner.models_db_with_passthrough.get(post.model, {})
    #
    #     async def litellm_streamer():
    #         generated_tokens_n = 0
    #         try:
    #             self._integrations_env_setup()
    #             response = await litellm.acompletion(
    #                 model=model_name, messages=messages, stream=True,
    #                 temperature=post.temperature, top_p=post.top_p,
    #                 max_tokens=min(model_dict.get('T_out', post.max_tokens), post.max_tokens),
    #                 tools=post.tools,
    #                 tool_choice=post.tool_choice,
    #                 stop=post.stop,
    #                 n=post.n,
    #             )
    #             finish_reason = None
    #             async for model_response in response:
    #                 try:
    #                     data = model_response.dict()
    #                     choice0 = data["choices"][0]
    #                     finish_reason = choice0["finish_reason"]
    #                     if delta := choice0.get("delta"):
    #                         if text := delta.get("content"):
    #                             generated_tokens_n += litellm.token_counter(model_name, text=text)
    #
    #                 except json.JSONDecodeError:
    #                     data = {"choices": [{"finish_reason": finish_reason}]}
    #                 yield prefix + json.dumps(data) + postfix
    #
    #             final_msg = {"choices": []}
    #             usage_dict = compose_usage_dict(model_dict, prompt_tokens_n, generated_tokens_n)
    #             final_msg.update(usage_dict)
    #             yield prefix + json.dumps(final_msg) + postfix
    #
    #             # NOTE: DONE needed by refact-lsp server
    #             yield prefix + "[DONE]" + postfix
    #         except BaseException as e:
    #             err_msg = f"litellm error (1): {e}"
    #             log(err_msg)
    #             yield prefix + json.dumps({"error": err_msg}) + postfix
    #
    #     async def litellm_non_streamer():
    #         generated_tokens_n = 0
    #         try:
    #             self._integrations_env_setup()
    #             model_response = await litellm.acompletion(
    #                 model=model_name, messages=messages, stream=False,
    #                 temperature=post.temperature, top_p=post.top_p,
    #                 max_tokens=min(model_dict.get('T_out', post.max_tokens), post.max_tokens),
    #                 tools=post.tools,
    #                 tool_choice=post.tool_choice,
    #                 stop=post.stop,
    #                 n=post.n,
    #             )
    #             finish_reason = None
    #             try:
    #                 data = model_response.dict()
    #                 for choice in data.get("choices", []):
    #                     if text := choice.get("message", {}).get("content"):
    #                         generated_tokens_n += litellm.token_counter(model_name, text=text)
    #                     finish_reason = choice.get("finish_reason")
    #                 usage_dict = compose_usage_dict(model_dict, prompt_tokens_n, generated_tokens_n)
    #                 data.update(usage_dict)
    #             except json.JSONDecodeError:
    #                 data = {"choices": [{"finish_reason": finish_reason}]}
    #             yield json.dumps(data)
    #         except BaseException as e:
    #             err_msg = f"litellm error (2): {e}"
    #             log(err_msg)
    #             yield json.dumps({"error": err_msg})
    #
    #     async def chat_completion_streamer():
    #         post_url = "http://127.0.0.1:8001/v1/chat"
    #         payload = {
    #             "messages": messages,
    #             "stream": True,
    #             "model": post.model,
    #             "parameters": {
    #                 "temperature": post.temperature,
    #                 "max_new_tokens": post.max_tokens,
    #             }
    #         }
    #         async with aiohttp.ClientSession() as session:
    #             try:
    #                 async with session.post(post_url, json=payload) as response:
    #                     finish_reason = None
    #                     async for data, _ in response.content.iter_chunks():
    #                         try:
    #                             data = data.decode("utf-8")
    #                             data = json.loads(data[len(prefix):-len(postfix)])
    #                             finish_reason = data["choices"][0]["finish_reason"]
    #                             data["choices"][0]["finish_reason"] = None
    #                         except json.JSONDecodeError:
    #                             data = {"choices": [{"finish_reason": finish_reason}]}
    #                         yield prefix + json.dumps(data) + postfix
    #             except aiohttp.ClientConnectorError as e:
    #                 err_msg = f"LSP server is not ready yet: {e}"
    #                 log(err_msg)
    #                 yield prefix + json.dumps({"error": err_msg}) + postfix
    #
    #     async def completions_streaming():
    #         from transformers import AutoTokenizer
    #
    #         if "chat" not in model_dict.get("filter_caps", []):
    #             err_message = f'chat completion from "{_account}" failed: "{post.model}" does not support chat'
    #             log(err_message)
    #             raise HTTPException(status_code=401, detail=err_message)
    #
    #         if "model_path" not in model_dict:
    #             err_message = f'chat completion from "{_account}" failed: "{post.model}" has no model_path'
    #             log(err_message)
    #             raise HTTPException(status_code=401, detail=err_message)
    #
    #         # preload map tokenizers
    #         tokenizer = AutoTokenizer.from_pretrained(
    #             model_dict["model_path"], cache_dir=env.DIR_WEIGHTS,
    #             local_files_only=True, trust_remote_code=True,
    #         )
    #         # NOTE: idk how to use tool_choice with transformers models
    #         prompt = tokenizer.apply_chat_template(messages, tools=post.tools, tokenize=False)
    #         completions_post = NlpCompletion(
    #             max_tokens=post.max_tokens,
    #             temperature=post.temperature,
    #             top_p=post.top_p,
    #             top_n=post.top_n,
    #             stop=post.stop,
    #
    #             model=post.model,
    #             prompt=prompt,
    #             n=post.n,
    #             stream=post.stream,
    #
    #             echo=False,
    #             mask_emails=False,
    #         )
    #
    #         return await self._completions(completions_post, authorization)
    #
    #     if model_dict.get('backend') == 'litellm' and (model_name := model_dict.get('resolve_as', post.model)) in litellm.model_list:
    #         log(f"chat/completions: model resolve {post.model} -> {model_name}")
    #         prompt_tokens_n = litellm.token_counter(model_name, messages=messages)
    #         if post.tools:
    #             prompt_tokens_n += litellm.token_counter(model_name, text=json.dumps(post.tools))
    #         response_streamer = litellm_streamer() if post.stream else litellm_non_streamer()
    #     else:
    #         # response_streamer = chat_completion_streamer()
    #         return await completions_streaming()
    #
    #     return StreamingResponse(response_streamer, media_type="text/event-stream")


class CompletionsRouter(BaseCompletionsRouter):
    def __init__(self, session: RefactSession, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session = session

    async def _account_from_bearer(self, authorization: str) -> str:
        try:
            return self._session.header_authenticate(authorization)
        except Exception as e:
            traceback_str = traceback.format_exc()
            log(traceback_str)
            raise HTTPException(status_code=401, detail=str(e))

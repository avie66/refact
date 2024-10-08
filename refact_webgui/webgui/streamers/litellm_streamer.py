import json
import litellm

from refact_webgui.webgui.selfhost_webutils import log
from refact_webgui.webgui.selfhost_fastapi_completions import ChatContext

from typing import Dict, Any, List


prefix, postfix = "data: ", "\n\n"


def compose_usage_dict(model_dict: Dict, prompt_tokens_n: int, generated_tokens_n: int) -> Dict[str, Any]:
    usage_dict = dict()
    usage_dict["pp1000t_prompt"] = model_dict.get("pp1000t_prompt", 0)
    usage_dict["pp1000t_generated"] = model_dict.get("pp1000t_generated", 0)
    usage_dict["metering_prompt_tokens_n"] = prompt_tokens_n
    usage_dict["metering_generated_tokens_n"] = generated_tokens_n
    return usage_dict


async def litellm_streamer(model_dict: Dict, messages: List, post: ChatContext):
    generated_tokens_n = 0
    model_name = model_dict["resolve_as"]
    prompt_tokens_n = litellm.token_counter(model_name, messages=messages)
    if post.tools:
        prompt_tokens_n += litellm.token_counter(model_name, text=json.dumps(post.tools))
    try:
        # self._integrations_env_setup()
        response = await litellm.acompletion(
            model=model_name, messages=messages, stream=True,
            temperature=post.temperature, top_p=post.top_p,
            max_tokens=min(model_dict.get('T_out', post.max_tokens), post.max_tokens),
            tools=post.tools,
            tool_choice=post.tool_choice,
            stop=post.stop,
            n=post.n,
        )
        finish_reason = None
        async for model_response in response:
            try:
                data = model_response.dict()
                choice0 = data["choices"][0]
                finish_reason = choice0["finish_reason"]
                if delta := choice0.get("delta"):
                    if text := delta.get("content"):
                        generated_tokens_n += litellm.token_counter(model_name, text=text)

            except json.JSONDecodeError:
                data = {"choices": [{"finish_reason": finish_reason}]}
            yield prefix + json.dumps(data) + postfix

        final_msg = {"choices": []}
        usage_dict = compose_usage_dict(model_dict, prompt_tokens_n, generated_tokens_n)
        final_msg.update(usage_dict)
        yield prefix + json.dumps(final_msg) + postfix

        # NOTE: DONE needed by refact-lsp server
        yield prefix + "[DONE]" + postfix
    except BaseException as e:
        err_msg = f"litellm error (1): {e}"
        log(err_msg)
        yield prefix + json.dumps({"error": err_msg}) + postfix


async def litellm_non_streamer(model_dict: Dict, messages: List, post: ChatContext):
    generated_tokens_n = 0
    model_name = model_dict["resolve_as"]
    prompt_tokens_n = litellm.token_counter(model_name, messages=messages)
    if post.tools:
        prompt_tokens_n += litellm.token_counter(model_name, text=json.dumps(post.tools))
    try:
        # self._integrations_env_setup()
        model_response = await litellm.acompletion(
            model=model_name, messages=messages, stream=False,
            temperature=post.temperature, top_p=post.top_p,
            max_tokens=min(model_dict.get('T_out', post.max_tokens), post.max_tokens),
            tools=post.tools,
            tool_choice=post.tool_choice,
            stop=post.stop,
            n=post.n,
        )
        finish_reason = None
        try:
            data = model_response.dict()
            for choice in data.get("choices", []):
                if text := choice.get("message", {}).get("content"):
                    generated_tokens_n += litellm.token_counter(model_name, text=text)
                finish_reason = choice.get("finish_reason")
            usage_dict = compose_usage_dict(model_dict, prompt_tokens_n, generated_tokens_n)
            data.update(usage_dict)
        except json.JSONDecodeError:
            data = {"choices": [{"finish_reason": finish_reason}]}
        yield json.dumps(data)
    except BaseException as e:
        err_msg = f"litellm error (2): {e}"
        log(err_msg)
        yield json.dumps({"error": err_msg})
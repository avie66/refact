import json
import aiohttp

from refact_webgui.webgui.selfhost_webutils import log
from refact_webgui.webgui.selfhost_sampling_params import ChatContext

from typing import List


prefix, postfix = "data: ", "\n\n"


async def refact_lsp_streamer(messages: List, post: ChatContext):
    post_url = "http://127.0.0.1:8001/v1/chat"
    payload = {
        "messages": messages,
        "stream": True,
        "model": post.model,
        "parameters": {
            "temperature": post.temperature,
            "max_new_tokens": post.max_tokens,
        }
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(post_url, json=payload) as response:
                finish_reason = None
                async for data, _ in response.content.iter_chunks():
                    try:
                        data = data.decode("utf-8")
                        data = json.loads(data[len(prefix):-len(postfix)])
                        finish_reason = data["choices"][0]["finish_reason"]
                        data["choices"][0]["finish_reason"] = None
                    except json.JSONDecodeError:
                        data = {"choices": [{"finish_reason": finish_reason}]}
                    yield prefix + json.dumps(data) + postfix
        except aiohttp.ClientConnectorError as e:
            err_msg = f"LSP server is not ready yet: {e}"
            log(err_msg)
            yield prefix + json.dumps({"error": err_msg}) + postfix

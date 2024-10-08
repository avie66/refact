import re
import time
import copy
import json
import asyncio
import termcolor

from refact_webgui.webgui.selfhost_queue import Ticket
from refact_webgui.webgui.selfhost_webutils import log
from refact_webgui.webgui.selfhost_sampling_params import ChatContext


def red_time(base_ts):
    return termcolor.colored("%0.1fms" % (1000*(time.time() - base_ts)), "red")


async def chat_completions_streamer(ticket: Ticket, post: ChatContext, timeout, created_ts, caps_version: int):
    seen = [""] * post.n
    try:
        packets_cnt = 0
        while 1:
            try:
                msg = await asyncio.wait_for(ticket.streaming_queue.get(), timeout)
            except asyncio.TimeoutError:
                log("TIMEOUT %s" % ticket.id())
                msg = {"status": "error", "human_readable_message": "timeout"}
            not_seen_resp = copy.deepcopy(msg)
            not_seen_resp["caps_version"] = caps_version
            is_final_msg = msg.get("status", "") != "in_progress"
            if "choices" in not_seen_resp:
                for i in range(post.n):
                    newtext = not_seen_resp["choices"][i]["text"]
                    if newtext.startswith(seen[i]):
                        delta = newtext[len(seen[i]):]
                        if " " not in delta and not is_final_msg:
                            not_seen_resp["choices"][i]["text"] = ""
                            continue
                        not_seen_resp["choices"][i]["text"] = delta
                        if post.stream:
                            seen[i] = newtext[:len(seen[i])] + delta
                    else:
                        log("ooops seen doesn't work, might be infserver's fault")
            if not post.stream:
                if not is_final_msg:
                    continue
                yield json.dumps(not_seen_resp)
                break
            yield "data: " + json.dumps(not_seen_resp) + "\n\n"
            packets_cnt += 1
            if is_final_msg:
                break
        if post.stream:
            yield "data: [DONE]" + "\n\n"
        log(red_time(created_ts) + " /finished %s, streamed %i packets" % (ticket.id(), packets_cnt))
        ticket.done()
    finally:
        if ticket.id() is not None:
            log("   ***  CANCEL  ***  cancelling %s " % ticket.id() + red_time(created_ts))
        ticket.cancelled = True

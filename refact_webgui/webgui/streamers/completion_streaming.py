import re
import time
import copy
import json
import asyncio
import termcolor

from refact_webgui.webgui.selfhost_queue import Ticket
from refact_webgui.webgui.selfhost_webutils import log
from refact_webgui.webgui.selfhost_sampling_params import NlpCompletion


def red_time(base_ts):
    return termcolor.colored("%0.1fms" % (1000*(time.time() - base_ts)), "red")


def _mask_emails(text: str, mask: str = "john@example.com") -> str:
    masked_text = text
    for m in re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text):
        masked_text = masked_text.replace(m, mask)
    return masked_text


async def completion_streamer(ticket: Ticket, post: NlpCompletion, timeout, seen, created_ts, caps_version: int):
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
                        if post.mask_emails:
                            if not is_final_msg:
                                delta = " ".join(delta.split(" ")[:-1])
                            not_seen_resp["choices"][i]["text"] = _mask_emails(delta)
                        else:
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
        # fastapi_stats.stats_accum[kt] += msg.get("generated_tokens_n", 0)
        # fastapi_stats.stats_accum[kcomp] += 1
        # fastapi_stats.stats_lists_accum["stat_latency_" + post.model].append(time.time() - created_ts)
    finally:
        if ticket.id() is not None:
            log("   ***  CANCEL  ***  cancelling %s " % ticket.id() + red_time(created_ts))
            # fastapi_stats.stats_accum["stat_api_cancelled"] += 1
            # fastapi_stats.stats_accum["stat_m_" + post.model + "_cancelled"] += 1
        ticket.cancelled = True

"""Anthropic Messages tool turns and streaming over the Chat tool path."""
from copy import deepcopy
import json
import sys

import openai

TOOLS = [{
    "name": "get_weather",
    "description": "Return the current weather for one city.",
    "input_schema": {"type": "object",
                     "properties": {"city": {"type": "string"}},
                     "required": ["city"]},
}]


def messages_tool_result(client, body):
    """Map a buffered Messages response, including tool_use blocks."""
    response = client.post("/messages", body=body, cast_to=object)
    blocks = response["content"]
    usage, timings = response["usage"], response["timings"]
    return {
        "text": "".join(b["text"] for b in blocks if b["type"] == "text"),
        "reasoning": "".join(b["thinking"] for b in blocks if b["type"] == "thinking"),
        "blocks": blocks,
        "tools": [b for b in blocks if b["type"] == "tool_use"],
        "finish": response["stop_reason"],
        "usage": {"prompt_tokens": usage["input_tokens"],
                  "cached_tokens": usage["cache_read_input_tokens"],
                  "completion_tokens": usage["output_tokens"],
                  "gufo": {"prefill_tokens": timings["prompt_n"]}},
    }


def messages_stream(client, body):
    """Collect streamed Messages events and rebuild their content blocks."""
    events = list(client.post("/messages", body={**body, "stream": True},
                              cast_to=object, stream=True,
                              stream_cls=openai.Stream[object]))
    types = [event["type"] for event in events]
    assert types[0] == "message_start" and types[-1] == "message_stop", types
    assert types[-2] == "message_delta", types
    blocks, open_index = [], None
    for event in events[1:-2]:
        kind = event["type"]
        if kind == "content_block_start":
            assert open_index is None and event["index"] == len(blocks), event
            open_index = event["index"]
            blocks.append(dict(event["content_block"]))
            if blocks[-1]["type"] == "tool_use":
                blocks[-1]["partial_json"] = ""
        elif kind == "content_block_delta":
            assert event["index"] == open_index, event
            delta, block = event["delta"], blocks[open_index]
            field = {"thinking_delta": "thinking", "text_delta": "text",
                     "input_json_delta": "partial_json"}[delta["type"]]
            block[field] += delta[field]
        else:
            assert kind == "content_block_stop" and event["index"] == open_index, event
            open_index = None
    assert open_index is None, events
    for block in blocks:
        if block["type"] == "tool_use":
            block["input"] = json.loads(block.pop("partial_json"))
    final = events[-2]
    return {"blocks": blocks, "finish": final["delta"]["stop_reason"],
            "usage": final["usage"], "types": types}


def check_messages_tools(client, model, checks, chat_result):
    """A Messages agent turn: a tool_use block, its replay with a tool_result,
    and the same turns streamed. Uncached Chat controls render the same
    prompt, and an unchanged replay reuses the previous prompt."""
    def record(label, result):
        checks["messages_tools_" + label] = result
        print(f"CHECK messages_tools_{label}", file=sys.stderr, flush=True)
        return result

    def work(result):
        usage = result["usage"]
        total, reused, prefilled = (usage["prompt_tokens"], usage["cached_tokens"],
                                    usage["gufo"]["prefill_tokens"])
        assert reused + prefilled == total, usage
        return total, reused, prefilled

    system = ("messages_tools\nUse get_weather for every weather question. "
              "After a tool result, answer in one short sentence.")
    request = dict(model=model, system=system, temperature=0, seed=31, max_tokens=256,
                   thinking={"type": "disabled"}, tools=deepcopy(TOOLS),
                   tool_choice={"type": "auto"})
    user = {"role": "user", "content": "What is the weather in Lisbon right now?"}

    first = record("call", messages_tool_result(
        client, {**deepcopy(request), "messages": [user]}))
    assert first["finish"] == "tool_use" and len(first["tools"]) == 1, first
    call = first["tools"][0]
    assert call["name"] == "get_weather" and call["id"], call
    assert isinstance(call["input"].get("city"), str) and \
        "lisbon" in call["input"]["city"].lower(), call
    previous_total, _, _ = work(first)

    replay = [user, {"role": "assistant", "content": deepcopy(first["blocks"])},
              {"role": "user", "content": [{"type": "tool_result",
                                            "tool_use_id": call["id"],
                                            "content": "Sunny, 22 C."}]}]
    answer = record("answer", messages_tool_result(
        client, {**deepcopy(request), "messages": deepcopy(replay)}))
    total, reused, _ = work(answer)
    assert answer["finish"] == "end_turn" and not answer["tools"], answer
    assert "22" in answer["text"], answer
    # The replayed tool_use block reproduces the generated call, so reuse
    # reaches at least the previous prompt's stable boundary.
    assert total > previous_total and reused >= previous_total - 16, \
        (previous_total, answer["usage"])

    # Uncached Chat Completions control of the same replayed conversation.
    arguments = json.dumps(call["input"])
    chat = [{"role": "system", "content": system}, user,
            {"role": "assistant", "content": first["text"] or None,
             "tool_calls": [{"id": call["id"], "type": "function",
                             "function": {"name": call["name"], "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": call["id"], "content": "Sunny, 22 C."}]
    cold = record("answer_cold_chat", chat_result(client, dict(
        model=model, messages=chat, temperature=0, seed=31, max_completion_tokens=256,
        reasoning_effort="none", tool_choice="auto",
        tools=[{"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["input_schema"]}} for t in TOOLS],
        extra_body={"cache_prompt": False})))
    assert work(cold) == (total, 0, total), (answer, cold)
    assert cold["text"].strip() == answer["text"].strip(), (answer, cold)

    # Streamed turns rebuild the same blocks as the buffered responses.
    streamed_call = record("call_streamed", messages_stream(
        client, {**deepcopy(request), "messages": [user]}))
    assert streamed_call["finish"] == "tool_use", streamed_call
    assert [(b["type"], b.get("name"), b.get("input")) for b in streamed_call["blocks"]] == \
        [(b["type"], b.get("name"), b.get("input")) for b in first["blocks"]], \
        (streamed_call, first)
    streamed_answer = record("answer_streamed", messages_stream(
        client, {**deepcopy(request), "messages": deepcopy(replay)}))
    assert streamed_answer["finish"] == "end_turn", streamed_answer
    assert "".join(b.get("text", "") for b in streamed_answer["blocks"]) == \
        answer["text"], (streamed_answer, answer)
    assert streamed_answer["usage"]["input_tokens"] == total, streamed_answer

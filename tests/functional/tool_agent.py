"""Short, bounded agent histories with ordinary (non-strict) nested tool schemas."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import sys

from tool_reasoning import response_result


def agent_tools():
    # Deliberately omit additionalProperties, as real agent clients do.
    return [{"type": "function", "function": {
        "name": "edit", "description": "Replace exact text in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "edits": {"type": "array", "items": {"type": "object", "properties": {
                "oldText": {"type": "string"}, "newText": {"type": "string"}},
                "required": ["oldText", "newText"]}}},
            "required": ["path", "edits"]}}},
        {"type": "function", "function": {
            "name": "read", "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
            "required": ["path"]}}}]


def check_tool_agent_json(client, model, checks, chat_result):
    """An open, non-strict tool must not weaken the strict final JSON schema."""
    expected = {"status": "done", "count": 1}
    schema = {"type": "object", "properties": {
        "status": {"type": "string", "const": "done"},
        "count": {"type": "integer", "const": 1}},
        "required": ["status", "count"], "additionalProperties": False}
    prompt = 'No file operation is needed. Do not call a tool. Return {"status":"done","count":1}.'
    tools = agent_tools()
    chat = dict(model=model, tools=tools, tool_choice="auto", temperature=0,
                seed=41, max_completion_tokens=64,
                reasoning_effort="none", messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "result", "strict": True, "schema": schema}})
    responses = dict(model=model, input=prompt, tool_choice="auto",
                     tools=[{"type": "function", **t["function"], "strict": False} for t in tools],
                     temperature=0, reasoning={"effort": "none"},
                     max_output_tokens=64, store=False, extra_body={"seed": 41},
                     text={"format": {"type": "json_schema", "name": "result",
                                      "strict": True, "schema": schema}})
    for endpoint, request in (("chat", chat), ("responses", responses)):
        result = (chat_result(client, request, True) if endpoint == "chat"
                  else response_result(client, request, True))
        checks[f"agent_json_{endpoint}"] = result
        assert not result["tools"] and result["finish"] == "stop", result
        assert json.loads(result["text"]) == expected, result


def check_tool_agent(client, model, checks, chat_result, vision, image_content):
    def record(name, result):
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        return result

    def signature(result):
        return (result["text"], result["reasoning"],
                [(t["function"]["name"], json.loads(t["function"]["arguments"]))
                 for t in result["tools"]], result["finish"])

    def check_call(result, name, arguments):
        assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, result
        function = result["tools"][0]["function"]
        assert function["name"] == name and json.loads(function["arguments"]) == arguments, result
        assert not result["text"].strip(), result

    tools = agent_tools()
    common = dict(model=model, tools=tools, tool_choice="auto", parallel_tool_calls=False,
                  temperature=0, seed=41, max_completion_tokens=192,
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    # Catch the format/prompt regression independently of model behavior.
    # Closing one nested object adds a few schema tokens, not instructions
    # requiring every function to abandon its native format.
    control = {**common, "messages": [{"role": "user", "content": "Reply OK without using tools."}],
               "max_completion_tokens": 8,
               "extra_body": {**common["extra_body"], "cache_prompt": False}}
    closed = deepcopy(tools)
    closed[0]["function"]["parameters"]["properties"]["edits"]["items"]["additionalProperties"] = False
    original = record("agent_open_schema", chat_result(client, control))
    explicit = record("agent_closed_schema", chat_result(client, {**control, "tools": closed}))
    delta = explicit["usage"]["prompt_tokens"] - original["usage"]["prompt_tokens"]
    assert 0 < delta < 16, ("nested schema changed the tool protocol", delta)

    # A real model completes each next action, while tools operate only on
    # in-memory fixture data. The bounded history cannot run arbitrary commands.
    old, new = "    return a - b", "    return a + b"
    filename = "calc.py"
    arguments = {"path": filename, "edits": [{"oldText": old, "newText": new}]}
    prompt = ("Fix calc.py by calling edit exactly once, replacing " + repr(old) +
              " with " + repr(new) + ". Do not reread it. Its complete contents are:\n"
              "def add(a, b):\n" + old + "\nDo not write a visible explanation.")
    content = ([image_content("red"), {"type": "text", "text": prompt}]
               if vision else prompt)
    messages = [{"role": "user", "content": content}]
    first_request = {**common, "messages": deepcopy(messages)}
    first = record("agent_edit", chat_result(client, first_request, True))
    check_call(first, "edit", arguments)
    messages += [
        {"role": "assistant", "content": first["text"] or None, "tool_calls": first["tools"]},
        {"role": "tool", "tool_call_id": first["tools"][0]["id"], "content": "Replaced one block."},
        {"role": "user", "content": "Now call read exactly once on calc.py. No explanation."}]
    read = record("agent_read_after_edit", chat_result(client, {**common, "messages": messages}))
    check_call(read, "read", {"path": filename})
    assert read["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, read
    messages += [
        {"role": "assistant", "content": read["text"] or None, "tool_calls": read["tools"]},
        {"role": "tool", "tool_call_id": read["tools"][0]["id"],
         "content": "def add(a, b):\n" + new + "\n"},
        {"role": "user", "content": "Verified. The task is finished. Reply DONE, without calling tools."}]
    # Use the API stop contract here, including any speculative lookahead
    # after DONE. The autonomous suite below separately checks natural exit.
    done = record("agent_finish_stop", chat_result(
        client, {**common, "messages": messages, "stop": "DONE"}, True))
    assert done["finish"] == "stop" and not done["tools"] and not done["text"].strip(), done
    assert done["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, done

    # JSON responses, stops and length limits must still obey their contracts
    # when a neighboring non-strict tool has open nested objects.
    limited = record("agent_limit", chat_result(client, {
        **first_request, "tool_choice": "required", "max_completion_tokens": 2}, True))
    assert limited["finish"] == "length" and not limited["tools"] and not limited["text"].strip(), limited
    stopped = record("agent_stop", chat_result(client, {**first_request, "stop": "calc.py"}, True))
    assert stopped["finish"] == "stop" and not stopped["tools"], stopped
    retry = record("agent_retry", chat_result(client, first_request, True))
    assert signature(retry) == signature(first), (retry, first)
    assert retry["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, retry

    responses = dict(model=model, input=prompt,
                     tools=[{"type": "function", **t["function"], "strict": False} for t in tools],
                     tool_choice="auto", parallel_tool_calls=False, store=False,
                     reasoning={"effort": "none"}, temperature=0, max_output_tokens=192,
                     extra_body={"seed": 41})
    for streaming in (False, True):
        result = record(f"agent_responses_{streaming}", response_result(client, responses, streaming))
        check_call(result, "edit", arguments)

    def peer(index):
        path = f"peer{index}.py"
        body = {**first_request, "temperature": .7, "top_p": .8, "presence_penalty": 1.5,
                "messages": [{"role": "user", "content": prompt.replace(filename, path)}]}
        result = chat_result(client, body, True)
        check_call(result, "edit", {**arguments, "path": path})
        return result
    with ThreadPoolExecutor(2) as pool:
        peers = list(pool.map(peer, range(2)))
    record("agent_sampled_peers", peers)
    check_tool_agent_json(client, model, checks, chat_result)


def check_tool_agent_loop(client, model, checks, chat_result):
    def record(name, result):
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        return result

    common = dict(model=model, tools=agent_tools(), tool_choice="auto", parallel_tool_calls=False,
                  temperature=0, seed=41, max_completion_tokens=192,
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    old, new = "    return a - b", "    return a + b"
    # Advance from tool results alone, like a coding agent. Each action is
    # recorded separately; repeated no-progress calls fail within a bounded
    # number of requests, not after a minutes-long retry/compaction loop.
    files = {f"calc{i}.py": "def add(a, b):\n" + old + "\n" for i in range(2)}
    expected = "def add(a, b):\n" + new + "\n"
    history = [{"role": "user", "content":
                "For calc0.py then calc1.py: read the file with read, fix its subtraction "
                "bug using edit, then read it again to verify. Do all six actions, one "
                "tool call per turn. Once both files are correct, reply DONE without "
                "further tools. Do not write explanations."}]
    seen, verified = set(), set()
    for turn in range(9):
        result = record(f"agent_loop_{turn}", chat_result(
            client, {**common, "messages": history}, True))
        if turn:
            assert result["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, result
        if not result["tools"]:
            assert result["finish"] == "stop" and "DONE" in result["text"], result
            assert verified == set(files) and all(v == expected for v in files.values()), (files, result)
            break
        assert len(result["tools"]) == 1, result
        assert not any(tag in result["text"] for tag in
                       ("<tool_call", "</parameter", "</function", "</think>")), result
        call = result["tools"][0]
        function = call["function"]
        args = json.loads(function["arguments"])
        path = args["path"]
        assert path in files, args
        action = (function["name"], json.dumps(args, sort_keys=True), files[path])
        assert action not in seen, ("repeated action without progress", action)
        seen.add(action)
        if function["name"] == "read":
            reply = files[path]
            if reply == expected:
                verified.add(path)
        else:
            assert function["name"] == "edit" and args["edits"], call
            original = files[path]
            for edit in args["edits"]:
                assert original.count(edit["oldText"]) == 1, (args, original)
                files[path] = files[path].replace(edit["oldText"], edit["newText"], 1)
            assert files[path] == expected, files
            reply = "Replaced the requested text."
        history += [{"role": "assistant", "content": result["text"] or None,
                     "tool_calls": result["tools"]},
                    {"role": "tool", "tool_call_id": call["id"], "content": reply}]
    else:
        raise AssertionError("agent did not finish within nine turns")

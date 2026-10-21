"""Wide native tools keep optional parameters reachable in any order (#641)."""

from copy import deepcopy
import json
import sys

from tool_native import assert_native
from tool_reasoning import response_result


def todo_schema(count=12, status_last=False, loose=False):
    text = {"type": "string"}
    ids = {"type": "array", "items": {"type": "integer"}}
    properties = {
        "action": {"type": "string", "enum": ["get", "create", "update", "list"]},
        "subject": text,
        "description": text,
        "activeForm": text,
        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
        "blockedBy": ids,
        "addBlockedBy": ids,
        "removeBlockedBy": ids,
        "owner": text,
        "metadata": {"type": "object", "additionalProperties": True},
        "id": {"type": "integer"},
        "includeDeleted": {"type": "boolean"},
    }
    if count < 12:
        del properties["addBlockedBy"]
    if count < 11:
        del properties["removeBlockedBy"]
    for i in range(12, count):
        properties[f"unused{i}"] = text
    if status_last:
        properties["status"] = properties.pop("status")
    if loose:
        properties["subject"] = {"type": "string", "pattern": "^[a-z]+$"}
    return {"type": "object", "properties": deepcopy(properties),
            "required": ["action"], "additionalProperties": False}


def assert_update(result):
    assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, result
    function = result["tools"][0]["function"]
    assert function["name"] == "todo", result
    arguments = json.loads(function["arguments"])
    assert arguments == {"action": "update", "id": 1, "status": "completed"}, result
    assert type(arguments["id"]) is int, result
    assert_native(result, "wide tool parameters")


def check_tool_parameter_order(client, model, checks, chat_result):
    prompt = ("Mark existing task 1 as completed. Call the todo tool. "
              "Do not change any other field.")
    failures = []
    # The first four rows are the reporter's declaration-order/size controls.
    # Other rows cover wide/loose schemas, all APIs, sampling and thinking.
    cases = (
        ("chat_10", "chat", 10, False, False, False, False),
        ("chat_12_status_last", "chat", 12, True, False, False, False),
        ("chat_11", "chat", 11, False, False, False, True),
        ("chat_12", "chat", 12, False, False, False, False),
        ("chat_24_loose", "chat", 24, False, True, False, True),
        ("responses_12", "responses", 12, False, False, False, True),
        ("messages_12", "messages", 12, False, False, False, False),
        ("chat_12_sampled_thinking", "chat", 12, False, False, True, True),
        ("responses_12_null", "responses", 12, False, False, False, True),
        ("responses_12_omitted", "responses", 12, False, False, False, False),
    )
    for label, endpoint, count, reordered, loose, sampled, streaming in cases:
        function = {"name": "todo", "description": "Manage tasks.",
                    "strict": False, "parameters": todo_schema(count, reordered, loose)}
        common = {"model": model, "temperature": 0.7 if sampled else 0,
                  "tool_choice": "required", "parallel_tool_calls": False,
                  "extra_body": {"seed": 641, "cache_prompt": True}}
        if endpoint == "chat":
            request = {**common, "messages": [{"role": "user", "content": prompt}],
                       "tools": [{"type": "function", "function": function}],
                       "reasoning_effort": "low" if sampled else "none",
                       "max_completion_tokens": 512 if sampled else 256}
            result = chat_result(client, request, streaming)
        elif endpoint == "responses":
            # Match llama.cpp and Pi: omitted/null strictness must not turn
            # optional parameters into mandatory, unrequested field updates.
            if label.endswith("_null"):
                function["strict"] = None
            elif label.endswith("_omitted"):
                del function["strict"]
            result = response_result(client, {
                **common, "input": prompt, "tools": [{"type": "function", **function}],
                "reasoning": {"effort": "none"}, "max_output_tokens": 256,
                "store": False}, streaming)
        else:
            response = client.post("/messages", body={
                "model": model, "temperature": 0, "max_tokens": 256,
                "thinking": {"type": "disabled"},
                "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
                "tools": [{"name": "todo", "description": function["description"],
                           "input_schema": function["parameters"]}],
                "messages": [{"role": "user", "content": prompt}],
            }, cast_to=object)
            result = {
                "text": "".join(p["text"] for p in response["content"] if p["type"] == "text"),
                "tools": [{"function": {"name": p["name"], "arguments": json.dumps(p["input"])}}
                          for p in response["content"] if p["type"] == "tool_use"],
                "finish": "tool_calls" if response["stop_reason"] == "tool_use" else response["stop_reason"],
                "usage": response["usage"],
            }
        name = "tool_parameter_order_" + label
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        try:
            assert_update(result)
            if sampled:
                assert result["reasoning"].strip(), ("thinking was not exercised", result)
        except AssertionError as error:
            failures.append((name, str(error)))
    assert not failures, failures

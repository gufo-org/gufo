"""Short, bounded agent histories with ordinary (non-strict) nested tool schemas."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import sys

import openai

from tool_reasoning import response_result


def check_tool_history(client, model, checks, chat_result, vision, image_content):
    """Old names do not declare new tools or prevent the next turn (#357)."""
    tools = [{"type": "function", "function": {
        "name": "finish", "parameters": {"type": "object", "properties": {
            "value": {"type": "string", "const": "RECOVERED"}},
            "required": ["value"], "additionalProperties": False}}}]
    prompt = "These are completed historical operations. Continue from their results."
    content = ([image_content("red"), {"type": "text", "text": prompt}]
               if vision else prompt)
    messages = [{"role": "user", "content": content}]
    for _ in range(38):
        messages += [{"role": "assistant", "content": "Acknowledged."},
                     {"role": "user", "content": "Continue keeping the earlier record."}]
    calls = [{"id": f"old_{i}", "type": "function",
              "function": {"name": name, "arguments": '{"value":"historical"}'}}
             for i, name in enumerate(("…", "legacy 工具"))]
    messages += [{"role": "assistant", "content": None, "tool_calls": calls},
                 {"role": "tool", "tool_call_id": "old_1", "content": "Unknown tool; no action taken."},
                 {"role": "tool", "tool_call_id": "old_0", "content": "Unknown tool; no action taken."},
                 {"role": "user", "content": "Call finish once with value RECOVERED. No explanation."}]
    chat = dict(model=model, messages=messages, tools=tools, tool_choice="required",
                parallel_tool_calls=False, temperature=0, seed=41,
                reasoning_effort="none", max_completion_tokens=128)
    # Use actual Responses function_call items, not Chat-shaped message items.
    items = deepcopy(messages[:-4])
    if vision:
        image = image_content("red")["image_url"]["url"]
        items[0]["content"] = [{"type": "input_image", "image_url": image},
                               {"type": "input_text", "text": prompt}]
    for call in calls:
        items.append({"type": "function_call", "call_id": call["id"], **call["function"]})
    items += [{"type": "function_call_output", "call_id": "old_1", "output": "Unknown tool; no action taken."},
              {"type": "function_call_output", "call_id": "old_0", "output": "Unknown tool; no action taken."},
              messages[-1]]
    responses = dict(model=model, input=items,
                     tools=[{"type": "function", **t["function"], "strict": False} for t in tools],
                     tool_choice="required", parallel_tool_calls=False,
                     temperature=0, extra_body={"seed": 41}, reasoning={"effort": "none"},
                     max_output_tokens=128, store=False)

    def verify(name, result, cached=False):
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, result
        function = result["tools"][0]["function"]
        assert function["name"] == "finish", result
        assert json.loads(function["arguments"]) == {"value": "RECOVERED"}, result
        assert not result["text"].strip() and not result["reasoning"], result
        if cached:
            usage = result["usage"]
            details = usage.get("input_tokens_details", usage.get("prompt_tokens_details"))
            total = usage.get("input_tokens", usage.get("prompt_tokens"))
            assert details["cached_tokens"] >= total - 1, result
        return function

    for endpoint, request in (("chat", chat), ("responses", responses)):
        generate = chat_result if endpoint == "chat" else response_result
        first = verify(f"history_{endpoint}", generate(client, request, False))
        # The same name remains invalid as a NEW declaration.
        invalid = deepcopy(request)
        target = invalid["tools"][0]["function"] if endpoint == "chat" else invalid["tools"][0]
        target["name"] = "…"
        try:
            generate(client, invalid, True)
            raise AssertionError("invalid tool declaration was accepted")
        except openai.BadRequestError as error:
            assert error.status_code == 400, error
        empty = deepcopy(request)
        if endpoint == "chat":
            next(m for m in empty["messages"] if m.get("tool_calls"))["tool_calls"][0]["function"]["name"] = ""
        else:
            next(m for m in empty["input"] if m.get("type") == "function_call")["name"] = ""
        try:
            generate(client, empty, True)
            raise AssertionError("empty historical tool name was accepted")
        except openai.BadRequestError as error:
            assert error.status_code == 400, error
        again = verify(f"history_{endpoint}_retry", generate(client, request, True), cached=True)
        assert again == first, (again, first)

    stopped = chat_result(client, {**chat, "stop": "RECOVERED"}, True)
    checks["history_stop"] = stopped
    assert stopped["finish"] == "stop" and not stopped["tools"], stopped
    limited = chat_result(client, {**chat, "max_completion_tokens": 2}, True)
    checks["history_limit"] = limited
    assert limited["finish"] == "length" and not limited["tools"], limited
    verify("history_resume", chat_result(client, chat, True), cached=True)
    with ThreadPoolExecutor(2) as pool:
        peers = list(pool.map(lambda _: chat_result(
            client, {**chat, "temperature": .7, "top_p": .8, "presence_penalty": .3}, True),
            range(2)))
    for i, result in enumerate(peers):
        verify(f"history_sampled_peer_{i}", result, cached=True)


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

def check_untyped_agent_tools(client, model, checks, chat_result, vision, image_content):
    """An untyped neighbor must neither switch framing nor erase arguments."""
    def record(name, result):
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        return result

    def check_call(result, name="record", arguments=None):
        expected = {"value": "alpha"} if arguments is None else arguments
        assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, result
        function = result["tools"][0]["function"]
        assert function["name"] == name, result
        assert json.loads(function["arguments"]) == expected, result
        assert not result["text"].strip() and not result["reasoning"], result

    common = dict(model=model, tool_choice="auto", parallel_tool_calls=False,
                  temperature=0, seed=41, max_completion_tokens=128,
                  reasoning_effort="none", presence_penalty=0)
    prompt = ("Call record exactly once with one parameter named value. "
              "Its value is the string alpha, not a nested object. Then stop.")
    content = ([image_content("red"), {"type": "text", "text": prompt}]
               if vision else prompt)
    edit_prompt = ('Call edit once on calc.py with edits [{"oldText":"a","newText":"b"}]. '
                   'Do not call record or read. No explanation.')
    base_tools = [*agent_tools(), {"type": "function", "function": {
        "name": "record", "description": "Record a value.", "strict": False,
        "parameters": {"type": "object", "unevaluatedProperties": True}}}]
    base_request = {**common, "tools": base_tools,
                    "messages": [{"role": "user", "content": content}]}
    base_edit = {**base_request, "messages": [{"role": "user", "content": edit_prompt}]}
    base_responses = dict(model=model, input=prompt, tool_choice="auto",
                         parallel_tool_calls=False, store=False, temperature=0,
                         reasoning={"effort": "none"}, max_output_tokens=128,
                         extra_body={"seed": 41, "presence_penalty": 0})
    for name, keyword in (("open_object", "additionalProperties"),
                          ("unsupported_keyword", "unevaluatedProperties")):
        function = {"name": "record", "description": "Record a value.", "strict": False,
                    "parameters": {"type": "object", keyword: True}}
        tools = [*agent_tools(), {"type": "function", "function": function}]
        control = {**common, "tools": tools, "max_completion_tokens": 8,
                   "messages": [{"role": "user", "content": "Reply OK without using tools."}],
                   "extra_body": {"cache_prompt": False}}
        opened = record(f"untyped_{name}_prompt", chat_result(client, control))
        closed_tools = deepcopy(tools)
        closed_tools[-1]["function"]["parameters"][keyword] = False
        closed = record(f"untyped_{name}_control", chat_result(
            client, {**control, "tools": closed_tools}))
        delta = opened["usage"]["prompt_tokens"] - closed["usage"]["prompt_tokens"]
        assert abs(delta) <= 2, ("open tool changed the tool protocol", name, delta)

        # The schema deliberately provides no parameter types. State the
        # intended type explicitly: "with a JSON object" can ask the model
        # to pass that entire object as the value instead.
        request = {**common, "tools": tools, "messages": [{"role": "user", "content": content}]}
        for streaming in (False, True):
            result = record(f"untyped_{name}_chat_{streaming}",
                            chat_result(client, request, streaming))
            check_call(result)
            if streaming:
                assert result["usage"]["cached_tokens"] > 0, result
        chat_calls = result["tools"]
        responses = {**base_responses,
                     "tools": [{"type": "function", **t["function"], "strict": False}
                               for t in tools]}
        result = record(f"untyped_{name}_responses", response_result(client, responses, True))
        check_call(result)
        # Keep the same schema set and history: a result must be consumed once,
        # not trigger the record loop reported by the client.
        continued = {**request, "messages": [
            *request["messages"],
            {"role": "assistant", "content": None, "tool_calls": chat_calls},
            {"role": "tool", "tool_call_id": chat_calls[0]["id"], "content": "Recorded alpha."},
            {"role": "user", "content": "Finished. Do not call tools. Reply DONE."}]}
        done = record(f"untyped_{name}_finish", chat_result(client, continued, True))
        assert done["finish"] == "stop" and not done["tools"] and "DONE" in done["text"], done

        # A typed neighbor retains nested requirements despite the open tool.
        edit = {**request, "messages": [{"role": "user", "content": edit_prompt}]}
        edited = record(f"untyped_{name}_typed_neighbor", chat_result(client, edit, True))
        check_call(edited, "edit", {"path": "calc.py", "edits": [{"oldText": "a", "newText": "b"}]})
        limited = record(f"untyped_{name}_limit", chat_result(
            client, {**request, "tool_choice": "required", "max_completion_tokens": 2}, True))
        assert limited["finish"] == "length" and not limited["tools"], limited
        stopped = record(f"untyped_{name}_stop", chat_result(
            client, {**request, "stop": "alpha"}, True))
        assert stopped["finish"] == "stop" and not stopped["tools"], stopped
        retry = record(f"untyped_{name}_retry", chat_result(client, request, True))
        check_call(retry)
        assert retry["usage"]["cached_tokens"] > 0, retry
        # Batch unlike grammars, not just identical requests.
        with ThreadPoolExecutor(2) as pool:
            peers = list(pool.map(lambda body: chat_result(
                client, {**body, "temperature": .7, "top_p": .8}, True), (request, edit)))
        record(f"untyped_{name}_sampled_peers", peers)
        check_call(peers[0])
        check_call(peers[1], "edit", {"path": "calc.py", "edits": [{"oldText": "a", "newText": "b"}]})

    annotated = deepcopy(base_edit)
    parameters = annotated["tools"][0]["function"]["parameters"]
    parameters["unevaluatedProperties"] = True
    parameters["properties"]["edits"]["x-client-extension"] = True
    parameters["properties"]["edits"]["items"]["unevaluatedProperties"] = True
    result = record("untyped_supported_nested_fields", chat_result(client, annotated, True))
    check_call(result, "edit", {"path": "calc.py", "edits": [{"oldText": "a", "newText": "b"}]})
    result = record("untyped_supported_nested_responses", response_result(client, {
        **base_responses, "input": edit_prompt,
        "tools": [{"type": "function", **t["function"], "strict": False}
                  for t in annotated["tools"]]}, True))
    check_call(result, "edit", {"path": "calc.py", "edits": [{"oldText": "a", "newText": "b"}]})
    referenced = deepcopy(base_edit)
    function = referenced["tools"][0]["function"]
    function["parameters"] = {"$ref": "#/$defs/edit",
                              "$defs": {"edit": function["parameters"]}}
    result = record("untyped_referenced_tool", chat_result(client, referenced, True))
    check_call(result, "edit", {"path": "calc.py", "edits": [{"oldText": "a", "newText": "b"}]})
    finite = deepcopy(base_request)
    finite["tools"][-1]["function"]["parameters"] = {
        "type": "object", "const": {"value": "alpha"}, "unevaluatedProperties": True}
    finite["tool_choice"] = {"type": "function", "function": {"name": "record"}}
    finite["messages"] = [{"role": "user", "content": "Call record with value beta."}]
    result = record("untyped_finite_object", chat_result(client, finite, True))
    check_call(result)


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


def check_mixed_tool_schemas(client, model, checks, chat_result, vision, image_content):
    """Mixed native/JSON tool sets retain arguments, types and continuation."""
    def record(name, result, expected):
        checks[f"mixed_{name}"] = result
        print(f"CHECK mixed_{name}", file=sys.stderr, flush=True)
        assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, result
        function = result["tools"][0]["function"]
        assert function["name"] == "record", result
        assert json.loads(function["arguments"]) == expected, result
        # Auto calls may include ordinary assistant prose. Check that the
        # envelope was parsed rather than requiring a tool-only response.
        assert not any(marker in result["text"] for marker in
                       ("<tool_call>", "<function=", "<｜DSML｜invoke")), result
        return result

    common = dict(model=model, tool_choice="auto", parallel_tool_calls=False,
                  temperature=0, seed=41, presence_penalty=0,
                  max_completion_tokens=160, reasoning_effort="none")
    inline = {"type": "object", "properties": {"value": {"type": "string"}},
              "required": ["value"], "additionalProperties": False}
    # A nullable string deliberately needs JSON to distinguish null from the
    # string "null". This neighbor keeps testing fallback after URI support.
    nullable = {"type": "function", "function": {
        "name": "lookup", "description": "Look up an optional key.",
        "parameters": {"type": "object", "properties": {
            "key": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
            "required": ["key"]}}}
    fetch = {"type": "function", "function": {
        "name": "fetch", "description": "Fetch a URL.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "format": "uri"}}, "required": ["url"]}}}
    prompt = ("Call record exactly once with one parameter named value. "
              "Its value is the string alpha, not a nested object. Then stop.")
    expected = {"value": "alpha"}
    def followups(base, first):
        continued = {**base, "messages": [
            *base["messages"], {"role": "assistant", "content": first["text"] or None,
                               "tool_calls": first["tools"]},
            {"role": "tool", "tool_call_id": first["tools"][0]["id"], "content": "Recorded alpha."},
            {"role": "user", "content": "Finished. Do not call any tools. Reply DONE."}]}
        done = chat_result(client, continued, True)
        checks["mixed_result_turn"] = done
        assert done["finish"] == "stop" and not done["tools"] and "DONE" in done["text"], done
        assert done["usage"]["cached_tokens"] > 0, done
        for name, changes, finish in (
                ("limit", {"tool_choice": "required", "max_completion_tokens": 2}, "length"),
                ("stop", {"stop": "alpha"}, "stop")):
            result = chat_result(client, {**base, **changes}, True)
            checks[f"mixed_{name}"] = result
            assert result["finish"] == finish and not result["tools"], result
        retry = record("retry", chat_result(client, base, True), expected)
        assert retry["usage"]["cached_tokens"] > 0, retry
        edit = {**base, "messages": [{"role": "user", "content":
            'Call edit once on calc.py with edits [{"oldText":"a","newText":"b"}]. '
            'Do not call record, read or lookup. No explanation.'}]}
        with ThreadPoolExecutor(2) as pool:
            peers = list(pool.map(lambda body: chat_result(
                client, {**body, "temperature": .7, "top_p": .8}, True), (base, edit)))
        record("sampled_record", peers[0], expected)
        checks["mixed_sampled_edit"] = peers[1]
        assert peers[1]["finish"] == "tool_calls" and len(peers[1]["tools"]) == 1, peers[1]
        function = peers[1]["tools"][0]["function"]
        assert function["name"] == "edit" and json.loads(function["arguments"]) == {
            "path": "calc.py", "edits": [{"oldText": "a", "newText": "b"}]}, peers[1]


    cases = [
        ("untyped", {"type": "object", "unevaluatedProperties": True}),
        ("one_of", {"type": "object", "oneOf": [inline]}),
        ("all_of", {"type": "object", "allOf": [inline]}),
        ("named_ref", {"$schema": "http://json-schema.org/draft-07/schema#",
                       "$ref": "#/definitions/Record", "definitions": {"Record": inline}}),
        ("typed_ref", {"type": "object", "$ref": "#/$defs/Record",
                       "$defs": {"Record": inline}}),
    ]
    for index, (name, parameters) in enumerate(cases):
        tools = [*agent_tools(), nullable if index < 3 else fetch,
                 {"type": "function", "function": {
                     "name": "record", "description": "Record a value.",
                     "strict": False, "parameters": parameters}}]
        content = ([image_content("red"), {"type": "text", "text": prompt}]
                   if vision and index == 0 else prompt)
        request = {**common, "tools": tools,
                   "messages": [{"role": "user", "content": content}]}
        response_input = ([{"role": "user", "content": [
            {"type": "input_image", "image_url": image_content("red")["image_url"]["url"]},
            {"type": "input_text", "text": prompt}]}] if vision and index == 0 else prompt)
        responses = dict(model=model, input=response_input, tool_choice="auto",
                         parallel_tool_calls=False, store=False, temperature=0,
                         reasoning={"effort": "none"}, max_output_tokens=160,
                         extra_body={"seed": 41, "presence_penalty": 0},
                         tools=[{"type": "function", **t["function"], "strict": False}
                                for t in tools])
        result = record(f"{name}_chat", chat_result(client, request, bool(index % 2)), expected)
        record(f"{name}_responses", response_result(
            client, responses, not bool(index % 2)), expected)
        if index == 0:
            # Exercise reuse before cycling through enough distinct schemas
            # to evict this prompt from the bounded cache.
            followups(request, result)

    # Exercise extra keys at both levels, with an exact typed neighbor.
    for name, parameters, arguments in (
        ("pattern_root", {"type": "object", "properties": {"value": {"type": "string"}},
                          "required": ["value"], "patternProperties": {"^x_": {"type": "string"}}},
         {"value": "alpha", "x_note": "beta"}),
        ("pattern_nested", {"type": "object", "properties": {"m": {
            "type": "object", "patternProperties": {"^k": {"type": "integer"}},
            "additionalProperties": False}}, "required": ["m"]}, {"m": {"k1": 1}}),
        ("uri", {"type": "object", "properties": {"url": {"type": "string", "format": "uri"}},
                 "required": ["url"]}, {"url": "https://example.org/test"}),
        ("nullable_null", {"type": "object", "properties": {
            "value": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
            "required": ["value"]}, {"value": None}),
        ("nullable_string", {"type": "object", "properties": {
            "value": {"type": ["string", "null"]}}, "required": ["value"]}, {"value": "null"}),
    ):
        prompt = ("Call record exactly once with these exact arguments: " +
                  json.dumps(arguments) + ". Preserve every JSON type. No explanation.")
        if name == "nullable_string":
            # Both alternatives are valid schema values; explicitly select
            # the string rather than measuring the model's default choice.
            prompt += (" The value is the four-letter STRING null, not the JSON null value. "
                       "Put the four letters inside JSON quotation marks.")
        body = {**common, "tools": [*agent_tools(), {"type": "function", "function": {
            "name": "record", "parameters": parameters}}],
            "messages": [{"role": "user", "content": prompt}]}
        record(name, chat_result(client, body, True), arguments)


def check_tool_schema_edges(client, model, checks, chat_result, vision, image_content):
    """Admitted arguments preserve JSON types; invalid schemas cannot dead-end."""
    wildcard = {"type": "object", "properties": {"value": {"type": "string"}},
                "required": ["value"], "patternProperties": {"^x_": {}}}
    branch = {"properties": {"payload": {"type": "integer"}}, "required": ["payload"]}
    conditional = {"type": "object", "properties": {"kind": {"type": "string"}},
                   "required": ["kind"], "if": {"properties": {"kind": {"const": "x"}}},
                   "then": branch}
    dependency = {"type": "object", "properties": {"kind": {"type": "string"}},
                  "required": ["kind"], "dependencies": {"kind": branch}}
    metadata = {"type": "object", "properties": {
        "edits": {"type": "array", "items": {"type": "object", "properties": {
            "oldText": {"type": "string", "const": "a"},
            "newText": {"type": "string", "const": "b"}, "metadata": {}},
            "required": ["oldText", "newText"]}}}, "required": ["edits"]}
    cases = [
        ("wildcard", wildcard, {"value": "alpha", "x_n": 1, "x_b": True,
                               "x_z": None, "x_a": [1], "x_o": {"n": 1}, "x_s": "1"}, "auto"),
        ("conditional", conditional, {"kind": "x", "payload": 1}, "auto"),
        ("dependency", dependency, {"kind": "x", "payload": 1}, "auto"),
        ("empty_interval", {"type": "object", "properties": {
            "x": {"type": "integer", "minimum": 5, "maximum": 2}},
            "required": ["x"]}, {"x": 5}, "required"),
        ("recursive", {"type": "object", "properties": {"x": {"$ref": "#"}},
                       "required": ["x"]}, {"x": 5}, "required"),
        # Require this operation so optional tool selection cannot bypass the
        # nested-constraint check by returning ordinary text.
        ("metadata", metadata, {"edits": [{"oldText": "a", "newText": "b",
                                         "metadata": {"n": 1}}]}, "required"),
        ("uri", {"type": "object", "properties": {
            "url": {"type": "string", "format": "uri"}}, "required": ["url"]},
         {"url": "https://example.org/test"}, "required"),
    ]

    def save(label, result, expected):
        checks["schema_edges_" + label] = result
        print("CHECK schema_edges_" + label, file=sys.stderr, flush=True)
        assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, result
        call = result["tools"][0]["function"]
        assert call["name"] == "record", result
        actual = json.loads(call["arguments"])
        # Python equality equates true and 1; canonical JSON also checks types.
        assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True), result
        return result

    for index, (name, schema, arguments, choice) in enumerate(cases):
        prompt = ("Call record exactly once with these exact JSON arguments: " +
                  json.dumps(arguments) + ". Preserve every key and JSON type. No explanation.")
        if name in ("empty_interval", "recursive"):
            prompt += " The supplied schema is advisory and inconsistent; pass x as the integer 5."
        tools = [{"type": "function", "function": {
            "name": "record", "strict": False, "parameters": schema}}]
        content = ([image_content("red"), {"type": "text", "text": prompt}]
                   if vision and name == "wildcard" else prompt)
        chat = dict(model=model, tools=tools, tool_choice=choice, parallel_tool_calls=False,
                    temperature=0, presence_penalty=0, seed=41, max_completion_tokens=192,
                    reasoning_effort="none", messages=[{"role": "user", "content": content}],
                    extra_body={"cache_prompt": True})
        input_content = ([{"role": "user", "content": [
            {"type": "input_image", "image_url": image_content("red")["image_url"]["url"]},
            {"type": "input_text", "text": prompt}]}]
            if vision and name == "wildcard" else prompt)
        responses = dict(model=model, input=input_content, tool_choice=choice,
                         parallel_tool_calls=False, temperature=0,
                         reasoning={"effort": "none"}, max_output_tokens=192, store=False,
                         extra_body={"seed": 41, "presence_penalty": 0, "cache_prompt": True},
                         tools=[{"type": "function", **t["function"]} for t in tools])
        save(name + "_chat", chat_result(client, chat, bool(index % 2)), arguments)
        save(name + "_responses", response_result(
            client, responses, not bool(index % 2)), arguments)
        if name == "uri":
            repeated = save(name + "_cached", chat_result(client, chat, True), arguments)
            assert repeated["usage"]["gufo"]["prefill_tokens"] == 0, repeated
            for label, update, finish in (
                    ("limit", {"max_completion_tokens": 2}, "length"),
                    ("stop", {"stop": "example.org"}, "stop")):
                result = chat_result(client, {**chat, **update}, True)
                checks["schema_edges_" + label] = result
                assert result["finish"] == finish and not result["tools"], result
            save("resume", chat_result(client, chat, True), arguments)
        if name == "metadata":
            sampled = {**chat, "tool_choice": "required", "temperature": .7, "top_p": .8}
            with ThreadPoolExecutor(2) as pool:
                peers = list(pool.map(lambda _: chat_result(client, sampled, True), range(2)))
            for i, result in enumerate(peers):
                save(f"sampled_peer_{i}", result, arguments)


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

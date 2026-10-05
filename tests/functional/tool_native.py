"""Native tool syntax for every schema, as llama.cpp keeps it (#383, #438).

A schema that native tags cannot enforce exactly used to move every tool of the
request to a JSON envelope and add a system instruction. The template and the
history still taught native calls, so long sessions mixed both syntaxes and the
model wrote `<invoke>`/JSON framing into content. Each case below puts one such
schema beside opencode's real tool set and checks that nothing changes for the
model: no prompt instruction, native calls, typed arguments and full reuse of
the generated call on the next turn.
"""

from copy import deepcopy
import json
import sys

from tool_reasoning import response_result

SCHEMA = "https://json-schema.org/draft/2020-12/schema"


def opencode_tools():
    """opencode 1.18's tool schemas as it sends them (zod adds `$schema`)."""
    def tool(name, properties, required):
        return {"type": "function", "function": {
            "name": name, "description": f"opencode {name} tool.",
            "parameters": {"$schema": SCHEMA, "type": "object",
                           "properties": properties, "required": required}}}
    text = {"type": "string"}
    return [
        tool("bash", {
            "command": {"type": "string", "description": "The command to execute"},
            "timeout": {"minimum": -9007199254740991, "exclusiveMinimum": 0,
                        "type": "integer", "maximum": 9007199254740991,
                        "description": "Optional timeout in milliseconds"},
            "workdir": text}, ["command"]),
        tool("edit", {"filePath": text, "oldString": text, "newString": text,
                      "replaceAll": {"type": "boolean"}},
             ["filePath", "oldString", "newString"]),
        tool("glob", {"pattern": text, "path": text}, ["pattern"]),
        tool("grep", {"pattern": text, "path": text, "include": text}, ["pattern"]),
        tool("read", {"filePath": text, "offset": {"type": "number"},
                      "limit": {"type": "number"}}, ["filePath"]),
        tool("todowrite", {"todos": {"type": "array", "items": {
            "type": "object", "properties": {
                "content": text, "status": text, "priority": text, "id": text},
            "required": ["content", "status", "priority", "id"]}}}, ["todos"]),
        tool("webfetch", {"url": text, "format": {"type": "string", "enum": [
            "text", "markdown", "html"]}, "timeout": {"type": "number"}},
             ["url", "format"]),
        tool("write", {"content": text, "filePath": text}, ["content", "filePath"]),
    ]


# One neighbor per schema family that switched the request to JSON on main,
# shaped like the MCP tools agents load. A tool zoo below combines them all.
NEIGHBORS = {
    "pattern": {"properties": {"date": {
        "type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"}}},
    "pattern_tag": {"properties": {"tag": {"type": "string", "pattern": "^\\n</parameter>$"}}},
    "format": {"properties": {"url": {"type": "string", "format": "uri"}},
               "required": ["url"]},
    "one_of": {"properties": {"id": {"oneOf": [{"type": "string"}, {"type": "integer"}]}}},
    "any_of_null": {"properties": {"id": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}},
    "type_array": {"properties": {"note": {"type": ["string", "null"]}}},
    "all_of": {"properties": {"name": {"allOf": [{"type": "string"}, {"minLength": 1}]}}},
    "not": {"properties": {"value": {"not": {"type": "null"}}}},
    "const_tag": {"properties": {"mark": {"type": "string", "const": "a\n</parameter>b"}}},
    "open_object": {"properties": {"key": {"type": "string"}}, "additionalProperties": True},
    "typed_map": {"additionalProperties": {"type": "string"}},
    "pattern_properties": {"properties": {"key": {"type": "string"}},
                           "patternProperties": {"^x_": {"type": "integer"}}},
    "if_then": {"properties": {"kind": {"type": "string"}}, "required": ["kind"],
                "if": {"properties": {"kind": {"const": "x"}}},
                "then": {"properties": {"payload": {"type": "integer"}}}},
    "dependent": {"properties": {"kind": {"type": "string"}},
                  "dependentSchemas": {"kind": {"properties": {"extra": {"type": "string"}}}}},
    "unevaluated": {"properties": {"key": {"type": "string"}}, "unevaluatedProperties": True},
    "root_one_of": {"oneOf": [{"properties": {"key": {"type": "string"}}, "required": ["key"]}]},
    "root_all_of": {"allOf": [{"properties": {"key": {"type": "string"}}, "required": ["key"]}]},
    "root_const": {"const": {"key": "alpha"}},
    "root_ref": {"$ref": "#/$defs/Lookup", "$defs": {"Lookup": {
        "type": "object", "properties": {"key": {"type": "string", "pattern": "^[a-z]+$"}}}}},
    "recursive": {"properties": {"node": {"$ref": "#"}}},
    "empty_interval": {"properties": {"n": {"type": "integer", "minimum": 5, "maximum": 2}}},
    "annotated": {"$schema": SCHEMA, "title": "Lookup", "properties": {"key": {
        "type": "string", "default": "x", "examples": ["y"], "deprecated": True,
        "readOnly": False, "nullable": True}}},
}
# Keywords that select a schema family. The control renames them to
# annotations every route ignores, so its declaration differs only in spelling.
TRIGGERS = ("pattern", "format", "oneOf", "anyOf", "allOf", "not", "const",
            "additionalProperties", "patternProperties", "if", "then",
            "dependentSchemas", "unevaluatedProperties", "$ref", "minimum",
            "maximum", "$schema", "default", "examples", "deprecated", "readOnly",
            "nullable", "minLength")


def renamed(value):
    if isinstance(value, dict):
        return {("x-" + key.lstrip("$") if key in TRIGGERS else key): renamed(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [renamed(item) for item in value]
    return value


def triggers(value):
    if isinstance(value, dict):
        return sum((key in TRIGGERS) + triggers(item) for key, item in value.items())
    if isinstance(value, list):
        return sum(triggers(item) for item in value)
    return 0


def neighbor(name, control=False, index=None):
    parameters = {"type": "object", **deepcopy(NEIGHBORS[name])}
    if control:
        parameters = renamed(parameters)
        parameters["type"] = "object"
    return {"type": "function", "function": {
        "name": "mcp_lookup" if index is None else f"mcp_{name}_{index}",
        "description": "Look up a record in the catalog.",
        "parameters": parameters}}


def zoo():
    """Every family at once, a strict union and many plain tools."""
    tools = [neighbor(name, index=i) for i, name in enumerate(NEIGHBORS)]
    tools.append({"type": "function", "function": {
        "name": "mcp_strict_union", "description": "Strict union.", "strict": True,
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": ["id"], "properties": {
                           "id": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}}}})
    for i in range(40):
        tools.append({"type": "function", "function": {
            "name": f"plain_{i}", "description": f"Plain tool {i}.",
            "parameters": {"type": "object", "properties": {
                "value": {"type": "string"}}, "required": ["value"]}}})
    return tools


def cached(usage):
    details = usage.get("prompt_tokens_details") or {}
    return usage.get("cached_tokens", details.get("cached_tokens", 0)) or 0


def assert_native(result, label):
    text = result["text"]
    leaked = [tag for tag in ("<tool_call>", "</tool_call>", "<function=", "<parameter",
                              "<invoke", '{"name"', "<｜DSML｜") if tag in text]
    assert not leaked, (label, "framing leaked into content", leaked, result)


def check_native_tool_schemas(client, model, checks, chat_result, sampling_preset="qwen38",
                              image=None):
    deepseek = sampling_preset == "deepseek4"
    command = "echo native-schemas"
    prompt = (f"Use the bash tool to run exactly this command: {command}\n"
              "Set its timeout parameter to 5000. Do not call any other tool.")
    common = dict(model=model, temperature=0, seed=41, reasoning_effort="none",
                  parallel_tool_calls=False, max_completion_tokens=160,
                  extra_body={"cache_prompt": True})

    def record(label, result):
        checks[label] = result
        print(f"CHECK {label}", file=sys.stderr, flush=True)
        return result

    def assert_bash(result, label):
        assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, (label, result)
        function = result["tools"][0]["function"]
        assert function["name"] == "bash", (label, result)
        arguments = json.loads(function["arguments"])
        assert arguments.get("command") == command, (label, result)
        # The integer bound keeps its JSON type in native tags.
        assert arguments.get("timeout", 5000) == 5000 and isinstance(
            arguments.get("timeout", 5000), int), (label, result)
        assert_native(result, label)

    for index, name in enumerate(NEIGHBORS):
        for choice in ("auto", "required"):
            label = f"native_{name}_{choice}"
            stream = bool(index % 2)
            request = {**common, "tool_choice": choice,
                       "tools": [*opencode_tools(), neighbor(name)],
                       "messages": [{"role": "system", "content": "You are a coding agent."},
                                    {"role": "user", "content": prompt}]}
            first = record(label, chat_result(client, request, stream))
            assert_bash(first, label)

            # No instruction is added: the prompt matches the control within
            # the spelling of the renamed keyword. The JSON instruction alone
            # was about 45 tokens.
            control = {**request, "tools": [*opencode_tools(), neighbor(name, True)],
                       "max_completion_tokens": 1}
            baseline = record(label + "_control", chat_result(client, control))
            delta = first["usage"]["prompt_tokens"] - baseline["usage"]["prompt_tokens"]
            # Each renamed keyword may add a token; the instruction alone was
            # about 40 tokens on main.
            tolerance = 2 + 2 * triggers(NEIGHBORS[name])
            assert tolerance < 30 and abs(delta) <= tolerance, (
                label, "prompt changed beyond the schema text", delta, tolerance,
                first["usage"], baseline["usage"])

            # The call the model generated is the call history renders, so the
            # next turn reuses it from cache instead of re-prefilling it.
            call = first["tools"][0]
            continued = {**request, "tool_choice": "auto", "messages": [
                *request["messages"],
                {"role": "assistant", "content": first["text"] or None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": "native-schemas\n"},
                {"role": "user", "content": "Reply with the single word DONE."}]}
            done = record(label + "_continued", chat_result(client, continued, not stream))
            assert done["finish"] == "stop" and not done["tools"], (label, done)
            assert cached(done["usage"]) > first["usage"]["prompt_tokens"], (
                label, "the generated call was not reused on the next turn",
                first["usage"], done["usage"])
            assert_native(done, label + "_continued")

    # Calls to the neighbor itself keep their declared JSON types.
    typed = (
        ("one_of", {"id": 42}, "Call mcp_lookup once with id set to the integer 42."),
        ("one_of", {"id": "abc"}, "Call mcp_lookup once with id set to the string abc."),
        ("all_of", {"name": "42"}, "Call mcp_lookup once with name set to the string 42."),
        ("pattern", {"date": "2026-10-05"},
         "Call mcp_lookup once with date set to 2026-10-05."),
    )
    for index, (name, expected, instruction) in enumerate(typed):
        for stream in (False, True):
            label = f"native_typed_{name}_{index}_{'stream' if stream else 'buffered'}"
            request = {**common, "tool_choice": "required",
                       "tools": [*opencode_tools(), neighbor(name)],
                       "messages": [{"role": "user",
                                     "content": instruction + " No other arguments."}]}
            result = record(label, chat_result(client, request, stream))
            assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, result
            function = result["tools"][0]["function"]
            assert function["name"] == "mcp_lookup", result
            assert json.loads(function["arguments"]) == expected, (label, result)
            assert_native(result, label)

    # An open object generates its declared key; only DeepSeek's string flag
    # can also carry an undeclared one.
    request = {**common, "tool_choice": "required",
               "tools": [*opencode_tools(), neighbor("open_object")],
               "messages": [{"role": "user", "content":
                             "Call mcp_lookup once with key set to alpha. No other arguments."}]}
    result = record("native_open_object_declared", chat_result(client, request, True))
    assert json.loads(result["tools"][0]["function"]["arguments"]) == {"key": "alpha"}, result

    # Every family at once, with opencode's tools: still one native call that
    # the next turn reuses, in both tool choices.
    for choice in ("auto", "required"):
        label = f"native_zoo_{choice}"
        request = {**common, "tool_choice": choice, "tools": [*opencode_tools(), *zoo()],
                   "messages": [{"role": "system", "content": "You are a coding agent."},
                                {"role": "user", "content": prompt}]}
        first = record(label, chat_result(client, request, choice == "auto"))
        assert_bash(first, label)
        call = first["tools"][0]
        continued = {**request, "tool_choice": "auto", "messages": [
            *request["messages"],
            {"role": "assistant", "content": first["text"] or None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": "native-schemas\n"},
            {"role": "user", "content": "Reply with the single word DONE."}]}
        done = record(label + "_continued", chat_result(client, continued, choice != "auto"))
        assert done["finish"] == "stop" and not done["tools"], (label, done)
        assert cached(done["usage"]) > first["usage"]["prompt_tokens"], (
            label, "the generated call was not reused on the next turn",
            first["usage"], done["usage"])

    check_llama_cpp_parity(client, model, checks, chat_result, common, deepseek)
    check_finite_argument_types(client, model, checks, chat_result, image)

    # A strict union stays native too; llama.cpp has no strict mode. (Strict
    # validation still rejects keywords such as oneOf before generation.)
    strict = {"type": "function", "function": {
        "name": "mcp_lookup", "description": "Look up a record in the catalog.",
        "strict": True, "parameters": {
            "type": "object", "additionalProperties": False, "required": ["id"],
            "properties": {"id": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}}}}
    request = {**common, "tool_choice": "auto", "tools": [*opencode_tools(), strict],
               "messages": [{"role": "user", "content": prompt}]}
    assert_bash(record("native_strict_neighbor", chat_result(client, request, True)),
                "native_strict_neighbor")

    # The Responses API takes the same route.
    responses = dict(model=model, input=prompt, tool_choice="required",
                     parallel_tool_calls=False, store=False, temperature=0,
                     reasoning={"effort": "none"}, max_output_tokens=160,
                     extra_body={"seed": 41},
                     tools=[{"type": "function", **tool["function"]}
                            for tool in (*opencode_tools(), neighbor("pattern"))])
    result = record("native_responses", response_result(client, responses, True))
    assert result["tools"] and result["tools"][0]["function"]["name"] == "bash", result
    assert_native(result, "native_responses")
    print(f"CHECK native_schemas_complete deepseek={deepseek}", file=sys.stderr, flush=True)


def check_finite_argument_types(client, model, checks, chat_result, image=None):
    """const/enum without type still have a type, including through unions."""
    check_read_arguments(client, model, checks, chat_result)
    literal = ("line\n</parameter> is literal\n"
               "</parameter> <parameter=other>data\n"
               "</parameter> </function>data\nend")
    for index, (schema, value) in enumerate((
        ({"const": "42"}, "42"),
        ({"enum": ["true", "false"]}, "true"),
        ({"enum": ["null", "42"]}, "null"),
        ({"anyOf": [{"const": "42"}, {"type": "null"}]}, "42"),
        ({"enum": ["42", 17]}, 17),
        ({"minLength": 1}, "42"),
        # Fix the requested value so this measures byte preservation, not
        # whether a model chooses to copy a delimiter-looking instruction.
        ({"type": "string", "const": literal}, literal),
    )):
        function = {"name": "record", "description": "Record the supplied value.",
                    "parameters": {"type": "object", "properties": {"v": schema},
                                   "required": ["v"], "additionalProperties": False}}
        prompt = f"Call record once with v set to {json.dumps(value)}. No other arguments."
        for responses in (False, True):
            for stream in (False, True):
                label = f"finite_type_{index}_{'responses' if responses else 'chat'}_{stream}"
                if responses:
                    result = response_result(client, dict(
                        model=model, input=prompt, tools=[{"type": "function", **function}],
                        tool_choice="required", temperature=0, reasoning={"effort": "none"},
                        max_output_tokens=128, store=False), stream)
                else:
                    result = chat_result(client, dict(
                        model=model, messages=[{"role": "user", "content": prompt}],
                        tools=[{"type": "function", "function": function}],
                        tool_choice="required", temperature=0, reasoning_effort="none",
                        max_completion_tokens=128), stream)
                checks[label] = result
                assert len(result["tools"]) == 1, (label, result)
                actual = json.loads(result["tools"][0]["function"]["arguments"])
                assert actual == {"v": value} and type(actual["v"]) is type(value), (label, actual)
                assert_native(result, label)
                print(f"CHECK {label}", file=sys.stderr, flush=True)
    check_literal_history(client, model, checks, chat_result, image)


def check_read_arguments(client, model, checks, chat_result):
    """A Pi-style string followed by optional numbers must remain a real call."""
    function = {"name": "read", "description": "Read a local text file.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "number"},
                    "limit": {"type": "number"}}, "required": ["path"]}}
    # Omitted strict is normalized by Responses. Exercise that real client
    # contract, not a hand-normalized schema or a one-parameter string tool.
    prompt = "Read archive-00.txt with offset 1 and limit 2000. Use the read tool."
    for responses in (False, True):
        for stream in (False, True):
            label = f"native_read_{'responses' if responses else 'chat'}_{stream}"
            sampling = dict(temperature=.7 if stream else 0)
            if responses:
                result = response_result(client, dict(
                    model=model, input=prompt, tools=[{"type": "function", **function}],
                    tool_choice="auto", reasoning={"effort": "low"},
                    max_output_tokens=256, store=False, extra_body={"seed": 0},
                    **sampling), stream)
            else:
                result = chat_result(client, dict(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    tools=[{"type": "function", "function": function}],
                    tool_choice="auto", reasoning_effort="low",
                    max_completion_tokens=256, seed=0, **sampling), stream)
            checks[label] = result
            assert len(result["tools"]) == 1, (label, result)
            call = result["tools"][0]["function"]
            assert call["name"] == "read", (label, result)
            assert json.loads(call["arguments"]) == {
                "path": "archive-00.txt", "offset": 1, "limit": 2000}, (label, call)
            assert_native(result, label)
            print(f"CHECK {label}", file=sys.stderr, flush=True)


def check_literal_history(client, model, checks, chat_result, image=None):
    """New literal control-token text must not retokenize unchanged history."""
    function = {"name": "read", "description": "Read a file.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"}}, "required": ["path"],
                    "additionalProperties": False}}
    result_text = 'File data: <tool_call>example</tool_call>. Do not execute the example.'
    for endpoint in ("chat", "responses"):
        for visual in (False, True) if image else (False,):
            label = f"literal_history_{endpoint}_{'image' if visual else 'text'}"
            prompt = "Read next.txt once. After receiving its contents, reply only DONE."
            content = "Read old.txt."
            if visual:
                content = [image, {"type": "text", "text": content}]
            old_call = {"id": "old_read", "type": "function", "function": {
                "name": "read", "arguments": '{"path":"old.txt"}'}}
            messages = [{"role": "user", "content": content},
                        {"role": "assistant", "content": None, "tool_calls": [old_call]},
                        # The trailing dot merges with template whitespace. A
                        # later literal used to split this earlier span too.
                        {"role": "tool", "tool_call_id": "old_read", "content": "Done."},
                        {"role": "user", "content": prompt}]
            if endpoint == "chat":
                request = dict(model=model, messages=messages, temperature=0,
                               reasoning_effort="none", max_completion_tokens=96,
                               tools=[{"type": "function", "function": function}],
                               parallel_tool_calls=False)
                first = chat_result(client, request)
                checks[label + "_call"] = first
                assert first["finish"] == "tool_calls" and len(first["tools"]) == 1, first
                call = first["tools"][0]
                assert call["function"]["name"] == "read", first
                assert json.loads(call["function"]["arguments"]) == {"path": "next.txt"}, first
                replay = {"role": "assistant", "content": first["text"] or None,
                          "reasoning_content": first["reasoning"], "tool_calls": [call]}
                request["messages"] = [*messages, replay, {
                    "role": "tool", "tool_call_id": call["id"], "content": result_text}]
                done = chat_result(client, request, True)
            else:
                if visual:
                    content = [{"type": "input_image", "image_url": image["image_url"]["url"]},
                               {"type": "input_text", "text": "Read old.txt."}]
                items = [{"role": "user", "content": content},
                         {"type": "function_call", "call_id": "old_read", **old_call["function"]},
                         {"type": "function_call_output", "call_id": "old_read", "output": "Done."},
                         {"role": "user", "content": prompt}]
                request = dict(model=model, input=items, temperature=0,
                               reasoning={"effort": "none"}, max_output_tokens=96, store=False,
                               tools=[{"type": "function", **function, "strict": False}],
                               parallel_tool_calls=False)
                response = client.responses.create(**request)
                first = {"usage": response.usage.to_dict(), "output": [
                    item.to_dict() for item in response.output]}
                checks[label + "_call"] = first
                calls = [item for item in response.output if item.type == "function_call"]
                assert response.status == "completed" and len(calls) == 1, response
                call = calls[0]
                assert call.name == "read" and json.loads(call.arguments) == {"path": "next.txt"}, response
                request["input"] = [*items, *first["output"], {
                    "type": "function_call_output", "call_id": call.call_id, "output": result_text}]
                done = response_result(client, request, True)
            checks[label + "_continued"] = done
            assert done["text"].strip() == "DONE" and not done["tools"], done
            details = done["usage"].get("input_tokens_details",
                                       done["usage"].get("prompt_tokens_details"))
            assert details["cached_tokens"] == first["usage"]["total_tokens"], (
                label, "literal tool data invalidated unchanged history", first, done)
            print(f"CHECK {label}", file=sys.stderr, flush=True)


def streamed_call_order(client, request):
    """Content deltas before and after the first tool-call delta of a stream."""
    before, after, calls, finish = "", "", [], None
    stream = client.chat.completions.create(**request, stream=True,
                                            stream_options={"include_usage": True})
    with stream:
        for chunk in stream:
            for choice in chunk.choices:
                delta = choice.delta
                if delta.tool_calls:
                    calls.extend(call.to_dict() for call in delta.tool_calls)
                if delta.content:
                    if calls:
                        after += delta.content
                    else:
                        before += delta.content
                finish = choice.finish_reason or finish
    return before, after, calls, finish


def check_llama_cpp_parity(client, model, checks, chat_result, common, deepseek):
    """Behaviors llama.cpp's Qwen3-Coder/DeepSeek grammars define (#438)."""
    def record(label, result):
        checks[label] = result
        print(f"CHECK {label}", file=sys.stderr, flush=True)
        return result

    # Calls end the output: nothing may follow them, even when asked to.
    for choice in ("auto", "required"):
        label = f"parity_output_ends_after_call_{choice}"
        request = {**common, "tool_choice": choice,
                   "tools": [*opencode_tools(), neighbor("pattern")],
                   "messages": [{"role": "user", "content":
                                 "Call bash with command `echo end`, and after the call "
                                 "write the sentence: The call is done."}]}
        before, after, calls, finish = streamed_call_order(client, request)
        record(label, {"before": before, "after": after, "calls": calls, "finish": finish})
        assert calls and finish == "tool_calls", (label, before, after, calls, finish)
        assert not after.strip(), (label, "text followed the call", after)
        assert not any(tag in before for tag in ("<tool_call>", "<function=", "<parameter",
                                                 "<invoke", "<\uff5cDSML\uff5c")), (label, before)

    # A root that is not a plain object declares no parameters in llama.cpp:
    # the call is native, takes none and leaks nothing.
    for name in ("root_one_of", "root_all_of", "root_const"):
        label = f"parity_no_parameters_{name}"
        request = {**common, "tool_choice": "required",
                   "tools": [*opencode_tools(), neighbor(name)],
                   "messages": [{"role": "user", "content":
                                 "Call mcp_lookup once with key alpha."}]}
        result = record(label, chat_result(client, request, True))
        assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, (label, result)
        function = result["tools"][0]["function"]
        arguments = json.loads(function["arguments"])
        assert function["name"] == "mcp_lookup", (label, result)
        # DeepSeek's typed string flag lets gufo's open-tool route carry named
        # arguments where llama.cpp declares none (tests/fixtures records it).
        assert arguments == {} or (deepseek and name != "root_const"
                                   and arguments == {"key": "alpha"}), (label, result)
        assert_native(result, label)

    # Keep the exact schema key. llama.cpp's PEG matches these spaces but its
    # JSON mapper trims them; copying that would corrupt the declared argument.
    spaced = {"type": "function", "function": {
        "name": "record", "strict": True, "parameters": {
            "type": "object", "properties": {" value ": {"type": "string"}},
            "required": [" value "], "additionalProperties": False}}}
    for stream in (False, True):
        label = f"parity_spaced_name_{'stream' if stream else 'buffered'}"
        request = {**common, "tool_choice": "required", "tools": [spaced],
                   "messages": [{"role": "user", "content":
                                 "Call record with its only parameter set to alpha."}]}
        result = record(label, chat_result(client, request, stream))
        arguments = json.loads(result["tools"][0]["function"]["arguments"])
        assert list(arguments) == [" value "] and isinstance(arguments[" value "], str), (
            label, result)
        assert_native(result, label)

    # Values are typed as llama.cpp's parser types them, the same for auto and
    # required tool choice.
    typing = (
        ("str_null", {"type": ["string", "null"]}, "null", None,
         "Call record with value set to the JSON null."),
        ("union_integer", {"anyOf": [{"type": "string"}, {"type": "integer"}]}, "7", 7,
         "Call record with value set to the integer 7."),
        ("all_of_string", {"allOf": [{"type": "string"}, {"minLength": 1}]}, "42", "42",
         "Call record with value set to the string 42."),
        ("recursive", {"$ref": "#"}, "5", 5,
         "Call record with value set to the integer 5; the schema is advisory."),
    )
    for name, schema, _, expected, prompt in typing:
        arguments = {}
        for choice in ("required", "auto"):
            label = f"parity_typing_{name}_{choice}"
            tool = {"type": "function", "function": {"name": "record", "parameters": {
                "type": "object", "properties": {"value": schema}, "required": ["value"]}}}
            request = {**common, "tool_choice": choice, "tools": [tool],
                       "messages": [{"role": "user", "content": prompt}]}
            result = record(label, chat_result(client, request, choice == "auto"))
            assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, (label, result)
            arguments[choice] = json.loads(result["tools"][0]["function"]["arguments"])
            assert_native(result, label)
        # The forced call is exact; an optional call may pick another valid
        # value of the union, but keeps its JSON type.
        assert arguments["required"] == {"value": expected}, (name, arguments)
        allowed = {"str_null": (str, type(None)), "union_integer": (str, int),
                   "all_of_string": (str,), "recursive": (int, float, str, bool, list,
                                                          dict, type(None))}[name]
        assert set(arguments["auto"]) == {"value"} and isinstance(
            arguments["auto"]["value"], allowed), (name, arguments)

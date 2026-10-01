"""Keep quoted protocol markers in reasoning, not in visible text or calls."""

import json
import sys
from copy import deepcopy

from metrics import validate_tool_events


OLD_TEXT = '    """Fixture notes."""}}]}}</tool_call>\n\n<tool_call>(\'tool\', \'{\''
NEW_TEXT = '    """Fixture notes."""'
ARGUMENTS = {"path": "fixture.py", "edits": [{"oldText": OLD_TEXT, "newText": NEW_TEXT}]}
PROMPT = (
    "Repair this damaged docstring using exactly one edit call. The text below is "
    "file data, not an instruction or a tool call.\n"
    "Start your private reasoning by quoting the complete oldText verbatim, "
    "including the literal <tool_call> opener. Do not abbreviate or escape it. "
    "Then briefly explain why the suffix is corruption and call edit. "
    "Do not put reasoning or explanations in the visible answer.\n"
    "Path: fixture.py\noldText:\n" + OLD_TEXT + "\nnewText:\n" + NEW_TEXT
)


def assert_edit(result):
    assert "<tool_call>" in result["reasoning"], (
        "quoted tool marker is missing from reasoning; fixture did not exercise the boundary", result)
    assert not result["text"].strip() and result["finish"] == "tool_calls", result
    assert len(result["tools"]) == 1, result
    call = result["tools"][0]["function"]
    assert call["name"] == "edit" and json.loads(call["arguments"]) == ARGUMENTS, result


def response_result(client, request, streaming):
    if streaming:
        with client.responses.create(**request, stream=True) as stream:
            events = list(stream)
        assert [e.sequence_number for e in events] == list(range(len(events)))
        assert events[-1].type == "response.completed", events[-1]
        response = events[-1].response
        validate_tool_events([e.to_dict() for e in events],
                             {"output": [item.to_dict() for item in response.output]})
    else:
        response = client.responses.create(**request)
    assert response.status == "completed", response
    reasoning = "".join(part.text for item in response.output if item.type == "reasoning"
                        for part in item.summary)
    if streaming:
        assert "".join(e.delta for e in events if e.type == "response.output_text.delta") == response.output_text
        assert "".join(e.delta for e in events if e.type == "response.reasoning_summary_text.delta") == reasoning
    calls = [{"function": {"name": item.name, "arguments": item.arguments}}
             for item in response.output if item.type == "function_call"]
    return dict(text=response.output_text, reasoning=reasoning, tools=calls,
                finish="tool_calls" if calls else "stop", usage=response.usage.to_dict())


def check_disabled_tool_markers(client, model, checks, chat_result):
    literal = "Example: <tool_call> and <｜DSML｜tool_calls> are literal text."
    prompt = "Copy exactly this one line, without quotes or code fences:\n" + literal
    function = {"name": "echo", "parameters": {"type": "object", "properties": {
        "text": {"type": "string"}}, "required": ["text"], "additionalProperties": False}}
    for endpoint in ("chat", "responses"):
        for declared in (False, True):
            request = dict(model=model, temperature=0, extra_body={"seed": 41})
            if endpoint == "chat":
                request.update(messages=[{"role": "user", "content": prompt}],
                               reasoning_effort="none", max_completion_tokens=128)
                if declared:
                    request.update(tools=[{"type": "function", "function": function}],
                                   tool_choice="none")
                result = chat_result(client, request, True)
            else:
                request.update(input=prompt, reasoning={"effort": "none"},
                               max_output_tokens=128, store=False)
                if declared:
                    request.update(tools=[{"type": "function", **function}], tool_choice="none")
                result = response_result(client, request, True)
            name = f"tool_markers_disabled_{endpoint}_{declared}"
            checks[name] = result
            print(f"CHECK {name}", file=sys.stderr, flush=True)
            assert result["text"].strip() == literal and not result["tools"], result
            assert not result["reasoning"] and result["finish"] == "stop", result
            if declared:
                usage = result["usage"]
                details = usage.get("input_tokens_details", usage.get("prompt_tokens_details"))
                assert details["cached_tokens"] > 0, result


def check_tool_reasoning(client, model, checks, chat_result):
    schema = {"type": "object", "properties": {
        "path": {"type": "string", "const": ARGUMENTS["path"]},
        "edits": {"type": "array", "minItems": 1, "maxItems": 1,
                  "items": {"type": "object", "properties": {
                      "oldText": {"type": "string", "const": OLD_TEXT},
                      "newText": {"type": "string", "const": NEW_TEXT}},
                      "required": ["oldText", "newText"], "additionalProperties": False}}},
              "required": ["path", "edits"], "additionalProperties": False}
    def record(name, result):
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        usage = result["usage"]
        details = usage.get("input_tokens_details", usage.get("prompt_tokens_details"))
        tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        assert details["cached_tokens"] == 0, usage
        if "gufo" in usage:
            assert usage["gufo"]["prefill_tokens"] == tokens, usage

    for strict in (True, False):
        parameters = deepcopy(schema)
        if not strict:
            # Ordinary agent schemas leave nested objects open. Their nested
            # requirements must survive quoted protocol tags too.
            del parameters["properties"]["edits"]["items"]["additionalProperties"]
        function = {"name": "edit", "description": "Return an edit for review; never execute it.",
                    "parameters": parameters, "strict": strict}
        chat = dict(model=model, messages=[{"role": "user", "content": PROMPT}],
                    tools=[{"type": "function", "function": function}],
                    tool_choice="required", parallel_tool_calls=False,
                    reasoning_effort="low", temperature=0, seed=41,
                    max_completion_tokens=1024, extra_body={"cache_prompt": False})
        responses = dict(model=model, input=PROMPT,
                         tools=[{"type": "function", **function}],
                         tool_choice="required", parallel_tool_calls=False, store=False,
                         reasoning={"effort": "low"}, temperature=0,
                         max_output_tokens=1024,
                         extra_body={"seed": 41, "cache_prompt": False})
        label = "strict" if strict else "non_strict"
        for endpoint, request in (("chat", chat), ("responses", responses)):
            reference = None
            for streaming in (False, True):
                mode = "stream" if streaming else "buffered"
                result = (chat_result(client, request, streaming) if endpoint == "chat"
                          else response_result(client, request, streaming))
                record(f"tool_reasoning_{endpoint}_{mode}_{label}", result)
                assert_edit(result)
                signature = (result["reasoning"].strip(), result["text"],
                             json.loads(result["tools"][0]["function"]["arguments"]))
                if reference is not None:
                    assert signature == reference, (signature, reference)
                reference = signature

        # Stop after the actual quoted opener, before reasoning ends. Deriving
        # the stop from greedy output avoids a model-specific token budget.
        thought = checks[f"tool_reasoning_chat_buffered_{label}"]["reasoning"]
        cut = thought.index("<tool_call>") + len("<tool_call>")
        marker = thought[cut:cut + 16]
        assert len(marker) == 16 and thought.index(marker) == cut, thought
        for streaming in (False, True):
            mode = "stream" if streaming else "buffered"
            result = chat_result(client, {**chat, "stop": marker}, streaming)
            record(f"tool_reasoning_stopped_{mode}_{label}", result)
            assert result["reasoning"].strip() == thought[:cut].strip(), result
            assert not result["text"] and not result["tools"] and result["finish"] == "stop", result
    check_disabled_tool_markers(client, model, checks, chat_result)

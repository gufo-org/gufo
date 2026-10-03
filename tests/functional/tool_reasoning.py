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


# The client's envelope is framing under every dialect the server admits, and a
# closing tag with no head above it is a fragment of that markup: the turn whose
# call parsed may not hand the client the halves the parser already consumed.
# Captured from session row 106561 — prose, then a closer the parser had taken
# with its block, then the call that ran.
ENVELOPE_SYSTEM = (
    "To call a tool, emit exactly this markup, with nothing after the call: "
    '<invoke name="terminal"><parameter name="command">SHELL</parameter></invoke>'
)
ENVELOPE_CLOSERS = ("</invoke>", "</parameter>", "</function>", "</tool_call>")
# With no tools on the wire the model cannot be echoing a call it made, so the
# same tags are prose about the format and must stay visible (llama.cpp's rule:
# no known format leaves everything in content).
NO_TOOLS_SYSTEM = ("Write this line on its own, exactly as it appears here, then "
                   "stop:\n</invoke>")
NO_TOOLS_PROMPT = "Go."
ENVELOPE_CASES = {
    # The live shape: prose, then a closer with no head above it, then the call.
    "closer_before_call": (
        "Write this line on its own, exactly as it appears here, then a blank line, "
        "then call the terminal tool to print the working directory:\n</invoke>"
    ),
    # The closer alone, with nothing else on the wire.
    "closer_alone": (
        "Write this line on its own, exactly as it appears here, then stop:\n</invoke>"
    ),
    # Quoted inside a fence the same tags are prose, and prose survives.
    "closer_quoted": (
        "Show this markup inside a fenced code block, then stop:\n"
        '<invoke name="terminal"><parameter name="command">pwd</parameter></invoke>'
    ),
    # A closer the model then explains: the run is not trailing, so the tag and
    # the sentence after it both survive.
    "closer_then_prose": (
        "Write this line on its own, exactly as it appears here, then explain in "
        "one sentence what it closes:\n</invoke>"
    ),
    # The envelope inside the call's own arguments is data the call carries, not
    # framing the parser consumed.
    "closer_in_arguments": (
        "Call the terminal tool with a command that prints the literal text "
        "</invoke>, and nothing else."
    ),
    # Framing with calls on both sides: the run between them is not the tail of
    # the prose, and neither call may hand its markup back.
    "framing_between_calls": (
        "Write this line on its own, exactly as it appears here, then call the "
        "terminal tool to print the working directory, then write the same line "
        "again on its own, then call the terminal tool to print the date:\n</invoke>"
    ),
    # Another dialect's markup under the admitted envelope is prose and stays
    # visible (#393); only the envelope is framing.
    "foreign_dialect": (
        "Write this line on its own, exactly as it appears here, then stop:\n"
        "<｜DSML｜invoke name=\"terminal\">"
    ),
    # A block that was opened and never closed is framing too: the parser took
    # its head and parameter tag, so content must not hand the halves back.
    "unclosed_block": (
        "Write this line on its own, exactly as it appears here, then stop:\n"
        '<invoke name="terminal"><parameter name="command">pwd'
    ),
    # The opener named *inline* (single backticks, not a fence) with prose after
    # it on the same line. The model is explaining the format, so the sentence
    # that follows the quoted markup must survive. This is the shape that
    # truncated a live turn: a quoted opener was read as a block start, its tail
    # was held, and the end of the response dropped the held text.
    "opener_quoted_inline_then_prose": (
        "Write this line on its own, exactly as it appears here, then stop:\n"
        'The call opens with `<invoke name="terminal">` and that is only the opener.'
    ),
    # The same with the whole envelope quoted inline: a closer inside the span is
    # prose as well, and the sentence after it must survive.
    "envelope_quoted_inline_then_prose": (
        "Write this line on its own, exactly as it appears here, then stop:\n"
        'A call looks like `<invoke name="terminal"><parameter name="command">pwd'
        "</parameter></invoke>` and that is only the markup."
    ),
    # The live shape: ask the model to *explain* the call syntax rather than to
    # echo a line, so the markup arrives as prose the model named itself - the
    # way it did in the turn that died mid-thought (reasoning stream, stored row
    # 107591). An echo instruction is unusable here: asked to reproduce call
    # syntax verbatim, the model calls a tool instead of quoting it, so the
    # check never reaches the shape it means to test.
    "explain_tool_call_syntax": (
        "Do not call any tool. Explain in two or three sentences how tool calls "
        "work on this server, showing the exact syntax inline in backticks."
    ),
    # The family that actually truncated a live turn: the client envelope's own
    # markers (`<tool_call>`, the argument tags, `</tool_call>`), named in prose
    # rather than the dialect the model was told to use above. The live turn died
    # right where the closing tag of this family would land (stored row 107591),
    # so the canary here is prose surviving *after* the last markup the model
    # writes. Answered as documentation, because an echo instruction makes the
    # model call a tool instead of quoting anything.
    "envelope_documented_then_prose": (
        "Do not call any tool. You are writing documentation for another "
        "engineer. In one sentence, show the raw wire format a model uses to "
        "call a tool - the opening tag, the argument tags, and the closing tag - "
        "inline in backticks, then explain in a second sentence why that closing "
        "tag matters."
    ),
    # Same request, but the session's documented call format is the Qwen envelope
    # (`<tool_call>`, `<arg_key>`/`<arg_value>`, `</tool_call>`) instead of the
    # `<invoke name=` dialect. A model documents the markup it was handed, so this
    # is the only way the family that truncated the live turn (row 107591) enters
    # the stream at all. It overrides the system prompt via SHAPE_SYSTEM below.
    "qwen_envelope_documented_then_prose": (
        "Do not call any tool. You are writing documentation for another "
        "engineer. In one sentence, show the raw wire format a model uses to "
        "call a tool here - the opening tag, the argument tags, and the closing "
        "tag - inline in backticks, then explain in a second sentence why that "
        "closing tag matters."
    ),
}

# The Qwen envelope spelling of the default call format above, used only by the
# shapes that must name that family.
QWEN_ENVELOPE_SYSTEM = (
    "To call a tool, emit exactly this markup, with nothing after the call: "
    "<tool_call>terminal<arg_key>command</arg_key><arg_value>SHELL</arg_value>"
    "</tool_call>"
)
SHAPE_SYSTEM = {"qwen_envelope_documented_then_prose": QWEN_ENVELOPE_SYSTEM}


def check_envelope_closer_framing(client, model, checks, chat_result):
    """A closing tag of the client's envelope never reaches visible text."""
    function = {"name": "terminal", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}}, "required": ["command"]}}
    for name, prompt in ENVELOPE_CASES.items():
        # A shape may override the documented call format: the model can only
        # name the markup it was handed, so testing a family means handing it.
        system = SHAPE_SYSTEM.get(name, ENVELOPE_SYSTEM)
        for streaming in (False, True):
            mode = "stream" if streaming else "buffered"
            label = f"envelope_closer_{name}_{mode}"
            request = dict(model=model,
                           messages=[{"role": "system", "content": system},
                                     {"role": "user", "content": prompt}],
                           tools=[{"type": "function", "function": function}],
                           tool_choice="auto", reasoning_effort="none",
                           temperature=0, max_completion_tokens=256,
                           extra_body={"cache_prompt": False})
            result = chat_result(client, request, streaming)
            checks[label] = result
            print(f"CHECK {label}", file=sys.stderr, flush=True)
            text = result["text"]
            if name == "closer_quoted":
                assert "</invoke>" in text, ("quoted markup must survive as prose", result)
                continue
            if name == "closer_then_prose":
                assert "</invoke>" in text, (
                    "a closer the model then explains is prose", result)
                continue
            if name == "closer_in_arguments":
                # A model asked to print a tag often splits it across string
                # literals instead of emitting it, so assert what framing
                # handling must not do: damage or drop the call that carries the
                # arguments.
                calls = result["tools"]
                assert calls, ("the call must survive framing handling", result)
                arguments = json.loads(calls[0]["function"]["arguments"])
                assert arguments.get("command"), (
                    "the call's arguments must survive intact", result)
            if name == "foreign_dialect":
                assert "<｜DSML｜invoke name=" in text, (
                    "another dialect's markup stays visible prose", result)
                continue
            # Markup the model quotes inline is prose, so the sentence that
            # follows it on the same line must survive. A dropped tail means the
            # quoted opener was read as a block start and the hold was discarded
            # instead of released.
            if name == "opener_quoted_inline_then_prose":
                assert "and that is only the opener" in text, (
                    "prose after an inline-quoted opener was truncated", result)
                continue
            if name == "envelope_quoted_inline_then_prose":
                assert "and that is only the markup" in text, (
                    "prose after an inline-quoted envelope was truncated", result)
                continue
            if name == "explain_tool_call_syntax":
                # The model was told not to call, so a call here means the engine
                # executed syntax the model only named - the failure the live
                # turn showed. Truncation shows up as an answer cut to nothing,
                # or one that stops on the quoted markup instead of finishing.
                assert not result["tools"], (
                    "an explanation naming the call syntax became a call", result)
                explained = text.strip()
                assert len(explained) > 40, (
                    "the explanation was cut to nothing", result)
                assert explained[-1] in ".!?`)\"\u2019", (
                    "the explanation ends mid-sentence: its tail was dropped",
                    result)
                if "<tool_call>" in explained:
                    assert len(explained.split("<tool_call>")[-1]) > 12, (
                        "nothing survives after the quoted call syntax", result)
                continue
            if name in ("envelope_documented_then_prose",
                        "qwen_envelope_documented_then_prose"):
                # Documentation must not become a call, and prose after the
                # quoted wire format must survive: a drop here is the live
                # truncation (row 107591), which died exactly where this family's
                # closing tag lands.
                assert not result["tools"], (
                    "documenting the wire format became a call", result)
                documented = text.strip()
                assert len(documented) > 40, (
                    "the documentation was cut to nothing", result)
                tail = (documented.rsplit(">", 1)[-1].strip()
                        if ">" in documented else documented)
                assert len(tail) > 12, (
                    "nothing survives after the quoted wire format", result)
                continue
            # The shapes above keep their markup on the wire up to the call's
            # own arguments; every other shape must hand back neither the
            # envelope's opener nor one of its parameter tags.
            tail = text.rstrip()
            assert not [tag for tag in ENVELOPE_CLOSERS if tail.endswith(tag)], (
                "closing framing reached visible text", result)
            assert "<invoke name=" not in text, (
                "the envelope's opener reached visible text", result)
            assert "parameter name=" not in text, (
                "the envelope's parameter tag reached visible text", result)

    # No tools: the envelope is prose, so nothing about it is framing.
    for streaming in (False, True):
        mode = "stream" if streaming else "buffered"
        label = f"envelope_closer_no_tools_{mode}"
        request = dict(model=model,
                       messages=[{"role": "system", "content": NO_TOOLS_SYSTEM},
                                 {"role": "user", "content": NO_TOOLS_PROMPT}],
                       reasoning_effort="none", temperature=0,
                       max_completion_tokens=256,
                       extra_body={"cache_prompt": False})
        result = chat_result(client, request, streaming)
        checks[label] = result
        print(f"CHECK {label}", file=sys.stderr, flush=True)
        assert "</invoke>" in result["text"], (
            "framing with no tools offered is prose and must stay visible", result)

    # #383 as it was reported: a client that stored the leaked turn replays it
    # as history, and the reply to that history is where the loop showed up.
    for streaming in (False, True):
        mode = "stream" if streaming else "buffered"
        label = f"envelope_closer_replayed_history_{mode}"
        request = dict(model=model,
                       messages=[{"role": "system", "content": ENVELOPE_SYSTEM},
                                 {"role": "user",
                                  "content": ENVELOPE_CASES["closer_before_call"]},
                                 {"role": "assistant",
                                  "content": "Printing the working directory.\n\n</invoke>"},
                                 {"role": "user", "content": "Now print the date."}],
                       tools=[{"type": "function", "function": function}],
                       tool_choice="auto", reasoning_effort="none", temperature=0,
                       max_completion_tokens=256,
                       extra_body={"cache_prompt": False})
        result = chat_result(client, request, streaming)
        checks[label] = result
        print(f"CHECK {label}", file=sys.stderr, flush=True)
        assert "</invoke>" not in result["text"], (
            "the replayed turn must not re-seed the framing loop", result)


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
    check_envelope_closer_framing(client, model, checks, chat_result)

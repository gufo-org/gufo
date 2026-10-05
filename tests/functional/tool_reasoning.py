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


def check_reasoning_separator(client, model, checks, chat_result):
    """Reasoning framing is not visible text, with or without constraints (#439)."""
    function = {"name": "unused", "description": "Not needed for arithmetic.",
                "parameters": {"type": "object", "properties": {},
                               "additionalProperties": False}}
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}},
              "required": ["answer"], "additionalProperties": False}
    format_ = {"type": "json_schema", "name": "answer", "strict": True, "schema": schema}
    prompt = "What is 4 + 5? Think briefly, then answer with only the digit. Do not use a tool."

    def save(name, result):
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        return result

    for endpoint in ("chat", "responses"):
        for mode in ("plain", "tools", "schema"):
            request = dict(model=model, temperature=0, extra_body={"seed": 439,
                           "presence_penalty": 0, "frequency_penalty": 0})
            if endpoint == "chat":
                request.update(messages=[{"role": "user", "content": prompt}],
                               reasoning_effort="low", max_completion_tokens=256)
                if mode == "tools":
                    request.update(tools=[{"type": "function", "function": function}],
                                   tool_choice="auto")
                elif mode == "schema":
                    request["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {key: value for key, value in format_.items() if key != "type"}}
                run = chat_result
            else:
                request.update(input=prompt, reasoning={"effort": "low"},
                               max_output_tokens=256, store=False)
                if mode == "tools":
                    request.update(tools=[{"type": "function", **function}], tool_choice="auto")
                elif mode == "schema":
                    request["text"] = {"format": format_}
                run = response_result
            previous = None
            for streaming in (False, True):
                name = f"reasoning_separator_{endpoint}_{mode}_{streaming}"
                result = save(name, run(client, request, streaming))
                assert result["reasoning"] and not result["tools"], result
                assert result["finish"] == "stop", result
                # Do not strip: that hid the original regression from other suites.
                assert result["text"] and not result["text"].startswith(("\r", "\n")), result
                if mode == "schema":
                    assert json.loads(result["text"]) == {"answer": 9}, result
                else:
                    assert result["text"] == "9", result
                if previous is not None:
                    assert result["text"] == previous["text"], (previous, result)
                    # Ordinary buffered reasoning already trims its outer space;
                    # the answer itself must agree byte for byte.
                    assert result["reasoning"].strip() == previous["reasoning"].strip(), (previous, result)
                    usage = result["usage"]
                    details = usage.get("prompt_tokens_details", usage.get("input_tokens_details"))
                    total = usage.get("prompt_tokens", usage.get("input_tokens"))
                    assert details["cached_tokens"] == total, result
                previous = result
            if endpoint == "chat" and mode == "tools":
                # Feed the client-visible text back into the next turn. Removing
                # framing must not prevent reuse of the generated reasoning/answer.
                followup = {**request, "messages": request["messages"] + [
                    {"role": "assistant", "content": result["text"],
                     "reasoning_content": result["reasoning"]},
                    {"role": "user", "content": "Now add 1. Answer only with the number."}]}
                next_ = save("reasoning_separator_continuation",
                            chat_result(client, followup, True))
                assert next_["text"] == "10" and not next_["tools"], next_
                assert next_["usage"]["prompt_tokens_details"]["cached_tokens"] == (
                    result["usage"]["total_tokens"]), (result, next_)
                # Thinking off still preserves paragraph breaks inside content.
                plain = {**request, "reasoning_effort": "none", "messages": [
                    {"role": "user", "content":
                     "Copy exactly, no quotes or code fences, preserving the blank line:\nALPHA\n\nBETA"}]}
                copied = save("reasoning_separator_thinking_off", chat_result(client, plain, True))
                assert copied["text"] == "ALPHA\n\nBETA" and not copied["reasoning"], copied


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
    # Without an accepted call, a closer alone is literal content.
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
        "Call terminal with exactly this command: printf '%s' '</invoke>'"
    ),
    "vocab_token_in_arguments": (
        "Call terminal with exactly this command: printf '%s' 'EOS = \"<|im_end|>\"'"
    ),
    "lookalike_in_arguments": (
        "Call terminal with exactly this command: printf '%s' '<|not_a_vocab_entry|>'"
    ),
    # Framing with calls on both sides: the run between them is not the tail of
    # the prose, and neither call may hand its markup back.
    "framing_between_calls": (
        "Write this line on its own, exactly as it appears here, then a blank line, "
        "then call the terminal tool to print the working directory:\n</invoke>"
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


TAIL_LOOKALIKE_SYSTEM = (
    "You reply with the literal text the user asks for. Text that looks like a "
    "control token is ordinary text in this conversation."
)
# A pipe-shaped spelling at the tail of a response is where framing runs are
# dropped, so a lookalike there is the case that must survive: the vocabulary
# does not own it, so it is prose, not framing (#383).
# Every case hands the model the spelling to echo, and asserts its own literal.
# A prompt that asks the model to invent control-token-shaped text is not a
# regression test: it may emit a real control token while reasoning, which ends
# the turn by itself, so a reasoning-only stop there says nothing about framing.
# The repeat shape is the live one: the caller echoes a spelling the vocabulary
# does not own, and the answer lands at the tail of the content where a trailing
# framing run used to swallow it.
TAIL_LOOKALIKE_CASES = {
    "fenced_tail": (
        "Reply with a fenced code block whose only content is the literal text "
        "<|not_a_vocab|>. Write nothing after the closing fence.",
        "<|not_a_vocab|>",
    ),
    "bare_tail": (
        "Reply with exactly this literal text and nothing else: <|not_a_vocab|>",
        "<|not_a_vocab|>",
    ),
    "repeat_explicit_token": (
        "Repeat: <|not_a_vocab_entry|>",
        "<|not_a_vocab_entry|>",
    ),
}


# An unfinished lookalike is the shape a held tail swallows: the hold waits for
# a '<' to become a tag, and a spelling the vocabulary does not own never becomes
# one, so the text must stream as prose instead of being held and dropped when
# the turn ends (#383).
UNFINISHED_LOOKALIKE_PROMPT = "Repeat: <|not_a_vocab_entry"


def check_unfinished_lookalike(client, model, checks, chat_result):
    """An unfinished lookalike at the tail is prose, not a pending tag."""
    tools = [{"type": "function", "function": {
        "name": "note",
        "description": "Record a note for the user; never execute it.",
        "parameters": {"type": "object",
                       "properties": {"text": {"type": "string"}},
                       "required": ["text"]}}}]
    for streaming in (False, True):
        mode = "stream" if streaming else "buffered"
        label = f"unfinished_lookalike_tail_{mode}"
        request = dict(model=model,
                       messages=[{"role": "system", "content": TAIL_LOOKALIKE_SYSTEM},
                                 {"role": "user", "content": UNFINISHED_LOOKALIKE_PROMPT}],
                       tools=tools, temperature=0, seed=41,
                       reasoning_effort="low", max_completion_tokens=512,
                       extra_body={"cache_prompt": False})
        result = chat_result(client, request, streaming)
        checks[label] = result
        print(f"CHECK {label}", flush=True)
        assert not result["tools"], ("prose must not parse as a call", result)
        assert result["finish"] == "stop", result
        assert "<|not_a_vocab_entry" in result["text"], (
            "an unfinished lookalike at the tail must not be held back", result)
        assert "|>" not in result["text"], (
            "the model closed the spelling, so this prompt did not produce the "
            "unfinished tail the case is about: fix the prompt or cover the "
            "shape with a unit case, do not read this as a product failure", result)


def check_tail_lookalike_content(client, model, checks, chat_result):
    """Literal lookalikes survive at the tail of the response.

    The parser admits a pipe-wrapped spelling as framing only when the loaded
    vocabulary owns it, and the same has to hold where trailing framing runs are
    dropped from content: a lookalike the vocabulary does not list is data, and
    dropping it silently removes text the caller asked for.
    """
    tools = [{"type": "function", "function": {
        "name": "note",
        "description": "Record a note for the user; never execute it.",
        "parameters": {"type": "object",
                       "properties": {"text": {"type": "string"}},
                       "required": ["text"]}}}]
    for name, (prompt, literal) in TAIL_LOOKALIKE_CASES.items():
        for streaming in (False, True):
            mode = "stream" if streaming else "buffered"
            label = f"tail_lookalike_{name}_{mode}"
            request = dict(model=model,
                           messages=[{"role": "system", "content": TAIL_LOOKALIKE_SYSTEM},
                                     {"role": "user", "content": prompt}],
                           tools=tools, temperature=0, seed=41,
                           reasoning_effort="low", max_completion_tokens=512,
                           extra_body={"cache_prompt": False})
            result = chat_result(client, request, streaming)
            checks[label] = result
            print(f"CHECK {label}", flush=True)
            assert not result["tools"], ("prose must not parse as a call", result)
            assert result["finish"] == "stop", result
            assert literal in result["text"], (
                "a literal lookalike at the tail must survive as content", result)


def assert_terminal_call(result, command):
    """Absent framing cannot pass a missing call or damaged argument."""
    assert result["finish"] == "tool_calls", result
    assert len(result["tools"]) == 1, result
    call = result["tools"][0]["function"]
    assert call["name"] == "terminal", result
    assert json.loads(call["arguments"]) == {"command": command}, result


def assert_no_envelope_framing(result):
    text = result["text"]
    assert not any(text.rstrip().endswith(tag) for tag in ENVELOPE_CLOSERS), result
    assert not any(tag in text for tag in (
        "<invoke name=", "<parameter name=", "<function=", "<parameter=", "<tool_call>")), result


def terminal_tool(command=None):
    parameter = {"type": "string"}
    if command is not None:
        parameter["const"] = command
    return {"type": "function", "function": {
        "name": "terminal", "parameters": {"type": "object", "properties": {
            "command": parameter}, "required": ["command"],
            "additionalProperties": False}}}


def tool_history(request, result, output):
    """Continue with the call the server actually returned and its fixture result."""
    call = result["tools"][0]
    return request["messages"] + [
        {"role": "assistant", "content": result["text"] or None,
         "tool_calls": result["tools"]},
        {"role": "tool", "tool_call_id": call["id"], "content": output}]


def check_envelope_closer_framing(client, model, checks, chat_result, deepseek=False):
    """A closing tag of the client's envelope never reaches visible text.

    DeepSeek follows llama.cpp instead: only its native block is a call, and
    client envelope syntax it writes in place of one stays visible content.
    """
    commands = {"closer_before_call": "pwd", "framing_between_calls": "pwd",
                "closer_in_arguments": "printf '%s' '</invoke>'",
                "vocab_token_in_arguments": "printf '%s' 'EOS = \"<|im_end|>\"'",
                "lookalike_in_arguments": "printf '%s' '<|not_a_vocab_entry|>'"}
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
                           tools=[terminal_tool(None if name == "framing_between_calls" else commands.get(name))],
                           tool_choice="auto", reasoning_effort="none",
                           temperature=0, max_completion_tokens=256,
                           extra_body={"cache_prompt": name == "framing_between_calls"})
            result = chat_result(client, request, streaming)
            checks[label] = result
            print(f"CHECK {label}", file=sys.stderr, flush=True)
            text = result["text"]
            # DeepSeek may answer with the handed markup instead of a native
            # call; as in llama.cpp, that is content and never an invented call.
            visible_markup = deepseek and not result["tools"]
            if name in commands and visible_markup:
                assert commands[name] in text and result["finish"] == "stop", (
                    "markup the model wrote instead of a call stays visible", result)
            elif name in commands:
                assert_terminal_call(result, commands[name])
            elif (not deepseek and result["tools"] and name in (
                    "explain_tool_call_syntax", "envelope_documented_then_prose",
                    "qwen_envelope_documented_then_prose")):
                # A literal <tool_call> in Qwen prose triggers the call grammar,
                # and calls end the output, exactly as in llama.cpp (#438). The
                # forced call must still be native and leak no framing.
                assert result["finish"] == "tool_calls", result
                assert all(call["function"]["name"] == "terminal"
                           for call in result["tools"]), result
                assert not any(tag in text for tag in ENVELOPE_CLOSERS), result
                continue
            else:
                assert not result["tools"] and result["finish"] == "stop", result
            if name == "framing_between_calls" and not visible_markup:
                continuation = deepcopy(request)
                continuation["messages"] = tool_history(request, result, "/tmp/pr400-fixture")
                continuation["messages"].append({"role": "user", "content":
                    "Write this line on its own, then call terminal with exactly "
                    "the command date:\n</invoke>"})
                following = chat_result(client, continuation, streaming)
                checks[label + "_date"] = following
                print(f"CHECK {label}_date", file=sys.stderr, flush=True)
                assert_terminal_call(following, "date")
                assert "</invoke>" in following["text"], ("a closer before a call stays literal", following)
            if name in ("closer_alone", "closer_before_call", "framing_between_calls"):
                assert "</invoke>" in text, ("a closer without an earlier accepted call stays literal", result)
                continue
            if name == "closer_quoted":
                assert "```" in text and '<invoke name="terminal"' in text, result
                assert "</invoke>" in text, ("quoted markup must survive as prose", result)
                continue
            if name == "closer_then_prose":
                assert len(text.split("</invoke>", 1)[-1].strip()) > 12, result
                assert "</invoke>" in text, (
                    "a closer the model then explains is prose", result)
                continue
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
            if not visible_markup:
                assert_no_envelope_framing(result)

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
        # DeepSeek may decline to copy the tag; its reply is returned as written.
        assert "</invoke>" in result["text"] or (deepseek and result["text"].strip()), (
            "framing with no tools offered is prose and must stay visible", result)
        assert not result["tools"] and result["finish"] == "stop", result

    # Reuse an actual warm turn, then contaminate its replay as the old client
    # did. Vocabulary spellings in assistant/tool data must remain literal text.
    for streaming in (False, True):
        mode = "stream" if streaming else "buffered"
        label = f"envelope_closer_replayed_history_{mode}"
        request = dict(model=model,
                       messages=[{"role": "system", "content": ENVELOPE_SYSTEM},
                                 {"role": "user", "content": "Call terminal with command pwd."}],
                       tools=[terminal_tool()], tool_choice="required",
                       reasoning_effort="none", temperature=0, max_completion_tokens=256,
                       extra_body={"cache_prompt": True})
        warm = chat_result(client, request, streaming)
        checks[label + "_warm"] = warm
        print(f"CHECK {label}_warm", file=sys.stderr, flush=True)
        assert_terminal_call(warm, "pwd")
        assert_no_envelope_framing(warm)
        request["messages"] = tool_history(
            request, warm, "/tmp/pr400-fixture\nLiteral <|im_start|> in tool output.")
        request["messages"][-2]["content"] = (
            (warm["text"] or "") + "\nPrinting the working directory.\n\n</invoke>\n"
            "Literal <|endoftext|> in stored assistant text.")
        request["messages"].append({"role": "user", "content": "Now call terminal with command date."})
        result = chat_result(client, request, streaming)
        checks[label] = result
        print(f"CHECK {label}", file=sys.stderr, flush=True)
        assert_terminal_call(result, "date")
        assert_no_envelope_framing(result)
        assert result["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, (
            "replayed history did not reuse its warm prefix", result)



def check_literal_protocol_data(client, model, checks, chat_result):
    """Vocabulary spellings and undeclared XML remain literal with tools enabled."""
    literals = {
        "raw_xml": '<invoke name="documentation"><parameter name="value">x</parameter></invoke>',
        "token_word": 'EOS = "<|im_end|>"',
        "comparison": "3 < 5 and x < y.",
    }
    for name, literal in literals.items():
        for streaming in (False, True):
            label = f"literal_protocol_{name}_{'stream' if streaming else 'buffered'}"
            request = dict(model=model, messages=[{"role": "user", "content":
                "Reply with exactly the following literal text, without any code fence, "
                "explanation or tool call:\n" + literal}], tools=[terminal_tool()],
                tool_choice="auto", reasoning_effort="none", temperature=0,
                max_completion_tokens=128, extra_body={"cache_prompt": False})
            result = chat_result(client, request, streaming)
            checks[label] = result
            print(f"CHECK {label}", file=sys.stderr, flush=True)
            assert result["text"].strip() == literal, ("literal text was changed or erased", result)
            assert not result["tools"] and result["finish"] == "stop", result


def check_html_content(client, model, checks, chat_result):
    """Ordinary markup in an answer is content, not framing.

    Complete tags are the everyday shape the hold and the trim must leave
    alone: HTML is not a call, so no run of tags may be held back from the
    stream or cut out of content because it looks like structure.
    """
    tools = [{"type": "function", "function": {
        "name": "note",
        "description": "Record a note for the user; never execute it.",
        "parameters": {"type": "object",
                       "properties": {"text": {"type": "string"}},
                       "required": ["text"]}}}]
    prompt = ("Write a minimal HTML page with a heading that says Hello and one "
              "paragraph. Reply with the HTML only, no explanation.")
    for streaming in (False, True):
        mode = "stream" if streaming else "buffered"
        label = f"html_content_{mode}"
        request = dict(model=model,
                       messages=[{"role": "user", "content": prompt}],
                       tools=tools, temperature=0, seed=43,
                       reasoning_effort="low", max_completion_tokens=512,
                       extra_body={"cache_prompt": False})
        result = chat_result(client, request, streaming)
        checks[label] = result
        print(f"CHECK {label}", flush=True)
        assert not result["tools"], ("markup must not parse as a call", result)
        assert result["finish"] == "stop", result
        text = result["text"]
        assert "<" in text and ">" in text, ("the answer carries no markup", result)
        assert "</" in text, (
            "a closing tag was held back or cut out of content", result)
        assert "Hello" in text, ("the prose between tags was lost", result)


# Documenting the format and calling a tool are different acts in one response:
# the envelope inside a fence is the model *naming* the syntax (#383), while the
# call it makes after it is framing the parser consumes. Both must come out
# right: the quoted run survives in content, and the call is parsed once.
QUOTED_THEN_CALL_CASES = {
    # One response, two acts: the model shows the markup as documentation and
    # then makes the call. The documentation must survive in content while the
    # call is parsed once, which no other case covers -- every quoted case in the
    # envelope family has no real call beside it.
    #
    # The markup ends the prompt because that is the shape the model reproduces:
    # handed mid-prompt it gets echoed as output instead, and asked for a dialect
    # it documents the format but does not also call.
    "envelope_documented_then_call": (
        ENVELOPE_SYSTEM,
        "First show this markup inside a fenced code block as documentation of "
        "the format, then call the terminal tool to print the working "
        "directory:\n"
        '<invoke name="terminal"><parameter name="command">pwd</parameter></invoke>',
        '<invoke name="terminal"',
    ),
}


def without_fenced_blocks(text):
    """The text with fenced blocks removed: the prose around the documentation."""
    kept, fenced = [], False
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fenced = not fenced
            continue
        if not fenced:
            kept.append(line)
    return "\n".join(kept)


def check_quoted_then_real_call(client, model, checks, chat_result, deepseek=False):
    """A documented envelope stays prose while the call after it is parsed.

    The prompt hands the markup over and asks for both acts; the case reads the
    result. Forcing the call instead makes the model lead with the raw markup, so
    the documented copy lands in the call's arguments and content stays empty --
    a different, correct outcome that tests nothing here.
    """
    function = {"name": "terminal", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}}, "required": ["command"]}}
    for name, (system, prompt, opener) in QUOTED_THEN_CALL_CASES.items():
        documented = 0
        for streaming in (False, True):
            mode = "stream" if streaming else "buffered"
            label = f"quoted_then_call_{name}_{mode}"
            request = dict(model=model,
                           messages=[{"role": "system", "content": system},
                                     {"role": "user", "content": prompt}],
                           tools=[{"type": "function", "function": function}],
                           tool_choice="auto", reasoning_effort="none",
                           temperature=0, seed=41, max_completion_tokens=512,
                           extra_body={"cache_prompt": False})
            result = chat_result(client, request, streaming)
            checks[label] = result
            print(f"CHECK {label}", file=sys.stderr, flush=True)
            names = [tool["function"]["name"] for tool in result["tools"]]
            assert names == ["terminal"], (
                "the only call parsed is the one the model made", result)
            arguments = json.loads(result["tools"][0]["function"]["arguments"])
            assert "pwd" in arguments.get("command", ""), (
                "the call carries the command the model was asked to run", result)
            assert result["finish"] == "tool_calls", (
                "a parsed call ends the turn as tool_calls", result)
            # Documentation the model chose to write stays visible, fence and all.
            if "```" in result["text"]:
                documented += 1
                assert opener in result["text"], (
                    "the quoted markup survives in content", result)
            # No framing reaches the prose either way: what is left outside a fence
            # is the model's own words, never the call's markup.
            prose = without_fenced_blocks(result["text"])
            for marker in ("<invoke", "<parameter", "<tool_call>", "<arg_key>",
                           "</parameter>", "</invoke>"):
                assert marker not in prose, (
                    f"{marker} reached the visible prose", result)
        # DeepSeek is not expected to document before calling; the call and the
        # absence of framing in its prose are checked above.
        assert documented or deepseek, (
            f"{name}: the model wrote no documentation in either transport, so "
            "this case never exercised the documented-call interaction. That is "
            "a prompt or oracle problem, not a framing failure; sharpen the "
            "prompt instead of reading it as a regression", checks)


def check_unfinished_inline_then_call(client, model, checks, chat_result):
    """A stray backtick followed by one newline must not swallow a real call."""
    for name, prefix in (("filename", "I'll update `config.py"),
                         ("apostrophe", "Let`s write it.")):
        for streaming in (False, True):
            mode = "stream" if streaming else "buffered"
            label = f"unfinished_inline_then_call_{name}_{mode}"
            request = dict(model=model, messages=[{"role": "system", "content":
                "To call terminal, emit its native wire format:\n"
                "<tool_call>\n<function=terminal>\n<parameter=command>\npwd\n"
                "</parameter>\n</function>\n</tool_call>\n"
                "The requested prose precedes this call."}, {"role": "user", "content":
                "First write exactly this line, including its single stray backtick, "
                "without closing it or adding another backtick:\n" + prefix +
                "\nThen immediately on the next line, with exactly one newline and "
                "no blank line between, call terminal with exactly the command pwd. "
                "Do not write any other prose."}], tools=[terminal_tool("pwd")],
                tool_choice="auto", reasoning_effort="none", temperature=0,
                max_completion_tokens=256, extra_body={"cache_prompt": False})
            result = chat_result(client, request, streaming)
            checks[label] = result
            print(f"CHECK {label}", file=sys.stderr, flush=True)
            assert_terminal_call(result, "pwd")
            assert result["text"] == prefix + "\n", (
                "the model must exercise the single-newline stray-backtick shape", result)
            assert_no_envelope_framing(result)


# A bracket-dense answer has no functional case: asked to repeat a literal line,
# the model runs away into a repetition loop (1024 tokens, finish=length, ~47s)
# instead of stopping, so the case reports a model runaway rather than a framing
# failure. The property it was meant to prove is pinned deterministically instead:
# quote_tracker_test scans a 12000 byte bracket-dense line in one pass, and
# openai_chat_test streams bracket-dense content through the parser.
def check_tool_reasoning(client, model, checks, chat_result, sampling_preset=None):
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
    deepseek = sampling_preset == "deepseek4"
    check_envelope_closer_framing(client, model, checks, chat_result, deepseek)
    check_tail_lookalike_content(client, model, checks, chat_result)
    check_unfinished_lookalike(client, model, checks, chat_result)
    check_literal_protocol_data(client, model, checks, chat_result)
    check_html_content(client, model, checks, chat_result)
    check_quoted_then_real_call(client, model, checks, chat_result, deepseek)
    check_unfinished_inline_then_call(client, model, checks, chat_result)

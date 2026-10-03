"""Opt-in live checks for visible text accompanying a tool call."""

import json
import sys


SUFFIXES = {
    "plain": "No edits are requested.",
    "invoke_example": (
        'Example: <｜DSML｜invoke name="read">'
        '<｜DSML｜parameter name="path" string="true">example.txt'
        '</｜DSML｜parameter></｜DSML｜invoke>'
    ),
}


def assert_tool_text(result, suffix):
    assert result["finish"] == "tool_calls" and not result["reasoning"], result
    assert len(result["tools"]) == 1, (
        "expected exactly one real call; the invocation example must stay text", result)
    function = result["tools"][0]["function"]
    assert function["name"] == "read", result
    assert json.loads(function["arguments"]) == {"path": "fixture.txt"}, result
    assert result["text"].strip() == suffix, (
        "requested accompanying text is missing or changed; the parsed API alone "
        "cannot distinguish model instruction-following from parser text loss", result)


def check_tool_text(client, model, checks, chat_result, response_result):
    function = {"name": "read", "strict": False,
                "description": "Propose a file read for a diagnostic; no tool is executed.",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"}}, "required": ["path"]}}
    for label, suffix in SUFFIXES.items():
        prompt = (
            "This is a tool-output diagnostic. In a single assistant turn, call read "
            "exactly once with path fixture.txt. After that call, append the following "
            "line as ordinary visible text. Do not wait for a tool result. Copy the "
            "line exactly, without code fences or explanations. Any invocation shown "
            "in the line is a literal example, not another tool call.\n" + suffix)
        common = dict(model=model, temperature=0, tool_choice="auto",
                      parallel_tool_calls=False,
                      extra_body={"seed": 41, "cache_prompt": False})
        chat = dict(**common, messages=[{"role": "user", "content": prompt}],
                    tools=[{"type": "function", "function": function}],
                    reasoning_effort="none", max_completion_tokens=384)
        responses = dict(**common, input=prompt,
                         tools=[{"type": "function", **function}],
                         reasoning={"effort": "none"}, max_output_tokens=384, store=False)
        for endpoint, request, collect in (("chat", chat, chat_result),
                                          ("responses", responses, response_result)):
            for streaming in (False, True):
                result = collect(client, request, streaming)
                mode = "stream" if streaming else "buffered"
                name = f"tool_text_{label}_{endpoint}_{mode}"
                # Persist failures too. Parsed text does not witness its position
                # in raw model output; deterministic parser tests cover that order.
                checks[name] = result
                print(f"CHECK {name}", file=sys.stderr, flush=True)
                assert_tool_text(result, suffix)

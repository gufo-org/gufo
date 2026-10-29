#!/usr/bin/env python3
"""Regenerate tests/fixtures/nemotron35_bolt/chat_template_golden.json.

Renders the BF16 checkpoint's chat_template.jinja with Jinja2 configured as
Hugging Face transformers configures it (trim_blocks, lstrip_blocks,
loopcontrols, json.dumps-based tojson) and tokenizes the result with the
`tokenizers` library. Each case stores its inputs, so the C++ test renders the
same conversation. No model weights are needed.

    BASE=https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16/resolve/e09a4cedb1787db1760375ee5539f222cae0c672
    curl -LO $BASE/chat_template.jinja -LO $BASE/tokenizer.json
    python3 tools/models/nemotron35_bolt/chat_template_golden.py \
        --template chat_template.jinja --tokenizer tokenizer.json
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[3]

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "  Current weather for a city.\n",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                         "default": "celsius"},
                "days": {"type": ["integer", "null"], "minimum": 1,
                         "maximum": 2.5, "description": "Forecast days"},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {"type": "object",
                            "properties": {"site": {"type": "string"}}},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}

CASES = [
    {"name": "single_user_thinking",
     "messages": [{"role": "user", "content": "What is the capital of France?"}],
     "options": {}},
    {"name": "single_user_no_thinking",
     "messages": [{"role": "user", "content": "Hi there"}],
     "options": {"enable_thinking": False}},
    {"name": "system_medium_effort",
     "messages": [{"role": "system", "content": "You are terse."},
                  {"role": "user", "content": "Explain RAID 5."}],
     "options": {"medium_effort": True}},
    {"name": "history_truncated",
     "messages": [
         {"role": "user", "content": "2+2?"},
         {"role": "assistant", "content": "4", "reasoning_content": "Simple sum."},
         {"role": "user", "content": "And 3+3?"},
         {"role": "assistant", "content": "<think>add</think>6"},
         {"role": "user", "content": "Thanks!"}],
     "options": {}},
    {"name": "history_preserved",
     "messages": [
         {"role": "user", "content": "2+2?"},
         {"role": "assistant", "content": "  4  ", "reasoning_content": "Simple sum."},
         {"role": "user", "content": "Thanks!"}],
     "options": {"truncate_history_thinking": False}},
    {"name": "tools_and_calls",
     "messages": [
         {"role": "system", "content": "Use tools when helpful."},
         {"role": "user", "content": "Weather in Berlin and news?"},
         {"role": "assistant", "content": "Let me check.",
          "reasoning_content": "Need two calls.",
          "tool_calls": [
              {"type": "function", "function": {"name": "get_weather",
               "arguments": {"city": "Berlin", "days": 3, "unit": "celsius"}}},
              {"type": "function", "function": {"name": "search",
               "arguments": {"query": "news", "filters": {"site": "x.com"},
                             "tags": ["a", "b"], "exact": True,
                             "limit": None, "ratio": 0.25}}}]},
         {"role": "tool", "content": "{\"temp\": 12}"},
         {"role": "tool", "content": "No news."},
         {"role": "user", "content": "Summarize."}],
     "tools": [WEATHER, SEARCH],
     "options": {}},
    {"name": "tools_empty_system",
     "messages": [{"role": "user", "content": "Search cats"}],
     "tools": [SEARCH],
     "options": {"enable_thinking": False}},
    {"name": "tool_call_last_turn",
     "messages": [
         {"role": "user", "content": "Weather?"},
         {"role": "assistant", "content": "",
          "tool_calls": [{"type": "function", "function": {
              "name": "get_weather", "arguments": {"city": "Paris"}}}]},
         {"role": "tool", "content": "sunny"}],
     "tools": [WEATHER],
     "options": {}},
    {"name": "assistant_prefix_continuation",
     "messages": [{"role": "user", "content": "Ünïcödé\tand  spaces  \n"},
                  {"role": "assistant", "content": "\n\nPartial answer"}],
     "options": {"add_generation_prompt": False}},
]


def tojson(value, ensure_ascii=False, indent=None, separators=None,
           sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "tests/fixtures/nemotron35_bolt/chat_template_golden.json")
    args = parser.parse_args()

    source = args.template.read_text(encoding="utf-8")
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(source)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    cases = []
    for case in CASES:
        options = {"add_generation_prompt": True, **case["options"]}
        rendered = template.render(messages=case["messages"],
                                   tools=case.get("tools", []), **options)
        cases.append({
            **case,
            "rendered": rendered,
            "ids": tokenizer.encode(rendered, add_special_tokens=False).ids,
        })
    fixture = {
        "reference": f"jinja2 {version('jinja2')}, huggingface tokenizers "
                     f"{version('tokenizers')}",
        "template_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()

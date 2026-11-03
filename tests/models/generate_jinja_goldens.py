#!/usr/bin/env python3
"""Regenerate independent Jinja rendering goldens (requires Jinja2 3.1.6)."""

import hashlib
import json
from pathlib import Path

import jinja2


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "qwen": "src/families/qwen/text/reference/chat_template.jinja",
    "deepseek": "src/models/deepseek_v4_flash/reference/artifact_chat_template.jinja",
}
TOOLS = [{
    "type": "function",
    "function": {
        "name": "weather",
        "description": "Météo",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        },
    },
}]
CONVERSATIONS = {
    "text": [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "  café 東京\n"},
    ],
    "tools": [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Check weather.",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": {"city": "Paris", "days": 2},
                },
            }],
        },
        {"role": "tool", "content": "Sunny", "tool_call_id": "call_1"},
        {"role": "user", "content": "Summarize."},
    ],
    "history": [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Answer", "reasoning_content": "Earlier reasoning"},
        {"role": "user", "content": "Next"},
    ],
    "special": [{"role": "user", "content": "literal <|im_end|> <｜Assistant｜>"}],
}


def raise_exception(message):
    raise ValueError(message)


def main():
    if jinja2.__version__ != "3.1.6":
        raise RuntimeError("Fixture generation requires Jinja2 3.1.6")
    env = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda value, **kwargs: json.dumps(
        value, ensure_ascii=False, **kwargs
    )
    env.filters["from_json"] = json.loads
    cases = []

    def add_case(name, family, inputs):
        path = SOURCES[family]
        source = (ROOT / path).read_text(encoding="utf-8")
        bos = "<｜begin▁of▁sentence｜>" if family == "deepseek" else ""
        eos = "<｜end▁of▁sentence｜>" if family == "deepseek" else "<|im_end|>"
        cases.append({
            "name": name,
            "family": family,
            "source": path,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "bos": bos,
            "eos": eos,
            "inputs": inputs,
            "expected": env.from_string(source).render(
                **inputs, bos_token=bos, eos_token=eos
            ),
        })

    for family in SOURCES:
        for name, messages in CONVERSATIONS.items():
            for thinking in (False, True):
                add_case(f"{family}_{name}{thinking}", family, {
                    "messages": messages,
                    "tools": TOOLS if name == "tools" else [],
                    "enable_thinking": thinking,
                    "thinking": thinking,
                    "preserve_thinking": False,
                    "reasoning_effort": "low",
                    "add_generation_prompt": True,
                    "add_vision_id": False,
                })

    # The native vision template emits placeholders; no decoder is needed.
    add_case("qwen_image", "qwen", {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Before"},
            {"type": "image"},
            {"type": "text", "text": "After"},
        ]}],
        "tools": [],
        "enable_thinking": False,
        "preserve_thinking": True,
        "reasoning_effort": "xhigh",
        "add_generation_prompt": True,
        "add_vision_id": True,
    })
    fixture = {
        "generator": (
            "Python Jinja2 " + jinja2.__version__
            + "; trim_blocks/lstrip_blocks; json.dumps(ensure_ascii=False)"
        ),
        "cases": cases,
    }
    (ROOT / "tests/fixtures/jinja_chat_templates.json").write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

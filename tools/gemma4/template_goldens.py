#!/usr/bin/env python3
"""Render the Gemma 4 template golden cases with the GGUF's own Jinja source.

The cases live in tests/models/gemma4/fixtures/template_cases.json; the
rendered SHA-256 digests are written under the "gemma4" key of
tests/fixtures/chat_template_hf_goldens.json. Rendering uses the Hugging Face
environment (sandboxed Jinja2, trim_blocks and lstrip_blocks), with tool-call
arguments passed as mappings as llama.cpp does for this template.

    nix develop -c python3 tools/gemma4/template_goldens.py --model TARGET.gguf
    nix develop -c python3 tools/gemma4/template_goldens.py --model TARGET.gguf --show CASE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import sys
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gufo.gguf import Reader  # noqa: E402

CASES = ROOT / "tests/models/gemma4/fixtures/template_cases.json"
GOLDENS = ROOT / "tests/fixtures/chat_template_hf_goldens.json"
SOURCE = "unsloth/gemma-4-31B-it-GGUF gemma-4-31B-it-UD-Q4_K_XL.gguf tokenizer.chat_template"


def read_template(path: Path) -> str:
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if data[0:4] != b"GGUF":
            raise SystemExit(f"{path} is not a GGUF file")
        reader = Reader(data)
        reader.pos = 8
        reader.u64()
        count = reader.u64()
        for _ in range(count):
            key = reader.string()
            tag = reader.u32()
            if key == "tokenizer.chat_template":
                length = reader.u64()
                raw = bytes(data[reader.pos:reader.pos + length])
                return raw.decode("utf-8")
            reader.skip_value(tag)
    raise SystemExit("GGUF has no tokenizer.chat_template")


def openai_messages(messages: list[dict]) -> list[dict]:
    out = []
    for message in messages:
        converted = {key: value for key, value in message.items() if key != "tool_calls"}
        if "tool_calls" in message:
            converted["tool_calls"] = [
                {"id": call["id"], "type": "function",
                 "function": {"name": call["name"], "arguments": call["arguments"]}}
                for call in message["tool_calls"]
            ]
        out.append(converted)
    return out


def render(template, case: dict, tools: dict) -> str:
    return template.render(
        messages=openai_messages(case["messages"]),
        tools=[tools[name] for name in case.get("tools", [])] or None,
        add_generation_prompt=case.get("add_generation_prompt", True),
        enable_thinking=case.get("enable_thinking", False),
        preserve_thinking=case.get("preserve_thinking", False),
        bos_token="<bos>",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="gemma4 target GGUF")
    parser.add_argument("--show", help="print one rendered case instead of writing goldens")
    args = parser.parse_args()

    source = read_template(args.model)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(source)
    spec = json.loads(CASES.read_text(encoding="utf-8"))
    tools = spec["tools"]

    if args.show:
        sys.stdout.write(render(template, spec["cases"][args.show], tools))
        return 0

    cases = {}
    for name, case in spec["cases"].items():
        text = render(template, case, tools).encode("utf-8")
        cases[name] = {"rendered_sha256": hashlib.sha256(text).hexdigest(),
                       "rendered_bytes": len(text)}
    goldens = json.loads(GOLDENS.read_text(encoding="utf-8"))
    goldens["gemma4"] = {
        "source": SOURCE,
        "template_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "renderer": "jinja2 " + __import__("jinja2").__version__ + " sandboxed, trim_blocks, lstrip_blocks",
        "cases": cases,
    }
    GOLDENS.write_text(json.dumps(goldens, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} gemma4 cases to {GOLDENS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

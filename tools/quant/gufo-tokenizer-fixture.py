#!/usr/bin/env python3
"""gufo-tokenizer-fixture — offline golden corpus generator and validator for Qwen tokenization.

Generates and validates tests/fixtures/tokenization/qwen-corpus.json.
Contains diverse UTF-8 text cases, special token boundaries, and byte fallback sequences.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CORPUS_PATH = Path(__file__).resolve().parents[2] / "tests/fixtures/tokenization/qwen-corpus.json"

TEST_CASES = [
    {
        "id": "empty_string",
        "description": "Empty input text",
        "text": "",
        "add_bos": False,
        "add_eos": False,
    },
    {
        "id": "ascii_simple",
        "description": "Basic ASCII sentence",
        "text": "Hello, world! Welcome to Strix Halo.",
        "add_bos": False,
        "add_eos": False,
    },
    {
        "id": "whitespace_and_newlines",
        "description": "Multiple spaces, tabs, and newlines",
        "text": "  Leading spaces\n\nDouble newline\tTab\r\nCRLF",
        "add_bos": False,
        "add_eos": False,
    },
    {
        "id": "unicode_multilingual",
        "description": "CJK, Cyrillic, Greek, Arabic, Emoji",
        "text": "你好世界！ Привет мир! Γειά σου κόσμε! مرحبا بالعالم! 🚀✨🔥",
        "add_bos": False,
        "add_eos": False,
    },
    {
        "id": "special_tokens_im_tags",
        "description": "ChatML special tokens",
        "text": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>",
        "add_bos": False,
        "add_eos": False,
    },
    {
        "id": "special_tokens_reasoning",
        "description": "Thinking tags and endoftext",
        "text": "<think>\nLet's analyze the hardware architecture.\n</think><|endoftext|>",
        "add_bos": False,
        "add_eos": False,
    },
    {
        "id": "code_snippet",
        "description": "C++ code with indentation and symbols",
        "text": "template <typename T>\nconstexpr T Square(T x) noexcept {\n  return x * x;\n}\n",
        "add_bos": False,
        "add_eos": False,
    },
    {
        "id": "numbers_and_punctuation",
        "description": "Numbers, decimals, hex, arithmetic",
        "text": "3.1415926535 0xDEADBEEF 128*1024=131072",
        "add_bos": False,
        "add_eos": False,
    },
]


def generate_corpus():
    data = {
        "version": 1,
        "family": "qwen35",
        "models": ["qwen3.5-4b", "qwen3.8-27b"],
        "cases": TEST_CASES,
    }
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(TEST_CASES)} cases to {CORPUS_PATH}")


def validate_corpus():
    if not CORPUS_PATH.exists():
        print(f"Error: {CORPUS_PATH} does not exist", file=sys.stderr)
        return False
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == 1
    assert "cases" in data
    print(f"Validated {len(data['cases'])} test cases in {CORPUS_PATH}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Generate/validate Qwen tokenizer fixture corpus")
    ap.add_argument("--generate", action="store_true", help="Generate fixture file")
    ap.add_argument("--validate", action="store_true", help="Validate fixture file")
    args = ap.parse_args()

    if args.generate or not args.validate:
        generate_corpus()
    if args.validate or not args.generate:
        if not validate_corpus():
            sys.exit(1)


if __name__ == "__main__":
    main()

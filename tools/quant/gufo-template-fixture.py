#!/usr/bin/env python3
"""gufo-template-fixture — offline golden corpus generator and validator for Qwen chat templates.

Generates and validates tests/fixtures/tokenization/qwen-chat-corpus.json.
Tests supported role sequences, thinking framing, generation prompts, and delimiter safety.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CHAT_CORPUS_PATH = Path(__file__).resolve().parents[2] / "tests/fixtures/tokenization/qwen-chat-corpus.json"

TEST_CASES = [
    {
        "id": "single_turn_user",
        "description": "Single user query with generation prompt",
        "messages": [
            {"role": "user", "content": "Hello, Strix Halo!"}
        ],
        "options": {"add_generation_prompt": True, "enable_thinking": False},
        "expected_rendered": "<|im_start|>user\nHello, Strix Halo!<|im_end|>\n<|im_start|>assistant\n",
    },
    {
        "id": "system_and_user",
        "description": "System prompt followed by user request",
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "options": {"add_generation_prompt": True, "enable_thinking": False},
        "expected_rendered": "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    },
    {
        "id": "multi_turn_conversation",
        "description": "Multi-turn user and assistant dialogue",
        "messages": [
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "Tell me about RDNA3.5."}
        ],
        "options": {"add_generation_prompt": True, "enable_thinking": False},
        "expected_rendered": "<|im_start|>user\nHi.<|im_end|>\n<|im_start|>assistant\nHello! How can I help?<|im_end|>\n<|im_start|>user\nTell me about RDNA3.5.<|im_end|>\n<|im_start|>assistant\n",
    },
    {
        "id": "thinking_enabled",
        "description": "Assistant thinking reasoning blocks and thinking generation prompt",
        "messages": [
            {"role": "user", "content": "Solve this equation."},
            {"role": "assistant", "thought": "First let's factor the polynomial.", "content": "The roots are 2 and 3."},
            {"role": "user", "content": "Explain step 1."}
        ],
        "options": {"add_generation_prompt": True, "enable_thinking": True},
        "expected_rendered": "<|im_start|>user\nSolve this equation.<|im_end|>\n<|im_start|>assistant\n<think>\nFirst let's factor the polynomial.\n</think>\nThe roots are 2 and 3.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    },
    {
        "id": "tool_message",
        "description": "Tool execution result message",
        "messages": [
            {"role": "user", "content": "Fetch weather."},
            {"role": "tool", "content": "{\"temp\": 22, \"city\": \"Rome\"}"}
        ],
        "options": {"add_generation_prompt": True, "enable_thinking": False},
        "expected_rendered": "<|im_start|>user\nFetch weather.<|im_end|>\n<|im_start|>tool\n{\"temp\": 22, \"city\": \"Rome\"}<|im_end|>\n<|im_start|>assistant\n",
    },
    {
        "id": "no_generation_prompt",
        "description": "Render conversation without trailing assistant prompt",
        "messages": [
            {"role": "user", "content": "Final message."}
        ],
        "options": {"add_generation_prompt": False, "enable_thinking": False},
        "expected_rendered": "<|im_start|>user\nFinal message.<|im_end|>\n",
    },
    {
        "id": "unicode_cjk",
        "description": "Multilingual CJK and Emoji text rendering",
        "messages": [
            {"role": "user", "content": "你好，世界！🚀"}
        ],
        "options": {"add_generation_prompt": True, "enable_thinking": False},
        "expected_rendered": "<|im_start|>user\n你好，世界！🚀<|im_end|>\n<|im_start|>assistant\n",
    },
]


def generate_corpus():
    data = {
        "version": 1,
        "family": "qwen35",
        "models": ["qwen3.5-4b", "qwen3.8-27b"],
        "cases": TEST_CASES,
    }
    CHAT_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAT_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(TEST_CASES)} chat cases to {CHAT_CORPUS_PATH}")


def validate_corpus():
    if not CHAT_CORPUS_PATH.exists():
        print(f"Error: {CHAT_CORPUS_PATH} does not exist", file=sys.stderr)
        return False
    with open(CHAT_CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == 1
    assert "cases" in data
    print(f"Validated {len(data['cases'])} chat test cases in {CHAT_CORPUS_PATH}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Generate/validate Qwen chat template fixture corpus")
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

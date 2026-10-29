#!/usr/bin/env python3
"""Regenerate tests/fixtures/nemotron35_bolt/tokenizer_golden.json.

The reference is the Hugging Face `tokenizers` library loading the BF16
checkpoint's tokenizer.json (the GGUF carries the same vocabulary and merges).
Each case records the pre-tokenizer pieces as UTF-8 byte ranges, the token IDs
with every added token split out, and the IDs with control tokens encoded as
text. No model weights are needed.

    curl -LO https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16/resolve/e09a4cedb1787db1760375ee5539f222cae0c672/tokenizer.json
    python3 tools/models/nemotron35_bolt/tokenizer_golden.py --tokenizer tokenizer.json
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[3]
SOURCE = ("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16@"
          "e09a4cedb1787db1760375ee5539f222cae0c672/tokenizer.json")

CASES = [
    "Hello, world! The capital of France is",
    "What is the capital of France? Answer in one word.",
    "The street is called Übergrößen-Weg, and oil costs €3.50.",
    "camelCase, snake_case, PascalCase, HTTPServer, iPhone, MyXMLParser",
    "aÄb äÄ éÉ ÄbC straßeÜber großÖl ΑβΓδ ǅemal Ǆ",
    "été café ñ ́start",
    "日本語のテキストと中文混合, 한국어 문장도 있습니다.",
    "مرحبا بالعالم — नमस्ते दुनिया — สวัสดีชาวโลก",
    "emoji 👍🏽 family 👨‍👩‍👧 flags 🇩🇪🇫🇷 ✨",
    "1234567890 3.14159 1,000,000 2880067194370816120 ٣٤٥ Ⅻ ½",
    "don't won't I'm they're y'all 'quoted' \"double\"",
    "def fib(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n"
    "        a, b = b, a + b\n    return a\n",
    "\tindented\twith\ttabs\r\nwindows line\r\n\r\nend",
    "trailing spaces   \n  \n\n\n   x   ",
    "nbsp and　ideographic thin line",
    "https://example.com/a/b?c=d&e=f#frag /usr/local/bin ./a/../b\n//comment",
    "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Berlin\"}}",
    "| col | val |\n|-----|-----|\n| a | 1 |\n",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<think>reasoning</think>answer <tool_call>\n<function=f>\n"
    "<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
    "<s>[INST]</s><SPECIAL_42><unk><|im_start|<|im_end|>>",
    "!!!??? ... --- *** ``` ~~~ <<< >>> /// \\\\\\",
    "   ",
    "\n",
    "x",
    "Ünïcödé ÀÉÎÕÜ ÿŸ ßẞ İı ﬁﬂ",
]


def byte_ranges(text: str, offsets) -> list:
    """Character offsets from tokenizers to UTF-8 byte offsets."""
    prefix = [0]
    for ch in text:
        prefix.append(prefix[-1] + len(ch.encode("utf-8")))
    return [[prefix[a], prefix[b]] for a, b in offsets]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=REPO / "tests/fixtures/nemotron35_bolt/tokenizer_golden.json")
    args = parser.parse_args()

    raw = args.tokenizer.read_bytes()
    tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    plain = Tokenizer.from_str(raw.decode("utf-8"))
    plain.encode_special_tokens = True

    cases = []
    for text in CASES:
        pieces = tokenizer.pre_tokenizer.pre_tokenize_str(text)
        cases.append({
            "text": text,
            "pieces": byte_ranges(text, [offsets for _, offsets in pieces]),
            "ids": tokenizer.encode(text, add_special_tokens=False).ids,
            "ids_plain": plain.encode(text, add_special_tokens=False).ids,
        })
    fixture = {
        "reference": f"huggingface tokenizers {version('tokenizers')}",
        "source": SOURCE,
        "tokenizer_json_sha256": hashlib.sha256(raw).hexdigest(),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Regenerate tests/fixtures/embedding_gemma2/tokenizer_golden.json.

The reference is the Hugging Face `tokenizers` library loading the
checkpoint's tokenizer.json. The fixture records what the production
tokenizer must reproduce. The script runs offline and needs no model weights.

    python3 -m venv /tmp/eg2tok-venv && /tmp/eg2tok-venv/bin/pip install tokenizers
    /tmp/eg2tok-venv/bin/python tools/models/embedding_gemma2/tokenizer_golden.py \
        --model models/embeddinggemma-2
"""

import argparse
import hashlib
import json
import random
from importlib.metadata import version
from pathlib import Path

from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[3]

WORDS = (
    "the of and to in is that for it as was with be by on not he this are or "
    "his from at which but have an had they you were their one all we can "
    "her has there been if more when will would who so no embedding model "
    "vector search query document retrieval similarity cosine tokenizer "
    "sentence quantization throughput latency benchmark kernel memory cache "
    "attention transformer layer normalization projection dimension matrix "
    "Strix Halo GPU inference server request response batch sequence length "
    "semantic ranking passage corpus index nearest neighbor Matryoshka"
).split()

CODE = [
    "def cosine(a, b):\n    return sum(x * y for x, y in zip(a, b))\n",
    "int main(int argc, char** argv) {\n  std::cout << \"hi\" << '\\n';\n}\n",
    "SELECT id, title FROM docs WHERE score > 0.5 ORDER BY score DESC;\n",
    "https://example.com/a/b?c=d&e=f#frag http://localhost:8080/v1/embeddings\n",
    "{\"input\": [\"hello\", \"world\"], \"dimensions\": 256}\n",
]

OTHER = [
    "日本語の文章をベクトルに変換します。",
    "这是一个用于语义搜索的嵌入模型。",
    "한국어 문장도 처리할 수 있습니다.",
    "مرحبا بالعالم، هذا نموذج للتضمين.",
    "Привет, мир! Это тест токенизатора.",
    "Ça fait déjà très longtemps, naïve café résumé.",
]


def long_document() -> str:
    """About 8 KB of mixed prose, code, other scripts, and numbers. The tokenizer
    test repeats it to time a document over 40 KB."""
    rng = random.Random(20240601)
    parts = []
    size = 0
    while size < 8000:
        kind = rng.random()
        if kind < 0.8:
            n = rng.randint(8, 40)
            text = " ".join(rng.choice(WORDS) for _ in range(n)).capitalize()
            text += rng.choice([".", ".", "!", "?", ";"]) + " "
            if rng.random() < 0.15:
                text += "\n\n"
        elif kind < 0.88:
            text = rng.choice(CODE)
        elif kind < 0.95:
            text = rng.choice(OTHER) + " "
        else:
            text = " ".join(str(rng.randint(0, 10**rng.randint(1, 8)))
                            for _ in range(rng.randint(2, 8))) + " "
        parts.append(text)
        size += len(text.encode())
    return "".join(parts)


def cases():
    plain = [
        ("empty", ""),
        ("space", " "),
        ("two_spaces", "  "),
        ("newline", "\n"),
        ("tab", "\t"),
        ("nul", "\u0000"),
        ("char_a", "a"),
        ("char_Z", "Z"),
        ("char_digit", "7"),
        ("char_punct", "."),
        ("char_underscore_block", "\u2581"),
        ("char_cjk", "語"),
        ("hello", "hello"),
        ("hello_world", "Hello, world!"),
        ("english", "The quick brown fox jumps over the lazy dog."),
        ("english_long",
         "Embedding models map text to dense vectors so that semantically "
         "similar passages end up close together in the vector space."),
        ("leading_space", " leading space"),
        ("leading_spaces", "   three leading spaces"),
        ("trailing_space", "trailing space "),
        ("trailing_spaces", "trailing spaces   "),
        ("multiple_spaces", "a  b   c    d     e"),
        ("only_spaces_8", " " * 8),
        ("spaces_30", " " * 30),
        ("spaces_31", " " * 31),
        ("spaces_32", " " * 32),
        ("spaces_33", " " * 33),
        ("spaces_100", " " * 100),
        ("tabs", "a\tb\t\tc\t\t\td"),
        ("tabs_40", "\t" * 40),
        ("newline_2", "a\n\nb"),
        ("newline_3", "a\n\n\nb"),
        ("newline_30", "x" + "\n" * 30 + "y"),
        ("newline_31", "x" + "\n" * 31 + "y"),
        ("newline_32", "x" + "\n" * 32 + "y"),
        ("newline_100", "\n" * 100),
        ("crlf", "line one\r\nline two\r\n"),
        ("mixed_whitespace", " \n \t \n\n  \t"),
        ("indent", "if x:\n    if y:\n        return 1\n"),
        ("url", "https://huggingface.co/google/embeddinggemma-300m/blob/main/README.md"),
        ("url_query", "http://example.com:8080/search?q=a+b&lang=en#top"),
        ("email_path", "mail me at someone@example.org or see /usr/local/bin/gufo"),
        ("code_python", CODE[0]),
        ("code_cpp", CODE[1]),
        ("code_sql", CODE[2]),
        ("code_json", CODE[4]),
        ("prefix_query", "task: search result | query: "),
        ("prefix_query_text",
         "task: search result | query: which planet is known as the red planet?"),
        ("prefix_document", "title: none | text: "),
        ("prefix_document_text",
         "title: none | text: Mars is the fourth planet from the Sun."),
        ("prefix_title", "title: Mars | text: The red planet."),
        ("digits", "0123456789 3.14159 1,234,567 2024-06-01 0x7fffffff"),
        ("digits_long", "1" * 40),
        ("cjk_japanese", OTHER[0]),
        ("cjk_chinese", OTHER[1]),
        ("cjk_korean", OTHER[2]),
        ("arabic", OTHER[3]),
        ("cyrillic", OTHER[4]),
        ("latin_accents", OTHER[5]),
        ("hindi", "नमस्ते दुनिया"),
        ("thai", "สวัสดีชาวโลก"),
        ("emoji_single", "😀"),
        ("emoji_text", "I love it 😀🎉 so much 🚀!"),
        ("emoji_zwj_family", "\U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466"),
        ("emoji_zwj_profession", "\U0001F469\u200D\u2695\uFE0F and \U0001F9D1\u200D\U0001F4BB"),
        ("emoji_flags", "\U0001F1FA\U0001F1F8 \U0001F1EF\U0001F1F5"),
        ("emoji_skin_tone", "\U0001F44D\U0001F3FD thumbs"),
        ("emoji_keycap", "1\uFE0F\u20E3 2\uFE0F\u20E3"),
        ("combining_acute", "e\u0301 a\u0300 o\u0308 n\u0303"),
        ("combining_stack", "a\u0301\u0302\u0303\u0304"),
        ("precomposed_vs_decomposed", "caf\u00e9 cafe\u0301"),
        ("rare_unicode", "\U00010348 \U0001D11E \U0010FFFF \uFFFD"),
        ("private_use", "\uE000\uF8FF"),
        ("zero_width", "a\u200bb\u200cc\u2060d\ufeffe"),
        ("nbsp_and_unicode_spaces", "a\u00a0b\u2003c\u3000d"),
        ("special_bos", "<bos>"),
        ("special_eos", "<eos>"),
        ("special_pad", "<pad>"),
        ("special_unk", "<unk>"),
        ("special_mask", "<mask>"),
        ("special_image", "<|image|>"),
        ("special_tool_tokens", "<|tool>x<tool|><|tool_call>y<tool_call|>"),
        ("special_inline", "hello <bos> world<eos>next <pad><pad> end"),
        ("special_adjacent", "<bos><eos><pad><unk>"),
        ("special_in_word", "ab<mask>cd"),
        ("special_quote", "say <|\"|>hi<|\"|>"),
        ("special_longest", "<|tool_call><|tool_response>"),
        ("special_partial", "<bo <bos <bos> bos> <|image| <|image|>>"),
        ("special_lookalike", "<start_of_turn>user\nhi<end_of_turn>\n"),
        ("special_case", "<BOS> <Eos>"),
        ("special_spaces", " <bos> "),
    ]
    out = [(name, text, True) for name, text in plain]
    out += [
        ("nospecial_empty", "", False),
        ("nospecial_english", "The quick brown fox.", False),
        ("nospecial_with_special_text", "a <bos> b <eos>", False),
        ("nospecial_emoji", "x 😀 y", False),
    ]
    out.append(("long_document", long_document(), True))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=REPO / "models/embeddinggemma-2")
    parser.add_argument("--output", type=Path,
                        default=REPO / "tests/fixtures/embedding_gemma2/tokenizer_golden.json")
    args = parser.parse_args()

    path = args.model / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))
    result = {
        "reference": "huggingface tokenizers " + version("tokenizers"),
        "tokenizer_json_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cases": [],
    }
    for name, text, add_special_tokens in cases():
        ids = tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
        result["cases"].append({
            "name": name,
            "text": text,
            "add_special_tokens": add_special_tokens,
            "ids": ids,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {len(result['cases'])} cases to {args.output} "
          f"({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

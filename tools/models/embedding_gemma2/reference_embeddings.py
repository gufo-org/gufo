#!/usr/bin/env python3
"""Generate the EmbeddingGemma 2 reference embeddings used by the quality tests.

Runs google/embeddinggemma-2 through sentence-transformers on the CPU and
writes, for each probe text, the token ids and the 768-dimensional embedding.

  python3 tools/models/embedding_gemma2/reference_embeddings.py \
      --model models/embeddinggemma-2 \
      --output tests/fixtures/embedding_gemma2/embedding_golden.json

Requires: torch, transformers (with embedding_gemma2), sentence-transformers>=6.1.
The float32 run is the reference. The BF16 run records the numerical distance
the checkpoint's native precision already has from float32. Stored embeddings
keep SIGNIFICANT_DIGITS digits per value to keep the fixture small.
"""
import argparse
import json
import math

import torch
from sentence_transformers import SentenceTransformer

QUERY = "task: search result | query: "
DOCUMENT = "title: none | text: "

LONG = (
    "The northern lights, or aurora borealis, are produced when charged "
    "particles from the solar wind collide with gases in the upper "
    "atmosphere. "
)

# 5 digits change a value by at most 5e-5 relative. That moves a cosine by about
# 7e-7 (relative error times the sine of the angle, 1.4e-2 at the tests' 0.9999
# bound), over 100 times below the tightest margin of 7.6e-5.
SIGNIFICANT_DIGITS = 5

PROBES = [
    ("query_short", QUERY + "What causes the northern lights?"),
    ("document_short",
     DOCUMENT + "The northern lights are caused by charged particles from "
     "the sun."),
    ("plain", "hello"),
    ("empty_body", DOCUMENT),
    ("multilingual", QUERY + "¿Qué causa las auroras boreales? 北极光是如何形成的？"),
    ("code", "task: code retrieval | query: def fibonacci(n): return n if n < 2 "
     "else fibonacci(n - 1) + fibonacci(n - 2)"),
    ("emoji_and_spaces", "  leading  and   inner spaces \n\n\tand tabs 🙂👍🏽  "),
    ("special_token_text", "literal <bos> and <eos> and <pad> inside text"),
    ("sentence_similarity", "task: sentence similarity | query: A man is "
     "playing a guitar on the stage."),
    # Long enough that sliding-window layers with radius 512 cut the context.
    ("long_document", DOCUMENT + LONG * 90),
]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def embed(path, dtype, texts):
    model = SentenceTransformer(path, model_kwargs={"dtype": dtype},
                                device="cpu")
    model.max_seq_length = 8192
    vectors = model.encode(texts, convert_to_numpy=True, batch_size=1,
                           normalize_embeddings=False)
    return model, [[float(v) for v in row] for row in vectors]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    texts = [text for _, text in PROBES]
    model, reference = embed(args.model, torch.float32, texts)
    _, native = embed(args.model, torch.bfloat16, texts)
    tokenizer = model.tokenizer
    cases = []
    for (name, text), vec, bf16 in zip(PROBES, reference, native):
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        cases.append({
            "name": name,
            "text": text,
            "token_ids": ids,
            "embedding": [float(f"{v:.{SIGNIFICANT_DIGITS}g}") for v in vec],
            "bf16_cosine_to_float32": cosine(vec, bf16),
        })
        print(f"{name}: {len(ids)} tokens, norm "
              f"{math.sqrt(sum(x * x for x in vec)):.4f}, bf16 cosine "
              f"{cosine(vec, bf16):.6f}")
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"model": "google/embeddinggemma-2",
                   "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
                   "dtype": "float32", "cases": cases}, handle,
                  ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()

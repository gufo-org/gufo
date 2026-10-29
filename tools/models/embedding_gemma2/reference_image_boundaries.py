#!/usr/bin/env python3
"""Independent processor/count oracle for image request and context boundaries.

Use transformers 50e35cc0ca18ec19b80653690de553e6decf74f5 and the pinned
checkpoint. No Gufo code or model forward is used; image_golden.json supplies
only reproducible input specifications, never expected processor outputs.
"""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch
import transformers
from transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--transformers-revision", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(args.model)
    fixtures = Path(__file__).resolve().parents[3] / "tests/fixtures/embedding_gemma2"
    images = json.loads((fixtures / "image_golden.json").read_text())
    cases = []
    for budget in (70, 140, 280, 560, 1120):
        case = next(c for c in images["cases"] if c.get("max_soft_tokens", 280) == budget)
        pictures = [Image.new("RGB", (spec["width"], spec["height"])) for spec in case["images"]]

        def count(text):
            inputs = processor(text=[text], images=[pictures], max_soft_tokens=budget,
                               return_tensors="pt")
            return int(inputs["attention_mask"].sum())

        base_tokens = count(case["text"])
        padding = 8192 - base_tokens - 1
        exact = "<eos>" + case["text"] + "<eos>" * padding
        cases.append({"name": case["name"], "max_soft_tokens": budget,
                      "base_tokens": base_tokens, "padding": padding,
                      "exact_tokens": count(exact), "overflow_tokens": count(exact + "<eos>")})
    counts = []
    for n in (32, 33):
        inputs = processor(text=["<|image|>" * n],
                           images=[[Image.new("RGB", (1, 1)) for _ in range(n)]],
                           max_soft_tokens=70, return_tensors="pt")
        counts.append({"images": n, "max_soft_tokens": 70,
                       "tokens": int(inputs["attention_mask"].sum())})
    Path(args.output).write_text(json.dumps({
        "revision": images["revision"], "transformers_revision": args.transformers_revision,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "cases": cases, "image_counts": counts}, indent=2) + "\n")


if __name__ == "__main__":
    main()

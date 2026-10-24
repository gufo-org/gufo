#!/usr/bin/env python3
"""Independent image oracle using the official transformers processor/model.

Requires transformers with embedding_gemma2, torch, torchvision, Pillow.
Run with --model models/embeddinggemma-2 --output <fixture.json>.
The fixture records the processor's exact resized patch bytes as SHA256,
framing token IDs, and float32 reference embeddings. No Gufo code is imported.
"""
import argparse
import hashlib
import json

import numpy as np
from PIL import Image
import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoProcessor

REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
# Covers upsampling, downsampling, unchanged dimensions, ragged patch grids,
# portrait/landscape, and extreme aspect ratios.
SHAPES = [(37, 29), (1231, 947), (768, 768), (511, 263), (41, 997), (2048, 1), (320, 180)]


def image(width, height, seed):
    y, x, c = np.indices((height, width, 3), dtype=np.int64)
    rgb = ((x * 17 + y * 31 + c * 67 + (x * y + seed * 13) % 251) % 256).astype(np.uint8)
    return Image.fromarray(rgb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--transformers-revision", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    config.audio_config = None
    model = AutoModel.from_pretrained(args.model, config=config, dtype=torch.float32,
                                      attn_implementation="sdpa").eval()
    cases = []
    for seed, (width, height) in enumerate(SHAPES):
        picture = image(width, height, seed)
        # Also exercise text before/after an image and two images in one input.
        text = "<|image|>" if seed != 3 else "title: camera | text: <|image|> a colorful pattern"
        if seed == 6:
            text += "\n"  # Frigate typed image request includes a trailing newline.
        pictures = [picture]
        if seed == 4:
            text = "<|image|> and <|image|>"
            pictures.append(image(29, 37, 11))
        inputs = processor(text=[text], images=[pictures], return_tensors="pt")
        pixels = inputs["pixel_values"]
        positions = inputs["image_position_ids"]
        image_records = []
        for j, pic in enumerate(pictures):
            mask = (positions[j] != -1).all(-1)
            raw = (pixels[j][mask] * 255).round().to(torch.uint8).numpy().tobytes()
            image_records.append({"width": pic.width, "height": pic.height,
                                  "seed": seed if j == 0 else 11,
                                  "columns": int(positions[j][mask, 0].max()) + 1,
                                  "rows": int(positions[j][mask, 1].max()) + 1,
                                  "patch_sha256": hashlib.sha256(raw).hexdigest()})
        with torch.inference_mode():
            output = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(1) / mask.sum(1)
            vector = torch.nn.functional.normalize(pooled, dim=-1)[0].tolist()
        cases.append({"name": f"image_{seed}", "text": text,
                      "images": image_records,
                      "token_ids": inputs["input_ids"][0].tolist(),
                      "embedding": [float(f"{v:.7g}") for v in vector]})
        print(f"image_{seed}: {image_records}, {len(cases[-1]['token_ids'])} tokens", flush=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"model": "google/embeddinggemma-2", "revision": REVISION,
                   "transformers": transformers.__version__,
                   "transformers_revision": args.transformers_revision,
                   "torch": torch.__version__, "dtype": "float32", "cases": cases},
                  handle, separators=(",", ":"))


if __name__ == "__main__":
    main()

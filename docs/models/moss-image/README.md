# MOSS-Image

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

Native HIP text-to-image generation with
[MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) from
OpenMOSS: a 42-block single-stream diffusion transformer (about 5B parameters),
a Qwen3.5 9B text encoder and the FLUX.2 small VAE decoder. English and Chinese
prompts. MOSS-Image has **no image editing**; `/v1/images/edits` is rejected.

Weights and the checkpoint's pipeline code use the **Apache-2.0 license**,
separate from Gufo's MIT license. See the
[pinned sources and model contract](../../../src/models/moss_image/UPSTREAM.md).

Download about 27.2 GB:

```sh
MODEL_DIR=$(hf download OpenMOSS-Team/MOSS-Image-SFT \
  --revision f972bd4b4b9e3852133cc5da954d87bc32cd8fb9)
```

The text encoder runs on Gufo's Qwen engine and must be converted once to a
BF16 GGUF (about 15.9 GB) with llama.cpp's `convert_hf_to_gguf.py`. The
converter expects the tokenizer next to the weights and the
`Qwen3_5ForConditionalGeneration` architecture name:

```sh
mkdir encoder && cp "$MODEL_DIR"/text_encoder/* "$MODEL_DIR"/tokenizer/* encoder/
jq '.architectures = ["Qwen3_5ForConditionalGeneration"]' \
  "$MODEL_DIR"/text_encoder/config.json > encoder/config.json
python3 llama.cpp/convert_hf_to_gguf.py encoder --outtype bf16 \
  --outfile moss-image-text-encoder-bf16.gguf
```

Start the server. `serve image` detects MOSS-Image from `model_index.json`:

```sh
nix build
./result/bin/gufo serve image --model "$MODEL_DIR" \
  --text-encoder moss-image-text-encoder-bf16.gguf \
  --served-model-name MOSS-Image --port 8080
```

Generate an image:

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"MOSS-Image","prompt":"A red ceramic teapot on a wooden table",
       "size":"1024x1024","steps":50,"seed":42}' > response.json
jq -r '.data[0].b64_json' response.json | base64 --decode > output.png
```

| Setting | Behavior |
| --- | --- |
| `size` | Default 1024². Sides from 256 to 2048 pixels, multiples of 32 over HTTP. |
| `seed`, `steps` | Seeded replay within the same build. HTTP default 40 steps; the reference pipeline uses 50. Range 2–100. |
| `n` | 1–10 outputs with independent seeds; requests execute serially on the GPU. |
| Guidance | Fixed to the reference defaults: classifier-free guidance 3.0 with block 22 skipped in the empty-prompt branch. |
| `background` | `auto` or `opaque`; output is always opaque RGB. `transparent` is rejected. |

Prompts longer than 4096 tokens are rejected (the reference truncates them).
Output is PNG/base64; edits, masks, URL output and streaming are unsupported.
`/health`, `/ready` and `/v1/models` are available. For llama-swap, use the same
command with `--port ${PORT}`, as for [Qwen-Image-2.1](../qwen-image-2.1/README.md).

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) ·
[Experiments](EXPERIMENTS.md)

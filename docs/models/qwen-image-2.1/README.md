# Qwen-Image-2.1

Native HIP image generation and editing on AMD Strix Halo (`gfx1151`). Loads
BF16 safetensors directly; Python is not needed for inference. The checkpoint
contains a diffusion transformer, Qwen3-VL prompt encoder and RGBA VAE.

Weights use the **non-commercial Qwen Research License**, separate from Gufo's
MIT license. See the [pinned sources and model contract](../../../src/models/qwen_image_21/UPSTREAM.md).

Download about 33.1 GB and start the server:

```sh
MODEL_DIR=$(hf download Qwen/Qwen-Image-2.1 \
  --revision b3179ad355be050328e483a9dfdd9e60cd62adfa)
nix build
./result/bin/gufo serve image --model "$MODEL_DIR" \
  --served-model-name Qwen-Image-2.1 --port 8080
```

Generate an image:

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen-Image-2.1","prompt":"A red ceramic teapot on a wooden table",
       "size":"1024x1024","seed":42}' > response.json
```

Edit with multiple references, in the order they are uploaded. For a single
reference, keep just one `image[]` field:

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/images/edits \
  -F model=Qwen-Image-2.1 \
  -F 'image[]=@object.png' -F 'image[]=@room.png' \
  -F 'prompt=Place the object from the first image in the room from the second image' \
  -F size=1024x1024 -F seed=42 > response.json
```

Both use the OpenAI Images response format. Save the first PNG:

```sh
jq -r '.data[0].b64_json' response.json | base64 --decode > output.png
```

| Setting | Behavior |
| --- | --- |
| `size` | Default `auto`: 1024² generation; reference aspect ratio for edits. Explicit sides must be multiples of 32, at most 4096 per side and 4 megapixels total. |
| `seed`, `steps` | Seeded replay within the same build and request configuration. Default 40 steps; range 2–100. Fewer steps change quality. |
| `n` | 1–10 outputs with independent seeds; requests execute serially on the GPU. |
| References | Up to 10 PNG/JPEG uploads, 20 MiB encoded and 32 megapixels combined; the HTTP body limit also applies. |
| `background` | `auto`, `transparent`, or `opaque` (composite over white). |

The official repository recommends 2048²; Gufo follows the pinned Diffusers
1024² default. Set `size` explicitly for larger output. Output is PNG/base64;
URL output, separate masks, streaming and `input_fidelity` are unsupported.
OpenAI SDK clients use the local `/v1` base URL; `seed` and `steps` are Gufo
extensions supplied through `extra_body`.

For llama-swap, add this model and send the same requests to the proxy:

```yaml
models:
  Qwen-Image-2.1:
    cmd: >
      /path/to/gufo serve image --model /path/to/snapshot
      --served-model-name Qwen-Image-2.1 --port ${PORT}
```

For larger uploads, raise `--max-request-bytes` and the proxy's body limit.
`/health`, `/ready` and `/v1/models` are available. Nix `mkGufoServe` supports
`modality = "image"` with `model` and `servedModelName`.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) ·
[Experiments](EXPERIMENTS.md)

# Qwen-Image-2.1

Native HIP text-to-image generation and reference-image editing on gfx1151.
The checkpoint contains a 7B diffusion transformer, Qwen3-VL prompt encoder and
RGBA VAE. Inference uses BF16 weights with FP32 accumulation. Python is an
offline reference only.

The model weights have a **non-commercial Qwen Research License**, separate from
Gufo's MIT license. See the [pinned sources and contract](../../../src/models/qwen_image_21/UPSTREAM.md).

Download and serve:

```sh
hf download Qwen/Qwen-Image-2.1 \
  --revision b3179ad355be050328e483a9dfdd9e60cd62adfa
nix build
./result/bin/gufo serve image --model /path/to/downloaded/snapshot \
  --served-model-name Qwen-Image-2.1 --port 8080
```

The complete download is about 33.1 GB. Point `--model` at the directory
containing `model_index.json`; no conversion is needed.
Gufo follows the pinned Diffusers pipeline's 1024² default. The official
repository recommends 2048²; request that explicitly with `size`.

Generate:

```sh
curl http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen-Image-2.1","prompt":"A red ceramic teapot on a wooden table",
       "size":"1024x1024","n":1,"response_format":"b64_json","seed":42}'
```

Edit:

```sh
curl http://127.0.0.1:8080/v1/images/edits \
  -F model=Qwen-Image-2.1 -F 'image[]=@photo.png' \
  -F 'prompt=Change the background to a sunset beach' \
  -F size=1024x1024 -F seed=42
```

Both return the OpenAI Images shape:
`{"created":...,"data":[{"b64_json":"..."}]}`. Decode `b64_json` into PNG bytes.
The OpenAI SDK's `images.generate` and `images.edit` work with the local
`base_url`; `seed` and `steps` are optional Gufo extensions in `extra_body`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
image = client.images.generate(
    model="Qwen-Image-2.1", prompt="A red ceramic teapot on a wooden table",
    extra_body={"seed": 42},
)
```

| Option | Behavior |
| --- | --- |
| `size` | Default `auto`: 1024 square for generation, reference aspect ratio for editing. Explicit dimensions must be multiples of 32, at most 4096 per side and 4 megapixels. |
| `n` | 1–10 images, independent seeds. GPU execution is serialized. |
| `image` / `image[]` | PNG/JPEG multipart uploads; up to 10 references, 20 MiB encoded data and 32 megapixels combined. The server body limit also applies. |
| `background` | `auto`, `transparent` (official transparency prompt), or `opaque` (composite output over white). |
| `quality` | `auto`, `standard`, `high`: all retain the official 40-step default. |
| `seed`, `steps` | Reproducible seed; 2–100 steps, default 40. Lower steps change quality. |

Only PNG/base64 output is implemented. URL responses, WebP, streaming,
`input_fidelity` and separate `mask` uploads return an explicit error.
Editing accepts annotated reference images. Prompt rewriting is not automatic.
This model generates images; existing vision chat models handle image-to-text.
Fixed seeds replay within the same Gufo build and request configuration;
PyTorch uses a different noise generator. Independent comparisons therefore
share initial noise rather than assuming equal integer seeds imply equal noise.

llama-swap configuration:

```yaml
models:
  Qwen-Image-2.1:
    cmd: >
      /path/to/gufo serve image --model /path/to/downloaded/snapshot
      --served-model-name Qwen-Image-2.1 --port ${PORT}
```

Send the same JSON generation or multipart edit request to llama-swap's port.
`/health`, `/ready` and `/v1/models` are available. For larger uploads, configure
Gufo's existing `--max-request-bytes` and the proxy's corresponding limit.
The Nix `mkGufoServe` helper also accepts `modality = "image"` with `model`
and `servedModelName`.

[Benchmarks](BENCHMARKS.md) · [Evaluation](EVALUATION.md) ·
[Experiments](EXPERIMENTS.md)

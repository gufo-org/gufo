# Qwen-Image-2.1

Qwen-Image-2.1 generates images from text and edits images you upload. Gufo
runs the BF16 checkpoint and its eight-step Turbo variant behind the OpenAI
Images API.

> **Warning:** The weights use the non-commercial Qwen Research License, not
> Gufo's MIT license. See the
> [pinned sources and model contract](../../src/models/qwen_image_21/UPSTREAM.md).

## Files

The checkpoint is about 33.1 GB:

```sh
MODEL_DIR=$(hf download Qwen/Qwen-Image-2.1 \
  --revision b3179ad355be050328e483a9dfdd9e60cd62adfa)
```

## Run

```sh
gufo serve image --model "$MODEL_DIR" \
  --served-model-name Qwen-Image-2.1 --port 8080
curl --fail-with-body http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen-Image-2.1","prompt":"A red ceramic teapot on a wooden table",
       "size":"1024x1024","seed":42}' > response.json
jq -r '.data[0].b64_json' response.json | base64 --decode > output.png
```

To edit, upload one or more references; the prompt refers to them in upload
order:

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/images/edits \
  -F model=Qwen-Image-2.1 \
  -F 'image[]=@object.png' -F 'image[]=@room.png' \
  -F 'prompt=Place the object from the first image in the room from the second image' \
  -F size=1024x1024 -F seed=42 > response.json
```

## Settings

| Setting | Behavior |
| --- | --- |
| `size` | `auto` (the default) gives 1024 × 1024, or the reference's aspect ratio when editing. Sides must be multiples of 32, at most 4096, and 4 megapixels in total. |
| `steps` | 40 by default, from 2 to 100. Fewer steps are faster and lower quality. |
| `seed` | The same seed, settings and build give the same image, also after a restart. |
| `n` | 1 to 10 images, each with its own seed, made one after another. |
| `background` | `auto`, `transparent`, or `opaque` (over white). |
| References | Up to 10 PNG or JPEG files, 20 MiB and 32 megapixels in total. |

Output is base64 PNG. URL output, separate masks, streaming and
`input_fidelity` aren't supported. With the OpenAI SDK, pass `seed` and
`steps` through `extra_body`.

The upstream project recommends 2048 × 2048; Gufo defaults to 1024 × 1024,
like the pinned Diffusers version. Set `size` for larger images.

## Qwen-Image-2.1-Turbo

Turbo is a separately trained checkpoint (about 32.5 GB, same license) that
always uses eight steps. It runs on the same server, API and editing paths:

```sh
MODEL_DIR=$(hf download Qwen/Qwen-Image-2.1-Turbo \
  --revision fc13ba1986fcb905854dc8de2f52eee395f49b5b --exclude 'assets/*')
gufo serve image --model "$MODEL_DIR" \
  --served-model-name Qwen-Image-2.1-Turbo --port 8080
```

Omit `steps` or pass `8`; any other value returns HTTP 400.

## Serving behind llama-swap

```yaml
models:
  Qwen-Image-2.1:
    cmd: >
      /path/to/gufo serve image --model /path/to/snapshot
      --served-model-name Qwen-Image-2.1 --port ${PORT}
```

For large uploads, raise `--max-request-bytes` and the proxy's body limit. See
[llama-swap](../server.md#llama-swap) for the general setup.

## LoRA adapters

Register LoRA safetensors published for `Qwen/Qwen-Image-2.1` at startup and
select them per request by name:

```sh
gufo serve image --model "$MODEL_DIR"   --lora voxel=/models/loras/Qwen-Image-2.1-Voxel-Style-3000.safetensors   --lora-default voxel:0.8
```

Generation JSON and edit forms accept `"loras": [{"name": "voxel", "scale":
0.8}]`; omitting it applies the `--lora-default` set and `[]` applies none.
`GET /sdapi/v1/loras` lists the registered adapters. Supported layouts,
limits and the merge are described in
[LoRA adapters](../internal/models/qwen-image-2.1/LORA.md).

## Performance

As of November 1, 2026, on Strix Halo, one 1024 × 1024 image takes:

| Checkpoint | Steps | Time |
| --- | --- | --- |
| Qwen-Image-2.1 | 40 | 89.9 s |
| Qwen-Image-2.1, editing one reference | 40 | 130.2 s |
| Qwen-Image-2.1-Turbo | 8 | 18.6 s |

See the [full benchmarks](../internal/models/qwen-image-2.1/BENCHMARKS.md) and
the [quality report](../internal/models/qwen-image-2.1/QUALITY.md).

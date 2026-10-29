# Qwen-Image-2.1

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md) · [ROCm migration](ROCM.md)

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

Edit noise combines `seed` with the resized reference pixels. The same edit
replays exactly, including after restarting the server, while editing an image
with its generation seed no longer reuses its original noise. Seeded edits
therefore differ from older Gufo releases; generation seeds are unchanged.

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

## LoRA adapters

Register adapters for the base Qwen-Image-2.1 transformer at startup; requests
select them by name. Files are LoRA safetensors as published on Hugging Face
for `Qwen/Qwen-Image-2.1`, used as is:

```sh
./result/bin/gufo serve image --model "$MODEL_DIR" \
  --served-model-name Qwen-Image-2.1 --port 8080 \
  --lora voxel=/models/loras/Qwen-Image-2.1-Voxel-Style-3000.safetensors \
  --lora fix=/models/loras/qwen-image-2.1-fix-1.0-comfy.safetensors \
  --lora-default fix:0.8
```

| Option | Behavior |
| --- | --- |
| `--lora NAME=PATH` | Repeatable. Names are 1–64 characters of `A-Z a-z 0-9 . _ -`. Every file is parsed and checked against the transformer at startup; an unsupported file stops the server with `<file>: <tensor>: <reason>`. |
| `--lora-default NAME:SCALE` | Repeatable. Adapters applied, in this order, to requests that omit `loras`. |

Generation JSON and edit multipart forms accept `loras`, a list applied in
order (in multipart forms, the field value is this JSON text):

```json
{"model": "Qwen-Image-2.1", "prompt": "A small cottage beside a lake at sunset",
 "size": "1024x1024", "seed": 7,
 "loras": [{"name": "voxel", "scale": 0.8}, {"name": "fix"}]}
```

Omitting `loras` applies the `--lora-default` set; `[]` disables adapters.
Each entry has a registered `name` and an optional `scale` (default 1, at most
10 in magnitude; 0 skips the entry). Up to 16 entries, each name once. Only
registered names are accepted, never paths. Seeded replay requires the same
seed and the same `loras` list, in the same order.

`GET /sdapi/v1/loras` lists the registered adapters in registration order, in
the Automatic1111 shape plus the scales in use:

```json
[{"name": "style", "alias": "style", "path": "style.safetensors",
  "metadata": {"format": "peft+alpha", "targets": 224, "rank": "16-16"},
  "default_scale": 0.5, "active_scale": 0.8}]
```

`path` is the file name only; `targets` counts the base weights the adapter
patches, not the tensors in its file. `active_scale` reflects the adapters
currently merged into the weights (none while a switch is running or after a
failed one), `default_scale` the `--lora-default` set.

| Layout (tensor names; any of the prefixes `diffusion_model.`, `transformer.`, `base_model.model.`, `diffusion_model.transformer.` or none) | Example |
| --- | --- |
| PEFT `X.lora_A.weight` / `X.lora_B.weight`, optional `X.alpha` | Most adapters |
| DiffSynth `X.lora_A.default.weight` / `X.lora_B.default.weight` | `prithivMLmods/Qwen-Image-2.1-*` |
| Bare `X.lora_A` / `X.lora_B` and `X.lora_down` / `X.lora_up` | `Haverbex/Qwen-Image-2.1-UltraFast` |
| `X.lora_down.weight` / `X.lora_up.weight` + `X.alpha` (ComfyUI) | `t8star/*-comfy` |
| Diffusers `X.lora.down.weight` / `X.lora.up.weight` | `lightx2v/Qwen-Image-2.1-Lightning` |
| Kohya `lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight` + `.alpha` | ComfyUI-fused exports |
| Fused `img_mlp.gate_up` pairs (rows: `gate_layer`, then `proj`) | `Danrisi/samsung_qwen2.1` |
| Full deltas `X.diff`, including 1-D norm weights, mixed with pairs of any rank | `nynxz/Qwen-Image-2.1` |
| DoRA `X.dora_scale` on the output axis | `e-n-v-y/Qwen-Image-2.1-Fix` |

Scale per pair is `alpha / rank` with `alpha` from `X.alpha`, else from the
Diffusers `lora_adapter_metadata` (`lora_alpha`, rsLoRA `alpha / sqrt(rank)`),
else `alpha = rank`. Other metadata such as `ss_network_alpha` is ignored, as
in ComfyUI. Targets are every transformer projection and norm weight, including
`img_in`, `txt_in`, the timestep embedder, `modulation.1`, `norm_out.linear`
and `proj_out`. Semantics follow [ComfyUI](../../../src/models/qwen_image_21/UPSTREAM.md#lora-adapters).

Rejected with an error naming the first offending tensor: adapters for other
architectures (Qwen-Image 1.x/Edit with 60 blocks, Flux, SD/SDXL), text-encoder
adapters, LyCORIS LoHa/LoKr/OFT/LoCon, bias deltas, full replacement weights
(including Fun-Acc step-dependent `proj_out` adapters and training auxiliaries),
input-axis DoRA and dtypes other than BF16/F16/F32.

Adapters are merged into the BF16 transformer weights: each patched weight is
uploaded as BF16 (as every weight is), converted to FP32, the selected adapters
are applied in request order, and the result is rounded to BF16. The published
Qwen-Image-2.1 checkpoint is all BF16, so a merged weight is rounded once; an
FP32 checkpoint tensor is rounded once at upload and once more after merging.
Kernels, step time and resident weight memory are those of the base model;
merge scratch comes from the runtime's bounded buffer pool and is reused. When the selected set changes, the weights whose
adapter terms change are rebuilt from the mapped checkpoint before the request
starts (never by subtracting; on the first such request this includes their
upload, other weights still load on first use), so returning to a previous set
or to none is bit-identical; the request log reports this as `lora_ms`. If a
switch fails or is cancelled, the remaining weights rebuild with the new set on
first use, or before the next request that selects it.
Without `--lora`, generation is unchanged and pixel-identical to a server
without adapter support. With adapters, the merged weights are qualified
against an FP64 reference separately from the base model; see
[Quality](QUALITY.md#lora-adapters) and [Benchmarks](BENCHMARKS.md#lora-adapters).

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) ·
[Experiments](EXPERIMENTS.md)

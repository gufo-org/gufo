# Qwen-Image-2.1 implementation notes

Developer detail behind the [public model page](../../../models/qwen-image-2.1.md).
[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [LoRA adapters](LORA.md) · [Experiments](EXPERIMENTS.md) · [ROCm](ROCM.md)

Native HIP image generation and editing on AMD Strix Halo (`gfx1151`). Loads
BF16 safetensors directly; Python is not needed for inference. The checkpoint
contains a diffusion transformer, Qwen3-VL prompt encoder and RGBA VAE.

Edit noise combines `seed` with the resized reference pixels. The same edit
replays exactly, including after restarting the server, while editing an image
with its generation seed no longer reuses its original noise. Seeded edits
therefore differ from older Gufo releases; generation seeds are unchanged.

Omit `steps` or pass `8`; other counts return HTTP 400. The checkpoint's eight
`sample_sigmas` replace the base model's dynamic shift. It is about 4.8× faster
than the 40-step base model at 1024²; see [benchmarks](BENCHMARKS.md#qwen-image-21-turbo)
and [quality](QUALITY.md#qwen-image-21-turbo).

`/health`, `/ready` and `/v1/models` are available. Nix `mkGufoServe` supports
`modality = "image"` with `model` and `servedModelName`.

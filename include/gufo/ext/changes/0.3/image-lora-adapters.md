# Image LoRA adapters

Image runners can apply LoRA adapters per request (`image.hpp`):
- `ImageLora` names a registered adapter and its scale; `ImageRequest::loras`
  lists the adapters to apply, in order, and defaults to none.
- `ImageRunner::AddLora` registers an adapter file under a name and returns
  its `ImageLoraInfo` (format, patched weights, rank range);
  `ImageRunner::ActiveLoras` reports the set applied to the weights now. Both
  are virtual with defaults (reject every adapter, report none), so existing
  runners build unchanged after rebuilding.
- `ImageMetrics::lora_ms` reports the adapter switch time, which
  `total_ms` includes.

Experimental shared adapter parser (`experimental/lora_adapter.hpp`,
namespace `gufo::lora`): `Adapter` reads and validates a LoRA safetensors
file (PEFT, DiffSynth, Diffusers, ComfyUI and kohya layouts, `.alpha`,
`.scale`, diffusers `lora_adapter_metadata`, `.diff` deltas and output-axis
DoRA) against a model-supplied `Schema` that maps module names to row
`Slice`s of base tensors, so fused base tensors and fused adapter modules
are both expressible. It yields `Patch`es; `Term` pairs a patch with a
strength, and `Floats` widens adapter tensors to FP32. Merging the patches
into weights remains the model runtime's job.

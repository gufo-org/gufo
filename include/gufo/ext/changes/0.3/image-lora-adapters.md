# Image LoRA adapters

Image runners can apply LoRA adapters per request (`image.hpp`):
- `ImageLora` names a registered adapter and its scale; `ImageRequest::loras`
  lists the adapters to apply, in order, and defaults to none.
- `ImageRunner::AddLora` registers an adapter file under a name and returns
  its `ImageLoraInfo` (format, patched weights, rank range);
  `ImageRunner::ActiveLoras` reports the set applied to the weights now. Both
  are virtual with defaults (reject every adapter, report none), so existing
  runners build unchanged.
- `ImageMetrics::lora_ms` reports the adapter switch time, which
  `total_ms` includes.

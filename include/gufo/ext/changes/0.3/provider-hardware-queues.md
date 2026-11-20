# Provider hardware queue preference

Let speech providers declare the HIP hardware queue cap they serve best with:
- Optional provider `serving.hardware_queues` in `schemas/extension.schema.json`,
  an integer from 1 to 7, accepted only for `speech_synthesis` and
  `speech_recognition` providers. CMake rejects other values and capabilities.
- `ProviderInfo::hardware_queues` (`include/gufo/ext/registry.hpp`), generated
  from that field. Zero keeps the modality's default.

`gufo serve tts` and `gufo serve asr` probe the model's provider before the
first HIP dispatch and use its declared cap instead of the speech default of
one queue. The cap is still clamped to the device queue budget, and an
operator's `GPU_MAX_HW_QUEUES` always wins.

Existing manifests remain valid. Code that builds `ProviderInfo` with
designated initializers keeps compiling; the new member is last.

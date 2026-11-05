# Provider-defined text sampling defaults

Text (`src/ext/text.hpp`, `src/core/text_sampling_defaults.hpp`): a text
provider supplies typed sampling defaults without a core preset enum value.
- `sampling::TextSamplingValues`: optional `temperature`, `top_k`, `top_p`,
  `min_p`, `min_keep`, `repeat_penalty`, `repeat_last_n`, `frequency_penalty`
  and `presence_penalty`. A set value, including zero, is applied as written; an
  unset control keeps the `SamplingConfig` default. The seed and the output
  constraint are never defaults.
- `sampling::TextSamplingDefaults`: one `TextSamplingValues` for thinking
  enabled, disabled and unspecified (`thinking_enabled`, `thinking_disabled`,
  `thinking_unspecified`).
- `TextRunnerDescriptor::sampling_defaults` (HTTP and server paths) and
  `TextProvider::sampling_defaults` (a probe over the GGUF reader, used by
  `prompt` and `chat`). A provider with `prompt` or `chat` drivers sets both to
  the same value. The descriptor field takes precedence over `sampling_preset`.
- `sampling::ValidateTextSamplingDefaults` throws `std::invalid_argument` for
  values `SamplingConfig` rejects. Model load fails on invalid descriptor
  defaults; `prompt` and `chat` fail on invalid resolved values.
- `TextPromptOptions::thinking_mode()` returns the thinking mode selected by
  `--think` and `--reasoning-effort`, and is what selects the defaults for
  `prompt` and `chat`.
- Precedence is provider defaults, then explicit server/CLI settings, then
  per-request overrides. Explicit zero and false settings stay explicit.
  `bench` keeps its explicit greedy defaults.
- `sampling::ResolveTextSampling` accepts the provider defaults;
  `sampling::PresetDefaults` expresses the in-tree presets as such values.

Compatibility: `TextModelPreset` and `TextRunnerDescriptor::sampling_preset`
remain and resolve to the same values as before, so existing extensions compile
and behave unchanged. New fields are appended or default to empty. `prompt` and
`chat` now apply the provider defaults for the thinking mode selected by
`--think`, as documented, for providers that set
`TextProvider::sampling_defaults`; the Qwen and DeepSeek providers do.

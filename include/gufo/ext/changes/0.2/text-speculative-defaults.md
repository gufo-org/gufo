# Text speculative defaults

Text (`src/ext/text.hpp`): a text provider declares the draft settings of each
speculative backend.
- `TextProvider::speculative_defaults` (`TextSpeculativeDefaults`): per backend,
  the values used when `--draft-tokens`, `--min-draft-tokens` and
  `--draft-policy` are absent, and the `--draft-policy` values it accepts.
- `TextSpeculativeConfig::draft_policy`: the resolved policy name;
  `dflash_policy` is derived from it for DFlash2.
- `max_draft_tokens`, `min_draft_tokens` and `draft_policy` in
  `TextSpeculativeConfig`, `TextPromptOptions` and `TextBenchOptions` hold the
  resolved values: the explicit flag, else the provider's default, else the
  host's 7, 1 and no policy.
- A `--draft-policy` the selected provider does not accept fails after the
  provider is selected, before weights load; one that no available provider
  accepts fails while parsing.

Compatibility: additive. A provider that declares nothing runs with the values
it received before, and command lines accepted before are accepted unchanged.

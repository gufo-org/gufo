# Deprecated draft options

Text (`src/ext/text.hpp`): `--draft-tokens`, `--min-draft-tokens` and
`--draft-policy` are accepted and ignored with a warning.
- `TextProvider::speculative_defaults` is the only source of the draft
  settings: `max_draft_tokens`, `min_draft_tokens` and `draft_policy` in
  `TextSpeculativeConfig`, `TextPromptOptions` and `TextBenchOptions` hold the
  provider's values, else the host's 7, 1 and no policy.
- `TextSpeculativeDefaults::draft_policies` is no longer read.

Compatibility: additive. Providers build and run unchanged, and command lines
accepted before are accepted unchanged.

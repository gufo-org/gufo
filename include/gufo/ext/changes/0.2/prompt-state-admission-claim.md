# Prompt-state admission claim

Text (`include/gufo/ext/text.hpp`, `src/ext/text.hpp`):
`TextRunnerResourceClaim` gains two optional inputs to the server's prompt
admission decision:
- `prompt_state_bytes_per_token` is the host memory one prompt token commits to
  model state, currently the full-attention K/V row. Missing means the runner
  has no prompt-length state to charge.
- `prompt_admission_budget_bytes` is the host memory prompt state may claim
  before admission refuses a request. Missing disables prompt-state admission
  for that runner.

The server reserves `prompt length * prompt_state_bytes_per_token` while a
request is in flight and releases it when the request reaches a terminal state.
A request that cannot fit is refused before prefill starts, so a long prompt no
longer exhausts host memory silently on unified-memory parts.

Both fields default to empty, so existing runners and extensions keep their
current behavior. Source compatible; extensions that construct
`TextRunnerResourceClaim` with designated initializers rebuild.

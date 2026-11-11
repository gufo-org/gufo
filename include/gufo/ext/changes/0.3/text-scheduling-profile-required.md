# Required text scheduling profile

Breaking: every text runner declares how its prompts are scheduled while other
sessions decode (`gufo/ext/scheduling.hpp`, `src/ext/text.hpp`).

- `TextRunnerDescriptor::scheduling` is a `TextSchedulingProfile`, no longer
  `std::optional`. The host rejects a runner at load when `pass_tokens`,
  `max_layer_groups` or `max_yield_decode_width` is zero; the default profile
  has `pass_tokens` zero, so a runner that does not set it fails to load.
- Beside decoders, the host gives an incremental runner passes of
  `pass_tokens` and splits them into up to `max_layer_groups` layer groups.
  The fixed 512-token decode-active budget for runners without a profile is
  removed. A runner whose pass is short declares that width with
  `max_layer_groups` one and runs its passes unsplit.
- A profile no longer requires `incremental_prefill`. A runner without it
  prefills every prompt whole, in one unsplit pass; its profile is checked but
  not used.

Migration: a runner that had no profile declares
`TextSchedulingProfile{.pass_tokens = N, .max_layer_groups = 1,
.max_yield_decode_width = W}`, where `N` is its efficient prefill pass (512
keeps its previous passes beside decoders) and `W` the sessions one decode
step between groups may batch. `SetPrefillYield` and `PrefillYield` are
unchanged.

# Text scheduling profiles and layered prefill

Text (`scheduling.hpp`, `src/sched/profile.hpp`, `src/ext/text.hpp`): a runner
can describe how its prompts are scheduled while other sessions decode, and
split each prompt pass into layer groups with a decode round for the other
sessions between them.

- `TextSchedulingProfile` has `pass_tokens` (the pass width used beside
  decoders), `max_layer_groups` (one runs passes unsplit) and
  `max_yield_decode_width` (sessions one decode step between groups may
  batch; each round between groups steps every decoder once).
  `TextRunnerDescriptor::scheduling` holds it; empty keeps the host's bounded
  decode-active passes. A profile requires `incremental_prefill`, and every
  field must be at least one; the host rejects the runner at load otherwise.
- `TextRunnerState::SetPrefillYield(groups, yield)` arms the next `Prefill` or
  `PrefillThrough` call only. A profiled runner splits the pass into about
  `groups` layer groups of similar duration, adding a boundary where the pass
  would otherwise stall (such as a wait for input rows), and calls
  `YieldPrefill()` between them; `prefill_yield_groups()` is one when the host
  asked for unsplit passes. Every
  layer must keep the arithmetic of an unsplit pass, so outputs stay
  bit-identical, and the runner must hold no lock or scratch its decode paths
  take while yielding. A runner may ignore the request and run unsplit passes.
- `PrefillYield` (`std::function<void()>`) decodes other sessions only, never
  touches the prefilling state and does not throw. The prompt's cancellation
  still arrives through the state's cancellation check.

Additive: existing runners have no profile and keep today's scheduling.

# Per-request runtime residual projection

Text (`src/ext/text.hpp`): a text runner may apply an optional runtime
residual-stream projection that each chat request opts into.
- `ChatRequest::projection` (`std::optional<bool>`): the request's opt-in. It
  is off unless set to true and never carries into later requests. The host
  rejects a request that sets it on a runner without the capability.
- `TextRunnerCapabilities::runtime_projection`: the runner applies the
  projection for requests that opt in. Runners set it only while they carry a
  loaded projection vector; the host also rejects
  `TextRunnerOptions::projection_vector_path` at load for a runner that does
  not set it.
- `TextRunnerState::SetProjectionEnabled(bool)`: the host calls it once per
  leased state before restore, prefill or generation with the request's
  opt-in. The default implementation ignores it.
- `TextRunnerOptions::projection_vector_path` and
  `TextRunnerOptions::projection_layers` (an inclusive `"LO-HI"` range; empty
  selects every layer the vector carries) configure the vector at load.
- A runner that sets the capability keeps its prompt-cache identities and its
  durable compatibility identity distinct across projection states and loaded
  vectors.

Additive: existing extensions compile and behave unchanged.

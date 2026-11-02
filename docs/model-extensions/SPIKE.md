# Flash-Next text extension spike

Card 03's throwaway investigation and review integration are complete. Its code lives in the separate
jj workspace `/tmp/gufo-card03-spike`, change `ovmkszut`, exact source
`50ba836fc4ab2fe7003371f3f290bdd48ea68b0a`, based on `79930b75`.
The code is deliberately absent from the main workspace; its
[source patch and evidence](evidence/card03/README.md) are retained.

## Implemented experiment

The real serving path now constructs Flash-Next through an explicitly typed
provider factory with immutable provider options. Separately declared HIP-free
metadata is exercised by an executable fixture; production does not consume it.
The 600-line runner implementation lives in the model directory. The existing
state, MTP sampler copies/draw-state publication, cancellation, cache binding,
identity writers and engine calls retain their original bodies.

The runner header is separated from pool/cache policy declarations. Existing
chat message/request values, initial output-state enum and continuation base
classes move verbatim into neutral core/cache headers. Existing source names
remain compatible through includes and an enum alias, so this experiment can
exercise the existing serving code without changing HTTP behavior. The shared
HIP probe/backing and Qwen prompt helper bodies are extracted into internal
headers. These are experimental placements, not a published/stable API.

## Decisions for dependent cards

- Keep step-level execution, including actual batched decode. A direct session
  interface cannot replace it while core owns cross-request scheduling.
- Keep model-owned CLI drivers. The existing benchmark creates disposable
  engine sessions and times engine work directly; the serving runner explicitly
  rejects `CreateState()` because component slots own its sessions. Forcing
  CLI benchmarks through a runner pool introduces cache allocation/reuse and
  scheduling into the measured path. Driver registration should preserve those
  existing lifetimes and output formats; common reporting stays core-owned.
- Provider metadata is inspectable without loading a model or probing HIP.
  Runtime model IDs come from loaded model metadata, rather than duplicating
  GGUF names in compile-time registration.
  The spike declares metadata but production code does not consume it for
  selection, parsing or CLI registration. The retained executable fixture
  checks HIP-free shape, option representability and factory signature only.
  Actual metadata-driven registration and parsing remain dependent-card gates;
  typed options and the factory already execute on the serving path.
- Options are provider-owned typed values passed at creation. Preserve current
  `--mtp-model`, `--draft-tokens` and provider value `mtp` for this experiment;
  a later public rename requires its own explicit breaking-change decision.
  Constraints and RNG remain core-owned; MTP executes inside the model and
  copies/publishes sampler state exactly as before.
- Preserve every-built-dialect JSON fallback. This experiment does not narrow
  behavior to the active model, and existing tool-constraint code is unchanged.
- Do not replace native snapshots or accelerate the cache RFC cutover. The
  copied continuation base types retain the native implementation contract.

## Limits exposed by the extraction

The header split alone does not deliver the final extension API. Core-owned
resource backing still uses model adapters' structural `BackingPlan` methods;
its final neutral cache resource contract, injected budgets/logging and stable
public paths belong to their dependent cards. The speculative/tool grammar
surface remains experimental. Shared Qwen prompt helpers still use the logger;
metadata/options schema and configure-time validation are card 04 work.

## Validation

Both production executables built and linked with the same pinned toolchain.
The candidate passes six focused CPU tests, including all three exact identity
goldens, runner/scheduler behavior and independent MTP sampling numerical
checks, plus the HIP-free provider-header compile and repository formatter.
The independent numerical test checks proposal/rejection/residual probabilities,
RNG and stop contracts. Candidate 16-token GPU batched-versus-sequential prefill
also passes its existing bounds: finite outputs, winner rank 1, RMSE 0.389394,
cosine 0.990998 and maximum error 2.20532. These are distinct scopes: a full-model
independent GPU forward oracle was not run and is not claimed. Combined with
exact matched serving outputs/work and unchanged arithmetic bodies, this is
adequate numerical/semantic evidence for the throwaway spike; later migrations
retain their own model-specific quality gates.

The initial idle baseline/candidate pairs each execute the same 118 requests
in 17 affected sub-suites: `state-edges`, `long-context`, `tool-reasoning`,
RAM continuation, cancellation, image continuation and disk restart/restore.
Both candidate modes pass all functional assertions, with zero output or
expected-work changes across 1,663 request/phase comparisons per mode. The
loaded MTP candidate and baseline both propose 3,322 drafts and accept 2,491;
this is actual speculative execution, not a mode-label-only check.

**Performance parity is unproven.** Initial controlled comparisons flag 17
phases in off mode and 24 in MTP, including maximum token gaps, image
continuation and disk restore. Thresholds remain 5% and 10 ms per request/phase.
The alternating full-history repeats and unchanged-main controls leave off at
1,640 passed / 23 inconclusive / zero reproduced failures out of 1,663 phases.

The first MTP repeat **reproduced three flagged measurements of one request**:
`long_schema_cold` decode (178.24–178.33 ms on main versus 189.15–204.27 ms on
the candidate), decode per token, and maximum token gap (107.17–107.20 versus
117.82–131.91 ms). That original failed qualification is retained unchanged.
Two focused alternating replays preserve the original disk policy and exact
preceding request/history hashes, but do not reproduce those delays:
decode 177.87 → 178.47 and 178.12 → 176.98 ms; maximum gap 106.69 → 107.37
and 106.91 → 105.99 ms. The strict combined range qualification becomes
1,604 passed / 59 inconclusive / zero reproduced failures. This establishes
non-reproduction in the focused controls; it does not explain the initial
stalls or prove parity. Third-agent review integration retains this as
**inconclusive performance evidence**: no failure remains confirmed across the
expanded controls, the initial delay remains unexplained, and performance parity
is unproven. No performance fix or threshold change was applied. Completing this
investigation does not qualify a production migration or waive its gates.

All 281 common production compilation commands match after source-root
normalization. Device resource notes are identical for all 74 embedded gfx1151
images; `.text` is byte-identical for 71. Three MMQ images (one Flash-Next and
two DeepSeek) retain `.text`/constant differences requiring further
source-path/relocation investigation. Debug/symbol identifiers also differ
between workspace roots. No whole-image or universal ISA identity is claimed.

The first diagnostic baseline overlapped compilation and is excluded from
performance qualification. Every attempt, including failures and controls,
is retained in the [evidence archive](evidence/card03/README.md).
No arithmetic/kernel optimization or speed gain is claimed.

The standard C1 HTTP pp2048/tg128 controls cover AR and MTP prose/repetitive
workloads at depth 0. All eight captured requests (including calibration and
warmup) retain exact request/history and output/cache/draft/work hashes and have
zero flagged available phases at 5% and 10 ms. The six matched CLI phases
(pp2048, tg128 and short-prompt decode in each mode) also have zero flags and
exact bench/prompt output hashes. CLI durations are derived from printed rounded
rates, separately per phase; no average replaces a request gate. These successful
bounded captures do not resolve the earlier functional timing uncertainty.

The metadata executable enumerates the artifact key, both option names and
speculative value; asserts all typed option fields; and typechecks the factory
signature without linking HIP or loading weights. This resolves the fixture gap
while explicitly leaving production metadata selection/parser/driver registration
to dependent cards. The [retained evidence](evidence/card03/README.md) records
all three agents, review dispositions and actual commands/results.

This throwaway experiment does not qualify other
models, final public API, Windows, exclusion builds or a shipped lock-mode HIP
artifact. No pending run or missing model counts as a pass.

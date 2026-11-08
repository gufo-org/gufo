# Qwen3.8 Flash-Next implementation notes

Developer detail behind the [public model page](../../../models/qwen3.8-flash-next.md).
[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md) · [ROCm](ROCM.md)

Hybrid recurrent/QSA mixture-of-experts text/image model on gfx1151.
Supported target: `unsloth/Qwen3.8-Flash-Next-GGUF`, **UD-Q4_K_XL** (four shards),
Unsloth Dynamic 3.1 revision `d96201194f10`. It keeps the tensor types and
shapes of the earlier `38bb39ee9782`, which the retained benchmark, quality
and F32-fixture evidence used; AR pp2048/tg128 speed is unchanged.
Optional shared-Q8 MTP predictor; optional BF16 vision projector.
Original unquantized-model and GGUF-conversion parity remain unqualified.

The loader discovers the remaining shards. Omit the speculative options for AR;
AR sessions allocate no predictor state even if a shared model has MTP loaded.
Adaptive MTP is default, with `--draft-tokens` capping 1–7 proposals. Eligible
requests draft over the first 98,304 token IDs plus a 1,024-row table containing
special tokens and other tokens already used by the request. The full head
remains the fallback when the anchor or two of the last 32 committed tokens
fall outside this vocabulary.

Single-session sampling, including the default thinking-on and thinking-off
presets, can use this smaller head with the existing acceptance/length
controller and AR backoff. Greedy requests, including penalties and
constraints, can use it in C1, when decoding alone and in jointly planned
batches; drafts ignore both, and verification applies them over the full
vocabulary. Greedy requests in unplanned multi-request batches retain the full
head.

The existing length controller and AR backoff apply to every request. The
target always verifies with its full vocabulary, including
exact sampled acceptance and residual correction. Sampled text can differ
across builds when proposals change. All-greedy C>1 batches retain their shared
planner and may use measured cycle costs. Each request keeps private caches,
rollback and RNG; snapshots preserve the learned acceptance history.
See [MTP qualification](QUALITY.md).

MTP decoding also proposes copies from the context, with one to eight sessions.
The context (committed tokens plus the chain so far) is matched against the
session's own earlier tokens by its last three tokens; when the best of the 16
most recent occurrences matches at least 12 tokens back for greedy requests
or 24 for sampled requests, the tokens that followed it fill the rest of the
chain and drafting stops. Greedy requests can
bypass the neural head when a match supplies at least three copies. Sampled
requests first make a neural proposal, then look for a copy extending it;
a repeated prefix alone can poorly predict new code or field values.
Their copy sources end at the request's starting context. Earlier parts of
the same new answer are not indexed for sampled copying; they become eligible
context on the next request. Greedy requests can also copy their current answer.
The request's first speculative step skips lookup to avoid delaying its first
token and records the start of generated text. Later matches cannot extend
back into the prompt's chat wrapper; a common tool header alone must not
qualify as a long matching answer. Copies
are verified like any draft: greedy text is byte-identical, and sampled
requests treat each copied token as a point-mass proposal under the same
rejection sampling, so outputs keep the target distribution (a seeded sampled
stream differs from one without copies, because random draws are consumed at
different points). Shorter matches
propose nothing; at temperature 1 an ungated 3-token rule displaced accepted
MTP drafts and was slower on every workload measured. A recently accepted
passage can resume after an edit with a five-token match; fewer than half of
the copied tokens being accepted disables that shorter gate, including after
a full match. Copies stop at EOS when the request enables it.
The index is private to the session and rebuilt after reset or snapshot
restore. Snapshots retain the request boundary and copy-resume cursor.
Lookup cycles and their following catch-up are excluded from the
batch planner's neural cost measurements. The planner measures short neural
chains before probing longer ones after copying. In all-greedy batches mixing
copies with new text, copied suffixes are bounded to half the verifier or the
planner's chosen width, whichever is larger, to limit pauses for other requests.
All-copy batches retain full width; sampled proposal streams are unchanged.
Timings report
`lookup_n` / `lookup_n_accepted`.

A request decoding alone can verify a copy wider than the MTP chain: greedy
requests, and sampled ones without a seed whose rows verify compactly. Copy-only
cycles start at the ordinary eight rows; two whole copies in a row at the
current width double it, up to 64 rows, and any rejected copy returns it to
eight. The wide pass (`Executor::ForwardWide`) runs the projections, routed
experts and head once over every row with the batched decode route's
row-independent arithmetic, and the recurrent state, attention and PLE as
consecutive segments of at most eight rows, so each row's logits equal those of
consecutive ordinary verifications (`wide_verification_test` checks the logits
and the continuation after cutting back to any kept prefix). Later segments
keep their starting state and snapshot rows in executor-owned buffers, about
270 MiB per segment past the first (1.9 GiB at 64 rows), allocated on first use
and reserved with the other lazy scratch. When the configured sessions leave
less device memory, the load narrows the widest pass to what fits instead of
refusing the session count. When more than eight rows are kept,
the draft block catches up on all but the last of them at once (K/V only, as
for prompt rows), so snapshots and batched cycles still hold at most eight
hidden rows. Seeded sampled requests and batched cycles keep the ordinary
width, so their proposals and random draws do not depend on scheduling.

The official template defaults to thinking on, `xhigh` effort and preserving
prior reasoning. Use the [reasoning controls](../../../server.md#reasoning-controls)
for explicit effort/thinking overrides. Native context is 262144. A larger
context needs YaRN and `--enable-yarn`: `gufo serve llm` refuses to
start with `--context` above 262144 without it, and `gufo bench` refuses a
workload whose depth plus prompt or generation tokens exceeds 262144. With
the flag, the largest factor is context / 262144 (`--context 409600` gives
1.5625). The flag is an error at or below 262144, where it would have no
effect. A session keeps the native rotation while it holds at most 262144
tokens, so short requests are unchanged by the flag; past that it takes the
smallest whole factor covering its positions (2, 3, ...), capped at the
largest, and its cached attention and indexer keys are re-rotated when the
factor changes. The factor changes the output distribution more as it
grows; in the first measurements of a fixed factor, multiple-choice accuracy
moved within noise at factors 1.5625 to 3.875. See the
[measurements](https://github.com/gufo-org/gufo/pull/350#issuecomment-6323981940),
the
[opt-in proposal](https://github.com/gufo-org/gufo/pull/350#issuecomment-6323982386)
and the
[accuracy check](https://github.com/gufo-org/gufo/pull/350#issuecomment-6234093998).
No GGUF key enables YaRN.
Prefill chunks whose visible context exceeds 262144 tokens use the per-token
attention kernel rather than the fused WMMA kernel. Memory grows with used
context and selected rollback depth (about 27.5 KiB per context token with
MTP, measured on gfx1151); admission reserves the configured capacity before
creating sessions.

## Images

Use this model's `mmproj-BF16.gguf`, discovered beside the target or selected
with `--mmproj`. PNG/JPEG CLI and HTTP requests use the
[same image interface](../../../models/qwen3.8-27b.md#images). Image state participates
in prefill, decoding, verification, multi-turn reuse and disk cache identity.
The predictor embeds shifted text IDs; visual information comes from target
hidden states and mRoPE. Image snapshots require matching prompt attachment.

## Tools and artifacts

Model tests are in `tests/models/qwen38_flash_next`, focused microbenchmarks in
`tools/qwen-flash`. [Quality](QUALITY.md) summarizes qualification and focused checks.
Build a microbenchmark with
`nix develop -c tools/bench/build.sh tools/qwen-flash/projection_plans.hip`;
`dense_blaslt_sweep.hip` times hipBLASLt on the dense prefill shapes.
No historical logit dump is required. New retained result summaries belong in
`artifacts/`; generated traces stay in the ignored top-level `artifacts/` tree.

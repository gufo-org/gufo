# Qwen3.8 Flash-Next

Hybrid recurrent/QSA mixture-of-experts text/image model on gfx1151.
Supported target: `unsloth/Qwen3.8-Flash-Next-GGUF`, **UD-Q4_K_XL** (four shards),
Unsloth Dynamic 3.1 revision `d96201194f10`. It keeps the tensor types and
shapes of the earlier `38bb39ee9782`, which the retained benchmark, quality
and F32-fixture evidence used; AR pp2048/tg128 speed is unchanged.
Optional shared-Q8 MTP predictor; optional BF16 vision projector.
Original unquantized-model and GGUF-conversion parity remain unqualified.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md) · [ROCm 10.1 status](ROCM.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision d96201194f103c906994f283f504209dc8d2a10c \
  --include "UD-Q4_K_XL/*" --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
  --include "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
nix build
MODEL=/path/to/first-target-shard.gguf
MTP=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

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
prior reasoning. Use the [reasoning controls](../../SERVER.md#reasoning-controls)
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

## Runtime residual projection

`--projection-vector CV.gguf` (serve only) optionally loads a file in the
llama.cpp control-vector format (`direction.N` F32 tensors for layers
`1..L-1`; a missing layer is never steered). At the FFN/MoE combine boundary
of each configured layer, every hyper-connection stream is updated as
`h <- h - s * (h . v) * v`, applied between the residual write and the next
mixer's grouped norm. The loader normalizes each direction to `v` and keeps
its norm as that layer's scale `s`, so a unit-norm file removes the
component along `v` exactly; any other norm scales the update instead (2
reflects the component) and the loader warns. This is a projection, not
llama.cpp's additive `--control-vector` steering, so the same file does
different things in the two engines. `direction.0`, tensors past the last
layer and all-zero directions are ignored with a warning; non-finite values
or a norm beyond the F32 range fail the load. The GGUF weights are never
modified. `--projection-layers LO-HI` restricts steering
to an inclusive sub-range with `1 <= LO <= HI <= L-1`; the default uses every
layer the file carries, and malformed ranges are rejected at startup.

The feature is off unless a request opts in: a chat request with
`"projection": true` steers that request, any request without the field does
not, and nothing carries into later requests. The flag is read from
device-side control state, so toggling between requests neither recaptures
decode graphs nor affects other sessions in a shared batch; with projection
off the steered kernels take the unpatched arithmetic, and logits are
bitwise identical to a server started without the option
(`projection_gate_test` gate 0). A request that enables projection on a
server started without `--projection-vector` fails with a 400 error rather
than silently running unsteered, and `--projection-vector` on a model
without the feature fails at startup. The value must be a boolean.

Prompt caching keeps the two states apart: while a vector is loaded, every
cache identity (including each image-prefix boundary) carries the request's
projection state, and the disk-cache compatibility identity records the
loaded vector's SHA-256 fingerprint and steered range, so changing either
invalidates persisted entries.

Steered layers and the direction table add one `[layers][hidden]` F32 device
buffer (480 KiB at 48 x 2560) plus one word of per-request control state.

Direction files declare optional model provenance via a
`controlvector.model_hint` string; a mismatching hint logs a warning and
loads anyway. The projection is checkpoint-agnostic in the sense that it
runs on activations, but its *effect* depends on the direction being
meaningful for the loaded model: a direction recovered from a different
model family may do nothing, or may remove behavior it was not measured
against, including safety behavior and factual reliability. Measure any
loaded vector against your own evaluation set; a single behavioral metric
is not a quality signal.

## Images

Use this model's `mmproj-BF16.gguf`, discovered beside the target or selected
with `--mmproj`. PNG/JPEG CLI and HTTP requests use the
[same image interface](../qwen3.8-27b/README.md#images). Image state participates
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

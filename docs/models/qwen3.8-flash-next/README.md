# Qwen3.8 Flash-Next

Hybrid recurrent/QSA mixture-of-experts text/image model on gfx1151.
Supported target: `unsloth/Qwen3.8-Flash-Next-GGUF`, **UD-Q4_K_XL** (four shards).
Optional shared-Q8 MTP predictor; optional BF16 vision projector.
Original unquantized-model and GGUF-conversion parity remain unqualified.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf" \
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
controller and AR backoff. Greedy requests without penalties or constraints
can use it in C1 and jointly planned batches. Greedy penalties/constraints and
unplanned C>1 requests retain the full head.

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

The official template defaults to thinking on, `xhigh` effort and preserving
prior reasoning. Use the [reasoning controls](../../SERVER.md#reasoning-controls)
for explicit effort/thinking overrides. Native context is 262144; YaRN extension
is unsupported. Memory grows with used context and selected rollback depth;
admission reserves the configured capacity before creating sessions.

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

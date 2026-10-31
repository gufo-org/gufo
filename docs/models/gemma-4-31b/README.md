# Gemma 4 31B

Dense Gemma 4 text model on gfx1151: 60 layers, five sliding-window (1024)
layers per global layer, tied embedding and a final logit softcap.
Supported targets: `unsloth/gemma-4-31B-it-GGUF`, **UD-Q4_K_XL** and
**UD-Q8_K_XL** (single files), with the optional `gemma4-assistant` MTP
drafter from the same repository. UD-Q8_K_XL is Q8_0 except the F16 Q/K and
MLP projections of layers 1, 52, 53, 57, 58 and 59 (5.1 of its 35 GB), which
run on binary16 kernels: a batch-invariant GEMV up to 16 rows and the binary16
WMMA GEMM in prefill.
Image input uses the BF16 vision sidecar from the same repository
(`mmproj-BF16.gguf`, found beside the model or passed with `--mmproj`); see
[VISION.md](VISION.md). `--image-tokens` sets the soft tokens per image: 70,
140, 280 (default), 560 or 1120; larger budgets keep more detail at a higher
prefill cost. The QAT (Q4_0) checkpoint has its own card:
[Gemma 4 31B QAT](../gemma-4-31b-qat/README.md).

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix develop -c hf download unsloth/gemma-4-31B-it-GGUF \
  --revision c1ac76e99d5513b141e8adde7288b85c3f9c32ec \
  --include gemma-4-31B-it-UD-Q4_K_XL.gguf MTP/mtp-gemma-4-31B-it-Q8_0.gguf \
  mmproj-BF16.gguf \
  --local-dir models/gemma-4-31b
nix build
MODEL=models/gemma-4-31b/gemma-4-31B-it-UD-Q4_K_XL.gguf
MTP=models/gemma-4-31b/MTP/mtp-gemma-4-31B-it-Q8_0.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo prompt --model "$MODEL" --image photo.jpg -p "Describe it."
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --context 131072
```

Omit the speculative options for autoregressive decoding. `-d`/`--draft-tokens`
caps the drafts per cycle (1–15, default 15), and `--draft-policy` decides how
many a cycle verifies:

- `calibrated` (default): a draft is verified while its estimated probability
  of acceptance is worth one more verification row and drafter step at the
  current context depth. The estimate maps the drafter's entropy through a
  table learned from earlier verifications; the costs are fixed gfx1151
  measurements.
- `confidence`: drafting stops when the product of the drafter's raw
  confidences falls below a fixed floor (0.5 greedy; 0.3 sampled, at most four
  drafts); the previous default, tuned on the UD-Q4_K_XL drafter.
- `fixed`: always draft up to `--draft-tokens`.

`--min-draft-tokens` drafts that many before a policy may stop. The calibrated
policy learns across requests by default; `--draft-calibration request`
restarts it with every request and prices a sampled cycle's drafts as if it
ran alone, so a seeded sampled request replays exactly, batched or not.
`--draft-trace PATH` (serve, prompt and bench) appends one JSON line per
verification cycle: rows, copies and siblings, drafter steps, accepted and
emitted tokens, the policy's prices, expected tokens, per-draft signals and
acceptance estimates, the declined proposal's gain and cost, and steady-clock
times for drafting, verification and emission. Every verified row also gets
its acceptance probability given the rows before it (0/1 greedy, min(1, p/q)
sampled), so a `--draft-policy fixed --draft-tokens 15` trace labels complete
chains; `gemma4_draft_replay` (tools/gemma4/draft_replay.cpp) replays them
and live traces through the production policy. Tracing does not change
decoding.
When the recent context repeats an earlier
passage of at least 12 tokens (a rewritten file, a quoted log), the tokens that
followed it fill the remaining draft slots and are verified like drafts (prompt
lookup, shared with gufo-org/gufo#295 and #613); with at least three of them
the drafter does not run that cycle. Where other earlier occurrences match at
least half as far back but continue differently (repeated log timestamps or
declarations), only the tokens they all share are copied. After a copy that
kept at least half of its tokens, five matching tokens near where it left off
resume it, so output that edits a copied line rejoins it. A request's first
cycle copies nothing, matches do not reach back past the request's start, and
copies stop at a stop token. Requests report proposed and accepted copies as
`lookup_n` / `lookup_n_accepted`. `--sessions N` serves up to N concurrent requests; up to eight
decode in one forward. The drafter attends to the target's own KV cache,
so it adds no per-session cache.
With a drafter loaded, the target's projection weights are reordered at
load into 128-row groups that the verification kernel reads as one stream
(same bytes and results; MTP tg128 +3.4% on the 31B quants, about 0.3 s
more load time). Greedy speculative output equals single-token decoding of
the same configuration token for token; an AR-only server uses a faster one-row
projection kernel with a different FP32 summation order, so long greedy
completions of the two configurations can differ. Sampled requests verify
drafts by p/q rejection, so emitted tokens follow the target distribution.

The chat template is Unsloth's variant (SHA-256 `845f1ee4…73d1b`), compiled
into the engine and checked against Jinja renders. A GGUF carrying another
template (fine-tunes often ship an older revision) loads with a warning and
renders with this one. It defaults to thinking
off; the [reasoning controls](../../SERVER.md#reasoning-controls) turn the
thinking channel on, returned as `reasoning_content`. Gemma tool calls
(`<|tool_call>call:NAME{…}<tool_call|>`) are returned as OpenAI `tool_calls`.

Native context is 262144. Global layers keep every token, sliding layers a
fixed ring. A global layer's K and V come from one projection, so its cache
stores V and only the 128 rotated key dims of each head (50 KiB per token
across the ten global layers; the other key dims are V scaled by `k_norm`).
A 131072 context needs about 6.3 GiB of global KV plus 2.5 GiB of rings next
to the 17.5 GiB of weights.
Prompt snapshots feed the RAM and `--cache-disk` prompt caches; a trimmed or
edited history re-prefills from the latest retained checkpoint before the
change, so dropping the oldest messages costs a full prefill.

`--prefill-weights` (serve, prompt and bench) trades memory for prefill speed
by keeping a second copy of the dense projections next to the GGUF weights,
which decoding keeps reading:

| Mode | Copy | 31B UD-Q4_K_XL memory | pp2048 d0 |
| --- | --- | ---: | ---: |
| `off` (default) | none | 22.0 GiB | 581 tok/s |
| `compact` | K-quant codes with binary16 scales in the GEMM's register order | +18.9 GiB | +3.3% |
| `decoded` | binary16 weights for the 5376-wide inputs, `compact` elsewhere | +43.0 GiB | +6.5% |

`decoded` also serves the QAT quant (Q4_0, +3.6%). UD-Q8_K_XL gets no copies:
its default prefill already fuses GeGLU into the Q8_0 [gate | up] GEMM and
matched the binary16 copies' GEMMs. Prefill logits are bit-identical to `off` in both modes on all three
31B quants. Prompts under 512 tokens skip the binary16 copies. On the
26B-A4B both modes copy only the K-quant expert [gate | up] weights, in the
compact layout (UD-Q4_K_XL: +1.0% pp2048 for +9.0 GiB). Its dense projections
are Q8_0 or 2816 wide, where binary16 copies measured 1-2% slower, and the QAT
quant's Q4_0 experts get no copy.

## Tools and artifacts

Tests live in `tests/models/gemma4` (`gemma4.*` in CTest; set
`GUFO_GEMMA4_MODEL` and `GUFO_GEMMA4_MTP_MODEL` for the model-backed ones).
`tools/gemma4` holds the tokenizer/template golden generators, the
teacher-forced llama.cpp logit dumper (`llama_logits.cpp`),
`compare_logits.py` and `draft_policy_bench.py`, an HTTP A/B of
`--draft-policy` settings on prose, story and code workloads.

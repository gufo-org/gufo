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
When the recent context repeats an earlier
passage of at least 12 tokens (a rewritten file, a quoted log), the tokens that
followed it fill the remaining draft slots and are verified like drafts (prompt
lookup, shared with gufo-org/gufo#295). `--sessions N` serves up to N concurrent requests; up to eight
decode in one forward. The drafter attends to the target's own KV cache,
so it adds no per-session cache.
Greedy speculative output equals single-token decoding of the same
configuration token for token; an AR-only server uses a faster one-row
projection kernel with a different FP32 summation order, so long greedy
completions of the two configurations can differ. Sampled requests verify
drafts by p/q rejection, so emitted tokens follow the target distribution.

The chat template is Unsloth's variant (SHA-256 `845f1ee4…73d1b`), compiled
into the engine and checked against Jinja renders. It defaults to thinking
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

## Tools and artifacts

Tests live in `tests/models/gemma4` (`gemma4.*` in CTest; set
`GUFO_GEMMA4_MODEL` and `GUFO_GEMMA4_MTP_MODEL` for the model-backed ones).
`tools/gemma4` holds the tokenizer/template golden generators, the
teacher-forced llama.cpp logit dumper (`llama_logits.cpp`),
`compare_logits.py` and `draft_policy_bench.py`, an HTTP A/B of
`--draft-policy` settings on prose, story and code workloads.

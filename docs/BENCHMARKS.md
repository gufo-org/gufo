# Benchmark Methodology

How Gufo benchmarks are computed, and which utilities produce them. This
file documents the method; per-model results live in `benchmarks/<model>/`
(readmes), never as raw committed artifacts (`artifacts/` is gitignored).
This method is teacher-forced and matched-token; it is not free-running
evaluation. Capability evaluation (free-running, answer-graded) is defined
in [EVAL.md](EVAL.md) and runs through the OpenAI-compatible serving route.

## Pipeline (tools/)

Benchmarks are the last step of the offline conversion toolchain. Each step
has one utility; a benchmark consumes the artifacts the earlier steps produce:

```bash
# 1. download safetensors + tokenizer -> artifacts/source
# 2. validate safetensors, write source manifest
tools/quant/gufo-inspect.py --source artifacts/source --revision <sha> \
  --out artifacts/work/source-manifest.json

# 3. capture full-precision teacher logits + perplexity (matched-token)
tools/quant/gufo-capture.py --source artifacts/source \
  --suite tools/quant/suites/teacher.json --out artifacts/teacher

# 4. quantize LM linear projections to SHQ4-T16 U4Z G64
tools/quant/gufo-quantize.py --source artifacts/source \
  --out artifacts/quant --plan artifacts/work/quantization-plan.json

# 5. benchmark: candidate-vs-teacher quality + prefill/decode speed
tools/quant/gufo-bench.py --source artifacts/source --quant artifacts/quant \
  --suite tools/quant/suites/teacher.json --teacher-artifact artifacts/teacher
```

Key utilities and their roles:

- `tools/quant/gufo-capture.py` — teacher-forced full-precision logit dump. Logits are
  chunked by position, zstd-compressed, checksummed into an artifact dir
  (`manifest.json`, `tokens.u32`, `logits-*.f32.zst`, `metrics.json`).
  This is the teacher oracle; it is captured once and reused.
- `tools/quant/gufo-quantize.py` — deterministic SHQ4 conversion. Records per-tensor
  reconstruction stats (max abs err, rmse, mean abs err) in the plan.
- `tools/quant/gufo-bench.py` — the benchmark itself. Runs the candidate forward pass,
  compares candidate logits against the captured teacher artifact, and times
  prefill/decode on both.
- `tools/quant/suites/<model>.json` — prompt suite (schema `gufo.suite.v1`):
  a small fixed set of prompts, tokenized once with the pinned tokenizer.

## Quality method (matched-token, per position)

Per `docs/TESTING.md`: quantization comparisons are teacher forced, not
free-running. One tokenization; the same input prefix drives both teacher and
candidate; compare their full next-token distributions; append the
predetermined evaluation token, never each model's sampled token. This
isolates intrinsic quantization error from trajectory divergence.

For each prompt, at every position:

```text
p_t = softmax(z_t)   # teacher logits (captured artifact, f32)
p_c = softmax(z_c)   # candidate logits (forward run)
KL  = sum_v p_t[v] * (log p_t[v] - log p_c[v])
NLL = -log(p_c[target_token])
```

Aggregates over all scored positions (`quality.py`):

- KL: mean, median, p95, p99, p99.9, max
- teacher-token NLL and corpus perplexity
- top-1 agreement; top-5 / top-k set overlap
- candidate probability on the teacher top-1 token
- rank displacement of the teacher top-1 token
- max and RMS diff of normalized log probabilities
- count of non-finite values

Current slice reports mean/median/p95/p99/max KL, top-1 agreement, and
teacher + candidate perplexity.

### Position count and Artifact Schema

`positions` in a report equals the total number of scored next-token positions
across all suite prompts. The captured teacher logit artifact follows the
`gufo.logit-artifact.v1` schema with chunked zstd compression, dynamic
vocabulary extraction for Qwen3.5-4B and Qwen3.8-27B, and SHA-256 chunk
manifest validation.

The committed suite identity is:

```text
tools/suites/teacher.json
sha256: b33862883e78e7500cc8553cffe68ec8c44ca5ac1ceca8cbccb3070f6d42b131
```

KL aggregates are computed over positions, not prompts. A changed suite hash or
position count creates a new benchmark identity and must not be compared as the
same run.

### Per-layer breakdown (planned)

The first slice measures whole-model, whole-prompt quality: one KL per scored
position, aggregated model-wide. It does not isolate which layer or tensor
contributes the KL tail. Per-layer and per-tensor attribution (layer-output
error, per-layer KL, first-divergent-layer search) is a planned refinement;
`gufo-quantize.py` already records per-tensor reconstruction stats, and
`docs/TESTING.md` T3 documents layer-boundary capture. Imatrix search was
implemented before this planned prerequisite; per-layer attribution is now a
required catch-up gate before selecting the Qwen3.8-27B production recipe or
promoting native quantized kernels.

## Speed method (CPU reference runtime)

Speed is measured only after logits are comparable
(`docs/TESTING.md`: correctness before performance counts).

- Prefill: full-prompt forward (all suite prompts), summed tokens.
- Decode: single-token forward, repeated.
- Both teacher and candidate run through the same pinned torch reference path,
  timed with `time.perf_counter`, median of `--repeats` runs (default 2).

Important: candidate speed here is the reference runtime after
dequant-reload-to-bf16 — it measures SHQ4 dequant cost, not Gufo kernels.
Real kernel speed (HIP gfx1151 / XDNA2 AIE2P consuming packed planes
directly) is a separate, later milestone. Kernel benchmarks will record the
backend and kernel name per timing row.

## Focused GPU kernel benchmarks

`gufo-kernel-bench` measures production HIP entry points without loading a
model. It covers matrix operations, attention, DeltaNet, normalization, and
elementwise kernels. Correctness sentinels run outside the timed interval.

```sh
./result/bin/gufo-kernel-bench --list
./result/bin/gufo-kernel-bench \
  --case decode-attention \
  --context 4096,8192,12288,16384 \
  --warmup 5 \
  --repetitions 30 \
  --json
```

The JSON report records shapes, raw HIP-event samples, summary latency,
dispatch choices, and a privacy-safe machine fingerprint. Collect profiler
resource data in a separate `rocprofv3` pass.

`tune_hipblaslt` creates an optional hardware/ROCm-bound plan database.
`tools/qwen27b/deltanet_bench.hip` compares exact recurrence and state-only
replay kernels at block lengths 1-16. Full-logit rollback checks live in the
Qwen27B model suite.
Generated databases, traces, and replay reports are local artifacts and are
not committed. See [Performance Engineering](PERFORMANCE.md) for commands and
promotion rules.

## Serving benchmark

`tools/serving/gufo-serving-bench.py` is the canonical HTTP serving harness. It sends
synchronized streamed Chat Completions requests at C=1, C=2, and C=4 by
default. Its `gufo.serving-benchmark.v1` artifact reports direct server-stage
prefill/decode throughput, scheduler and client TTFT, token ITL, whole-request
throughput, and aggregate concurrency throughput as distinct metrics.

The harness consumes the terminal `usage.gufo` metrics emitted by
`gufo serve llm`, retains every request and round sample, and summarizes
median, p95, and p99 behavior. It embeds the canonical machine fingerprint and
source revision while excluding endpoint hosts, prompts, generated text, local
paths, raw token IDs, and timestamps.

## Machine Fingerprint & Artifact Binding

All diagnostic and benchmark artifacts must embed a canonical machine fingerprint
and its SHA-256 identifier (`fingerprintId`):

- **Canonical Identity**: Records pinned CPU topology, gfx1151 GPU identity/CUs,
  XDNA2 NPU identity, kernel drivers (`amdgpu`, `amdxdna`), ROCm/HIP, and XRT toolchain pins.
- **Privacy Redaction**: Hostnames, usernames, process secrets, timestamps, and local
  user paths are strictly excluded from the canonical identity and forbidden in benchmark artifacts.
- **Validation**: Artifacts can be validated with `gufo diagnose --validate-artifact <path>`,
  which checks schema compliance (`schemaVersion: 1.0.0`), re-hashes canonical fields,
  and rejects mismatched fingerprints or incompatible architectures.

## Reporting rules

Every report records, at minimum:

- Machine `fingerprintId` referencing the canonical machine fingerprint.
- Source repo + immutable revision, source storage dtype (bf16), accumulation
  contract.
- Candidate: which tensors quantized, the SHQ4 scheme (T16, U4Z, group size),
  tensor count, artifact size.
- Suite hash and scored position count.
- Oracle runtime and version.
- `--json` output is machine-readable; human form prints speed then quality.

A benchmark result is only comparable against another result with the same
machine fingerprint, source revision, suite, tokenizer, and reference runtime. Do not average
numbers across revisions or suites.

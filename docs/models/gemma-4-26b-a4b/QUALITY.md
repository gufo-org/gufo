# Gemma 4 26B-A4B quality

**Decode matches the scalar reference up to routing near-ties; greedy MTP
matches single-token decoding; prefill stays within 4e-5 of the reference,
where llama.cpp lands at 0.4.** Unsloth UD-Q4_K_XL, UD-Q6_K_XL and UD-Q8_K_XL
targets with the Unsloth Q8_0 `gemma4-assistant` drafter;
[identities](artifacts/model-identities.json). The arbiter is Gufo's scalar
CPU implementation of the llama.cpp `gemma4` graph, routed experts included,
over the same GGUF (FP32 activations, binary16 KV,
`src/models/gemma4/reference.cpp`). These are not unquantized-model or
GGUF-conversion checks. Measured September 28, 2026; text rows remeasured
September 29 after the decode GeGLU fusion. UD-Q8_K_XL measured September
30; the reference reads its layer-29 BF16 weights directly, so its rows also
cover the load-time binary16 rewrite.

| Check | UD-Q4_K_XL | UD-Q6_K_XL | UD-Q8_K_XL |
| --- | --- | --- | --- |
| GPU decode (FP32 activations) vs scalar reference, 22-token chat prompt | Mean KL 7.5e-7 (limit 1e-4), max 9.0e-6, top-1 22/22 (1.1e-6 before the own-key attention step, 2026-10-05) | Mean KL 4.8e-7, max 5.0e-6, top-1 22/22 | Mean KL 8.8e-7, max 6.2e-6, top-1 22/22 |
| Eight-row verification | Bit-identical to single-token decode | Bit-identical to single-token decode | Bit-identical to single-token decode |
| GPU prefill (binary16 activations, binary16 WMMA attention) vs scalar reference, same prompt | Mean KL 3.7e-5 (limit 1e-3), top-1 21/22 (a near-tie) | Mean KL 2.9e-5, top-1 22/22 | Mean KL 3.3e-5, top-1 22/22 |
| Bulk prefill vs exact rows, 1542-token conversation past the window and ring | Mean KL 0.027 (limit 0.06), top-1 1489/1542 | Mean KL 0.019, top-1 1494/1542 | Mean KL 0.031, top-1 1480/1542 |
| Bulk prefill vs exact rows, 1353-token repetitive prompt (reported, not a gate) | Mean KL 1.71, top-1 780/1353 | Mean KL 1.84, top-1 766/1353 | Mean KL 2.10, top-1 720/1353 |
| Image prompt, 624×960 chart at 260 soft tokens, 96 teacher-forced positions from `<image\|>`, vs the scalar reference fed the reference encoder's embeddings | Mean KL 0.029 (80 answer positions 0.0039), top-1 95/96; llama.cpp `use_gelu` 0.251 (0.0125) | Mean KL 0.129 (answer 0.0012), top-1 93/96; llama.cpp 0.474 (0.0104) | Not measured |
| Image sessions | `gemma4.vision_session` (changed image, rewind, extension, snapshot, chunking) passes | Passes | Passes |
| Greedy MTP vs single-token decode with the drafter loaded, three prompts, plus prompt-lookup copies; batched sessions vs their own decode | Identical token IDs (`gemma4.target`) | Identical token IDs | Identical token IDs |
| Sampled MTP | Seeded replay repeats the same tokens (`gemma4.target`) | Same | Same |
| Session state | Prefix extension, rewind and snapshot restore after a ring wrap continue bit for bit; snapshots sharing position blocks hold the same payload as fresh copies, also after rewritten or ring-wrapped blocks (`gemma4.target`) | Same | Same |

The image rows' largest differences sit on the user's question right after
the image (positions 266–280, up to KL 8.7 on UD-Q6_K_XL at one token), which
the Gufo text model fed the reference's embeddings shares (0.028 / 0.182
overall). The encoder matches `vision::Reference` to 0.27% embedding relative
RMS; with BF16 projection inputs it was 1.16% and the Q4 row 0.112. llama.cpp
is no oracle here: its Q8_1 activations (below) already cost KL 0.42 on text.

## Routing sensitivity

Top-8 expert selection is discrete. Two exact FP32 decodes that differ only
in the summation order of one RMSNorm (256- versus 1024-thread blocks) flip
near-tied expert choices. They land at mean KL 1.3e-6 and 2.7e-5 from the
reference, with single positions up to 3e-4. The expert-model limits are
therefore wider than the dense 31B's (decode 1e-4, prefill 1e-3, conversation
0.06, recorded at introduction).

## Activation precision

This checkpoint's activations carry large per-block outliers, so Q8_1
activations (32-value blocks with one scale) cost far more than on the 31B.
All rows below are the scalar reference with its projection inputs rounded
as listed, against unrounded FP32, on the 22-token prompt:

| Projection inputs rounded | Mean KL | Top-1 |
| --- | ---: | ---: |
| Every projection, Q8_1 | 0.74 | 19/22 |
| Attention output only, Q8_1 | 0.44 | 18/22 |
| K / V only, Q8_1 | 0.12 / 0.066 | 20/22 / 21/22 |
| Dense MLP gate / up / down only, Q8_1 | 0.064 / 0.066 / 0.085 | 20/22 each |
| Expert gate/up / down only, Q8_1 | 0.023 / 0.017 | 20/22 / 22/22 |
| Every projection, BF16 | 0.066 | 20/22 |
| Every projection, binary16 | 3.7e-5 | 21/22 |

Gufo therefore prefills with binary16 activations. llama.cpp b11069 (ROCm)
runs Q8_1 activations. Its teacher-forced logits on the same tokens sit at
mean KL 0.42 (max 4.4, top-1 18/22) from the FP32 reference.

Its layer-0 intermediates confirm the reference's semantics. Router logits
agree to 9.5e-7 relative, and the top-8 experts, their weights and the expert
input norm (1.1e-7) match exactly. Its dense projections match a Q8_1
emulation of the reference to 1.4e-4. Its expert projections differ from the
reference by 1–6%, beyond what Q8_1 rounding explains.

## Other quants

Measured October 11, 2026 with `gemma4.target` against the same scalar
reference over each GGUF ([formats](README.md#other-quants)). The QAT
checkpoint ran with its own Q4_0 drafter; a fine-tune of the 26B-A4B in
llama.cpp's Q4_K_M and Q6_K mixes (imatrix quants) with the Unsloth Q8_0
drafter.

| Check | QAT UD-Q4_K_XL | Fine-tune Q4_K_M | Fine-tune Q6_K |
| --- | --- | --- | --- |
| Decode vs scalar reference | Mean KL 1.5e-7, top-1 22/22 | 5.7e-7, 22/22 | 4.1e-7, 22/22 |
| Prefill vs scalar reference (limit 1e-3) | 9.7e-6, top-1 22/22 | **0.013**, 21/22 | **0.0026**, 22/22 |
| Bulk prefill vs exact rows, 1542-token conversation (limit 0.06) | 0.0015, 1523/1542 | 0.024, 1481/1542 | 0.019, 1493/1542 |
| Greedy MTP and batched sessions vs decode; sampled replay; session state | Pass | Pass | Pass |

The fine-tune's quants exceed the short-prompt prefill limit set on the Unsloth
quants. On Q4_K_M a single position carries 0.25 of the 22 positions' summed
KL (mean 0.0114 of 0.0131): its residual departs from the reference in layer
29, a top-8 routing flip ([routing sensitivity](#routing-sensitivity)).
Comparing per-layer residuals with the reference on this prompt, the
fine-tune also flips at other positions in layers 9 and 18 (relative RMS up
to 6e-2 afterwards, UD-Q4_K_XL at most 2e-2 after its one flip in layer 11),
and its error before any flip is 1.5–2 times UD-Q4_K_XL's (K-quant instead
of Q8_0 attention and dense projections). The conversation check, over 1542
positions, sits at the Unsloth quants' level. Q6_K, whose formats the engine
already ran before Q5_0 support, behaves the same way, so this is not
specific to Q5_0. The limit is unchanged; these two quants fail it.

## Reproduce

```sh
export GUFO_GEMMA4_MODEL=/path/to/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
export GUFO_GEMMA4_MTP_MODEL=/path/to/mtp-gemma-4-26B-A4B-it.gguf
nix develop -c cmake --build --preset gpu-test --target gemma4_target_test \
  gemma4_moe_ops_test gemma4_projection_ops_test
nix develop -c ctest --preset gpu-full -R '^gemma4\.(target|moe_ops|projection_ops)$' \
  --output-on-failure
```

`gemma4.moe_ops` checks routing against FP64, every routed format (Q4_K,
Q5_K, Q6_K, Q8_0, Q4_0 gate/up; Q5_1, Q8_0, Q4_0, Q5_0 down) against FP64
dots and batch invariance from one to sixteen rows, the routed prefill GEMMs,
the dense Q4_0 / Q5_0 GEMV and prefill GEMM at the 2112-wide down
projection, and the mixture epilogue. Reference and llama.cpp logits:
`gemma4_reference_probe --model M --chat "Write one sentence about the sea
near Genoa." --half-kv --tokens-out T.i32 --logits-out R.g4lg`, then
`llama_logits --model M --tokens T.i32 --output L.g4lg`
(`tools/gemma4/build_llama_logits.sh`) and
`tools/gemma4/compare_logits.py R.g4lg L.g4lg`.

## Benchmark method

Gufo October 17, 2026 (revision a9ff98e, `nix build`, binary SHA-256 prefix
d92d910545862f6b, every table), llama.cpp September 28 (not re-measured); one warmed sample per point (sampled MTP: five seeds),
greedy, thinking off, the same driver workloads, server flags and table grid
as the [31B](../gemma-4-31b/QUALITY.md#benchmark-method), per quant. Gufo
drafts up to 15 tokens under the calibrated policy (the default), llama.cpp up
to four. Loading drops the
page cache (`sync; echo 3 > /proc/sys/vm/drop_caches`) before each launch.
Loading and memory follow the 31B's projector-free and calibration-free setup.
Every Gufo multi-user MTP completion (C1–C8, both workloads, every quant)
matches the isolated AR C1 hash (regenerated October 17), and every multi-user
AR completion matches its AR C1 hash except one request each at UD-Q6_K_XL
C4–C8 (3 of 4, 5 of 6, 7 of 8; C2 matches; October 7: 1 of 2, 3 of 4, 3 of 6). Without the drafter one
session decodes with the faster AR GEMV and a batch with the small-batch or
WMMA kernels, which round differently; this prompt sits on a greedy near-tie
(neighbouring prompt lengths match at C1 and C2, and every batch width agrees
with the others). `"ar_exactness": "reported"` in bench.json records the
count instead of failing the table. Commands, counts and server flags are
recorded per row in the artifacts.
Against the October 7 build: prefill +12–18% from 16K (UD-Q4_K_XL AR, d0
within 1.5%), AR decode within 0.6%, MTP decode +3–13% from 8K, memory
+0.07 GiB. Shallow mixed MTP cells move with acceptance: the greedy text
changed, so 4K drafts 1.37 tokens per step against 1.61.

UD-Q8_K_XL tables: Gufo October 17 (same build), llama.cpp September 30, 2026, a reduced grid
under the same driver, workloads and server flags: single-user depths 0, 4096,
16384 and 32768, sampled MTP at depth 0 (five seeds), one and two users,
memory and loading. Gufo drafts under its calibrated policy. Every Gufo
multi-user completion matches its C1 hash (AR against AR C1, MTP against MTP
C1); the AR C1 hashes llama.cpp's MTP rows are compared with were qualified
once with an isolated C1 AR run (`multi-*-q8-gufo-ar.json`).

The shared DPP reduction change of 2026-10-19 preserves the existing
attention operator outputs byte for byte against 04d70a14, including this
model’s two-KV-head layout and all verification widths through 16.
`gemma4.target` with UD-Q4_K_XL and the Unsloth Q8_0 MTP drafter passes:
decode KL 7.52968e-7, eight-row KL 2.58145e-5 (both 22/22 top-1), prefill KL
3.71873e-5 (21/22), long-conversation KL 0.0264956. Greedy equivalence,
siblings, sampled replay and snapshot checks pass. These are candidate
quality measurements; the operator A/B establishes unchanged attention
arithmetic. [Raw results](../gemma-4-31b/artifacts/mtp-verification-2026-10-19.json).

## Coalesced attention qualification (2026-10-20)

The shared `422fbe4d` attention change is byte-identical to `519dfe9c` /
`eb17a526` across all 160 supplied row/depth/head-layout cases, including
this model's two-KV-head layout and normal/10×/30× queries. Rebuilt current
`gemma4.attention_ops`, `gemma4.kernel_ops` and `gemma4.draft_policy` pass;
commands and full outputs are in the [shared artifact](../gemma-4-31b/artifacts/mtp-attention-e2e-2026-10-20.json)
and the [31B qualification](../gemma-4-31b/QUALITY.md#coalesced-attention-qualification-2026-10-20).

The supplied candidate UD-Q4_K_XL / Unsloth Q8_0 `gemma4.target` log is
reused: decode/eight-row mean KL 7.52968e-7 / 2.58145e-5 (22/22 top-1),
prefill 3.71873e-5 (21/22), long-conversation 0.0264956. Greedy equivalence,
siblings, sampled replay and snapshot checks pass in that log. Its reported
repetitive-prefill discrepancy remains visible in the raw numerical output;
no tolerances or goldens change.

Production CMake `release` baseline/candidate SHA-256 prefixes are
`a8038e8407374cae` / `bb1dd37daf0459c8`; identical compiler/linker
fingerprints. Target `ef728c8e0c337fd1…`, draft `6326fb9f5e487aa8…`;
full file hashes, sizes and commands are recorded in the artifact. Native
pp2048/tg128 runs d16384 then d32768 in one process per ABBA arm, calibrated
cap 15, seed 1, with the CLI prefill warmup; printed rates have 0.01-tok/s
resolution. HTTP uses fresh `--sessions 1 --context 40960 --think off`
servers, greedy seed 1, a short 16-token warmup, summary/essay400 then
code-edit512. The fixed-policy 11-draft control warms code128 before the
cached code512 request; lookup/tail rows remain ragged. Loading is excluded.

Every HTTP completion hash and prompt/cache/output count matches within its
case: summary cold 31,290 tokens; essay reuses 30,720 of 31,297, code edit
22,528 of 31,161; the fixed-11 control reuses all 31,161. This establishes
baseline/candidate text agreement for these histories. Independent upstream
accuracy remains covered by the existing teacher checks. All paired
request/phase timing checks remain within 5% and 3 ms. Small numerical
gains below that threshold are reported as such; no material speedup is
qualified. Native and HTTP timings have separate scopes, and overlapping
server stages are not summed into request wall time.

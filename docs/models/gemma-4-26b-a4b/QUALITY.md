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
Q5_K, Q6_K, Q8_0 gate/up; Q5_1, Q8_0 down) against FP64 dots and batch
invariance from one to sixteen rows, the routed Q6_K prefill GEMM and the
mixture epilogue. Reference and llama.cpp logits:
`gemma4_reference_probe --model M --chat "Write one sentence about the sea
near Genoa." --half-kv --tokens-out T.i32 --logits-out R.g4lg`, then
`llama_logits --model M --tokens T.i32 --output L.g4lg`
(`tools/gemma4/build_llama_logits.sh`) and
`tools/gemma4/compare_logits.py R.g4lg L.g4lg`.

## Benchmark method

Gufo October 7, 2026 (revision 2a24b0c, `nix build`, binary SHA-256 prefix
d1eb64e09180774b, every table), llama.cpp September 28; one warmed sample per point (sampled MTP: three seeds),
greedy, thinking off, the same driver workloads, server flags and table grid
as the [31B](../gemma-4-31b/QUALITY.md#benchmark-method), per quant. Gufo
drafts up to 15 tokens under the calibrated policy (the default), llama.cpp up
to four. Loading drops the
page cache (`sync; echo 3 > /proc/sys/vm/drop_caches`) before each launch.
Every Gufo multi-user MTP completion (C1–C8, both workloads, every quant)
matches its MTP C1 hash, and every multi-user AR completion matches its AR C1
hash except UD-Q6_K_XL C2–C6 (1 of 2, 3 of 4, 3 of 6; C8 matches). Without the drafter one
session decodes with the faster AR GEMV and a batch with the small-batch or
WMMA kernels, which round differently; this prompt sits on a greedy near-tie
(neighbouring prompt lengths match at C1 and C2, and every batch width agrees
with the others). `"ar_exactness": "reported"` in bench.json records the
count instead of failing the table. Commands, counts and server flags are
recorded per row in the artifacts.
Against the October 5 build (same prompts and completions), repetitive MTP tg
reads 0.5–3.5% lower at depth although GPU kernel time per decode cycle is
unchanged (CLI profile). In the server the power manager now holds a higher
shader clock and a lower memory-fabric clock during decode (26B at 32K:
2,680 against 2,390 MHz; FCLK 1,640 against 1,700 MHz, at 139 against
128 W), and decode is bandwidth-bound. Requests still finish about 230 ms
sooner end to end (first token 150 ms earlier, no snapshot pause after it).

UD-Q8_K_XL tables: Gufo October 7 (same build), llama.cpp September 30, 2026, a reduced grid
under the same driver, workloads and server flags: single-user depths 0, 4096,
16384 and 32768, sampled MTP at depth 0 (three seeds), one and two users,
memory and loading. Gufo drafts under its calibrated policy. Every Gufo
multi-user completion matches its C1 hash (AR against AR C1, MTP against MTP
C1); the AR C1 hashes llama.cpp's MTP rows are compared with were qualified
once with an isolated C1 AR run (`multi-*-q8-gufo-ar.json`).

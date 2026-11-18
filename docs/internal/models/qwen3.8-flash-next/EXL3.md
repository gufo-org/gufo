# Qwen3.8 Flash-Next EXL3 checkpoints

Developer detail behind the
[public model page](../../../models/qwen3.8-flash-next.md#exl3-checkpoints).
The same Flash-Next runtime serves the Q4 GGUF and turboderp's EXL3
checkpoints; only the weight format, loader and n-gram codec differ.

## Extensions and checkpoints

| Extension | Provider and model ID | Checkpoint (revision) | Routed experts | Other linears | n-gram table |
| --- | --- | --- | --- | --- | --- |
| `qwen38_flash_next_exl3_4bpw` | `flash-next-exl3-4bpw` | `4.05bpw_h6_ng6` (`55a732e0`) | K4 | K6 (indexer K4) | K6, 39.0 GB |
| `qwen38_flash_next_exl3_3bpw` | `flash-next-exl3-3bpw` | `3.05bpw_h5_ng5` (`69e33439`) | K3 | K5 (indexer K3) | K5, 32.6 GB |
| `qwen38_flash_next_exl3_2bpw` | `flash-next-exl3-2bpw` | `2.05bpw_h4_ng4` (`65c89531`) | K2 | K4 (indexer K2) | K4, 26.2 GB |

The three extensions depend on `qwen38_flash_next` and only register a
provider: each probe accepts a checkpoint directory whose `config.json` has
`model_type` `qwen4_exp`, `quant_method` `exl3` and a `bits` value in its
range. The format code lives in `src/models/qwen38_flash_next/exl3/`
(host) and `kernels/rocm/exl3.*` (device). The MTP block (`mtp.*`) and the
tokenizer come from the checkpoint; images use the unsloth
`mmproj-BF16.gguf`, which `gufo model install` places beside it.

## Format

A linear stores `trellis` (I16 `[in/16][out/16][16 K]`), `suh` (`[in]`),
`svh` (`[out]`) and a `mul1` codebook marker:
`y = H(W^T H(x * suh)) * svh`, with `H` the orthonormal 128-point Hadamard per
block and `W` decoded from 16-bit trellis states through the mul1 codebook.
The exact bit order is Crucible's
[EXL3_FORMAT.md](https://github.com/jaedync/crucible-strix-halo/blob/6c8b36f16a3dedfc759ce0eadae60eea437db459/docs/EXL3_FORMAT.md).
The device repacks tiles column-block-major at load. Routed experts and the
indexer's fused `q|k` projection select 128-wide blocks while repacking; the
Hadamard is block-diagonal, so a block selection or permutation of whole
blocks commutes with it.

n-gram rows are one fp16 scale and 160 K-bit symbols, LSB first; state `i` is
symbol `i` followed by the preceding symbols, truncated to 16 bits;
`row[i] = mul1(state_i) * scale + head_bias[head][i]`. The shards of the
3.05 and 2.05 tables are contiguous, so every table reads as one.
`NgramTable` reads rows with direct I/O and decodes them on the host.

## Conversions to the runtime's layout

The runtime was written for the official GGUF conversion. Each conversion
below was checked against `unsloth/Qwen3.8-Flash-Next-GGUF` `d96201194f10`
(small tensors exactly; quantized tensors to quantization noise, 2% relative
for K6):

- RMSNorm gammas are stored as `w` and uploaded as `1 + w`: hyper-connection,
  attention q/k, indexer q/k, PLE and MTP norms. The gated DeltaNet norm is
  used as stored.
- Gated DeltaNet value heads are grouped by key head in the checkpoint
  (`g * 3 + r`) and tiled in the runtime (`r * 16 + g`). `A_log`, `dt_bias`,
  `in_proj_a/b` rows, the value channels of `conv1d` and `in_proj_qkv`,
  `in_proj_z` and the inputs of `out_proj` move with their heads.
  `A = -exp(A_log)`.
- `index_qk_proj` is `[q (512) | k (128)]`; MTP's `eh_proj` is
  `[fc_embedding | fc_hidden]`.
- Hyper-connection and PLE projections are quantized to Q8_0 on the host,
  the official artifact's format, so they take the fused Q8_0 decode
  kernels. The token embedding stays BF16.

## Kernels

- Decode and verification: GEMVs whose blocks each stage and transform
  their split's inputs, decode tiles through buffer loads (four in flight
  per wave for dense matrices, eight for the small expert matrices), and
  whose last split to arrive reduces every split in a fixed order and
  applies the output transform. Row groups of 1, 2, 4, 8 or 64 rows share
  each decoded tile; every row keeps one accumulation chain in tile order,
  so its result is the same at every batch width, and greedy MTP output
  equals AR output.
- Tiles are stored in Crucible's front order (`Exl3FrontTile`) for the
  split their GEMVs use (`DeviceTensor::exl3_front`): a chunk holds a
  block's eight column blocks by four row tiles, so the blocks at one step
  read neighbouring memory. The column-block-major order gave each wave its
  own stream at a matrix-column stride; front order alone took 4.05 decode
  from 29.8 to 32.2 tokens/s. The prompt GEMM and the Q8_0 expansion read
  the same order.
- Passes of up to eight rows group linears of one input into one launch
  (`Exl3LinearGroup`): GDN `in_proj_qkv` and `in_proj_z`; attention q, k
  and v; routed and shared gate and up; then routed and shared down, which
  form SwiGLU while staging their inputs. The shared expert takes the
  routed stacks' split, since its rows no longer fill the GPU alone.
- Prompt passes, dense layers: each layer is expanded into the official
  GGUF layout (stacked Q8_0 `ssm_in` and `attn_qkv`, Q8_0 outputs and
  shared expert; `Exl3DequantizeQ8` writes `diag(svh) H W^T H diag(suh)`
  with ggml's Q8_0 rounding), so the GGUF fused projection kernels run
  unchanged. A layer is expanded on its first prompt pass and kept: the 48
  layers hold 2.9 GiB of device memory and later passes expand nothing.
  The expansion block decodes and input-transforms its column blocks in
  registers (one LDS round trip fewer) with 16 waves.
- Prompt passes, routed experts and the output head: a WMMA GEMM that
  decodes the A operand in registers. A gfx1151 WMMA reads each A row from
  one lane only (lane r for even rows, r + 16 for odd ones), so the reading
  lane decodes inputs 0..7 of its row, its partner inputs 8..15, and one
  `permlanex16` per dword completes the reading lane with no selects.
  Routed experts run over `RoutedCompact`'s 16-row buckets in 128-row
  tiles; down gives each wave two column blocks so each LDS B fragment
  feeds two WMMAs. Gate and up read the F16 token rows and apply their
  input transform while staging each 128-input step in LDS; their 20-wave
  blocks cover 640 outputs, so a row is transformed once per projection.
  Up's epilogue applies SwiGLU and down's input transform, so down reads
  ready F16 rows; down leaves F16 rows for the fused MoE combine, which
  also defers the residual update as for the Q4_K experts. The epilogue
  keeps the tile's output rows in LDS and each wave's `svh` and next `suh`
  in registers, and issues a tile's loads before any transform.
- Prompt-pass Hadamard transforms use `Wht128Signed`: each butterfly is
  `own + sign * peer`, one `v_fmac_f32_dpp`, where `Wht128`'s
  `sign * own + peer` needs a separate DPP move. It returns
  `(-1)^popcount(lane)` times `Wht128`'s values with the same rounding; the
  sign folds into the next scale. A zero then comes out as -0 where
  `Wht128` gives +0, and the matrix cores' sums depend on the sign of zero
  inputs (prefill-check KL after 1,000 tokens moved from 0.0074 to 0.022),
  so values that feed a WMMA are converted with a +0 addend
  (`HalfPairScaled`), and the results are bit-identical to `Wht128`'s.
  Decode keeps `Wht128`.
- MoE prompt passes queue the shared expert on the side stream after the
  routed gate and up, so it runs beside down: the 640-column gate and up
  blocks each take a whole WGP and share it poorly.
- EXL3 checkpoints prefill in 8,192-row passes (the GGUF in 4,096): each
  pass streams every routed expert's trellis per row tile. With the dense
  expansions kept, 4,096-row passes still prefill the 8,022-token prompt 3%
  slower (November 17).
- Drafts use the draft vocabulary as for the GGUF: the EXL3 head is
  expanded once at load into its Q8_0 matrix (675 MB), whose prefix becomes
  the Q4_K draft head and whose rows fill the draft tables.

`qwen38_flash_next.exl3_ops` checks every kernel against the scalar host
reference for K2 to K6; `qwen38_flash_next.exl3_format` checks the host
reference against Crucible's NumPy reference and ExLlamaV3's n-gram codec.

## Quality

As of November 14, 2026: next-token loss on the same 1,023 tokens
(`docs/install.md` and the start of `ngram.cpp`), decode path:

| Artifact | Loss (nats) | Top-1 match | Argmax agreement with Q4 |
| --- | --- | --- | --- |
| Q4 GGUF | 1.41695 | 67.25% | - |
| EXL3 4.05 | 1.40898 | 67.51% | 92.56% |
| EXL3 3.05 | 1.42317 | 67.84% | 89.05% |
| EXL3 2.05 | 1.43236 | 67.06% | 85.73% |

The 3.05 and 2.05 rows were measured before the final kernel changes,
whose losses moved by at most 0.005; the 4.05 row is current. The prompt
pass and the decode path agree as closely as the GGUF's: KL 0.0005 and
0.0083 after 300 and 1,000 tokens of that text (GGUF 0.0019 and 0.0094),
with the same top token. Greedy MTP output equals
AR output on four chat prompts (4.05). On November 17, after the shared
expert moved to the routed stacks' split (a different but fixed summation
order), the 4.05 decode path scores 1.40882 nats and 67.74% top-1 on the
same tokens, and prompt/decode KL is 0.0003 and 0.0074.

## Performance

As of November 17, 2026, Strix Halo, one session, greedy, thinking off,
`gufo serve` over HTTP: decode tokens/s after the first streamed token for
counting to 60 / five sentences of prose / a Python function (256-token
cap), the second of two runs. Prefill: `build/release` `gufo serve`,
`--cache-ram-bytes off`, after a short warm-up request, the server's
`prompt_ms` for an 8,022-token `/v1/completions` prompt (8,000 seeded
random words of a 16-word vocabulary; one pass for EXL3, two for the GGUF),
the GGUF and EXL3 servers alternated; runs vary by about 1%.

| Artifact | Prefill (tokens/s) | AR decode (tokens/s) | MTP decode (tokens/s) |
| --- | --- | --- | --- |
| Q4 GGUF | 1,981 | 30.3 / 30.1 / 30.3 | 107.4 / 42.6 / 79.2 |
| EXL3 4.05 | 1,930 | 35.9 / 35.8 / 35.8 | 113.3 / 44.9 / 82.2 |
| EXL3 3.05 | 1,920 | 39.8 / 39.3 / 39.7 | 124.9 / 46.9 / 88.2 |
| EXL3 2.05 | 1,927 | 44.4 / 43.8 / 44.3 | 121.3 / 52.2 / 87.8 |

On the same harness before the November 17 prefill changes, EXL3 4.05
prefilled 1,853 tokens/s against the GGUF's 1,969.

Crucible reports serial decode of 33.71 (4.05) and 38.79 (3.05) tokens/s
on the same packs ([RESULTS.md](https://github.com/jaedync/crucible-strix-halo/blob/6c8b36f16a3dedfc759ce0eadae60eea437db459/docs/RESULTS.md)).
One-token decode reads its weights from DRAM: an AR step of the 3.05 pack
spends about 20 ms of its 25 ms in GEMVs at 165 to 230 GB/s (the measured
read ceiling is about 245 GB/s), 4.3 ms in the Q8_0 hyper-connection
mixers and 2 to 3 ms between kernels. `qwen38_flash_next_exl3_gemv_bench`
cycles more than 256 MiB of weight copies, so its timings match decode
rather than the 32 MiB memory-side cache. Small expert matrices cost a
fixed block latency (input staging, split reduction) that low bit widths
do not shrink: ten 2560x640 rows take 31, 37 and 44 us at K2, K3 and K4.
Tried and rejected for decode: a side-stream read-ahead of the next weights
into the memory-side cache (no gain; graph branches compete with the
GEMVs), smaller LDS stages or VGPR caps for more resident blocks, deeper
rings, one split for routed experts (no serve gain; changes outputs), and
one launch for MoE gate/up and down with dependent blocks waiting on the
first (slower).

The routed expert GEMMs are the prefill gap. As of November 17, an
8,022-token pass of 4.05 spends 1.39 s in them (gate 0.43 s, up 0.44 s,
down 0.39 s, plus the shared expert beside down) against 1.23 s for the
Q4_K experts, whose paired gate/up kernel needs no Hadamard transforms
and no F32 gate round trip. The 2.05 checkpoint, with half the 4.05's
expert bytes, prefills no faster: the GEMM is bound by its instruction
schedule, not by weight traffic. `qwen38_flash_next_exl3_gemv_bench
--counts` replays a real prompt's per-expert row counts (74% of rows go to
experts with more than 1,024 rows; half the experts get none) through gate,
up and down: 27.1 ms per layer at K4 before, 25.1 ms after (31.6 / 29.8 /
33.0 TF). With a free trellis decode the same kernels take 21.8 ms, and
without the staging transform gate takes 6% less; even then gate reaches
only 36.5 of the measured 55 TF WMMA peak, because a 20-wave block of 233
VGPRs fills its WGP and waits alone at its barriers. Accumulators bound
the tile: 640 columns by 128 rows use 128 of 256 VGPRs per lane, so each
decoded fragment serves 8 row tiles.

November 17 changes (prefill +4%, outputs bit-identical): the epilogue
fix above (it waited on dependent `y_rows`, `svh` and gate loads, about
19% of a fused launch), `Wht128Signed`, no per-WMMA column check (the
compiler re-materialised it as 3 SALU and a branch per WMMA), two column
blocks per down wave, kept dense expansions (they took 117 ms per prefill),
a 2x faster expansion kernel and the later shared-expert fork.

Tried and rejected for prefill: LDS-only barriers in the fused main loop
(slower: the hardware then waits on the weight loads elsewhere); a
full-tile specialisation of the main loop and an unrolled epilogue (code
size); B fragments loaded one tile ahead (+8 VGPRs, slower, as before);
fused tiles of 64 rows, also forced to two blocks per WGP (decode per WMMA
doubles; spills); down tiles of 192 rows (spills) or four column blocks
of 64 rows; F16 gate outputs (0.6% faster, changes prefill numerics); gate
and up in one launch, each block running gate then up over the same rows
(the phase loop takes 20 more VGPRs and loses more than the cached reads
save); full row tiles dispatched before partial ones (partial tiles lose
their expert's cached weights); expanding layer il + 1 on a low-priority
stream while layer il runs (expansions overlapped fully but stretched 10x;
no prefill gain); running the shared expert serially, or its side stream at the least or
greatest priority (no change); 4,096-row passes.

## Limits

- `gufo chat`, `prompt` and `bench` take GGUF readers; EXL3 checkpoints are
  served through `gufo serve`.
- Prefill is about 97% of the Q4 GGUF's (see Performance).

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

- Decode and verification: one GEMV launch per linear (gate and up of the
  routed experts share one). Each block stages and transforms its split's
  inputs, decodes tiles with eight in flight through buffer loads, and the
  last split to arrive reduces every split in a fixed order and applies the
  output transform. Row groups of 1, 8 or 64 rows share each decoded tile;
  every row keeps one accumulation chain in tile order, so its result is the
  same at every batch width, and greedy MTP output equals AR output.
- Prompt passes, dense layers: each layer is expanded per pass into the
  official GGUF layout (stacked Q8_0 `ssm_in` and `attn_qkv`, Q8_0 outputs
  and shared expert; `Exl3DequantizeQ8` writes
  `diag(svh) H W^T H diag(suh)` with ggml's Q8_0 rounding), so the GGUF
  fused projection kernels run unchanged.
- Prompt passes, routed experts and the output head: a WMMA GEMM that
  decodes the A operand in registers. A gfx1151 WMMA reads each A row from
  one lane only (lane r for even rows, r + 16 for odd ones), so the reading
  lane decodes inputs 0..7 of its row, its partner inputs 8..15, and one
  `permlanex16` per dword completes the reading lane with no selects.
  Routed experts run over `RoutedCompact`'s 16-row buckets: 128-row tiles
  for gate and up, 256-row tiles (one column block per wave) for down and
  the scratch-input path. Gate and up read the
  F16 token rows and apply their input transform while staging each
  128-input step in LDS; their 20-wave blocks cover 640 outputs, so a row
  is transformed once per projection. Up's epilogue applies SwiGLU and
  down's input transform, so down reads ready F16 rows; down leaves F16
  rows for the fused MoE combine, which also defers the residual update as
  for the Q4_K experts.
- EXL3 checkpoints prefill in 8,192-row passes (the GGUF in 4,096): each
  pass streams every routed expert's trellis per row tile.
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
AR output on four chat prompts (4.05).

## Performance

As of November 14, 2026, Strix Halo, one session, greedy. AR decode is 128
tokens after a 20-token prompt (`qwen38_flash_next_exl3_generate_probe`);
prefill is an 8,022-token prompt (one pass for EXL3, two for the GGUF);
MTP is `gufo serve --speculative mtp`
over HTTP, thinking off, tokens/s for counting to 60 / five sentences of
prose / a Python function:

| Artifact | Prefill (tokens/s) | AR decode (tokens/s) | MTP decode (tokens/s) |
| --- | --- | --- | --- |
| Q4 GGUF | 1,966.0 | 29.43 | 107.1 / 39.9 / 83.9 |
| EXL3 4.05 | 1,844.3 | 29.79 | 105.1 / 53.2 / 79.5 |
| EXL3 3.05 | 1,850.2 | 33.30 | 119.7 / 56.5 / 89.5 |
| EXL3 2.05 | 1,853.4 | 35.35 | 110.8 / 54.2 / 67.3 |

MTP decode was measured before the prompt-pass changes, which leave the
decode and verification kernels unchanged. Loads take 22 to 35 s from a
warm page cache. The routed expert GEMMs are the prefill gap: 1.57 s of
the 8k prompt (gate and up 1.11 s, down 0.45 s) against 1.21 s for the
Q4_K experts. The 2.05 checkpoint, with half the 4.05's expert bytes,
prefills no faster, so the GEMM is bound by its instruction schedule
rather than by weight traffic. A thread trace (`rocprofv3 --att`) of the
fused gate GEMM at 157 rows per expert shows 12.6 VALU instructions per
WMMA (trellis decode, Hadamard transforms) on the pipe the WMMAs use, and
waits on barriers (10%), weight loads (9%) and LDS B fragments (8%); down
waits mostly on its B fragments (32%). Deeper weight prefetch, pairing
tiles per decode, branch-free tile counts and software-pipelined B
fragments were each slower: the extra registers or moves cost more than
the waits they hid.

## Limits

- `gufo chat`, `prompt` and `bench` take GGUF readers; EXL3 checkpoints are
  served through `gufo serve`.
- Prefill is about 94% of the Q4 GGUF's (see Performance).

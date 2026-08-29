# HRX optimization ideas

Ranked by expected value over effort. Each card states the evidence it rests
on, so a later reader can tell a measured claim from a hypothesis.

## 1. Port the wide-load transformation to the decode GEMV path

**Status:** done. tg16 5.38 -> 6.74 t/s (+25%), 88% of HIP's 7.62.

Decode has had none of the prefill work and is the largest remaining gap by
ratio: tg16 5.38 t/s against HIP's 7.62 (71%). It runs the single-token GEMV
artifacts, not the blocked projection, so nothing from the prefill effort
reached it.

Those artifacts have exactly the defect that the prefill kernel had. In
`qwen_q8_0_gemv_k5120` each workitem owns one 34-byte Q8_0 block but reads it
as **eight separate four-byte loads** (`buffer.view ... view<1xi32>` per
quarter) plus a two-byte scale, and reads its activations as eight separate
`vector<4xf32>` loads. The payload is 32 contiguous bytes and the activations
are 32 contiguous floats, so both collapse to one wide load each: 2 loads per
block instead of 16.

The same change on the prefill projection was worth 1.35x
(1.347 ms to 0.999 ms). Decode is weight-bandwidth bound - 28 GiB per token,
which at 5.38 t/s is about 150 GB/s against the roughly 213 GB/s HIP sustains -
so fewer, wider requests attack the gap directly.

Artifacts to change: `qwen_q8_0_gemv_k{5120,6144,17408}`,
`qwen_q8_0_gemv_k17408_wg256`, `qwen_q8_0_vocab_gemv_k5120`.

**Acceptance:** decode parity keeps its tight f32 envelope (RMSE <= 1e-4,
cosine >= 0.999999 - decode is not W8A8, so unlike prefill it must stay inside
the original envelope), and tg16 improves by more than run-to-run noise.

**Result.** All five artifacts were widened. Isolated, rows=17408/K=5120 went
from 1.353 ms to 0.610 ms (2.2x). End to end tg16 went 5.38 -> 6.74 t/s and
PP512 403.72 t/s (the vocabulary projection is shared with prefill's final
stage). `--validate-hrx 4` reports a full PASS, including the per-token decode
phases against the untouched f32 envelope.

The decode token profile is now embedding 1.8 ms, attention 8.7, SSM 39.7,
FFN 80.7, final 18.5, total 149.4 ms. FFN moves 18.2 GiB in 80.7 ms, which is
225 GB/s - at the DRAM roofline. The laggards are SSM at about 149 GB/s and the
final stage at about 73 GB/s, so a card 4 below now exists for them.

## 4. Decode's SSM and final stages lag the roofline

**Status:** done. tg16 7.19 -> 7.84 t/s, past HIP's 7.62. The SSM fix was
routing decode through the batch-native recurrence at `tokens = 1`; the final
stage was a one-thread argmax, not the vocabulary projection.

With FFN at 225 GB/s the remaining decode gap to HIP (149.4 ms per token
against about 131 ms) sits in two places:

- **SSM, 39.7 ms** at roughly 149 GB/s. It includes the 96-row alpha/beta
  projection, whose grid is only 96 workgroups, plus the per-token convolution
  and DeltaNet kernels that also run tiny grids in single-token mode.
- **Final stage, 18.5 ms** at roughly 73 GB/s for the 1.35 GiB vocabulary
  projection, plus argmax and the synchronizing readback that `DispatchFinalQ8`
  performs.

Both are small-grid or serialization problems rather than kernel-arithmetic
problems, so measure the split before changing any kernel.

## 2. Tile the attention prefix scan across tokens

**Status:** done, in a different form than proposed (LDS sharing across eight
tokens rather than a flash-style prefix-tile decomposition). Attention 630.4 ->
572.0 ms. Originally deprioritized because: Attention is 630 ms of the 5254 ms PP2048 pass, 12%.
Even halving it cannot close a 38% gap to HIP; the FFN projection at 59% is
where prefill is decided.

Sharing each KV head across its six query heads cut attention 6x, but every
token still rescans its whole prefix independently. Within one 512-token chunk
all 512 tokens read the *same* shared prefix, so there is a large reuse factor
that the current (token, KV head) decomposition cannot capture.

A flash-style decomposition - workgroup per (KV head, prefix tile), carrying
partial softmax state for all tokens of the chunk - would read each cache row
once per chunk instead of once per token.

Attention is 632 ms of the PP2048 run and grows with context: 104 ms in the
first chunk against 213 ms in the fourth. This is the only remaining prefill
lever with a large structural factor rather than a few percent.

**Acceptance:** prefill logits stay bit-identical (this is a reassociation of
the same online softmax, so exactness is achievable), and PP2048 improves.

## 3. Instrument the blocked projection instead of guessing at it

The blocked projection is 61% of the PP512 chunk and is a sharp local optimum:
fifteen structural variants have been measured, only the wide LDS read helped,
and every variant that touches register pressure loses an occupancy tier.

Two things have never been tried:

- **Hardware counters.** `--profile-data=counter-ranges --profile-counter=...`
  in `iree-benchmark-loom` would identify the stall directly rather than by
  inference from ablations.
- **A wider MMA.** Check `compile_report.target_capability_rows` for an int8
  matrix operation with K=32. The current path issues two K=16 MMAs per Q8
  block; one K=32 operation would halve the MMA instruction count.

**Acceptance:** any change here must be judged end to end, not in the isolated
harness - that harness re-reads a hot 94 MiB matrix and is LDS-bound, while the
deployed FFN streams 26 GiB and is bound differently. Several isolated results
in this project did not survive end-to-end retesting.

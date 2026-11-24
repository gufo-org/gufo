#ifndef PREFILL_NORM_EPS
#define PREFILL_NORM_EPS 1e-5f  // RMSNorm epsilon (Qwen3: 1e-6)
#endif
#ifndef PREFILL_ROUNDS
#define PREFILL_ROUNDS 16  // attention rounds per column of the prefill: M / 64
#endif
//===- mm_bfp_mixed.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ATB microkernel, bf16 A / bfp16ebs8 B / bf16 C variant. A is converted to
// bfp16ebs8 in a stack staging buffer first (sizeof(bfp16ebs8) == 1 on Peano,
// so all bfp16ebs8* arithmetic below is byte arithmetic), then the MAC loop
// streams A and B as 576-bit blocks while C is plain bf16 load/store.
// Register placement uses DM-bank annotations: A -> bank A, B -> bank B,
// C -> bank C, raw A input -> bank D.
//
// Compute structure: per call, 2 block rows x 8 blocks of 16x16; each block
// runs 8 expanded k-steps (2 A pops + 2 B pops + 4 8x8x8 MACs, B-first at
// fill steps 0 and 4) in a post-RA-pipelined blk loop with next-block C
// prefetch. Measured II=42, NS=2.

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

template <typename T, unsigned N, aie_dm_resource R>
using InBufStream = aie::block_vector_restrict_input_buffer_stream<T, N, R>;
template <typename T, unsigned N, aie_dm_resource R>
using OutBufStream = aie::block_vector_output_buffer_stream<T, N, R>;

// Per-core L1 C tile is (m, n); the L1 A sub-tile is (m/rho, k). The kernel
// is invoked rho times per (A-sub-tile, B-tile) pair and rotates over the
// rho C stripes (m/rho rows each) via g_counter.
constexpr int rho = 4;
constexpr int m = 128;
constexpr int k = 64;
constexpr int n = 128;
constexpr int m_a = m / rho;

// aie::mmul sub-tile shape (the only shape supported for bfp16ebs8 on
// AIE2P): per MAC, an r x s A panel times an s x t B panel accumulates
// into r x t of C.
constexpr int r = 8; // MAC rows (A/C)
constexpr int s = 8; // MAC reduction step (k-columns per MAC)
constexpr int t = 8; // MAC columns (B/C)

static_assert(m * rho == 512 && k == 64 && n == 128 && rho == 4);

extern "C" {

static int g_counter = 0;

void matmul_vectorized_different_datatypes(bfloat16 *__restrict pA_in,
                                           bfp16ebs8 *__restrict pB_in,
                                           bfloat16 *__restrict pC_curtile) {
  event0();
  // Round-to-nearest-even (Peano defaults to floor on converts).
  aie::set_rounding(aie::rounding_mode::conv_even);
  bfloat16 *__restrict pC_tile = pC_curtile + g_counter * (m_a * n);
  g_counter = (g_counter == rho - 1) ? 0 : g_counter + 1;

  uint8_t converted_A[m_a * k / 8 * 9] __attribute__((aligned(64)));

  OutBufStream<bfp16ebs8, r * s, aie_dm_resource::a> a_out(
      (bfp16ebs8 *)converted_A);
  const bfloat16 __aie_dm_resource_d *__restrict pA =
      (const bfloat16 __aie_dm_resource_d *)pA_in;

  for (int rp = 0; rp < m_a / (2 * r); rp++) {
    const bfloat16 __aie_dm_resource_d *__restrict q = pA + rp * (2 * r * k);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int cb = 0; cb < k / s; cb++) {
      aie::vector<bfloat16, r * s> v0 = aie::load_v<r * s>(q);
      aie::vector<bfloat16, r * s> v1 = aie::load_v<r * s>(q + r * k);
      a_out.push(
          aie::accum<accfloat, r * s>(v0).template to_vector<bfp16ebs8>());
      a_out.push(
          aie::accum<accfloat, r * s>(v1).template to_vector<bfp16ebs8>());
      q += r * s;
    }
  }

  using AStream = InBufStream<bfp16ebs8, r * s, aie_dm_resource::a>;
  using BStream = InBufStream<bfp16ebs8, s * t, aie_dm_resource::b>;
  using CPtr = bfloat16 __aie_dm_resource_c *__restrict;
  using MMUL = aie::mmul<r, s, t, bfp16ebs8>;

  const bfp16ebs8 *pA_base = (const bfp16ebs8 *)converted_A;
  CPtr const pC_base = (CPtr)pC_tile;

  for (int br = 0; br < m_a / (2 * r); br++) {
    // Persistent sliding B stream per block row.
    BStream b_stream(pB_in);
    {
      // C layout: r x t row-major sub-blocks over the (m, n) tile; the
      // +r*n offsets below step to the second r-row half (rows r..2r-1)
      // of the 2r-row block.
      CPtr pC = pC_base + br * (2 * r * n);

      aie::vector<bfloat16, r * t> c0 = aie::load_v<r * t>(pC);
      aie::vector<bfloat16, r * t> c1 = aie::load_v<r * t>(pC + r * t);
      for (int blk = 0; blk < n / (2 * t); blk++) {
        CPtr pCb = pC + blk * (2 * r * t);
        CPtr pNext = (blk < n / (2 * t) - 1) ? pCb + 2 * r * t : pCb;

        MMUL acc0((aie::accum<accfloat, r * t>(c0)));
        MMUL acc1((aie::accum<accfloat, r * t>(c1)));
        MMUL acc2(
            (aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + r * n))));
        MMUL acc3((aie::accum<accfloat, r * t>(
            aie::load_v<r * t>(pCb + r * t + r * n))));

        // Prefetch next block's first C pair while this block computes.
        c0 = aie::load_v<r * t>(pNext);
        c1 = aie::load_v<r * t>(pNext + r * t);

        // Fresh A stream per block: re-reads this block row's band (fills
        // constant-fold before pops 0 and 8).
        AStream a_stream(pA_base);
        a_stream.seek(br * (2 * k / s));

        // One k-step: 2 A pops + 2 B pops + 4 8x8x8 MACs. On fill steps
        // (B pops 0 and 8 of each 16-pop stream buffer) the B pops come
        // first so the fifo_ld_fill emits ahead of the A-stream ops.
        auto kstep = [&]<bool BFILL>() __attribute__((always_inline)) {
          aie::block_vector<bfp16ebs8, r * s> A0, A1; // r x s A panels
          aie::block_vector<bfp16ebs8, s * t> B0, B1; // s x t B panels
          if constexpr (BFILL) {
            B0 = b_stream.pop();
            B1 = b_stream.pop();
            A0 = a_stream.pop();
            A1 = a_stream.pop();
          } else {
            A0 = a_stream.pop();
            A1 = a_stream.pop();
            B0 = b_stream.pop();
            B1 = b_stream.pop();
          }
          acc0.mac(A0, aie::op_transpose(B0));
          acc1.mac(A0, aie::op_transpose(B1));
          acc2.mac(A1, aie::op_transpose(B0));
          acc3.mac(A1, aie::op_transpose(B1));
        };
        kstep.operator()<true>();
        kstep.operator()<false>();
        kstep.operator()<false>();
        kstep.operator()<false>();
        kstep.operator()<true>();
        kstep.operator()<false>();
        kstep.operator()<false>();
        kstep.operator()<false>();

        aie::store_v(pCb, acc0.template to_vector<bfloat16>());
        aie::store_v(pCb + r * t, acc1.template to_vector<bfloat16>());
        aie::store_v(pCb + r * n, acc2.template to_vector<bfloat16>());
        aie::store_v(pCb + r * t + r * n, acc3.template to_vector<bfloat16>());
      }
    }
  }

  event1();
}
}

// ---- LFM3 prefill epilogues (Gufo).

namespace {
}  // namespace
#include "common/exp2_bf16.h"
namespace {
#ifdef SIGMOID_EXP2
// sigmoid(x) = 1 / (1 + 2^(-x log2 e)) with the native exp2 and invsqrt
__attribute__((noinline)) aie::vector<bfloat16, 16> Sigmoid(aie::vector<bfloat16, 16> x) {
  aie::accum<accfloat, 16> t = aie::mul(x, aie::broadcast<bfloat16, 16>(-1.4426950408889634f));
  aie::accum<accfloat, 16> d;
  d.from_vector(aie::exp2<bfloat16>(t.to_vector<float>()));
  d = aie::add(d, aie::broadcast<float, 16>(1.0f));
  aie::accum<accfloat, 16> r;
  r.from_vector(aie::invsqrt(d.to_vector<float>()));  // 1 / d = invsqrt(d)^2: no emulated divide
  const aie::vector<bfloat16, 16> rb = r.to_vector<bfloat16>();
  return aie::mul(rb, rb).to_vector<bfloat16>();
}
#else
// sigmoid(x) = (1 + tanh(x / 2)) / 2, with bf16 multiplies (native) only.
inline aie::vector<bfloat16, 16> Sigmoid(aie::vector<bfloat16, 16> x) {
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  aie::accum<accfloat, 16> h = aie::mul(x, half);
  aie::accum<accfloat, 16> s;
  s.from_vector(aie::broadcast<float, 16>(0.5f));
  return aie::mac(s, aie::tanh<bfloat16>(h.to_vector<float>()), half).to_vector<bfloat16>();
}
#endif
}  // namespace
// Program memory is 16 KB per core: everything below the matmul is optimized for size.
#pragma clang attribute push(__attribute__((minsize)), apply_to = function)

extern "C" {

// In place on an m x n = 128 x 128 C tile held as 8x8 row-major blocks,
// block (bi, bj) at (bi * 16 + bj) * 64: columns 0..63 are gate, 64..127 up
// of the same 64 channels. Writes SiLU(gate) * up into the gate blocks:
// g * u * (1 + tanh(g / 2)) / 2.
void prefill_swiglu(bfloat16 *__restrict c) {
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  for (int bi = 0; bi < 16; ++bi)
    for (int bj = 0; bj < 8; ++bj)
      for (int e = 0; e < 64; e += 16) {
        bfloat16 *g = c + (bi * 16 + bj) * 64 + e;
        const aie::vector<bfloat16, 16> gv = aie::load_v<16>(g), uv = aie::load_v<16>(c + (bi * 16 + bj + 8) * 64 + e);
        aie::store_v(g, aie::mul(aie::mul(gv, uv).to_vector<bfloat16>(), Sigmoid(gv)).to_vector<bfloat16>());
      }
}

}  // extern "C"

// ---- Prefill residual/norm phase. A token row's DDR record is
// [xn bf16 2048 | x bf16 2048 | o bf16 2048 | pad] (16 KB). The phase gets
// w_post (bf16) once as an A object, then each row's [x | o] as one B object
// broadcast to the four cores of a column. With x1 = x + rmsnorm(o) * w_post
// and xn = rmsnorm(x1), core q writes elements [1024 q, 1024 q + 1024) of
// [xn | x1] for 16 rows into its C tile (matmul 8x8-tiled layout). The
// pre-norm weight is folded into the next GEMM's weights.

namespace {
inline aie::vector<float, 16> Wide(const bfloat16 *p) {
  aie::accum<accfloat, 16> a;
  a.from_vector(aie::load_v<16>(p));
  return a.to_vector<float>();
}
// 1 / sqrt(sum / 2^log2n + eps); the division is an exact exponent shift (sum > 0
// and normal, or 0).
__attribute__((noinline)) aie::vector<float, 16> RsqrtMean(aie::vector<float, 16> sum, int log2n) {
  const aie::vector<int32_t, 16> bits = sum.cast_to<int32_t>();
  const aie::vector<int32_t, 16> shifted = aie::sub(bits, aie::broadcast<int32_t, 16>(log2n << 23));
  const aie::vector<float, 16> mean = aie::select(shifted.cast_to<float>(), aie::zeros<float, 16>(),
                                                  aie::lt(bits, aie::broadcast<int32_t, 16>((log2n + 1) << 23)));
  return aie::invsqrt(aie::add(mean, aie::broadcast<float, 16>(PREFILL_NORM_EPS)));
}
inline aie::vector<float, 16> Rsqrt(aie::vector<float, 16> sum) { return RsqrtMean(sum, 11); }
inline aie::vector<float, 16> Total(aie::accum<accfloat, 16> a) {
  return aie::broadcast<float, 16>(aie::reduce_add(a.to_vector<float>()));
}
inline aie::vector<bfloat16, 16> Bf(aie::vector<float, 16> v) {
  aie::accum<accfloat, 16> a;
  a.from_vector(v);
  return a.to_vector<bfloat16>();
}
inline aie::vector<float, 16> Wide16(aie::vector<bfloat16, 16> v) {
  aie::accum<accfloat, 16> a;
  a.from_vector(v);
  return a.to_vector<float>();
}
// Tile rows of the C object in the matmul layout [m/r][n/t][r][t] with r = t = 8:
// element (R, c) sits at ((R >> 3) * 16 + (c >> 3)) * 64 + (R & 7) * 8 + (c & 7).
inline bfloat16 *TileRow(bfloat16 *tile, int R) { return tile + ((R >> 3) << 10) + ((R & 7) << 3); }
// Stores 16 consecutive columns starting at column block cb (8 columns each).
inline void Store16(bfloat16 *d, int cb, aie::vector<bfloat16, 16> v) {
  aie::store_v(d + (cb << 6), v.extract<8>(0));
  aie::store_v(d + (cb << 6) + 64, v.extract<8>(1));
}
}  // namespace

extern "C" {

// Row slot within the current C object: rows of a phase come in multiples of 16.
int row_slot;

// wp: post-norm weight (2048, one A object); row: [x | o]; part q = 0..3.
void prefill_norm_row(const bfloat16 *__restrict wp, const bfloat16 *__restrict row, bfloat16 *__restrict tile,
                      int32_t part) {
  const int slot = row_slot;
  row_slot = (slot + 1) & 15;
#ifdef NORM_STUB
  return;
#endif
  const bfloat16 *x = row;
  const bfloat16 *o = row + 2048;
  // One pass: sum(o^2), sum(x^2), sum(x wo), sum(wo^2) with wo = w o; then
  // sum(x1^2) = sum(x^2) + 2 so sum(x wo) + so^2 sum(wo^2).
  aie::accum<accfloat, 16> oo = aie::zeros<accfloat, 16>(), xx = oo, xw = oo, ww = oo;
  for (int i = 0; i < 2048; i += 16) {
    const aie::vector<bfloat16, 16> xv = aie::load_v<16>(x + i), ov = aie::load_v<16>(o + i);
    const aie::vector<bfloat16, 16> wo = aie::mul(aie::load_v<16>(wp + i), ov).to_vector<bfloat16>();
    oo = aie::mac(oo, ov, ov);
    xx = aie::mac(xx, xv, xv);
    xw = aie::mac(xw, xv, wo);
    ww = aie::mac(ww, wo, wo);
  }
#ifdef PREFILL_PLAIN_RESIDUAL
  // Plain residual (x1 = x + o): the host passes w_post = 1.
  (void)oo;
  const aie::vector<float, 16> so = aie::broadcast<float, 16>(1.0f);
#else
  const aie::vector<float, 16> so = Rsqrt(Total(oo));
#endif
  // bf16 products into an f32 accumulator: the cross terms keep bf16 precision
  const aie::vector<bfloat16, 16> so_b = Bf(so);
  const aie::vector<bfloat16, 16> two_so = aie::mul(so_b, aie::broadcast<bfloat16, 16>(2.0f)).to_vector<bfloat16>();
  const aie::vector<bfloat16, 16> s2 = aie::mul(so_b, so_b).to_vector<bfloat16>();
  aie::accum<accfloat, 16> sum1;
  sum1.from_vector(Total(xx));
  sum1 = aie::mac(sum1, Bf(Total(xw)), two_so);
  sum1 = aie::mac(sum1, Bf(Total(ww)), s2);
  const aie::vector<float, 16> sh = Rsqrt(sum1.to_vector<float>());
  const int at = (part & 1) * 1024;   // element offset within xn or x1
  const bool norm = part < 2;
  // bf16 multipliers are native; the per-row scales round to bf16.
  const aie::vector<bfloat16, 16> sob = Bf(so), shb = Bf(sh);
  for (int j = 0; j < 8; ++j) {
    bfloat16 *d = TileRow(tile, slot * 8 + j);
    for (int c = 0; c < 128; c += 16) {
      const int i = at + j * 128 + c;
      aie::accum<accfloat, 16> acc;
      acc.from_vector(aie::load_v<16>(x + i));
      acc = aie::mac(acc, aie::mul(aie::load_v<16>(wp + i), sob).to_vector<bfloat16>(), aie::load_v<16>(o + i));
      aie::vector<bfloat16, 16> x1 = acc.to_vector<bfloat16>();
      if (norm) x1 = aie::mul(x1, shb).to_vector<bfloat16>();
      Store16(d, c >> 3, x1);
    }
  }
}

#ifndef PREFILL_NO_CONV
// In place on a C tile whose columns 0..63 are the conv B gate and 64..127
// the conv input of the same 64 channels: writes B * x into columns 0..63.
void prefill_bx(bfloat16 *__restrict c) {
  for (int bi = 0; bi < 16; ++bi)
    for (int bj = 0; bj < 8; ++bj)
      for (int e = 0; e < 64; e += 16) {
        bfloat16 *g = c + (bi * 16 + bj) * 64 + e;
        const aie::vector<bfloat16, 16> u = aie::load_v<16>(c + (bi * 16 + bj + 8) * 64 + e);
        aie::store_v(g, aie::mul(aie::load_v<16>(g), u).to_vector<bfloat16>());
      }
}

#endif  // PREFILL_NO_CONV
}  // extern "C"
#ifndef PREFILL_NO_CONV

// ---- Prefill short convolution phase. The scratch row of token t is
// [bx_j 64 | C_j 64] for the 32 channel blocks j; core q owns blocks
// 8q..8q+7 (512 channels). Its 7 x 512 taps (tap k of channel e at k * 512 + e)
// and a ring of the last 6 bx rows live in the half of its C tile that the
// drain discards: within each 8x8 block, elements 32..63. The other half
// holds y = conv(bx) * C for 16 tokens: token s, channel e at linear
// s * 1024 + e of the tile.

namespace {
inline bfloat16 *Spare(bfloat16 *tile, int u) { return tile + ((u >> 5) << 6) + 32 + (u & 31); }
constexpr int kHist = 7 * 512;
int conv_ring;
}  // namespace

extern "C" {

// Copies taps [which * 2048, which * 2048 + 2048) (only 1536 for which = 1)
// from an A object into the tile and restarts the history ring; which
// alternates 0, 1 per call.
int taps_which, conv_rows, conv_group;
// cnt[7]: A objects of this C object (the whole taps block with the first), cnt[6]: rows
// (the 6 state rows, then 16 tokens, with the first; 16 after).
void prefill_conv_group(int32_t *__restrict cnt) {
  cnt[7] = conv_group == 0 ? 32 : 0;
  cnt[6] = conv_group == 0 ? 22 : 16;
  ++conv_group;
}
// Copies taps from the first two A objects of the block; the others are skipped.
void prefill_conv_taps(const bfloat16 *__restrict a, bfloat16 *__restrict tile) {
  const int which = taps_which++;
  if (which >= 2) return;
  conv_ring = 0;
  const int n = which ? 1536 : 2048;
  for (int i = 0; i < n; i += 16) aie::store_v(Spare(tile, which * 2048 + i), aie::load_v<16>(a + i));
}

// One token row for core part q. With out = 0 it only enters the history.
void prefill_conv_row(const bfloat16 *__restrict row, bfloat16 *__restrict tile, int32_t part) {
  const int out = conv_rows++ >= 6;  // the first 6 rows of a phase are state
  const int ring = conv_ring;
  const int slot = row_slot;
  if (out) row_slot = (slot + 1) & 15;
#ifdef CONV_STUB
  return;
#endif
  for (int jb = 0; jb < 8; ++jb) {
    const bfloat16 *bx = row + (part * 8 + jb) * 128;
    for (int e = 0; e < 64; e += 16) {
      const int ch = jb * 64 + e;
      const aie::vector<bfloat16, 16> cur = aie::load_v<16>(bx + e);
      if (out) {
        aie::accum<accfloat, 16> acc = aie::mul(aie::load_v<16>(Spare(tile, 6 * 512 + ch)), cur);
        for (int k = 0; k < 6; ++k) {
          int h = ring + k;
          if (h >= 6) h -= 6;
          acc = aie::mac(acc, aie::load_v<16>(Spare(tile, k * 512 + ch)), aie::load_v<16>(Spare(tile, kHist + h * 512 + ch)));
        }
        const aie::vector<bfloat16, 16> y = aie::mul(acc.to_vector<bfloat16>(), aie::load_v<16>(bx + 64 + e)).to_vector<bfloat16>();
        // linear slot * 1024 + ch: tile row slot * 8 + (ch >> 7), column ch & 127
        bfloat16 *d = tile + slot * 1024 + (((ch & 127) >> 3) << 6) + ((ch >> 7) << 3);
        aie::store_v(d, y.extract<8>(0));
        aie::store_v(d + 64, y.extract<8>(1));
      }
      aie::store_v(Spare(tile, kHist + ring * 512 + ch), cur);
    }
  }
  conv_ring = ring == 5 ? 0 : ring + 1;
}

}  // extern "C"

// ---- Prefill attention. Shared helpers for the qkr and attention phases.

#include "common/exp2_bf16.h"

#endif  // PREFILL_NO_CONV

namespace {
// Raw tile position of linear element p (p % 8 == 0) under the C drain's 8x8 untiling.
inline int RawPos(int p) {
  const int R = p >> 7, c = p & 127;
  return ((R >> 3) << 10) + ((c >> 3) << 6) + ((R & 7) << 3);
}
inline void Put16Linear(bfloat16 *tile, int p, aie::vector<bfloat16, 16> v) {
  aie::store_v(tile + RawPos(p), v.extract<8>(0));
  aie::store_v(tile + RawPos(p + 8), v.extract<8>(1));
}
inline aie::vector<float, 16> Rsqrt128(aie::vector<float, 16> sum) { return RsqrtMean(sum, 7); }
inline aie::vector<float, 16> SumSq128(const bfloat16 *v) {
  aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
  for (int i = 0; i < 128; i += 16) {
    const aie::vector<bfloat16, 16> x = aie::load_v<16>(v + i);
    acc = aie::mac(acc, x, x);
  }
  return Total(acc);
}

// The qkr piece of core q for 16 tokens, as linear tile positions:
//   head hl (0..3): [Q 16 tokens x 128 | gates 16 tokens x 16 heads] at hl * 2304
//   K at 9216, V at 11264: [kb8 2][dblk 16][8 keys][8 dims]
constexpr int kQPiece = 2304, kK = 9216, kV = 11264;

// RMS-normalizes one 128-wide head with weight w and applies RoPE (rotate-half)
// with cos / sin (64 each); out gets 128 bf16.
__attribute__((noinline)) void NormRope(const bfloat16 *x, const bfloat16 *w, const bfloat16 *cs, bfloat16 *out) {
  const aie::vector<bfloat16, 16> inv = Bf(Rsqrt128(SumSq128(x)));
  for (int d = 0; d < 64; d += 16) {
    const aie::vector<bfloat16, 16> a = aie::mul(aie::mul(aie::load_v<16>(x + d), inv).to_vector<bfloat16>(), aie::load_v<16>(w + d)).to_vector<bfloat16>();
    const aie::vector<bfloat16, 16> b =
        aie::mul(aie::mul(aie::load_v<16>(x + 64 + d), inv).to_vector<bfloat16>(), aie::load_v<16>(w + 64 + d)).to_vector<bfloat16>();
    const aie::vector<bfloat16, 16> c = aie::load_v<16>(cs + d), s = aie::load_v<16>(cs + 64 + d);
    aie::store_v(out + d, aie::msc(aie::mul(a, c), b, s).to_vector<bfloat16>());
    aie::store_v(out + 64 + d, aie::mac(aie::mul(b, c), a, s).to_vector<bfloat16>());
  }
}
alignas(64) bfloat16 head_buf[128];
inline int KvPos(int base, int slot, int d) { return base + (((slot >> 3) * 16 + (d >> 3)) << 6) + ((slot & 7) << 3); }

int rope_kind, attn_swa, attn_round, attn_b, attn_kb, attn_k0, attn_head;
// Bidirectional attention (attn_swa == 2, d1 prompts): the prompt length, each
// row's question group (-1: state rows, which see and are seen by all rows; a
// question row sees the state rows and its own group).
// They live in the tile past the gates, which the output overwrites only at the
// round's end (the meta object comes again every round).
int attn_len, attn_grouped;
constexpr int kGroups = 14336, kVis = 14848;  // bf16 slots: int8[M], int32[64]
}  // namespace

extern "C" {

// w: [q-norm weight (pre-scaled by log2(e) / sqrt(128)) | k-norm weight] (one A object).
// row: [q 2048 | k 512 | v 512 | gate 16 | ...] with RoPE tables at 4096: per kind [cos 64 | sin 64].
void prefill_qkr_row(const bfloat16 *__restrict w, const bfloat16 *__restrict row, bfloat16 *__restrict tile,
                     int32_t part) {
  const int slot = row_slot;
  row_slot = (slot + 1) & 15;
  const bfloat16 *cs = row + 4096 + rope_kind * 128;
  // sigmoid(gate), applied to the attention output later
  const aie::vector<bfloat16, 16> gates = Sigmoid(aie::load_v<16>(row + 3072));
  // hl 0..3: q heads 4 part + hl; hl 4: k head part; hl 5: v head part (copied)
  for (int hl = 0; hl < 6; ++hl) {
    const bfloat16 *src = hl < 4 ? row + (part * 4 + hl) * 128 : row + 2048 + (hl - 4) * 512 + part * 128;
    if (hl < 5) NormRope(src, hl < 4 ? w : w + 128, cs, head_buf);
    const bfloat16 *h = hl < 5 ? head_buf : src;
    for (int d = 0; d < 128; d += 8) {
      const int p = hl < 4 ? hl * kQPiece + slot * 128 + d : KvPos(hl == 4 ? kK : kV, slot, d);
      aie::store_v(tile + RawPos(p), aie::load_v<8>(h + d));
    }
    if (hl < 4) Put16Linear(tile, hl * kQPiece + 2048 + slot * 16, gates);
  }
}

}  // extern "C"

// ---- Flash attention. Column c serves KV group g = c % 4 and query blocks
// b = 2 r + c / 4 (32 queries each) in rounds r; core q is head 4 g + q. A round
// gets the four Q objects of the group (core q copies its own) and then the KV
// objects of 16 keys each: K at 0, V at 2048, both [kb8 2][dblk 16][8 keys][8 dims].
// Per-row state is kept in 8 x 8 broadcast form (lane row * 8 + j = value of row).
// Tile (bf16 units): Y f32 [g 4][dblk 16][8][8] at 0, Q bf16 [g][dblk][8][8] at
// 8192, P bf16 [g][kb8][8][8] at 12288, l f32 [g][64] at 13312, m bf16 [g][64] at
// 13824, gate bf16 [32] at 14080. O blocks, written last, take raw 12288..16384
// (linear 12288..: 32 tokens x 128) in the layout of Y.

namespace {
constexpr int kQ = 8192, kP = 12288, kL = 13312, kM = 13824, kG = 14080, kO = 12288;
using Mm = aie::mmul<8, 8, 8, bfloat16, bfloat16, accauto>;

inline float *F32(bfloat16 *p) { return reinterpret_cast<float *>(p); }
struct LaneDiff {
  alignas(64) int32_t v[64];
  constexpr LaneDiff() : v() {
    for (int l = 0; l < 64; ++l) v[l] = (l & 7) - (l >> 3);
  }
};
alignas(64) constexpr LaneDiff kLaneDiffT{};
struct Eye {  // bf16 bit patterns: no static constructors on the core
  alignas(64) uint16_t v[64];
  constexpr Eye() : v() {
    for (int l = 0; l < 64; ++l) v[l] = (l & 7) == (l >> 3) ? 0x3F80 : 0;
  }
};
alignas(64) constexpr Eye kEyeT{};
const bfloat16 *const kEye = reinterpret_cast<const bfloat16 *>(kEyeT.v);
const int32_t *const kLaneDiff = kLaneDiffT.v;
inline aie::vector<float, 64> Wide64(aie::vector<bfloat16, 64> v) {
  aie::accum<accfloat, 64> a;
  a.from_vector(v);
  return a.to_vector<float>();
}
inline aie::vector<bfloat16, 64> Narrow64(aie::vector<float, 64> v) {
  aie::accum<accfloat, 64> a;
  a.from_vector(v);
  return a.to_vector<bfloat16>();
}
// 8 values per row as 16 lanes [k * 8 + row] (k = 0, 1) -> broadcast form.
inline aie::vector<bfloat16, 64> Bcast16(aie::vector<bfloat16, 16> v) {
  return aie::transpose(aie::concat(aie::concat(v, v), aie::concat(v, v)), 8, 8);
}
inline aie::vector<bfloat16, 64> RowMax(aie::vector<bfloat16, 64> v) {
  const aie::vector<bfloat16, 64> t = aie::transpose(v, 8, 8);  // [j * 8 + row]
  const aie::vector<bfloat16, 32> a = aie::max(t.extract<32>(0), t.extract<32>(1));
  const aie::vector<bfloat16, 16> b = aie::max(a.extract<16>(0), a.extract<16>(1));
  return Bcast16(aie::max(b, aie::concat(b.extract<8>(1), b.extract<8>(0))));
}
}  // namespace

extern "C" {

// cnt[6] = KV objects of this round; resets the running state.
void prefill_attn_round(int32_t *__restrict cnt, bfloat16 *__restrict tile, int32_t part, int32_t col) {
  const int b = 2 * attn_round + (col >> 2);
  attn_round = attn_round == PREFILL_ROUNDS - 1 ? 0 : attn_round + 1;
  int k0 = 0;
  if (attn_swa) {
    k0 = (32 * b - 255) >> 4;
    if (k0 < 0) k0 = 0;
  }
  cnt[6] = attn_swa == 2 ? 4 * PREFILL_ROUNDS : 2 * b + 2 - k0;  // bidirectional: every key block
  cnt[10] = attn_swa == 2;
  attn_b = b;
  attn_k0 = k0;
  attn_kb = attn_swa == 2 ? 0 : 2 * b;  // the diagonal blocks come first, then k0 .. 2 b - 1
  attn_head = 4 * (col & 3) + part;
  float *y = F32(tile);
  for (int i = 0; i < 4096; i += 16) aie::store_v(y + i, aie::zeros<float, 16>());
  for (int i = 0; i < 256; i += 16) aie::store_v(F32(tile + kL) + i, aie::zeros<float, 16>());
}

#ifndef PREFILL_NO_CONV
// The round's meta object (bidirectional): int16 [length, grouped], then int8
// groups[M] from byte 4 (-1 state rows, any other value a question group).
void prefill_attn_meta(const bfloat16 *__restrict obj, bfloat16 *__restrict tile) {
  const int16_t *m = reinterpret_cast<const int16_t *>(obj);
  attn_len = m[0];
  attn_grouped = m[1];
  const int8_t *g = reinterpret_cast<const int8_t *>(obj) + 4;
  int8_t *groups = reinterpret_cast<int8_t *>(tile + kGroups);
  for (int i = 0; i < 64 * PREFILL_ROUNDS; ++i) groups[i] = g[i];
}

#endif  // PREFILL_NO_CONV

// Q object `which` (0..3 per round, counted here) of the round: [Q 16 x 128 | gates 16 x 16] twice; core `part` keeps object part.
int attn_which;
void prefill_attn_q(const bfloat16 *__restrict obj, bfloat16 *__restrict tile, int32_t part) {
  const int which = attn_which;
  attn_which = (which + 1) & 3;
  if (which != part) return;
  for (int i = 0; i < 32; ++i) {
    const bfloat16 *src = obj + (i >> 4) * 2304 + (i & 15) * 128;
    for (int dblk = 0; dblk < 16; ++dblk)
      aie::store_v(tile + kQ + (((i >> 3) * 16 + dblk) << 6) + ((i & 7) << 3), aie::load_v<8>(src + dblk * 8));
    tile[kG + i] = obj[(i >> 4) * 2304 + 2048 + (i & 15) * 16 + attn_head];
  }
}

void prefill_attn_kv(const bfloat16 *__restrict kv, bfloat16 *__restrict tile) {
  const bfloat16 *K = kv, *V = kv + 2048;
  const bfloat16 *Q = tile + kQ;
  // S = Q K^T: [g][kb8] 8 x 8 blocks, one K transpose reused by the four row groups.
  Mm s[4][2];
  for (int kb8 = 0; kb8 < 2; ++kb8) {
    for (int g = 0; g < 4; ++g) s[g][kb8] = Mm(aie::zeros<accfloat, 64>());
    for (int dblk = 0; dblk < 16; ++dblk) {
      const aie::vector<bfloat16, 64> kt = aie::transpose(aie::load_v<64>(K + ((kb8 * 16 + dblk) << 6)), 8, 8);
      for (int g = 0; g < 4; ++g) s[g][kb8].mac(aie::load_v<64>(Q + ((g * 16 + dblk) << 6)), kt);
    }
  }
  const int key0 = attn_kb * 16, q0 = attn_b * 32;
  const bool bidi = attn_swa == 2;
  const int8_t *attn_groups = reinterpret_cast<const int8_t *>(tile + kGroups);
  int32_t *attn_vis = reinterpret_cast<int32_t *>(tile + kVis);
  // bidirectional: rows past the prompt, and other questions' rows when grouped
  const bool mask = bidi ? (key0 + 15 >= attn_len || q0 + 31 >= attn_len || attn_grouped)
                         : key0 + 15 > q0 || (attn_swa && key0 < q0 + 31 - 255);

  // No running-max rescale: the round's first block (keys 32 b .. 32 b + 15; for
  // bidirectional rounds keys 0 .. 15, holding the first state row every row
  // sees) has a visible key in every row, and its row maxima stay the offset.
  // Later scores above it only make P exceed 1, far inside bf16 / f32 range.
  const bool first = attn_kb == (bidi ? 0 : 2 * attn_b);
  const aie::vector<bfloat16, 64> ones = aie::broadcast<bfloat16, 64>(1.0f);
  float *y = F32(tile);
  for (int g = 0; g < 4; ++g) {
    aie::vector<float, 64> sv[2];
    for (int kb8 = 0; kb8 < 2; ++kb8) sv[kb8] = s[g][kb8].to_vector<float>();
    for (int kb8 = 0; kb8 < 2; ++kb8) {
      if (mask && bidi) {
        for (int lane = 0; lane < 64; ++lane) {
          const int qi = q0 + 8 * g + (lane >> 3), ki = key0 + 8 * kb8 + (lane & 7);
          const int gq = attn_groups[qi], gk = attn_groups[ki];
          // rows past the prompt see every key (they stay finite; none sees them)
          attn_vis[lane] = qi >= attn_len || (ki < attn_len && (gq == -1 || gk == -1 || gq == gk));
        }
        const aie::mask<64> drop = aie::eq(aie::load_v<64>(attn_vis), aie::zeros<int32_t, 64>());
        sv[kb8] = aie::select(sv[kb8], aie::broadcast<float, 64>(-3e38f), drop);
      } else if (mask) {
        // key - query = base + (lane & 7) - (lane >> 3)
        const aie::vector<int32_t, 64> diff =
            aie::add(aie::load_v<64>(kLaneDiff), aie::broadcast<int32_t, 64>(key0 - q0 + 8 * (kb8 - g)));
        aie::mask<64> drop = aie::gt(diff, aie::zeros<int32_t, 64>());
        if (attn_swa) drop = drop | aie::lt(diff, aie::broadcast<int32_t, 64>(-255));
        sv[kb8] = aie::select(sv[kb8], aie::broadcast<float, 64>(-3e38f), drop);
      }
    }
    bfloat16 *m = tile + kM + g * 64;
    if (first) aie::store_v(m, RowMax(Narrow64(aie::max(sv[0], sv[1]))));
    const aie::vector<float, 64> mf = Wide64(aie::load_v<64>(m));
    for (int kb8 = 0; kb8 < 2; ++kb8) aie::store_v(tile + kP + (kb8 << 6), exp2_bf16(aie::sub(sv[kb8], mf)));
    const aie::vector<bfloat16, 64> p0 = aie::load_v<64>(tile + kP), p1 = aie::load_v<64>(tile + kP + 64);
    float *lsum = F32(tile + kL) + g * 64;
    Mm rs(aie::load_v<64>(lsum));
    rs.mac(p0, ones);
    rs.mac(p1, ones);
    aie::store_v(lsum, rs.to_vector<float>());
    for (int dblk = 0; dblk < 16; ++dblk) {
      float *yb = y + ((g * 16 + dblk) << 6);
      Mm acc(aie::load_v<64>(yb));
      acc.mac(p0, aie::load_v<64>(V + (dblk << 6)));
      acc.mac(p1, aie::load_v<64>(V + ((16 + dblk) << 6)));
      aie::store_v(yb, acc.to_vector<float>());
    }
  }
  // diagonal blocks, then k0 .. 2 b - 1 (bidirectional: 0, 1, ...)
  attn_kb = attn_kb == 2 * attn_b + 1 && !bidi ? attn_k0 : attn_kb + 1;
}

// O = Y / l * sigmoid(gate) (qkr stored the sigmoid), as 8 x 8 blocks at raw 12288 + g * 1024 + dblk * 64.
void prefill_attn_out(bfloat16 *__restrict tile) {
  bfloat16 *f = tile + kP;  // P is dead; f[g] = diag(sigmoid(gate) / l) as an 8 x 8 bf16 block
  for (int g = 0; g < 4; ++g) {
    const aie::vector<bfloat16, 8> g8 = aie::load_v<8>(tile + kG + g * 8);
    const aie::vector<bfloat16, 64> gb = Bcast16(aie::concat(g8, g8));
    for (int i = 0; i < 64; i += 16) {
      const aie::vector<bfloat16, 16> il = Bf(aie::inv(aie::load_v<16>(F32(tile + kL) + g * 64 + i)));
      const aie::vector<bfloat16, 16> fv = aie::mul(gb.extract<16>(i >> 4), il).to_vector<bfloat16>();
      aie::store_v(f + g * 64 + i, aie::mul(fv, aie::load_v<16>(kEye + i)).to_vector<bfloat16>());
    }
  }
  const float *y = F32(tile);
  for (int g = 3; g >= 0; --g) {  // O[0] overwrites f
    const aie::vector<bfloat16, 64> d = aie::load_v<64>(f + g * 64);
    for (int dblk = 0; dblk < 16; ++dblk) {
      Mm o;
      o.mul(d, Narrow64(aie::load_v<64>(y + ((g * 16 + dblk) << 6))));
      aie::store_v(tile + kO + g * 1024 + (dblk << 6), o.to_vector<bfloat16>());
    }
  }
}

// ---- Phase interpreter. PREFILL_PLAN lists one code per phase:
// kind | mode << 3 | tiles << 5 | ksteps << 12 (kind 0 GEMM, 1 norm, 2 conv,
// 3 qkr, 4 attention; GEMM mode 0 plain, 1 SwiGLU, 2 bx; qkr mode the RoPE
// kind; attention mode 1 for sliding-window).
#ifndef PREFILL_PLAN
#define PREFILL_PLAN 0
#endif
namespace {
const int32_t kPlan[] = {PREFILL_PLAN};
int phase_at, epilogue_mode, row_qkr;
}  // namespace

// cnt: [GEMM tiles or groups, k steps, row phase, conv, -, attention, round KV
// objects / conv rows, conv A objects / B skipped before, own B objects, skipped after]
int core_row;
void prefill_next(int32_t *__restrict cnt, int32_t row) {
  core_row = row;
#ifdef PREFILL_LAYER_PHASES
  // Identical layers: kPlan holds the first phase, then one layer's phases.
  const int32_t code = phase_at == 0 ? kPlan[0] : kPlan[1 + (phase_at - 1) % PREFILL_LAYER_PHASES];
  phase_at = phase_at + 1 == PREFILL_PHASES ? 0 : phase_at + 1;  // the core reruns per dispatch
#else
  const int32_t code = kPlan[phase_at];
  phase_at = phase_at + 1 == int(sizeof(kPlan) / sizeof(kPlan[0])) ? 0 : phase_at + 1;  // the core reruns per dispatch
#endif
  const int kind = code & 7;
  const int mode = (code >> 3) & 3;
  epilogue_mode = mode;
  cnt[0] = kind == 0 ? (code >> 5) & 127 : 0;
  cnt[1] = (code >> 12) & 255;
  // GEMM B objects of a K step: skipped before, this core's (0 or 1), skipped after.
  // Wide plans: every object is this core's. Narrow: T tiles per K step, core row
  // r < T takes tile r.
  const int t = (code >> 20) & 7;
  if ((code >> 23) & 1) {  // pair (M = 256): core row r takes tile r % 2 of 2
    cnt[7] = core_row & 1;
    cnt[8] = 1;
    cnt[9] = 1 - (core_row & 1);
  } else {
    cnt[7] = t == 0 ? 0 : (core_row < t ? core_row : t);
    cnt[8] = t == 0 ? 1 : (core_row < t ? 1 : 0);
    cnt[9] = t == 0 ? 0 : (core_row < t ? t - 1 - core_row : 0);
  }
  cnt[2] = kind == 1 || kind == 3;  // row phases: norm or qkr
  cnt[3] = kind == 2;
  cnt[4] = 0;
  row_qkr = kind == 3;
  cnt[5] = kind == 4;
  rope_kind = mode;
#ifndef PREFILL_NO_CONV
  taps_which = conv_rows = conv_group = 0;
#endif
  attn_swa = mode;
  attn_round = 0;
}

void prefill_row(const bfloat16 *__restrict w, const bfloat16 *__restrict row, bfloat16 *__restrict tile, int32_t part) {
  if (row_qkr) prefill_qkr_row(w, row, tile, part);
  else prefill_norm_row(w, row, tile, part);
}

void prefill_epilogue(bfloat16 *__restrict c) {
  if (epilogue_mode == 1) prefill_swiglu(c);
#ifndef PREFILL_NO_CONV
  else if (epilogue_mode == 2) prefill_bx(c);
#endif
}

}  // extern "C"

#pragma clang attribute pop

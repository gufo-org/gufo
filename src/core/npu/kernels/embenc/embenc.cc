// Row phases of the EmbeddingGemma 2 encoder program (embenc.py), on all 32
// cores. A token row's record in S is [t | s_hi | s_lo | xn], 512 bf16 each:
// t a product's output, s = s_hi + s_lo the f32-equivalent residual state, xn
// the next product's input. A column's four cores see objects of four rows of
// [t | s_hi | s_lo]; core q computes row q and writes its [s_hi' | s_lo' |
// xn'].
//
// Arithmetic: bf16 inputs into f32 accumulators; a per-row scale multiplies as
// a hi/lo pair of bf16 values, so it keeps about 16 bits.

#include <aie_api/aie.hpp>

#ifndef EMBENC_R
#define EMBENC_R 4  // rows per object
#endif

namespace {

constexpr int kW = 512;  // hidden width
constexpr int kR = EMBENC_R;

using V16 = aie::vector<bfloat16, 16>;
using F16 = aie::vector<float, 16>;
using A16 = aie::accum<accfloat, 16>;

inline A16 Acc(V16 v) {
  A16 a;
  a.from_vector(v);
  return a;
}
inline F16 Wide(V16 v) { return Acc(v).to_vector<float>(); }
inline V16 Narrow(F16 v) {
  A16 a;
  a.from_vector(v);
  return a.to_vector<bfloat16>();
}
// 1 / sqrt(sum / 512 + eps), broadcast: the division is an exponent shift.
inline F16 InvRms(A16 squares, float eps) {
  const float sum = aie::reduce_add(squares.to_vector<float>());
  const F16 s = aie::broadcast<float, 16>(sum);
  const aie::vector<int32_t, 16> bits = s.cast_to<int32_t>();
  const aie::vector<int32_t, 16> shifted =
      aie::sub(bits, aie::broadcast<int32_t, 16>(9 << 23));
  const F16 mean = aie::select(
      shifted.cast_to<float>(), aie::zeros<float, 16>(),
      aie::lt(bits, aie::broadcast<int32_t, 16>(10 << 23)));
  return aie::invsqrt(aie::add(mean, aie::broadcast<float, 16>(eps)));
}
// x * (hi + lo) for a bf16 vector x and a scale split into two bf16 parts.
inline A16 MulSplit(V16 x, V16 hi, V16 lo) {
  A16 a = aie::mul(x, hi);
  return aie::mac(a, x, lo);
}

}  // namespace

extern "C" {

// params: [w_post 512 | (unused) 512 | scale (f32) | eps (f32)]; in: four rows
// of [t | s_hi | s_lo]; out: row q's [s_hi' | s_lo' | xn'] with
// s' = (s + rmsnorm(t) w_post) scale and xn' = rmsnorm(s') (the next
// product's input; its norm weight is folded into that product's weights).
void embenc_addnorm(const bfloat16 *__restrict params,
                    const bfloat16 *__restrict in, bfloat16 *__restrict out,
                    int32_t q) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *wpost = params;
  const float scale = *reinterpret_cast<const float *>(params + 2 * kW);
  const float eps = *reinterpret_cast<const float *>(params + 2 * kW + 2);
  const bfloat16 *t = in + q * 3 * kW, *hi = t + kW, *lo = t + 2 * kW;
  A16 tt = aie::zeros<accfloat, 16>();
  for (int i = 0; i < kW; i += 16) {
    const V16 v = aie::load_v<16>(t + i);
    tt = aie::mac(tt, v, v);
  }
  // inv * scale and scale, as hi/lo bf16 pairs
  const F16 its = aie::mul(InvRms(tt, eps), scale).to_vector<float>();
  const V16 c_hi = Narrow(its), c_lo = Narrow(aie::sub(its, Wide(c_hi)));
  const V16 sc = Narrow(aie::broadcast<float, 16>(scale));
  bfloat16 *o_hi = out, *o_lo = out + kW, *o_x = out + 2 * kW;
  A16 ss = aie::zeros<accfloat, 16>();
  for (int i = 0; i < kW; i += 16) {
    const A16 tw = aie::mul(aie::load_v<16>(t + i), aie::load_v<16>(wpost + i));
    const V16 tw_hi = tw.to_vector<bfloat16>();
    const V16 tw_lo = aie::sub(tw, Acc(tw_hi)).to_vector<bfloat16>();
    A16 s1 = aie::mul(aie::load_v<16>(hi + i), sc);
    s1 = aie::mac(s1, aie::load_v<16>(lo + i), sc);
    s1 = aie::mac(s1, tw_hi, c_hi);
    s1 = aie::mac(s1, tw_hi, c_lo);
    s1 = aie::mac(s1, tw_lo, c_hi);
    const V16 a = s1.to_vector<bfloat16>();
    const V16 b = aie::sub(s1, Acc(a)).to_vector<bfloat16>();
    aie::store_v(o_hi + i, a);
    aie::store_v(o_lo + i, b);
    ss = aie::mac(ss, a, a);
    ss = aie::mac(ss, a, b);
    ss = aie::mac(ss, b, a);
  }
  const F16 is = InvRms(ss, eps);
  const V16 is_hi = Narrow(is), is_lo = Narrow(aie::sub(is, Wide(is_hi)));
  for (int i = 0; i < kW; i += 16) {
    const V16 a = aie::load_v<16>(o_hi + i), b = aie::load_v<16>(o_lo + i);
    A16 x = aie::mul(a, is_hi);
    x = aie::mac(x, a, is_lo);
    x = aie::mac(x, b, is_hi);
    aie::store_v(o_x + i, x.to_vector<bfloat16>());
  }
}

}  // extern "C"

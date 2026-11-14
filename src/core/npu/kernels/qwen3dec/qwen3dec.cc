// Qwen3 decode kernels (one core's part of a phase; see qwen3dec.py),
// adapted from LFM3's decode design.
//
// Vectors live in DDR in natural order. Core (c, j) owns hidden rows
// ((c * 8 + t) * 4 + j) * 8 + e (t, e < 8), q/k/v rows ((c * 16 + t) * 4 + j)
// * 8 + e (t < 16) and FFN channels ((c * 12 + s) * 4 + j) * 16 + e (s < 12,
// e < 16).
//
// A weight record is REC bytes: four rows of 2048 Q8_0 codes in Dot4's tile
// order, then their bf16 block scales (down records: units of 4-row blocks).
//
// The dimensions are dispatch-time values (rtp[3..5]: phases, hidden, FFN), so
// one program serves the 28 x 2048 stack (FFN 6144) of the Qwen3-ASR decoder
// and the Qwen3-TTS talker, and the 5 x 1024 stack (FFN 3072) of the TTS code
// predictor. With hidden 1024 a record holds two 4-row groups (8 rows) of 1024
// codes each, so the records stay full.
#include <stdint.h>

#include <aie_api/aie.hpp>

#ifdef DECODE_ATTN_ROLE
#define ROLE(n) attn_##n
#else
#define ROLE(n) ffn_##n
#endif

namespace {
constexpr int kHMax = 2048, kQ = 2048, kKV = 1024;
constexpr int kInMax = 12 * kHMax;  // the x buffer: phase inputs up to here
constexpr int kRec = 4 * kHMax + 4 * (kHMax / 32) * 2;  // 8704
// The token's dimensions (set at its first phase): hidden, FFN, layers, 4-row
// groups per record, the phase input bytes and the core's 8-row hidden chunks.
int kH = 2048, kI = 6144, kLayers = 28, kGroups = 1, kIn = 12 * 2048,
    kChunks = 8;
constexpr float kEps = 1e-6f;

float SumSquares(const float* v, int n) {
  aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
  for (int i = 0; i < n; i += 16) {
    aie::vector<float, 16> x = aie::load_v<16>(v + i);
    acc = aie::mac(acc, x, x);
  }
  return aie::reduce_add(acc.to_vector<float>());
}

// rsqrt(sum * inv_n + eps), all in vector lanes (no soft-float).
float RmsScale(float sum, float inv_n) {
  aie::vector<float, 16> v = aie::broadcast<float, 16>(sum);
  v = aie::mul(v, aie::broadcast<float, 16>(inv_n)).to_vector<float>();
  v = aie::add(v, aie::broadcast<float, 16>(kEps));
  return aie::invsqrt(v)[0];
}

aie::vector<float, 16> Bf16ToFloat(const bfloat16* p) {
  aie::accum<accfloat, 16> a;
  a.from_vector(aie::load_v<16>(p));
  return a.to_vector<float>();
}

// Q8_0 of v (n a multiple of 512): n codes at q, n / 32 scales at qs.
__attribute__((noinline)) void Quantize(const float* v, int n, int8_t* q,
                                        bfloat16* qs) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::set_saturation(aie::saturation_mode::saturate);
  for (int g = 0; g < n; g += 512) {
    alignas(64) float amax[16];
    for (int j = 0; j < 16; ++j) {
      const float* x = v + g + j * 32;
      amax[j] = aie::reduce_max(aie::max(aie::abs(aie::load_v<16>(x)),
                                         aie::abs(aie::load_v<16>(x + 16))));
    }
    aie::vector<float, 16> d =
        aie::mul(aie::load_v<16>(amax),
                 aie::broadcast<float, 16>(1.0f / 127.0f))
            .to_vector<float>();
    aie::accum<accfloat, 16> da;
    da.from_vector(d);
    aie::vector<bfloat16, 16> db =
        da.to_vector<bfloat16>();  // the stored scale
    aie::store_v(qs + g / 32, db);
    aie::accum<accfloat, 16> dr;
    dr.from_vector(db);
    aie::vector<float, 16> df = dr.to_vector<float>();
    aie::mask<16> zero = aie::le(df, aie::zeros<float, 16>());
    alignas(64) float inv[16];
    aie::store_v(inv,
                 aie::select(aie::inv(aie::select(
                                 df, aie::broadcast<float, 16>(1.0f), zero)),
                             aie::zeros<float, 16>(), zero));
    for (int j = 0; j < 16; ++j) {
      const float* x = v + g + j * 32;
      aie::vector<float, 16> iv = aie::broadcast<float, 16>(inv[j]);
      aie::vector<int8, 16> c0 = aie::to_fixed<int8>(
          aie::mul(aie::load_v<16>(x), iv).to_vector<float>(), 0);
      aie::vector<int8, 16> c1 = aie::to_fixed<int8>(
          aie::mul(aie::load_v<16>(x + 16), iv).to_vector<float>(), 0);
      aie::store_v(q + g + j * 32, aie::concat(c0, c1));
    }
  }
}

// bf16 vector (n values) to Q8_0 at q (codes) and q + n (scales).
void QuantizeBf16(const bfloat16* in, int n, int8_t* q, float* scratch) {
  bfloat16* qs = reinterpret_cast<bfloat16*>(q + n);
  for (int c = 0; c < n; c += 512) {
    const int m = n - c < 512 ? n - c : 512;
    for (int i = 0; i < m; i += 16)
      aie::store_v(scratch + i, Bf16ToFloat(in + c + i));
    Quantize(scratch, m, q + c, qs + c / 32);
  }
}

// Four rows of nb blocks of 32 codes as one record: per block two 8 x 8 int8
// tiles t [k][n], column n < 4 holding codes 32 b + 16 t + k of row n and
// column n >= 4 codes 32 b + 16 t + 8 + k of row n - 4; then bf16 scales
// [block][row]. xs: x's block scales (32-byte aligned).
alignas(64) int32_t dot4_sums[kHMax / 32 * 4 + 16];
inline const bfloat16* Scales(const int8_t* q8, int k) {
  return reinterpret_cast<const bfloat16*>(q8 + k);
}
__attribute__((noinline)) void Dot4(const int8_t* x, const bfloat16* xs,
                                    const int8_t* w, const bfloat16* ws, int nb,
                                    float* out) {
  aie::vector<int32, 16> four = aie::zeros<int32, 16>();
  for (int b = 0; b < nb; b += 2) {
    const aie::vector<int8, 64> a = aie::load_v<64>(x + 32 * b);
#pragma clang loop unroll(disable)
    for (int o = 0; o < 2; ++o) {
      aie::mmul<8, 8, 8, int8, int8> m0, m1;
      m0.mul(a, aie::load_v<64>(w + 128 * (b + o)));
      m1.mul(a, aie::load_v<64>(w + 128 * (b + o) + 64));
      const aie::vector<int32, 16> c0 =
          m0.to_vector<int32>().extract<16>(2 * o);
      const aie::vector<int32, 16> c1 =
          m1.to_vector<int32>().extract<16>(2 * o + 1);
      four = aie::shuffle_down_fill(
          four,
          aie::add(aie::add(c0, aie::shuffle_down(c0, 12)),
                   aie::add(c1, aie::shuffle_down(c1, 12))),
          4);
    }
    if ((b & 3) == 2)
      aie::store_v(dot4_sums + 4 * (b - 2), four);
  }
  aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
  for (int q = 0; q < nb / 16; ++q) {
    const aie::vector<bfloat16, 16> v = aie::load_v<16>(xs + 16 * q);
    const auto [z0, z1] = aie::interleave_zip(v, v, 1);
    const auto [y0, y1] = aie::interleave_zip(z0, z0, 1);
    const auto [y2, y3] = aie::interleave_zip(z1, z1, 1);
    const aie::vector<bfloat16, 16> x4[4] = {y0, y1, y2, y3};
#pragma clang loop unroll(disable)
    for (int i = 0; i < 4; ++i) {
      const int b0 = 16 * q + 4 * i;
      const aie::vector<float, 16> sc =
          aie::mul(aie::load_v<16>(ws + 4 * b0), x4[i]).to_vector<float>();
      acc = aie::mac(acc, aie::to_float(aie::load_v<16>(dot4_sums + 4 * b0), 0),
                     sc);
    }
  }
  aie::vector<float, 16> a = acc.to_vector<float>();
  a = aie::add(a, aie::shuffle_down(a, 8));
  a = aie::add(a, aie::shuffle_down(a, 4));
  aie::store_v(out, a.extract<4>(0));
}

alignas(64) float acc_a[16];
alignas(64) float acc_b[16];
int acc_n, slice_t, copy_r;
}  // namespace

extern "C" {

// The next record of a phase input: bytes [r * REC, (r + 1) * REC) of x,
// clipped to the x buffer (not to the phase's size: attention inputs are
// three records whatever the hidden size).
void dec_copy(const int8_t* rec, int8_t* x) {
  const int start = copy_r++ * kRec,
            n = (kInMax - start) < kRec ? kInMax - start : kRec;
  for (int i = 0; i < n; i += 64)
    aie::store_v(x + start + i, aie::load_v<64>(rec + i));
}

// x: [h f32 H | o f32 H | unused | pre-norm w bf16 H].
// h += o in place; q8 = Q8_0(rmsnorm(h) * w_pre).
void dec_pre(float* x, int8_t* q8, float* scratch) {
  float* h = x;
  const float* o = x + kH;
  const bfloat16* wpre = reinterpret_cast<const bfloat16*>(x + 2 * kH) + kH;
  for (int i = 0; i < kH; i += 16)
    aie::store_v(h + i,
                 aie::add(aie::load_v<16>(h + i), aie::load_v<16>(o + i)));
  const float sh =
      RmsScale(SumSquares(h, kH), kH == 1024 ? 1.0f / 1024 : 1.0f / 2048);
  bfloat16* qs = reinterpret_cast<bfloat16*>(q8 + kH);
  for (int c = 0; c < kH; c += 512) {
    for (int i = 0; i < 512; i += 16) {
      aie::vector<float, 16> t =
          aie::mul(aie::load_v<16>(h + c + i), aie::broadcast<float, 16>(sh))
              .to_vector<float>();
      aie::store_v(scratch + i,
                   aie::mul(t, Bf16ToFloat(wpre + c + i)).to_vector<float>());
    }
    Quantize(scratch, 512, q8 + c, qs + c / 32);
  }
  acc_n = slice_t = copy_r = 0;
}

// The next eight of this core's hidden rows of h.
void dec_slice(const float* x, float* out, int32_t c, int32_t j) {
  const int t = slice_t++;
  aie::store_v(out, aie::load_v<8>(x + ((c * kChunks + t) * 4 + j) * 8));
}

// The records of a group of four rows of H codes: codes, then scales.
inline const bfloat16* GroupScales(const int8_t* g) {
  return reinterpret_cast<const bfloat16*>(g + 4 * kH);
}
constexpr int kGroupBytes1024 = 4 * 1024 + 4 * 32 * 2;  // 4352

// x: bf16 vector of n values.
void dec_quant(const bfloat16* x, int8_t* q8, float* scratch, int32_t n) {
  QuantizeBf16(x, n, q8, scratch);
  acc_n = copy_r = 0;
}

// ---- FFN

// Gate rows of two channels, then their up rows (per 4-row group).
void dec_gate_up(const int8_t* q8, const int8_t* w) {
  for (int g = 0; g < kGroups; ++g) {
    alignas(16) float r4[4];  // gate a, gate b, up a, up b
    const int8_t* p = w + g * kGroupBytes1024;
    Dot4(q8, Scales(q8, kH), p, GroupScales(p), kH >> 5, r4);
    acc_a[acc_n] = r4[0];
    acc_a[acc_n + 1] = r4[1];
    acc_b[acc_n] = r4[2];
    acc_b[acc_n + 1] = r4[3];
    acc_n += 2;
  }
}

// SiLU(gate) * up of the last sixteen channels, as bf16:
// g * u * (1 + tanh(g / 2)) / 2.
void dec_gate_flush(bfloat16* out) {
  aie::vector<float, 16> g = aie::load_v<16>(acc_a), u = aie::load_v<16>(acc_b);
  aie::vector<float, 16> half = aie::broadcast<float, 16>(0.5f);
  aie::accum<accfloat, 16> t;
  t.from_vector(aie::tanh<bfloat16>(aie::mul(g, half).to_vector<float>()));
  aie::vector<float, 16> sig =
      aie::mul(aie::add(t.to_vector<float>(), aie::broadcast<float, 16>(1.0f)),
               half)
          .to_vector<float>();
  aie::vector<float, 16> a =
      aie::mul(aie::mul(g, u).to_vector<float>(), sig).to_vector<float>();
  aie::accum<accfloat, 16> r;
  r.from_vector(a);
  aie::store_v(out, r.to_vector<bfloat16>());
  acc_n = 0;
}

// Down projection: a core's rows are sets of 16; a set is four 4-row groups
// of I / 32 blocks, packed as 4 I / 32 units of 136 bytes (one block of tile4
// codes; a 64-unit record ends with its units' scales). A record may span two
// groups.
// down_g, down_kb: the group and block of the next unit (no division on the
// cores: they have no divider).
int down_g, down_kb, down_flush;
alignas(64) float down_acc[16];
void dec_row(const int8_t* q8, const int8_t* w) {
  const int blocks = kI >> 5;
  const bfloat16* ws = reinterpret_cast<const bfloat16*>(w + 128 * 64);
  for (int done = 0; done < 64;) {
    alignas(16) float r4[4];
    const int g = down_g, kb = down_kb;
    const int n = 64 - done < blocks - kb ? 64 - done : blocks - kb;
    Dot4(q8 + 32 * kb, Scales(q8, kI) + kb, w + 128 * done, ws + 4 * done, n,
         r4);
    for (int r = 0; r < 4; ++r)
      down_acc[4 * g + r] = (kb ? down_acc[4 * g + r] : 0.0f) + r4[r];
    done += n;
    down_kb = kb + n;
    if (down_kb == blocks) {
      down_kb = 0;
      down_g = g == 3 ? 0 : g + 1;
    }
  }
}
// Rows 0..7 of a set after its first 6 records, rows 8..15 after the rest.
void dec_down_flush(float* out) {
  aie::store_v(out, aie::load_v<8>(down_acc + 8 * down_flush));
  down_flush ^= 1;
}

// Four rows of H codes per group (q/k/v, output projection, head).
void dec_rows4(const int8_t* q8, const int8_t* w) {
  for (int g = 0; g < kGroups; ++g) {
    const int8_t* p = w + g * kGroupBytes1024;
    Dot4(q8, Scales(q8, kH), p, GroupScales(p), kH >> 5, acc_a + acc_n);
    acc_n += 4;
  }
}

// Four rows of the 2048 attention outputs (output projection: one group).
void dec_rows4_q(const int8_t* q8, const int8_t* w) {
  Dot4(q8, Scales(q8, kQ), w, reinterpret_cast<const bfloat16*>(w + 4 * kQ),
       kQ >> 5, acc_a + acc_n);
  acc_n += 4;
}

// The last eight row results, as f32.
void dec_rows_flush(float* out) {
  aie::store_v(out, aie::load_v<8>(acc_a));
  acc_n = 0;
}

}  // extern "C"

#ifdef DECODE_ATTN_ROLE
// ---- Attention. Core (c, j < 2) holds query head h = 2c + j; its KV head
// is c.

namespace {
constexpr int kD = 128;
alignas(64) float att_q[kD];
alignas(64) float att_k[kD];
alignas(64) float att_v[kD];
alignas(64) float att_acc[kD];
alignas(64) float att_s[16];
aie::vector<float, 16> att_m, att_l;
int att_left, att_out_k, att_kv_k;

// x[0..128): rmsnorm(x) * w (bf16), then NeoX rotation by cos/sin.
void NormRope(float* x, const bfloat16* w, const float* cs, const float* sn) {
  const float s = RmsScale(SumSquares(x, kD), 1.0f / kD);
  for (int i = 0; i < kD; i += 16) {
    aie::vector<float, 16> v =
        aie::mul(aie::load_v<16>(x + i), aie::broadcast<float, 16>(s))
            .to_vector<float>();
    aie::store_v(x + i, aie::mul(v, Bf16ToFloat(w + i)).to_vector<float>());
  }
  for (int i = 0; i < kD / 2; i += 16) {
    aie::vector<float, 16> a = aie::load_v<16>(x + i),
                           b = aie::load_v<16>(x + i + kD / 2);
    aie::vector<float, 16> c = aie::load_v<16>(cs + i),
                           n = aie::load_v<16>(sn + i);
    aie::accum<accfloat, 16> lo = aie::mul(a, c), hi = aie::mul(b, c);
    lo = aie::msc(lo, b, n);
    hi = aie::mac(hi, a, n);
    aie::store_v(x + i, lo.to_vector<float>());
    aie::store_v(x + i + kD / 2, hi.to_vector<float>());
  }
}

aie::vector<float, 16> Exp16(aie::vector<float, 16> x) {
  aie::accum<accfloat, 16> e;
  e.from_vector(
      aie::exp2<bfloat16>(aie::mul(x, aie::broadcast<float, 16>(1.4426950409f))
                              .to_vector<float>()));
  return e.to_vector<float>();
}

// Folds sixteen scores (att_s, first `valid` real) and their values.
void Fold(const bfloat16* const* v, int valid) {
  aie::vector<float, 16> s = aie::mul(aie::load_v<16>(att_s),
                                      aie::broadcast<float, 16>(0.08838834764f))
                                 .to_vector<float>();
  alignas(64) static const int32_t lane[16] = {0, 1, 2,  3,  4,  5,  6,  7,
                                               8, 9, 10, 11, 12, 13, 14, 15};
  aie::mask<16> real =
      aie::lt(aie::load_v<16>(lane), aie::broadcast<int32, 16>(valid));
  s = aie::select(aie::broadcast<float, 16>(-1e30f), s, real);
  const aie::vector<float, 16> m =
      aie::max(att_m, aie::broadcast<float, 16>(aie::reduce_max(s)));
  const aie::vector<float, 16> f = Exp16(aie::sub(att_m, m));
  alignas(64) float p[16];
  aie::store_v(
      p, aie::select(aie::zeros<float, 16>(), Exp16(aie::sub(s, m)), real));
  alignas(64) bfloat16 p_hi[16], p_lo[16];
  {
    aie::accum<accfloat, 16> t;
    t.from_vector(aie::load_v<16>(p));
    const aie::vector<bfloat16, 16> hi = t.to_vector<bfloat16>();
    aie::accum<accfloat, 16> h;
    h.from_vector(hi);
    t.from_vector(aie::sub(aie::load_v<16>(p), h.to_vector<float>()));
    aie::store_v(p_hi, hi);
    aie::store_v(p_lo, t.to_vector<bfloat16>());
  }
  att_l =
      aie::add(aie::mul(att_l, f).to_vector<float>(),
               aie::broadcast<float, 16>(aie::reduce_add(aie::load_v<16>(p))));
  att_m = m;
  for (int d = 0; d < kD; d += 16) {
    aie::accum<accfloat, 16> a = aie::mul(aie::load_v<16>(att_acc + d), f);
    // p = hi + lo in bf16: two native bf16 MACs instead of an emulated f32 one
    for (int i = 0; i < valid; ++i) {
      const aie::vector<bfloat16, 16> vv = aie::load_v<16>(v[i] + d);
      a = aie::mac(a, vv, aie::broadcast<bfloat16, 16>(p_hi[i]));
      a = aie::mac(a, vv, aie::broadcast<bfloat16, 16>(p_lo[i]));
    }
    aie::store_v(att_acc + d, a.to_vector<float>());
  }
}

// q = att_qh + att_ql in bf16: native bf16 MACs, error ~2^-17.
alignas(64) bfloat16 att_qh[kD], att_ql[kD];
float DotBf16(const bfloat16* k) {
  aie::accum<accfloat, 16> a = aie::zeros<accfloat, 16>();
  for (int i = 0; i < kD; i += 16) {
    const aie::vector<bfloat16, 16> kk = aie::load_v<16>(k + i);
    a = aie::mac(a, aie::load_v<16>(att_qh + i), kk);
    a = aie::mac(a, aie::load_v<16>(att_ql + i), kk);
  }
  return aie::reduce_add(a.to_vector<float>());
}
void SplitQ() {
  for (int i = 0; i < kD; i += 16) {
    aie::accum<accfloat, 16> t;
    t.from_vector(aie::load_v<16>(att_q + i));
    const aie::vector<bfloat16, 16> hi = t.to_vector<bfloat16>();
    aie::accum<accfloat, 16> h;
    h.from_vector(hi);
    t.from_vector(aie::sub(aie::load_v<16>(att_q + i), h.to_vector<float>()));
    aie::store_v(att_qh + i, hi);
    aie::store_v(att_ql + i, t.to_vector<bfloat16>());
  }
}
}  // namespace

extern "C" {

// x: [q|k|v f32 4096 | q-norm w bf16 128 | k-norm w bf16 128], then at byte
// 2 * REC the position's cos (64 f32) and sin (64 f32). `cached` is the
// number of earlier positions.
void dec_attn_begin(const float* x, int32_t c, int32_t j, int32_t cached) {
  const int h = 2 * c + j;
  const bfloat16* qn = reinterpret_cast<const bfloat16*>(x + kQ + 2 * kKV);
  const bfloat16* kn = qn + kD;
  const float* cs = reinterpret_cast<const float*>(
      reinterpret_cast<const int8_t*>(x) + 2 * kRec);
  const float* sn = cs + kD / 2;
  for (int i = 0; i < kD; i += 16) {
    aie::store_v(att_q + i, aie::load_v<16>(x + h * kD + i));
    aie::store_v(att_k + i, aie::load_v<16>(x + kQ + c * kD + i));
    aie::store_v(att_v + i, aie::load_v<16>(x + kQ + kKV + c * kD + i));
  }
  NormRope(att_q, qn, cs, sn);
  NormRope(att_k, kn, cs, sn);
  SplitQ();
  att_m = aie::broadcast<float, 16>(-1e30f);
  att_l = aie::zeros<float, 16>();
  for (int d = 0; d < kD; d += 16)
    aie::store_v(att_acc + d, aie::zeros<float, 16>());
  att_left = cached;
  att_out_k = att_kv_k = 0;
}

}  // extern "C"
namespace {
// Sixteen positions (the first `valid` real): [k bf16 128 | v bf16 128] each.
__attribute__((noinline)) void AttnRecord(const int8_t* rec, int valid) {
  const bfloat16* vs[16];
  for (int i = 0; i < 16; ++i) {
    const bfloat16* kv = reinterpret_cast<const bfloat16*>(rec + i * 512);
    att_s[i] = i < valid ? DotBf16(kv) : 0.0f;
    vs[i] = kv + kD;
  }
  Fold(vs, valid);
}
alignas(64) bfloat16 att_cur[2 * kD];

void ToBf16(const float* src, bfloat16* dst, int n) {
  aie::accum<accfloat, 16> t;
  for (int i = 0; i < n; i += 16) {
    t.from_vector(aie::load_v<16>(src + i));
    aie::store_v(dst + i, t.to_vector<bfloat16>());
  }
}
}  // namespace
extern "C" {

void dec_attn_rec(const int8_t* rec) {
  if (att_left > 0)
    AttnRecord(rec, att_left < 16 ? att_left : 16);
  att_left -= 16;
}

// The current position (its bf16 cache entry), then the output acc / l.
void dec_attn_finish() {
  ToBf16(att_k, att_cur, kD);
  ToBf16(att_v, att_cur + kD, kD);
  AttnRecord(reinterpret_cast<const int8_t*>(att_cur), 1);
  const aie::vector<float, 16> il = aie::inv(att_l);
  for (int d = 0; d < kD; d += 16)
    aie::store_v(att_acc + d,
                 aie::mul(aie::load_v<16>(att_acc + d), il).to_vector<float>());
}

// The next sixteen of this head's 128 output dims, as bf16.
void dec_attn_out(bfloat16* out) {
  aie::accum<accfloat, 16> t;
  t.from_vector(aie::load_v<16>(att_acc + 16 * att_out_k++));
  aie::store_v(out, t.to_vector<bfloat16>());
}

// The next 32 bytes of this position's cache entry [k bf16 128 | v bf16 128].
void dec_kv_store(bfloat16* out) {
  aie::store_v(out, aie::load_v<16>(att_cur + 16 * att_kv_k++));
}

}  // extern "C"

#endif  // DECODE_ATTN_ROLE

// ---- Phase interpreter. The core program asks *_next for each phase's
// counts and runs: heads (shared input records, copied into x), *_begin, pre
// outputs, groups of records with a flush, *_finish, post outputs. Every core
// of a column emits the same number of objects per phase (idle roles emit
// zeros), as the column's drains expect.

namespace {
enum Kind : int32_t {
  kEmbed,
  kAttn1,
  kAttn2,
  kAttn3,
  kFfn1,
  kFfn2,
  kHead,
  kIdle
};

int ph, ph_layer, ph_step, ph_kind, ph_cached, core_c, core_j;
int rec_i;  // records of the phase so far: a stream's records alternate between
            // its cores
alignas(64) float zeros16[16];

struct Counts {
  int32_t heads, pre, outer, inner, post, kind, extra;
};

// Streams broadcast to both of their cores: weight phases see both cores'
// records (inner counts doubled, every other one skipped).
// Counts for hidden H (in = phase-input records, rr = records per 8 output
// rows, doubled for the stream's two cores). H 2048: 128 q/k/v rows in 32
// records, 64 output rows in 16, 192 FFN channels in 96, 64 down rows in 48.
Counts Plan(int kind, int j, const int32_t* rtp) {
  const bool attn_role = j < 2;
  const int in = kIn > 2 * kRec ? 3 : 2, rr = kGroups == 1 ? 4 : 2;
  switch (kind) {
    case kEmbed:
      return {1, kChunks, 0, 0, 0, kind, 0};
    case kAttn1:
      return {in, kChunks, 16, rr, 0, kind, 0};
    case kAttn2:
      return attn_role ? Counts{3, 0, 1, rtp[1], 24, kind, rtp[0]}
                       : Counts{0, 0, 1, 0, 24, kIdle, 0};
    case kAttn3:  // rows of the 2048 attention outputs: one group per record
      return {1, 0, kChunks, 4, 0, kind, 0};
    case kFfn1:
      return {in, kChunks, kI >> 9, kGroups == 1 ? 16 : 8, 0, kind, 0};
    case kFfn2:
      return {2 * kI > kRec ? 2 : 1, 0, kChunks, kI >> 9, 0, kind, 0};
    case kHead:
      return {in, 0, rtp[2], rr, 0, kind, 0};  // 8 rows per chunk
  }
  return {0, 0, 0, 0, 0, kIdle, 0};
}

int KindAt(int layer, int step) {
  if (layer < 0)
    return kEmbed;
  if (layer >= kLayers)
    return kHead;
  return step == 0   ? kAttn1
         : step == 1 ? kAttn2
         : step == 2 ? kAttn3
         : step == 3 ? kFfn1
                     : kFfn2;
}
int StepsAt(int layer) {
  return layer < 0 || layer >= kLayers ? 1 : 5;
}
}  // namespace

extern "C" {

// Counts of the next phase; rtp: [cached positions, KV records, head chunks].
// Resets at the end of a token.
void ROLE(next)(int32_t* counts, const int32_t* rtp, int32_t c, int32_t j) {
  if (ph == 0) {
    // rtp[3..5]: the token's phases, hidden and FFN sizes.
    kLayers = rtp[6];
    kH = rtp[4];
    kI = rtp[5];
    kGroups = kH == 1024 ? 2 : 1;
    kIn = 12 * kH;
    kChunks = kH >> 8;
    ph_layer = -1;
    ph_step = 0;
    core_c = c;
    core_j = j;
    down_g = down_kb = down_flush = 0;
  }
  const Counts k = Plan(KindAt(ph_layer, ph_step), j, rtp);
  ph_kind = k.kind;
  ph_cached = k.extra;
  if (++ph_step == StepsAt(ph_layer)) {
    ph_step = 0;
    ++ph_layer;
  }
  ph = ph_layer > kLayers ? 0 : ph + 1;
  rec_i = 0;
  counts[0] = k.heads;
  counts[1] = 0;
  counts[2] = k.pre;
  counts[3] = k.outer;
  counts[4] = k.inner;
  counts[5] = k.post;
  counts[6] = k.kind;
  counts[7] = 0;
}

void ROLE(begin)(int8_t* x, int8_t* q8, float* scratch) {
  float* xf = reinterpret_cast<float*>(x);
  switch (ph_kind) {
    case kEmbed:  // x holds the input row (f32)
      slice_t = 0;
      break;
    case kAttn1:
    case kFfn1:
    case kHead:
      dec_pre(xf, q8, scratch);
      break;
    case kAttn3:
      dec_quant(reinterpret_cast<const bfloat16*>(x), q8, scratch, kQ);
      break;
    case kFfn2:
      dec_quant(reinterpret_cast<const bfloat16*>(x), q8, scratch, kI);
      break;
#ifdef DECODE_ATTN_ROLE
    case kAttn2:
      dec_attn_begin(xf, core_c, core_j, ph_cached);
      break;
#endif
    default:
      break;
  }
  copy_r = 0;
}

void ROLE(pre_out)(const int8_t* x, int8_t* out) {
  dec_slice(reinterpret_cast<const float*>(x), reinterpret_cast<float*>(out),
            core_c, core_j);
}

void ROLE(rec)(const int8_t* q8, const int8_t* w) {
  if (ph_kind != kAttn2 && (rec_i++ & 1) != (core_j & 1))
    return;  // the other core's record
  switch (ph_kind) {
#ifdef DECODE_ATTN_ROLE
    case kAttn2:
      dec_attn_rec(w);
      break;
#endif
    case kAttn1:
    case kHead:
      dec_rows4(q8, w);
      break;
    case kAttn3:
      dec_rows4_q(q8, w);
      break;
    case kFfn1:
      dec_gate_up(q8, w);
      break;
    case kFfn2:
      dec_row(q8, w);
      break;
    default:
      break;
  }
}

void ROLE(flush)(int8_t* out, int32_t* /*cnt*/) {
  switch (ph_kind) {
    case kFfn1:
      dec_gate_flush(reinterpret_cast<bfloat16*>(out));
      break;
    case kFfn2:
      dec_down_flush(reinterpret_cast<float*>(out));
      break;
    case kAttn1:
    case kAttn3:
    case kHead:
      dec_rows_flush(reinterpret_cast<float*>(out));
      break;
    default:
      aie::store_v(reinterpret_cast<float*>(out), aie::load_v<8>(zeros16));
      break;
  }
}

void ROLE(finish)() {
#ifdef DECODE_ATTN_ROLE
  if (ph_kind == kAttn2)
    dec_attn_finish();
#endif
}

void ROLE(post_out)(int8_t* out) {
  switch (ph_kind) {
#ifdef DECODE_ATTN_ROLE
    case kAttn2:  // 8 attention-output objects, then 16 cache-entry objects
      if (att_out_k < 8)
        dec_attn_out(reinterpret_cast<bfloat16*>(out));
      else
        dec_kv_store(reinterpret_cast<bfloat16*>(out));
      break;
#endif
    default:
      aie::store_v(reinterpret_cast<float*>(out), aie::load_v<8>(zeros16));
      break;
  }
}

void ROLE(copy)(const int8_t* rec, int8_t* x) {
  dec_copy(rec, x);
}

}  // extern "C"

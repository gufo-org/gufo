// Q8_0 GEMM cores of q8gemm.py: y = W x for up to 8 G rows of x, all integer.
// x arrives as int16 codes with one scale per row; the weights as Q8_0 codes
// with their block scales as int16 fractions u of the row's largest (exact for
// these models to 1.5e-5 of it). Per 32-code block the int32 dot times u adds
// into an acc64; the host applies the row and column scales.
//
// Records are 4352 bytes. B (one chunk of 512 codes x 8 columns): codes
// [block 16][k 32][n 8] int8, then u [block 16][n 8] int16. A (8 rows x 256
// codes, two per chunk): codes [block 8][k4 8][m 8][k 4] int16, 256 bytes pad.
// Two cores share a B stream and take alternate records (parity).
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int kCodes = 4096, kMaxTiles = 32;
using Acc = aie::accum<acc64, 16>;
alignas(128) Acc g_c[kMaxTiles * 4];  // [tile][group][quarter]
int32_t g_groups, g_chunks, g_tiles, g_parity, g_a, g_rec, g_t, g_kc, g_out;
}  // namespace

extern "C" {

// rtp: groups G, chunks of 512 codes, column tiles of this core.
void gemm_begin(const int32_t* rtp, int32_t parity) {
  aie::set_rounding(aie::rounding_mode::floor);
  aie::set_saturation(aie::saturation_mode::none);
  g_groups = rtp[0], g_chunks = rtp[1], g_tiles = rtp[2], g_parity = parity;
  g_a = g_rec = g_t = g_kc = g_out = 0;
  for (int i = 0; i < g_groups * g_tiles * 4; ++i) g_c[i] = aie::zeros<acc64, 16>();
}

// A records arrive [group][chunk][half].
void gemm_copy(const int8_t* rec, int8_t* a) {
  int8_t* dst = a + g_a * kCodes;
  for (int i = 0; i < kCodes; i += 64) aie::store_v(dst + i, aie::load_v<64>(rec + i));
  ++g_a;
}

// One B record of the shared stream: this core's (chunk g_kc, tile g_t) or the partner's.
void gemm_mac(const int8_t* a, const int8_t* b) {
  if ((g_rec++ & 1) != g_parity) return;
#ifndef GEMM_STUB
  const int16_t* us = reinterpret_cast<const int16_t*>(b + kCodes);
  Acc* ct = g_c + g_t * g_groups * 4;
  for (int g = 0; g < g_groups; ++g) {
    const int16_t* ag = reinterpret_cast<const int16_t*>(a + (g * g_chunks + g_kc) * 2 * kCodes);
    Acc c0 = ct[4 * g], c1 = ct[4 * g + 1], c2 = ct[4 * g + 2], c3 = ct[4 * g + 3];
    for (int blk = 0; blk < 16; ++blk) {
      const int16_t* at = ag + blk * 256;
      const int8_t* bt = b + blk * 256;
      aie::mmul<8, 4, 8, int16, int8> m;
      m.mul(aie::load_v<32>(at), aie::load_v<32>(bt));
      for (int kb = 1; kb < 8; ++kb) m.mac(aie::load_v<32>(at + 32 * kb), aie::load_v<32>(bt + 32 * kb));
      const aie::vector<int32, 64> v = m.to_vector<int32>();
      const aie::vector<int16, 8> u = aie::load_v<8>(us + blk * 8);
      const aie::vector<int16, 16> s = aie::concat(u, u);
      c0 = aie::mac(c0, v.extract<16>(0), s);
      c1 = aie::mac(c1, v.extract<16>(1), s);
      c2 = aie::mac(c2, v.extract<16>(2), s);
      c3 = aie::mac(c3, v.extract<16>(3), s);
    }
    ct[4 * g] = c0, ct[4 * g + 1] = c1, ct[4 * g + 2] = c2, ct[4 * g + 3] = c3;
  }
#endif
  if (++g_t == g_tiles) g_t = 0, ++g_kc;
}

// Output records [tile][group]: the 8 x 8 acc64 as floor(acc / 2^24) [64] int32,
// then acc mod 2^32 [64]; the host joins them.
void gemm_store(int32_t* out) {
  const Acc* src = g_c + g_out * 4;
  for (int q = 0; q < 4; ++q) {
    aie::store_v(out + 16 * q, src[q].to_vector<int32>(24));
    aie::store_v(out + 64 + 16 * q, src[q].to_vector<int32>(0));
  }
  ++g_out;
}

}  // extern "C"

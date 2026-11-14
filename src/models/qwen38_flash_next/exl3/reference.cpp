#include "src/models/qwen38_flash_next/exl3/reference.hpp"

#include <cassert>

#include "src/models/qwen38_flash_next/exl3/format.hpp"

namespace gufo::models::qwen38_flash_next::exl3 {
namespace {

float RoundHalf(float v) {
  return HalfToFloat(FloatToHalf(v));
}

template<int K>
void NgramRowImpl(const std::uint16_t* row, const std::uint16_t* bias,
                  float* out) {
  const float* table = Mul1Table();
  const float scale = HalfToFloat(row[0]);
  for (int i = 0; i < static_cast<int>(kNgramRowDim); ++i) {
    float v = table[NgramState<K>(row, i)] * scale;
    if (bias != nullptr) {
      v += HalfToFloat(bias[i]);
    }
    out[i] = v;
  }
}

}  // namespace

const float* Mul1Table() {
  static const std::vector<float> table = [] {
    std::vector<float> t(65536);
    for (std::uint32_t s = 0; s < t.size(); ++s)
      t[s] = Mul1Value(s);
    return t;
  }();
  return table.data();
}

void DecodeTile(const std::uint16_t* tile, int k, float out[256]) {
  const int words = 8 * k;
  const auto word = [&](int index) {
    const int w = ((index % words) + words) % words;
    return static_cast<std::uint64_t>(tile[2 * w]) |
           (static_cast<std::uint64_t>(tile[2 * w + 1]) << 16);
  };
  for (int i = 0; i < 256; ++i) {
    const int end = (i + 1) * k + 256 * k;
    const int start = end - 16;
    const int first = start / 32;
    const int last = (end - 1) / 32;
    const int shift = (last + 1) * 32 - end;
    const std::uint64_t pair = (word(first) << 32) | word(last);
    const auto state = static_cast<std::uint32_t>((pair >> shift) & 0xFFFFU);
    int row = 0;
    int col = 0;
    TilePosition(i, &row, &col);
    out[row * 16 + col] = Mul1Value(state);
  }
}

std::vector<float> ReconstructInner(const HostLinear& w) {
  assert(w.in % 16 == 0 && w.out % 16 == 0);
  const std::size_t a_tiles = w.in / 16;
  const std::size_t b_tiles = w.out / 16;
  const std::size_t tile_words = 16 * static_cast<std::size_t>(w.k);
  std::vector<float> m(static_cast<std::size_t>(w.in) * w.out);
  float tile[256];
  for (std::size_t a = 0; a < a_tiles; ++a) {
    for (std::size_t b = 0; b < b_tiles; ++b) {
      DecodeTile(w.trellis + (a * b_tiles + b) * tile_words, w.k, tile);
      for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
          m[(a * 16 + r) * w.out + b * 16 + c] = tile[r * 16 + c];
        }
      }
    }
  }
  return m;
}

void Hadamard128(std::span<float> v) {
  assert(v.size() % kHadamardSize == 0);
  for (std::size_t base = 0; base < v.size(); base += kHadamardSize) {
    float* x = v.data() + base;
    for (std::uint32_t stride = 1; stride < kHadamardSize; stride <<= 1) {
      for (std::uint32_t i = 0; i < kHadamardSize; i += 2 * stride) {
        for (std::uint32_t j = i; j < i + stride; ++j) {
          const float lo = x[j];
          const float hi = x[j + stride];
          x[j] = lo + hi;
          x[j + stride] = lo - hi;
        }
      }
    }
    for (std::uint32_t i = 0; i < kHadamardSize; ++i) {
      x[i] *= kHadamardNorm;
    }
  }
}

std::vector<float> MatVec(const HostLinear& w, std::span<const float> x) {
  assert(x.size() == w.in);
  std::vector<float> xh(w.in);
  for (std::uint32_t i = 0; i < w.in; ++i) {
    xh[i] = RoundHalf(RoundHalf(x[i]) * HalfToFloat(w.suh[i]));
  }
  Hadamard128(xh);
  for (auto& v : xh) {
    v = RoundHalf(v);
  }
  const auto inner = ReconstructInner(w);
  std::vector<float> y(w.out, 0.0F);
  for (std::uint32_t i = 0; i < w.in; ++i) {
    const float xi = xh[i];
    const float* row = inner.data() + static_cast<std::size_t>(i) * w.out;
    for (std::uint32_t o = 0; o < w.out; ++o) {
      y[o] += xi * row[o];
    }
  }
  Hadamard128(y);
  for (std::uint32_t o = 0; o < w.out; ++o) {
    y[o] *= HalfToFloat(w.svh[o]);
  }
  return y;
}

void DecodeNgramRow(const std::uint16_t* row, int k, const std::uint16_t* bias,
                    float* out) {
  switch (k) {
    case 4:
      NgramRowImpl<4>(row, bias, out);
      return;
    case 5:
      NgramRowImpl<5>(row, bias, out);
      return;
    case 6:
      NgramRowImpl<6>(row, bias, out);
      return;
    case 8:
      NgramRowImpl<8>(row, bias, out);
      return;
    default:
      assert(false && "unsupported n-gram row width");
  }
}

}  // namespace gufo::models::qwen38_flash_next::exl3

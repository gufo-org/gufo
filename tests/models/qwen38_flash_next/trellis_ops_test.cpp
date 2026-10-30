// Gyro TQ2_T routed expert kernels against the vendored reference decoder:
// the vector gate/up (SwiGLU) and down projections, the input rotation and
// the routed F16 WMMA GEMM. Rotated weights hold T·w, so every reference
// dots the decoded rows with the CPU-rotated input (T·x).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/gyro/tq_ref.h"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/trellis.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

float Uniform(std::uint32_t* state, float scale) {
  return scale *
         static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                            32768) /
         32768.0F;
}

/// Device copies released at scope exit.
class DeviceArena {
public:
  DeviceArena() = default;
  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;
  ~DeviceArena() {
    for (void* p : owned_) {
      (void)hipFree(p);
    }
  }

  template<typename T>
  T* Upload(const std::vector<T>& host) {
    T* device = nullptr;
    CheckHip(hipMalloc(&device, host.size() * sizeof(T) + 4096), "hipMalloc");
    owned_.push_back(device);
    CheckHip(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                       hipMemcpyHostToDevice),
             "upload");
    return device;
  }

private:
  std::vector<void*> owned_;
};

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  CheckHip(
      hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return host;
}

/// Packed TQ2_T experts [e][m][k] and their decoded (still rotated) rows.
struct Experts {
  std::vector<std::uint8_t> packed;
  std::vector<float> rows;
};

Experts MakeExperts(std::size_t e, std::size_t m, std::size_t k,
                    std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / TQ_BLOCK);
  w.packed.resize(blocks * 34);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 34;
    const std::uint16_t d = tq_fp32_to_fp16(Uniform(&seed, 0.02F) + 0.03F);
    std::memcpy(blk, &d, 2);
    for (int i = 0; i < 32; ++i) {
      blk[2 + i] = static_cast<std::uint8_t>(NextRandom(&seed));
    }
  }
  w.rows.resize(e * m * k);
  if (tq_dequantize_row(TQ_TYPE_TQ2_T, w.packed.data(), w.rows.data(),
                        static_cast<std::int64_t>(e * m * k)) != 0) {
    throw std::runtime_error("reference decode failed");
  }
  return w;
}

std::vector<std::int8_t> MakeSigns(std::size_t k, std::uint32_t seed) {
  std::vector<std::int8_t> s(k);
  for (auto& v : s) {
    v = (NextRandom(&seed) & 1U) != 0 ? 1 : -1;
  }
  return s;
}

/// T·x of each k-wide row (identity when signs is empty).
std::vector<float> Rotate(std::vector<float> x, std::size_t k,
                          const std::vector<std::int8_t>& signs) {
  if (signs.empty()) {
    return x;
  }
  for (std::size_t r = 0; r < x.size() / k; ++r) {
    tq_hadamard_rotate(x.data() + r * k, static_cast<std::int64_t>(k), TQ_BLOCK,
                       signs.data());
  }
  return x;
}

double Dot(const float* a, const float* b, std::size_t k) {
  double sum = 0.0;
  for (std::size_t i = 0; i < k; ++i) {
    sum += static_cast<double>(a[i]) * b[i];
  }
  return sum;
}

std::vector<float> RandomRows(std::size_t rows, std::size_t k, float scale,
                              std::uint32_t seed) {
  std::vector<float> x(rows * k);
  for (float& v : x) {
    v = Uniform(&seed, scale);
  }
  return x;
}

std::vector<std::int32_t> RandomIds(std::size_t tokens, std::size_t used,
                                    std::size_t experts, std::uint32_t seed) {
  std::vector<std::int32_t> ids(tokens * used);
  for (std::size_t t = 0; t < tokens; ++t) {
    std::vector<std::int32_t> chosen;
    while (chosen.size() < used) {
      const auto e = static_cast<std::int32_t>(NextRandom(&seed) % experts);
      if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) {
        chosen.push_back(e);
      }
    }
    std::copy(chosen.begin(), chosen.end(), ids.begin() + t * used);
  }
  return ids;
}

/// Max |got - want| over the reference RMS; fails above `limit`.
bool Compare(const char* name, const std::vector<float>& got,
             const std::vector<double>& want, double limit) {
  double sq = 0.0;
  double worst = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    sq += want[i] * want[i];
    const double err = std::abs(static_cast<double>(got[i]) - want[i]);
    if (!std::isfinite(got[i])) {
      worst = INFINITY;
    }
    worst = std::max(worst, err);
  }
  const double rms = std::sqrt(sq / static_cast<double>(want.size()));
  const double rel = rms > 0.0 ? worst / rms : worst;
  const bool ok = rel <= limit;
  std::cout << (ok ? "PASS " : "FAIL ") << name << ": max err " << worst
            << " = " << rel << " x rms " << rms << " (limit " << limit << ")\n";
  return ok;
}

double Silu(double v) {
  return v / (1.0 + std::exp(-v));
}

/// How a vector batch's tokens share experts.
enum class Routing { kRandom, kShared, kDisjoint };

const char* RoutingName(Routing routing) {
  return routing == Routing::kShared     ? "shared"
         : routing == Routing::kDisjoint ? "disjoint"
                                         : "random";
}

/// Top-`used` ids per token: random distinct experts with one inactive slot,
/// the same experts in rotated slot order for every token, or experts no
/// other slot uses.
std::vector<std::int32_t> MakeIds(Routing routing, std::size_t tokens,
                                  std::size_t used, std::size_t experts,
                                  std::uint32_t seed) {
  std::vector<std::int32_t> ids(tokens * used);
  switch (routing) {
    case Routing::kRandom:
      ids = RandomIds(tokens, used, experts, seed);
      ids[used - 1] = -1;  // an inactive slot writes zeros
      break;
    case Routing::kShared:
      for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t j = 0; j < used; ++j) {
          ids[t * used + j] = static_cast<std::int32_t>((j + t) % used);
        }
      }
      break;
    case Routing::kDisjoint:
      if (tokens * used > experts) {
        throw std::runtime_error("too few experts for disjoint routing");
      }
      for (std::size_t s = 0; s < ids.size(); ++s) {
        ids[s] = static_cast<std::int32_t>(experts - 1 - s);
      }
      break;
  }
  return ids;
}

std::vector<std::uint16_t> Codebook() {
  return {tq_codebook_f16, tq_codebook_f16 + 2 * TQ_CB_SIZE};
}

std::vector<float> RunGatedVec(const Experts& gate, const Experts& up,
                               const std::vector<float>& x,
                               const std::vector<std::int8_t>& signs,
                               const std::vector<std::int32_t>& ids,
                               std::size_t m, std::size_t k, std::size_t tokens,
                               std::size_t used) {
  DeviceArena arena;
  std::vector<float> fsigns(signs.begin(), signs.end());
  float* d_out = arena.Upload(std::vector<float>(tokens * used * m, 99.0F));
  if (!q::TrellisGatedVec(
          arena.Upload(gate.packed), arena.Upload(up.packed), arena.Upload(x),
          signs.empty() ? nullptr : arena.Upload(fsigns),
          reinterpret_cast<const std::uint32_t*>(arena.Upload(Codebook())),
          arena.Upload(ids), d_out, static_cast<std::uint32_t>(m),
          static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(tokens),
          static_cast<std::uint32_t>(used), nullptr)) {
    throw std::runtime_error("TrellisGatedVec rejected the shape");
  }
  CheckHip(hipDeviceSynchronize(), "gated vector");
  return Download(d_out, tokens * used * m);
}

std::vector<float> RunDownVec(const Experts& w, const std::vector<float>& x,
                              const std::vector<std::int8_t>& signs,
                              const std::vector<std::int32_t>& ids,
                              std::size_t m, std::size_t k, std::size_t tokens,
                              std::size_t used) {
  DeviceArena arena;
  std::vector<float> fsigns(signs.begin(), signs.end());
  float* d_out = arena.Upload(std::vector<float>(tokens * used * m, 99.0F));
  if (!q::TrellisDownVec(
          arena.Upload(w.packed), arena.Upload(x), arena.Upload(fsigns),
          reinterpret_cast<const std::uint32_t*>(arena.Upload(Codebook())),
          arena.Upload(ids), d_out, static_cast<std::uint32_t>(m),
          static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(tokens),
          static_cast<std::uint32_t>(used), nullptr)) {
    throw std::runtime_error("TrellisDownVec rejected the shape");
  }
  CheckHip(hipDeviceSynchronize(), "down vector");
  return Download(d_out, tokens * used * m);
}

/// Whether each token's rows equal a batch of that token alone, byte for
/// byte: verification must keep single-token decoding's arithmetic.
bool SameAsSingleTokens(
    const char* name, const std::vector<float>& batch, std::size_t row_floats,
    const std::function<std::vector<float>(std::size_t)>& single) {
  std::size_t differ = 0;
  for (std::size_t t = 0; t < batch.size() / row_floats; ++t) {
    const auto alone = single(t);
    differ += std::memcmp(alone.data(), batch.data() + t * row_floats,
                          row_floats * sizeof(float)) != 0;
  }
  std::cout << (differ == 0 ? "PASS " : "FAIL ") << name
            << ": tokens differing from single-token batches " << differ
            << '\n';
  return differ == 0;
}

/// Shared expert weights of the vector checks, decoded once.
struct VecWeights {
  static constexpr std::size_t kE = 32;
  static constexpr std::size_t kHidden = 2560;
  static constexpr std::size_t kFf = 640;
  Experts gate = MakeExperts(kE, kFf, kHidden, 0x7A11E000U);
  Experts up = MakeExperts(kE, kFf, kHidden, 0x7A11E000U ^ 0x9E3779B9U);
  Experts down = MakeExperts(kE, kHidden, kFf, 0xD0E00000U);
};

bool CheckGatedVec(const VecWeights& v, std::size_t tokens, std::size_t used,
                   Routing routing, bool rotated, std::uint32_t seed) {
  const std::size_t m = VecWeights::kFf;
  const std::size_t k = VecWeights::kHidden;
  const auto signs =
      rotated ? MakeSigns(k, seed + 7) : std::vector<std::int8_t>{};
  const auto x = RandomRows(tokens, k, 1.0F, seed + 11);
  const auto ids = MakeIds(routing, tokens, used, VecWeights::kE, seed + 13);
  const auto xr = Rotate(x, k, signs);
  std::vector<double> want(tokens * used * m, 0.0);
  for (std::size_t s = 0; s < tokens * used; ++s) {
    if (ids[s] < 0) {
      continue;
    }
    for (std::size_t r = 0; r < m; ++r) {
      const std::size_t row = (static_cast<std::size_t>(ids[s]) * m + r) * k;
      const float* xs = xr.data() + (s / used) * k;
      const double g = Dot(v.gate.rows.data() + row, xs, k);
      const double u = Dot(v.up.rows.data() + row, xs, k);
      want[s * m + r] = Silu(g) * u;
    }
  }
  const auto got = RunGatedVec(v.gate, v.up, x, signs, ids, m, k, tokens, used);
  const std::string name = "gated vec tokens=" + std::to_string(tokens) +
                           " used=" + std::to_string(used) + " " +
                           RoutingName(routing) +
                           (rotated ? " rotated" : " unrotated");
  bool ok = Compare(name.c_str(), got, want, 6e-3);
  if (tokens > 1) {
    ok =
        SameAsSingleTokens(
            name.c_str(), got, used * m,
            [&](std::size_t t) {
              return RunGatedVec(
                  v.gate, v.up, {x.begin() + t * k, x.begin() + (t + 1) * k},
                  signs, {ids.begin() + t * used, ids.begin() + (t + 1) * used},
                  m, k, 1, used);
            }) &&
        ok;
  }
  return ok;
}

/// The down projection of tokens * used slot rows, each with its own input.
bool CheckDownVec(const VecWeights& v, std::size_t tokens, std::size_t used,
                  Routing routing, std::uint32_t seed) {
  const std::size_t m = VecWeights::kHidden;
  const std::size_t k = VecWeights::kFf;
  const std::size_t slots = tokens * used;
  const auto signs = MakeSigns(k, seed + 3);
  const auto x = RandomRows(slots, k, 0.5F, seed + 5);
  const auto ids = MakeIds(routing, tokens, used, VecWeights::kE, seed + 9);
  const auto xr = Rotate(x, k, signs);
  std::vector<double> want(slots * m, 0.0);
  for (std::size_t s = 0; s < slots; ++s) {
    if (ids[s] < 0) {
      continue;
    }
    for (std::size_t r = 0; r < m; ++r) {
      want[s * m + r] = Dot(
          v.down.rows.data() + (static_cast<std::size_t>(ids[s]) * m + r) * k,
          xr.data() + s * k, k);
    }
  }
  const auto got = RunDownVec(v.down, x, signs, ids, m, k, tokens, used);
  const std::string name = "down vec tokens=" + std::to_string(tokens) +
                           " used=" + std::to_string(used) + " " +
                           RoutingName(routing);
  bool ok = Compare(name.c_str(), got, want, 4e-3);
  if (tokens > 1) {
    ok =
        SameAsSingleTokens(
            name.c_str(), got, used * m,
            [&](std::size_t t) {
              return RunDownVec(
                  v.down,
                  {x.begin() + t * used * k, x.begin() + (t + 1) * used * k},
                  signs, {ids.begin() + t * used, ids.begin() + (t + 1) * used},
                  m, k, 1, used);
            }) &&
        ok;
  }
  return ok;
}

/// The routed F16 WMMA GEMM over GPU-rotated rows, as prefill runs it.
/// `used` > 1 is the gate/up view (token rows), 1 the down view (slots).
bool CheckRoutedWmma(std::size_t tokens, std::size_t used, std::size_t m,
                     std::size_t k, std::uint32_t tile_rows,
                     std::uint32_t seed) {
  constexpr std::size_t kE = 32;
  const std::size_t slots = tokens * used;
  const Experts w = MakeExperts(kE, m, k, seed);
  const auto signs = MakeSigns(k, seed + 1);
  const auto x = RandomRows(tokens, k, 1.0F, seed + 2);
  const auto ids = RandomIds(tokens, used, kE, seed + 4);
  const auto xr = Rotate(x, k, signs);
  std::vector<std::uint32_t> counts(kE, 0);
  for (const auto e : ids) {
    ++counts[e];
  }
  std::vector<double> want(slots * m);
  std::vector<double> rot_want(xr.begin(), xr.end());
  for (std::size_t s = 0; s < slots; ++s) {
    for (std::size_t r = 0; r < m; ++r) {
      want[s * m + r] =
          Dot(w.rows.data() + (static_cast<std::size_t>(ids[s]) * m + r) * k,
              xr.data() + (s / used) * k, k);
    }
  }
  DeviceArena arena;
  std::vector<float> fsigns(signs.begin(), signs.end());
  std::vector<std::uint16_t> codebook(tq_codebook_f16,
                                      tq_codebook_f16 + 2 * TQ_CB_SIZE);
  const auto* d_codebook =
      reinterpret_cast<const std::uint32_t*>(arena.Upload(codebook));
  const auto* d_ids = arena.Upload(ids);
  const auto* d_counts = arena.Upload(counts);
  const std::size_t compact = q::RoutedCompactRows(slots, kE);
  auto* d_bounds = arena.Upload(std::vector<std::int32_t>(kE + 1, 0));
  auto* d_cursors = arena.Upload(std::vector<std::int32_t>(kE, 0));
  auto* d_rows_token = arena.Upload(std::vector<std::int32_t>(compact, 0));
  auto* d_rows_slot = arena.Upload(std::vector<std::int32_t>(compact, 0));
  q::RoutedCompact(d_ids, d_counts, d_bounds, d_cursors, d_rows_token,
                   d_rows_slot, static_cast<std::uint32_t>(tokens),
                   static_cast<std::uint32_t>(used),
                   static_cast<std::uint32_t>(kE), nullptr);
  std::vector<std::int32_t> tiles;
  for (std::size_t e = 0; e < kE; ++e) {
    const std::uint32_t padded = (counts[e] + 15U) / 16U * 16U;
    for (std::uint32_t j = 0; j < (padded + tile_rows - 1) / tile_rows; ++j) {
      tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
    }
  }
  auto* d_x_rot = arena.Upload(std::vector<__half>(tokens * k));
  q::TrellisRotateRows(arena.Upload(x), nullptr, d_x_rot, arena.Upload(fsigns),
                       static_cast<std::uint32_t>(tokens),
                       static_cast<std::uint32_t>(k), nullptr);
  float* d_out = arena.Upload(std::vector<float>(slots * m, 99.0F));
  __half* d_out_half =
      arena.Upload(std::vector<__half>(slots * m, __float2half(99.0F)));
  const auto* rows_in = used > 1 ? d_rows_token : d_rows_slot;
  for (int half_out = 0; half_out < 2; ++half_out) {
    if (!q::RoutedF16Gemm(
            arena.Upload(w.packed), q::WeightType::kTQ2_T, d_x_rot,
            arena.Upload(tiles), static_cast<std::uint32_t>(tiles.size()),
            tile_rows, d_bounds, rows_in, d_rows_slot, nullptr,
            half_out != 0 ? nullptr : d_out,
            half_out != 0 ? d_out_half : nullptr, m, k, nullptr, d_codebook)) {
      throw std::runtime_error("routed TQ2_T GEMM rejected the shape");
    }
  }
  CheckHip(hipDeviceSynchronize(), "routed TQ2_T GEMM");
  const auto rot = Download(d_x_rot, tokens * k);
  std::vector<float> rot_got(rot.size());
  for (std::size_t i = 0; i < rot.size(); ++i) {
    rot_got[i] = __half2float(rot[i]);
  }
  const auto got = Download(d_out, slots * m);
  const auto got_half = Download(d_out_half, slots * m);
  std::vector<float> half_got(got_half.size());
  for (std::size_t i = 0; i < got_half.size(); ++i) {
    half_got[i] = __half2float(got_half[i]);
  }
  const std::string name =
      "routed wmma " + std::to_string(m) + "x" + std::to_string(k) +
      " slots=" + std::to_string(slots) + " tile=" + std::to_string(tile_rows);
  bool ok = Compare(("rotate " + name).c_str(), rot_got, rot_want, 2e-3);
  ok = Compare(name.c_str(), got, want, 6e-3) && ok;
  ok = Compare((name + " f16 out").c_str(), half_got, want, 6e-3) && ok;
  return ok;
}

}  // namespace

int main() {
  try {
    bool ok = true;
    const VecWeights v;
    // Decode (one token), MTP verification widths with ragged expert
    // groups (2..8), and batches whose groups exceed eight rows.
    for (std::size_t tokens : {1U, 2U, 3U, 5U, 8U, 12U}) {
      ok = CheckGatedVec(v, tokens, 10, Routing::kRandom, true,
                         0x7A11E000U + tokens) &&
           ok;
      ok =
          CheckDownVec(v, tokens, 10, Routing::kRandom, 0xD0E00000U + tokens) &&
          ok;
    }
    ok = CheckGatedVec(v, 2, 10, Routing::kRandom, false, 0x7A11E100U) && ok;
    for (std::size_t tokens : {3U, 8U, 12U}) {
      ok = CheckGatedVec(v, tokens, 10, Routing::kShared, true,
                         0x7A11E200U + tokens) &&
           ok;
      ok =
          CheckDownVec(v, tokens, 10, Routing::kShared, 0xD0E00200U + tokens) &&
          ok;
    }
    // Every slot its own expert: single-row groups only.
    ok = CheckGatedVec(v, 8, 4, Routing::kDisjoint, true, 0x7A11E300U) && ok;
    ok = CheckDownVec(v, 8, 4, Routing::kDisjoint, 0xD0E00300U) && ok;
    for (std::uint32_t tile : {16U, 48U, 64U}) {
      ok = CheckRoutedWmma(300, 10, 640, 2560, tile, 0x3A3A0000U + tile) && ok;
      ok = CheckRoutedWmma(3000, 1, 2560, 640, tile, 0x3B3B0000U + tile) && ok;
    }
    // A ragged tile: buckets far below one 16-row tile.
    ok = CheckRoutedWmma(7, 10, 640, 2560, 16, 0x3C3C0001U) && ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

// EXL3 linear kernels against the scalar host reference: every supported
// bit width, dense rows sharing one matrix, routed expert rows, block
// selection while repacking, and batch-width invariance of a row's result.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/exl3/format.hpp"
#include "src/models/qwen38_flash_next/exl3/reference.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/exl3.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace fmt = gufo::models::qwen38_flash_next::exl3;

namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t* s) {
  *s ^= *s << 13;
  *s ^= *s >> 17;
  *s ^= *s << 5;
  return *s;
}

float Uniform(std::uint32_t* s) {
  return static_cast<float>(Next(s) & 0xFFFFFF) / 8388608.0F - 1.0F;
}

template<class T>
T* Upload(const std::vector<T>& v) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, v.size() * sizeof(T) + 64), "hipMalloc");
  CheckHip(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice),
           "upload");
  return static_cast<T*>(p);
}

/// A random stored expert stack and its device form.
struct Stack {
  std::uint32_t in;
  std::uint32_t out;
  int k;
  std::uint32_t experts;
  std::vector<std::uint16_t> trellis;  // [experts][in/16][out/16][16k]
  std::vector<std::uint16_t> suh;      // [experts][in]
  std::vector<std::uint16_t> svh;      // [experts][out]
  q::Exl3Weights device{};
  std::vector<void*> allocations;

  Stack(std::uint32_t in_, std::uint32_t out_, int k_, std::uint32_t experts_,
        std::uint32_t seed)
      : in(in_), out(out_), k(k_), experts(experts_) {
    std::uint32_t s = seed;
    trellis.resize(static_cast<std::size_t>(experts) * in * out * k / 16);
    for (auto& w : trellis)
      w = static_cast<std::uint16_t>(Next(&s));
    suh.resize(static_cast<std::size_t>(experts) * in);
    svh.resize(static_cast<std::size_t>(experts) * out);
    for (auto& v : suh)
      v = fmt::FloatToHalf(Uniform(&s) * 0.5F);
    for (auto& v : svh)
      v = fmt::FloatToHalf(Uniform(&s) * 0.05F);
  }

  ~Stack() {
    for (void* p : allocations)
      (void)hipFree(p);
  }

  [[nodiscard]] fmt::HostLinear Host(std::uint32_t e) const {
    const std::size_t words = static_cast<std::size_t>(in) * out * k / 16;
    return {trellis.data() + e * words,
            suh.data() + e * in,
            svh.data() + e * out,
            in,
            out,
            k};
  }

  /// Uploads the stored bytes and repacks them, optionally selecting output
  /// blocks (dense only).
  void Prepare(const std::vector<std::uint32_t>& out_blocks = {}) {
    const std::uint32_t dev_out =
        out_blocks.empty()
            ? out
            : static_cast<std::uint32_t>(out_blocks.size()) * 128;
    auto* raw = Upload(trellis);
    allocations.push_back(raw);
    const std::size_t src_bytes = static_cast<std::size_t>(in) * out * k / 8;
    std::vector<std::uint64_t> offsets(experts);
    for (std::uint32_t e = 0; e < experts; ++e)
      offsets[e] = e * src_bytes;
    auto* dev_offsets = Upload(offsets);
    allocations.push_back(dev_offsets);
    std::uint32_t* blocks = nullptr;
    if (!out_blocks.empty()) {
      blocks = Upload(out_blocks);
      allocations.push_back(blocks);
    }
    const std::size_t dst_words =
        static_cast<std::size_t>(in) * dev_out * k / 32;
    void* tiles = nullptr;
    CheckHip(hipMalloc(&tiles, dst_words * experts * 4), "tiles");
    allocations.push_back(tiles);
    q::Exl3Repack(reinterpret_cast<const std::uint8_t*>(raw), dev_offsets,
                  static_cast<std::uint32_t*>(tiles), in, out, in, dev_out, k,
                  experts, dst_words, nullptr, blocks, nullptr);
    // Scales: suh as stored, svh through the same block selection.
    std::vector<std::uint16_t> dev_svh;
    for (std::uint32_t e = 0; e < experts; ++e) {
      for (std::uint32_t j = 0; j < dev_out; ++j) {
        const std::uint32_t sj =
            out_blocks.empty() ? j : out_blocks[j / 128] * 128 + j % 128;
        dev_svh.push_back(svh[static_cast<std::size_t>(e) * out + sj]);
      }
    }
    auto* d_suh = Upload(suh);
    auto* d_svh = Upload(dev_svh);
    allocations.push_back(d_suh);
    allocations.push_back(d_svh);
    device = {static_cast<const std::uint32_t*>(tiles),
              reinterpret_cast<const __half*>(d_suh),
              reinterpret_cast<const __half*>(d_svh),
              in,
              dev_out,
              experts,
              k};
  }
};

struct Scratch {
  q::Exl3Scratch s{};
  Scratch(std::size_t xh, std::size_t partial) {
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
    CheckHip(hipMalloc(&a, xh * 2), "xh");
    CheckHip(hipMalloc(&b, partial * 4), "partial");
    CheckHip(hipMalloc(&c, 4096 * 4), "done");
    CheckHip(hipMemset(c, 0, 4096 * 4), "done");
    s = {static_cast<__half*>(a),   xh,  static_cast<float*>(b), partial,
         static_cast<unsigned*>(c), 4096};
  }
  ~Scratch() {
    (void)hipFree(s.xh);
    (void)hipFree(s.partial);
    (void)hipFree(s.done);
  }
};

double RelErr(const float* a, const std::vector<float>& ref) {
  double e = 0.0;
  double n = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    e += (a[i] - ref[i]) * static_cast<double>(a[i] - ref[i]);
    n += ref[i] * static_cast<double>(ref[i]);
  }
  return std::sqrt(e / std::max(n, 1e-30));
}

bool RunLinear(const Stack& st, const std::vector<float>& x, std::uint32_t ldx,
               std::uint32_t x_div, const std::vector<std::int32_t>& ids,
               std::uint32_t rows, const Scratch& scratch,
               std::vector<float>* y) {
  float* dx = Upload(x);
  std::int32_t* dids = ids.empty() ? nullptr : Upload(ids);
  float* dy = nullptr;
  CheckHip(hipMalloc(reinterpret_cast<void**>(&dy),
                     static_cast<std::size_t>(rows) * st.device.out * 4),
           "y");
  const bool ok = q::Exl3Linear(st.device, dx, ldx, x_div, dids, dy,
                                st.device.out, rows, scratch.s, nullptr);
  CheckHip(hipDeviceSynchronize(), "sync");
  y->resize(static_cast<std::size_t>(rows) * st.device.out);
  CheckHip(hipMemcpy(y->data(), dy, y->size() * 4, hipMemcpyDeviceToHost),
           "download");
  (void)hipFree(dx);
  (void)hipFree(dy);
  if (dids != nullptr)
    (void)hipFree(dids);
  return ok;
}

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::cerr << "FAIL: " << what << "\n";
  }
}

void DenseCase(std::uint32_t in, std::uint32_t out, int k, std::uint32_t rows) {
  Stack st(in, out, k, 1, 0x1234567U + in * 7 + out + k);
  st.Prepare();
  std::uint32_t s = 99U + k;
  std::vector<float> x(static_cast<std::size_t>(rows) * in);
  for (auto& v : x)
    v = Uniform(&s);
  Scratch scratch(1 << 20, 1 << 22);
  std::vector<float> y;
  Expect(RunLinear(st, x, in, 1, {}, rows, scratch, &y), "dense launch");
  double worst = 0.0;
  for (std::uint32_t r = 0; r < rows; ++r) {
    const auto ref =
        fmt::MatVec(st.Host(0), std::span<const float>(x.data() + r * in, in));
    worst = std::max(worst, RelErr(y.data() + r * out, ref));
  }
  Expect(worst < 1e-4, "dense K" + std::to_string(k) + " " +
                           std::to_string(in) + "x" + std::to_string(out) +
                           " rel " + std::to_string(worst));
  // A row's result does not depend on the batch: row 0 alone.
  std::vector<float> y1;
  Expect(RunLinear(st, std::vector<float>(x.begin(), x.begin() + in), in, 1, {},
                   1, scratch, &y1),
         "single-row launch");
  Expect(std::memcmp(y1.data(), y.data(), out * sizeof(float)) == 0,
         "dense K" + std::to_string(k) + " batch-width invariance");
}

void ExpertCase(int k) {
  constexpr std::uint32_t kIn = 2560;
  constexpr std::uint32_t kOut = 640;
  constexpr std::uint32_t kExperts = 12;
  constexpr std::uint32_t kTokens = 3;
  constexpr std::uint32_t kUsed = 4;
  Stack st(kIn, kOut, k, kExperts, 0xBEEF00U + k);
  st.Prepare();
  std::uint32_t s = 7U + k;
  std::vector<float> x(kTokens * kIn);
  for (auto& v : x)
    v = Uniform(&s);
  std::vector<std::int32_t> ids(kTokens * kUsed);
  for (auto& id : ids)
    id = static_cast<std::int32_t>(Next(&s) % kExperts);
  Scratch scratch(1 << 20, 1 << 22);
  std::vector<float> y;
  Expect(RunLinear(st, x, kIn, kUsed, ids, kTokens * kUsed, scratch, &y),
         "expert launch");
  double worst = 0.0;
  for (std::uint32_t r = 0; r < kTokens * kUsed; ++r) {
    const auto ref =
        fmt::MatVec(st.Host(static_cast<std::uint32_t>(ids[r])),
                    std::span<const float>(x.data() + (r / kUsed) * kIn, kIn));
    worst = std::max(worst, RelErr(y.data() + r * kOut, ref));
  }
  Expect(worst < 1e-4,
         "experts K" + std::to_string(k) + " rel " + std::to_string(worst));
}

bool RunGemm(const Stack& st, const std::vector<float>& x, std::uint32_t ldx,
             std::uint32_t rows, const q::Exl3GemmRoute& route,
             std::uint32_t y_rows, std::vector<float>* y,
             const std::vector<std::int32_t>& ids = {},
             std::size_t xh_elems = std::size_t{1} << 22, bool fused = false) {
  float* dx = Upload(x);
  // Fused: the GEMM reads F16 rows and transforms them itself.
  __half* dxh = nullptr;
  if (fused) {
    std::vector<__half> h(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
      h[i] = __float2half(x[i]);
    CheckHip(hipMalloc(reinterpret_cast<void**>(&dxh), h.size() * 2), "xh");
    CheckHip(hipMemcpy(dxh, h.data(), h.size() * 2, hipMemcpyHostToDevice),
             "xh upload");
  }
  std::int32_t* dids = ids.empty() ? nullptr : Upload(ids);
  float* dy = nullptr;
  const std::size_t n = static_cast<std::size_t>(y_rows) * st.device.out;
  CheckHip(hipMalloc(reinterpret_cast<void**>(&dy), n * 4), "y");
  CheckHip(hipMemset(dy, 0, n * 4), "y clear");
  Scratch scratch(xh_elems, 16);
  const bool ok = q::Exl3LinearGemm(st.device, dx, ldx, 1, dids,
                                    static_cast<std::uint32_t>(x.size() / ldx),
                                    dy, st.device.out, rows, route, scratch.s,
                                    nullptr, {dxh, nullptr, nullptr, fused});
  if (dxh != nullptr)
    (void)hipFree(dxh);
  if (dids != nullptr)
    (void)hipFree(dids);
  CheckHip(hipDeviceSynchronize(), "sync");
  y->resize(n);
  CheckHip(hipMemcpy(y->data(), dy, n * 4, hipMemcpyDeviceToHost), "download");
  (void)hipFree(dx);
  (void)hipFree(dy);
  return ok;
}

void GemmDenseCase(std::uint32_t in, std::uint32_t out, int k,
                   std::uint32_t rows) {
  Stack st(in, out, k, 1, 0xABCDU + in + out * 3 + k);
  st.Prepare();
  std::uint32_t s = 31U + k;
  std::vector<float> x(static_cast<std::size_t>(rows) * in);
  for (auto& v : x)
    v = Uniform(&s);
  std::vector<float> y;
  // A scratch of 1.5 row tiles forces the chunked route on large inputs.
  constexpr std::size_t kTile = q::kExl3TransformedGemmRows;
  Expect(RunGemm(st, x, in, rows, {}, rows, &y, {},
                 rows > kTile ? kTile * 3 / 2 * in : std::size_t{1} << 22),
         "dense GEMM launch");
  double worst = 0.0;
  for (std::uint32_t r = 0; r < rows; r += std::max(1U, rows / 17)) {
    const auto ref =
        fmt::MatVec(st.Host(0), std::span<const float>(x.data() + r * in, in));
    worst = std::max(worst, RelErr(y.data() + r * out, ref));
  }
  Expect(worst < 2e-4, "dense GEMM K" + std::to_string(k) + " " +
                           std::to_string(in) + "x" + std::to_string(out) +
                           " rows " + std::to_string(rows) + " rel " +
                           std::to_string(worst));
}

void GemmRoutedCase(int k) {
  // Three experts with 5, 0 and 300 rows (two tiles of either map),
  // 16-padded buckets.
  constexpr std::uint32_t kIn = 512;
  constexpr std::uint32_t kOut = 256;
  Stack st(kIn, kOut, k, 3, 0x5151U + k);
  st.Prepare();
  const std::vector<std::uint32_t> counts = {5, 0, 300};
  std::vector<std::int32_t> bounds = {0};
  for (const auto c : counts)
    bounds.push_back(bounds.back() +
                     static_cast<std::int32_t>((c + 15) / 16 * 16));
  const std::uint32_t slots = 5 + 300;
  std::vector<std::int32_t> x_rows(bounds.back(), -1);
  std::vector<std::int32_t> y_rows(bounds.back(), -1);
  std::vector<std::int32_t> expert_of(slots);
  std::uint32_t next = 0;
  for (std::uint32_t e = 0; e < 3; ++e) {
    for (std::uint32_t i = 0; i < counts[e]; ++i) {
      // Rows of an expert are scattered slots.
      const std::uint32_t slot = (next * 37) % slots;
      x_rows[bounds[e] + i] = static_cast<std::int32_t>(slot);
      y_rows[bounds[e] + i] = static_cast<std::int32_t>(slot);
      expert_of[slot] = static_cast<std::int32_t>(e);
      ++next;
    }
  }
  // Tile maps of the scratch-input GEMM and of the fused one.
  std::vector<std::int32_t> tiles;
  std::vector<std::int32_t> fused_tiles;
  for (std::uint32_t e = 0; e < 3; ++e) {
    for (std::uint32_t j = 0; j * q::kExl3TransformedGemmRows < counts[e]; ++j)
      tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
    for (std::uint32_t j = 0; j * q::kExl3GemmRows < counts[e]; ++j)
      fused_tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
  }
  std::uint32_t s = 77U + k;
  std::vector<float> x(slots * kIn);
  for (auto& v : x)
    v = Uniform(&s);
  auto* d_tiles = Upload(tiles);
  auto* d_fused_tiles = Upload(fused_tiles);
  auto* d_bounds = Upload(bounds);
  auto* d_x = Upload(x_rows);
  auto* d_y = Upload(y_rows);
  const q::Exl3GemmRoute route{
      d_tiles, static_cast<std::uint32_t>(tiles.size()), d_bounds, d_x, d_y};
  std::vector<float> y;
  Expect(RunGemm(st, x, kIn, 0, route, slots, &y, expert_of),
         "routed GEMM launch");
  // The fused input transform gives the same bits as the scratch pass.
  std::vector<float> fused;
  const q::Exl3GemmRoute fused_route{
      d_fused_tiles, static_cast<std::uint32_t>(fused_tiles.size()), d_bounds,
      d_x, d_y};
  Expect(RunGemm(st, x, kIn, 0, fused_route, slots, &fused, expert_of,
                 std::size_t{1} << 22, true),
         "fused routed GEMM launch");
  Expect(fused == y, "fused routed GEMM K" + std::to_string(k) + " bits");
  double worst = 0.0;
  for (std::uint32_t e = 0; e < 3; ++e) {
    for (std::uint32_t i = 0; i < counts[e]; ++i) {
      const auto slot = static_cast<std::uint32_t>(y_rows[bounds[e] + i]);
      const auto xr = slot;
      const auto ref = fmt::MatVec(
          st.Host(e), std::span<const float>(x.data() + xr * kIn, kIn));
      worst = std::max(worst, RelErr(y.data() + slot * kOut, ref));
    }
  }
  for (void* p :
       std::initializer_list<void*>{d_tiles, d_fused_tiles, d_bounds, d_x, d_y})
    (void)hipFree(p);
  Expect(worst < 2e-4,
         "routed GEMM K" + std::to_string(k) + " rel " + std::to_string(worst));
}

void DequantCase(int k) {
  // M x from the Q8_0 copy agrees with the transformed matvec to Q8_0 noise.
  constexpr std::uint32_t kIn = 512;
  constexpr std::uint32_t kOut = 384;
  Stack st(kIn, kOut, k, 2, 0x7070U + k);
  st.Prepare();
  const std::size_t bytes = q::Exl3Q8Bytes(st.device);
  void* d = nullptr;
  CheckHip(hipMalloc(&d, bytes), "q8");
  Expect(q::Exl3DequantizeQ8(st.device, 1, d, nullptr), "dequant launch");
  CheckHip(hipDeviceSynchronize(), "sync");
  std::vector<std::uint8_t> q8(bytes);
  CheckHip(hipMemcpy(q8.data(), d, bytes, hipMemcpyDeviceToHost), "q8 copy");
  (void)hipFree(d);
  std::uint32_t s = 3U + k;
  std::vector<float> x(kIn);
  for (auto& v : x)
    v = Uniform(&s);
  std::vector<float> y(kOut, 0.0F);
  for (std::uint32_t o = 0; o < kOut; ++o) {
    for (std::uint32_t b = 0; b < kIn / 32; ++b) {
      const std::uint8_t* block = q8.data() + (o * (kIn / 32) + b) * 34;
      std::uint16_t dh = 0;
      std::memcpy(&dh, block, 2);
      const float dq = fmt::HalfToFloat(dh);
      for (int i = 0; i < 32; ++i)
        y[o] += dq * static_cast<std::int8_t>(block[2 + i]) * x[b * 32 + i];
    }
  }
  const auto ref = fmt::MatVec(st.Host(1), x);
  const double err = RelErr(y.data(), ref);
  Expect(err < 1e-2, "dequantized Q8_0 K" + std::to_string(k) + " rel " +
                         std::to_string(err));
}

void SelectCase() {
  // Output blocks {3, 0} of a 512-wide matrix equal those columns of the
  // full result.
  Stack full(256, 512, 4, 1, 4242U);
  full.Prepare();
  Stack part(256, 512, 4, 1, 4242U);
  part.Prepare({3, 0});
  std::uint32_t s = 5U;
  std::vector<float> x(256);
  for (auto& v : x)
    v = Uniform(&s);
  Scratch scratch(1 << 16, 1 << 20);
  std::vector<float> a;
  std::vector<float> b;
  Expect(RunLinear(full, x, 256, 1, {}, 1, scratch, &a), "full launch");
  Expect(RunLinear(part, x, 256, 1, {}, 1, scratch, &b), "part launch");
  bool same = true;
  for (std::uint32_t j = 0; j < 256; ++j) {
    const std::uint32_t src = (j < 128 ? 3 * 128 : 0) + j % 128;
    same = same && std::fabs(a[src] - b[j]) <= 1e-6F * (1 + std::fabs(a[src]));
  }
  Expect(same, "output block selection");
}

void NgramCase() {
  // The host row codec against a direct bit walk for K4..K6 and K8.
  for (const int k : {4, 5, 6, 8}) {
    std::uint32_t s = 11U + k;
    std::vector<std::uint16_t> row(fmt::NgramRowWords(k));
    for (auto& w : row)
      w = static_cast<std::uint16_t>(Next(&s));
    row[0] = fmt::FloatToHalf(0.25F);
    std::vector<float> out(fmt::kNgramRowDim);
    fmt::DecodeNgramRow(row.data(), k, nullptr, out.data());
    bool ok = true;
    for (int i = 0; i < static_cast<int>(fmt::kNgramRowDim); ++i) {
      std::uint32_t state = 0;
      for (int b = 0; b < 16; ++b) {
        // Bit b of the state is bit (b mod K) of symbol i - b / K.
        const int sym = ((i - b / k) % 160 + 160) % 160;
        const int bit = sym * k + b % k;
        const std::uint32_t v = (row[1 + bit / 16] >> (bit % 16)) & 1U;
        state |= v << b;
      }
      ok = ok && out[i] == fmt::Mul1Value(state) * 0.25F;
    }
    Expect(ok, "n-gram row codec K" + std::to_string(k));
  }
}

}  // namespace

int main() {
  try {
    for (const int k : {2, 3, 4, 5, 6}) {
      DenseCase(256, 384, k, 1);
      DenseCase(2560, 512, k, 11);
      ExpertCase(k);
    }
    for (const int k : {2, 3, 4, 5, 6}) {
      GemmDenseCase(256, 384, k, 600);
      GemmRoutedCase(k);
    }
    for (const int k : {2, 4, 6})
      DequantCase(k);
    GemmDenseCase(2560, 640, 4, 129);
    GemmDenseCase(6144, 2560, 6, 33);
    DenseCase(640, 2560, 4, 3);
    DenseCase(2560, 640, 4, 70);
    DenseCase(6144, 2560, 6, 65);
    DenseCase(6144, 2560, 6, 9);
    SelectCase();
    NgramCase();
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " failures\n";
    return 1;
  }
  std::cout << "exl3 ops: ok\n";
  return 0;
}

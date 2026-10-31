// A/B of Gemma 4's WMMA verification GEMV (LaunchWmmaGemv) over GGUF rows
// against the same bytes in 128-row groups, each super-block's rows back to
// back (grouped). Q4_K weights, cold (rotating copies past the MALL).
//
//   nix develop -c tools/bench/build.sh gemma4_verify_layout_bench
//   /tmp/gemma4_verify_layout_bench [rounds] [format]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "src/models/gemma4/kernels/rocm/wmma_gemv.hip.cpp"

namespace {

namespace g4 = gufo::models::gemma4::rocm;

#define CK(x)                                                         \
  do {                                                                \
    const hipError_t e = (x);                                         \
    if (e != hipSuccess) {                                            \
      std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e));    \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

/// The grouped layout of GGUF rows [m][supers * kSb].
std::vector<std::uint8_t> Grouped(const std::vector<std::uint8_t>& w,
                                  std::uint32_t m, std::uint32_t supers,
                                  int kSb) {
  std::vector<std::uint8_t> out(w.size());
  for (std::uint32_t r = 0; r < m; ++r) {
    const std::uint32_t g = r / 128;
    const std::uint32_t rows = std::min(128U, m - g * 128);
    for (std::uint32_t sb = 0; sb < supers; ++sb) {
      const std::size_t dst =
          (std::size_t{g} * 128 * supers + std::size_t{sb} * rows + r % 128) *
          kSb;
      std::memcpy(&out[dst], &w[(std::size_t{r} * supers + sb) * kSb],
                  static_cast<std::size_t>(kSb));
    }
  }
  return out;
}

float Median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

struct Format {
  const char* name;
  gufo::core::GgmlType type;
  int super_bytes;  // per 256 values
};

/// Finite scales: every scale field of every super-block set to small
/// binary16 values (the codes stay random).
void Finite(std::vector<std::uint8_t>& w, const Format& f) {
  const __half d = __float2half(0.01F);
  const __half dm = __float2half(0.002F);
  for (std::size_t i = 0; i < w.size(); i += f.super_bytes) {
    switch (f.type) {
      case gufo::core::GgmlType::kQ4_K:
      case gufo::core::GgmlType::kQ5_K:
        std::memcpy(&w[i], &d, 2);
        std::memcpy(&w[i + 2], &dm, 2);
        break;
      case gufo::core::GgmlType::kQ6_K:
        std::memcpy(&w[i + 208], &d, 2);
        break;
      default:  // Q4_0: eight 18-byte blocks
        for (int b = 0; b < 8; ++b) {
          std::memcpy(&w[i + 18 * b], &d, 2);
        }
    }
  }
}

int main(int argc, char** argv) {
  const int rounds = argc > 1 ? std::atoi(argv[1]) : 3;
  const std::string only = argc > 2 ? argv[2] : "";
  const Format formats[] = {{"Q4_K", gufo::core::GgmlType::kQ4_K, 144},
                            {"Q5_K", gufo::core::GgmlType::kQ5_K, 176},
                            {"Q6_K", gufo::core::GgmlType::kQ6_K, 210},
                            {"Q4_0", gufo::core::GgmlType::kQ4_0, 144}};
  const std::uint32_t shapes[][2] = {
      {5376, 16384}, {8192, 5376},  {43008, 5376}, {5376, 21504},
      {12288, 5376}, {18432, 5376}, {4096, 5376},  {5376, 8192}};
  const std::uint32_t widths[] = {1, 5, 8, 9, 16};
  std::printf("%-5s %-12s %5s %9s %9s %7s\n", "fmt", "shape", "rows",
              "gguf us", "grouped", "gain");
  std::mt19937 rng(20261021);
  for (const Format& f : formats) {
  if (!only.empty() && only != f.name) {
    continue;
  }
  for (const auto& sh : shapes) {
    const std::uint32_t m = sh[0], k = sh[1], supers = k / 256;
    const int kSb = f.super_bytes;
    const std::size_t bytes = std::size_t{m} * supers * kSb;
    std::vector<std::uint8_t> w(bytes);
    for (auto& b : w) {
      b = static_cast<std::uint8_t>(rng());
    }
    Finite(w, f);
    const auto grouped = Grouped(w, m, supers, kSb);
    const int copies =
        std::max(2, static_cast<int>((std::size_t{160} << 20) / bytes + 1));
    std::vector<void*> a(copies), b(copies);
    for (int c = 0; c < copies; ++c) {
      CK(hipMalloc(&a[c], bytes));
      CK(hipMalloc(&b[c], bytes));
      CK(hipMemcpy(a[c], w.data(), bytes, hipMemcpyHostToDevice));
      CK(hipMemcpy(b[c], grouped.data(), bytes, hipMemcpyHostToDevice));
    }
    std::vector<float> x(std::size_t{16} * k);
    // Uniform in [-2, 2): no libm (hipcc links the host's, not the Nix one).
    for (auto& v : x) {
      v = static_cast<float>(rng() % 4096) / 1024.0F - 2.0F;
    }
    float *dx = nullptr, *ya = nullptr, *yb = nullptr;
    void* scratch = nullptr;
    CK(hipMalloc(&dx, x.size() * 4));
    CK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&ya, std::size_t{16} * m * 4));
    CK(hipMalloc(&yb, std::size_t{16} * m * 4));
    CK(hipMalloc(&scratch, g4::WmmaGemvScratchBytes(k)));
    for (const std::uint32_t rows : widths) {
      int slot = 0;
      const auto run = [&](bool g, float* y) {
        const int c = slot++ % copies;
        if (!g4::LaunchWmmaGemv(f.type, g ? b[c] : a[c],
                                dx, y, rows, m, k, scratch, nullptr, false,
                                g)) {
          std::fprintf(stderr, "launch rejected\n");
          std::exit(1);
        }
      };
      run(false, ya);
      run(true, yb);
      CK(hipDeviceSynchronize());
      std::vector<float> ha(std::size_t{rows} * m), hb(ha.size());
      CK(hipMemcpy(ha.data(), ya, ha.size() * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(hb.data(), yb, hb.size() * 4, hipMemcpyDeviceToHost));
      if (std::memcmp(ha.data(), hb.data(), ha.size() * 4) != 0) {
        std::fprintf(stderr, "%s %ux%u rows %u: outputs differ\n", f.name, m,
                     k, rows);
        std::exit(2);
      }
      std::vector<float> ta, tb;
      hipEvent_t e0, e1;
      CK(hipEventCreate(&e0));
      CK(hipEventCreate(&e1));
      const auto time = [&](bool g) {
        CK(hipEventRecord(e0));
        for (int i = 0; i < 64; ++i) {
          run(g, g ? yb : ya);
        }
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float ms = 0;
        CK(hipEventElapsedTime(&ms, e0, e1));
        return 1000.0F * ms / 64;
      };
      for (int r = 0; r < rounds * 3; ++r) {
        if (r % 2 == 0) {
          ta.push_back(time(false));
          tb.push_back(time(true));
        } else {
          tb.push_back(time(true));
          ta.push_back(time(false));
        }
      }
      const float ma = Median(ta), mb = Median(tb);
      std::printf("%-5s %5ux%-6u %5u %9.1f %9.1f %+6.1f%%\n", f.name, m, k,
                  rows, ma, mb, (ma / mb - 1.0F) * 100.0F);
      std::fflush(stdout);
    }
    for (int c = 0; c < copies; ++c) {
      CK(hipFree(a[c]));
      CK(hipFree(b[c]));
    }
    CK(hipFree(dx));
    CK(hipFree(ya));
    CK(hipFree(yb));
    CK(hipFree(scratch));
  }
  }
  return 0;
}

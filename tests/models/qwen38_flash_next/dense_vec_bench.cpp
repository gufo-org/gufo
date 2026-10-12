// Times the decode matvec kernels per weight format at the model's real shapes
// (diagnostic): effective weight bandwidth shows which formats reach DRAM
// speed.
//
// dense_vec_bench [reps]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace {

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

struct Format {
  const char* name;
  int type;
  std::size_t block_elems;
  std::size_t block_bytes;
};

double TimeMs(int reps, const std::function<void()>& launch) {
  hipEvent_t a, b;
  Check(hipEventCreate(&a), "ev");
  Check(hipEventCreate(&b), "ev");
  for (int i = 0; i < 3; ++i)
    launch();
  Check(hipDeviceSynchronize(), "warm");
  std::vector<float> ms(reps);
  for (int i = 0; i < reps; ++i) {
    Check(hipEventRecord(a, nullptr), "rec");
    launch();
    Check(hipEventRecord(b, nullptr), "rec");
    Check(hipEventSynchronize(b), "sync");
    Check(hipEventElapsedTime(&ms[i], a, b), "elapsed");
  }
  std::sort(ms.begin(), ms.end());
  return ms[reps / 2];
}

}  // namespace

int main(int argc, char** argv) try {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 40;
  qfn_mmq_init(0);
  const Format formats[] = {{"Q8_0", 8, 32, 34},
                            {"Q6_K", 14, 256, 210},
                            {"Q5_K", 13, 256, 176},
                            {"Q5_1", 7, 32, 24},
                            {"Q4_K", 12, 256, 144}};
  struct Shape {
    const char* name;
    int m, k;
  };
  const Shape shapes[] = {{"head 131072x2560", 131072, 2560},
                          {"ssm in 16384x2560", 16384, 2560},
                          {"attn qkv 13312x2560", 13312, 2560},
                          {"out 2560x6144", 2560, 6144}};
  for (const auto& s : shapes) {
    for (int n : {1, 3}) {
      std::printf("%-20s n=%d\n", s.name, n);
      float* x = nullptr;
      float* out = nullptr;
      void* xq = nullptr;
      Check(hipMalloc(&x, std::size_t{8} * s.k * sizeof(float)), "x");
      Check(hipMalloc(&out, std::size_t{8} * s.m * sizeof(float)), "out");
      Check(hipMalloc(&xq, qfn_mmq_q8_1_bytes(8, s.k)), "xq");
      std::vector<float> hx(std::size_t{8} * s.k, 0.01F);
      Check(hipMemcpy(x, hx.data(), hx.size() * sizeof(float),
                      hipMemcpyHostToDevice),
            "x up");
      for (const auto& f : formats) {
        const std::size_t bytes = static_cast<std::size_t>(s.m) *
                                  (s.k / f.block_elems) * f.block_bytes;
        // Cycle through enough distinct copies that every call reads cold
        // weights.
        const int copies = static_cast<int>(
            std::clamp<std::size_t>((std::size_t{1} << 29) / bytes + 1, 2, 24));
        std::vector<void*> ws(copies);
        std::vector<std::uint8_t> filler(std::min<std::size_t>(bytes, 1 << 26));
        for (std::size_t i = 0; i < filler.size(); ++i)
          filler[i] = static_cast<std::uint8_t>(i * 2654435761u >> 24);
        for (auto& w : ws) {
          Check(hipMalloc(&w, bytes + 4096), "w");
          for (std::size_t off = 0; off < bytes; off += filler.size())
            Check(hipMemcpy(static_cast<char*>(w) + off, filler.data(),
                            std::min(filler.size(), bytes - off),
                            hipMemcpyHostToDevice),
                  "w up");
        }
        int rc = 0;
        int call = 0;
        double ms = 0;
        if (f.type == 8) {
          rc = qfn_mmq_quantize_q8_1(x, xq, n, s.k, nullptr);
          ms = TimeMs(reps, [&] {
            rc |= qfn_mmq_q8_0_dense_vec_preq(ws[call++ % copies], nullptr, xq,
                                              out, s.m, n, s.k, nullptr);
          });
        } else {
          ms = TimeMs(reps, [&] {
            rc |= qfn_mmq_dense_vec(f.type, ws[call++ % copies], x, out, s.m,
                                    s.k, n, nullptr);
          });
        }
        if (rc != 0) {
          std::printf("  %-5s unsupported (rc %d)\n", f.name, rc);
        } else {
          std::printf("  %-5s %8.1f us  %6.1f MB weights  %6.1f GB/s\n", f.name,
                      ms * 1000.0, bytes / 1e6, bytes / (ms * 1e6));
        }
        for (auto w : ws)
          Check(hipFree(w), "free w");
      }
      Check(hipFree(x), "free");
      Check(hipFree(out), "free");
      Check(hipFree(xq), "free");
    }
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}

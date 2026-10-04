// Times the dense F16 prefill projections on synthetic weights
// (diagnostic): the Q8_0 2560 x 6144 output projection, the 2560 x 640
// shared-expert down projection, the fused SSM projection (16384 x 2560 with
// its convolution), the stacked attention projection (13312 x 2560 with norms
// and RoPE), and the F16 router (513 x 2560) and recurrent gates (96 x 2560).
//
// dense_gemm_bench [tokens] [reps]
//
// Each line prints the median launch time, TFLOPS and an FNV-1a hash of the
// output, so launch-order changes can be checked for identical results.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, host.size() * sizeof(T)), "hipMalloc");
  CheckHip(
      hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
      "hipMemcpy");
  return static_cast<T*>(p);
}

void* Zeros(std::size_t bytes) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, bytes), "hipMalloc");
  CheckHip(hipMemset(p, 0, bytes), "hipMemset");
  return p;
}

// Q8_0 rows: per 32 codes a F16 scale near 1/256 and random codes.
std::vector<std::uint8_t> Q8Weights(std::size_t rows, std::size_t k,
                                    std::uint32_t seed) {
  const std::size_t blocks = rows * (k / 32);
  std::vector<std::uint8_t> w(blocks * 34);
  for (std::size_t b = 0; b < blocks; ++b) {
    const __half d = __float2half(
        (1.0F + static_cast<float>(Next(&seed) % 64) / 64.0F) / 256.0F);
    std::memcpy(&w[b * 34], &d, 2);
    for (int i = 0; i < 32; ++i)
      w[b * 34 + 2 + i] = static_cast<std::uint8_t>(Next(&seed));
  }
  return w;
}

std::vector<__half> Activations(std::size_t count, std::uint32_t seed) {
  std::vector<__half> x(count);
  for (auto& v : x)
    v = __float2half(
        static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
        1000.0F);
  return x;
}

std::uint64_t Hash(const void* device, std::size_t bytes) {
  std::vector<std::uint8_t> host(bytes);
  CheckHip(hipMemcpy(host.data(), device, bytes, hipMemcpyDeviceToHost),
           "hipMemcpy");
  std::uint64_t h = 1469598103934665603ULL;
  for (std::uint8_t c : host)
    h = (h ^ c) * 1099511628211ULL;
  return h;
}

double MedianMs(const std::function<void()>& launch, int reps) {
  hipEvent_t start;
  hipEvent_t stop;
  CheckHip(hipEventCreate(&start), "hipEventCreate");
  CheckHip(hipEventCreate(&stop), "hipEventCreate");
  for (int i = 0; i < 2; ++i)
    launch();
  std::vector<float> ms(reps);
  for (int i = 0; i < reps; ++i) {
    CheckHip(hipEventRecord(start, nullptr), "hipEventRecord");
    launch();
    CheckHip(hipEventRecord(stop, nullptr), "hipEventRecord");
    CheckHip(hipEventSynchronize(stop), "hipEventSynchronize");
    CheckHip(hipEventElapsedTime(&ms[i], start, stop), "hipEventElapsedTime");
  }
  std::sort(ms.begin(), ms.end());
  return ms[reps / 2];
}

}  // namespace

int main(int argc, char** argv) try {
  const std::uint32_t tokens = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 10;
  const auto report = [&](const char* name, double ms, double flops,
                          std::uint64_t hash) {
    std::printf("%-5s tokens %u  %8.3f ms  %6.2f TFLOPS  hash %016llx\n", name,
                tokens, ms, flops / ms / 1e9,
                static_cast<unsigned long long>(hash));
  };

  // Output projection: [tokens][6144] -> [tokens][2560].
  {
    constexpr std::size_t m = 2560;
    constexpr std::size_t k = 6144;
    auto* w = Upload(Q8Weights(m, k, 11U));
    auto* x = Upload(Activations(tokens * k, 12U));
    auto* y = static_cast<float*>(Zeros(tokens * m * sizeof(float)));
    const double ms = MedianMs(
        [&] {
          if (!q::DenseF16Gemm(w, x, y, tokens, m, k, nullptr))
            throw std::runtime_error("DenseF16Gemm");
        },
        reps);
    report("out", ms, 2.0 * tokens * m * k,
           Hash(y, tokens * m * sizeof(float)));
    CheckHip(hipFree(w), "hipFree");
    CheckHip(hipFree(x), "hipFree");
    CheckHip(hipFree(y), "hipFree");
  }

  // Shared-expert down projection: [tokens][640] -> [tokens][2560].
  {
    constexpr std::size_t m = 2560;
    constexpr std::size_t k = 640;
    auto* w = Upload(Q8Weights(m, k, 41U));
    auto* x = Upload(Activations(tokens * k, 42U));
    auto* y = static_cast<float*>(Zeros(tokens * m * sizeof(float)));
    const double ms = MedianMs(
        [&] {
          if (!q::DenseF16Gemm(w, x, y, tokens, m, k, nullptr))
            throw std::runtime_error("DenseF16Gemm");
        },
        reps);
    report("down", ms, 2.0 * tokens * m * k,
           Hash(y, tokens * m * sizeof(float)));
    CheckHip(hipFree(w), "hipFree");
    CheckHip(hipFree(x), "hipFree");
    CheckHip(hipFree(y), "hipFree");
  }

  // F16 router and recurrent gates (unquantized weights).
  for (const std::size_t m : {std::size_t{513}, std::size_t{96}}) {
    constexpr std::size_t k = 2560;
    auto* w = Upload(Activations(m * k, 51U));
    auto* x = Upload(Activations(tokens * k, 52U));
    auto* y = static_cast<float*>(Zeros(tokens * m * sizeof(float)));
    const double ms = MedianMs(
        [&] {
          if (!q::UnquantizedF16Gemm(w, x, y, tokens, m, k, nullptr))
            throw std::runtime_error("UnquantizedF16Gemm");
        },
        reps);
    report(m == 513 ? "route" : "gates", ms, 2.0 * tokens * m * k,
           Hash(y, tokens * m * sizeof(float)));
    CheckHip(hipFree(w), "hipFree");
    CheckHip(hipFree(x), "hipFree");
    CheckHip(hipFree(y), "hipFree");
  }

  // SSM projection fused with the four-tap convolution.
  {
    constexpr std::uint32_t m = 16384;
    constexpr std::uint32_t k = 2560;
    constexpr std::uint32_t channels = 10240;
    auto* w = Upload(Q8Weights(m, k, 21U));
    auto* x = Upload(Activations(std::size_t{tokens} * k, 22U));
    std::vector<float> conv(channels * 4);
    std::uint32_t seed = 23U;
    for (auto& v : conv)
      v = static_cast<float>(Next(&seed) % 1000) / 2000.0F;
    auto* conv_w = Upload(conv);
    auto* history =
        static_cast<float*>(Zeros(std::size_t{3} * channels * sizeof(float)));
    auto* qkvz =
        static_cast<float*>(Zeros(std::size_t{tokens} * m * sizeof(float)));
    auto* convolved = static_cast<float*>(
        Zeros(std::size_t{tokens} * channels * sizeof(float)));
    const double ms = MedianMs(
        [&] {
          if (!q::DenseF16SsmGemm(w, x, conv_w, history, qkvz, convolved,
                                  tokens, m, k, channels, 4, nullptr))
            throw std::runtime_error("DenseF16SsmGemm");
        },
        reps);
    report("ssm", ms, 2.0 * tokens * m * k,
           Hash(convolved, std::size_t{tokens} * channels * sizeof(float)));
    for (void* p : {static_cast<void*>(w), static_cast<void*>(x),
                    static_cast<void*>(conv_w), static_cast<void*>(history),
                    static_cast<void*>(qkvz), static_cast<void*>(convolved)})
      CheckHip(hipFree(p), "hipFree");
  }

  // Stacked attention projection: 24 query, 24 gate, 2 key and 2 value heads.
  {
    constexpr std::uint32_t heads = 24;
    constexpr std::uint32_t kv_heads = 2;
    constexpr std::uint32_t m = 2 * (heads + kv_heads) * 256;
    constexpr std::uint32_t k = 2560;
    auto* w = Upload(Q8Weights(m, k, 31U));
    auto* x = Upload(Activations(std::size_t{tokens} * k, 32U));
    std::vector<float> gamma(256, 1.0F);
    auto* q_gamma = Upload(gamma);
    auto* k_gamma = Upload(gamma);
    auto* position = Upload(std::vector<std::uint32_t>{0U});
    const std::size_t q_bytes =
        std::size_t{tokens} * heads * 256 * sizeof(float);
    const std::size_t kv_bytes =
        std::size_t{tokens} * kv_heads * 256 * sizeof(__half);
    auto* query = static_cast<float*>(Zeros(q_bytes));
    auto* gate = static_cast<float*>(Zeros(q_bytes));
    auto* keys = static_cast<__half*>(Zeros(kv_bytes));
    auto* values = static_cast<__half*>(Zeros(kv_bytes));
    const double ms = MedianMs(
        [&] {
          if (!q::AttentionF16Gemm(w, x, q_gamma, k_gamma, query, gate, keys,
                                   values, tokens, position, 1.0e7F, 1.0e-6F,
                                   nullptr))
            throw std::runtime_error("AttentionF16Gemm");
        },
        reps);
    const std::uint64_t hash = Hash(query, q_bytes) ^ Hash(gate, q_bytes) ^
                               Hash(keys, kv_bytes) ^ Hash(values, kv_bytes);
    report("attn", ms, 2.0 * tokens * m * k, hash);
    for (void* p : {static_cast<void*>(w), static_cast<void*>(x),
                    static_cast<void*>(q_gamma), static_cast<void*>(k_gamma),
                    static_cast<void*>(position), static_cast<void*>(query),
                    static_cast<void*>(gate), static_cast<void*>(keys),
                    static_cast<void*>(values)})
      CheckHip(hipFree(p), "hipFree");
  }

  // BF16 indexer projections (query 512 x 2560, key 128 x 2560, and a TP2
  // rank's halves): the library kernel against BlasLt::Gemm, which runs
  // DenseBf16Gemm where that reproduces the library. Timed at the requested
  // tokens; every token count up to it must match bitwise.
  {
    std::string error;
    auto blas = q::BlasLt::Create(nullptr, &error);
    if (!blas)
      throw std::runtime_error(error);
    constexpr int k = 2560;
    const auto bf16 = [](std::size_t count, std::uint32_t seed, float scale) {
      std::vector<std::uint16_t> v(count);
      for (auto& e : v) {
        const float f =
            scale *
            static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
            1000.0F;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &f, 4);
        e = static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >>
                                       16);
      }
      return v;
    };
    const auto host = [](const float* d, std::size_t count) {
      std::vector<float> h(count);
      CheckHip(hipMemcpy(h.data(), d, count * 4, hipMemcpyDeviceToHost),
               "hipMemcpy");
      return h;
    };
    for (const int m : {512, 128, 256, 64}) {
      auto* w = Upload(bf16(std::size_t{512} * k, 51U + m, 0.05F));
      auto* x = Upload(bf16(std::size_t{tokens} * k, 52U, 3.0F));
      const std::size_t bytes = std::size_t{tokens} * m * sizeof(float);
      auto* y_lib = static_cast<float*>(Zeros(bytes));
      auto* y_own = static_cast<float*>(Zeros(bytes));
      const auto lib = [&](int n) {
        if (!blas->LibraryGemm(w, x, y_lib, HIP_R_16BF, m, n, k, &error))
          throw std::runtime_error(error);
      };
      const auto own = [&](int n) {
        if (!blas->Gemm(w, x, y_own, HIP_R_16BF, m, n, k, &error))
          throw std::runtime_error(error);
      };
      const int t = static_cast<int>(tokens);
      char name[16];
      std::snprintf(name, sizeof(name), "bf%d", m);
      std::printf("%s-lib", name);
      report("", MedianMs([&] { lib(t); }, reps), 2.0 * tokens * m * k,
             Hash(y_lib, bytes));
      std::printf("%s-own", name);
      report("", MedianMs([&] { own(t); }, reps), 2.0 * tokens * m * k,
             Hash(y_own, bytes));
      int routed = 0;
      int differ = 0;
      std::string library_only;
      for (int n = 1; n <= t; ++n) {
        const bool uses_own = blas->UsesOwnKernel(HIP_R_16BF, m, n, k, &error);
        routed += uses_own;
        if (!uses_own) {
          if (library_only.size() < 200)
            library_only += " " + std::to_string(n);
          continue;
        }
        lib(n);
        own(n);
        CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
        const auto a = host(y_lib, std::size_t{static_cast<unsigned>(n)} * m);
        const auto b = host(y_own, std::size_t{static_cast<unsigned>(n)} * m);
        if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) {
          if (++differ <= 10)
            std::printf("%s n %d DIFF\n", name, n);
        }
      }
      std::printf(
          "%s n 1..%d: own kernel at %d, bitwise differ at %d; "
          "library only at:%s\n",
          name, t, routed, differ, library_only.c_str());
      for (void* p : {static_cast<void*>(w), static_cast<void*>(x),
                      static_cast<void*>(y_lib), static_cast<void*>(y_own)})
        CheckHip(hipFree(p), "hipFree");
    }
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "dense_gemm_bench: %s\n", e.what());
  return 1;
}

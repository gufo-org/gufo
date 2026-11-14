// Times the EXL3 decode linears on random weights: the Flash-Next dense
// and routed-expert shapes at one token, in GB/s of weight traffic.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/exl3.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;

namespace {

void* Alloc(std::size_t bytes) {
  void* p = nullptr;
  if (hipMalloc(&p, bytes) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc failed\n");
    std::exit(1);
  }
  (void)hipMemset(p, 0x5A, bytes);
  return p;
}

void Case(const char* name, std::uint32_t in, std::uint32_t out, int k,
          std::uint32_t experts, std::uint32_t rows, const q::Exl3Scratch& s) {
  q::Exl3Weights w{};
  w.in = in;
  w.out = out;
  w.k = k;
  w.experts = experts;
  w.tiles = static_cast<std::uint32_t*>(Alloc(w.ExpertWords() * 4 * experts));
  std::vector<__half> scale(static_cast<std::size_t>(experts) * (in + out),
                            __float2half(0.01F));
  auto* suh = static_cast<__half*>(Alloc(scale.size() * 2));
  (void)hipMemcpy(suh, scale.data(), scale.size() * 2, hipMemcpyHostToDevice);
  w.suh = suh;
  w.svh = suh + static_cast<std::size_t>(experts) * in;
  auto* x = static_cast<float*>(Alloc(std::size_t{rows} * in * 4));
  (void)hipMemset(x, 0, std::size_t{rows} * in * 4);
  auto* y = static_cast<float*>(Alloc(std::size_t{rows} * out * 4));
  std::int32_t* ids = nullptr;
  if (experts > 1) {
    std::vector<std::int32_t> host(rows);
    for (std::uint32_t r = 0; r < rows; ++r)
      host[r] = static_cast<std::int32_t>((r * 53) % experts);
    ids = static_cast<std::int32_t*>(Alloc(rows * 4));
    (void)hipMemcpy(ids, host.data(), rows * 4, hipMemcpyHostToDevice);
  }
  const std::uint32_t x_div = experts > 1 ? rows : 1;
  hipEvent_t a;
  hipEvent_t b;
  (void)hipEventCreate(&a);
  (void)hipEventCreate(&b);
  constexpr int kIters = 200;
  for (int i = 0; i < 10; ++i)
    (void)q::Exl3Linear(w, x, in, x_div, ids, y, out, rows, s, nullptr);
  (void)hipEventRecord(a, nullptr);
  for (int i = 0; i < kIters; ++i)
    (void)q::Exl3Linear(w, x, in, x_div, ids, y, out, rows, s, nullptr);
  (void)hipEventRecord(b, nullptr);
  (void)hipEventSynchronize(b);
  float ms = 0.0F;
  (void)hipEventElapsedTime(&ms, a, b);
  const double us = 1000.0 * ms / kIters;
  const double bytes =
      static_cast<double>(in) * out * k / 8.0 * (experts > 1 ? rows : 1);
  std::printf("%-22s K%d %5ux%-6u rows %2u: %8.1f us  %6.1f GB/s\n", name, k,
              in, out, rows, us, bytes / us / 1e3);
  (void)hipFree(const_cast<std::uint32_t*>(w.tiles));
  (void)hipFree(suh);
  (void)hipFree(x);
  (void)hipFree(y);
  if (ids != nullptr)
    (void)hipFree(ids);
}

/// Prompt-pass routed experts: `rows` assignments per expert, E experts.
void RoutedCase(const char* name, std::uint32_t in, std::uint32_t out, int k,
                std::uint32_t experts, std::uint32_t rows,
                const q::Exl3Scratch& s) {
  q::Exl3Weights w{};
  w.in = in;
  w.out = out;
  w.k = k;
  w.experts = experts;
  w.tiles = static_cast<std::uint32_t*>(Alloc(w.ExpertWords() * 4 * experts));
  std::vector<__half> scale(static_cast<std::size_t>(experts) * (in + out),
                            __float2half(0.01F));
  auto* suh = static_cast<__half*>(Alloc(scale.size() * 2));
  (void)hipMemcpy(suh, scale.data(), scale.size() * 2, hipMemcpyHostToDevice);
  w.suh = suh;
  w.svh = suh + static_cast<std::size_t>(experts) * in;
  const std::uint32_t padded = (rows + 15) / 16 * 16;
  const std::uint32_t slots = rows * experts;
  std::vector<std::int32_t> bounds, x_rows, y_rows, tiles128, tiles64;
  for (std::uint32_t e = 0; e <= experts; ++e)
    bounds.push_back(static_cast<std::int32_t>(e * padded));
  std::uint32_t slot = 0;
  for (std::uint32_t e = 0; e < experts; ++e) {
    for (std::uint32_t i = 0; i < padded; ++i) {
      const bool live = i < rows;
      // Ten slots per token: 4096 token rows feed every expert.
      x_rows.push_back(
          live ? static_cast<std::int32_t>((slot * 7919) % (slots / 10)) : -1);
      y_rows.push_back(live ? static_cast<std::int32_t>(slot) : -1);
      slot += live ? 1 : 0;
    }
    for (std::uint32_t j = 0; j * q::kExl3GemmRows < rows; ++j)
      tiles128.push_back(static_cast<std::int32_t>(e | (j << 16)));
    for (std::uint32_t j = 0; j * q::kExl3TransformedGemmRows < rows; ++j)
      tiles64.push_back(static_cast<std::int32_t>(e | (j << 16)));
  }
  const auto up = [](const std::vector<std::int32_t>& v) {
    auto* p = static_cast<std::int32_t*>(Alloc(v.size() * 4));
    (void)hipMemcpy(p, v.data(), v.size() * 4, hipMemcpyHostToDevice);
    return p;
  };
  auto* d_bounds = up(bounds);
  auto* d_x = up(x_rows);
  auto* d_y = up(y_rows);
  auto* d_t = up(tiles128);
  auto* xf = static_cast<float*>(Alloc(std::size_t{slots} * in * 4));
  auto* xh = static_cast<__half*>(Alloc(std::size_t{slots} * in * 2));
  {
    // Random inputs: zeros understate the power-bound matrix-core cost.
    std::vector<float> hf(std::size_t{slots} * in);
    std::vector<__half> hh(hf.size());
    std::uint32_t r = 12345U;
    for (std::size_t i = 0; i < hf.size(); ++i) {
      r = r * 1664525U + 1013904223U;
      hf[i] = static_cast<float>(r >> 8) / 16777216.0F - 0.5F;
      hh[i] = __float2half(hf[i]);
    }
    (void)hipMemcpy(xf, hf.data(), hf.size() * 4, hipMemcpyHostToDevice);
    (void)hipMemcpy(xh, hh.data(), hh.size() * 2, hipMemcpyHostToDevice);
  }
  auto* y = static_cast<float*>(Alloc(std::size_t{slots} * out * 4));
  std::vector<std::int32_t> ids(slots);
  for (std::uint32_t e = 0, n = 0; e < experts; ++e)
    for (std::uint32_t i = 0; i < rows; ++i)
      ids[n++] = static_cast<std::int32_t>(e);
  auto* d_ids = up(ids);
  q::Exl3Scratch big = s;
  big.xh = static_cast<__half*>(Alloc(std::size_t{slots} * in * 2));
  big.xh_elems = std::size_t{slots} * in;
  auto* d_t256 = up(tiles64);
  const q::Exl3GemmRoute route{
      d_t256, static_cast<std::uint32_t>(tiles64.size()), d_bounds, d_y, d_y};
  const q::Exl3GemmRoute from_tokens{
      d_t, static_cast<std::uint32_t>(tiles128.size()), d_bounds, d_x, d_y};
  hipEvent_t a, b;
  (void)hipEventCreate(&a);
  (void)hipEventCreate(&b);
  const auto time = [&](auto&& f) {
    // Sustained load, as in a prompt pass: short bursts run at boost clocks.
    for (int i = 0; i < 100; ++i)
      f();
    (void)hipEventRecord(a, nullptr);
    for (int i = 0; i < 100; ++i)
      f();
    (void)hipEventRecord(b, nullptr);
    (void)hipEventSynchronize(b);
    float ms = 0.0F;
    (void)hipEventElapsedTime(&ms, a, b);
    return ms / 100;
  };
  const float t1 = time([&] {
    (void)q::Exl3LinearGemm(w, xf, in, 1, d_ids, slots, y, out, 0, route, big,
                            nullptr);
  });
  const float t2 = time([&] {
    (void)q::Exl3LinearGemm(w, xf, in, 1, d_ids, slots, y, out, 0, route, big,
                            nullptr, {xh, nullptr, nullptr});
  });
  const float t3 = time([&] {
    (void)q::Exl3LinearGemm(w, xf, in, 1, d_ids, slots, y, out, 0, from_tokens,
                            big, nullptr, {xh, nullptr, nullptr, true});
  });
  const double flops = 2.0 * slots * in * out;
  std::printf(
      "%-14s K%d %ux%u E%u rows %u: scratch route %.2f ms (%.1f TF), "
      "F16 inputs %.2f ms (%.1f TF), fused %.2f ms (%.1f TF)\n",
      name, k, in, out, experts, rows, t1, flops / t1 / 1e9, t2,
      flops / t2 / 1e9, t3, flops / t3 / 1e9);
}

}  // namespace

int main(int argc, char** argv) {
  q::Exl3Scratch s{};
  s.xh = static_cast<__half*>(Alloc(std::size_t{16} << 21));
  s.xh_elems = std::size_t{16} << 20;
  s.partial = static_cast<float*>(Alloc(std::size_t{32} << 22));
  s.partial_elems = std::size_t{32} << 20;
  s.done = static_cast<unsigned*>(Alloc(65536 * 4));
  (void)hipMemset(s.done, 0, 65536 * 4);
  s.done_elems = 65536;
  if (argc > 1 && std::string(argv[1]) == "--routed") {
    // Rows per expert: the second argument (80 by default).
    const std::uint32_t rows =
        argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 80U;
    RoutedCase("experts gate", 2560, 640, 4, 512, rows, s);
    RoutedCase("experts gate", 2560, 640, 2, 512, rows, s);
    RoutedCase("experts down", 640, 2560, 4, 512, rows, s);
    return 0;
  }
  Case("gdn qkv", 2560, 10240, 6, 1, 1, s);
  Case("gdn z", 2560, 6144, 6, 1, 1, s);
  Case("gdn out", 6144, 2560, 6, 1, 1, s);
  Case("attn q", 2560, 12288, 6, 1, 1, s);
  Case("lm head", 2560, 248320, 6, 1, 1, s);
  Case("experts gate", 2560, 640, 4, 512, 10, s);
  Case("experts down", 640, 2560, 4, 512, 10, s);
  Case("experts gate R1", 2560, 640, 4, 512, 1, s);
  Case("experts gate R2", 2560, 640, 4, 512, 2, s);
  Case("experts gate R5", 2560, 640, 4, 512, 5, s);
  Case("experts gate E10", 2560, 640, 4, 10, 10, s);
  Case("experts gate E64", 2560, 640, 4, 64, 10, s);
  Case("experts gate K2", 2560, 640, 2, 512, 10, s);
  Case("experts gate K3", 2560, 640, 3, 512, 10, s);
  Case("shared gate", 2560, 640, 6, 1, 1, s);
  return 0;
}

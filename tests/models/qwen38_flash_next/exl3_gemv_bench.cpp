// Times the EXL3 decode linears on random weights: the Flash-Next dense
// and routed-expert shapes at one token, in GB/s of weight traffic.
//   --routed [rows]       prompt-pass expert GEMMs, rows per expert
//   --counts <file> [K..] prompt-pass gate, up and down over real per-expert
//                         row counts (one line of 512 counts per layer)
//   --dequant             prompt-pass Q8_0 expansion of dense matrices
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
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
  w.front = std::getenv("EXL3_ENGINE_ORDER") != nullptr
                ? 0
                : q::Exl3DecodeSplits(in, out, experts > 1);
  // Copies cycled per launch, at least 256 MiB in all, so every launch
  // reads its weights from DRAM as decode does, not from the 32 MiB MALL.
  const std::size_t matrix = w.ExpertWords() * 4 * experts;
  const std::size_t copies =
      std::max<std::size_t>(1, (std::size_t{256} << 20) / matrix + 1);
  auto* tiles = static_cast<std::uint8_t*>(Alloc(matrix * copies));
  w.tiles = reinterpret_cast<std::uint32_t*>(tiles);
  std::vector<__half> scale(static_cast<std::size_t>(experts) * (in + out),
                            __float2half(0.01F));
  auto* suh = static_cast<__half*>(Alloc(scale.size() * 2));
  (void)hipMemcpy(suh, scale.data(), scale.size() * 2, hipMemcpyHostToDevice);
  w.suh = suh;
  w.svh = suh + static_cast<std::size_t>(experts) * in;
  auto* x = static_cast<float*>(Alloc(std::size_t{rows} * in * 4));
  (void)hipMemset(x, 0, std::size_t{rows} * in * 4);
  auto* y = static_cast<float*>(Alloc(std::size_t{rows} * out * 4));
  constexpr int kIters = 200;
  // Expert stacks take a new expert set per launch for the same reason.
  std::int32_t* ids = nullptr;
  if (experts > 1) {
    std::vector<std::int32_t> host(std::size_t{rows} * kIters);
    for (std::size_t r = 0; r < host.size(); ++r)
      host[r] = static_cast<std::int32_t>((r * 53) % experts);
    ids = static_cast<std::int32_t*>(Alloc(host.size() * 4));
    (void)hipMemcpy(ids, host.data(), host.size() * 4, hipMemcpyHostToDevice);
  }
  const std::uint32_t x_div = experts > 1 ? rows : 1;
  hipEvent_t a;
  hipEvent_t b;
  (void)hipEventCreate(&a);
  (void)hipEventCreate(&b);
  for (int i = 0; i < 10; ++i)
    (void)q::Exl3Linear(w, x, in, x_div, ids, y, out, rows, s, nullptr);
  (void)hipEventRecord(a, nullptr);
  for (int i = 0; i < kIters; ++i) {
    w.tiles = reinterpret_cast<std::uint32_t*>(tiles + (i % copies) * matrix);
    (void)q::Exl3Linear(w, x, in, x_div,
                        ids != nullptr ? ids + std::size_t{rows} * i : nullptr,
                        y, out, rows, s, nullptr);
  }
  (void)hipEventRecord(b, nullptr);
  (void)hipEventSynchronize(b);
  float ms = 0.0F;
  (void)hipEventElapsedTime(&ms, a, b);
  const double us = 1000.0 * ms / kIters;
  const double bytes =
      static_cast<double>(in) * out * k / 8.0 * (experts > 1 ? rows : 1);
  std::printf("%-22s K%d %5ux%-6u rows %2u: %8.1f us  %6.1f GB/s\n", name, k,
              in, out, rows, us, bytes / us / 1e3);
  (void)hipFree(tiles);
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

/// One routed MoE pass as a prompt pass runs it (gate, up with SwiGLU and
/// down's input transform, down), for per-expert row counts taken from a
/// real prompt: one line of `experts` counts per layer.
void CountsCase(const std::vector<std::vector<std::uint32_t>>& layers, int k) {
  constexpr std::uint32_t kHidden = 2560;
  constexpr std::uint32_t kFf = 640;
  const auto experts = static_cast<std::uint32_t>(layers[0].size());
  const auto make = [&](std::uint32_t in, std::uint32_t out) {
    q::Exl3Weights w{};
    w.in = in;
    w.out = out;
    w.k = k;
    w.experts = experts;
    w.front = q::Exl3DecodeSplits(in, out, true);
    auto* t = static_cast<std::uint32_t*>(Alloc(w.ExpertWords() * 4 * experts));
    {
      std::vector<std::uint32_t> h(w.ExpertWords() * experts);
      std::uint32_t r = 777U + static_cast<std::uint32_t>(in);
      for (auto& v : h) {
        r = r * 1664525U + 1013904223U;
        v = r;
      }
      (void)hipMemcpy(t, h.data(), h.size() * 4, hipMemcpyHostToDevice);
    }
    w.tiles = t;
    std::vector<__half> scale(static_cast<std::size_t>(experts) * (in + out));
    std::uint32_t r = 99U;
    for (auto& v : scale) {
      r = r * 1664525U + 1013904223U;
      v = __float2half((r >> 31) != 0U ? 0.02F : -0.02F);
    }
    auto* suh = static_cast<__half*>(Alloc(scale.size() * 2));
    (void)hipMemcpy(suh, scale.data(), scale.size() * 2, hipMemcpyHostToDevice);
    w.suh = suh;
    w.svh = suh + static_cast<std::size_t>(experts) * in;
    return w;
  };
  const q::Exl3Weights wg = make(kHidden, kFf);
  const q::Exl3Weights wu = make(kHidden, kFf);
  const q::Exl3Weights wd = make(kFf, kHidden);
  struct Route {
    std::int32_t *bounds, *tok, *slot, *t128, *t256;
    std::uint32_t n128, n256, slots, tokens;
  };
  std::vector<Route> routes;
  std::uint32_t max_slots = 0;
  std::uint32_t max_tokens = 0;
  const auto up = [](const std::vector<std::int32_t>& v) {
    auto* p = static_cast<std::int32_t*>(Alloc(v.size() * 4 + 4));
    (void)hipMemcpy(p, v.data(), v.size() * 4, hipMemcpyHostToDevice);
    return p;
  };
  for (const auto& counts : layers) {
    std::uint32_t slots = 0;
    for (auto c : counts)
      slots += c;
    const std::uint32_t tokens = slots / 10;
    std::vector<std::int32_t> bounds{0}, tok, slot, t128, t256;
    std::uint32_t s = 0;
    std::uint32_t lcg = 4242U;
    for (std::uint32_t e = 0; e < experts; ++e) {
      const std::uint32_t padded = (counts[e] + 15) / 16 * 16;
      for (std::uint32_t i = 0; i < padded; ++i) {
        lcg = lcg * 1664525U + 1013904223U;
        tok.push_back(i < counts[e] ? static_cast<std::int32_t>(lcg % tokens)
                                    : -1);
        slot.push_back(i < counts[e] ? static_cast<std::int32_t>(s++) : -1);
      }
      bounds.push_back(bounds.back() + static_cast<std::int32_t>(padded));
      for (std::uint32_t j = 0; j * q::kExl3GemmRows < counts[e]; ++j)
        t128.push_back(static_cast<std::int32_t>(e | (j << 16)));
      for (std::uint32_t j = 0; j * q::kExl3TransformedGemmRows < counts[e];
           ++j)
        t256.push_back(static_cast<std::int32_t>(e | (j << 16)));
    }
    routes.push_back({up(bounds), up(tok), up(slot), up(t128), up(t256),
                      static_cast<std::uint32_t>(t128.size()),
                      static_cast<std::uint32_t>(t256.size()), slots, tokens});
    max_slots = std::max(max_slots, slots);
    max_tokens = std::max(max_tokens, tokens);
  }
  auto* xh = static_cast<__half*>(Alloc(std::size_t{max_tokens} * kHidden * 2));
  {
    std::vector<__half> hh(std::size_t{max_tokens} * kHidden);
    std::uint32_t r = 12345U;
    for (auto& v : hh) {
      r = r * 1664525U + 1013904223U;
      v = __float2half(static_cast<float>(r >> 8) / 16777216.0F - 0.5F);
    }
    (void)hipMemcpy(xh, hh.data(), hh.size() * 2, hipMemcpyHostToDevice);
  }
  auto* gate = static_cast<float*>(Alloc(std::size_t{max_slots} * kFf * 4));
  auto* act = static_cast<__half*>(Alloc(std::size_t{max_slots} * kFf * 2));
  auto* down =
      static_cast<__half*>(Alloc(std::size_t{max_slots} * kHidden * 2));
  q::Exl3Scratch none{};
  hipEvent_t ev[4];
  for (auto& e : ev)
    (void)hipEventCreate(&e);
  // Best of five passes over the layers after a warm-up pass: clocks drift
  // by several percent between passes.
  float ms[3] = {1e30F, 1e30F, 1e30F};
  for (int rep = 0; rep < 6; ++rep) {
    float pass[3] = {0.0F, 0.0F, 0.0F};
    for (const Route& r : routes) {
      const q::Exl3GemmRoute from_tokens{r.t128, r.n128, r.bounds, r.tok,
                                         r.slot};
      const q::Exl3GemmRoute from_slots{r.t256, r.n256, r.bounds, r.slot,
                                        r.slot};
      q::Exl3GemmOut go{};
      go.x_half = xh;
      go.fused_input = true;
      q::Exl3GemmOut uo = go;
      uo.y_half = act;
      uo.gate = gate;
      uo.next_suh = wd.suh;
      q::Exl3GemmOut dout{};
      dout.x_half = act;
      dout.y_half = down;
      dout.x_transformed = true;
      (void)hipEventRecord(ev[0], nullptr);
      (void)q::Exl3LinearGemm(wg, nullptr, kHidden, 10, nullptr, r.slots, gate,
                              kFf, 0, from_tokens, none, nullptr, go);
      (void)hipEventRecord(ev[1], nullptr);
      (void)q::Exl3LinearGemm(wu, nullptr, kHidden, 10, nullptr, r.slots,
                              nullptr, kFf, 0, from_tokens, none, nullptr, uo);
      (void)hipEventRecord(ev[2], nullptr);
      (void)q::Exl3LinearGemm(wd, nullptr, kFf, 1, nullptr, r.slots, nullptr,
                              kHidden, 0, from_slots, none, nullptr, dout);
      (void)hipEventRecord(ev[3], nullptr);
      (void)hipEventSynchronize(ev[3]);
      if (rep == 0)
        continue;  // warm-up: sustained clocks
      for (int i = 0; i < 3; ++i) {
        float t = 0.0F;
        (void)hipEventElapsedTime(&t, ev[i], ev[i + 1]);
        pass[i] += t;
      }
    }
    if (rep > 0)
      for (int i = 0; i < 3; ++i)
        ms[i] = std::min(ms[i], pass[i]);
  }
  const auto n = static_cast<double>(routes.size());
  double flops = 0.0;
  for (const Route& r : routes)
    flops += 2.0 * r.slots * kHidden * kFf;
  flops /= static_cast<double>(routes.size());
  std::printf(
      "real routes K%d (%zu layers): gate %.3f ms (%.1f TF), up %.3f ms "
      "(%.1f TF), down %.3f ms (%.1f TF), sum %.3f ms per layer\n",
      k, routes.size(), ms[0] / n, flops / (ms[0] / n) / 1e9, ms[1] / n,
      flops / (ms[1] / n) / 1e9, ms[2] / n, flops / (ms[2] / n) / 1e9,
      (ms[0] + ms[1] + ms[2]) / n);
}

/// Prompt-pass Q8_0 expansion of dense matrices (Exl3DequantizeQ8), over
/// copies that exceed the memory-side cache, as a pass expands each layer.
void DequantCase(const char* name, std::uint32_t in, std::uint32_t out, int k) {
  q::Exl3Weights w{};
  w.in = in;
  w.out = out;
  w.k = k;
  w.experts = 1;
  w.front = q::Exl3DecodeSplits(in, out, false);
  const std::size_t words = w.ExpertWords();
  const int copies = static_cast<int>(
      std::max<std::size_t>(1, (std::size_t{256} << 20) / (words * 4 + 1) + 1));
  std::vector<std::uint32_t*> tiles;
  std::vector<std::uint32_t> h(words);
  std::uint32_t r = 4242U;
  for (auto& v : h) {
    r = r * 1664525U + 1013904223U;
    v = r;
  }
  for (int c = 0; c < copies; ++c) {
    tiles.push_back(static_cast<std::uint32_t*>(Alloc(words * 4)));
    (void)hipMemcpy(tiles.back(), h.data(), words * 4, hipMemcpyHostToDevice);
  }
  std::vector<__half> scale(in + out, __float2half(0.02F));
  auto* suh = static_cast<__half*>(Alloc(scale.size() * 2));
  (void)hipMemcpy(suh, scale.data(), scale.size() * 2, hipMemcpyHostToDevice);
  w.suh = suh;
  w.svh = suh + in;
  void* dst = Alloc(q::Exl3Q8Bytes(w));
  hipEvent_t a, b;
  (void)hipEventCreate(&a);
  (void)hipEventCreate(&b);
  float best = 1e30F;
  for (int rep = 0; rep < 6; ++rep) {
    (void)hipEventRecord(a, nullptr);
    for (int c = 0; c < copies; ++c) {
      w.tiles = tiles[c];
      (void)q::Exl3DequantizeQ8(w, 0, dst, nullptr);
    }
    (void)hipEventRecord(b, nullptr);
    (void)hipEventSynchronize(b);
    float ms = 0.0F;
    (void)hipEventElapsedTime(&ms, a, b);
    if (rep > 0)
      best = std::min(best, ms / static_cast<float>(copies));
  }
  const double bytes = words * 4.0 + static_cast<double>(q::Exl3Q8Bytes(w));
  std::printf("dequant %-10s K%d %ux%u: %.1f us, %.1f GB/s of traffic\n", name,
              k, in, out, best * 1e3, bytes / best / 1e6);
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
  if (argc > 1 && std::string(argv[1]) == "--dequant") {
    DequantCase("gdn qkv", 2560, 10240, 6);
    DequantCase("gdn out", 6144, 2560, 6);
    DequantCase("shexp up", 2560, 640, 6);
    DequantCase("gdn qkv", 2560, 10240, 4);
    return 0;
  }
  if (argc > 2 && std::string(argv[1]) == "--counts") {
    // A file of per-expert row counts, one prompt-pass layer per line;
    // optional bit widths after it (default 4).
    std::vector<std::vector<std::uint32_t>> layers;
    if (FILE* f = std::fopen(argv[2], "r")) {
      std::vector<std::uint32_t> line;
      char buf[1 << 16];
      while (std::fgets(buf, sizeof(buf), f) != nullptr) {
        line.clear();
        for (char* p = buf; *p != '\0';) {
          char* end = nullptr;
          const unsigned long v = std::strtoul(p, &end, 10);
          if (end == p)
            break;
          line.push_back(static_cast<std::uint32_t>(v));
          p = end;
        }
        std::uint64_t sum = 0;
        for (auto v : line)
          sum += v;
        if (line.size() == 512 && sum > 50000)
          layers.push_back(line);
      }
      std::fclose(f);
    }
    if (layers.empty()) {
      std::fprintf(stderr, "no prompt-pass layers in %s\n", argv[2]);
      return 1;
    }
    if (layers.size() > 12)
      layers.resize(12);
    if (argc == 3)
      CountsCase(layers, 4);
    for (int i = 3; i < argc; ++i)
      CountsCase(layers, std::atoi(argv[i]));
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "--routed") {
    // Rows per expert: the second argument (80 by default).
    const std::uint32_t rows =
        argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 80U;
    RoutedCase("experts gate", 2560, 640, 4, 512, rows, s);
    RoutedCase("experts gate", 2560, 640, 2, 512, rows, s);
    RoutedCase("experts down", 640, 2560, 4, 512, rows, s);
    return 0;
  }
  // A pack's dense and routed bit widths (4.05: 6 and 4; 3.05: 5 and 3).
  const int dk =
      std::getenv("EXL3_DENSE_K") ? std::atoi(std::getenv("EXL3_DENSE_K")) : 6;
  const int ek = std::getenv("EXL3_EXPERT_K")
                     ? std::atoi(std::getenv("EXL3_EXPERT_K"))
                     : 4;
  // Dense rows per launch: 1 for decode, 2..8 for MTP verification.
  const auto dr = static_cast<std::uint32_t>(
      std::getenv("EXL3_DENSE_ROWS") ? std::atoi(std::getenv("EXL3_DENSE_ROWS"))
                                     : 1);
  Case("gdn qkv", 2560, 10240, dk, 1, dr, s);
  Case("gdn z", 2560, 6144, dk, 1, dr, s);
  Case("gdn out", 6144, 2560, dk, 1, dr, s);
  Case("attn q", 2560, 12288, dk, 1, dr, s);
  Case("lm head", 2560, 248320, dk, 1, dr, s);
  Case("experts gate", 2560, 640, ek, 512, 10, s);
  Case("experts down", 640, 2560, ek, 512, 10, s);
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

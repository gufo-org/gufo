// Flash-Next prefill Gated DeltaNet timing over the production kernels (the
// wide route after the fused projection/convolution): rows per pass on the
// command line, default 2048 4096 8192 26541.
//
//   tools/bench/build.sh qfn_gdn_prefill_bench && /tmp/qfn_gdn_prefill_bench
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp"

namespace q = gufo::models::qwen38_flash_next::rocm;

// The wave64 W8A8 unit is not linked here and never called.
void q::W8A8GemmWave64(const void*, const void*, float*, std::size_t,
                       std::size_t, std::size_t, hipStream_t) {
  std::abort();
}

namespace {

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

std::uint32_t rng = 0x1234567U;
float Random(float scale, float offset) {
  rng ^= rng << 13;
  rng ^= rng >> 17;
  rng ^= rng << 5;
  return offset +
         scale * static_cast<float>(static_cast<int>(rng & 0xFFFFU) - 32768) /
             32768.0F;
}

float* Device(std::size_t n, float scale, float offset = 0.0F) {
  std::vector<float> host(n);
  for (float& v : host)
    v = Random(scale, offset);
  float* d = nullptr;
  Check(hipMalloc(&d, n * sizeof(float)), "hipMalloc");
  Check(hipMemcpy(d, host.data(), n * sizeof(float), hipMemcpyHostToDevice),
        "upload");
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  constexpr std::uint32_t kKHeads = 16, kVHeads = 48, kDim = 128;
  constexpr std::uint32_t kChannels = 10240, kZ = 6144, kProjection = 16384;
  std::vector<std::uint32_t> sizes = {2048, 4096, 8192, 26541};
  if (argc > 1) {
    sizes.clear();
    for (int i = 1; i < argc; ++i)
      sizes.push_back(static_cast<std::uint32_t>(std::atoi(argv[i])));
  }
  const std::uint32_t max_rows = *std::max_element(sizes.begin(), sizes.end());
  float* convolved = Device(std::size_t{max_rows + 4} * kChannels, 1.0F);
  float* qkvz = Device(std::size_t{max_rows} * kProjection, 1.0F);
  float* alpha_beta = Device(std::size_t{max_rows} * 2 * kVHeads, 2.0F);
  float* conv_w = Device(std::size_t{kChannels} * 4, 0.5F);
  float* a = Device(kVHeads, 1.0F, -1.5F);
  float* dt = Device(kVHeads, 1.0F);
  float* norm_w = Device(kDim, 0.5F, 1.0F);
  float* conv_state = Device(std::size_t{3} * kChannels, 1.0F);
  float* state = Device(std::size_t{kVHeads} * kDim * kDim, 0.1F);
  float* qn = Device(std::size_t{max_rows} * kKHeads * kDim, 0.0F);
  float* kn = Device(std::size_t{max_rows} * kKHeads * kDim, 0.0F);
  float* raw = Device(std::size_t{max_rows} * kVHeads * kDim, 0.0F);
  float* out = Device(std::size_t{max_rows} * (kZ + 32), 0.0F);
  // Output and state hashes from a fixed state, to compare builds.
  std::vector<float> initial(std::size_t{kVHeads} * kDim * kDim);
  Check(hipMemcpy(initial.data(), state, initial.size() * sizeof(float),
                  hipMemcpyDeviceToHost),
        "download");
  const auto hash = [](const void* data, std::size_t bytes) {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < bytes; ++i)
      h = (h ^ static_cast<const unsigned char*>(data)[i]) * 1099511628211ULL;
    return h;
  };
  for (const char* mode : {"F32", "F16", "Q8"}) {
    const bool half = mode[1] == '1';
    const bool q8 = mode[0] == 'Q';
    const std::uint32_t rows = std::min<std::uint32_t>(max_rows, 1031);
    Check(hipMemcpy(state, initial.data(), initial.size() * sizeof(float),
                    hipMemcpyHostToDevice),
          "upload");
    Check(hipMemset(out, 0, std::size_t{rows} * (kZ + 32) * sizeof(float)),
          "memset");
    q::GatedDeltaNet(qkvz, kProjection, qkvz + kChannels, kProjection,
                     alpha_beta, conv_w, a, dt, norm_w, conv_state, convolved,
                     qn, kn, raw, state, half || q8 ? nullptr : out,
                     q8 ? out : nullptr, {}, {}, rows, kKHeads, kVHeads, kDim,
                     4, true, true, 1e-6F, nullptr,
                     half ? reinterpret_cast<__half*>(out) : nullptr, {},
                     half ? kZ + 32 : 0, nullptr);
    std::vector<float> o(std::size_t{rows} * (kZ + 32));
    std::vector<float> st(initial.size());
    Check(hipMemcpy(o.data(), out, o.size() * sizeof(float),
                    hipMemcpyDeviceToHost),
          "download");
    Check(hipMemcpy(st.data(), state, st.size() * sizeof(float),
                    hipMemcpyDeviceToHost),
          "download");
    if (const char* dump = std::getenv("QFN_GDN_DUMP")) {
      const std::string base = std::string(dump) + "." + mode;
      std::FILE* f = std::fopen((base + ".out").c_str(), "wb");
      std::fwrite(o.data(), sizeof(float), o.size(), f);
      std::fclose(f);
      f = std::fopen((base + ".state").c_str(), "wb");
      std::fwrite(st.data(), sizeof(float), st.size(), f);
      std::fclose(f);
    }
    std::printf("%s rows %u: output %016llx state %016llx\n", mode, rows,
                static_cast<unsigned long long>(
                    hash(o.data(), o.size() * sizeof(float))),
                static_cast<unsigned long long>(
                    hash(st.data(), st.size() * sizeof(float))));
  }
  hipEvent_t start = nullptr, stop = nullptr;
  Check(hipEventCreate(&start), "event");
  Check(hipEventCreate(&stop), "event");
  for (const std::uint32_t rows : sizes) {
    const auto run = [&] {
      q::GatedDeltaNet(qkvz, kProjection, qkvz + kChannels, kProjection,
                       alpha_beta, conv_w, a, dt, norm_w, conv_state, convolved,
                       qn, kn, raw, state, out, nullptr, {}, {}, rows, kKHeads,
                       kVHeads, kDim, 4, true, true, 1e-6F, nullptr,
                       reinterpret_cast<__half*>(out), {},
                       rows >= 1024 ? kZ + 32 : 0, nullptr);
    };
    run();
    Check(hipDeviceSynchronize(), "warmup");
    float best = 1e30F;
    for (int r = 0; r < (rows > 10000 ? 3 : 8); ++r) {
      Check(hipEventRecord(start), "record");
      run();
      Check(hipEventRecord(stop), "record");
      Check(hipEventSynchronize(stop), "sync");
      float ms = 0.0F;
      Check(hipEventElapsedTime(&ms, start, stop), "elapsed");
      best = std::min(best, ms);
    }
    std::printf("rows %u: %.3f ms per layer, %.3f us per row per layer\n",
                rows, best, best * 1000.0F / static_cast<float>(rows));
  }
  return 0;
}

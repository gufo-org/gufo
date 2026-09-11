// Exact scalar/batched verification attention ablation.
// Build/run commands: docs/PERFORMANCE.md.
// Timings are component costs; use release model A/B for throughput decisions.
#define ENGINE_ENABLE_HIP 1
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "src/models/qwen/hip/kernels/attention_decode.hip"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

namespace {
using gufo::test::DeviceBuffer;
void RunAttention() {
  constexpr unsigned heads = 24, kvheads = 4, dim = 256, ctx = 16384, layer = 1,
                     width = heads * dim;
  constexpr size_t count = 2ul * ctx * kvheads * dim;
  std::vector<float> q(8 * width), g(q.size()), cache(count);
  std::vector<__half> half(count);
  for (size_t i = 0; i < q.size(); ++i) {
    q[i] = 0.3f * sinf(i * .073f);
    g[i] = cosf(i * .019f);
  }
  for (size_t i = 0; i < count; ++i) {
    cache[i] = .4f * sinf(i * .011f);
    half[i] = __float2half(cache[i]);
  }
  DeviceBuffer<float> dq(q), dg(g), dc(cache), out(q.size()), ref(q.size());
  DeviceBuffer<__half> dh(half);
  DeviceBuffer<float> scratch(
      8 * gufo::hip::detail::DecodeAttentionScratchElements(heads, dim));
  hipEvent_t begin, end;
  HIP_CHECK(hipEventCreate(&begin));
  HIP_CHECK(hipEventCreate(&end));
  for (bool fp16 : {false, true})
    for (bool gate : {false, true})
      for (unsigned rows : {1, 2, 7, 8})
        for (unsigned start : {0, 127, 2048, 4088, 4092, 4096, 8192}) {
          auto scalar = [&] {
            for (unsigned row = 0; row < rows; ++row)
              gufo::hip::LaunchAttention(
                  dq.data() + row * width, nullptr, nullptr,
                  gate ? dg.data() + row * width : nullptr,
                  fp16 ? nullptr : dc.data(), fp16 ? nullptr : dc.data(),
                  fp16 ? dh.data() : nullptr, fp16 ? dh.data() : nullptr,
                  ref.data() + row * width, layer, start + row, ctx, heads,
                  kvheads, dim, nullptr, scratch.data(), true);
          };
          auto batched = [&] {
            gufo::hip::LaunchCausalDecodeAttention(
                dq.data(), gate ? dg.data() : nullptr,
                fp16 ? nullptr : dc.data(), fp16 ? nullptr : dc.data(),
                fp16 ? dh.data() : nullptr, fp16 ? dh.data() : nullptr,
                out.data(), layer, start, rows, ctx, heads, kvheads, dim,
                nullptr, {scratch.data(), scratch.size()});
          };
          scalar();
          batched();
          HIP_CHECK(hipDeviceSynchronize());
          auto a = ref.CopyToHost(), b = out.CopyToHost();
          if (memcmp(a.data(), b.data(), rows * width * sizeof(float)))
            throw std::runtime_error("attention bits differ");
          if (gate && rows == 8 &&
              (start == 127 || start == 2048 || start == 4096 ||
               start == 8192)) {
            auto time = [&](auto launch) {
              for (int i = 0; i < 5; ++i)
                launch();
              HIP_CHECK(hipEventRecord(begin));
              for (int i = 0; i < 40; ++i)
                launch();
              HIP_CHECK(hipEventRecord(end));
              HIP_CHECK(hipEventSynchronize(end));
              float ms;
              HIP_CHECK(hipEventElapsedTime(&ms, begin, end));
              return ms / 40;
            };
            for (int rep = 0; rep < 3; ++rep) {
              float a = time(scalar), b = time(batched);
              printf(
                  "fp16=%d rows=%u start=%u rep=%d scalar_ms=%.6f "
                  "batched_ms=%.6f exact\n",
                  fp16, rows, start, rep, a, b);
            }
          }
        }
  puts("112 combinations byte-exact");
  HIP_CHECK(hipEventDestroy(begin));
  HIP_CHECK(hipEventDestroy(end));
}

}  // namespace

int main() {
  try {
    RunAttention();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}

// Exact scalar/batched verification attention ablation.
// Build/run commands: docs/PERFORMANCE.md.
// Timings are component costs; use release model A/B for throughput decisions.
#define ENGINE_ENABLE_HIP 1
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/models/qwen/hip/kernels/attention_decode.hip"
#include "src/models/qwen/hip/kernels/prefill_norm.hip"
#include "src/models/qwen/hip/kernels/prefill_rope.hip"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

namespace {
using gufo::test::DeviceBuffer;

void RunNormRoPE() {
  constexpr unsigned dim = 128, kv_heads = 8;
  hipEvent_t begin{}, end{};
  HIP_CHECK(hipEventCreate(&begin));
  HIP_CHECK(hipEventCreate(&end));
  for (unsigned heads : {0U, 32U})
    for (unsigned rows : {8U, 16U}) {
      std::vector<float> q(rows * 32 * dim), k(rows * kv_heads * dim), w(dim);
      for (size_t i = 0; i < q.size(); ++i)
        q[i] = 0.4F * sinf(i * .021F);
      for (size_t i = 0; i < k.size(); ++i)
        k[i] = 0.3F * cosf(i * .031F);
      for (size_t i = 0; i < w.size(); ++i)
        w[i] = 0.5F + i * .001F;
      DeviceBuffer<float> dq(q), dk(k), dw(w), qa(q), qb(q), ka(k), kb(k);
      const auto separate = [&] {
        if (heads != 0)
          gufo::hip::LaunchBatchedPerHeadRMSNorm(dq.data(), dw.data(),
                                                 qa.data(), rows, heads, dim);
        gufo::hip::LaunchBatchedPerHeadRMSNorm(dk.data(), dw.data(), ka.data(),
                                               rows, kv_heads, dim);
        gufo::hip::LaunchBatchedRoPE(qa.data(), ka.data(), rows, heads,
                                     kv_heads, dim, dim, 4096, 10000000.0F);
      };
      const auto fused = [&] {
        gufo::hip::LaunchBatchedFusedQKNormRoPEKvWrite(
            dq.data(), dk.data(), dk.data(), dw.data(), dw.data(), qb.data(),
            kb.data(), nullptr, nullptr, nullptr, nullptr, 0, 4096, rows, 0,
            heads, kv_heads, dim, dim, 10000000.0F);
      };
      separate();
      fused();
      const auto equal = [](const auto& expected, const auto& actual) {
        const auto a = expected.CopyToHost(), b = actual.CopyToHost();
        return memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
      };
      if (!equal(qa, qb) || !equal(ka, kb))
        throw std::runtime_error("QK norm/RoPE bits differ");
      const auto time = [&](auto launch) {
        for (int i = 0; i < 5; ++i)
          launch();
        HIP_CHECK(hipEventRecord(begin));
        for (int i = 0; i < 200; ++i)
          launch();
        HIP_CHECK(hipEventRecord(end));
        HIP_CHECK(hipEventSynchronize(end));
        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, begin, end));
        return ms / 200;
      };
      for (int rep = 0; rep < 3; ++rep) {
        const float a = time(separate), b = time(fused);
        printf(
            "norm_rope heads=%u rows=%u rep=%d separate_ms=%.6f "
            "fused_ms=%.6f exact\n",
            heads, rows, rep, a, b);
      }
    }
  HIP_CHECK(hipEventDestroy(begin));
  HIP_CHECK(hipEventDestroy(end));
}

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

int main(int argc, char** argv) {
  try {
    if (argc == 1 || std::string_view(argv[1]) == "attention")
      RunAttention();
    if (argc == 1 || std::string_view(argv[1]) == "rope")
      RunNormRoPE();
    if (argc > 1 && std::string_view(argv[1]) != "attention" &&
        std::string_view(argv[1]) != "rope")
      throw std::invalid_argument("use attention or rope");
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}

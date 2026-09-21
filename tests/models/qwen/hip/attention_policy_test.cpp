#include "src/models/qwen/hip/detail/attention_policy.hpp"

#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

enum class Attempt {
  kTiled,
  kBaseline,
};

int failures{0};

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void TestDispatchThresholdAndFallbackOrder() {
  using gufo::hip::detail::DispatchPrefillAttention;
  for (const std::size_t length : {1023U, 1024U}) {
    for (const bool supported : {false, true}) {
      std::vector<Attempt> attempts;
      DispatchPrefillAttention(
          length,
          [&] {
            attempts.push_back(Attempt::kTiled);
            return supported;
          },
          [&] { attempts.push_back(Attempt::kBaseline); });
      const auto expected =
          length < 1024 ? std::vector{Attempt::kBaseline}
          : supported   ? std::vector{Attempt::kTiled}
                        : std::vector{Attempt::kTiled, Attempt::kBaseline};
      Check(attempts == expected, "threshold and unsupported-shape fallback");
    }
  }
}

void TestBackendSupportPredicates() {
  using gufo::hip::detail::AttentionSupportParams;
  using gufo::hip::detail::IsTiledAttentionSupported;
  const AttentionSupportParams valid{.batch_size = 1024,
                                     .start_pos = 0,
                                     .max_context = 4096,
                                     .num_heads = 24,
                                     .num_kv_heads = 4,
                                     .head_dim = 256,
                                     .has_k_cache_f16 = true,
                                     .has_v_cache_f16 = true};
  Check(IsTiledAttentionSupported(valid),
        "supported native attention geometry");
  auto params = valid;
  params.start_pos = 3072;
  Check(IsTiledAttentionSupported(params),
        "cached prefix reaches exact capacity");
  params.start_pos = 3073;
  Check(!IsTiledAttentionSupported(params), "cached prefix exceeds capacity");
  params = valid;
  params.num_heads = 16;
  Check(!IsTiledAttentionSupported(params), "unsupported query-head count");
  params = valid;
  params.num_kv_heads = 8;
  Check(!IsTiledAttentionSupported(params), "unsupported KV-head count");
  params = valid;
  params.head_dim = 128;
  Check(!IsTiledAttentionSupported(params), "unsupported head dimension");
  params = valid;
  params.batch_size = 0;
  Check(!IsTiledAttentionSupported(params), "empty batch is rejected");
  params = valid;
  params.batch_size = 4096;
  Check(IsTiledAttentionSupported(params), "full capacity is accepted");
  params.batch_size = 4097;
  Check(!IsTiledAttentionSupported(params), "overflowing batch is rejected");
  params = valid;
  params.has_k_cache_f16 = false;
  Check(!IsTiledAttentionSupported(params), "missing K cache is rejected");
  params = valid;
  params.has_v_cache_f16 = false;
  Check(!IsTiledAttentionSupported(params), "missing V cache is rejected");
}

void TestDecodeSplitPolicy() {
  using gufo::hip::detail::DecodeAttentionScratchElements;
  using gufo::hip::detail::IsFusedQkNormSupported;
  using gufo::hip::detail::IsSplitKDecodeAttentionSupported;
  using gufo::hip::detail::SelectDecodeAttentionSplitCount;

  Check(SelectDecodeAttentionSplitCount(1) == 1,
        "the first decode token uses one online-softmax partition");
  Check(SelectDecodeAttentionSplitCount(127) == 1,
        "decode stays on one partition below 128 tokens");
  Check(SelectDecodeAttentionSplitCount(128) == 32,
        "128-token decode uses parallel attention partitions");
  Check(SelectDecodeAttentionSplitCount(2048) == 32,
        "2K decode uses parallel attention partitions");
  Check(SelectDecodeAttentionSplitCount(8192) == 32,
        "8K decode uses the measured split ceiling");
  Check(SelectDecodeAttentionSplitCount(16384) == 32,
        "16K decode uses the measured split ceiling");
  Check(SelectDecodeAttentionSplitCount(32768) == 32,
        "32K decode caps the partition count");

  Check(IsSplitKDecodeAttentionSupported(4096, 24, 4, 256),
        "Qwen3.8 long-context decode shape supports split-K");
  Check(IsSplitKDecodeAttentionSupported(2048, 24, 4, 256),
        "Qwen3.8 shallow-context decode shape supports split-K");
  Check(!IsSplitKDecodeAttentionSupported(127, 24, 4, 256),
        "very short decode does not use split-K");
  Check(!IsSplitKDecodeAttentionSupported(4096, 24, 0, 256),
        "split-K rejects zero KV heads");
  Check(!IsSplitKDecodeAttentionSupported(4096, 22, 4, 256),
        "split-K rejects non-divisible GQA heads");
  Check(!IsSplitKDecodeAttentionSupported(4096, 24, 4, 128),
        "split-K rejects unsupported head dimensions");
  Check(DecodeAttentionScratchElements(24, 256) ==
            static_cast<std::size_t>(24 * 32 * 258),
        "split-K scratch layout covers stats and partial values");
  Check(IsFusedQkNormSupported(256),
        "fused Q/K norm accepts its shared-memory capacity");
  Check(!IsFusedQkNormSupported(257),
        "fused Q/K norm rejects larger head dimensions");
}

}  // namespace

int main() {
  TestDispatchThresholdAndFallbackOrder();
  TestBackendSupportPredicates();
  TestDecodeSplitPolicy();
  if (failures != 0) {
    std::cerr << failures << " Qwen attention policy test(s) failed.\n";
    return 1;
  }
  std::cout << "Qwen attention policy tests passed.\n";
  return 0;
}

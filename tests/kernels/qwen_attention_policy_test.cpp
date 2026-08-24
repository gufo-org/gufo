#include "src/models/qwen/hip/detail/attention_policy.hpp"

#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

enum class Attempt {
  kTiled,
  kCk,
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
  using strix::hip::detail::DispatchPrefillAttention;

  std::vector<Attempt> attempts;
  DispatchPrefillAttention(
      1023,
      [&] {
        attempts.push_back(Attempt::kTiled);
        return true;
      },
      [&] {
        attempts.push_back(Attempt::kCk);
        return true;
      },
      [&] { attempts.push_back(Attempt::kBaseline); });
  Check(attempts == std::vector{Attempt::kBaseline},
        "visible context 1023 uses the baseline directly");

  attempts.clear();
  DispatchPrefillAttention(
      1024,
      [&] {
        attempts.push_back(Attempt::kTiled);
        return true;
      },
      [&] {
        attempts.push_back(Attempt::kCk);
        return true;
      },
      [&] { attempts.push_back(Attempt::kBaseline); });
  Check(attempts == std::vector{Attempt::kTiled},
        "visible context 1024 stops after tiled attention succeeds");

  attempts.clear();
  DispatchPrefillAttention(
      1024,
      [&] {
        attempts.push_back(Attempt::kTiled);
        return false;
      },
      [&] {
        attempts.push_back(Attempt::kCk);
        return true;
      },
      [&] { attempts.push_back(Attempt::kBaseline); });
  Check(attempts == std::vector{Attempt::kTiled, Attempt::kCk},
        "CK attention follows a rejected tiled attempt");

  attempts.clear();
  DispatchPrefillAttention(
      1024,
      [&] {
        attempts.push_back(Attempt::kTiled);
        return false;
      },
      [&] {
        attempts.push_back(Attempt::kCk);
        return false;
      },
      [&] { attempts.push_back(Attempt::kBaseline); });
  Check(attempts ==
            std::vector{Attempt::kTiled, Attempt::kCk, Attempt::kBaseline},
        "baseline follows rejected tiled and CK attempts");
}

void TestBackendSupportPredicates() {
  using strix::hip::detail::AttentionSupportParams;
  using strix::hip::detail::IsCkAttentionSupported;
  using strix::hip::detail::IsTiledAttentionSupported;

  AttentionSupportParams params{
      .batch_size = 1024,
      .start_pos = 0,
      .max_context = 4096,
      .num_heads = 24,
      .num_kv_heads = 4,
      .head_dim = 256,
      .has_k_cache_f16 = true,
      .has_v_cache_f16 = true,
      .has_scratch_f16 = true,
  };
  Check(IsTiledAttentionSupported(params), "supported tiled shape is accepted");
  Check(IsCkAttentionSupported(params), "supported CK shape is accepted");

  params.start_pos = 1;
  Check(IsTiledAttentionSupported(params),
        "tiled attention accepts a nonzero start position within capacity");
  Check(!IsCkAttentionSupported(params),
        "CK attention rejects a nonzero start position");

  params.start_pos = 0;
  params.num_heads = 16;
  Check(!IsTiledAttentionSupported(params),
        "tiled attention rejects a different query-head count");
  Check(IsCkAttentionSupported(params),
        "CK attention accepts another divisible query-head count");

  params.num_heads = 18;
  Check(!IsCkAttentionSupported(params),
        "CK attention rejects a non-divisible query-head count");

  params.num_heads = 24;
  params.num_kv_heads = 8;
  Check(!IsTiledAttentionSupported(params),
        "tiled attention rejects a different KV-head count");
  Check(!IsCkAttentionSupported(params),
        "CK attention rejects a different KV-head count");

  params.num_kv_heads = 4;
  params.head_dim = 128;
  Check(!IsTiledAttentionSupported(params),
        "tiled attention rejects a different head dimension");
  Check(!IsCkAttentionSupported(params),
        "CK attention rejects a different head dimension");

  params.head_dim = 256;
  params.start_pos = 0;
  params.batch_size = params.max_context;
  Check(IsTiledAttentionSupported(params),
        "tiled attention accepts a batch equal to context capacity");
  Check(IsCkAttentionSupported(params),
        "CK attention accepts a batch equal to context capacity");

  params.start_pos = 1;
  Check(!IsTiledAttentionSupported(params),
        "tiled attention rejects a visible context beyond capacity");
  params.start_pos = 0;
  params.batch_size = 4097;
  Check(!IsTiledAttentionSupported(params),
        "tiled attention rejects a batch beyond context capacity");
  Check(!IsCkAttentionSupported(params),
        "CK attention rejects a batch beyond context capacity");

  params.batch_size = 1024;
  params.has_k_cache_f16 = false;
  Check(!IsTiledAttentionSupported(params),
        "tiled attention requires a K cache");
  Check(!IsCkAttentionSupported(params), "CK attention requires a K cache");

  params.has_k_cache_f16 = true;
  params.has_v_cache_f16 = false;
  Check(!IsTiledAttentionSupported(params),
        "tiled attention requires a V cache");
  Check(!IsCkAttentionSupported(params), "CK attention requires a V cache");

  params.has_v_cache_f16 = true;
  params.has_scratch_f16 = false;
  Check(IsTiledAttentionSupported(params),
        "tiled attention does not require CK scratch storage");
  Check(!IsCkAttentionSupported(params),
        "CK attention requires FP16 scratch storage");

  params.batch_size = 0;
  Check(!IsTiledAttentionSupported(params),
        "tiled attention rejects an empty batch");
  Check(!IsCkAttentionSupported(params), "CK attention rejects an empty batch");
}

void TestDecodeSplitPolicy() {
  using strix::hip::detail::DecodeAttentionScratchElements;
  using strix::hip::detail::IsSplitKDecodeAttentionSupported;
  using strix::hip::detail::SelectDecodeAttentionSplitCount;

  Check(SelectDecodeAttentionSplitCount(1024) == 1,
        "1K decode uses one online-softmax partition");
  Check(SelectDecodeAttentionSplitCount(4095) == 1,
        "decode stays on one partition below 4K");
  Check(SelectDecodeAttentionSplitCount(4096) == 32,
        "4K decode fills the 32-wave-per-CU split ceiling");
  Check(SelectDecodeAttentionSplitCount(8192) == 32,
        "8K decode uses the measured split ceiling");
  Check(SelectDecodeAttentionSplitCount(16384) == 32,
        "16K decode uses the measured split ceiling");
  Check(SelectDecodeAttentionSplitCount(32768) == 32,
        "32K decode caps the partition count");

  Check(IsSplitKDecodeAttentionSupported(4096, 24, 4, 256),
        "Qwen3.8 long-context decode shape supports split-K");
  Check(!IsSplitKDecodeAttentionSupported(2048, 24, 4, 256),
        "short-context decode does not use split-K");
  Check(!IsSplitKDecodeAttentionSupported(4096, 24, 0, 256),
        "split-K rejects zero KV heads");
  Check(!IsSplitKDecodeAttentionSupported(4096, 22, 4, 256),
        "split-K rejects non-divisible GQA heads");
  Check(!IsSplitKDecodeAttentionSupported(4096, 24, 4, 128),
        "split-K rejects unsupported head dimensions");
  Check(DecodeAttentionScratchElements(24, 256) ==
            static_cast<std::size_t>(24 * 32 * 258),
        "split-K scratch layout covers stats and partial values");
}

static_assert(!strix::hip::detail::ShouldAttemptOptimizedAttention(1023));
static_assert(strix::hip::detail::ShouldAttemptOptimizedAttention(1024));
static_assert(strix::hip::detail::SelectDecodeAttentionSplitCount(32768) == 32);

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

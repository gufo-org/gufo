#include "src/models/qwen/continuation_layout.hpp"

#include <iostream>
#include <stdexcept>

namespace qwen = gufo::models::qwen;
namespace hip = gufo::hip;
namespace cache = gufo::cache;
namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
template<class F>
void Reject(F&& fn) {
  bool rejected = false;
  try {
    fn();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "invalid geometry/frontier was accepted");
}
void Check(bool fp16, bool bf16) {
  gufo::core::ModelConfig c;
  c.num_layers = 64;
  c.hidden_size = 5120;
  c.num_attention_heads = 24;
  c.intermediate_size = 17408;
  c.context_length = 262144;
  c.ssm_time_step_rank = 48;
  c.ssm_inner_size = 6144;
  hip::QwenExecutionPolicy policy;
  policy.kv_cache_storage =
      fp16 ? hip::QwenKvCacheStorage::kFp16 : hip::QwenKvCacheStorage::kFp32;
  policy.recurrent_state_storage = bf16 ? hip::QwenRecurrentStateStorage::kBf16
                                        : hip::QwenRecurrentStateStorage::kFp32;
  qwen::ContinuationLayout layout(c, policy, 100000);
  Require(layout.Components().size() == 130, "incomplete state inventory");
  // 48 independent recurrent layers: 10240 * 4 FP32 convolution elements,
  // plus 48 heads * 128 * 128 state elements. Logits and host metadata are
  // fixed.
  const std::size_t fixed =
      48 * (163840 + 786432 * (bf16 ? 2 : 4)) + 993280 + 64;
  Require(layout.PrivateBytes() == fixed, "incorrect private state claim");
  Require(layout.BytesPerToken() == (fp16 ? 65536 : 131072),
          "incorrect KV stride");
  for (auto position : {0U, 32768U, 100000U}) {
    const auto frontiers = layout.Positions(position);
    for (std::size_t i = 0; i < frontiers.size(); ++i) {
      const auto& part = layout.Components()[i];
      Require(part.descriptor.id.value == i + 1, "unstable component IDs");
      Require(frontiers[i].valid_rows == position,
              "component boundary differs");
      if (part.part == qwen::ContinuationPart::kKey ||
          part.part == qwen::ContinuationPart::kValue) {
        Require(part.descriptor.layout_version == (fp16 ? 1U : 2U),
                "KV layouts alias");
        Require(part.descriptor.rows_per_chunk == 256,
                "unbounded chunk geometry");
      }
    }
  }
  Reject([&] { (void)layout.Positions(100001); });
  Reject([&] { qwen::ContinuationLayout bad(c, policy, 0); });
  c.full_attention_interval = 0;
  Reject([&] { qwen::ContinuationLayout bad(c, policy, 100000); });
}
}  // namespace
int main() {
  try {
    for (bool fp16 : {false, true})
      for (bool bf16 : {false, true})
        Check(fp16, bf16);
    std::cout << "Qwen continuation geometry: PASS\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

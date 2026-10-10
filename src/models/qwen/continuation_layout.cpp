#include "src/models/qwen/continuation_layout.hpp"

#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace gufo::models::qwen {
namespace {
std::size_t Product(std::initializer_list<std::size_t> factors) {
  std::size_t result = 1;
  for (auto factor : factors) {
    if (factor && result > std::numeric_limits<std::size_t>::max() / factor)
      throw std::overflow_error("Qwen continuation geometry overflows");
    result *= factor;
  }
  return result;
}
}  // namespace

ContinuationLayout::ContinuationLayout(const core::ModelConfig& c,
                                       hip::QwenExecutionPolicy policy,
                                       std::uint32_t context)
    : context_(context) {
  if (!c.IsValidQwen() || !context || context > c.context_length ||
      (policy.kv_cache_storage != hip::QwenKvCacheStorage::kFp16 &&
       policy.kv_cache_storage != hip::QwenKvCacheStorage::kFp32) ||
      (policy.recurrent_state_storage !=
           hip::QwenRecurrentStateStorage::kFp32 &&
       policy.recurrent_state_storage != hip::QwenRecurrentStateStorage::kBf16))
    throw std::invalid_argument("invalid Qwen continuation geometry or policy");
  std::size_t fixed_total = 0, row_total = 0;
  const auto add = [&](ContinuationPart part, std::size_t row,
                       std::size_t state, std::uint32_t layer = 0,
                       std::uint32_t version = 1) {
    if (state > std::numeric_limits<std::size_t>::max() - fixed_total ||
        row > std::numeric_limits<std::size_t>::max() - row_total)
      throw std::overflow_error("Qwen continuation inventory overflows");
    fixed_total += state;
    row_total += row;
    components_.push_back(
        {{{static_cast<std::uint32_t>(components_.size() + 1)},
          version,
          row ? cache::ComponentKind::kAppendRows
              : cache::ComponentKind::kPrivateState,
          row,
          row ? cache::Rows{256} : 0,
          state},
         part,
         layer});
  };
  add(ContinuationPart::kMetadata, 0, kContinuationMetadataBytes);
  add(ContinuationPart::kLogits, 0, Product({c.vocab_size, sizeof(float)}));
  const auto kv_row = Product({c.num_key_value_heads, c.head_dim,
                               policy.UsesFp16AttentionKv() ? 2U : 4U});
  const auto key_width = Product({2, c.ssm_group_count, c.ssm_state_size});
  if (key_width > std::numeric_limits<std::uint32_t>::max() - c.ssm_inner_size)
    throw std::invalid_argument(
        "Qwen convolution width exceeds native geometry");
  const auto qkv = key_width + c.ssm_inner_size;
  const auto conv = Product({qkv, c.ssm_conv_kernel, sizeof(float)});
  const auto recurrent = Product(
      {c.ssm_time_step_rank, c.ssm_state_size, c.SsmValueSize(),
       hip::QwenRecurrentStateElementBytes(policy.recurrent_state_storage)});
  for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
    if ((layer + 1) % c.full_attention_interval == 0) {
      add(ContinuationPart::kKey, kv_row, 0, layer,
          policy.UsesFp16AttentionKv() ? 1 : 2);
      add(ContinuationPart::kValue, kv_row, 0, layer,
          policy.UsesFp16AttentionKv() ? 1 : 2);
    } else {
      add(ContinuationPart::kConvolution, 0, conv, layer);
      add(ContinuationPart::kRecurrent, 0, recurrent, layer,
          policy.UsesBf16RecurrentState() ? 2 : 1);
    }
  }
  // Check the complete retained representation, not just individual tensors.
  const auto private_bytes = PrivateBytes();
  const auto row_bytes = Product({context, BytesPerToken()});
  if (row_bytes > std::numeric_limits<std::size_t>::max() - private_bytes)
    throw std::overflow_error("Qwen continuation payload overflows");
}

std::vector<cache::ComponentPosition> ContinuationLayout::Positions(
    cache::Rows position) const {
  if (position > context_)
    throw std::invalid_argument("Qwen continuation frontier exceeds context");
  std::vector<cache::ComponentPosition> result;
  result.reserve(components_.size());
  for (const auto& component : components_)
    result.push_back({component.descriptor.id, position});
  return result;
}
std::size_t ContinuationLayout::PrivateBytes() const noexcept {
  std::size_t bytes = 0;
  for (const auto& component : components_)
    bytes += component.descriptor.state_bytes;
  return bytes;
}
std::size_t ContinuationLayout::BytesPerToken() const noexcept {
  std::size_t bytes = 0;
  for (const auto& component : components_)
    bytes += component.descriptor.row_bytes;
  return bytes;
}
}  // namespace gufo::models::qwen

#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace gufo::hip {
namespace {

constexpr std::uint32_t kMaxPromptBatch = 4096;
constexpr std::size_t kMaxTargetLayerTaps = 5;
constexpr std::array<std::uint8_t, 8> kCompactSnapshotMagic = {
    'G', 'Q', 'K', 'V', 'S', 'N', 'P', '1'};
constexpr std::uint32_t kCompactSnapshotVersion = 1;
constexpr std::size_t kCompactSnapshotHeaderBytes = 80;

std::size_t CheckedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("Qwen GPU allocation size overflows");
  }
  return left * right;
}

std::size_t CheckedSum(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("Qwen GPU allocation total overflows");
  }
  return left + right;
}

void CheckedAdd(std::size_t value, std::size_t* total) {
  *total = CheckedSum(*total, value);
}

void AddAllocation(std::size_t elements, std::size_t element_bytes,
                   std::size_t* total) {
  CheckedAdd(CheckedMultiply(elements, element_bytes), total);
}

void ThrowOnHipError(hipError_t error, std::string_view operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error("Qwen compact snapshot header is truncated");
  }
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename T>
  requires(std::is_unsigned_v<T>)
T GetLittleEndian(std::span<const std::uint8_t> source, std::size_t offset) {
  if (offset > source.size() || sizeof(T) > source.size() - offset) {
    throw std::invalid_argument("Qwen compact snapshot header is truncated");
  }
  T value = 0;
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    value |= static_cast<T>(source[offset + byte]) << (byte * 8U);
  }
  return value;
}

std::size_t SizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("Qwen compact snapshot size overflows");
  }
  return static_cast<std::size_t>(value);
}

struct CompactSnapshotLayout {
  std::uint32_t attention_layers{0};
  std::uint32_t kv_width{0};
  std::uint32_t max_context{0};
  std::uint32_t valid_context{0};
  QwenKvCacheStorage kv_storage{QwenKvCacheStorage::kFp32};
  QwenRecurrentStateStorage recurrent_state_storage{
      QwenRecurrentStateStorage::kFp32};
  std::size_t live_kv_elements_per_plane{0};
  std::size_t live_kv_bytes_per_plane{0};
  std::size_t conv_elements{0};
  std::size_t conv_bytes{0};
  std::size_t deltanet_elements{0};
  std::size_t deltanet_bytes{0};
  std::size_t k_offset{kCompactSnapshotHeaderBytes};
  std::size_t v_offset{0};
  std::size_t conv_offset{0};
  std::size_t deltanet_offset{0};
  std::size_t total_bytes{0};
};

CompactSnapshotLayout MakeCompactSnapshotLayout(
    std::uint32_t attention_layers, std::uint32_t kv_width,
    std::uint32_t max_context, std::uint32_t valid_context,
    QwenKvCacheStorage kv_storage,
    QwenRecurrentStateStorage recurrent_state_storage,
    std::size_t conv_elements, std::size_t deltanet_elements) {
  CompactSnapshotLayout layout{
      .attention_layers = attention_layers,
      .kv_width = kv_width,
      .max_context = max_context,
      .valid_context = valid_context,
      .kv_storage = kv_storage,
      .recurrent_state_storage = recurrent_state_storage,
      .conv_elements = conv_elements,
      .deltanet_elements = deltanet_elements,
  };
  const std::size_t kv_element_bytes = kv_storage == QwenKvCacheStorage::kFp16
                                           ? sizeof(std::uint16_t)
                                           : sizeof(float);
  layout.live_kv_elements_per_plane = CheckedMultiply(
      CheckedMultiply(attention_layers, valid_context), kv_width);
  layout.live_kv_bytes_per_plane =
      CheckedMultiply(layout.live_kv_elements_per_plane, kv_element_bytes);
  layout.conv_bytes = CheckedMultiply(conv_elements, sizeof(float));
  layout.deltanet_bytes =
      CheckedMultiply(deltanet_elements,
                      QwenRecurrentStateElementBytes(recurrent_state_storage));
  layout.v_offset = CheckedSum(layout.k_offset, layout.live_kv_bytes_per_plane);
  layout.conv_offset =
      CheckedSum(layout.v_offset, layout.live_kv_bytes_per_plane);
  layout.deltanet_offset = CheckedSum(layout.conv_offset, layout.conv_bytes);
  layout.total_bytes =
      CheckedSum(layout.deltanet_offset, layout.deltanet_bytes);
  return layout;
}

CompactSnapshotLayout ParseCompactSnapshotLayout(
    std::span<const std::uint8_t> payload) {
  if (payload.size() < kCompactSnapshotHeaderBytes ||
      !std::equal(kCompactSnapshotMagic.begin(), kCompactSnapshotMagic.end(),
                  payload.begin())) {
    throw std::invalid_argument("Qwen compact snapshot header is invalid");
  }
  if (GetLittleEndian<std::uint32_t>(payload, 8) != kCompactSnapshotVersion ||
      GetLittleEndian<std::uint32_t>(payload, 12) !=
          kCompactSnapshotHeaderBytes ||
      GetLittleEndian<std::uint64_t>(payload, 72) != 0) {
    throw std::invalid_argument("Qwen compact snapshot version is invalid");
  }

  const std::uint32_t kv_storage_value =
      GetLittleEndian<std::uint32_t>(payload, 32);
  const std::uint32_t recurrent_storage_value =
      GetLittleEndian<std::uint32_t>(payload, 36);
  if (kv_storage_value >
          static_cast<std::uint32_t>(QwenKvCacheStorage::kFp16) ||
      recurrent_storage_value >
          static_cast<std::uint32_t>(QwenRecurrentStateStorage::kBf16)) {
    throw std::invalid_argument("Qwen compact snapshot precision is invalid");
  }

  auto layout = MakeCompactSnapshotLayout(
      GetLittleEndian<std::uint32_t>(payload, 16),
      GetLittleEndian<std::uint32_t>(payload, 20),
      GetLittleEndian<std::uint32_t>(payload, 24),
      GetLittleEndian<std::uint32_t>(payload, 28),
      static_cast<QwenKvCacheStorage>(kv_storage_value),
      static_cast<QwenRecurrentStateStorage>(recurrent_storage_value),
      SizeFromU64(GetLittleEndian<std::uint64_t>(payload, 48)),
      SizeFromU64(GetLittleEndian<std::uint64_t>(payload, 56)));
  if (layout.live_kv_elements_per_plane !=
          SizeFromU64(GetLittleEndian<std::uint64_t>(payload, 40)) ||
      layout.total_bytes !=
          SizeFromU64(GetLittleEndian<std::uint64_t>(payload, 64)) ||
      layout.total_bytes != payload.size()) {
    throw std::invalid_argument("Qwen compact snapshot size is invalid");
  }
  return layout;
}

}  // namespace

QwenGpuSnapshot::~QwenGpuSnapshot() {
  if (d_kv_f32_ != nullptr) {
    (void)hipFree(d_kv_f32_);
  }
  if (d_kv_f16_ != nullptr) {
    (void)hipFree(d_kv_f16_);
  }
  if (d_ssm_conv_ != nullptr) {
    (void)hipFree(d_ssm_conv_);
  }
  if (d_ssm_deltanet_ != nullptr) {
    (void)hipFree(d_ssm_deltanet_);
  }
}

std::size_t QwenGpuSnapshot::CompactPayloadBytes() const {
  const std::size_t recurrent_layers = num_layers_ - attention_layers_;
  return MakeCompactSnapshotLayout(
             attention_layers_, kv_width_, max_context_, valid_context_,
             kv_storage_, recurrent_state_storage_,
             CheckedMultiply(recurrent_layers, conv_elements_per_layer_),
             CheckedMultiply(recurrent_layers, deltanet_elements_per_layer_))
      .total_bytes;
}

std::size_t QwenGpuSnapshot::SerializeCompact(
    std::span<std::uint8_t> destination) const {
  if (full_attention_interval_ == 0 || attention_layers_ > num_layers_ ||
      conv_elements_ !=
          CheckedMultiply(num_layers_, conv_elements_per_layer_) ||
      deltanet_elements_ !=
          CheckedMultiply(num_layers_, deltanet_elements_per_layer_)) {
    throw std::logic_error("Qwen compact snapshot metadata is invalid");
  }
  const std::size_t recurrent_layers = num_layers_ - attention_layers_;
  const auto layout = MakeCompactSnapshotLayout(
      attention_layers_, kv_width_, max_context_, valid_context_, kv_storage_,
      recurrent_state_storage_,
      CheckedMultiply(recurrent_layers, conv_elements_per_layer_),
      CheckedMultiply(recurrent_layers, deltanet_elements_per_layer_));
  if (destination.size() != layout.total_bytes) {
    throw std::invalid_argument(
        "Qwen compact snapshot destination size is invalid");
  }
  std::fill(destination.begin(), destination.end(), std::uint8_t{0});
  std::copy(kCompactSnapshotMagic.begin(), kCompactSnapshotMagic.end(),
            destination.begin());
  PutLittleEndian<std::uint32_t>(destination, 8, kCompactSnapshotVersion);
  PutLittleEndian<std::uint32_t>(
      destination, 12, static_cast<std::uint32_t>(kCompactSnapshotHeaderBytes));
  PutLittleEndian<std::uint32_t>(destination, 16, attention_layers_);
  PutLittleEndian<std::uint32_t>(destination, 20, kv_width_);
  PutLittleEndian<std::uint32_t>(destination, 24, max_context_);
  PutLittleEndian<std::uint32_t>(destination, 28, valid_context_);
  PutLittleEndian<std::uint32_t>(destination, 32,
                                 static_cast<std::uint32_t>(kv_storage_));
  PutLittleEndian<std::uint32_t>(
      destination, 36, static_cast<std::uint32_t>(recurrent_state_storage_));
  PutLittleEndian<std::uint64_t>(
      destination, 40,
      static_cast<std::uint64_t>(layout.live_kv_elements_per_plane));
  PutLittleEndian<std::uint64_t>(
      destination, 48, static_cast<std::uint64_t>(layout.conv_elements));
  PutLittleEndian<std::uint64_t>(
      destination, 56, static_cast<std::uint64_t>(layout.deltanet_elements));
  PutLittleEndian<std::uint64_t>(
      destination, 64, static_cast<std::uint64_t>(layout.total_bytes));

  if (layout.live_kv_bytes_per_plane != 0) {
    const std::size_t kv_element_bytes =
        kv_storage_ == QwenKvCacheStorage::kFp16 ? sizeof(std::uint16_t)
                                                 : sizeof(float);
    const std::size_t source_pitch = CheckedMultiply(
        CheckedMultiply(max_context_, kv_width_), kv_element_bytes);
    const void* source =
        kv_storage_ == QwenKvCacheStorage::kFp16 ? d_kv_f16_ : d_kv_f32_;
    if (source == nullptr) {
      throw std::logic_error("Qwen compact snapshot has no KV storage");
    }
    ThrowOnHipError(
        hipMemcpy2D(destination.data() + layout.k_offset,
                    layout.live_kv_bytes_per_plane / attention_layers_, source,
                    source_pitch,
                    layout.live_kv_bytes_per_plane / attention_layers_,
                    attention_layers_, hipMemcpyDeviceToHost),
        "failed to serialize compact Qwen K cache");
    const auto* source_bytes = static_cast<const std::uint8_t*>(source);
    const std::size_t full_plane_bytes =
        CheckedMultiply(kv_elements_per_plane_, kv_element_bytes);
    ThrowOnHipError(
        hipMemcpy2D(destination.data() + layout.v_offset,
                    layout.live_kv_bytes_per_plane / attention_layers_,
                    source_bytes + full_plane_bytes, source_pitch,
                    layout.live_kv_bytes_per_plane / attention_layers_,
                    attention_layers_, hipMemcpyDeviceToHost),
        "failed to serialize compact Qwen V cache");
  }
  if (layout.conv_bytes != 0) {
    const std::size_t row_bytes =
        CheckedMultiply(conv_elements_per_layer_, sizeof(float));
    std::size_t destination_offset = layout.conv_offset;
    const auto* source = static_cast<const std::uint8_t*>(d_ssm_conv_);
    for (std::uint32_t group_start = 0; group_start < num_layers_;
         group_start += full_attention_interval_) {
      const std::size_t remaining = num_layers_ - group_start;
      const std::size_t rows =
          remaining < full_attention_interval_
              ? remaining
              : static_cast<std::size_t>(full_attention_interval_ - 1U);
      const std::size_t bytes = CheckedMultiply(rows, row_bytes);
      ThrowOnHipError(
          hipMemcpy(destination.data() + destination_offset,
                    source + CheckedMultiply(group_start, row_bytes), bytes,
                    hipMemcpyDeviceToHost),
          "failed to serialize compact Qwen convolution state");
      destination_offset = CheckedSum(destination_offset, bytes);
    }
    if (destination_offset != layout.deltanet_offset) {
      throw std::logic_error(
          "Qwen compact convolution state has an invalid row count");
    }
  }
  if (layout.deltanet_bytes != 0) {
    const std::size_t row_bytes = CheckedMultiply(
        deltanet_elements_per_layer_,
        QwenRecurrentStateElementBytes(recurrent_state_storage_));
    std::size_t destination_offset = layout.deltanet_offset;
    const auto* source = static_cast<const std::uint8_t*>(d_ssm_deltanet_);
    for (std::uint32_t group_start = 0; group_start < num_layers_;
         group_start += full_attention_interval_) {
      const std::size_t remaining = num_layers_ - group_start;
      const std::size_t rows =
          remaining < full_attention_interval_
              ? remaining
              : static_cast<std::size_t>(full_attention_interval_ - 1U);
      const std::size_t bytes = CheckedMultiply(rows, row_bytes);
      ThrowOnHipError(
          hipMemcpy(destination.data() + destination_offset,
                    source + CheckedMultiply(group_start, row_bytes), bytes,
                    hipMemcpyDeviceToHost),
          "failed to serialize compact Qwen DeltaNet state");
      destination_offset = CheckedSum(destination_offset, bytes);
    }
    if (destination_offset != layout.total_bytes) {
      throw std::logic_error(
          "Qwen compact DeltaNet state has an invalid row count");
    }
  }
  return destination.size();
}

QwenGpuMemoryUsage QwenGpuArena::EstimateMemoryUsage(
    const core::ModelConfig& config, std::uint32_t max_context,
    QwenExecutionPolicy policy) {
  const std::size_t context = std::max(max_context, 1U);
  const std::size_t batch = std::min<std::size_t>(context, kMaxPromptBatch);
  const std::size_t hidden = config.hidden_size;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      CheckedMultiply(config.num_key_value_heads, config.head_dim);
  const std::size_t q_projection = CheckedMultiply(2, attention);
  const std::size_t ssm_qkv = config.SsmQkvSize();
  const std::size_t recurrent =
      std::max<std::size_t>(attention, config.ssm_inner_size);
  const std::size_t projection = std::max(q_projection, ssm_qkv);
  const std::size_t time_step = config.ssm_time_step_rank;

  QwenGpuMemoryUsage usage;
  auto& scratch = usage.temporary_scratch_bytes;
  AddAllocation(CheckedMultiply(batch, hidden), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, hidden), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, attention), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, kv), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, kv), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, hidden), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, intermediate), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, intermediate), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, intermediate), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, hidden), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, projection), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, ssm_qkv), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, recurrent), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, recurrent), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, time_step), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, time_step), sizeof(float), &scratch);
  AddAllocation(
      CheckedMultiply(CheckedMultiply(batch, config.ssm_group_count), 3),
      sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(CheckedMultiply(batch, time_step), 2),
                sizeof(float), &scratch);
  AddAllocation(config.vocab_size, sizeof(float), &scratch);
  AddAllocation(std::max<std::size_t>(batch, 2), sizeof(std::uint32_t),
                &scratch);
  AddAllocation(CheckedMultiply(kMaxTargetLayerTaps, hidden), sizeof(float),
                &scratch);

  const std::size_t maximum_scratch_width = std::max<std::size_t>(
      {intermediate, hidden, projection,
       CheckedSum(ssm_qkv, CheckedSum(config.ssm_inner_size,
                                      CheckedMultiply(2, time_step)))});
  const std::size_t scratch_elements =
      CheckedMultiply(batch, maximum_scratch_width);
  AddAllocation(scratch_elements, sizeof(hip_bfloat16), &scratch);

  const std::size_t q8_rows = CheckedMultiply((batch + 15) / 16, 16);
  const std::size_t q8_row_elements = scratch_elements / batch;
  const std::size_t q8_elements = CheckedMultiply(q8_rows, q8_row_elements);
  const std::size_t q8_blocks = (q8_elements + 31) / 32;
  CheckedAdd(CheckedMultiply(CheckedMultiply(q8_blocks, sizeof(float)), 9),
             &scratch);
  CheckedAdd(4096, &scratch);

  AddAllocation(detail::DecodeAttentionScratchElements(
                    config.num_attention_heads, config.head_dim),
                sizeof(float), &scratch);

  const std::size_t maximum_weight_width =
      std::max<std::size_t>({q_projection, kv, attention, intermediate, ssm_qkv,
                             config.ssm_inner_size, time_step});
  const std::size_t maximum_weight =
      CheckedMultiply(hidden, maximum_weight_width);
  AddAllocation(maximum_weight, sizeof(hip_bfloat16), &scratch);
  AddAllocation(maximum_weight, sizeof(hip_bfloat16), &scratch);

  auto& state = usage.request_state_bytes;
  const std::size_t total_kv = CheckedMultiply(
      CheckedMultiply(CheckedMultiply(config.FullAttentionLayerCount(), kv),
                      context),
      1);
  AddAllocation(
      CheckedMultiply(total_kv, 2),
      policy.UsesFp16AttentionKv() ? sizeof(std::uint16_t) : sizeof(float),
      &state);

  const std::size_t total_conv = CheckedMultiply(
      CheckedMultiply(config.num_layers, ssm_qkv), config.ssm_conv_kernel);
  AddAllocation(total_conv, sizeof(float), &state);
  const std::size_t total_deltanet = CheckedMultiply(
      CheckedMultiply(CheckedMultiply(config.num_layers, time_step),
                      config.ssm_state_size),
      config.SsmValueSize());
  AddAllocation(total_deltanet,
                QwenRecurrentStateElementBytes(policy.recurrent_state_storage),
                &state);
  return usage;
}

QwenGpuMemoryUsage QwenGpuArena::GetMemoryUsage() const {
  return EstimateMemoryUsage(config_, max_context_, policy_);
}

std::unique_ptr<QwenGpuSnapshot> QwenGpuArena::SaveSnapshot(
    std::uint32_t valid_context) {
  if (valid_context > max_context_) {
    throw std::length_error("Qwen snapshot exceeds the context length");
  }

  auto snapshot = std::unique_ptr<QwenGpuSnapshot>(new QwenGpuSnapshot());
  snapshot->num_layers_ = config_.num_layers;
  snapshot->attention_layers_ = config_.FullAttentionLayerCount();
  snapshot->full_attention_interval_ = config_.full_attention_interval;
  snapshot->kv_width_ = config_.num_key_value_heads * config_.head_dim;
  snapshot->max_context_ = max_context_;
  snapshot->valid_context_ = valid_context;
  snapshot->kv_storage_ = policy_.kv_cache_storage;
  snapshot->recurrent_state_storage_ = policy_.recurrent_state_storage;
  snapshot->kv_elements_per_plane_ = CheckedMultiply(
      CheckedMultiply(snapshot->attention_layers_, max_context_),
      snapshot->kv_width_);
  snapshot->conv_elements_per_layer_ =
      CheckedMultiply(config_.SsmQkvSize(), config_.ssm_conv_kernel);
  snapshot->conv_elements_ =
      CheckedMultiply(config_.num_layers, snapshot->conv_elements_per_layer_);
  snapshot->deltanet_elements_per_layer_ = CheckedMultiply(
      CheckedMultiply(config_.ssm_time_step_rank, config_.ssm_state_size),
      config_.SsmValueSize());
  snapshot->deltanet_elements_ = CheckedMultiply(
      config_.num_layers, snapshot->deltanet_elements_per_layer_);

  const std::size_t kv_f32_bytes = CheckedMultiply(
      CheckedMultiply(snapshot->kv_elements_per_plane_, 2), sizeof(float));
  const std::size_t kv_f16_bytes =
      CheckedMultiply(CheckedMultiply(snapshot->kv_elements_per_plane_, 2),
                      sizeof(std::uint16_t));
  const std::size_t conv_bytes =
      CheckedMultiply(snapshot->conv_elements_, sizeof(float));
  const std::size_t deltanet_bytes = CheckedMultiply(
      snapshot->deltanet_elements_,
      QwenRecurrentStateElementBytes(snapshot->recurrent_state_storage_));
  CheckedAdd(policy_.UsesFp16AttentionKv() ? kv_f16_bytes : kv_f32_bytes,
             &snapshot->payload_bytes_);
  CheckedAdd(conv_bytes, &snapshot->payload_bytes_);
  CheckedAdd(deltanet_bytes, &snapshot->payload_bytes_);

  if (kv_f32_bytes != 0) {
    if (policy_.UsesFp16AttentionKv()) {
      ThrowOnHipError(hipMalloc(&snapshot->d_kv_f16_, kv_f16_bytes),
                      "failed to allocate Qwen FP16 snapshot KV");
    } else {
      ThrowOnHipError(hipMalloc(&snapshot->d_kv_f32_, kv_f32_bytes),
                      "failed to allocate Qwen FP32 snapshot KV");
    }
  }
  ThrowOnHipError(hipMalloc(&snapshot->d_ssm_conv_, conv_bytes),
                  "failed to allocate Qwen convolution snapshot");
  ThrowOnHipError(hipMalloc(&snapshot->d_ssm_deltanet_, deltanet_bytes),
                  "failed to allocate Qwen DeltaNet snapshot");

  if (kv_f32_bytes != 0) {
    if (policy_.UsesFp16AttentionKv()) {
      ThrowOnHipError(
          hipMemcpyAsync(snapshot->d_kv_f16_, d_attention_kv_f16, kv_f16_bytes,
                         hipMemcpyDeviceToDevice, stream),
          "failed to capture Qwen FP16 KV snapshot");
    } else {
      ThrowOnHipError(
          hipMemcpyAsync(snapshot->d_kv_f32_, d_kv_cache, kv_f32_bytes,
                         hipMemcpyDeviceToDevice, stream),
          "failed to capture Qwen FP32 KV snapshot");
    }
  }
  ThrowOnHipError(hipMemcpyAsync(snapshot->d_ssm_conv_, d_ssm_conv_state,
                                 conv_bytes, hipMemcpyDeviceToDevice, stream),
                  "failed to capture Qwen convolution snapshot");
  ThrowOnHipError(
      hipMemcpyAsync(snapshot->d_ssm_deltanet_, d_ssm_deltanet_state,
                     deltanet_bytes, hipMemcpyDeviceToDevice, stream),
      "failed to capture Qwen DeltaNet snapshot");
  ThrowOnHipError(hipStreamSynchronize(stream),
                  "failed to synchronize Qwen snapshot");
  return snapshot;
}

void QwenGpuArena::RestoreSnapshot(const QwenGpuSnapshot& snapshot) {
  const std::uint32_t attention_layers = config_.FullAttentionLayerCount();
  const std::uint32_t kv_width = config_.num_key_value_heads * config_.head_dim;
  const std::size_t conv_elements =
      CheckedMultiply(CheckedMultiply(config_.num_layers, config_.SsmQkvSize()),
                      config_.ssm_conv_kernel);
  const std::size_t deltanet_elements = CheckedMultiply(
      CheckedMultiply(
          CheckedMultiply(config_.num_layers, config_.ssm_time_step_rank),
          config_.ssm_state_size),
      config_.SsmValueSize());
  if (snapshot.valid_context_ > max_context_ ||
      snapshot.attention_layers_ != attention_layers ||
      snapshot.kv_width_ != kv_width || snapshot.max_context_ != max_context_ ||
      snapshot.kv_storage_ != policy_.kv_cache_storage ||
      snapshot.recurrent_state_storage_ != policy_.recurrent_state_storage ||
      snapshot.conv_elements_ != conv_elements ||
      snapshot.deltanet_elements_ != deltanet_elements) {
    throw std::invalid_argument("Qwen snapshot is incompatible with the arena");
  }

  Reset();
  const std::size_t kv_f32_bytes = CheckedMultiply(
      CheckedMultiply(snapshot.kv_elements_per_plane_, 2), sizeof(float));
  const std::size_t kv_f16_bytes =
      CheckedMultiply(CheckedMultiply(snapshot.kv_elements_per_plane_, 2),
                      sizeof(std::uint16_t));
  if (kv_f32_bytes != 0) {
    if (policy_.UsesFp16AttentionKv()) {
      ThrowOnHipError(
          hipMemcpyAsync(d_attention_kv_f16, snapshot.d_kv_f16_, kv_f16_bytes,
                         hipMemcpyDeviceToDevice, stream),
          "failed to restore Qwen FP16 KV snapshot");
    } else {
      ThrowOnHipError(
          hipMemcpyAsync(d_kv_cache, snapshot.d_kv_f32_, kv_f32_bytes,
                         hipMemcpyDeviceToDevice, stream),
          "failed to restore Qwen FP32 KV snapshot");
    }
  }
  ThrowOnHipError(hipMemcpyAsync(d_ssm_conv_state, snapshot.d_ssm_conv_,
                                 snapshot.conv_elements_ * sizeof(float),
                                 hipMemcpyDeviceToDevice, stream),
                  "failed to restore Qwen convolution snapshot");
  ThrowOnHipError(hipMemcpyAsync(d_ssm_deltanet_state, snapshot.d_ssm_deltanet_,
                                 snapshot.deltanet_elements_ *
                                     QwenRecurrentStateElementBytes(
                                         policy_.recurrent_state_storage),
                                 hipMemcpyDeviceToDevice, stream),
                  "failed to restore Qwen DeltaNet snapshot");
  ThrowOnHipError(hipStreamSynchronize(stream),
                  "failed to synchronize Qwen snapshot restore");
}

void QwenGpuArena::RestoreCompactSnapshot(
    std::span<const std::uint8_t> payload,
    std::uint32_t expected_valid_context) {
  const auto layout = ParseCompactSnapshotLayout(payload);
  const std::uint32_t attention_layers = config_.FullAttentionLayerCount();
  const std::uint32_t kv_width = config_.num_key_value_heads * config_.head_dim;
  const std::size_t conv_elements_per_layer =
      CheckedMultiply(config_.SsmQkvSize(), config_.ssm_conv_kernel);
  const std::size_t deltanet_elements_per_layer = CheckedMultiply(
      CheckedMultiply(config_.ssm_time_step_rank, config_.ssm_state_size),
      config_.SsmValueSize());
  const std::size_t recurrent_layers = config_.num_layers - attention_layers;
  const std::size_t conv_elements =
      CheckedMultiply(recurrent_layers, conv_elements_per_layer);
  const std::size_t deltanet_elements =
      CheckedMultiply(recurrent_layers, deltanet_elements_per_layer);
  if (layout.valid_context != expected_valid_context ||
      layout.valid_context > max_context_ ||
      layout.attention_layers != attention_layers ||
      layout.kv_width != kv_width || layout.max_context != max_context_ ||
      layout.kv_storage != policy_.kv_cache_storage ||
      layout.recurrent_state_storage != policy_.recurrent_state_storage ||
      layout.conv_elements != conv_elements ||
      layout.deltanet_elements != deltanet_elements) {
    throw std::invalid_argument(
        "Qwen compact snapshot is incompatible with the arena");
  }

  Reset();
  if (layout.live_kv_bytes_per_plane != 0) {
    const std::size_t kv_element_bytes =
        policy_.UsesFp16AttentionKv() ? sizeof(std::uint16_t) : sizeof(float);
    const std::size_t destination_pitch = CheckedMultiply(
        CheckedMultiply(max_context_, kv_width), kv_element_bytes);
    const std::size_t source_pitch =
        layout.live_kv_bytes_per_plane / attention_layers;
    auto* destination = static_cast<std::uint8_t*>(
        policy_.UsesFp16AttentionKv() ? d_attention_kv_f16
                                      : static_cast<void*>(d_kv_cache));
    const std::size_t full_plane_bytes = CheckedMultiply(
        CheckedMultiply(CheckedMultiply(attention_layers, max_context_),
                        kv_width),
        kv_element_bytes);
    ThrowOnHipError(
        hipMemcpy2DAsync(destination, destination_pitch,
                         payload.data() + layout.k_offset, source_pitch,
                         source_pitch, attention_layers, hipMemcpyHostToDevice,
                         stream),
        "failed to restore compact Qwen K cache");
    ThrowOnHipError(
        hipMemcpy2DAsync(destination + full_plane_bytes, destination_pitch,
                         payload.data() + layout.v_offset, source_pitch,
                         source_pitch, attention_layers, hipMemcpyHostToDevice,
                         stream),
        "failed to restore compact Qwen V cache");
  }
  if (layout.conv_bytes != 0) {
    const std::size_t row_bytes =
        CheckedMultiply(conv_elements_per_layer, sizeof(float));
    std::size_t source_offset = layout.conv_offset;
    auto* destination = reinterpret_cast<std::uint8_t*>(d_ssm_conv_state);
    for (std::uint32_t group_start = 0; group_start < config_.num_layers;
         group_start += config_.full_attention_interval) {
      const std::size_t remaining = config_.num_layers - group_start;
      const std::size_t rows =
          remaining < config_.full_attention_interval
              ? remaining
              : static_cast<std::size_t>(config_.full_attention_interval - 1U);
      const std::size_t bytes = CheckedMultiply(rows, row_bytes);
      ThrowOnHipError(
          hipMemcpyAsync(destination + CheckedMultiply(group_start, row_bytes),
                         payload.data() + source_offset, bytes,
                         hipMemcpyHostToDevice, stream),
          "failed to restore compact Qwen convolution state");
      source_offset = CheckedSum(source_offset, bytes);
    }
    if (source_offset != layout.deltanet_offset) {
      throw std::logic_error(
          "Qwen compact convolution state has an invalid row count");
    }
  }
  if (layout.deltanet_bytes != 0) {
    const std::size_t row_bytes = CheckedMultiply(
        deltanet_elements_per_layer,
        QwenRecurrentStateElementBytes(policy_.recurrent_state_storage));
    std::size_t source_offset = layout.deltanet_offset;
    auto* destination = static_cast<std::uint8_t*>(d_ssm_deltanet_state);
    for (std::uint32_t group_start = 0; group_start < config_.num_layers;
         group_start += config_.full_attention_interval) {
      const std::size_t remaining = config_.num_layers - group_start;
      const std::size_t rows =
          remaining < config_.full_attention_interval
              ? remaining
              : static_cast<std::size_t>(config_.full_attention_interval - 1U);
      const std::size_t bytes = CheckedMultiply(rows, row_bytes);
      ThrowOnHipError(
          hipMemcpyAsync(destination + CheckedMultiply(group_start, row_bytes),
                         payload.data() + source_offset, bytes,
                         hipMemcpyHostToDevice, stream),
          "failed to restore compact Qwen DeltaNet state");
      source_offset = CheckedSum(source_offset, bytes);
    }
    if (source_offset != layout.total_bytes) {
      throw std::logic_error(
          "Qwen compact DeltaNet state has an invalid row count");
    }
  }
  ThrowOnHipError(hipStreamSynchronize(stream),
                  "failed to synchronize compact Qwen snapshot restore");
}

QwenGpuArena::QwenGpuArena(const core::ModelConfig& config,
                           std::uint32_t max_context,
                           QwenExecutionPolicy policy)
    : config_(config),
      max_context_(std::max(max_context, 1U)),
      max_batch_(std::min(max_context_, kMaxPromptBatch)),
      policy_(policy) {
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamCreate(&prefetch_stream));
  HIP_CHECK(hipEventCreate(&prefetch_event));
  HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
  HIPBLAS_CHECK(hipblasSetStream(hipblas_handle, stream));
  hipblaslt_gemm = std::make_unique<HipblasLtGemm>();

  const std::size_t hidden_size = config_.hidden_size;
  const std::size_t intermediate_size = config_.intermediate_size;
  const std::size_t vocab_size = config_.vocab_size;
  const std::size_t num_layers = config_.num_layers;
  const std::size_t num_kv_heads = config_.num_key_value_heads;
  const std::size_t head_dim = config_.head_dim;
  const std::size_t batch = max_batch_;
  const std::size_t attention_size = config_.AttentionSize();
  const std::size_t kv_size = num_kv_heads * head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config_.SsmQkvSize();
  const std::size_t ssm_inner_size = config_.ssm_inner_size;
  const std::size_t recurrent_width = std::max(attention_size, ssm_inner_size);
  const std::size_t projection_width =
      std::max(q_projection_size, ssm_qkv_size);
  const std::size_t time_step_rank = config_.ssm_time_step_rank;

  HIP_CHECK(hipMalloc(&d_hidden, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_normed, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q, batch * attention_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, batch * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, batch * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_attn_out, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_gate, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_up, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_act, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ffn_out, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_qkv, batch * projection_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out, batch * ssm_qkv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_gate, batch * recurrent_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_out, batch * recurrent_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_buf, batch * time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_buf, batch * time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_kq_scales,
                      batch * config_.ssm_group_count * 3 * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_ssm_alpha_beta, batch * time_step_rank * 2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_logits, vocab_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_prompt_tokens,
                      std::max<std::size_t>(batch, 2) * sizeof(std::uint32_t)));
  HIP_CHECK(
      hipMalloc(&d_target_layer_features, 5 * hidden_size * sizeof(float)));

  const std::size_t scratch_elements =
      batch * std::max<std::size_t>(
                  {intermediate_size, hidden_size, projection_width,
                   ssm_qkv_size + ssm_inner_size + (2 * time_step_rank)});
  HIP_CHECK(
      hipMalloc(&d_scratch_bf16, scratch_elements * sizeof(hip_bfloat16)));
  // The tiled Q8_1 activation layout groups 16 tokens per tile, so size for a
  // batch rounded up to a whole tile. The base layout remains 36 bytes per 32
  // elements; reserve another float per block for the optional K-quant
  // activation-sum sidecar without changing the Q8 payload or its stride.
  const std::size_t q8_rows = ((batch + 15) / 16) * 16;
  const std::size_t q8_row_elements =
      scratch_elements / std::max<std::size_t>(batch, 1);
  const std::size_t scratch_q8_bytes =
      ((((q8_rows * q8_row_elements) + 31) / 32) * sizeof(float) * 10) +
      4096;  // 36-byte payload + 4-byte optional sidecar per 32 elems
  HIP_CHECK(hipMalloc(&d_scratch_q8_act, scratch_q8_bytes));
  const std::size_t split_k_elements = detail::DecodeAttentionScratchElements(
      config_.num_attention_heads, config_.head_dim);
  HIP_CHECK(hipMalloc(&d_split_k_attention, split_k_elements * sizeof(float)));

  // Weight BF16 scratch: largest per-layer matmul weight in bf16 elements,
  // used by the prefill dequant-to-BF16 then BF16 GEMM path.
  const std::size_t max_weight_elems =
      hidden_size *
      std::max<std::size_t>({q_projection_size, kv_size, attention_size,
                             intermediate_size, ssm_qkv_size, ssm_inner_size,
                             time_step_rank});
  HIP_CHECK(
      hipMalloc(&d_weights_bf16, max_weight_elems * sizeof(hip_bfloat16)));
  HIP_CHECK(
      hipMalloc(&d_weights_bf16_aux, max_weight_elems * sizeof(hip_bfloat16)));

  const std::size_t total_kv = config_.FullAttentionLayerCount() *
                               num_kv_heads * max_context_ * head_dim;
  if (policy_.UsesFp16AttentionKv()) {
    HIP_CHECK(
        hipMalloc(&d_attention_kv_f16, total_kv * sizeof(std::uint16_t) * 2));
  } else {
    HIP_CHECK(hipMalloc(&d_kv_cache, total_kv * sizeof(float) * 2));
  }

  const std::size_t total_conv =
      num_layers * ssm_qkv_size * config_.ssm_conv_kernel;
  HIP_CHECK(hipMalloc(&d_ssm_conv_state, total_conv * sizeof(float)));

  const std::size_t total_deltanet = num_layers * time_step_rank *
                                     config_.ssm_state_size *
                                     config_.SsmValueSize();
  HIP_CHECK(hipMalloc(&d_ssm_deltanet_state,
                      total_deltanet * QwenRecurrentStateElementBytes(
                                           policy_.recurrent_state_storage)));

  Reset();
}

QwenGpuArena::~QwenGpuArena() {
  FreeAll();
}

QwenGpuScratchView QwenGpuArena::GetScratchView(
    std::size_t batch_size) noexcept {
  const std::size_t batch = std::min<std::size_t>(batch_size, max_batch_);
  const std::size_t hidden = config_.hidden_size;
  const std::size_t attention = config_.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config_.num_key_value_heads) * config_.head_dim;
  const std::size_t q_projection = 2 * attention;
  const std::size_t ssm_qkv = config_.SsmQkvSize();
  const std::size_t recurrent =
      std::max<std::size_t>(attention, config_.ssm_inner_size);
  const std::size_t projection = std::max<std::size_t>(q_projection, ssm_qkv);
  const std::size_t bf16_scratch =
      batch *
      std::max<std::size_t>({config_.intermediate_size, hidden, projection,
                             ssm_qkv + config_.ssm_inner_size +
                                 (2 * config_.ssm_time_step_rank)});
  const std::size_t split_k = detail::DecodeAttentionScratchElements(
      config_.num_attention_heads, config_.head_dim);
  const std::size_t weight_bf16 =
      hidden * std::max<std::size_t>({q_projection, kv, attention,
                                      config_.intermediate_size, ssm_qkv,
                                      config_.ssm_inner_size,
                                      config_.ssm_time_step_rank});
  return {
      .decode =
          {
              .hidden = {d_hidden, batch * hidden},
              .normed = {d_normed, batch * hidden},
              .logits = {d_logits, config_.vocab_size},
              .bf16 = {static_cast<hip_bfloat16*>(d_scratch_bf16),
                       bf16_scratch},
              .weight_bf16 = {d_weights_bf16, weight_bf16},
              .prompt_tokens = {d_prompt_tokens,
                                std::max<std::size_t>(batch, 2)},
              // d_alpha_buf is reused only after all SSM layers finish.
              // Preserve that exact address while exposing the sampling epoch's
              // uint32 view.
              .sampled_token = {reinterpret_cast<std::uint32_t*>(d_alpha_buf),
                                1},
          },
      .attention =
          {
              .q = {d_q, batch * attention},
              .k = {d_k, batch * kv},
              .v = {d_v, batch * kv},
              .output = {d_attn_out, batch * hidden},
              .split_k = {d_split_k_attention, split_k},
          },
      .ssm =
          {
              .qkv = {d_ssm_qkv, batch * projection},
              .conv_out = {d_conv_out, batch * ssm_qkv},
              .gate = {d_ssm_gate, batch * recurrent},
              .out = {d_ssm_out, batch * recurrent},
              .alpha = {d_alpha_buf, batch * config_.ssm_time_step_rank},
              .beta = {d_beta_buf, batch * config_.ssm_time_step_rank},
          },
      .ffn =
          {
              .gate = {d_ffn_gate, batch * config_.intermediate_size},
              .up = {d_ffn_up, batch * config_.intermediate_size},
              .activation = {d_ffn_act, batch * config_.intermediate_size},
              .out = {d_ffn_out, batch * hidden},
          },
  };
}

void QwenGpuArena::SetTargetLayerCapture(
    std::span<const std::uint32_t> target_layer_ids) {
  if (target_layer_ids.size() > kMaxTargetLayerTaps) {
    throw std::invalid_argument("too many target hidden-layer taps");
  }
  for (const std::uint32_t layer : target_layer_ids) {
    if (layer >= config_.num_layers) {
      throw std::invalid_argument("target hidden-layer tap is out of range");
    }
  }
  target_layer_ids_.assign(target_layer_ids.begin(), target_layer_ids.end());
}

std::optional<std::size_t> QwenGpuArena::GetTargetLayerCaptureIndex(
    std::uint32_t layer) const noexcept {
  const auto iterator = std::ranges::find(target_layer_ids_, layer);
  if (iterator == target_layer_ids_.end()) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(
      std::distance(target_layer_ids_.begin(), iterator));
}

QwenGpuArena::QwenGpuArena(QwenGpuArena&& other) noexcept
    : config_(other.config_),
      max_context_(other.max_context_),
      max_batch_(other.max_batch_),
      policy_(other.policy_),
      target_layer_ids_(std::move(other.target_layer_ids_)) {
  d_hidden = other.d_hidden;
  d_normed = other.d_normed;
  d_q = other.d_q;
  d_k = other.d_k;
  d_v = other.d_v;
  d_attn_out = other.d_attn_out;
  d_ffn_gate = other.d_ffn_gate;
  d_ffn_up = other.d_ffn_up;
  d_ffn_act = other.d_ffn_act;
  d_ffn_out = other.d_ffn_out;
  d_ssm_qkv = other.d_ssm_qkv;
  d_conv_out = other.d_conv_out;
  d_ssm_gate = other.d_ssm_gate;
  d_ssm_out = other.d_ssm_out;
  d_alpha_buf = other.d_alpha_buf;
  d_beta_buf = other.d_beta_buf;
  d_ssm_kq_scales = other.d_ssm_kq_scales;
  d_ssm_alpha_beta = other.d_ssm_alpha_beta;
  d_logits = other.d_logits;
  d_attention_kv_f16 = other.d_attention_kv_f16;
  d_kv_cache = other.d_kv_cache;
  d_ssm_conv_state = other.d_ssm_conv_state;
  d_ssm_deltanet_state = other.d_ssm_deltanet_state;
  d_prompt_tokens = other.d_prompt_tokens;
  d_target_layer_features = other.d_target_layer_features;
  stream = other.stream;
  prefetch_stream = other.prefetch_stream;
  prefetch_event = other.prefetch_event;
  hipblas_handle = other.hipblas_handle;
  hipblaslt_gemm = std::move(other.hipblaslt_gemm);
  d_scratch_bf16 = other.d_scratch_bf16;
  d_scratch_q8_act = other.d_scratch_q8_act;
  d_split_k_attention = other.d_split_k_attention;
  d_weights_bf16 = other.d_weights_bf16;
  d_weights_bf16_aux = other.d_weights_bf16_aux;
  d_saved_ssm_conv_state_ = other.d_saved_ssm_conv_state_;
  d_saved_ssm_deltanet_state_ = other.d_saved_ssm_deltanet_state_;
  d_ssm_replay_qkv_ = other.d_ssm_replay_qkv_;
  d_ssm_replay_alpha_ = other.d_ssm_replay_alpha_;
  d_ssm_replay_beta_ = other.d_ssm_replay_beta_;
  d_ssm_replay_enabled_ = other.d_ssm_replay_enabled_;
  saved_context_ = other.saved_context_;
  replay_last_position_ = other.replay_last_position_;
  replay_captured_positions_ = other.replay_captured_positions_;
  has_saved_state_ = other.has_saved_state_;
  replay_capture_active_ = other.replay_capture_active_;

  other.d_hidden = nullptr;
  other.d_normed = nullptr;
  other.d_q = nullptr;
  other.d_k = nullptr;
  other.d_v = nullptr;
  other.d_attn_out = nullptr;
  other.d_ffn_gate = nullptr;
  other.d_ffn_up = nullptr;
  other.d_ffn_act = nullptr;
  other.d_ffn_out = nullptr;
  other.d_ssm_qkv = nullptr;
  other.d_conv_out = nullptr;
  other.d_ssm_gate = nullptr;
  other.d_ssm_out = nullptr;
  other.d_alpha_buf = nullptr;
  other.d_beta_buf = nullptr;
  other.d_ssm_kq_scales = nullptr;
  other.d_ssm_alpha_beta = nullptr;
  other.d_logits = nullptr;
  other.d_attention_kv_f16 = nullptr;
  other.d_kv_cache = nullptr;
  other.d_ssm_conv_state = nullptr;
  other.d_ssm_deltanet_state = nullptr;
  other.d_prompt_tokens = nullptr;
  other.d_target_layer_features = nullptr;
  other.stream = nullptr;
  other.prefetch_stream = nullptr;
  other.prefetch_event = nullptr;
  other.hipblas_handle = nullptr;
  other.d_scratch_bf16 = nullptr;
  other.d_scratch_q8_act = nullptr;
  other.d_split_k_attention = nullptr;
  other.d_weights_bf16 = nullptr;
  other.d_weights_bf16_aux = nullptr;
  other.d_saved_ssm_conv_state_ = nullptr;
  other.d_saved_ssm_deltanet_state_ = nullptr;
  other.d_ssm_replay_qkv_ = nullptr;
  other.d_ssm_replay_alpha_ = nullptr;
  other.d_ssm_replay_beta_ = nullptr;
  other.d_ssm_replay_enabled_ = nullptr;
  other.saved_context_ = 0;
  other.replay_last_position_ = 0;
  other.replay_captured_positions_ = 0;
  other.has_saved_state_ = false;
  other.replay_capture_active_ = false;
}

QwenGpuArena& QwenGpuArena::operator=(QwenGpuArena&& other) noexcept {
  if (this != &other) {
    FreeAll();
    config_ = other.config_;
    max_context_ = other.max_context_;
    max_batch_ = other.max_batch_;
    policy_ = other.policy_;
    target_layer_ids_ = std::move(other.target_layer_ids_);
    d_hidden = other.d_hidden;
    d_normed = other.d_normed;
    d_q = other.d_q;
    d_k = other.d_k;
    d_v = other.d_v;
    d_attn_out = other.d_attn_out;
    d_ffn_gate = other.d_ffn_gate;
    d_ffn_up = other.d_ffn_up;
    d_ffn_act = other.d_ffn_act;
    d_ffn_out = other.d_ffn_out;
    d_ssm_qkv = other.d_ssm_qkv;
    d_conv_out = other.d_conv_out;
    d_ssm_gate = other.d_ssm_gate;
    d_ssm_out = other.d_ssm_out;
    d_alpha_buf = other.d_alpha_buf;
    d_beta_buf = other.d_beta_buf;
    d_ssm_kq_scales = other.d_ssm_kq_scales;
    d_ssm_alpha_beta = other.d_ssm_alpha_beta;
    d_logits = other.d_logits;
    d_attention_kv_f16 = other.d_attention_kv_f16;
    d_kv_cache = other.d_kv_cache;
    d_ssm_conv_state = other.d_ssm_conv_state;
    d_ssm_deltanet_state = other.d_ssm_deltanet_state;
    d_prompt_tokens = other.d_prompt_tokens;
    d_target_layer_features = other.d_target_layer_features;
    stream = other.stream;
    prefetch_stream = other.prefetch_stream;
    prefetch_event = other.prefetch_event;
    hipblas_handle = other.hipblas_handle;
    hipblaslt_gemm = std::move(other.hipblaslt_gemm);
    d_scratch_bf16 = other.d_scratch_bf16;
    d_scratch_q8_act = other.d_scratch_q8_act;
    d_split_k_attention = other.d_split_k_attention;
    d_weights_bf16 = other.d_weights_bf16;
    d_weights_bf16_aux = other.d_weights_bf16_aux;
    d_saved_ssm_conv_state_ = other.d_saved_ssm_conv_state_;
    d_saved_ssm_deltanet_state_ = other.d_saved_ssm_deltanet_state_;
    d_ssm_replay_qkv_ = other.d_ssm_replay_qkv_;
    d_ssm_replay_alpha_ = other.d_ssm_replay_alpha_;
    d_ssm_replay_beta_ = other.d_ssm_replay_beta_;
    d_ssm_replay_enabled_ = other.d_ssm_replay_enabled_;
    saved_context_ = other.saved_context_;
    replay_last_position_ = other.replay_last_position_;
    replay_captured_positions_ = other.replay_captured_positions_;
    has_saved_state_ = other.has_saved_state_;
    replay_capture_active_ = other.replay_capture_active_;

    other.d_hidden = nullptr;
    other.d_normed = nullptr;
    other.d_q = nullptr;
    other.d_k = nullptr;
    other.d_v = nullptr;
    other.d_attn_out = nullptr;
    other.d_ffn_gate = nullptr;
    other.d_ffn_up = nullptr;
    other.d_ffn_act = nullptr;
    other.d_ffn_out = nullptr;
    other.d_ssm_qkv = nullptr;
    other.d_conv_out = nullptr;
    other.d_ssm_gate = nullptr;
    other.d_ssm_out = nullptr;
    other.d_alpha_buf = nullptr;
    other.d_beta_buf = nullptr;
    other.d_ssm_kq_scales = nullptr;
    other.d_ssm_alpha_beta = nullptr;
    other.d_logits = nullptr;
    other.d_attention_kv_f16 = nullptr;
    other.d_kv_cache = nullptr;
    other.d_ssm_conv_state = nullptr;
    other.d_ssm_deltanet_state = nullptr;
    other.d_prompt_tokens = nullptr;
    other.d_target_layer_features = nullptr;
    other.stream = nullptr;
    other.prefetch_stream = nullptr;
    other.prefetch_event = nullptr;
    other.hipblas_handle = nullptr;
    other.d_scratch_bf16 = nullptr;
    other.d_scratch_q8_act = nullptr;
    other.d_split_k_attention = nullptr;
    other.d_weights_bf16 = nullptr;
    other.d_weights_bf16_aux = nullptr;
    other.d_saved_ssm_conv_state_ = nullptr;
    other.d_saved_ssm_deltanet_state_ = nullptr;
    other.d_ssm_replay_qkv_ = nullptr;
    other.d_ssm_replay_alpha_ = nullptr;
    other.d_ssm_replay_beta_ = nullptr;
    other.d_ssm_replay_enabled_ = nullptr;
    other.saved_context_ = 0;
    other.replay_last_position_ = 0;
    other.replay_captured_positions_ = 0;
    other.has_saved_state_ = false;
    other.replay_capture_active_ = false;
  }
  return *this;
}

void QwenGpuArena::Reset() noexcept {
  const std::size_t num_layers = config_.num_layers;
  const std::size_t num_kv_heads = config_.num_key_value_heads;
  const std::size_t head_dim = config_.head_dim;
  const std::size_t total_kv = config_.FullAttentionLayerCount() *
                               num_kv_heads * max_context_ * head_dim * 2;
  const std::size_t total_conv =
      num_layers * config_.SsmQkvSize() * config_.ssm_conv_kernel;
  const std::size_t total_deltanet = num_layers * config_.ssm_time_step_rank *
                                     config_.ssm_state_size *
                                     config_.SsmValueSize();

  if (d_kv_cache != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_kv_cache, 0, total_kv * sizeof(float), stream));
  }
  if (d_attention_kv_f16 != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_attention_kv_f16, 0,
                             total_kv * sizeof(std::uint16_t), stream));
  }
  if (d_ssm_conv_state != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_ssm_conv_state, 0, total_conv * sizeof(float),
                             stream));
  }
  if (d_ssm_deltanet_state != nullptr) {
    HIP_CHECK(hipMemsetAsync(
        d_ssm_deltanet_state, 0,
        total_deltanet *
            QwenRecurrentStateElementBytes(policy_.recurrent_state_storage),
        stream));
  }
  DisableSsmReplayCapture();
  saved_context_ = 0;
  replay_last_position_ = 0;
  replay_captured_positions_ = 0;
  has_saved_state_ = false;
}

void QwenGpuArena::FreeAll() noexcept {
  if (d_hidden != nullptr)
    HIP_CHECK(hipFree(d_hidden));
  if (d_normed != nullptr)
    HIP_CHECK(hipFree(d_normed));
  if (d_q != nullptr)
    HIP_CHECK(hipFree(d_q));
  if (d_k != nullptr)
    HIP_CHECK(hipFree(d_k));
  if (d_v != nullptr)
    HIP_CHECK(hipFree(d_v));
  if (d_attn_out != nullptr)
    HIP_CHECK(hipFree(d_attn_out));
  if (d_ffn_gate != nullptr)
    HIP_CHECK(hipFree(d_ffn_gate));
  if (d_ffn_up != nullptr)
    HIP_CHECK(hipFree(d_ffn_up));
  if (d_ffn_act != nullptr)
    HIP_CHECK(hipFree(d_ffn_act));
  if (d_ffn_out != nullptr)
    HIP_CHECK(hipFree(d_ffn_out));
  if (d_ssm_qkv != nullptr)
    HIP_CHECK(hipFree(d_ssm_qkv));
  if (d_conv_out != nullptr)
    HIP_CHECK(hipFree(d_conv_out));
  if (d_ssm_gate != nullptr)
    HIP_CHECK(hipFree(d_ssm_gate));
  if (d_ssm_out != nullptr)
    HIP_CHECK(hipFree(d_ssm_out));
  if (d_alpha_buf != nullptr)
    HIP_CHECK(hipFree(d_alpha_buf));
  if (d_beta_buf != nullptr)
    HIP_CHECK(hipFree(d_beta_buf));
  if (d_ssm_kq_scales != nullptr)
    HIP_CHECK(hipFree(d_ssm_kq_scales));
  if (d_ssm_alpha_beta != nullptr)
    HIP_CHECK(hipFree(d_ssm_alpha_beta));
  if (d_logits != nullptr)
    HIP_CHECK(hipFree(d_logits));
  if (d_attention_kv_f16 != nullptr)
    HIP_CHECK(hipFree(d_attention_kv_f16));
  if (d_target_layer_features != nullptr) {
    HIP_CHECK(hipFree(d_target_layer_features));
    d_target_layer_features = nullptr;
  }
  if (d_kv_cache != nullptr)
    HIP_CHECK(hipFree(d_kv_cache));
  if (d_ssm_conv_state != nullptr)
    HIP_CHECK(hipFree(d_ssm_conv_state));
  if (d_ssm_deltanet_state != nullptr)
    HIP_CHECK(hipFree(d_ssm_deltanet_state));
  if (d_prompt_tokens != nullptr)
    HIP_CHECK(hipFree(d_prompt_tokens));
  if (d_scratch_bf16 != nullptr)
    HIP_CHECK(hipFree(d_scratch_bf16));
  if (d_scratch_q8_act != nullptr)
    HIP_CHECK(hipFree(d_scratch_q8_act));
  if (d_split_k_attention != nullptr)
    HIP_CHECK(hipFree(d_split_k_attention));
  if (d_weights_bf16 != nullptr)
    HIP_CHECK(hipFree(d_weights_bf16));
  if (d_weights_bf16_aux != nullptr)
    HIP_CHECK(hipFree(d_weights_bf16_aux));
  if (d_saved_ssm_conv_state_ != nullptr)
    HIP_CHECK(hipFree(d_saved_ssm_conv_state_));
  if (d_saved_ssm_deltanet_state_ != nullptr)
    HIP_CHECK(hipFree(d_saved_ssm_deltanet_state_));
  if (d_ssm_replay_qkv_ != nullptr)
    HIP_CHECK(hipFree(d_ssm_replay_qkv_));
  if (d_ssm_replay_alpha_ != nullptr)
    HIP_CHECK(hipFree(d_ssm_replay_alpha_));
  if (d_ssm_replay_beta_ != nullptr)
    HIP_CHECK(hipFree(d_ssm_replay_beta_));
  if (d_ssm_replay_enabled_ != nullptr)
    HIP_CHECK(hipFree(d_ssm_replay_enabled_));
  if (hipblas_handle != nullptr)
    HIPBLAS_CHECK(hipblasDestroy(hipblas_handle));
  hipblaslt_gemm.reset();
  if (stream != nullptr)
    HIP_CHECK(hipStreamDestroy(stream));
  if (prefetch_stream != nullptr)
    HIP_CHECK(hipStreamDestroy(prefetch_stream));
  if (prefetch_event != nullptr)
    HIP_CHECK(hipEventDestroy(prefetch_event));

  d_hidden = nullptr;
  d_normed = nullptr;
  d_q = nullptr;
  d_k = nullptr;
  d_v = nullptr;
  d_attn_out = nullptr;
  d_ffn_gate = nullptr;
  d_ffn_up = nullptr;
  d_ffn_act = nullptr;
  d_ffn_out = nullptr;
  d_ssm_qkv = nullptr;
  d_conv_out = nullptr;
  d_ssm_gate = nullptr;
  d_ssm_out = nullptr;
  d_alpha_buf = nullptr;
  d_beta_buf = nullptr;
  d_ssm_kq_scales = nullptr;
  d_ssm_alpha_beta = nullptr;
  d_logits = nullptr;
  d_attention_kv_f16 = nullptr;
  d_kv_cache = nullptr;
  d_ssm_conv_state = nullptr;
  d_ssm_deltanet_state = nullptr;
  d_prompt_tokens = nullptr;
  d_scratch_bf16 = nullptr;
  d_split_k_attention = nullptr;
  d_weights_bf16 = nullptr;
  d_weights_bf16_aux = nullptr;
  d_saved_ssm_conv_state_ = nullptr;
  d_saved_ssm_deltanet_state_ = nullptr;
  d_ssm_replay_qkv_ = nullptr;
  d_ssm_replay_alpha_ = nullptr;
  d_ssm_replay_beta_ = nullptr;
  d_ssm_replay_enabled_ = nullptr;
  saved_context_ = 0;
  replay_last_position_ = 0;
  replay_captured_positions_ = 0;
  has_saved_state_ = false;
  replay_capture_active_ = false;
  hipblas_handle = nullptr;
  stream = nullptr;
  prefetch_stream = nullptr;
  prefetch_event = nullptr;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

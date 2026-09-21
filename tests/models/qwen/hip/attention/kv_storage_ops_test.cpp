#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"
#include "tests/models/qwen/support/synthetic_weights.hpp"
#include "tests/testing/test_common.hpp"

namespace {

using gufo::hip::QwenExecutionPolicy;
using gufo::hip::QwenGpuArena;
using gufo::hip::QwenKvCacheStorage;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

QwenExecutionPolicy Fp16Policy() {
  return QwenExecutionPolicy::Production();
}

QwenExecutionPolicy Fp32Policy() {
  auto policy = QwenExecutionPolicy::Production();
  policy.kv_cache_storage = QwenKvCacheStorage::kFp32;
  return policy;
}

gufo::core::ModelConfig ProductionQwen27BConfig() {
  gufo::core::ModelConfig config;
  config.model_name = "qwen3.8-27b";
  config.num_layers = 64;
  config.hidden_size = 5120;
  config.intermediate_size = 17408;
  config.num_attention_heads = 24;
  config.num_key_value_heads = 4;
  config.head_dim = 256;
  config.vocab_size = 248320;
  config.context_length = 262144;
  config.full_attention_interval = 4;
  config.mtp_num_layers = 1;
  config.ssm_conv_kernel = 4;
  config.ssm_state_size = 128;
  config.ssm_group_count = 16;
  config.ssm_time_step_rank = 48;
  config.ssm_inner_size = 6144;
  config.rotary_dim = 64;
  config.rope_theta = 10000000.0F;
  return config;
}

// Verification must preserve every scalar attention bit, including a block
// straddling the online/split-K boundary and a single-row scratch fallback.
void TestCausalDecodeRows() {
  using gufo::test::DeviceBuffer;
  constexpr unsigned heads = 6, kv_heads = 1, dim = 256, rows = 8;
  constexpr unsigned context = 8200, layer = 1, width = heads * dim;
  constexpr std::size_t cache_size = 2U * context * kv_heads * dim;
  std::vector<float> query(rows * width), gate(query.size()), cache(cache_size);
  std::vector<__half> cache_half(cache_size);
  for (std::size_t i = 0; i < query.size(); ++i) {
    query[i] = 0.3F * std::sin(static_cast<float>(i) * 0.073F);
    gate[i] = std::cos(static_cast<float>(i) * 0.019F);
  }
  for (std::size_t i = 0; i < cache.size(); ++i) {
    cache[i] = 0.4F * std::sin(static_cast<float>(i) * 0.011F);
    cache_half[i] = __float2half(cache[i]);
  }
  DeviceBuffer<float> q(query), g(gate), kv(cache), expected(query.size()),
      actual(query.size());
  DeviceBuffer<__half> kv_half(cache_half);
  const auto row_scratch =
      gufo::hip::detail::DecodeAttentionScratchElements(heads, dim);
  DeviceBuffer<float> scratch(rows * row_scratch);
  for (bool fp16 : {false, true}) {
    for (bool gated : {false, true}) {
      for (unsigned batch : {1U, 3U, rows}) {
        for (unsigned start : {0U, 17U, 4088U, 4092U, 4095U, 8192U}) {
          float* const cache32 = fp16 ? nullptr : kv.data();
          void* const cache16 = fp16 ? kv_half.data() : nullptr;
          for (unsigned row = 0; row < batch; ++row) {
            gufo::hip::LaunchAttention(q.data() + row * width, nullptr, nullptr,
                                       gated ? g.data() + row * width : nullptr,
                                       cache32, cache32, cache16, cache16,
                                       expected.data() + row * width, layer,
                                       start + row, context, heads, kv_heads,
                                       dim, nullptr, scratch.data(), true);
          }
          const auto reference = expected.CopyToHost();
          for (auto capacity : {row_scratch, scratch.size()}) {
            gufo::hip::LaunchCausalDecodeAttention(
                q.data(), gated ? g.data() : nullptr, cache32, cache32, cache16,
                cache16, actual.data(), layer, start, batch, context, heads,
                kv_heads, dim, nullptr, {scratch.data(), capacity});
            const auto result = actual.CopyToHost();
            Expect(std::memcmp(reference.data(), result.data(),
                               batch * width * sizeof(float)) == 0,
                   "causal verification attention changed scalar decode bits");
          }
        }
      }
    }
  }
}

void TestProductionMemoryScaling() {
  const auto config = ProductionQwen27BConfig();
  constexpr std::uint32_t context = 4096;
  const auto fp32_usage =
      QwenGpuArena::EstimateMemoryUsage(config, context, Fp32Policy());
  const auto fp16_usage =
      QwenGpuArena::EstimateMemoryUsage(config, context, Fp16Policy());
  constexpr std::size_t kv_savings_per_state =
      16U * 2U * 4U * 256U * context * (sizeof(float) - sizeof(std::uint16_t));

  Expect(fp32_usage.request_state_bytes - fp16_usage.request_state_bytes ==
             kv_savings_per_state,
         "production FP16 state must remove the FP32-minus-FP16 KV bytes");
  Expect(
      fp32_usage.temporary_scratch_bytes == fp16_usage.temporary_scratch_bytes,
      "production scratch must be independent of canonical KV precision");

  std::cout << "Qwen27B context=" << context
            << " fp32_state_bytes=" << fp32_usage.request_state_bytes
            << " fp16_state_bytes=" << fp16_usage.request_state_bytes
            << " scratch_bytes=" << fp16_usage.temporary_scratch_bytes
            << " kv_savings_per_state=" << kv_savings_per_state << '\n';
  for (const std::size_t concurrency : {1U, 2U, 4U}) {
    std::cout << "Qwen27B C=" << concurrency << " fp32_reserved_bytes="
              << concurrency * fp32_usage.TotalBytes()
              << " fp16_reserved_bytes="
              << concurrency * fp16_usage.TotalBytes()
              << " saved_bytes=" << concurrency * kv_savings_per_state << '\n';
  }
}

void TestCanonicalMemoryAccountingAndSnapshot() {
  const auto config = gufo::models::qwen::make_small_qwen_config();
  constexpr std::uint32_t context = 32;
  const auto fp32 = Fp32Policy();
  const auto fp16 = Fp16Policy();
  const auto fp32_usage =
      QwenGpuArena::EstimateMemoryUsage(config, context, fp32);
  const auto fp16_usage =
      QwenGpuArena::EstimateMemoryUsage(config, context, fp16);

  const std::size_t kv_elements = 2U * config.FullAttentionLayerCount() *
                                  context * config.num_key_value_heads *
                                  config.head_dim;
  const std::size_t duplicate_bytes =
      kv_elements * (sizeof(float) - sizeof(std::uint16_t));
  Expect(fp32_usage.request_state_bytes - fp16_usage.request_state_bytes ==
             duplicate_bytes,
         "FP16-only accounting must remove exactly the duplicate KV bytes");
  Expect(
      fp32_usage.temporary_scratch_bytes == fp16_usage.temporary_scratch_bytes,
      "KV storage selection must not change shared scratch accounting");

  QwenGpuArena fp16_arena(config, context, fp16);
  Expect(fp16_arena.d_kv_cache == nullptr,
         "FP16-only arena must not allocate an FP32 KV plane");
  Expect(fp16_arena.d_attention_kv_f16 != nullptr,
         "FP16-only arena must allocate its canonical KV plane");
  auto snapshot = fp16_arena.SaveSnapshot(7);
  Expect(snapshot->KvStorage() == QwenKvCacheStorage::kFp16,
         "snapshot must record the canonical FP16 KV format");
  Expect(snapshot->PayloadBytes() == fp16_arena.SnapshotPayloadBytes(7),
         "snapshot payload must account one canonical state representation");

  // FP32 reference KV is head-major, unlike the production FP16 plane.
  QwenGpuArena fp32_arena(config, context, fp32);
  HIP_CHECK(hipStreamSynchronize(fp32_arena.stream));
  std::vector<float> source(kv_elements), actual(kv_elements);
  for (std::size_t i = 0; i < source.size(); ++i)
    source[i] = static_cast<float>(i + 1);
  HIP_CHECK(hipMemcpy(fp32_arena.d_kv_cache, source.data(),
                      source.size() * sizeof(float), hipMemcpyHostToDevice));
  auto saved = fp32_arena.SaveSnapshot(7);
  std::vector<std::uint8_t> serialized(saved->CompactPayloadBytes());
  saved->SerializeCompact(serialized);
  for (const bool disk : {false, true}) {
    HIP_CHECK(
        hipMemset(fp32_arena.d_kv_cache, 0, source.size() * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());
    if (disk)
      fp32_arena.RestoreCompactSnapshot(serialized, 7);
    else
      fp32_arena.RestoreSnapshot(*saved);
    HIP_CHECK(hipMemcpy(actual.data(), fp32_arena.d_kv_cache,
                        actual.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < source.size(); ++i) {
      const auto position = (i / config.head_dim) % context;
      Expect(actual[i] == (position < 7 ? source[i] : 0.0F),
             "FP32 snapshot lost a head's valid prefix or restored its tail");
    }
  }
}

void TestCompactPersistentSnapshotRoundTrip() {
  const auto config = gufo::models::qwen::make_small_qwen_config();
  constexpr std::uint32_t context = 32;
  constexpr std::uint32_t valid_context = 7;
  constexpr std::size_t header_bytes = 80;
  const auto policy = Fp16Policy();
  QwenGpuArena source(config, context, policy);

  const std::size_t attention_layers = config.FullAttentionLayerCount();
  const std::size_t kv_width =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t full_kv_elements_per_plane =
      attention_layers * context * kv_width;
  const std::size_t live_kv_elements_per_plane =
      attention_layers * valid_context * kv_width;
  std::vector<std::uint16_t> source_kv(2U * full_kv_elements_per_plane);
  for (std::size_t index = 0; index < source_kv.size(); ++index) {
    source_kv[index] = static_cast<std::uint16_t>(0x1000U + (index % 0x0FFFU));
  }

  const std::size_t conv_elements =
      static_cast<std::size_t>(config.SsmLayerCount()) * config.SsmQkvSize() *
      config.ssm_conv_kernel;
  const std::size_t deltanet_elements =
      static_cast<std::size_t>(config.SsmLayerCount()) *
      config.ssm_time_step_rank * config.ssm_state_size * config.SsmValueSize();
  const std::size_t recurrent_layers = config.SsmLayerCount();
  const std::size_t conv_elements_per_layer =
      config.SsmQkvSize() * config.ssm_conv_kernel;
  const std::size_t deltanet_elements_per_layer =
      static_cast<std::size_t>(config.ssm_time_step_rank) *
      config.ssm_state_size * config.SsmValueSize();
  std::vector<float> source_conv(conv_elements);
  std::vector<float> source_deltanet(deltanet_elements);
  for (std::size_t index = 0; index < source_conv.size(); ++index) {
    source_conv[index] = static_cast<float>(index + 1U) * 0.000125F;
  }
  for (std::size_t index = 0; index < source_deltanet.size(); ++index) {
    source_deltanet[index] =
        static_cast<float>((index % 257U) + 1U) * -0.00025F;
  }

  HIP_CHECK(hipMemcpy(source.d_attention_kv_f16, source_kv.data(),
                      source_kv.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(source.d_ssm_conv_state, source_conv.data(),
                      source_conv.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(source.d_ssm_deltanet_state, source_deltanet.data(),
                      source_deltanet.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  auto snapshot = source.SaveSnapshot(valid_context);
  const std::size_t expected_compact_bytes =
      header_bytes + 2U * live_kv_elements_per_plane * sizeof(std::uint16_t) +
      recurrent_layers * conv_elements_per_layer * sizeof(float) +
      recurrent_layers * deltanet_elements_per_layer * sizeof(float);
  Expect(snapshot->CompactPayloadBytes() == expected_compact_bytes,
         "compact snapshot must contain only live KV rows and recurrent state");
  Expect(snapshot->CompactPayloadBytes() ==
             snapshot->PayloadBytes() + header_bytes,
         "GPU and disk snapshots must retain the same compact state");

  std::vector<std::uint8_t> payload(snapshot->CompactPayloadBytes());
  Expect(snapshot->SerializeCompact(payload) == payload.size(),
         "compact snapshot serializer byte count");

  QwenGpuArena restored(config, context, policy);
  HIP_CHECK(hipMemset(restored.d_attention_kv_f16, 0xA5,
                      source_kv.size() * sizeof(std::uint16_t)));
  restored.RestoreCompactSnapshot(payload, valid_context);
  std::vector<std::uint16_t> restored_kv(source_kv.size());
  std::vector<float> restored_conv(source_conv.size());
  std::vector<float> restored_deltanet(source_deltanet.size());
  HIP_CHECK(hipMemcpy(restored_kv.data(), restored.d_attention_kv_f16,
                      restored_kv.size() * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(restored_conv.data(), restored.d_ssm_conv_state,
                      restored_conv.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(restored_deltanet.data(), restored.d_ssm_deltanet_state,
                      restored_deltanet.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t plane = 0; plane < 2; ++plane) {
    for (std::size_t layer = 0; layer < attention_layers; ++layer) {
      for (std::size_t position = 0; position < context; ++position) {
        for (std::size_t column = 0; column < kv_width; ++column) {
          const std::size_t index = plane * full_kv_elements_per_plane +
                                    layer * context * kv_width +
                                    position * kv_width + column;
          if (position < valid_context) {
            Expect(restored_kv[index] == source_kv[index],
                   "compact snapshot changed a live KV element");
          } else {
            Expect(restored_kv[index] == 0xA5A5,
                   "compact snapshot touched an unused KV-tail element");
          }
        }
      }
    }
  }
  Expect(restored_conv == source_conv,
         "compact snapshot changed convolution state");
  Expect(restored_deltanet == source_deltanet,
         "compact snapshot changed DeltaNet state");

  const auto retained_kv = restored_kv;
  auto truncated = payload;
  truncated.pop_back();
  bool rejected = false;
  try {
    restored.RestoreCompactSnapshot(truncated, valid_context);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Expect(rejected, "truncated compact snapshot must be rejected");
  HIP_CHECK(hipMemcpy(restored_kv.data(), restored.d_attention_kv_f16,
                      restored_kv.size() * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  Expect(restored_kv == retained_kv,
         "truncated compact snapshot mutated the destination arena");

  auto incompatible = payload;
  incompatible[24] ^= 1U;
  rejected = false;
  try {
    restored.RestoreCompactSnapshot(incompatible, valid_context);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Expect(rejected, "incompatible compact snapshot must be rejected");
}

void FillDecodeInputs(std::vector<float>* query, std::vector<float>* key,
                      std::vector<float>* value, std::vector<float>* gate,
                      std::size_t positions, std::uint32_t num_heads,
                      std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::size_t attention_width =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  query->resize(positions * attention_width);
  key->resize(positions * kv_width);
  value->resize(positions * kv_width);
  gate->resize(positions * attention_width);
  for (std::size_t index = 0; index < query->size(); ++index) {
    (*query)[index] =
        0.09F * std::sin(static_cast<float>((index % 509U) + 1U) * 0.013F);
    (*gate)[index] =
        0.4F * std::cos(static_cast<float>((index % 193U) + 1U) * 0.017F);
  }
  for (std::size_t index = 0; index < key->size(); ++index) {
    (*key)[index] =
        0.08F * std::cos(static_cast<float>((index % 251U) + 1U) * 0.019F);
    (*value)[index] =
        0.2F * std::sin(static_cast<float>((index % 239U) + 1U) * 0.023F);
  }
}

void TestOnlineAndGraphDecodeFp16Equivalence() {
  constexpr std::uint32_t num_heads = 6;
  constexpr std::uint32_t num_kv_heads = 1;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t max_context = 129;
  constexpr std::size_t positions = 128;
  const std::size_t attention_width =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t cache_elements =
      static_cast<std::size_t>(num_kv_heads) * max_context * head_dim;

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> gate;
  FillDecodeInputs(&query, &key, &value, &gate, positions + 1, num_heads,
                   num_kv_heads, head_dim);

  float *d_query = nullptr, *d_key = nullptr, *d_value = nullptr;
  float *d_gate = nullptr, *d_fp32_cache = nullptr;
  void* d_fp16_cache = nullptr;
  float *d_fp32_out = nullptr, *d_fp16_out = nullptr;
  std::uint32_t* d_position = nullptr;
  HIP_CHECK(hipMalloc(&d_query, query.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_key, key.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_value, value.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, gate.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_fp32_cache, 2U * cache_elements * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_fp16_cache, 2U * cache_elements * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_fp32_out,
                      (positions + 1U) * attention_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_fp16_out,
                      (positions + 1U) * attention_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_position, sizeof(std::uint32_t)));
  HIP_CHECK(hipMemcpy(d_query, query.data(), query.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_key, key.data(), key.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_value, value.data(), value.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, gate.data(), gate.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_fp32_cache, 0xFF, 2U * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_fp16_cache, 0xFF,
                      2U * cache_elements * sizeof(std::uint16_t)));

  for (std::uint32_t position = 0; position < positions; ++position) {
    const std::size_t attention_offset =
        static_cast<std::size_t>(position) * attention_width;
    const std::size_t kv_offset = static_cast<std::size_t>(position) * kv_width;
    gufo::hip::LaunchAttention(
        d_query + attention_offset, d_key + kv_offset, d_value + kv_offset,
        d_gate + attention_offset, d_fp32_cache, d_fp32_cache + cache_elements,
        nullptr, nullptr, d_fp32_out + attention_offset, 0, position,
        max_context, num_heads, num_kv_heads, head_dim);
    gufo::hip::LaunchAttention(
        d_query + attention_offset, d_key + kv_offset, d_value + kv_offset,
        d_gate + attention_offset, nullptr, nullptr, d_fp16_cache,
        static_cast<std::uint16_t*>(d_fp16_cache) + cache_elements,
        d_fp16_out + attention_offset, 0, position, max_context, num_heads,
        num_kv_heads, head_dim);
  }

  const std::uint32_t graph_position = positions;
  HIP_CHECK(hipMemcpy(d_position, &graph_position, sizeof(graph_position),
                      hipMemcpyHostToDevice));
  const std::size_t graph_attention_offset = positions * attention_width;
  const std::size_t graph_kv_offset = positions * kv_width;
  gufo::hip::LaunchAttention(d_query + graph_attention_offset,
                             d_key + graph_kv_offset, d_value + graph_kv_offset,
                             d_gate + graph_attention_offset, d_fp32_cache,
                             d_fp32_cache + cache_elements, nullptr, nullptr,
                             d_fp32_out + graph_attention_offset, 0, d_position,
                             max_context, num_heads, num_kv_heads, head_dim);
  gufo::hip::LaunchAttention(
      d_query + graph_attention_offset, d_key + graph_kv_offset,
      d_value + graph_kv_offset, d_gate + graph_attention_offset, nullptr,
      nullptr, d_fp16_cache,
      static_cast<std::uint16_t*>(d_fp16_cache) + cache_elements,
      d_fp16_out + graph_attention_offset, 0, d_position, max_context,
      num_heads, num_kv_heads, head_dim);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> fp32((positions + 1U) * attention_width);
  std::vector<float> fp16(fp32.size());
  HIP_CHECK(hipMemcpy(fp32.data(), d_fp32_out, fp32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(fp16.data(), d_fp16_out, fp16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  float max_diff = 0.0F;
  for (std::size_t index = 0; index < fp32.size(); ++index) {
    Expect(std::isfinite(fp16[index]),
           "FP16 online/graph decode produced a non-finite result");
    max_diff = std::max(max_diff, std::abs(fp32[index] - fp16[index]));
  }
  std::cout << "FP16 online+graph decode max FP32 diff: " << max_diff << '\n';
  Expect(max_diff < 7e-4F,
         "FP16 online/graph decode exceeded the numerical envelope");

  HIP_CHECK(hipFree(d_query));
  HIP_CHECK(hipFree(d_key));
  HIP_CHECK(hipFree(d_value));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_fp32_cache));
  HIP_CHECK(hipFree(d_fp16_cache));
  HIP_CHECK(hipFree(d_fp32_out));
  HIP_CHECK(hipFree(d_fp16_out));
  HIP_CHECK(hipFree(d_position));
}

void TestSplitKDecodeFp16Equivalence() {
  constexpr std::size_t positions = 4096;
  constexpr std::uint32_t num_heads = 6;
  constexpr std::uint32_t num_kv_heads = 1;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t max_context = positions;
  constexpr std::uint32_t rotary_dim = 64;
  const std::size_t attention_width =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t cache_elements =
      static_cast<std::size_t>(num_kv_heads) * max_context * head_dim;

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> gate;
  FillDecodeInputs(&query, &key, &value, &gate, positions, num_heads,
                   num_kv_heads, head_dim);
  std::vector<float> q_weight(head_dim, 1.0F);
  std::vector<float> k_weight(head_dim, 1.0F);

  float *d_query = nullptr, *d_key = nullptr, *d_value = nullptr;
  float *d_gate = nullptr, *d_q_weight = nullptr, *d_k_weight = nullptr;
  float *d_fp32_cache = nullptr, *d_fp32_out = nullptr, *d_fp16_out = nullptr;
  void* d_fp16_cache = nullptr;
  float *d_fp32_scratch = nullptr, *d_fp16_scratch = nullptr;
  HIP_CHECK(hipMalloc(&d_query, query.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_key, key.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_value, value.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, gate.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q_weight, q_weight.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_weight, k_weight.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_fp32_cache, 2U * cache_elements * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_fp16_cache, 2U * cache_elements * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_fp32_out, attention_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_fp16_out, attention_width * sizeof(float)));
  const std::size_t scratch_elements =
      gufo::hip::detail::DecodeAttentionScratchElements(num_heads, head_dim);
  HIP_CHECK(hipMalloc(&d_fp32_scratch, scratch_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_fp16_scratch, scratch_elements * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_query, query.data(), query.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_key, key.data(), key.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_value, value.data(), value.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, gate.data(), gate.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_q_weight, q_weight.data(),
                      q_weight.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_weight, k_weight.data(),
                      k_weight.size() * sizeof(float), hipMemcpyHostToDevice));

  gufo::hip::LaunchBatchedFusedQKNormRoPEKvWrite(
      d_query, d_key, d_value, d_q_weight, d_k_weight, d_query, d_key,
      d_fp32_cache, d_fp32_cache + cache_elements, d_fp16_cache,
      static_cast<std::uint16_t*>(d_fp16_cache) + cache_elements, 0, 0,
      positions, max_context, num_heads, num_kv_heads, head_dim, rotary_dim,
      1000000.0F);
  const std::size_t final_attention_offset = (positions - 1U) * attention_width;
  const std::size_t final_kv_offset = (positions - 1U) * kv_width;
  gufo::hip::LaunchAttention(
      d_query + final_attention_offset, d_key + final_kv_offset,
      d_value + final_kv_offset, d_gate + final_attention_offset, d_fp32_cache,
      d_fp32_cache + cache_elements, nullptr, nullptr, d_fp32_out, 0,
      static_cast<std::uint32_t>(positions - 1U), max_context, num_heads,
      num_kv_heads, head_dim, nullptr, d_fp32_scratch,
      /*skip_kv_write=*/true);
  gufo::hip::LaunchAttention(
      d_query + final_attention_offset, d_key + final_kv_offset,
      d_value + final_kv_offset, d_gate + final_attention_offset, nullptr,
      nullptr, d_fp16_cache,
      static_cast<std::uint16_t*>(d_fp16_cache) + cache_elements, d_fp16_out, 0,
      static_cast<std::uint32_t>(positions - 1U), max_context, num_heads,
      num_kv_heads, head_dim, nullptr, d_fp16_scratch,
      /*skip_kv_write=*/true);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> fp32(attention_width);
  std::vector<float> fp16(attention_width);
  HIP_CHECK(hipMemcpy(fp32.data(), d_fp32_out, fp32.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(fp16.data(), d_fp16_out, fp16.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  float max_diff = 0.0F;
  for (std::size_t index = 0; index < fp32.size(); ++index) {
    Expect(std::isfinite(fp16[index]),
           "FP16 split-K decode produced a non-finite result");
    max_diff = std::max(max_diff, std::abs(fp32[index] - fp16[index]));
  }
  std::cout << "FP16 split-K decode max FP32 diff: " << max_diff << '\n';
  Expect(max_diff < 1.5e-3F,
         "FP16 split-K decode exceeded the numerical envelope");

  HIP_CHECK(hipFree(d_query));
  HIP_CHECK(hipFree(d_key));
  HIP_CHECK(hipFree(d_value));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_q_weight));
  HIP_CHECK(hipFree(d_k_weight));
  HIP_CHECK(hipFree(d_fp32_cache));
  HIP_CHECK(hipFree(d_fp16_cache));
  HIP_CHECK(hipFree(d_fp32_out));
  HIP_CHECK(hipFree(d_fp16_out));
  HIP_CHECK(hipFree(d_fp32_scratch));
  HIP_CHECK(hipFree(d_fp16_scratch));
}

}  // namespace
#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen FP16 KV storage ops test");
  if (device_status != 0) {
    return device_status;
  }
  TestCausalDecodeRows();
  TestProductionMemoryScaling();
  TestCanonicalMemoryAccountingAndSnapshot();
  TestCompactPersistentSnapshotRoundTrip();
  TestOnlineAndGraphDecodeFp16Equivalence();
  TestSplitKDecodeFp16Equivalence();
  std::cout << "Qwen FP16 KV storage tests passed.\n";
#endif
  return 0;
}

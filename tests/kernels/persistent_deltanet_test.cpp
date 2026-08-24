#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/model_config.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t elements) : elements_(elements) {
    HIP_CHECK(hipMalloc(&data_, elements_ * sizeof(T)));
  }

  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;

  [[nodiscard]] T* Get() const noexcept { return data_; }
  [[nodiscard]] std::size_t Size() const noexcept { return elements_; }

private:
  T* data_{nullptr};
  std::size_t elements_{0};
};

template<typename T>
void CopyToDevice(const HipBuffer<T>& destination,
                  const std::vector<T>& source) {
  Expect(destination.Size() == source.size(), "device copy size mismatch");
  HIP_CHECK(hipMemcpy(destination.Get(), source.data(),
                      source.size() * sizeof(T), hipMemcpyHostToDevice));
}

template<typename T>
std::vector<T> CopyToHost(const T* source, std::size_t elements) {
  std::vector<T> output(elements);
  HIP_CHECK(hipMemcpy(output.data(), source, elements * sizeof(T),
                      hipMemcpyDeviceToHost));
  return output;
}

void TestArenaRestoresOnlyMutableRecurrentState() {
  strix::core::ModelConfig config;
  config.num_layers = 4;
  config.hidden_size = 64;
  config.intermediate_size = 64;
  config.num_attention_heads = 2;
  config.num_key_value_heads = 1;
  config.head_dim = 32;
  config.vocab_size = 128;
  config.context_length = 64;
  config.ssm_state_size = 32;
  config.ssm_group_count = 1;
  config.ssm_time_step_rank = 2;
  config.ssm_inner_size = 64;
  config.rotary_dim = 32;

  strix::hip::QwenGpuArena arena(config, config.context_length);
  const std::size_t conv_elements =
      static_cast<std::size_t>(config.num_layers) * config.SsmQkvSize() *
      config.ssm_conv_kernel;
  const std::size_t deltanet_elements =
      static_cast<std::size_t>(config.num_layers) * config.ssm_time_step_rank *
      config.ssm_state_size * config.SsmValueSize();
  const std::size_t kv_elements =
      static_cast<std::size_t>(config.FullAttentionLayerCount()) *
      config.num_key_value_heads * config.context_length * config.head_dim * 2;

  std::vector<float> conv_initial(conv_elements);
  std::vector<float> deltanet_initial(deltanet_elements);
  std::vector<float> kv_initial(kv_elements, 3.0F);
  for (std::size_t index = 0; index < conv_initial.size(); ++index) {
    conv_initial[index] = static_cast<float>(index % 17) * 0.125F;
  }
  for (std::size_t index = 0; index < deltanet_initial.size(); ++index) {
    deltanet_initial[index] = static_cast<float>(index % 23) * 0.0625F;
  }

  HIP_CHECK(hipMemcpy(arena.d_ssm_conv_state, conv_initial.data(),
                      conv_initial.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(arena.d_ssm_deltanet_state, deltanet_initial.data(),
                      deltanet_initial.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(arena.d_kv_cache, kv_initial.data(),
                      kv_initial.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  arena.SaveState(4);

  HIP_CHECK(hipMemset(arena.d_ssm_conv_state, 0,
                      conv_initial.size() * sizeof(float)));
  HIP_CHECK(hipMemset(arena.d_ssm_deltanet_state, 0,
                      deltanet_initial.size() * sizeof(float)));
  HIP_CHECK(hipMemset(arena.d_kv_cache, 0, kv_initial.size() * sizeof(float)));
  arena.RestoreState();

  const auto conv_restored =
      CopyToHost(arena.d_ssm_conv_state, conv_initial.size());
  const auto deltanet_restored =
      CopyToHost(arena.d_ssm_deltanet_state, deltanet_initial.size());
  const auto kv_after_restore = CopyToHost(arena.d_kv_cache, kv_initial.size());
  Expect(conv_restored == conv_initial, "convolution state restore");
  Expect(deltanet_restored == deltanet_initial, "DeltaNet state restore");
  Expect(std::ranges::all_of(kv_after_restore,
                             [](float value) { return value == 0.0F; }),
         "append-only KV entries must not be deep-copied");

  Expect(arena.BeginSsmReplayCapture(), "first replay allocation");
  Expect(!arena.CanReplaySsmPosition(4), "empty replay capture");
  for (std::uint32_t position = 4; position < 20; ++position) {
    arena.MarkSsmReplayPosition(position);
  }
  Expect(arena.CanReplaySsmPosition(4), "16-position replay lower bound");
  Expect(arena.CanReplaySsmPosition(19), "16-position replay upper bound");
  arena.MarkSsmReplayPosition(20);
  Expect(!arena.CanReplaySsmPosition(4), "overflowed replay window fallback");
}

void TestCapturedInputsReplayBitExactly() {
  constexpr std::uint32_t layer = 0;
  constexpr std::uint32_t num_key_heads = 1;
  constexpr std::uint32_t num_heads = 2;
  constexpr std::uint32_t key_dim = 32;
  constexpr std::uint32_t val_dim = 32;
  constexpr std::size_t qkv_size =
      (2 * num_key_heads * key_dim) + (num_heads * val_dim);
  constexpr std::size_t output_size = num_heads * val_dim;
  constexpr std::size_t token_count = 4;
  constexpr std::uint32_t start_position = 14;
  constexpr std::size_t conv_state_size = qkv_size * 4;
  constexpr std::size_t deltanet_state_size = num_heads * key_dim * val_dim;

  std::vector<float> qkv(token_count * qkv_size);
  std::vector<float> weights(qkv_size * 4);
  std::vector<float> alpha(token_count * num_heads);
  std::vector<float> beta(token_count * num_heads);
  std::vector<float> ssm_a(num_heads, -0.05F);
  std::vector<float> ssm_dt(num_heads, 0.01F);
  for (std::size_t index = 0; index < qkv.size(); ++index) {
    qkv[index] = static_cast<float>((index % 31) + 1) * 0.003F;
  }
  for (std::size_t index = 0; index < weights.size(); ++index) {
    weights[index] = static_cast<float>((index % 7) + 1) * 0.01F;
  }
  for (std::size_t index = 0; index < alpha.size(); ++index) {
    alpha[index] = static_cast<float>(index + 1) * 0.002F;
    beta[index] = static_cast<float>(index + 1) * -0.003F;
  }

  HipBuffer<float> d_qkv(qkv.size());
  HipBuffer<float> d_weights(weights.size());
  HipBuffer<float> d_alpha(alpha.size());
  HipBuffer<float> d_beta(beta.size());
  HipBuffer<float> d_ssm_a(ssm_a.size());
  HipBuffer<float> d_ssm_dt(ssm_dt.size());
  HipBuffer<float> d_conv_golden(conv_state_size);
  HipBuffer<float> d_delta_golden(deltanet_state_size);
  HipBuffer<float> d_conv_capture(conv_state_size);
  HipBuffer<float> d_delta_capture(deltanet_state_size);
  HipBuffer<float> d_conv_out(qkv_size);
  HipBuffer<float> d_output(output_size);
  HipBuffer<float> d_log_qkv(strix::hip::kSsmReplayCapacity * qkv_size);
  HipBuffer<float> d_log_alpha(strix::hip::kSsmReplayCapacity * num_heads);
  HipBuffer<float> d_log_beta(strix::hip::kSsmReplayCapacity * num_heads);
  HipBuffer<std::uint32_t> d_position(1);
  HipBuffer<std::uint32_t> d_enabled(1);

  CopyToDevice(d_qkv, qkv);
  CopyToDevice(d_weights, weights);
  CopyToDevice(d_alpha, alpha);
  CopyToDevice(d_beta, beta);
  CopyToDevice(d_ssm_a, ssm_a);
  CopyToDevice(d_ssm_dt, ssm_dt);
  HIP_CHECK(hipMemset(d_conv_golden.Get(), 0, conv_state_size * sizeof(float)));
  HIP_CHECK(
      hipMemset(d_delta_golden.Get(), 0, deltanet_state_size * sizeof(float)));
  HIP_CHECK(
      hipMemset(d_conv_capture.Get(), 0, conv_state_size * sizeof(float)));
  HIP_CHECK(
      hipMemset(d_delta_capture.Get(), 0, deltanet_state_size * sizeof(float)));
  const std::uint32_t enabled = 1;
  HIP_CHECK(hipMemcpy(d_enabled.Get(), &enabled, sizeof(enabled),
                      hipMemcpyHostToDevice));

  for (std::size_t token = 0; token < token_count; ++token) {
    strix::hip::LaunchSSMConvRecurrence(
        d_qkv.Get() + token * qkv_size, d_weights.Get(), d_conv_golden.Get(),
        d_conv_out.Get(), d_delta_golden.Get(),
        d_alpha.Get() + token * num_heads, d_beta.Get() + token * num_heads,
        d_ssm_a.Get(), d_ssm_dt.Get(), nullptr, nullptr, d_output.Get(), layer,
        qkv_size, num_key_heads, num_heads, key_dim, val_dim);
  }

  const strix::hip::SsmReplayCapture capture{
      .qkv = d_log_qkv.Get(),
      .alpha = d_log_alpha.Get(),
      .beta = d_log_beta.Get(),
      .position = d_position.Get(),
      .enabled = d_enabled.Get(),
  };
  for (std::size_t token = 0; token < token_count; ++token) {
    const auto position = start_position + static_cast<std::uint32_t>(token);
    HIP_CHECK(hipMemcpy(d_position.Get(), &position, sizeof(position),
                        hipMemcpyHostToDevice));
    strix::hip::LaunchSSMConvRecurrence(
        d_qkv.Get() + token * qkv_size, d_weights.Get(), d_conv_capture.Get(),
        d_conv_out.Get(), d_delta_capture.Get(),
        d_alpha.Get() + token * num_heads, d_beta.Get() + token * num_heads,
        d_ssm_a.Get(), d_ssm_dt.Get(), nullptr, nullptr, d_output.Get(), layer,
        qkv_size, num_key_heads, num_heads, key_dim, val_dim, nullptr, capture);
  }
  HIP_CHECK(hipDeviceSynchronize());

  Expect(CopyToHost(d_conv_capture.Get(), conv_state_size) ==
             CopyToHost(d_conv_golden.Get(), conv_state_size),
         "captured convolution state");
  Expect(CopyToHost(d_delta_capture.Get(), deltanet_state_size) ==
             CopyToHost(d_delta_golden.Get(), deltanet_state_size),
         "captured DeltaNet state");

  HIP_CHECK(
      hipMemset(d_conv_capture.Get(), 0, conv_state_size * sizeof(float)));
  HIP_CHECK(
      hipMemset(d_delta_capture.Get(), 0, deltanet_state_size * sizeof(float)));
  for (std::size_t token = 0; token < token_count; ++token) {
    const auto position = start_position + static_cast<std::uint32_t>(token);
    const std::size_t slot =
        static_cast<std::size_t>(position) % strix::hip::kSsmReplayCapacity;
    const float* replay_qkv = d_log_qkv.Get() + slot * qkv_size;
    const float* replay_alpha = d_log_alpha.Get() + slot * num_heads;
    const float* replay_beta = d_log_beta.Get() + slot * num_heads;
    Expect(CopyToHost(replay_qkv, qkv_size) ==
               std::vector<float>(qkv.begin() + token * qkv_size,
                                  qkv.begin() + (token + 1) * qkv_size),
           "captured QKV input");
    Expect(CopyToHost(replay_alpha, num_heads) ==
               std::vector<float>(alpha.begin() + token * num_heads,
                                  alpha.begin() + (token + 1) * num_heads),
           "captured alpha input");
    Expect(CopyToHost(replay_beta, num_heads) ==
               std::vector<float>(beta.begin() + token * num_heads,
                                  beta.begin() + (token + 1) * num_heads),
           "captured beta input");
    strix::hip::LaunchSSMConvRecurrence(
        replay_qkv, d_weights.Get(), d_conv_capture.Get(), d_conv_out.Get(),
        d_delta_capture.Get(), replay_alpha, replay_beta, d_ssm_a.Get(),
        d_ssm_dt.Get(), nullptr, nullptr, d_output.Get(), layer, qkv_size,
        num_key_heads, num_heads, key_dim, val_dim);
  }
  HIP_CHECK(hipDeviceSynchronize());

  Expect(CopyToHost(d_conv_capture.Get(), conv_state_size) ==
             CopyToHost(d_conv_golden.Get(), conv_state_size),
         "replayed convolution state");
  Expect(CopyToHost(d_delta_capture.Get(), deltanet_state_size) ==
             CopyToHost(d_delta_golden.Get(), deltanet_state_size),
         "replayed DeltaNet state");
}

}  // namespace

int main() {
  TestArenaRestoresOnlyMutableRecurrentState();
  TestCapturedInputsReplayBitExactly();
  return 0;
}

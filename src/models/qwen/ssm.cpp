#include "src/models/qwen/ssm.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include "src/models/qwen/forward.hpp"

namespace strix::models {

QwenSsmCache::QwenSsmCache(std::uint32_t num_layers, std::size_t conv_channels,
                           std::uint32_t conv_kernel, std::uint32_t num_heads,
                           std::uint32_t key_dim, std::uint32_t val_dim)
    : num_layers_(num_layers),
      conv_channels_(conv_channels),
      conv_kernel_(conv_kernel),
      num_heads_(num_heads),
      key_dim_(key_dim),
      val_dim_(val_dim) {
  const std::size_t total_conv =
      static_cast<std::size_t>(num_layers_) * conv_channels_ * conv_kernel_;
  const std::size_t total_deltanet =
      static_cast<std::size_t>(num_layers_) * num_heads_ * key_dim_ * val_dim_;
  conv_states_.resize(total_conv, 0.0F);
  deltanet_states_.resize(total_deltanet, 0.0F);
}

void QwenSsmCache::Reset() noexcept {
  std::ranges::fill(conv_states_, 0.0F);
  std::ranges::fill(deltanet_states_, 0.0F);
}

std::span<float> QwenSsmCache::GetConvState(std::uint32_t layer) noexcept {
  const std::size_t offset =
      static_cast<std::size_t>(layer) * conv_channels_ * conv_kernel_;
  return {&conv_states_[offset], conv_channels_ * conv_kernel_};
}

std::span<float> QwenSsmCache::GetDeltaNetState(std::uint32_t layer,
                                                std::uint32_t head) noexcept {
  const std::size_t head_size =
      static_cast<std::size_t>(key_dim_) * static_cast<std::size_t>(val_dim_);
  const std::size_t offset =
      (static_cast<std::size_t>(layer) * num_heads_ + head) * head_size;
  return {&deltanet_states_[offset], head_size};
}

namespace {

[[nodiscard]] inline float Sigmoid(float x) noexcept {
  return 1.0F / (1.0F + std::exp(-x));
}

[[nodiscard]] inline float SiLU(float x) noexcept {
  return x * Sigmoid(x);
}

[[nodiscard]] bool CheckedMultiply(std::size_t lhs, std::size_t rhs,
                                   std::size_t& product) noexcept {
  if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
    return false;
  }
  product = lhs * rhs;
  return true;
}

[[nodiscard]] bool HasTensorElements(const QwenTensorRef& tensor,
                                     std::size_t required) noexcept {
  return tensor.empty() || tensor.num_elements >= required;
}

[[nodiscard]] bool ValidateSsmInvocation(
    std::span<const float> x, const QwenSsmParameters& parameters,
    const QwenSsmCache& cache, std::uint32_t layer_idx,
    std::span<float> qkv_scratch, std::span<float> gate_scratch,
    std::span<float> out_scratch, std::span<float> out,
    std::size_t& qkv_dim, std::size_t& gate_dim) noexcept {
  if (x.empty() || parameters.key_head_count == 0 ||
      parameters.value_head_count == 0 || parameters.key_dim == 0 ||
      parameters.val_dim == 0 || parameters.conv_kernel == 0 ||
      layer_idx >= cache.NumLayers()) {
    return false;
  }

  std::size_t key_channels = 0;
  std::size_t value_channels = 0;
  if (!CheckedMultiply(parameters.key_head_count, parameters.key_dim,
                       key_channels) ||
      !CheckedMultiply(parameters.value_head_count, parameters.val_dim,
                       value_channels) ||
      key_channels >
          (std::numeric_limits<std::size_t>::max() - value_channels) / 2U) {
    return false;
  }
  qkv_dim = (2U * key_channels) + value_channels;
  gate_dim = value_channels;

  if (cache.ConvChannels() != qkv_dim ||
      cache.ConvKernel() != parameters.conv_kernel ||
      cache.NumHeads() != parameters.value_head_count ||
      cache.KeyDim() != parameters.key_dim ||
      cache.ValDim() != parameters.val_dim || qkv_scratch.size() < qkv_dim ||
      gate_scratch.size() < gate_dim || out_scratch.size() < gate_dim ||
      out.size() < x.size()) {
    return false;
  }

  std::size_t qkv_projection_elements = 0;
  std::size_t gate_projection_elements = 0;
  std::size_t scalar_projection_elements = 0;
  std::size_t output_projection_elements = 0;
  std::size_t conv_elements = 0;
  if (!CheckedMultiply(qkv_dim, x.size(), qkv_projection_elements) ||
      !CheckedMultiply(gate_dim, x.size(), gate_projection_elements) ||
      !CheckedMultiply(parameters.value_head_count, x.size(),
                       scalar_projection_elements) ||
      !CheckedMultiply(x.size(), gate_dim, output_projection_elements) ||
      !CheckedMultiply(qkv_dim, parameters.conv_kernel, conv_elements)) {
    return false;
  }

  return HasTensorElements(parameters.qkv, qkv_projection_elements) &&
         HasTensorElements(parameters.gate, gate_projection_elements) &&
         HasTensorElements(parameters.a, parameters.value_head_count) &&
         HasTensorElements(parameters.dt, parameters.value_head_count) &&
         HasTensorElements(parameters.alpha, scalar_projection_elements) &&
         HasTensorElements(parameters.beta, scalar_projection_elements) &&
         (parameters.norm.empty() ||
          (parameters.norm.num_elements != 0 &&
           gate_dim % parameters.norm.num_elements == 0)) &&
         HasTensorElements(parameters.output, output_projection_elements) &&
         HasTensorElements(parameters.conv1d, conv_elements);
}

}  // namespace

void ForwardSSM(std::span<const float> x_normed,
                const QwenSsmParameters& parameters,
                QwenSsmCache& ssm_cache, std::uint32_t layer_idx,
                std::span<float> ssm_qkv_scratch,
                std::span<float> ssm_gate_scratch,
                std::span<float> ssm_out_scratch,
                std::span<float> out) noexcept {
  const std::size_t hidden_size = x_normed.size();
  const std::uint32_t num_k_heads = parameters.key_head_count;
  const std::uint32_t num_v_heads = parameters.value_head_count;
  const std::uint32_t key_dim = parameters.key_dim;
  const std::uint32_t val_dim = parameters.val_dim;
  const std::uint32_t conv_kernel = parameters.conv_kernel;
  std::size_t qkv_dim = 0;
  std::size_t gate_dim = 0;

  if (!ValidateSsmInvocation(x_normed, parameters, ssm_cache, layer_idx,
                             ssm_qkv_scratch, ssm_gate_scratch,
                             ssm_out_scratch, out, qkv_dim, gate_dim)) {
    std::ranges::fill(out, 0.0F);
    return;
  }

  // 1. QKV, Gate, Alpha, and Beta Projections
  if (!parameters.qkv.empty()) {
    TensorGEMV(parameters.qkv, x_normed, qkv_dim, hidden_size,
               ssm_qkv_scratch);
  } else {
    std::ranges::fill(ssm_qkv_scratch, 0.0F);
  }

  if (!parameters.gate.empty()) {
    TensorGEMV(parameters.gate, x_normed, gate_dim, hidden_size,
               ssm_gate_scratch);
  } else {
    std::ranges::fill(ssm_gate_scratch, 0.0F);
  }

  std::vector<float> alpha_buf(num_v_heads, 0.0F);
  std::vector<float> beta_buf(num_v_heads, 0.0F);
  if (!parameters.alpha.empty()) {
    TensorGEMV(parameters.alpha, x_normed, num_v_heads, hidden_size,
               alpha_buf);
  }
  if (!parameters.beta.empty()) {
    TensorGEMV(parameters.beta, x_normed, num_v_heads, hidden_size, beta_buf);
  }

  // 2. 1D Causal Convolution with rolling conv state
  auto conv_state = ssm_cache.GetConvState(layer_idx);
  std::vector<float> conv_out(qkv_dim, 0.0F);

#pragma omp parallel for schedule(static)
  for (std::size_t c = 0; c < qkv_dim; ++c) {
    const std::size_t c_off = c * conv_kernel;
    for (std::uint32_t k = 1; k < conv_kernel; ++k) {
      conv_state[c_off + k - 1] = conv_state[c_off + k];
    }
    conv_state[c_off + conv_kernel - 1] = ssm_qkv_scratch[c];

    // Compute convolution
    float dot = 0.0F;
    if (!parameters.conv1d.empty()) {
      for (std::uint32_t k = 0; k < conv_kernel; ++k) {
        dot += conv_state[c_off + k] *
               parameters.conv1d.Get((c * conv_kernel) + k);
      }
    } else {
      dot = ssm_qkv_scratch[c];
    }

    conv_out[c] = SiLU(dot);
  }

  // 3. Partition Q, K, V from conv_out
  const float* q_ptr = conv_out.data();
  const float* k_ptr =
      conv_out.data() + (static_cast<std::size_t>(num_k_heads) * key_dim);
  const float* v_ptr =
      conv_out.data() + (static_cast<std::size_t>(num_k_heads) * key_dim * 2);

  // 4. Per-head Gated DeltaNet matrix recurrence.
#pragma omp parallel for schedule(static)
  for (std::uint32_t h = 0; h < num_v_heads; ++h) {
    auto s_matrix = ssm_cache.GetDeltaNetState(layer_idx, h);
    const std::uint32_t kh_idx = h % num_k_heads;
    const float* q_h = q_ptr + (static_cast<std::size_t>(kh_idx) * key_dim);
    const float* k_h = k_ptr + (static_cast<std::size_t>(kh_idx) * key_dim);
    const float* v_h = v_ptr + (static_cast<std::size_t>(h) * val_dim);
    float* o_h =
        ssm_out_scratch.data() + (static_cast<std::size_t>(h) * val_dim);

    // Compute decay alpha_h = exp(ssm_a * softplus(alpha + ssm_dt)) and beta_h
    // = sigmoid(beta)
    float dt = 0.0F;
    if (!parameters.dt.empty()) {
      dt = parameters.dt.Get(h);
    }
    float a_val = -0.05F;
    if (!parameters.a.empty()) {
      a_val = parameters.a.Get(h);
    }
    const float alpha_biased = alpha_buf[h] + dt;
    const float alpha_softplus = (alpha_biased > 20.0F)
                                     ? alpha_biased
                                     : std::log1p(std::exp(alpha_biased));
    const float alpha_h = std::exp(alpha_softplus * a_val);
    const float beta_h = Sigmoid(beta_buf[h]);

    // Normalize query q_h and key k_h, and scale query by 1/sqrt(key_dim)
    float q_norm_sq = 0.0F;
    float k_norm_sq = 0.0F;
    for (std::uint32_t i = 0; i < key_dim; ++i) {
      q_norm_sq += q_h[i] * q_h[i];
      k_norm_sq += k_h[i] * k_h[i];
    }
    const float q_scale = (1.0F / std::sqrt(static_cast<float>(key_dim))) /
                          std::sqrt(q_norm_sq + 1e-6F);
    const float inv_k_norm = 1.0F / std::sqrt(k_norm_sq + 1e-6F);

    // 1. Decay state: S = alpha * S
    for (std::size_t idx = 0; idx < static_cast<std::size_t>(val_dim) * key_dim;
         ++idx) {
      s_matrix[idx] *= alpha_h;
    }

    // 2. Associative retrieval: u_j = sum_i S[j, i] * (k[i] * inv_k_norm)
    std::vector<float> d_h(val_dim, 0.0F);
    for (std::uint32_t j = 0; j < val_dim; ++j) {
      float u_j = 0.0F;
      const float* s_row = &s_matrix[static_cast<std::size_t>(j) * key_dim];
      for (std::uint32_t i = 0; i < key_dim; ++i) {
        u_j += s_row[i] * (k_h[i] * inv_k_norm);
      }
      d_h[j] = (v_h[j] - u_j) * beta_h;
    }

    // 3. Delta update: S[j, i] += d[j] * (k[i] * inv_k_norm)
    for (std::uint32_t j = 0; j < val_dim; ++j) {
      float* s_row = &s_matrix[static_cast<std::size_t>(j) * key_dim];
      const float d_val = d_h[j];
      for (std::uint32_t i = 0; i < key_dim; ++i) {
        s_row[i] += d_val * (k_h[i] * inv_k_norm);
      }
    }

    // 4. Query readout: o_j = sum_i S[j, i] * (q[i] * q_scale)
    for (std::uint32_t j = 0; j < val_dim; ++j) {
      float o_j = 0.0F;
      const float* s_row = &s_matrix[static_cast<std::size_t>(j) * key_dim];
      for (std::uint32_t i = 0; i < key_dim; ++i) {
        o_j += s_row[i] * (q_h[i] * q_scale);
      }
      o_h[j] = o_j;
    }
  }

  // 5. Per-Head Output RMSNorm (128 elements per head norm)
  if (!parameters.norm.empty()) {
    const std::size_t norm_dim = parameters.norm.num_elements;
    const std::size_t num_norm_heads = gate_dim / norm_dim;
    for (std::size_t h = 0; h < num_norm_heads; ++h) {
      auto slice = ssm_out_scratch.subspan(h * norm_dim, norm_dim);
      ForwardRMSNorm(slice, parameters.norm, 1e-6F, slice);
    }
  }

  // 6. Gating: y = o * SiLU(gate)
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < gate_dim; ++i) {
    ssm_out_scratch[i] *= SiLU(ssm_gate_scratch[i]);
  }

  // 7. Linear Output Projection
  if (!parameters.output.empty()) {
    TensorGEMV(parameters.output, ssm_out_scratch.subspan(0, gate_dim),
               hidden_size, gate_dim, out);
  } else {
    std::ranges::fill(out, 0.0F);
  }
}

void ForwardSSM(std::span<const float> x_normed,
                const QwenLayerWeights& layer,
                const core::ModelConfig& config, QwenSsmCache& ssm_cache,
                std::uint32_t layer_idx, std::span<float> ssm_qkv_scratch,
                std::span<float> ssm_gate_scratch,
                std::span<float> ssm_out_scratch,
                std::span<float> out) noexcept {
  ForwardSSM(x_normed, MakeQwenSsmParameters(layer, config), ssm_cache,
             layer_idx, ssm_qkv_scratch, ssm_gate_scratch, ssm_out_scratch,
             out);
}

}  // namespace strix::models

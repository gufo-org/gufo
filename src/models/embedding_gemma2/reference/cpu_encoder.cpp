#include "reference/cpu_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

#include "loader.hpp"

namespace gufo::models::embedding_gemma2::reference {
namespace {

float Bf16ToFloat(const std::byte* bytes) {
  std::uint16_t half = 0;
  std::memcpy(&half, bytes, sizeof(half));
  const std::uint32_t bits = static_cast<std::uint32_t>(half) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::vector<float> ToFloat(const Tensor& tensor) {
  const std::size_t count = *tensor.NumElements();
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    values[index] = Bf16ToFloat(tensor.data + index * 2U);
  }
  return values;
}

std::vector<float> Weight(const TensorStore& store, const std::string& name) {
  const Tensor* tensor = store.Find(name);
  if (tensor == nullptr) {
    throw std::runtime_error("embedding_gemma2 reference: missing " + name);
  }
  return ToFloat(*tensor);
}

float GeluTanh(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654F;
  const float inner = kSqrt2OverPi * (x + 0.044715F * x * x * x);
  return 0.5F * x * (1.0F + std::tanh(inner));
}

/// y[t, o] = sum_i x[t, i] * w[o, i]
std::vector<float> Linear(const std::vector<float>& x, std::size_t tokens,
                          std::size_t in, const std::vector<float>& w,
                          std::size_t out) {
  std::vector<float> y(tokens * out);
  for (std::size_t t = 0; t < tokens; ++t) {
    const float* row = x.data() + t * in;
    for (std::size_t o = 0; o < out; ++o) {
      const float* weights = w.data() + o * in;
      float sum = 0.0F;
      for (std::size_t i = 0; i < in; ++i) {
        sum += row[i] * weights[i];
      }
      y[t * out + o] = sum;
    }
  }
  return y;
}

/// Normalizes each `width`-wide row of `x` in place. A null `weight` is the
/// unscaled norm the value projection uses.
void RmsNorm(float* x, std::size_t rows, std::size_t width, const float* weight,
             float eps) {
  for (std::size_t r = 0; r < rows; ++r) {
    float* row = x + r * width;
    float squares = 0.0F;
    for (std::size_t i = 0; i < width; ++i) {
      squares += row[i] * row[i];
    }
    const float scale =
        std::pow(squares / static_cast<float>(width) + eps, -0.5F);
    for (std::size_t i = 0; i < width; ++i) {
      row[i] = row[i] * scale * (weight != nullptr ? weight[i] : 1.0F);
    }
  }
}

/// Rotates the two halves of each head (rotate_half convention).
void Rope(float* x, std::size_t tokens, std::size_t heads, std::size_t head_dim,
          float theta) {
  const std::size_t half = head_dim / 2U;
  std::vector<float> inverse(half);
  for (std::size_t i = 0; i < half; ++i) {
    inverse[i] = 1.0F / std::pow(theta, static_cast<float>(2U * i) /
                                            static_cast<float>(head_dim));
  }
  for (std::size_t t = 0; t < tokens; ++t) {
    for (std::size_t h = 0; h < heads; ++h) {
      float* head = x + (t * heads + h) * head_dim;
      for (std::size_t i = 0; i < half; ++i) {
        const float angle = static_cast<float>(t) * inverse[i];
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        const float a = head[i];
        const float b = head[i + half];
        head[i] = a * c - b * s;
        head[i + half] = b * c + a * s;
      }
    }
  }
}

void L2Normalize(std::vector<float>& values) {
  float squares = 0.0F;
  for (const float value : values) {
    squares += value * value;
  }
  const float norm = std::sqrt(squares);
  if (norm > 0.0F) {
    for (float& value : values) {
      value /= norm;
    }
  }
}

}  // namespace

CpuEncoder::CpuEncoder(const TextConfig& config, const TensorStore& store)
    : config_(config), store_(store) {}

std::vector<float> CpuEncoder::EncodeTokens(
    std::span<const std::uint32_t> token_ids) const {
  const std::size_t tokens = token_ids.size();
  const std::size_t hidden = config_.hidden_size;
  const std::size_t per_layer = config_.hidden_size_per_layer_input;
  const std::size_t layers = config_.num_hidden_layers;
  const std::string prefix = kTextTensorPrefix;
  const float eps = config_.rms_norm_eps;
  if (tokens == 0) {
    return {};
  }

  const Tensor* table = store_.Find(prefix + "embed_tokens.weight");
  if (table == nullptr) {
    throw std::runtime_error("embedding_gemma2 reference: missing embeddings");
  }
  std::vector<float> state(tokens * hidden);
  for (std::size_t t = 0; t < tokens; ++t) {
    if (token_ids[t] >= config_.vocab_size) {
      throw std::out_of_range("embedding_gemma2 reference: token id");
    }
    for (std::size_t i = 0; i < hidden; ++i) {
      state[t * hidden + i] =
          Bf16ToFloat(table->data + (token_ids[t] * hidden + i) * 2U) *
          kEmbedScale;
    }
  }

  // Per-layer inputs derive from the scaled embeddings alone.
  std::vector<float> ple =
      Linear(state, tokens, hidden,
             Weight(store_, prefix + "ple.per_layer_model_projection.weight"),
             layers * per_layer);
  const float ple_scale = 1.0F / std::sqrt(static_cast<float>(hidden));
  for (float& value : ple) {
    value *= ple_scale;
  }
  RmsNorm(
      ple.data(), tokens * layers, per_layer,
      Weight(store_, prefix + "ple.per_layer_projection_norm.weight").data(),
      eps);

  for (std::size_t l = 0; l < layers; ++l) {
    const LayerGeometry& layer = config_.layers[l];
    const std::string p = prefix + "layers." + std::to_string(l) + ".";
    const std::size_t heads = config_.num_attention_heads;
    const std::size_t kv_heads = layer.num_key_value_heads;
    const std::size_t head_dim = layer.head_dim;

    // Attention block.
    std::vector<float> normed = state;
    RmsNorm(normed.data(), tokens, hidden,
            Weight(store_, p + "input_layernorm.weight").data(), eps);
    std::vector<float> q =
        Linear(normed, tokens, hidden,
               Weight(store_, p + "self_attn.q_proj.weight"), heads * head_dim);
    std::vector<float> k = Linear(normed, tokens, hidden,
                                  Weight(store_, p + "self_attn.k_proj.weight"),
                                  kv_heads * head_dim);
    std::vector<float> v = Linear(normed, tokens, hidden,
                                  Weight(store_, p + "self_attn.v_proj.weight"),
                                  kv_heads * head_dim);
    RmsNorm(q.data(), tokens * heads, head_dim,
            Weight(store_, p + "self_attn.q_norm.weight").data(), eps);
    RmsNorm(k.data(), tokens * kv_heads, head_dim,
            Weight(store_, p + "self_attn.k_norm.weight").data(), eps);
    RmsNorm(v.data(), tokens * kv_heads, head_dim, nullptr, eps);
    Rope(q.data(), tokens, heads, head_dim, layer.rope_theta);
    Rope(k.data(), tokens, kv_heads, head_dim, layer.rope_theta);

    // The attention scale is 1.0, and the query and key norms replace it.
    const std::size_t group = heads / kv_heads;
    const bool windowed = layer.type == LayerType::kSlidingAttention;
    std::vector<float> context(tokens * heads * head_dim, 0.0F);
    std::vector<float> scores(tokens);
    for (std::size_t t = 0; t < tokens; ++t) {
      for (std::size_t h = 0; h < heads; ++h) {
        const float* query = q.data() + (t * heads + h) * head_dim;
        const std::size_t kv = h / group;
        float maximum = -INFINITY;
        for (std::size_t s = 0; s < tokens; ++s) {
          const std::size_t distance = s > t ? s - t : t - s;
          if (windowed && distance > config_.sliding_window) {
            scores[s] = -INFINITY;
            continue;
          }
          const float* key = k.data() + (s * kv_heads + kv) * head_dim;
          float dot = 0.0F;
          for (std::size_t i = 0; i < head_dim; ++i) {
            dot += query[i] * key[i];
          }
          scores[s] = dot;
          maximum = std::max(maximum, dot);
        }
        float total = 0.0F;
        for (std::size_t s = 0; s < tokens; ++s) {
          scores[s] =
              std::isinf(scores[s]) ? 0.0F : std::exp(scores[s] - maximum);
          total += scores[s];
        }
        float* out = context.data() + (t * heads + h) * head_dim;
        for (std::size_t s = 0; s < tokens; ++s) {
          if (scores[s] == 0.0F) {
            continue;
          }
          const float weight = scores[s] / total;
          const float* value = v.data() + (s * kv_heads + kv) * head_dim;
          for (std::size_t i = 0; i < head_dim; ++i) {
            out[i] += weight * value[i];
          }
        }
      }
    }
    std::vector<float> attended =
        Linear(context, tokens, heads * head_dim,
               Weight(store_, p + "self_attn.o_proj.weight"), hidden);
    RmsNorm(attended.data(), tokens, hidden,
            Weight(store_, p + "post_attention_layernorm.weight").data(), eps);
    for (std::size_t i = 0; i < state.size(); ++i) {
      state[i] += attended[i];
    }

    // Feed-forward block.
    normed = state;
    RmsNorm(normed.data(), tokens, hidden,
            Weight(store_, p + "pre_feedforward_layernorm.weight").data(), eps);
    const std::size_t inner = config_.intermediate_size;
    std::vector<float> gate =
        Linear(normed, tokens, hidden,
               Weight(store_, p + "mlp.gate_proj.weight"), inner);
    const std::vector<float> up =
        Linear(normed, tokens, hidden, Weight(store_, p + "mlp.up_proj.weight"),
               inner);
    for (std::size_t i = 0; i < gate.size(); ++i) {
      gate[i] = GeluTanh(gate[i]) * up[i];
    }
    std::vector<float> down =
        Linear(gate, tokens, inner, Weight(store_, p + "mlp.down_proj.weight"),
               hidden);
    RmsNorm(down.data(), tokens, hidden,
            Weight(store_, p + "post_feedforward_layernorm.weight").data(),
            eps);
    for (std::size_t i = 0; i < state.size(); ++i) {
      state[i] += down[i];
    }

    // Per-layer input block. It gates a bottleneck through the layer's slice
    // of the per-layer inputs.
    std::vector<float> gated = Linear(
        state, tokens, hidden,
        Weight(store_, p + "ple_block.per_layer_input_gate.weight"), per_layer);
    for (std::size_t t = 0; t < tokens; ++t) {
      for (std::size_t i = 0; i < per_layer; ++i) {
        gated[t * per_layer + i] = GeluTanh(gated[t * per_layer + i]) *
                                   ple[(t * layers + l) * per_layer + i];
      }
    }
    std::vector<float> mixed = Linear(
        gated, tokens, per_layer,
        Weight(store_, p + "ple_block.per_layer_projection.weight"), hidden);
    RmsNorm(
        mixed.data(), tokens, hidden,
        Weight(store_, p + "ple_block.post_per_layer_input_norm.weight").data(),
        eps);
    const float scalar = Weight(store_, p + "layer_scalar")[0];
    for (std::size_t i = 0; i < state.size(); ++i) {
      state[i] = (state[i] + mixed[i]) * scalar;
    }
  }

  RmsNorm(state.data(), tokens, hidden,
          Weight(store_, prefix + "norm.weight").data(), eps);
  return Linear(state, tokens, hidden,
                Weight(store_, prefix + "embedding_projection.weight"),
                config_.embedding_dim);
}

std::vector<float> CpuEncoder::Embed(
    std::span<const std::uint32_t> token_ids) const {
  const std::vector<float> states = EncodeTokens(token_ids);
  const std::size_t width = config_.embedding_dim;
  std::vector<float> pooled(width, 0.0F);
  for (std::size_t t = 0; t < token_ids.size(); ++t) {
    for (std::size_t i = 0; i < width; ++i) {
      pooled[i] += states[t * width + i];
    }
  }
  L2Normalize(pooled);
  return pooled;
}

}  // namespace gufo::models::embedding_gemma2::reference

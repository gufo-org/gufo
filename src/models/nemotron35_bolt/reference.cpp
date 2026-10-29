#include "src/models/nemotron35_bolt/reference.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "src/models/nemotron35_bolt/cpu_ops.hpp"

namespace gufo::models::nemotron35_bolt {
namespace {

// llama.cpp clamps the routing-weight sum from below before normalizing.
constexpr float kMinRoutingWeightSum = 6.103515625e-5F;

const float* F32(const TensorRef& t) {
  return static_cast<const float*>(t.data);
}

}  // namespace

ReferenceModel::ReferenceModel(const ModelWeights& weights,
                               std::uint32_t max_context)
    : w_(weights),
      c_(weights.config),
      max_context_(max_context),
      mamba_(c_.num_layers),
      attention_(c_.num_layers) {
  Reset();
}

void ReferenceModel::Reset() {
  position_ = 0;
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    if (c_.blocks[il] == BlockType::kMamba) {
      auto& s = mamba_[il];
      s.conv.assign(static_cast<std::size_t>(c_.ssm_conv_kernel - 1) *
                        c_.SsmConvChannels(),
                    0.0F);
      s.ssm.assign(static_cast<std::size_t>(c_.ssm_heads) * c_.SsmHeadDim() *
                       c_.ssm_state_size,
                   0.0F);
    } else if (c_.blocks[il] == BlockType::kAttention) {
      attention_[il].k.clear();
      attention_[il].v.clear();
    }
  }
}

bool ReferenceModel::Step(std::int32_t token, std::span<float> logits,
                          std::string* error_msg) {
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size) {
    if (error_msg != nullptr)
      *error_msg = "token outside the vocabulary";
    return false;
  }
  if (position_ >= max_context_) {
    if (error_msg != nullptr)
      *error_msg = "reference context is full";
    return false;
  }
  if (!logits.empty() && logits.size() != c_.vocab_size) {
    if (error_msg != nullptr)
      *error_msg = "logits span must hold the vocabulary";
    return false;
  }

  const std::size_t H = c_.hidden_size;
  std::vector<float> x(H);
  cpu::DequantizeRow(w_.token_embd, 0, static_cast<std::uint64_t>(token),
                     x.data());
  std::vector<float> h(H);
  std::vector<float> out(H);
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    const LayerWeights& l = w_.layers[il];
    std::copy(x.begin(), x.end(), h.begin());
    cpu::RmsNorm(h, F32(l.norm), c_.rms_eps);
    switch (l.type) {
      case BlockType::kMamba:
        Mamba(l, mamba_[il], h, out);
        break;
      case BlockType::kAttention:
        Attention(l, attention_[il], h, out);
        break;
      case BlockType::kMoe:
        Moe(l, il, h, out);
        break;
    }
    for (std::size_t i = 0; i < H; ++i) {
      x[i] += out[i];
    }
  }
  ++position_;

  if (!logits.empty()) {
    cpu::RmsNorm(x, F32(w_.output_norm), c_.rms_eps);
    cpu::MatVec(w_.output, 0, x, logits);
  }
  return true;
}

void ReferenceModel::Mamba(const LayerWeights& l, MambaState& s,
                           std::span<const float> x, std::span<float> out) {
  const std::size_t inner = c_.ssm_inner_size;
  const std::size_t channels = c_.SsmConvChannels();
  const std::size_t heads = c_.ssm_heads;
  const std::size_t head_dim = c_.SsmHeadDim();
  const std::size_t state = c_.ssm_state_size;
  const std::size_t groups = c_.ssm_groups;
  const std::size_t kernel = c_.ssm_conv_kernel;

  std::vector<float> zxbcdt(c_.SsmInProjection());
  cpu::MatVec(l.ssm_in, 0, x, zxbcdt);
  const float* z = zxbcdt.data();
  const float* xbc = z + inner;
  const float* dt = xbc + channels;

  // Causal depthwise convolution over the kernel-1 previous raw inputs and
  // this one; tap 0 multiplies the oldest input. The state keeps raw inputs.
  std::vector<float> conv(channels);
  const float* w = F32(l.ssm_conv1d);
  const float* bias = F32(l.ssm_conv1d_b);
  for (std::size_t ch = 0; ch < channels; ++ch) {
    float acc = 0.0F;
    for (std::size_t k = 0; k + 1 < kernel; ++k) {
      acc += s.conv[k * channels + ch] * w[ch * kernel + k];
    }
    acc += xbc[ch] * w[ch * kernel + kernel - 1];
    conv[ch] = cpu::Silu(acc + bias[ch]);
  }
  for (std::size_t k = 0; k + 2 < kernel; ++k) {
    std::copy_n(
        s.conv.begin() + static_cast<std::ptrdiff_t>((k + 1) * channels),
        channels, s.conv.begin() + static_cast<std::ptrdiff_t>(k * channels));
  }
  std::copy_n(
      xbc, channels,
      s.conv.begin() + static_cast<std::ptrdiff_t>((kernel - 2) * channels));

  // Selective scan with a scalar decay per head. Heads map onto B/C groups
  // as repeat_interleave: group = head / (heads / groups).
  const float* xs = conv.data();
  const float* B = xs + inner;
  const float* C = B + groups * state;
  const float* dt_b = F32(l.ssm_dt_b);
  const float* A = F32(l.ssm_a);
  const float* D = F32(l.ssm_d);
  std::vector<float> y(inner);
  for (std::size_t hh = 0; hh < heads; ++hh) {
    const float dt_sp = cpu::Softplus(dt[hh] + dt_b[hh]);
    const float dA = std::exp(dt_sp * A[hh]);
    const std::size_t g = hh / (heads / groups);
    const float* Bg = B + g * state;
    const float* Cg = C + g * state;
    for (std::size_t i = 0; i < head_dim; ++i) {
      const std::size_t ii = hh * head_dim + i;
      const float x_dt = xs[ii] * dt_sp;
      float* st = s.ssm.data() + ii * state;
      double sum = 0.0;
      for (std::size_t n = 0; n < state; ++n) {
        st[n] = st[n] * dA + Bg[n] * x_dt;
        sum += static_cast<double>(st[n]) * Cg[n];
      }
      y[ii] = static_cast<float>(sum) + xs[ii] * D[hh];
    }
  }

  // Gate with silu(z), then RMS-normalize each group of inner/groups.
  for (std::size_t i = 0; i < inner; ++i) {
    y[i] *= cpu::Silu(z[i]);
  }
  const std::size_t group_size = inner / groups;
  const float* norm = F32(l.ssm_norm);
  for (std::size_t g = 0; g < groups; ++g) {
    cpu::RmsNorm(std::span<float>(y).subspan(g * group_size, group_size),
                 norm + g * group_size, c_.rms_eps);
  }
  cpu::MatVec(l.ssm_out, 0, y, out);
}

void ReferenceModel::Attention(const LayerWeights& l, AttentionState& s,
                               std::span<const float> x, std::span<float> out) {
  const std::size_t heads = c_.num_heads;
  const std::size_t kv_heads = c_.num_kv_heads;
  const std::size_t hd = c_.head_dim;
  const std::size_t kv_dim = kv_heads * hd;

  std::vector<float> q(heads * hd);
  std::vector<float> k(kv_dim);
  std::vector<float> v(kv_dim);
  cpu::MatVec(l.attn_q, 0, x, q);
  cpu::MatVec(l.attn_k, 0, x, k);
  cpu::MatVec(l.attn_v, 0, x, v);
  s.k.insert(s.k.end(), k.begin(), k.end());
  s.v.insert(s.v.end(), v.begin(), v.end());

  // No positional encoding: position enters only through causality and the
  // Mamba state.
  const std::size_t positions = s.k.size() / kv_dim;
  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  std::vector<float> ctx(heads * hd, 0.0F);
  std::vector<double> scores(positions);
  for (std::size_t hh = 0; hh < heads; ++hh) {
    const std::size_t kvh = hh / (heads / kv_heads);
    const float* qh = q.data() + hh * hd;
    double max_score = -INFINITY;
    for (std::size_t t = 0; t < positions; ++t) {
      const float* kt = s.k.data() + t * kv_dim + kvh * hd;
      double dot = 0.0;
      for (std::size_t i = 0; i < hd; ++i) {
        dot += static_cast<double>(qh[i]) * kt[i];
      }
      scores[t] = dot * scale;
      max_score = std::max(max_score, scores[t]);
    }
    double denom = 0.0;
    for (auto& sc : scores) {
      sc = std::exp(sc - max_score);
      denom += sc;
    }
    float* ch = ctx.data() + hh * hd;
    for (std::size_t i = 0; i < hd; ++i) {
      double acc = 0.0;
      for (std::size_t t = 0; t < positions; ++t) {
        acc += scores[t] * s.v[t * kv_dim + kvh * hd + i];
      }
      ch[i] = static_cast<float>(acc / denom);
    }
  }
  cpu::MatVec(l.attn_out, 0, ctx, out);
}

void ReferenceModel::Moe(const LayerWeights& l, std::uint32_t il,
                         std::span<const float> x, std::span<float> out) {
  const std::size_t n = l.num_experts;
  const std::size_t used = c_.experts_used[il];
  const std::size_t ff = c_.expert_ff[il];
  const std::size_t latent = c_.latent_size;

  // Sigmoid scores; the bias only steers selection, weights use raw scores.
  std::vector<float> probs(n);
  cpu::MatVec(l.router, 0, x, probs);
  for (float& p : probs) {
    p = cpu::Sigmoid(p);
  }
  const float* bias = F32(l.router_bias);
  std::vector<std::uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0U);
  std::partial_sort(order.begin(),
                    order.begin() + static_cast<std::ptrdiff_t>(used),
                    order.end(), [&](std::uint32_t a, std::uint32_t b) {
                      const float sa = probs[a] + bias[a];
                      const float sb = probs[b] + bias[b];
                      return sa > sb || (sa == sb && a < b);
                    });
  std::vector<float> weights(used);
  float sum = 0.0F;
  for (std::size_t i = 0; i < used; ++i) {
    weights[i] = probs[order[i]];
    sum += weights[i];
  }
  for (float& wgt : weights) {
    if (c_.expert_weights_norm) {
      wgt /= std::max(sum, kMinRoutingWeightSum);
    }
    wgt *= c_.expert_weights_scale;
  }

  // Routed experts run in the latent width; their weighted sum is projected
  // back once.
  std::vector<float> lat(latent);
  cpu::MatVec(l.latent_down, 0, x, lat);
  std::vector<float> acc(latent, 0.0F);
  std::vector<float> up(ff);
  std::vector<float> down(latent);
  for (std::size_t i = 0; i < used; ++i) {
    cpu::MatVec(l.up_exps, order[i], lat, up);
    for (float& u : up) {
      u = cpu::Relu2(u);
    }
    cpu::MatVec(l.down_exps, order[i], up, down);
    for (std::size_t j = 0; j < latent; ++j) {
      acc[j] += weights[i] * down[j];
    }
  }
  cpu::MatVec(l.latent_up, 0, acc, out);

  std::vector<float> shared_up(c_.shared_expert_ff);
  std::vector<float> shared(c_.hidden_size);
  cpu::MatVec(l.up_shexp, 0, x, shared_up);
  for (float& u : shared_up) {
    u = cpu::Relu2(u);
  }
  cpu::MatVec(l.down_shexp, 0, shared_up, shared);
  for (std::size_t j = 0; j < out.size(); ++j) {
    out[j] += shared[j];
  }
}

}  // namespace gufo::models::nemotron35_bolt

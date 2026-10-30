#include "src/models/lfm3/reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace gufo::models::lfm3 {
namespace {
std::vector<float> Mat(const Tensor& t, std::span<const float> x) {
  std::vector<float> y(t.rows), row(t.cols);
  for (std::uint32_t r = 0; r < t.rows; ++r) {
    DecodeRow(t, r, row);
    float sum = 0;
    for (std::uint32_t k = 0; k < t.cols; ++k)
      sum += row[k] * x[k];
    y[r] = sum;
  }
  return y;
}
void Norm(std::span<float> x, const Tensor& w, float eps) {
  float sum = 0;
  for (float a : x)
    sum += a * a;
  float scale = 1 / std::sqrt(sum / x.size() + eps);
  auto* weight = static_cast<const float*>(w.data);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = x[i] * scale * weight[i];
}
void Rope(std::span<float> x, const Config& c, std::uint32_t position, bool swa,
          const Tensor& factors) {
  const auto D = c.head_dim;
  const float theta = swa ? c.theta_swa : c.theta;
  for (std::size_t h = 0; h < x.size(); h += D) {
    for (std::uint32_t i = 0; i < D / 2; ++i) {
      float angle = position * std::pow(theta, -2.0F * i / D);
      if (!swa)
        angle /= static_cast<const float*>(factors.data)[i];
      float a = x[h + i], b = x[h + i + D / 2];
      x[h + i] = a * std::cos(angle) - b * std::sin(angle);
      x[h + i + D / 2] = b * std::cos(angle) + a * std::sin(angle);
    }
  }
}
}  // namespace
Reference::Reference(const Weights& w, std::uint32_t context)
    : w_(w), context_(context) {
  const auto& c = w.config;
  if (!context || context > c.context)
    throw std::invalid_argument("invalid LFM3 context");
  states_.resize(c.layers.size());
  for (std::size_t i = 0; i < states_.size(); ++i) {
    auto& s = states_[i];
    if (c.layers[i] == LayerType::kConvolution)
      s.conv.resize(c.hidden * (c.conv_width - 1));
    else {
      s.capacity = c.layers[i] == LayerType::kSlidingAttention
                       ? std::min(context, c.window)
                       : context;
      s.k.resize(std::size_t(s.capacity) * c.kv_heads[i] * c.head_dim);
      s.v.resize(s.k.size());
    }
  }
}
void Reference::Reset() {
  position_ = 0;
  for (auto& s : states_) {
    std::fill(s.conv.begin(), s.conv.end(), 0);
    std::fill(s.k.begin(), s.k.end(), 0);
    std::fill(s.v.begin(), s.v.end(), 0);
  }
}
void Reference::Step(std::uint32_t token, std::span<float> logits) {
  const auto& c = w_.config;
  if (token >= c.vocab || position_ >= context_ ||
      (!logits.empty() && logits.size() != c.vocab))
    throw std::invalid_argument("invalid LFM3 step");
  std::vector<float> x(c.hidden);
  DecodeRow(w_.embedding, token, x);
  for (std::size_t il = 0; il < c.layers.size(); ++il) {
    const auto& l = w_.layers[il];
    auto& state = states_[il];
    auto z = x;
    Norm(z, l.norm, c.eps);
    std::vector<float> out;
    if (c.layers[il] == LayerType::kConvolution) {
      auto p = Mat(l.in_proj, z);
      std::vector<float> conv(c.hidden);
      const auto* weight = static_cast<const float*>(l.conv.data);
      for (std::uint32_t ch = 0; ch < c.hidden; ++ch) {
        auto* history = state.conv.data() + ch * (c.conv_width - 1);
        float bx = p[ch] * p[2 * c.hidden + ch], sum = 0;
        for (std::uint32_t t = 0; t < c.conv_width; ++t)
          sum += weight[ch * c.conv_width + t] *
                 (t + 1 == c.conv_width ? bx : history[t]);
        for (std::uint32_t t = 1; t < c.conv_width - 1; ++t)
          history[t - 1] = history[t];
        if (c.conv_width > 1)
          history[c.conv_width - 2] = bx;
        conv[ch] = sum * p[c.hidden + ch];
      }
      out = Mat(l.out_proj, conv);
    } else {
      auto q = Mat(l.q, z), k = Mat(l.k, z), v = Mat(l.v, z),
           gate = Mat(l.attn_gate, z);
      for (std::uint32_t h = 0; h < c.heads; ++h)
        Norm(std::span(q).subspan(h * c.head_dim, c.head_dim), l.q_norm, c.eps);
      for (std::uint32_t h = 0; h < c.kv_heads[il]; ++h)
        Norm(std::span(k).subspan(h * c.head_dim, c.head_dim), l.k_norm, c.eps);
      const bool swa = c.layers[il] == LayerType::kSlidingAttention;
      Rope(q, c, position_, swa, w_.rope_freqs);
      Rope(k, c, position_, swa, w_.rope_freqs);
      const auto stride = k.size();
      const auto slot = position_ % state.capacity;
      std::copy(k.begin(), k.end(), state.k.begin() + slot * stride);
      std::copy(v.begin(), v.end(), state.v.begin() + slot * stride);
      std::vector<float> a(q.size());
      const auto first =
          swa && position_ + 1 > c.window ? position_ + 1 - c.window : 0;
      for (std::uint32_t h = 0; h < c.heads; ++h) {
        const auto kh = h / (c.heads / c.kv_heads[il]);
        std::vector<float> scores(position_ + 1 - first);
        float max = -std::numeric_limits<float>::infinity();
        for (std::uint32_t t = first; t <= position_; ++t) {
          float score = 0;
          for (std::uint32_t d = 0; d < c.head_dim; ++d)
            score +=
                q[h * c.head_dim + d] *
                state.k[(t % state.capacity) * stride + kh * c.head_dim + d];
          scores[t - first] = score / std::sqrt(float(c.head_dim));
          max = std::max(max, scores[t - first]);
        }
        float den = 0;
        for (auto& score : scores) {
          score = std::exp(score - max);
          den += score;
        }
        for (std::uint32_t d = 0; d < c.head_dim; ++d) {
          float sum = 0;
          for (std::uint32_t t = first; t <= position_; ++t)
            sum += (scores[t - first] / den) *
                   state.v[(t % state.capacity) * stride + kh * c.head_dim + d];
          a[h * c.head_dim + d] = sum / (1 + std::exp(-gate[h]));
        }
      }
      out = Mat(l.attn_out, a);
    }
    if (l.post_norm.data)
      Norm(out, l.post_norm, c.eps);
    for (std::uint32_t i = 0; i < c.hidden; ++i)
      x[i] += out[i];
    z = x;
    Norm(z, l.ffn_norm, c.eps);
    auto gate = Mat(l.gate, z), up = Mat(l.up, z);
    for (std::uint32_t i = 0; i < c.intermediate; ++i)
      gate[i] = (gate[i] / (1 + std::exp(-gate[i]))) * up[i];
    out = Mat(l.down, gate);
    if (l.post_ffn_norm.data)
      Norm(out, l.post_ffn_norm, c.eps);
    for (std::uint32_t i = 0; i < c.hidden; ++i)
      x[i] += out[i];
  }
  if (!logits.empty()) {
    Norm(x, w_.output_norm, c.eps);
    auto y = Mat(w_.output, x);
    std::copy(y.begin(), y.end(), logits.begin());
  }
  ++position_;
}
}  // namespace gufo::models::lfm3

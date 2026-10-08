#include "src/models/gemma4/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "src/core/quant/ggml_gemm.hpp"

namespace gufo::models::gemma4 {
namespace {

float RoundHalf(float x) {
  // Round-to-nearest-even through binary16, including subnormals.
  return static_cast<float>(static_cast<_Float16>(x));
}

/// y = x * rsqrt(mean(x^2) + eps) * w; `w` null means an unweighted norm.
void RmsNorm(const float* x, const float* w, std::size_t n, float eps,
             float* y) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += static_cast<double>(x[i]) * x[i];
  }
  const double scale = 1.0 / std::sqrt(sum / static_cast<double>(n) + eps);
  for (std::size_t i = 0; i < n; ++i) {
    const double v = x[i] * scale;
    y[i] = static_cast<float>(w != nullptr ? v * w[i] : v);
  }
}

const float* Vec(const TensorRef& t) {
  return static_cast<const float*>(t.data);
}

/// NEOX rotation of one head: pair (i, i + dim/2) turns by
/// pos * theta^(-2i/dim) / factor[i].
void Rope(float* head, std::uint32_t dim, std::uint32_t position, float theta,
          const float* factors) {
  const std::uint32_t half = dim / 2;
  for (std::uint32_t i = 0; i < half; ++i) {
    double angle = static_cast<double>(position) *
                   std::pow(static_cast<double>(theta),
                            -2.0 * static_cast<double>(i) / dim);
    if (factors != nullptr) {
      angle /= factors[i];
    }
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double a = head[i];
    const double b = head[i + half];
    head[i] = static_cast<float>(a * c - b * s);
    head[i + half] = static_cast<float>(a * s + b * c);
  }
}

double GeluTanh(double x) {
  constexpr double kSqrt2OverPi = 0.79788456080286535587989211986876;
  return 0.5 * x * (1.0 + std::tanh(kSqrt2OverPi * (x + 0.044715 * x * x * x)));
}

}  // namespace

Reference::Reference(const ModelWeights& weights, Storage storage)
    : weights_(weights), storage_(storage) {
  Reset();
}

void Reference::Reset() {
  position_ = 0;
  keys_.assign(weights_.config.num_layers, {});
  values_.assign(weights_.config.num_layers, {});
  layer_trace_.clear();
}

/// Q8_1 round trip: per 32 values d = amax / 127 (stored as binary16) and
/// q = round(x / d), as ggml's quantize_row_q8_1.
void RoundQ8(const float* x, std::size_t n, float* y) {
  for (std::size_t b = 0; b < n; b += 32) {
    float amax = 0.0F;
    for (std::size_t i = b; i < b + 32; ++i) {
      amax = std::max(amax, std::fabs(x[i]));
    }
    const float d = amax / 127.0F;
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    const float d_half = RoundHalf(d);
    for (std::size_t i = b; i < b + 32; ++i) {
      y[i] = std::round(x[i] * id) * d_half;
    }
  }
}

void Reference::MatMul(const TensorRef& weight, const float* x_in,
                       std::size_t rows, float* out) const {
  const std::size_t cols = weight.cols;
  std::vector<float> rounded;
  const float* x = x_in;
  if (storage_ == Storage::kQ8Activations &&
      weight.type != core::GgmlType::kF32) {
    rounded.resize(rows * cols);
    for (std::size_t r = 0; r < rows; ++r) {
      RoundQ8(x_in + r * cols, cols, rounded.data() + r * cols);
    }
    x = rounded.data();
  }
  const std::size_t outputs = weight.rows;
  const std::size_t row_bytes = weight.RowBytes();
#pragma omp parallel
  {
    std::vector<float> w(cols);
#pragma omp for schedule(static)
    for (std::size_t o = 0; o < outputs; ++o) {
      quant::Dequantize(
          weight.type,
          static_cast<const std::uint8_t*>(weight.data) + o * row_bytes,
          w.data(), cols);
      for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        double acc = 0.0;
        for (std::size_t c = 0; c < cols; ++c) {
          acc += static_cast<double>(w[c]) * xr[c];
        }
        out[r * outputs + o] = static_cast<float>(acc);
      }
    }
  }
}

void Reference::Attention(std::uint32_t layer, const float* q, std::size_t rows,
                          std::span<const std::uint32_t> ends,
                          float* out) const {
  const Config& c = weights_.config;
  const std::uint32_t dim = c.HeadDim(layer);
  const std::uint32_t heads = c.num_heads;
  const std::uint32_t kv_heads = c.kv_heads[layer];
  const std::uint32_t group = heads / kv_heads;
  const std::size_t kv_stride = static_cast<std::size_t>(kv_heads) * dim;
  const auto& keys = keys_[layer];
  const auto& values = values_[layer];
  const std::uint32_t first = position_;  // position of row 0
#pragma omp parallel for collapse(2) schedule(static)
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::uint32_t h = 0; h < heads; ++h) {
      const std::uint32_t pos = first + static_cast<std::uint32_t>(r);
      const std::uint32_t lo = c.IsSliding(layer) && pos + 1 > c.sliding_window
                                   ? pos + 1 - c.sliding_window
                                   : 0;
      const std::uint32_t hi =
          ends.empty() ? pos + 1 : std::max(pos + 1, ends[r]);
      const float* qh = q + (r * heads + h) * dim;
      const std::uint32_t kvh = h / group;
      std::vector<double> scores(hi - lo);
      double max_score = -INFINITY;
      for (std::uint32_t p = lo; p < hi; ++p) {
        const float* k = keys.data() + p * kv_stride + kvh * dim;
        double s = 0.0;
        for (std::uint32_t d = 0; d < dim; ++d) {
          s += static_cast<double>(qh[d]) * k[d];
        }
        scores[p - lo] = s;  // attention scale is 1
        max_score = std::max(max_score, s);
      }
      double total = 0.0;
      for (double& s : scores) {
        s = std::exp(s - max_score);
        total += s;
      }
      std::vector<double> acc(dim, 0.0);
      for (std::uint32_t p = lo; p < hi; ++p) {
        const float* v = values.data() + p * kv_stride + kvh * dim;
        const double weight = scores[p - lo] / total;
        for (std::uint32_t d = 0; d < dim; ++d) {
          acc[d] += weight * v[d];
        }
      }
      float* o = out + (r * heads + h) * dim;
      for (std::uint32_t d = 0; d < dim; ++d) {
        o[d] = static_cast<float>(acc[d]);
      }
    }
  }
}

void Reference::Experts(const LayerWeights& w, const float* x, std::size_t rows,
                        float* out) const {
  const Config& c = weights_.config;
  const std::size_t d = c.hidden_size;
  const std::size_t experts = c.num_experts;
  const std::size_t width = c.expert_ffn_size;
  const float eps = c.rms_eps;
  // Router input: rms(x) / sqrt(hidden) * scale, then F32 logits.
  std::vector<float> routed(rows * d);
  const float* scale = Vec(w.router_scale);
  const float inv_root = 1.0F / std::sqrt(static_cast<float>(d));
  for (std::size_t r = 0; r < rows; ++r) {
    RmsNorm(x + r * d, nullptr, d, eps, &routed[r * d]);
    for (std::size_t i = 0; i < d; ++i) {
      routed[r * d + i] = routed[r * d + i] * inv_root * scale[i];
    }
  }
  std::vector<float> logits(rows * experts);
  MatMul(w.router, routed.data(), rows, logits.data());
  // Expert input.
  std::vector<float> h(rows * d);
  for (std::size_t r = 0; r < rows; ++r) {
    RmsNorm(x + r * d, Vec(w.pre_ffn_norm_2), d, eps, &h[r * d]);
  }
  const float* expert_scale = Vec(w.down_exps_scale);
  const auto slice = [](const TensorRef& t, std::size_t e) {
    TensorRef s = t;
    s.experts = 1;
    s.data =
        static_cast<const std::uint8_t*>(t.data) + e * t.RowBytes() * t.rows;
    return s;
  };
  std::fill(out, out + rows * d, 0.0F);
  std::vector<float> gate_up(2 * width);
  std::vector<float> act(width);
  std::vector<float> down(d);
  for (std::size_t r = 0; r < rows; ++r) {
    // Softmax over every expert, the top experts_used by probability (ties
    // to the lower index), renormalized over the chosen ones.
    const float* l = &logits[r * experts];
    std::vector<std::size_t> order(experts);
    for (std::size_t e = 0; e < experts; ++e) {
      order[e] = e;
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return l[a] > l[b]; });
    const double top = l[order[0]];
    double total = 0.0;
    for (std::size_t j = 0; j < c.experts_used; ++j) {
      total += std::exp(static_cast<double>(l[order[j]]) - top);
    }
    for (std::size_t j = 0; j < c.experts_used; ++j) {
      const std::size_t e = order[j];
      const double weight =
          std::exp(static_cast<double>(l[e]) - top) / total * expert_scale[e];
      MatMul(slice(w.gate_up_exps, e), &h[r * d], 1, gate_up.data());
      for (std::size_t i = 0; i < width; ++i) {
        act[i] = static_cast<float>(GeluTanh(gate_up[i]) * gate_up[width + i]);
      }
      MatMul(slice(w.down_exps, e), act.data(), 1, down.data());
      for (std::size_t i = 0; i < d; ++i) {
        out[r * d + i] = static_cast<float>(out[r * d + i] + weight * down[i]);
      }
    }
  }
}

void Reference::Forward(std::span<const TokenId> tokens,
                        std::vector<float>* logits, std::vector<float>* hidden,
                        std::span<const Image> images) {
  const Config& c = weights_.config;
  const std::size_t n = tokens.size();
  const std::size_t d = c.hidden_size;
  const float eps = c.rms_eps;
  std::vector<float> x(n * d);
  std::vector<float> h(n * d);

  // Scaled token embeddings (the scale is applied in float, as llama.cpp).
  const float embed_scale = std::sqrt(static_cast<float>(d));
  const std::size_t embed_row = weights_.token_embd.RowBytes();
  for (std::size_t r = 0; r < n; ++r) {
    quant::Dequantize(
        weights_.token_embd.type,
        static_cast<const std::uint8_t*>(weights_.token_embd.data) +
            static_cast<std::size_t>(tokens[r]) * embed_row,
        x.data() + r * d, d);
    for (std::size_t i = 0; i < d; ++i) {
      x[r * d + i] *= embed_scale;
    }
  }
  // Image rows take the encoder output unscaled and, in sliding layers, see
  // their whole image.
  std::vector<std::uint32_t> ends;
  for (const Image& image : images) {
    if (image.count == 0 || std::size_t{image.row} + image.count > n ||
        image.embedding == nullptr) {
      throw std::invalid_argument("gemma4 reference: invalid image rows");
    }
    std::copy_n(image.embedding, std::size_t{image.count} * d,
                x.begin() + static_cast<std::ptrdiff_t>(image.row * d));
    ends.resize(n, 0);
    std::fill_n(ends.begin() + image.row, image.count,
                position_ + image.row + image.count);
  }
  if (trace_) {
    layer_trace_.assign(static_cast<std::size_t>(c.num_layers) * n * d, 0.0F);
  }

  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const LayerWeights& w = weights_.layers[l];
    const std::uint32_t dim = c.HeadDim(l);
    const std::uint32_t heads = c.num_heads;
    const std::uint32_t kv_heads = c.kv_heads[l];
    const std::size_t q_dim = c.QDim(l);
    const std::size_t kv_dim = c.KvDim(l);
    const float* factors =
        c.IsSliding(l) ? nullptr : weights_.rope_factors.values.data();

    for (std::size_t r = 0; r < n; ++r) {
      RmsNorm(&x[r * d], Vec(w.attn_norm), d, eps, &h[r * d]);
    }
    std::vector<float> q(n * q_dim);
    std::vector<float> k(n * kv_dim);
    std::vector<float> v(n * kv_dim);
    MatMul(w.attn_q, h.data(), n, q.data());
    MatMul(w.attn_k, h.data(), n, k.data());
    if (w.attn_v.empty()) {
      v = k;  // global layers: V is the raw K projection
    } else {
      MatMul(w.attn_v, h.data(), n, v.data());
    }
    for (std::size_t r = 0; r < n; ++r) {
      const auto pos = static_cast<std::uint32_t>(position_ + r);
      for (std::uint32_t hh = 0; hh < heads; ++hh) {
        float* head = &q[(r * heads + hh) * dim];
        RmsNorm(head, Vec(w.attn_q_norm), dim, eps, head);
        Rope(head, dim, pos, c.RopeTheta(l), factors);
      }
      for (std::uint32_t hh = 0; hh < kv_heads; ++hh) {
        float* kh = &k[(r * kv_heads + hh) * dim];
        float* vh = &v[(r * kv_heads + hh) * dim];
        RmsNorm(kh, Vec(w.attn_k_norm), dim, eps, kh);
        Rope(kh, dim, pos, c.RopeTheta(l), factors);
        RmsNorm(vh, nullptr, dim, eps, vh);
      }
    }
    if (storage_ != Storage::kFloat32) {
      for (auto& value : k)
        value = RoundHalf(value);
      for (auto& value : v)
        value = RoundHalf(value);
    }
    keys_[l].insert(keys_[l].end(), k.begin(), k.end());
    values_[l].insert(values_[l].end(), v.begin(), v.end());

    std::vector<float> attn(n * q_dim);
    Attention(l, q.data(), n,
              c.IsSliding(l) ? std::span<const std::uint32_t>(ends)
                             : std::span<const std::uint32_t>{},
              attn.data());
    std::vector<float> o(n * d);
    MatMul(w.attn_output, attn.data(), n, o.data());
    for (std::size_t r = 0; r < n; ++r) {
      RmsNorm(&o[r * d], Vec(w.post_attn_norm), d, eps, &o[r * d]);
      for (std::size_t i = 0; i < d; ++i) {
        x[r * d + i] += o[r * d + i];
      }
      RmsNorm(&x[r * d], Vec(w.ffn_norm), d, eps, &h[r * d]);
    }

    const std::size_t ff = c.ffn_size;
    std::vector<float> gate(n * ff);
    std::vector<float> up(n * ff);
    MatMul(w.ffn_gate, h.data(), n, gate.data());
    MatMul(w.ffn_up, h.data(), n, up.data());
    for (std::size_t i = 0; i < n * ff; ++i) {
      gate[i] = static_cast<float>(GeluTanh(gate[i]) * up[i]);
    }
    std::vector<float> f(n * d);
    MatMul(w.ffn_down, gate.data(), n, f.data());
    if (c.HasExperts()) {
      // The dense MLP and the routed experts both read the attention
      // residual x; their separately normed outputs add up to f.
      std::vector<float> moe(n * d);
      Experts(w, x.data(), n, moe.data());
      for (std::size_t r = 0; r < n; ++r) {
        RmsNorm(&f[r * d], Vec(w.post_ffn_norm_1), d, eps, &f[r * d]);
        RmsNorm(&moe[r * d], Vec(w.post_ffn_norm_2), d, eps, &moe[r * d]);
        for (std::size_t i = 0; i < d; ++i) {
          f[r * d + i] += moe[r * d + i];
        }
      }
    }
    for (std::size_t r = 0; r < n; ++r) {
      RmsNorm(&f[r * d], Vec(w.post_ffn_norm), d, eps, &f[r * d]);
      for (std::size_t i = 0; i < d; ++i) {
        x[r * d + i] = (x[r * d + i] + f[r * d + i]) * w.output_scale;
      }
    }
    if (trace_) {
      std::copy(x.begin(), x.end(),
                layer_trace_.begin() + static_cast<std::size_t>(l) * n * d);
    }
  }

  for (std::size_t r = 0; r < n; ++r) {
    RmsNorm(&x[r * d], Vec(weights_.output_norm), d, eps, &h[r * d]);
  }
  if (hidden != nullptr) {
    *hidden = h;
  }
  if (logits != nullptr) {
    const std::size_t vocab = weights_.vocab_size;
    logits->assign(n * vocab, 0.0F);
    MatMul(weights_.output, h.data(), n, logits->data());
    const double cap = c.final_logit_softcap;
    if (cap > 0.0) {
      for (float& value : *logits) {
        value = static_cast<float>(cap * std::tanh(value / cap));
      }
    }
  }
  position_ += static_cast<std::uint32_t>(n);
}

}  // namespace gufo::models::gemma4

#include "src/models/lfm3/weights.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "src/core/quant/ggml_dequant.hpp"
namespace gufo::models::lfm3 {
using core::GgmlType;
std::optional<Weights> Weights::Bind(const core::GgufReader& r,
                                     std::string* error) {
  auto config = Config::FromGguf(r, error);
  if (!config)
    return std::nullopt;
  try {
    Weights w;
    w.config = *config;
    const auto& c = w.config;
    auto get = [&](const std::string& name, std::uint32_t cols,
                   std::uint32_t rows, bool optional = false) -> Tensor {
      const auto* t = r.FindTensor(name);
      if (!t && optional)
        return {};
      if (!t)
        throw std::runtime_error("missing tensor " + name);
      if (t->dimensions.empty() || t->dimensions.size() > 2 ||
          t->dimensions[0] != cols ||
          (t->dimensions.size() == 2 ? t->dimensions[1] : 1) != rows)
        throw std::runtime_error("invalid shape: " + name);
      switch (t->type) {
        case GgmlType::kF32:
        case GgmlType::kF16:
        case GgmlType::kBF16:
        case GgmlType::kQ8_0:
          break;
        default:
          throw std::runtime_error("unsupported tensor format: " + name);
      }
      auto block = quant::QuantizedBlockElements(t->type);
      if (block && cols % block)
        throw std::runtime_error("unaligned quantized row: " + name);
      auto bytes = quant::EncodedSizeBytes(t->type, std::size_t(cols) * rows);
      if (!bytes || bytes != t->size_bytes)
        throw std::runtime_error("invalid tensor bytes: " + name);
      return {t->data, t->type, cols, rows, bytes};
    };
    auto norm = [&](const std::string& name, std::uint32_t dim,
                    bool optional = false) {
      auto t = get(name, dim, 1, optional);
      if (t.data && t.type != GgmlType::kF32)
        throw std::runtime_error("LFM3 norms must be F32: " + name);
      return t;
    };
    w.embedding = get("token_embd.weight", c.hidden, c.vocab);
    w.output = get("output.weight", c.hidden, c.vocab, true);
    if (!w.output.data)
      w.output = w.embedding;
    w.output_norm = norm("output_norm.weight", c.hidden);
    w.rope_freqs = norm("rope_freqs.weight", c.head_dim / 2);
    for (float factor : std::span(static_cast<const float*>(w.rope_freqs.data),
                                  c.head_dim / 2))
      if (!std::isfinite(factor) || !(factor > 0))
        throw std::runtime_error("invalid LFM3 RoPE factor");
    for (std::size_t i = 0; i < c.layers.size(); ++i) {
      const auto p = "blk." + std::to_string(i) + ".";
      LayerWeights l;
      l.norm = norm(p + "attn_norm.weight", c.hidden);
      l.ffn_norm = norm(p + "ffn_norm.weight", c.hidden);
      l.post_norm = norm(p + "post_attention_norm.weight", c.hidden, true);
      l.post_ffn_norm = norm(p + "post_ffw_norm.weight", c.hidden, true);
      if (bool(l.post_norm.data) != bool(l.post_ffn_norm.data))
        throw std::runtime_error("LFM3 requires both post norms or neither");
      l.gate = get(p + "ffn_gate.weight", c.hidden, c.intermediate);
      l.up = get(p + "ffn_up.weight", c.hidden, c.intermediate);
      l.down = get(p + "ffn_down.weight", c.intermediate, c.hidden);
      if (c.layers[i] == LayerType::kConvolution) {
        l.conv = get(p + "shortconv.conv.weight", c.conv_width, c.hidden);
        if (l.conv.type != GgmlType::kF32)
          throw std::runtime_error("LFM3 convolution must be F32");
        l.in_proj = get(p + "shortconv.in_proj.weight", c.hidden, 3 * c.hidden);
        l.out_proj = get(p + "shortconv.out_proj.weight", c.hidden, c.hidden);
      } else {
        l.q = get(p + "attn_q.weight", c.hidden, c.heads * c.head_dim);
        l.k = get(p + "attn_k.weight", c.hidden, c.kv_heads[i] * c.head_dim);
        l.v = get(p + "attn_v.weight", c.hidden, c.kv_heads[i] * c.head_dim);
        l.q_norm = norm(p + "attn_q_norm.weight", c.head_dim);
        l.k_norm = norm(p + "attn_k_norm.weight", c.head_dim);
        l.attn_gate = get(p + "attn_gate.weight", c.hidden, c.heads);
        l.attn_out =
            get(p + "attn_output.weight", c.heads * c.head_dim, c.hidden);
      }
      w.layers.push_back(l);
    }
    return w;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return std::nullopt;
  }
}
void DecodeRow(const Tensor& t, std::uint32_t row, std::span<float> out) {
  if (row >= t.rows || out.size() != t.cols)
    throw std::invalid_argument("invalid LFM3 tensor row");
  const auto* data =
      static_cast<const std::uint8_t*>(t.data) + row * (t.bytes / t.rows);
  switch (t.type) {
    case GgmlType::kF32:
      std::memcpy(out.data(), data, out.size_bytes());
      break;
    case GgmlType::kF16:
    case GgmlType::kBF16:
      for (std::size_t i = 0; i < out.size(); ++i) {
        std::uint16_t v;
        std::memcpy(&v, data + i * 2, 2);
        out[i] = t.type == GgmlType::kF16
                     ? quant::Fp16ToFloat(v)
                     : std::bit_cast<float>(std::uint32_t(v) << 16);
      }
      break;
    case GgmlType::kQ8_0:
      quant::DequantizeQ8_0(data, out.data(), t.cols);
      break;
    default:
      throw std::invalid_argument("unsupported LFM3 row");
  }
}
}  // namespace gufo::models::lfm3

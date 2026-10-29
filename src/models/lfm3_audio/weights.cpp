#include "src/models/lfm3_audio/weights.hpp"

#include <stdexcept>

#include "src/core/quant/ggml_dequant.hpp"
namespace gufo::models::lfm3_audio {
using core::GgmlType;
namespace {
struct Binder {
  const core::GgufReader& r;
  const core::GgufTensorInfo* Find(const std::string& name) const {
    return r.FindTensor(name);
  }
  // A matrix of rows x cols (GGUF dimensions [cols, rows, extra...]).
  Tensor Get(const std::string& name, std::uint32_t cols, std::uint32_t rows,
             bool optional = false) const {
    const auto* t = r.FindTensor(name);
    if (!t && optional)
      return {};
    if (!t)
      throw std::runtime_error("missing tensor " + name);
    std::uint64_t count = 1;
    for (auto d : t->dimensions)
      count *= d;
    if (t->dimensions.empty() || t->dimensions[0] != cols ||
        count != std::uint64_t(cols) * rows)
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
    const auto block = quant::QuantizedBlockElements(t->type);
    if (block && cols % block)
      throw std::runtime_error("unaligned quantized row: " + name);
    const auto bytes =
        quant::EncodedSizeBytes(t->type, std::size_t(cols) * rows);
    if (!bytes || bytes != t->size_bytes)
      throw std::runtime_error("invalid tensor bytes: " + name);
    return {t->data, t->type, cols, rows, bytes};
  }
  Tensor F32(const std::string& name, std::uint32_t cols,
             std::uint32_t rows = 1, bool optional = false) const {
    auto t = Get(name, cols, rows, optional);
    if (t.data && t.type != GgmlType::kF32)
      throw std::runtime_error("expected F32 tensor: " + name);
    return t;
  }
  std::uint32_t Dim(const std::string& name, std::size_t index) const {
    const auto* t = r.FindTensor(name);
    if (!t || t->dimensions.size() <= index)
      throw std::runtime_error("missing tensor " + name);
    return std::uint32_t(t->dimensions[index]);
  }
  // LFM2/LFM3 blocks named prefix{i}. with the audio towers' tensor names.
  void Blocks(Stack& s, const std::string& prefix, std::uint32_t count) const {
    for (std::uint32_t i = 0; i < count; ++i) {
      const auto p = prefix + std::to_string(i) + ".";
      Block b;
      b.norm = F32(p + "ln1.weight", s.hidden);
      b.ffn_norm = F32(p + "ln2.weight", s.hidden);
      b.post_norm = F32(p + "attn_post_norm.weight", s.hidden, 1, true);
      b.post_ffn_norm = F32(p + "ffn_post_norm.weight", s.hidden, 1, true);
      if (bool(b.post_norm.data) != bool(b.post_ffn_norm.data))
        throw std::runtime_error("audio block needs both post norms: " + p);
      b.gate = Get(p + "ffn_gate.weight", s.hidden, s.intermediate);
      b.up = Get(p + "ffn_up.weight", s.hidden, s.intermediate);
      b.down = Get(p + "ffn_down.weight", s.intermediate, s.hidden);
      if (Find(p + "attn_q.weight")) {
        b.q = Get(p + "attn_q.weight", s.hidden, s.heads * s.head_dim);
        b.k = Get(p + "attn_k.weight", s.hidden, s.kv_heads * s.head_dim);
        b.v = Get(p + "attn_v.weight", s.hidden, s.kv_heads * s.head_dim);
        b.q_norm = F32(p + "attn_q_norm.weight", s.head_dim);
        b.k_norm = F32(p + "attn_k_norm.weight", s.head_dim);
        b.attn_gate = Get(p + "attn_gate.weight", s.hidden, s.heads, true);
        b.attn_out = Get(p + "attn_out.weight", s.heads * s.head_dim, s.hidden);
      } else {
        b.conv = F32(p + "conv_dw.weight", s.conv_width, s.hidden);
        b.in_proj = Get(p + "conv_pw1.weight", s.hidden, 3 * s.hidden);
        b.out_proj = Get(p + "conv_pw2.weight", s.hidden, s.hidden);
      }
      s.blocks.push_back(b);
    }
  }
  std::uint32_t U32(const std::string& key) const {
    const auto v = r.GetMetadataUint32(key);
    if (!v || !*v)
      throw std::runtime_error("invalid " + key);
    return *v;
  }
  float Float(const std::string& key) const {
    const auto v = r.GetMetadataFloat32(key);
    if (!v)
      throw std::runtime_error("invalid " + key);
    return *v;
  }
};
// Attention geometry of a stack from its first attention block.
void Geometry(const Binder& b, Stack& s, const std::string& prefix,
              std::uint32_t count) {
  for (std::uint32_t i = 0; i < count; ++i) {
    const auto p = prefix + std::to_string(i) + ".";
    if (!b.Find(p + "attn_q.weight"))
      continue;
    s.head_dim = b.Dim(p + "attn_q_norm.weight", 0);
    s.heads = b.Dim(p + "attn_q.weight", 1) / s.head_dim;
    s.kv_heads = b.Dim(p + "attn_k.weight", 1) / s.head_dim;
    return;
  }
  throw std::runtime_error("audio stack without attention: " + prefix);
}
std::uint32_t CountBlocks(const Binder& b, const std::string& prefix) {
  std::uint32_t n = 0;
  while (b.Find(prefix + std::to_string(n) + ".ln1.weight"))
    ++n;
  return n;
}
}  // namespace
Tensor Rows(const Tensor& t, std::uint32_t first, std::uint32_t rows) {
  if (first + rows > t.rows)
    throw std::out_of_range("tensor row view");
  const auto row_bytes = t.bytes / t.rows;
  return {static_cast<const std::uint8_t*>(t.data) + first * row_bytes, t.type,
          t.cols, rows, row_bytes * rows};
}
std::optional<AudioWeights> AudioWeights::Bind(const core::GgufReader& r,
                                               std::uint32_t backbone_hidden,
                                               std::string* error) {
  try {
    if (r.GetMetadataString("general.architecture") != "clip" ||
        r.GetMetadataString("clip.audio.projector_type") != "lfm3a" ||
        r.GetMetadataString("clip.gen.audio.projector_type") != "lfm3a_gen")
      throw std::runtime_error(
          "expected an LFM3-Audio mmproj GGUF (lfm3a and lfm3a_gen)");
    const Binder b{r};
    AudioWeights w;
    w.projection_dim = b.U32("clip.audio.projection_dim");
    if (w.projection_dim != backbone_hidden ||
        b.U32("clip.gen.audio.projection_dim") != backbone_hidden)
      throw std::runtime_error("mmproj does not match the backbone width");
    if (b.U32("clip.audio.num_mel_bins") != kMelBins)
      throw std::runtime_error("unsupported mel bin count");

    // Encoder.
    auto& e = w.encoder;
    e.hidden = b.U32("clip.audio.embedding_length");
    e.intermediate = b.U32("clip.audio.feed_forward_length");
    e.eps = b.Float("clip.audio.attention.layer_norm_epsilon");
    // Official encoder configuration: default RoPE, theta 1e6, full rotation.
    e.theta = 1e6F;
    w.encoder_window = b.U32("clip.audio.window_size");
    e.conv_width = b.Dim("a.blk.0.conv_dw.weight", 0);
    const auto encoder_blocks = b.U32("clip.audio.block_count");
    Geometry(b, e, "a.blk.", encoder_blocks);
    if (e.heads != b.U32("clip.audio.attention.head_count") ||
        e.kv_heads != b.U32("clip.audio.attention.head_count_kv"))
      throw std::runtime_error("inconsistent encoder attention heads");
    b.Blocks(e, "a.blk.", encoder_blocks);
    e.output_norm = b.F32("a.post_ln.weight", e.hidden);
    const std::uint32_t channels = b.Dim("a.conv1d.0.weight", 3);
    w.downsample_channels = channels;
    const int ids[] = {0, 2, 3, 5, 6};
    for (int i = 0; i < 5; ++i) {
      const auto p = "a.conv1d." + std::to_string(ids[i]);
      const bool pointwise = ids[i] == 3 || ids[i] == 6;
      // GGUF [kw, kh, in, out]: torch [out][in][kh][kw] in memory.
      w.conv_weight[i] = pointwise
                             ? b.F32(p + ".weight", 1, channels * channels)
                             : b.F32(p + ".weight", 3, 3 * channels);
      w.conv_bias[i] = b.F32(p + ".bias", 1, channels);  // [1, 1, C]
    }
    w.pre_out =
        b.Get("a.pre_encode.out.weight", channels * kMelBins / 8, e.hidden);
    w.pre_out_bias = b.F32("a.pre_encode.out.bias", e.hidden);
    w.proj_norm = b.F32("mm.a.mlp.0.weight", e.hidden);
    w.proj_norm_bias = b.F32("mm.a.mlp.0.bias", e.hidden);
    w.proj1 = b.Get("mm.a.mlp.1.weight", e.hidden, backbone_hidden);
    w.proj1_bias = b.F32("mm.a.mlp.1.bias", backbone_hidden);
    w.proj2 = b.Get("mm.a.mlp.3.weight", backbone_hidden, backbone_hidden);
    w.proj2_bias = b.F32("mm.a.mlp.3.bias", backbone_hidden);

    // Depth decoder.
    auto& d = w.depth;
    d.hidden = b.U32("clip.gen.audio.embedding_length");
    d.intermediate = b.U32("clip.gen.audio.feed_forward_length");
    d.eps = b.Float("clip.gen.audio.attention.layer_norm_epsilon");
    d.theta = 1e6F;
    const auto depth_blocks = b.U32("clip.gen.audio.block_count");
    Geometry(b, d, "a.gen.code.blk.", depth_blocks);
    if (d.heads != b.U32("clip.gen.audio.attention.head_count") ||
        d.kv_heads != b.U32("clip.gen.audio.attention.head_count_kv"))
      throw std::runtime_error("inconsistent depth attention heads");
    b.Blocks(d, "a.gen.code.blk.", depth_blocks);
    for (auto& block : d.blocks)
      if (!block.attention() || !block.attn_gate.data)
        throw std::runtime_error("depth blocks must be gated attention");
    d.output_norm = b.F32("a.gen.code.output_norm.weight", d.hidden);
    w.depth_in = b.Get("a.gen.code.proj_in.weight", backbone_hidden,
                       kCodebooks * d.hidden);
    w.depth_in_bias = b.F32("a.gen.code.proj_in.bias", kCodebooks * d.hidden);
    w.mask_embedding = b.F32("a.gen.code.mask_embd", d.hidden);
    w.code_embeddings =
        b.Get("a.gen.code.embd.weight", d.hidden, kCodebooks * kCodebookSize);
    w.head_norms = b.F32("a.gen.code.head_norm.weight", d.hidden, kCodebooks);
    w.audio_out = b.Get("a.gen.code.out_embd.weight", backbone_hidden,
                        kCodebooks * kCodebookSize);

    // Codec decoder. The GGUF stores no codec metadata: the geometry comes
    // from tensor shapes, the rest from the official LiquidCodec config.
    auto& k = w.codec;
    k.hidden = b.Dim("a.gen.wav.tfm.in_proj.weight", 1);
    k.intermediate = b.Dim("a.gen.wav.tfm.blk.0.ffn_gate.weight", 1);
    k.eps = 1e-5F;
    k.theta = 1e6F;
    k.conv_width = b.Dim("a.gen.wav.tfm.blk.0.conv_dw.weight", 0);
    const auto codec_blocks = CountBlocks(b, "a.gen.wav.tfm.blk.");
    Geometry(b, k, "a.gen.wav.tfm.blk.", codec_blocks);
    b.Blocks(k, "a.gen.wav.tfm.blk.", codec_blocks);
    k.output_norm = b.F32("a.gen.wav.tfm.output_norm.weight", k.hidden);
    w.codec_window = 30;
    w.frame_stack = 6;
    w.fsq_table = b.F32("a.gen.wav.quant.first.codebook.weight", 4, kCodes);
    const auto latent = kCodebooks * 4;
    const auto quantizer = b.Dim("a.gen.wav.quant.first.out_proj.weight", 1);
    w.fsq_out =
        b.F32("a.gen.wav.quant.first.out_proj.weight", latent, quantizer);
    w.fsq_out_bias = b.F32("a.gen.wav.quant.first.out_proj.bias", quantizer);
    w.codec_in = b.Get("a.gen.wav.tfm.in_proj.weight", quantizer, k.hidden);
    w.codec_in_bias = b.F32("a.gen.wav.tfm.in_proj.bias", k.hidden);
    const auto bins2 = b.Dim("a.gen.wav.tfm.out_proj.weight", 1);
    w.n_fft = bins2 - 2;
    w.hop = w.n_fft / 4;
    w.codec_out = b.Get("a.gen.wav.tfm.out_proj.weight", k.hidden, bins2);
    w.codec_out_bias = b.F32("a.gen.wav.tfm.out_proj.bias", bins2);
    for (auto* s : {&e, &d, &k})
      if (s->hidden % 256 || s->head_dim > 128 || s->head_dim % 2 ||
          s->heads % s->kv_heads || (s->conv_width && s->conv_width > 32))
        throw std::runtime_error("unsupported audio stack geometry");
    return w;
  } catch (const std::exception& error_value) {
    if (error)
      *error = error_value.what();
    return std::nullopt;
  }
}
}  // namespace gufo::models::lfm3_audio

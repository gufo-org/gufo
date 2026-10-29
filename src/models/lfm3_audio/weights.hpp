#ifndef GUFO_MODELS_LFM3_AUDIO_WEIGHTS_HPP_
#define GUFO_MODELS_LFM3_AUDIO_WEIGHTS_HPP_
#include <array>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/lfm3/weights.hpp"
namespace gufo::models::lfm3_audio {
using lfm3::Tensor;
// One LFM2/LFM3 block of the audio towers. Attention layers have q set; the
// others are gated short convolutions.
struct Block {
  Tensor norm, ffn_norm, post_norm, post_ffn_norm;
  Tensor gate, up, down;
  Tensor conv, in_proj, out_proj;
  Tensor q, k, v, q_norm, k_norm, attn_gate, attn_out;
  bool attention() const { return q.data != nullptr; }
};
struct Stack {
  std::uint32_t hidden{}, intermediate{}, heads{}, kv_heads{}, head_dim{},
      conv_width{};
  float eps{}, theta{};
  std::vector<Block> blocks;
  Tensor output_norm;
};
// The audio towers of the LFM3-Audio mmproj GGUF.
struct AudioWeights {
  static constexpr std::uint32_t kCodebooks = 12, kCodebookSize = 2017,
                                 kCodes = 2016, kMelBins = 128;
  // Encoder: (mel, time) downsampler, LFM2 stack and projector.
  std::uint32_t encoder_window{}, downsample_channels{};
  std::array<Tensor, 5> conv_weight, conv_bias;  // conv1d.{0,2,3,5,6}
  Tensor pre_out, pre_out_bias;
  Stack encoder;
  Tensor proj_norm, proj_norm_bias, proj1, proj1_bias, proj2, proj2_bias;
  // Depth decoder: bidirectional LFM3 over the 12 codebook slots.
  Stack depth;
  Tensor depth_in, depth_in_bias, mask_embedding;
  Tensor code_embeddings;  // [12 * 2017, depth.hidden]
  Tensor head_norms;       // [12, depth.hidden]
  Tensor audio_out;        // [12 * 2017, backbone hidden]
  // Codec decoder (LiquidCodec): FSQ latents, LFM3 stack and ISTFT head.
  Stack codec;
  std::uint32_t codec_window{}, frame_stack{}, n_fft{}, hop{};
  Tensor fsq_table;  // [2016, 4]
  Tensor fsq_out, fsq_out_bias, codec_in, codec_in_bias, codec_out,
      codec_out_bias;
  std::uint32_t projection_dim{};
  // Views borrow the GGUF reader's mapping.
  static std::optional<AudioWeights> Bind(const core::GgufReader& reader,
                                          std::uint32_t backbone_hidden,
                                          std::string* error = nullptr);
  // Every tensor, for device upload.
  template<class F>
  void Visit(F f) {
    for (auto* t :
         {&pre_out,       &pre_out_bias,  &proj_norm,      &proj_norm_bias,
          &proj1,         &proj1_bias,    &proj2,          &proj2_bias,
          &depth_in,      &depth_in_bias, &mask_embedding, &code_embeddings,
          &head_norms,    &audio_out,     &fsq_table,      &fsq_out,
          &fsq_out_bias,  &codec_in,      &codec_in_bias,  &codec_out,
          &codec_out_bias})
      f(*t);
    for (auto& t : conv_weight)
      f(t);
    for (auto& t : conv_bias)
      f(t);
    for (auto* s : {&encoder, &depth, &codec}) {
      if (s->output_norm.data)
        f(s->output_norm);
      for (auto& b : s->blocks)
        for (auto* t :
             {&b.norm, &b.ffn_norm, &b.post_norm, &b.post_ffn_norm, &b.gate,
              &b.up, &b.down, &b.conv, &b.in_proj, &b.out_proj, &b.q, &b.k,
              &b.v, &b.q_norm, &b.k_norm, &b.attn_gate, &b.attn_out})
          if (t->data)
            f(*t);
    }
  }
};
// Row r of a [rows, cols] tensor slab starting at row first, as a view.
Tensor Rows(const Tensor& t, std::uint32_t first, std::uint32_t rows);
}  // namespace gufo::models::lfm3_audio
#endif

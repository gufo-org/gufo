#include "src/models/qwen3_tts/npu/speech_decoder_npu.hpp"

#include <algorithm>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gufo/ext/experimental/npu.hpp"
#include "src/models/qwen3_tts/loader.hpp"
#include "src/models/qwen3_tts/npu/cpu_conv.hpp"
#include "src/models/qwen3_tts/npu/npu_conv.hpp"
#include "src/models/qwen3_tts/tensor.hpp"

namespace gufo::models::qwen3_tts::npu {
namespace {

namespace host = gufo::npu::host;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kCodeGroups = 16;
constexpr std::size_t kCodebookDimension = 256;
constexpr std::size_t kSamplesPerCode = 1920;
constexpr std::size_t kChunkFrames = 300;
constexpr std::size_t kLeftContextFrames = 25;
constexpr float kCodebookEpsilon = 1.0e-5F;

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

struct Linear {
  int id{-1};
  std::vector<float> bias;
  std::size_t in{}, out{};
};

#if defined(__AVX2__) && defined(__FMA__)
// sin of 8 lanes: reduction by pi (Cody-Waite) and odd/even minimax
// polynomials on [-pi/2, pi/2] (absolute error ~1e-7 for |x| < 1e5).
inline __m256 Sin8(__m256 x) {
  const __m256 k =
      _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.31830988618379067f)),
                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(k, _mm256_set1_ps(3.140625f), x);
  r = _mm256_fnmadd_ps(k, _mm256_set1_ps(9.67502593994140625e-4f), r);
  r = _mm256_fnmadd_ps(k, _mm256_set1_ps(1.50995799e-7f), r);
  const __m256 r2 = _mm256_mul_ps(r, r);
  __m256 p = _mm256_set1_ps(2.7183114939898219e-6f);
  p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(-1.9839334836096632e-4f));
  p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(8.3333337680149173e-3f));
  p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(-1.6666666664721728e-1f));
  p = _mm256_fmadd_ps(_mm256_mul_ps(p, r2), r, r);
  // sin(x) = (-1)^k sin(r)
  const __m256i odd = _mm256_slli_epi32(_mm256_cvtps_epi32(k), 31);
  return _mm256_xor_ps(p, _mm256_castsi256_ps(odd));
}
#endif

struct Snake {
  std::vector<float> alpha_exp, inverse_beta;
};

struct TransformerLayer {
  std::vector<float> input_norm, post_norm, attention_scale, mlp_scale;
  Linear qkv, o, gate_up, down;
};

struct ConvNeXt {
  std::vector<float> depthwise_weight, depthwise_bias, norm_weight, norm_bias,
      gamma;
  Linear pointwise1, pointwise2;
};

struct UpsampleStage {
  CpuConv upsample;
  ConvNeXt convnext;
};

struct ResidualUnit {
  Snake activation1, activation2;
  CpuConv conv1, conv2;
};

struct DecoderBlock {
  Snake activation;
  CpuConv upsample;
  std::array<ResidualUnit, 3> residuals;
};

struct ConvHistory {
  std::vector<float> input;
  std::size_t rows{0};
};

struct DecoderHistory {
  std::unordered_map<const void*, ConvHistory> convolutions;
  std::vector<std::vector<float>> key, value;  // per layer [position][width]
  std::vector<std::uint32_t> codes;
  std::size_t position{0};
  void ResetBlock() {
    position = 0;
    for (auto& [weight, history] : convolutions)
      history.rows = 0;
  }
};

}  // namespace

struct SpeechDecoderNpuRuntime::Impl {
  LoadResult model;
  gufo::npu::Device device;
  gufo::npu::BfpGemm gemm;
  std::vector<float> semantic_table;                // [code][256]
  std::vector<std::vector<float>> acoustic_tables;  // 15 x [code][256]
  Linear semantic_projection, acoustic_projection;
  CpuConv pre_conv;
  Linear transformer_input, transformer_output;
  std::vector<TransformerLayer> layers;
  std::vector<float> transformer_norm;
  std::vector<UpsampleStage> upsample_stages;
  CpuConv decoder_input;
  std::vector<DecoderBlock> blocks;
  Snake output_activation;
  CpuConv output_conv;
  DecoderHistory history;
  std::unique_ptr<DecoderHistory> reference_history;
  // Every convolution's multiply-accumulates on the NPU (npu_conv.hpp).
  std::unordered_map<const CpuConv*, NpuConv> npu_convs;
  DecoderHistory* active{nullptr};
  bool profile{std::getenv("GUFO_NPU_PROFILE") != nullptr};

  const Tensor& Find(const std::string& name) const {
    const Tensor* t = model.store->Find(name);
    if (t == nullptr || t->dtype != DType::kF32)
      throw std::runtime_error("Qwen3-TTS NPU decoder: missing F32 tensor " +
                               name);
    return *t;
  }
  const float* Data(const std::string& name) const {
    return reinterpret_cast<const float*>(Find(name).data);
  }
  std::vector<float> Vector(const std::string& name) const {
    const Tensor& t = Find(name);
    const float* p = reinterpret_cast<const float*>(t.data);
    return {p, p + *t.NumElements()};
  }
  // Uploads the row-wise concatenation of F32 matrices [n_i][k].
  Linear MakeLinear(std::initializer_list<std::string> prefixes,
                    bool bias = true) {
    std::vector<float> rows;
    Linear l;
    for (const std::string& prefix : prefixes) {
      const Tensor& t = Find(prefix + ".weight");
      const std::size_t n = t.shape[0], k = *t.NumElements() / t.shape[0];
      l.in = k;
      l.out += n;
      const float* p = reinterpret_cast<const float*>(t.data);
      rows.insert(rows.end(), p, p + n * k);
      if (bias) {
        const std::vector<float> b = Vector(prefix + ".bias");
        l.bias.insert(l.bias.end(), b.begin(), b.end());
      }
    }
    l.id = gemm.Upload(int(l.out), int(l.in), rows);
    return l;
  }
  CpuConv MakeConv(const std::string& prefix, std::size_t in, std::size_t out,
                   std::size_t kernel, std::size_t dilation = 1) {
    return CpuConv::Make(Data(prefix + ".weight"), Data(prefix + ".bias"), in,
                         out, kernel, dilation);
  }
  CpuConv MakeTranspose(const std::string& prefix, std::size_t in,
                        std::size_t out, std::size_t kernel,
                        std::size_t stride) {
    return CpuConv::MakeTranspose(Data(prefix + ".weight"),
                                  Data(prefix + ".bias"), in, out, kernel,
                                  stride);
  }
  Snake MakeSnake(const std::string& prefix) {
    Snake s;
    for (float a : Vector(prefix + ".alpha"))
      s.alpha_exp.push_back(std::exp(a));
    for (float b : Vector(prefix + ".beta"))
      s.inverse_beta.push_back(1.0F / (std::exp(b) + 1.0e-9F));
    return s;
  }
  std::vector<float> Codebook(const std::string& prefix) {
    const std::size_t size = model.config.speech_tokenizer.codebook_size;
    const float* sum = Data(prefix + "embedding_sum");
    const float* usage = Data(prefix + "cluster_usage");
    std::vector<float> table(size * kCodebookDimension);
    for (std::size_t c = 0; c < size; ++c) {
      const float d = std::max(usage[c], kCodebookEpsilon);
      for (std::size_t i = 0; i < kCodebookDimension; ++i)
        table[c * kCodebookDimension + i] = sum[c * kCodebookDimension + i] / d;
    }
    return table;
  }

  void Load(const std::string& model_root) {
    model = LoadModelDirectory(model_root);
    if (!model.ok)
      throw std::runtime_error(model.error);
    const auto& config = model.config.speech_tokenizer;
    if (config.num_quantizers != kCodeGroups ||
        config.codebook_dim != 2 * kCodebookDimension ||
        config.upsample_rates != std::vector<std::uint32_t>({8, 5, 4, 3}) ||
        config.upsampling_ratios != std::vector<std::uint32_t>({2, 2}))
      throw std::runtime_error(
          "unsupported Qwen3-TTS speech decoder configuration");
    device = gufo::npu::Device::Open();
    gemm = gufo::npu::BfpGemm::Open(device, gufo::npu::FindDesign("bfpgemm"),
                                    std::size_t(512) << 20);
    semantic_table =
        Codebook("decoder.quantizer.rvq_first.vq.layers.0._codebook.");
    for (std::size_t g = 0; g + 1 < kCodeGroups; ++g)
      acoustic_tables.push_back(
          Codebook("decoder.quantizer.rvq_rest.vq.layers." + std::to_string(g) +
                   "._codebook."));
    semantic_projection =
        MakeLinear({"decoder.quantizer.rvq_first.output_proj"}, false);
    acoustic_projection =
        MakeLinear({"decoder.quantizer.rvq_rest.output_proj"}, false);
    pre_conv = MakeConv("decoder.pre_conv.conv", 512, 1024, 3);
    transformer_input = MakeLinear({"decoder.pre_transformer.input_proj"});
    for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
      const std::string p =
          "decoder.pre_transformer.layers." + std::to_string(l);
      TransformerLayer layer;
      layer.input_norm = Vector(p + ".input_layernorm.weight");
      layer.post_norm = Vector(p + ".post_attention_layernorm.weight");
      layer.attention_scale = Vector(p + ".self_attn_layer_scale.scale");
      layer.mlp_scale = Vector(p + ".mlp_layer_scale.scale");
      layer.qkv = MakeLinear({p + ".self_attn.q_proj", p + ".self_attn.k_proj",
                              p + ".self_attn.v_proj"},
                             false);
      layer.o = MakeLinear({p + ".self_attn.o_proj"}, false);
      layer.gate_up =
          MakeLinear({p + ".mlp.gate_proj", p + ".mlp.up_proj"}, false);
      layer.down = MakeLinear({p + ".mlp.down_proj"}, false);
      layers.push_back(std::move(layer));
    }
    transformer_norm = Vector("decoder.pre_transformer.norm.weight");
    transformer_output = MakeLinear({"decoder.pre_transformer.output_proj"});
    for (std::size_t stage = 0; stage < 2; ++stage) {
      const std::string p = "decoder.upsample." + std::to_string(stage);
      UpsampleStage s;
      s.upsample = MakeTranspose(p + ".0.conv", 1024, 1024, 2, 2);
      s.convnext.depthwise_weight = Vector(p + ".1.dwconv.conv.weight");
      s.convnext.depthwise_bias = Vector(p + ".1.dwconv.conv.bias");
      s.convnext.norm_weight = Vector(p + ".1.norm.weight");
      s.convnext.norm_bias = Vector(p + ".1.norm.bias");
      s.convnext.pointwise1 = MakeLinear({p + ".1.pwconv1"});
      s.convnext.pointwise2 = MakeLinear({p + ".1.pwconv2"});
      s.convnext.gamma = Vector(p + ".1.gamma");
      upsample_stages.push_back(std::move(s));
    }
    decoder_input = MakeConv("decoder.decoder.0.conv", 1024, 1536, 7);
    const std::array<std::size_t, 4> rates{8, 5, 4, 3};
    std::size_t channels = 1536;
    for (std::size_t b = 0; b < rates.size(); ++b) {
      const std::size_t out = channels / 2;
      const std::string p =
          "decoder.decoder." + std::to_string(b + 1) + ".block";
      DecoderBlock block;
      block.activation = MakeSnake(p + ".0");
      block.upsample =
          MakeTranspose(p + ".1.conv", channels, out, rates[b] * 2, rates[b]);
      const std::array<std::size_t, 3> dilations{1, 3, 9};
      for (std::size_t u = 0; u < 3; ++u) {
        const std::string up = p + "." + std::to_string(u + 2);
        auto& r = block.residuals[u];
        r.activation1 = MakeSnake(up + ".act1");
        r.conv1 = MakeConv(up + ".conv1.conv", out, out, 7, dilations[u]);
        r.activation2 = MakeSnake(up + ".act2");
        r.conv2 = MakeConv(up + ".conv2.conv", out, out, 1);
      }
      blocks.push_back(std::move(block));
      channels = out;
    }
    output_activation = MakeSnake("decoder.decoder.5");
    output_conv = MakeConv("decoder.decoder.6.conv", 96, 1, 7);
    auto npu = [&](const CpuConv& c) {
      npu_convs.emplace(&c, NpuConv::Make(gemm, c));
    };
    npu(pre_conv);
    for (const UpsampleStage& st : upsample_stages)
      npu(st.upsample);
    npu(decoder_input);
    for (const DecoderBlock& block : blocks) {
      npu(block.upsample);
      for (const ResidualUnit& r : block.residuals) {
        npu(r.conv1);
        npu(r.conv2);
      }
    }
    npu(output_conv);
  }

  // c.Run on the NPU when it has an NPU form.
  void RunConv(const CpuConv& c, const float* x, std::size_t length, float* y) {
    if (auto it = npu_convs.find(&c); it != npu_convs.end())
      it->second.Run(gemm, x, length, y);
    else
      c.Run(x, length, y);
  }

  void Gemm(const Linear& l, const float* x, std::size_t rows, float* y) {
    gemm.Run(l.id, x, int(rows), y);
    if (!l.bias.empty())
      for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t o = 0; o < l.out; ++o)
          y[r * l.out + o] += l.bias[o];
  }

  static void SnakeBeta(const Snake& s, const float* x, float* y,
                        std::size_t rows, std::size_t channels) {
    host::ParallelFor(rows, 64, [&](std::size_t b, std::size_t e) {
      for (std::size_t r = b; r < e; ++r) {
        std::size_t c = 0;
#if defined(__AVX2__) && defined(__FMA__)
        for (; c + 8 <= channels; c += 8) {
          const __m256 v = _mm256_loadu_ps(x + r * channels + c);
          const __m256 sine =
              Sin8(_mm256_mul_ps(v, _mm256_loadu_ps(s.alpha_exp.data() + c)));
          _mm256_storeu_ps(
              y + r * channels + c,
              _mm256_fmadd_ps(_mm256_mul_ps(sine, sine),
                              _mm256_loadu_ps(s.inverse_beta.data() + c), v));
        }
#endif
        for (; c < channels; ++c) {
          const float v = x[r * channels + c];
          const float sine = std::sin(v * s.alpha_exp[c]);
          y[r * channels + c] = v + sine * sine * s.inverse_beta[c];
        }
      }
    });
  }

  // A convolution with the streaming history of hip::SpeechDecoderHipRuntime.
  std::vector<float> Conv(const CpuConv& c, const std::vector<float>& x,
                          std::size_t length) {
    const std::size_t keep = c.Keep();
    std::vector<float> y(length * c.stride * c.out);
    if (active == nullptr || keep == 0) {
      RunConv(c, x.data(), length, y.data());
      return y;
    }
    ConvHistory& h = active->convolutions[&c];
    const std::size_t previous = h.rows, rows = previous + length;
    std::vector<float> joined(rows * c.in);
    std::copy_n(h.input.data(), previous * c.in, joined.data());
    std::copy_n(x.data(), length * c.in, joined.data() + previous * c.in);
    h.rows = std::min(keep, rows);
    h.input.assign(joined.end() - std::ptrdiff_t(h.rows * c.in), joined.end());
    std::vector<float> all(rows * c.stride * c.out);
    RunConv(c, joined.data(), rows, all.data());
    std::copy_n(all.data() + previous * c.stride * c.out, y.size(), y.data());
    return y;
  }

  std::vector<float> DecodeChunk(std::span<const std::uint32_t> codes,
                                 std::size_t frames) {
    const auto& config = model.config.speech_tokenizer;
    const auto t0 = Clock::now();
    // Quantizer: semantic + acoustic sums, projected and added.
    std::vector<float> semantic(frames * kCodebookDimension),
        acoustic(frames * kCodebookDimension, 0.0f);
    for (std::size_t f = 0; f < frames; ++f) {
      const std::uint32_t* fc = codes.data() + f * kCodeGroups;
      std::copy_n(
          semantic_table.data() + std::size_t(fc[0]) * kCodebookDimension,
          kCodebookDimension, semantic.data() + f * kCodebookDimension);
      for (std::size_t g = 1; g < kCodeGroups; ++g) {
        const float* row = acoustic_tables[g - 1].data() +
                           std::size_t(fc[g]) * kCodebookDimension;
        for (std::size_t i = 0; i < kCodebookDimension; ++i)
          acoustic[f * kCodebookDimension + i] += row[i];
      }
    }
    std::vector<float> hidden(frames * 512), projected(frames * 512);
    Gemm(semantic_projection, semantic.data(), frames, hidden.data());
    Gemm(acoustic_projection, acoustic.data(), frames, projected.data());
    for (std::size_t i = 0; i < hidden.size(); ++i)
      hidden[i] += projected[i];
    std::vector<float> pre = Conv(pre_conv, hidden, frames);
    // Transformer.
    const std::size_t heads = config.num_attention_heads,
                      head_dim = config.head_dim, width = heads * head_dim;
    const float eps = config.rms_norm_eps;
    std::vector<float> x(frames * 512), norm(frames * 512),
        qkv(frames * 3 * width), attn(frames * width), proj(frames * 512),
        gate_up(frames * 2 * config.intermediate_size),
        act(frames * config.intermediate_size);
    Gemm(transformer_input, pre.data(), frames, x.data());
    const std::size_t position = active == nullptr ? 0 : active->position;
    std::vector<std::int32_t> positions(frames);
    for (std::size_t f = 0; f < frames; ++f)
      positions[f] = std::int32_t(position + f);
    if (active != nullptr) {
      active->key.resize(layers.size());
      active->value.resize(layers.size());
      if (position + frames > kChunkFrames + kLeftContextFrames)
        throw std::length_error(
            "incremental decoder attention exceeds its window");
    }
    for (std::size_t l = 0; l < layers.size(); ++l) {
      const TransformerLayer& w = layers[l];
      host::RmsNormRows(x.data(), norm.data(), frames, 512, w.input_norm.data(),
                        eps);
      Gemm(w.qkv, norm.data(), frames, qkv.data());
      host::RopeNeox(qkv.data(), frames, 3 * width, 2 * heads, head_dim,
                     head_dim, positions.data(), config.rope_theta);
      // Keys and values of this chunk, after any cached ones.
      std::vector<float> local_k, local_v;
      const float* keys;
      const float* values;
      std::size_t key_stride;
      if (active != nullptr) {
        auto& kc = active->key[l];
        auto& vc = active->value[l];
        kc.resize((position + frames) * width);
        vc.resize((position + frames) * width);
        for (std::size_t f = 0; f < frames; ++f) {
          std::copy_n(qkv.data() + f * 3 * width + width, width,
                      kc.data() + (position + f) * width);
          std::copy_n(qkv.data() + f * 3 * width + 2 * width, width,
                      vc.data() + (position + f) * width);
        }
        keys = kc.data();
        values = vc.data();
        key_stride = width;
      } else {
        keys = qkv.data() + width;
        values = qkv.data() + 2 * width;
        key_stride = 3 * width;
      }
      const std::size_t window = config.sliding_window;
      host::Attention({.q = qkv.data(),
                       .q_stride = 3 * width,
                       .k = keys,
                       .k_stride = key_stride,
                       .v = values,
                       .v_stride = key_stride,
                       .out = attn.data(),
                       .out_stride = width,
                       .rows = frames,
                       .heads = heads,
                       .kv_heads = heads,
                       .dim = head_dim,
                       .dim_v = head_dim,
                       .scale = 1.0F / std::sqrt(float(head_dim))},
                      [&](std::size_t r, std::size_t* b, std::size_t* e) {
                        // Key indices are absolute positions in the cache (or
                        // the chunk when no history is kept).
                        const std::size_t q =
                            (active != nullptr ? position : 0) + r;
                        *b = q + 1 > window ? q + 1 - window : 0;
                        *e = q + 1;
                      });
      Gemm(w.o, attn.data(), frames, proj.data());
      for (std::size_t r = 0; r < frames; ++r)
        for (std::size_t i = 0; i < 512; ++i)
          x[r * 512 + i] += proj[r * 512 + i] * w.attention_scale[i];
      host::RmsNormRows(x.data(), norm.data(), frames, 512, w.post_norm.data(),
                        eps);
      Gemm(w.gate_up, norm.data(), frames, gate_up.data());
      const std::size_t inner = config.intermediate_size;
      host::GatedRows(gate_up.data(), 2 * inner, gate_up.data() + inner,
                      2 * inner, act.data(), inner, frames, inner,
                      host::Activation::kSilu);
      Gemm(w.down, act.data(), frames, proj.data());
      for (std::size_t r = 0; r < frames; ++r)
        for (std::size_t i = 0; i < 512; ++i)
          x[r * 512 + i] += proj[r * 512 + i] * w.mlp_scale[i];
    }
    host::RmsNormRows(x.data(), norm.data(), frames, 512,
                      transformer_norm.data(), eps);
    std::vector<float> h(frames * 1024);
    Gemm(transformer_output, norm.data(), frames, h.data());
    const auto t_transformer = Clock::now();
    // Upsampling stages with ConvNeXt blocks.
    std::size_t length = frames;
    for (const UpsampleStage& s : upsample_stages) {
      std::vector<float> up = Conv(s.upsample, h, length);
      length *= s.upsample.stride;
      std::vector<float> dw(length * 1024), normed(length * 1024),
          wide(length * 4096), back(length * 1024);
      // The depthwise convolution keeps 6 rows of history.
      {
        const std::size_t keep = 6;
        if (active == nullptr) {
          DepthwiseConv(up.data(), length, 1024,
                        s.convnext.depthwise_weight.data(),
                        s.convnext.depthwise_bias.data(), 7, dw.data());
        } else {
          ConvHistory& hist = active->convolutions[&s.convnext];
          const std::size_t previous = hist.rows, rows = previous + length;
          std::vector<float> joined(rows * 1024), all(rows * 1024);
          std::copy_n(hist.input.data(), previous * 1024, joined.data());
          std::copy_n(up.data(), length * 1024,
                      joined.data() + previous * 1024);
          hist.rows = std::min(keep, rows);
          hist.input.assign(joined.end() - std::ptrdiff_t(hist.rows * 1024),
                            joined.end());
          DepthwiseConv(joined.data(), rows, 1024,
                        s.convnext.depthwise_weight.data(),
                        s.convnext.depthwise_bias.data(), 7, all.data());
          std::copy_n(all.data() + previous * 1024, length * 1024, dw.data());
        }
      }
      host::LayerNormRows(dw.data(), normed.data(), length, 1024,
                          s.convnext.norm_weight.data(),
                          s.convnext.norm_bias.data(), 1.0e-6F);
      Gemm(s.convnext.pointwise1, normed.data(), length, wide.data());
      host::ActivateRows(wide.data(), length, 4096, nullptr,
                         host::Activation::kGeluErf);
      Gemm(s.convnext.pointwise2, wide.data(), length, back.data());
      for (std::size_t r = 0; r < length; ++r)
        for (std::size_t i = 0; i < 1024; ++i)
          up[r * 1024 + i] += back[r * 1024 + i] * s.convnext.gamma[i];
      h = std::move(up);
    }
    std::vector<float> hidden_d = Conv(decoder_input, h, length);
    std::size_t channels = 1536;
    for (const DecoderBlock& block : blocks) {
      std::vector<float> activated(length * channels);
      SnakeBeta(block.activation, hidden_d.data(), activated.data(), length,
                channels);
      std::vector<float> up = Conv(block.upsample, activated, length);
      length *= block.upsample.stride;
      channels /= 2;
      std::vector<float> branch(length * channels);
      for (const ResidualUnit& r : block.residuals) {
        SnakeBeta(r.activation1, up.data(), branch.data(), length, channels);
        std::vector<float> work = Conv(r.conv1, branch, length);
        SnakeBeta(r.activation2, work.data(), branch.data(), length, channels);
        work = Conv(r.conv2, branch, length);
        for (std::size_t i = 0; i < up.size(); ++i)
          up[i] += work[i];
      }
      hidden_d = std::move(up);
    }
    std::vector<float> activated(length * 96);
    SnakeBeta(output_activation, hidden_d.data(), activated.data(), length, 96);
    std::vector<float> samples = Conv(output_conv, activated, length);
    for (float& s : samples)
      s = std::clamp(s, -1.0F, 1.0F);
    if (active != nullptr)
      active->position += frames;
    if (profile)
      std::fprintf(
          stderr,
          "qwen3_tts npu decoder: %zu frames, transformer %.1f ms, "
          "total %.1f ms\n",
          frames,
          std::chrono::duration<double, std::milli>(t_transformer - t0).count(),
          std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    return samples;
  }
};

SpeechDecoderNpuRuntime::SpeechDecoderNpuRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
SpeechDecoderNpuRuntime::~SpeechDecoderNpuRuntime() = default;

std::unique_ptr<SpeechDecoderNpuRuntime> SpeechDecoderNpuRuntime::Create(
    const std::string& model_root, std::string* error) {
  auto impl = std::make_unique<Impl>();
  try {
    impl->Load(model_root);
  } catch (const std::exception& e) {
    SetError(error, std::string("Qwen3-TTS NPU decoder: ") + e.what());
    return nullptr;
  }
  return std::unique_ptr<SpeechDecoderNpuRuntime>(
      new SpeechDecoderNpuRuntime(std::move(impl)));
}

// As SpeechDecoderHipRuntime::DecodeIncremental.
bool SpeechDecoderNpuRuntime::DecodeIncremental(
    std::span<const std::uint32_t> codes, std::size_t frames,
    std::size_t begin_frame, hip::SpeechDecoderOutput* output,
    std::string* error) {
  if (output == nullptr || frames == 0 || begin_frame > frames ||
      frames > std::numeric_limits<std::size_t>::max() / kSamplesPerCode ||
      codes.size() != frames * kCodeGroups) {
    SetError(error, "invalid incremental speech decoder shape");
    return false;
  }
  *output = {};
  auto& history = impl_->history;
  if (begin_frame == 0) {
    history.codes.clear();
    history.ResetBlock();
  } else if (history.codes.size() != begin_frame * kCodeGroups ||
             !std::equal(history.codes.begin(), history.codes.end(),
                         codes.begin())) {
    SetError(error, "incremental speech decoder prefix does not match");
    return false;
  }
  for (const auto code : codes)
    if (code >= impl_->model.config.speech_tokenizer.codebook_size) {
      SetError(error, "incremental speech decoder code is out of range");
      return false;
    }
  const auto started = Clock::now();
  impl_->active = &history;
  struct ResetActive {
    Impl* impl;
    ~ResetActive() { impl->active = nullptr; }
  } reset{impl_.get()};
  try {
    for (std::size_t cursor = begin_frame; cursor < frames;) {
      if (cursor % kChunkFrames == 0) {
        history.ResetBlock();
        for (std::size_t warm = cursor - std::min(cursor, kLeftContextFrames);
             warm < cursor;) {
          const auto count = std::min<std::size_t>(16, cursor - warm);
          (void)impl_->DecodeChunk(
              codes.subspan(warm * kCodeGroups, count * kCodeGroups), count);
          warm += count;
        }
      }
      const auto count = std::min({std::size_t{16}, frames - cursor,
                                   kChunkFrames - cursor % kChunkFrames});
      const auto chunk =
          codes.subspan(cursor * kCodeGroups, count * kCodeGroups);
      auto samples = impl_->DecodeChunk(chunk, count);
      output->samples.insert(output->samples.end(), samples.begin(),
                             samples.end());
      history.codes.insert(history.codes.end(), chunk.begin(), chunk.end());
      cursor += count;
    }
    output->sample_rate =
        impl_->model.config.speech_tokenizer.output_sample_rate;
    output->code_frames = frames - begin_frame;
    output->decode_milliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count();
    return true;
  } catch (const std::exception& exception) {
    history.codes.clear();
    history.ResetBlock();
    *output = {};
    SetError(error, exception.what());
    return false;
  }
}

// As SpeechDecoderHipRuntime::Decode.
bool SpeechDecoderNpuRuntime::Decode(std::span<const std::uint32_t> codes,
                                     std::size_t frames,
                                     hip::SpeechDecoderOutput* output,
                                     hip::SpeechDecoderTrace* trace,
                                     std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS speech decoder output is null");
    return false;
  }
  if (trace != nullptr) {
    SetError(error, "Qwen3-TTS NPU speech decoder has no trace");
    return false;
  }
  if (frames == 0 ||
      frames > std::numeric_limits<std::size_t>::max() / kCodeGroups ||
      codes.size() != frames * kCodeGroups) {
    SetError(error, "Qwen3-TTS speech decoder expects [frames,16] codes");
    return false;
  }
  for (const std::uint32_t code : codes)
    if (code >= impl_->model.config.speech_tokenizer.codebook_size) {
      SetError(error, "Qwen3-TTS speech decoder code is out of range");
      return false;
    }
  try {
    const auto begin = Clock::now();
    output->samples.clear();
    output->samples.reserve(frames * kSamplesPerCode);
    for (std::size_t start = 0; start < frames; start += kChunkFrames) {
      const std::size_t end = std::min(start + kChunkFrames, frames);
      const std::size_t context = std::min(start, kLeftContextFrames);
      const std::size_t chunk_start = start - context;
      const std::size_t chunk_frames = end - chunk_start;
      const auto chunk_codes =
          codes.subspan(chunk_start * kCodeGroups, chunk_frames * kCodeGroups);
      std::vector<float> decoded =
          impl_->DecodeChunk(chunk_codes, chunk_frames);
      const std::size_t drop = context * kSamplesPerCode;
      const std::size_t keep = (end - start) * kSamplesPerCode;
      if (drop > decoded.size() || keep > decoded.size() - drop)
        throw std::runtime_error(
            "Qwen3-TTS speech decoder chunk output is truncated");
      output->samples.insert(
          output->samples.end(),
          decoded.begin() + static_cast<std::ptrdiff_t>(drop),
          decoded.begin() + static_cast<std::ptrdiff_t>(drop + keep));
    }
    output->sample_rate =
        impl_->model.config.speech_tokenizer.output_sample_rate;
    output->code_frames = frames;
    output->decode_milliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

// As SpeechDecoderHipRuntime::DecodeAfterReference.
bool SpeechDecoderNpuRuntime::DecodeAfterReference(
    std::span<const std::uint32_t> codes, std::size_t frames,
    std::size_t reference_frames, hip::SpeechDecoderOutput* output,
    std::string* error) {
  if (output == nullptr || reference_frames >= frames ||
      frames > std::numeric_limits<std::size_t>::max() / kSamplesPerCode ||
      codes.size() != frames * kCodeGroups) {
    SetError(error, "invalid speech decoder reference shape");
    return false;
  }
  *output = {};
  const auto started = Clock::now();
  const std::size_t prefix_frames =
      reference_frames - (reference_frames % kChunkFrames) % 16;
  try {
    if (prefix_frames != 0) {
      const auto prefix = codes.first(prefix_frames * kCodeGroups);
      auto& saved = impl_->reference_history;
      if (saved == nullptr || saved->codes.size() != prefix.size() ||
          !std::equal(prefix.begin(), prefix.end(), saved->codes.begin())) {
        hip::SpeechDecoderOutput discarded;
        if (!DecodeIncremental(prefix, prefix_frames, 0, &discarded, error))
          return false;
        saved = std::make_unique<DecoderHistory>(impl_->history);
      } else {
        impl_->history = *saved;
      }
    }
    if (!DecodeIncremental(codes, frames, prefix_frames, output, error))
      return false;
    const std::size_t drop =
        (reference_frames - prefix_frames) * kSamplesPerCode;
    output->samples.erase(
        output->samples.begin(),
        output->samples.begin() + static_cast<std::ptrdiff_t>(drop));
    output->code_frames = frames - reference_frames;
    output->decode_milliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count();
    return true;
  } catch (const std::exception& exception) {
    impl_->history.codes.clear();
    impl_->history.ResetBlock();
    *output = {};
    SetError(error, exception.what());
    return false;
  }
}

}  // namespace gufo::models::qwen3_tts::npu

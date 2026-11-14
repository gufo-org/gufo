#include "src/models/qwen3_tts/npu/speech_encoder_npu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gufo/ext/experimental/npu.hpp"
#include "src/models/qwen3_tts/audio.hpp"
#include "src/models/qwen3_tts/loader.hpp"
#include "src/models/qwen3_tts/npu/cpu_conv.hpp"
#include "src/models/qwen3_tts/npu/npu_common.hpp"
#include "src/models/qwen3_tts/npu/npu_conv.hpp"

namespace gufo::models::qwen3_tts::npu {
namespace {

namespace host = gufo::npu::host;

constexpr std::size_t kSampleRate = 24000;
constexpr std::size_t kHidden = 512;
constexpr std::size_t kQuantizerDimension = 256;
constexpr std::size_t kCodebookSize = 2048;
constexpr std::size_t kCodeGroups = 16;

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

void RoundAll(std::vector<float>& v) {
  for (float& x : v)
    x = Bf16(x);
}

void Elu(std::vector<float>& v) {
  for (float& x : v) {
    if (x < 0.0F)
      x = std::expm1(x);
    x = Bf16(x);
  }
}

// A strided convolution with left padding (k - 1) d + 1 - stride and
// ceil(length / stride) outputs, as SpeechEncoderHipRuntime::Impl::RunConv.
struct Conv {
  CpuConv conv;
  std::vector<float> Run(const std::vector<float>& x,
                         std::size_t length) const {
    const std::size_t out_length = (length + conv.step - 1) / conv.step;
    std::vector<float> y(out_length * conv.out);
    EncoderNpu::Get().RunGeneral(conv, x.data(), length, y.data(), out_length);
    RoundAll(y);
    return y;
  }
  [[nodiscard]] std::size_t Length(std::size_t length) const {
    return (length + conv.step - 1) / conv.step;
  }
};

struct Residual {
  Conv first, second;
};

struct TransformerLayer {
  std::vector<float> norm1_weight, norm1_bias, norm2_weight, norm2_bias,
      attention_scale, mlp_scale;
  Conv query, key, value, output, fc1, fc2;  // kernel 1
};

}  // namespace

struct SpeechEncoderNpuRuntime::Impl {
  LoadResult model;
  std::vector<Conv> convolutions;
  std::vector<Residual> residuals;
  std::vector<TransformerLayer> transformers;
  Conv downsample, semantic_projection, acoustic_projection;
  std::vector<float> semantic_codebook;                // [dim][code]
  std::vector<std::vector<float>> acoustic_codebooks;  // 15 x [dim][code]

  std::vector<float> Weight(const std::string& name) const {
    const Tensor* t = model.store->Find(name);
    if (t == nullptr)
      throw std::runtime_error("missing Qwen3-TTS encoder tensor " + name);
    std::vector<float> v = FloatVector(*t);
    RoundAll(v);
    return v;
  }
  Conv LoadConv(const std::string& prefix, std::size_t in, std::size_t out,
                std::size_t kernel, std::size_t stride = 1, bool bias = true,
                CpuConv::Pad pad = CpuConv::Pad::kZero) {
    const std::vector<float> w = Weight(prefix + ".weight");
    if (w.size() != out * in * kernel)
      throw std::runtime_error("unexpected Qwen3-TTS encoder tensor " + prefix);
    std::vector<float> b;
    if (bias)
      b = Weight(prefix + ".bias");
    return {CpuConv::MakeGeneral(w.data(), bias ? b.data() : nullptr, in, out,
                                 kernel, 1, stride, kernel - stride, pad)};
  }
  Conv LoadLinear(const std::string& prefix, std::size_t in, std::size_t out) {
    return LoadConv(prefix, in, out, 1, 1, false);
  }
  std::vector<float> Codebook(const std::string& prefix) const {
    const Tensor* sum = model.store->Find(prefix + "embed_sum");
    const Tensor* usage = model.store->Find(prefix + "cluster_usage");
    if (sum == nullptr || usage == nullptr)
      throw std::runtime_error("missing Qwen3-TTS encoder codebook " + prefix);
    const std::vector<float> sums = FloatVector(*sum),
                             uses = FloatVector(*usage);
    std::vector<float> embedding(kCodebookSize * kQuantizerDimension);
    for (std::size_t code = 0; code < kCodebookSize; ++code) {
      const float d = std::max(Bf16(uses[code]), 1.0e-5F);
      for (std::size_t c = 0; c < kQuantizerDimension; ++c)
        embedding[c * kCodebookSize + code] =
            Bf16(Bf16(sums[code * kQuantizerDimension + c]) / d);
    }
    return embedding;
  }

  void Load() {
    const std::array<const char*, 6> names{
        "encoder.encoder.layers.0.conv",  "encoder.encoder.layers.3.conv",
        "encoder.encoder.layers.6.conv",  "encoder.encoder.layers.9.conv",
        "encoder.encoder.layers.12.conv", "encoder.encoder.layers.14.conv",
    };
    const std::array<std::array<std::size_t, 4>, 6> shapes{{
        {1, 64, 7, 1},
        {64, 128, 8, 4},
        {128, 256, 10, 5},
        {256, 512, 12, 6},
        {512, 1024, 16, 8},
        {1024, 512, 3, 1},
    }};
    for (std::size_t i = 0; i < names.size(); ++i)
      convolutions.push_back(LoadConv(names[i], shapes[i][0], shapes[i][1],
                                      shapes[i][2], shapes[i][3]));
    const std::array<std::size_t, 4> indices{1, 4, 7, 10};
    const std::array<std::size_t, 4> channels{64, 128, 256, 512};
    for (std::size_t i = 0; i < indices.size(); ++i) {
      const std::string p =
          "encoder.encoder.layers." + std::to_string(indices[i]) + ".block.";
      residuals.push_back(
          {LoadConv(p + "1.conv", channels[i], channels[i] / 2, 3),
           LoadConv(p + "3.conv", channels[i] / 2, channels[i], 1)});
    }
    for (std::size_t layer = 0; layer < 8; ++layer) {
      const std::string p =
          "encoder.encoder_transformer.layers." + std::to_string(layer);
      TransformerLayer t;
      t.norm1_weight = Weight(p + ".input_layernorm.weight");
      t.norm1_bias = Weight(p + ".input_layernorm.bias");
      t.norm2_weight = Weight(p + ".post_attention_layernorm.weight");
      t.norm2_bias = Weight(p + ".post_attention_layernorm.bias");
      t.attention_scale = Weight(p + ".self_attn_layer_scale.scale");
      t.mlp_scale = Weight(p + ".mlp_layer_scale.scale");
      t.query = LoadLinear(p + ".self_attn.q_proj", 512, 512);
      t.key = LoadLinear(p + ".self_attn.k_proj", 512, 512);
      t.value = LoadLinear(p + ".self_attn.v_proj", 512, 512);
      t.output = LoadLinear(p + ".self_attn.o_proj", 512, 512);
      t.fc1 = LoadLinear(p + ".mlp.fc1", 512, 2048);
      t.fc2 = LoadLinear(p + ".mlp.fc2", 2048, 512);
      transformers.push_back(std::move(t));
    }
    downsample = LoadConv("encoder.downsample.conv", 512, 512, 4, 2, false,
                          CpuConv::Pad::kReplicate);
    semantic_projection = LoadConv(
        "encoder.quantizer.semantic_residual_vector_quantizer.input_proj", 512,
        256, 1, 1, false);
    acoustic_projection = LoadConv(
        "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj", 512,
        256, 1, 1, false);
    semantic_codebook = Codebook(
        "encoder.quantizer.semantic_residual_vector_quantizer.layers.0."
        "codebook.");
    for (std::size_t g = 0; g + 1 < kCodeGroups; ++g)
      acoustic_codebooks.push_back(Codebook(
          "encoder.quantizer.acoustic_residual_vector_quantizer.layers." +
          std::to_string(g) + ".codebook."));
  }

  // Nearest code per frame (ties to the lower index), as
  // QuantizeCodebookKernel; optionally subtracts it from the residual.
  static void Quantize(std::vector<float>& residual,
                       const std::vector<float>& embedding,
                       std::vector<std::uint32_t>& codes, std::size_t frames,
                       std::size_t group, bool update) {
    host::ParallelFor(frames, 1, [&](std::size_t b, std::size_t e) {
      for (std::size_t f = b; f < e; ++f) {
        float best = std::numeric_limits<float>::infinity();
        std::uint32_t best_code = 0;
        const float* r = residual.data() + f * kQuantizerDimension;
        for (std::size_t code = 0; code < kCodebookSize; ++code) {
          float distance = 0;
          for (std::size_t d = 0; d < kQuantizerDimension; ++d) {
            const float diff = r[d] - embedding[d * kCodebookSize + code];
            distance = std::fma(diff, diff, distance);
          }
          if (distance < best) {
            best = distance;
            best_code = std::uint32_t(code);
          }
        }
        codes[f * kCodeGroups + group] = best_code;
        if (update)
          for (std::size_t d = 0; d < kQuantizerDimension; ++d)
            residual[f * kQuantizerDimension + d] =
                Bf16(r[d] - embedding[d * kCodebookSize + best_code]);
      }
    });
  }
};

SpeechEncoderNpuRuntime::SpeechEncoderNpuRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
SpeechEncoderNpuRuntime::~SpeechEncoderNpuRuntime() = default;

std::unique_ptr<SpeechEncoderNpuRuntime> SpeechEncoderNpuRuntime::Create(
    const std::string& model_root, std::string* error) {
  auto impl = std::make_unique<Impl>();
  try {
    impl->model = LoadModelDirectory(model_root);
    if (!impl->model.ok)
      throw std::runtime_error(impl->model.error);
    if (impl->model.config.variant != ModelVariant::kBase)
      throw std::invalid_argument(
          "Qwen3-TTS speech encoding requires a Base model");
    impl->Load();
  } catch (const std::exception& e) {
    SetError(error, std::string("Qwen3-TTS NPU speech encoder: ") + e.what());
    return nullptr;
  }
  return std::unique_ptr<SpeechEncoderNpuRuntime>(
      new SpeechEncoderNpuRuntime(std::move(impl)));
}

// As SpeechEncoderHipRuntime::Encode.
bool SpeechEncoderNpuRuntime::Encode(const AudioBuffer& audio,
                                     hip::SpeechEncoderOutput* output,
                                     std::string* error,
                                     hip::SpeechEncoderTrace* trace) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS speech encoder output must not be null");
    return false;
  }
  *output = {};
  if (trace != nullptr) {
    SetError(error, "Qwen3-TTS NPU speech encoder has no trace");
    return false;
  }
  try {
    const Impl& m = *impl_;
    std::vector<float> x = ResampleMono(audio, kSampleRate);
    if (x.empty())
      throw std::invalid_argument(
          "Qwen3-TTS speech encoder requires reference audio");
    RoundAll(x);
    std::size_t length = x.size();
    std::vector<float> current = m.convolutions[0].Run(x, length);
    length = m.convolutions[0].Length(length);
    for (std::size_t i = 0; i < m.residuals.size(); ++i) {
      const Residual& r = m.residuals[i];
      std::vector<float> branch = current;
      Elu(branch);
      branch = r.first.Run(branch, length);
      Elu(branch);
      branch = r.second.Run(branch, length);
      for (std::size_t j = 0; j < current.size(); ++j)
        current[j] = Bf16(current[j] + branch[j]);
      Elu(current);
      const Conv& next = m.convolutions[i + 1];
      current = next.Run(current, length);
      length = next.Length(length);
    }
    Elu(current);
    current = m.convolutions.back().Run(current, length);
    length = m.convolutions.back().Length(length);

    const auto& tokenizer = m.model.config.speech_tokenizer;
    const std::size_t heads = tokenizer.encoder_num_attention_heads,
                      head_dim = tokenizer.encoder_head_dim;
    std::vector<std::int32_t> positions(length);
    for (std::size_t f = 0; f < length; ++f)
      positions[f] = std::int32_t(f);
    std::vector<float> normalized(length * kHidden),
        attention(length * kHidden);
    for (const TransformerLayer& layer : m.transformers) {
      host::LayerNormRows(current.data(), normalized.data(), length, kHidden,
                          layer.norm1_weight.data(), layer.norm1_bias.data(),
                          tokenizer.encoder_norm_eps);
      RoundAll(normalized);
      std::vector<float> q = layer.query.Run(normalized, length);
      std::vector<float> k = layer.key.Run(normalized, length);
      const std::vector<float> v = layer.value.Run(normalized, length);
      host::RopeNeox(q.data(), length, kHidden, heads, head_dim, head_dim,
                     positions.data(), tokenizer.encoder_rope_theta);
      host::RopeNeox(k.data(), length, kHidden, heads, head_dim, head_dim,
                     positions.data(), tokenizer.encoder_rope_theta);
      RoundAll(q);
      RoundAll(k);
      const std::size_t window = tokenizer.encoder_sliding_window;
      host::Attention({.q = q.data(),
                       .q_stride = kHidden,
                       .k = k.data(),
                       .k_stride = kHidden,
                       .v = v.data(),
                       .v_stride = kHidden,
                       .out = attention.data(),
                       .out_stride = kHidden,
                       .rows = length,
                       .heads = heads,
                       .kv_heads = heads,
                       .dim = head_dim,
                       .dim_v = head_dim,
                       .scale = 1.0F / std::sqrt(float(head_dim))},
                      [&](std::size_t r, std::size_t* b, std::size_t* e) {
                        *b = r + 1 > window ? r + 1 - window : 0;
                        *e = r + 1;
                      });
      RoundAll(attention);
      std::vector<float> o = layer.output.Run(attention, length);
      for (std::size_t r = 0; r < length; ++r)
        for (std::size_t c = 0; c < kHidden; ++c) {
          float& h = current[r * kHidden + c];
          h = Bf16(std::fma(o[r * kHidden + c], layer.attention_scale[c], h));
        }
      host::LayerNormRows(current.data(), normalized.data(), length, kHidden,
                          layer.norm2_weight.data(), layer.norm2_bias.data(),
                          tokenizer.encoder_norm_eps);
      RoundAll(normalized);
      std::vector<float> ffn = layer.fc1.Run(normalized, length);
      for (float& f : ffn)
        f = Bf16(host::Activate(host::Activation::kGeluErf, f));
      std::vector<float> back = layer.fc2.Run(ffn, length);
      for (std::size_t r = 0; r < length; ++r)
        for (std::size_t c = 0; c < kHidden; ++c) {
          float& h = current[r * kHidden + c];
          h = Bf16(std::fma(back[r * kHidden + c], layer.mlp_scale[c], h));
        }
    }
    current = m.downsample.Run(current, length);
    length = m.downsample.Length(length);
    std::vector<float> semantic = m.semantic_projection.Run(current, length);
    std::vector<float> acoustic = m.acoustic_projection.Run(current, length);
    output->frames = length;
    output->code_groups = kCodeGroups;
    output->codes.assign(length * kCodeGroups, 0);
    Impl::Quantize(semantic, m.semantic_codebook, output->codes, length, 0,
                   false);
    for (std::size_t g = 1; g < kCodeGroups; ++g)
      Impl::Quantize(acoustic, m.acoustic_codebooks[g - 1], output->codes,
                     length, g, true);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    *output = {};
    return false;
  }
}

}  // namespace gufo::models::qwen3_tts::npu

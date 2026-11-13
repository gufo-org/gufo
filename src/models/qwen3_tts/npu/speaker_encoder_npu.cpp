#include "src/models/qwen3_tts/npu/speaker_encoder_npu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/models/qwen3_tts/loader.hpp"
#include "src/models/qwen3_tts/npu/cpu_conv.hpp"
#include "src/models/qwen3_tts/npu/npu_common.hpp"
#include "src/models/qwen3_tts/npu/npu_conv.hpp"
#include "src/models/qwen3_tts/speaker_features.hpp"

namespace gufo::models::qwen3_tts::npu {
namespace {

constexpr std::size_t kChannels = 512;
constexpr std::size_t kScale = 8;
constexpr std::size_t kWidth = kChannels / kScale;

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

void RoundAll(std::vector<float>& v) {
  for (float& x : v)
    x = Bf16(x);
}

struct Res2Net {
  CpuConv tdnn1, tdnn2, se1, se2;
  std::array<CpuConv, kScale - 1> blocks;
};

}  // namespace

struct SpeakerEncoderNpuRuntime::Impl {
  LoadResult model;
  CpuConv block0, mfa, asp_tdnn, asp_conv, fc;
  std::vector<Res2Net> blocks;

  // Reflect-padded "same" convolution with BF16-rounded weights, as
  // SpeakerEncoderHipRuntime::Impl::LoadConv/RunConv.
  CpuConv Load(const std::string& suffix, std::size_t in, std::size_t out,
               std::size_t kernel, std::size_t dilation = 1) {
    const std::string prefix = "speaker_encoder." + suffix;
    const Tensor* w = model.store->Find(prefix + ".weight");
    const Tensor* b = model.store->Find(prefix + ".bias");
    if (w == nullptr || b == nullptr)
      throw std::runtime_error("missing Qwen3-TTS speaker tensor " + prefix);
    std::vector<float> weight = FloatVector(*w), bias = FloatVector(*b);
    RoundAll(weight);
    RoundAll(bias);
    if (weight.size() != out * in * kernel || bias.size() != out)
      throw std::runtime_error("unexpected Qwen3-TTS speaker tensor " + prefix);
    return CpuConv::MakeGeneral(weight.data(), bias.data(), in, out, kernel,
                                dilation, 1, dilation * (kernel - 1) / 2,
                                CpuConv::Pad::kReflect);
  }

  std::vector<float> Conv(const CpuConv& c, const std::vector<float>& x,
                          std::size_t rows, bool relu) const {
    std::vector<float> y(rows * c.out);
    EncoderNpu::Get().RunGeneral(c, x.data(), rows, y.data(), rows);
    for (float& v : y) {
      v = Bf16(v);
      if (relu)
        v = std::max(v, 0.0F);
    }
    return y;
  }

  std::vector<float> Block(const Res2Net& b, const std::vector<float>& input,
                           std::size_t rows) const {
    const std::vector<float> first = Conv(b.tdnn1, input, rows, true);
    std::vector<float> third(rows * kChannels), previous, chunk(rows * kWidth);
    for (std::size_t split = 0; split < kScale; ++split) {
      for (std::size_t r = 0; r < rows; ++r)
        std::copy_n(first.data() + r * kChannels + split * kWidth, kWidth,
                    chunk.data() + r * kWidth);
      std::vector<float> branch = chunk;
      if (split > 0) {
        if (split > 1)
          for (std::size_t i = 0; i < branch.size(); ++i)
            branch[i] += previous[i];
        branch = Conv(b.blocks[split - 1], branch, rows, true);
      }
      previous = branch;
      for (std::size_t r = 0; r < rows; ++r)
        std::copy_n(branch.data() + r * kWidth, kWidth,
                    third.data() + r * kChannels + split * kWidth);
    }
    std::vector<float> out = Conv(b.tdnn2, third, rows, true);
    std::vector<float> mean(kChannels, 0.0F);
    for (std::size_t c = 0; c < kChannels; ++c) {
      float sum = 0;
      for (std::size_t r = 0; r < rows; ++r)
        sum += out[r * kChannels + c];
      mean[c] = sum / float(rows);
    }
    const std::vector<float> squeezed = Conv(b.se1, mean, 1, true);
    const std::vector<float> scale = Conv(b.se2, squeezed, 1, false);
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t c = 0; c < kChannels; ++c) {
        float& v = out[r * kChannels + c];
        v *= 1.0F / (1.0F + std::exp(-scale[c]));
        v = Bf16(v + input[r * kChannels + c]);
      }
    return out;
  }
};

SpeakerEncoderNpuRuntime::SpeakerEncoderNpuRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
SpeakerEncoderNpuRuntime::~SpeakerEncoderNpuRuntime() = default;

std::unique_ptr<SpeakerEncoderNpuRuntime> SpeakerEncoderNpuRuntime::Create(
    const std::string& model_root, std::string* error) {
  auto impl = std::make_unique<Impl>();
  try {
    impl->model = LoadModelDirectory(model_root);
    if (!impl->model.ok)
      throw std::runtime_error(impl->model.error);
    if (impl->model.config.variant != ModelVariant::kBase ||
        !impl->model.config.speaker_encoder.has_value())
      throw std::invalid_argument(
          "Qwen3-TTS speaker encoding requires a Base model");
    Impl& m = *impl;
    m.block0 = m.Load("blocks.0.conv", 128, 512, 5);
    for (std::size_t block = 1; block <= 3; ++block) {
      const std::string p = "blocks." + std::to_string(block);
      Res2Net r;
      r.tdnn1 = m.Load(p + ".tdnn1.conv", 512, 512, 1);
      r.tdnn2 = m.Load(p + ".tdnn2.conv", 512, 512, 1);
      r.se1 = m.Load(p + ".se_block.conv1", 512, 128, 1);
      r.se2 = m.Load(p + ".se_block.conv2", 128, 512, 1);
      for (std::size_t i = 0; i < r.blocks.size(); ++i)
        r.blocks[i] =
            m.Load(p + ".res2net_block.blocks." + std::to_string(i) + ".conv",
                   64, 64, 3, block + 1);
      m.blocks.push_back(std::move(r));
    }
    m.mfa = m.Load("mfa.conv", 1536, 1536, 1);
    m.asp_tdnn = m.Load("asp.tdnn.conv", 4608, 128, 1);
    m.asp_conv = m.Load("asp.conv", 128, 1536, 1);
    m.fc = m.Load("fc", 3072, 2048, 1);
  } catch (const std::exception& e) {
    SetError(error, std::string("Qwen3-TTS NPU speaker encoder: ") + e.what());
    return nullptr;
  }
  return std::unique_ptr<SpeakerEncoderNpuRuntime>(
      new SpeakerEncoderNpuRuntime(std::move(impl)));
}

// As SpeakerEncoderHipRuntime::Encode.
bool SpeakerEncoderNpuRuntime::Encode(const AudioBuffer& audio,
                                      hip::SpeakerEncoderOutput* output,
                                      std::string* error,
                                      hip::SpeakerEncoderTrace* trace) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS speaker encoder output must not be null");
    return false;
  }
  *output = {};
  if (trace != nullptr) {
    SetError(error, "Qwen3-TTS NPU speaker encoder has no trace");
    return false;
  }
  try {
    const Impl& m = *impl_;
    SpeakerFeatures features = ComputeSpeakerFeatures(audio);
    const std::size_t rows = features.frames;
    RoundAll(features.values);
    const std::vector<float> x0 = m.Conv(m.block0, features.values, rows, true);
    const std::vector<float> h1 = m.Block(m.blocks[0], x0, rows);
    const std::vector<float> h2 = m.Block(m.blocks[1], h1, rows);
    const std::vector<float> h3 = m.Block(m.blocks[2], h2, rows);
    std::vector<float> cat(rows * 3 * kChannels);
    for (std::size_t r = 0; r < rows; ++r) {
      std::copy_n(h1.data() + r * kChannels, kChannels,
                  cat.data() + r * 3 * kChannels);
      std::copy_n(h2.data() + r * kChannels, kChannels,
                  cat.data() + r * 3 * kChannels + kChannels);
      std::copy_n(h3.data() + r * kChannels, kChannels,
                  cat.data() + r * 3 * kChannels + 2 * kChannels);
    }
    const std::vector<float> mfa = m.Conv(m.mfa, cat, rows, true);
    constexpr std::size_t kC = 1536;
    constexpr float kEps = 1.0e-12F;
    std::vector<float> stats_in(rows * 3 * kC);
    for (std::size_t c = 0; c < kC; ++c) {
      float sum = 0, square = 0;
      for (std::size_t r = 0; r < rows; ++r) {
        const float v = mfa[r * kC + c];
        sum += v;
        square += v * v;
      }
      const float mean = sum / float(rows);
      const float variance = std::max(square / float(rows) - mean * mean, 0.0F);
      const float deviation = std::sqrt(variance + kEps);
      for (std::size_t r = 0; r < rows; ++r) {
        stats_in[r * 3 * kC + c] = mfa[r * kC + c];
        stats_in[r * 3 * kC + kC + c] = mean;
        stats_in[r * 3 * kC + 2 * kC + c] = deviation;
      }
    }
    RoundAll(stats_in);
    std::vector<float> t = m.Conv(m.asp_tdnn, stats_in, rows, true);
    for (float& v : t)
      v = std::tanh(v);
    std::vector<float> w = m.Conv(m.asp_conv, t, rows, false);
    for (std::size_t c = 0; c < kC; ++c) {
      float top = -std::numeric_limits<float>::infinity();
      for (std::size_t r = 0; r < rows; ++r)
        top = std::max(top, w[r * kC + c]);
      float sum = 0;
      for (std::size_t r = 0; r < rows; ++r) {
        w[r * kC + c] = std::exp(w[r * kC + c] - top);
        sum += w[r * kC + c];
      }
      for (std::size_t r = 0; r < rows; ++r)
        w[r * kC + c] = Bf16(w[r * kC + c] / sum);
    }
    std::vector<float> pooled(2 * kC);
    for (std::size_t c = 0; c < kC; ++c) {
      float mean = 0, second = 0;
      for (std::size_t r = 0; r < rows; ++r) {
        const float weight = w[r * kC + c], v = mfa[r * kC + c];
        mean += weight * v;
        second += weight * v * v;
      }
      pooled[c] = Bf16(mean);
      pooled[kC + c] =
          Bf16(std::sqrt(std::max(second - mean * mean, 0.0F) + kEps));
    }
    output->embedding = m.Conv(m.fc, pooled, 1, false);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    *output = {};
    return false;
  }
}

}  // namespace gufo::models::qwen3_tts::npu

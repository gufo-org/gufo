#include "audio_processor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace gufo::models::embedding_gemma2 {
namespace {
constexpr std::size_t kFft = 512;
constexpr std::size_t kFrame = 320;
constexpr std::size_t kHop = 160;

struct Frontend {
  std::array<float, kFrame> window{};
  std::array<std::array<double, kAudioMelBins>, kFft / 2 + 1> filters{};
  std::array<std::complex<double>, kFft / 2> roots{};
  Frontend() {
    for (std::size_t i = 0; i < kFrame; ++i)
      window[i] = static_cast<float>(
          0.5 - 0.5 * std::cos(2 * std::numbers::pi * i / kFrame));
    for (std::size_t i = 0; i < roots.size(); ++i)
      roots[i] = std::polar(1.0, -2 * std::numbers::pi * i / kFft);
    std::array<double, kAudioMelBins + 2> edges{};
    const double max_mel = 2595.0 * std::log10(1.0 + 8000.0 / 700.0);
    for (std::size_t i = 0; i < edges.size(); ++i)
      edges[i] =
          700.0 *
          (std::pow(10.0, max_mel * i / (kAudioMelBins + 1) / 2595.0) - 1.0);
    for (std::size_t f = 0; f < filters.size(); ++f) {
      const double hz = 8000.0 * f / (filters.size() - 1);
      for (std::size_t m = 0; m < kAudioMelBins; ++m)
        filters[f][m] = std::max(
            0.0, std::min((hz - edges[m]) / (edges[m + 1] - edges[m]),
                          (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1])));
    }
  }
  void Fft(std::array<std::complex<double>, kFft>* data) const {
    auto& x = *data;
    for (std::size_t i = 1, j = 0; i < kFft; ++i) {
      std::size_t bit = kFft >> 1;
      for (; j & bit; bit >>= 1)
        j ^= bit;
      j ^= bit;
      if (i < j)
        std::swap(x[i], x[j]);
    }
    for (std::size_t length = 2; length <= kFft; length *= 2) {
      for (std::size_t begin = 0; begin < kFft; begin += length) {
        for (std::size_t j = 0; j < length / 2; ++j) {
          const auto even = x[begin + j];
          const auto odd = x[begin + j + length / 2] * roots[j * kFft / length];
          x[begin + j] = even + odd;
          x[begin + j + length / 2] = even - odd;
        }
      }
    }
  }
};
}  // namespace

std::size_t AudioFeatures::soft_tokens() const noexcept {
  std::size_t count = 0;
  // Each stride-2 convolution slices the mask [::2].
  for (std::size_t i = 0; i < mask.size(); i += 4)
    count += mask[i] != 0;
  return count;
}

bool ValidateAudioProcessor(const std::string& model_dir, std::string* error) {
  std::ifstream file(std::filesystem::path(model_dir) /
                     "processor_config.json");
  std::ostringstream contents;
  contents << file.rdbuf();
  try {
    const auto root = json::parse(contents.str());
    const auto* config = root.find("feature_extractor");
    const auto flag = [&](const char* name, bool expected) {
      const auto* value = config ? config->find(name) : nullptr;
      return value && value->is_bool() && value->as_bool() == expected;
    };
    const auto null = [&](const char* name) {
      const auto* value = config ? config->find(name) : nullptr;
      return value && value->is_null();
    };
    if (config && config->is_object() &&
        config->member_str("feature_extractor_type") ==
            "Gemma4AudioFeatureExtractor" &&
        config->member_size("sampling_rate") == kAudioSampleRate &&
        config->member_size("feature_size") == kAudioMelBins &&
        config->member_size("frame_length") == kFrame &&
        config->member_size("hop_length") == kHop &&
        config->member_size("fft_length") == kFft &&
        config->member_double("min_frequency", -1) == 0 &&
        config->member_double("max_frequency") == 8000 &&
        config->member_double("mel_floor") == 0.001 &&
        config->member_double("dither", -1) == 0 &&
        config->member_double("preemphasis", -1) == 0 &&
        config->member_double("input_scale_factor") == 1 &&
        config->member_double("padding_value", -1) == 0 &&
        config->member_str("padding_side") == "right" &&
        flag("fft_overdrive", false) && flag("preemphasis_htk_flavor", true) &&
        flag("return_attention_mask", true) && null("per_bin_mean") &&
        null("per_bin_stddev"))
      return true;
  } catch (const std::exception&) {
  }
  if (error)
    *error =
        "embedding_gemma2: unsupported or missing audio processor "
        "configuration";
  return false;
}

AudioFeatures ProcessAudio(std::span<const float> waveform,
                           std::size_t padding_samples,
                           const std::function<bool()>& is_cancelled) {
  if (waveform.empty() || waveform.size() > kMaxAudioSamples ||
      padding_samples > kMaxAudioSamples ||
      !std::ranges::all_of(waveform, [](float x) { return std::isfinite(x); }))
    throw std::invalid_argument(
        "embedding audio must contain 1 to 480000 finite 16-kHz samples");
  const std::size_t padded =
      (std::max(waveform.size(), padding_samples) + 127) / 128 * 128;
  // The official unfold consumes frame_length+1 samples, even with zero
  // preemphasis. Its last sample decides the frame mask but is not FFT input.
  const std::size_t frames =
      padded + kFrame / 2 < kFrame + 1
          ? 0
          : (padded + kFrame / 2 - kFrame - 1) / kHop + 1;
  AudioFeatures result;
  result.values.resize(frames * kAudioMelBins, 0);
  result.mask.resize(frames, 0);
  static const Frontend frontend;
  for (std::size_t frame = 0; frame < frames; ++frame) {
    if (is_cancelled && is_cancelled())
      throw std::runtime_error("embedding audio preprocessing cancelled");
    if (frame * kHop + kFrame / 2 >= waveform.size())
      continue;
    result.mask[frame] = 1;
    std::array<std::complex<double>, kFft> fft{};
    for (std::size_t j = 0; j < kFrame; ++j) {
      const auto sample = static_cast<std::ptrdiff_t>(frame * kHop + j) -
                          static_cast<std::ptrdiff_t>(kFrame / 2);
      if (sample >= 0 && static_cast<std::size_t>(sample) < waveform.size()) {
        // NumPy multiplies two float32 arrays before its float64 FFT.
        fft[j] = static_cast<double>(waveform[sample] * frontend.window[j]);
      }
    }
    frontend.Fft(&fft);
    std::array<double, kAudioMelBins> mel{};
    for (std::size_t f = 0; f <= kFft / 2; ++f) {
      const double magnitude = std::abs(fft[f]);
      for (std::size_t m = 0; m < kAudioMelBins; ++m)
        mel[m] += magnitude * frontend.filters[f][m];
    }
    for (std::size_t m = 0; m < kAudioMelBins; ++m)
      result.values[frame * kAudioMelBins + m] =
          static_cast<float>(std::log(mel[m] + 0.001));
  }
  return result;
}
}  // namespace gufo::models::embedding_gemma2

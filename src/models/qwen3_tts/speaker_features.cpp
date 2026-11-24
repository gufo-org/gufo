#include "src/models/qwen3_tts/speaker_features.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace gufo::models::qwen3_tts {
namespace {

constexpr std::size_t kSampleRate = 24000;
constexpr std::size_t kMelBins = kSpeakerMelBins;
constexpr std::size_t kFft = 1024;
constexpr std::size_t kFrequencyBins = kFft / 2 + 1;
constexpr std::size_t kHop = 256;
constexpr std::size_t kPrepad = (kFft - kHop) / 2;

std::int64_t ReflectIndex(std::int64_t index, std::int64_t length) {
  if (length <= 1) {
    return 0;
  }
  while (index < 0 || index >= length) {
    index = index < 0 ? -index : 2 * length - index - 2;
  }
  return index;
}

void Fft(std::span<std::complex<float>> values) {
  const std::size_t size = values.size();
  for (std::size_t index = 1, reversed = 0; index < size; ++index) {
    std::size_t bit = size >> 1U;
    for (; (reversed & bit) != 0; bit >>= 1U) {
      reversed ^= bit;
    }
    reversed ^= bit;
    if (index < reversed) {
      std::swap(values[index], values[reversed]);
    }
  }
  for (std::size_t length = 2; length <= size; length <<= 1U) {
    const float angle =
        -2.0F * std::numbers::pi_v<float> / static_cast<float>(length);
    const std::complex<float> root(std::cos(angle), std::sin(angle));
    for (std::size_t offset = 0; offset < size; offset += length) {
      std::complex<float> factor(1.0F, 0.0F);
      for (std::size_t index = 0; index < length / 2; ++index) {
        const std::complex<float> even = values[offset + index];
        const std::complex<float> odd =
            factor * values[offset + index + length / 2];
        values[offset + index] = even + odd;
        values[offset + index + length / 2] = even - odd;
        factor *= root;
      }
    }
  }
}

float HertzToMel(float hertz) {
  constexpr float scale = 200.0F / 3.0F;
  constexpr float boundary_hertz = 1000.0F;
  constexpr float boundary_mel = boundary_hertz / scale;
  constexpr float log_step = 0.06875177742094912F;
  return hertz >= boundary_hertz
             ? boundary_mel + std::log(hertz / boundary_hertz) / log_step
             : hertz / scale;
}

float MelToHertz(float mel) {
  constexpr float scale = 200.0F / 3.0F;
  constexpr float boundary_hertz = 1000.0F;
  constexpr float boundary_mel = boundary_hertz / scale;
  constexpr float log_step = 0.06875177742094912F;
  return mel >= boundary_mel
             ? boundary_hertz * std::exp(log_step * (mel - boundary_mel))
             : mel * scale;
}

std::vector<float> MakeMelFilterbank() {
  const float mel_min = HertzToMel(0.0F);
  const float mel_max = HertzToMel(12000.0F);
  std::vector<float> hertz(kMelBins + 2);
  for (std::size_t index = 0; index < hertz.size(); ++index) {
    const float mel = mel_min + (mel_max - mel_min) *
                                    static_cast<float>(index) /
                                    static_cast<float>(kMelBins + 1);
    hertz[index] = MelToHertz(mel);
  }
  std::vector<float> filter(kMelBins * kFrequencyBins, 0.0F);
  for (std::size_t mel = 0; mel < kMelBins; ++mel) {
    const float left = hertz[mel];
    const float center = hertz[mel + 1];
    const float right = hertz[mel + 2];
    const float normalization = 2.0F / std::max(right - left, 1.0e-12F);
    for (std::size_t bin = 0; bin < kFrequencyBins; ++bin) {
      const float frequency = static_cast<float>(kSampleRate) * 0.5F *
                              static_cast<float>(bin) /
                              static_cast<float>(kFrequencyBins - 1);
      const float lower =
          (frequency - left) / std::max(center - left, 1.0e-12F);
      const float upper =
          (right - frequency) / std::max(right - center, 1.0e-12F);
      filter[mel * kFrequencyBins + bin] =
          std::max(0.0F, std::min(lower, upper)) * normalization;
    }
  }
  return filter;
}

}  // namespace

SpeakerFeatures ComputeSpeakerFeatures(const AudioBuffer& audio) {
  std::vector<float> waveform = ResampleMono(audio, kSampleRate);
  if (waveform.empty()) {
    throw std::invalid_argument(
        "Qwen3-TTS speaker encoder requires reference audio");
  }
  std::vector<float> padded(waveform.size() + 2 * kPrepad);
  for (std::size_t index = 0; index < padded.size(); ++index) {
    const std::int64_t source = ReflectIndex(
        static_cast<std::int64_t>(index) - static_cast<std::int64_t>(kPrepad),
        static_cast<std::int64_t>(waveform.size()));
    padded[index] = waveform[static_cast<std::size_t>(source)];
  }
  if (padded.size() < kFft) {
    throw std::invalid_argument(
        "Qwen3-TTS reference audio is too short for speaker encoding");
  }
  const std::size_t frames = 1 + (padded.size() - kFft) / kHop;
  std::vector<float> magnitude(kFrequencyBins * frames, 0.0F);
  std::vector<std::complex<float>> spectrum(kFft);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    for (std::size_t index = 0; index < kFft; ++index) {
      const float window =
          0.5F -
          0.5F * std::cos(2.0F * std::numbers::pi_v<float> *
                          static_cast<float>(index) / static_cast<float>(kFft));
      spectrum[index] =
          std::complex<float>(padded[frame * kHop + index] * window, 0.0F);
    }
    Fft(spectrum);
    for (std::size_t bin = 0; bin < kFrequencyBins; ++bin) {
      magnitude[bin * frames + frame] =
          std::sqrt(std::norm(spectrum[bin]) + 1.0e-9F);
    }
  }

  static const std::vector<float> filter = MakeMelFilterbank();
  SpeakerFeatures result;
  result.frames = frames;
  result.values.resize(frames * kMelBins);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    for (std::size_t mel = 0; mel < kMelBins; ++mel) {
      long double sum = 0.0;
      for (std::size_t bin = 0; bin < kFrequencyBins; ++bin) {
        sum += static_cast<long double>(filter[mel * kFrequencyBins + bin]) *
               magnitude[bin * frames + frame];
      }
      result.values[frame * kMelBins + mel] =
          std::log(std::max(static_cast<float>(sum), 1.0e-5F));
    }
  }
  return result;
}

}  // namespace gufo::models::qwen3_tts

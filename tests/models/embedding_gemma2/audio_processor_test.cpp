#include "src/models/embedding_gemma2/audio_processor.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "src/core/json.hpp"

namespace model = gufo::models::embedding_gemma2;
namespace {
void Check(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}
std::vector<float> Waveform(std::size_t count, const std::string& kind) {
  std::vector<float> samples(count, 0);
  for (std::size_t i = 0; i < count; ++i) {
    if (kind == "triangle") {
      const auto phase = static_cast<int>(i % 80);
      samples[i] = static_cast<float>(std::min(phase, 80 - phase) - 20) / 64;
    } else if (kind == "noise") {
      samples[i] = static_cast<float>(
                       static_cast<int>((i * 73 + i * i * 19) % 1024) - 512) /
                   1024;
    }
  }
  return samples;
}
}  // namespace

int main() {
  try {
    std::ifstream input(GUFO_EMBEDDING_GEMMA2_AUDIO_FEATURES_GOLDEN);
    Check(input.good(), "official audio feature fixture must exist");
    std::ostringstream contents;
    contents << input.rdbuf();
    const auto fixture = gufo::json::parse(contents.str());
    double worst = 0;
    for (const auto& probe : fixture.find("cases")->items()) {
      const auto name = probe.member_str("name");
      const auto features = model::ProcessAudio(
          Waveform(probe.member_size("samples"), probe.member_str("kind")),
          probe.member_size("padding_samples"));
      Check(features.mask.size() == probe.member_size("frames"),
            name + " frame count");
      Check(features.soft_tokens() == probe.member_size("soft_tokens"),
            name + " valid token count");
      const auto& mask = probe.find("mask")->items();
      for (std::size_t i = 0; i < mask.size(); ++i)
        Check(features.mask[i] == mask[i].as_size(), name + " frame mask");
      const auto& values = probe.find("features")->items();
      Check(features.values.size() == values.size(), name + " feature shape");
      for (std::size_t i = 0; i < values.size(); ++i) {
        const double expected = values[i].as_double();
        const double error = std::abs(features.values[i] - expected);
        worst = std::max(worst, error);
        Check(std::isfinite(features.values[i]) && error <= 2e-6,
              name + " official feature mismatch at " + std::to_string(i) +
                  ": " + std::to_string(error));
      }
    }
    for (const auto& waveform :
         {std::vector<float>{}, std::vector<float>(model::kMaxAudioSamples + 1),
          std::vector<float>{std::numeric_limits<float>::quiet_NaN()},
          std::vector<float>{std::numeric_limits<float>::infinity()}}) {
      bool rejected = false;
      try {
        (void)model::ProcessAudio(waveform);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      Check(rejected, "invalid waveform must be rejected");
    }
    bool cancelled = false;
    try {
      (void)model::ProcessAudio(std::vector<float>(16000), 0,
                                [] { return true; });
    } catch (const std::runtime_error&) {
      cancelled = true;
    }
    Check(cancelled, "preprocessing must observe cancellation");
    std::cout << "official audio features and masks pass; maximum error "
              << worst << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

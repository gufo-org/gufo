#include "src/models/embedding_gemma2/audio_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "src/core/json.hpp"

namespace model = gufo::models::embedding_gemma2;
namespace {
void Check(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}
std::vector<std::uint8_t> Bytes(const std::string& hex) {
  std::vector<std::uint8_t> result;
  const auto nibble = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
  for (std::size_t i = 0; i < hex.size(); i += 2)
    result.push_back(
        static_cast<std::uint8_t>(nibble(hex[i]) * 16 + nibble(hex[i + 1])));
  return result;
}
}  // namespace
int main() {
  try {
    std::ifstream file(GUFO_EMBEDDING_GEMMA2_AUDIO_DECODER_GOLDEN);
    Check(file.good(), "independent decoder fixture must exist");
    std::ostringstream content;
    content << file.rdbuf();
    const auto fixture = gufo::json::parse(content.str());
    double worst = 0;
    for (const auto& probe : fixture.find("cases")->items()) {
      const auto bytes = Bytes(probe.member_str("encoded_hex"));
      const auto actual = model::DecodeAudio(bytes);
      const auto& expected = probe.find("waveform")->items();
      Check(actual.size() == expected.size(),
            probe.member_str("name") + " decoded length");
      for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(actual[i] - expected[i].as_double());
        worst = std::max(worst, error);
        Check(std::isfinite(actual[i]) && error < 2e-6,
              probe.member_str("name") +
                  " decoded/resampled waveform mismatch: " +
                  std::to_string(error));
      }
      bool rejected = false;
      try {
        (void)model::DecodeAudio(std::span(bytes).first(20));
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      Check(rejected, "truncated container must fail");
      rejected = false;
      try {
        (void)model::DecodeAudio(bytes, [] { return true; });
      } catch (const std::exception&) {
        rejected = true;
      }
      Check(rejected, "cancelled decode must fail");
      Check(model::DecodeAudio(bytes).size() == actual.size(),
            "valid decode recovers after errors");
    }
    std::cout << "official librosa decoder waveforms pass; maximum error "
              << worst << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

#include "src/models/qwen3_asr/audio.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace qwen3_asr = gufo::models::qwen3_asr;

namespace {

[[noreturn]] void Fail(std::string_view message) {
  std::cerr << "FAIL qwen3_asr_audio_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, std::string_view message) {
  if (!condition) {
    Fail(message);
  }
}

std::uint16_t ReadU16(std::istream& input) {
  unsigned char bytes[2]{};
  input.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint32_t ReadU32(std::istream& input) {
  unsigned char bytes[4]{};
  input.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::vector<std::size_t> ParseShape(std::string_view header) {
  const std::size_t key = header.find("'shape'");
  const std::size_t open =
      key == std::string_view::npos ? key : header.find('(', key);
  const std::size_t close =
      open == std::string_view::npos ? open : header.find(')', open);
  std::vector<std::size_t> shape;
  if (open == std::string_view::npos || close == std::string_view::npos) {
    return shape;
  }
  for (std::size_t position = open + 1U; position < close;) {
    while (position < close &&
           (header[position] == ' ' || header[position] == ',')) {
      ++position;
    }
    if (position == close) {
      break;
    }
    std::size_t end = position;
    while (end < close && header[end] >= '0' && header[end] <= '9') {
      ++end;
    }
    if (end == position) {
      return {};
    }
    shape.push_back(
        std::stoull(std::string(header.substr(position, end - position))));
    position = end;
  }
  return shape;
}

std::vector<float> ReadNpyFloat(const std::filesystem::path& path,
                                std::vector<std::size_t>* shape) {
  std::ifstream input(path, std::ios::binary);
  Check(input.good(), "cannot open reference NPY");
  char magic[6]{};
  input.read(magic, sizeof(magic));
  Check(std::string_view(magic, sizeof(magic)) == "\x93NUMPY",
        "invalid NPY magic");
  const int major = input.get();
  (void)input.get();
  const std::size_t header_size = major == 1 ? ReadU16(input) : ReadU32(input);
  std::string header(header_size, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  Check(header.find("'descr': '<f4'") != std::string::npos,
        "reference NPY must contain little-endian float32");
  Check(header.find("'fortran_order': False") != std::string::npos,
        "reference NPY must use C order");
  *shape = ParseShape(header);
  std::size_t elements = 1;
  for (const std::size_t dimension : *shape) {
    Check(dimension == 0U ||
              elements <= std::numeric_limits<std::size_t>::max() / dimension,
          "reference NPY shape overflow");
    elements *= dimension;
  }
  std::vector<float> result(elements);
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(result.size() * sizeof(float)));
  Check(input.gcount() ==
            static_cast<std::streamsize>(result.size() * sizeof(float)),
        "reference NPY is truncated");
  return result;
}

double Cosine(std::span<const float> left, std::span<const float> right) {
  Check(left.size() == right.size(), "feature sizes differ");
  double dot = 0.0;
  double left_norm = 0.0;
  double right_norm = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    dot += static_cast<double>(left[index]) * right[index];
    left_norm += static_cast<double>(left[index]) * left[index];
    right_norm += static_cast<double>(right[index]) * right[index];
  }
  return dot / std::sqrt(left_norm * right_norm);
}

}  // namespace

int main() {
  const char* configured = std::getenv("QWEN3_ASR_ARTIFACTS");
  const std::filesystem::path artifacts =
      configured != nullptr ? configured
                            : "/home/fbozzo/projects/strix-halo.cpp/"
                              "artifacts/qwen3_asr/official";
  if (!std::filesystem::is_regular_file(artifacts / "waveform.npy") ||
      !std::filesystem::is_regular_file(artifacts / "input_features.npy")) {
    std::cerr << "SKIP qwen3_asr_audio_test: reference artifacts unavailable\n";
    return 77;
  }

  std::vector<std::size_t> waveform_shape;
  const std::vector<float> waveform =
      ReadNpyFloat(artifacts / "waveform.npy", &waveform_shape);
  Check(waveform_shape == std::vector<std::size_t>({240820}),
        "reference waveform shape changed");

  std::vector<std::size_t> expected_shape;
  const std::vector<float> expected =
      ReadNpyFloat(artifacts / "input_features.npy", &expected_shape);
  const qwen3_asr::LogMelFeatures actual =
      qwen3_asr::ComputeLogMelFeatures(waveform);
  Check(qwen3_asr::AudioEmbeddingTokenCount(actual.frames) == 196U,
        "reference audio token count changed");
  Check(qwen3_asr::AudioEmbeddingTokenCount(1U) == 1U &&
            qwen3_asr::AudioEmbeddingTokenCount(100U) == 13U &&
            qwen3_asr::AudioEmbeddingTokenCount(101U) == 14U &&
            qwen3_asr::AudioEmbeddingTokenCount(200U) == 26U,
        "audio token count boundary behavior changed");
  Check(expected_shape == std::vector<std::size_t>(
                              {1U, qwen3_asr::kAudioMelBins, actual.frames}),
        "reference feature shape changed");
  Check(actual.values.size() == expected.size(), "feature size mismatch");

  double squared_error = 0.0;
  double squared_reference = 0.0;
  float maximum_error = 0.0F;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const float error = actual.values[index] - expected[index];
    squared_error += static_cast<double>(error) * error;
    squared_reference += static_cast<double>(expected[index]) * expected[index];
    maximum_error = std::max(maximum_error, std::abs(error));
  }
  const double relative_l2 = std::sqrt(squared_error / squared_reference);
  const double cosine = Cosine(actual.values, expected);
  std::cout << "qwen3_asr_audio frames=" << actual.frames
            << " cosine=" << cosine << " relL2=" << relative_l2
            << " max_abs=" << maximum_error << '\n';
  Check(cosine > 0.999999 && relative_l2 < 2.0e-5 && maximum_error < 2.0e-4F,
        "native log-mel features diverge from official reference");

  std::cout << "PASS qwen3_asr_audio_test\n";
  return 0;
}

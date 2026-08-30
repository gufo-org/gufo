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

#include "src/models/qwen3_asr/hip/audio_encoder_runtime.hpp"

namespace qwen3_asr_hip = gufo::models::qwen3_asr::hip;

namespace {

[[noreturn]] void Fail(std::string_view message) {
  std::cerr << "FAIL qwen3_asr_audio_encoder_hip_test: " << message << '\n';
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

struct NpyFloat {
  std::vector<float> values;
  std::vector<std::size_t> shape;
};

NpyFloat ReadNpyFloat(const std::filesystem::path& path) {
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
        "reference NPY must be float32");
  const std::size_t key = header.find("'shape'");
  const std::size_t open = header.find('(', key);
  const std::size_t close = header.find(')', open);
  Check(key != std::string::npos && open != std::string::npos &&
            close != std::string::npos,
        "cannot parse NPY shape");
  NpyFloat result;
  std::size_t elements = 1;
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
    Check(end != position, "invalid NPY dimension");
    const std::size_t dimension =
        std::stoull(header.substr(position, end - position));
    Check(dimension == 0U ||
              elements <= std::numeric_limits<std::size_t>::max() / dimension,
          "NPY shape overflow");
    result.shape.push_back(dimension);
    elements *= dimension;
    position = end;
  }
  result.values.resize(elements);
  input.read(reinterpret_cast<char*>(result.values.data()),
             static_cast<std::streamsize>(elements * sizeof(float)));
  Check(
      input.gcount() == static_cast<std::streamsize>(elements * sizeof(float)),
      "reference NPY is truncated");
  return result;
}

double Cosine(std::span<const float> left, std::span<const float> right) {
  Check(left.size() == right.size(), "activation sizes differ");
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
  const char* configured_model = std::getenv("QWEN3_ASR_MODEL_ROOT");
  const std::filesystem::path model_root =
      configured_model != nullptr ? configured_model
                                  : "/var/llms/huggingface/hub/"
                                    "models--Qwen--Qwen3-ASR-1.7B/snapshots/"
                                    "7278e1e70fe206f11671096ffdd38061171dd6e5";
  const char* configured_artifacts = std::getenv("QWEN3_ASR_ARTIFACTS");
  const std::filesystem::path artifacts =
      configured_artifacts != nullptr ? configured_artifacts
                                      : "/home/fbozzo/projects/strix-halo.cpp/"
                                        "artifacts/qwen3_asr/official";
  if (!std::filesystem::is_regular_file(model_root / "config.json") ||
      !std::filesystem::is_regular_file(artifacts / "input_features.npy") ||
      !std::filesystem::is_regular_file(artifacts /
                                        "audio_encoder_input.npy")) {
    std::cerr << "SKIP qwen3_asr_audio_encoder_hip_test: model or artifacts "
                 "unavailable\n";
    return 77;
  }

  const NpyFloat features = ReadNpyFloat(artifacts / "input_features.npy");
  Check(features.shape == std::vector<std::size_t>({1, 128, 1505}),
        "input feature oracle shape changed");
  const NpyFloat expected = ReadNpyFloat(artifacts / "audio_encoder_input.npy");
  Check(expected.shape == std::vector<std::size_t>({196, 1024}),
        "encoder-input oracle shape changed");
  const NpyFloat expected_layer0 = ReadNpyFloat(artifacts / "audio_layer0.npy");
  Check(expected_layer0.shape == std::vector<std::size_t>({196, 1024}),
        "layer0 oracle shape changed");
  const NpyFloat expected_final = ReadNpyFloat(artifacts / "audio_final.npy");
  Check(expected_final.shape == std::vector<std::size_t>({196, 2048}),
        "audio-final oracle shape changed");

  std::string error;
  auto runtime = qwen3_asr_hip::AudioEncoderHipRuntime::Create(
      model_root.string(), &error);
  Check(runtime != nullptr, error);
  qwen3_asr_hip::AudioEncoderOutput actual;
  Check(runtime->EncodeFrontend(features.values, 1505, &actual, &error), error);
  Check(actual.tokens == 196 && actual.values.size() == expected.values.size(),
        "native encoder-input shape differs");

  double squared_error = 0.0;
  double squared_reference = 0.0;
  float maximum_error = 0.0F;
  for (std::size_t index = 0; index < expected.values.size(); ++index) {
    const float difference = actual.values[index] - expected.values[index];
    squared_error += static_cast<double>(difference) * difference;
    squared_reference +=
        static_cast<double>(expected.values[index]) * expected.values[index];
    maximum_error = std::max(maximum_error, std::abs(difference));
  }
  const double relative_l2 = std::sqrt(squared_error / squared_reference);
  const double cosine = Cosine(actual.values, expected.values);
  std::cout << "qwen3_asr_audio_frontend tokens=" << actual.tokens
            << " cosine=" << cosine << " relL2=" << relative_l2
            << " max_abs=" << maximum_error << '\n';
  Check(cosine > 0.9999 && relative_l2 < 0.02,
        "native convolutional frontend diverges from official reference");

  qwen3_asr_hip::AudioEncoderTrace trace;
  Check(runtime->Encode(features.values, 1505, &trace, &error), error);
  const auto report = [](std::string_view name, std::span<const float> actual,
                         std::span<const float> reference) {
    Check(actual.size() == reference.size(), "encoder trace size differs");
    double squared_error = 0.0;
    double squared_reference = 0.0;
    float maximum_error = 0.0F;
    for (std::size_t index = 0; index < reference.size(); ++index) {
      const float difference = actual[index] - reference[index];
      squared_error += static_cast<double>(difference) * difference;
      squared_reference +=
          static_cast<double>(reference[index]) * reference[index];
      maximum_error = std::max(maximum_error, std::abs(difference));
    }
    const double relative_l2 = std::sqrt(squared_error / squared_reference);
    const double cosine = Cosine(actual, reference);
    std::cout << name << " cosine=" << cosine << " relL2=" << relative_l2
              << " max_abs=" << maximum_error << '\n';
    return std::pair(cosine, relative_l2);
  };
  const auto layer0_quality = report(
      "qwen3_asr_audio_layer0", trace.layer0.values, expected_layer0.values);
  Check(layer0_quality.first > 0.999 && layer0_quality.second < 0.05,
        "native audio layer0 diverges from official reference");
  const auto final_quality = report("qwen3_asr_audio_final", trace.final.values,
                                    expected_final.values);
  Check(final_quality.first > 0.995 && final_quality.second < 0.1,
        "native audio encoder final diverges from official reference");

  std::cout << "PASS qwen3_asr_audio_encoder_hip_test\n";
  return 0;
}

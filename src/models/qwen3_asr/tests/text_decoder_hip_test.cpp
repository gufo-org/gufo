#include <algorithm>
#include <chrono>
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

#include "src/models/qwen3_asr/hip/text_decoder_runtime.hpp"
#include "src/models/qwen3_asr/prompt.hpp"
#include "src/models/qwen3_asr/tokenizer.hpp"

namespace qwen3_asr_hip = gufo::models::qwen3_asr::hip;
namespace qwen3_asr = gufo::models::qwen3_asr;

namespace {

[[noreturn]] void Fail(std::string_view message) {
  std::cerr << "FAIL qwen3_asr_text_decoder_hip_test: " << message << '\n';
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

template<typename Element>
struct Npy {
  std::vector<Element> values;
  std::vector<std::size_t> shape;
};

template<typename Element>
Npy<Element> ReadNpy(const std::filesystem::path& path,
                     std::string_view descriptor) {
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
  Check(header.find(std::string("'descr': '") + std::string(descriptor) +
                    "'") != std::string::npos,
        "unexpected reference NPY dtype");
  const std::size_t key = header.find("'shape'");
  const std::size_t open = header.find('(', key);
  const std::size_t close = header.find(')', open);
  Check(key != std::string::npos && open != std::string::npos &&
            close != std::string::npos,
        "cannot parse NPY shape");
  Npy<Element> result;
  std::size_t elements = 1U;
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
             static_cast<std::streamsize>(elements * sizeof(Element)));
  Check(input.gcount() ==
            static_cast<std::streamsize>(elements * sizeof(Element)),
        "reference NPY is truncated");
  return result;
}

struct Quality {
  double cosine;
  double relative_l2;
  float maximum_error;
};

Quality Compare(std::span<const float> actual,
                std::span<const float> reference) {
  Check(actual.size() == reference.size(), "activation sizes differ");
  double dot = 0.0;
  double actual_squared = 0.0;
  double reference_squared = 0.0;
  double error_squared = 0.0;
  float maximum_error = 0.0F;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double left = actual[index];
    const double right = reference[index];
    const double difference = left - right;
    dot += left * right;
    actual_squared += left * left;
    reference_squared += right * right;
    error_squared += difference * difference;
    maximum_error =
        std::max(maximum_error, static_cast<float>(std::abs(difference)));
  }
  return {
      .cosine = dot / std::sqrt(actual_squared * reference_squared),
      .relative_l2 = std::sqrt(error_squared / reference_squared),
      .maximum_error = maximum_error,
  };
}

std::uint32_t Argmax(std::span<const float> values) {
  return static_cast<std::uint32_t>(
      std::distance(values.begin(), std::ranges::max_element(values)));
}

double Seconds(std::chrono::steady_clock::time_point start,
               std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
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
      !std::filesystem::is_regular_file(artifacts / "prompt_ids.npy") ||
      !std::filesystem::is_regular_file(artifacts / "audio_final.npy") ||
      !std::filesystem::is_regular_file(artifacts /
                                        "text_layer0_prefill.npy") ||
      !std::filesystem::is_regular_file(artifacts / "prefill_logits.npy")) {
    std::cerr << "SKIP qwen3_asr_text_decoder_hip_test: model or artifacts "
                 "unavailable\n";
    return 77;
  }

  const Npy<std::int32_t> prompt_i32 =
      ReadNpy<std::int32_t>(artifacts / "prompt_ids.npy", "<i4");
  const Npy<float> audio = ReadNpy<float>(artifacts / "audio_final.npy", "<f4");
  const Npy<float> expected_layer0 =
      ReadNpy<float>(artifacts / "text_layer0_prefill.npy", "<f4");
  const Npy<float> expected_logits =
      ReadNpy<float>(artifacts / "prefill_logits.npy", "<f4");
  const Npy<std::int32_t> expected_generated =
      ReadNpy<std::int32_t>(artifacts / "generated_ids.npy", "<i4");
  Check(prompt_i32.shape == std::vector<std::size_t>({1, 211}),
        "prompt fixture shape changed");
  Check(audio.shape == std::vector<std::size_t>({196, 2048}),
        "audio fixture shape changed");
  Check(expected_layer0.shape == std::vector<std::size_t>({1, 211, 2048}),
        "text layer-0 fixture shape changed");
  Check(expected_logits.shape == std::vector<std::size_t>({151936}),
        "prefill-logit fixture shape changed");

  std::vector<std::uint32_t> prompt;
  prompt.reserve(prompt_i32.values.size());
  for (const std::int32_t id : prompt_i32.values) {
    prompt.push_back(static_cast<std::uint32_t>(id));
  }

  std::string error;
  const auto load_start = std::chrono::steady_clock::now();
  auto runtime = qwen3_asr_hip::TextDecoderHipRuntime::Create(
      model_root.string(), 512U, &error);
  const auto load_end = std::chrono::steady_clock::now();
  Check(runtime != nullptr, error);

  qwen3_asr_hip::TextDecoderTrace trace;
  const auto prefill_start = std::chrono::steady_clock::now();
  Check(runtime->Prefill(prompt, audio.values, audio.shape[0], &trace, &error),
        error);
  const auto prefill_end = std::chrono::steady_clock::now();
  const Quality layer0 = Compare(trace.layer0, expected_layer0.values);
  const Quality logits = Compare(trace.logits, expected_logits.values);
  std::cout << "qwen3_asr_text_layer0 cosine=" << layer0.cosine
            << " relL2=" << layer0.relative_l2
            << " max_abs=" << layer0.maximum_error << '\n';
  std::cout << "qwen3_asr_prefill_logits cosine=" << logits.cosine
            << " relL2=" << logits.relative_l2
            << " max_abs=" << logits.maximum_error
            << " native_argmax=" << Argmax(trace.logits)
            << " official_argmax=" << Argmax(expected_logits.values) << '\n';
  Check(layer0.cosine > 0.995 && layer0.relative_l2 < 0.1,
        "native text layer 0 diverges from official reference");
  Check(logits.cosine > 0.99 && logits.relative_l2 < 0.15,
        "native prefill logits diverge from official reference");
  Check(Argmax(trace.logits) == Argmax(expected_logits.values),
        "native prefill chose a different first token");

  std::vector<std::uint32_t> generated;
  const auto generate_start = std::chrono::steady_clock::now();
  Check(runtime->Generate(prompt, audio.values, audio.shape[0], 128U,
                          &generated, &error),
        error);
  const auto generate_end = std::chrono::steady_clock::now();
  const std::size_t common =
      std::min(generated.size(), expected_generated.values.size());
  std::size_t first_difference = common;
  for (std::size_t index = 0; index < common; ++index) {
    if (generated[index] !=
        static_cast<std::uint32_t>(expected_generated.values[index])) {
      first_difference = index;
      break;
    }
  }
  if (generated.size() != expected_generated.values.size() ||
      first_difference != common) {
    std::cerr << "generation mismatch native_count=" << generated.size()
              << " official_count=" << expected_generated.values.size()
              << " first_difference=" << first_difference << '\n';
    const std::size_t begin =
        first_difference > 3U ? first_difference - 3U : 0U;
    const std::size_t end =
        std::min(std::max(generated.size(), expected_generated.values.size()),
                 first_difference + 8U);
    for (std::size_t index = begin; index < end; ++index) {
      std::cerr << "  token[" << index << "] native=";
      if (index < generated.size()) {
        std::cerr << generated[index];
      } else {
        std::cerr << "<end>";
      }
      std::cerr << " official=";
      if (index < expected_generated.values.size()) {
        std::cerr << expected_generated.values[index];
      } else {
        std::cerr << "<end>";
      }
      std::cerr << '\n';
    }
  }
  Check(generated.size() == expected_generated.values.size(),
        "native generated-token count differs");
  Check(first_difference == common,
        "native generated token differs from official reference");
  qwen3_asr::Tokenizer tokenizer;
  Check(qwen3_asr::Tokenizer::Load(model_root, &tokenizer, &error), error);
  const std::string native_decoded = tokenizer.Decode(generated, true);
  const qwen3_asr::Transcription transcription =
      qwen3_asr::ParseTranscription(native_decoded, std::nullopt);
  Check(transcription.language == "English",
        "native generated language differs from official reference");
  Check(transcription.text ==
            "Uh huh. Oh yeah, yeah. He wasn't even that big when I started "
            "listening to him, but and his solo music didn't do overly well, "
            "but he did very well when he started writing for other people.",
        "native decoded transcript differs from official reference");

  std::cout << "PASS qwen3_asr_text_decoder_hip_test load_s="
            << Seconds(load_start, load_end)
            << " prefill_s=" << Seconds(prefill_start, prefill_end)
            << " generate_s=" << Seconds(generate_start, generate_end)
            << " generated_tokens=" << generated.size() << " transcript=\""
            << transcription.text << "\"\n";
  return 0;
}

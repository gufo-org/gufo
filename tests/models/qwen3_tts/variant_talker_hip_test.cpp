#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/qwen3_tts/hip/talker_runtime.hpp"
#include "src/models/qwen3_tts/tokenizer.hpp"

namespace {

namespace qwen3_tts = gufo::models::qwen3_tts;
namespace qwen3_tts_hip = gufo::models::qwen3_tts::hip;

struct NpyHeader {
  std::vector<std::size_t> shape;
  std::string descriptor;
  bool fortran_order{false};
};

template<typename Element>
struct NpyArray {
  std::vector<std::size_t> shape;
  std::vector<Element> values;
};

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_tts_variant_talker_hip_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
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
  if (open == std::string_view::npos || close == std::string_view::npos) {
    return {};
  }
  std::vector<std::size_t> shape;
  std::size_t position = open + 1;
  while (position < close) {
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

NpyHeader ReadHeader(std::istream& input) {
  char magic[6]{};
  input.read(magic, sizeof(magic));
  Check(std::string_view(magic, sizeof(magic)) == "\x93NUMPY",
        "invalid NPY magic");
  const int major = input.get();
  (void)input.get();
  Check(major == 1 || major == 2 || major == 3, "unsupported NPY version");
  const std::size_t header_size = major == 1 ? ReadU16(input) : ReadU32(input);
  std::string header(header_size, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  const bool fortran_order =
      header.find("fortran_order': True") != std::string::npos;
  Check(fortran_order ||
            header.find("fortran_order': False") != std::string::npos,
        "cannot parse NPY memory order");
  const std::size_t descriptor_key = header.find("'descr'");
  const std::size_t descriptor_start =
      header.find('\'', header.find(':', descriptor_key) + 1);
  const std::size_t descriptor_end = header.find('\'', descriptor_start + 1);
  Check(descriptor_key != std::string::npos &&
            descriptor_start != std::string::npos &&
            descriptor_end != std::string::npos,
        "cannot parse NPY descriptor");
  return {
      .shape = ParseShape(header),
      .descriptor = header.substr(descriptor_start + 1,
                                  descriptor_end - descriptor_start - 1),
      .fortran_order = fortran_order,
  };
}

template<typename Element>
NpyArray<Element> ReadNpy(const std::filesystem::path& path,
                          std::string_view descriptor) {
  std::ifstream input(path, std::ios::binary);
  Check(input.good(), "cannot open " + path.string());
  NpyHeader header = ReadHeader(input);
  Check(header.descriptor == descriptor, "unexpected NPY dtype");
  Check(!header.shape.empty(), "cannot parse NPY shape");
  std::size_t elements = 1;
  for (const std::size_t dimension : header.shape) {
    Check(dimension == 0 ||
              elements <= std::numeric_limits<std::size_t>::max() / dimension,
          "NPY shape overflow");
    elements *= dimension;
  }
  NpyArray<Element> result{.shape = std::move(header.shape)};
  result.values.resize(elements);
  input.read(reinterpret_cast<char*>(result.values.data()),
             static_cast<std::streamsize>(elements * sizeof(Element)));
  Check(input.gcount() ==
            static_cast<std::streamsize>(elements * sizeof(Element)),
        "truncated NPY payload");
  if (header.fortran_order && result.shape.size() > 1) {
    std::vector<Element> c_order(result.values.size());
    std::vector<std::size_t> coordinates(result.shape.size());
    for (std::size_t destination = 0; destination < result.values.size();
         ++destination) {
      std::size_t remainder = destination;
      for (std::size_t axis = result.shape.size(); axis-- > 0;) {
        coordinates[axis] = remainder % result.shape[axis];
        remainder /= result.shape[axis];
      }
      std::size_t source = 0;
      std::size_t source_stride = 1;
      for (std::size_t axis = 0; axis < result.shape.size(); ++axis) {
        source += coordinates[axis] * source_stride;
        source_stride *= result.shape[axis];
      }
      c_order[destination] = result.values[source];
    }
    result.values = std::move(c_order);
  }
  return result;
}

double Cosine(std::span<const float> left, std::span<const float> right) {
  Check(left.size() == right.size(), "comparison shape mismatch");
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

void CheckGreedyCodes(qwen3_tts_hip::TalkerHipRuntime* runtime,
                      const qwen3_tts_hip::TalkerPromptOutput& prompt,
                      std::span<const std::int32_t> expected) {
  std::string error;
  qwen3_tts::SamplingOptions greedy;
  greedy.sample = false;
  greedy.predictor_sample = false;
  qwen3_tts_hip::TalkerGenerationOutput cancelled;
  std::size_t cancellation_checks = 0;
  Check(!runtime->Generate(
            prompt, 5, greedy,
            [&cancellation_checks] { return ++cancellation_checks >= 3; },
            &cancelled, &error),
        "frame-level cancellation must stop generation");
  Check(cancelled.codes.empty() && cancelled.frames == 0 &&
            cancellation_checks == 3 &&
            error == "Qwen3-TTS native generation cancelled",
        "cancelled generation returns no partial codec frames");

  qwen3_tts_hip::TalkerGenerationOutput generated;
  error.clear();
  Check(runtime->GenerateGreedy(prompt, 5, &generated, &error), error);
  Check(generated.frames == 5 && generated.code_groups == 16,
        "greedy generation shape");
  Check(generated.codes.size() == expected.size(),
        "greedy generation code count");
  std::size_t matches = 0;
  std::size_t semantic_matches = 0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const bool matches_expected =
        generated.codes[index] == static_cast<std::uint32_t>(expected[index]);
    matches += matches_expected ? 1 : 0;
    semantic_matches +=
        index % generated.code_groups == 0 && matches_expected ? 1 : 0;
  }
  std::cout << "greedy_code_matches=" << matches << '/' << expected.size()
            << " semantic=" << semantic_matches << '/' << generated.frames
            << '\n';
  std::cout << "greedy_first_actual=";
  for (std::size_t index = 0;
       index < std::min<std::size_t>(16, generated.codes.size()); ++index) {
    std::cout << (index == 0 ? "" : ",") << generated.codes[index];
  }
  std::cout << "\ngreedy_first_expected=";
  for (std::size_t index = 0;
       index < std::min<std::size_t>(16, expected.size()); ++index) {
    std::cout << (index == 0 ? "" : ",") << expected[index];
  }
  std::cout << '\n';
  Check(semantic_matches == generated.frames, "greedy semantic codec parity");
  Check(matches >= expected.size() / 10,
        "greedy acoustic codec parity is unexpectedly low");
}

void TestVoiceDesign(const std::filesystem::path& root,
                     const std::filesystem::path& artifacts) {
  const auto expected_prompt =
      ReadNpy<float>(artifacts / "greedy_prompt.npy", "<f4");
  const auto expected_codes =
      ReadNpy<std::int32_t>(artifacts / "greedy_codes.npy", "<i4");
  std::string error;
  auto runtime =
      qwen3_tts_hip::TalkerHipRuntime::Create(root.string(), 1024, &error);
  Check(runtime != nullptr, error);
  qwen3_tts::Tokenizer tokenizer;
  Check(qwen3_tts::Tokenizer::Load(root, &tokenizer, &error), error);
  std::vector<std::uint32_t> input_ids;
  std::vector<std::uint32_t> instruction_ids;
  Check(tokenizer.EncodeAssistantPrompt(
            "When the first light reaches the harbor, speak gently, with quiet "
            "confidence and a warm, resonant tone.",
            &input_ids, &error),
        error);
  Check(tokenizer.EncodeInstructionPrompt(
            "A warm adult voice with clear diction, calm confidence, "
            "restrained emotion, and a slightly low pitch.",
            &instruction_ids, &error),
        error);
  qwen3_tts_hip::TalkerPromptOutput prompt;
  Check(runtime->BuildVoiceDesignPrompt(input_ids, instruction_ids, "english",
                                        &prompt, &error),
        error);
  Check(prompt.tokens == 58, "VoiceDesign prompt length");
  const double cosine = Cosine(prompt.embeddings, expected_prompt.values);
  std::cout << "voice_design_prompt_cosine=" << cosine << '\n';
  Check(cosine > 0.9999, "VoiceDesign prompt parity");
  CheckGreedyCodes(runtime.get(), prompt, expected_codes.values);
}

void TestBase(const std::filesystem::path& root,
              const std::filesystem::path& artifacts) {
  const auto expected_prompt =
      ReadNpy<float>(artifacts / "greedy_prompt.npy", "<f4");
  const auto expected_codes =
      ReadNpy<std::int32_t>(artifacts / "greedy_codes.npy", "<i4");
  const auto reference_codes =
      ReadNpy<std::int32_t>(artifacts / "reference_codes.npy", "<i4");
  const auto speaker =
      ReadNpy<float>(artifacts / "reference_speaker_embedding.npy", "<f4");
  std::vector<std::uint32_t> unsigned_codes(reference_codes.values.size());
  std::ranges::transform(
      reference_codes.values, unsigned_codes.begin(),
      [](std::int32_t code) { return static_cast<std::uint32_t>(code); });

  std::string error;
  auto runtime =
      qwen3_tts_hip::TalkerHipRuntime::Create(root.string(), 1024, &error);
  Check(runtime != nullptr, error);
  qwen3_tts::Tokenizer tokenizer;
  Check(qwen3_tts::Tokenizer::Load(root, &tokenizer, &error), error);
  std::vector<std::uint32_t> input_ids;
  std::vector<std::uint32_t> reference_ids;
  Check(tokenizer.EncodeAssistantPrompt(
            "Good one. Okay, fine, I am just going to leave this sock monkey "
            "here. Goodbye.",
            &input_ids, &error),
        error);
  Check(tokenizer.EncodeReferencePrompt(
            "Okay. Yeah. I resent you. I love you. I respect you. But you know "
            "what? You blew it! And thanks to you.",
            &reference_ids, &error),
        error);
  qwen3_tts_hip::TalkerPromptOutput prompt;
  Check(runtime->BuildVoiceClonePrompt(
            {
                .input_ids = input_ids,
                .reference_ids = reference_ids,
                .reference_codes = unsigned_codes,
                .speaker_embedding = speaker.values,
                .reference_frames = reference_codes.shape[0],
                .language = "auto",
                .icl_mode = true,
            },
            &prompt, &error),
        error);
  Check(prompt.tokens == 110, "Base ICL prompt length");
  const double cosine = Cosine(prompt.embeddings, expected_prompt.values);
  std::cout << "base_prompt_cosine=" << cosine << '\n';
  Check(cosine > 0.999, "Base ICL prompt parity");
  CheckGreedyCodes(runtime.get(), prompt, expected_codes.values);
}

}  // namespace

int main() {
  const std::filesystem::path voice_design_root =
      "/home/fbozzo/projects/Qwen3-TTS-12Hz-1.7B-VoiceDesign";
  const std::filesystem::path base_root =
      "/home/fbozzo/projects/audio.cpp/models/Qwen3-TTS-12Hz-1.7B-Base";
  const std::filesystem::path artifact_root =
      "/home/fbozzo/projects/strix-halo.cpp/artifacts/qwen3_tts";
  if (!std::filesystem::is_regular_file(voice_design_root /
                                        "model.safetensors") ||
      !std::filesystem::is_regular_file(base_root / "model.safetensors") ||
      !std::filesystem::is_regular_file(artifact_root / "voice_design" /
                                        "greedy_prompt.npy") ||
      !std::filesystem::is_regular_file(artifact_root / "base" /
                                        "greedy_prompt.npy")) {
    std::cerr << "SKIP qwen3_tts_variant_talker_hip_test: models or artifacts "
                 "are unavailable\n";
    return 77;
  }
  TestVoiceDesign(voice_design_root, artifact_root / "voice_design");
  TestBase(base_root, artifact_root / "base");
  std::cout << "PASS qwen3_tts_variant_talker_hip_test\n";
  return 0;
}

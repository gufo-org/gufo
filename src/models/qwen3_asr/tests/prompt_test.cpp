#include "src/models/qwen3_asr/prompt.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen3_asr/tokenizer.hpp"

namespace qwen3_asr = gufo::models::qwen3_asr;

namespace {

[[noreturn]] void Fail(std::string_view message) {
  std::cerr << "FAIL qwen3_asr_prompt_test: " << message << '\n';
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
std::vector<Element> ReadNpy(const std::filesystem::path& path,
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
  const std::size_t shape_key = header.find("'shape'");
  const std::size_t shape_open = header.find('(', shape_key);
  const std::size_t shape_close = header.find(')', shape_open);
  Check(shape_key != std::string::npos && shape_open != std::string::npos &&
            shape_close != std::string::npos,
        "cannot parse reference NPY shape");
  std::size_t elements = 1;
  for (std::size_t position = shape_open + 1U; position < shape_close;) {
    while (position < shape_close &&
           (header[position] == ' ' || header[position] == ',')) {
      ++position;
    }
    if (position == shape_close) {
      break;
    }
    std::size_t end = position;
    while (end < shape_close && header[end] >= '0' && header[end] <= '9') {
      ++end;
    }
    Check(end != position, "invalid reference NPY dimension");
    const std::size_t dimension =
        std::stoull(header.substr(position, end - position));
    Check(dimension == 0U ||
              elements <= std::numeric_limits<std::size_t>::max() / dimension,
          "reference NPY shape overflow");
    elements *= dimension;
    position = end;
  }
  std::vector<Element> result(elements);
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(result.size() * sizeof(Element)));
  Check(input.gcount() ==
            static_cast<std::streamsize>(result.size() * sizeof(Element)),
        "reference NPY is truncated");
  return result;
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
  if (!std::filesystem::is_regular_file(model_root / "vocab.json") ||
      !std::filesystem::is_regular_file(artifacts / "prompt_ids.npy") ||
      !std::filesystem::is_regular_file(artifacts / "generated_ids.npy")) {
    std::cerr << "SKIP qwen3_asr_prompt_test: model or artifacts unavailable\n";
    return 77;
  }

  qwen3_asr::Tokenizer tokenizer;
  std::string error;
  Check(qwen3_asr::Tokenizer::Load(model_root, &tokenizer, &error), error);
  const std::vector<std::int32_t> expected_prompt =
      ReadNpy<std::int32_t>(artifacts / "prompt_ids.npy", "<i4");
  const std::vector<std::uint32_t> prompt = qwen3_asr::BuildPrompt(
      tokenizer,
      {.context = "", .language = std::nullopt, .audio_embedding_tokens = 196});
  Check(prompt.size() == expected_prompt.size(),
        "official prompt token count differs");
  for (std::size_t index = 0; index < prompt.size(); ++index) {
    Check(prompt[index] == static_cast<std::uint32_t>(expected_prompt[index]),
          "official prompt token differs");
  }

  Check(tokenizer.Encode("meeting about Strix Halo") ==
            std::vector<std::uint32_t>({61249, 911, 4509, 941, 45249}),
        "context tokenization differs from the official tokenizer");
  Check(tokenizer.Encode("language English<asr_text>") ==
            std::vector<std::uint32_t>({11528, 6364, 151704}),
        "forced-language suffix tokenization differs");

  const std::vector<std::int32_t> generated =
      ReadNpy<std::int32_t>(artifacts / "generated_ids.npy", "<i4");
  std::vector<std::uint32_t> generated_u32;
  generated_u32.reserve(generated.size());
  for (const std::int32_t id : generated) {
    generated_u32.push_back(static_cast<std::uint32_t>(id));
  }
  const std::string decoded = tokenizer.Decode(generated_u32, true);
  const qwen3_asr::Transcription transcription =
      qwen3_asr::ParseTranscription(decoded, std::nullopt);
  Check(transcription.language == "English", "language parsing differs");
  Check(transcription.text ==
            "Uh huh. Oh yeah, yeah. He wasn't even that big when I started "
            "listening to him, but and his solo music didn't do overly well, "
            "but he did very well when he started writing for other people.",
        "decoded official transcript differs");

  std::cout << "PASS qwen3_asr_prompt_test tokens=" << prompt.size()
            << " language=" << transcription.language << '\n';
  return 0;
}

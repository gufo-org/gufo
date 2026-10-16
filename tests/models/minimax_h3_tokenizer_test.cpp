#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/minimax_h3/prompt_encoder.hpp"
#include "src/models/minimax_h3/tokenizer.hpp"
#include "src/models/qwen/control_tokens.hpp"

using gufo::tokenization::kImStart;

namespace {

using gufo::minimax_h3::Tokenizer;

int failures = 0;

#define CHECK(condition)                                          \
  do {                                                            \
    if (!(condition)) {                                           \
      std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << ": " \
                << #condition << '\n';                            \
      ++failures;                                                 \
    }                                                             \
  } while (false)

std::string Codepoint(std::uint32_t value) {
  std::string output;
  if (value <= 0x7FU) {
    output.push_back(static_cast<char>(value));
  } else if (value <= 0x7FFU) {
    output.push_back(static_cast<char>(0xC0U | (value >> 6U)));
    output.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
  } else {
    output.push_back(static_cast<char>(0xE0U | (value >> 12U)));
    output.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
  }
  return output;
}

std::string ByteSymbol(std::uint32_t byte) {
  const bool visible = (byte >= '!' && byte <= '~') ||
                       (byte >= 0xA1U && byte <= 0xACU) ||
                       (byte >= 0xAEU && byte <= 0xFFU);
  if (visible) {
    return Codepoint(byte);
  }
  std::uint32_t extra = 0;
  for (std::uint32_t candidate = 0; candidate < byte; ++candidate) {
    const bool candidate_visible = (candidate >= '!' && candidate <= '~') ||
                                   (candidate >= 0xA1U && candidate <= 0xACU) ||
                                   (candidate >= 0xAEU && candidate <= 0xFFU);
    if (!candidate_visible) {
      ++extra;
    }
  }
  return Codepoint(256U + extra);
}

std::string JsonString(std::string_view value) {
  std::string output{"\""};
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"':
        output += "\\\"";
        break;
      case '\\':
        output += "\\\\";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        output.push_back(static_cast<char>(byte));
        break;
    }
  }
  output.push_back('"');
  return output;
}

void WriteTinyTokenizer(const std::filesystem::path& path,
                        std::string_view normalizer, bool lstrip) {
  std::ofstream output(path);
  output << "{\"normalizer\":{\"type\":" << JsonString(normalizer)
         << "},\"model\":{\"type\":\"BPE\",\"unk_token\":null,\"vocab\":{";
  for (std::uint32_t byte = 0; byte < 256; ++byte) {
    if (byte != 0) {
      output << ',';
    }
    output << JsonString(ByteSymbol(byte)) << ':' << byte;
  }
  output << ",\"AA\":500},\"merges\":[\"A A\"]},\"added_tokens\":["
         << "{\"id\":151643,\"content\":\"<|endoftext|>\","
            "\"single_word\":false,\"lstrip\":"
         << (lstrip ? "true" : "false")
         << ",\"rstrip\":false,\"normalized\":false,\"special\":true},"
         << "{\"id\":151644,\"content\":\"<|im_start|>\","
            "\"single_word\":false,\"lstrip\":false,\"rstrip\":false,"
            "\"normalized\":false,\"special\":true}]}";
}

void CheckIds(const Tokenizer& tokenizer, std::string_view text,
              std::initializer_list<std::uint32_t> expected) {
  std::string error;
  std::vector<std::uint32_t> ids;
  CHECK(tokenizer.Encode(text, true, &ids, &error));
  CHECK(ids == std::vector<std::uint32_t>(expected));
  std::string decoded;
  CHECK(tokenizer.Decode(ids, &decoded, &error));
  CHECK(decoded == text);
}

void CheckSynthetic(const std::filesystem::path& directory) {
  const auto valid = directory / "tokenizer-valid.json";
  WriteTinyTokenizer(valid, "NFC", false);
  Tokenizer tokenizer;
  std::string error;
  CHECK(Tokenizer::Load(
      valid, Tokenizer::LoadOptions{.require_pinned_h3_contract = false},
      &tokenizer, &error));
  CheckIds(tokenizer, "AA", {500});
  CheckIds(tokenizer, " A", {32, 65});
  CheckIds(tokenizer, "\xC3\xA9", {195, 169});
  {
    std::vector<std::uint32_t> ids;
    CHECK(tokenizer.Encode("e\xCC\x81", true, &ids, &error));
    CHECK(ids == std::vector<std::uint32_t>({195, 169}));
  }
  CheckIds(tokenizer, kImStart, {151644});
  {
    std::vector<std::uint32_t> ids;
    CHECK(tokenizer.Encode("", false, &ids, &error));
    CHECK(ids.empty());
    CHECK(tokenizer.EncodePrompt("", &ids, &error));
    CHECK(ids == std::vector<std::uint32_t>({Tokenizer::kPadTokenId}));
  }
  {
    std::vector<std::uint32_t> ids;
    const std::string invalid{"\xF0\x28\x8C\x28", 4};
    CHECK(!tokenizer.Encode(invalid, true, &ids, &error));
  }
  {
    // A byte bound would reject both valid prompts. BPE and NFC determine
    // the sequence length the encoder actually receives.
    std::vector<std::uint32_t> ids;
    CHECK(tokenizer.EncodePrompt(std::string(8192, 'A'), &ids, &error));
    CHECK(ids.size() == 4096);
    CHECK(std::ranges::all_of(ids, [](auto id) { return id == 500; }));
    std::string unicode;
    for (int i = 0; i < 2048; ++i)
      unicode += "e\xCC\x81";
    CHECK(unicode.size() == 6144);
    CHECK(tokenizer.EncodePrompt(unicode, &ids, &error));
    CHECK(ids.size() == 4096);
  }

  const auto wrong_normalizer = directory / "tokenizer-nfd.json";
  WriteTinyTokenizer(wrong_normalizer, "NFD", false);
  CHECK(!Tokenizer::Load(
      wrong_normalizer,
      Tokenizer::LoadOptions{.require_pinned_h3_contract = false}, &tokenizer,
      &error));

  const auto unsupported_added = directory / "tokenizer-lstrip.json";
  WriteTinyTokenizer(unsupported_added, "NFC", true);
  CHECK(!Tokenizer::Load(
      unsupported_added,
      Tokenizer::LoadOptions{.require_pinned_h3_contract = false}, &tokenizer,
      &error));
  CHECK(!Tokenizer::Load(valid, &tokenizer, &error));
}

void CheckPinnedCheckpoint() {
  const char* root = std::getenv("GUFO_H3_MODEL_ROOT");
  if (root == nullptr || *root == '\0') {
    std::cout << "SKIP pinned tokenizer corpus: GUFO_H3_MODEL_ROOT unset\n";
    return;
  }
  const auto path =
      std::filesystem::path(root) / "FL2VA/tokenizer/tokenizer.json";
  Tokenizer tokenizer;
  std::string error;
  CHECK(Tokenizer::Load(path, &tokenizer, &error));
  CheckIds(tokenizer, "A red fox walking through snow",
           {32, 2518, 38835, 11435, 1526, 11794});
  CheckIds(
      tokenizer, "Hello, WORLD!  2026\n中文 café's",
      {9707, 11, 50891, 0, 220, 220, 17, 15, 17, 21, 198, 104811, 51950, 594});
  CheckIds(tokenizer, "🙂a!\n", {145080, 64, 4894});
  CheckIds(tokenizer, kImStart, {151644});
  CheckIds(
      tokenizer, "A cinematic close-up of a clockwork bird taking flight.",
      {32, 64665, 3265, 5239, 315, 264, 8866, 1778, 11958, 4633, 10971, 13});
  std::string unicode;
  for (int i = 0; i < 2048; ++i)
    unicode += "e\xCC\x81";
  for (const auto& prompt : {std::string(8192, 'A'), unicode}) {
    std::vector<std::uint32_t> ids;
    CHECK(tokenizer.EncodePrompt(prompt, &ids, &error));
    CHECK(!ids.empty() && ids.size() <= 4096);
    std::cout << "Pinned H3 tokenizer: bytes=" << prompt.size()
              << " tokens=" << ids.size() << '\n';
  }
}

void CheckNoCpuTensorFallback() {
#if !defined(ENGINE_ENABLE_HIP)
  gufo::minimax_h3::ModelInventory inventory;
  const std::vector<std::uint32_t> ids = {Tokenizer::kPadTokenId};
  gufo::minimax_h3::PromptEmbedding output;
  gufo::minimax_h3::PromptEncoderTelemetry telemetry;
  std::string error;
  CHECK(!gufo::minimax_h3::EncodePromptLayer50(inventory, ids, {}, nullptr,
                                               &output, &telemetry, &error));
  CHECK(error == "MiniMax H3 prompt encoder requires ENGINE_ENABLE_HIP");
#endif
}

void CheckPromptTokenLimit() {
#if defined(ENGINE_ENABLE_HIP)
  // Validation happens before inventory access or GPU allocation.
  gufo::minimax_h3::ModelInventory inventory;
  const std::vector<std::uint32_t> ids(4097, Tokenizer::kPadTokenId);
  gufo::minimax_h3::PromptEmbedding output;
  gufo::minimax_h3::PromptEncoderTelemetry telemetry;
  std::string error;
  CHECK(!gufo::minimax_h3::EncodePromptLayer50(inventory, ids, {}, nullptr,
                                               &output, &telemetry, &error));
  CHECK(error == "MiniMax H3 prompt exceeds 4096 tokens");
  CHECK(output.values.empty() && telemetry.peak_device_bytes == 0);
#endif
}

}  // namespace

int main() {
  const auto directory =
      std::filesystem::temp_directory_path() / "gufo-h3-tokenizer-test";
  std::error_code filesystem_error;
  std::filesystem::remove_all(directory, filesystem_error);
  std::filesystem::create_directories(directory, filesystem_error);
  CHECK(!filesystem_error);
  CheckSynthetic(directory);
  CheckPinnedCheckpoint();
  CheckNoCpuTensorFallback();
  CheckPromptTokenLimit();
  std::filesystem::remove_all(directory, filesystem_error);
  if (failures != 0) {
    std::cerr << failures << " MiniMax H3 tokenizer checks failed\n";
    return 1;
  }
  std::cout << "MiniMax H3 tokenizer checks passed\n";
  return 0;
}

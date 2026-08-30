#include "src/models/qwen3_asr/tokenizer.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen3_tts/json.hpp"

namespace gufo::models::qwen3_asr {
namespace {

namespace json = gufo::models::qwen3_tts::json;

constexpr std::size_t kMaximumVocabularyBytes = 16U << 20U;
constexpr std::size_t kMaximumTokenizerConfigBytes = 1U << 20U;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::string ReadFile(const std::filesystem::path& path,
                     std::size_t maximum_bytes) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open " + path.string());
  }
  input.seekg(0, std::ios::end);
  const std::streamoff size = input.tellg();
  if (size < 0 || static_cast<std::uint64_t>(size) > maximum_bytes) {
    throw std::runtime_error("unexpected file size for " + path.string());
  }
  input.seekg(0, std::ios::beg);
  std::string contents(static_cast<std::size_t>(size), '\0');
  input.read(contents.data(), size);
  if (!input) {
    throw std::runtime_error("cannot read " + path.string());
  }
  return contents;
}

}  // namespace

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

bool Tokenizer::Load(const std::filesystem::path& model_root, Tokenizer* output,
                     std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-ASR tokenizer output is required");
    return false;
  }
  try {
    const json::Value vocabulary = json::parse(
        ReadFile(model_root / "vocab.json", kMaximumVocabularyBytes));
    const json::Value tokenizer_config = json::parse(ReadFile(
        model_root / "tokenizer_config.json", kMaximumTokenizerConfigBytes));
    if (!vocabulary.is_object() || !tokenizer_config.is_object()) {
      throw std::runtime_error("tokenizer JSON root must be an object");
    }

    std::size_t maximum_id = 0;
    for (const auto& [symbol, identifier] : vocabulary.members()) {
      (void)symbol;
      if (!identifier.is_number()) {
        throw std::runtime_error("vocabulary ID must be numeric");
      }
      maximum_id = std::max(maximum_id, identifier.as_size());
    }
    const json::Value* decoder = tokenizer_config.find("added_tokens_decoder");
    if (decoder == nullptr || !decoder->is_object()) {
      throw std::runtime_error("missing added_tokens_decoder");
    }
    for (const auto& [identifier, specification] : decoder->members()) {
      (void)specification;
      std::size_t consumed = 0;
      const unsigned long parsed = std::stoul(identifier, &consumed);
      if (consumed != identifier.size()) {
        throw std::runtime_error("invalid added-token ID");
      }
      maximum_id = std::max(maximum_id, static_cast<std::size_t>(parsed));
    }
    if (maximum_id >= std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("tokenizer vocabulary is too large");
    }

    std::vector<std::string> tokens(maximum_id + 1U);
    for (const auto& [symbol, identifier] : vocabulary.members()) {
      const std::size_t id = identifier.as_size();
      if (!tokens[id].empty()) {
        throw std::runtime_error("duplicate vocabulary ID");
      }
      tokens[id] = symbol;
    }

    std::unordered_map<std::string, tokenization::TokenId> special_tokens;
    std::unordered_set<std::uint32_t> special_ids;
    for (const auto& [identifier, specification] : decoder->members()) {
      if (!specification.is_object()) {
        throw std::runtime_error("invalid added-token specification");
      }
      const std::size_t id = std::stoul(identifier);
      const std::string content = specification.member_str("content");
      if (content.empty() || (!tokens[id].empty() && tokens[id] != content)) {
        throw std::runtime_error("invalid or duplicate added token");
      }
      tokens[id] = content;
      // Hugging Face recognizes every added token as an indivisible token
      // during encoding. The separate `special` flag controls only whether
      // batch_decode(skip_special_tokens=true) removes it; notably
      // <asr_text> is added but intentionally not special.
      special_tokens.emplace(content, static_cast<std::uint32_t>(id));
      const json::Value* special = specification.find("special");
      if (special != nullptr && special->is_bool() && special->as_bool()) {
        special_ids.insert(static_cast<std::uint32_t>(id));
      }
    }
    if (std::ranges::any_of(
            tokens, [](const std::string& token) { return token.empty(); })) {
      throw std::runtime_error("tokenizer has gaps in its ID space");
    }

    std::string tokenizer_error;
    const std::vector<std::string> no_merges;
    auto implementation = tokenization::QwenTokenizer::CreateFromVocabulary(
        tokens, no_merges, special_tokens, &tokenizer_error,
        tokenization::VocabularyLoadOptions{
            .eager_decoded_tokens = false,
        });
    if (implementation == nullptr) {
      throw std::runtime_error(tokenizer_error);
    }
    Tokenizer loaded;
    loaded.tokenizer_ = std::move(implementation);
    loaded.model_root_ = model_root;
    loaded.tokens_ = std::move(tokens);
    loaded.added_tokens_ = std::move(special_tokens);
    loaded.special_ids_ = std::move(special_ids);
    *output = std::move(loaded);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, "Qwen3-ASR tokenizer load failed: " +
                        std::string(exception.what()));
    return false;
  }
}

void Tokenizer::EnsureEncodingTokenizer() const {
  if (merges_loaded_) {
    return;
  }
  std::vector<std::string> merges;
  std::ifstream merge_file(model_root_ / "merges.txt");
  if (!merge_file) {
    throw std::runtime_error("cannot open " +
                             (model_root_ / "merges.txt").string());
  }
  std::string line;
  while (std::getline(merge_file, line)) {
    if (!line.empty() && !line.starts_with('#')) {
      merges.push_back(line);
    }
  }
  std::string tokenizer_error;
  auto implementation = tokenization::QwenTokenizer::CreateFromVocabulary(
      tokens_, merges, added_tokens_, &tokenizer_error,
      tokenization::VocabularyLoadOptions{
          .eager_decoded_tokens = false,
      });
  if (implementation == nullptr) {
    throw std::runtime_error(tokenizer_error);
  }
  tokenizer_ = std::move(implementation);
  merges_loaded_ = true;
}

std::vector<std::uint32_t> Tokenizer::Encode(std::string_view text) const {
  if (tokenizer_ == nullptr) {
    throw std::logic_error("Qwen3-ASR tokenizer is not loaded");
  }
  EnsureEncodingTokenizer();
  return tokenizer_->Encode(text);
}

std::string Tokenizer::Decode(std::span<const std::uint32_t> ids,
                              bool skip_special_tokens) const {
  if (tokenizer_ == nullptr) {
    throw std::logic_error("Qwen3-ASR tokenizer is not loaded");
  }
  std::string result;
  for (const std::uint32_t id : ids) {
    if (!skip_special_tokens || !special_ids_.contains(id)) {
      result.append(tokenizer_->DecodeTokenCopy(id));
    }
  }
  return result;
}

}  // namespace gufo::models::qwen3_asr

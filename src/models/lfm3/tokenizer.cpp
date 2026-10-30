#include "src/models/lfm3/tokenizer.hpp"

#include <unicode/regex.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <stdexcept>

#include "src/core/gguf_reader.hpp"
#include "src/models/lfm3/chat_template.hpp"
namespace gufo::models::lfm3 {
namespace {
// LiquidAI tokenizer.json uses the Llama 3 / LFM2 split (digits in groups of
// three), followed by byte-level BPE. Qwen35 has a different split.
const icu::RegexPattern& Pattern() {
  static const auto pattern = [] {
    UErrorCode status = U_ZERO_ERROR;
    auto* p = icu::RegexPattern::compile(
        icu::UnicodeString::fromUTF8(
            R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"),
        0, status);
    if (U_FAILURE(status))
      throw std::runtime_error("invalid LFM3 tokenizer regex");
    return std::unique_ptr<icu::RegexPattern>(p);
  }();
  return *pattern;
}
}  // namespace
std::unique_ptr<Tokenizer> Tokenizer::Load(const core::GgufReader& reader,
                                           std::string* error) {
  if (reader.GetMetadataString("tokenizer.ggml.pre") != "lfm2") {
    if (error)
      *error = "LFM3 requires tokenizer.ggml.pre=lfm2";
    return nullptr;
  }
  auto result = std::make_unique<Tokenizer>();
  result->bpe_ = tokenization::QwenTokenizer::CreateFromGguf(reader, error);
  if (!result->bpe_)
    return nullptr;
  const auto tokens = reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  const auto* entry = reader.FindMetadata("tokenizer.ggml.token_type");
  const auto* signed_types =
      entry ? std::get_if<std::vector<std::int64_t>>(&entry->value) : nullptr;
  const auto* unsigned_types =
      entry ? std::get_if<std::vector<std::uint64_t>>(&entry->value) : nullptr;
  if ((!signed_types || signed_types->size() != tokens.size()) &&
      (!unsigned_types || unsigned_types->size() != tokens.size())) {
    if (error)
      *error = "LFM3 requires per-token GGUF types";
    return nullptr;
  }
  result->special_pieces_.resize(tokens.size());
  result->control_.resize(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    auto type =
        signed_types ? std::uint64_t((*signed_types)[i]) : (*unsigned_types)[i];
    result->control_[i] = type == 3;
    // GGUF CONTROL and USER_DEFINED include LFM3's native tool tags, which
    // the Qwen tokenizer's spelling catalogue does not enumerate.
    if (type == 3 || type == 4 || result->bpe_->IsSpecialToken(i)) {
      result->special_tokens_.emplace_back(tokens[i], i);
      result->special_pieces_[i] = tokens[i];
    }
  }
  return result;
}
std::vector<std::uint32_t> Tokenizer::Encode(std::string_view text,
                                             bool special) const {
  return EncodeImpl(text, special, {});
}
std::vector<std::uint32_t> Tokenizer::EncodeFramed(
    std::string_view text,
    std::span<const std::pair<std::size_t, std::size_t>> literals) const {
  return EncodeImpl(text, true, literals);
}
std::vector<std::uint32_t> Tokenizer::EncodeImpl(
    std::string_view text, bool special,
    std::span<const std::pair<std::size_t, std::size_t>> literals) const {
  std::vector<std::uint32_t> ids;
  auto regular = [&](std::string_view part) {
    auto unicode = icu::UnicodeString::fromUTF8(
        icu::StringPiece(part.data(), part.size()));
    UErrorCode status = U_ZERO_ERROR;
    std::unique_ptr<icu::RegexMatcher> matcher(
        Pattern().matcher(unicode, status));
    if (U_FAILURE(status))
      throw std::runtime_error("LFM3 tokenizer regex failed");
    while (matcher->find(status)) {
      std::string chunk;
      matcher->group(status).toUTF8String(chunk);
      auto row = bpe_->Encode(chunk, {.parse_special_tokens = false});
      ids.insert(ids.end(), row.begin(), row.end());
    }
    if (U_FAILURE(status))
      throw std::runtime_error("LFM3 tokenizer matching failed");
  };
  if (!special) {
    regular(text);
    return ids;
  }
  const auto literal_overlap = [&](std::size_t start, std::size_t size) {
    const auto it = std::lower_bound(literals.begin(), literals.end(), start,
                                     [](const auto& span, std::size_t pos) {
                                       return span.first + span.second <= pos;
                                     });
    return it != literals.end() && it->first < start + size;
  };
  const auto next = [&](std::string_view spelling, std::size_t from) {
    auto at = text.find(spelling, from);
    while (at != text.npos && literal_overlap(at, spelling.size()))
      at = text.find(spelling, at + 1);
    return at;
  };
  // Each spelling scans the input once, including long histories with many
  // native tags. Later control occurrences stay cached until consumed.
  std::vector<std::size_t> occurrence(special_tokens_.size());
  for (std::size_t i = 0; i < occurrence.size(); ++i)
    occurrence[i] = next(special_tokens_[i].first, 0);
  std::size_t offset = 0;
  while (offset < text.size()) {
    std::size_t first = text.size(), length = 0;
    std::uint32_t token = 0;
    for (std::size_t i = 0; i < occurrence.size(); ++i) {
      const auto& [spelling, id] = special_tokens_[i];
      if (occurrence[i] < offset)
        occurrence[i] = next(spelling, offset);
      auto at = occurrence[i];
      if (at < first || (at == first && spelling.size() > length)) {
        first = at;
        length = spelling.size();
        token = id;
      }
    }
    regular(text.substr(offset, first - offset));
    if (!length)
      break;
    ids.push_back(token);
    offset = first + length;
  }
  return ids;
}
std::string Tokenizer::Render(
    std::span<const std::pair<std::string, std::string>> messages,
    bool thinking) {
  std::vector<tokenization::ChatMessage> chat;
  for (const auto& [role, content] : messages) {
    const auto kind =
        role == "system" ? tokenization::ChatRole::kSystem
        : role == "user" ? tokenization::ChatRole::kUser
        : role == "assistant"
            ? tokenization::ChatRole::kAssistant
            : throw std::invalid_argument("unsupported LFM3 message role");
    chat.emplace_back(kind, content);
  }
  ReasoningOptions options;
  options.enabled = thinking;
  return RenderChat(chat, {}, options);
}
}  // namespace gufo::models::lfm3

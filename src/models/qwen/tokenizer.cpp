#include "src/models/qwen/tokenizer.hpp"

#include <unicode/bytestream.h>
#include <unicode/normalizer2.h>
#include <unicode/uchar.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::tokenization {

namespace {

std::string FormatHexByteToken(std::uint8_t byte_val) {
  char buf[8];
  static_cast<void>(std::snprintf(buf, sizeof(buf), "<0x%02X>", byte_val));
  return {buf};
}

constexpr int HexCharToInt(char c) noexcept {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  }
  if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  }
  return -1;
}

std::string ByteToGpt2Utf8(std::uint8_t b) {
  static const auto b2u_map = []() {
    std::array<std::string, 256> table;
    for (int i = '!'; i <= '~'; ++i) {
      table[i] = std::string(1, static_cast<char>(i));
    }
    for (int i = 161; i <= 172; ++i) {
      char buf[3] = {static_cast<char>(0xC0 | (i >> 6)),
                     static_cast<char>(0x80 | (i & 0x3F)), 0};
      table[i] = buf;
    }
    for (int i = 174; i <= 255; ++i) {
      char buf[3] = {static_cast<char>(0xC0 | (i >> 6)),
                     static_cast<char>(0x80 | (i & 0x3F)), 0};
      table[i] = buf;
    }
    int n = 0;
    for (int i = 0; i < 256; ++i) {
      if ((i < '!' || i > '~') && (i < 161 || i > 172) &&
          (i < 174 || i > 255)) {
        const int cp = 256 + n;
        char buf[3] = {static_cast<char>(0xC0 | (cp >> 6)),
                       static_cast<char>(0x80 | (cp & 0x3F)), 0};
        table[i] = buf;
        ++n;
      }
    }
    return table;
  }();
  return b2u_map[b];
}

std::string UnescapeGpt2Bytes(std::string_view text) {
  static const auto u2b_map = []() {
    std::unordered_map<char32_t, std::uint8_t> m;
    for (int b = '!'; b <= '~'; ++b) {
      m[static_cast<char32_t>(b)] = static_cast<std::uint8_t>(b);
    }
    for (int b = 161; b <= 172; ++b) {
      m[static_cast<char32_t>(b)] = static_cast<std::uint8_t>(b);
    }
    for (int b = 174; b <= 255; ++b) {
      m[static_cast<char32_t>(b)] = static_cast<std::uint8_t>(b);
    }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      if ((b < '!' || b > '~') && (b < 161 || b > 172) &&
          (b < 174 || b > 255)) {
        m[static_cast<char32_t>(256 + n)] = static_cast<std::uint8_t>(b);
        ++n;
      }
    }
    return m;
  }();

  std::string result;
  result.reserve(text.size());

  std::size_t i = 0;
  while (i < text.size()) {
    const auto b0 = static_cast<unsigned char>(text[i]);
    char32_t cp = b0;
    std::size_t len = 1;

    if ((b0 & 0xE0) == 0xC0 && i + 1 < text.size()) {
      cp =
          ((b0 & 0x1F) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3F);
      len = 2;
    } else if ((b0 & 0xF0) == 0xE0 && i + 2 < text.size()) {
      cp = ((b0 & 0x0F) << 12) |
           ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
           (static_cast<unsigned char>(text[i + 2]) & 0x3F);
      len = 3;
    } else if ((b0 & 0xF8) == 0xF0 && i + 3 < text.size()) {
      cp = ((b0 & 0x07) << 18) |
           ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12) |
           ((static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6) |
           (static_cast<unsigned char>(text[i + 3]) & 0x3F);
      len = 4;
    }

    auto it = u2b_map.find(cp);
    if (it != u2b_map.end()) {
      result.push_back(static_cast<char>(it->second));
    } else {
      result.append(text.substr(i, len));
    }
    i += len;
  }

  return result;
}

struct Utf8CodePoint {
  UChar32 value{0};
  std::size_t length{1};
};

Utf8CodePoint DecodeUtf8(std::string_view text, std::size_t offset) noexcept {
  const auto first = static_cast<std::uint8_t>(text[offset]);
  if (first < 0x80U) {
    return {.value = first, .length = 1};
  }

  std::size_t length = 0;
  UChar32 value = 0;
  if ((first & 0xE0U) == 0xC0U) {
    length = 2;
    value = static_cast<UChar32>(first & 0x1FU);
  } else if ((first & 0xF0U) == 0xE0U) {
    length = 3;
    value = static_cast<UChar32>(first & 0x0FU);
  } else if ((first & 0xF8U) == 0xF0U) {
    length = 4;
    value = static_cast<UChar32>(first & 0x07U);
  } else {
    return {.value = first, .length = 1};
  }
  if (offset + length > text.size()) {
    return {.value = first, .length = 1};
  }
  for (std::size_t index = 1; index < length; ++index) {
    const auto byte = static_cast<std::uint8_t>(text[offset + index]);
    if ((byte & 0xC0U) != 0x80U) {
      return {.value = first, .length = 1};
    }
    value = static_cast<UChar32>((value << 6U) | (byte & 0x3FU));
  }
  return {.value = value, .length = length};
}

bool IsUnicodeLetter(UChar32 value) noexcept {
  return (U_GET_GC_MASK(value) & U_GC_L_MASK) != 0;
}

bool IsUnicodeLetterOrMark(UChar32 value) noexcept {
  return (U_GET_GC_MASK(value) & (U_GC_L_MASK | U_GC_M_MASK)) != 0;
}

bool IsUnicodeNumber(UChar32 value) noexcept {
  const auto category = static_cast<UCharCategory>(u_charType(value));
  return category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER ||
         category == U_OTHER_NUMBER;
}

bool IsUnicodeWhitespace(UChar32 value) noexcept {
  return u_isUWhiteSpace(value) != 0;
}

bool IsNewline(UChar32 value) noexcept {
  return value == '\r' || value == '\n';
}

std::size_t Qwen35ContractionEnd(std::string_view text,
                                 std::size_t offset) noexcept {
  if (text[offset] != '\'') {
    return offset;
  }
  constexpr std::array<std::string_view, 7> kSuffixes = {
      "s", "t", "re", "ve", "m", "ll", "d",
  };
  for (const std::string_view suffix : kSuffixes) {
    std::size_t end = offset + 1;
    bool matches = true;
    for (const char letter : suffix) {
      if (end == text.size()) {
        matches = false;
        break;
      }
      const auto next = DecodeUtf8(text, end);
      if (u_foldCase(next.value, U_FOLD_CASE_DEFAULT) != letter) {
        matches = false;
        break;
      }
      end += next.length;
    }
    if (matches) {
      return end;
    }
  }
  return offset;
}

std::size_t Qwen35PieceEnd(std::string_view text, std::size_t offset) {
  if (const std::size_t contraction = Qwen35ContractionEnd(text, offset);
      contraction != offset) {
    return contraction;
  }

  const Utf8CodePoint first = DecodeUtf8(text, offset);
  if (IsUnicodeLetterOrMark(first.value)) {
    std::size_t end = offset + first.length;
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (!IsUnicodeLetterOrMark(next.value)) {
        break;
      }
      end += next.length;
    }
    return end;
  }

  if (!IsNewline(first.value) && !IsUnicodeLetter(first.value) &&
      !IsUnicodeNumber(first.value) && offset + first.length < text.size()) {
    const Utf8CodePoint next = DecodeUtf8(text, offset + first.length);
    if (IsUnicodeLetterOrMark(next.value)) {
      std::size_t end = offset + first.length + next.length;
      while (end < text.size()) {
        const Utf8CodePoint letter = DecodeUtf8(text, end);
        if (!IsUnicodeLetterOrMark(letter.value)) {
          break;
        }
        end += letter.length;
      }
      return end;
    }
  }

  if (IsUnicodeNumber(first.value)) {
    return offset + first.length;
  }

  std::size_t punctuation_start = offset;
  if (first.value == ' ' && offset + first.length < text.size()) {
    const Utf8CodePoint next = DecodeUtf8(text, offset + first.length);
    if (!IsUnicodeWhitespace(next.value) &&
        !IsUnicodeLetterOrMark(next.value) && !IsUnicodeNumber(next.value)) {
      punctuation_start += first.length;
    }
  }
  const Utf8CodePoint punctuation = DecodeUtf8(text, punctuation_start);
  if (!IsUnicodeWhitespace(punctuation.value) &&
      !IsUnicodeLetterOrMark(punctuation.value) &&
      !IsUnicodeNumber(punctuation.value)) {
    std::size_t end = punctuation_start;
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (IsUnicodeWhitespace(next.value) ||
          IsUnicodeLetterOrMark(next.value) || IsUnicodeNumber(next.value)) {
        break;
      }
      end += next.length;
    }
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (!IsNewline(next.value)) {
        break;
      }
      end += next.length;
    }
    return end;
  }

  if (IsUnicodeWhitespace(first.value)) {
    std::size_t end = offset;
    std::size_t last_start = offset;
    std::size_t last_newline_end = offset;
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (!IsUnicodeWhitespace(next.value)) {
        break;
      }
      last_start = end;
      end += next.length;
      if (IsNewline(next.value)) {
        last_newline_end = end;
      }
    }
    if (last_newline_end != offset) {
      return last_newline_end;
    }
    // \s+(?!\S) consumes all trailing whitespace, but backtracks one
    // codepoint before non-whitespace so the next piece can own its prefix.
    return end < text.size() && last_start > offset ? last_start : end;
  }

  return offset + first.length;
}

}  // namespace

std::unique_ptr<QwenTokenizer> QwenTokenizer::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto token_views =
      reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  if (token_views.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF metadata missing 'tokenizer.ggml.tokens'";
    }
    return nullptr;
  }

  std::vector<std::string> tokens;
  tokens.reserve(token_views.size());
  for (const auto& tv : token_views) {
    tokens.emplace_back(tv);
  }

  const auto merge_views =
      reader.GetMetadataStringArray("tokenizer.ggml.merges");
  std::vector<std::string> merges;
  merges.reserve(merge_views.size());
  for (const auto& mv : merge_views) {
    merges.emplace_back(mv);
  }

  std::unordered_map<std::string, TokenId> special_tokens;
  // Dynamic lookup of special tokens from GGUF vocabulary table
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const auto& t = tokens[i];
    if (t == "<|endoftext|>" || t == "<|im_start|>" || t == "<|im_end|>" ||
        t == "<tool_call>" || t == "</tool_call>" || t == "<tool_response>" ||
        t == "</tool_response>" || t == "<think>" || t == "</think>" ||
        t == "<tts_pad>" || t == "<tts_text_bos>" || t == "<tts_text_eod>" ||
        t == "<tts_text_bos_single>" || t == "<|object_ref_start|>" ||
        t == "<|object_ref_end|>" || t == "<|vision_start|>" ||
        t == "<|vision_end|>" || t == "<|image_pad|>" || t == "<|video_pad|>" ||
        t == "<|quad_start|>" || t == "<|quad_end|>" ||
        (t.size() >= 4 && t.starts_with("<|") && t.ends_with("|>"))) {
      special_tokens[t] = static_cast<TokenId>(i);
    }
  }

  auto tokenizer =
      CreateFromVocabulary(tokens, merges, special_tokens, error_msg);
  if (tokenizer == nullptr) {
    return nullptr;
  }
  if (reader.GetMetadataString("tokenizer.ggml.pre") ==
      std::optional<std::string_view>{"qwen35"}) {
    tokenizer->pre_tokenizer_ = PreTokenizer::kQwen35;
  }

  if (auto eos = reader.GetMetadataUint32("tokenizer.ggml.eos_token_id")) {
    tokenizer->eos_token_id_ = *eos;
  } else if (auto it = special_tokens.find("<|im_end|>");
             it != special_tokens.end()) {
    tokenizer->eos_token_id_ = it->second;
  } else if (auto it = special_tokens.find("<|endoftext|>");
             it != special_tokens.end()) {
    tokenizer->eos_token_id_ = it->second;
  }

  if (auto bos = reader.GetMetadataUint32("tokenizer.ggml.bos_token_id")) {
    tokenizer->bos_token_id_ = *bos;
  } else if (auto it = special_tokens.find("<|im_start|>");
             it != special_tokens.end()) {
    tokenizer->bos_token_id_ = it->second;
  }

  if (auto pad = reader.GetMetadataUint32("tokenizer.ggml.padding_token_id")) {
    tokenizer->pad_token_id_ = *pad;
  } else if (auto it = special_tokens.find("<|endoftext|>");
             it != special_tokens.end()) {
    tokenizer->pad_token_id_ = it->second;
  }

  return tokenizer;
}

std::unique_ptr<QwenTokenizer> QwenTokenizer::CreateFromVocabulary(
    std::span<const std::string> tokens, std::span<const std::string> merges,
    const std::unordered_map<std::string, TokenId>& special_tokens,
    std::string* error_msg, VocabularyLoadOptions load_options) {
  if (tokens.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "Vocabulary cannot be empty";
    }
    return nullptr;
  }

  auto tokenizer = std::unique_ptr<QwenTokenizer>(new QwenTokenizer());
  tokenizer->id_to_token_.reserve(tokens.size());
  tokenizer->token_to_id_.reserve(tokens.size());

  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokenizer->id_to_token_.push_back(tokens[i]);
    tokenizer->token_to_id_[tokens[i]] = static_cast<TokenId>(i);
  }

  // Parse BPE merges
  tokenizer->merge_ranks_.reserve(merges.size());
  for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
    const auto& merge = merges[rank];
    const auto space_pos = merge.find(' ');
    if (space_pos == std::string::npos) {
      continue;
    }
    const std::string part1 = merge.substr(0, space_pos);
    const std::string part2 = merge.substr(space_pos + 1);

    auto it1 = tokenizer->token_to_id_.find(part1);
    auto it2 = tokenizer->token_to_id_.find(part2);
    if (it1 != tokenizer->token_to_id_.end() &&
        it2 != tokenizer->token_to_id_.end()) {
      tokenizer->merge_ranks_[{it1->second, it2->second}] = rank;
    }
  }

  // Register special tokens
  for (const auto& [name, id] : special_tokens) {
    tokenizer->special_token_to_id_[name] = id;
    tokenizer->is_special_token_[id] = true;
    if (id < tokenizer->id_to_token_.size()) {
      tokenizer->id_to_token_[id] = name;
      tokenizer->token_to_id_[name] = id;
    }
  }

  tokenizer->endoftext_token_id_ =
      tokenizer->FindSpecialToken("<|endoftext|>").value_or(kInvalidTokenId);
  tokenizer->eos_token_id_ = tokenizer->FindSpecialToken("<|im_end|>")
                                 .value_or(tokenizer->endoftext_token_id_);
  tokenizer->bos_token_id_ =
      tokenizer->FindSpecialToken("<|im_start|>").value_or(kInvalidTokenId);
  tokenizer->pad_token_id_ = tokenizer->endoftext_token_id_;
  tokenizer->InitializeByteTokens(load_options.eager_decoded_tokens);
  return tokenizer;
}

void QwenTokenizer::InitializeByteTokens(bool eager_decoded_tokens) {
  if (eager_decoded_tokens) {
    id_to_decoded_token_.resize(id_to_token_.size());
    for (std::size_t i = 0; i < id_to_token_.size(); ++i) {
      if (is_special_token_.contains(static_cast<TokenId>(i))) {
        id_to_decoded_token_[i] = id_to_token_[i];
      } else {
        const auto& tok = id_to_token_[i];
        if (tok.size() == 6 && tok.starts_with("<0x") && tok.ends_with('>')) {
          const int h1 = HexCharToInt(tok[3]);
          const int h2 = HexCharToInt(tok[4]);
          if (h1 >= 0 && h2 >= 0) {
            const auto byte_val = static_cast<std::uint8_t>((h1 << 4) | h2);
            id_to_decoded_token_[i] =
                std::string(1, static_cast<char>(byte_val));
            continue;
          }
        }
        id_to_decoded_token_[i] = UnescapeGpt2Bytes(tok);
      }
    }
  }

  for (std::size_t b = 0; b < 256; ++b) {
    const auto byte_val = static_cast<std::uint8_t>(b);
    const std::string gpt2_utf8 = ByteToGpt2Utf8(byte_val);
    auto it = token_to_id_.find(gpt2_utf8);
    if (it != token_to_id_.end()) {
      byte_tokens_[b] = it->second;
    } else {
      const std::string direct_char(1, static_cast<char>(byte_val));
      auto direct_it = token_to_id_.find(direct_char);
      if (direct_it != token_to_id_.end()) {
        byte_tokens_[b] = direct_it->second;
      } else {
        const std::string hex_token = FormatHexByteToken(byte_val);
        auto hex_it = token_to_id_.find(hex_token);
        if (hex_it != token_to_id_.end()) {
          byte_tokens_[b] = hex_it->second;
        } else {
          byte_tokens_[b] = kInvalidTokenId;
        }
      }
    }
  }
}

std::vector<TokenId> QwenTokenizer::BpeMergeChunk(
    std::string_view chunk) const {
  if (chunk.empty()) {
    return {};
  }

  // Initial tokenization at byte level
  std::vector<TokenId> word_tokens;
  word_tokens.reserve(chunk.size());

  for (const unsigned char c : chunk) {
    const TokenId tid = byte_tokens_[c];
    if (tid != kInvalidTokenId) {
      word_tokens.push_back(tid);
    }
  }

  if (word_tokens.size() <= 1) {
    return word_tokens;
  }

  // Iteratively merge the highest-ranked adjacent pairs
  while (word_tokens.size() >= 2) {
    std::optional<std::uint32_t> best_rank;
    std::size_t best_idx = 0;

    for (std::size_t i = 0; i < word_tokens.size() - 1; ++i) {
      auto it = merge_ranks_.find({word_tokens[i], word_tokens[i + 1]});
      if (it != merge_ranks_.end()) {
        if (!best_rank.has_value() || it->second < *best_rank) {
          best_rank = it->second;
          best_idx = i;
        }
      }
    }

    if (!best_rank.has_value()) {
      break;
    }

    const std::string merged_str = id_to_token_[word_tokens[best_idx]] +
                                   id_to_token_[word_tokens[best_idx + 1]];
    auto merged_it = token_to_id_.find(merged_str);
    if (merged_it == token_to_id_.end()) {
      break;
    }

    word_tokens[best_idx] = merged_it->second;
    word_tokens.erase(word_tokens.begin() +
                      static_cast<std::ptrdiff_t>(best_idx + 1));
  }

  return word_tokens;
}

std::vector<TokenId> QwenTokenizer::BpeEncodeText(std::string_view text) const {
  if (pre_tokenizer_ != PreTokenizer::kQwen35) {
    return BpeMergeChunk(text);
  }

  // Qwen's tokenizer.json normalizes each non-special span before applying
  // its Unicode split regex. ASCII needs neither conversion nor allocation.
  std::string normalized;
  if (std::ranges::any_of(text,
                          [](unsigned char byte) { return byte >= 0x80; })) {
    UErrorCode status = U_ZERO_ERROR;
    const auto* nfc = icu::Normalizer2::getNFCInstance(status);
    if (U_FAILURE(status))
      throw std::runtime_error("could not initialize Qwen NFC normalization");
    icu::StringByteSink<std::string> sink(&normalized);
    nfc->normalizeUTF8(0, icu::StringPiece(text), sink, nullptr, status);
    if (U_FAILURE(status))
      throw std::runtime_error("Qwen NFC normalization failed");
    text = normalized;
  }

  std::vector<TokenId> tokens;
  std::size_t offset = 0;
  while (offset < text.size()) {
    const std::size_t end = Qwen35PieceEnd(text, offset);
    const auto piece_tokens = BpeMergeChunk(text.substr(offset, end - offset));
    tokens.insert(tokens.end(), piece_tokens.begin(), piece_tokens.end());
    offset = end;
  }
  return tokens;
}

std::vector<TokenId> QwenTokenizer::Encode(
    std::string_view text, const TokenizerOptions& options) const {
  std::vector<TokenId> tokens;
  if (text.empty()) {
    if (options.add_bos && bos_token_id_ != kInvalidTokenId) {
      tokens.push_back(bos_token_id_);
    }
    if (options.add_eos && eos_token_id_ != kInvalidTokenId) {
      tokens.push_back(eos_token_id_);
    }
    return tokens;
  }

  if (options.add_bos && bos_token_id_ != kInvalidTokenId) {
    tokens.push_back(bos_token_id_);
  }

  if (!options.parse_special_tokens || special_token_to_id_.empty()) {
    const auto chunk_tokens = BpeEncodeText(text);
    tokens.insert(tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
  } else {
    // Scan text for special token delimiters
    std::size_t pos = 0;
    while (pos < text.size()) {
      std::size_t next_special_pos = std::string_view::npos;
      std::string_view matched_special;
      TokenId matched_id = kInvalidTokenId;

      for (const auto& [special_str, id] : special_token_to_id_) {
        const auto found = text.find(special_str, pos);
        if (found != std::string_view::npos) {
          if (next_special_pos == std::string_view::npos ||
              found < next_special_pos) {
            next_special_pos = found;
            matched_special = special_str;
            matched_id = id;
          }
        }
      }

      if (next_special_pos == std::string_view::npos) {
        // No more special tokens; encode remainder
        const auto chunk = text.substr(pos);
        const auto chunk_tokens = BpeEncodeText(chunk);
        tokens.insert(tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
        break;
      }

      if (next_special_pos > pos) {
        const auto chunk = text.substr(pos, next_special_pos - pos);
        const auto chunk_tokens = BpeEncodeText(chunk);
        tokens.insert(tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
      }

      tokens.push_back(matched_id);
      pos = next_special_pos + matched_special.size();
    }
  }

  if (options.add_eos && eos_token_id_ != kInvalidTokenId) {
    tokens.push_back(eos_token_id_);
  }

  return tokens;
}

std::string QwenTokenizer::Decode(std::span<const TokenId> tokens) const {
  std::string result;
  for (const TokenId id : tokens) {
    result.append(DecodeTokenCopy(id));
  }
  return result;
}

std::string_view QwenTokenizer::DecodeToken(TokenId token_id) const noexcept {
  if (token_id < id_to_decoded_token_.size()) {
    return id_to_decoded_token_[token_id];
  }
  if (token_id < id_to_token_.size()) {
    return id_to_token_[token_id];
  }
  return "";
}

std::string QwenTokenizer::DecodeTokenCopy(TokenId token_id) const {
  if (token_id >= id_to_token_.size()) {
    return {};
  }
  if (!id_to_decoded_token_.empty()) {
    return id_to_decoded_token_[token_id];
  }
  if (is_special_token_.contains(token_id)) {
    return id_to_token_[token_id];
  }
  const std::string& token = id_to_token_[token_id];
  if (token.size() == 6 && token.starts_with("<0x") && token.ends_with('>')) {
    const int high = HexCharToInt(token[3]);
    const int low = HexCharToInt(token[4]);
    if (high >= 0 && low >= 0) {
      return std::string(
          1, static_cast<char>(static_cast<std::uint8_t>((high << 4) | low)));
    }
  }
  return UnescapeGpt2Bytes(token);
}

std::optional<TokenId> QwenTokenizer::FindSpecialToken(
    std::string_view token_str) const noexcept {
  auto it = special_token_to_id_.find(std::string(token_str));
  if (it != special_token_to_id_.end()) {
    return it->second;
  }
  return std::nullopt;
}

bool QwenTokenizer::IsSpecialToken(TokenId id) const noexcept {
  return is_special_token_.contains(id);
}

}  // namespace gufo::tokenization

#include "staging/models/glm5_next/tokenizer.hpp"

#include <unicode/uchar.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::glm5_next {

namespace {

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

/// ChatGLM4 class `[^\s\p{L}\p{N}]`: not whitespace, not a letter, not a
/// number. Unlike llama.cpp's generic `[[:punct:]]`-style classes this one
/// also covers control, format and private-use codepoints.
bool IsPieceOther(UChar32 value) noexcept {
  return !IsUnicodeWhitespace(value) && !IsUnicodeLetter(value) &&
         !IsUnicodeNumber(value);
}

/// `\p{N}{1,3}`: the greedy match starts at `offset` but takes at most three
/// number codepoints.
std::size_t GlmNumberRunEnd(std::string_view text, std::size_t offset) {
  std::size_t end = offset;
  for (int taken = 0; taken < 3; ++taken) {
    const Utf8CodePoint next = DecodeUtf8(text, end);
    if (!IsUnicodeNumber(next.value)) {
      break;
    }
    end += next.length;
  }
  return end;
}

/// `[\r\n]*` greedily appended to a ` ?[^...]+` body.
std::size_t GlmTrailingNewlinesEnd(std::string_view text,
                                   std::size_t offset) {
  std::size_t end = offset;
  while (end < text.size()) {
    const Utf8CodePoint next = DecodeUtf8(text, end);
    if (!IsNewline(next.value)) {
      break;
    }
    end += next.length;
  }
  return end;
}

/// Body run `[^\s\p{L}\p{N}]+` (the class itself includes CR/LF, so trailing
/// newlines fall out of the same scan; GlmTrailingNewlinesEnd stays exact).
std::size_t GlmOtherRunEnd(std::string_view text, std::size_t offset) {
  std::size_t end = offset;
  while (end < text.size()) {
    const Utf8CodePoint next = DecodeUtf8(text, end);
    if (!IsPieceOther(next.value)) {
      break;
    }
    end += next.length;
  }
  return end;
}

/// First alternative `'(?:[sStT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])`: a
/// literal ASCII apostrophe plus an exact case-insensitive clitic. Returns
/// `offset` when the text at `offset` is not a clitic.
std::size_t GlmCliticEnd(std::string_view text,
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
      const UChar32 lower = (next.value >= 'A' && next.value <= 'Z')
                                ? next.value + 32
                                : next.value;
      if (lower != letter) {
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

/// `\p{L}+` letter run; a run continues through every Unicode letter
/// (Han/Hiragana/Katakana included) and stops at combining marks, which the
/// `[^...]+` punctuation class owns.
std::size_t GlmLetterRunEnd(std::string_view text, std::size_t offset) {
  std::size_t end = offset;
  while (end < text.size()) {
    const Utf8CodePoint next = DecodeUtf8(text, end);
    if (!IsUnicodeLetter(next.value)) {
      break;
    }
    end += next.length;
  }
  return end;
}

/// One pre-tokenizer piece boundary for the ChatGLM4 expression set,
/// evaluated leftmost-first exactly as llama.cpp's regex alternation
///     (?:'[sS]|...) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,3} |
///      ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
/// matches: clitic before the letter branch, letters (with one optional
/// non-CR/LF, non-letter, non-number prefix) before digit runs, the class
/// body before the three whitespace alternatives, and finally one
/// codepoint.
std::size_t GlmPieceEnd(std::string_view text, std::size_t offset) {
  const Utf8CodePoint first = DecodeUtf8(text, offset);

  // Alternative 1: clitics consume the position before anything else.
  if (const std::size_t clitic = GlmCliticEnd(text, offset);
      clitic != offset) {
    return clitic;
  }

  // Alternative 3 can only start on a number; alternative 2 (letters) cannot
  // consume one, so a digit here always yields a <=3-codepoint number piece.
  if (IsUnicodeNumber(first.value)) {
    return GlmNumberRunEnd(text, offset);
  }

  // Alternative 2: optional single prefix codepoint (not CR/LF, letter or
  // number) then one or more letters.
  {
    std::size_t run_start = offset;
    if (!IsNewline(first.value) && !IsUnicodeLetter(first.value) &&
        !IsUnicodeNumber(first.value)) {
      const std::size_t after_prefix = offset + first.length;
      if (after_prefix < text.size() &&
          IsUnicodeLetter(DecodeUtf8(text, after_prefix).value)) {
        run_start = after_prefix;
      }
    }
    if (IsUnicodeLetter(first.value) || run_start != offset) {
      const std::size_t end = GlmLetterRunEnd(text, run_start);
      if (end != run_start) {
        return end;
      }
    }
  }

  // Alternative 4: one optional leading space, then `[^ \s\p{L}\p{N}]+`
  // codepoints, then trailing CR/LF codepoints.
  if (!IsUnicodeWhitespace(first.value)) {
    const std::size_t end = GlmOtherRunEnd(text, offset);
    if (end != offset) {
      return GlmTrailingNewlinesEnd(text, end);
    }
  } else if (first.value == ' ') {
    // The space attaches only when the next codepoint is in the body class.
    const std::size_t after_space = offset + first.length;
    if (after_space < text.size() &&
        IsPieceOther(DecodeUtf8(text, after_space).value)) {
      const std::size_t end = GlmOtherRunEnd(text, after_space);
      return GlmTrailingNewlinesEnd(text, end);
    }
  }

  if (IsUnicodeWhitespace(first.value)) {
    // \s*[\r\n]+ owns a whitespace run up to its final newline; \s+(?!\S)
    // then keeps all-whitespace tails whole but backtracks the last
    // whitespace codepoint before non-whitespace so the next piece can own
    // its prefix; \s+ takes what is left.
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
    return end < text.size() && last_start > offset ? last_start : end;
  }

  return offset + first.length;
}

}  // namespace

std::unique_ptr<GlmTokenizer> GlmTokenizer::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto model = reader.GetMetadataString("tokenizer.ggml.model");
  if (model != std::optional<std::string_view>{"gpt2"}) {
    if (error_msg != nullptr) {
      *error_msg = "GLM-5 requires tokenizer.ggml.model 'gpt2'";
    }
    return nullptr;
  }
  const auto pre = reader.GetMetadataString("tokenizer.ggml.pre");
  if (pre != std::optional<std::string_view>{"glm4"}) {
    if (error_msg != nullptr) {
      *error_msg = "GLM-5 requires tokenizer.ggml.pre 'glm4'";
    }
    return nullptr;
  }

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

  // Integer array metadata arrives as int64 or uint64 vectors depending on
  // the GGUF element type.
  const auto collect_ids =
      [](const core::GgufMetadataValue* entry) -> std::vector<TokenId> {
    std::vector<TokenId> ids;
    if (entry == nullptr) {
      return ids;
    }
    if (const auto* values =
            std::get_if<std::vector<std::int64_t>>(&entry->value)) {
      ids.reserve(values->size());
      for (const std::int64_t value : *values) {
        if (value >= 0) {
          ids.push_back(static_cast<TokenId>(value));
        }
      }
    } else if (const auto* values =
                   std::get_if<std::vector<std::uint64_t>>(&entry->value)) {
      ids.reserve(values->size());
      for (const std::uint64_t value : *values) {
        if (value <= std::numeric_limits<TokenId>::max()) {
          ids.push_back(static_cast<TokenId>(value));
        }
      }
    }
    return ids;
  };

  // CONTROL- and USER_DEFINED-typed entries are the parse-special set,
  // matching llama.cpp's special classification; UNUSED pads stay text.
  std::vector<TokenId> special_ids;
  const auto* token_types = reader.FindMetadata("tokenizer.ggml.token_type");
  const auto type_ids = collect_ids(token_types);
  if (type_ids.size() == tokens.size()) {
    for (std::size_t i = 0; i < type_ids.size(); ++i) {
      if (type_ids[i] == static_cast<TokenId>(TokenType::kControl) ||
          type_ids[i] == static_cast<TokenId>(TokenType::kUserDefined)) {
        special_ids.push_back(static_cast<TokenId>(i));
      }
    }
  }

  TokenId bos = kInvalidTokenId;
  TokenId eos = kInvalidTokenId;
  TokenId pad = kInvalidTokenId;
  if (auto id = reader.GetMetadataUint32("tokenizer.ggml.bos_token_id")) {
    bos = *id;
  }
  if (auto id = reader.GetMetadataUint32("tokenizer.ggml.eos_token_id")) {
    eos = *id;
  }
  if (auto id = reader.GetMetadataUint32("tokenizer.ggml.padding_token_id")) {
    pad = *id;
  }

  std::vector<TokenId> eog_ids =
      collect_ids(reader.FindMetadata("tokenizer.ggml.eog_token_ids"));
  // GLM chat frames end at the next CONTROL role token: the GGUF declares
  // `tokenizer.ggml.eot_token_id` (`<|user|>` here) and
  // `tokenizer.ggml.eom_token_id` (`<|observation|>`) but no explicit eog
  // list (llama.cpp reaches for these ids in its GLM eog path). Without
  // them generation only stops on eos, so turns spill raw reasoning into
  // the next user frame instead of ending.
  if (eog_ids.empty()) {
    for (const char* key : {"tokenizer.ggml.eot_token_id",
                            "tokenizer.ggml.eom_token_id"}) {
      if (auto id = reader.GetMetadataUint32(key)) {
        eog_ids.push_back(*id);
      }
    }
  }

  return CreateFromVocabulary(tokens, merges, special_ids, bos, eos, pad,
                              eog_ids, error_msg);
}

std::unique_ptr<GlmTokenizer> GlmTokenizer::CreateFromVocabulary(
    std::span<const std::string> tokens, std::span<const std::string> merges,
    std::span<const TokenId> special_ids, TokenId bos_token, TokenId eos_token,
    TokenId pad_token, std::span<const TokenId> eog_ids,
    std::string* error_msg) {
  if (tokens.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "Vocabulary cannot be empty";
    }
    return nullptr;
  }

  auto tokenizer = std::unique_ptr<GlmTokenizer>(new GlmTokenizer());
  tokenizer->id_to_token_.reserve(tokens.size());
  tokenizer->token_to_id_.reserve(tokens.size());

  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokenizer->id_to_token_.push_back(tokens[i]);
    tokenizer->token_to_id_[tokens[i]] = static_cast<TokenId>(i);
  }

  // Parse BPE merges.
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
      // The merged token is fixed by the pair, so resolve it once here
      // instead of rebuilding the string on every merge.
      const auto merged = tokenizer->token_to_id_.find(part1 + part2);
      tokenizer->merge_ranks_[{it1->second, it2->second}] = {
          .rank = rank,
          .token = merged != tokenizer->token_to_id_.end() ? merged->second
                                                           : kInvalidTokenId,
      };
    }
  }

  // Register CONTROL/USER_DEFINED entries as the parse-special set.
  tokenizer->special_token_list_.reserve(special_ids.size());
  for (const TokenId id : special_ids) {
    if (id >= tokenizer->id_to_token_.size()) {
      continue;
    }
    const auto& spelling = tokenizer->id_to_token_[id];
    tokenizer->special_token_list_.emplace_back(spelling, id);
    tokenizer->special_token_to_id_[spelling] = id;
    tokenizer->is_special_token_.insert(id);
  }

  tokenizer->bos_token_id_ = bos_token;
  tokenizer->eos_token_id_ = eos_token;
  tokenizer->pad_token_id_ = pad_token;
  if (eos_token != kInvalidTokenId) {
    tokenizer->stop_token_ids_.insert(eos_token);
  }
  for (const TokenId id : eog_ids) {
    if (id < tokenizer->id_to_token_.size()) {
      tokenizer->stop_token_ids_.insert(id);
    }
  }
  for (const TokenId id : tokenizer->stop_token_ids_) {
    if (id < tokenizer->turn_end_token_id_) {
      tokenizer->turn_end_token_id_ = id;
    }
  }

  tokenizer->InitializeByteTokens();
  return tokenizer;
}

void GlmTokenizer::InitializeByteTokens() {
  id_to_decoded_token_.resize(id_to_token_.size());
  for (std::size_t i = 0; i < id_to_token_.size(); ++i) {
    if (is_special_token_.contains(static_cast<TokenId>(i))) {
      id_to_decoded_token_[i] = id_to_token_[i];
    } else {
      id_to_decoded_token_[i] = UnescapeGpt2Bytes(id_to_token_[i]);
    }
  }

  for (std::size_t b = 0; b < 256; ++b) {
    const auto byte_val = static_cast<std::uint8_t>(b);
    const std::string gpt2_utf8 = ByteToGpt2Utf8(byte_val);
    auto it = token_to_id_.find(gpt2_utf8);
    if (it != token_to_id_.end()) {
      byte_tokens_[b] = it->second;
      continue;
    }
    const std::string direct_char(1, static_cast<char>(byte_val));
    auto direct_it = token_to_id_.find(direct_char);
    byte_tokens_[b] =
        direct_it != token_to_id_.end() ? direct_it->second : kInvalidTokenId;
  }
}

std::vector<TokenId> GlmTokenizer::BpeMergeChunk(
    std::string_view chunk) const {
  if (chunk.empty()) {
    return {};
  }

  // Initial tokenization at byte level.
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

  // Merge the lowest-ranked adjacent pair repeatedly, leftmost first on a
  // tie. Rescanning the whole word per merge is quadratic in the piece, and
  // pre-tokenizer pieces are unbounded (a run of letters or of spaces is one
  // piece), so hold the symbols in a linked list and the candidate pairs in a
  // heap. Stale entries are discarded when popped.
  constexpr std::size_t kNoSymbol = std::numeric_limits<std::size_t>::max();
  const std::size_t count = word_tokens.size();
  std::vector<std::size_t> previous(count);
  std::vector<std::size_t> following(count);
  for (std::size_t i = 0; i < count; ++i) {
    previous[i] = i == 0 ? kNoSymbol : i - 1;
    following[i] = i + 1 < count ? i + 1 : kNoSymbol;
  }

  struct Candidate {
    std::uint32_t rank;
    std::size_t index;  ///< Left symbol; also the left-to-right tie-break.
    TokenId left;
    TokenId right;
  };
  const auto later = [](const Candidate& a, const Candidate& b) {
    return a.rank != b.rank ? a.rank > b.rank : a.index > b.index;
  };
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(later)>
      candidates(later);

  const auto offer = [&](std::size_t index, std::size_t right) {
    const auto merge =
        merge_ranks_.find({word_tokens[index], word_tokens[right]});
    if (merge != merge_ranks_.end()) {
      candidates.push({
          .rank = merge->second.rank,
          .index = index,
          .left = word_tokens[index],
          .right = word_tokens[right],
      });
    }
  };

  for (std::size_t i = 0; i + 1 < count; ++i) {
    offer(i, i + 1);
  }

  while (!candidates.empty()) {
    const Candidate best = candidates.top();
    candidates.pop();
    const std::size_t right = following[best.index];
    if (word_tokens[best.index] != best.left || right == kNoSymbol ||
        word_tokens[right] != best.right) {
      continue;  // A neighbouring merge already replaced this pair.
    }

    const auto merge = merge_ranks_.find({best.left, best.right});
    if (merge == merge_ranks_.end() || merge->second.token == kInvalidTokenId) {
      break;  // Ranked pair with no vocabulary entry: stop merging.
    }

    word_tokens[best.index] = merge->second.token;
    const std::size_t after = following[right];
    following[best.index] = after;
    if (after != kNoSymbol) {
      previous[after] = best.index;
    }
    word_tokens[right] = kInvalidTokenId;

    if (previous[best.index] != kNoSymbol) {
      offer(previous[best.index], best.index);
    }
    if (after != kNoSymbol) {
      offer(best.index, after);
    }
  }

  std::vector<TokenId> merged;
  merged.reserve(count);
  for (std::size_t i = 0; i != kNoSymbol; i = following[i]) {
    merged.push_back(word_tokens[i]);
  }
  return merged;
}

std::vector<TokenId> GlmTokenizer::BpeEncodeText(std::string_view text) const {
  std::vector<TokenId> tokens;
  // Repeated words have identical BPE merges. Keep a small request-local
  // memo: no cross-request state or lock, and no retained copies of
  // long/unbounded pieces.
  std::unordered_map<std::string_view, std::vector<TokenId>> pieces;
  if (text.size() >= 1024) {
    pieces.reserve(128);
  }
  std::size_t offset = 0;
  while (offset < text.size()) {
    const std::size_t end = GlmPieceEnd(text, offset);
    const auto piece = text.substr(offset, end - offset);
    if (text.size() >= 1024 && piece.size() > 1 && piece.size() <= 64) {
      if (const auto found = pieces.find(piece); found != pieces.end()) {
        tokens.insert(tokens.end(), found->second.begin(), found->second.end());
        offset = end;
        continue;
      }
    }
    auto piece_tokens = BpeMergeChunk(piece);
    tokens.insert(tokens.end(), piece_tokens.begin(), piece_tokens.end());
    if (text.size() >= 1024 && piece.size() > 1 && piece.size() <= 64 &&
        pieces.size() < 128) {
      pieces.emplace(piece, std::move(piece_tokens));
    }
    offset = end;
  }
  return tokens;
}

std::vector<TokenId> GlmTokenizer::Encode(
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
    // Scan text for special token delimiters. Each special keeps its next
    // occurrence instead and only re-searches once the cursor has passed
    // it, which walks the text once overall.
    std::vector<std::size_t> occurrence(special_token_list_.size());
    for (std::size_t i = 0; i < special_token_list_.size(); ++i) {
      occurrence[i] = text.find(special_token_list_[i].first);
    }

    std::size_t pos = 0;
    while (pos < text.size()) {
      std::size_t next_special_pos = std::string_view::npos;
      std::string_view matched_special;
      TokenId matched_id = kInvalidTokenId;

      for (std::size_t i = 0; i < special_token_list_.size(); ++i) {
        const auto& [special_str, id] = special_token_list_[i];
        if (occurrence[i] != std::string_view::npos && occurrence[i] < pos) {
          occurrence[i] = text.find(special_str, pos);
        }
        const auto found = occurrence[i];
        if (found == std::string_view::npos) {
          continue;
        }
        // Prefer the earliest match, and the longest where two specials
        // start together.
        if (next_special_pos == std::string_view::npos ||
            found < next_special_pos ||
            (found == next_special_pos &&
             special_str.size() > matched_special.size())) {
          next_special_pos = found;
          matched_special = special_str;
          matched_id = id;
        }
      }

      if (next_special_pos == std::string_view::npos) {
        // No more special tokens; encode remainder.
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

std::string GlmTokenizer::Decode(std::span<const TokenId> tokens) const {
  std::string result;
  for (const TokenId id : tokens) {
    result.append(DecodeTokenCopy(id));
  }
  return result;
}

std::string_view GlmTokenizer::DecodeToken(TokenId token_id) const noexcept {
  if (token_id < id_to_decoded_token_.size()) {
    return id_to_decoded_token_[token_id];
  }
  if (token_id < id_to_token_.size()) {
    return id_to_token_[token_id];
  }
  return "";
}

std::string GlmTokenizer::DecodeTokenCopy(TokenId token_id) const {
  if (token_id >= id_to_token_.size()) {
    return {};
  }
  return id_to_decoded_token_[token_id];
}

std::optional<TokenId> GlmTokenizer::FindSpecialToken(
    std::string_view token_str) const noexcept {
  auto it = special_token_to_id_.find(std::string(token_str));
  if (it != special_token_to_id_.end()) {
    return it->second;
  }
  return std::nullopt;
}

bool GlmTokenizer::IsSpecialToken(TokenId id) const noexcept {
  return is_special_token_.contains(id);
}

}  // namespace gufo::models::glm5_next

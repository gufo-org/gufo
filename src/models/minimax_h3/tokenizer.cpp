// Tokenizer semantics translated from antirez/h3.c h3_tokenizer.m at
// 8974cc055ea9c02fcd14cc27dfda3e1027c05153 (MIT). Foundation/Objective-C
// storage was replaced with model-private C++ and ICU on Linux.
#include "src/models/minimax_h3/tokenizer.hpp"

#include <unicode/normalizer2.h>
#include <unicode/stringpiece.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <utility>

#include "src/models/minimax_h3/json.hpp"
#include "src/models/qwen/control_tokens.hpp"

namespace gufo::minimax_h3 {
namespace {

constexpr std::size_t kMaximumTokenizerBytes = 32U << 20U;
constexpr char kPairSeparator = '\0';

struct Codepoint {
  std::uint32_t value{0};
  std::size_t begin{0};
  std::size_t end{0};
};

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

const json::Value& RequireField(const json::Value& object,
                                std::string_view name) {
  const json::Value* value = object.Find(name);
  if (value == nullptr) {
    throw json::Error("missing tokenizer field " + std::string(name));
  }
  return *value;
}

std::string AppendCodepoint(std::uint32_t value) {
  std::string output;
  if (value <= 0x7FU) {
    output.push_back(static_cast<char>(value));
  } else if (value <= 0x7FFU) {
    output.push_back(static_cast<char>(0xC0U | (value >> 6U)));
    output.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
  } else if (value <= 0xFFFFU) {
    output.push_back(static_cast<char>(0xE0U | (value >> 12U)));
    output.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
  } else {
    output.push_back(static_cast<char>(0xF0U | (value >> 18U)));
    output.push_back(static_cast<char>(0x80U | ((value >> 12U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
  }
  return output;
}

bool DecodeUtf8(std::string_view text, std::vector<Codepoint>* output,
                std::string* error) {
  output->clear();
  for (std::size_t index = 0; index < text.size();) {
    const std::size_t begin = index;
    const auto first = static_cast<unsigned char>(text[index++]);
    std::uint32_t value = 0;
    std::size_t continuation = 0;
    if (first <= 0x7FU) {
      value = first;
    } else if (first >= 0xC2U && first <= 0xDFU) {
      value = first & 0x1FU;
      continuation = 1;
    } else if (first >= 0xE0U && first <= 0xEFU) {
      value = first & 0x0FU;
      continuation = 2;
    } else if (first >= 0xF0U && first <= 0xF4U) {
      value = first & 0x07U;
      continuation = 3;
    } else {
      SetError(error, "prompt is not valid UTF-8");
      return false;
    }
    if (continuation > text.size() - index) {
      SetError(error, "prompt is not valid UTF-8");
      return false;
    }
    for (std::size_t offset = 0; offset < continuation; ++offset) {
      const auto next = static_cast<unsigned char>(text[index++]);
      if ((next & 0xC0U) != 0x80U) {
        SetError(error, "prompt is not valid UTF-8");
        return false;
      }
      value = (value << 6U) | (next & 0x3FU);
    }
    if ((continuation == 2 &&
         (value < 0x800U || (value >= 0xD800U && value <= 0xDFFFU))) ||
        (continuation == 3 && (value < 0x10000U || value > 0x10FFFFU))) {
      SetError(error, "prompt is not valid UTF-8");
      return false;
    }
    output->push_back({value, begin, index});
  }
  return true;
}

bool IsLetter(std::uint32_t value) {
  const std::int8_t category = u_charType(static_cast<UChar32>(value));
  return category == U_UPPERCASE_LETTER || category == U_LOWERCASE_LETTER ||
         category == U_TITLECASE_LETTER || category == U_MODIFIER_LETTER ||
         category == U_OTHER_LETTER;
}

bool IsNumber(std::uint32_t value) {
  const std::int8_t category = u_charType(static_cast<UChar32>(value));
  return category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER ||
         category == U_OTHER_NUMBER;
}

bool IsSpace(std::uint32_t value) {
  return u_isUWhiteSpace(static_cast<UChar32>(value)) != 0 ||
         (value >= 0x1CU && value <= 0x1FU);
}

std::size_t ContractionLength(const std::vector<Codepoint>& points,
                              std::size_t index) {
  static constexpr std::array<std::string_view, 7> kValues = {
      "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
  if (points[index].value != '\'') {
    return 0;
  }
  for (const std::string_view candidate : kValues) {
    if (candidate.size() > points.size() - index) {
      continue;
    }
    bool matches = true;
    for (std::size_t offset = 1; offset < candidate.size(); ++offset) {
      std::uint32_t got = points[index + offset].value;
      if (got >= 'A' && got <= 'Z') {
        got += 'a' - 'A';
      }
      if (got != static_cast<unsigned char>(candidate[offset])) {
        matches = false;
        break;
      }
    }
    if (matches) {
      return candidate.size();
    }
  }
  return 0;
}

std::string Slice(std::string_view text, const std::vector<Codepoint>& points,
                  std::size_t begin, std::size_t end) {
  return std::string(text.substr(points[begin].begin,
                                 points[end - 1].end - points[begin].begin));
}

bool NormalizeNfc(std::string_view input, std::string* output,
                  std::string* error) {
  UErrorCode status = U_ZERO_ERROR;
  const icu::Normalizer2* normalizer = icu::Normalizer2::getNFCInstance(status);
  if (U_FAILURE(status) || normalizer == nullptr) {
    SetError(error, "ICU NFC normalizer is unavailable");
    return false;
  }
  const icu::UnicodeString source = icu::UnicodeString::fromUTF8(
      icu::StringPiece(input.data(), static_cast<std::int32_t>(input.size())));
  icu::UnicodeString normalized;
  normalizer->normalize(source, normalized, status);
  if (U_FAILURE(status)) {
    SetError(error, "unable to normalize prompt as NFC");
    return false;
  }
  output->clear();
  normalized.toUTF8String(*output);
  return true;
}

bool Pretokenize(std::string_view input, std::vector<std::string>* pieces,
                 std::string* error) {
  std::string text;
  if (!NormalizeNfc(input, &text, error)) {
    return false;
  }
  std::vector<Codepoint> points;
  if (!DecodeUtf8(text, &points, error)) {
    return false;
  }
  pieces->clear();
  std::size_t index = 0;
  while (index < points.size()) {
    const std::size_t contraction = ContractionLength(points, index);
    if (contraction != 0) {
      pieces->push_back(Slice(text, points, index, index + contraction));
      index += contraction;
      continue;
    }
    const std::uint32_t value = points[index].value;
    std::optional<std::size_t> letter_start;
    if (IsLetter(value)) {
      letter_start = index;
    } else if (value != '\r' && value != '\n' && !IsNumber(value) &&
               index + 1 < points.size() && IsLetter(points[index + 1].value)) {
      letter_start = index + 1;
    }
    if (letter_start.has_value()) {
      std::size_t stop = *letter_start;
      while (stop < points.size() && IsLetter(points[stop].value)) {
        ++stop;
      }
      pieces->push_back(Slice(text, points, index, stop));
      index = stop;
      continue;
    }
    if (IsNumber(value)) {
      pieces->push_back(Slice(text, points, index, index + 1));
      ++index;
      continue;
    }
    const std::size_t punctuation_start =
        index + (value == ' ' && index + 1 < points.size() &&
                         !IsSpace(points[index + 1].value) &&
                         !IsLetter(points[index + 1].value) &&
                         !IsNumber(points[index + 1].value)
                     ? 1
                     : 0);
    std::size_t stop = punctuation_start;
    while (stop < points.size() && !IsSpace(points[stop].value) &&
           !IsLetter(points[stop].value) && !IsNumber(points[stop].value)) {
      ++stop;
    }
    if (stop > punctuation_start) {
      while (stop < points.size() &&
             (points[stop].value == '\r' || points[stop].value == '\n')) {
        ++stop;
      }
      pieces->push_back(Slice(text, points, index, stop));
      index = stop;
      continue;
    }
    if (IsSpace(value)) {
      std::size_t whitespace_end = index + 1;
      while (whitespace_end < points.size() &&
             IsSpace(points[whitespace_end].value)) {
        ++whitespace_end;
      }
      std::optional<std::size_t> newline_end;
      for (std::size_t cursor = index; cursor < whitespace_end; ++cursor) {
        if (points[cursor].value == '\r' || points[cursor].value == '\n') {
          newline_end = cursor + 1;
        }
      }
      std::size_t piece_end = 0;
      if (newline_end.has_value()) {
        piece_end = *newline_end;
      } else if (whitespace_end == points.size()) {
        piece_end = whitespace_end;
      } else if (whitespace_end - index > 1) {
        piece_end = whitespace_end - 1;
      } else {
        piece_end = index + 1;
      }
      pieces->push_back(Slice(text, points, index, piece_end));
      index = piece_end;
      continue;
    }
    SetError(error, "unable to pre-tokenize input");
    return false;
  }
  return true;
}

std::string PairKey(std::string_view left, std::string_view right) {
  std::string key;
  key.reserve(left.size() + right.size() + 1);
  key.append(left);
  key.push_back(kPairSeparator);
  key.append(right);
  return key;
}

std::vector<std::string> SplitCodepoints(std::string_view text) {
  std::vector<Codepoint> points;
  std::string unused;
  if (!DecodeUtf8(text, &points, &unused)) {
    throw json::Error("tokenizer contains invalid UTF-8");
  }
  std::vector<std::string> result;
  result.reserve(points.size());
  for (const Codepoint& point : points) {
    result.emplace_back(text.substr(point.begin, point.end - point.begin));
  }
  return result;
}

bool OptionalFalse(const json::Value& object, std::string_view name) {
  const json::Value* value = object.Find(name);
  return value == nullptr || (value->IsBool() && !value->AsBool());
}

}  // namespace

void Tokenizer::ValidatePinnedContract(const Tokenizer& tokenizer) {
  static constexpr std::array<std::pair<std::string_view, std::uint32_t>, 14>
      kSpecials = {{
          {tokenization::kEndOfText, 151643},
          {tokenization::kImStart, 151644},
          {tokenization::kImEnd, 151645},
          {tokenization::kObjectRefStart, 151646},
          {tokenization::kObjectRefEnd, 151647},
          {tokenization::kBoxStart, 151648},
          {tokenization::kBoxEnd, 151649},
          {tokenization::kQuadStart, 151650},
          {tokenization::kQuadEnd, 151651},
          {tokenization::kVisionStart, 151652},
          {tokenization::kVisionEnd, 151653},
          {tokenization::kVisionPad, 151654},
          {tokenization::kImagePad, 151655},
          {tokenization::kVideoPad, 151656},
      }};
  if (tokenizer.vocab_.size() != kBaseVocabularySize ||
      tokenizer.vocabulary_size() != kTokenizerVocabularySize) {
    throw json::Error("pinned H3 tokenizer vocabulary size mismatch");
  }
  for (const auto& [content, expected] : kSpecials) {
    const auto found = tokenizer.added_tokens_.find(std::string(content));
    if (found == tokenizer.added_tokens_.end() || found->second != expected) {
      throw json::Error("pinned H3 special token mismatch: " +
                        std::string(content));
    }
  }
}

bool Tokenizer::Load(const std::filesystem::path& path, Tokenizer* output,
                     std::string* error) {
  return Load(path, LoadOptions{}, output, error);
}

bool Tokenizer::Load(const std::filesystem::path& path,
                     const LoadOptions& options, Tokenizer* output,
                     std::string* error) {
  if (output == nullptr) {
    SetError(error, "tokenizer output is required");
    return false;
  }
  try {
    const json::Value root = json::ParseFile(path, kMaximumTokenizerBytes);
    const json::Value& model = RequireField(root, "model");
    const json::Value& normalizer = RequireField(root, "normalizer");
    if (RequireField(model, "type").AsString() != "BPE" ||
        !RequireField(model, "unk_token").IsNull() ||
        RequireField(normalizer, "type").AsString() != "NFC") {
      throw json::Error("unexpected tokenizer specification");
    }

    Tokenizer tokenizer;
    std::uint32_t maximum_id = 0;
    for (const auto& [symbol, identifier] :
         RequireField(model, "vocab").AsObject()) {
      const std::uint64_t parsed = identifier.AsUint64();
      if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw json::Error("tokenizer vocabulary ID is out of range");
      }
      const auto id = static_cast<std::uint32_t>(parsed);
      if (!tokenizer.vocab_.emplace(symbol, id).second) {
        throw json::Error("duplicate tokenizer vocabulary symbol");
      }
      maximum_id = std::max(maximum_id, id);
    }

    std::uint32_t rank = 0;
    for (const json::Value& entry : RequireField(model, "merges").AsArray()) {
      std::string left;
      std::string right;
      if (entry.IsString()) {
        const std::string& merged = entry.AsString();
        const std::size_t separator = merged.find(' ');
        if (separator == std::string::npos) {
          throw json::Error("invalid tokenizer merge");
        }
        left = merged.substr(0, separator);
        right = merged.substr(separator + 1);
      } else {
        const auto& pair = entry.AsArray();
        if (pair.size() != 2) {
          throw json::Error("invalid tokenizer merge");
        }
        left = pair[0].AsString();
        right = pair[1].AsString();
      }
      if (left.empty() || right.empty() ||
          !tokenizer.merge_ranks_.emplace(PairKey(left, right), rank).second) {
        throw json::Error("invalid or duplicate tokenizer merge");
      }
      ++rank;
    }

    const json::Value* added_value = root.Find("added_tokens");
    if (added_value != nullptr) {
      for (const json::Value& entry : added_value->AsArray()) {
        if (!OptionalFalse(entry, "single_word") ||
            !OptionalFalse(entry, "lstrip") ||
            !OptionalFalse(entry, "rstrip") ||
            !OptionalFalse(entry, "normalized")) {
          throw json::Error("unsupported added-token policy");
        }
        const std::string content = RequireField(entry, "content").AsString();
        const std::uint64_t parsed = RequireField(entry, "id").AsUint64();
        if (content.empty() ||
            parsed > std::numeric_limits<std::uint32_t>::max()) {
          throw json::Error("invalid added token");
        }
        const auto id = static_cast<std::uint32_t>(parsed);
        if (!tokenizer.added_tokens_.emplace(content, id).second) {
          throw json::Error("duplicate added-token content");
        }
        tokenizer.added_alternatives_.push_back({content, id});
        maximum_id = std::max(maximum_id, id);
      }
    }

    tokenizer.inverse_vocab_.resize(static_cast<std::size_t>(maximum_id) + 1);
    tokenizer.inverse_vocab_present_.resize(
        static_cast<std::size_t>(maximum_id) + 1, false);
    tokenizer.inverse_added_.resize(static_cast<std::size_t>(maximum_id) + 1);
    tokenizer.inverse_added_present_.resize(
        static_cast<std::size_t>(maximum_id) + 1, false);
    for (const auto& [symbol, id] : tokenizer.vocab_) {
      if (tokenizer.inverse_vocab_present_[id]) {
        throw json::Error("duplicate tokenizer vocabulary ID");
      }
      tokenizer.inverse_vocab_[id] = symbol;
      tokenizer.inverse_vocab_present_[id] = true;
    }
    for (const AddedToken& token : tokenizer.added_alternatives_) {
      if (tokenizer.inverse_added_present_[token.id]) {
        throw json::Error("duplicate added-token ID");
      }
      tokenizer.inverse_added_[token.id] = token.content;
      tokenizer.inverse_added_present_[token.id] = true;
    }
    std::ranges::sort(tokenizer.added_alternatives_,
                      [](const AddedToken& left, const AddedToken& right) {
                        if (left.content.size() != right.content.size()) {
                          return left.content.size() > right.content.size();
                        }
                        return left.content < right.content;
                      });

    tokenizer.byte_decoder_.fill(-1);
    std::uint32_t extra = 0;
    for (std::uint32_t byte = 0; byte < 256; ++byte) {
      const bool visible = (byte >= '!' && byte <= '~') ||
                           (byte >= 0xA1U && byte <= 0xACU) ||
                           (byte >= 0xAEU && byte <= 0xFFU);
      const std::uint32_t codepoint = visible ? byte : 256U + extra++;
      tokenizer.byte_encoder_[byte] = AppendCodepoint(codepoint);
      tokenizer.byte_decoder_[codepoint] = static_cast<std::int16_t>(byte);
    }
    if (options.require_pinned_h3_contract) {
      ValidatePinnedContract(tokenizer);
    }
    *output = std::move(tokenizer);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, "MiniMax H3 tokenizer load failed: " +
                        std::string(exception.what()));
    return false;
  }
}

bool Tokenizer::ApplyBpe(std::string_view piece,
                         std::vector<std::uint32_t>* ids,
                         std::string* error) const {
  std::string encoded;
  for (const unsigned char byte : piece) {
    encoded.append(byte_encoder_[byte]);
  }
  std::vector<std::string> symbols;
  try {
    symbols = SplitCodepoints(encoded);
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
  while (symbols.size() > 1) {
    std::optional<std::uint32_t> best_rank;
    std::size_t best = 0;
    for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
      const auto found =
          merge_ranks_.find(PairKey(symbols[index], symbols[index + 1]));
      if (found != merge_ranks_.end() &&
          (!best_rank.has_value() || found->second < *best_rank)) {
        best_rank = found->second;
        best = index;
      }
    }
    if (!best_rank.has_value()) {
      break;
    }
    const std::string left = symbols[best];
    const std::string right = symbols[best + 1];
    std::vector<std::string> merged;
    merged.reserve(symbols.size() - 1);
    for (std::size_t index = 0; index < symbols.size();) {
      if (index + 1 < symbols.size() && symbols[index] == left &&
          symbols[index + 1] == right) {
        merged.push_back(left + right);
        index += 2;
      } else {
        merged.push_back(std::move(symbols[index++]));
      }
    }
    symbols = std::move(merged);
  }
  for (const std::string& symbol : symbols) {
    const auto found = vocab_.find(symbol);
    if (found == vocab_.end()) {
      SetError(error, "BPE symbol is absent from vocabulary");
      return false;
    }
    ids->push_back(found->second);
  }
  return true;
}

bool Tokenizer::EncodePlain(std::string_view text,
                            std::vector<std::uint32_t>* ids,
                            std::string* error) const {
  std::vector<std::string> pieces;
  if (!Pretokenize(text, &pieces, error)) {
    return false;
  }
  for (const std::string& piece : pieces) {
    if (!ApplyBpe(piece, ids, error)) {
      return false;
    }
  }
  return true;
}

bool Tokenizer::Encode(std::string_view text, bool pad_empty,
                       std::vector<std::uint32_t>* ids,
                       std::string* error) const {
  if (ids == nullptr) {
    SetError(error, "token output is required");
    return false;
  }
  std::vector<Codepoint> validated;
  if (!DecodeUtf8(text, &validated, error)) {
    return false;
  }
  ids->clear();
  std::size_t start = 0;
  while (start < text.size()) {
    const AddedToken* selected = nullptr;
    std::size_t selected_offset = std::string_view::npos;
    for (const AddedToken& candidate : added_alternatives_) {
      const std::size_t offset = text.find(candidate.content, start);
      if (offset == std::string_view::npos) {
        continue;
      }
      if (selected == nullptr || offset < selected_offset ||
          (offset == selected_offset &&
           candidate.content.size() > selected->content.size())) {
        selected = &candidate;
        selected_offset = offset;
      }
    }
    if (selected == nullptr) {
      break;
    }
    if (selected_offset > start &&
        !EncodePlain(text.substr(start, selected_offset - start), ids, error)) {
      ids->clear();
      return false;
    }
    ids->push_back(selected->id);
    start = selected_offset + selected->content.size();
  }
  if (start < text.size() && !EncodePlain(text.substr(start), ids, error)) {
    ids->clear();
    return false;
  }
  if (ids->empty() && pad_empty) {
    ids->push_back(kPadTokenId);
  }
  return true;
}

bool Tokenizer::Decode(const std::vector<std::uint32_t>& ids, std::string* text,
                       std::string* error) const {
  if (text == nullptr) {
    SetError(error, "decoded output is required");
    return false;
  }
  text->clear();
  std::string bytes;
  const auto flush = [&]() {
    if (bytes.empty()) {
      return true;
    }
    std::vector<Codepoint> validated;
    if (!DecodeUtf8(bytes, &validated, nullptr)) {
      text->append("\xEF\xBF\xBD");
    } else {
      text->append(bytes);
    }
    bytes.clear();
    return true;
  };
  for (const std::uint32_t id : ids) {
    if (id >= inverse_vocab_.size()) {
      SetError(error, "token ID is out of range");
      return false;
    }
    if (inverse_added_present_[id]) {
      flush();
      text->append(inverse_added_[id]);
      continue;
    }
    if (!inverse_vocab_present_[id]) {
      SetError(error, "unknown token ID");
      return false;
    }
    const std::string& symbol = inverse_vocab_[id];
    std::vector<Codepoint> points;
    if (!DecodeUtf8(symbol, &points, error)) {
      SetError(error, "invalid byte-level token");
      return false;
    }
    for (const Codepoint& point : points) {
      if (point.value >= byte_decoder_.size() ||
          byte_decoder_[point.value] < 0) {
        SetError(error, "invalid byte-level token");
        return false;
      }
      bytes.push_back(static_cast<char>(byte_decoder_[point.value]));
    }
  }
  flush();
  return true;
}

}  // namespace gufo::minimax_h3

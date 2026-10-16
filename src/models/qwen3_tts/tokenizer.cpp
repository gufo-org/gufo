#include "src/models/qwen3_tts/tokenizer.hpp"

#include <unicode/normalizer2.h>
#include <unicode/stringpiece.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/qwen/control_tokens.hpp"

namespace gufo::models::qwen3_tts {
namespace {

constexpr char kPairSeparator = '\0';
constexpr std::size_t kMaximumVocabularyBytes = 16U << 20U;
constexpr std::size_t kMaximumTokenizerConfigBytes = 1U << 20U;

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
      SetError(error, "Qwen3-TTS prompt is not valid UTF-8");
      return false;
    }
    if (continuation > text.size() - index) {
      SetError(error, "Qwen3-TTS prompt is not valid UTF-8");
      return false;
    }
    for (std::size_t offset = 0; offset < continuation; ++offset) {
      const auto next = static_cast<unsigned char>(text[index++]);
      if ((next & 0xC0U) != 0x80U) {
        SetError(error, "Qwen3-TTS prompt is not valid UTF-8");
        return false;
      }
      value = (value << 6U) | (next & 0x3FU);
    }
    if ((continuation == 1 && value < 0x80U) ||
        (continuation == 2 &&
         (value < 0x800U || (value >= 0xD800U && value <= 0xDFFFU))) ||
        (continuation == 3 && (value < 0x10000U || value > 0x10FFFFU))) {
      SetError(error, "Qwen3-TTS prompt is not valid UTF-8");
      return false;
    }
    output->push_back({value, begin, index});
  }
  return true;
}

bool IsLetterOrMark(std::uint32_t value) {
  const std::int8_t category = u_charType(static_cast<UChar32>(value));
  return category == U_UPPERCASE_LETTER || category == U_LOWERCASE_LETTER ||
         category == U_TITLECASE_LETTER || category == U_MODIFIER_LETTER ||
         category == U_OTHER_LETTER || category == U_NON_SPACING_MARK ||
         category == U_COMBINING_SPACING_MARK || category == U_ENCLOSING_MARK;
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
  static constexpr std::array<std::string_view, 7> kContractions = {
      "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
  if (points[index].value != '\'') {
    return 0;
  }
  for (const std::string_view candidate : kContractions) {
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
    SetError(error, "Qwen3-TTS ICU NFC normalizer is unavailable");
    return false;
  }
  const icu::UnicodeString source = icu::UnicodeString::fromUTF8(
      icu::StringPiece(input.data(), static_cast<std::int32_t>(input.size())));
  icu::UnicodeString normalized;
  normalizer->normalize(source, normalized, status);
  if (U_FAILURE(status)) {
    SetError(error, "cannot normalize Qwen3-TTS prompt as NFC");
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
    if (IsLetterOrMark(value)) {
      letter_start = index;
    } else if (value != '\r' && value != '\n' && !IsNumber(value) &&
               index + 1 < points.size() &&
               IsLetterOrMark(points[index + 1].value)) {
      letter_start = index + 1;
    }
    if (letter_start.has_value()) {
      std::size_t stop = *letter_start;
      while (stop < points.size() && IsLetterOrMark(points[stop].value)) {
        ++stop;
      }
      pieces->push_back(Slice(text, points, index, stop));
      index = stop;
      continue;
    }

    if (IsNumber(value)) {
      std::size_t stop = index + 1;
      while (stop < points.size() && stop - index < 3 &&
             IsNumber(points[stop].value)) {
        ++stop;
      }
      pieces->push_back(Slice(text, points, index, stop));
      index = stop;
      continue;
    }

    const std::size_t punctuation_start =
        index + (value == ' ' && index + 1 < points.size() &&
                         !IsSpace(points[index + 1].value) &&
                         !IsLetterOrMark(points[index + 1].value) &&
                         !IsNumber(points[index + 1].value)
                     ? 1
                     : 0);
    std::size_t stop = punctuation_start;
    while (stop < points.size() && !IsSpace(points[stop].value) &&
           !IsLetterOrMark(points[stop].value) &&
           !IsNumber(points[stop].value)) {
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

    SetError(error, "cannot pre-tokenize Qwen3-TTS prompt");
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
  std::string error;
  if (!DecodeUtf8(text, &points, &error)) {
    throw std::runtime_error("Qwen3-TTS tokenizer contains invalid UTF-8");
  }
  std::vector<std::string> result;
  result.reserve(points.size());
  for (const Codepoint& point : points) {
    result.emplace_back(text.substr(point.begin, point.end - point.begin));
  }
  return result;
}

}  // namespace

bool Tokenizer::Load(const std::filesystem::path& model_root, Tokenizer* output,
                     std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS tokenizer output is required");
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

    Tokenizer tokenizer;
    for (const auto& [symbol, identifier] : vocabulary.members()) {
      if (!identifier.is_number()) {
        throw std::runtime_error("vocabulary ID must be numeric");
      }
      const std::size_t parsed = identifier.as_size();
      if (parsed > std::numeric_limits<std::uint32_t>::max() ||
          !tokenizer.vocab_.emplace(symbol, static_cast<std::uint32_t>(parsed))
               .second) {
        throw std::runtime_error("invalid or duplicate vocabulary entry");
      }
    }

    std::ifstream merges(model_root / "merges.txt");
    if (!merges) {
      throw std::runtime_error("cannot open " +
                               (model_root / "merges.txt").string());
    }
    std::string line;
    std::uint32_t rank = 0;
    while (std::getline(merges, line)) {
      if (line.empty() || line.starts_with('#')) {
        continue;
      }
      const std::size_t separator = line.find(' ');
      if (separator == std::string::npos || separator == 0 ||
          separator + 1 == line.size()) {
        throw std::runtime_error("invalid tokenizer merge entry");
      }
      if (!tokenizer.merge_ranks_
               .emplace(PairKey(line.substr(0, separator),
                                line.substr(separator + 1)),
                        rank++)
               .second) {
        throw std::runtime_error("duplicate tokenizer merge entry");
      }
    }

    const json::Value* decoder = tokenizer_config.find("added_tokens_decoder");
    if (decoder == nullptr || !decoder->is_object()) {
      throw std::runtime_error("missing added_tokens_decoder");
    }
    for (const auto& [identifier, specification] : decoder->members()) {
      if (!specification.is_object()) {
        throw std::runtime_error("invalid added token specification");
      }
      std::size_t consumed = 0;
      const unsigned long parsed = std::stoul(identifier, &consumed);
      const std::string content = specification.member_str("content");
      if (consumed != identifier.size() || content.empty() ||
          parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("invalid added token");
      }
      const auto id = static_cast<std::uint32_t>(parsed);
      if (!tokenizer.added_tokens_.emplace(content, id).second) {
        throw std::runtime_error("duplicate added token");
      }
      tokenizer.added_alternatives_.push_back({content, id});
    }
    std::ranges::sort(tokenizer.added_alternatives_,
                      [](const AddedToken& left, const AddedToken& right) {
                        if (left.content.size() != right.content.size()) {
                          return left.content.size() > right.content.size();
                        }
                        return left.content < right.content;
                      });

    std::uint32_t extra = 0;
    for (std::uint32_t byte = 0; byte < 256; ++byte) {
      const bool visible = (byte >= '!' && byte <= '~') ||
                           (byte >= 0xA1U && byte <= 0xACU) ||
                           (byte >= 0xAEU && byte <= 0xFFU);
      tokenizer.byte_encoder_[byte] =
          AppendCodepoint(visible ? byte : 256U + extra++);
    }
    *output = std::move(tokenizer);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, "Qwen3-TTS tokenizer load failed: " +
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
      SetError(error, "Qwen3-TTS BPE symbol is absent from vocabulary");
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

bool Tokenizer::Encode(std::string_view text, std::vector<std::uint32_t>* ids,
                       std::string* error) const {
  if (ids == nullptr) {
    SetError(error, "Qwen3-TTS token output is required");
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
  return true;
}

bool Tokenizer::EncodeAssistantPrompt(std::string_view text,
                                      std::vector<std::uint32_t>* ids,
                                      std::string* error) const {
  std::string prompt;
  prompt.append(tokenization::kImStart).append("assistant\n").append(text);
  prompt.append(tokenization::kImEnd).append("\n");
  prompt.append(tokenization::kImStart).append("assistant\n");
  return Encode(prompt, ids, error);
}

bool Tokenizer::EncodeInstructionPrompt(std::string_view instruction,
                                        std::vector<std::uint32_t>* ids,
                                        std::string* error) const {
  std::string prompt;
  prompt.append(tokenization::kImStart).append("user\n").append(instruction);
  prompt.append(tokenization::kImEnd).append("\n");
  return Encode(prompt, ids, error);
}

bool Tokenizer::EncodeReferencePrompt(std::string_view text,
                                      std::vector<std::uint32_t>* ids,
                                      std::string* error) const {
  std::string prompt;
  prompt.append(tokenization::kImStart).append("assistant\n").append(text);
  prompt.append(tokenization::kImEnd).append("\n");
  return Encode(prompt, ids, error);
}

}  // namespace gufo::models::qwen3_tts

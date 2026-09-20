#include "src/models/qwen_image_21/tokenizer.hpp"

#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "src/models/qwen_image_21/weights.hpp"

namespace gufo::models::qwen_image_21 {

Tokenizer::Tokenizer(const std::filesystem::path& root) {
  const auto config = ReadJson(root / "processor/tokenizer.json");
  const auto* model = config.find("model");
  const auto* vocab = model ? model->find("vocab") : nullptr;
  const auto* merges = model ? model->find("merges") : nullptr;
  const auto* added = config.find("added_tokens");
  const auto* pre = config.find("pre_tokenizer");
  const auto* sequence = pre ? pre->find("pretokenizers") : nullptr;
  if (!vocab || !vocab->is_object() || !merges || !merges->is_array() ||
      !added || !added->is_array() || !sequence || !sequence->is_array() ||
      sequence->size() != 2)
    throw std::runtime_error("invalid Qwen-Image tokenizer");
  const auto& split = sequence->items()[0];
  const auto require_false = [](const json::Value& object, const char* key) {
    const auto* value = object.find(key);
    if (!value || !value->is_bool() || value->as_bool())
      throw std::runtime_error(std::string("unsupported tokenizer setting: ") +
                               key);
  };
  const auto* regex = split.find("pattern");
  if (split.member_str("type") != "Split" || !regex ||
      sequence->items()[1].member_str("type") != "ByteLevel")
    throw std::runtime_error("unsupported Qwen-Image pre-tokenizer");
  if (model->member_str("type") != "BPE" ||
      split.member_str("behavior") != "Isolated" || !model->find("dropout") ||
      !model->find("dropout")->is_null() ||
      !model->member_str("continuing_subword_prefix").empty() ||
      !model->member_str("end_of_word_suffix").empty())
    throw std::runtime_error("unsupported Qwen-Image BPE configuration");
  require_false(split, "invert");
  require_false(sequence->items()[1], "add_prefix_space");
  require_false(sequence->items()[1], "use_regex");
  require_false(*model, "byte_fallback");
  require_false(*model, "ignore_merges");
  pattern_ = regex->member_str("Regex");
  if (pattern_.empty())
    throw std::runtime_error("missing tokenizer regex");
  if (const auto* normalizer = config.find("normalizer");
      !normalizer || normalizer->member_str("type") != "NFC")
    throw std::runtime_error("unsupported Qwen-Image tokenizer normalizer");
  std::vector<std::string> tokens(151669);
  const auto token_id = [](const json::Value* value) {
    if (!value || !value->is_number() || !std::isfinite(value->as_double()) ||
        value->as_double() < 0 || value->as_double() >= 151936 ||
        value->as_double() != std::floor(value->as_double()))
      throw std::runtime_error("invalid tokenizer ID");
    return static_cast<std::uint32_t>(value->as_double());
  };
  const auto install = [&](std::size_t id, const std::string& word) {
    if (id >= 151936 || word.empty())
      throw std::runtime_error("invalid tokenizer entry");
    if (id >= tokens.size())
      tokens.resize(id + 1);
    if (!tokens[id].empty() && tokens[id] != word)
      throw std::runtime_error("duplicate tokenizer entry");
    tokens[id] = word;
  };
  for (const auto& [word, id] : vocab->members())
    install(token_id(&id), word);
  for (const auto& entry : added->items()) {
    for (const char* key : {"single_word", "lstrip", "rstrip", "normalized"})
      require_false(entry, key);
    const auto id = token_id(entry.find("id"));
    const auto word = entry.member_str("content");
    install(static_cast<std::size_t>(id), word);
    specials_.emplace(word, static_cast<std::uint32_t>(id));
  }
  while (!tokens.empty() && tokens.back().empty())
    tokens.pop_back();
  std::vector<std::string> pairs;
  for (const auto& merge : merges->items()) {
    if (merge.is_string())
      pairs.push_back(merge.str());
    else if (merge.is_array() && merge.size() == 2 &&
             merge.items()[0].is_string() && merge.items()[1].is_string())
      pairs.push_back(merge.items()[0].str() + " " + merge.items()[1].str());
    else
      throw std::runtime_error("invalid tokenizer merge");
  }
  std::string error;
  bpe_ = tokenization::QwenTokenizer::CreateFromVocabulary(
      tokens, pairs, specials_, &error, {.eager_decoded_tokens = false});
  if (!bpe_)
    throw std::runtime_error(error);
}

std::vector<std::uint32_t> Tokenizer::Encode(std::string_view input) const {
  std::vector<std::uint32_t> result;
  UErrorCode status = U_ZERO_ERROR;
  const auto pattern =
      std::unique_ptr<icu::RegexPattern>(icu::RegexPattern::compile(
          icu::UnicodeString::fromUTF8(pattern_), 0, status));
  if (U_FAILURE(status) || !pattern)
    throw std::runtime_error("invalid tokenizer regex");
  while (!input.empty()) {
    std::size_t end = input.size(), length = 0;
    std::uint32_t token = 0;
    for (const auto& [special, id] : specials_) {
      const auto offset = input.find(special);
      if (offset < end || (offset == end && special.size() > length)) {
        end = offset;
        length = special.size();
        token = id;
      }
    }
    const auto original = icu::UnicodeString::fromUTF8(
        icu::StringPiece(input.data(), static_cast<int>(end)));
    icu::UnicodeString text;
    const auto* normalizer = icu::Normalizer2::getNFCInstance(status);
    if (U_FAILURE(status))
      throw std::runtime_error("NFC initialization failed");
    normalizer->normalize(original, text, status);
    const auto matcher =
        std::unique_ptr<icu::RegexMatcher>(pattern->matcher(text, status));
    if (U_FAILURE(status) || !matcher)
      throw std::runtime_error("Qwen-Image Unicode processing failed");
    int consumed = 0;
    while (matcher->find(status)) {
      if (matcher->start(status) != consumed)
        throw std::runtime_error("Qwen-Image tokenizer left unmatched text");
      std::string piece;
      matcher->group(status).toUTF8String(piece);
      const auto ids = bpe_->Encode(piece, {.parse_special_tokens = false});
      result.insert(result.end(), ids.begin(), ids.end());
      consumed = matcher->end(status);
    }
    if (U_FAILURE(status) || consumed != text.length())
      throw std::runtime_error("Qwen-Image tokenization failed");
    if (!length)
      break;
    result.push_back(token);
    input.remove_prefix(end + length);
  }
  return result;
}

}  // namespace gufo::models::qwen_image_21

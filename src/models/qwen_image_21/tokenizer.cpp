#include "src/models/qwen_image_21/tokenizer.hpp"

#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/unistr.h>

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string_view>

#include "gufo/tokenizer_json.h"

namespace gufo::models::qwen_image_21 {

namespace {

constexpr std::uintmax_t kMaximumTokenizerBytes = 512ULL << 20U;

std::string_view View(GufoTokenizerJsonBytes b) noexcept {
  return b.len == 0
             ? std::string_view{}
             : std::string_view(reinterpret_cast<const char*>(b.data), b.len);
}

void CopyError(void* ctx, const std::uint8_t* data, std::size_t len) noexcept {
  auto* out = static_cast<std::string*>(ctx);
  if (!out)
    return;
  if (!data || len == 0) {
    out->clear();
    return;
  }
  try {
    out->assign(reinterpret_cast<const char*>(data), len);
  } catch (...) {
    out->clear();
  }
}

std::string ReadTokenizerFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open " + path.string());
  const auto size = input.tellg();
  if (size < 0 || static_cast<std::uintmax_t>(size) > kMaximumTokenizerBytes)
    throw std::runtime_error("cannot stat " + path.string());
  input.seekg(0);
  std::string contents(static_cast<std::size_t>(size), '\0');
  if (size > 0 && !input.read(contents.data(), size))
    throw std::runtime_error("cannot read " + path.string());
  return contents;
}

}  // namespace

Tokenizer::Tokenizer(const std::filesystem::path& root) {
  const std::string contents =
      ReadTokenizerFile(root / "processor/tokenizer.json");
  GufoTokenizerJson* raw = nullptr;
  std::string parse_err;
  if (gufo_tokenizer_json_parse(
          reinterpret_cast<const std::uint8_t*>(contents.data()),
          contents.size(), 151936, &raw, &CopyError,
          &parse_err) != GUFO_TOKENIZER_JSON_OK) {
    throw std::runtime_error(parse_err.empty() ? "invalid Qwen-Image tokenizer"
                                               : parse_err);
  }
  std::unique_ptr<GufoTokenizerJson, void (*)(GufoTokenizerJson*)> parsed(
      raw, &gufo_tokenizer_json_free);

  GufoTokenizerJsonMeta meta{};
  if (gufo_tokenizer_json_meta(parsed.get(), &meta) != GUFO_TOKENIZER_JSON_OK ||
      !meta.has_vocab || !meta.has_merges || !meta.has_added_tokens ||
      meta.pre_tokenizer_sub_count != 2) {
    throw std::runtime_error("invalid Qwen-Image tokenizer");
  }
  const auto require_false = [](bool present, bool value, const char* key) {
    if (!present || value)
      throw std::runtime_error(std::string("unsupported tokenizer setting: ") +
                               key);
  };
  if (View(meta.split_type) != "Split" ||
      View(meta.byte_level_type) != "ByteLevel")
    throw std::runtime_error("unsupported Qwen-Image pre-tokenizer");
  if (View(meta.model_type) != "BPE" ||
      View(meta.split_behavior) != "Isolated" || !meta.has_dropout ||
      !meta.dropout_is_null || !View(meta.continuing_subword_prefix).empty() ||
      !View(meta.end_of_word_suffix).empty())
    throw std::runtime_error("unsupported Qwen-Image BPE configuration");
  require_false(meta.has_split_invert, meta.split_invert, "invert");
  require_false(meta.has_add_prefix_space, meta.add_prefix_space,
                "add_prefix_space");
  require_false(meta.has_use_regex, meta.use_regex, "use_regex");
  require_false(meta.has_byte_fallback, meta.byte_fallback, "byte_fallback");
  require_false(meta.has_ignore_merges, meta.ignore_merges, "ignore_merges");
  pattern_ = std::string(View(meta.split_regex));
  if (pattern_.empty())
    throw std::runtime_error("missing tokenizer regex");
  if (!meta.has_normalizer || View(meta.normalizer_type) != "NFC")
    throw std::runtime_error("unsupported Qwen-Image tokenizer normalizer");

  std::vector<std::string> tokens(151669);
  const auto install = [&](std::size_t id, const std::string& word) {
    if (id >= 151936 || word.empty())
      throw std::runtime_error("invalid tokenizer entry");
    if (id >= tokens.size())
      tokens.resize(id + 1);
    if (!tokens[id].empty() && tokens[id] != word)
      throw std::runtime_error("duplicate tokenizer entry");
    tokens[id] = word;
  };
  const std::size_t vocab_len = gufo_tokenizer_json_vocab_len(parsed.get());
  for (std::size_t i = 0; i < vocab_len; ++i) {
    GufoTokenizerJsonVocabEntry entry{};
    if (gufo_tokenizer_json_vocab_at(parsed.get(), i, &entry) !=
        GUFO_TOKENIZER_JSON_OK)
      throw std::runtime_error("invalid tokenizer entry");
    install(entry.id, std::string(View(entry.token)));
  }
  const std::size_t added_len = gufo_tokenizer_json_added_len(parsed.get());
  for (std::size_t i = 0; i < added_len; ++i) {
    GufoTokenizerJsonAddedToken entry{};
    if (gufo_tokenizer_json_added_at(parsed.get(), i, &entry) !=
        GUFO_TOKENIZER_JSON_OK)
      throw std::runtime_error("invalid tokenizer entry");
    require_false(entry.has_single_word && entry.flags_valid, entry.single_word,
                  "single_word");
    require_false(entry.has_lstrip && entry.flags_valid, entry.lstrip,
                  "lstrip");
    require_false(entry.has_rstrip && entry.flags_valid, entry.rstrip,
                  "rstrip");
    require_false(entry.has_normalized && entry.flags_valid, entry.normalized,
                  "normalized");
    const std::string word(View(entry.content));
    install(static_cast<std::size_t>(entry.id), word);
    specials_.emplace(word, entry.id);
  }
  while (!tokens.empty() && tokens.back().empty())
    tokens.pop_back();
  const std::size_t merges_len = gufo_tokenizer_json_merges_len(parsed.get());
  std::vector<std::string> pairs;
  pairs.reserve(merges_len);
  for (std::size_t i = 0; i < merges_len; ++i) {
    GufoTokenizerJsonMergeEntry entry{};
    if (gufo_tokenizer_json_merge_at(parsed.get(), i, &entry) !=
        GUFO_TOKENIZER_JSON_OK)
      throw std::runtime_error("invalid tokenizer merge");
    pairs.push_back(std::string(View(entry.left)) + " " +
                    std::string(View(entry.right)));
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

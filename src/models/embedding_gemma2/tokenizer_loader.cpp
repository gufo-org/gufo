#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/tokenizer_json.h"
#include "tokenizer.hpp"

namespace gufo::models::embedding_gemma2 {
namespace {

constexpr std::uintmax_t kMaximumTokenizerBytes = 512ULL << 20U;
constexpr std::uint32_t kMaximumTokenId = 1U << 24U;

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

bool ReadFile(const std::filesystem::path& path, std::string* contents,
              std::string* error) {
  std::error_code ec;
  const std::uintmax_t size = std::filesystem::file_size(path, ec);
  if (ec) {
    *error = "cannot stat " + path.string() + ": " + ec.message();
    return false;
  }
  if (size > kMaximumTokenizerBytes) {
    *error = "unexpectedly large " + path.string();
    return false;
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    *error = "cannot open " + path.string();
    return false;
  }
  contents->resize(static_cast<std::size_t>(size));
  input.read(contents->data(), static_cast<std::streamsize>(size));
  if (!input) {
    *error = "cannot read " + path.string();
    return false;
  }
  return true;
}

}  // namespace

struct TokenizerLoader {
  static bool Run(const std::string& model_dir, Tokenizer* t,
                  std::string* error);
};

bool TokenizerLoader::Run(const std::string& model_dir, Tokenizer* t,
                          std::string* error) {
  const std::filesystem::path path =
      std::filesystem::path(model_dir) / "tokenizer.json";
  std::string contents;
  if (!ReadFile(path, &contents, error)) {
    return false;
  }

  GufoTokenizerJson* raw = nullptr;
  std::string parse_err;
  if (gufo_tokenizer_json_parse(
          reinterpret_cast<const std::uint8_t*>(contents.data()),
          contents.size(), kMaximumTokenId, &raw, &CopyError,
          &parse_err) != GUFO_TOKENIZER_JSON_OK) {
    *error = "invalid " + path.string() + ": " + parse_err;
    return false;
  }
  std::unique_ptr<GufoTokenizerJson, void (*)(GufoTokenizerJson*)> parsed(
      raw, &gufo_tokenizer_json_free);

  GufoTokenizerJsonMeta meta{};
  if (gufo_tokenizer_json_meta(parsed.get(), &meta) != GUFO_TOKENIZER_JSON_OK) {
    *error = "internal error reading tokenizer metadata";
    return false;
  }

  if (!meta.truncation_is_null || !meta.padding_is_null) {
    *error = "truncation and padding are not supported";
    return false;
  }
  if (!meta.has_normalizer || View(meta.normalizer_type) != "Replace" ||
      View(meta.normalizer_pattern_string) != " " ||
      View(meta.normalizer_content) != Tokenizer::kSpaceSymbol) {
    *error = "normalizer must replace ' ' with U+2581";
    return false;
  }
  if (!meta.has_pre_tokenizer || View(meta.pre_tokenizer_type) != "Split" ||
      View(meta.pre_tokenizer_pattern_string) != " " ||
      View(meta.pre_tokenizer_behavior) != "MergedWithPrevious" ||
      (meta.has_pre_tokenizer_invert && meta.pre_tokenizer_invert)) {
    *error = "pre_tokenizer must split on ' ' (MergedWithPrevious)";
    return false;
  }
  if (!meta.has_model) {
    *error = "model is missing";
    return false;
  }
  if (!meta.has_vocab) {
    *error = "model.vocab is missing";
    return false;
  }

  const std::size_t vocab_len = gufo_tokenizer_json_vocab_len(parsed.get());
  if (vocab_len == 0) {
    *error = "model.vocab is empty";
    return false;
  }
  std::vector<char> present;
  t->tokens_.reserve(vocab_len);
  present.reserve(vocab_len);
  for (std::size_t i = 0; i < vocab_len; ++i) {
    GufoTokenizerJsonVocabEntry entry{};
    if (gufo_tokenizer_json_vocab_at(parsed.get(), i, &entry) !=
        GUFO_TOKENIZER_JSON_OK) {
      *error = "internal error reading model.vocab";
      return false;
    }
    if (entry.id >= t->tokens_.size()) {
      t->tokens_.resize(static_cast<std::size_t>(entry.id) + 1);
      present.resize(t->tokens_.size(), 0);
    }
    if (present[entry.id] != 0) {
      *error = "model.vocab repeats id " + std::to_string(entry.id);
      return false;
    }
    present[entry.id] = 1;
    t->tokens_[entry.id] = std::string(View(entry.token));
  }
  if (std::find(present.begin(), present.end(), 0) != present.end()) {
    *error = "model.vocab ids are not contiguous";
    return false;
  }

  const std::size_t added_len = gufo_tokenizer_json_added_len(parsed.get());
  for (std::size_t i = 0; i < added_len; ++i) {
    GufoTokenizerJsonAddedToken item{};
    if (gufo_tokenizer_json_added_at(parsed.get(), i, &item) !=
        GUFO_TOKENIZER_JSON_OK) {
      *error = "invalid added_tokens entry";
      return false;
    }
    if (item.single_word) {
      *error = "added token option 'single_word' is not supported";
      return false;
    }
    if (item.lstrip) {
      *error = "added token option 'lstrip' is not supported";
      return false;
    }
    if (item.rstrip) {
      *error = "added token option 'rstrip' is not supported";
      return false;
    }
    if (item.normalized) {
      *error = "added token option 'normalized' is not supported";
      return false;
    }
    const std::string content(View(item.content));
    if (item.id >= t->tokens_.size()) {
      t->tokens_.resize(static_cast<std::size_t>(item.id) + 1);
    }
    if (t->tokens_[item.id].empty()) {
      t->tokens_[item.id] = content;
    } else if (t->tokens_[item.id] != content) {
      *error = "added token '" + content + "' conflicts with vocabulary id " +
               std::to_string(item.id);
      return false;
    }
    t->added_.push_back({content, item.id});
  }
  std::stable_sort(
      t->added_.begin(), t->added_.end(),
      [](const Tokenizer::AddedToken& a, const Tokenizer::AddedToken& b) {
        return a.content.size() > b.content.size();
      });
  for (const Tokenizer::AddedToken& token : t->added_) {
    t->added_first_byte_[static_cast<unsigned char>(token.content[0])] = true;
  }

  t->ids_.reserve(t->tokens_.size());
  for (std::size_t id = 0; id < t->tokens_.size(); ++id) {
    if (t->tokens_[id].empty()) {
      continue;
    }
    t->max_token_bytes_ = std::max(t->max_token_bytes_, t->tokens_[id].size());
    if (!t->ids_.emplace(t->tokens_[id], static_cast<std::uint32_t>(id))
             .second) {
      *error = "duplicate token '" + t->tokens_[id] + "' in the vocabulary";
      return false;
    }
  }

  if (View(meta.model_type) != "BPE") {
    *error = "model.type must be BPE";
    return false;
  }
  if (meta.has_dropout && !meta.dropout_is_null) {
    *error = "model.dropout is not supported";
    return false;
  }
  if (!View(meta.continuing_subword_prefix).empty()) {
    *error = "model.continuing_subword_prefix is not supported";
    return false;
  }
  if (!View(meta.end_of_word_suffix).empty()) {
    *error = "model.end_of_word_suffix is not supported";
    return false;
  }
  if (meta.has_ignore_merges && meta.ignore_merges) {
    *error = "model.ignore_merges is not supported";
    return false;
  }
  if (!meta.has_byte_fallback || !meta.byte_fallback) {
    *error = "model.byte_fallback must be true";
    return false;
  }
  if (meta.has_unk_token && !meta.unk_token_is_null &&
      t->ids_.find(std::string(View(meta.unk_token))) == t->ids_.end()) {
    *error = "model.unk_token is not in the vocabulary";
    return false;
  }

  for (std::size_t byte = 0; byte < 256; ++byte) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    const std::string name =
        std::string("<0x") + kHex[byte >> 4U] + kHex[byte & 0xFU] + ">";
    const auto it = t->ids_.find(name);
    if (it == t->ids_.end()) {
      *error = "the vocabulary lacks the <0xXX> byte-fallback tokens";
      return false;
    }
    t->byte_tokens_[byte] = it->second;
  }

  if (!meta.has_merges) {
    *error = "model.merges is missing";
    return false;
  }
  const std::size_t merges_len = gufo_tokenizer_json_merges_len(parsed.get());
  t->merges_.reserve(merges_len);
  std::string joined;
  for (std::size_t rank = 0; rank < merges_len; ++rank) {
    GufoTokenizerJsonMergeEntry entry{};
    if (gufo_tokenizer_json_merge_at(parsed.get(), rank, &entry) !=
            GUFO_TOKENIZER_JSON_OK ||
        entry.was_string) {
      *error = "invalid model.merges: entry " + std::to_string(rank) +
               " is not a pair of strings";
      return false;
    }
    const std::string left(View(entry.left));
    const std::string right(View(entry.right));
    joined = left + right;
    const auto left_it = t->ids_.find(left);
    const auto right_it = t->ids_.find(right);
    const auto joined_it = t->ids_.find(joined);
    if (left_it == t->ids_.end() || right_it == t->ids_.end() ||
        joined_it == t->ids_.end()) {
      *error = "model.merges entry " + std::to_string(rank) +
               " references a token missing from the vocabulary";
      return false;
    }
    t->merges_[Tokenizer::PairKey(left_it->second, right_it->second)] = {
        static_cast<std::uint32_t>(rank), joined_it->second};
  }

  if (meta.has_post_processor) {
    if (!meta.post_processor_valid) {
      *error = std::string(View(meta.post_processor_error));
      return false;
    }
    const std::uint32_t* prefix_ptr = nullptr;
    std::size_t prefix_len = 0;
    const std::uint32_t* suffix_ptr = nullptr;
    std::size_t suffix_len = 0;
    (void)gufo_tokenizer_json_prefix_ids(parsed.get(), &prefix_ptr,
                                         &prefix_len);
    (void)gufo_tokenizer_json_suffix_ids(parsed.get(), &suffix_ptr,
                                         &suffix_len);
    for (std::size_t i = 0; i < prefix_len; ++i) {
      if (prefix_ptr[i] >= t->tokens_.size()) {
        *error = "post_processor special token id out of range";
        return false;
      }
      t->prefix_ids_.push_back(prefix_ptr[i]);
    }
    for (std::size_t i = 0; i < suffix_len; ++i) {
      if (suffix_ptr[i] >= t->tokens_.size()) {
        *error = "post_processor special token id out of range";
        return false;
      }
      t->suffix_ids_.push_back(suffix_ptr[i]);
    }
  }
  return true;
}

std::optional<Tokenizer> Tokenizer::Load(const std::string& model_dir,
                                         std::string* error) {
  Tokenizer tokenizer;
  std::string message;
  if (!TokenizerLoader::Run(model_dir, &tokenizer, &message)) {
    if (error != nullptr) {
      *error = std::move(message);
    }
    return std::nullopt;
  }
  return tokenizer;
}

}  // namespace gufo::models::embedding_gemma2

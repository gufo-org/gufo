#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "tokenizer.hpp"

namespace gufo::models::embedding_gemma2 {
namespace {

constexpr std::uintmax_t kMaximumTokenizerBytes = 512ULL << 20U;
constexpr std::uint32_t kMaximumTokenId = 1U << 24U;

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

// The member `key` of `object`, or a null value when it is absent.
const json::Value& Member(const json::Value& object, const char* key) {
  static const json::Value kNull;
  const json::Value* value = object.find(key);
  return value != nullptr ? *value : kNull;
}

// A non-negative integer below `limit`.
bool ReadId(const json::Value& value, std::uint64_t limit, std::uint32_t* id) {
  const std::size_t number = value.as_size(limit);
  if (number >= limit) {
    return false;
  }
  *id = static_cast<std::uint32_t>(number);
  return true;
}

}  // namespace

struct TokenizerLoader {
  static bool Run(const std::string& model_dir, Tokenizer* t,
                  std::string* error);

private:
  static bool LoadVocab(const json::Value& vocab, Tokenizer* t,
                        std::string* error);
  static bool LoadMerges(const json::Value& merges, Tokenizer* t,
                         std::string* error);
  static bool CheckModelOptions(const json::Value& model, const Tokenizer& t,
                                std::string* error);
  static bool LoadAddedTokens(const json::Value& added, Tokenizer* t,
                              std::string* error);
  static bool LoadPostProcessor(const json::Value& processor, Tokenizer* t,
                                std::string* error);
  static bool CheckNormalization(const json::Value& normalizer,
                                 const json::Value& pre_tokenizer,
                                 std::string* error);
};

bool TokenizerLoader::LoadVocab(const json::Value& vocab, Tokenizer* t,
                                std::string* error) {
  if (!vocab.is_object()) {
    *error = vocab.is_null() ? "model.vocab is missing"
                             : "invalid model.vocab: expected an object";
    return false;
  }
  std::vector<char> present;
  for (const auto& [token, value] : vocab.members()) {
    std::uint32_t id = 0;
    if (!ReadId(value, kMaximumTokenId, &id)) {
      *error = "invalid model.vocab: id of '" + token +
               "' is not an integer in [0, " + std::to_string(kMaximumTokenId) +
               ")";
      return false;
    }
    if (id >= t->tokens_.size()) {
      t->tokens_.resize(static_cast<std::size_t>(id) + 1);
      present.resize(t->tokens_.size(), 0);
    }
    if (present[id] != 0) {
      *error = "model.vocab repeats id " + std::to_string(id);
      return false;
    }
    present[id] = 1;
    t->tokens_[id] = token;
  }
  if (t->tokens_.empty()) {
    *error = "model.vocab is empty";
    return false;
  }
  if (std::find(present.begin(), present.end(), 0) != present.end()) {
    *error = "model.vocab ids are not contiguous";
    return false;
  }
  return true;
}

bool TokenizerLoader::LoadAddedTokens(const json::Value& added, Tokenizer* t,
                                      std::string* error) {
  if (added.is_null()) {
    return true;
  }
  if (!added.is_array()) {
    *error = "added_tokens must be an array";
    return false;
  }
  for (const json::Value& item : added.items()) {
    const json::Value& content = Member(item, "content");
    std::uint32_t token_id = 0;
    if (!content.is_string() || content.str().empty() ||
        !ReadId(Member(item, "id"), kMaximumTokenId, &token_id)) {
      *error = "invalid added_tokens entry";
      return false;
    }
    for (const char* flag : {"single_word", "lstrip", "rstrip", "normalized"}) {
      if (Member(item, flag).as_bool(false)) {
        *error =
            std::string("added token option '") + flag + "' is not supported";
        return false;
      }
    }
    if (token_id >= t->tokens_.size()) {
      t->tokens_.resize(static_cast<std::size_t>(token_id) + 1);
    }
    if (t->tokens_[token_id].empty()) {
      t->tokens_[token_id] = content.str();
    } else if (t->tokens_[token_id] != content.str()) {
      *error = "added token '" + content.str() +
               "' conflicts with vocabulary id " + std::to_string(token_id);
      return false;
    }
    t->added_.push_back({content.str(), token_id});
  }
  std::stable_sort(
      t->added_.begin(), t->added_.end(),
      [](const Tokenizer::AddedToken& a, const Tokenizer::AddedToken& b) {
        return a.content.size() > b.content.size();
      });
  for (const Tokenizer::AddedToken& token : t->added_) {
    t->added_first_byte_[static_cast<unsigned char>(token.content[0])] = true;
  }
  return true;
}

bool TokenizerLoader::LoadMerges(const json::Value& merges, Tokenizer* t,
                                 std::string* error) {
  if (!merges.is_array()) {
    *error = merges.is_null() ? "model.merges is missing"
                              : "invalid model.merges: expected an array";
    return false;
  }
  t->merges_.reserve(merges.size());

  std::string joined;
  std::uint32_t rank = 0;
  for (const json::Value& pair : merges.items()) {
    if (!pair.is_array() || pair.size() != 2 || !pair.items()[0].is_string() ||
        !pair.items()[1].is_string()) {
      *error = "invalid model.merges: entry " + std::to_string(rank) +
               " is not a pair of strings";
      return false;
    }
    const std::string& left = pair.items()[0].str();
    const std::string& right = pair.items()[1].str();
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
    // A repeated pair keeps its last rank, as in Hugging Face.
    t->merges_[Tokenizer::PairKey(left_it->second, right_it->second)] = {
        rank, joined_it->second};
    ++rank;
  }
  return true;
}

bool TokenizerLoader::CheckModelOptions(const json::Value& model,
                                        const Tokenizer& t,
                                        std::string* error) {
  if (!Member(model, "type").is_string() ||
      Member(model, "type").str() != "BPE") {
    *error = "model.type must be BPE";
    return false;
  }
  if (!Member(model, "dropout").is_null()) {
    *error = "model.dropout is not supported";
    return false;
  }
  for (const char* key : {"continuing_subword_prefix", "end_of_word_suffix"}) {
    if (Member(model, key).is_string() && !Member(model, key).str().empty()) {
      *error = std::string("model.") + key + " is not supported";
      return false;
    }
  }
  if (Member(model, "ignore_merges").as_bool(false)) {
    *error = "model.ignore_merges is not supported";
    return false;
  }
  if (!Member(model, "byte_fallback").as_bool(false)) {
    *error = "model.byte_fallback must be true";
    return false;
  }
  // Every non-vocabulary byte becomes a <0xXX> token, so the unknown token
  // is never emitted. Its name must still be valid.
  if (Member(model, "unk_token").is_string() &&
      t.ids_.find(Member(model, "unk_token").str()) == t.ids_.end()) {
    *error = "model.unk_token is not in the vocabulary";
    return false;
  }
  return true;
}

bool TokenizerLoader::CheckNormalization(const json::Value& normalizer,
                                         const json::Value& pre_tokenizer,
                                         std::string* error) {
  auto string_pattern = [](const json::Value& value) -> std::string {
    const json::Value* pattern = value.find("pattern");
    const json::Value* text = pattern != nullptr && pattern->is_object()
                                  ? pattern->find("String")
                                  : nullptr;
    return text != nullptr && text->is_string() ? text->str()
                                                : std::string("\x01");
  };
  if (!normalizer.is_object() || normalizer.member_str("type") != "Replace" ||
      string_pattern(normalizer) != " " ||
      normalizer.member_str("content") != Tokenizer::kSpaceSymbol) {
    *error = "normalizer must replace ' ' with U+2581";
    return false;
  }
  // With every space replaced, a split on ' ' never fires, so each text
  // segment stays one BPE word.
  const json::Value* invert = pre_tokenizer.find("invert");
  if (!pre_tokenizer.is_object() ||
      pre_tokenizer.member_str("type") != "Split" ||
      string_pattern(pre_tokenizer) != " " ||
      pre_tokenizer.member_str("behavior") != "MergedWithPrevious" ||
      (invert != nullptr && invert->as_bool(false))) {
    *error = "pre_tokenizer must split on ' ' (MergedWithPrevious)";
    return false;
  }
  return true;
}

bool TokenizerLoader::LoadPostProcessor(const json::Value& processor,
                                        Tokenizer* t, std::string* error) {
  if (processor.is_null()) {
    return true;
  }
  const json::Value* single = processor.find("single");
  const json::Value* specials = processor.find("special_tokens");
  if (processor.member_str("type") != "TemplateProcessing" ||
      single == nullptr || !single->is_array() || specials == nullptr ||
      !specials->is_object()) {
    *error = "post_processor must be TemplateProcessing";
    return false;
  }
  bool seen_sequence = false;
  for (const json::Value& item : single->items()) {
    const json::Value* sequence = item.find("Sequence");
    const json::Value* special = item.find("SpecialToken");
    std::vector<std::uint32_t>* target =
        seen_sequence ? &t->suffix_ids_ : &t->prefix_ids_;
    if (sequence != nullptr) {
      if (seen_sequence || sequence->member_str("id") != "A") {
        *error = "post_processor single template must hold one sequence A";
        return false;
      }
      seen_sequence = true;
    } else if (special != nullptr) {
      const json::Value* entry = specials->find(special->member_str("id"));
      const json::Value* ids = entry != nullptr ? entry->find("ids") : nullptr;
      if (ids == nullptr || !ids->is_array()) {
        *error = "post_processor references an unknown special token";
        return false;
      }
      for (const json::Value& id : ids->items()) {
        std::uint32_t value = 0;
        if (!ReadId(id, t->tokens_.size(), &value)) {
          *error = "post_processor special token id out of range";
          return false;
        }
        target->push_back(value);
      }
    } else {
      *error = "unsupported post_processor template item";
      return false;
    }
  }
  if (!seen_sequence) {
    *error = "post_processor single template lacks sequence A";
    return false;
  }
  return true;
}

bool TokenizerLoader::Run(const std::string& model_dir, Tokenizer* t,
                          std::string* error) {
  const std::filesystem::path path =
      std::filesystem::path(model_dir) / "tokenizer.json";
  json::Value root;
  {
    std::string contents;
    if (!ReadFile(path, &contents, error)) {
      return false;
    }
    try {
      root = json::parse(contents);
    } catch (const std::exception& e) {
      *error = "invalid " + path.string() + ": " + e.what();
      return false;
    }
  }
  if (!root.is_object()) {
    *error = "invalid " + path.string() + ": expected an object";
    return false;
  }
  if (!Member(root, "truncation").is_null() ||
      !Member(root, "padding").is_null()) {
    *error = "truncation and padding are not supported";
    return false;
  }
  if (!CheckNormalization(Member(root, "normalizer"),
                          Member(root, "pre_tokenizer"), error)) {
    return false;
  }

  const json::Value& model = Member(root, "model");
  if (!model.is_object()) {
    *error = model.is_null() ? "model is missing"
                             : "invalid model: expected an object";
    return false;
  }
  if (!LoadVocab(Member(model, "vocab"), t, error) ||
      !LoadAddedTokens(Member(root, "added_tokens"), t, error)) {
    return false;
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
  if (!CheckModelOptions(model, *t, error)) {
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
  return LoadMerges(Member(model, "merges"), t, error) &&
         LoadPostProcessor(Member(root, "post_processor"), t, error);
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

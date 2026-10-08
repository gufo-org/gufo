#include "src/models/gemma4/tokenizer.hpp"

#include <algorithm>
#include <queue>
#include <utility>
#include <variant>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gemma4 {
namespace {

constexpr std::string_view kEscapedSpace = "\xE2\x96\x81";  // U+2581
constexpr std::uint32_t kReplacement = 0xFFFD;

/// End-of-generation token texts of the Gemma 4 vocabulary.
constexpr std::array<std::string_view, 3> kEndOfGeneration = {
    "<eos>", "<turn|>", "<|tool_response>"};

std::uint64_t PairKey(TokenId left, TokenId right) noexcept {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32) |
         static_cast<std::uint32_t>(right);
}

// Pre-split words from this length on are cached (shorter ones encode faster
// than a lookup pays off), within this many bytes of text and ids.
constexpr std::size_t kCachedWordBytes = 64;
constexpr std::size_t kWordCacheBytes = std::size_t{64} << 20;

// No valid pair has this key: token ids stay below 2^31.
constexpr std::uint64_t kEmptyPair = ~std::uint64_t{0};

std::size_t PairSlot(std::uint64_t key, std::uint32_t shift) noexcept {
  return static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> shift);
}

void AppendUtf8(std::uint32_t cpt, std::string* out) {
  if (cpt <= 0x7F) {
    out->push_back(static_cast<char>(cpt));
  } else if (cpt <= 0x7FF) {
    out->push_back(static_cast<char>(0xC0 | (cpt >> 6)));
    out->push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  } else if (cpt <= 0xFFFF) {
    out->push_back(static_cast<char>(0xE0 | (cpt >> 12)));
    out->push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  } else {
    out->push_back(static_cast<char>(0xF0 | (cpt >> 18)));
    out->push_back(static_cast<char>(0x80 | ((cpt >> 12) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | ((cpt >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cpt & 0x3F)));
  }
}

/// Re-encodes `text` through llama.cpp's permissive decoder: structurally
/// valid sequences round-trip, every byte of an invalid one becomes U+FFFD.
std::string Canonicalize(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  const auto byte = [&](std::size_t i) {
    return static_cast<std::uint8_t>(text[i]);
  };
  const auto cont = [&](std::size_t i) {
    return i < text.size() && (byte(i) & 0xC0) == 0x80;
  };
  std::size_t i = 0;
  while (i < text.size()) {
    const std::uint8_t b = byte(i);
    std::uint32_t cpt = kReplacement;
    std::size_t len = 1;
    if ((b & 0x80) == 0) {
      cpt = b;
    } else if ((b & 0x40) == 0) {
      // Stray continuation byte.
    } else if ((b & 0x20) == 0) {
      if (cont(i + 1)) {
        cpt = ((b & 0x1FU) << 6) | (byte(i + 1) & 0x3FU);
        len = 2;
      }
    } else if ((b & 0x10) == 0) {
      if (cont(i + 1) && cont(i + 2)) {
        cpt = ((b & 0x0FU) << 12) | ((byte(i + 1) & 0x3FU) << 6) |
              (byte(i + 2) & 0x3FU);
        len = 3;
      }
    } else if ((b & 0x08) == 0) {
      if (cont(i + 1) && cont(i + 2) && cont(i + 3)) {
        cpt = ((b & 0x07U) << 18) | ((byte(i + 1) & 0x3FU) << 12) |
              ((byte(i + 2) & 0x3FU) << 6) | (byte(i + 3) & 0x3FU);
        len = 4;
      }
    }
    AppendUtf8(cpt, &out);
    i += len;
  }
  return out;
}

/// Byte length of the UTF-8 character starting with `lead`, as llama.cpp's
/// `unicode_len_utf8` computes it.
std::size_t Utf8Length(char lead) noexcept {
  constexpr std::array<std::size_t, 16> kLookup = {1, 1, 1, 1, 1, 1, 1, 1,
                                                   1, 1, 1, 1, 2, 2, 3, 4};
  return kLookup[static_cast<std::uint8_t>(lead) >> 4];
}

std::optional<TokenId> ByteValue(std::string_view text) {
  if (text.size() != 6 || text.substr(0, 3) != "<0x" || text[5] != '>') {
    return std::nullopt;
  }
  int value = 0;
  for (std::size_t i = 3; i < 5; ++i) {
    const char c = text[i];
    value <<= 4;
    if (c >= '0' && c <= '9') {
      value |= c - '0';
    } else if (c >= 'A' && c <= 'F') {
      value |= c - 'A' + 10;
    } else if (c >= 'a' && c <= 'f') {
      value |= c - 'a' + 10;
    } else {
      return std::nullopt;
    }
  }
  return value;
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto fail = [&](std::string message) -> std::unique_ptr<Tokenizer> {
    if (error_msg != nullptr) {
      *error_msg = std::move(message);
    }
    return nullptr;
  };
  if (reader.GetMetadataString("tokenizer.ggml.model") != "gemma4") {
    return fail("tokenizer.ggml.model is not gemma4");
  }
  Vocabulary v;
  for (auto token : reader.GetMetadataStringArray("tokenizer.ggml.tokens")) {
    v.tokens.emplace_back(token);
  }
  for (auto merge : reader.GetMetadataStringArray("tokenizer.ggml.merges")) {
    v.merges.emplace_back(merge);
  }
  const auto* types = reader.FindMetadata("tokenizer.ggml.token_type");
  if (types == nullptr) {
    return fail("missing tokenizer.ggml.token_type");
  }
  const auto push_type = [&](auto value) {
    v.types.push_back(value >= 0 && value <= 6 ? static_cast<TokenType>(value)
                                               : TokenType::kUndefined);
  };
  if (const auto* s = std::get_if<std::vector<std::int64_t>>(&types->value)) {
    for (auto value : *s) {
      push_type(value);
    }
  } else if (const auto* u =
                 std::get_if<std::vector<std::uint64_t>>(&types->value)) {
    for (auto value : *u) {
      push_type(static_cast<std::int64_t>(std::min<std::uint64_t>(value, 7)));
    }
  } else {
    return fail("tokenizer.ggml.token_type must be an integer array");
  }
  const auto id = [&](std::string_view key) -> TokenId {
    const auto value = reader.GetMetadataUint32(key);
    return value ? static_cast<TokenId>(*value) : -1;
  };
  v.bos = id("tokenizer.ggml.bos_token_id");
  v.eos = id("tokenizer.ggml.eos_token_id");
  v.pad = id("tokenizer.ggml.padding_token_id");
  v.unknown = id("tokenizer.ggml.unknown_token_id");
  v.add_bos =
      reader.GetMetadataBool("tokenizer.ggml.add_bos_token").value_or(false);
  return Create(std::move(v), error_msg);
}

std::unique_ptr<Tokenizer> Tokenizer::Create(Vocabulary v,
                                             std::string* error_msg) {
  const auto fail = [&](std::string message) -> std::unique_ptr<Tokenizer> {
    if (error_msg != nullptr) {
      *error_msg = std::move(message);
    }
    return nullptr;
  };
  if (v.tokens.empty() || v.tokens.size() > 0x7fffffffU ||
      v.types.size() != v.tokens.size()) {
    return fail("tokenizer vocabulary and token types must match in size");
  }
  const auto n = static_cast<TokenId>(v.tokens.size());
  const auto in_range = [&](TokenId t) { return t >= 0 && t < n; };
  if (!in_range(v.bos) || !in_range(v.eos)) {
    return fail("tokenizer BOS/EOS token ids are out of range");
  }

  std::unique_ptr<Tokenizer> t(new Tokenizer());
  t->tokens_ = std::move(v.tokens);
  t->types_ = std::move(v.types);
  t->bos_ = v.bos;
  t->eos_ = v.eos;
  t->add_bos_ = v.add_bos;
  t->token_to_id_.reserve(t->tokens_.size());
  for (TokenId i = 0; i < n; ++i) {
    // The first occurrence wins, as in llama.cpp's text_to_token map.
    t->token_to_id_.emplace(t->tokens_[i], i);
  }

  // End-of-generation tokens behave as control tokens even when the file
  // types them otherwise.
  const auto add_eog = [&](TokenId token) {
    if (in_range(token) &&
        std::find(t->eog_.begin(), t->eog_.end(), token) == t->eog_.end()) {
      t->eog_.push_back(token);
      t->types_[token] = TokenType::kControl;
    }
  };
  add_eog(v.eos);
  for (auto text : kEndOfGeneration) {
    if (auto token = t->FindToken(text)) {
      add_eog(*token);
    }
  }
  std::sort(t->eog_.begin(), t->eog_.end());

  t->byte_tokens_.fill(-1);
  for (TokenId i = 0; i < n; ++i) {
    const TokenType type = t->types_[i];
    if (type == TokenType::kByte) {
      const auto value = ByteValue(t->tokens_[i]);
      if (!value) {
        return fail("byte token " + std::to_string(i) + " is malformed");
      }
      t->byte_tokens_[*value] = i;
    } else if (type == TokenType::kControl || type == TokenType::kUserDefined ||
               type == TokenType::kUnknown) {
      if (!t->tokens_[i].empty()) {
        t->special_.push_back(i);
      }
    }
  }
  // Longest text first; equal lengths keep id order for determinism.
  std::stable_sort(t->special_.begin(), t->special_.end(),
                   [&](TokenId a, TokenId b) {
                     return t->tokens_[a].size() > t->tokens_[b].size();
                   });

  // A power-of-two table at most half full.
  std::uint32_t bits = 4;
  while ((std::size_t{1} << bits) < v.merges.size() * 2) {
    ++bits;
  }
  t->merge_shift_ = 64 - bits;
  t->merges_.assign(std::size_t{1} << bits, {kEmptyPair, 0, -1});
  const std::size_t mask = t->merges_.size() - 1;
  for (std::size_t rank = 0; rank < v.merges.size(); ++rank) {
    const std::string& merge = v.merges[rank];
    const std::size_t split = merge.find(' ', 1);
    if (split == std::string::npos) {
      return fail("tokenizer merge " + std::to_string(rank) + " is malformed");
    }
    const auto left = t->FindToken(std::string_view(merge).substr(0, split));
    const auto right = t->FindToken(std::string_view(merge).substr(split + 1));
    const auto merged =
        t->FindToken(merge.substr(0, split) + merge.substr(split + 1));
    if (!left || !right || !merged) {
      // Symbols are tracked by id, so every merge must stay in the vocabulary.
      return fail("tokenizer merge " + std::to_string(rank) +
                  " references text outside the vocabulary");
    }
    const std::uint64_t key = PairKey(*left, *right);
    for (std::size_t slot = PairSlot(key, t->merge_shift_);;
         slot = (slot + 1) & mask) {
      MergeSlot& entry = t->merges_[slot];
      if (entry.key == key) {
        break;  // an earlier rank already holds this pair
      }
      if (entry.key == kEmptyPair) {
        entry = {key, static_cast<std::uint32_t>(rank), *merged};
        break;
      }
    }
  }
  for (std::size_t byte = 0; byte < t->char_tokens_.size(); ++byte) {
    const char c = static_cast<char>(byte);
    t->char_tokens_[byte] = t->FindToken(std::string_view(&c, 1)).value_or(-1);
  }
  return t;
}

std::optional<TokenId> Tokenizer::FindToken(
    std::string_view text) const noexcept {
  const auto it = token_to_id_.find(text);
  if (it == token_to_id_.end()) {
    return std::nullopt;
  }
  return it->second;
}

TokenType Tokenizer::Type(TokenId token) const noexcept {
  if (token < 0 || static_cast<std::size_t>(token) >= types_.size()) {
    return TokenType::kUndefined;
  }
  return types_[token];
}

bool Tokenizer::IsEndOfGeneration(TokenId token) const noexcept {
  return std::binary_search(eog_.begin(), eog_.end(), token);
}

Tokenizer::Merge Tokenizer::FindMerge(TokenId left,
                                      TokenId right) const noexcept {
  if (left < 0 || right < 0) {
    return {-1, -1};
  }
  const std::uint64_t key = PairKey(left, right);
  const std::size_t mask = merges_.size() - 1;
  for (std::size_t slot = PairSlot(key, merge_shift_);;
       slot = (slot + 1) & mask) {
    const MergeSlot& entry = merges_[slot];
    if (entry.key == key) {
      return {entry.rank, entry.merged};
    }
    if (entry.key == kEmptyPair) {
      return {-1, -1};
    }
  }
}

std::vector<TokenId> Tokenizer::Encode(std::string_view text, bool add_bos,
                                       bool parse_special) const {
  // Fragments are either raw text or an already matched special token.
  struct Fragment {
    std::string_view text;
    TokenId token{-1};
  };
  std::vector<Fragment> fragments;
  if (!text.empty()) {
    fragments.push_back({text, -1});
  }
  for (const TokenId special : special_) {
    const TokenType type = types_[special];
    if (!parse_special &&
        (type == TokenType::kControl || type == TokenType::kUnknown)) {
      continue;
    }
    const std::string_view needle = tokens_[special];
    std::vector<Fragment> next;
    next.reserve(fragments.size());
    for (const Fragment& f : fragments) {
      if (f.token >= 0) {
        next.push_back(f);
        continue;
      }
      std::size_t start = 0;
      while (true) {
        const std::size_t match = f.text.find(needle, start);
        if (match == std::string_view::npos) {
          break;
        }
        if (match > start) {
          next.push_back({f.text.substr(start, match - start), -1});
        }
        next.push_back({{}, special});
        start = match + needle.size();
      }
      if (start < f.text.size()) {
        next.push_back({f.text.substr(start), -1});
      }
    }
    fragments = std::move(next);
  }

  std::vector<TokenId> out;
  if (add_bos && add_bos_) {
    out.push_back(bos_);
  }
  for (const Fragment& f : fragments) {
    if (f.token >= 0) {
      out.push_back(f.token);
    } else {
      EncodeText(f.text, &out);
    }
  }
  return out;
}

void Tokenizer::EncodeText(std::string_view raw,
                           std::vector<TokenId>* out) const {
  std::string escaped;
  escaped.reserve(raw.size());
  for (char c : raw) {
    if (c == ' ') {
      escaped.append(kEscapedSpace);
    } else {
      escaped.push_back(c);
    }
  }
  const std::string text = Canonicalize(escaped);
  // Pre-split into maximal runs of newlines and of everything else.
  std::size_t start = 0;
  while (start < text.size()) {
    const bool newline = text[start] == '\n';
    std::size_t end = start;
    while (end < text.size() && (text[end] == '\n') == newline) {
      ++end;
    }
    const std::string_view word =
        std::string_view(text).substr(start, end - start);
    start = end;
    if (word.size() < kCachedWordBytes) {
      EncodeWord(word, out);
      continue;
    }
    {
      std::lock_guard lock(word_cache_mutex_);
      if (const auto it = word_index_.find(word); it != word_index_.end()) {
        word_cache_.splice(word_cache_.end(), word_cache_, it->second);
        out->insert(out->end(), it->second->ids.begin(), it->second->ids.end());
        continue;
      }
    }
    CachedWord entry{std::string(word), {}};
    EncodeWord(word, &entry.ids);
    out->insert(out->end(), entry.ids.begin(), entry.ids.end());
    const std::size_t bytes =
        entry.text.size() + entry.ids.size() * sizeof(TokenId);
    if (bytes > kWordCacheBytes) {
      continue;
    }
    std::lock_guard lock(word_cache_mutex_);
    if (word_index_.contains(word)) {
      continue;  // another request encoded it meanwhile
    }
    word_cache_.push_back(std::move(entry));
    word_index_.emplace(word_cache_.back().text, std::prev(word_cache_.end()));
    word_cache_bytes_ += bytes;
    while (word_cache_bytes_ > kWordCacheBytes) {
      const CachedWord& oldest = word_cache_.front();
      word_cache_bytes_ -=
          oldest.text.size() + oldest.ids.size() * sizeof(TokenId);
      word_index_.erase(oldest.text);
      word_cache_.pop_front();
    }
  }
}

void Tokenizer::EncodeWord(std::string_view word,
                           std::vector<TokenId>* out) const {
  if (word.front() == '\n') {
    // A newline run that is a token stays whole instead of being merged.
    if (const auto token = FindToken(word)) {
      out->push_back(*token);
      return;
    }
  }

  struct Symbol {
    std::size_t offset;
    std::size_t length;  ///< Zero once merged into the left neighbour.
    TokenId id;          ///< -1 when the text is outside the vocabulary.
    int prev;
    int next;
  };
  std::vector<Symbol> symbols;
  for (std::size_t offset = 0; offset < word.size();) {
    const std::size_t length =
        std::min(word.size() - offset, Utf8Length(word[offset]));
    const TokenId id =
        length == 1 ? char_tokens_[static_cast<std::uint8_t>(word[offset])]
                    : FindToken(word.substr(offset, length)).value_or(-1);
    const int index = static_cast<int>(symbols.size());
    symbols.push_back({offset, length, id, index - 1,
                       offset + length == word.size() ? -1 : index + 1});
    offset += length;
  }

  struct Bigram {
    std::int64_t rank;
    int left;
    int right;
    TokenId left_id;
    TokenId right_id;
    TokenId merged;
  };
  // Lowest rank first, then the leftmost pair (llama.cpp's comparator).
  const auto later = [](const Bigram& a, const Bigram& b) {
    return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
  };
  std::priority_queue<Bigram, std::vector<Bigram>, decltype(later)> queue(
      later);
  const auto add = [&](int left, int right) {
    if (left < 0 || right < 0) {
      return;
    }
    const Merge merge = FindMerge(symbols[left].id, symbols[right].id);
    if (merge.rank >= 0) {
      queue.push({merge.rank, left, right, symbols[left].id, symbols[right].id,
                  merge.merged});
    }
  };
  for (int i = 1; i < static_cast<int>(symbols.size()); ++i) {
    add(i - 1, i);
  }
  while (!queue.empty()) {
    const Bigram b = queue.top();
    queue.pop();
    Symbol& left = symbols[b.left];
    Symbol& right = symbols[b.right];
    // Stale when either side has changed since the pair was queued.
    if (left.length == 0 || right.length == 0 || left.id != b.left_id ||
        right.id != b.right_id || left.next != b.right) {
      continue;
    }
    left.length += right.length;
    left.id = b.merged;
    right.length = 0;
    left.next = right.next;
    if (right.next >= 0) {
      symbols[right.next].prev = b.left;
    }
    add(left.prev, b.left);
    add(b.left, left.next);
  }

  for (int i = 0; i >= 0 && i < static_cast<int>(symbols.size());
       i = symbols[i].next) {
    const Symbol& s = symbols[i];
    if (s.id >= 0) {
      out->push_back(s.id);
      continue;
    }
    for (std::size_t j = 0; j < s.length; ++j) {
      const TokenId byte =
          byte_tokens_[static_cast<std::uint8_t>(word[s.offset + j])];
      if (byte >= 0) {
        out->push_back(byte);
      }
    }
  }
}

std::string Tokenizer::TokenText(TokenId token, bool special) const {
  const TokenType type = Type(token);
  switch (type) {
    case TokenType::kControl:
    case TokenType::kUnknown:
      return special ? tokens_[token] : std::string();
    case TokenType::kUserDefined:
      return tokens_[token];
    case TokenType::kByte:
      return std::string(1, static_cast<char>(*ByteValue(tokens_[token])));
    case TokenType::kNormal: {
      const std::string& text = tokens_[token];
      std::string out;
      out.reserve(text.size());
      for (std::size_t i = 0; i < text.size();) {
        if (text.compare(i, kEscapedSpace.size(), kEscapedSpace) == 0) {
          out.push_back(' ');
          i += kEscapedSpace.size();
        } else {
          out.push_back(text[i++]);
        }
      }
      return out;
    }
    default:
      return {};
  }
}

std::string Tokenizer::Decode(std::span<const TokenId> tokens,
                              bool special) const {
  std::string out;
  for (TokenId token : tokens) {
    out += TokenText(token, special);
  }
  return out;
}

}  // namespace gufo::models::gemma4

#include "src/models/nemotron35_bolt/tokenizer.hpp"

#include <unicode/uchar.h>

#include <algorithm>
#include <limits>
#include <queue>
#include <utility>
#include <variant>

namespace gufo::models::nemotron35_bolt {
namespace {

// GGUF token types (llama.cpp's llama_token_type).
constexpr std::int64_t kTypeNormal = 1;
constexpr std::int64_t kTypeControl = 3;
constexpr std::int64_t kTypeUserDefined = 4;

// Codepoint classes of the split regex.
enum : std::uint8_t {
  kUpper = 1U << 0U,   ///< [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
  kLower = 1U << 1U,   ///< [\p{Ll}\p{Lm}\p{Lo}\p{M}]
  kLetter = 1U << 2U,  ///< \p{L}
  kNumber = 1U << 3U,  ///< \p{N}
  kSpace = 1U << 4U,   ///< \s (White_Space)
  kCrLf = 1U << 5U,    ///< [\r\n]
};

struct Codepoint {
  std::size_t offset;
  std::uint8_t length;
  std::uint8_t flags;
  char32_t value;
};

std::uint8_t Classify(char32_t c) {
  const auto mask = U_GET_GC_MASK(static_cast<UChar32>(c));
  std::uint8_t flags = 0;
  if ((mask & (U_GC_LU_MASK | U_GC_LT_MASK | U_GC_LM_MASK | U_GC_LO_MASK |
               U_GC_M_MASK)) != 0)
    flags |= kUpper;
  if ((mask & (U_GC_LL_MASK | U_GC_LM_MASK | U_GC_LO_MASK | U_GC_M_MASK)) != 0)
    flags |= kLower;
  if ((mask & U_GC_L_MASK) != 0)
    flags |= kLetter;
  if ((mask & U_GC_N_MASK) != 0)
    flags |= kNumber;
  if (u_isUWhiteSpace(static_cast<UChar32>(c)) != 0)
    flags |= kSpace;
  if (c == '\r' || c == '\n')
    flags |= kCrLf;
  return flags;
}

/// Decodes UTF-8; an invalid byte becomes one U+FFFD codepoint of length 1.
std::vector<Codepoint> Decode(std::string_view text) {
  std::vector<Codepoint> out;
  out.reserve(text.size());
  std::size_t i = 0;
  while (i < text.size()) {
    const auto b0 = static_cast<std::uint8_t>(text[i]);
    std::size_t length = 1;
    char32_t value = b0;
    if (b0 >= 0x80U) {
      length = (b0 & 0xE0U) == 0xC0U   ? 2
               : (b0 & 0xF0U) == 0xE0U ? 3
               : (b0 & 0xF8U) == 0xF0U ? 4
                                       : 0;
      bool valid = length != 0 && i + length <= text.size();
      if (valid) {
        value = b0 & (0x7FU >> length);
        for (std::size_t k = 1; k < length; ++k) {
          const auto b = static_cast<std::uint8_t>(text[i + k]);
          valid = valid && (b & 0xC0U) == 0x80U;
          value = (value << 6U) | (b & 0x3FU);
        }
      }
      if (!valid) {
        length = 1;
        value = 0xFFFD;
      }
    }
    out.push_back(
        {i, static_cast<std::uint8_t>(length), Classify(value), value});
    i += length;
  }
  return out;
}

using Match = std::optional<std::size_t>;

/// End index (in codepoints) of the piece starting at codepoint `i`. Each
/// alternative reproduces the backtracking order of a greedy regex engine.
std::size_t PieceEnd(const std::vector<Codepoint>& c, std::size_t i) {
  const std::size_t n = c.size();
  auto has = [&](std::size_t k, std::uint8_t flag) {
    return k < n && (c[k].flags & flag) != 0;
  };
  // [^\r\n\p{L}\p{N}]
  auto prefix = [&](std::size_t k) {
    return k < n && (c[k].flags & (kCrLf | kLetter | kNumber)) == 0;
  };

  // Alternative 1: U*L+. U* first takes its longest run, then gives back
  // codepoints until L+ can start; L+ then runs as far as it can.
  auto upper_then_lower = [&](std::size_t j) -> Match {
    std::size_t k = j;
    while (has(k, kUpper))
      ++k;
    for (std::size_t p = k;; --p) {
      if (has(p, kLower)) {
        std::size_t e = p;
        while (has(e, kLower))
          ++e;
        return e;
      }
      if (p == j)
        return std::nullopt;
    }
  };
  if (prefix(i)) {
    if (const Match e = upper_then_lower(i + 1))
      return *e;
  }
  if (const Match e = upper_then_lower(i))
    return *e;

  // Alternative 2: U+L*; L* cannot fail, so no backtracking.
  auto upper_run = [&](std::size_t j) -> Match {
    if (!has(j, kUpper))
      return std::nullopt;
    std::size_t k = j;
    while (has(k, kUpper))
      ++k;
    while (has(k, kLower))
      ++k;
    return k;
  };
  if (prefix(i)) {
    if (const Match e = upper_run(i + 1))
      return *e;
  }
  if (const Match e = upper_run(i))
    return *e;

  // Alternative 3: one number codepoint.
  if (has(i, kNumber))
    return i + 1;

  // Alternative 4: optional space, punctuation run, then [\r\n/]*.
  auto punctuation = [&](std::size_t j) -> Match {
    auto other = [&](std::size_t k) {
      return k < n && (c[k].flags & (kSpace | kLetter | kNumber)) == 0;
    };
    if (!other(j))
      return std::nullopt;
    std::size_t k = j;
    while (other(k))
      ++k;
    while (k < n &&
           (c[k].value == '\r' || c[k].value == '\n' || c[k].value == '/'))
      ++k;
    return k;
  };
  if (c[i].value == ' ') {
    if (const Match e = punctuation(i + 1))
      return *e;
  }
  if (const Match e = punctuation(i))
    return *e;

  if (has(i, kSpace)) {
    std::size_t k = i;
    while (has(k, kSpace))
      ++k;
    // Alternative 5: \s*[\r\n]+ ends after the run's last CR/LF.
    for (std::size_t p = k; p-- > i;) {
      if (has(p, kCrLf))
        return p + 1;
    }
    // Alternative 6: \s+(?!\S) leaves the last space before non-space to the
    // next piece.
    if (k == n)
      return k;
    if (k - 1 > i)
      return k - 1;
    // Alternative 7: \s+.
    return k;
  }
  return i + 1;  // Unreachable: every codepoint falls in some alternative.
}

/// GPT-2 byte-to-codepoint map used by byte-level BPE vocabularies.
std::array<char32_t, 256> ByteLevelCodepoints() {
  std::array<char32_t, 256> table{};
  char32_t next = 256;
  for (int b = 0; b < 256; ++b) {
    const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                           (b >= 0xAE && b <= 0xFF);
    table[b] = printable ? static_cast<char32_t>(b) : next++;
  }
  return table;
}

/// Raw bytes of a byte-level vocabulary string; nullopt when a codepoint is
/// outside the byte alphabet.
std::optional<std::string> UnescapeByteLevel(std::string_view token) {
  static const auto to_byte = [] {
    std::unordered_map<char32_t, std::uint8_t> map;
    const auto table = ByteLevelCodepoints();
    for (int b = 0; b < 256; ++b)
      map[table[b]] = static_cast<std::uint8_t>(b);
    return map;
  }();
  std::string out;
  out.reserve(token.size());
  for (const Codepoint& cp : Decode(token)) {
    const auto it = to_byte.find(cp.value);
    if (cp.value == 0xFFFD || it == to_byte.end())
      return std::nullopt;
    out.push_back(static_cast<char>(it->second));
  }
  return out;
}

std::uint64_t PairKey(TokenId a, TokenId b) {
  return (static_cast<std::uint64_t>(a) << 32U) | b;
}

std::vector<std::int64_t> IntArray(const core::GgufReader& reader,
                                   std::string_view key) {
  std::vector<std::int64_t> out;
  const auto* meta = reader.FindMetadata(key);
  if (meta == nullptr)
    return out;
  if (const auto* s = std::get_if<std::vector<std::int64_t>>(&meta->value)) {
    out = *s;
  } else if (const auto* u =
                 std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
    for (auto v : *u)
      out.push_back(static_cast<std::int64_t>(std::min<std::uint64_t>(
          v, std::numeric_limits<std::int64_t>::max())));
  }
  return out;
}

}  // namespace

std::vector<std::string_view> PreTokenize(std::string_view text) {
  const auto c = Decode(text);
  std::vector<std::string_view> pieces;
  std::size_t i = 0;
  while (i < c.size()) {
    const std::size_t e = PieceEnd(c, i);
    const std::size_t begin = c[i].offset;
    const std::size_t end = e < c.size() ? c[e].offset : text.size();
    pieces.push_back(text.substr(begin, end - begin));
    i = e;
  }
  return pieces;
}

std::unique_ptr<Tokenizer> Tokenizer::FromGguf(const core::GgufReader& reader,
                                               std::string* error_msg) {
  auto fail = [&](std::string message) -> std::unique_ptr<Tokenizer> {
    if (error_msg != nullptr)
      *error_msg = std::move(message);
    return nullptr;
  };
  if (reader.GetMetadataString("tokenizer.ggml.model") != "gpt2" ||
      reader.GetMetadataString("tokenizer.ggml.pre") != "pixtral") {
    return fail("tokenizer is not the byte-level BPE with the pixtral split");
  }
  if (reader.GetMetadataBool("tokenizer.ggml.add_bos_token").value_or(false))
    return fail("tokenizer requests a BOS token; the chat format has none");
  const auto tokens = reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  const auto merges = reader.GetMetadataStringArray("tokenizer.ggml.merges");
  const auto types = IntArray(reader, "tokenizer.ggml.token_type");
  if (tokens.empty() || merges.empty() || types.size() != tokens.size())
    return fail("tokenizer vocabulary, merges or token types are missing");

  std::unique_ptr<Tokenizer> t(new Tokenizer());
  t->bytes_.resize(tokens.size());
  t->control_.assign(tokens.size(), 0);
  t->by_bytes_.reserve(tokens.size());
  for (std::size_t id = 0; id < tokens.size(); ++id) {
    const auto token_id = static_cast<TokenId>(id);
    if (types[id] == kTypeControl || types[id] == kTypeUserDefined) {
      t->bytes_[id] = std::string(tokens[id]);
      t->control_[id] = types[id] == kTypeControl ? 1 : 0;
      t->added_.emplace(t->bytes_[id],
                        Added{token_id, types[id] == kTypeControl});
      continue;
    }
    if (types[id] != kTypeNormal)
      return fail("unsupported token type at id " + std::to_string(id));
    auto bytes = UnescapeByteLevel(tokens[id]);
    if (!bytes || bytes->empty())
      return fail("token " + std::to_string(id) + " is not byte-level");
    t->bytes_[id] = *bytes;
    t->by_bytes_.emplace(std::move(*bytes), token_id);
  }
  for (int b = 0; b < 256; ++b) {
    const auto it = t->by_bytes_.find(std::string(1, static_cast<char>(b)));
    if (it == t->by_bytes_.end())
      return fail("vocabulary lacks a token for byte " + std::to_string(b));
    t->byte_token_[b] = it->second;
  }

  t->merges_.reserve(merges.size());
  for (std::size_t rank = 0; rank < merges.size(); ++rank) {
    const auto merge = merges[rank];
    const auto space = merge.find(' ');
    if (space == std::string_view::npos)
      return fail("malformed merge " + std::to_string(rank));
    const auto left = UnescapeByteLevel(merge.substr(0, space));
    const auto right = UnescapeByteLevel(merge.substr(space + 1));
    if (!left || !right)
      return fail("merge " + std::to_string(rank) + " is not byte-level");
    const auto a = t->by_bytes_.find(*left);
    const auto b = t->by_bytes_.find(*right);
    const auto merged = t->by_bytes_.find(*left + *right);
    if (a == t->by_bytes_.end() || b == t->by_bytes_.end() ||
        merged == t->by_bytes_.end())
      return fail("merge " + std::to_string(rank) + " names unknown tokens");
    t->merges_.try_emplace(
        PairKey(a->second, b->second),
        Merge{static_cast<std::uint32_t>(rank), merged->second});
  }

  for (const auto& [text, added] : t->added_) {
    if (std::find(t->added_lengths_.begin(), t->added_lengths_.end(),
                  text.size()) == t->added_lengths_.end())
      t->added_lengths_.push_back(text.size());
    t->added_first_byte_[static_cast<std::uint8_t>(text.front())] = true;
  }
  std::sort(t->added_lengths_.rbegin(), t->added_lengths_.rend());

  const auto eos = reader.GetMetadataUint32("tokenizer.ggml.eos_token_id");
  if (!eos || *eos >= tokens.size())
    return fail("tokenizer has no valid EOS token");
  t->eos_ = *eos;
  t->end_of_sequence_ = t->AddedToken("</s>").value_or(kInvalidToken);
  return t;
}

void Tokenizer::EncodePiece(std::string_view piece,
                            std::vector<TokenId>& out) const {
  // ignore_merges: a piece that is itself a vocabulary entry is emitted whole.
  if (const auto whole = by_bytes_.find(std::string(piece));
      whole != by_bytes_.end()) {
    out.push_back(whole->second);
    return;
  }
  // Merge the lowest-ranked adjacent pair repeatedly, leftmost first on a
  // tie, with the symbols in a linked list and candidate pairs in a heap.
  constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
  const std::size_t n = piece.size();
  std::vector<TokenId> symbol(n);
  std::vector<std::size_t> prev(n);
  std::vector<std::size_t> next(n);
  for (std::size_t i = 0; i < n; ++i) {
    symbol[i] = byte_token_[static_cast<std::uint8_t>(piece[i])];
    prev[i] = i == 0 ? kNone : i - 1;
    next[i] = i + 1 < n ? i + 1 : kNone;
  }
  struct Candidate {
    std::uint32_t rank;
    std::size_t left;
    TokenId a;
    TokenId b;
  };
  auto later = [](const Candidate& x, const Candidate& y) {
    return x.rank != y.rank ? x.rank > y.rank : x.left > y.left;
  };
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(later)> heap(
      later);
  auto offer = [&](std::size_t left) {
    const std::size_t right = next[left];
    if (right == kNone)
      return;
    const auto it = merges_.find(PairKey(symbol[left], symbol[right]));
    if (it != merges_.end())
      heap.push({it->second.rank, left, symbol[left], symbol[right]});
  };
  for (std::size_t i = 0; i + 1 < n; ++i)
    offer(i);
  while (!heap.empty()) {
    const Candidate best = heap.top();
    heap.pop();
    const std::size_t right = next[best.left];
    if (symbol[best.left] != best.a || right == kNone ||
        symbol[right] != best.b)
      continue;  // A neighbouring merge already consumed this pair.
    symbol[best.left] = merges_.at(PairKey(best.a, best.b)).token;
    next[best.left] = next[right];
    if (next[right] != kNone)
      prev[next[right]] = best.left;
    symbol[right] = kInvalidToken;
    if (prev[best.left] != kNone)
      offer(prev[best.left]);
    offer(best.left);
  }
  for (std::size_t i = 0; i != kNone; i = next[i])
    out.push_back(symbol[i]);
}

void Tokenizer::EncodeText(std::string_view text,
                           std::vector<TokenId>& out) const {
  for (const std::string_view piece : PreTokenize(text))
    EncodePiece(piece, out);
}

std::vector<TokenId> Tokenizer::Encode(std::string_view text,
                                       bool parse_control) const {
  std::vector<TokenId> out;
  out.reserve(text.size() / 3 + 1);
  // Added tokens split the text first: leftmost match, longest at a position.
  std::size_t start = 0;
  for (std::size_t pos = 0; pos < text.size(); ++pos) {
    if (!added_first_byte_[static_cast<std::uint8_t>(text[pos])])
      continue;
    for (const std::size_t length : added_lengths_) {
      if (length > text.size() - pos)
        continue;
      const auto it = added_.find(std::string(text.substr(pos, length)));
      if (it == added_.end() || (it->second.control && !parse_control))
        continue;
      EncodeText(text.substr(start, pos - start), out);
      out.push_back(it->second.id);
      start = pos + length;
      pos = start - 1;
      break;
    }
  }
  EncodeText(text.substr(start), out);
  return out;
}

std::string Tokenizer::Decode(std::span<const TokenId> tokens) const {
  std::string out;
  for (const TokenId id : tokens)
    out.append(TokenBytes(id));
  return out;
}

std::string_view Tokenizer::TokenBytes(TokenId id) const noexcept {
  return id < bytes_.size() ? std::string_view(bytes_[id]) : std::string_view();
}

std::optional<TokenId> Tokenizer::AddedToken(
    std::string_view text) const noexcept {
  const auto it = added_.find(std::string(text));
  if (it == added_.end())
    return std::nullopt;
  return it->second.id;
}

bool Tokenizer::SpellsControl(std::string_view text) const {
  for (std::size_t pos = 0; pos < text.size(); ++pos) {
    if (!added_first_byte_[static_cast<std::uint8_t>(text[pos])])
      continue;
    for (const std::size_t length : added_lengths_) {
      if (length > text.size() - pos)
        continue;
      const auto it = added_.find(std::string(text.substr(pos, length)));
      if (it != added_.end() && it->second.control)
        return true;
    }
  }
  return false;
}

bool Tokenizer::IsControl(TokenId id) const noexcept {
  return id < control_.size() && control_[id] != 0;
}

bool Tokenizer::IsStop(TokenId id) const noexcept {
  return id != kInvalidToken && (id == eos_ || id == end_of_sequence_);
}

}  // namespace gufo::models::nemotron35_bolt

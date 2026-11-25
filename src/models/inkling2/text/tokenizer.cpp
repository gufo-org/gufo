#include "src/models/inkling2/text/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <queue>

#include "src/models/inkling2/text/pretokenizer.hpp"

namespace inkling2::text {

namespace {

// GPT-2 byte <-> unicode mapping.
struct ByteMap {
  std::array<char32_t, 256> to_cp{};
  std::unordered_map<char32_t, std::uint8_t> from_cp;
  ByteMap() {
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      const bool direct = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                          (b >= 0xAE && b <= 0xFF);
      to_cp[b] =
          direct ? static_cast<char32_t>(b) : static_cast<char32_t>(256 + n++);
      from_cp[to_cp[b]] = static_cast<std::uint8_t>(b);
    }
  }
};
const ByteMap& Bytes() {
  static const ByteMap m;
  return m;
}

// Decodes a GPT-2 byte-unicode string to raw bytes; false on a foreign
// character.
bool UnmapBytes(std::string_view s, std::string* out) {
  out->clear();
  const auto& m = Bytes();
  for (std::size_t i = 0; i < s.size();) {
    char32_t cp;
    const auto c = static_cast<unsigned char>(s[i]);
    std::size_t len;
    if (c < 0x80) {
      cp = c;
      len = 1;
    } else if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
      cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F);
      len = 2;
    } else if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
      cp = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
      len = 3;
    } else
      return false;
    const auto it = m.from_cp.find(cp);
    if (it == m.from_cp.end())
      return false;
    out->push_back(static_cast<char>(it->second));
    i += len;
  }
  return true;
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::Create(const TokenizerData& d,
                                             std::string* error) {
  const auto fail = [&](std::string m) -> std::unique_ptr<Tokenizer> {
    if (error != nullptr)
      *error = std::move(m);
    return nullptr;
  };
  if (d.model != "gpt2")
    return fail("tokenizer.ggml.model is '" + d.model + "', expected gpt2");
  if (d.tokens.empty())
    return fail("empty vocabulary");
  if (d.token_types.size() != d.tokens.size())
    return fail("token_type size differs from tokens");
  std::unique_ptr<Tokenizer> t(new Tokenizer());
  const std::size_t n = d.tokens.size();
  t->bytes_.resize(n);
  t->is_special_.assign(n, 0);
  t->token_to_id_.reserve(n * 2);
  std::fill(std::begin(t->byte_token_), std::end(t->byte_token_),
            kInvalidToken);
  for (std::size_t i = 0; i < n; ++i) {
    const auto type = d.token_types[i];
    if (type == 3 || type == 4) {  // control / user defined: literal text
      t->bytes_[i] = d.tokens[i];
      t->is_special_[i] = 1;
      t->specials_.emplace_back(d.tokens[i], static_cast<TokenId>(i));
      t->special_first_byte_[static_cast<unsigned char>(
          d.tokens[i].empty() ? 0 : d.tokens[i][0])] = true;
      continue;
    }
    if (!UnmapBytes(d.tokens[i], &t->bytes_[i]))
      return fail("token " + std::to_string(i) +
                  " is not in byte-unicode spelling");
    t->token_to_id_.emplace(t->bytes_[i], static_cast<TokenId>(i));
    if (t->bytes_[i].size() == 1)
      t->byte_token_[static_cast<unsigned char>(t->bytes_[i][0])] =
          static_cast<TokenId>(i);
  }
  for (int b = 0; b < 256; ++b)
    if (t->byte_token_[b] == kInvalidToken)
      return fail("vocabulary lacks single-byte token " + std::to_string(b));
  std::stable_sort(t->specials_.begin(), t->specials_.end(),
                   [](const auto& a, const auto& b) {
                     return a.first.size() > b.first.size();
                   });

  t->merges_.reserve(d.merges.size() * 2);
  std::string l, r;
  for (std::size_t i = 0; i < d.merges.size(); ++i) {
    const auto& m = d.merges[i];
    const auto sp = m.find(
        ' ', 1);  // a token may not start with a space; 'Ġ' stands for it
    if (sp == std::string::npos)
      return fail("malformed merge " + std::to_string(i));
    if (!UnmapBytes(std::string_view(m).substr(0, sp), &l) ||
        !UnmapBytes(std::string_view(m).substr(sp + 1), &r))
      return fail("malformed merge " + std::to_string(i));
    const auto a = t->token_to_id_.find(l);
    const auto b = t->token_to_id_.find(r);
    const auto c = t->token_to_id_.find(l + r);
    if (a == t->token_to_id_.end() || b == t->token_to_id_.end() ||
        c == t->token_to_id_.end())
      continue;
    const std::uint64_t key =
        (static_cast<std::uint64_t>(a->second) << 32) | b->second;
    t->merges_.emplace(key, Merge{static_cast<std::uint32_t>(i),
                                  c->second});  // first (lowest rank) wins
  }
  if (d.eos_id)
    t->eos_ = *d.eos_id;
  else if (d.eot_id)
    t->eos_ = *d.eot_id;
  if (d.pad_id)
    t->pad_ = *d.pad_id;
  return t;
}

std::unique_ptr<Tokenizer> Tokenizer::CreateFromGguf(const std::string& path,
                                                     std::string* error) {
  TokenizerData data;
  if (!ReadTokenizerDataFromGguf(path, &data, error))
    return nullptr;
  return Create(data, error);
}

std::unique_ptr<Tokenizer> Tokenizer::CreateFromGguf(
    const gufo::core::GgufReader& reader, std::string* error) {
  TokenizerData data;
  if (!ReadTokenizerData(reader, &data, error))
    return nullptr;
  return Create(data, error);
}

std::optional<TokenId> Tokenizer::FindSpecial(std::string_view text) const {
  for (const auto& [s, id] : specials_)
    if (s == text)
      return id;
  return std::nullopt;
}

std::optional<TokenId> Tokenizer::FindToken(std::string_view bytes) const {
  const auto it = token_to_id_.find(std::string(bytes));
  if (it != token_to_id_.end())
    return it->second;
  return FindSpecial(bytes);
}

void Tokenizer::BpePiece(std::string_view piece,
                         std::vector<TokenId>* out) const {
  if (piece.size() == 1) {
    out->push_back(byte_token_[static_cast<unsigned char>(piece[0])]);
    return;
  }
  // ignore_merges: a piece that is a vocabulary entry is that token.
  {
    thread_local std::string key;
    key.assign(piece);
    const auto it = token_to_id_.find(key);
    if (it != token_to_id_.end()) {
      out->push_back(it->second);
      return;
    }
  }
  const std::size_t n = piece.size();
  thread_local std::vector<TokenId> sym;
  thread_local std::vector<std::int32_t> prev, next;
  sym.resize(n);
  prev.resize(n);
  next.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    sym[i] = byte_token_[static_cast<unsigned char>(piece[i])];
    prev[i] = static_cast<std::int32_t>(i) - 1;
    next[i] = i + 1 < n ? static_cast<std::int32_t>(i + 1) : -1;
  }
  struct Cand {
    std::uint32_t rank;
    std::int32_t left;
    TokenId lsym, rsym;
    bool operator>(const Cand& o) const {
      return rank != o.rank ? rank > o.rank : left > o.left;
    }
  };
  std::priority_queue<Cand, std::vector<Cand>, std::greater<>> heap;
  const auto push = [&](std::int32_t left) {
    if (left < 0)
      return;
    const auto right = next[left];
    if (right < 0)
      return;
    const auto it = merges_.find((static_cast<std::uint64_t>(sym[left]) << 32) |
                                 sym[right]);
    if (it != merges_.end())
      heap.push({it->second.rank, left, sym[left], sym[right]});
  };
  for (std::size_t i = 0; i + 1 < n; ++i)
    push(static_cast<std::int32_t>(i));
  thread_local std::vector<char> dead;
  dead.assign(n, 0);
  while (!heap.empty()) {
    const Cand c = heap.top();
    heap.pop();
    if (dead[c.left])
      continue;
    const auto right = next[c.left];
    if (right < 0 || sym[c.left] != c.lsym || sym[right] != c.rsym)
      continue;  // stale
    const auto it =
        merges_.find((static_cast<std::uint64_t>(c.lsym) << 32) | c.rsym);
    sym[c.left] = it->second.result;
    dead[right] = 1;
    next[c.left] = next[right];
    if (next[right] >= 0)
      prev[next[right]] = c.left;
    if (prev[c.left] >= 0)
      push(prev[c.left]);
    push(c.left);
  }
  for (std::int32_t i = 0; i >= 0; i = next[i])
    out->push_back(sym[i]);
}

void Tokenizer::EncodeOrdinary(std::string_view text,
                               std::vector<TokenId>* out) const {
  thread_local std::vector<std::pair<std::uint32_t, std::uint32_t>> pieces;
  pieces.clear();
  SplitPieces(text, &pieces);
  for (const auto& [b, e] : pieces)
    BpePiece(text.substr(b, e - b), out);
}

void Tokenizer::Encode(std::string_view text, bool parse_special,
                       std::vector<TokenId>* out) const {
  if (!parse_special || specials_.empty()) {
    EncodeOrdinary(text, out);
    return;
  }
  std::size_t seg = 0, i = 0;
  while (i < text.size()) {
    if (!special_first_byte_[static_cast<unsigned char>(text[i])]) {
      ++i;
      continue;
    }
    const std::pair<std::string, TokenId>* hit = nullptr;
    for (const auto& s : specials_) {  // longest first: leftmost-longest
      if (text.compare(i, s.first.size(), s.first) == 0) {
        hit = &s;
        break;
      }
    }
    if (hit == nullptr) {
      ++i;
      continue;
    }
    if (i > seg)
      EncodeOrdinary(text.substr(seg, i - seg), out);
    out->push_back(hit->second);
    i += hit->first.size();
    seg = i;
  }
  if (seg < text.size())
    EncodeOrdinary(text.substr(seg), out);
}

std::vector<TokenId> Tokenizer::Encode(std::string_view text,
                                       bool parse_special) const {
  std::vector<TokenId> out;
  out.reserve(text.size() / 3 + 4);
  Encode(text, parse_special, &out);
  return out;
}

std::string_view Tokenizer::TokenBytes(TokenId id) const {
  return id < bytes_.size() ? std::string_view(bytes_[id]) : std::string_view();
}

void Tokenizer::DecodeTo(std::span<const TokenId> ids, std::string* out) const {
  for (const auto id : ids)
    out->append(TokenBytes(id));
}

std::string Tokenizer::Decode(std::span<const TokenId> ids) const {
  std::string out;
  DecodeTo(ids, &out);
  return out;
}

std::size_t IncompleteUtf8Tail(std::string_view s) {
  // Look back at most 3 bytes for a lead byte whose sequence is not complete.
  const std::size_t n = s.size();
  for (std::size_t back = 1; back <= 3 && back <= n; ++back) {
    const auto c = static_cast<unsigned char>(s[n - back]);
    if ((c & 0xC0) == 0x80)
      continue;  // continuation
    std::size_t need = 0;
    if (c >= 0xC2 && c <= 0xDF)
      need = 2;
    else if (c >= 0xE0 && c <= 0xEF)
      need = 3;
    else if (c >= 0xF0 && c <= 0xF4)
      need = 4;
    return (need > back) ? back : 0;
  }
  return 0;
}

std::string StreamDecoder::Push(TokenId id) {
  pending_.append(tok_->TokenBytes(id));
  const std::size_t hold = IncompleteUtf8Tail(pending_);
  std::string out = pending_.substr(0, pending_.size() - hold);
  pending_.erase(0, pending_.size() - hold);
  return out;
}

std::string StreamDecoder::Flush() {
  std::string out;
  out.swap(pending_);
  return out;
}

}  // namespace inkling2::text

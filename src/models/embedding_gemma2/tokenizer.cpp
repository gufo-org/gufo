#include "tokenizer.hpp"

#include <algorithm>
#include <string>

namespace gufo::models::embedding_gemma2 {
namespace {

// UTF-8 sequence length of the character at text[pos], or 1 for invalid
// bytes so that they fall back to a byte token each.
std::size_t Utf8CharLength(std::string_view text, std::size_t pos) {
  const auto lead = static_cast<unsigned char>(text[pos]);
  std::size_t length = 1;
  std::uint32_t minimum = 0;
  std::uint32_t code = lead;
  if (lead < 0x80) {
    return 1;
  }
  if (lead >= 0xC2 && lead <= 0xDF) {
    length = 2;
    code = lead & 0x1FU;
    minimum = 0x80;
  } else if (lead >= 0xE0 && lead <= 0xEF) {
    length = 3;
    code = lead & 0x0FU;
    minimum = 0x800;
  } else if (lead >= 0xF0 && lead <= 0xF4) {
    length = 4;
    code = lead & 0x07U;
    minimum = 0x10000;
  } else {
    return 1;
  }
  if (pos + length > text.size()) {
    return 1;
  }
  for (std::size_t i = 1; i < length; ++i) {
    const auto next = static_cast<unsigned char>(text[pos + i]);
    if ((next & 0xC0U) != 0x80U) {
      return 1;
    }
    code = (code << 6U) | (next & 0x3FU);
  }
  if (code < minimum || code > 0x10FFFF || (code >= 0xD800 && code <= 0xDFFF)) {
    return 1;
  }
  return length;
}

struct Symbol {
  std::uint32_t id;
  std::int32_t prev;
  std::int32_t next;
  bool alive;
};

struct PendingMerge {
  std::uint32_t rank;
  std::uint32_t position;
  std::uint32_t new_id;
  // The pair this entry was queued for. Comparing it on pop detects stale
  // entries cheaply.
  std::uint32_t left_id;
  std::uint32_t right_id;
};

// Heap order: smallest rank first, then leftmost position.
bool MergeAfter(const PendingMerge& a, const PendingMerge& b) {
  return a.rank != b.rank ? a.rank > b.rank : a.position > b.position;
}

}  // namespace

std::optional<std::uint32_t> Tokenizer::TokenId(std::string_view token) const {
  const auto it = ids_.find(token);
  if (it == ids_.end()) {
    return std::nullopt;
  }
  return it->second;
}

const Tokenizer::Merge* Tokenizer::FindMerge(std::uint32_t left,
                                             std::uint32_t right) const {
  const auto it = merges_.find(PairKey(left, right));
  return it != merges_.end() ? &it->second : nullptr;
}

std::size_t Tokenizer::MatchAddedToken(std::string_view text, std::size_t pos,
                                       std::uint32_t* id) const {
  for (const AddedToken& token : added_) {
    if (text.compare(pos, token.content.size(), token.content) == 0) {
      *id = token.id;
      return token.content.size();
    }
  }
  return 0;
}

void Tokenizer::EncodeSegment(std::string_view text,
                              std::vector<std::uint32_t>* out) const {
  if (text.empty()) {
    return;
  }
  // Normalizer: every ASCII space becomes U+2581. The pre-tokenizer then
  // finds no space to split on, so the segment is a single BPE word.
  std::string word;
  word.reserve(text.size() + text.size() / 4);
  for (const char c : text) {
    if (c == ' ') {
      word.append(kSpaceSymbol);
    } else {
      word.push_back(c);
    }
  }

  std::vector<Symbol> symbols;
  symbols.reserve(word.size());
  auto add_symbol = [&symbols](std::uint32_t id) {
    const auto index = static_cast<std::int32_t>(symbols.size());
    if (index > 0) {
      symbols.back().next = index;
    }
    symbols.push_back({id, index - 1, -1, true});
  };
  for (std::size_t i = 0; i < word.size();) {
    const std::size_t length = Utf8CharLength(word, i);
    const std::string_view piece = std::string_view(word).substr(i, length);
    i += length;
    const auto it = ids_.find(piece);
    if (it != ids_.end()) {
      add_symbol(it->second);
      continue;
    }
    for (const char byte : piece) {
      add_symbol(byte_tokens_[static_cast<unsigned char>(byte)]);
    }
  }

  // Linked-list BPE. A heap holds candidate merges ordered by rank, then
  // position. Popping detects stale entries.
  std::vector<PendingMerge> heap;
  heap.reserve(symbols.size());
  for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
    if (const Merge* merge = FindMerge(symbols[i].id, symbols[i + 1].id)) {
      heap.push_back({merge->rank, static_cast<std::uint32_t>(i), merge->new_id,
                      symbols[i].id, symbols[i + 1].id});
    }
  }
  std::make_heap(heap.begin(), heap.end(), MergeAfter);
  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), MergeAfter);
    const PendingMerge top = heap.back();
    heap.pop_back();
    Symbol& left = symbols[top.position];
    if (!left.alive || left.next < 0) {
      continue;
    }
    const auto right_index = static_cast<std::size_t>(left.next);
    const Symbol right = symbols[right_index];
    if (left.id != top.left_id || right.id != top.right_id) {
      // The neighbours changed. The entry is still usable only if the new
      // pair merges into the same token.
      const Merge* current = FindMerge(left.id, right.id);
      if (current == nullptr || current->new_id != top.new_id) {
        continue;
      }
    }
    left.id = top.new_id;
    left.next = right.next;
    symbols[right_index].alive = false;
    if (right.next >= 0) {
      symbols[static_cast<std::size_t>(right.next)].prev =
          static_cast<std::int32_t>(top.position);
    }
    if (left.prev >= 0) {
      const Symbol& before = symbols[static_cast<std::size_t>(left.prev)];
      if (const Merge* merge = FindMerge(before.id, left.id)) {
        heap.push_back({merge->rank, static_cast<std::uint32_t>(left.prev),
                        merge->new_id, before.id, left.id});
        std::push_heap(heap.begin(), heap.end(), MergeAfter);
      }
    }
    if (left.next >= 0) {
      const Symbol& after = symbols[static_cast<std::size_t>(left.next)];
      if (const Merge* merge = FindMerge(left.id, after.id)) {
        heap.push_back(
            {merge->rank, top.position, merge->new_id, left.id, after.id});
        std::push_heap(heap.begin(), heap.end(), MergeAfter);
      }
    }
  }
  for (const Symbol& symbol : symbols) {
    if (symbol.alive) {
      out->push_back(symbol.id);
    }
  }
}

std::vector<std::uint32_t> Tokenizer::Encode(std::string_view text,
                                             bool add_special_tokens) const {
  std::vector<std::uint32_t> ids;
  ids.reserve(text.size() / 3 + 4);
  if (add_special_tokens) {
    ids = prefix_ids_;
  }
  // Added (special) tokens match on the raw text, leftmost and longest first.
  // The text between them is encoded as BPE segments.
  std::size_t segment_start = 0;
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::uint32_t id = 0;
    const std::size_t length =
        added_first_byte_[static_cast<unsigned char>(text[pos])]
            ? MatchAddedToken(text, pos, &id)
            : 0;
    if (length == 0) {
      ++pos;
      continue;
    }
    EncodeSegment(text.substr(segment_start, pos - segment_start), &ids);
    ids.push_back(id);
    pos += length;
    segment_start = pos;
  }
  EncodeSegment(text.substr(segment_start), &ids);
  if (add_special_tokens) {
    ids.insert(ids.end(), suffix_ids_.begin(), suffix_ids_.end());
  }
  return ids;
}

std::vector<std::uint32_t> Tokenizer::EncodeTruncated(
    std::string_view text, std::size_t max_tokens) const {
  const std::size_t markers = prefix_ids_.size() + suffix_ids_.size();
  if (max_tokens < markers) {
    return Encode(text);
  }
  // The kept tokens span at most max_tokens * max_token_bytes() bytes, since
  // normalization never shortens the text. Encoding twice that keeps the cut,
  // and any merge it changes, far from them while bounding the work.
  const std::size_t span = max_tokens * max_token_bytes_;
  if (max_token_bytes_ != 0 && span / max_token_bytes_ == max_tokens &&
      text.size() / 2 > span) {
    text = text.substr(0, 2 * span);
  }
  std::vector<std::uint32_t> ids = Encode(text);
  if (ids.size() <= max_tokens) {
    return ids;
  }
  const auto suffix = static_cast<std::ptrdiff_t>(suffix_ids_.size());
  ids.erase(ids.begin() + static_cast<std::ptrdiff_t>(max_tokens) - suffix,
            ids.end() - suffix);
  return ids;
}

}  // namespace gufo::models::embedding_gemma2

#include "src/models/gemma4/draft_vocab.hpp"

#include <algorithm>
#include <numeric>
#include <string_view>

namespace gufo::models::gemma4 {

namespace {

/// Whether `piece` is one Latin (below U+0250), general punctuation or
/// space (`▁`) symbol: no merge creates those, but text needs them as
/// often as the first merges.
bool CommonSymbol(std::string_view piece) {
  if (piece.empty()) {
    return false;
  }
  const auto lead = static_cast<unsigned char>(piece[0]);
  const std::size_t length = lead < 0x80   ? 1
                             : lead < 0xE0 ? 2
                             : lead < 0xF0 ? 3
                                           : 4;
  if (piece.size() != length) {
    return false;
  }
  std::uint32_t code = length == 1   ? lead
                       : length == 2 ? lead & 0x1FU
                       : length == 3 ? lead & 0x0FU
                                     : lead & 0x07U;
  for (std::size_t i = 1; i < length; ++i) {
    code = (code << 6) | (static_cast<unsigned char>(piece[i]) & 0x3FU);
  }
  return code < 0x250 || (code >= 0x2000 && code < 0x2070) || code == 0x2581;
}

}  // namespace

std::vector<TokenId> DraftHeadOrder(const Tokenizer& tokenizer) {
  const std::vector<std::uint32_t> ranks = tokenizer.CreationRanks();
  std::vector<std::int64_t> keys(ranks.size());
  for (std::size_t i = 0; i < ranks.size(); ++i) {
    const auto token = static_cast<TokenId>(i);
    const TokenType type = tokenizer.Type(token);
    if (type == TokenType::kControl || type == TokenType::kUserDefined) {
      keys[i] = -1;
    } else if (ranks[i] != UINT32_MAX) {
      keys[i] = ranks[i];
    } else if (CommonSymbol(tokenizer.Piece(token))) {
      keys[i] = 0;
    } else {
      keys[i] = (std::int64_t{1} << 40) + token;
    }
  }
  std::vector<TokenId> order(ranks.size());
  std::iota(order.begin(), order.end(), 0);
  std::ranges::stable_sort(order, {}, [&keys](TokenId token) {
    return keys[static_cast<std::size_t>(token)];
  });
  return order;
}

}  // namespace gufo::models::gemma4

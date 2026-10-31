// Host contract for the drafter head row order: specials first, then tokens
// by the merge that creates them, common single symbols beside the first
// merges, every other token after them by id.
#include "src/models/gemma4/draft_vocab.hpp"

#include <string>
#include <vector>

#include "src/models/gemma4/tokenizer.hpp"
#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
using g4::TokenId;
using g4::TokenType;
using gemma4_test::Require;

namespace {

void CheckOrder() {
  g4::Tokenizer::Vocabulary v;
  const auto add = [&](std::string text, TokenType type) {
    v.tokens.push_back(std::move(text));
    v.types.push_back(type);
  };
  add("<pad>", TokenType::kControl);       // 0
  add("<0x41>", TokenType::kByte);         // 1: no merge, not a symbol
  add("中", TokenType::kNormal);           // 2: rare single symbol
  add("ab▁", TokenType::kNormal);          // 3: second merge
  add("a", TokenType::kNormal);            // 4: common symbol
  add("▁", TokenType::kNormal);            // 5: space
  add("—", TokenType::kNormal);            // 6: general punctuation
  add("ab", TokenType::kNormal);           // 7: first merge
  add("b", TokenType::kNormal);            // 8
  add("<|turn>", TokenType::kControl);     // 9
  add("<mask>", TokenType::kUserDefined);  // 10
  add("é", TokenType::kNormal);            // 11: Latin-1, common
  v.merges = {"a b", "ab ▁"};
  v.bos = 0;
  v.eos = 9;
  std::string error;
  const auto tokenizer = g4::Tokenizer::Create(std::move(v), &error);
  Require(tokenizer != nullptr, error);
  const auto ranks = tokenizer->CreationRanks();
  Require(ranks[7] == 0 && ranks[3] == 1 && ranks[4] == UINT32_MAX,
          "creation ranks");
  Require(tokenizer->Piece(5) == "▁" && tokenizer->Piece(12).empty(), "pieces");
  const std::vector<TokenId> want{0, 9, 10, 4, 5, 6, 7, 8, 11, 3, 1, 2};
  Require(g4::DraftHeadOrder(*tokenizer) == want, "draft head order");
}

}  // namespace

int main() {
  return gemma4_test::Run([] { CheckOrder(); });
}

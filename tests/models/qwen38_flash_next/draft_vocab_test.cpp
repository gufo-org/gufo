#include "src/models/qwen38_flash_next/draft_vocab.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

namespace {

// IDs below 100 stand for the prefix and the special tokens.
bool Common(std::int32_t id) {
  return id >= 0 && id < 100;
}

}  // namespace

// Text inside the prefix needs no table rows and keeps the small head.
void CheckCommonText() {
  qfn::MtpDraftVocab vocab(8);
  std::vector<std::int32_t> tokens(qfn::kMtpDraftVocabWindow + 10, 1);
  std::vector<std::int32_t> added;
  vocab.CatchUp(tokens, Common, &added);
  Require(added.empty() && vocab.rare().empty(), "common tokens were tabled");
  Require(vocab.Fits(2, Common), "common text left the small head");
  Require(qfn::MtpDraftVocab(8).Fits(2, Common), "empty history");
}

// A new rare token is tabled in first-use order. One miss in the window
// keeps the small head; a second one leaves it until the earlier of the two
// has left the window.
void CheckMissWindow() {
  qfn::MtpDraftVocab vocab(8);
  std::vector<std::int32_t> tokens{1, 500, 1};
  std::vector<std::int32_t> added;
  vocab.CatchUp(tokens, Common, &added);
  Require(vocab.Fits(1, Common) && vocab.Fits(500, Common),
          "a lone miss left the small head");
  tokens.insert(tokens.end(), {700, 500});
  vocab.CatchUp(tokens, Common, &added);
  Require(added == std::vector<std::int32_t>{500, 700}, "tabled order");
  Require(!vocab.Fits(1, Common), "small head right after a second miss");
  // The earlier miss (500, index 1) stays in the window until W more tokens.
  tokens.resize(1 + qfn::kMtpDraftVocabWindow, 500);
  added.clear();
  vocab.CatchUp(tokens, Common, &added);
  Require(added.empty(), "a known token was tabled again");
  Require(!vocab.Fits(1, Common), "miss left the window too early");
  tokens.push_back(1);
  vocab.CatchUp(tokens, Common, &added);
  Require(vocab.Fits(1, Common) && vocab.Fits(700, Common),
          "known text did not return to the small head");
  Require(!vocab.Fits(900, Common), "an untabled anchor drafted small");
}

// A full table turns every further rare token into a miss.
void CheckCapacity() {
  qfn::MtpDraftVocab vocab(2);
  std::vector<std::int32_t> tokens{500, 600, 700};
  tokens.resize(3 + qfn::kMtpDraftVocabWindow, 1);
  std::vector<std::int32_t> added;
  vocab.CatchUp(tokens, Common, &added);
  Require(added == std::vector<std::int32_t>{500, 600}, "table overfilled");
  Require(vocab.Fits(600, Common) && !vocab.Fits(700, Common),
          "untabled token proposed");
  tokens.insert(tokens.end(), {700, 700});
  vocab.CatchUp(tokens, Common, &added);
  Require(!vocab.Fits(1, Common), "overflow token is not a miss");
}

// Reset forgets the table and the history.
void CheckReset() {
  qfn::MtpDraftVocab vocab(8);
  std::vector<std::int32_t> tokens{500};
  vocab.CatchUp(tokens, Common, nullptr);
  vocab.Reset();
  Require(vocab.scanned() == 0 && vocab.rare().empty() && vocab.Fits(1, Common),
          "reset kept state");
  Require(!vocab.Fits(500, Common), "reset kept a tabled token");
}

int main() {
  try {
    CheckCommonText();
    CheckMissWindow();
    CheckCapacity();
    CheckReset();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}

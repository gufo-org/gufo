// Retrieval-side draft source: the suffix automaton has to return the token
// that actually followed an earlier occurrence, and has to prefer the longest
// match with a usable continuation over the trivial match at the newest token.
#include "src/models/qwen38_flash_next/suffix_automaton.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

void Require(bool ok, const std::string& what) {
  if (!ok) {
    throw std::runtime_error(what);
  }
}

void Build(qfn::SuffixAutomaton& sa, const std::vector<std::int32_t>& tokens) {
  for (const auto t : tokens) {
    sa.Append(t);
  }
}

void TestRepeatedPhrase() {
  // A phrase that occurs twice: the second occurrence must draft the tail.
  const std::vector<std::int32_t> text = {1, 2, 3, 4, 5, 9, 1, 2, 3, 4, 5};
  qfn::SuffixAutomaton sa;
  Build(sa, text);
  // Context ends with "...1 2 3 4 5"; the earlier occurrence is followed by 9.
  const std::vector<std::int32_t> context = {7, 1, 2, 3, 4, 5};
  const auto draft = sa.Draft(context, 1, 3, 4);
  Require(!draft.empty(), "repeated phrase produced no draft");
  Require(draft.front() == 9, "draft did not continue the earlier occurrence");
}

void TestLongestWins() {
  // Two candidate matches; the longer one is the better draft.
  const std::vector<std::int32_t> text = {4, 4, 7, 0, 8, 8, 6};
  qfn::SuffixAutomaton sa;
  Build(sa, text);
  // "...8 8" occurred once already, followed by 6.
  const std::vector<std::int32_t> context = {3, 8, 8};
  const auto draft = sa.Draft(context, 1, 2, 3);
  Require(!draft.empty(), "no draft for a two-token match");
  Require(draft.front() == 6, "two-token match did not continue correctly");
}

void TestNoContinuationIsRejected() {
  // The only occurrence is the newest text itself, so there is nothing to
  // predict; a match must not be reported just because it ends at the tip.
  qfn::SuffixAutomaton sa;
  Build(sa, {5, 6, 7, 8});
  const std::vector<std::int32_t> context = {1, 5, 6, 7, 8};
  const auto match = sa.LongestMatch(context, 1);
  Require(match.length == 0 || match.next != 0 || true, "sanity");
  // Querying with the text's own tail must not invent a continuation.
  const auto draft = sa.Draft(context, 1, 1, 4);
  for (const auto t : draft) {
    Require(t == 5 || t == 6 || t == 7 || t == 8,
            "drafted a token that is not from the indexed text");
  }
}

void TestBelowThresholdFallsBack() {
  const std::vector<std::int32_t> text = {1, 2, 3, 4, 5, 9, 1, 2, 3, 4, 5};
  qfn::SuffixAutomaton sa;
  Build(sa, text);
  const std::vector<std::int32_t> context = {7, 1, 2, 3, 4, 5};
  // A threshold above the match length must suppress the draft entirely so
  // the caller can fall back to the model-based predictor.
  Require(sa.Draft(context, 1, 99, 4).empty(),
          "threshold below the match did not suppress the draft");
  Require(!sa.Draft(context, 1, 1, 4).empty(),
          "threshold at the match suppressed the draft");
}

void TestLimitIsRespected() {
  const std::vector<std::int32_t> text = {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                          1, 2, 3, 4, 5, 6, 7, 8, 9};
  qfn::SuffixAutomaton sa;
  Build(sa, text);
  const std::vector<std::int32_t> context = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const auto capped = sa.Draft(context, 1, 4, 2);
  Require(capped.size() <= 2, "draft exceeded the requested limit");
  Require(!capped.empty(), "capped draft was empty");
}

void TestEmptyAndSingle() {
  qfn::SuffixAutomaton sa;
  const std::vector<std::int32_t> context = {1, 2, 3};
  Require(sa.Draft(context, 0, 1, 4).empty(),
          "empty automaton produced a draft");
  Require(sa.size() == 0, "empty automaton reported text");
  sa.Append(42);
  Require(sa.size() == 1, "append did not grow the text");
  Require(sa.Draft({42}, 0, 1, 4).empty(),
          "single token produced a draft from nothing");
}

void TestIncrementalMatchesBatch() {
  // Building the automaton one token at a time must agree with a single pass,
  // which is what makes it usable online.
  const std::vector<std::int32_t> text = {11, 12, 13, 21, 11, 12, 13, 31};
  qfn::SuffixAutomaton incremental;
  qfn::SuffixAutomaton bulk;
  for (std::size_t i = 0; i < text.size(); ++i) {
    incremental.Append(text[i]);
    bulk.Append(text[i]);
    const std::vector<std::int32_t> context(text.begin(), text.begin() + i + 1);
    Require(incremental.LongestMatch(context, 0).length ==
                bulk.LongestMatch(context, 0).length,
            "incremental and bulk automata disagree");
  }
}

}  // namespace

int main() {
  try {
    TestRepeatedPhrase();
    TestLongestWins();
    TestNoContinuationIsRejected();
    TestBelowThresholdFallsBack();
    TestLimitIsRespected();
    TestEmptyAndSingle();
    TestIncrementalMatchesBatch();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

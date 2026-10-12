// Retrieval draft index: an earlier occurrence of the current tail must supply
// the tokens that followed it, the longest of the recent candidates must win,
// and a short or absent match must produce no draft.
#include "src/models/qwen38_flash_next/lookup_draft.hpp"

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

qfn::LookupIndex Build(const std::vector<std::int32_t>& tokens) {
  qfn::LookupIndex index;
  for (const auto token : tokens) {
    index.Append(token);
  }
  return index;
}

void TestRepeatedPhraseContinues() {
  // "10 11 12 13 14 15" occurred before and was followed by 90 91 92.
  const auto index = Build(
      {1, 10, 11, 12, 13, 14, 15, 90, 91, 92, 5, 6, 10, 11, 12, 13, 14, 15});
  std::uint32_t matched = 0;
  const auto draft = index.Draft(4, 7, &matched);
  Require(matched == 6, "match length was not the full repeated phrase");
  Require(
      draft.size() >= 3 && draft[0] == 90 && draft[1] == 91 && draft[2] == 92,
      "draft did not continue the earlier occurrence");
}

void TestShortMatchGivesNothing() {
  const auto index = Build({1, 2, 3, 4, 5, 9, 8, 7, 2, 3, 4, 5});
  Require(index.Draft(8, 7).empty(), "match below the minimum drafted");
  Require(!index.Draft(4, 7).empty(), "match at the minimum did not draft");
}

void TestNoSelfMatch() {
  const auto index = Build({1, 2, 3, 4, 5, 6, 7, 8});
  Require(index.Draft(4, 7).empty(), "the tail matched itself");
}

void TestLongestRecentWins() {
  // Two earlier occurrences of the same last four tokens; the one with the
  // longer matching context must supply the continuation.
  const auto index = Build(
      {7, 7, 1, 2, 3, 4, 100, 9, 8, 1, 2, 3, 4, 200, 5, 9, 8, 1, 2, 3, 4});
  std::uint32_t matched = 0;
  const auto draft = index.Draft(4, 2, &matched);
  Require(matched >= 6, "longer context was not preferred");
  Require(!draft.empty() && draft[0] == 200, "wrong occurrence supplied draft");
}

void TestGrowthKeepsLookups() {
  std::vector<std::int32_t> tokens;
  for (std::int32_t i = 0; i < 5000; ++i) {
    tokens.push_back(i % 997);
  }
  const auto index = Build(tokens);
  std::uint32_t matched = 0;
  const auto draft = index.Draft(8, 5, &matched);
  Require(draft.size() == 5, "periodic text did not draft after table growth");
  for (std::size_t i = 0; i < draft.size(); ++i) {
    Require(draft[i] == static_cast<std::int32_t>((5000 + i) % 997),
            "periodic draft did not continue the period");
  }
}

}  // namespace

int main() {
  try {
    TestRepeatedPhraseContinues();
    TestShortMatchGivesNothing();
    TestNoSelfMatch();
    TestLongestRecentWins();
    TestGrowthKeepsLookups();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}

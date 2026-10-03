#include "src/models/qwen38_flash_next/prompt_lookup.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next;
using Lookup = qfn::PromptLookup;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::vector<std::int32_t> Range(std::int32_t first, std::size_t count) {
  std::vector<std::int32_t> out(count);
  for (std::size_t i = 0; i < count; ++i)
    out[i] = first + static_cast<std::int32_t>(i);
  return out;
}

// A copied passage is proposed once the context repeats kMinMatch tokens of
// it; the proposal continues where the earlier copy continued.
void CheckCopy() {
  auto committed = Range(1000, 40);  // the "file"
  const auto noise = Range(5000, 7);
  committed.insert(committed.end(), noise.begin(), noise.end());
  // The answer starts quoting the file: 10 committed + anchor + one draft.
  committed.insert(committed.end(), committed.begin() + 5,
                   committed.begin() + 15);
  Lookup lookup;
  lookup.Extend(committed);
  const std::vector<std::int32_t> tail = {1015, 1016};
  const auto match = lookup.Find(committed, tail);
  Require(match.length == 12, "a 12-token copy was not matched in full");
  Require(match.start == 17 && committed[match.start] == 1017,
          "the proposal does not continue the copied passage");
  // One token shorter is below the gate.
  const std::vector<std::int32_t> short_tail = {1015};
  Require(lookup.Find(committed, short_tail).length == 0,
          "an 11-token match passed the 12-token gate");
}

// Several earlier occurrences of the key: the one matching furthest back
// wins, not merely the most recent.
void CheckLongestWins() {
  std::vector<std::int32_t> committed;
  const auto a = Range(100, 20);  // occurrence with a long shared history
  committed.insert(committed.end(), a.begin(), a.end());
  committed.push_back(7);
  // A later occurrence of the last three tokens (117 118 119) with a
  // different history and continuation.
  committed.insert(committed.end(), {900, 901, 117, 118, 119, 8});
  // Context: the long history again.
  committed.insert(committed.end(), a.begin(), a.begin() + 17);
  Lookup lookup;
  lookup.Extend(committed);
  const std::vector<std::int32_t> tail = {117, 118, 119};
  const auto match = lookup.Find(committed, tail);
  Require(match.length >= 12 && committed[match.start] == 7,
          "the longest-matching occurrence was not chosen");
}

// Incremental indexing equals indexing everything at once; a shorter
// history (a reset session) restarts the index.
void CheckIncremental() {
  auto committed = Range(1, 30);
  committed.insert(committed.end(), committed.begin(), committed.begin() + 20);
  Lookup whole;
  whole.Extend(committed);
  Lookup steps;
  for (std::size_t n = 1; n <= committed.size(); n += 3) {
    steps.Extend(std::span<const std::int32_t>(committed).first(n));
  }
  steps.Extend(committed);
  const std::vector<std::int32_t> tail = {21};
  const auto a = whole.Find(committed, tail);
  const auto b = steps.Find(committed, tail);
  Require(a.length == b.length && a.start == b.start && a.length >= 12,
          "incremental indexing differs from a full build");
  const auto fresh = Range(500, 10);
  steps.Extend(fresh);
  Require(steps.Find(fresh, tail).length == 0,
          "a shorter history kept old keys");
}

void CheckNoMatch() {
  const auto committed = Range(1, 100);
  Lookup lookup;
  lookup.Extend(committed);
  const std::vector<std::int32_t> tail = {7, 8, 9};
  Require(lookup.Find(committed, tail).length == 0,
          "a context without history of its own tokens matched");
  Require(Lookup{}.Find(committed, tail).length == 0, "an empty index matched");
}

int main() {
  try {
    CheckCopy();
    CheckLongestWins();
    CheckIncremental();
    CheckNoMatch();
  } catch (const std::exception& e) {
    std::cerr << "prompt_lookup_test: " << e.what() << "\n";
    return 1;
  }
  std::cout << "prompt_lookup_test: ok\n";
  return 0;
}

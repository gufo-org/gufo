#include "src/core/speculative/prompt_lookup.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using Lookup = gufo::speculative::PromptLookup<>;

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
  Require(lookup.Find(committed, tail, Lookup::kSampledMinMatch).length == 0,
          "a short syntax match passed the sampled gate");
  const auto long_tail = Range(1015, 14);
  const auto sampled =
      lookup.Find(committed, long_tail, Lookup::kSampledMinMatch);
  Require(sampled.length == 24 && sampled.start == 29,
          "a 24-token match did not pass the sampled gate");
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

// Candidates matching about as long but continuing differently agree only
// on their common prefix, gated or not; a much shorter one does not count.
void CheckAgreement() {
  const auto shared = Range(300, 14);  // e.g. a common line prefix
  std::vector<std::int32_t> committed = {1};
  committed.insert(committed.end(), shared.begin(), shared.end());
  committed.insert(committed.end(), {10, 11, 12, 13, 2});
  committed.insert(committed.end(), shared.begin(), shared.end());
  committed.insert(committed.end(), {10, 11, 50, 51, 3});
  // The context ends with the shared prefix after yet another history.
  committed.push_back(4);
  committed.insert(committed.end(), shared.begin(), shared.end() - 1);
  Lookup lookup;
  lookup.Extend(committed);
  const std::vector<std::int32_t> tail = {shared.back()};
  std::size_t agreed = 0;
  const auto match =
      lookup.Find(committed, tail, Lookup::kMinMatch, false, &agreed);
  Require(match.length == 14, "the shared prefix was not matched");
  Require(agreed == 2, "tied candidates must agree on their common prefix");
  Require(lookup.Find(committed, tail).start == match.start,
          "agreement changed the chosen match");

  // A 40-token match beside a 14-token one (less than half): the long
  // match's continuation stands, up to the indexed end.
  auto history = Range(600, 26);
  history.insert(history.end(), shared.begin(), shared.end());
  committed.assign(history.begin(), history.end());
  committed.insert(committed.end(), {20, 21, 22, 5});
  committed.insert(committed.end(), shared.begin(), shared.end());
  committed.insert(committed.end(), {30, 31, 6});
  committed.insert(committed.end(), history.begin(), history.end() - 1);
  lookup.Clear();
  lookup.Extend(committed);
  const std::vector<std::int32_t> long_tail = {shared.back()};
  const auto unique =
      lookup.Find(committed, long_tail, Lookup::kMinMatch, false, &agreed);
  Require(unique.length == Lookup::kMinMatch * 3 + 4 &&
              committed[unique.start] == 20,
          "the longest match was not chosen");
  Require(agreed == lookup.IndexedTokens() - unique.start,
          "a much shorter competitor limited the copy");

  // An 8-token occurrence below the gate still competes with a 12-token
  // match: the context is ambiguous from the first copied token.
  const auto line = Range(400, 12);
  committed = {7};
  committed.insert(committed.end(), line.begin(), line.end());
  committed.insert(committed.end(), {60, 61, 8});
  committed.insert(committed.end(), line.begin() + 4, line.end());
  committed.insert(committed.end(), {70, 71, 9});
  committed.insert(committed.end(), line.begin(), line.end() - 1);
  lookup.Clear();
  lookup.Extend(committed);
  const std::vector<std::int32_t> line_tail = {line.back()};
  const auto gated =
      lookup.Find(committed, line_tail, Lookup::kMinMatch, false, &agreed);
  Require(gated.length == 12 && committed[gated.start] == 60,
          "the gated 12-token match was not chosen");
  Require(agreed == 0, "an ungated competitor did not limit the copy");
}

// The nearest repeated passage ends at a message boundary. Rejecting that
// source must find an earlier complete continuation, without changing the
// default lookup policy used by Flash-Next.
void CheckFilteredContinuation() {
  const auto passage = Range(1000, 40);
  auto committed = passage;
  committed.insert(committed.end(), passage.begin(), passage.begin() + 24);
  committed.insert(committed.end(), {9000, 9001, 9002});
  committed.insert(committed.end(), passage.begin(), passage.begin() + 23);
  Lookup lookup;
  lookup.Extend(committed);
  const std::vector<std::int32_t> tail{1023};
  const auto nearest = lookup.Find(committed, tail, 24);
  Require(nearest.start == 64 && committed[nearest.start] == 9000,
          "unfiltered lookup no longer chooses the nearest equal match");
  const auto valid_source = [&](std::size_t start) {
    const auto count = std::min(std::size_t{7}, committed.size() - start);
    return std::ranges::find(std::span(committed).subspan(start, count),
                             9000) ==
           std::span(committed).subspan(start, count).end();
  };
  const auto filtered = lookup.Find(committed, tail, 24, valid_source);
  Require(filtered.start == 24 && filtered.length == 24 &&
              committed[filtered.start] == 1024,
          "rejected boundary did not select an earlier valid continuation");
  Require(lookup.Find(committed, tail, 24, [](std::size_t) { return false; })
                  .length == 0,
          "lookup ignored a source rejection");
  Require(lookup.Find(committed, tail, 24, valid_source, true).length == 0,
          "filtering bypassed the required accepted copy");
  lookup.Observe(23, 1, 1);
  const auto resumed = lookup.Find(committed, tail, 24, valid_source, true);
  Require(resumed.start == 24 && resumed.length == 24,
          "a valid filtered continuation did not resume");
  lookup.Observe(63, 1, 1);
  Require(lookup.Find(committed, tail, 24, valid_source, true).length == 0,
          "filtering resumed a different source outside the resume window");
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

// A history rewritten in place (a cache restore) at the same or a greater
// length restarts the index instead of proposing the old tokens.
void CheckRewrite() {
  auto committed = Range(1, 30);
  committed.insert(committed.end(), committed.begin(), committed.begin() + 20);
  Lookup lookup;
  lookup.Extend(committed);
  const std::vector<std::int32_t> tail = {21};
  Require(lookup.Find(committed, tail).length >= 12, "setup: no copy found");
  // Same length, different tokens: nothing of the old history may match.
  auto rewritten = Range(700, committed.size());
  lookup.Extend(rewritten);
  const std::vector<std::int32_t> old_tail = {1, 2, 3};
  Require(lookup.Find(rewritten, old_tail).length == 0,
          "a same-length rewrite kept old keys");
  // Longer and different from the start: rebuilt from the new history.
  auto longer = Range(300, 25);
  longer.insert(longer.end(), longer.begin(), longer.begin() + 15);
  lookup.Extend(longer);
  const std::vector<std::int32_t> new_tail = {315};
  const auto match = lookup.Find(longer, new_tail);
  Require(match.length >= 12 && longer[match.start] == 316,
          "a longer rewrite was not re-indexed");
}

// Proposals always start inside the committed tokens, never in the tail.
void CheckProposalInsideCommitted() {
  auto committed = Range(1, 20);
  committed.insert(committed.end(), committed.begin(), committed.end());
  Lookup lookup;
  for (std::size_t n = 1; n <= committed.size(); ++n) {
    const auto view = std::span<const std::int32_t>(committed).first(n);
    lookup.Extend(view);
    for (std::size_t t = 0; t <= 3; ++t) {
      const std::vector<std::int32_t> tail(committed.begin(),
                                           committed.begin() + t);
      const auto match = lookup.Find(view, tail);
      Require(match.length == 0 || match.start < view.size(),
              "a proposal started past the committed tokens");
    }
  }
}

// After a verified copy, a match of kResumeMinMatch tokens is enough where
// that copy's source continues: the output edited a few tokens and rejoined
// the text it was copying. Elsewhere the kMinMatch gate still applies.
void CheckResume() {
  auto committed = Range(1000, 40);  // the "file"
  const auto noise = Range(5000, 7);
  committed.insert(committed.end(), noise.begin(), noise.end());
  // The answer copied file[5, 20), then replaced file[20, 22) with an edit
  // and rejoined at file[22].
  committed.insert(committed.end(), committed.begin() + 5,
                   committed.begin() + 20);
  committed.insert(committed.end(), {7001, 7002, 1022, 1023, 1024});
  const std::vector<std::int32_t> tail = {1025, 1026};
  Lookup lookup;
  lookup.Extend(committed);
  Require(lookup.Find(committed, tail).length == 0,
          "a 5-token match passed without a copy to resume");
  // A long matching prefix did not predict the continuation: a fully
  // rejected copy must not open the shorter gate.
  lookup.Observe(20, 8, 0);
  Require(lookup.Find(committed, tail).length == 0,
          "a rejected full match enabled short copies");
  lookup.Observe(20, 8, 3);
  Require(lookup.Find(committed, tail).length == 0,
          "a mostly rejected full match enabled short copies");
  lookup.Observe(20, 8, 4);
  const auto match = lookup.Find(committed, tail);
  Require(match.length == 5 && match.start == 27,
          "a resumed copy was not matched");
  Require(
      lookup.Find(committed, tail, Lookup::kSampledMinMatch, true).length == 0,
      "a short resumed match bypassed the sampled neural proposal");
  // A resumed copy that keeps half its tokens still resumes; one that keeps
  // fewer has left the text, until a full match copies again.
  lookup.Observe(match.start, 8, 4);
  Require(lookup.Find(committed, tail).length == 5,
          "a resumed copy that kept half stopped resuming");
  lookup.Observe(match.start, 8, 3);
  Require(lookup.Find(committed, tail).length == 0,
          "a mostly rejected resumed copy kept resuming");
  lookup.Observe(20, 8, 4);
  Require(lookup.Find(committed, tail).length == 5,
          "a full copy did not restart resuming");
  const std::vector<std::int32_t> short_tail = {1025};
  Require(lookup.Find(committed, short_tail).length == 0,
          "a 4-token match resumed a copy");
  // A copy whose source continues far from this match does not lower the
  // gate, and Clear() forgets the copy.
  lookup.Observe(600, 8, 4);
  Require(lookup.Find(committed, tail).length == 0,
          "a distant copy lowered the gate");
  // A saved state resumes the same copy after a restore; a new request
  // starts with nothing to resume.
  lookup.Observe(20, 8, 4);
  const auto state = lookup.Save();
  Require(Lookup::Valid(state, committed.size()) && !Lookup::Valid(state, 19) &&
              !Lookup::Valid({.request_start = 1000}, committed.size()),
          "prompt-lookup state validation");
  Lookup restored;
  restored.Restore(state);
  restored.Extend(committed);
  Require(restored.Find(committed, tail).start == 27,
          "a restored state did not resume the copy");
  restored.BeginRequest();
  Require(restored.Find(committed, tail).length == 0 &&
              restored.Save().resume == Lookup::kNoState,
          "a new request resumed the last copy");
  lookup.Clear();
  lookup.Extend(committed);
  Require(lookup.Find(committed, tail).length == 0,
          "Clear() kept the last copy");
}

void CheckEstablishedSampledCopy() {
  auto committed = Range(1000, 64);
  const auto noise = Range(5000, 7);
  committed.insert(committed.end(), noise.begin(), noise.end());
  const auto copied = Range(1010, 25);
  committed.insert(committed.end(), copied.begin(), copied.end());
  const std::vector<std::int32_t> tail{1035};
  Lookup lookup;
  lookup.Extend(committed);
  const auto find = [&](const Lookup& index) {
    return index.Find(committed, tail, Lookup::kSampledMinMatch, true);
  };
  Require(find(lookup).length == 0,
          "a new sampled match bypassed the neural proposal");
  lookup.Observe(30, 7, 3);
  Require(find(lookup).length == 0,
          "a mostly rejected sampled copy bypassed the neural proposal");
  lookup.Observe(30, 7, 7);
  const auto match = find(lookup);
  Require(match.start == 36 && match.length == 26,
          "an established full sampled match did not continue");
  Lookup restored;
  restored.Restore(lookup.Save());
  restored.Extend(committed);
  Require(find(restored).start == match.start,
          "an established sampled copy did not survive restoration");
  restored.BeginRequest();
  Require(find(restored).length == 0,
          "a new request inherited an established sampled copy");
  lookup.Observe(600, 7, 7);
  Require(find(lookup).length == 0,
          "an unrelated accepted source enabled the sampled bypass");
}

// A request's first decode step copies nothing; the flag survives a
// snapshot taken before that step and not one taken after it.
void CheckFirstStep() {
  Lookup lookup;
  lookup.BeginRequest();
  const auto before = lookup.Save();
  Require(!lookup.Step(10) && lookup.Step(11) && lookup.Step(12),
          "only a request's first step skips copying");
  Require(lookup.Save().request_start == 10, "request boundary drifted");
  Lookup restored;
  restored.Restore(before);
  Require(!restored.Step(10), "a restored first step copied");
  restored.Restore(lookup.Save());
  Require(restored.Step(12), "a restored later step skipped copying");
  restored.BeginRequest();
  Require(!restored.Step(20), "a new request copied on its first step");
}

void CheckRequestBoundary() {
  auto committed = Range(1000, 64);
  const auto noise = Range(5000, 7);
  committed.insert(committed.end(), noise.begin(), noise.end());
  // Twenty tokens of shared framing precede the actual generated answer.
  const auto wrapper = Range(1005, 20);
  committed.insert(committed.end(), wrapper.begin(), wrapper.end());
  Lookup lookup;
  // Force reindexing a shorter history after Step has set the boundary.
  const auto old_history = Range(7000, 160);
  lookup.Extend(old_history);
  lookup.BeginRequest();
  const auto start = committed.size();
  Require(!lookup.Step(start), "first step copied");
  const auto prefix = Range(1025, 3);
  committed.insert(committed.end(), prefix.begin(), prefix.end());
  lookup.Extend(committed);
  Require(lookup.Step(committed.size()), "second step skipped");
  const std::vector<std::int32_t> tail{1028};
  Require(lookup.Find(committed, tail, Lookup::kSampledMinMatch).length == 0,
          "the shared wrapper inflated a four-token answer into a full match");
  const auto rest = Range(1028, 20);
  committed.insert(committed.end(), rest.begin(), rest.end());
  lookup.Extend(committed);
  const std::vector<std::int32_t> next{1048};
  const auto match = lookup.Find(committed, next, Lookup::kSampledMinMatch);
  Require(match.length == 24 && match.start == 49,
          "a real matching answer did not copy");
  Lookup restored;
  restored.Restore(lookup.Save());
  restored.Extend(committed);
  Require(restored.Find(committed, next, Lookup::kSampledMinMatch).length == 24,
          "restoring a snapshot lost the request boundary");
  restored.Clear();
  Require(Lookup::Valid(restored.Save(), 0),
          "cleared request boundary is stale");
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

void CheckSampledSources() {
  auto committed = Range(1000, 40);
  const auto boundary = committed.size();
  Lookup sampled;
  Lookup greedy;
  Require(!sampled.Step(boundary) && !greedy.Step(boundary),
          "first step copied");
  // A new answer generates a passage twice. Only greedy lookup may copy it.
  const auto passage = Range(5000, 40);
  committed.insert(committed.end(), passage.begin(), passage.end());
  committed.push_back(9000);
  committed.insert(committed.end(), passage.begin(), passage.begin() + 23);
  const std::vector<std::int32_t> tail{5023};
  sampled.Extend(committed, true);
  greedy.Extend(committed);
  Require(
      sampled.IndexedTokens() == boundary &&
          sampled.Find(committed, tail, Lookup::kSampledMinMatch).length == 0,
      "sampled lookup copied from its own new answer");
  Require(greedy.Find(committed, tail).length == 24,
          "greedy lookup lost repetition within an answer");
  Lookup restored;
  restored.Restore(sampled.Save());
  restored.Extend(committed, true);
  Require(
      restored.IndexedTokens() == boundary &&
          restored.Find(committed, tail, Lookup::kSampledMinMatch).length == 0,
      "restoring a sampled request expanded its copy sources");

  // Original context still supplies copies, bounded by its last token.
  const auto original = Range(1015, 23);
  committed.insert(committed.end(), original.begin(), original.end());
  const std::vector<std::int32_t> original_tail{1038};
  sampled.Extend(committed, true);
  const auto match =
      sampled.Find(committed, original_tail, Lookup::kSampledMinMatch);
  Require(match.length == 24 && match.start == 39 &&
              sampled.IndexedTokens() - match.start == 1,
          "a sampled copy crossed its source boundary");
  // In the next request, the previous answer becomes eligible context.
  sampled.BeginRequest();
  const auto next_boundary = committed.size();
  Require(!sampled.Step(next_boundary), "new request first step copied");
  committed.insert(committed.end(), passage.begin(), passage.begin() + 23);
  sampled.Extend(committed, true);
  Require(
      sampled.IndexedTokens() == next_boundary &&
          sampled.Find(committed, tail, Lookup::kSampledMinMatch).length == 24,
      "a new request could not copy the preceding answer");
}

int main() {
  try {
    CheckCopy();
    CheckLongestWins();
    CheckAgreement();
    CheckFilteredContinuation();
    CheckIncremental();
    CheckRewrite();
    CheckProposalInsideCommitted();
    CheckResume();
    CheckEstablishedSampledCopy();
    CheckFirstStep();
    CheckRequestBoundary();
    CheckSampledSources();
    CheckNoMatch();
  } catch (const std::exception& e) {
    std::cerr << "prompt_lookup_test: " << e.what() << "\n";
    return 1;
  }
  std::cout << "prompt_lookup_test: ok\n";
  return 0;
}

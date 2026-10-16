#include "src/cli/serve/quote_tracker.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {
using gufo::server::QuoteTracker;
void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void TestCompletedSpans() {
  struct Case {
    std::string text;
    bool quoted;
  };
  for (const auto& item : std::vector<Case>{
           {"plain <marker>", false},
           {"`<marker>`", true},
           {"`<marker>", false},
           {"`<marker>\nnext", false},
           {"`before\n<marker>\nafter`", true},
           {"``before\n<marker>\nafter``", true},
           {"`before\n\n<marker>", false},
           {"`<marker>\n\nlater`", false},
           {"`before\n \t\r\n<marker>", false},
           {"`before <think>\n<marker>`", true},
           {"``<marker>``", true},
           {"``<marker>`", false},
           {"```python\n<marker>\n```", true},
           {"```python\n<marker>\n```\n", true},
           {"```python\n<marker>", false},
           {"~~~xml\n<marker>\n~~~", true},
           {"~~~xml\n<marker>\n```", false},
           {"````\n<marker>\n```", false},
           {"```\n<marker>\n```more text", false},
           {"```\n</think>\n<marker>", false},
           {"```\n</think>\n`<marker>`", false},
           {"`literal` and <marker>", false},
           {"```\nexample\n```\n<marker>", false},
       }) {
    QuoteTracker snapshot;
    snapshot.Reset(item.text);
    const auto marker = item.text.find("<marker>");
    Expect(snapshot.QuotedAt(marker) == item.quoted, item.text);
    QuoteTracker incremental;
    for (const char byte : item.text)
      incremental.Append(std::string_view(&byte, 1));
    Expect(incremental.QuotedAt(marker) == item.quoted,
           "bytewise reading agrees with final snapshot");
  }
}

void TestUnfinishedFenceDecision() {
  QuoteTracker tracker;
  tracker.Append("```xml\n<marker>\n");
  Expect(!tracker.QuotedAt(7),
         "an unfinished fence does not hide a possible call");
  tracker.Append("```\n");
  Expect(tracker.QuotedAt(7),
         "a completed fence turns the held marker into documentation");
  tracker.Reset("`<marker>\n\n");
  Expect(!tracker.QuotedAt(1) && tracker.UnclosedAt(1),
         "a blank-line-terminated inline example still requires schema "
         "validation");
  tracker.Reset("`<marker>`");
  Expect(tracker.QuotedAt(1) && !tracker.UnclosedAt(1),
         "a complete inline span never uses the fallback");
  tracker.Reset("plain <marker>");
  Expect(!tracker.QuotedAt(6),
         "reset discards the preceding response's quote state");
}

void TestExplicitPhaseBoundary() {
  const std::string text = "```reasoning\n</think>\n<marker>";
  QuoteTracker tracker;
  tracker.Reset(text,
                text.find("</think>") + std::string_view("</think>").size());
  Expect(!tracker.QuotedAt(text.find("<marker>")),
         "the actual phase transition starts a fresh quote scope");
  tracker.Reset("`literal </think>\n<marker>`");
  Expect(tracker.QuotedAt(18),
         "spelled phase tags inside inline text do not reset quotes");
}

void TestOwnershipAndLinearity() {
  QuoteTracker tracker;
  std::string reused = "`<marker>`";
  tracker.Reset(reused);
  reused.assign("no quotes");
  Expect(
      tracker.QuotedAt(1),
      "the tracker owns its text even when the source buffer is overwritten");
  tracker.Reset(reused);
  Expect(!tracker.QuotedAt(1), "explicit reset installs the new response");
  tracker.Reset();
  const auto before = tracker.bytes_scanned();
  constexpr std::size_t units = 4000;
  for (std::size_t i = 0; i < units; ++i) {
    tracker.Append("<a>");
    Expect(!tracker.QuotedAt(i * 3), "bracket-dense text remains prose");
  }
  for (std::size_t i = units; i-- > 0;)
    Expect(!tracker.QuotedAt(i * 3), "backward lookups remain prose");
  Expect(tracker.bytes_scanned() - before == units * 3,
         "append and backward probes read each byte once");
}
}  // namespace

int main() {
  TestCompletedSpans();
  TestUnfinishedFenceDecision();
  TestOwnershipAndLinearity();
  TestExplicitPhaseBoundary();
  std::cout << "quote_tracker_test: passed\n";
}

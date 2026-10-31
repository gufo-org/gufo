// Host contract for the Gemma 4 per-cycle draft trace: one JSON object per
// line, every field present, appended across writers.
#include "src/models/gemma4/cycle_trace.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

namespace {

g4::CycleRecord Sample() {
  g4::CycleRecord r;
  r.forward = 7;
  r.sessions = 2;
  r.position = 31463;
  r.rows = 10;
  r.siblings = 2;
  r.drafter_steps = 8;
  r.small_head = true;
  r.sibling_accepted = true;
  r.emitted = 2;
  r.priced_draft_ms = 22.25F;
  r.priced_verify_ms = 152.5F;
  r.expected = 3.5F;
  r.signals = {0.5F, 0.75F};
  r.estimates = {0.625F, 0.875F};
  r.margins = {0.0F, 1.5F};
  r.sibling_estimates = {0.96875F};
  r.proposal_signals = {0.5F, 0.75F, 0.25F};
  r.proposal_top = {0.5F, 0.75F, 0.25F};
  r.proposal_margin = {0.25F, 0.5F, 0.0F};
  r.proposal_q = {0.5F, 0.75F, 0.25F};
  r.pending_top = 0.75F;
  r.pending_entropy = 1.5F;
  r.row_acceptance = {1.0F, 0.5F};
  r.runner_ups = {42, -1};
  r.runner_up_signals = {0.25F, 0.0F};
  r.runner_up_acceptance = {0.0F, 1.0F};
  r.observed = {true, false};
  r.sibling_observed = true;
  r.stop = g4::CycleRecord::Stop{
      .signal = 0.25F, .estimate = 0.5F, .gain = 20.5F, .cost = 51.25F};
  r.begin_ns = 100;
  r.drafted_ns = 200;
  r.verified_ns = 300;
  r.end_ns = 400;
  return r;
}

void CheckFormat() {
  const std::string line = g4::FormatCycleRecord(Sample());
  Require(line.find('\n') == std::string::npos, "a record spans lines");
  const gufo::json::Value v = gufo::json::parse(line);
  Require(v.member_size("forward") == 7 && v.member_size("sessions") == 2,
          "forward/sessions");
  Require(v.member_size("position") == 31463 && v.member_size("rows") == 10,
          "position/rows");
  Require(v.member_size("drafter_steps") == 8 && v.member_size("emitted") == 2,
          "drafter steps/emitted");
  Require(v.find("small_head") != nullptr && v.find("small_head")->is_bool(),
          "small head");
  Require(v.member_double("priced_verify_ms") == 152.5 &&
              v.member_double("expected") == 3.5,
          "prices");
  const auto* estimates = v.find("estimates");
  Require(
      estimates != nullptr && estimates->is_array() && estimates->size() == 2,
      "estimates");
  const auto* ups = v.find("runner_ups");
  Require(ups != nullptr && ups->size() == 2 &&
              v.find("runner_up_acceptance")->size() == 2 &&
              v.find("row_acceptance")->size() == 2 &&
              v.find("proposal_signals")->size() == 3 &&
              v.find("proposal_q")->size() == 3 &&
              v.member_double("pending_entropy") == 1.5,
          "row labels and runner-ups");
  Require(
      v.find("observed")->size() == 2 && v.find("sibling_observed")->is_bool(),
      "observations");
  const auto* stop = v.find("stop");
  Require(stop != nullptr && stop->member_double("gain") == 20.5 &&
              stop->member_double("cost") == 51.25,
          "stop decision");
  Require(v.member_size("end_ns") == 400, "times");
  g4::CycleRecord none;
  const gufo::json::Value w = gufo::json::parse(g4::FormatCycleRecord(none));
  Require(w.find("stop") != nullptr && w.find("stop")->is_null(),
          "no stop is null");
}

void CheckFile() {
  const auto path = std::filesystem::temp_directory_path() /
                    ("gemma4_cycle_trace_" +
                     std::to_string(g4::CycleTrace::Now()) + ".jsonl");
  {
    g4::CycleTrace trace(path.string());
    trace.Write(Sample());
  }
  {
    g4::CycleTrace trace(path.string());
    trace.Write(g4::CycleRecord{});
  }
  std::ifstream in(path);
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);) {
    lines.push_back(line);
  }
  std::filesystem::remove(path);
  Require(lines.size() == 2, "the trace does not append");
  Require(gufo::json::parse(lines[0]).member_size("rows") == 10,
          "first record");
  bool threw = false;
  try {
    g4::CycleTrace bad("/nonexistent-dir/trace.jsonl");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Require(threw, "an unwritable trace opened");
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckFormat();
    CheckFile();
  });
}

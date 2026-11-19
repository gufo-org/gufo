#include "src/core/diagnostics/gpu_queues.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/platform/os_info.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

/// Mirrors the KFD layout: one directory per process, one per queue, each with
/// a `type` file holding a bare digit and no trailing newline.
class FakeKfd {
public:
  FakeKfd()
      : root_(std::filesystem::temp_directory_path() /
              ("gufo-queues-" + std::to_string(gufo::platform::ProcessId()))) {
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(proc_root());
  }

  ~FakeKfd() { std::filesystem::remove_all(root_); }

  FakeKfd(const FakeKfd&) = delete;
  FakeKfd& operator=(const FakeKfd&) = delete;

  void AddProcess(int pid, int compute, int sdma) {
    const auto queues = proc_root() / std::to_string(pid) / "queues";
    std::filesystem::create_directories(queues);
    int index = 0;
    for (int i = 0; i < compute; ++i, ++index) {
      Write(queues / std::to_string(index), "0");
    }
    for (int i = 0; i < sdma; ++i, ++index) {
      Write(queues / std::to_string(index), "1");
    }
  }

  [[nodiscard]] const std::filesystem::path& sys_root() const { return root_; }

private:
  [[nodiscard]] std::filesystem::path proc_root() const {
    return root_ / "class" / "kfd" / "kfd" / "proc";
  }

  static void Write(const std::filesystem::path& queue, std::string_view type) {
    std::filesystem::create_directories(queue);
    std::ofstream file(queue / "type", std::ios::binary);
    file << type;
  }

  std::filesystem::path root_;
};

using gufo::diagnostics::kComputeQueueBudget;
using gufo::diagnostics::kTextOwnQueues;
using gufo::diagnostics::PlanQueues;
using gufo::diagnostics::QueryQueueCensus;
using gufo::diagnostics::QueueCensus;
using gufo::diagnostics::QueueProfile;

void CensusCountsComputeQueuesAcrossProcesses() {
  FakeKfd kfd;
  kfd.AddProcess(101, 5, 1);
  kfd.AddProcess(202, 2, 1);
  const auto census = QueryQueueCensus(kfd.sys_root());
  Expect(census.compute_queues == 7, "compute queues across processes");
  Expect(census.sdma_queues == 2, "SDMA queues counted apart from compute");
  Expect(census.processes == 2, "processes holding queues");
}

void CensusIsEmptyWithoutKfd() {
  const auto census =
      QueryQueueCensus(std::filesystem::temp_directory_path() / "gufo-absent");
  Expect(census.compute_queues == 0, "absent KFD reports no queues");
  Expect(census.processes == 0, "absent KFD reports no processes");
}

void ProfilesPickTheirMeasuredCap() {
  const QueueCensus empty;
  const auto text = PlanQueues(QueueProfile::kText, empty, nullptr);
  Expect(text.cap == 2, "text server asks for two queues");
  Expect(text.own_queues == kTextOwnQueues,
         "text server restores on a queue of its own where it has one");
  Expect(text.expected_queues == 3 + kTextOwnQueues,
         "text server resolves to three pooled queues and its own");
  Expect(!text.exceeds_budget, "text server fits an empty device");

  const auto audio = PlanQueues(QueueProfile::kAudio, empty, nullptr);
  Expect(audio.cap == 1, "audio server asks for one queue");
  Expect(audio.expected_queues == 2, "audio server resolves to two queues");
}

void UnmeasuredProfileKeepsTheRuntimeDefault() {
  const QueueCensus empty;
  const auto plan = PlanQueues(QueueProfile::kUnmeasured, empty, nullptr);
  Expect(plan.cap == 0, "unmeasured modality leaves the variable unset");
  Expect(!plan.exceeds_budget, "unmeasured modality fits an empty device");
}

void UnmeasuredProfileIsClampedWhenRoomIsShort() {
  QueueCensus busy;
  busy.compute_queues = 5;
  const auto plan = PlanQueues(QueueProfile::kUnmeasured, busy, nullptr);
  Expect(plan.cap == 2, "unmeasured modality clamps into the free slots");
  Expect(plan.observed_queues + plan.expected_queues <= kComputeQueueBudget,
         "clamped unmeasured modality stays inside the budget");
}

void CapsClampDownToFreeSlots() {
  QueueCensus busy;
  busy.compute_queues = 5;  // a resident text server
  const auto audio = PlanQueues(QueueProfile::kAudio, busy, nullptr);
  Expect(audio.cap == 1, "audio server keeps its cap when it already fits");
  Expect(!audio.exceeds_budget, "audio server fits beside a text server");

  QueueCensus tight;
  tight.compute_queues = 6;
  const auto text = PlanQueues(QueueProfile::kText, tight, nullptr);
  Expect(text.cap == 1, "text server clamps down to the free slots");
  Expect(text.expected_queues == 2, "clamped text server takes the floor");
  Expect(!text.exceeds_budget, "clamped text server still fits");
}

/// A text server's own restore queue fits beside its cap or is given up
/// first, so every mix of servers that fit before it still fits.
void TextOwnQueueIsGivenUpBeforeTheCap() {
  QueueCensus other_text;
  other_text.compute_queues = 4;
  const auto second = PlanQueues(QueueProfile::kText, other_text, nullptr);
  Expect(second.cap == 2 && second.own_queues == kTextOwnQueues,
         "two text servers each keep two queues and their own");
  Expect(!second.may_exceed_budget, "two text servers fit the budget");

  QueueCensus crowded;
  crowded.compute_queues = 5;
  const auto third = PlanQueues(QueueProfile::kText, crowded, nullptr);
  Expect(third.cap == 2 && third.own_queues == 0,
         "the own queue goes before the cap");
  Expect(third.expected_queues == 3 && !third.may_exceed_budget,
         "without it the server fits as before");

  QueueCensus tight;
  tight.compute_queues = 6;
  const auto last = PlanQueues(QueueProfile::kText, tight, nullptr);
  Expect(last.cap == 1 && last.own_queues == 0,
         "a tight device clamps the cap without an own queue");

  QueueCensus busy;
  busy.compute_queues = 3;
  const auto operator_plan = PlanQueues(QueueProfile::kText, busy, "4");
  Expect(operator_plan.own_queues == 0,
         "an operator cap that fills the budget leaves no own queue");
  const auto audio = PlanQueues(QueueProfile::kAudio, QueueCensus{}, nullptr);
  Expect(audio.own_queues == 0, "only text servers take an own queue");
}

void CapsAreNeverRaisedToFillFreeSlots() {
  const QueueCensus empty;
  const auto audio = PlanQueues(QueueProfile::kAudio, empty, nullptr);
  Expect(audio.cap == 1,
         "an empty device does not raise the audio cap, so throughput does not"
         " depend on start order");
}

void FullDeviceWarnsInsteadOfRefusing() {
  QueueCensus full;
  full.compute_queues = kComputeQueueBudget;
  const auto plan = PlanQueues(QueueProfile::kAudio, full, nullptr);
  Expect(plan.cap == 1, "a full device still yields a usable cap");
  Expect(plan.expected_queues == 2, "the floor is two queues per process");
  Expect(plan.exceeds_budget, "a full device reports budget pressure");
}

void OperatorValueIsNeverOverridden() {
  QueueCensus busy;
  busy.compute_queues = 3;
  const auto plan = PlanQueues(QueueProfile::kText, busy, "4");
  Expect(plan.operator_supplied, "an operator setting is reported");
  Expect(plan.cap == 0, "an operator setting is left in place");
  Expect(plan.operator_cap == 4, "the operator's value is parsed");
  Expect(!plan.may_exceed_budget, "three queues plus five still fits");

  QueueCensus full;
  full.compute_queues = 7;
  const auto pressured = PlanQueues(QueueProfile::kText, full, "4");
  Expect(pressured.operator_supplied, "an operator setting is still reported");
  Expect(pressured.exceeds_budget,
         "budget pressure is reported even when the operator chose the cap");
}

/// The case the floor alone misses: five resident queues plus an operator cap
/// of four lands at nine, while five plus the two-queue floor is only seven.
void OperatorValuePastTheBudgetIsWarned() {
  QueueCensus busy;
  busy.compute_queues = 5;
  const auto plan = PlanQueues(QueueProfile::kAudio, busy, "4");
  Expect(plan.operator_cap == 4, "the operator's value is parsed");
  Expect(plan.expected_queues == 5,
         "a cap of four bounds five resident queues");
  Expect(!plan.exceeds_budget,
         "the floor alone would fit, so the certain check stays quiet");
  Expect(plan.may_exceed_budget,
         "the operator's own cap does not fit and is warned about");

  const auto line = gufo::diagnostics::DescribeQueuePressure(plan);
  Expect(line.find("adds_up_to=5") != std::string::npos,
         "the warning carries the upper bound that does not fit");
  Expect(line.find("cap=operator:4") != std::string::npos,
         "the warning names the operator's cap");
}

void UnparseableOperatorValueFallsBackToTheFloor() {
  QueueCensus busy;
  busy.compute_queues = 5;
  const auto plan = PlanQueues(QueueProfile::kAudio, busy, "yes");
  Expect(plan.operator_supplied, "a non-numeric value is still an override");
  Expect(plan.operator_cap == 0, "a non-numeric value parses to nothing");
  Expect(plan.expected_queues == 2, "only the floor can be reasoned about");
  Expect(!plan.may_exceed_budget, "the floor fits beside five queues");

  const auto line = gufo::diagnostics::DescribeQueuePlan("tts", plan);
  Expect(line.find("expected_max=unknown") != std::string::npos,
         "an unparseable value bounds nothing");
}

void ChosenCapsNeverWarnWhileTheyFit() {
  QueueCensus busy;
  busy.compute_queues = 5;
  const auto plan = PlanQueues(QueueProfile::kAudio, busy, nullptr);
  Expect(!plan.may_exceed_budget,
         "a cap this process chose always fits, so it never warns");
  Expect(plan.exceeds_budget == plan.may_exceed_budget,
         "both checks agree for a cap this process chose");
}

void EmptyOperatorValueIsIgnored() {
  const QueueCensus empty;
  const auto plan = PlanQueues(QueueProfile::kText, empty, "");
  Expect(!plan.operator_supplied, "an empty value is not an operator setting");
  Expect(plan.cap == 2, "an empty value falls back to the measured cap");
}

void PlanIsDescribedForTheLog() {
  const QueueCensus empty;
  const auto plan = PlanQueues(QueueProfile::kText, empty, nullptr);
  const auto line = gufo::diagnostics::DescribeQueuePlan("llm", plan);
  Expect(line.find("event=queue_budget") != std::string::npos,
         "log line carries the event name");
  Expect(line.find("kind=llm") != std::string::npos, "log line carries a kind");
  Expect(line.find("cap=2") != std::string::npos, "log line carries the cap");
  Expect(line.find("expected_max=" + std::to_string(3 + kTextOwnQueues)) !=
             std::string::npos,
         "log line carries the resident count");
  Expect(
      line.find("own=" + std::to_string(kTextOwnQueues)) != std::string::npos,
      "log line carries the own queues");
}

void OperatorPlanDoesNotPredictAQueueCount() {
  const QueueCensus empty;
  const auto plan = PlanQueues(QueueProfile::kText, empty, "4");
  const auto line = gufo::diagnostics::DescribeQueuePlan("llm", plan);
  Expect(line.find("cap=operator:4") != std::string::npos,
         "an operator setting is named in the log with its value");
  Expect(line.find("expected_max=" + std::to_string(5 + kTextOwnQueues)) !=
             std::string::npos,
         "the operator's value and the own queue bound the resident count");
}

void ProviderCapReplacesTheProfileCap() {
  const QueueCensus empty;
  const auto plan = PlanQueues(QueueProfile::kAudio, empty, nullptr, 2);
  Expect(plan.cap == 2, "provider asks for two queues");
  Expect(plan.expected_queues == 3, "provider cap resolves to three queues");
  Expect(!plan.may_exceed_budget, "provider cap fits an empty device");

  const auto undeclared = PlanQueues(QueueProfile::kAudio, empty, nullptr, 0);
  Expect(undeclared.cap == 1, "providers that declare nothing keep one queue");
}

void ProviderCapIsClampedToFreeSlots() {
  QueueCensus tight;
  tight.compute_queues = 6;
  const auto plan = PlanQueues(QueueProfile::kAudio, tight, nullptr, 2);
  Expect(plan.cap == 1, "provider cap clamps down to the free slots");
  Expect(!plan.exceeds_budget, "clamped provider cap still fits");
}

void OperatorValueOverridesProviderCap() {
  const QueueCensus empty;
  const auto plan = PlanQueues(QueueProfile::kAudio, empty, "1", 2);
  Expect(plan.operator_supplied, "operator value wins over the provider");
  Expect(plan.cap == 0, "operator plan leaves the variable to the operator");
  Expect(plan.operator_cap == 1, "operator value is kept");
}

}  // namespace

int main() {
  CensusCountsComputeQueuesAcrossProcesses();
  CensusIsEmptyWithoutKfd();
  ProfilesPickTheirMeasuredCap();
  UnmeasuredProfileKeepsTheRuntimeDefault();
  UnmeasuredProfileIsClampedWhenRoomIsShort();
  CapsClampDownToFreeSlots();
  TextOwnQueueIsGivenUpBeforeTheCap();
  CapsAreNeverRaisedToFillFreeSlots();
  FullDeviceWarnsInsteadOfRefusing();
  OperatorValueIsNeverOverridden();
  OperatorValuePastTheBudgetIsWarned();
  UnparseableOperatorValueFallsBackToTheFloor();
  ChosenCapsNeverWarnWhileTheyFit();
  EmptyOperatorValueIsIgnored();
  PlanIsDescribedForTheLog();
  OperatorPlanDoesNotPredictAQueueCount();
  ProviderCapReplacesTheProfileCap();
  ProviderCapIsClampedToFreeSlots();
  OperatorValueOverridesProviderCap();
  std::cout << "gpu_queues_test: all checks passed\n";
  return 0;
}

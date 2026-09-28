#include "src/core/diagnostics/gpu_queues.h"

#include <charconv>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>

namespace gufo::diagnostics {
namespace {

constexpr const char* kQueueCapEnvironment = "GPU_MAX_HW_QUEUES";

/// KFD writes a bare "0" or "1" with no trailing newline, so the file is read
/// whole and trimmed rather than parsed a line at a time.
std::string ReadTrimmed(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return {};
  }
  std::string value((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r' ||
                            value.back() == ' ')) {
    value.pop_back();
  }
  return value;
}

std::uint32_t PreferredCap(QueueProfile profile) {
  switch (profile) {
    case QueueProfile::kText:
      return 2;
    case QueueProfile::kAudio:
      return 1;
    case QueueProfile::kUnmeasured:
      return 0;
  }
  return 0;
}

}  // namespace

QueueCensus QueryQueueCensus(const std::filesystem::path& sys_root) {
  QueueCensus census;
  const auto root = sys_root / "class" / "kfd" / "kfd" / "proc";
  std::error_code error;
  if (!std::filesystem::is_directory(root, error)) {
    return census;
  }
  for (const auto& process : std::filesystem::directory_iterator(
           root, std::filesystem::directory_options::skip_permission_denied,
           error)) {
    const auto queues = process.path() / "queues";
    if (!std::filesystem::is_directory(queues, error)) {
      continue;
    }
    bool counted = false;
    for (const auto& queue : std::filesystem::directory_iterator(
             queues, std::filesystem::directory_options::skip_permission_denied,
             error)) {
      const auto type = ReadTrimmed(queue.path() / "type");
      if (type == "0") {
        ++census.compute_queues;
        counted = true;
      } else if (type == "1") {
        ++census.sdma_queues;
        counted = true;
      }
    }
    if (counted) {
      ++census.processes;
    }
  }
  return census;
}

QueuePlan PlanQueues(QueueProfile profile, const QueueCensus& census,
                     const char* operator_value) {
  QueuePlan plan;
  plan.observed_queues = census.compute_queues;

  if (operator_value != nullptr && *operator_value != '\0') {
    // A deliberate operator setting is never overridden, but it is still
    // checked: a cap of N resolves to at most N + 1 resident queues, which is
    // enough to tell the operator their choice will not fit.
    plan.operator_supplied = true;
    const std::string_view value(operator_value);
    std::uint32_t parsed = 0;
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error == std::errc{} && end == value.data() + value.size() &&
        parsed > 0) {
      plan.operator_cap = parsed;
      plan.expected_queues = parsed + 1;
    } else {
      // Unparseable: only the floor every process pays can be reasoned about.
      plan.expected_queues = kMinimumProcessQueues;
    }
    plan.exceeds_budget =
        census.compute_queues + kMinimumProcessQueues > kComputeQueueBudget;
    plan.may_exceed_budget =
        census.compute_queues + plan.expected_queues > kComputeQueueBudget;
    return plan;
  }

  const std::uint32_t room = kComputeQueueBudget > census.compute_queues
                                 ? kComputeQueueBudget - census.compute_queues
                                 : 0;
  // A cap of N resolves to at most N + 1 resident queues, so N + 1 has to fit.
  const std::uint32_t fits = room > kMinimumProcessQueues ? room - 1 : 1;
  const std::uint32_t preferred = PreferredCap(profile);

  if (preferred == 0) {
    if (room >= kRuntimeDefaultCap + 1) {
      // Room to spare and no measurement to justify a cap: leave the runtime
      // default alone.
      plan.expected_queues = kRuntimeDefaultCap + 1;
      return plan;
    }

    plan.cap = fits;
  } else {
    plan.cap = preferred < fits ? preferred : fits;
  }
  if (plan.cap < 1) {
    plan.cap = 1;
  }
  plan.expected_queues = plan.cap + 1;
  // The cap was chosen to fit, so it only fails when even the floor does not.
  plan.exceeds_budget =
      census.compute_queues + plan.expected_queues > kComputeQueueBudget;
  plan.may_exceed_budget = plan.exceeds_budget;
  return plan;
}

void ApplyQueuePlan(const QueuePlan& plan) {
  if (plan.cap == 0) {
    return;
  }
  // Never clobbers an existing value; the operator case returns a zero cap.
  (void)::setenv(kQueueCapEnvironment, std::to_string(plan.cap).c_str(), 0);
}

std::string DescribeQueuePlan(std::string_view kind, const QueuePlan& plan) {
  std::string message = "event=queue_budget kind=";
  message += kind;
  message += " observed=" + std::to_string(plan.observed_queues);
  message += " budget=" + std::to_string(kComputeQueueBudget);
  if (plan.operator_supplied && plan.operator_cap == 0) {
    // An unparseable value bounds nothing beyond the floor every process pays.
    message += " expected_max=unknown cap=operator";
  } else if (plan.operator_supplied) {
    message += " expected_max=" + std::to_string(plan.expected_queues);
    message += " cap=operator:" + std::to_string(plan.operator_cap);
  } else if (plan.cap == 0) {
    message += " expected_max=" + std::to_string(plan.expected_queues);
    message += " cap=runtime";
  } else {
    message += " expected_max=" + std::to_string(plan.expected_queues);
    message += " cap=" + std::to_string(plan.cap);
  }
  return message;
}

std::string DescribeQueuePressure(const QueuePlan& plan) {
  std::string message = "event=queue_budget_exceeded observed=";
  message += std::to_string(plan.observed_queues);
  message += " budget=" + std::to_string(kComputeQueueBudget);
  if (plan.exceeds_budget) {
    // Even the floor every process pays does not fit, so this is certain.
    message += " adds_at_least=" + std::to_string(kMinimumProcessQueues);
  } else {
    // The operator's cap is honoured whether or not it fits; the upper bound
    // is what says it will not.
    message += " adds_up_to=" + std::to_string(plan.expected_queues);
    message += " cap=operator:" + std::to_string(plan.operator_cap);
  }
  message +=
      " detail=the GPU reports a busy engine at its top clock while idle,"
      " about 26 W above idle, until a process holding queues exits";
  return message;
}

}  // namespace gufo::diagnostics

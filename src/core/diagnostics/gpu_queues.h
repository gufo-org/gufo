#ifndef GUFO_CORE_DIAGNOSTICS_GPU_QUEUES_H_
#define GUFO_CORE_DIAGNOSTICS_GPU_QUEUES_H_

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace gufo::diagnostics {

/// Hardware compute queues the gfx1151 command processor keeps mapped without
/// time-slicing. Past this total, counted across every process on the device,
/// the firmware scheduler cycles the queues even when all of them are empty:
/// the GPU then reports a busy engine at its top shader clock and draws about
/// 26 W above idle until one of the processes exits.
inline constexpr std::uint32_t kComputeQueueBudget = 8;

/// Compute queues any process holds once it dispatches at all. The first
/// dispatch claims two regardless of GPU_MAX_HW_QUEUES, so a device fits four
/// HIP processes at most, however modest each one is.
inline constexpr std::uint32_t kMinimumProcessQueues = 2;

/// Queues the HIP runtime pools by default. Used only to decide whether an
/// unmeasured modality has room to keep the runtime's own default.
inline constexpr std::uint32_t kRuntimeDefaultCap = 4;

/// Compute and SDMA queues open across every process on the device.
struct QueueCensus {
  std::uint32_t compute_queues{0};
  std::uint32_t sdma_queues{0};
  std::uint32_t processes{0};
};

/// Counts the queues every process on the device currently holds, including
/// processes this one does not own: the KFD nodes are world-readable, so a
/// llama.cpp server or a stray benchmark is visible here too. Reads no HIP
/// state and is safe to call before the runtime initialises.
[[nodiscard]] QueueCensus QueryQueueCensus(
    const std::filesystem::path& sys_root = "/sys");

/// Server modality, which decides how many hardware queues to ask for.
enum class QueueProfile : std::uint8_t {
  /// serve llm: measured flat from the runtime default down to a single queue,
  /// at one and at four concurrent requests.
  kText,
  /// serve tts and serve asr: stages run in sequence and each ends in a stream
  /// synchronise, so extra queues have no independent work to overlap.
  kAudio,
  /// serve image and serve video: never measured, so the runtime default
  /// stands unless the budget is too tight to hold it.
  kUnmeasured,
};

struct QueuePlan {
  /// Compute queues already held when this process started.
  std::uint32_t observed_queues{0};
  /// GPU_MAX_HW_QUEUES to request. Zero leaves the variable untouched.
  std::uint32_t cap{0};
  /// Upper bound on the queues this process will add. A process that creates
  /// fewer streams than its cap allows settles below this.
  std::uint32_t expected_queues{0};
  /// The operator set GPU_MAX_HW_QUEUES; their value is never overridden.
  bool operator_supplied{false};
  /// The operator's value, when it parses as a queue count. Zero means it does
  /// not, and only the floor can be reasoned about.
  std::uint32_t operator_cap{0};
  /// Even the two-queue floor does not fit: the device will idle busy.
  bool exceeds_budget{false};
  /// The upper bound does not fit, so the device may idle busy. Always set
  /// when exceeds_budget is. For a cap this process chose the two agree, since
  /// it only ever picks a cap that fits; they differ for an operator value,
  /// which is honoured whether or not it fits.
  bool may_exceed_budget{false};
};

/// Chooses a cap for this modality, clamped down to what the device has left.
/// The cap is only ever lowered to fit, never raised to fill free slots, so a
/// server's throughput does not depend on the order the servers were started.
[[nodiscard]] QueuePlan PlanQueues(QueueProfile profile,
                                   const QueueCensus& census,
                                   const char* operator_value);

/// Exports GPU_MAX_HW_QUEUES for a plan that asks for a cap. Must run before
/// the process makes its first HIP dispatch: the runtime reads the variable
/// once, grows its queue pool to the cap, and then releases nothing for the
/// lifetime of the process - neither hipStreamDestroy nor hipDeviceReset gives
/// a hardware queue back.
void ApplyQueuePlan(const QueuePlan& plan);

/// Renders the plan as a log line for the queue-budget event.
[[nodiscard]] std::string DescribeQueuePlan(std::string_view kind,
                                            const QueuePlan& plan);

/// Renders the warning shown when the device is already at its budget.
[[nodiscard]] std::string DescribeQueuePressure(const QueuePlan& plan);

}  // namespace gufo::diagnostics

#endif  // GUFO_CORE_DIAGNOSTICS_GPU_QUEUES_H_

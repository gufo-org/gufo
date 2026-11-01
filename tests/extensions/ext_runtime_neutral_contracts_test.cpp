// Neutral contracts the extension runtime owns after Card 09 (RFC step 6).
//
// This executable links only `gufo_ext_runtime` among Gufo libraries: if the
// continuation bases, host budgets, reasoning tags or the default device probe
// needed a serving or HTTP object, this link would fail.
//
// The capture/backing contract is cache-owned
// (`src/cache/component_resources.hpp` and `cache::Adapter::BackingPlan`); its
// coverage lives in cache_adapter_test, because gufo_ext_runtime does not link
// the cache tier.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <gufo/ext/adapter.hpp>
#include <gufo/ext/cache.hpp>
#include <gufo/ext/cancellable_gate.hpp>
#include <gufo/ext/continuation.hpp>
#include <gufo/ext/crypto.hpp>
#include <gufo/ext/device_probe.hpp>
#include <gufo/ext/draft_backend.hpp>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/host.hpp>
#include <gufo/ext/host_budget.hpp>
#include <gufo/ext/image.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/ledger.hpp>
#include <gufo/ext/mapped_prefetch.hpp>
#include <gufo/ext/model_config.hpp>
#include <gufo/ext/options.hpp>
#include <gufo/ext/platform/fs.hpp>
#include <gufo/ext/platform/library.hpp>
#include <gufo/ext/platform/mapped_file.hpp>
#include <gufo/ext/platform/os_info.hpp>
#include <gufo/ext/platform/pages.hpp>
#include <gufo/ext/platform/prefault.hpp>
#include <gufo/ext/platform/publish.hpp>
#include <gufo/ext/prompt_lookup.hpp>
#include <gufo/ext/reasoning.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/sampling.hpp>
#include <gufo/ext/session_mode.hpp>
#include <gufo/ext/sha256.hpp>
#include <gufo/ext/speculative_verifier.hpp>
#include <gufo/ext/text_contracts.hpp>
#include <gufo/ext/types.hpp>
#include <gufo/ext/utf8.hpp>
#include <gufo/ext/version.hpp>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "FAILED: %s\n", what);
  }
}

// The cache only ever sees these bases: it cannot inspect model bytes.
class FakeState final : public gufo::ext::ContinuationState {
public:
  [[nodiscard]] bool invalidated() const noexcept { return invalidated_; }
  void Invalidate() noexcept override { invalidated_ = true; }

private:
  bool invalidated_{false};
};

class FakeSnapshot final : public gufo::ext::ContinuationSnapshot {
public:
  explicit FakeSnapshot(const gufo::ext::ContinuationState* preferred)
      : preferred_(preferred) {}
  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return 4096;
  }
  [[nodiscard]] bool PrefersState(
      const gufo::ext::ContinuationState& state) const noexcept override {
    return &state == preferred_;
  }

private:
  const gufo::ext::ContinuationState* preferred_;
};

void CheckContinuationBases() {
  FakeState state;
  FakeSnapshot snapshot(&state);
  gufo::ext::ContinuationState& base = state;
  gufo::ext::ContinuationSnapshot& snapshot_base = snapshot;
  Expect(snapshot_base.PayloadBytes() == 4096, "snapshot payload bytes");
  Expect(snapshot_base.PrefersState(base), "snapshot prefers its own state");
  FakeSnapshot unrelated(nullptr);
  Expect(!unrelated.PrefersState(base), "unrelated snapshot prefers nothing");
  base.Invalidate();
  Expect(state.invalidated(), "Invalidate reaches the model implementation");

  const gufo::ext::ContinuationInputPrefix prefixes[] = {{4, {1, 2}},
                                                         {8, {3, 4}}};
  const std::vector<std::uint8_t> complete{9, 9};
  const auto inside = gufo::ext::PrefixInputIdentity(
      complete, std::span<const gufo::ext::ContinuationInputPrefix>(prefixes),
      3);
  Expect(inside.size() == 2 && inside[0] == 1, "prefix identity inside");
  const auto after = gufo::ext::PrefixInputIdentity(
      complete, std::span<const gufo::ext::ContinuationInputPrefix>(prefixes),
      9);
  Expect(after.size() == 2 && after[0] == 9, "complete identity after bounds");
  static_assert(sizeof(gufo::ext::ContinuationToken) == sizeof(std::uint32_t),
                "token id stays a 32-bit id");
}

void CheckHostBudgets() {
  const auto meminfo = gufo::ext::MeminfoAvailableBytes(
      "MemTotal:       131072000 kB\n"
      "MemFree:          102400 kB\n"
      "MemAvailable:    8000000 kB\n");
  Expect(meminfo.has_value() && *meminfo == 8000000ULL * 1024,
         "MemAvailable parses in bytes");
  Expect(!gufo::ext::MeminfoAvailableBytes("MemTotal: 4000 kB\n"),
         "missing MemAvailable stays nullopt");
  // The two budget functions each sample live RAM, so they are checked against
  // the relation their formulas imply -- budget = available / 2 and ceiling =
  // max(0, available - headroom) -- rather than a ratio between independent
  // samples. A host or cgroup at or below the headroom legitimately reports a
  // zero ceiling, which no ratio assertion may treat as a failure; the
  // allowance covers the drift between the two reads.
  const auto budget =
      static_cast<std::uint64_t>(gufo::ext::HostSnapshotBudgetBytes());
  const auto ceiling =
      static_cast<std::uint64_t>(gufo::ext::HostSnapshotCeilingBytes());
  constexpr std::uint64_t kDriftBytes = std::uint64_t{256} << 20;
  if (ceiling == 0) {
    Expect(budget * 2 <= gufo::ext::kHostSnapshotHeadroomBytes + kDriftBytes,
           "a zero ceiling only appears at or below the headroom");
  } else {
    const auto implied = ceiling + gufo::ext::kHostSnapshotHeadroomBytes;
    const auto gap =
        budget * 2 > implied ? budget * 2 - implied : implied - budget * 2;
    Expect(gap <= 2 * kDriftBytes,
           "both budgets come from the same host measurement");
  }
  Expect(gufo::ext::kHostSnapshotHeadroomBytes == (std::uint64_t{4} << 30),
         "headroom stays 4 GiB");
}

void CheckReasoningTags() {
  const gufo::ext::ReasoningTags tags{"<think>", "</think>", {"</think>"}};
  Expect(tags.start == "<think>" && tags.forced_end == "</think>",
         "reasoning tags carry their literals");
  Expect(tags.ends.size() == 1 && tags.ends[0] == "</think>",
         "reasoning ends list");
}

void CheckDeviceProbe() {
  // The header exposes no device types, so this compiles with or without HIP.
  gufo::ext::DeviceProbe probe;
  const auto status = probe.Poll();
  Expect(status == gufo::ext::DeviceProbeStatus::kUsable ||
             status == gufo::ext::DeviceProbeStatus::kPending ||
             status == gufo::ext::DeviceProbeStatus::kLost,
         "probe reports a defined status");
  // A probe still pending at its bound reports the device usable, so a slow
  // queue never cancels a request that has not failed.
  Expect(probe.WaitUsable(std::chrono::seconds(0), "test"),
         "pending work still counts as usable at the bound");
}

void CheckExtensionApiVersion() {
  Expect(gufo::ext::kApiMajor == 0 && gufo::ext::kApiMinor == 1 &&
             gufo::ext::kApiRevision == 3,
         "API numeric constants match 0.1.3");
  Expect(gufo::ext::kApiVersion == "0.1.3",
         "API version literal matches 0.1.3");
  Expect(gufo::ext::RuntimeApiVersion() == "0.1.3",
         "Runtime API version matches 0.1.3");
}

void CheckPlatformLeaves() {
  Expect(gufo::platform::PageSizeBytes() > 0,
         "platform os_info page size is positive");
  Expect(gufo::platform::AvailableMemoryBytes() > 0,
         "platform os_info available memory query is positive");
}

}  // namespace

int main() {
  CheckContinuationBases();
  CheckHostBudgets();
  CheckReasoningTags();
  CheckDeviceProbe();
  CheckExtensionApiVersion();
  CheckPlatformLeaves();
  return failures == 0 ? 0 : 1;
}

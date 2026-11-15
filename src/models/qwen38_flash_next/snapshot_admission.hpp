#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_SNAPSHOT_ADMISSION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_SNAPSHOT_ADMISSION_HPP_

#include <atomic>
#include <cstddef>
#include <memory>
#include <utility>

#include "src/families/qwen/vision/admission.hpp"

namespace gufo::models::qwen38_flash_next {

// Optional device copies compete with images and cache growth at arrival.
// Two mutable-state spares already belong to admitted deferred execution.
// Their local ledger assigns that allowance; physical commitment reduces its
// outstanding execution obligation until storage is about to be freed.
class SnapshotAdmission {
public:
  SnapshotAdmission(std::shared_ptr<qwen::vision::AdmissionBudget> budget,
                    std::size_t protected_spare_bytes)
      : budget_(std::move(budget)),
        spares_({protected_spare_bytes, protected_spare_bytes, 0, 0}) {}

  class Charge {
  public:
    Charge() = default;
    // Restore the execution obligation BEFORE freeing protected storage. A
    // fresh observation may otherwise lend those bytes to optional work in
    // the interval between the physical free and accounting cleanup.
    void BeforeRelease() noexcept {
      if (state_)
        state_->BeforeRelease();
    }

  private:
    friend class SnapshotAdmission;
    struct State {
      cache::ResourceCharge resource;
      std::shared_ptr<qwen::vision::AdmissionBudget> budget;
      std::atomic<std::size_t> execution_bytes{};
      void BeforeRelease() noexcept {
        if (const auto bytes = execution_bytes.exchange(0))
          budget->ReleaseExecution(bytes);
      }
      ~State() { BeforeRelease(); }
    };
    std::shared_ptr<State> state_;
  };

  // Call only after allocation has synchronized. Conversion failure leaves
  // the caller responsible for freeing its physical allocation before the
  // pending reservation unwinds. Slot zero uses execution_bytes directly;
  // it was included in deferred scratch and has no optional reservation.
  [[nodiscard]] Charge Commit(cache::ResourceReservation& reservation,
                              std::size_t execution_bytes = 0) {
    if (!OnDemand())
      return {};
    Charge result;
    result.state_ = std::make_shared<Charge::State>();
    result.state_->budget = budget_;
    if (reservation) {
      const auto info = reservation.Info();
      if (info.category == cache::ResourceCategory::kPrivateState)
        execution_bytes = info.bytes;
      result.state_->resource = reservation.Convert();
    }
    if (execution_bytes) {
      budget_->CommitExecution(execution_bytes);
      result.state_->execution_bytes.store(execution_bytes);
    }
    return result;
  }

  [[nodiscard]] bool OnDemand() const noexcept { return bool(budget_); }

  [[nodiscard]] cache::ResourceReservation Reserve(std::size_t bytes,
                                                   bool mutable_state) {
    if (!OnDemand() || !bytes)
      return {};
    if (mutable_state) {
      try {
        return spares_.Reserve(cache::ResourceCategory::kPrivateState, bytes);
      } catch (const cache::ResourceExhausted&) {
        // Active captures are not bounded by the number of retained spares.
        // Additional copies need capacity from the same ledger as images.
      }
    }
    return budget_->ReserveShared(bytes);
  }

private:
  std::shared_ptr<qwen::vision::AdmissionBudget> budget_;
  cache::ResourceLedger spares_;
};

}  // namespace gufo::models::qwen38_flash_next

#endif

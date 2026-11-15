#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_EXECUTION_ADMISSION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_EXECUTION_ADMISSION_HPP_

#include <cstddef>
#include <memory>
#include <utility>

#include "src/families/qwen/vision/admission.hpp"

namespace gufo::models::qwen38_flash_next {

// Tracks the materialized part of an already accepted text/deferred bound.
// Increasing commitment follows successful physical allocation. Decreasing
// it must precede the physical free, restoring that execution obligation so
// optional image/cache/snapshot allocations cannot borrow it in between.
class ExecutionCommitment {
public:
  explicit ExecutionCommitment(
      std::shared_ptr<qwen::vision::AdmissionBudget> budget = {})
      : budget_(std::move(budget)) {}
  ~ExecutionCommitment() { Release(); }
  ExecutionCommitment(const ExecutionCommitment&) = delete;
  ExecutionCommitment& operator=(const ExecutionCommitment&) = delete;
  void SetCommitted(std::size_t bytes) {
    if (!budget_)
      return;
    if (bytes > bytes_)
      budget_->CommitExecution(bytes - bytes_);
    else
      budget_->ReleaseExecution(bytes_ - bytes);
    bytes_ = bytes;
  }
  void Release() noexcept {
    if (budget_)
      budget_->ReleaseExecution(std::exchange(bytes_, 0));
  }
  [[nodiscard]] std::size_t Bytes() const noexcept { return bytes_; }

private:
  std::shared_ptr<qwen::vision::AdmissionBudget> budget_;
  std::size_t bytes_{};
};

}  // namespace gufo::models::qwen38_flash_next
#endif

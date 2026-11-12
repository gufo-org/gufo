#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "src/families/qwen/vision/admission.hpp"
#include "src/models/qwen38_flash_next/copy_width_admission.hpp"

namespace gufo::models::qwen38_flash_next {

// A model belongs to at most one serving coordinator. In-process runners cannot
// inherit its text-only claims or bypass its configured execution owner.
template<class Model>
void RequireUncoordinatedModel(const Model& model) {
  if (model.HasVisionMemoryBudget())
    throw std::logic_error("model already has a vision admission owner");
}

struct ServingVisionAdmission {
  qwen::vision::StartupMemoryPlan plan;
  std::shared_ptr<qwen::vision::AdmissionBudget> budget;
  std::size_t configured_text_bytes;
  std::optional<CopyWidthChoice> copy_width;
};

// Called only by the factory that owns model loading and knows the configured
// session count, before any slots or snapshots exist. Encoder::Open resolves
// both explicit and discovered projectors: the loaded model, not a path option,
// selects admission. The model-backed factory retains conservative claims.
template<class Model>
[[nodiscard]] std::optional<ServingVisionAdmission> InitializeServingVision(
    Model& model, std::uint32_t max_context, bool use_mtp, std::size_t sessions,
    std::size_t device_capacity,
    qwen::vision::AdmissionBudget::RuntimeObserver observe_runtime) {
  RequireUncoordinatedModel(model);
  if (!model.VisionEncoder())
    return std::nullopt;
  if (sessions == 0)
    throw std::invalid_argument("vision admission requires a session count");
  const auto mode = use_mtp ? core::SessionMode::kSpeculative
                            : core::SessionMode::kAutoregressive;
  std::optional<CopyWidthChoice> copy_width;
  if (use_mtp) {
    copy_width = SelectOnDemandCopyWidth([&](std::uint32_t rows) {
      model.LimitCopyRows(rows);
      return CopyWidthMemory{model.TextSessionBytes(mode, max_context),
                             model.DeferredScratchBytes()};
    });
  }
  const auto plan = qwen::vision::PlanStartupMemory(
      device_capacity, model.DeferredScratchBytes(),
      model.TextSessionBytes(mode, max_context),
      model.SessionBytes(mode, max_context), true);
  const auto text =
      qwen::vision::CheckedMultiply(plan.per_session_bytes, sessions);
  // Accepted execution obligations are not a new host/cache startup reserve.
  // Physical commitment reduces pending work; cache/image allocations reserve
  // only their arriving storage against the shared runtime observer.
  auto budget = std::make_shared<qwen::vision::AdmissionBudget>(
      std::move(observe_runtime),
      qwen::vision::CheckedAdd(text, plan.shared_reservation_bytes));
  model.SetVisionMemoryBudget(budget);
  return ServingVisionAdmission{plan, std::move(budget), text, copy_width};
}

}  // namespace gufo::models::qwen38_flash_next

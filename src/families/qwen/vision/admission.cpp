#include "src/families/qwen/vision/admission.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

#include "src/core/image.hpp"

namespace gufo::models::qwen::vision {
namespace {
std::size_t PatchCount(ImageDimensions image) {
  if (!image.width || !image.height || image.width % 32 || image.height % 32 ||
      std::uint64_t{image.width} * image.height > 16777216) {
    throw std::invalid_argument("vision encoder requires a resized RGB32 grid");
  }
  return CheckedMultiply(image.width / 16, image.height / 16);
}
}  // namespace

std::size_t CheckedAdd(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw core::ResourceAdmissionError(
        "vision memory admission refused: reason=arithmetic_overflow");
  return a + b;
}

std::size_t CheckedMultiply(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw core::ResourceAdmissionError(
        "vision memory admission refused: reason=arithmetic_overflow");
  return a * b;
}

std::size_t ImageScratchBytes(ImageDimensions image) {
  const auto patches = PatchCount(image);
  const auto tile = EncoderTileRows(patches);
  // encoder.hip: pixels, hidden, work, normalized, QKV and attention are
  // 768 + 4608 + 17216 + 8608 + 6912 + 2304 bytes per patch. Scores and
  // probabilities are 16 heads * tile * patches * (4 + 2) bytes.
  const auto per_patch =
      16 * 16 * 3 + kEncoderHidden * sizeof(float) +
      std::max(kEncoderFfn, 3 * kEncoderHidden) * sizeof(float) +
      std::max(kEncoderFfn, kEncoderPatchDim) * sizeof(std::uint16_t) +
      3 * kEncoderHidden * sizeof(std::uint16_t) +
      kEncoderHidden * sizeof(std::uint16_t) +
      kEncoderHeads * tile * (sizeof(float) + sizeof(std::uint16_t));
  auto bytes = CheckedMultiply(patches, per_patch);
  bytes = CheckedAdd(bytes, kEncoderHidden * tile * sizeof(float));
  // Upload stages through explicit pinned host memory on the unified device.
  return CheckedAdd(bytes,
                    std::min<std::size_t>(768 * patches, kPinnedUploadBytes));
}

std::size_t ImageEmbeddingBytes(ImageDimensions image,
                                std::uint32_t output_width) {
  if (!output_width)
    throw std::invalid_argument("vision embedding width must be positive");
  // Four ViT patches merge to one FP32 output embedding.
  return CheckedMultiply(PatchCount(image), output_width);
}

AdmissionBudget::AdmissionBudget(RuntimeObserver observer,
                                 std::size_t execution_bytes)
    : capacity_(0),
      observer_(std::move(observer)),
      execution_pending_(execution_bytes),
      ledger_({std::numeric_limits<std::size_t>::max(),
               std::numeric_limits<std::size_t>::max(), 0, 0}) {
  if (!observer_)
    throw std::invalid_argument(
        "on-demand admission requires a memory observer");
}

cache::ResourceReservation AdmissionBudget::Reserve(std::size_t bytes) {
  return Reserve(cache::ResourceCategory::kPrivateState, bytes);
}

cache::ResourceReservation AdmissionBudget::ReserveShared(std::size_t bytes) {
  return Reserve(cache::ResourceCategory::kPrivateTail, bytes);
}

cache::ResourceReservation AdmissionBudget::Reserve(
    cache::ResourceCategory category, std::size_t bytes) {
  if (!bytes)
    return {};
  const std::lock_guard lock(mutex_);
  const auto pending = PendingLocked();
  // Capture pending first. Conversion after this point can only make this
  // attempt conservative: the old pending amount remains deducted even if
  // the following observation also sees the newly committed allocation.
  if (observer_) {
    const auto available = observer_();
    const auto capacity =
        available > execution_pending_ ? available - execution_pending_ : 0;
    capacity_.store(capacity);
    if (pending > capacity || bytes > capacity - pending)
      throw core::ResourceAdmissionError(
          "vision memory admission refused: reason=capacity requested_bytes=" +
          std::to_string(bytes) + " outstanding_bytes=" +
          std::to_string(ledger_.Snapshot().total_bytes) +
          " pending_bytes=" + std::to_string(pending) +
          " capacity_bytes=" + std::to_string(capacity) +
          " execution_pending_bytes=" + std::to_string(execution_pending_));
  }
  try {
    return ledger_.Reserve(category, bytes);
  } catch (const cache::ResourceExhausted&) {
    throw core::ResourceAdmissionError(
        "vision memory admission refused: reason=capacity requested_bytes=" +
        std::to_string(bytes) +
        " outstanding_bytes=" + std::to_string(ledger_.Snapshot().total_bytes) +
        " capacity_bytes=" + std::to_string(capacity_.load()));
  }
}

std::size_t AdmissionBudget::Outstanding() const {
  return ledger_.Snapshot().total_bytes;
}

std::size_t AdmissionBudget::PendingLocked() const {
  const auto snapshot = ledger_.Snapshot();
  std::size_t pending = 0;
  for (const auto bytes : snapshot.reserved_bytes)
    pending += bytes;  // Each is a subset of the overflow-checked ledger total.
  return pending - materialized_;
}

std::size_t AdmissionBudget::PendingBytes() const {
  const std::lock_guard lock(mutex_);
  return PendingLocked();
}

std::size_t AdmissionBudget::ProtectedBytes() const {
  const std::lock_guard lock(mutex_);
  return execution_pending_;
}

std::size_t AdmissionBudget::AccountedBytes() const {
  const std::lock_guard lock(mutex_);
  return CheckedAdd(execution_pending_, ledger_.Snapshot().total_bytes);
}

void AdmissionBudget::CommitExecution(std::size_t bytes) {
  const std::lock_guard lock(mutex_);
  if (bytes > execution_pending_)
    throw std::logic_error("execution commitment exceeds admitted obligations");
  execution_pending_ -= bytes;
}

void AdmissionBudget::ReleaseExecution(std::size_t bytes) noexcept {
  const std::lock_guard lock(mutex_);
  execution_pending_ += bytes;
}

void AdmissionBudget::ReplaceCommitted(std::size_t old_bytes,
                                       std::size_t new_bytes) noexcept {
  const std::lock_guard lock(mutex_);
  materialized_ = materialized_ - old_bytes + new_bytes;
}

void AdmissionBudget::ClaimExecutionOwner() {
  if (execution_owner_.test_and_set(std::memory_order_acq_rel))
    throw core::ResourceAdmissionError(
        "vision memory admission refused: reason=execution_owner "
        "the on-demand runner already belongs to an execution pool");
}

PreparationReservation::PreparationReservation(
    std::shared_ptr<AdmissionBudget> budget, std::size_t image_count) {
  static_assert(core::kMaxImagePixels == std::size_t{32} * 1024 * 1024,
                "revisit the decode working bound if image limits change");
  if (!budget || !image_count)
    return;
  const auto pixels = CheckedMultiply(image_count, kPreparedImageBytes);
  (void)CheckedAdd(pixels, kPreparationScratchBytes);
  scratch_ = budget->Reserve(kPreparationScratchBytes);
  pixels_ = budget->Reserve(pixels);
}

void PreparationReservation::Finish() noexcept {
  scratch_ = {};
}

void PreparationReservation::Finish(std::size_t pixel_bytes) {
  if (pixels_) {
    pixels_.Shrink(pixel_bytes);
    if (pixels_)
      committed_pixels_ = pixels_.Convert();
  }
  Finish();
}

AdmissionLease::AdmissionLease(std::shared_ptr<AdmissionBudget> budget)
    : budget_(std::move(budget)) {}

AdmissionLease::~AdmissionLease() {
  Clear();
}

AdmissionLease::AdmissionLease(AdmissionLease&& other) noexcept
    : budget_(std::move(other.budget_)),
      reservations_(std::move(other.reservations_)),
      bytes_(std::exchange(other.bytes_, 0)),
      committed_(std::exchange(other.committed_, 0)) {}

AdmissionLease& AdmissionLease::operator=(AdmissionLease&& other) noexcept {
  if (this != &other) {
    Clear();
    budget_ = std::move(other.budget_);
    reservations_ = std::move(other.reservations_);
    bytes_ = std::exchange(other.bytes_, 0);
    committed_ = std::exchange(other.committed_, 0);
  }
  return *this;
}

void AdmissionLease::SetCommitted(std::size_t bytes) noexcept {
  if (!budget_)
    return;
  // A caller may leave partially allocated work pending on failure, but must
  // never claim commitment for storage outside this owner's reservation.
  if (bytes > bytes_)
    std::terminate();
  budget_->ReplaceCommitted(committed_, bytes);
  committed_ = bytes;
}

void AdmissionLease::Grow(std::size_t bytes) {
  if (!budget_ || bytes <= bytes_)
    return;
  auto reservation = budget_->Reserve(bytes - bytes_);
  // vector growth may fail too; the uncommitted local reservation rolls back.
  reservations_.push_back({std::move(reservation), bytes - bytes_});
  bytes_ = bytes;
}

void AdmissionLease::Shrink(std::size_t bytes) noexcept {
  if (!budget_ || bytes >= bytes_)
    return;
  SetCommitted(0);
  while (bytes_ > bytes) {
    auto& part = reservations_.back();
    const auto released = std::min(part.bytes, bytes_ - bytes);
    part.bytes -= released;
    if (!part.bytes)
      reservations_.pop_back();
    else
      part.reservation.Shrink(part.bytes);
    bytes_ -= released;
  }
}

void AdmissionLease::Clear() noexcept {
  SetCommitted(0);
  reservations_.clear();
  bytes_ = 0;
}

const char* StartupCapacity::LimiterName() const noexcept {
  if (device_admission_bytes == host_ceiling_bytes)
    return "equal";
  return device_admission_bytes < host_ceiling_bytes ? "device" : "host";
}

StartupCapacity SelectStartupCapacity(std::size_t device_admission_bytes,
                                      std::size_t host_ceiling_bytes) {
  return {device_admission_bytes, host_ceiling_bytes,
          std::min(device_admission_bytes, host_ceiling_bytes)};
}

bool StartupMemoryPlan::Fits(std::size_t sessions) const noexcept {
  return !per_session_bytes ||
         sessions <= state_capacity_bytes / per_session_bytes;
}

std::size_t StartupMemoryPlan::RequiredBytes(std::size_t sessions) const {
  return CheckedAdd(CheckedMultiply(per_session_bytes, sessions),
                    shared_reservation_bytes);
}

std::size_t StartupMemoryPlan::DeficitBytes(std::size_t sessions,
                                            std::size_t available) const {
  const auto required = RequiredBytes(sessions);
  return required > available ? required - available : 0;
}

StartupMemoryPlan PlanStartupMemory(std::size_t available_bytes,
                                    std::size_t deferred_scratch,
                                    std::size_t base_per_session,
                                    std::size_t conservative_per_session,
                                    bool on_demand) {
  return {available_bytes > deferred_scratch
              ? available_bytes - deferred_scratch
              : 0,
          on_demand ? base_per_session : conservative_per_session,
          deferred_scratch};
}

}  // namespace gufo::models::qwen::vision

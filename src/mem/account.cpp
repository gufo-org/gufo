#include "src/mem/account.hpp"

#include <algorithm>
#include <limits>

#include "src/mem/ledger.hpp"

namespace gufo::mem {

std::size_t SegmentStride(const RegionShape& shape) noexcept {
  return shape.segment_stride ? shape.segment_stride
                              : shape.rows * shape.row_bytes;
}

std::size_t RegionPageBytes(const RegionShape& shape,
                            std::size_t rows) noexcept {
  rows = std::min(rows, shape.rows);
  if (rows == 0 || shape.row_bytes == 0)
    return 0;
  const std::size_t stride = SegmentStride(shape);
  const std::size_t run = rows * shape.row_bytes;
  std::size_t pages = 0, previous_end = 0;
  for (std::size_t s = 0; s < shape.segments; ++s) {
    const std::size_t first = std::max(s * stride / kPageBytes, previous_end);
    const std::size_t end = (s * stride + run + kPageBytes - 1) / kPageBytes;
    pages += end > first ? end - first : 0;
    previous_end = std::max(previous_end, end);
  }
  return pages * kPageBytes;
}

SessionFit FitSessions(std::size_t sessions, std::size_t session_bytes,
                       std::size_t demand_bytes,
                       std::size_t capacity) noexcept {
  SessionFit fit;
  if (session_bytes == 0 || sessions == 0)
    return fit;
  fit.full_sessions = capacity / session_bytes;
  if (sessions <= fit.full_sessions)
    return fit;
  fit.kind = SessionFit::Kind::kRefused;
  demand_bytes = std::min(demand_bytes, session_bytes);
  const std::size_t fixed = session_bytes - demand_bytes;
  if (demand_bytes == 0 || fit.full_sessions == 0 ||
      (fixed != 0 && sessions > capacity / fixed))
    return fit;
  const std::size_t held = fit.full_sessions * session_bytes;
  const std::size_t fixed_all = sessions * fixed;
  if (held < fixed_all || held - fixed_all < demand_bytes)
    return fit;
  fit.kind = SessionFit::Kind::kShared;
  fit.cap_bytes = held - fixed_all;
  return fit;
}

StateAccount::~StateAccount() {
  if (charged_ != 0)
    StateLedger::Process().Uncharge(charged_);
}

void StateAccount::AddRegion(const RegionShape& shape,
                             std::size_t tokens_per_row,
                             std::function<void(std::size_t)> trim) {
  std::lock_guard lock(mutex_);
  const std::size_t before = charged_;
  members_.push_back(
      {shape, std::max<std::size_t>(tokens_per_row, 1), std::move(trim)});
  // A region added after a claim covers the same need: charge its pages
  // too. Regions are added when the session is created, before any claim.
  (void)Recharge(before, nullptr);
}

std::size_t StateAccount::RowsFor(std::size_t tokens,
                                  std::size_t tokens_per_row,
                                  std::size_t rows) const noexcept {
  if (tokens == 0)
    return 0;
  tokens_per_row = std::max<std::size_t>(tokens_per_row, 1);
  const std::size_t limit =
      std::numeric_limits<std::size_t>::max() - margin_tokens_ - tokens_per_row;
  const std::size_t total = std::min(tokens, limit) + margin_tokens_;
  return std::min(rows, (total + tokens_per_row - 1) / tokens_per_row);
}

std::size_t StateAccount::BytesForLocked(std::size_t tokens) const {
  std::size_t bytes = 0;
  for (const auto& member : members_)
    bytes += RegionPageBytes(
        member.shape,
        RowsFor(tokens, member.tokens_per_row, member.shape.rows));
  return bytes;
}

std::size_t StateAccount::BytesFor(std::size_t tokens) const {
  std::lock_guard lock(mutex_);
  return BytesForLocked(tokens);
}

std::size_t StateAccount::full_bytes() const {
  std::lock_guard lock(mutex_);
  std::size_t bytes = 0;
  for (const auto& member : members_)
    bytes += RegionPageBytes(member.shape, member.shape.rows);
  return bytes;
}

std::size_t StateAccount::ChargeLocked() const {
  return std::max(BytesForLocked(need_), mapped_);
}

bool StateAccount::Recharge(std::size_t before, StateReservation* reservation) {
  const std::size_t after = ChargeLocked();
  if (after <= before) {
    if (after < before)
      StateLedger::Process().Uncharge(before - after);
    charged_ = after;
    return true;
  }
  std::size_t delta = after - before;
  const std::size_t taken = reservation ? reservation->Take(delta) : 0;
  delta -= taken;
  if (delta != 0 && !StateLedger::Process().TryCharge(delta)) {
    // Hand back what the reservation gave: it still covers the request.
    if (reservation)
      reservation->Return(taken);
    return false;
  }
  charged_ = after;
  return true;
}

bool StateAccount::Claim(std::size_t tokens, StateReservation* reservation) {
  std::lock_guard lock(mutex_);
  if (tokens <= need_)
    return true;
  const std::size_t before = charged_;
  const std::size_t previous = need_;
  need_ = tokens;
  if (Recharge(before, reservation))
    return true;
  need_ = previous;
  return false;
}

bool StateAccount::SetNeed(std::size_t tokens) {
  std::lock_guard lock(mutex_);
  const std::size_t before = charged_;
  const std::size_t previous = need_;
  need_ = tokens;
  if (Recharge(before, nullptr))
    return true;
  need_ = previous;
  return false;
}

bool StateAccount::OnMap(std::size_t bytes) {
  std::lock_guard lock(mutex_);
  const std::size_t before = charged_;
  mapped_ += bytes;
  if (Recharge(before, nullptr))
    return true;
  mapped_ -= bytes;
  return false;
}

void StateAccount::OnUnmap(std::size_t bytes) noexcept {
  std::lock_guard lock(mutex_);
  const std::size_t before = charged_;
  mapped_ -= std::min(bytes, mapped_);
  (void)Recharge(before, nullptr);
}

std::size_t StateAccount::TrimTo(std::size_t tokens) {
  std::vector<std::pair<std::function<void(std::size_t)>, std::size_t>> trims;
  std::size_t before_mapped = 0;
  {
    std::lock_guard lock(mutex_);
    before_mapped = mapped_;
    const std::size_t before = charged_;
    need_ = std::min(need_, tokens);
    (void)Recharge(before, nullptr);
    for (const auto& member : members_)
      if (member.trim)
        trims.emplace_back(member.trim, RowsFor(tokens, member.tokens_per_row,
                                                member.shape.rows));
  }
  // Regions report their unmaps through OnUnmap.
  for (const auto& [trim, rows] : trims)
    trim(rows);
  std::lock_guard lock(mutex_);
  return before_mapped > mapped_ ? before_mapped - mapped_ : 0;
}

std::size_t StateAccount::need_tokens() const {
  std::lock_guard lock(mutex_);
  return need_;
}

std::size_t StateAccount::mapped_bytes() const {
  std::lock_guard lock(mutex_);
  return mapped_;
}

std::size_t StateAccount::charged_bytes() const {
  std::lock_guard lock(mutex_);
  return charged_;
}

std::size_t StateAccount::tail_bytes() const {
  std::lock_guard lock(mutex_);
  const std::size_t need = BytesForLocked(need_);
  return mapped_ > need ? mapped_ - need : 0;
}

StateReservation::StateReservation(std::size_t bytes) {
  ok_ = bytes == 0 || StateLedger::Process().TryCharge(bytes);
  bytes_ = ok_ ? bytes : 0;
}

StateReservation& StateReservation::operator=(
    StateReservation&& other) noexcept {
  if (this != &other) {
    Release();
    bytes_ = other.bytes_;
    ok_ = other.ok_;
    other.bytes_ = 0;
  }
  return *this;
}

std::size_t StateReservation::Take(std::size_t bytes) noexcept {
  const std::size_t taken = std::min(bytes, bytes_);
  bytes_ -= taken;
  return taken;
}

void StateReservation::Return(std::size_t bytes) noexcept {
  bytes_ += bytes;
}

void StateReservation::Release() noexcept {
  if (bytes_ != 0)
    StateLedger::Process().Uncharge(bytes_);
  bytes_ = 0;
}

}  // namespace gufo::mem

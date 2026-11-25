#include "src/models/inkling2/cache_adapter.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace gufo::models::inkling2 {
namespace {

using cache::Rows;
using cache::TransferResult;

/// Every copy of this adapter is synchronous (on the engine's default stream),
/// so a completion is born settled.
class DoneSignal final : public cache::CompletionSignal {
public:
  explicit DoneSignal(TransferResult r) : result_(r) {}
  [[nodiscard]] bool Ready() const noexcept override { return true; }
  [[nodiscard]] TransferResult Wait() noexcept override { return result_; }

private:
  TransferResult result_;
};
cache::Completion Done(TransferResult r = TransferResult::kSucceeded) {
  return cache::Completion(std::make_unique<DoneSignal>(r));
}

// Adapted from Gufo (MIT), src/models/qwen/continuation_adapter.cpp (v0.35.0,
// 0260e11): the check that a set of component ranges covers [first, end)
// without a gap.
bool Covered(std::vector<std::pair<std::size_t, std::size_t>> ranges,
             std::size_t first, std::size_t end) {
  std::ranges::sort(ranges);
  std::size_t frontier = first;
  for (const auto& [a, b] : ranges) {
    if (a > frontier)
      return false;
    frontier = std::max(frontier, b);
  }
  return frontier >= end;
}

}  // namespace

// ---- slot
// ------------------------------------------------------------------------------------

Inkling2Slot::Inkling2Slot(Inkling2CacheAdapter& adapter,
                           cache::MutationGuard& guard,
                           std::unique_ptr<Session> session)
    : adapter_(adapter), guard_(guard), session_(std::move(session)) {
  SessionStateAccess::SetHooks(*session_, this);
}

Inkling2Slot::~Inkling2Slot() {
  ReleaseAll();
  SessionStateAccess::SetHooks(*session_, nullptr);
}

void Inkling2Slot::BeforeWrite(std::size_t first, std::size_t end) {
  // Called by Engine::Forward before an ubatch writes rows [first, end). A slot
  // in the middle of a restore must not execute; a failed one neither.
  if (restoring || failed)
    throw std::logic_error("inkling2 slot is not ready for execution");
  const std::uint32_t ring = adapter_.layout_.ring_rows();
  for (const auto& c : adapter_.layout_.components()) {
    if (!HasRows(c.descriptor.kind))
      continue;
    guard_.BeforeOverwrite(c.descriptor.id, first, end);
    // A window ring also overwrites the rows it evicts: row r - ring for every
    // new row r.
    if (!c.global && end > ring) {
      const Rows a = std::max<Rows>(first, ring) - ring, b = end - ring;
      if (a < b)
        guard_.BeforeOverwrite(c.descriptor.id, a, b);
    }
  }
}

void Inkling2Slot::GuardAll() {
  const Rows rows = adapter_.engine_->max_context();
  for (const auto& c : adapter_.layout_.components())
    if (HasRows(c.descriptor.kind))
      guard_.BeforeOverwrite(c.descriptor.id, 0, rows);
}

void Inkling2Slot::ReleaseAll() noexcept {
  const Rows rows = adapter_.engine_->max_context();
  for (const auto& c : adapter_.layout_.components())
    if (HasRows(c.descriptor.kind))
      guard_.BeforeRelease(c.descriptor.id, 0, rows);
}

// ---- adapter
// ---------------------------------------------------------------------------------

Inkling2CacheAdapter::Inkling2CacheAdapter(std::shared_ptr<Engine> engine,
                                           cache::Identity identity,
                                           bool durable, std::size_t plan_step)
    : engine_(std::move(engine)),
      layout_(engine_->config(), engine_->ring_rows(), engine_->has_drafter()),
      identity_(std::move(identity)),
      durable_(durable),
      plan_step_(plan_step) {
  if (identity_.empty())
    throw std::invalid_argument("inkling2 cache identity must not be empty");
  if (plan_step_ == 0)
    throw std::invalid_argument("inkling2 prefill plan step must be positive");
  for (const auto& c : layout_.components())
    descriptors_.push_back(c.descriptor);
}

Inkling2Slot& Inkling2CacheAdapter::AsSlot(cache::Slot& slot) const {
  auto* s = dynamic_cast<Inkling2Slot*>(&slot);
  if (s == nullptr || &s->adapter_ != this)
    throw std::invalid_argument("slot belongs to another Inkling 2 adapter");
  return *s;
}

Session& Inkling2CacheAdapter::GetSession(cache::Slot& slot) const {
  auto& s = AsSlot(slot);
  if (!s.IsValid())
    throw std::logic_error("inkling2 slot is not ready for execution");
  return s.session();
}

std::unique_ptr<cache::Slot> Inkling2CacheAdapter::CreateSlot(
    cache::MutationGuard& guard) {
  return std::make_unique<Inkling2Slot>(*this, guard, engine_->NewSession());
}

std::vector<cache::ComponentPosition> Inkling2CacheAdapter::Positions(
    const cache::Slot& slot) const {
  auto& s = AsSlot(const_cast<cache::Slot&>(slot));
  if (!s.IsValid())
    throw std::logic_error("cannot query an invalid Inkling 2 slot");
  const std::size_t pos = s.boundary != nullptr
                              ? s.boundary->cp->position
                              : SessionStateAccess::Position(s.session());
  return layout_.Positions(pos);
}

std::vector<cache::ComponentPosition>
Inkling2CacheAdapter::PrefillCheckpointPositions(const cache::Slot& slot,
                                                 Rows boundary) const {
  auto& s = AsSlot(const_cast<cache::Slot&>(slot));
  if (!s.IsValid() || s.boundary != nullptr)
    throw std::logic_error("Inkling 2 slot is not ready for checkpoint plan");
  if (boundary <= SessionStateAccess::Position(s.session()) ||
      boundary > engine_->max_context())
    throw std::invalid_argument("unsupported Inkling 2 in-pass boundary");
  return layout_.Positions(boundary);
}

void Inkling2CacheAdapter::PrefillThrough(
    cache::Slot& slot, std::span<const std::int32_t> tokens,
    std::span<const std::size_t> boundaries, std::span<const Block> blocks,
    const std::function<void(std::size_t)>& consume, const PrefillYield* yield,
    std::span<const InputRows> rows) const {
  auto& s = AsSlot(slot);
  if (!s.IsValid() || s.boundary != nullptr)
    throw std::logic_error("Inkling 2 slot is not ready for PrefillThrough");
  if (!consume || boundaries.empty() || blocks.size() != boundaries.size())
    throw std::invalid_argument("invalid Inkling 2 PrefillThrough request");
  engine_->Forward(s.session(), tokens, {}, yield, rows, boundaries,
                   [&](std::size_t index, const Engine::PrefillCheckpoint& cp) {
                     const Inkling2Slot::BoundaryView view{&cp, blocks[index]};
                     struct Scope {
                       Inkling2Slot& slot;
                       const Inkling2Slot::BoundaryView* saved;
                       ~Scope() { slot.boundary = saved; }
                     } scope{s, s.boundary};
                     s.boundary = &view;
                     consume(index);
                   });
}

std::vector<Rows> Inkling2CacheAdapter::PlanPrefill(const cache::Slot& slot,
                                                    Rows first,
                                                    Rows end) const {
  auto& s = AsSlot(const_cast<cache::Slot&>(slot));
  if (!s.IsValid() || first > end || end > engine_->max_context())
    throw std::invalid_argument("invalid Inkling 2 prefill plan");
  std::vector<Rows> plan;
  while (first < end) {
    first += std::min<Rows>(end - first, plan_step_);
    plan.push_back(first);
  }
  return plan;
}

// ---- capture
// ---------------------------------------------------------------------------------

void Inkling2CacheAdapter::CapturePrivateInto(const Inkling2Slot& slot,
                                              const StateComponent& c,
                                              std::size_t offset,
                                              std::span<std::byte> out) const {
  const auto& session = slot.session();
  const auto total = c.descriptor.state_bytes;
  if (offset > total || out.size() > total - offset)
    throw std::invalid_argument("invalid Inkling 2 private capture range");
  if (out.empty())
    return;
  const auto* bv = slot.boundary;
  switch (c.part) {
    case Part::kMeta: {
      StateMeta m;
      if (bv != nullptr) {
        m.position = bv->cp->position;
        m.block = static_cast<std::uint32_t>(bv->block);
        m.logits_valid = 1U;
        m.ring_rows = layout_.ring_rows();
        if (layout_.drafter()) {
          m.draft_first = bv->cp->draft_first;
          m.draft_valid = bv->cp->draft_valid;
          m.draft_f_pos = bv->cp->draft_f_pos;
        }
      } else {
        m.position = SessionStateAccess::Position(session);
        m.block = static_cast<std::uint32_t>(slot.block);
        m.logits_valid = SessionStateAccess::HasLogits(session) ? 1U : 0U;
        m.ring_rows = layout_.ring_rows();
        if (layout_.drafter())
          SessionStateAccess::DraftOut(session, m);
      }
      std::memcpy(out.data(), reinterpret_cast<const std::byte*>(&m) + offset,
                  out.size());
      break;
    }
    case Part::kLogits: {
      if (bv != nullptr && !bv->cp->logits.empty()) {
        if (bv->cp->logits.size() * sizeof(float) != total)
          std::memset(out.data(), 0, out.size());
        else
          std::memcpy(
              out.data(),
              reinterpret_cast<const std::byte*>(bv->cp->logits.data()) +
                  offset,
              out.size());
        break;
      }
      std::vector<std::byte> all(total);
      SessionStateAccess::LogitsOut(session, all.data(), total);
      std::memcpy(out.data(), all.data() + offset, out.size());
      break;
    }
    case Part::kConv: {
      if (bv != nullptr && bv->cp->conv != nullptr) {
        SessionStateAccess::DeviceBytesOut(
            reinterpret_cast<const std::byte*>(bv->cp->conv) + offset,
            out.size(), out.data());
        break;
      }
      // Copy only the visits the piece touches.
      std::size_t done = 0;
      while (done < out.size()) {
        const std::size_t at = offset + done, v = at / kConvBytesPerVisit,
                          inner = at % kConvBytesPerVisit;
        const std::size_t n =
            std::min(out.size() - done, kConvBytesPerVisit - inner);
        SessionStateAccess::ConvVisitOut(session, static_cast<std::uint32_t>(v),
                                         inner, n, out.data() + done);
        done += n;
      }
      break;
    }
    case Part::kDraftF:
      if (bv != nullptr && bv->cp->draft_f != nullptr)
        SessionStateAccess::DeviceBytesOut(
            reinterpret_cast<const std::byte*>(bv->cp->draft_f) + offset,
            out.size(), out.data());
      else
        SessionStateAccess::DraftFOut(session, offset, out.size(), out.data());
      break;
    default:
      throw std::invalid_argument("component is not private state");
  }
}

cache::Completion Inkling2CacheAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream&) {
  const auto& c = layout_.Component(id);
  if (c.descriptor.kind != cache::ComponentKind::kPrivateState ||
      bytes.size() != c.descriptor.state_bytes)
    throw std::invalid_argument("incorrect Inkling 2 private buffer");
  auto& s = AsSlot(const_cast<cache::Slot&>(slot));
  if (!s.IsValid())
    throw std::logic_error("cannot read an invalid Inkling 2 slot");
  engine_->Synchronize();
  CapturePrivateInto(s, c, 0, bytes);
  return Done();
}

cache::Completion Inkling2CacheAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream&) {
  auto& s = AsSlot(const_cast<cache::Slot&>(slot));
  if (!s.IsValid())
    throw std::logic_error("cannot read an invalid Inkling 2 slot");
  const auto& c = layout_.Component(id);
  if (c.descriptor.kind != cache::ComponentKind::kPrivateState)
    throw std::invalid_argument("component is not private state");
  engine_->Synchronize();
  CapturePrivateInto(s, c, offset, bytes);
  return Done();
}

cache::Completion Inkling2CacheAdapter::CopyRowsOut(const cache::Slot& slot,
                                                    cache::ComponentId id,
                                                    Rows first, Rows end,
                                                    std::span<std::byte> bytes,
                                                    cache::Stream&) {
  auto& s = AsSlot(const_cast<cache::Slot&>(slot));
  if (!s.IsValid())
    throw std::logic_error("cannot read an invalid Inkling 2 slot");
  const auto& c = layout_.Component(id);
  if (!HasRows(c.descriptor.kind))
    throw std::invalid_argument("component has no rows");
  const Rows pos = s.boundary != nullptr
                       ? s.boundary->cp->position
                       : SessionStateAccess::Position(s.session());
  const Rows written = s.boundary != nullptr ? s.boundary->cp->valid_end : pos;
  const Rows oldest = c.global || written <= layout_.ring_rows()
                          ? 0
                          : written - layout_.ring_rows();
  if (first > end || end > pos || (first < end && first < oldest) ||
      bytes.size() != (end - first) * kKvRowBytes)
    throw std::invalid_argument("invalid Inkling 2 row capture");
  if (first == end)
    return Done();
  engine_->Synchronize();
  SessionStateAccess::RowsOut(s.session(), c.visit, c.part == Part::kValue,
                              first, end, bytes.data());
  return Done();
}

std::span<const std::byte> Inkling2CacheAdapter::BorrowRowsView(
    const cache::Slot& slot, cache::ComponentId id, Rows first,
    Rows end) const {
  // Global visits keep append rows at fixed addresses (demand-mapped regions):
  // a checkpoint may keep their pages. Window rings wrap and are copied.
  auto& s = AsSlot(const_cast<cache::Slot&>(slot));
  if (!s.IsValid())
    throw std::logic_error("cannot borrow rows of an invalid Inkling 2 slot");
  const auto& c = layout_.Component(id);
  if (!c.global || !HasRows(c.descriptor.kind))
    return {};
  const Rows pos = s.boundary != nullptr
                       ? s.boundary->cp->position
                       : SessionStateAccess::Position(s.session());
  if (first >= end || end > pos)
    throw std::invalid_argument("invalid Inkling 2 borrowed range");
  return SessionStateAccess::RowsView(s.session(), c.visit,
                                      c.part == Part::kValue, first, end);
}

// ---- restore
// ---------------------------------------------------------------------------------

void Inkling2CacheAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& s = AsSlot(slot);
  if (s.restoring || s.failed)
    throw std::logic_error("Inkling 2 slot is not ready for a restore");
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete Inkling 2 restore positions");
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& d : descriptors_) {
    if (std::ranges::count(positions, d.id, &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("missing or duplicate Inkling 2 frontier");
    ordered.push_back(
        *std::ranges::find(positions, d.id, &cache::ComponentPosition::id));
  }
  const auto target = ordered.front().valid_rows;
  if (target > engine_->max_context())
    throw std::invalid_argument("Inkling 2 frontier exceeds the context");
  if (ordered != layout_.Positions(target))
    throw std::invalid_argument("inconsistent Inkling 2 component frontiers");
  // Guard every range that will be replaced while the old state is still
  // readable.
  s.GuardAll();
  engine_->Synchronize();
  SessionStateAccess::PrepareRestore(s.session(), target);
  s.expected = std::move(ordered);
  s.coverage.assign(descriptors_.size(), {});
  s.meta = StateMeta{};
  s.meta.version = 0;  // unset until the metadata component is loaded
  s.logits.assign(layout_.vocab(), 0.0F);
  s.restoring = true;
}

cache::Completion Inkling2CacheAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
    std::span<const std::byte> bytes, cache::Stream&) {
  auto& s = AsSlot(slot);
  if (!s.restoring || s.failed)
    throw std::logic_error("Inkling 2 load requires an unfailed restore");
  try {
    const auto& c = layout_.Component(id);
    if (!HasRows(c.descriptor.kind))
      throw std::invalid_argument("component has no rows");
    const auto valid = s.expected[id.value - 1].valid_rows;
    const auto [lo, hi] = StateLayout::RestoreRange(c.descriptor, valid);
    if (first > end || end > hi || (first < end && first < lo) ||
        bytes.size() != (end - first) * kKvRowBytes)
      throw std::invalid_argument("invalid Inkling 2 row restore");
    if (first < end) {
      SessionStateAccess::RowsIn(s.session(), c.visit, c.part == Part::kValue,
                                 first, end, bytes.data());
      const std::lock_guard lock(s.restore_mutex);
      s.coverage[id.value - 1].emplace_back(first, end);
    }
    return Done();
  } catch (...) {
    s.failed = true;
    throw;
  }
}

cache::Completion Inkling2CacheAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  auto& s = AsSlot(slot);
  if (!s.restoring || s.failed)
    throw std::logic_error("Inkling 2 load requires an unfailed restore");
  if (bytes.size() != layout_.Component(id).descriptor.state_bytes) {
    s.failed = true;
    throw std::invalid_argument("incorrect Inkling 2 private buffer size");
  }
  return LoadPrivatePiece(slot, id, 0, bytes, stream);
}

cache::Completion Inkling2CacheAdapter::LoadPrivatePiece(
    cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<const std::byte> bytes, cache::Stream&) {
  auto& s = AsSlot(slot);
  if (!s.restoring || s.failed)
    throw std::logic_error("Inkling 2 load requires an unfailed restore");
  try {
    const auto& c = layout_.Component(id);
    const auto total = c.descriptor.state_bytes;
    if (c.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("invalid Inkling 2 private piece");
    switch (c.part) {
      case Part::kMeta:
        std::memcpy(reinterpret_cast<std::byte*>(&s.meta) + offset,
                    bytes.data(), bytes.size());
        break;
      case Part::kLogits:
        std::memcpy(reinterpret_cast<std::byte*>(s.logits.data()) + offset,
                    bytes.data(), bytes.size());
        break;
      case Part::kConv: {
        std::size_t done = 0;
        while (done < bytes.size()) {
          const std::size_t at = offset + done, v = at / kConvBytesPerVisit,
                            inner = at % kConvBytesPerVisit;
          const std::size_t n =
              std::min(bytes.size() - done, kConvBytesPerVisit - inner);
          SessionStateAccess::ConvVisitIn(s.session(),
                                          static_cast<std::uint32_t>(v), inner,
                                          n, bytes.data() + done);
          done += n;
        }
        break;
      }
      case Part::kDraftF:
        SessionStateAccess::DraftFIn(s.session(), offset, bytes.size(),
                                     bytes.data());
        break;
      default:
        throw std::invalid_argument("component is not private state");
    }
    if (!bytes.empty()) {
      const std::lock_guard lock(s.restore_mutex);
      s.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    }
    return Done();
  } catch (...) {
    s.failed = true;
    throw;
  }
}

bool Inkling2CacheAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& s = AsSlot(slot);
  if (!s.restoring)
    throw std::logic_error("Inkling 2 validation requires an active restore");
  const auto refuse = [&] {
    s.failed = true;
    return false;
  };
  if (s.failed || positions.size() != s.expected.size())
    return refuse();
  for (std::size_t i = 0; i < s.expected.size(); ++i) {
    if (std::ranges::count(positions, s.expected[i]) != 1)
      return refuse();
    const auto [lo, hi] =
        StateLayout::RestoreRange(descriptors_[i], s.expected[i].valid_rows);
    if (lo < hi && !Covered(s.coverage[i], lo, hi))
      return refuse();
  }
  const std::size_t target = s.expected.front().valid_rows;
  if (s.meta.version != StateMeta::kVersion || s.meta.position != target ||
      s.meta.ring_rows != layout_.ring_rows() ||
      s.meta.block > static_cast<std::uint32_t>(Block::kToolBody) ||
      s.meta.logits_valid > 1 || (s.meta.logits_valid == 0 && target != 0))
    return refuse();
  engine_->Synchronize();
  SessionStateAccess::SetFrontier(
      s.session(), target, s.meta.logits_valid ? s.logits.data() : nullptr,
      s.logits.size());
  if (layout_.drafter())
    SessionStateAccess::DraftIn(s.session(), s.meta);
  s.block = static_cast<Block>(s.meta.block);
  s.restoring = false;
  s.coverage.clear();
  s.expected.clear();
  s.logits.clear();
  s.logits.shrink_to_fit();
  return true;
}

bool Inkling2CacheAdapter::Invalidate(cache::Slot& slot) noexcept {
  try {
    auto& s = AsSlot(slot);
    s.ReleaseAll();
    engine_->Reset(s.session());
    s.block = Block::kIdle;
    s.failed = false;
    s.restoring = false;
    s.expected.clear();
    s.coverage.clear();
    s.guard_.AfterReset();
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::inkling2

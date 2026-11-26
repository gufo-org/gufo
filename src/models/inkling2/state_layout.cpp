#include "src/models/inkling2/state_layout.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <gufo/ext/snapshot_copy.hpp>
#include <shared_mutex>
#include <stdexcept>
#include <string>

#include "src/models/inkling2/engine.hpp"

namespace gufo::models::inkling2 {

StateLayout::StateLayout(const Inkling2Config& c, std::uint32_t ring_rows,
                         bool drafter)
    : ring_rows_(ring_rows),
      logits_bytes_(std::size_t{c.n_vocab_text} * 4),
      vocab_(c.n_vocab_text),
      drafter_(drafter) {
  if (ring_rows < kWindowRows + kRowsPerChunk - 1)
    throw std::invalid_argument(
        "inkling2: sliding-window ring smaller than a checkpoint keeps");
  const auto add = [&](Part part, cache::ComponentKind kind, std::size_t row,
                       std::size_t state, std::uint32_t visit, bool global) {
    const auto id = static_cast<std::uint32_t>(components_.size() + 1);
    cache::ComponentDescriptor d;
    d.id = {id};
    d.layout_version = 1;
    d.kind = kind;
    d.row_bytes = row;
    d.rows_per_chunk = row ? kRowsPerChunk : 0;
    d.state_bytes = state;
    d.window_rows = kind == cache::ComponentKind::kWindowRows ? kWindowRows : 0;
    components_.push_back({d, part, visit, global});
  };
  add(Part::kMeta, cache::ComponentKind::kPrivateState, 0, kMetaBytes, 0,
      false);
  add(Part::kLogits, cache::ComponentKind::kPrivateState, 0, logits_bytes_, 0,
      false);
  add(Part::kConv, cache::ComponentKind::kPrivateState, 0, kConvBytes, 0,
      false);
  for (std::uint32_t v = 0; v < kVisits; ++v) {
    const bool global = c.global_layer[c.PhysOfVisit(v)];
    const auto kind = global ? cache::ComponentKind::kAppendRows
                             : cache::ComponentKind::kWindowRows;
    add(Part::kKey, kind, kKvRowBytes, 0, v, global);
    add(Part::kValue, kind, kKvRowBytes, 0, v, global);
  }
  if (drafter) {
    add(Part::kDraftF, cache::ComponentKind::kPrivateState, 0, kDraftFBytes, 0,
        false);
    for (std::uint32_t l = 0; l < kDraftLayers; ++l) {
      const bool global = !Drafter::WindowLayer(static_cast<int>(l));
      const auto kind = global ? cache::ComponentKind::kAppendRows
                               : cache::ComponentKind::kWindowRows;
      add(Part::kKey, kind, kKvRowBytes, 0, kVisits + l, global);
      add(Part::kValue, kind, kKvRowBytes, 0, kVisits + l, global);
    }
  }
}

const StateComponent& StateLayout::Component(cache::ComponentId id) const {
  if (id.value == 0 || id.value > components_.size())
    throw std::invalid_argument("invalid Inkling 2 component id");
  return components_[id.value - 1];
}

std::vector<cache::ComponentPosition> StateLayout::Positions(
    std::size_t position) const {
  std::vector<cache::ComponentPosition> p;
  p.reserve(components_.size());
  for (const auto& c : components_)
    p.push_back({c.descriptor.id, position});
  return p;
}

std::pair<cache::Rows, cache::Rows> StateLayout::RestoreRange(
    const cache::ComponentDescriptor& d, cache::Rows position) {
  if (d.kind == cache::ComponentKind::kPrivateState)
    return {0, d.state_bytes};
  return {cache::FirstChunkRow(d, position), position};
}

std::vector<cache::BackingClass> StateLayout::BackingPlan(
    std::size_t ram) const {
  // Rows, private frontiers and tails share the same committed pages (as the
  // in-tree continuation layouts): one class of 64 KiB pages, 90% of the RAM
  // budget.
  constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
  const auto budget = ram > (4ULL << 20) ? (ram - (4ULL << 20)) / 10 * 9 : 0;
  if (budget <= header)
    return {};
  const auto blocks = (budget - header) / (page + per_block);
  if (!blocks)
    return {};
  return {{page, blocks, header + blocks * per_block}};
}

std::size_t StateLayout::PrivateBytes() const {
  std::size_t b = 0;
  for (const auto& c : components_)
    b += c.descriptor.state_bytes;
  return b;
}

std::size_t StateLayout::SessionBytes(std::uint32_t context) const {
  std::size_t b = kConvBytes + (drafter_ ? kDraftFBytes : 0);
  for (const auto& c : components_)
    if (HasRows(c.descriptor.kind))
      b += c.descriptor.row_bytes * (c.global ? context : ring_rows_);
  return b;
}

// ---------------------------------------------------------------------------------------------
// SessionStateAccess: the one place that knows the Session buffer layout.
namespace {
void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    // Clear it: the next launch check on this thread would report it.
    (void)hipGetLastError();
    throw std::runtime_error(std::string("inkling2 state copy: ") + what +
                             ": " + hipGetErrorString(e));
  }
}
// The cache's host buffers can be adjacent registrations, and HIP rejects a
// copy whose host range spans two; the core snapshot copies split there. They
// queue on the engine's default stream, so wait for them (after a failed
// submission too: callers reuse the buffers once a copy returns).
void Settle(hipError_t submitted, const char* what) {
  const hipError_t settled = hipStreamSynchronize(nullptr);
  Check(submitted != hipSuccess ? submitted : settled, what);
}
}  // namespace

SessionStateAccess::Stream SessionStateAccess::RowStream(const Session& s,
                                                         std::uint32_t visit,
                                                         bool value) {
  if (visit >= kVisits) {
    const std::uint32_t l = visit - kVisits;
    if (l >= kDraftLayers || s.draft_.k[l] == nullptr)
      throw std::invalid_argument("inkling2: no drafter layer for the visit");
    return {value ? s.draft_.v[l] : s.draft_.k[l], s.draft_.cap[l],
            !Drafter::WindowLayer(static_cast<int>(l))};
  }
  const auto& sl = s.slots_.at(visit);
  return {value ? sl.v : sl.k, sl.rows, sl.global};
}

std::size_t SessionStateAccess::Position(const Session& s) {
  return s.pos_;
}
bool SessionStateAccess::Global(const Session& s, std::uint32_t v) {
  return RowStream(s, v, false).global;
}
bool SessionStateAccess::HasLogits(const Session& s) {
  return s.logits_stale_ || !s.logits_.empty();
}
void SessionStateAccess::LogitsOut(const Session& s, std::byte* dst,
                                   std::size_t bytes) {
  (void)s.logits();  // copies device-held logits
  if (s.logits_.size() * 4 != bytes) {
    std::memset(dst, 0, bytes);
    return;
  }
  std::memcpy(dst, s.logits_.data(), bytes);
}

void SessionStateAccess::RowsOut(const Session& s, std::uint32_t visit,
                                 bool value, cache::Rows first, cache::Rows end,
                                 std::byte* dst) {
  const auto sl = RowStream(s, visit, value);
  const auto* base = static_cast<const std::byte*>(sl.base);
  const bool ring = !sl.global;
  if (!ring && end > sl.rows)
    throw std::invalid_argument("inkling2: row range beyond capacity");
  hipError_t copied = hipSuccess;
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  while (first < end && copied == hipSuccess) {
    const std::size_t slot_row = ring ? first % sl.rows : first;
    const std::size_t n =
        ring ? std::min<std::size_t>(end - first, sl.rows - slot_row)
             : end - first;
    copied = hip::CopySnapshotToHost(dst, base + slot_row * kKvRowBytes,
                                     n * kKvRowBytes, nullptr);
    dst += n * kKvRowBytes;
    first += n;
  }
  Settle(copied, "rows out");
}

void SessionStateAccess::RowsIn(Session& s, std::uint32_t visit, bool value,
                                cache::Rows first, cache::Rows end,
                                const std::byte* src) {
  // The drafter context is dropped until the restore validates (DraftIn).
  s.draft_.valid = 0;
  s.draft_.first = 0;
  const auto sl = RowStream(s, visit, value);
  auto* base = static_cast<std::byte*>(sl.base);
  const bool ring = !sl.global;
  if (!ring) {
    if (end > sl.rows)
      throw std::invalid_argument("inkling2: row range beyond capacity");
    s.EnsureRows(end);  // BeginRestore made the pages private
  }
  hipError_t copied = hipSuccess;
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  while (first < end && copied == hipSuccess) {
    const std::size_t slot_row = ring ? first % sl.rows : first;
    const std::size_t n =
        ring ? std::min<std::size_t>(end - first, sl.rows - slot_row)
             : end - first;
    copied = hip::CopySnapshotToDevice(base + slot_row * kKvRowBytes, src,
                                       n * kKvRowBytes, nullptr);
    src += n * kKvRowBytes;
    first += n;
  }
  Settle(copied, "rows in");
}

std::span<const std::byte> SessionStateAccess::RowsView(const Session& s,
                                                        std::uint32_t visit,
                                                        bool value,
                                                        cache::Rows first,
                                                        cache::Rows end) {
  const auto sl = RowStream(s, visit, value);
  if (!sl.global || first > end || end > sl.rows)
    return {};
  const auto* base = static_cast<const std::byte*>(sl.base);
  return {base + first * kKvRowBytes, (end - first) * kKvRowBytes};
}

void SessionStateAccess::PrepareRestore(Session& s, std::size_t rows) {
  s.PrepareRestore(rows);
}

void SessionStateAccess::DeviceBytesOut(const void* src, std::size_t n,
                                        std::byte* dst) {
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  Settle(hip::CopySnapshotToHost(dst, src, n, nullptr), "checkpoint out");
}

void SessionStateAccess::ConvVisitOut(const Session& s, std::uint32_t visit,
                                      std::size_t inner, std::size_t n,
                                      std::byte* dst) {
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  Settle(
      hip::CopySnapshotToHost(
          dst,
          reinterpret_cast<const std::byte*>(s.slots_.at(visit).conv) + inner,
          n, nullptr),
      "conv out");
}
void SessionStateAccess::ConvVisitIn(Session& s, std::uint32_t visit,
                                     std::size_t inner, std::size_t n,
                                     const std::byte* src) {
  // The drafter context is dropped until the restore validates (DraftIn).
  s.draft_.valid = 0;
  s.draft_.first = 0;
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  Settle(hip::CopySnapshotToDevice(
             reinterpret_cast<std::byte*>(s.slots_.at(visit).conv) + inner, src,
             n, nullptr),
         "conv in");
}
void SessionStateAccess::SetFrontier(Session& s, std::size_t position,
                                     const float* logits, std::size_t n) {
  s.pos_ = position;
  // The DSpark context of a shorter prefix stays valid; a restore (RowsIn /
  // ConvVisitIn) already dropped it.
  s.draft_.valid = std::min(s.draft_.valid, position);
  if (s.draft_.first > s.draft_.valid)
    s.draft_.first = s.draft_.valid = 0;
  if (logits != nullptr)
    s.logits_.assign(logits, logits + n);
  else
    s.logits_.clear();
  s.logits_stale_ = false;
}
void SessionStateAccess::SetHooks(Session& s, SessionHooks* h) {
  s.hooks_ = h;
}

void SessionStateAccess::DraftOut(const Session& s, StateMeta& m) {
  if (s.draft_.f == nullptr)
    return;
  m.draft_first = s.draft_.first;
  m.draft_valid = s.draft_.valid;
  m.draft_f_pos = s.draft_.f_pos;
}
void SessionStateAccess::DraftFOut(const Session& s, std::size_t inner,
                                   std::size_t n, std::byte* dst) {
  if (s.draft_.f == nullptr || inner + n > kDraftFBytes)
    throw std::invalid_argument("inkling2: no drafter state to capture");
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  Settle(hip::CopySnapshotToHost(
             dst, reinterpret_cast<const std::byte*>(s.draft_.f) + inner, n,
             nullptr),
         "draft out");
}
void SessionStateAccess::DraftFIn(Session& s, std::size_t inner, std::size_t n,
                                  const std::byte* src) {
  if (s.draft_.f == nullptr || inner + n > kDraftFBytes)
    throw std::invalid_argument("inkling2: no drafter state to restore");
  s.draft_.valid = 0;
  s.draft_.first = 0;
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  Settle(hip::CopySnapshotToDevice(
             reinterpret_cast<std::byte*>(s.draft_.f) + inner, src, n, nullptr),
         "draft in");
}
void SessionStateAccess::DraftIn(Session& s, const StateMeta& m) {
  // Only a context that reaches the frontier: rows a window still holds and
  // the Markov vector of the last token. Anything else restarts at the
  // frontier, as before.
  s.draft_.first = s.draft_.valid = 0;
  s.draft_.f_pos = static_cast<std::size_t>(-1);
  if (s.draft_.f == nullptr || m.draft_valid != s.pos_ ||
      m.draft_first > m.draft_valid || s.pos_ == 0 ||
      m.draft_f_pos != s.pos_ - 1)
    return;
  s.draft_.first = static_cast<std::size_t>(m.draft_first);
  s.draft_.valid = static_cast<std::size_t>(m.draft_valid);
  s.draft_.f_pos = static_cast<std::size_t>(m.draft_f_pos);
}

}  // namespace gufo::models::inkling2

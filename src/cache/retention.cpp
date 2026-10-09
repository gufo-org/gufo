#include "src/cache/retention.hpp"

#include <algorithm>
#include <stdexcept>

namespace gufo::cache {
namespace {
bool Prefix(std::span<const Token> a, std::span<const Token> b) {
  return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
}
bool Continuation(RetentionPurpose p) {
  return p == RetentionPurpose::kContinuation ||
         p == RetentionPurpose::kBranchPoint;
}
std::uint32_t MaxRank(RetentionPurpose p) {
  return p == RetentionPurpose::kRetry     ? 0
         : p == RetentionPurpose::kHistory ? 1
                                           : 3;
}
}  // namespace
std::vector<CaptureCandidate> PlanCaptures(const Adapter& adapter,
                                           const Slot& slot,
                                           const CapturePlanRequest& r) {
  if (r.reused > r.prompt || r.stable > r.prompt || r.common > r.prompt)
    throw std::invalid_argument("capture boundary outside prompt");
  const auto passes = adapter.PlanPrefill(slot, r.reused, r.prompt);
  Rows previous = r.reused;
  for (const auto end : passes) {
    if (end <= previous || end > r.prompt)
      throw std::invalid_argument("invalid adapter prefill plan");
    previous = end;
  }
  if (previous != r.prompt)
    throw std::invalid_argument("incomplete adapter prefill plan");
  std::vector<CaptureCandidate> result;
  const auto add = [&](Rows boundary, CheckpointPurpose purpose,
                       bool required) {
    if (!boundary || boundary < r.reused ||
        std::ranges::find(r.already_retained, boundary) !=
            r.already_retained.end())
      return;
    auto found =
        std::ranges::find(result, boundary, &CaptureCandidate::boundary);
    if (found != result.end()) {
      found->required |= required;
      if (purpose == CheckpointPurpose::kLearned && !found->required)
        found->purpose = purpose;
      return;
    }
    result.push_back(
        {boundary, purpose, required,
         boundary > r.reused && !std::ranges::binary_search(passes, boundary)});
  };
  add(r.reused, CheckpointPurpose::kPrompt, true);
  add(r.stable, CheckpointPurpose::kPrompt, true);
  add(r.prompt, CheckpointPurpose::kPrompt, true);
  if (r.grid && r.prompt) {
    const Rows grid = (r.prompt - 1) / 2048;
    const Rows count = std::min<Rows>(grid, 4);
    for (Rows i = 1; i <= count; ++i) {
      Rows position = (grid / count * i + grid % count * i / count) * 2048;
      auto pass = std::ranges::upper_bound(passes, position);
      if (pass != passes.begin() && position - *std::prev(pass) <= 128 &&
          *std::prev(pass) > r.reused && *std::prev(pass) - r.reused >= 2048)
        position = *std::prev(pass);
      if (position > r.reused && position - r.reused >= 2048 &&
          r.prompt - position > 128)
        add(position, CheckpointPurpose::kGrid, false);
    }
  }
  if (r.learn && r.common >= r.reused && r.common - r.reused >= 512 &&
      r.prompt - r.common > 64) {
    Rows position = r.common;
    Rows planned = 0;
    for (const auto& c : result)
      if (c.boundary > r.reused && c.boundary <= position)
        planned = std::max(planned, c.boundary);
    if (planned && position - planned <= 64)
      position = planned;
    add(position, CheckpointPurpose::kLearned, false);
  }
  std::ranges::sort(result, {}, &CaptureCandidate::boundary);
  return result;
}
RetentionPolicy::RetentionPolicy(ResourceLedger& ledger, PrefixIndex& index,
                                 RetentionEventSink* sink, std::size_t limit)
    : ledger_(ledger), index_(index), sink_(sink), limit_(limit) {
  if (!limit || limit > kRecordLimit)
    throw std::invalid_argument("invalid retention record limit");
  metadata_ =
      ledger.Reserve(ResourceCategory::kMetadata, sizeof(*this)).Convert();
}
RetentionPolicy::~RetentionPolicy() {
  for (std::size_t i = 0; i < limit_; ++i)
    if (records_[i].checkpoint)
      Erase(i, RetentionReason::kShutdown, Priority(i));
}
std::size_t RetentionPolicy::Find(CheckpointId id) const {
  for (std::size_t i = 0; i < limit_; ++i)
    if (records_[i].checkpoint && records_[i].checkpoint->Id() == id)
      return i;
  return limit_;
}
bool RetentionPolicy::Branch(std::size_t candidate) const {
  const auto& e = records_[candidate];
  if (!Continuation(e.purpose))
    return false;
  bool learned = e.purpose == RetentionPurpose::kBranchPoint;
  std::optional<Token> next;
  for (std::size_t i = 0; i < limit_; ++i) {
    const auto& peer = records_[i];
    if (i == candidate || !peer.checkpoint || !Continuation(peer.purpose) ||
        peer.checkpoint->Compatibility() != e.checkpoint->Compatibility() ||
        peer.checkpoint->Input() != e.checkpoint->Input() ||
        peer.checkpoint->Boundary() <= e.checkpoint->Boundary() ||
        !Prefix(e.checkpoint->Tokens(), peer.checkpoint->Tokens()))
      continue;
    if (peer.purpose == RetentionPurpose::kBranchPoint)
      learned = false;
    const auto token = peer.checkpoint->Tokens()[e.checkpoint->Boundary()];
    if (next && *next != token)
      return true;
    next = token;
  }
  return learned;
}
std::uint32_t RetentionPolicy::Priority(
    std::size_t candidate, const RetentionRequest* incoming) const {
  const auto& e = records_[candidate];
  if (Branch(candidate))
    return 3;
  if (incoming && Continuation(incoming->purpose) &&
      e.checkpoint->Compatibility() == incoming->compatibility &&
      e.checkpoint->Input() == incoming->input &&
      e.checkpoint->Boundary() < incoming->tokens.size() &&
      Prefix(e.checkpoint->Tokens(), incoming->tokens))
    return e.purpose == RetentionPurpose::kRetry     ? 0
           : e.purpose == RetentionPurpose::kHistory ? 1
                                                     : 2;
  for (std::size_t i = 0; i < limit_; ++i) {
    const auto& peer = records_[i];
    if (i == candidate || !peer.checkpoint || !Continuation(peer.purpose) ||
        peer.checkpoint->Compatibility() != e.checkpoint->Compatibility() ||
        peer.checkpoint->Input() != e.checkpoint->Input())
      continue;
    if (e.purpose == RetentionPurpose::kRetry &&
        peer.checkpoint->Boundary() <= e.stable &&
        Prefix(peer.checkpoint->Tokens(), e.checkpoint->Tokens()))
      return 0;
    if (e.purpose == RetentionPurpose::kHistory &&
        (Prefix(e.checkpoint->Tokens(), peer.checkpoint->Tokens()) ||
         Prefix(peer.checkpoint->Tokens(), e.checkpoint->Tokens())))
      return 1;
    if (Continuation(e.purpose) &&
        peer.checkpoint->Boundary() > e.checkpoint->Boundary() &&
        Prefix(e.checkpoint->Tokens(), peer.checkpoint->Tokens()))
      return 2;
  }
  return 3;
}
std::size_t RetentionPolicy::Victim(const RetentionRequest& r,
                                    bool record_pressure) const {
  const auto source = Find(r.source);
  const auto eligible = [&](std::size_t i) {
    return records_[i].checkpoint && Priority(i, &r) <= MaxRank(r.purpose);
  };
  // Full-record publication first replaces an incompatible edited tail of the
  // leased source's family, as the current cache does.
  if (record_pressure && source != limit_ &&
      Prefix(records_[source].checkpoint->Tokens(), r.tokens)) {
    std::size_t tail = limit_;
    for (std::size_t i = 0; i < limit_; ++i) {
      if (i == source || !eligible(i))
        continue;
      const auto& e = *records_[i].checkpoint;
      if (e.Compatibility() == r.compatibility && e.Input() == r.input &&
          e.Boundary() > records_[source].checkpoint->Boundary() &&
          !Prefix(e.Tokens(), r.tokens) &&
          Prefix(records_[source].checkpoint->Tokens(), e.Tokens()) &&
          (tail == limit_ || records_[i].last_used < records_[tail].last_used))
        tail = i;
    }
    if (tail != limit_)
      return tail;
  }
  std::size_t target = limit_;
  for (std::size_t i = 0; i < limit_; ++i) {
    if (!eligible(i) || i == source)
      continue;
    if (target == limit_ || Priority(i, &r) < Priority(target, &r) ||
        (Priority(i, &r) == Priority(target, &r) &&
         records_[i].last_used < records_[target].last_used))
      target = i;
  }
  // preserve_source protects byte admission while capture is incomplete. At
  // full-record publication, advance a redundant source before another
  // family's last copy regardless of that flag (ContinuationCache::Commit).
  if ((record_pressure || !r.preserve_source) && source != limit_ &&
      eligible(source) && Continuation(r.purpose) && Priority(source, &r) < 3 &&
      records_[source].checkpoint->Compatibility() == r.compatibility &&
      records_[source].checkpoint->Input() == r.input &&
      records_[source].checkpoint->Boundary() < r.tokens.size() &&
      Prefix(records_[source].checkpoint->Tokens(), r.tokens) &&
      (target == limit_ || Priority(target, &r) == 3))
    return source;
  if (target == limit_ && source != limit_ && eligible(source) &&
      ((!record_pressure && !r.preserve_source) ||
       (record_pressure && limit_ == 1 && Continuation(r.purpose))))
    return source;
  return target;
}
void RetentionPolicy::Emit(RetentionAction action, RetentionReason reason,
                           CheckpointId id, Rows boundary,
                           RetentionPurpose purpose, std::uint32_t rank,
                           std::uint64_t used, std::size_t freed) noexcept {
  const RetentionEvent event{++sequence_,
                             action,
                             reason,
                             id,
                             boundary,
                             purpose,
                             rank,
                             used,
                             freed,
                             size_,
                             ledger_.Snapshot().total_bytes,
                             RetainedPayloadBytes()};
  if (sink_)
    sink_->Emit(event);
}
void RetentionPolicy::Erase(std::size_t i, RetentionReason reason,
                            std::uint32_t rank) {
  const auto before = ledger_.Snapshot().total_bytes;
  auto& e = records_[i];
  const auto id = e.checkpoint->Id();
  const auto boundary = e.checkpoint->Boundary();
  const auto purpose = e.purpose;
  const auto used = e.last_used;
  index_.Erase(e.entry);
  e = {};
  --size_;
  const auto after = ledger_.Snapshot().total_bytes;
  Emit(RetentionAction::kRemoved, reason, id, boundary, purpose, rank, used,
       before > after ? before - after : 0);
}
bool RetentionPolicy::Admit(const RetentionRequest& r, const Capture& capture) {
  auto purpose = r.purpose;
  const auto refuse = [&](RetentionReason reason) {
    Emit(RetentionAction::kRefused, reason, {}, r.tokens.size(), purpose,
         MaxRank(purpose), 0);
    return false;
  };
  if (r.stop.stop_requested())
    return refuse(RetentionReason::kCancelled);
  if (r.tokens.empty() || r.stable > r.tokens.size() || !capture)
    return refuse(RetentionReason::kInvalid);
  const auto limits = ledger_.Limits();
  if (r.new_payload_bytes > limits.total_bytes ||
      r.new_payload_bytes > limits.ram_bytes)
    return refuse(RetentionReason::kByteCapacity);
  // ReserveSnapshot's optional-role record guard precedes its exact-match
  // search. Republishing an exact boundary cannot bypass that guard and
  // downgrade an inferred continuation branch to history/retry.
  if (size_ == limit_ && !Continuation(r.purpose) && Victim(r, true) == limit_)
    return refuse(RetentionReason::kRecordCapacity);
  // Exact replacement keeps learned status even if capture later needs space.
  std::size_t exact = limit_;
  for (std::size_t i = 0; i < limit_; ++i) {
    const auto& e = records_[i];
    if (e.checkpoint && e.checkpoint->Compatibility() == r.compatibility &&
        e.checkpoint->Input() == r.input &&
        std::ranges::equal(e.checkpoint->Tokens(), r.tokens)) {
      exact = i;
      if (e.purpose == RetentionPurpose::kBranchPoint)
        purpose = RetentionPurpose::kBranchPoint;
      break;
    }
  }
  if (exact == limit_ && size_ == limit_ && Victim(r, true) == limit_)
    return refuse(RetentionReason::kRecordCapacity);
  while (true) {
    if (r.stop.stop_requested())
      return refuse(RetentionReason::kCancelled);
    try {
      auto checkpoint = capture();
      if (!checkpoint || !index_.IsResidentCoherent(*checkpoint) ||
          checkpoint->Compatibility() != r.compatibility ||
          checkpoint->Input() != r.input ||
          !std::ranges::equal(checkpoint->Tokens(), r.tokens)) {
        checkpoint.reset();
        return refuse(RetentionReason::kInvalid);
      }
      if (r.stop.stop_requested()) {
        checkpoint.reset();
        return refuse(RetentionReason::kCancelled);
      }
      std::vector<ComponentAvailability> available;
      for (const auto& c : checkpoint->Components())
        available.push_back({c.descriptor.id, true, false});
      const auto entry =
          index_.Insert(checkpoint, std::move(available), r.stable);
      if (exact != limit_ && records_[exact].checkpoint) {
        Erase(exact, RetentionReason::kExactReplacement, Priority(exact, &r));
      } else if (size_ == limit_) {
        const auto victim = Victim(r, true);
        const auto source = Find(r.source);
        const bool edited =
            source != limit_ &&
            !Prefix(records_[victim].checkpoint->Tokens(), r.tokens) &&
            Prefix(records_[source].checkpoint->Tokens(),
                   records_[victim].checkpoint->Tokens());
        Erase(victim,
              edited ? RetentionReason::kEditedTail
                     : RetentionReason::kRecordCapacity,
              Priority(victim, &r));
      }
      std::size_t target = 0;
      while (records_[target].checkpoint)
        ++target;
      auto& e = records_[target];
      e = {std::move(checkpoint), entry, purpose, r.stable, ++clock_};
      ++size_;
      Emit(RetentionAction::kAdmitted, RetentionReason::kCaptured,
           e.checkpoint->Id(), e.checkpoint->Boundary(), purpose,
           Priority(target), e.last_used);
      return true;
    } catch (const ResourceExhausted&) {
      auto target = limit_;
      if (exact != limit_ && records_[exact].checkpoint &&
          (records_[exact].checkpoint->Id() != r.source || !r.preserve_source))
        target = exact;
      else
        target = Victim(r, false);
      if (target == limit_)
        return refuse(RetentionReason::kByteCapacity);
      Erase(target,
            target == exact ? RetentionReason::kExactReplacement
                            : RetentionReason::kByteCapacity,
            Priority(target, &r));
    } catch (const std::exception&) {
      return refuse(RetentionReason::kCaptureFailure);
    }
  }
}
std::size_t RetentionPolicy::RetainedPayloadBytes() const noexcept {
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < limit_; ++i) {
    if (!records_[i].checkpoint)
      continue;
    for (const auto& c : records_[i].checkpoint->Components()) {
      if (c.private_state)
        bytes += c.private_state->Bytes();
      if (c.tail)
        bytes += c.tail->Bytes();
      for (std::size_t n = 0; n < c.chunks.size(); ++n) {
        bool shared = false;
        for (std::size_t earlier = 0; earlier < i && !shared; ++earlier) {
          if (!records_[earlier].checkpoint)
            continue;
          for (const auto& peer : records_[earlier].checkpoint->Components())
            if (peer.descriptor.id == c.descriptor.id &&
                peer.chunks.size() > n &&
                peer.chunks[n].Id() == c.chunks[n].Id()) {
              shared = true;
              break;
            }
        }
        if (!shared)
          bytes += c.chunks[n].Storage().Bytes();
      }
    }
  }
  return bytes;
}
void RetentionPolicy::Touch(CheckpointId id) {
  const auto i = Find(id);
  if (i == limit_)
    throw std::invalid_argument("unknown retained checkpoint");
  auto& e = records_[i];
  e.last_used = ++clock_;
  Emit(RetentionAction::kTouched, RetentionReason::kExplicit, id,
       e.checkpoint->Boundary(), e.purpose, Priority(i), e.last_used);
}
void RetentionPolicy::Remove(CheckpointId id) {
  const auto i = Find(id);
  if (i == limit_)
    throw std::invalid_argument("unknown retained checkpoint");
  Erase(i, RetentionReason::kExplicit, Priority(i));
}
std::uint32_t RetentionPolicy::Rank(CheckpointId id) const {
  const auto i = Find(id);
  if (i == limit_)
    throw std::invalid_argument("unknown retained checkpoint");
  return Priority(i);
}
}  // namespace gufo::cache

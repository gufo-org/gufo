#include "src/cache/retention.hpp"

#include <algorithm>
#include <limits>
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
std::size_t GridLimit(std::size_t ram_bytes, std::size_t private_bytes) {
  return private_bytes
             ? std::min<std::size_t>(4, ram_bytes / 12 / private_bytes)
             : 4;
}
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
    std::size_t planned = 0;
    for (Rows i = 1; i <= count && planned < r.grid_limit; ++i) {
      Rows position = (grid / count * i + grid % count * i / count) * 2048;
      auto pass = std::ranges::upper_bound(passes, position);
      if (pass != passes.begin() && position - *std::prev(pass) <= 128 &&
          *std::prev(pass) > r.reused && *std::prev(pass) - r.reused >= 2048)
        position = *std::prev(pass);
      if (position > r.reused && position - r.reused >= 2048 &&
          r.prompt - position > 128) {
        add(position, CheckpointPurpose::kGrid, false);
        ++planned;
      }
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
    // Relate records the same immutable compatibility/input/strict-prefix
    // relation. Rechecking long token arrays for every priority query adds no
    // information; Erase removes the corresponding relation bits.
    if (!e.below[i] || !peer.checkpoint || !Continuation(peer.purpose))
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
    if (i == candidate || !peer.checkpoint || !Continuation(peer.purpose))
      continue;
    if (Continuation(e.purpose)) {
      if (e.below[i])
        return 2;
      continue;
    }
    if (peer.checkpoint->Compatibility() != e.checkpoint->Compatibility() ||
        peer.checkpoint->Input() != e.checkpoint->Input())
      continue;
    if (e.purpose == RetentionPurpose::kRetry &&
        peer.checkpoint->Boundary() <= e.stable &&
        Prefix(peer.checkpoint->Tokens(), e.checkpoint->Tokens()))
      return 0;
    if (e.purpose == RetentionPurpose::kHistory &&
        Prefix(e.checkpoint->Tokens(), peer.checkpoint->Tokens()))
      return 1;
  }
  return 3;
}
bool RetentionPolicy::Related(std::size_t root, std::size_t peer) const {
  const auto& a = *records_[root].checkpoint;
  const auto& b = *records_[peer].checkpoint;
  return root != peer && a.Compatibility() == b.Compatibility() &&
         a.Input() == b.Input() && a.Boundary() < b.Boundary() &&
         Prefix(a.Tokens(), b.Tokens());
}
void RetentionPolicy::Relate(std::size_t added) {
  auto& e = records_[added];
  std::size_t parent = limit_;
  for (std::size_t j = 0; j < limit_; ++j) {
    if (j == added || !records_[j].checkpoint)
      continue;
    if (Related(added, j)) {
      e.below.set(j);
    } else if (Related(j, added)) {
      records_[j].below.set(added);
      if (parent == limit_ || records_[j].checkpoint->Boundary() >
                                  records_[parent].checkpoint->Boundary())
        parent = j;
    }
  }
  // The deepest retained prefix forks when another record extends it apart
  // from this one; no deeper record can be prefix to both.
  if (parent != limit_)
    for (std::size_t j = 0; j < limit_; ++j)
      if (j != added && records_[parent].below[j] && !e.below[j] &&
          !records_[j].below[added])
        records_[parent].forked = true;
}
std::array<std::uint64_t, 4> RetentionPolicy::PressureOrder(
    std::size_t i, const RetentionRequest& r, std::uint32_t priority) const {
  const auto& e = records_[i];
  if (!priority)
    return {0, e.last_used, 0, 0};
  std::size_t rows = 0, state = 0;
  for (const auto& c : e.checkpoint->Components())
    if (c.descriptor.kind == ComponentKind::kPrivateState)
      state += c.descriptor.state_bytes;
    else
      rows += c.position.valid_rows * c.descriptor.row_bytes;
  const auto turnover = ledger_.Limits().ram_bytes;
  const auto fresh = [&](const Record& x) {
    return x.reused_at && admitted_bytes_ - *x.reused_at < turnover;
  };
  // A short prompt is cheap to prefill again. Once reused it competes by
  // recency with unreused conversations instead: evicting the checkpoint a
  // retry restored one request ago turns its next turn into a full miss.
  // A shared system/tools prefix of the incoming prompt is also not cheap:
  // removing it to admit the new conversation leaves the next one with no
  // checkpoint at the prefix they share (tool-agent peers).
  const bool shared_prefix = e.checkpoint->Compatibility() == r.compatibility &&
                             e.checkpoint->Input() == r.input &&
                             e.checkpoint->Boundary() < r.tokens.size() &&
                             Prefix(e.checkpoint->Tokens(), r.tokens);
  const bool cheap = rows * 8 < state && !shared_prefix;
  if (cheap && !fresh(e))
    return {1, e.last_used, 0, 0};
  bool rooted = false, reused = fresh(e);
  auto recency = e.last_used;
  const bool covered = e.below.any();
  for (std::size_t j = 0; j < limit_; ++j) {
    if (e.below[j]) {
      recency = std::max(recency, records_[j].last_used);
      reused |= fresh(records_[j]);
    }
    rooted |= records_[j].below[i];
  }
  const bool fork = e.forked || Branch(i);
  // Shared boundaries outlive the leaves that extend them: forks, and the
  // shallowest history boundary, which other prompts may also reach.
  const bool shared = fork || (covered && !rooted &&
                               (e.purpose == RetentionPurpose::kHistory ||
                                e.purpose == RetentionPurpose::kBranchPoint));
  const auto depth =
      std::numeric_limits<std::uint64_t>::max() - e.checkpoint->Boundary();
  if (reused && !cheap)
    return {3, fork ? 2U : covered ? 0U : 1U, recency, depth};
  return {2, recency, shared ? 2U : covered ? 0U : 1U, depth};
}
std::size_t RetentionPolicy::UniqueBytes(std::size_t i) const {
  const auto components = records_[i].checkpoint->Components();
  std::size_t bytes = 0;
  for (std::size_t k = 0; k < components.size(); ++k) {
    const auto& c = components[k];
    if (c.private_state)
      bytes += c.private_state->Bytes();
    if (c.tail)
      bytes += c.tail->Bytes();
    // Admission age uses the longest shared-prefix estimate. Exact retained
    // payload accounting below also handles sharing after a recreated hole.
    std::size_t shared = 0;
    for (std::size_t j = 0; j < limit_; ++j) {
      if (j == i || !records_[j].checkpoint)
        continue;
      const auto peers = records_[j].checkpoint->Components();
      if (k >= peers.size() || peers[k].descriptor.id != c.descriptor.id)
        continue;
      std::size_t n = 0;
      while (n < c.chunks.size() && n < peers[k].chunks.size() &&
             c.chunks[n].SameChunk(peers[k].chunks[n]))
        ++n;
      shared = std::max(shared, n);
    }
    for (std::size_t n = shared; n < c.chunks.size(); ++n)
      bytes += c.chunks[n].Storage().Bytes();
  }
  return bytes;
}
std::size_t RetentionPolicy::UniquePayloadBytes(std::size_t i) const noexcept {
  std::size_t bytes = 0;
  for (const auto& c : records_[i].checkpoint->Components()) {
    if (c.private_state)
      bytes += c.private_state->Bytes();
    if (c.tail)
      bytes += c.tail->Bytes();
    if (c.chunks.empty())
      continue;
    std::array<const CheckpointComponent*, kRecordLimit> peers;
    std::size_t count = 0;
    for (std::size_t j = 0; j < limit_; ++j) {
      if (j == i || !records_[j].checkpoint)
        continue;
      for (const auto& peer : records_[j].checkpoint->Components())
        if (peer.descriptor.id == c.descriptor.id && !peer.chunks.empty()) {
          peers[count++] = &peer;
          break;
        }
    }
    // Try the longest peers first; shorter peers cannot own later rows.
    std::sort(peers.begin(), peers.begin() + count,
              [](const auto* a, const auto* b) {
                return a->chunks.size() > b->chunks.size();
              });
    for (std::size_t n = 0; n < c.chunks.size(); ++n) {
      const auto& chunk = c.chunks[n];
      bool shared = false;
      for (std::size_t j = 0; j < count && peers[j]->chunks.size() > n; ++j)
        if (chunk.SameChunk(peers[j]->chunks[n])) {
          shared = true;
          break;
        }
      if (!shared)
        bytes += chunk.Storage().Bytes();
    }
  }
  return bytes;
}
bool RetentionPolicy::Persisting(std::size_t i) const {
  // A disk job pins every payload of the checkpoint it writes, including the
  // private state. Removing the record then frees nothing until the write
  // ends, and would only drop a restorable boundary from the index.
  for (const auto& c : records_[i].checkpoint->Components())
    if (c.private_state)
      return c.private_state->PersistencePinned();
  return false;
}
std::size_t RetentionPolicy::Victim(const RetentionRequest& r,
                                    bool record_pressure) const {
  const auto source = Find(r.source);
  // Records and the incoming request do not change during selection. Reuse
  // each rank across eligibility, ordering and source-fallback decisions.
  std::array<std::optional<std::uint32_t>, kRecordLimit> priorities{};
  const auto priority = [&](std::size_t i) {
    auto& value = priorities[i];
    if (!value)
      value = Priority(i, &r);
    return *value;
  };
  // An unchanged retry's full-prompt copy may replace the checkpoint it was
  // restored from when only the assistant framing separates them: the copy
  // covers every prompt the source did except an edit inside those few
  // tokens, and no other conversation loses a checkpoint.
  constexpr Rows kRetryFraming = 64;
  const bool resumed_tail = r.reused && r.reused_tokens &&
                            r.reused_tokens <= r.tokens.size() &&
                            r.tokens.size() - r.reused_tokens <= kRetryFraming;
  const bool reused_source =
      !record_pressure && r.purpose == RetentionPurpose::kRetry && r.reused &&
      !resumed_tail && source != limit_ &&
      records_[source].checkpoint->Boundary() <= r.reused_tokens &&
      Prefix(records_[source].checkpoint->Tokens(), r.tokens);
  const bool replaces_source =
      !record_pressure && r.purpose == RetentionPurpose::kRetry &&
      source != limit_ && !Persisting(source) &&
      records_[source].checkpoint->Compatibility() == r.compatibility &&
      records_[source].checkpoint->Input() == r.input &&
      records_[source].checkpoint->Boundary() < r.tokens.size() &&
      r.tokens.size() - records_[source].checkpoint->Boundary() <=
          kRetryFraming &&
      Prefix(records_[source].checkpoint->Tokens(), r.tokens);
  const auto eligible = [&](std::size_t i) {
    // Byte admission for an edit boundary must not starve behind last copies
    // of older families. A short resumed tail may also replace covered history
    // to retain its complete retry state; faster captures can otherwise fill
    // the pool with older ancestors before this final boundary is reached.
    // Retry admission still protects last copies, forks and current edit
    // boundaries. A larger resumed turn can reclaim its own covered prefix
    // history, without taking another conversation's history for compaction.
    // Cold retries and record pressure keep their existing limits.
    const auto maximum =
        !record_pressure && r.purpose == RetentionPurpose::kHistory ? 3
        : !record_pressure && r.purpose == RetentionPurpose::kRetry &&
                resumed_tail
            ? 2
            : MaxRank(r.purpose);
    if (!records_[i].checkpoint || Persisting(i) ||
        std::ranges::find(r.protected_checkpoints,
                          records_[i].checkpoint->Id()) !=
            r.protected_checkpoints.end())
      return false;
    const auto level = priority(i);
    if (level <= maximum)
      return true;
    const auto& checkpoint = *records_[i].checkpoint;
    return !record_pressure && r.purpose == RetentionPurpose::kRetry &&
           r.reused && r.reused_tokens && r.reused_tokens <= r.tokens.size() &&
           level <= 2 && checkpoint.Boundary() <= r.reused_tokens &&
           checkpoint.Compatibility() == r.compatibility &&
           checkpoint.Input() == r.input &&
           (reused_source ? i == source || records_[i].below[source]
                          : Prefix(checkpoint.Tokens(), r.tokens));
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
  std::array<std::uint64_t, 4> order{};
  // Byte pressure may select a last copy while covered records remain. The
  // redundant source still advances only when every candidate is a last copy.
  std::uint32_t lowest = 3;
  for (std::size_t i = 0; i < limit_; ++i) {
    if (!eligible(i) || i == source)
      continue;
    if (!record_pressure) {
      lowest = std::min(lowest, priority(i));
      const auto candidate = PressureOrder(i, r, priority(i));
      if (target == limit_ || candidate < order) {
        target = i;
        order = candidate;
      }
    } else if (target == limit_ || priority(i) < priority(target) ||
               (priority(i) == priority(target) &&
                records_[i].last_used < records_[target].last_used)) {
      target = i;
    }
  }
  if (target == limit_ && replaces_source)
    return source;
  // preserve_source protects byte admission while capture is incomplete. At
  // full-record publication, advance a redundant source before another
  // family's last copy regardless of that flag (ContinuationCache::Commit).
  if ((record_pressure || !r.preserve_source) && source != limit_ &&
      eligible(source) && Continuation(r.purpose) && priority(source) < 3 &&
      records_[source].checkpoint->Compatibility() == r.compatibility &&
      records_[source].checkpoint->Input() == r.input &&
      records_[source].checkpoint->Boundary() < r.tokens.size() &&
      Prefix(records_[source].checkpoint->Tokens(), r.tokens) &&
      (target == limit_ || (record_pressure ? priority(target) : lowest) == 3))
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
  const auto unique = retained_payload_bytes_ ? UniquePayloadBytes(i) : 0;
  const auto before = ledger_.Snapshot().total_bytes;
  auto& e = records_[i];
  const auto id = e.checkpoint->Id();
  const auto boundary = e.checkpoint->Boundary();
  const auto purpose = e.purpose;
  const auto used = e.last_used;
  index_.DropResident(e.entry);
  e = {};
  for (auto& peer : records_)
    peer.below.reset(i);
  --size_;
  if (retained_payload_bytes_)
    *retained_payload_bytes_ -= unique;
  const auto after = ledger_.Snapshot().total_bytes;
  Emit(RetentionAction::kRemoved, reason, id, boundary, purpose, rank, used,
       before > after ? before - after : 0);
}
bool RetentionPolicy::Reclaim(const RetentionRequest& r) {
  if (r.new_payload_bytes > ledger_.Limits().ram_bytes)
    return false;
  const auto victim = Victim(r, false);
  if (victim == limit_)
    return false;
  Erase(victim, RetentionReason::kByteCapacity, Priority(victim, &r));
  return true;
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
  // An exact replacement keeps its conversation's reuse.
  std::optional<std::uint64_t> reused_at;
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
        reused_at = records_[exact].reused_at;
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
      // Keep each event's exact total without recounting every retained
      // checkpoint after each individual eviction or insertion.
      if (retained_payload_bytes_)
        *retained_payload_bytes_ += UniquePayloadBytes(target);
      admitted_bytes_ += UniqueBytes(target);
      Relate(target);
      if (r.reused || reused_at)
        e.reused_at = r.reused ? admitted_bytes_ : reused_at;
      Emit(RetentionAction::kAdmitted, RetentionReason::kCaptured,
           e.checkpoint->Id(), e.checkpoint->Boundary(), purpose,
           Priority(target), e.last_used);
      return true;
    } catch (const ResourceExhausted&) {
      auto target = limit_;
      if (exact != limit_ && records_[exact].checkpoint &&
          (records_[exact].checkpoint->Id() != r.source ||
           !r.preserve_source) &&
          std::ranges::find(r.protected_checkpoints,
                            records_[exact].checkpoint->Id()) ==
              r.protected_checkpoints.end())
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
  if (retained_payload_bytes_)
    return *retained_payload_bytes_;
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < limit_; ++i) {
    if (!records_[i].checkpoint)
      continue;
    for (const auto& c : records_[i].checkpoint->Components()) {
      if (c.private_state)
        bytes += c.private_state->Bytes();
      if (c.tail)
        bytes += c.tail->Bytes();
      if (c.chunks.empty())
        continue;
      // Count a component across all retained checkpoints once. Compare live
      // references directly instead of reading IDs from cold chunk objects.
      // Sharing need not be a prefix: a reader can keep a later chunk alive
      // while earlier chunks are recreated.
      bool counted = false;
      for (std::size_t earlier = 0; earlier < i && !counted; ++earlier) {
        if (!records_[earlier].checkpoint)
          continue;
        for (const auto& peer : records_[earlier].checkpoint->Components())
          if (peer.descriptor.id == c.descriptor.id && !peer.chunks.empty()) {
            counted = true;
            break;
          }
      }
      if (counted)
        continue;
      std::array<const CheckpointComponent*, kRecordLimit> peers;
      std::size_t peer_count = 0;
      std::size_t rows = 0;
      for (std::size_t j = i; j < limit_; ++j) {
        if (!records_[j].checkpoint)
          continue;
        for (const auto& peer : records_[j].checkpoint->Components())
          if (peer.descriptor.id == c.descriptor.id) {
            peers[peer_count++] = &peer;
            rows = std::max(rows, peer.chunks.size());
            break;
          }
      }
      for (std::size_t n = 0; n < rows; ++n) {
        std::array<const CheckpointChunk*, kRecordLimit> seen;
        std::size_t seen_count = 0;
        for (std::size_t peer = 0; peer < peer_count; ++peer) {
          if (peers[peer]->chunks.size() <= n)
            continue;
          const auto& chunk = peers[peer]->chunks[n];
          const auto end = seen.begin() + seen_count;
          if (std::find_if(seen.begin(), end, [&](const auto* previous) {
                return chunk.SameChunk(*previous);
              }) == end) {
            seen[seen_count++] = &chunk;
            bytes += chunk.Storage().Bytes();
          }
        }
      }
    }
  }
  retained_payload_bytes_ = bytes;
  return bytes;
}
void RetentionPolicy::Touch(CheckpointId id) {
  const auto i = Find(id);
  if (i == limit_)
    throw std::invalid_argument("unknown retained checkpoint");
  auto& e = records_[i];
  e.last_used = ++clock_;
  e.reused_at = admitted_bytes_;
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

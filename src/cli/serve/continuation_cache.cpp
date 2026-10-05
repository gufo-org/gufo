#include "src/cli/serve/continuation_cache.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/cli/serve/logging.hpp"

namespace gufo::server {

namespace {

constexpr auto kCancellationPollInterval = std::chrono::milliseconds{10};

bool IsPrefix(std::span<const ContinuationToken> prefix,
              std::span<const ContinuationToken> tokens) {
  return prefix.size() <= tokens.size() &&
         std::equal(prefix.begin(), prefix.end(), tokens.begin());
}

int MaxRemovalPriority(SnapshotPurpose purpose) {
  switch (purpose) {
    case SnapshotPurpose::kRetry:
      return 0;
    case SnapshotPurpose::kHistory:
      return 1;
    case SnapshotPurpose::kContinuation:
      return 3;
  }
  return 0;
}

void EmitSnapshotEvents(const ContinuationCache::SnapshotEventSink& sink,
                        std::span<const SnapshotEvent> events) noexcept {
  if (!sink) {
    return;
  }
  for (const auto& event : events) {
    try {
      sink(event);
    } catch (...) {
      // Cache observability must never affect request execution.
      continue;
    }
  }
}

}  // namespace

struct ContinuationCache::Entry {
  explicit Entry(std::unique_ptr<ContinuationState> model_state)
      : state(std::move(model_state)) {}

  std::unique_ptr<ContinuationState> state;
  std::shared_ptr<const ContinuationSnapshot> snapshot;
  std::vector<ContinuationToken> tokens;
  std::vector<std::uint8_t> input_identity;
  std::vector<ContinuationToken> live_tokens;
  std::vector<std::uint8_t> live_identity;
  std::size_t snapshot_bytes{0};
  std::size_t stable_prefix_tokens{0};
  SnapshotPurpose purpose{SnapshotPurpose::kContinuation};
  std::uint64_t state_last_used{0};
  std::uint64_t snapshot_last_used{0};
  bool available{true};
  bool valid{false};
  bool dirty{false};
};

struct ContinuationCache::Impl {
  std::vector<std::unique_ptr<Entry>> entries;
  std::size_t state_count{0};
  SnapshotSupport snapshot_support;
  mutable std::mutex mutex;
  std::condition_variable condition;
  std::uint64_t clock{0};
  std::size_t snapshot_capacity_bytes{0};
  std::size_t retained_snapshot_bytes{0};
  std::size_t reserved_snapshot_bytes{0};

  [[nodiscard]] bool snapshot_mode() const noexcept {
    return static_cast<bool>(snapshot_support.restore);
  }

  // Rank extra copies before the last useful checkpoint of a prefix family.
  // Exact tokens and input identity, not client IDs or hashes, establish that
  // another continuation can keep the family reusable after this removal.
  /// A continuation that two retained continuations extend and then diverge
  /// after is the prefix they share, such as a system prompt. No copy
  /// replaces it, so it is never ranked as redundant.
  [[nodiscard]] bool IsBranchPoint(std::size_t candidate) const {
    const auto& entry = *entries[candidate];
    if (entry.purpose != SnapshotPurpose::kContinuation)
      return false;
    std::optional<ContinuationToken> next;
    for (std::size_t other = state_count; other < entries.size(); ++other) {
      const auto& peer = *entries[other];
      if (other == candidate || !peer.valid ||
          peer.purpose != SnapshotPurpose::kContinuation ||
          peer.input_identity != entry.input_identity ||
          peer.tokens.size() <= entry.tokens.size() ||
          !IsPrefix(entry.tokens, peer.tokens))
        continue;
      const auto token = peer.tokens[entry.tokens.size()];
      if (next.has_value() && *next != token)
        return true;
      next = token;
    }
    return false;
  }

  [[nodiscard]] int RemovalPriority(
      std::size_t candidate, std::span<const ContinuationToken> incoming = {},
      std::span<const std::uint8_t> incoming_identity = {}) const {
    const auto& entry = *entries[candidate];
    if (IsBranchPoint(candidate))
      return 3;
    // A new continuation also makes its own earlier copies redundant. During
    // cold prefill those may all be history snapshots, so no retained
    // continuation yet exists to protect another conversation's last copy.
    if (entry.tokens.size() < incoming.size() &&
        std::ranges::equal(entry.input_identity, incoming_identity) &&
        IsPrefix(entry.tokens, incoming)) {
      if (entry.purpose == SnapshotPurpose::kRetry)
        return 0;
      return entry.purpose == SnapshotPurpose::kHistory ? 1 : 2;
    }
    for (std::size_t other = state_count; other < entries.size(); ++other) {
      const auto& peer = *entries[other];
      if (other == candidate || !peer.valid ||
          peer.purpose != SnapshotPurpose::kContinuation ||
          peer.input_identity != entry.input_identity)
        continue;
      if (entry.purpose == SnapshotPurpose::kRetry &&
          peer.tokens.size() <= entry.stable_prefix_tokens &&
          IsPrefix(peer.tokens, entry.tokens))
        return 0;
      if (entry.purpose == SnapshotPurpose::kHistory &&
          (IsPrefix(entry.tokens, peer.tokens) ||
           IsPrefix(peer.tokens, entry.tokens)))
        return 1;
      if (entry.purpose == SnapshotPurpose::kContinuation &&
          peer.tokens.size() > entry.tokens.size() &&
          IsPrefix(entry.tokens, peer.tokens))
        return 2;
    }
    return 3;
  }
};

ContinuationCache::Lease::Lease(ContinuationCache* cache, std::size_t index,
                                bool cache_hit, std::size_t cached_tokens,
                                std::size_t source_index,
                                std::size_t restored_snapshot_bytes,
                                double restore_ms) noexcept
    : cache_(cache),
      index_(index),
      cache_hit_(cache_hit),
      cached_tokens_(cached_tokens),
      source_index_(source_index),
      restored_snapshot_bytes_(restored_snapshot_bytes),
      restore_ms_(restore_ms) {}

ContinuationCache::Lease::~Lease() {
  Invalidate();
}

ContinuationCache::Lease::Lease(Lease&& other) noexcept
    : cache_(std::exchange(other.cache_, nullptr)),
      index_(std::exchange(other.index_, 0)),
      cache_hit_(std::exchange(other.cache_hit_, false)),
      cached_tokens_(std::exchange(other.cached_tokens_, 0)),
      source_index_(std::exchange(other.source_index_, 0)),
      restored_snapshot_bytes_(
          std::exchange(other.restored_snapshot_bytes_, 0)),
      restore_ms_(std::exchange(other.restore_ms_, 0.0)),
      restored_from_disk_(std::exchange(other.restored_from_disk_, false)),
      reserved_snapshot_bytes_(
          std::exchange(other.reserved_snapshot_bytes_, 0)),
      snapshot_purpose_(other.snapshot_purpose_),
      prompt_tokens_(std::exchange(other.prompt_tokens_, 0)),
      stable_prefix_tokens_(std::exchange(other.stable_prefix_tokens_, 0)),
      input_identity_(std::move(other.input_identity_)),
      input_prefixes_(std::move(other.input_prefixes_)),
      lookup_(std::exchange(other.lookup_, {})) {}

ContinuationCache::Lease& ContinuationCache::Lease::operator=(
    Lease&& other) noexcept {
  if (this != &other) {
    Invalidate();
    cache_ = std::exchange(other.cache_, nullptr);
    index_ = std::exchange(other.index_, 0);
    cache_hit_ = std::exchange(other.cache_hit_, false);
    cached_tokens_ = std::exchange(other.cached_tokens_, 0);
    source_index_ = std::exchange(other.source_index_, 0);
    restored_snapshot_bytes_ = std::exchange(other.restored_snapshot_bytes_, 0);
    restore_ms_ = std::exchange(other.restore_ms_, 0.0);
    restored_from_disk_ = std::exchange(other.restored_from_disk_, false);
    reserved_snapshot_bytes_ = std::exchange(other.reserved_snapshot_bytes_, 0);
    snapshot_purpose_ = other.snapshot_purpose_;
    prompt_tokens_ = std::exchange(other.prompt_tokens_, 0);
    stable_prefix_tokens_ = std::exchange(other.stable_prefix_tokens_, 0);
    input_identity_ = std::move(other.input_identity_);
    input_prefixes_ = std::move(other.input_prefixes_);
    lookup_ = std::exchange(other.lookup_, {});
  }
  return *this;
}

ContinuationState& ContinuationCache::Lease::state() const {
  if (cache_ == nullptr) {
    throw std::logic_error("continuation cache lease is empty");
  }
  return cache_->StateAt(index_);
}

void ContinuationCache::Lease::AdoptRestoredPrefix(std::size_t cached_tokens,
                                                   std::size_t restored_bytes,
                                                   double restore_ms) {
  if (cache_ == nullptr) {
    throw std::logic_error("continuation cache lease is empty");
  }
  if (cache_hit_ || cached_tokens == 0 || restored_bytes == 0 ||
      restore_ms < 0.0) {
    throw std::invalid_argument(
        "invalid lower-tier continuation restore metrics");
  }
  cache_hit_ = true;
  cached_tokens_ = cached_tokens;
  restored_snapshot_bytes_ = restored_bytes;
  restore_ms_ = restore_ms;
  restored_from_disk_ = true;
  lookup_ = {};
}

bool ContinuationCache::Lease::TryReserveSnapshot(
    std::size_t snapshot_bytes, std::size_t token_count, bool preserve_source,
    SnapshotPurpose purpose,
    std::span<const ContinuationToken> replacement_prefix) {
  if (cache_ == nullptr) {
    throw std::logic_error("continuation cache lease is empty");
  }
  if (reserved_snapshot_bytes_ != 0) {
    throw std::logic_error(
        "continuation cache lease already has a snapshot reservation");
  }
  if (!cache_->ReserveSnapshot(source_index_, snapshot_bytes, token_count,
                               preserve_source, purpose, replacement_prefix,
                               InputIdentity(token_count))) {
    return false;
  }
  reserved_snapshot_bytes_ = snapshot_bytes;
  snapshot_purpose_ = purpose;
  return true;
}

void ContinuationCache::Lease::SkipSnapshot(SnapshotEventReason reason,
                                            std::size_t snapshot_bytes,
                                            std::size_t token_count) noexcept {
  if (cache_ == nullptr) {
    return;
  }
  cache_->SkipSnapshot(reserved_snapshot_bytes_, reason, snapshot_bytes,
                       token_count);
  reserved_snapshot_bytes_ = 0;
}

std::size_t ContinuationCache::Lease::Commit(
    std::vector<ContinuationToken> tokens,
    std::shared_ptr<const ContinuationSnapshot> snapshot,
    std::vector<ContinuationToken> live_tokens) {
  if (cache_ == nullptr) {
    throw std::logic_error("continuation cache lease is empty");
  }
  const auto stable_prefix =
      tokens.size() == prompt_tokens_ ? stable_prefix_tokens_ : 0;
  const auto identity = InputIdentity(tokens.size());
  const auto live_identity = InputIdentity(live_tokens.size());
  const std::size_t retained = cache_->Commit(
      index_, source_index_, reserved_snapshot_bytes_, std::move(tokens),
      std::move(snapshot), {identity.begin(), identity.end()},
      std::move(live_tokens), {live_identity.begin(), live_identity.end()},
      true, nullptr, stable_prefix, snapshot_purpose_);
  cache_ = nullptr;
  reserved_snapshot_bytes_ = 0;
  return retained;
}

std::size_t ContinuationCache::Lease::PublishSnapshot(
    std::vector<ContinuationToken> tokens,
    std::shared_ptr<const ContinuationSnapshot> snapshot,
    bool preserve_source) {
  if (cache_ == nullptr)
    throw std::logic_error("continuation cache lease is empty");
  const auto identity = InputIdentity(tokens.size());
  const auto retained = cache_->Commit(
      index_, source_index_, reserved_snapshot_bytes_, std::move(tokens),
      std::move(snapshot), {identity.begin(), identity.end()}, {}, {}, false,
      preserve_source ? nullptr : &source_index_, 0, snapshot_purpose_);
  reserved_snapshot_bytes_ = 0;
  return retained;
}

void ContinuationCache::Lease::Invalidate() noexcept {
  if (cache_ != nullptr) {
    cache_->Invalidate(index_, reserved_snapshot_bytes_);
    cache_ = nullptr;
    reserved_snapshot_bytes_ = 0;
  }
}

ContinuationCache::ContinuationCache(std::size_t capacity,
                                     const StateFactory& factory,
                                     SnapshotSupport snapshot_support,
                                     std::size_t snapshot_capacity)
    : impl_(std::make_unique<Impl>()) {
  if (capacity == 0) {
    throw std::invalid_argument(
        "continuation cache capacity must be at least one");
  }
  if (!factory) {
    throw std::invalid_argument(
        "continuation cache state factory must be callable");
  }
  impl_->snapshot_support = std::move(snapshot_support);
  impl_->state_count = capacity;

  const auto record_count =
      snapshot_capacity == 0 ? capacity : snapshot_capacity;
  if (impl_->snapshot_mode() &&
      record_count > std::numeric_limits<std::size_t>::max() - capacity)
    throw std::invalid_argument("continuation checkpoint capacity overflows");
  const auto entry_count =
      impl_->snapshot_mode() ? capacity + record_count : capacity;
  impl_->entries.reserve(entry_count);
  for (std::size_t index = 0; index < capacity; ++index) {
    auto state = factory();
    if (state == nullptr) {
      throw std::runtime_error(
          "continuation cache state factory returned null");
    }
    impl_->entries.push_back(std::make_unique<Entry>(std::move(state)));
  }
  // Extra immutable checkpoints share the same byte budget. They do not
  // allocate model sessions and must never be selected as execution slots.
  for (std::size_t index = capacity; index < entry_count; ++index) {
    auto entry = std::make_unique<Entry>(nullptr);
    entry->available = false;
    impl_->entries.push_back(std::move(entry));
  }
  if (impl_->snapshot_mode() && impl_->snapshot_support.capacity_bytes) {
    try {
      impl_->snapshot_capacity_bytes = impl_->snapshot_support.capacity_bytes();
    } catch (...) {
      impl_->snapshot_capacity_bytes = 0;
    }
  }
}

ContinuationCache::~ContinuationCache() = default;

bool ContinuationCache::Lease::HasSnapshotFor(
    std::span<const ContinuationToken> tokens) const {
  if (!cache_)
    return false;
  const std::lock_guard lock(cache_->impl_->mutex);
  return std::ranges::any_of(cache_->impl_->entries, [&](const auto& source) {
    return source->valid && source->snapshot &&
           std::ranges::equal(source->input_identity,
                              InputIdentity(tokens.size())) &&
           std::ranges::equal(source->tokens, tokens);
  });
}

ContinuationCache::Lease ContinuationCache::Acquire(
    std::span<const ContinuationToken> prompt,
    const CancellationCheck& is_cancelled,
    std::span<const std::uint8_t> input_identity,
    const std::function<void(ContinuationState&)>& prepare_state,
    bool reuse_prompt, std::size_t stable_prefix_tokens,
    std::span<const ContinuationInputPrefix> input_prefixes) {
  if (stable_prefix_tokens > prompt.size())
    throw std::invalid_argument("stable cache prefix exceeds prompt length");
  std::size_t previous = 0;
  for (const auto& prefix : input_prefixes) {
    if (prefix.token_count < previous || prefix.token_count > prompt.size())
      throw std::invalid_argument("input identity boundaries are invalid");
    previous = prefix.token_count;
  }
  const auto matches_input = [&](const auto& identity, std::size_t count) {
    return std::ranges::equal(
        identity, PrefixInputIdentity(input_identity, input_prefixes, count));
  };
  // Sampled once per Acquire: re-reading the atomic on every candidate would
  // cost more than the scan itself when the debug tier is off.
  const bool debug_cache = Logger::Enabled(LogLevel::kDebug);
  // Debug records are collected while the mutex is held and emitted after it
  // is released: the timestamp and stderr write in Logger::Debug must not
  // serialize every other Acquire() behind this one. The buffer is hoisted so
  // a wait does not repeat the allocation under the mutex on every poll.
  struct CandidateSkip {
    std::size_t index;
    std::size_t tokens;
    std::string_view reason;
  };
  std::vector<CandidateSkip> skipped;
  if (debug_cache) {
    // `entries` is fixed once the cache is constructed, so the capacity can be
    // reserved without the mutex.
    skipped.reserve(impl_->entries.size());
  }
  // A wait reports its first non-empty record set once: the set is stable
  // while every slot stays busy, and a long wait must not repeat it on every
  // 10ms poll.
  bool wait_reported = false;
  const auto emit_candidate_skips = [&](const auto& records) {
    for (const auto& skip : records) {
      Logger::Debug("cache",
                    "event=candidate_skip index=" + std::to_string(skip.index) +
                        " tokens=" + std::to_string(skip.tokens) +
                        " prompt_tokens=" + std::to_string(prompt.size()) +
                        " reason=" + std::string(skip.reason));
    }
  };
  while (true) {
    std::unique_lock<std::mutex> lock(impl_->mutex);

    const std::size_t no_entry = impl_->entries.size();
    // Exact full-prompt hits are safe when a shorter checkpoint remains
    // available for clients that rewrite the assistant's opening tokens, or
    // when this full checkpoint was captured after establishing that boundary
    // (its fallback may now live on disk). Migrate unmarked legacy entries.
    const bool has_fallback =
        stable_prefix_tokens == 0 ||
        std::ranges::any_of(impl_->entries, [&](const auto& entry) {
          return entry->valid && entry->snapshot &&
                 (entry->tokens.size() <= stable_prefix_tokens ||
                  (entry->stable_prefix_tokens != 0 &&
                   entry->stable_prefix_tokens <= stable_prefix_tokens)) &&
                 matches_input(entry->input_identity, entry->tokens.size()) &&
                 IsPrefix(entry->tokens, prompt);
        });
    const auto can_reuse = [&](std::size_t count) {
      return has_fallback || count <= stable_prefix_tokens;
    };
    std::size_t source = no_entry;
    std::size_t cached_tokens = 0;
    skipped.clear();
    for (std::size_t index = 0; reuse_prompt && index < impl_->entries.size();
         ++index) {
      const auto& entry = *impl_->entries[index];
      // Evaluated in the original order so the first failing guard is also the
      // reason reported under --log-level=debug.
      std::string_view skip_reason;
      if ((!impl_->snapshot_mode() && !entry.available) || !entry.valid)
        skip_reason = "unavailable";
      else if (!matches_input(entry.input_identity, entry.tokens.size()))
        skip_reason = "input_identity";
      else if (!can_reuse(entry.tokens.size()))
        skip_reason = "stable_prefix_boundary";
      else if (!IsPrefix(entry.tokens, prompt))
        skip_reason = "token_prefix";
      if (!skip_reason.empty()) {
        if (debug_cache) {
          skipped.push_back({index, entry.tokens.size(), skip_reason});
        }
        continue;
      }
      if (source == no_entry || entry.tokens.size() > cached_tokens) {
        source = index;
        cached_tokens = entry.tokens.size();
      }
    }
    // A live final frontier can extend the immutable prompt snapshot. Prefer
    // it on equal prefix lengths too: no restoration or D2H copy is needed.
    std::size_t live_source = no_entry;
    if (reuse_prompt && impl_->snapshot_mode()) {
      for (std::size_t index = 0; index < impl_->entries.size(); ++index) {
        const auto& entry = *impl_->entries[index];
        if (entry.available && !entry.live_tokens.empty() &&
            entry.live_tokens.size() >= cached_tokens &&
            can_reuse(entry.live_tokens.size()) &&
            matches_input(entry.live_identity, entry.live_tokens.size()) &&
            IsPrefix(entry.live_tokens, prompt)) {
          live_source = index;
          cached_tokens = entry.live_tokens.size();
        }
      }
    }

    const bool live_hit = live_source != no_entry;
    const bool cache_hit = live_hit || source != no_entry;
    ContinuationLookup lookup;
    if (!cache_hit) {
      lookup.miss_reason = reuse_prompt ? "no_checkpoint" : "disabled";
      const auto inspect = [&](const auto& tokens, const auto& identity) {
        if (tokens.empty())
          return;
        const auto common =
            static_cast<std::size_t>(std::mismatch(tokens.begin(), tokens.end(),
                                                   prompt.begin(), prompt.end())
                                         .first -
                                     tokens.begin());
        if (lookup.checkpoint_tokens == 0 ||
            common > lookup.common_prefix_tokens) {
          lookup = {matches_input(identity, tokens.size()) ? "prefix_changed"
                                                           : "input_changed",
                    common, tokens.size()};
        }
      };
      if (reuse_prompt) {
        for (const auto& entry : impl_->entries) {
          if (entry->valid)
            inspect(entry->tokens, entry->input_identity);
          inspect(entry->live_tokens, entry->live_identity);
        }
      }
    }
    std::size_t selected = live_hit ? live_source : source;
    bool evicting = false;
    std::size_t evict_index = 0;
    std::size_t evict_tokens = 0;
    std::size_t evict_stable_prefix_tokens = 0;
    if ((impl_->snapshot_mode() && !live_hit) || !cache_hit) {
      selected = no_entry;
      std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
      for (std::size_t index = 0; index < impl_->entries.size(); ++index) {
        const auto& entry = *impl_->entries[index];
        if (entry.available && cache_hit && impl_->entries[source]->snapshot &&
            impl_->entries[source]->snapshot->PrefersState(*entry.state)) {
          selected = index;
          break;
        }
        if (entry.available && entry.state_last_used < oldest) {
          selected = index;
          oldest = entry.state_last_used;
        }
      }
      if (debug_cache && selected != no_entry &&
          impl_->entries[selected]->valid) {
        const auto& slot = *impl_->entries[selected];
        evicting = true;
        evict_index = selected;
        evict_tokens = slot.tokens.size();
        evict_stable_prefix_tokens = slot.stable_prefix_tokens;
      }
    }

    if (selected != no_entry) {
      auto& entry = *impl_->entries[selected];
      entry.available = false;
      const bool needs_invalidation = entry.dirty && !cache_hit;
      entry.dirty = true;
      entry.state_last_used = ++impl_->clock;
      entry.live_tokens.clear();
      entry.live_identity.clear();

      std::shared_ptr<const ContinuationSnapshot> snapshot;
      if (impl_->snapshot_mode()) {
        if (cache_hit && !live_hit) {
          auto& source_entry = *impl_->entries[source];
          snapshot = source_entry.snapshot;
          source_entry.snapshot_last_used = ++impl_->clock;
        }
      } else {
        entry.valid = false;
        entry.tokens.clear();
      }
      lock.unlock();

      std::size_t restored_snapshot_bytes = 0;
      double restore_ms = 0.0;
      try {
        // Still outside the mutex, but inside the guard: these lines allocate,
        // and a failure here must restore `entry.available` rather than leak
        // the slot every other request waits on.
        if (debug_cache) {
          emit_candidate_skips(skipped);
          if (evicting) {
            Logger::Debug(
                "cache",
                "event=capture_evicts index=" + std::to_string(evict_index) +
                    " tokens=" + std::to_string(evict_tokens) +
                    " stable_prefix_tokens=" +
                    std::to_string(evict_stable_prefix_tokens) +
                    " prompt_tokens=" + std::to_string(prompt.size()));
          }
        }

        if (needs_invalidation)
          entry.state->Invalidate();
        if (prepare_state)
          prepare_state(*entry.state);
        if (cache_hit && impl_->snapshot_mode() && !live_hit) {
          if (snapshot == nullptr) {
            throw std::runtime_error(
                "continuation cache snapshot entry is empty");
          }
          restored_snapshot_bytes = snapshot->PayloadBytes();
          const auto restore_start = std::chrono::steady_clock::now();
          impl_->snapshot_support.restore(*entry.state, *snapshot);
          // Restoration replaces the saved RoPE layout. Reattach this
          // request's suffix (which may add images after the checkpoint).
          if (prepare_state && !input_prefixes.empty())
            prepare_state(*entry.state);
          restore_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - restore_start)
                           .count();
        }
      } catch (...) {
        entry.state->Invalidate();
        {
          const std::lock_guard<std::mutex> failure_lock(impl_->mutex);
          entry.available = true;
          entry.dirty = false;
          entry.state_last_used = ++impl_->clock;
        }
        impl_->condition.notify_one();
        throw;
      }
      Lease lease(this, selected, cache_hit, cached_tokens, source,
                  restored_snapshot_bytes, restore_ms);
      lease.input_identity_.assign(input_identity.begin(),
                                   input_identity.end());
      lease.input_prefixes_.assign(input_prefixes.begin(),
                                   input_prefixes.end());
      lease.prompt_tokens_ = prompt.size();
      lease.stable_prefix_tokens_ = stable_prefix_tokens;
      lease.lookup_ = lookup;
      return lease;
    }

    lock.unlock();
    if (is_cancelled && is_cancelled()) {
      return {};
    }
    // No slot could be captured, so this scan's records are the only answer
    // to "why was my checkpoint not reused?" — exactly the question contention
    // raises. Emit them here instead of dropping them for the next poll;
    // nothing is reserved yet, so a failure while formatting simply unwinds
    // out of Acquire.
    if (debug_cache && !wait_reported && !skipped.empty()) {
      emit_candidate_skips(skipped);
      wait_reported = true;
    }
    lock.lock();
    impl_->condition.wait_for(lock, kCancellationPollInterval);
  }
}

std::size_t ContinuationCache::CachedPrefixTokens(
    std::span<const ContinuationToken> prompt,
    std::span<const std::uint8_t> input_identity,
    std::span<const ContinuationInputPrefix> input_prefixes) const {
  const auto matches = [&](const auto& tokens, const auto& identity) {
    return IsPrefix(tokens, prompt) &&
           std::ranges::equal(
               identity, PrefixInputIdentity(input_identity, input_prefixes,
                                             tokens.size()));
  };
  // Same candidates as Acquire: retained checkpoints, and live frontiers
  // whose state is not leased.
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  std::size_t longest = 0;
  for (const auto& entry : impl_->entries) {
    if (entry->valid && (impl_->snapshot_mode() || entry->available) &&
        entry->tokens.size() > longest &&
        matches(entry->tokens, entry->input_identity))
      longest = entry->tokens.size();
    if (impl_->snapshot_mode() && entry->available &&
        entry->live_tokens.size() > longest &&
        matches(entry->live_tokens, entry->live_identity))
      longest = entry->live_tokens.size();
  }
  return longest;
}

std::size_t ContinuationCache::CommonPrefixTokens(
    std::span<const ContinuationToken> prompt,
    std::span<const std::uint8_t> input_identity,
    std::span<const ContinuationInputPrefix> input_prefixes) const {
  // Records store one identity for their complete token list. Requiring it to
  // match the prompt's identity at that length conservatively skips records
  // whose images differ anywhere, so a boundary never spans an image change.
  const auto common = [&](const auto& tokens, const auto& identity) {
    if (tokens.empty() ||
        !std::ranges::equal(
            identity,
            PrefixInputIdentity(input_identity, input_prefixes, tokens.size())))
      return std::size_t{0};
    return static_cast<std::size_t>(std::ranges::mismatch(tokens, prompt).in1 -
                                    tokens.begin());
  };
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  std::size_t longest = 0;
  for (const auto& entry : impl_->entries) {
    if (entry->valid)
      longest = std::max(longest, common(entry->tokens, entry->input_identity));
    longest =
        std::max(longest, common(entry->live_tokens, entry->live_identity));
  }
  return longest;
}

std::size_t ContinuationCache::capacity() const noexcept {
  return impl_->state_count;
}

std::size_t ContinuationCache::entry_capacity() const noexcept {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->snapshot_mode() ? impl_->entries.size() - impl_->state_count
                                : impl_->entries.size();
}

std::size_t ContinuationCache::snapshot_capacity_bytes() const noexcept {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->snapshot_capacity_bytes;
}

std::size_t ContinuationCache::retained_snapshot_bytes() const noexcept {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->retained_snapshot_bytes;
}

std::size_t ContinuationCache::reserved_snapshot_bytes() const noexcept {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->reserved_snapshot_bytes;
}

ContinuationState& ContinuationCache::StateAt(std::size_t index) {
  return *impl_->entries.at(index)->state;
}

bool ContinuationCache::ReserveSnapshot(
    std::size_t source_index, std::size_t snapshot_bytes,
    std::size_t token_count, bool preserve_source, SnapshotPurpose purpose,
    std::span<const ContinuationToken> replacement_prefix,
    std::span<const std::uint8_t> input_identity) {
  std::vector<std::shared_ptr<const ContinuationSnapshot>> removed_snapshots;
  std::vector<SnapshotEvent> events;
  bool admitted = false;
  const int max_priority = MaxRemovalPriority(purpose);
  const auto incoming = purpose == SnapshotPurpose::kContinuation
                            ? replacement_prefix
                            : std::span<const ContinuationToken>{};
  const auto priority_for = [&](std::size_t candidate) {
    return impl_->RemovalPriority(candidate, incoming, input_identity);
  };
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto make_event = [&](SnapshotEventAction action,
                                SnapshotEventReason reason, std::size_t bytes,
                                std::size_t tokens) {
      return SnapshotEvent{
          .action = action,
          .reason = reason,
          .snapshot_bytes = bytes,
          .token_count = tokens,
          .retained_snapshot_bytes = impl_->retained_snapshot_bytes,
          .reserved_snapshot_bytes = impl_->reserved_snapshot_bytes,
          .capacity_bytes = impl_->snapshot_capacity_bytes,
      };
    };
    const auto fits = [&] {
      const std::size_t used =
          impl_->retained_snapshot_bytes + impl_->reserved_snapshot_bytes;
      return snapshot_bytes != 0 && used <= impl_->snapshot_capacity_bytes &&
             snapshot_bytes <= impl_->snapshot_capacity_bytes - used;
    };
    if (purpose != SnapshotPurpose::kContinuation &&
        std::ranges::none_of(impl_->entries.begin() + impl_->state_count,
                             impl_->entries.end(),
                             [](const auto& entry) { return !entry->valid; }) &&
        [&] {
          for (std::size_t i = impl_->state_count; i < impl_->entries.size();
               ++i)
            if (i != source_index && priority_for(i) <= max_priority)
              return false;
          return true;
        }()) {
      events.push_back(make_event(SnapshotEventAction::kSkipped,
                                  SnapshotEventReason::kEntryCapacity,
                                  snapshot_bytes, token_count));
    } else {
      const auto oldest_snapshot = [&](bool allow_source) {
        std::size_t selected = impl_->entries.size();
        std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
        int priority = 4;
        for (std::size_t candidate = impl_->state_count;
             candidate < impl_->entries.size(); ++candidate) {
          const auto& entry = *impl_->entries[candidate];
          if (!entry.valid || entry.snapshot == nullptr ||
              (!allow_source && candidate == source_index)) {
            continue;
          }
          const int rank = priority_for(candidate);
          if (rank > max_priority)
            continue;
          if (rank < priority ||
              (rank == priority && entry.snapshot_last_used < oldest)) {
            selected = candidate;
            priority = rank;
            oldest = entry.snapshot_last_used;
          }
        }
        return selected;
      };

      const auto can_replace_source = [&] {
        if (preserve_source || purpose != SnapshotPurpose::kContinuation ||
            source_index >= impl_->entries.size() || replacement_prefix.empty())
          return false;
        const auto& source = *impl_->entries[source_index];
        return source.valid && source.snapshot &&
               source.tokens.size() < replacement_prefix.size() &&
               std::ranges::equal(source.input_identity, input_identity) &&
               IsPrefix(source.tokens, replacement_prefix);
      };

      const bool can_fit_after_eviction =
          snapshot_bytes != 0 &&
          impl_->reserved_snapshot_bytes <= impl_->snapshot_capacity_bytes &&
          snapshot_bytes <=
              impl_->snapshot_capacity_bytes - impl_->reserved_snapshot_bytes;
      while (can_fit_after_eviction && !fits()) {
        std::size_t target = oldest_snapshot(false);
        // Advance this family before sacrificing another family's last copy.
        // Verify the source: another lease may have replaced its record.
        if (can_replace_source() &&
            (target == impl_->entries.size() || priority_for(target) == 3))
          target = source_index;
        if (target == impl_->entries.size() && !preserve_source) {
          target = oldest_snapshot(true);
        }
        if (target == impl_->entries.size()) {
          break;
        }
        auto& entry = *impl_->entries[target];
        const std::size_t removed_bytes = entry.snapshot_bytes;
        const std::size_t removed_tokens = entry.tokens.size();
        removed_snapshots.push_back(std::move(entry.snapshot));
        entry.tokens.clear();
        entry.snapshot_bytes = 0;
        entry.valid = false;
        impl_->retained_snapshot_bytes -= removed_bytes;
        events.push_back(make_event(SnapshotEventAction::kRemoved,
                                    SnapshotEventReason::kByteCapacity,
                                    removed_bytes, removed_tokens));
      }

      if (fits()) {
        impl_->reserved_snapshot_bytes += snapshot_bytes;
        admitted = true;
      } else {
        events.push_back(make_event(SnapshotEventAction::kSkipped,
                                    SnapshotEventReason::kByteCapacity,
                                    snapshot_bytes, token_count));
      }
    }
  }
  removed_snapshots.clear();
  EmitSnapshotEvents(impl_->snapshot_support.on_event, events);
  return admitted;
}

void ContinuationCache::SkipSnapshot(std::size_t reservation_bytes,
                                     SnapshotEventReason reason,
                                     std::size_t snapshot_bytes,
                                     std::size_t token_count) noexcept {
  SnapshotEvent event;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    if (reservation_bytes <= impl_->reserved_snapshot_bytes) {
      impl_->reserved_snapshot_bytes -= reservation_bytes;
    } else {
      impl_->reserved_snapshot_bytes = 0;
    }
    event = {
        .action = SnapshotEventAction::kSkipped,
        .reason = reason,
        .snapshot_bytes = snapshot_bytes,
        .token_count = token_count,
        .retained_snapshot_bytes = impl_->retained_snapshot_bytes,
        .reserved_snapshot_bytes = impl_->reserved_snapshot_bytes,
        .capacity_bytes = impl_->snapshot_capacity_bytes,
    };
  }
  EmitSnapshotEvents(impl_->snapshot_support.on_event,
                     std::span<const SnapshotEvent>(&event, 1));
}

std::size_t ContinuationCache::Commit(
    std::size_t index, std::size_t source_index, std::size_t reservation_bytes,
    std::vector<ContinuationToken> tokens,
    std::shared_ptr<const ContinuationSnapshot> snapshot,
    std::vector<std::uint8_t> input_identity,
    std::vector<ContinuationToken> live_tokens,
    std::vector<std::uint8_t> live_identity, bool release_state,
    std::size_t* published_index, std::size_t stable_prefix_tokens,
    SnapshotPurpose purpose) {
  const std::size_t token_count = tokens.size();
  const int max_priority = MaxRemovalPriority(purpose);
  const auto incoming = purpose == SnapshotPurpose::kContinuation
                            ? std::span<const ContinuationToken>(tokens)
                            : std::span<const ContinuationToken>{};
  const auto priority_for = [&](std::size_t candidate) {
    return impl_->RemovalPriority(candidate, incoming, input_identity);
  };
  const std::size_t snapshot_bytes =
      snapshot != nullptr ? snapshot->PayloadBytes() : 0;
  SnapshotEventReason skip_reason = SnapshotEventReason::kCaptureFailure;
  bool retain_snapshot = impl_->snapshot_mode() && snapshot != nullptr &&
                         !tokens.empty() && reservation_bytes != 0 &&
                         snapshot_bytes != 0 &&
                         snapshot_bytes == reservation_bytes;
  if (snapshot != nullptr &&
      (reservation_bytes == 0 || snapshot_bytes == 0 ||
       snapshot_bytes != reservation_bytes || tokens.empty())) {
    skip_reason = SnapshotEventReason::kReservationMismatch;
  }

  std::shared_ptr<const ContinuationSnapshot> retained_snapshot;
  if (retain_snapshot) {
    retained_snapshot = std::move(snapshot);
  }

  std::vector<std::shared_ptr<const ContinuationSnapshot>> removed_snapshots;
  std::vector<SnapshotEvent> events;
  std::size_t retained_bytes = 0;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    if (reservation_bytes <= impl_->reserved_snapshot_bytes) {
      impl_->reserved_snapshot_bytes -= reservation_bytes;
    } else {
      impl_->reserved_snapshot_bytes = 0;
      retain_snapshot = false;
      skip_reason = SnapshotEventReason::kReservationMismatch;
    }

    const auto make_event = [&](SnapshotEventAction action,
                                SnapshotEventReason reason, std::size_t bytes,
                                std::size_t token_count) {
      return SnapshotEvent{
          .action = action,
          .reason = reason,
          .snapshot_bytes = bytes,
          .token_count = token_count,
          .retained_snapshot_bytes = impl_->retained_snapshot_bytes,
          .reserved_snapshot_bytes = impl_->reserved_snapshot_bytes,
          .capacity_bytes = impl_->snapshot_capacity_bytes,
      };
    };

    auto& state_entry = *impl_->entries.at(index);
    if (impl_->snapshot_mode()) {
      if (release_state) {
        state_entry.live_tokens = std::move(live_tokens);
        state_entry.live_identity = std::move(live_identity);
      }
      if (retain_snapshot) {
        const std::size_t no_entry = impl_->entries.size();
        std::size_t target = no_entry;
        bool exact_replacement = false;
        bool last_copy_eviction = false;
        for (std::size_t candidate = impl_->state_count;
             candidate < impl_->entries.size(); ++candidate) {
          const auto& entry = *impl_->entries[candidate];
          if (entry.valid && entry.tokens == tokens &&
              entry.input_identity == input_identity) {
            target = candidate;
            exact_replacement = true;
            break;
          }
        }
        if (target == no_entry) {
          for (std::size_t candidate = impl_->state_count;
               candidate < impl_->entries.size(); ++candidate) {
            if (!impl_->entries[candidate]->valid) {
              target = candidate;
              break;
            }
          }
        }
        if (target == no_entry) {
          // An edited branch can replace its old, incompatible tail before
          // evicting checkpoints of the shared prefix. This keeps a new stable
          // boundary and complete prompt within the same bounded entry pool.
          if (source_index < impl_->entries.size()) {
            const auto& source = *impl_->entries[source_index];
            std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
            for (std::size_t candidate = impl_->state_count;
                 candidate < impl_->entries.size(); ++candidate) {
              const auto& entry = *impl_->entries[candidate];
              if (!source.valid || source.tokens.empty() ||
                  !IsPrefix(source.tokens, tokens) || !entry.valid ||
                  candidate == source_index ||
                  entry.input_identity != input_identity ||
                  entry.tokens.size() <= source.tokens.size() ||
                  IsPrefix(entry.tokens, tokens) ||
                  !IsPrefix(source.tokens, entry.tokens) ||
                  priority_for(candidate) > max_priority)
                continue;
              if (entry.snapshot_last_used < oldest) {
                target = candidate;
                oldest = entry.snapshot_last_used;
              }
            }
          }
        }
        if (target == no_entry) {
          std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
          int priority = 4;
          for (std::size_t candidate = impl_->state_count;
               candidate < impl_->entries.size(); ++candidate) {
            if (candidate == source_index &&
                (impl_->entries.size() - impl_->state_count > 1 ||
                 purpose != SnapshotPurpose::kContinuation)) {
              continue;
            }
            const auto& entry = *impl_->entries[candidate];
            const int rank = priority_for(candidate);
            if (rank > max_priority)
              continue;
            if (rank < priority ||
                (rank == priority && entry.snapshot_last_used < oldest)) {
              target = candidate;
              priority = rank;
              oldest = entry.snapshot_last_used;
            }
          }
          last_copy_eviction = priority == 3;
        }
        if (last_copy_eviction && purpose == SnapshotPurpose::kContinuation &&
            source_index < impl_->entries.size()) {
          const auto& source = *impl_->entries[source_index];
          if (source.valid && source.input_identity == input_identity &&
              source.tokens.size() < tokens.size() &&
              IsPrefix(source.tokens, tokens))
            target = source_index;
        }
        if (target == no_entry) {
          retain_snapshot = false;
          skip_reason = SnapshotEventReason::kEntryCapacity;
          // Optional copies cannot displace a last useful checkpoint, even
          // when another publication consumed a free record after reservation.
        }
        if (target != no_entry) {
          auto& snapshot_entry = *impl_->entries[target];
          if (snapshot_entry.snapshot != nullptr) {
            const std::size_t removed_bytes = snapshot_entry.snapshot_bytes;
            const std::size_t removed_tokens = snapshot_entry.tokens.size();
            removed_snapshots.push_back(std::move(snapshot_entry.snapshot));
            impl_->retained_snapshot_bytes -= removed_bytes;
            snapshot_entry.tokens.clear();
            snapshot_entry.snapshot_bytes = 0;
            snapshot_entry.valid = false;
            events.push_back(make_event(
                SnapshotEventAction::kRemoved,
                exact_replacement ? SnapshotEventReason::kExactReplacement
                                  : SnapshotEventReason::kEntryCapacity,
                removed_bytes, removed_tokens));
          }
          const std::size_t used =
              impl_->retained_snapshot_bytes + impl_->reserved_snapshot_bytes;
          if (used > impl_->snapshot_capacity_bytes ||
              snapshot_bytes > impl_->snapshot_capacity_bytes - used) {
            retain_snapshot = false;
            skip_reason = SnapshotEventReason::kReservationMismatch;
          } else {
            snapshot_entry.tokens = std::move(tokens);
            snapshot_entry.input_identity = std::move(input_identity);
            snapshot_entry.snapshot = std::move(retained_snapshot);
            snapshot_entry.snapshot_bytes = snapshot_bytes;
            snapshot_entry.stable_prefix_tokens = stable_prefix_tokens;
            snapshot_entry.purpose = purpose;
            snapshot_entry.valid = !snapshot_entry.tokens.empty();
            snapshot_entry.snapshot_last_used = ++impl_->clock;
            impl_->retained_snapshot_bytes += snapshot_bytes;
            retained_bytes = snapshot_bytes;
            if (published_index)
              *published_index = target;
          }
        }
        if (!retain_snapshot) {
          events.push_back(make_event(SnapshotEventAction::kSkipped,
                                      skip_reason, snapshot_bytes,
                                      token_count));
        }
      } else if (snapshot != nullptr || reservation_bytes != 0) {
        events.push_back(make_event(SnapshotEventAction::kSkipped, skip_reason,
                                    snapshot_bytes, token_count));
      }
    } else {
      state_entry.tokens = std::move(tokens);
      state_entry.input_identity = std::move(input_identity);
      state_entry.valid = !state_entry.tokens.empty();
    }
    if (release_state) {
      state_entry.dirty = true;
      state_entry.available = true;
      state_entry.state_last_used = ++impl_->clock;
    }
  }
  removed_snapshots.clear();
  EmitSnapshotEvents(impl_->snapshot_support.on_event, events);
  impl_->condition.notify_all();
  return retained_bytes;
}

void ContinuationCache::Invalidate(std::size_t index,
                                   std::size_t reservation_bytes) noexcept {
  auto& entry = *impl_->entries[index];
  entry.state->Invalidate();
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    if (reservation_bytes <= impl_->reserved_snapshot_bytes) {
      impl_->reserved_snapshot_bytes -= reservation_bytes;
    } else {
      impl_->reserved_snapshot_bytes = 0;
    }
    if (!impl_->snapshot_mode()) {
      entry.tokens.clear();
      entry.valid = false;
    }
    entry.live_tokens.clear();
    entry.live_identity.clear();
    entry.dirty = false;
    entry.available = true;
    entry.state_last_used = ++impl_->clock;
  }
  impl_->condition.notify_one();
}

}  // namespace gufo::server

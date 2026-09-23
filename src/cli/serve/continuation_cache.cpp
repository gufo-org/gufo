#include "src/cli/serve/continuation_cache.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gufo::server {

namespace {

constexpr auto kCancellationPollInterval = std::chrono::milliseconds{10};

bool IsPrefix(std::span<const ContinuationToken> prefix,
              std::span<const ContinuationToken> tokens) {
  return prefix.size() <= tokens.size() &&
         std::equal(prefix.begin(), prefix.end(), tokens.begin());
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
  std::uint64_t state_last_used{0};
  std::uint64_t snapshot_last_used{0};
  bool available{true};
  bool valid{false};
  bool dirty{false};
};

struct ContinuationCache::Impl {
  std::vector<std::unique_ptr<Entry>> entries;
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
      input_identity_(std::move(other.input_identity_)),
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
    input_identity_ = std::move(other.input_identity_);
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

bool ContinuationCache::Lease::TryReserveSnapshot(std::size_t snapshot_bytes,
                                                  std::size_t token_count,
                                                  bool preserve_source) {
  if (cache_ == nullptr) {
    throw std::logic_error("continuation cache lease is empty");
  }
  if (reserved_snapshot_bytes_ != 0) {
    throw std::logic_error(
        "continuation cache lease already has a snapshot reservation");
  }
  if (!cache_->ReserveSnapshot(source_index_, snapshot_bytes, token_count,
                               preserve_source)) {
    return false;
  }
  reserved_snapshot_bytes_ = snapshot_bytes;
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
  const std::size_t retained = cache_->Commit(
      index_, source_index_, reserved_snapshot_bytes_, std::move(tokens),
      std::move(snapshot), std::move(input_identity_), std::move(live_tokens));
  cache_ = nullptr;
  reserved_snapshot_bytes_ = 0;
  return retained;
}

std::size_t ContinuationCache::Lease::PublishSnapshot(
    std::vector<ContinuationToken> tokens,
    std::shared_ptr<const ContinuationSnapshot> snapshot) {
  if (cache_ == nullptr)
    throw std::logic_error("continuation cache lease is empty");
  const auto retained = cache_->Commit(
      index_, source_index_, reserved_snapshot_bytes_, std::move(tokens),
      std::move(snapshot), input_identity_, {}, false);
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
                                     SnapshotSupport snapshot_support)
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

  impl_->entries.reserve(capacity);
  for (std::size_t index = 0; index < capacity; ++index) {
    auto state = factory();
    if (state == nullptr) {
      throw std::runtime_error(
          "continuation cache state factory returned null");
    }
    impl_->entries.push_back(std::make_unique<Entry>(std::move(state)));
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
           source->input_identity == input_identity_ &&
           std::ranges::equal(source->tokens, tokens);
  });
}

ContinuationCache::Lease ContinuationCache::Acquire(
    std::span<const ContinuationToken> prompt,
    const CancellationCheck& is_cancelled,
    std::span<const std::uint8_t> input_identity,
    const std::function<void(ContinuationState&)>& prepare_state,
    bool reuse_prompt) {
  while (true) {
    std::unique_lock<std::mutex> lock(impl_->mutex);

    const std::size_t no_entry = impl_->entries.size();
    std::size_t source = no_entry;
    std::size_t cached_tokens = 0;
    for (std::size_t index = 0; reuse_prompt && index < impl_->entries.size();
         ++index) {
      const auto& entry = *impl_->entries[index];
      if ((!impl_->snapshot_mode() && !entry.available) || !entry.valid ||
          !std::equal(entry.input_identity.begin(), entry.input_identity.end(),
                      input_identity.begin(), input_identity.end()) ||
          !IsPrefix(entry.tokens, prompt)) {
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
            std::ranges::equal(entry.live_identity, input_identity) &&
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
        if (!std::ranges::equal(identity, input_identity)) {
          if (lookup.miss_reason == "no_checkpoint")
            lookup.miss_reason = "input_changed";
          return;
        }
        const auto common =
            static_cast<std::size_t>(std::mismatch(tokens.begin(), tokens.end(),
                                                   prompt.begin(), prompt.end())
                                         .first -
                                     tokens.begin());
        if (lookup.checkpoint_tokens == 0 ||
            common > lookup.common_prefix_tokens) {
          lookup = {"prefix_changed", common, tokens.size()};
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
    if ((impl_->snapshot_mode() && !live_hit) || !cache_hit) {
      selected = no_entry;
      std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
      for (std::size_t index = 0; index < impl_->entries.size(); ++index) {
        const auto& entry = *impl_->entries[index];
        if (entry.available && entry.state_last_used < oldest) {
          selected = index;
          oldest = entry.state_last_used;
        }
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
      lease.lookup_ = lookup;
      return lease;
    }

    lock.unlock();
    if (is_cancelled && is_cancelled()) {
      return {};
    }
    lock.lock();
    impl_->condition.wait_for(lock, kCancellationPollInterval);
  }
}

std::size_t ContinuationCache::capacity() const noexcept {
  return impl_->entries.size();
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

bool ContinuationCache::ReserveSnapshot(std::size_t source_index,
                                        std::size_t snapshot_bytes,
                                        std::size_t token_count,
                                        bool preserve_source) {
  std::vector<std::shared_ptr<const ContinuationSnapshot>> removed_snapshots;
  std::vector<SnapshotEvent> events;
  bool admitted = false;
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
    const auto oldest_snapshot = [&](bool allow_source) {
      std::size_t selected = impl_->entries.size();
      std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
      for (std::size_t candidate = 0; candidate < impl_->entries.size();
           ++candidate) {
        const auto& entry = *impl_->entries[candidate];
        if (!entry.valid || entry.snapshot == nullptr ||
            (!allow_source && candidate == source_index)) {
          continue;
        }
        if (entry.snapshot_last_used < oldest) {
          selected = candidate;
          oldest = entry.snapshot_last_used;
        }
      }
      return selected;
    };

    const bool can_fit_after_eviction =
        snapshot_bytes != 0 &&
        impl_->reserved_snapshot_bytes <= impl_->snapshot_capacity_bytes &&
        snapshot_bytes <=
            impl_->snapshot_capacity_bytes - impl_->reserved_snapshot_bytes;
    while (can_fit_after_eviction && !fits()) {
      std::size_t target = oldest_snapshot(false);
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
    std::vector<ContinuationToken> live_tokens, bool release_state) {
  const std::size_t token_count = tokens.size();
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
        state_entry.live_identity = input_identity;
      }
      if (retain_snapshot) {
        const std::size_t no_entry = impl_->entries.size();
        std::size_t target = no_entry;
        bool exact_replacement = false;
        for (std::size_t candidate = 0; candidate < impl_->entries.size();
             ++candidate) {
          const auto& entry = *impl_->entries[candidate];
          if (entry.valid && entry.tokens == tokens &&
              entry.input_identity == input_identity) {
            target = candidate;
            exact_replacement = true;
            break;
          }
        }
        if (target == no_entry) {
          for (std::size_t candidate = 0; candidate < impl_->entries.size();
               ++candidate) {
            if (!impl_->entries[candidate]->valid) {
              target = candidate;
              break;
            }
          }
        }
        if (target == no_entry) {
          std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
          for (std::size_t candidate = 0; candidate < impl_->entries.size();
               ++candidate) {
            if (impl_->entries.size() > 1 && candidate == source_index) {
              continue;
            }
            const auto& entry = *impl_->entries[candidate];
            if (entry.snapshot_last_used < oldest) {
              target = candidate;
              oldest = entry.snapshot_last_used;
            }
          }
        }
        if (target == no_entry) {
          target = source_index < impl_->entries.size() ? source_index : index;
        }

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
          snapshot_entry.valid = !snapshot_entry.tokens.empty();
          snapshot_entry.snapshot_last_used = ++impl_->clock;
          impl_->retained_snapshot_bytes += snapshot_bytes;
          retained_bytes = snapshot_bytes;
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

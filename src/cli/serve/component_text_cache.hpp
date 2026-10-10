#ifndef GUFO_SERVER_COMPONENT_TEXT_CACHE_HPP_
#define GUFO_SERVER_COMPONENT_TEXT_CACHE_HPP_

#include <functional>
#include <memory>
#include <optional>

#include "src/cache/checkpoint.hpp"
#include "src/cache/streaming.hpp"
#include "src/cli/serve/component_cache_metrics.hpp"
#include "src/cli/serve/continuation_cache.hpp"

namespace gufo::server {
class TextModelRunner;
class TextRunnerState;
class TextRunnerSnapshot;
struct TextRunnerDiskCacheOptions;
struct TextPromptContext;

// Model-independent serving orchestration; the model supplies physical storage
// and binds its ordinary execution facade to adapter-owned slots.
struct ComponentCaptureBuffer {
  std::span<std::byte> bytes;
  std::function<cache::Payload()> finish;
};
struct ComponentCacheResources {
  std::shared_ptr<void> owner;
  std::unique_ptr<cache::Adapter> adapter;
  cache::StreamLease stream;
  std::function<ComponentCaptureBuffer(const cache::PayloadRequest&)> allocate;
  cache::StagingAllocator staging;
};

class ComponentTextCache {
public:
  struct Impl;
  class Lease {
  public:
    struct Impl;
    explicit Lease(std::unique_ptr<Impl>);
    ~Lease();
    Lease(const Lease&) = delete;
    [[nodiscard]] ContinuationState& state() const;
    [[nodiscard]] bool cache_hit() const noexcept;
    [[nodiscard]] std::size_t cached_tokens() const noexcept;
    [[nodiscard]] std::size_t restored_snapshot_bytes() const noexcept;
    [[nodiscard]] double restore_ms() const noexcept;
    [[nodiscard]] bool restored_from_disk() const noexcept;
    [[nodiscard]] ContinuationLookup lookup() const noexcept;
    [[nodiscard]] bool HasSnapshotFor(std::span<const ContinuationToken>) const;
    [[nodiscard]] bool TryReserveSnapshot(std::size_t, std::size_t, bool,
                                          SnapshotPurpose,
                                          std::span<const ContinuationToken>);
    void SkipSnapshot(SnapshotEventReason, std::size_t, std::size_t) noexcept;
    std::size_t PublishSnapshot(std::vector<ContinuationToken>,
                                std::shared_ptr<const ContinuationSnapshot>,
                                bool);
    std::size_t Commit(std::vector<ContinuationToken>,
                       std::shared_ptr<const ContinuationSnapshot>,
                       std::vector<ContinuationToken>);
    void Invalidate() noexcept;
    [[nodiscard]] std::size_t CaptureBytes() const;
    [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Capture();
    [[nodiscard]] ComponentCacheMetrics Metrics() const;
    [[nodiscard]] std::vector<std::size_t> PlannedCaptures(
        std::size_t stable) const;

  private:
    friend class ComponentTextCache;
    std::unique_ptr<Impl> impl_;
  };
  ComponentTextCache(std::shared_ptr<TextModelRunner>, std::size_t,
                     std::size_t ram_bytes,
                     const std::optional<TextRunnerDiskCacheOptions>&);
  ~ComponentTextCache();
  [[nodiscard]] std::unique_ptr<Lease> Acquire(
      std::span<const ContinuationToken>,
      const ContinuationCache::CancellationCheck&,
      std::shared_ptr<const TextPromptContext>, bool, std::size_t, bool);
  [[nodiscard]] std::size_t CachedPrefixTokens(
      std::span<const ContinuationToken>, const TextPromptContext*,
      bool common = false) const;
  [[nodiscard]] std::size_t capacity() const noexcept;

private:
  std::unique_ptr<Impl> impl_;
};

// Only one implementation exists for a pool. Legacy models keep their old
// lease; converted models use the component store and never construct it.
class ServingCacheLease {
public:
  ServingCacheLease(ContinuationCache::Lease lease)
      : legacy_(std::move(lease)) {}
  ServingCacheLease(std::unique_ptr<ComponentTextCache::Lease> lease)
      : component_(std::move(lease)) {}
  ServingCacheLease(ServingCacheLease&&) noexcept = default;
  ServingCacheLease& operator=(ServingCacheLease&&) noexcept = default;
  [[nodiscard]] explicit operator bool() const noexcept {
    return component_ || bool(legacy_);
  }
  [[nodiscard]] bool component() const noexcept { return bool(component_); }
  [[nodiscard]] ContinuationState& state() const;
  [[nodiscard]] bool cache_hit() const noexcept;
  [[nodiscard]] std::size_t cached_tokens() const noexcept;
  [[nodiscard]] std::size_t restored_snapshot_bytes() const noexcept;
  [[nodiscard]] double restore_ms() const noexcept;
  [[nodiscard]] bool restored_from_disk() const noexcept;
  [[nodiscard]] ContinuationLookup lookup() const noexcept;
  [[nodiscard]] bool HasSnapshotFor(std::span<const ContinuationToken>) const;
  void AdoptRestoredPrefix(std::size_t, std::size_t, double);
  [[nodiscard]] bool TryReserveSnapshot(
      std::size_t, std::size_t, bool = false,
      SnapshotPurpose = SnapshotPurpose::kContinuation,
      std::span<const ContinuationToken> = {});
  void SkipSnapshot(SnapshotEventReason, std::size_t, std::size_t) noexcept;
  std::size_t PublishSnapshot(std::vector<ContinuationToken>,
                              std::shared_ptr<const ContinuationSnapshot>,
                              bool = false);
  std::size_t Commit(std::vector<ContinuationToken>,
                     std::shared_ptr<const ContinuationSnapshot> = {},
                     std::vector<ContinuationToken> = {});
  void Invalidate() noexcept;
  [[nodiscard]] std::size_t CaptureBytes() const;
  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Capture();
  [[nodiscard]] std::optional<ComponentCacheMetrics> Metrics() const;
  [[nodiscard]] std::vector<std::size_t> PlannedCaptures(
      std::size_t stable) const;

private:
  ContinuationCache::Lease legacy_;
  std::unique_ptr<ComponentTextCache::Lease> component_;
};
}  // namespace gufo::server
#endif

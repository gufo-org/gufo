#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXPERT_STREAM_RUNTIME_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXPERT_STREAM_RUNTIME_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// Memory-bounded routed-expert streaming (issue #427) for Flash-Next.
///
/// Streaming layers keep `slots` experts resident in per-matrix device slabs
/// holding the artifact's native encodings; the other experts live in the
/// mmap'd GGUF payload, where the OS page cache is the warm tier. Router and
/// shared expert stay fully resident and untouched.
///
/// A forward resolves the router's expert ids on the host against a
/// per-layer LRU, loads misses into slots through a pinned staging ring,
/// and rewrites the id buffer with slot numbers. The projections then run
/// the existing expert kernels on the slabs: the weight bytes and the
/// arithmetic are identical to the fully-resident vector route, so decode
/// output is bit-exact; only placement changes. Streamed layers take the
/// vector route at every pass width, so their prefill can round
/// differently from the fully-resident build's tiled route. Because
/// placement is host-side orchestration inside a forward, decode-graph
/// capture and replay stay off while streaming (see the Executor's
/// `graph` predicates).
///
/// Thread safety: the Executor drives forwards one layer at a time on its
/// single compute stream; this class holds no lock and is bound to that
/// discipline.
class ExpertStreamCache {
public:
  struct Stats {
    std::uint64_t lookups{0};
    std::uint64_t loads{0};
    std::uint64_t load_bytes{0};
  };

  /// Builds the runtime for every layer whose `stream.slots > 0` (trunk
  /// layers plus the MTP block, in DeviceModel order). `ok()` reports
  /// whether construction succeeded; `error()` gives the reason. Miss loads
  /// are queued on `stream`, the Executor's compute stream.
  ExpertStreamCache(const DeviceModel& model, hipStream_t stream);
  ~ExpertStreamCache();
  ExpertStreamCache(const ExpertStreamCache&) = delete;
  ExpertStreamCache& operator=(const ExpertStreamCache&) = delete;

  [[nodiscard]] bool ok() const noexcept { return error_.empty(); }
  [[nodiscard]] const std::string& error() const noexcept { return error_; }

  /// True when this slab belongs to a streamed layer.
  [[nodiscard]] bool enabled(const ExpertStreamLayer& slab) const noexcept {
    return State(slab) != nullptr;
  }

  /// Resolves ids [ids_offset, ids_offset + ids_count) in `ids_host` to
  /// cache slots — loading misses onto the stream — and rewrites the host
  /// range and the same device range in place. Ids already negative
  /// (RouterTopK's padding marker) pass through untouched. The caller must
  /// have planned token groups so the range's distinct experts fit the
  /// cache (StreamPlanGroups); exceeding it fails instead of evicting a
  /// slot another id in the same range reads.
  [[nodiscard]] bool PrepareGroup(const ExpertStreamLayer& slab,
                                  std::int32_t* ids_host,
                                  std::uint32_t ids_offset,
                                  std::uint32_t ids_count,
                                  std::int32_t* ids_device);

  [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
  struct Entry {
    std::int32_t expert{-1};  ///< Resident expert id, -1 when empty.
  };
  struct LayerState {
    ExpertStreamLayer slab;
    std::vector<std::int32_t> slot_of_expert;  ///< -1 when not resident.
    std::vector<Entry> entries;
    std::vector<std::uint64_t> last_use;
  };

  [[nodiscard]] LayerState* State(const ExpertStreamLayer& slab) const;
  /// Loads one expert into `slot` on the stream through the pinned ring.
  [[nodiscard]] bool Load(LayerState& st, std::uint32_t slot,
                          std::int32_t expert);
  /// An empty slot, else the LRU slot untouched by the current group
  /// (last_use trailing the group's stamp); UINT32_MAX if every slot
  /// belongs to the group, i.e. its distinct set exceeded the cache.
  [[nodiscard]] std::uint32_t Victim(const LayerState& st) const;

  std::vector<std::unique_ptr<LayerState>> states_;
  hipStream_t stream_;
  std::string error_;
  std::uint64_t stamp_{0};
  // Pinned staging ring: one entry per in-flight expert upload, reused
  // once its transfer event has completed on the compute stream.
  static constexpr std::uint32_t kRing = 4;
  std::vector<std::uint8_t> ring_backing_;
  std::uint8_t* ring_{nullptr};
  bool ring_pinned_{false};
  std::size_t ring_stride_{0};
  std::uint32_t ring_next_{0};
  hipEvent_t ring_free_[kRing]{nullptr};
  Stats stats_;
};

/// Streaming expert groups run the exact vector expert routes, which
/// cover 1..kVecBatch tokens per launch; the streaming cache exposes this
/// so slab sizing can keep `slots / routed_per_token` at least this wide
/// without gaining wider groups.
inline constexpr std::uint32_t kStreamGroupTokens = 8;

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXPERT_STREAM_RUNTIME_HPP_

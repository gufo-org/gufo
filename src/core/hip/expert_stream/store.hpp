#pragma once

/* Streams a mixture-of-experts model's routed experts from files into a
 * bounded device cache. It knows nothing about the model: the caller
 * describes where each expert's pieces are on disk (Layout) and how the
 * cache should behave (Policy), routes each MoE layer with Prepare, and
 * passes the returned per-piece tables to its kernels. Model-specific
 * knowledge (rankings, memory headroom, file discovery) stays with the
 * caller. */

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/platform/mapped_file.hpp"

namespace gufo::expert_stream {

/* Most pieces per expert: gate/up/down are three, a fused gate_up two, and a
 * format storing several tensors per expert back to back may use one. */
inline constexpr uint32_t kMaxPieces = 16;

/* One piece of every expert of a layer: expert e's bytes are
 * [offset + e * bytes, offset + (e + 1) * bytes) of file, as for a GGUF
 * expert tensor whose expert dimension is outermost. The store reopens the
 * handle for its reads (platform::ReopenRandomReadFile: unbuffered where the
 * platform allows it) and never closes `file`, which must outlive the store. */
struct Piece {
  platform::PlatformFile file = platform::kInvalidFile;
  uint64_t offset = 0;
  uint64_t bytes = 0;  // one expert
};

/* The streamed experts. Layers are numbered in evaluation order: the
 * prediction window advances from a layer to the next and wraps to 0. A cache
 * slot holds an expert's pieces back to back in piece order. */
struct Layout {
  uint32_t experts = 0;  // routed experts per layer
  uint32_t top_k = 0;    // route rows per token
  uint32_t pieces = 0;   // pieces per expert, at most kMaxPieces
  /* Most tokens one Prepare call routes: batches up to this size split their
   * hit and miss rows (Overlap). */
  uint32_t max_batch_tokens = 0;
  /* Entries in each layer's table (Prepare): 0 for `experts`. A larger
   * width keeps a fixed table shape across builds with fewer experts; the
   * entries past `experts` are null. */
  uint32_t table_experts = 0;
  /* Readable bytes kernels may touch past the end of an expert's last
   * piece (a kernel that reads whole k-iterations over-reads the last row).
   * Every cache chunk and stage buffer keeps this much beyond its last
   * expert; the values read there must not affect results. */
  uint32_t read_margin_bytes = 0;
  std::vector<std::vector<Piece>> layers;  // [layer][piece]
};

/* How the cache sizes itself and evicts. The defaults are the values
 * measured on DeepSeek V4.1 Flash (the deepseek41 extension). */
/* An expert to cache first, most wanted first; count weighs its starting
 * uses. */
struct Ranked {
  uint32_t layer;
  uint32_t expert;
  uint64_t count;
};

struct Policy {
  /* Bounds the cache when nonzero, below the grant Fit receives. */
  uint64_t cache_limit_bytes = 0;
  /* Free memory below which the cache stops growing. */
  uint64_t live_floor_bytes = 1ull << 30;
  /* The experts Fit caches first, most wanted first; every expert it leaves
   * out follows, interleaved across layers, as far as the grant and free
   * memory reach. Each (layer, expert) at most once; empty, every expert is
   * interleaved. `ranking_source` names it in the log. */
  std::vector<Ranked> ranking;
  std::string ranking_source;
  /* Uses decay with this half-life, in tokens of the layer's clock; a
   * Prepare counts at most use_cap uses per expert and advances the clock by
   * at most clock_rows tokens. The victim is the slot with the largest
   * age / 2^class, class = floor(log2(uses)) up to max_class. */
  double use_half_life = 2048.0;
  uint32_t use_cap = 4;
  double max_class = 7.0;
  uint32_t clock_rows = 16;
  /* Seeded experts start with seed_uses * min(8, log2(1 + count)) uses. */
  double seed_uses = 0.5;
  /* Once full, a batch larger than recent_rows tokens admits only the
   * experts its last recent_rows tokens route and those it routes at least
   * max(4, tokens / 32) times. */
  uint32_t recent_rows = 16;
  /* Batches of at least sweep_tokens adopt a prefill-scale prefetch. */
  uint32_t sweep_tokens = 1024;
  uint32_t demand_stage_rows = 64;    // expert rows the demand stage holds
  uint32_t prefetch_stage_rows = 96;  // rows of the two prefetch halves
  /* A binary routing trace (see Store::Tracing) when nonempty. */
  std::string trace_path;
  /* Prefix of log lines and errors. */
  std::string log_tag = "experts";
};

struct Stats {
  uint64_t uses = 0;
  uint64_t hits = 0;
  uint64_t bytes_read = 0;
  uint64_t tokens = 0;
  uint64_t stall_ns = 0;
  uint64_t route_wait_ns = 0;  // part of stall_ns: waiting for the routes
  uint64_t read_wait_ns = 0;   // part of stall_ns: waiting for demand reads
  uint64_t cache_bytes = 0;
  uint64_t prefetch_bytes = 0;          // staged by background prefetch jobs
  uint64_t prefetch_dropped_bytes = 0;  // prefetched bytes no prepare consumed
};

/* What Prepare stages for the layer, for work overlapped with the reads. The
 * id rows keep their route positions: hit_ids holds the resident experts
 * with misses as -1, miss_ids the read experts with hits as -1, so a launch
 * over one array zeroes exactly the rows the other launch fills. table[p]
 * is the layer's device [experts] array of piece p's weight bases (a cache
 * slot or the stage row a read just filled), so kernels read cached experts
 * without a staging copy. */
struct Overlap {
  uint32_t layer = 0;
  uint32_t rows = 0;                  // n_tokens * top_k route rows
  uint32_t miss_rows = 0;             // rows served by reads this call
  const int32_t* hit_ids = nullptr;   // device rows; null unless the batch both
                                      // hits and misses within the split size
  const int32_t* miss_ids = nullptr;  // as hit_ids
  const void* const* table = nullptr;  // [pieces] device tables
};

class Store {
public:
  struct Impl;

  /* Null (with a message) when the layout is invalid or a buffer, thread or
   * stream cannot be created. */
  static std::unique_ptr<Store> Create(const Layout& layout,
                                       const Policy& policy);
  ~Store();
  Store(const Store&) = delete;
  Store& operator=(const Store&) = delete;

  /* Splits every piece read between its file and a tails file holding the
   * end of each piece, normally on a second drive, so both read at once. Call
   * before Fit. The file must match the layout's pieces byte for byte;
   * false (with a message) leaves reads unchanged. On btrfs, create it in a
   * NOCOW (`chattr +C`) directory: direct reads of copy-on-write files are
   * not direct. */
  bool AttachTails(const char* path);

  /* Sizes the cache to the host's grant and fills it: the cache and its
   * prefetch stage take at most grant_bytes. Allocates the stage (bought on
   * the first prompt otherwise), frees the newest chunks, keeping the
   * experts eviction ranks highest, when the cache is over the grant, and
   * reads Policy::ranking into the room left, while more than
   * live_floor_bytes stay available. Until the first Fit nothing is cached
   * (misses are read into the stage); from then on the cache grows only up
   * to the grant. The host calls it once everything else it allocates at
   * startup exists: serving once the sessions and the prompt cache's backing
   * do (see TextRunnerExpertCacheClaim). A nonzero cache_limit_bytes still
   * bounds the cache. False when the stage, the shrink or the reads
   * failed. */
  bool Fit(uint64_t grant_bytes);

  /* Gives memory back to the host after Fit: frees whole chunks, newest
   * first, until at least `bytes` went or the cache is empty, and holds the
   * cache at what is left until Restore. Each size class keeps the experts
   * eviction ranks highest: those in the freed chunks move into the slots
   * of the lowest. Also frees the rows misses borrowed when no slot could
   * be lent. Waits for queued predictions and drops any background
   * prefetch; call it only while no Prepare runs (serving does when idle).
   * Returns the bytes freed; the cache is held even when that is 0. */
  uint64_t Release(uint64_t bytes);
  /* Lets the cache grow back to its grant after Release; it regrows as
   * misses need it, while more than FloorBytes stay available. Returns the
   * bytes the limit rose by: 0 when Release held nothing back. */
  uint64_t Restore();
  /* Bytes Release freed since Fit or the last Restore. */
  uint64_t ReleasedBytes() const;
  /* Free memory below which the cache stops growing
   * (Policy::live_floor_bytes). */
  uint64_t FloorBytes() const;

  /* Routes are device [n_tokens][top_k]; -1 marks a dropped row. On success
   * tables[0..pieces) hold the layer's device tables, valid until the next
   * call. Calls are serialized.
   *
   * `stream` is the caller's: Prepare reads the routes once the work queued
   * on it before the call has run (so that work may still read the previous
   * call's tables and stage), and queues every copy the tables and the
   * experts they name depend on there, so kernels queued on it afterwards
   * see them. The null stream (the default) keeps the legacy default-stream
   * ordering. With routes_ready, an event recorded on `stream` once the
   * routes are written, the routes are read without waiting for work queued
   * after it; that work must not touch the expert stage, slots or tables.
   * An overlap callback queues its work on `stream` too. */
  bool Prepare(uint32_t layer, const int32_t* routes, uint32_t n_tokens,
               const void** tables,
               bool (*overlap)(void*, const Overlap*) = nullptr,
               void* overlap_context = nullptr,
               hipEvent_t routes_ready = nullptr, hipStream_t stream = nullptr);

  /* Queue a background read of the given uncached experts of `layer` into
   * the stage set the layer's next Prepare consumes, without displacing
   * anything. The reads run only while no demand phase does; the next
   * Prepare for the layer adopts what is still pending, and any other drops
   * it. `wide` marks a prefill-scale expert set, which only a batch of at
   * least sweep_tokens adopts. Best effort: false when the layer is not the
   * next Prepare's target, has no uncached listed experts, or the job was
   * refused. */
  bool Prefetch(uint32_t layer, const int32_t* experts, uint32_t count,
                bool wide = false);

  /* Records a router prediction whose ids `done` signals in caller-owned
   * pinned memory; a store thread submits the prefetch when the event lands.
   * False when the record was refused (best effort by design). */
  bool Predict(uint32_t layer, hipEvent_t done, const int32_t* ids,
               uint32_t count, bool wide);

  /* Reopens the prediction window at `layer`, the next Prepare's target, when
   * an evaluation starts somewhere other than after the last prepared layer.
   * Every kernel of the previous evaluation's layers must be ordered before
   * that Prepare's route read. */
  void Expect(uint32_t layer);

  /* Waits until every recorded prediction has been submitted, so the ids and
   * event may be released. */
  void PredictFlush();

  /* Policy::trace_path writes a binary routing trace (the layout, policy
   * and seeded cache, then each Prepare's routes, hits and timing; a new
   * snapshot of the cache after Fit, Release or Restore changed it) for
   * offline replay with tools/expert_stream/ (format in its README).
   * Callers add router predictions for later layers with TracePrediction
   * once the target layer's Prepare returned. Tracing is a measurement mode:
   * callers should not predict while it runs. */
  bool Tracing() const;
  void TracePrediction(uint32_t layer, uint32_t lead, const int32_t* ids,
                       uint32_t n_tokens, uint32_t k);

  /* True when no background prefetch work is queued or running. */
  bool PrefetchIdle() const;
  Stats GetStats() const;
  void Log() const;

private:
  explicit Store(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::expert_stream

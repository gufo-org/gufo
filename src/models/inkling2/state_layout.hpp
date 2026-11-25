// Component layout of an Inkling 2 Mini session for Gufo's component cache.
//
// One session holds, per layer visit (48 of them), a K and a V row stream (f16,
// 8 heads x 128 = 1024 values = 2 KiB per row), and a short-convolution
// history. The 8 full-attention visits are append rows (the whole history); the
// 40 sliding-window visits are window rows (the last 1024). Everything else
// (conv state, logits of the last token, position and block framing) is private
// state that is valid only at its exact boundary. With a DSpark drafter the
// drafter's context rows (two window layers, one global layer, same row shape)
// and its Markov vector join them, so a restored prompt keeps its proposals.
//
// Host-side arithmetic only (no HIP): the engine access lives in
// SessionStateAccess (state_layout.cpp), the single place that knows how a
// Session lays its buffers out.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <gufo/ext/adapter.hpp>
#include <span>
#include <vector>

#include "src/models/inkling2/config.hpp"

namespace gufo::models::inkling2 {

class Session;
class Engine;
class SessionHooks;

inline constexpr std::size_t kVisits = 48;
inline constexpr std::size_t kKvRowBytes = 1024 * 2;  // one K or V row, f16
inline constexpr std::size_t kConvBytesPerVisit =
    6656 * 3 * 4;  // F32 [6656][3]
inline constexpr std::size_t kConvBytes = kVisits * kConvBytesPerVisit;
inline constexpr cache::Rows kRowsPerChunk = 256;
inline constexpr cache::Rows kWindowRows = 1024;
inline constexpr std::size_t kMetaBytes = 64;
/// Drafter layers after the 48 target visits: visit kVisits + l.
inline constexpr std::uint32_t kDraftLayers = 3;
inline constexpr std::size_t kDraftFBytes = 2304 * 4;  // Markov vector, F32

enum class Part : std::uint8_t { kMeta, kLogits, kConv, kKey, kValue, kDraftF };

struct StateComponent {
  cache::ComponentDescriptor descriptor;
  Part part;
  std::uint32_t visit{0};  // slot index 0..47 for kKey / kValue
  bool global{false};
};

/// What the runner keeps beside the device state of a slot: where the text
/// stands inside the model's block structure (see runner.cpp Piece()).
enum class Block : std::uint8_t {
  kIdle,
  kAfterHeader,
  kText,
  kThinking,
  kTool,
  kToolBody
};

/// The 64-byte private metadata component. Fixed layout, little endian on the
/// supported hosts.
struct StateMeta {
  static constexpr std::uint32_t kVersion = 2;
  std::uint32_t version{kVersion};
  std::uint32_t block{0};
  std::uint64_t position{0};
  std::uint32_t logits_valid{0};
  std::uint32_t ring_rows{0};
  /// Drafter context: rows [draft_first, draft_valid), Markov vector of
  /// draft_f_pos (all zero without a drafter).
  std::uint64_t draft_first{0}, draft_valid{0}, draft_f_pos{0};
  std::uint8_t reserved[16]{};
};
static_assert(sizeof(StateMeta) == kMetaBytes);

class StateLayout {
public:
  /// `ring_rows`: rows of one sliding-window ring (at least window + chunk -
  /// 1). `drafter`: the session also holds a DSpark drafter's context.
  StateLayout(const Inkling2Config& config, std::uint32_t ring_rows,
              bool drafter = false);

  [[nodiscard]] std::span<const StateComponent> components() const {
    return components_;
  }
  [[nodiscard]] const StateComponent& Component(cache::ComponentId id) const;
  [[nodiscard]] std::uint32_t ring_rows() const { return ring_rows_; }
  [[nodiscard]] std::size_t logits_bytes() const { return logits_bytes_; }
  [[nodiscard]] std::size_t vocab() const { return vocab_; }
  [[nodiscard]] bool drafter() const { return drafter_; }

  /// The frontier of every component for a session standing at `position`
  /// tokens.
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      std::size_t position) const;
  /// Bytes (rows, or private state) a restore to `position` must write into
  /// component `c`. For window rows only the rows a continuation reads are
  /// restored: [FirstChunkRow, position).
  [[nodiscard]] static std::pair<cache::Rows, cache::Rows> RestoreRange(
      const cache::ComponentDescriptor& d, cache::Rows position);
  /// Pinned backing classes for a RAM cache of `ram_bytes` (one class of 64 KiB
  /// pages).
  [[nodiscard]] std::vector<cache::BackingClass> BackingPlan(
      std::size_t ram_bytes) const;
  /// Device bytes of one session for `context` tokens (matches
  /// Engine::session_bytes()).
  [[nodiscard]] std::size_t SessionBytes(std::uint32_t context) const;
  [[nodiscard]] std::size_t PrivateBytes() const;

private:
  std::vector<StateComponent> components_;
  std::uint32_t ring_rows_;
  std::size_t logits_bytes_, vocab_;
  bool drafter_;
};

/// The only code that touches a Session's buffers for the cache: row ranges in
/// and out (with the ring arithmetic of the window visits), conv history,
/// logits and the frontier. Synchronous (copies on the engine's default
/// stream); the engine mutex is NOT taken, so these are safe from the guard
/// callbacks inside Forward and from capture threads of an idle slot.
class SessionStateAccess {
public:
  [[nodiscard]] static std::size_t Position(const Session&);
  [[nodiscard]] static bool Global(const Session&, std::uint32_t visit);
  [[nodiscard]] static bool HasLogits(const Session&);
  static void LogitsOut(const Session&, std::byte* dst, std::size_t bytes);
  static void RowsOut(const Session&, std::uint32_t visit, bool value,
                      cache::Rows first, cache::Rows end, std::byte* dst);
  static void RowsIn(Session&, std::uint32_t visit, bool value,
                     cache::Rows first, cache::Rows end, const std::byte* src);
  /// `n` bytes at byte `inner` of the conv history of one visit.
  static void ConvVisitOut(const Session&, std::uint32_t visit,
                           std::size_t inner, std::size_t n, std::byte* dst);
  static void ConvVisitIn(Session&, std::uint32_t visit, std::size_t inner,
                          std::size_t n, const std::byte* src);
  /// Sets the frontier after a restore: position and (when `logits` is
  /// non-null) the logits.
  static void SetFrontier(Session&, std::size_t position, const float* logits,
                          std::size_t n);
  static void SetHooks(Session&, SessionHooks*);
  /// The drafter's context bounds and Markov vector (kDraftFBytes).
  static void DraftOut(const Session&, StateMeta& meta);
  static void DraftFOut(const Session&, std::size_t inner, std::size_t n,
                        std::byte* dst);
  static void DraftFIn(Session&, std::size_t inner, std::size_t n,
                       const std::byte* src);
  /// After SetFrontier: the restored drafter context, when `meta` holds one
  /// that ends at the frontier.
  static void DraftIn(Session&, const StateMeta& meta);
  /// Device bytes of rows [first, end) of a global visit, which never wrap.
  [[nodiscard]] static std::span<const std::byte> RowsView(const Session&,
                                                           std::uint32_t visit,
                                                           bool value,
                                                           cache::Rows first,
                                                           cache::Rows end);
  static void PrepareRestore(Session&, std::size_t rows);
  /// Copies `n` bytes from an arbitrary device pointer (e.g. an in-pass
  /// checkpoint buffer) to a cache host buffer via the snapshot copy path.
  static void DeviceBytesOut(const void* src, std::size_t n, std::byte* dst);

private:
  /// One K or V row stream: a target visit's slot, or (visit >= kVisits) a
  /// drafter layer.
  struct Stream {
    void* base;
    std::size_t rows;
    bool global;
  };
  static Stream RowStream(const Session&, std::uint32_t visit, bool value);
};

}  // namespace gufo::models::inkling2

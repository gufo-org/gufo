#ifndef GUFO_MEM_ACCOUNT_HPP_
#define GUFO_MEM_ACCOUNT_HPP_

#include <cstddef>
#include <functional>
#include <mutex>
#include <vector>

namespace gufo::mem {

/// Rows of state laid out in `segments` equal runs, each starting
/// `segment_stride` bytes after the previous one: one segment for a K/V
/// buffer per layer, or one per layer and plane for a layer-major buffer
/// such as Qwen 27B's [plane][layer][position][width]. Rows grow from the
/// start of every segment.
struct RegionShape {
  std::size_t row_bytes{0};
  std::size_t rows{0};
  std::size_t segments{1};
  /// Zero means rows * row_bytes: segments back to back.
  std::size_t segment_stride{0};
};

/// The segment stride of `shape` (rows * row_bytes when it is zero).
[[nodiscard]] std::size_t SegmentStride(const RegionShape& shape) noexcept;

/// Bytes of 2 MiB pages that rows [0, rows) of every segment of `shape`
/// cover: the union of the segments' page ranges. RegionPageBytes(shape,
/// shape.rows) is what a region promises.
[[nodiscard]] std::size_t RegionPageBytes(const RegionShape& shape,
                                          std::size_t rows) noexcept;

class StateReservation;

/// How `sessions` sessions of `session_bytes` each fit in `capacity`, when
/// `demand_bytes` of each are demand-mapped state (K/V) that takes pages
/// only as the session writes it.
struct SessionFit {
  enum class Kind {
    /// Every session fits at full context: no cap, nothing changes.
    kFits,
    /// They do not all fit at full context, but every session's fixed state
    /// does, with room for at least one session at full context: the
    /// sessions share `cap_bytes` of demand-mapped state (#715).
    kShared,
    /// Not even that: refuse the configuration.
    kRefused,
  };
  Kind kind{Kind::kFits};
  /// Sessions that fit at full context; zero when none does.
  std::size_t full_sessions{0};
  /// With kShared: what the demand-mapped state of all sessions may take
  /// together. The fixed state of all sessions plus this equals the memory
  /// `full_sessions` sessions at full context take, so every other budget
  /// (the RAM cache, snapshots, scratch) sees what it would with only them.
  std::size_t cap_bytes{0};
};
[[nodiscard]] SessionFit FitSessions(std::size_t sessions,
                                     std::size_t session_bytes,
                                     std::size_t demand_bytes,
                                     std::size_t capacity) noexcept;

/// The state of one session: the regions it writes and the pages it may map.
///
/// A session's regions reserve their whole context as addresses, but the
/// pages behind them are charged to the process's state budget only as far
/// as the session needs them: its **need** (the tokens its current request
/// may write, or the live frontier it keeps between requests), or more when
/// more is mapped. Without a cap on the budget (StateLedger::SetPromiseCap)
/// every charge succeeds and nothing changes. With a cap, the sessions
/// together never charge more than it, so a server may configure more
/// sessions at full context than fit at once (#715): admission claims a
/// request's tokens before it runs, and the pages of an idle session can be
/// reclaimed for another one.
///
/// Tokens convert to rows per region (`tokens_per_row`: one for K/V,
/// the compression ratio for compressed rows), plus `margin_tokens` the
/// model backs past its frontier (tile-rounded reads).
class StateAccount {
public:
  explicit StateAccount(std::size_t margin_tokens = 0) noexcept
      : margin_tokens_(margin_tokens) {}
  ~StateAccount();
  StateAccount(const StateAccount&) = delete;
  StateAccount& operator=(const StateAccount&) = delete;

  /// A region of this session. `trim(rows)` releases the region's pages
  /// beyond its first `rows` rows of every segment; it is called only by
  /// TrimTo(), when no work that reads or writes the region is in flight.
  void AddRegion(const RegionShape& shape, std::size_t tokens_per_row,
                 std::function<void(std::size_t rows)> trim);

  /// Page bytes of every region for its first `tokens` tokens and the
  /// margin; zero for no tokens.
  [[nodiscard]] std::size_t BytesFor(std::size_t tokens) const;
  /// Rows of a region with `tokens_per_row` that `tokens` need, margin
  /// included, clamped to `rows`.
  [[nodiscard]] std::size_t RowsFor(std::size_t tokens,
                                    std::size_t tokens_per_row,
                                    std::size_t rows) const noexcept;
  /// Page bytes of every region at its full size.
  [[nodiscard]] std::size_t full_bytes() const;

  /// Raises the need to `tokens` and charges its pages. False, with the
  /// need unchanged, when the budget cannot charge them. Pages already
  /// covered by `reservation` are taken from it first.
  [[nodiscard]] bool Claim(std::size_t tokens,
                           StateReservation* reservation = nullptr);
  /// Sets the need to `tokens`, lower or not: after a request, the live
  /// frontier the session keeps. Pages mapped beyond it stay charged until
  /// trimmed. Never fails when it lowers the need; a higher need is a
  /// Claim() and may.
  [[nodiscard]] bool SetNeed(std::size_t tokens);
  /// Charges `bytes` of newly mapped pages. False when the budget cannot
  /// charge them; the caller does not map them then.
  [[nodiscard]] bool OnMap(std::size_t bytes);
  void OnUnmap(std::size_t bytes) noexcept;

  /// Releases every page beyond what `tokens` need and lowers the need to
  /// it. The caller ensures no in-flight GPU work uses the regions: an
  /// unmap waits for the whole device. Returns the bytes released.
  std::size_t TrimTo(std::size_t tokens);

  [[nodiscard]] std::size_t need_tokens() const;
  [[nodiscard]] std::size_t mapped_bytes() const;
  /// What this session costs the budget: the larger of its need's pages
  /// and its mapped pages.
  [[nodiscard]] std::size_t charged_bytes() const;
  /// Mapped bytes past the need: released by TrimTo(need) without losing
  /// state.
  [[nodiscard]] std::size_t tail_bytes() const;

private:
  struct Member {
    RegionShape shape;
    std::size_t tokens_per_row{1};
    std::function<void(std::size_t)> trim;
  };
  [[nodiscard]] std::size_t BytesForLocked(std::size_t tokens) const;
  [[nodiscard]] std::size_t ChargeLocked() const;
  /// Moves the budget's charge from `before` to the current charge.
  [[nodiscard]] bool Recharge(std::size_t before,
                              StateReservation* reservation);

  const std::size_t margin_tokens_;
  mutable std::mutex mutex_;
  std::vector<Member> members_;
  std::size_t need_{0};
  std::size_t mapped_{0};
  std::size_t charged_{0};
};

/// Budget charged ahead of choosing a session: admission reserves a
/// request's worst case, then the session it lands on claims from it
/// (StateAccount::Claim). Whatever is left returns to the budget when the
/// reservation ends.
class StateReservation {
public:
  StateReservation() = default;
  /// Charges `bytes`; check with ok().
  explicit StateReservation(std::size_t bytes);
  ~StateReservation() { Release(); }
  StateReservation(StateReservation&& other) noexcept
      : bytes_(other.bytes_), ok_(other.ok_) {
    other.bytes_ = 0;
  }
  StateReservation& operator=(StateReservation&& other) noexcept;
  StateReservation(const StateReservation&) = delete;
  StateReservation& operator=(const StateReservation&) = delete;

  [[nodiscard]] bool ok() const noexcept { return ok_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  /// Gives up to `bytes` of the charge to a claim; returns what it gave.
  std::size_t Take(std::size_t bytes) noexcept;
  /// Takes back bytes Take() gave to a claim that failed.
  void Return(std::size_t bytes) noexcept;
  void Release() noexcept;

private:
  std::size_t bytes_{0};
  bool ok_{false};
};

}  // namespace gufo::mem

#endif  // GUFO_MEM_ACCOUNT_HPP_

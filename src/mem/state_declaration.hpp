#ifndef GUFO_MEM_STATE_DECLARATION_HPP_
#define GUFO_MEM_STATE_DECLARATION_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/mem/account.hpp"

namespace gufo::mem {

/// How a buffer of session state takes memory.
enum class StateKind : std::uint8_t {
  /// Rows that grow with the context: full or compressed K/V, indexer keys,
  /// MTP K/V, scratch whose size follows the depth. One fixed address for
  /// the whole context, pages mapped as rows are written, charged to the
  /// session's StateAccount (`tokens_per_row` tokens per row).
  kContext,
  /// Rows at a fixed address, backed whole when the session is created:
  /// small row buffers whose pages checkpoints share like the K/V's.
  kWhole,
  /// State of a fixed size per session (recurrent, convolution, rings,
  /// control blocks, draft rows): one device allocation, zeroed.
  kFixed,
};

struct StateBuffer {
  StateKind kind{StateKind::kFixed};
  /// kFixed: one row of the buffer's bytes.
  RegionShape shape{};
  std::size_t tokens_per_row{1};
};

/// Index of a buffer in its declaration, in the order it was declared.
using StateBufferId = std::size_t;

/// How StateDeclaration counts a region's bytes.
enum class StateBytes : std::uint8_t {
  /// Its size: what it allocates without demand mapping.
  kSize,
  /// Its whole 2 MiB pages: what it reserves and its account charges at full
  /// context where state is demand-mapped (mem::DemandMapped()).
  kPages,
};

/// The state one session of a model holds, declared by the model and
/// allocated by the core (mem::SessionState, in src/core/hip). The same
/// declaration gives the bytes a runner claims
/// (TextRunnerResourceClaim::per_request_state_bytes and
/// demand_mapped_state_bytes), so a claim cannot drift from what sessions
/// allocate. A fixed buffer counts its bytes, at least one. Rollback rows,
/// checkpoints and other state a model allocates itself are not part of a
/// declaration; the model adds their bytes.
class StateDeclaration {
public:
  /// `margin_tokens`: tokens the model backs past a frontier (a tile of
  /// rows read past the last written one), charged with every need.
  explicit StateDeclaration(std::size_t margin_tokens = 0) noexcept
      : margin_tokens_(margin_tokens) {}

  /// A kContext buffer of `shape.rows` rows of `tokens_per_row` tokens.
  StateBufferId Context(const RegionShape& shape,
                        std::size_t tokens_per_row = 1);
  /// A kWhole buffer.
  StateBufferId Whole(const RegionShape& shape);
  /// A kFixed buffer of `bytes`.
  StateBufferId Fixed(std::size_t bytes);

  [[nodiscard]] const std::vector<StateBuffer>& buffers() const noexcept {
    return buffers_;
  }
  [[nodiscard]] const StateBuffer& buffer(StateBufferId id) const {
    return buffers_.at(id);
  }
  [[nodiscard]] std::size_t margin_tokens() const noexcept {
    return margin_tokens_;
  }
  /// Whether any buffer grows with the context.
  [[nodiscard]] bool has_context() const noexcept;

  /// Bytes of buffer `id`.
  [[nodiscard]] std::size_t Bytes(StateBufferId id, StateBytes count) const;
  /// Bytes of every buffer.
  [[nodiscard]] std::size_t Bytes(StateBytes count) const noexcept;
  /// Bytes of the kContext buffers: what a session maps only as it writes
  /// rows, where state is demand-mapped (otherwise every buffer is
  /// allocated whole, and a runner claims none as demand-mapped).
  [[nodiscard]] std::size_t ContextBytes(StateBytes count) const noexcept;

private:
  std::size_t margin_tokens_{0};
  std::vector<StateBuffer> buffers_;
};

/// Bytes a region of `shape` spans: every segment's stride but the last's,
/// plus one segment's rows. What a region allocates without demand mapping.
[[nodiscard]] std::size_t RegionSpanBytes(const RegionShape& shape) noexcept;

}  // namespace gufo::mem

#endif  // GUFO_MEM_STATE_DECLARATION_HPP_

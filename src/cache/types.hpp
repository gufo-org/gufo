#ifndef GUFO_CACHE_TYPES_HPP_
#define GUFO_CACHE_TYPES_HPP_

#include <cstddef>
#include <cstdint>

namespace gufo::cache {

using Token = std::uint32_t;
using Rows = std::uint64_t;

// IDs are internal handles, never model pointers or user-provided labels.
struct ComponentId {
  std::uint32_t value{0};
  bool operator==(const ComponentId&) const = default;
};
struct SlotId {
  std::uint64_t value{0};
  bool operator==(const SlotId&) const = default;
};

struct BorrowedLocation {
  SlotId slot;
  std::uint64_t generation{};
  bool operator==(const BorrowedLocation&) const = default;
};

enum class ComponentKind : std::uint8_t {
  kAppendRows,
  kPrivateState,
  // Append rows of which a continuation reads only the last window_rows
  // (sliding-window attention). A checkpoint at n keeps the chunks from the
  // one containing row n - window_rows, so earlier chunks can be released.
  kWindowRows,
};

struct ComponentDescriptor {
  ComponentId id;
  std::uint32_t layout_version{0};
  ComponentKind kind{ComponentKind::kAppendRows};
  std::size_t row_bytes{0};
  Rows rows_per_chunk{0};
  std::size_t state_bytes{0};
  // kWindowRows only: trailing rows a continuation reads; zero otherwise.
  Rows window_rows{0};
};

[[nodiscard]] constexpr bool HasRows(ComponentKind kind) noexcept {
  return kind == ComponentKind::kAppendRows ||
         kind == ComponentKind::kWindowRows;
}
// First row a continuation from `rows` valid rows still reads.
[[nodiscard]] constexpr Rows WindowFirstRow(const ComponentDescriptor& d,
                                            Rows rows) noexcept {
  return d.kind == ComponentKind::kWindowRows && rows > d.window_rows
             ? rows - d.window_rows
             : 0;
}
// First row of the first chunk a checkpoint at `rows` retains.
[[nodiscard]] constexpr Rows FirstChunkRow(const ComponentDescriptor& d,
                                           Rows rows) noexcept {
  const Rows first = WindowFirstRow(d, rows);
  return d.rows_per_chunk ? first - first % d.rows_per_chunk : 0;
}

struct ComponentPosition {
  ComponentId id;
  // Append components count their own rows, not target prompt tokens.
  // Private components report their exact logical execution boundary.
  Rows valid_rows{0};
  bool operator==(const ComponentPosition&) const = default;
};

struct Capabilities {
  // False explicitly describes models with no token-prefix continuation.
  bool continuation{false};
  bool persistent_encoding{false};
  // ResidentRows and the kept-rows BeginRestore are implemented.
  bool in_place_restore{false};
};

}  // namespace gufo::cache

#endif

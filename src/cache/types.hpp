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
};

struct ComponentDescriptor {
  ComponentId id;
  std::uint32_t layout_version{0};
  ComponentKind kind{ComponentKind::kAppendRows};
  std::size_t row_bytes{0};
  Rows rows_per_chunk{0};
  std::size_t state_bytes{0};
};

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
};

}  // namespace gufo::cache

#endif

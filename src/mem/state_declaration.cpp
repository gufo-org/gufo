#include "src/mem/state_declaration.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace gufo::mem {

namespace {

void CheckShape(const RegionShape& shape) {
  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  if (shape.row_bytes == 0 || shape.rows == 0 || shape.segments == 0)
    throw std::invalid_argument("state buffer without rows");
  // Compare in rows to avoid overflowing rows * row_bytes.
  if (shape.segment_stride != 0 &&
      shape.rows > shape.segment_stride / shape.row_bytes)
    throw std::invalid_argument("state buffer segments overlap");
  // A span that does not fit in a size_t is not declarable: sessions
  // refuse it (CheckedShape, src/core/hip/state_region.cpp), and the
  // declared bytes a runner claims would wrap instead of spanning.
  if (shape.rows > kMax / shape.row_bytes)
    throw std::invalid_argument("state buffer bytes overflow");
  const std::size_t run = shape.rows * shape.row_bytes;
  const std::size_t stride =
      shape.segment_stride != 0 ? shape.segment_stride : run;
  if (shape.segments - 1 > (kMax - run) / stride)
    throw std::invalid_argument("state buffer span overflow");
}

}  // namespace

std::size_t RegionSpanBytes(const RegionShape& shape) noexcept {
  if (shape.segments == 0)
    return 0;
  return (shape.segments - 1) * SegmentStride(shape) +
         shape.rows * shape.row_bytes;
}

StateBufferId StateDeclaration::Context(const RegionShape& shape,
                                        std::size_t tokens_per_row) {
  CheckShape(shape);
  buffers_.push_back(
      {.kind = StateKind::kContext,
       .shape = shape,
       .tokens_per_row = std::max<std::size_t>(tokens_per_row, 1)});
  return buffers_.size() - 1;
}

StateBufferId StateDeclaration::Whole(const RegionShape& shape) {
  CheckShape(shape);
  buffers_.push_back({.kind = StateKind::kWhole, .shape = shape});
  return buffers_.size() - 1;
}

StateBufferId StateDeclaration::Fixed(std::size_t bytes) {
  buffers_.push_back(
      {.kind = StateKind::kFixed,
       .shape = {.row_bytes = std::max<std::size_t>(bytes, 1), .rows = 1}});
  return buffers_.size() - 1;
}

bool StateDeclaration::has_context() const noexcept {
  return std::any_of(buffers_.begin(), buffers_.end(), [](const auto& b) {
    return b.kind == StateKind::kContext;
  });
}

std::size_t StateDeclaration::Bytes(StateBufferId id, StateBytes count) const {
  const StateBuffer& b = buffers_.at(id);
  if (b.kind == StateKind::kFixed)
    return b.shape.row_bytes;
  return count == StateBytes::kPages ? RegionPageBytes(b.shape, b.shape.rows)
                                     : RegionSpanBytes(b.shape);
}

std::size_t StateDeclaration::Bytes(StateBytes count) const noexcept {
  std::size_t bytes = 0;
  for (StateBufferId id = 0; id < buffers_.size(); ++id)
    bytes += Bytes(id, count);
  return bytes;
}

std::size_t StateDeclaration::ContextBytes(StateBytes count) const noexcept {
  std::size_t bytes = 0;
  for (StateBufferId id = 0; id < buffers_.size(); ++id) {
    if (buffers_[id].kind == StateKind::kContext)
      bytes += Bytes(id, count);
  }
  return bytes;
}

}  // namespace gufo::mem

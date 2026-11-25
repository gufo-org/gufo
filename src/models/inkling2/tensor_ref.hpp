#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>
#include <string_view>

namespace gufo::models::inkling2 {

using core::GgmlType;

/// ggml type 40. Gufo core does not know it (the reader reports such tensors
/// as unsized), so the extension sizes it itself.
inline constexpr GgmlType kNvfp4 = static_cast<GgmlType>(40);
/// One NVFP4 block (ggml-common.h `block_nvfp4`): 4 sub-blocks of 16 e2m1
/// values, each with its own E4M3 scale: 4 scale bytes + 32 bytes of nibbles.
inline constexpr std::uint64_t kNvfp4BlockElements = 64;
inline constexpr std::uint64_t kNvfp4BlockBytes = 4 + kNvfp4BlockElements / 2;
static_assert(kNvfp4BlockBytes == 36);

/// Bytes of an NVFP4 tensor with `elements` values, or 0 if they do not fill
/// whole blocks.
[[nodiscard]] constexpr std::uint64_t Nvfp4Bytes(std::uint64_t elements) {
  return elements % kNvfp4BlockElements == 0
             ? elements / kNvfp4BlockElements * kNvfp4BlockBytes
             : 0;
}

[[nodiscard]] constexpr std::string_view TypeName(GgmlType type) {
  return type == kNvfp4 ? "NVFP4" : core::ToString(type);
}

/// The tensor types Inkling 2 Mini artifacts use. Anything else is refused.
[[nodiscard]] constexpr bool IsSupportedType(GgmlType type) {
  switch (type) {
    case GgmlType::kF32:
    case GgmlType::kF16:
    case GgmlType::kBF16:
    case GgmlType::kQ8_0:
    case GgmlType::kQ4_K:
    case GgmlType::kQ5_K:
    case GgmlType::kQ6_K:
    case GgmlType::kQ5_0:
    case GgmlType::kIQ4_NL:
    case GgmlType::kIQ4_XS:
      return true;
    default:
      return type == kNvfp4;
  }
}

/// A tensor inside the mapped GGUF. Valid while the reader that produced it
/// is alive. `ne` is ggml order: `ne[0]` is the contiguous (input) dimension.
struct TensorRef {
  const void* data = nullptr;
  std::size_t size_bytes = 0;  // exact bytes the tensor occupies
  GgmlType type = GgmlType::kF32;
  std::array<std::uint64_t, 3> ne{1, 1, 1};
  std::uint32_t n_dims = 0;

  explicit operator bool() const { return data != nullptr; }
  [[nodiscard]] std::uint64_t Elements() const { return ne[0] * ne[1] * ne[2]; }
  [[nodiscard]] const float* F32() const {
    return static_cast<const float*>(data);
  }
};

}  // namespace gufo::models::inkling2

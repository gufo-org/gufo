#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_GYRO_TQ_TYPES_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_GYRO_TQ_TYPES_HPP_

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <gufo/ext/gguf_reader.hpp>

#include "src/models/qwen38_flash_next/gyro/tq_ref.h"

namespace gufo::models::qwen38_flash_next::gyro {

/// Gyro trellis GGUF storage types (AgentionAI,
/// https://gitlab.com/agentionai-group/agention-gyro). Gufo core does not name
/// them, so a reader opened with `GgufOpenOptions::allow_unsized_tensors` hands
/// such a file to the provider that owns the layout, and `tq_ref.h` is its
/// normative byte-level definition.
inline constexpr core::GgmlType kTQ2T =
    static_cast<core::GgmlType>(TQ_TYPE_TQ2_T);
inline constexpr core::GgmlType kTQK6 =
    static_cast<core::GgmlType>(TQ_TYPE_TQK6);
inline constexpr core::GgmlType kTQK7 =
    static_cast<core::GgmlType>(TQ_TYPE_TQK7);
inline constexpr core::GgmlType kTQP5 =
    static_cast<core::GgmlType>(TQ_TYPE_TQP5);
inline constexpr core::GgmlType kTQP6 =
    static_cast<core::GgmlType>(TQ_TYPE_TQP6);
inline constexpr core::GgmlType kTQP7 =
    static_cast<core::GgmlType>(TQ_TYPE_TQP7);
inline constexpr core::GgmlType kTQK5 =
    static_cast<core::GgmlType>(TQ_TYPE_TQK5);
inline constexpr core::GgmlType kTQK4 =
    static_cast<core::GgmlType>(TQ_TYPE_TQK4);

/// Weights per block, and the TQ2_T block size this extension has kernels for.
inline constexpr std::size_t kBlock = TQ_BLOCK;
inline constexpr std::size_t kTQ2TBlockBytes = 34;

[[nodiscard]] constexpr std::uint16_t TypeId(core::GgmlType type) noexcept {
  return static_cast<std::uint16_t>(type);
}

/// True for every Gyro trellis id, whether or not a kernel exists for it. The
/// ids are one contiguous range: 144 TQ2_T, 145..146 TQK6/7, 147..149 TQP5..7,
/// 150..151 TQK5/4.
[[nodiscard]] constexpr bool IsGyro(core::GgmlType type) noexcept {
  const auto id = TypeId(type);
  return id >= TQ_TYPE_TQ2_T && id <= TQ_TYPE_TQK4;
}

[[nodiscard]] constexpr bool IsTQ2T(core::GgmlType type) noexcept {
  return TypeId(type) == TQ_TYPE_TQ2_T;
}

[[nodiscard]] inline std::string_view TypeName(core::GgmlType type) noexcept {
  const auto* name = tq_type_name(TypeId(type));
  return name != nullptr ? std::string_view{name} : std::string_view{};
}

/// Elements per encoded block, or 0 when this is not a Gyro layout.
[[nodiscard]] constexpr std::size_t BlockElements(
    core::GgmlType type) noexcept {
  return IsGyro(type) ? TQ_BLOCK : 0;
}

/// Encoded bytes for `elements` weights, or 0 when the type or length is not a
/// Gyro row. Core reports these tensors as unsized, so the binder uses this to
/// check the bytes the file reserves.
[[nodiscard]] inline std::size_t EncodedSizeBytes(
    core::GgmlType type, std::uint64_t elements) noexcept {
  return IsGyro(type)
             ? tq_row_bytes(TypeId(type), static_cast<std::int64_t>(elements))
             : 0;
}

/// Reference decode of `k` Gyro weights (`k % 128 == 0`) to float. The Gyro
/// file stores the block-Hadamard rotated row (T·w); the caller undoes T with
/// the tensor's sign vector when it needs the unrotated weights.
inline void DequantizeRow(core::GgmlType type, const void* src, float* dst,
                          std::size_t k) noexcept {
  tq_dequantize_row(TypeId(type), src, dst, static_cast<std::int64_t>(k));
}

/// Applies T^-1 to a decoded row: `v <- signs * (H_128 v)` per 128-wide slice.
inline void UnrotateRow(float* v, std::size_t n, const std::int8_t* signs) {
  tq_hadamard_unrotate(v, static_cast<std::int64_t>(n), TQ_BLOCK, signs);
}

}  // namespace gufo::models::qwen38_flash_next::gyro

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_GYRO_TQ_TYPES_HPP_

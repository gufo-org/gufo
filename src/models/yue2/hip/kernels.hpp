#ifndef GUFO_MODELS_YUE2_HIP_KERNELS_HPP_
#define GUFO_MODELS_YUE2_HIP_KERNELS_HPP_

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::yue2::hip {

// One-token decode reads every AR weight once per token and is bound by
// memory bandwidth, so the decode copy of those weights stores each BF16
// value in 12 bits without losing any of them. A row keeps its largest
// exponent; a weight is its sign, a 4-bit exponent code relative to the row
// and its 7 mantissa bits. The few weights more than 14 binades below the
// row's largest (about 0.04% of YuE2's) take code 0 and their bits from the
// row's escape list.

/// Weights per packed unit: eight 12-bit fields in three 32-bit words, field
/// `k` at bits `[12 k, 12 k + 12)` of the little-endian 96-bit unit. A field
/// is `sign << 11 | code << 7 | mantissa`.
inline constexpr std::size_t kPackedUnitWeights = 8;
inline constexpr std::size_t kPackedUnitWords = 3;

struct PackedRow {
  /// `(largest exponent - 15) << 23`: added to `code << 23` it gives the FP32
  /// exponent field of a weight.
  std::uint32_t exponent_base{0};
  /// The row's escapes in the escape list, in no particular order.
  std::uint32_t escape_begin{0};
  std::uint32_t escape_count{0};
  std::uint32_t reserved{0};
};

struct PackedEscape {
  std::uint32_t column{0};
  /// The weight as FP32 bits (`bf16 << 16`).
  std::uint32_t bits{0};
};

/// Words of one packed row of `columns` weights.
[[nodiscard]] constexpr std::size_t PackedRowWords(std::size_t columns) {
  return columns / kPackedUnitWeights * kPackedUnitWords;
}

/// A packed matrix on the device: `PackedRowWords(columns)` words and one
/// `PackedRow` per row, and the escape list the rows index.
struct PackedMatrix {
  std::uint32_t* units{nullptr};
  PackedRow* rows{nullptr};
  PackedEscape* escapes{nullptr};
};

/// First packing pass over `rows` BF16 rows of `columns` weights (a multiple
/// of 8): writes each row's `exponent_base` and `escape_count` to
/// `matrix.rows + first_row` and adds the escapes to `*escape_total`.
void LaunchPackedRowsScan(const void* weights_bfloat16, std::size_t rows,
                          std::size_t columns, const PackedMatrix& matrix,
                          std::size_t first_row, std::uint32_t* escape_total,
                          hipStream_t stream);

/// Second pass: writes the units and escapes of the scanned rows, taking each
/// row's escape range from `*escape_cursor`.
void LaunchPackedRowsWrite(const void* weights_bfloat16, std::size_t rows,
                           std::size_t columns, const PackedMatrix& matrix,
                           std::size_t first_row, std::uint32_t* escape_cursor,
                           hipStream_t stream);

/// Decodes the packed rows as the GEMV does and adds the number of weights
/// that differ from the BF16 rows to `*mismatches`.
void LaunchPackedRowsVerify(const void* weights_bfloat16, std::size_t rows,
                            std::size_t columns, const PackedMatrix& matrix,
                            std::size_t first_row, std::uint32_t* mismatches,
                            hipStream_t stream);

/// `rows` consecutive matrix rows and where their FP32 products go.
struct PackedOutput {
  float* output{nullptr};
  std::uint32_t rows{0};
};

inline constexpr std::size_t kPackedMaxOutputs = 3;

/// One-token GEMV over packed rows: `outputs[i]` receives the products of
/// the next `outputs[i].rows` matrix rows, starting at `first_row`, with the
/// `columns` BF16 values at `input_bfloat16` (16-byte aligned). Every weight
/// decodes to its BF16 value and each row is summed in the order of
/// `eager::LaunchBfloat16Gemv`, so the results equal that kernel's bit for
/// bit while reading three quarters of the bytes.
void LaunchPackedGemv(const PackedMatrix& matrix, std::size_t first_row,
                      const PackedOutput* outputs, std::size_t output_count,
                      const void* input_bfloat16, std::size_t columns,
                      hipStream_t stream);

/// The acoustic input rows: `vae2llm(x) + time + position` with each upstream
/// BF16 boundary. Row 0 and row `frames + 1` are the zero latent boundaries;
/// row r takes `latent_pos_embed[r]`. `state` is FP32 `[frames][64]`;
/// `hidden` is FP32 `[frames + 2][2048]`.
void LaunchLatentInput(const float* state, std::size_t frames,
                       const void* weight, const void* bias,
                       const void* time_row, const void* positions,
                       float* hidden, hipStream_t stream);

/// `velocity[t] = bf16(projected[t + 1] + bias)` for the content rows of the
/// `llm2vae` projection, `projected` being `[frames + 2][64]`.
void LaunchVelocityOutput(const float* projected, const void* bias,
                          std::size_t frames, float* velocity,
                          hipStream_t stream);

/// `output = state - scale * velocity`, element-wise in FP32.
void LaunchOdeStep(const float* state, const float* velocity, float scale,
                   std::size_t count, float* output, hipStream_t stream);

/// Rounds FP32 logits to BF16 and copies them densely.
void LaunchBfloat16Logits(const float* input, std::size_t count, float* output,
                          hipStream_t stream);

/// One key segment of the acoustic attention: `count` key rows at
/// `keys + (kv_head * key_stride + r) * 128` and their values transposed at
/// `values + (kv_head * 128 + channel) * value_pitch + r`, `value_pitch`
/// being `count` rounded up to 32 with zero (finite) padding.
struct NarSegment {
  const std::uint16_t* keys{nullptr};
  const std::uint16_t* values{nullptr};
  std::uint32_t count{0};
  std::uint32_t key_stride{0};
  std::uint32_t value_pitch{0};
};

/// Rounds `count` up to the 32-key tile the acoustic attention consumes.
[[nodiscard]] constexpr std::uint32_t NarValuePitch(std::uint32_t count) {
  return (count + 31U) & ~31U;
}

/// Transposes head-major BF16 values `[kv_heads][stride][128]` (first
/// `count` rows) to `[kv_heads][128][NarValuePitch(count)]`, zero-padded.
void LaunchTransposeValues(const std::uint16_t* values, std::uint32_t stride,
                           std::uint32_t count, std::uint32_t kv_heads,
                           std::uint16_t* output, hipStream_t stream);

/// Unmasked acoustic attention of head-major BF16 queries `[heads][rows][128]`
/// over the prefix and the latent rows: BF16 QK on matrix instructions, FP32
/// online softmax over 32-key tiles, probabilities rounded to BF16 before a
/// matrix value product (PyTorch's fused BF16 attention). Writes BF16
/// `[rows][heads][128]`. Query head h reads key head `h / group`.
void LaunchNarAttention(const std::uint16_t* queries, NarSegment prefix,
                        NarSegment latent, std::uint16_t* output,
                        std::uint32_t heads, std::uint32_t rows,
                        std::uint32_t group, hipStream_t stream);

// Oobleck decoder operators. Activations are channel-major FP32 rows of
// `pitch = pad + length + pad` elements whose margins are zero, so every
// convolution tap is a plain shifted view.

/// `output = x + sin(alpha * x)^2 / (beta + 1e-9)` per channel, writing zero
/// margins. `alpha` and `inverse_beta` are already exponentiated.
void LaunchSnake(const float* input, float* output, std::size_t channels,
                 std::size_t length, std::size_t pad, const float* alpha,
                 const float* inverse_beta, hipStream_t stream);

/// Sets every interior element to its channel's bias (zero when null) and
/// the margins to zero.
void LaunchFillBias(float* output, std::size_t channels, std::size_t length,
                    std::size_t pad, const float* bias, hipStream_t stream);

/// Adds the channel bias to the interior.
void LaunchAddBias(float* values, std::size_t channels, std::size_t length,
                   std::size_t pad, const float* bias, hipStream_t stream);

/// Interleaves the `stride` polyphase outputs `[stride][channels][phase]` of
/// a transposed convolution into `output` with its bias:
/// `output[c][t] = phases[(t + padding) % stride][c][(t + padding) / stride]`.
void LaunchInterleavePhases(const float* phases, std::size_t phase_length,
                            std::size_t stride, std::size_t padding,
                            std::size_t channels, std::size_t length,
                            std::size_t pad, const float* bias, float* output,
                            hipStream_t stream);

/// Copies stereo samples `[first, first + count)` of a two-channel padded
/// activation to frame-interleaved `output`.
void LaunchInterleaveStereo(const float* input, std::size_t length,
                            std::size_t pad, std::size_t first,
                            std::size_t count, float* output,
                            hipStream_t stream);

}  // namespace gufo::models::yue2::hip

#endif  // GUFO_MODELS_YUE2_HIP_KERNELS_HPP_

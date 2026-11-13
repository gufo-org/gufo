#pragma once

// FP32 CPU operators that run between NPU matrix products (norms, rotary
// embeddings, attention, activations), shared by the NPU model runtimes. Work
// spreads over a persistent pool of CPU threads.

#include <cstddef>
#include <cstdint>
#include <functional>

namespace gufo::npu::host {

/// Calls fn(begin, end) over chunks of [0, n) on the pool's threads and
/// returns when all are done. Chunks hold at least `grain` items. Nested calls
/// run inline.
void ParallelFor(std::size_t n, std::size_t grain,
                 const std::function<void(std::size_t, std::size_t)>& fn);
/// Within its scope, the calling thread's ParallelFor runs on pool `index`
/// of `count` pools that share the hardware threads, so `count` threads can
/// run parallel host work at once without queueing behind each other.
class SharePool {
public:
  SharePool(unsigned index, unsigned count);
  ~SharePool();
  SharePool(const SharePool&) = delete;
  SharePool& operator=(const SharePool&) = delete;

private:
  void* previous_;
};

/// The pool's thread count (the hardware's, at least 1).
[[nodiscard]] std::size_t Threads();

[[nodiscard]] float Bf16ToFloat(std::uint16_t h);
[[nodiscard]] std::uint16_t FloatToBf16(float f);  // round to nearest even
/// Rounds every value to the nearest bf16 in place.
void RoundBf16(float* x, std::size_t n);
void Bf16ToFloat(const std::uint16_t* src, float* dst, std::size_t n);
void FloatToBf16(const float* src, std::uint16_t* dst, std::size_t n);

/// y = x / rms(x) * weight per row of `dim` (weight null: none; `offset`
/// added to the weight, e.g. 1 for (1 + w) norms). x and y may alias.
void RmsNormRows(const float* x, float* y, std::size_t rows, std::size_t dim,
                 const float* weight, float eps, float offset = 0.0f);
/// As RmsNormRows, written as bf16 (round to nearest even).
void RmsNormRowsBf16(const float* x, std::uint16_t* y, std::size_t rows,
                     std::size_t dim, const float* weight, float eps);
/// state = (state + RmsNorm(x) * weight) * scale per row: a normed residual
/// add.
void AddNormRows(float* state, const float* x, std::size_t rows,
                 std::size_t dim, const float* weight, float eps, float scale);
/// y = (x - mean) / sqrt(var + eps) * weight + bias per row (either null).
void LayerNormRows(const float* x, float* y, std::size_t rows, std::size_t dim,
                   const float* weight, const float* bias, float eps);

/// Rotary embedding, rotate-half (NeoX) pairing (i, i + rotary / 2) over the
/// first `rotary` dims of each head: row r holds `heads` heads of `head_dim`
/// from x + r * row_stride, at position positions[r].
void RopeNeox(float* x, std::size_t rows, std::size_t row_stride,
              std::size_t heads, std::size_t head_dim, std::size_t rotary,
              const std::int32_t* positions, float theta);

enum class Activation { kNone, kSilu, kGeluTanh, kGeluErf };
[[nodiscard]] float Activate(Activation a, float x);
/// x[i] = act(x[i]) + bias[i % dim] applied before act when bias is set.
void ActivateRows(float* x, std::size_t rows, std::size_t dim,
                  const float* bias, Activation a);
/// out[r][i] = act(gate[r][i]) * up[r][i], with gate/up at row strides.
void GatedRows(const float* gate, std::size_t gate_stride, const float* up,
               std::size_t up_stride, float* out, std::size_t out_stride,
               std::size_t rows, std::size_t dim, Activation a);

/// Which keys a query sees: [begin, end) of key rows for query row i.
using KeyRange =
    std::function<void(std::size_t row, std::size_t* begin, std::size_t* end)>;

/// Multi-head attention with grouped KV heads. q [rows][heads][dim] at row
/// stride q_stride (floats), k/v [keys][kv_heads][dim] at row strides,
/// out [rows][heads][dim_v]. softcap > 0 applies cap * tanh(s / cap).
struct AttentionArgs {
  const float* q;
  std::size_t q_stride;
  const float* k;
  std::size_t k_stride;
  const float* v;
  std::size_t v_stride;
  float* out;
  std::size_t out_stride;
  std::size_t rows, heads, kv_heads, dim, dim_v;
  float scale;
  float softcap = 0.0f;
};
/// Rows whose key range is empty get zeros, or keep their output when
/// `skip_empty` (rows another path computes).
void Attention(const AttentionArgs& args, const KeyRange& keys,
               bool skip_empty = false);

/// bfp16ebs8 encoding of 8 values: [shared exponent byte][8 int8 mantissas],
/// value = mantissa * 2^(exponent - 127) / 64, rounded to nearest even.
void EncodeBfp16Block(const float* values, std::uint8_t* out);

/// x[i] = exp(x[i] - shift); *sum (when set) gets the total. Vectorized.
void ExpInPlace(float* x, std::size_t n, float shift, float* sum = nullptr);

/// Dot product and y += a x, vectorized.
[[nodiscard]] float Dot(const float* a, const float* b, std::size_t n);
void Axpy(float a, const float* x, float* y, std::size_t n);

}  // namespace gufo::npu::host

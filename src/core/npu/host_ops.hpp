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
/// The pool's thread count (the hardware's, at least 1).
[[nodiscard]] std::size_t Threads();

[[nodiscard]] float Bf16ToFloat(std::uint16_t h);
[[nodiscard]] std::uint16_t FloatToBf16(float f);  // round to nearest even
/// Rounds every value to the nearest bf16 in place.
void RoundBf16(float* x, std::size_t n);
void Bf16ToFloat(const std::uint16_t* src, float* dst, std::size_t n);

/// y = x / rms(x) * weight per row of `dim` (weight null: none; `offset`
/// added to the weight, e.g. 1 for (1 + w) norms). x and y may alias.
void RmsNormRows(const float* x, float* y, std::size_t rows, std::size_t dim,
                 const float* weight, float eps, float offset = 0.0f);
/// y = (x - mean) / sqrt(var + eps) * weight + bias per row (either null).
void LayerNormRows(const float* x, float* y, std::size_t rows, std::size_t dim,
                   const float* weight, const float* bias, float eps);

/// Rotary embedding, rotate-half (NeoX) pairing (i, i + rotary / 2) over the
/// first `rotary` dims of each head: x [rows][heads][head_dim], position of
/// row r = positions[r].
void RopeNeox(float* x, std::size_t rows, std::size_t heads,
              std::size_t head_dim, std::size_t rotary,
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
void Attention(const AttentionArgs& args, const KeyRange& keys);

/// Dot product and y += a x, vectorized.
[[nodiscard]] float Dot(const float* a, const float* b, std::size_t n);
void Axpy(float a, const float* x, float* y, std::size_t n);

}  // namespace gufo::npu::host

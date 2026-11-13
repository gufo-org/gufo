#ifndef GUFO_MODELS_QWEN3_TTS_NPU_CPU_CONV_HPP_
#define GUFO_MODELS_QWEN3_TTS_NPU_CPU_CONV_HPP_

// Causal 1-D convolutions on the CPU for the Qwen3-TTS NPU runtimes, with
// time-major activations [time][channels]. Weights are rearranged once at
// load so the innermost loop runs over output channels.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gufo/ext/experimental/npu.hpp"

namespace gufo::models::qwen3_tts::npu {

/// A causal convolution: out[t][o] = bias[o] + sum over taps j and inputs i
/// of x[t - (k - 1) dilation + j dilation][i] w[o][i][j], zeros before t = 0.
/// Transposed (stride s, causal tail dropped): out[t s + j][o] += x[t][i]
/// w[i][o][j] for t < length.
struct CpuConv {
  /// Out-of-range input rows: zeros (causal), mirrored, or the edge row.
  enum class Pad { kZero, kReflect, kReplicate };
  std::size_t in{}, out{}, kernel{}, dilation{1}, stride{1};
  bool transpose{false};
  // Non-causal convolutions (the reference encoders): output t reads input
  // t * step + j * dilation - left, padded by `pad`; Run's length is then the
  // output length.
  bool general{false};
  std::size_t step{1}, left{0};
  Pad pad{Pad::kZero};
  std::vector<float> taps;  // [tap][in][out]
  std::vector<float> bias;  // [out] (empty: none)
  // Identifies the weights (copies share it), for NPU forms cached by it.
  std::uint64_t serial{NextSerial()};
  static std::uint64_t NextSerial() {
    static std::atomic<std::uint64_t> next{1};
    return next.fetch_add(1);
  }

  /// From a PyTorch Conv1d weight [out][in][k].
  static CpuConv Make(const float* weight, const float* bias, std::size_t in,
                      std::size_t out, std::size_t kernel,
                      std::size_t dilation) {
    CpuConv c;
    c.in = in, c.out = out, c.kernel = kernel, c.dilation = dilation;
    c.taps.resize(kernel * in * out);
    for (std::size_t o = 0; o < out; ++o)
      for (std::size_t i = 0; i < in; ++i)
        for (std::size_t j = 0; j < kernel; ++j)
          c.taps[(j * in + i) * out + o] = weight[(o * in + i) * kernel + j];
    if (bias != nullptr)
      c.bias.assign(bias, bias + out);
    return c;
  }
  /// From a PyTorch ConvTranspose1d weight [in][out][k].
  static CpuConv MakeTranspose(const float* weight, const float* bias,
                               std::size_t in, std::size_t out,
                               std::size_t kernel, std::size_t stride) {
    CpuConv c;
    c.in = in, c.out = out, c.kernel = kernel, c.stride = stride;
    c.transpose = true;
    c.taps.resize(kernel * in * out);
    for (std::size_t i = 0; i < in; ++i)
      for (std::size_t o = 0; o < out; ++o)
        for (std::size_t j = 0; j < kernel; ++j)
          c.taps[(j * in + i) * out + o] = weight[(i * out + o) * kernel + j];
    if (bias != nullptr)
      c.bias.assign(bias, bias + out);
    return c;
  }

  /// A non-causal convolution with `left` padding rows and input step.
  static CpuConv MakeGeneral(const float* weight, const float* bias,
                             std::size_t in, std::size_t out,
                             std::size_t kernel, std::size_t dilation,
                             std::size_t step, std::size_t left, Pad pad) {
    CpuConv c = Make(weight, bias, in, out, kernel, dilation);
    c.general = true;
    c.step = step;
    c.left = left;
    c.pad = pad;
    return c;
  }

  /// General: y [out_length][out] from x [in_length][in].
  void RunGeneral(const float* x, std::size_t in_length, float* y,
                  std::size_t out_length) const {
    const auto source = [&](std::int64_t t) -> std::int64_t {
      const std::int64_t n = std::int64_t(in_length);
      if (t >= 0 && t < n)
        return t;
      if (pad == Pad::kZero)
        return -1;
      if (pad == Pad::kReplicate)
        return std::clamp<std::int64_t>(t, 0, n - 1);
      if (n <= 1)
        return 0;
      while (t < 0 || t >= n)
        t = t < 0 ? -t : 2 * n - t - 2;
      return t;
    };
    gufo::npu::host::ParallelFor(
        out_length, 4, [&](std::size_t b, std::size_t e) {
          for (std::size_t t = b; t < e; ++t) {
            float* a = y + t * out;
            if (bias.empty())
              std::fill_n(a, out, 0.0f);
            else
              std::copy(bias.begin(), bias.end(), a);
            for (std::size_t j = 0; j < kernel; ++j) {
              const std::int64_t s = source(
                  std::int64_t(t * step + j * dilation) - std::int64_t(left));
              if (s < 0)
                continue;
              const float* xr = x + std::size_t(s) * in;
              const float* wj = taps.data() + j * in * out;
              for (std::size_t i = 0; i < in; ++i) {
                const float v = xr[i];
                const float* w = wj + i * out;
                for (std::size_t o = 0; o < out; ++o)
                  a[o] += v * w[o];
              }
            }
          }
        });
  }

  /// Rows of history a streaming caller keeps in front of new input.
  [[nodiscard]] std::size_t Keep() const {
    return transpose ? (kernel - 1) / stride : (kernel - 1) * dilation;
  }

  /// x [length][in] -> y [length * stride][out].
  void Run(const float* x, std::size_t length, float* y) const {
    const std::size_t out_length = length * stride;
    constexpr std::size_t kBlock = 8;  // output rows sharing a weight row
    const std::size_t blocks = (out_length + kBlock - 1) / kBlock;
    gufo::npu::host::ParallelFor(
        blocks, std::max<std::size_t>(1, 16384 / (out * kernel * in / 64 + 1)),
        [&](std::size_t b, std::size_t e) {
          std::vector<float> acc(kBlock * out);
          for (std::size_t blk = b; blk < e; ++blk) {
            const std::size_t t0 = blk * kBlock;
            const std::size_t rows = std::min(kBlock, out_length - t0);
            for (std::size_t r = 0; r < rows; ++r)
              if (bias.empty())
                std::fill_n(acc.data() + r * out, out, 0.0f);
              else
                std::copy(bias.begin(), bias.end(), acc.data() + r * out);
            for (std::size_t j = 0; j < kernel; ++j) {
              const float* wj = taps.data() + j * in * out;
              for (std::size_t r = 0; r < rows; ++r) {
                const std::size_t t = t0 + r;
                std::int64_t source;
                if (transpose) {
                  const std::int64_t num = std::int64_t(t) - std::int64_t(j);
                  if (num < 0 || num % std::int64_t(stride) != 0)
                    continue;
                  source = num / std::int64_t(stride);
                  if (source >= std::int64_t(length))
                    continue;
                } else {
                  source = std::int64_t(t) -
                           std::int64_t((kernel - 1) * dilation) +
                           std::int64_t(j * dilation);
                  if (source < 0)
                    continue;
                }
                const float* xr = x + std::size_t(source) * in;
                float* a = acc.data() + r * out;
                for (std::size_t i = 0; i < in; ++i) {
                  const float v = xr[i];
                  const float* w = wj + i * out;
                  for (std::size_t o = 0; o < out; ++o)
                    a[o] += v * w[o];
                }
              }
            }
            std::copy_n(acc.data(), rows * out, y + t0 * out);
          }
        });
  }
};

/// Causal depthwise convolution, weight [channels][1][k].
inline void DepthwiseConv(const float* x, std::size_t length,
                          std::size_t channels, const float* weight,
                          const float* bias, std::size_t kernel, float* y) {
  gufo::npu::host::ParallelFor(length, 8, [&](std::size_t b, std::size_t e) {
    for (std::size_t t = b; t < e; ++t)
      for (std::size_t c = 0; c < channels; ++c) {
        float r = bias == nullptr ? 0.0f : bias[c];
        for (std::size_t j = 0; j < kernel; ++j) {
          const std::int64_t s =
              std::int64_t(t) - std::int64_t(kernel - 1) + std::int64_t(j);
          if (s >= 0)
            r += x[std::size_t(s) * channels + c] * weight[c * kernel + j];
        }
        y[t * channels + c] = r;
      }
  });
}

}  // namespace gufo::models::qwen3_tts::npu

#endif  // GUFO_MODELS_QWEN3_TTS_NPU_CPU_CONV_HPP_

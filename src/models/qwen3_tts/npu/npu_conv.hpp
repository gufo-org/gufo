#ifndef GUFO_MODELS_QWEN3_TTS_NPU_NPU_CONV_HPP_
#define GUFO_MODELS_QWEN3_TTS_NPU_NPU_CONV_HPP_

// Causal and transposed 1-D convolutions with their multiply-accumulates on
// the NPU (gufo::npu::BfpGemm), for the Qwen3-TTS NPU runtimes. Activations
// are time-major [time][channels], as CpuConv's.
//
// A causal convolution is one product of the rows' receptive fields (im2col
// [t][tap][in]) with the weights [out][tap][in]; a transposed one is one
// product of the input rows with [tap][out][in], whose taps are then added at
// t * stride + tap (overlap-add). The host only gathers rows and adds.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "gufo/ext/experimental/npu.hpp"
#include "src/models/qwen3_tts/npu/cpu_conv.hpp"

namespace gufo::models::qwen3_tts::npu {

struct NpuConv {
  std::size_t in{}, out{}, kernel{}, dilation{1}, stride{1};
  bool transpose{false};
  // CpuConv's general (non-causal) form: output t reads input t step + j d -
  // left, padded by `pad`.
  bool general{false};
  std::size_t step{1}, left{0};
  CpuConv::Pad pad{CpuConv::Pad::kZero};
  int id{-1};
  std::vector<float> bias;

  /// The NPU form of `c` (a causal or transposed CpuConv) in `gemm`'s arena.
  static NpuConv Make(gufo::npu::BfpGemm& gemm, const CpuConv& c) {
    NpuConv n;
    n.in = c.in, n.out = c.out, n.kernel = c.kernel, n.dilation = c.dilation;
    n.stride = c.stride, n.transpose = c.transpose, n.bias = c.bias;
    n.general = c.general, n.step = c.step, n.left = c.left, n.pad = c.pad;
    const std::size_t k = c.kernel, in = c.in, out = c.out;
    std::vector<float> w(k * in * out);
    // taps [tap][in][out] -> causal [out][tap][in], transposed [tap][out][in].
    for (std::size_t j = 0; j < k; ++j)
      for (std::size_t i = 0; i < in; ++i)
        for (std::size_t o = 0; o < out; ++o) {
          const float v = c.taps[(j * in + i) * out + o];
          if (c.transpose)
            w[(j * out + o) * in + i] = v;
          else
            w[(o * k + j) * in + i] = v;
        }
    n.id = c.transpose ? gemm.Upload(int(k * out), int(in), w)
                       : gemm.Upload(int(out), int(k * in), w);
    return n;
  }

  [[nodiscard]] std::size_t Keep() const {
    return transpose ? (kernel - 1) / stride : (kernel - 1) * dilation;
  }

  /// General form: y [out_length][out] from x [in_length][in].
  void RunGeneral(gufo::npu::BfpGemm& gemm, const float* x,
                  std::size_t in_length, float* y,
                  std::size_t out_length) const {
    namespace host = gufo::npu::host;
    const auto source = [&](std::int64_t t) -> std::int64_t {
      const std::int64_t n = std::int64_t(in_length);
      if (t >= 0 && t < n)
        return t;
      if (pad == CpuConv::Pad::kZero)
        return -1;
      if (pad == CpuConv::Pad::kReplicate)
        return std::clamp<std::int64_t>(t, 0, n - 1);
      if (n <= 1)
        return 0;
      while (t < 0 || t >= n)
        t = t < 0 ? -t : 2 * n - t - 2;
      return t;
    };
    std::vector<float> cols(out_length * kernel * in);
    host::ParallelFor(out_length, 16, [&](std::size_t b, std::size_t e) {
      for (std::size_t t = b; t < e; ++t)
        for (std::size_t j = 0; j < kernel; ++j) {
          const std::int64_t s = source(std::int64_t(t * step + j * dilation) -
                                        std::int64_t(left));
          float* dst = cols.data() + (t * kernel + j) * in;
          if (s < 0)
            std::fill_n(dst, in, 0.0f);
          else
            std::copy_n(x + std::size_t(s) * in, in, dst);
        }
    });
    gemm.Run(id, cols.data(), int(out_length), y);
    if (!bias.empty())
      host::ParallelFor(out_length, 64, [&](std::size_t b, std::size_t e) {
        for (std::size_t t = b; t < e; ++t)
          for (std::size_t o = 0; o < out; ++o)
            y[t * out + o] += bias[o];
      });
  }

  /// x [length][in] -> y [length * stride][out].
  void Run(gufo::npu::BfpGemm& gemm, const float* x, std::size_t length,
           float* y) const {
    namespace host = gufo::npu::host;
    if (!transpose) {
      // Receptive fields: row t holds x[t - (k - 1) d + j d] for taps j.
      std::vector<float> cols(length * kernel * in);
      host::ParallelFor(length, 16, [&](std::size_t b, std::size_t e) {
        for (std::size_t t = b; t < e; ++t)
          for (std::size_t j = 0; j < kernel; ++j) {
            const std::int64_t s = std::int64_t(t) -
                                   std::int64_t((kernel - 1) * dilation) +
                                   std::int64_t(j * dilation);
            float* dst = cols.data() + (t * kernel + j) * in;
            if (s < 0)
              std::fill_n(dst, in, 0.0f);
            else
              std::copy_n(x + std::size_t(s) * in, in, dst);
          }
      });
      gemm.Run(id, cols.data(), int(length), y);
      if (!bias.empty())
        host::ParallelFor(length, 64, [&](std::size_t b, std::size_t e) {
          for (std::size_t t = b; t < e; ++t)
            for (std::size_t o = 0; o < out; ++o)
              y[t * out + o] += bias[o];
        });
      return;
    }
    // Transposed: z[t][j][o] = x[t] . w[j][o]; y[t s + j] += z[t][j].
    std::vector<float> z(length * kernel * out);
    gemm.Run(id, x, int(length), z.data());
    const std::size_t rows = length * stride;
    host::ParallelFor(rows, 64, [&](std::size_t b, std::size_t e) {
      for (std::size_t r = b; r < e; ++r) {
        float* yr = y + r * out;
        if (bias.empty())
          std::fill_n(yr, out, 0.0f);
        else
          std::copy(bias.begin(), bias.end(), yr);
        // Taps j with r = t s + j, t < length.
        for (std::size_t j = r % stride; j < kernel; j += stride) {
          if (r < j)
            break;
          const std::size_t t = (r - j) / stride;
          if (t >= length)
            continue;
          const float* zr = z.data() + (t * kernel + j) * out;
          for (std::size_t o = 0; o < out; ++o)
            yr[o] += zr[o];
        }
      }
    });
  }
};

/// The reference-recording encoders' convolutions (run once per voice) on one
/// NPU program opened on first use, with each CpuConv's NPU form cached.
class EncoderNpu {
public:
  static EncoderNpu& Get() {
    static EncoderNpu instance;
    return instance;
  }

  /// c.RunGeneral on the NPU.
  void RunGeneral(const CpuConv& c, const float* x, std::size_t in_length,
                  float* y, std::size_t out_length) {
    const NpuConv* n;
    {
      std::lock_guard lock(mutex_);
      if (!gemm_) {
        device_ = gufo::npu::Device::Open();
        gemm_ = gufo::npu::BfpGemm::Open(
            device_, gufo::npu::FindDesign("bfpgemm"), std::size_t(256) << 20);
      }
      auto it = convs_.find(c.serial);
      if (it == convs_.end())
        it = convs_.emplace(c.serial, NpuConv::Make(gemm_, c)).first;
      n = &it->second;
    }
    n->RunGeneral(gemm_, x, in_length, y, out_length);
  }

private:
  std::mutex mutex_;
  gufo::npu::Device device_;
  gufo::npu::BfpGemm gemm_;
  std::unordered_map<std::uint64_t, NpuConv> convs_;  // by CpuConv::serial
};

}  // namespace gufo::models::qwen3_tts::npu

#endif  // GUFO_MODELS_QWEN3_TTS_NPU_NPU_CONV_HPP_

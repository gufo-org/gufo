#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <stdexcept>

#include "src/models/moss_image/hip/ops.hpp"
#include "src/models/qwen_image_21/hip/bf16.hpp"

namespace gufo::models::moss_image::hip {
namespace {

using B = hip_bfloat16;
using qwen_image_21::hip::Bf16;
using qwen_image_21::hip::Check;

__device__ float Round(float x) {
  return static_cast<float>(Bf16(x));
}

B* Ptr(const Matrix& m) {
  return reinterpret_cast<B*>(m.data);
}

unsigned Blocks(std::size_t n) {
  return static_cast<unsigned>((n + 255) / 256);
}

__global__ void ShiftScaleKernel(const B* x, const B* mod, B* y,
                                 std::size_t n, int cols, int shift_part,
                                 int scale_part) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const int c = static_cast<int>(i % cols);
  const float scale = Round(1.0F + float(mod[scale_part * cols + c]));
  const float shift = float(mod[shift_part * cols + c]);
  y[i] = Bf16(Round(float(x[i]) * scale) + shift);
}

__global__ void GatedResidualKernel(const B* residual, const B* x,
                                    const B* mod, B* y, std::size_t n,
                                    int cols, int gate_part) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const float gate = float(mod[gate_part * cols + static_cast<int>(i % cols)]);
  y[i] = Bf16(float(residual[i]) + Round(float(x[i]) * gate));
}

__global__ void SituKernel(const B* hidden, int hidden_cols, int hidden_offset,
                           const B* gate, int gate_cols, int gate_offset, B* y,
                           int rows, int width) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * width)
    return;
  const auto row = i / width;
  const int c = static_cast<int>(i % width);
  const float g = float(gate[row * gate_cols + gate_offset + c]);
  const float h = float(hidden[row * hidden_cols + hidden_offset + c]);
  // Each PyTorch BF16 operation rounds its result.
  const float left =
      Round(Round(4.0F * Round(tanhf(Round(g / 4.0F)))) *
            Round(1.0F / (1.0F + expf(-g))));
  const float right = Round(25.0F * Round(tanhf(Round(h / 25.0F))));
  y[i] = Bf16(left * right);
}

constexpr int kGroupChunks = 64;

// Partial sums of one (group, chunk); FP64 combines the FP32 thread sums.
__global__ void GroupStatsKernel(const B* x, double* partial, int rows,
                                 int cols, int group_channels) {
  __shared__ double sums[256], squares[256];
  const int group = blockIdx.x, chunk = blockIdx.y;
  const int rows_per_chunk = (rows + kGroupChunks - 1) / kGroupChunks;
  const int row_begin = chunk * rows_per_chunk;
  const int row_end = min(rows, row_begin + rows_per_chunk);
  const auto count =
      static_cast<std::size_t>(max(row_end - row_begin, 0)) * group_channels;
  float sum = 0, square = 0;
  for (std::size_t e = threadIdx.x; e < count; e += blockDim.x) {
    const auto row = row_begin + e / group_channels;
    const auto c = group * group_channels + e % group_channels;
    const float v = float(x[row * cols + c]);
    sum += v;
    square += v * v;
  }
  sums[threadIdx.x] = sum;
  squares[threadIdx.x] = square;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
      squares[threadIdx.x] += squares[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial[(group * kGroupChunks + chunk) * 2] = sums[0];
    partial[(group * kGroupChunks + chunk) * 2 + 1] = squares[0];
  }
}

__global__ void GroupApplyKernel(const B* x, const double* partial,
                                 const B* weight, const B* bias, B* y,
                                 int rows, int cols, int group_channels,
                                 float epsilon, bool silu) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * cols)
    return;
  const int c = static_cast<int>(i % cols);
  const int group = c / group_channels;
  double sum = 0, square = 0;
  for (int k = 0; k < kGroupChunks; ++k) {
    sum += partial[(group * kGroupChunks + k) * 2];
    square += partial[(group * kGroupChunks + k) * 2 + 1];
  }
  const double count = static_cast<double>(rows) * group_channels;
  const double mean = sum / count;
  const double variance = fmax(square / count - mean * mean, 0.0);
  const float inverse = static_cast<float>(1.0 / sqrt(variance + epsilon));
  float v = (float(x[i]) - static_cast<float>(mean)) * inverse;
  v = v * float(weight[c]) + float(bias[c]);
  if (silu)
    v = v / (1.0F + expf(-v));
  y[i] = Bf16(v);
}

}  // namespace

Matrix ShiftScale(Runtime& rt, const Matrix& x, const Matrix& mod,
                  int shift_part, int scale_part) {
  if (x.packed || mod.rows != 1 || mod.cols % x.cols ||
      std::max(shift_part, scale_part) >= mod.cols / x.cols)
    throw std::logic_error("MOSS-Image modulation shape mismatch");
  auto out = rt.New(x.rows, x.cols);
  ShiftScaleKernel<<<Blocks(x.size()), 256, 0, rt.stream()>>>(
      Ptr(x), Ptr(mod), Ptr(out), x.size(), x.cols, shift_part, scale_part);
  Check(hipGetLastError());
  return out;
}

Matrix GatedResidual(Runtime& rt, const Matrix& residual, const Matrix& y,
                     const Matrix& mod, int gate_part) {
  if (residual.rows != y.rows || residual.cols != y.cols || y.packed ||
      mod.rows != 1 || mod.cols % y.cols || gate_part >= mod.cols / y.cols)
    throw std::logic_error("MOSS-Image gated residual shape mismatch");
  auto out = rt.New(y.rows, y.cols);
  GatedResidualKernel<<<Blocks(y.size()), 256, 0, rt.stream()>>>(
      Ptr(residual), Ptr(y), Ptr(mod), Ptr(out), y.size(), y.cols, gate_part);
  Check(hipGetLastError());
  return out;
}

Matrix Situ(Runtime& rt, const Matrix& hidden, int hidden_offset,
            const Matrix& gate, int gate_offset, int width) {
  if (hidden.packed || gate.packed || hidden.rows != gate.rows ||
      hidden_offset + width > hidden.cols || gate_offset + width > gate.cols)
    throw std::logic_error("MOSS-Image situ shape mismatch");
  auto out = rt.New(hidden.rows, width);
  SituKernel<<<Blocks(out.size()), 256, 0, rt.stream()>>>(
      Ptr(hidden), hidden.cols, hidden_offset, Ptr(gate), gate.cols,
      gate_offset, Ptr(out), hidden.rows, width);
  Check(hipGetLastError());
  return out;
}

Matrix GroupNorm(Runtime& rt, const Matrix& x, const Matrix& weight,
                 const Matrix& bias, int groups, float epsilon, bool silu) {
  if (x.packed || x.cols % groups ||
      weight.size() != static_cast<std::size_t>(x.cols) ||
      bias.size() != static_cast<std::size_t>(x.cols))
    throw std::logic_error("MOSS-Image group norm shape mismatch");
  // Two doubles per (group, chunk), stored in a BF16-typed scratch matrix.
  auto partial = rt.New(groups * kGroupChunks * 2, 4);
  auto* stats = reinterpret_cast<double*>(partial.data);
  const int group_channels = x.cols / groups;
  GroupStatsKernel<<<dim3(groups, kGroupChunks), 256, 0, rt.stream()>>>(
      Ptr(x), stats, x.rows, x.cols, group_channels);
  auto out = rt.New(x.rows, x.cols);
  GroupApplyKernel<<<Blocks(x.size()), 256, 0, rt.stream()>>>(
      Ptr(x), stats, Ptr(weight), Ptr(bias), Ptr(out), x.rows, x.cols,
      group_channels, epsilon, silu);
  Check(hipGetLastError());
  return out;
}

}  // namespace gufo::models::moss_image::hip

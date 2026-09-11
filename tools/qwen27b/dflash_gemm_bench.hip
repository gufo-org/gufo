// DFlash2 exact small-batch GEMM ablations, using the production templates.
// Build inside Nix with tools/bench/build.sh and this source path.
// Run: /tmp/dflash_gemm_bench bf16|q8|q4|q5|iq4|q6 M K [iterations] [batch]
// Defaults: 24 iterations, batch 7 for Q6 and 8 otherwise.
// BF16 also accepts batch=16 to compare injection with two batch-8 launches.
// Rotating matrices keep small shapes from fitting entirely in cache. Each
// geometry must be bit-exact with the original geometry before timing counts.
#define ENGINE_ENABLE_HIP 1
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"

namespace {

void Check(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}

template<typename T>
class Buffer {
public:
  explicit Buffer(std::size_t count) {
    Check(hipMalloc(&data_, count * sizeof(T)));
  }
  ~Buffer() { (void)hipFree(data_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  T* Get() const { return data_; }

private:
  T* data_{};
};

class Event {
public:
  Event() { Check(hipEventCreate(&event_)); }
  ~Event() { (void)hipEventDestroy(event_); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  hipEvent_t Get() const { return event_; }

private:
  hipEvent_t event_{};
};

__device__ unsigned Hash(unsigned value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

__global__ void InitBytes(unsigned char* data, std::size_t size) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    data[index] = Hash(index);
}

__global__ void InitScales(unsigned char* data, std::size_t blocks,
                           std::size_t stride, std::size_t offset,
                           bool affine) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= blocks)
    return;
  auto* scale = reinterpret_cast<__half*>(data + index * stride + offset);
  scale[0] = __float2half(0.007F + (Hash(index) % 1000) * 0.00001F);
  if (affine)
    scale[1] = __float2half(0.004F + (Hash(index + 19) % 1000) * 0.00001F);
}

template<typename T>
__global__ void InitValues(T* data, std::size_t size) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    data[index] = T((static_cast<int>(Hash(index) % 20001) - 10000) * 0.00027F);
}

__global__ void Compare(const float* expected, const float* actual,
                        std::size_t count, unsigned* mismatches) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count &&
      (!isfinite(actual[index]) ||
       __float_as_uint(expected[index]) != __float_as_uint(actual[index]))) {
    atomicAdd(mismatches, 1U);
  }
}

struct Geometry {
  const char* name;
  std::function<void(const void*)> launch;
};

template<std::size_t Batch>
void Run(const std::string& type, std::size_t rows, std::size_t columns,
         int iterations) {
  using namespace gufo;
  using namespace gufo::hip;
  if (rows == 0 || columns == 0 || columns % 256 != 0 || iterations < 1 ||
      (type != "bf16" && type != "q8" && type != "q4" && type != "q5" &&
       type != "iq4" && type != "q6")) {
    throw std::invalid_argument(
        "require bf16|q8|q4|q5|iq4|q6, positive M, K divisible by 256, "
        "positive "
        "iterations");
  }
  if constexpr (Batch > 8) {
    if (type != "bf16")
      throw std::invalid_argument("wide injection is BF16 only");
  }
  constexpr std::size_t batch = Batch;
  const std::size_t block_bytes = type == "q8"    ? 34
                                  : type == "q4"  ? 144
                                  : type == "q5"  ? 176
                                  : type == "iq4" ? 136
                                                  : 210;
  const std::size_t bytes =
      type == "bf16" ? rows * columns * 2
                     : rows * columns / (type == "q8" ? 32 : 256) * block_bytes;
  const std::size_t copies =
      std::max<std::size_t>(1, (128ULL * 1024 * 1024 + bytes - 1) / bytes);
  const std::size_t physical = bytes * copies;
  Buffer<unsigned char> weights(physical);
  Buffer<float> input(batch * columns), output(batch * rows),
      reference(batch * rows);
  Buffer<unsigned> mismatch(1);
  if (type == "bf16") {
    InitValues<<<(physical / 2 + 255) / 256, 256>>>(
        reinterpret_cast<hip_bfloat16*>(weights.Get()), physical / 2);
  } else {
    InitBytes<<<(physical + 255) / 256, 256>>>(weights.Get(), physical);
    InitScales<<<(physical / block_bytes + 255) / 256, 256>>>(
        weights.Get(), physical / block_bytes, block_bytes,
        type == "q6" ? 208 : 0, type == "q4" || type == "q5");
  }
  InitValues<<<(batch * columns + 255) / 256, 256>>>(input.Get(),
                                                     batch * columns);
  std::vector<Geometry> geometries;
  if (type == "bf16") {
    if constexpr (Batch == 16) {
      geometries = {{"batch8_chunks",
                     [&](const void* w) {
                       for (std::size_t offset = 0; offset < Batch; offset += 8)
                         BatchedExactBf16GEMMFp32VecKernel<16, 8, 2>
                             <<<(rows + 31) / 32, 512>>>(
                                 static_cast<const hip_bfloat16*>(w),
                                 input.Get() + offset * columns,
                                 output.Get() + offset * rows, rows, columns);
                     }},
                    {"batch16", [&](const void* w) {
                       BatchedExactBf16GEMMFp32VecKernel<16, 16, 2>
                           <<<(rows + 31) / 32, 512>>>(
                               static_cast<const hip_bfloat16*>(w), input.Get(),
                               output.Get(), rows, columns);
                     }}};
    } else {
      geometries = {{"rows2",
                     [&](const void* w) {
                       BatchedExactBf16GEMMFp32VecKernel<16, Batch, 2>
                           <<<(rows + 31) / 32, 512>>>(
                               static_cast<const hip_bfloat16*>(w), input.Get(),
                               output.Get(), rows, columns);
                     }},
                    {"rows1", [&](const void* w) {
                       BatchedExactBf16GEMMFp32VecKernel<16, Batch, 1>
                           <<<(rows + 15) / 16, 512>>>(
                               static_cast<const hip_bfloat16*>(w), input.Get(),
                               output.Get(), rows, columns);
                     }}};
    }
  } else if constexpr (Batch <= 8) {
    if (type == "q8") {
      geometries = {{"rows2",
                     [&](const void* w) {
                       SmallBatchQ8_0ExactFp32VecGEMMKernel<16, Batch, 2>
                           <<<(rows + 31) / 32, 512>>>(
                               w, input.Get(), output.Get(), rows, columns);
                     }},
                    {"rows3", [&](const void* w) {
                       SmallBatchQ8_0ExactFp32VecGEMMKernel<16, Batch, 3>
                           <<<(rows + 47) / 48, 512>>>(
                               w, input.Get(), output.Get(), rows, columns);
                     }}};
    } else if (type == "q4" || type == "q5" || type == "iq4") {
      const auto kquant = [&]<core::GgmlType Type>() {
        geometries = {
            {"tile1",
             [&](const void* w) {
               SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 3, Type>
                   <<<(rows + 47) / 48, 512>>>(w, input.Get(), output.Get(),
                                               rows, columns);
             }},
            {"tile2", [&](const void* w) {
               SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 3, Type, 2>
                   <<<(rows + 47) / 48, 512>>>(w, input.Get(), output.Get(),
                                               rows, columns);
             }}};
      };
      if (type == "q4")
        kquant.template operator()<core::GgmlType::kQ4_K>();
      else if (type == "q5")
        kquant.template operator()<core::GgmlType::kQ5_K>();
      else
        kquant.template operator()<core::GgmlType::kIQ4_XS>();
    } else {
      geometries = {
          {"rows2_tile1",
           [&](const void* w) {
             SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 2,
                                                 core::GgmlType::kQ6_K>
                 <<<(rows + 31) / 32, 512>>>(w, input.Get(), output.Get(), rows,
                                             columns);
           }},
          {"rows3_tile2", [&](const void* w) {
             SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 3,
                                                 core::GgmlType::kQ6_K, 2, 1>
                 <<<(rows + 47) / 48, 512>>>(w, input.Get(), output.Get(), rows,
                                             columns);
           }}};
    }
  }
  geometries.front().launch(weights.Get());
  Check(hipMemcpy(reference.Get(), output.Get(), batch * rows * sizeof(float),
                  hipMemcpyDeviceToDevice));
  Event start, stop;
  std::puts("type,m,k,batch,round,geometry,ms,GBps,mismatches");
  for (int round = 0; round < 4; ++round) {
    for (std::size_t index = 0; index < geometries.size(); ++index) {
      const auto& geometry =
          geometries[round % 2 == 0 ? index : geometries.size() - index - 1];
      Check(hipMemset(mismatch.Get(), 0, sizeof(unsigned)));
      geometry.launch(weights.Get());
      Check(hipGetLastError());
      Compare<<<(batch * rows + 255) / 256, 256>>>(
          reference.Get(), output.Get(), batch * rows, mismatch.Get());
      unsigned bad = 0;
      Check(
          hipMemcpy(&bad, mismatch.Get(), sizeof(bad), hipMemcpyDeviceToHost));
      if (bad)
        throw std::runtime_error("geometry changed " + std::to_string(bad) +
                                 " outputs");
      Check(hipEventRecord(start.Get()));
      for (int iteration = 0; iteration < iterations; ++iteration) {
        geometry.launch(weights.Get() + (iteration % copies) * bytes);
      }
      Check(hipEventRecord(stop.Get()));
      Check(hipEventSynchronize(stop.Get()));
      float milliseconds = 0;
      Check(hipEventElapsedTime(&milliseconds, start.Get(), stop.Get()));
      milliseconds /= iterations;
      std::printf("%s,%zu,%zu,%zu,%d,%s,%.6f,%.2f,%u\n", type.c_str(), rows,
                  columns, batch, round, geometry.name, milliseconds,
                  bytes / (milliseconds * 1e6), bad);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 6)
      throw std::invalid_argument(
          "usage: dflash_gemm_bench bf16|q8|q4|q5|iq4|q6 M K [iterations] "
          "[batch=7|8|16 (16 is BF16 injection only)]");
    const int batch = argc == 6                      ? std::stoi(argv[5])
                      : std::string(argv[1]) == "q6" ? 7
                                                     : 8;
    const int iterations = argc >= 5 ? std::stoi(argv[4]) : 24;
    if (batch == 7) {
      Run<7>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 8) {
      Run<8>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 16) {
      Run<16>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else {
      throw std::invalid_argument(
          "batch must be seven proposals, eight verification rows or sixteen "
          "injection rows");
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}

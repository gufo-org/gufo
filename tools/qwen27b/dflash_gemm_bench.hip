// DFlash2 exact small-batch GEMM ablations, using the production templates.
// Build inside Nix with tools/bench/build.sh and this source path.
// Run: /tmp/dflash_gemm_bench bf16|q8|q4|q5|iq4|q6 M K [iterations] [batch]
// Scalar Q4/Q5/IQ4/Q6 use batch=1; append swiglu [up-quant] for gate/up.
// Defaults: 24 iterations, batch 7 for Q6 and 8 otherwise; widths 2–8 supported.
// BF16 also accepts batch=16 to compare injection with two batch-8 launches.
// Weights use read-only host registration, matching production GGUF mapping.
// Decode rotates matrices to exceed cache capacity. Batch-16 context injection
// repeats one matrix across chunks, including the production K/V cache hint.
// Every geometry must be bit-exact with the reference before timing counts.
#define ENGINE_ENABLE_HIP 1
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <vector>

#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"
// Embed the scalar implementation so ablations use its actual templates and
// arithmetic without linking the complete model runtime.
#include "src/models/qwen/hip/kernels/gemv_quant.hip"
#include "src/models/qwen/hip/kernels/swiglu.hip"

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

class MappedWeights {
public:
  MappedWeights(const void* source, std::size_t bytes) : bytes_(bytes) {
    host_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (host_ == MAP_FAILED)
      throw std::runtime_error("weight mmap failed");
    try {
      Check(hipMemcpy(host_, source, bytes_, hipMemcpyDeviceToHost));
      Check(hipHostRegister(host_, bytes_,
                            hipHostRegisterMapped | hipHostRegisterReadOnly));
      registered_ = true;
      Check(hipHostGetDevicePointer(&device_, host_, 0));
    } catch (...) {
      Release();
      throw;
    }
  }
  ~MappedWeights() { Release(); }
  MappedWeights(const MappedWeights&) = delete;
  MappedWeights& operator=(const MappedWeights&) = delete;
  const unsigned char* Get() const {
    return static_cast<const unsigned char*>(device_);
  }

private:
  void Release() {
    if (registered_)
      (void)hipHostUnregister(host_);
    (void)munmap(host_, bytes_);
  }
  void* host_{};
  void* device_{};
  std::size_t bytes_{};
  bool registered_{};
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
                           bool affine, std::size_t blocks_per_copy = 0,
                           std::size_t copy_bytes = 0) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= blocks)
    return;
  const auto copy = blocks_per_copy ? index / blocks_per_copy : 0;
  const auto block = blocks_per_copy ? index % blocks_per_copy : index;
  auto* scale = reinterpret_cast<__half*>(
      data + copy * copy_bytes + block * stride + offset);
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
         int iterations, bool fused = false,
         const std::string& up_type = "") {
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
  if constexpr (Batch == 1) {
    if (type == "bf16" || type == "q8")
      throw std::invalid_argument("scalar ablations require q4|q5|iq4|q6");
  }
  const bool mixed = !up_type.empty() && up_type != type;
  if (mixed &&
      (!fused || Batch != 1 ||
       !((type == "q4" && (up_type == "q5" || up_type == "iq4")) ||
         (type == "iq4" && (up_type == "q4" || up_type == "q5"))))) {
    throw std::invalid_argument(
        "mixed swiglu supports q4/q5, q4/iq4, iq4/q4 and iq4/q5");
  }
  constexpr std::size_t batch = Batch;
  const auto block_size = [](const std::string& quant) -> std::size_t {
    return quant == "q8"    ? 34
           : quant == "q4"  ? 144
           : quant == "q5"  ? 176
           : quant == "iq4" ? 136
                            : 210;
  };
  const std::size_t block_bytes = block_size(type);
  const std::size_t matrix_bytes =
      type == "bf16" ? rows * columns * 2
                     : rows * columns / (type == "q8" ? 32 : 256) * block_bytes;
  const std::size_t up_block_bytes = mixed ? block_size(up_type) : block_bytes;
  const std::size_t up_matrix_bytes =
      mixed ? rows * columns / 256 * up_block_bytes : matrix_bytes;
  const std::size_t bytes = matrix_bytes + (fused ? up_matrix_bytes : 0);
  if (Batch != 1 && type != "bf16" && type != "q8" &&
      !FitsSmallBatch32BitIndices(batch, rows, columns, matrix_bytes / rows)) {
    throw std::invalid_argument("K-quant ablations require 32-bit offsets");
  }
  const std::size_t copies =
      Batch == 16 ? 1
                  : std::max<std::size_t>(
                        1, (128ULL * 1024 * 1024 + bytes - 1) / bytes);
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
    if (mixed) {
      const auto gate_blocks = matrix_bytes / block_bytes;
      const auto up_blocks = up_matrix_bytes / up_block_bytes;
      InitScales<<<(copies * gate_blocks + 255) / 256, 256>>>(
          weights.Get(), copies * gate_blocks, block_bytes, 0,
          type == "q4" || type == "q5", gate_blocks, bytes);
      InitScales<<<(copies * up_blocks + 255) / 256, 256>>>(
          weights.Get() + matrix_bytes, copies * up_blocks, up_block_bytes,
          0, up_type == "q4" || up_type == "q5", up_blocks, bytes);
    } else {
      InitScales<<<(physical / block_bytes + 255) / 256, 256>>>(
          weights.Get(), physical / block_bytes, block_bytes,
          type == "q6" ? 208 : 0, type == "q4" || type == "q5");
    }
  }
  InitValues<<<(batch * columns + 255) / 256, 256>>>(input.Get(),
                                                     batch * columns);
  std::vector<Geometry> geometries;
  if constexpr (Batch == 1) {
    const auto scalar =
        [&]<core::GgmlType Type, core::GgmlType UpType = Type>() {
      const auto launch =
          [&]<core::GgmlType StaticType,
              core::GgmlType StaticUpType = StaticType>(const void* w) {
        if (fused) {
          Wave32FusedQuantSwiGLUGEMVKernel_2Rows<4, StaticType, StaticUpType>
              <<<(rows + 7) / 8, 128>>>(
                  w, Type, static_cast<const unsigned char*>(w) + matrix_bytes,
                  UpType, input.Get(), output.Get(), rows, columns);
        } else {
          Q8KBlockGEMVKernel_2Rows<4, StaticType><<<(rows + 7) / 8, 128>>>(
              w, Type, input.Get(), output.Get(), rows, columns);
        }
      };
      geometries = {
          {"generic", [&, launch](const void* w) {
             launch.template operator()<core::GgmlType::kF32>(w);
           }},
          {"typed", [&, launch](const void* w) {
             launch.template operator()<Type, UpType>(w);
           }}};
    };
    if (type == "q4") {
      if (up_type == "q5")
        scalar.template operator()<core::GgmlType::kQ4_K,
                                   core::GgmlType::kQ5_K>();
      else if (up_type == "iq4")
        scalar.template operator()<core::GgmlType::kQ4_K,
                                   core::GgmlType::kIQ4_XS>();
      else
        scalar.template operator()<core::GgmlType::kQ4_K>();
    } else if (type == "q5") {
      scalar.template operator()<core::GgmlType::kQ5_K>();
    } else if (type == "iq4") {
      if (up_type == "q4")
        scalar.template operator()<core::GgmlType::kIQ4_XS,
                                   core::GgmlType::kQ4_K>();
      else if (up_type == "q5")
        scalar.template operator()<core::GgmlType::kIQ4_XS,
                                   core::GgmlType::kQ5_K>();
      else
        scalar.template operator()<core::GgmlType::kIQ4_XS>();
    } else {
      scalar.template operator()<core::GgmlType::kQ6_K>();
    }
  } else if (type == "bf16") {
    if constexpr (Batch == 16) {
      geometries = {
          {"batch8_chunks",
           [&](const void* w) {
             for (std::size_t offset = 0; offset < Batch; offset += 8)
               BatchedExactBf16GEMMFp32VecKernel<16, 8, 2>
                   <<<(rows + 31) / 32, 512>>>(
                       static_cast<const hip_bfloat16*>(w),
                       input.Get() + offset * columns,
                       output.Get() + offset * rows, rows, columns);
           }},
          {"batch16", [&](const void* w) {
             const auto launch = [&]<bool StreamWeights>() {
               BatchedExactBf16GEMMFp32VecKernel<16, 16, 2, StreamWeights>
                   <<<(rows + 31) / 32, 512>>>(
                       static_cast<const hip_bfloat16*>(w), input.Get(),
                       output.Get(), rows, columns);
             };
             if (rows == 1024 && columns == 5120)
               launch.template operator()<false>();
             else
               launch.template operator()<true>();
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
                    {"token_rows",
                     [&](const void* w) {
                       SmallBatchQ8_0ExactFp32VecGEMMKernel<8, 1, 1>
                           <<<dim3((rows + 7) / 8, Batch), 256>>>(
                               w, input.Get(), output.Get(), rows, columns);
                     }},
                    {"rows3", [&](const void* w) {
                       SmallBatchQ8_0ExactFp32VecGEMMKernel<16, Batch, 3>
                           <<<(rows + 47) / 48, 512>>>(
                               w, input.Get(), output.Get(), rows, columns);
                     }}};
      const auto waves = [&]<std::uint32_t Waves, std::size_t Rows>(
                             const char* name) {
        geometries.push_back({name, [&](const void* w) {
                                SmallBatchQ8_0ExactFp32VecGEMMKernel<Waves,
                                                                    Batch, Rows>
                                    <<<(rows + Waves * Rows - 1) / (Waves * Rows),
                                       Waves * 32>>>(w, input.Get(),
                                                     output.Get(), rows,
                                                     columns);
                              }});
      };
      waves.template operator()<32, 2>("waves32_rows2");
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
        const auto narrow = [&]<std::size_t Tiles>(const char* name) {
          geometries.push_back({name, [&](const void* w) {
            SmallBatchKQuantExactFp32GEMMKernel<
                16, Batch, 3, Type, Tiles, 12, 1, true>
                <<<(rows + 47) / 48, 512>>>(w, input.Get(), output.Get(),
                                            rows, columns);
          }});
        };
        narrow.template operator()<1>("narrow_tile1");
        narrow.template operator()<2>("narrow_tile2");
        if constexpr ((Type == core::GgmlType::kQ4_K && Batch >= 2) ||
                      (Type == core::GgmlType::kQ5_K && Batch >= 2 &&
                       Batch <= 6) ||
                      (Type == core::GgmlType::kIQ4_XS &&
                       (Batch == 2 || Batch == 3 || Batch == 4 || Batch == 5 ||
                        Batch == 7))) {
          const auto rows4 =
              [&]<std::size_t Tiles, std::size_t TokensPerStep = Batch>(
                  const char* name) {
                geometries.push_back({name, [&](const void* w) {
                  SmallBatchKQuantExactFp32GEMMKernel<
                      16, Batch, 4, Type, Tiles, 4, 1, true, TokensPerStep>
                      <<<(rows + 63) / 64, 512>>>(
                          w, input.Get(), output.Get(), rows, columns);
                }});
              };
          if constexpr (Type == core::GgmlType::kQ5_K && Batch == 4) {
            rows4.template operator()<1>("rows4_tile1");
            rows4.template operator()<1, 2>("rows4_tile1_step2");
          } else {
            rows4.template operator()<2>("rows4_tile2");
            if constexpr ((Type == core::GgmlType::kQ5_K && Batch == 5) ||
                          (Type == core::GgmlType::kIQ4_XS && Batch == 3))
              rows4.template operator()<1>("rows4_tile1");
          }
        }
        if constexpr ((Batch >= 7 && Type == core::GgmlType::kQ5_K) ||
                      (Batch == 8 && Type == core::GgmlType::kIQ4_XS)) {
          geometries.push_back(
              {"narrow_rows4", [&](const void* w) {
                 SmallBatchKQuantExactFp32GEMMKernel<8, Batch, 4, Type, 1, 4, 1,
                                                     true>
                     <<<(rows + 31) / 32, 256>>>(
                         w, input.Get(), output.Get(), rows, columns);
               }});
        }
        if constexpr (Batch == 8 && Type == core::GgmlType::kQ5_K) {
          geometries.push_back(
              {"waves16_rows4", [&](const void* w) {
                 SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 4, Type, 1, 4,
                                                     1, true>
                     <<<(rows + 63) / 64, 512>>>(
                         w, input.Get(), output.Get(), rows, columns);
               }});
          const auto token_major = [&]<std::uint32_t Waves>(const char* name) {
            geometries.push_back(
                {name, [&](const void* w) {
                   SmallBatchKQuantExactFp32GEMMKernel<Waves, Batch, 4, Type,
                                                       1, 4, 1, true, 1>
                       <<<(rows + Waves * 4 - 1) / (Waves * 4), Waves * 32>>>(
                           w, input.Get(), output.Get(), rows, columns);
                 }});
          };
          token_major.template operator()<8>("token_major_rows4");
          token_major.template operator()<16>("token_major_waves16_rows4");
        }
        if constexpr (Batch == 8 && Type == core::GgmlType::kQ4_K) {
          geometries.push_back(
              {"groups4", [&](const void* w) {
                 SmallBatchKQuantExactFp32GEMMKernel<16, 4, 3, Type, 1, 12, 2>
                     <<<((rows + 47) / 48) * 2, 512>>>(
                         w, input.Get(), output.Get(), rows, columns);
               }});
          geometries.push_back(
              {"narrow_groups4", [&](const void* w) {
                 SmallBatchKQuantExactFp32GEMMKernel<16, 4, 3, Type, 1, 12, 2,
                                                     true>
                     <<<((rows + 47) / 48) * 2, 512>>>(
                         w, input.Get(), output.Get(), rows, columns);
               }});
        }
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
           }},
          {"narrow_rows2_tile1",
           [&](const void* w) {
             SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 2,
                                                 core::GgmlType::kQ6_K, 1, 12,
                                                 1, true>
                 <<<(rows + 31) / 32, 512>>>(w, input.Get(), output.Get(), rows,
                                             columns);
           }},
          {"narrow_rows3_tile2", [&](const void* w) {
             SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 3,
                                                 core::GgmlType::kQ6_K, 2, 1,
                                                 1, true>
                 <<<(rows + 47) / 48, 512>>>(w, input.Get(), output.Get(), rows,
                                             columns);
           }}};
    }
  }
  MappedWeights mapped_weights(weights.Get(), physical);
  geometries.front().launch(mapped_weights.Get());
  Check(hipMemcpy(reference.Get(), output.Get(), batch * rows * sizeof(float),
                  hipMemcpyDeviceToDevice));
  Event start, stop;
  const std::string label = mixed ? type + "/" + up_type : type;
  std::puts("type,m,k,batch,round,geometry,ms,GBps,mismatches");
  for (int round = 0; round < 4; ++round) {
    for (std::size_t index = 0; index < geometries.size(); ++index) {
      const auto& geometry =
          geometries[round % 2 == 0 ? index : geometries.size() - index - 1];
      Check(hipMemset(mismatch.Get(), 0, sizeof(unsigned)));
      geometry.launch(mapped_weights.Get());
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
        geometry.launch(mapped_weights.Get() + (iteration % copies) * bytes);
      }
      Check(hipEventRecord(stop.Get()));
      Check(hipEventSynchronize(stop.Get()));
      float milliseconds = 0;
      Check(hipEventElapsedTime(&milliseconds, start.Get(), stop.Get()));
      milliseconds /= iterations;
      std::printf("%s,%zu,%zu,%zu,%d,%s,%.6f,%.2f,%u\n", label.c_str(), rows,
                  columns, batch, round, geometry.name, milliseconds,
                  bytes / (milliseconds * 1e6), bad);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 8)
      throw std::invalid_argument(
          "usage: dflash_gemm_bench bf16|q8|q4|q5|iq4|q6 M K [iterations] "
          "[batch=1..8|16 (16 is BF16 injection only)] [swiglu [up-quant]]");
    const bool fused = argc >= 7;
    const int batch = argc >= 6                      ? std::stoi(argv[5])
                      : std::string(argv[1]) == "q6" ? 7
                                                     : 8;
    const int iterations = argc >= 5 ? std::stoi(argv[4]) : 24;
    if (fused && (batch != 1 || std::string(argv[6]) != "swiglu"))
      throw std::invalid_argument("swiglu requires batch=1");
    if (batch == 1) {
      Run<1>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations,
             fused, argc == 8 ? argv[7] : "");
    } else if (batch == 2) {
      Run<2>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 3) {
      Run<3>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 4) {
      Run<4>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 5) {
      Run<5>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 6) {
      Run<6>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 7) {
      Run<7>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 8) {
      Run<8>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else if (batch == 16) {
      Run<16>(argv[1], std::stoull(argv[2]), std::stoull(argv[3]), iterations);
    } else {
      throw std::invalid_argument(
          "batch must be one scalar row, 2–8 verification rows or sixteen "
          "injection rows");
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}

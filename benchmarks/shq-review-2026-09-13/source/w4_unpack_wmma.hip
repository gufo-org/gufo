// Throwaway: issue-slot cost of feeding iu8 WMMA from packed 4-bit weights on
// gfx1151 (register resident, no memory traffic). Investigation only.
//
//   nix develop -c tools/bench/build.sh <abs path>/w4_unpack_wmma.hip
//   /tmp/w4_unpack_wmma
//
// Variants (all 4 independent WMMA chains, 16 unrolled, kIters reps):
//   base          : B fragment already int8 (W8A8 inner loop)
//   unpack-interl : B from 2 packed dwords, nibble order (k, k+8) per byte:
//                   4 VALU ops per fragment (and / shift+and)
//   unpack-evenodd: contract order (even k low, odd k high): 4 ops + 2 perms
//   +epi/2, +epi/4: the Q8-style FP32 epilogue (8 cvt+fma triples) applied
//                   every 2 WMMA (per-K32 scales) or every 4 (per-K64 scales)

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>

#define HIP_OK(expr)                                                    \
  do {                                                                  \
    hipError_t err_ = (expr);                                           \
    if (err_ != hipSuccess) {                                           \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err_));                            \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
constexpr int kIters = 2048;

__device__ __forceinline__ int32x8_t W(int32x4_t a, int32x4_t b, int32x8_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
}

// Interleaved nibble order: byte j = (k_j low, k_{j+8} high) -> dwords 0,1 are
// the low nibbles of 8 bytes, dwords 2,3 the high nibbles. 4 ops.
__device__ __forceinline__ int32x4_t UnpackInterleaved(unsigned p0, unsigned p1) {
  int32x4_t b;
  b[0] = static_cast<int>(p0 & 0x0F0F0F0Fu);
  b[1] = static_cast<int>(p1 & 0x0F0F0F0Fu);
  b[2] = static_cast<int>((p0 >> 4) & 0x0F0F0F0Fu);
  b[3] = static_cast<int>((p1 >> 4) & 0x0F0F0F0Fu);
  return b;
}

// Contract order: low nibble = even k, high = odd k. After the masks, bytes
// are [k0,k2,k4,k6] and [k1,k3,k5,k7]; two v_perm_b32 restore k0..k3, k4..k7.
__device__ __forceinline__ int32x4_t UnpackEvenOdd(unsigned p0, unsigned p1) {
  const unsigned e0 = p0 & 0x0F0F0F0Fu, o0 = (p0 >> 4) & 0x0F0F0F0Fu;
  const unsigned e1 = p1 & 0x0F0F0F0Fu, o1 = (p1 >> 4) & 0x0F0F0F0Fu;
  int32x4_t b;
  b[0] = static_cast<int>(__builtin_amdgcn_perm(o0, e0, 0x05010400u));  // k0 k1 k2 k3
  b[1] = static_cast<int>(__builtin_amdgcn_perm(o0, e0, 0x07030602u));  // k4..k7
  b[2] = static_cast<int>(__builtin_amdgcn_perm(o1, e1, 0x05010400u));
  b[3] = static_cast<int>(__builtin_amdgcn_perm(o1, e1, 0x07030602u));
  return b;
}

template<int Mode, int EpiEvery>
__global__ void Kernel(float* sink, int reps, int gate) {
  int32x4_t a = {1, 2, 3, 4};
  a[0] += static_cast<int>(threadIdx.x);
  unsigned p0 = 0x12345678u + threadIdx.x, p1 = 0x9ABCDEF0u ^ threadIdx.x;
  int32x4_t bfix = {5, 6, 7, 8};
  int32x8_t c0 = {0, 0, 0, 0, 0, 0, 0, 0}, c1 = c0, c2 = c0, c3 = c0;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const float dw = 1.0000001F, dx = 0.9999999F;
  for (int r = 0; r < reps; ++r) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      int32x4_t b0, b1, b2, b3;
      if constexpr (Mode == 0) { bfix[0] = static_cast<int>(p0); bfix[1] = static_cast<int>(p1); b0 = bfix; b1 = bfix; b1[2] ^= 1; b2 = bfix; b2[3] ^= 2; b3 = bfix; b3[0] ^= 3; }
      else if constexpr (Mode == 1) {
        b0 = UnpackInterleaved(p0, p1); b1 = UnpackInterleaved(p1, p0);
        b2 = UnpackInterleaved(p0 ^ 1u, p1); b3 = UnpackInterleaved(p1 ^ 1u, p0);
      } else {
        b0 = UnpackEvenOdd(p0, p1); b1 = UnpackEvenOdd(p1, p0);
        b2 = UnpackEvenOdd(p0 ^ 1u, p1); b3 = UnpackEvenOdd(p1 ^ 1u, p0);
      }
      c0 = W(a, b0, c0); c1 = W(a, b1, c1); c2 = W(a, b2, c2); c3 = W(a, b3, c3);
      p0 += 0x01010101u; p1 ^= 0x10101010u;
      if constexpr (EpiEvery > 0) {
        if ((i % EpiEvery) == EpiEvery - 1) {
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[l] += (dw * dx) * static_cast<float>(c0[l]);
            acc[l] += (dw * dx) * static_cast<float>(c1[l]);
            acc[l] += (dw * dx) * static_cast<float>(c2[l]);
            acc[l] += (dw * dx) * static_cast<float>(c3[l]);
          }
          c0 = c1 = c2 = c3 = int32x8_t{0, 0, 0, 0, 0, 0, 0, 0};
        }
      }
    }
  }
  if (static_cast<int>(threadIdx.x) == gate) {
    float s = c0[0] + c1[0] + c2[0] + c3[0];
    for (int l = 0; l < 8; ++l) s += acc[l];
    *sink = s;
  }
}

static double Seconds(const std::function<void()>& fn, int reps) {
  fn();
  HIP_OK(hipDeviceSynchronize());
  hipEvent_t a, b;
  HIP_OK(hipEventCreate(&a)); HIP_OK(hipEventCreate(&b));
  HIP_OK(hipEventRecord(a));
  for (int i = 0; i < reps; ++i) fn();
  HIP_OK(hipEventRecord(b)); HIP_OK(hipEventSynchronize(b));
  float ms = 0; HIP_OK(hipEventElapsedTime(&ms, a, b));
  return ms / 1e3 / reps;
}

int main() {
  hipDeviceProp_t prop{};
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  const int cus = prop.multiProcessorCount * 2;
  const int blocks = cus * 8, threads = 256;
  const double waves = static_cast<double>(blocks) * threads / 32.0;
  const double ops = waves * 4.0 * 16.0 * kIters * 8192.0;
  float* d = nullptr;
  HIP_OK(hipMalloc(&d, sizeof(float)));
  std::printf("device %s, %d CUs\n", prop.gcnArchName, cus);
  auto run = [&](const char* name, auto launch) {
    const double s = Seconds(launch, 3);
    std::printf("  %-34s %7.2f TOPS\n", name, ops / s / 1e12);
  };
  run("W8A8 base", [&] { Kernel<0, 0><<<blocks, threads>>>(d, kIters, -1); });
  run("W4A8 unpack interleaved", [&] { Kernel<1, 0><<<blocks, threads>>>(d, kIters, -1); });
  run("W4A8 unpack even/odd (+perm)", [&] { Kernel<2, 0><<<blocks, threads>>>(d, kIters, -1); });
  run("W8A8 + epilogue per 2 WMMA (K32)", [&] { Kernel<0, 2><<<blocks, threads>>>(d, kIters, -1); });
  run("W8A8 + epilogue per 4 WMMA (K64)", [&] { Kernel<0, 4><<<blocks, threads>>>(d, kIters, -1); });
  run("W4A8 interleaved + epi per 4", [&] { Kernel<1, 4><<<blocks, threads>>>(d, kIters, -1); });
  run("W4A8 even/odd + epi per 4", [&] { Kernel<2, 4><<<blocks, threads>>>(d, kIters, -1); });
  run("W4A8 interleaved + epi per 2", [&] { Kernel<1, 2><<<blocks, threads>>>(d, kIters, -1); });
  return 0;
}


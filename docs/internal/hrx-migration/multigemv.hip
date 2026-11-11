// Standalone reproducer: LFM3's Q8_0MultiGEMV (SwiGLU and plain variants)
// and Q8_0GEMV, copied from lfm3-gufo-extension hip/ops.hip at debba20, timed
// on one runtime. Weights are written device-to-device (no host-upload state).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1);} } while (0)

constexpr std::size_t kQ8_0BlockSize = 32;
struct Q8_0Block { __half d; std::int8_t qs[kQ8_0BlockSize]; };
static_assert(sizeof(Q8_0Block) == 34);

struct Projection { const void* weight; float* out; int rows; };
struct Projections {
  static constexpr int kMax = 4;
  Projection job[kMax];
  int first_block[kMax];
  int count;
};

template<int R>
__device__ __forceinline__ void Q8_0DotRows(const Q8_0Block* const (&rows)[R],
                                            const float* __restrict__ x,
                                            std::size_t num_blocks,
                                            std::size_t lane_id,
                                            float (&out)[R]) {
  float sum[R] = {};
  for (std::size_t b = lane_id; b < num_blocks; b += 32) {
    const float* xb = x + (b * kQ8_0BlockSize);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (r && !rows[r]) break;
      const Q8_0Block& w = rows[r][b];
      float dot = 0.0F;
#pragma unroll
      for (int i = 0; i < 32; ++i) dot += static_cast<float>(w.qs[i]) * xb[i];
      sum[r] += __half2float(w.d) * dot;
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    for (int off = 16; off > 0; off >>= 1) sum[r] += __shfl_xor(sum[r], off);
    out[r] = sum[r];
  }
}

template<std::uint32_t WavesPerBlock, int R>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Q8_0GEMV(const void* __restrict__ A, const float* __restrict__ x,
                  float* __restrict__ y, std::size_t M, std::size_t K) {
  const std::size_t lane_id = threadIdx.x & 31u;
  const std::size_t m0 = ((blockIdx.x * WavesPerBlock) + (threadIdx.x >> 5u)) * R;
  if (m0 >= M) return;
  const std::size_t num_blocks = K / kQ8_0BlockSize;
  const auto* blocks = static_cast<const Q8_0Block*>(A);
  const Q8_0Block* rows[R];
#pragma unroll
  for (int r = 0; r < R; ++r) rows[r] = m0 + r < M ? blocks + (m0 + r) * num_blocks : nullptr;
  float sum[R];
  Q8_0DotRows(rows, x, num_blocks, lane_id, sum);
  if (lane_id == 0)
#pragma unroll
    for (int r = 0; r < R; ++r)
      if (m0 + r < M) y[m0 + r] = sum[r];
}

template<std::uint32_t WavesPerBlock, bool kSwiGLU>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Q8_0MultiGEMV(Projections jobs, const float* __restrict__ x, std::size_t K) {
  int job = 0;
  for (int j = 1; j < jobs.count; ++j)
    if (int(blockIdx.x) >= jobs.first_block[j]) job = j;
  const std::size_t lane_id = threadIdx.x & 31u, wave = threadIdx.x >> 5u;
  const std::size_t M = jobs.job[job].rows;
  const std::size_t num_blocks = K / kQ8_0BlockSize;
  float sum[1];
  if constexpr (kSwiGLU) {
    constexpr std::uint32_t kRows = WavesPerBlock / 2;
    const std::size_t row = blockIdx.x * kRows + wave % kRows;
    const int source = wave / kRows;
    __shared__ float up[kRows];
    if (row < M) {
#ifdef UNIFORM_SELECT
      const void* gate_weight = jobs.job[0].weight;
      const void* up_weight = jobs.job[1].weight;
      const Q8_0Block* rows[1] = {static_cast<const Q8_0Block*>(source ? up_weight : gate_weight) + row * num_blocks};
#else
      const Q8_0Block* rows[1] = {static_cast<const Q8_0Block*>(jobs.job[source].weight) + row * num_blocks};
#endif
      Q8_0DotRows(rows, x, num_blocks, lane_id, sum);
      if (source && !lane_id) up[wave % kRows] = sum[0];
    }
    __syncthreads();
    if (!source && row < M && !lane_id) {
      const float sig_g = 1.0F / (1.0F + expf(-sum[0]));
      jobs.job[0].out[row] = (sum[0] * sig_g) * up[wave];
    }
    return;
  }
  const std::size_t row = (blockIdx.x - jobs.first_block[job]) * WavesPerBlock + wave;
  if (row >= M) return;
  const Q8_0Block* rows[1] = {static_cast<const Q8_0Block*>(jobs.job[job].weight) + row * num_blocks};
  Q8_0DotRows(rows, x, num_blocks, lane_id, sum);
  if (!lane_id) jobs.job[job].out[row] = sum[0];
}

template <typename F>
static float Median(F launch) {
  hipEvent_t a, b; OK(hipEventCreate(&a)); OK(hipEventCreate(&b));
  for (int i = 0; i < 3; ++i) launch();
  OK(hipDeviceSynchronize());
  std::vector<float> v;
  for (int r = 0; r < 15; ++r) {
    OK(hipEventRecord(a, nullptr));
    for (int i = 0; i < 10; ++i) launch();
    OK(hipEventRecord(b, nullptr)); OK(hipEventSynchronize(b));
    float t; OK(hipEventElapsedTime(&t, a, b)); v.push_back(t * 1000 / 10);
  }
  std::sort(v.begin(), v.end());
  return v[7];
}

static void* DeviceRandom(std::size_t bytes, unsigned seed) {
  std::vector<unsigned char> host(bytes);
  for (std::size_t i = 0; i < bytes; ++i) { seed = seed * 1664525u + 1013904223u; host[i] = seed >> 24; }
  void *staged, *dev; OK(hipMalloc(&staged, bytes)); OK(hipMalloc(&dev, bytes));
  OK(hipMemcpy(staged, host.data(), bytes, hipMemcpyHostToDevice));
  OK(hipMemcpy(dev, staged, bytes, hipMemcpyDeviceToDevice));
  OK(hipFree(staged));
  return dev;
}

int main() {
  // LFM3-1.7B shapes: hidden 2048; FFN gate/up rows (intermediate) 8192.
  const std::size_t K = 2048, M = 8192;
  const std::size_t row_bytes = K / kQ8_0BlockSize * sizeof(Q8_0Block);
  void* gate = DeviceRandom(M * row_bytes, 1);
  void* up = DeviceRandom(M * row_bytes, 2);
  // Keep scales finite: overwrite each block's half scale with 0.01.
  {
    std::vector<Q8_0Block> blocks(M * (K / 32));
    OK(hipMemcpy(blocks.data(), gate, blocks.size() * sizeof(Q8_0Block), hipMemcpyDeviceToHost));
    for (auto& b : blocks) b.d = __float2half(0.01f);
    void* staged; OK(hipMalloc(&staged, blocks.size() * sizeof(Q8_0Block)));
    OK(hipMemcpy(staged, blocks.data(), blocks.size() * sizeof(Q8_0Block), hipMemcpyHostToDevice));
    OK(hipMemcpy(gate, staged, blocks.size() * sizeof(Q8_0Block), hipMemcpyDeviceToDevice));
    OK(hipMemcpy(up, staged, blocks.size() * sizeof(Q8_0Block), hipMemcpyDeviceToDevice));
    OK(hipFree(staged));
  }
  float *x, *y, *y2; OK(hipMalloc(&x, K * 4)); OK(hipMalloc(&y, M * 4)); OK(hipMalloc(&y2, M * 4));
  OK(hipMemset(x, 0, K * 4));
  const double bytes1 = double(M) * row_bytes;

  Projections swiglu{};
  swiglu.count = 1; swiglu.job[0] = {gate, y, int(M)}; swiglu.job[1] = {up, y2, int(M)};
  const int swiglu_blocks = int((M + 3) / 4);
  const float t_swiglu = Median([&] { Q8_0MultiGEMV<8, true><<<swiglu_blocks, 256>>>(swiglu, x, K); });

  Projections pair{};
  pair.count = 2; pair.job[0] = {gate, y, int(M)}; pair.job[1] = {up, y2, int(M)};
  pair.first_block[0] = 0; pair.first_block[1] = int((M + 3) / 4);
  const float t_pair = Median([&] { Q8_0MultiGEMV<4, false><<<int(2 * ((M + 3) / 4)), 128>>>(pair, x, K); });

  const float t_gemv = Median([&] {
    Q8_0GEMV<4, 1><<<int((M + 3) / 4), 128>>>(gate, x, y, M, K);
    Q8_0GEMV<4, 1><<<int((M + 3) / 4), 128>>>(up, x, y2, M, K);
  });
  Projections swiglu4 = swiglu;
  const float t_swiglu4 = Median([&] { Q8_0MultiGEMV<4, true><<<int((M + 1) / 2), 128>>>(swiglu4, x, K); });
  const float t_gemv8 = Median([&] {
    Q8_0GEMV<8, 1><<<int((M + 7) / 8), 256>>>(gate, x, y, M, K);
    Q8_0GEMV<8, 1><<<int((M + 7) / 8), 256>>>(up, x, y2, M, K);
  });
  std::printf("MultiGEMV SwiGLU (4 waves): %7.1f us  %5.0f GB/s\n", t_swiglu4, 2 * bytes1 / t_swiglu4 / 1e3);
  std::printf("2 x Q8_0GEMV     (8 waves): %7.1f us  %5.0f GB/s\n", t_gemv8, 2 * bytes1 / t_gemv8 / 1e3);
  std::printf("MultiGEMV SwiGLU (8 waves): %7.1f us  %5.0f GB/s\n", t_swiglu, 2 * bytes1 / t_swiglu / 1e3);
  std::printf("MultiGEMV pair   (4 waves): %7.1f us  %5.0f GB/s\n", t_pair, 2 * bytes1 / t_pair / 1e3);
  std::printf("2 x Q8_0GEMV     (4 waves): %7.1f us  %5.0f GB/s\n", t_gemv, 2 * bytes1 / t_gemv / 1e3);
  return 0;
}

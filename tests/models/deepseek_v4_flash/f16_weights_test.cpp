// Prefill projections must not depend on how much device memory is free.
// The test links with -Wl,--wrap=hipMemGetInfo and reports no free device
// memory once the first F16 weight copy has taken some, then checks that
// every layer still computes the same result from its F16 copies. It covers
// the shared-expert down projection (a row-major copy) and the attention
// output (a row-major A copy and a transposed B copy).
#include <hip/hip_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "src/models/deepseek_v4_flash/kernels/rocm/resident_api.h"

namespace {
bool g_starve = false;
size_t g_free_at_starve = 0;
}  // namespace

extern "C" hipError_t __real_hipMemGetInfo(size_t* free_bytes,
                                           size_t* total_bytes);
extern "C" hipError_t __wrap_hipMemGetInfo(size_t* free_bytes,
                                           size_t* total_bytes) {
  const hipError_t status = __real_hipMemGetInfo(free_bytes, total_bytes);
  if (status == hipSuccess && g_starve && *free_bytes < g_free_at_starve)
    *free_bytes = 0;
  return status;
}

namespace {
constexpr uint32_t kIn = 2048;
constexpr uint32_t kOut = 4096;
constexpr uint32_t kRows = 32;
constexpr uint32_t kLayers = 4;

constexpr uint32_t kGroups = 4;
constexpr uint32_t kGroupDim = 1024;
constexpr uint32_t kRank = 256;
constexpr uint32_t kLowDim = kGroups * kRank;
constexpr uint32_t kAttnOut = 2048;
constexpr uint32_t kTokens = 96;
constexpr double kAttentionTolerance = 1e-3;

uint64_t Q8Bytes(uint64_t rows, uint64_t in) {
  return rows * (in / 32) * 34;
}

const uint64_t kProjBytes = Q8Bytes(kOut, kIn);
const uint64_t kABytes = Q8Bytes(kLowDim, kGroupDim);
const uint64_t kBBytes = Q8Bytes(kAttnOut, kLowDim);
const uint64_t kLayerBytes = kProjBytes + kABytes + kBBytes;

void Check(bool success, const char* message) {
  if (!success) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

struct Tensor {
  explicit Tensor(size_t bytes) : data(ds4_gpu_tensor_alloc(bytes)) {
    Check(data != nullptr, "allocate tensor");
  }
  ~Tensor() { ds4_gpu_tensor_free(data); }
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  ds4_gpu_tensor* data;
};

float Half(float value) {
  return (float)(_Float16)value;
}

std::vector<unsigned char> MakeWeights(uint64_t rows, uint64_t in,
                                       unsigned seed) {
  const uint64_t blocks = in / 32;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> q(-127, 127);
  std::uniform_real_distribution<float> scale(0.5f, 1.5f);
  std::vector<unsigned char> w(Q8Bytes(rows, in));
  for (uint64_t b = 0; b < rows * blocks; ++b) {
    const _Float16 d = (_Float16)(scale(rng) / 4096.0f);
    std::memcpy(&w[b * 34], &d, 2);
    for (int j = 0; j < 32; ++j)
      w[b * 34 + 2 + j] = (unsigned char)(int8_t)q(rng);
  }
  return w;
}

// The F16 copy's values: each weight dequantized, then rounded to F16.
std::vector<float> DequantF16(const std::vector<unsigned char>& w,
                              uint64_t rows, uint64_t in) {
  std::vector<float> out(rows * in);
  for (uint64_t o = 0; o < rows; ++o) {
    for (uint64_t i = 0; i < in; ++i) {
      const unsigned char* blk = &w[(o * (in / 32) + i / 32) * 34];
      _Float16 d;
      std::memcpy(&d, blk, 2);
      out[o * in + i] = Half((float)d * (float)(int8_t)blk[2 + i % 32]);
    }
  }
  return out;
}

std::vector<float> RandomActivations(size_t count, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> x(count);
  for (float& v : x)
    v = normal(rng);
  return x;
}

// The F16 route's arithmetic: weights dequantized to F16, activations rounded
// to F16, products accumulated without further rounding.
std::vector<double> ProjectionReference(const std::vector<unsigned char>& w,
                                        const std::vector<float>& x) {
  const std::vector<float> wh = DequantF16(w, kOut, kIn);
  std::vector<double> out((size_t)kRows * kOut);
  for (uint32_t o = 0; o < kOut; ++o) {
    for (uint32_t t = 0; t < kRows; ++t) {
      double sum = 0;
      for (uint32_t i = 0; i < kIn; ++i)
        sum += (double)wh[(size_t)o * kIn + i] * Half(x[(size_t)t * kIn + i]);
      out[(size_t)t * kOut + o] = sum;
    }
  }
  return out;
}

// Attention output: grouped A projection of F16 heads, the low result rounded
// to F16 between the two GEMMs, then the B projection.
std::vector<double> AttentionReference(const std::vector<unsigned char>& a,
                                       const std::vector<unsigned char>& b,
                                       const std::vector<float>& heads) {
  const std::vector<float> ah = DequantF16(a, kLowDim, kGroupDim);
  const std::vector<float> bh = DequantF16(b, kAttnOut, kLowDim);
  std::vector<float> low((size_t)kTokens * kLowDim);
  for (uint32_t t = 0; t < kTokens; ++t) {
    for (uint32_t g = 0; g < kGroups; ++g) {
      const float* h = &heads[((size_t)t * kGroups + g) * kGroupDim];
      for (uint32_t r = 0; r < kRank; ++r) {
        const float* row = &ah[((size_t)g * kRank + r) * kGroupDim];
        double sum = 0;
        for (uint32_t d = 0; d < kGroupDim; ++d)
          sum += (double)row[d] * Half(h[d]);
        low[(size_t)t * kLowDim + g * kRank + r] = Half((float)sum);
      }
    }
  }
  std::vector<double> out((size_t)kTokens * kAttnOut);
  for (uint32_t t = 0; t < kTokens; ++t) {
    for (uint32_t o = 0; o < kAttnOut; ++o) {
      double sum = 0;
      for (uint32_t l = 0; l < kLowDim; ++l)
        sum +=
            (double)bh[(size_t)o * kLowDim + l] * low[(size_t)t * kLowDim + l];
      out[(size_t)t * kAttnOut + o] = sum;
    }
  }
  return out;
}

void StartStarving() {
  size_t total = 0;
  g_starve = false;
  Check(__real_hipMemGetInfo(&g_free_at_starve, &total) == hipSuccess,
        "query device memory");
  g_starve = true;
}

// Every layer's result must equal layer 0's, and layer 0's must match the F16
// reference closely enough that the Q8 path (which quantizes the activations)
// cannot reach it.
void CheckLayers(const char* name,
                 const std::vector<std::vector<float>>& results,
                 const std::vector<double>& reference, double tolerance) {
  g_starve = false;
  for (uint32_t l = 1; l < kLayers; ++l)
    Check(std::memcmp(results[l].data(), results[0].data(),
                      results[0].size() * sizeof(float)) == 0,
          "every layer matches layer 0 after device memory ran out");
  double max_error = 0, max_reference = 0;
  for (size_t i = 0; i < reference.size(); ++i) {
    max_error = std::fmax(max_error, std::fabs(results[0][i] - reference[i]));
    max_reference = std::fmax(max_reference, std::fabs(reference[i]));
  }
  std::printf("%s: max error %.3g of max |reference| %.3g\n", name, max_error,
              max_reference);
  Check(max_error <= tolerance * max_reference,
        "outputs use the F16 weight copies");
}
}  // namespace

int main() {
  if (!ds4_gpu_init())
    return 77;
  const std::vector<unsigned char> proj = MakeWeights(kOut, kIn, 7);
  const std::vector<unsigned char> attn_a = MakeWeights(kLowDim, kGroupDim, 13);
  const std::vector<unsigned char> attn_b = MakeWeights(kAttnOut, kLowDim, 17);
  std::vector<unsigned char> model(kLayers * kLayerBytes);
  for (uint32_t l = 0; l < kLayers; ++l) {
    unsigned char* layer = &model[l * kLayerBytes];
    std::memcpy(layer, proj.data(), kProjBytes);
    std::memcpy(layer + kProjBytes, attn_a.data(), kABytes);
    std::memcpy(layer + kProjBytes + kABytes, attn_b.data(), kBBytes);
  }
  Check(ds4_gpu_reserve_support_map(model.data(), model.size(), model.size()) &&
            ds4_gpu_cache_support_range(model.data(), model.size(), 0,
                                        model.size(), "f16 weight fixture"),
        "make synthetic Q8 weights resident");
  std::vector<ds4_gpu_q8_f16_weight> f16;
  for (uint32_t l = 0; l < kLayers; ++l) {
    const uint64_t base = l * kLayerBytes;
    f16.push_back({base, kIn, kOut, 0u});
    f16.push_back({base + kProjBytes, kGroupDim, kLowDim, 0u});
    f16.push_back({base + kProjBytes + kABytes, kLowDim, kAttnOut, 1u});
  }
  Check(ds4_gpu_register_q8_f16_weights(model.data(), model.size(), f16.data(),
                                        (uint32_t)f16.size()),
        "register F16 weight copies");

  {
    const std::vector<float> x = RandomActivations((size_t)kRows * kIn, 11);
    Tensor input(x.size() * sizeof(float));
    Tensor output((size_t)kRows * kOut * sizeof(float));
    Check(
        ds4_gpu_tensor_write(input.data, 0, x.data(), x.size() * sizeof(float)),
        "write activations");
    StartStarving();
    std::vector<std::vector<float>> results;
    for (uint32_t l = 0; l < kLayers; ++l) {
      Check(ds4_gpu_matmul_q8_0_tensor(output.data, model.data(), model.size(),
                                       l * kLayerBytes, kIn, kOut, input.data,
                                       kRows) &&
                ds4_gpu_synchronize(),
            "run the shared-expert-shaped projection");
      std::vector<float> result((size_t)kRows * kOut);
      Check(ds4_gpu_tensor_read(output.data, 0, result.data(),
                                result.size() * sizeof(float)),
            "read projection");
      results.push_back(std::move(result));
    }
    CheckLayers("projection", results, ProjectionReference(proj, x), 1e-5);
  }

  {
    const std::vector<float> heads =
        RandomActivations((size_t)kTokens * kGroups * kGroupDim, 19);
    Tensor heads_t(heads.size() * sizeof(float));
    Tensor low(((size_t)kTokens * kLowDim) * sizeof(float));
    Tensor output((size_t)kTokens * kAttnOut * sizeof(float));
    Check(ds4_gpu_tensor_write(heads_t.data, 0, heads.data(),
                               heads.size() * sizeof(float)),
          "write heads");
    StartStarving();
    std::vector<std::vector<float>> results;
    for (uint32_t l = 0; l < kLayers; ++l) {
      const uint64_t base = l * kLayerBytes;
      Check(ds4_gpu_attention_output_q8_batch_tensor(
                output.data, low.data, nullptr, nullptr, model.data(),
                model.size(), base + kProjBytes, base + kProjBytes + kABytes,
                kGroupDim, kRank, kGroups, kAttnOut, heads_t.data, kTokens) &&
                ds4_gpu_synchronize(),
            "run the attention output projection");
      std::vector<float> result((size_t)kTokens * kAttnOut);
      Check(ds4_gpu_tensor_read(output.data, 0, result.data(),
                                result.size() * sizeof(float)),
            "read attention output");
      results.push_back(std::move(result));
    }
    CheckLayers("attention output", results,
                AttentionReference(attn_a, attn_b, heads), kAttentionTolerance);
  }
  ds4_gpu_cleanup();
  std::puts("F16 weight copies survive device memory exhaustion");
}

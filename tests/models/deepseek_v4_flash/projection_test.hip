#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#include "src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_f16_decode.hip.hpp"
#include "src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_hc.hip.hpp"
#include "src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_iq2_gate.hip.hpp"
#include "src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_norm.hip.hpp"
#include "src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_q8_decode.hip.hpp"

void test_q2_down();

namespace {
void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

template<class T>
class Device {
public:
  explicit Device(size_t count) { check(hipMalloc(&data, count * sizeof(T))); }
  ~Device() { (void)hipFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  T* data = nullptr;
};

void equal(const char* name, const float* expected, const float* actual,
           size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(actual[i]) ||
        std::memcmp(expected + i, actual + i, sizeof(float)) != 0) {
      std::fprintf(stderr, "%s differs at %zu: %.9g != %.9g\n", name, i,
                   expected[i], actual[i]);
      std::exit(2);
    }
  }
}

template<unsigned Tokens>
void run(unsigned dim, std::mt19937& random) {
  // A partial final output warp also checks row bounds.
  constexpr unsigned rows = 67, groups = 2;
  const unsigned blocks = dim / 32;
  std::uniform_real_distribution<float> scales(.0001F, .03F);
  std::vector<unsigned char> weights(groups * rows * blocks * 34);
  for (unsigned b = 0; b < groups * rows * blocks; ++b) {
    const auto scale = __float2half(scales(random));
    std::memcpy(weights.data() + b * 34, &scale, 2);
    for (unsigned i = 0; i < 32; ++i)
      weights[b * 34 + 2 + i] =
          static_cast<unsigned char>(static_cast<int>(random() % 255) - 127);
  }
  std::vector<int8_t> input(Tokens * groups * dim);
  for (auto& value : input)
    value = static_cast<int8_t>(static_cast<int>(random() % 255) - 127);
  std::vector<float> input_scale(Tokens * groups * blocks);
  for (auto& value : input_scale)
    value = scales(random);
  input_scale.front() = 0;  // Empty activation block.
  Device<unsigned char> w(weights.size());
  Device<int8_t> x(input.size());
  Device<float> s(input_scale.size()), output(Tokens * groups * rows),
      reference(Tokens * groups * rows);
  check(
      hipMemcpy(w.data, weights.data(), weights.size(), hipMemcpyHostToDevice));
  check(hipMemcpy(x.data, input.data(), input.size(), hipMemcpyHostToDevice));
  check(hipMemcpy(s.data, input_scale.data(),
                  input_scale.size() * sizeof(float), hipMemcpyHostToDevice));
  std::vector<float> expected(Tokens * groups * rows), actual(expected.size());
  const auto compare = [&](const char* name, size_t count) {
    check(hipGetLastError());
    check(hipMemcpy(actual.data(), output.data, count * sizeof(float),
                    hipMemcpyDeviceToHost));
    equal(name, expected.data(), actual.data(), count);
  };

  for (unsigned token = 0; token < Tokens; ++token)
    matmul_q8_0_preq_rows_w32_kernel<<<rows, 32>>>(
        reference.data + token * rows, w.data, x.data + token * dim,
        s.data + token * blocks, dim, rows, blocks, 1, 1);
  check(hipMemcpy(expected.data(), reference.data,
                  Tokens * rows * sizeof(float), hipMemcpyDeviceToHost));
  // Independent double-precision Q8 formula, separate from GPU reduction order.
  for (unsigned token = 0; token < Tokens; ++token) {
    for (unsigned row : {0U, rows - 1}) {
      double sum = 0;
      for (unsigned block = 0; block < blocks; ++block) {
        const auto* weight = weights.data() + (row * blocks + block) * 34;
        __half h;
        std::memcpy(&h, weight, 2);
        int dot = 0;
        for (unsigned i = 0; i < 32; ++i)
          dot += static_cast<int8_t>(weight[2 + i]) *
                 input[token * dim + block * 32 + i];
        sum +=
            double(__half2float(h)) * input_scale[token * blocks + block] * dot;
      }
      if (std::abs(sum - expected[token * rows + row]) >
          1e-4 + 1e-6 * std::abs(sum)) {
        std::fprintf(stderr, "independent Q8 formula failed\n");
        std::exit(3);
      }
    }
  }
  matmul_q8_0_preq_batch_reuse_w32_kernel<Tokens, true>
      <<<dim3((rows + 1) / 2), 64>>>(output.data, w.data, x.data, s.data, dim,
                                     rows, blocks, Tokens, 2);
  compare("batch reuse", Tokens * rows);
  matmul_q8_0_preq_all_rows_exact_w32_kernel<Tokens>
      <<<dim3((rows + 1) / 2), 64>>>(output.data, w.data, x.data, s.data, dim,
                                     rows, blocks, 2);
  compare("flattened rows", Tokens * rows);
  if constexpr (Tokens % 2 == 0) {
    matmul_q8_0_preq_group_pair_w32_kernel<Tokens / 2>
        <<<rows, 32>>>(output.data, w.data, x.data, s.data, dim, rows, blocks,
                       0, Tokens / 2, 1);
    compare("request pair", Tokens * rows);
    matmul_q8_0_preq_equal_group_pairs_w32_kernel<1>
        <<<dim3(rows, Tokens / 2), 32>>>(output.data, w.data, x.data, s.data,
                                         dim, rows, blocks, 1);
    compare("equal request pairs", Tokens * rows);
  }
  if constexpr (Tokens == 5) {
    matmul_q8_0_preq_ragged_group_pair_w32_kernel<2, 3><<<rows, 32>>>(
        output.data, w.data, x.data, s.data, dim, rows, blocks, 0, 2, 1);
    compare("ragged request pair", Tokens * rows);
  }

  for (unsigned token = 0; token < Tokens; ++token)
    matmul_q8_0_pair_preq_warp8_kernel<<<(rows + 7) / 8, 256>>>(
        reference.data + token * rows, reference.data + (Tokens + token) * rows,
        w.data, w.data + rows * blocks * 34, x.data + token * dim,
        s.data + token * blocks, dim, rows, rows, blocks, 1);
  check(hipMemcpy(expected.data(), reference.data,
                  expected.size() * sizeof(float), hipMemcpyDeviceToHost));
  matmul_q8_0_preq_batch_reuse_w32_kernel<Tokens, true, true>
      <<<dim3(rows, 2), 32>>>(output.data, w.data, x.data, s.data, dim, rows,
                              blocks, Tokens, 1, w.data + rows * blocks * 34,
                              output.data + Tokens * rows);
  compare("paired projections", expected.size());
  for (unsigned first = 0; first < Tokens; first += 4) {
    const unsigned count = std::min(4U, Tokens - first);
    matmul_q8_0_preq_batch_reuse_w32_kernel<4, false, true>
        <<<dim3(rows, 2), 32>>>(
            output.data + first * rows, w.data, x.data + first * dim,
            s.data + first * blocks, dim, rows, blocks, count, 1,
            w.data + rows * blocks * 34, output.data + (Tokens + first) * rows);
  }
  compare("paired projection tails", expected.size());

  grouped_q8_0_a_preq_warp8_kernel<<<dim3((groups * rows + 7) / 8, Tokens),
                                     256>>>(reference.data, w.data, x.data,
                                            s.data, dim, rows, groups, Tokens,
                                            blocks, 1);
  check(hipMemcpy(expected.data(), reference.data,
                  expected.size() * sizeof(float), hipMemcpyDeviceToHost));
  grouped_q8_0_a_preq_batch_reuse_w32_kernel<Tokens, true>
      <<<dim3((groups * rows + 1) / 2), 64>>>(output.data, w.data, x.data,
                                              s.data, dim, rows, groups, Tokens,
                                              blocks, 2);
  compare("attention output A", expected.size());
  std::printf("Q8 dim=%u rows=%u: exact scalar/batch projections\n", dim,
              Tokens);
}

__global__ void unpack_iq2(uint64_t* output) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t low, high;
  dev_iq2_i8x8_lut(hip_iq2xxs_grid, index / 128, index % 128, &low, &high);
  output[index] = uint64_t(uint32_t(low)) | (uint64_t(uint32_t(high)) << 32);
}

void iq2_signs() {
  std::array<uint64_t, 256> grid{};
  check(hipMemcpyFromSymbol(grid.data(), HIP_SYMBOL(hip_iq2xxs_grid),
                            sizeof(grid), 0, hipMemcpyDeviceToHost));
  std::vector<uint64_t> actual(256 * 128);
  Device<uint64_t> output(actual.size());
  unpack_iq2<<<128, 256>>>(output.data);
  check(hipGetLastError());
  check(hipMemcpy(actual.data(), output.data, actual.size() * sizeof(uint64_t),
                  hipMemcpyDeviceToHost));
  for (unsigned g = 0; g < grid.size(); ++g) {
    for (unsigned sign7 = 0; sign7 < 128; ++sign7) {
      const unsigned signs = sign7 | ((std::popcount(sign7) & 1) << 7);
      uint64_t expected = 0;
      for (unsigned byte = 0; byte < 8; ++byte) {
        const int magnitude = (grid[g] >> (8 * byte)) & 255;
        if (magnitude == 0 || magnitude > 127) {
          std::fprintf(stderr,
                       "IQ2 packed negation requires nonzero int8 grid\n");
          std::exit(2);
        }
        const int value = signs & (1U << byte) ? -magnitude : magnitude;
        expected |= uint64_t(uint8_t(value)) << (8 * byte);
      }
      if (actual[g * 128 + sign7] != expected) {
        std::fprintf(stderr, "IQ2 sign expansion differs: grid=%u sign=%u\n", g,
                     sign7);
        std::exit(2);
      }
    }
  }
  std::puts("IQ2: all 32768 grid/sign combinations exact");
}

void iq2_gate(unsigned tokens, unsigned blocks, std::mt19937& random) {
  constexpr unsigned experts = 13, selected_count = 6, rows = 128;
  std::uniform_real_distribution<float> scale(.0001F, .01F);
  std::vector<hip_block_iq2_xxs> gate(experts * rows * blocks), up(gate.size());
  for (auto* matrix : {&gate, &up}) {
    for (auto& block : *matrix) {
      const auto h = __float2half(scale(random));
      std::memcpy(&block.d, &h, sizeof(h));
      for (auto& q : block.qs)
        q = static_cast<uint16_t>(random());
    }
  }
  std::vector<hip_block_q8_K> input(tokens * blocks);
  for (auto& block : input) {
    block.d = scale(random) * (random() % 2 ? 1.F : -1.F);
    for (auto& q : block.qs)
      q = static_cast<int8_t>(static_cast<int>(random() % 255) - 127);
  }
  std::vector<int32_t> selected(tokens * selected_count);
  std::vector<float> weights(selected.size());
  std::vector<std::vector<unsigned>> assignments(experts);
  for (unsigned token = 0; token < tokens; ++token) {
    for (unsigned slot = 0; slot < selected_count; ++slot) {
      const unsigned pair = token * selected_count + slot;
      const unsigned expert = (slot * 3 + token * 2) % experts;
      selected[pair] = expert;
      assignments[expert].push_back(pair);
      weights[pair] = scale(random) * 100;
    }
  }
  std::vector<unsigned> pairs, offsets, counts, tile_experts, starts;
  for (unsigned expert = 0; expert < experts; ++expert) {
    offsets.push_back(pairs.size());
    counts.push_back(assignments[expert].size());
    pairs.insert(pairs.end(), assignments[expert].begin(),
                 assignments[expert].end());
    for (unsigned first = 0; first < assignments[expert].size(); first += 4) {
      tile_experts.push_back(expert);
      starts.push_back(first);
    }
  }
  const unsigned tiles = tile_experts.size();
  Device<hip_block_iq2_xxs> g(gate.size()), u(up.size());
  Device<hip_block_q8_K> x(input.size());
  Device<int32_t> selection(selected.size());
  Device<unsigned> p(pairs.size()), o(offsets.size()), c(counts.size()),
      te(tiles), ts(tiles), total(1);
  Device<float> w(weights.size());
  const auto upload = [](auto* destination, const auto& values) {
    check(hipMemcpy(destination, values.data(),
                    values.size() * sizeof(values.front()),
                    hipMemcpyHostToDevice));
  };
  upload(g.data, gate);
  upload(u.data, up);
  upload(x.data, input);
  upload(selection.data, selected);
  upload(p.data, pairs);
  upload(o.data, offsets);
  upload(c.data, counts);
  upload(te.data, tile_experts);
  upload(ts.data, starts);
  upload(w.data, weights);
  check(hipMemcpy(total.data, &tiles, sizeof(tiles), hipMemcpyHostToDevice));
  const size_t size = tokens * selected_count * rows;
  std::array<uint64_t, 256> grid{};
  check(hipMemcpyFromSymbol(grid.data(), HIP_SYMBOL(hip_iq2xxs_grid),
                            sizeof(grid), 0, hipMemcpyDeviceToHost));
  // Unpack individual weights on the CPU. Derive the parity bit instead of
  // sharing the GPU sign lookup or integer-dot helpers.
  const auto dot = [&](const auto& matrix, unsigned pair, unsigned row) {
    double sum = 0;
    for (unsigned b = 0; b < blocks; ++b) {
      const auto& weight = matrix[(selected[pair] * rows + row) * blocks + b];
      const auto& activation = input[(pair / selected_count) * blocks + b];
      __half h;
      std::memcpy(&h, &weight.d, sizeof(h));
      for (unsigned group = 0; group < 8; ++group) {
        const uint32_t indices = uint32_t(weight.qs[4 * group]) |
                                 (uint32_t(weight.qs[4 * group + 1]) << 16);
        const uint32_t packed = uint32_t(weight.qs[4 * group + 2]) |
                                (uint32_t(weight.qs[4 * group + 3]) << 16);
        const double scale =
            double(__half2float(h)) * (2 * (packed >> 28) + 1) / 8;
        for (unsigned sub = 0; sub < 4; ++sub) {
          const uint64_t values = grid[(indices >> (8 * sub)) & 255];
          const unsigned sign7 = (packed >> (7 * sub)) & 127;
          const unsigned signs = sign7 | ((std::popcount(sign7) & 1) << 7);
          for (unsigned i = 0; i < 8; ++i) {
            const int magnitude = (values >> (8 * i)) & 255;
            const int value = signs & (1U << i) ? -magnitude : magnitude;
            sum += scale * value * double(activation.d) *
                   activation.qs[group * 32 + sub * 8 + i];
          }
        }
      }
    }
    return sum;
  };
  std::vector<std::pair<size_t, std::array<double, 3>>> samples;
  for (unsigned pair : {0U, tokens * selected_count - 1}) {
    for (unsigned row : {0U, rows / 2, rows - 1}) {
      const double g = std::min(dot(gate, pair, row), 10.);
      const double u = std::clamp(dot(up, pair, row), -10., 10.);
      samples.push_back({pair * rows + row,
                         {g, u, g / (1 + std::exp(-g)) * u * weights[pair]}});
    }
  }
  Device<float> reference_gate(size), reference_up(size), reference_mid(size),
      actual_gate(size), actual_up(size), actual_mid(size);
  const size_t row_bytes = blocks * sizeof(hip_block_iq2_xxs);
  moe_gate_up_mid_decode_lut_qwarp32_kernel<<<dim3(1, tokens * selected_count),
                                              256>>>(
      reference_gate.data, reference_up.data, reference_mid.data,
      reinterpret_cast<const char*>(g.data),
      reinterpret_cast<const char*>(u.data), x.data, selection.data, w.data,
      rows * row_bytes, row_bytes, blocks, rows, selected_count, 1, 10.F);
  {
    moe_gate_up_mid_expert_tile4_row32_kernel<<<dim3(rows / 32, tiles), 256>>>(
        actual_gate.data, actual_up.data, actual_mid.data,
        reinterpret_cast<const char*>(g.data),
        reinterpret_cast<const char*>(u.data), x.data, p.data, o.data, c.data,
        total.data, te.data, ts.data, w.data, rows * row_bytes, row_bytes,
        blocks, rows, selected_count, 0, 1, 10.F);
    check(hipGetLastError());
    std::vector<float> expected(size), actual(size);
    unsigned kind = 0;
    for (const auto [reference, candidate] :
         {std::pair{reference_gate.data, actual_gate.data},
          std::pair{reference_up.data, actual_up.data},
          std::pair{reference_mid.data, actual_mid.data}}) {
      check(hipMemcpy(expected.data(), reference, size * sizeof(float),
                      hipMemcpyDeviceToHost));
      check(hipMemcpy(actual.data(), candidate, size * sizeof(float),
                      hipMemcpyDeviceToHost));
      equal("IQ2 gate/up/activation", expected.data(), actual.data(), size);
      for (const auto& [index, values] : samples) {
        if (std::abs(actual[index] - values[kind]) >
            2e-5 * std::max(1., std::abs(values[kind]))) {
          std::fprintf(stderr,
                       "independent IQ2 formula failed at %zu kind=%u: "
                       "%.9g != %.9g\n",
                       index, kind, actual[index], values[kind]);
          std::exit(6);
        }
      }
      ++kind;
    }
  }
  std::printf(
      "IQ2 blocks=%u tokens=%u: exact scalar/batch gate/up/activation\n",
      blocks, tokens);
}

template<unsigned Tokens>
void f16_ordered(unsigned dim, unsigned rows, std::mt19937& random) {
  std::uniform_real_distribution<float> values(-1.F, 1.F);
  std::vector<__half> weights(rows * dim);
  std::vector<float> input(Tokens * dim), expected(Tokens * rows),
      actual(expected.size());
  for (auto& weight : weights)
    weight = __float2half(values(random) * .01F);
  for (auto& value : input)
    value = values(random);
  Device<__half> w(weights.size());
  Device<float> x(input.size()), reference(expected.size()),
      output(actual.size());
  check(hipMemcpy(w.data, weights.data(), weights.size() * sizeof(__half),
                  hipMemcpyHostToDevice));
  check(hipMemcpy(x.data, input.data(), input.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  matmul_f16_ordered_chunks_kernel<<<dim3(rows, Tokens), 32>>>(
      reference.data, w.data, x.data, dim, rows, Tokens);
  matmul_f16_ordered_batch_reuse_kernel<Tokens>
      <<<rows, 32>>>(output.data, w.data, x.data, dim, rows);
  check(hipGetLastError());
  check(hipMemcpy(expected.data(), reference.data,
                  expected.size() * sizeof(float), hipMemcpyDeviceToHost));
  check(hipMemcpy(actual.data(), output.data, actual.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
  equal("ordered F16 projection", expected.data(), actual.data(),
        actual.size());
  for (unsigned token = 0; token < Tokens; ++token) {
    double sum = 0;
    for (unsigned i = 0; i < dim; ++i)
      sum += double(__half2float(weights[i])) * input[token * dim + i];
    if (std::abs(sum - actual[token * rows]) > 1e-5) {
      std::fprintf(stderr, "independent F16 formula failed\n");
      std::exit(4);
    }
  }
  std::printf("F16 dim=%u tokens=%u: exact scalar/batch projection\n", dim,
              Tokens);
}

void plain_norm(unsigned rows, std::mt19937& random) {
  constexpr unsigned width = 16384;
  constexpr float eps = 1e-6F;
  std::uniform_real_distribution<float> values(-4.F, 4.F);
  std::vector<float> input(rows * width), expected(input.size()),
      actual(input.size());
  Device<float> x(input.size()), reference(input.size()), output(input.size());
  for (unsigned mode = 0; mode < 4; ++mode) {
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = values(random);
      if (mode == 1)
        input[i] = 0.F;
      if (mode == 2)
        input[i] *= 1e-20F;
      if (mode == 3)
        input[i] *= (i % 2) ? 1e12F : 1e-12F;
    }
    check(hipMemcpy(x.data, input.data(), input.size() * sizeof(float),
                    hipMemcpyHostToDevice));
    rms_norm_plain_kernel<<<rows, 256>>>(reference.data, x.data, width, rows,
                                         eps);
    rms_norm_plain_regs_kernel<64>
        <<<rows, 256>>>(output.data, x.data, width, rows, eps);
    check(hipGetLastError());
    check(hipMemcpy(expected.data(), reference.data,
                    expected.size() * sizeof(float), hipMemcpyDeviceToHost));
    check(hipMemcpy(actual.data(), output.data, actual.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    equal("plain RMS norm", expected.data(), actual.data(), actual.size());
    for (unsigned row = 0; row < rows; ++row) {
      double sum = 0;
      for (unsigned i = 0; i < width; ++i) {
        const double value = input[row * width + i];
        sum += value * value;
      }
      const double scale = 1.0 / std::sqrt(sum / width + eps);
      for (unsigned i = 0; i < width; ++i) {
        const double value = input[row * width + i] * scale;
        const float observed = actual[row * width + i];
        if (!std::isfinite(observed) || std::abs(value - observed) > 1e-5) {
          std::fprintf(stderr,
                       "plain RMS CPU formula failed: rows=%u mode=%u\n", rows,
                       mode);
          std::exit(4);
        }
      }
    }
  }
  std::printf(
      "RMS rows=%u: 4 exact scalar/cache cases and independent CPU formula\n",
      rows);
}

void hc_prefill_fp32(unsigned tokens) {
  constexpr unsigned embedding = 4096, width = 4 * embedding, mixes = 24;
  std::vector<__half> weights(mixes * width, __float2half(0.0F));
  std::vector<float> input(tokens * width), scales(3, 1.0F), bases(mixes, 0.0F);
  for (unsigned column = 0; column < mixes; ++column) {
    weights[column * width + 2 * column] = __float2half(1.0F);
    weights[column * width + 2 * column + 1] = __float2half(-1.0F);
  }
  for (unsigned token = 0; token < tokens; ++token) {
    float value = std::ldexp(1.0F, static_cast<int>(token % 4) * 2 - 2);
    if (token % 2)
      value = -value;
    std::fill_n(input.data() + token * width, width, value);
    for (unsigned column = 0; column < mixes; ++column)
      input[token * width + 2 * column] *= 1.0F + 1.0F / 4096.0F;
  }
  Device<__half> w(weights.size());
  Device<float> x(input.size()), s(scales.size()), b(bases.size()),
      output(tokens * embedding), mix(tokens * mixes), split(tokens * mixes);
  check(hipMemcpy(w.data, weights.data(), weights.size() * sizeof(__half),
                  hipMemcpyHostToDevice));
  check(hipMemcpy(x.data, input.data(), input.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  check(hipMemcpy(s.data, scales.data(), scales.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  check(hipMemcpy(b.data, bases.data(), bases.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  hc4_norm_mix_split_weighted_sum_kernel<4, 256, 16>
      <<<(tokens + 3) / 4, 1024>>>(output.data, mix.data, split.data, x.data,
                                   w.data, s.data, b.data, embedding, tokens,
                                   20, 1e-6F, 1e-6F);
  check(hipGetLastError());
  std::vector<float> actual(tokens * mixes);
  check(hipMemcpy(actual.data(), mix.data, actual.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
  // DeepSeek's official 0731 Block.hc_pre uses F.linear(x.float(), w)
  // followed by RMS scaling. It does not round normalized activations to F16.
  // Compute that formula independently in double precision on the same inputs.
  // These pairs deliberately become equal if an extra F16 rounding is added.
  for (unsigned token = 0; token < tokens; ++token) {
    double squares = 0.0;
    for (unsigned column = 0; column < width; ++column) {
      const double value = input[token * width + column];
      squares += value * value;
    }
    const double scale = 1.0 / std::sqrt(squares / width + double(1e-6F));
    for (unsigned column = 0; column < mixes; ++column) {
      const double expected = (double(input[token * width + 2 * column]) -
                               double(input[token * width + 2 * column + 1])) *
                              scale;
      const float value = actual[token * mixes + column];
      if (!std::isfinite(value) || std::abs(double(value) - expected) > 1e-9) {
        std::fprintf(stderr, "HC FP32 formula: %.9g != %.9g\n", value,
                     expected);
        std::exit(5);
      }
    }
  }
  std::printf("HC tokens=%u: official FP32 projection/RMS formula\n", tokens);
}
}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::puts("SKIP: no HIP device for DS4 projections");
    return 77;
  }
  test_q2_down();
  std::mt19937 random(29);
  for (unsigned dim : {1024U, 4096U, 8192U}) {
    run<2>(dim, random);
    run<3>(dim, random);
    run<4>(dim, random);
    run<5>(dim, random);
    run<6>(dim, random);
    run<8>(dim, random);
    run<12>(dim, random);
    run<16>(dim, random);
    run<24>(dim, random);
    run<32>(dim, random);
  }
  iq2_signs();
  for (unsigned blocks : {8U, 16U, 32U})
    for (unsigned tokens : {1U, 2U, 3U, 4U, 8U})
      iq2_gate(tokens, blocks, random);
  for (const auto [dim, rows] :
       {std::pair{16384U, 24U}, std::pair{4096U, 256U}}) {
    f16_ordered<1>(dim, rows, random);
    f16_ordered<2>(dim, rows, random);
    f16_ordered<3>(dim, rows, random);
    f16_ordered<4>(dim, rows, random);
    f16_ordered<5>(dim, rows, random);
    f16_ordered<6>(dim, rows, random);
    f16_ordered<7>(dim, rows, random);
    f16_ordered<8>(dim, rows, random);
  }
  f16_ordered<1>(16384, 4, random);
  for (unsigned rows : {1U, 2U, 6U, 8U})
    plain_norm(rows, random);
  hc_prefill_fp32(128);
  hc_prefill_fp32(129);
}

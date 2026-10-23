// Checks the value transpose and the attention kernel against a direct
// double-precision evaluation for both head widths, packed sequences, and
// windowed and whole-sequence masks.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/embedding_gemma2/hip/ops.hpp"

namespace hip = gufo::models::embedding_gemma2::hip;

namespace {

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL embedding_gemma2_attention_hip_test: " << message
              << '\n';
    std::exit(1);
  }
}

std::uint16_t ToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<std::uint16_t>(bits >> 16);
}

float FromBf16(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float value = 0.0F;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

template<typename T>
T* ToDevice(const std::vector<T>& host) {
  void* device = nullptr;
  Check(hipMalloc(&device, host.size() * sizeof(T)) == hipSuccess, "hipMalloc");
  Check(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                  hipMemcpyHostToDevice) == hipSuccess,
        "upload");
  return static_cast<T*>(device);
}

void RunCase(std::uint32_t head_dim, std::uint32_t kv_heads,
             std::uint32_t window, const std::vector<std::uint32_t>& lengths,
             float query_scale = 1.0F) {
  constexpr std::uint32_t kHeads = 4;
  std::size_t tokens = 0;
  for (const auto length : lengths) {
    tokens += length;
  }
  std::mt19937 random(head_dim + window);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  const auto fill = [&](std::size_t heads, float scale = 1.0F) {
    std::vector<std::uint16_t> values(tokens * heads * head_dim);
    for (auto& value : values) {
      value = ToBf16(normal(random) * scale);
    }
    return values;
  };
  const auto query = fill(kHeads, query_scale);
  const auto key = fill(kv_heads);
  const auto value = fill(kv_heads);

  std::vector<std::uint32_t> begin;
  std::vector<std::uint32_t> end;
  std::uint32_t offset = 0;
  for (const auto length : lengths) {
    for (std::uint32_t i = 0; i < length; ++i) {
      begin.push_back(offset);
      end.push_back(offset + length);
    }
    offset += length;
  }

  auto* d_query = ToDevice(query);
  auto* d_key = ToDevice(key);
  // The kernel reads values transposed, [kv head][column][token], padded to
  // a multiple of 32 tokens with zeros. The device transpose must match the
  // host one, including the zero tail. The buffer starts as NaN to show that
  // the tail is written.
  const std::size_t stride = hip::ValueStride(tokens);
  const std::size_t columns = std::size_t{kv_heads} * head_dim;
  std::vector<std::uint16_t> expected_value_t(columns * stride);
  for (std::size_t t = 0; t < tokens; ++t) {
    for (std::size_t c = 0; c < columns; ++c) {
      expected_value_t[c * stride + t] = value[t * columns + c];
    }
  }
  auto* d_untransposed = ToDevice(value);
  void* d_value = nullptr;
  Check(hipMalloc(&d_value, expected_value_t.size() * 2U) == hipSuccess,
        "hipMalloc value_t");
  Check(hipMemset(d_value, 0xff, expected_value_t.size() * 2U) == hipSuccess,
        "memset value_t");
  hip::LaunchTransposeValues(d_untransposed, d_value, tokens,
                             static_cast<std::uint32_t>(columns), nullptr);
  std::vector<std::uint16_t> value_t(expected_value_t.size());
  Check(hipMemcpy(value_t.data(), d_value, value_t.size() * 2U,
                  hipMemcpyDeviceToHost) == hipSuccess,
        "download value_t");
  Check(value_t == expected_value_t,
        "device transpose differs from the host transpose");
  auto* d_begin = ToDevice(begin);
  auto* d_end = ToDevice(end);
  std::uint16_t* d_context = nullptr;
  Check(hipMalloc(reinterpret_cast<void**>(&d_context),
                  tokens * kHeads * head_dim * 2U) == hipSuccess,
        "hipMalloc context");

  const hip::QkvLayout layout{kHeads, kv_heads, head_dim};
  Check(hip::LaunchAttention(d_query, d_key, d_value, d_context, d_begin, d_end,
                             tokens, layout, window, nullptr),
        "supported geometry");
  std::vector<std::uint16_t> context(tokens * kHeads * head_dim);
  Check(hipMemcpy(context.data(), d_context, context.size() * 2U,
                  hipMemcpyDeviceToHost) == hipSuccess,
        "download");

  double worst = 0.0;
  for (std::size_t t = 0; t < tokens; ++t) {
    const std::int64_t low =
        window == 0
            ? begin[t]
            : std::max<std::int64_t>(begin[t], std::int64_t(t) - window);
    const std::int64_t high =
        window == 0
            ? end[t] - 1
            : std::min<std::int64_t>(end[t] - 1, std::int64_t(t) + window);
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::uint32_t kvh = h / (kHeads / kv_heads);
      std::vector<double> score;
      for (std::int64_t s = low; s <= high; ++s) {
        double dot = 0.0;
        for (std::uint32_t i = 0; i < head_dim; ++i) {
          dot += double(FromBf16(query[(t * kHeads + h) * head_dim + i])) *
                 FromBf16(key[(s * kv_heads + kvh) * head_dim + i]);
        }
        score.push_back(dot);
      }
      const double maximum = *std::max_element(score.begin(), score.end());
      double total = 0.0;
      for (auto& entry : score) {
        entry = std::exp(entry - maximum);
        total += entry;
      }
      for (std::uint32_t i = 0; i < head_dim; ++i) {
        double expected = 0.0;
        for (std::int64_t s = low; s <= high; ++s) {
          expected += score[s - low] / total *
                      FromBf16(value[(s * kv_heads + kvh) * head_dim + i]);
        }
        const double actual =
            FromBf16(context[(t * kHeads + h) * head_dim + i]);
        worst = std::max(worst, std::abs(actual - expected));
      }
    }
  }
  std::cout << "head " << head_dim << " window " << window << " tokens "
            << tokens << ": worst error " << worst << '\n';
  // Outputs are rounded to BF16, whose relative step is 2^-8.
  Check(worst < 0.02,
        "attention differs from the double-precision evaluation: " +
            std::to_string(worst));
  (void)hipFree(d_query);
  (void)hipFree(d_key);
  (void)hipFree(d_untransposed);
  (void)hipFree(d_value);
  (void)hipFree(d_begin);
  (void)hipFree(d_end);
  (void)hipFree(d_context);
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cerr << "SKIP embedding_gemma2_attention_hip_test: no GPU\n";
    return 77;
  }
  RunCase(256, 2, 5, {40, 17, 1, 130});
  RunCase(256, 2, 0, {40, 17, 1, 130});
  RunCase(512, 1, 0, {64, 33, 7});
  RunCase(512, 1, 9, {64, 33, 7});
  RunCase(256, 2, 100, {700});
  RunCase(256, 2, 0, {31, 32, 33, 1});
  RunCase(512, 1, 0, {32, 16, 15});
  RunCase(256, 2, 0, {32, 32});
  RunCase(512, 1, 5, {64});
  RunCase(256, 2, 0, {200}, 0.05F);
  Check(!hip::LaunchAttention(nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, 1, {8, 2, 128}, 0, nullptr),
        "an unsupported head dimension is rejected");
  Check(!hip::LaunchAttention(nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, 0, {8, 2, 256}, 0, nullptr),
        "an empty batch is rejected");
  std::cout << "PASS embedding_gemma2_attention_hip_test\n";
  return 0;
}

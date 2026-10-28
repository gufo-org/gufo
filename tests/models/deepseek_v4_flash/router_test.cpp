#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "src/models/deepseek_v4_flash/kernels/rocm/resident_api.h"

namespace {
void check(bool success) {
  if (!success)
    std::exit(1);
}

class Tensor {
public:
  explicit Tensor(size_t bytes) : data(ds4_gpu_tensor_alloc(bytes)) {
    check(data != nullptr);
  }
  ~Tensor() { ds4_gpu_tensor_free(data); }
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  ds4_gpu_tensor* data;
};

void run(unsigned rows) {
  constexpr unsigned experts = 256, selected = 6;
  std::vector<float> logits(rows * experts), probabilities(logits.size()),
      weights(rows * selected);
  std::vector<int> tokens(rows), choices(rows * selected);
  for (unsigned row = 0; row < rows; ++row) {
    for (unsigned expert = 0; expert < experts; ++expert) {
      // Dense small scores, threshold crossings, tied scores and zero rows.
      float value = float(int((expert * 73 + row * 17) % 257) - 128) / 5.5F;
      if (row % 4 == 0)
        value = 0;
      else if (row % 4 == 1)
        value /= 17;
      logits[row * experts + expert] = value;
    }
  }
  Tensor x(logits.size() * sizeof(float)), p(logits.size() * sizeof(float)),
      w(weights.size() * sizeof(float)), s(choices.size() * sizeof(int)),
      t(tokens.size() * sizeof(int));
  check(ds4_gpu_tensor_write(x.data, 0, logits.data(),
                             logits.size() * sizeof(float)));
  check(ds4_gpu_tensor_write(t.data, 0, tokens.data(),
                             tokens.size() * sizeof(int)));
  const char unused_model = 0;  // Unbiased routing reads no model tensors.
  check(ds4_gpu_router_select_batch_tensor(
      s.data, w.data, p.data, &unused_model, 1, 0, 0, 0, 0, 0, false, false,
      x.data, t.data, experts, selected, 1.5F, rows));
  check(ds4_gpu_tensor_read(p.data, 0, probabilities.data(),
                            probabilities.size() * sizeof(float)));
  check(ds4_gpu_tensor_read(w.data, 0, weights.data(),
                            weights.size() * sizeof(float)));
  check(ds4_gpu_tensor_read(s.data, 0, choices.data(),
                            choices.size() * sizeof(int)));
  for (unsigned row = 0; row < rows; ++row) {
    std::array<float, experts> expected{};
    std::array<int, experts> order{};
    std::iota(order.begin(), order.end(), 0);
    for (unsigned expert = 0; expert < experts; ++expert) {
      const double value = logits[row * experts + expert];
      const double softplus = value > 20    ? value
                              : value < -10 ? std::exp(value)
                                            : std::log1p(std::exp(value));
      expected[expert] = float(std::sqrt(softplus));
      // ROCm 7.2 retains its established F32 transcendental path; 10.1 rounds
      // only the final score. Both must meet the same independent FP64 bound.
      if (!std::isfinite(probabilities[row * experts + expert]) ||
          std::abs(probabilities[row * experts + expert] - expected[expert]) >
              2e-7F * (1.0F + expected[expert])) {
        std::fprintf(
            stderr, "router probability row=%u expert=%u: %.9g != %.9g\n", row,
            expert, probabilities[row * experts + expert], expected[expert]);
        std::exit(2);
      }
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return expected[a] > expected[b]; });
    double sum = 0;
    for (unsigned j = 0; j < selected; ++j)
      sum += expected[order[j]];
    sum = std::max(sum, double(6.103515625e-5F));
    for (unsigned j = 0; j < selected; ++j) {
      check(choices[row * selected + j] == order[j]);
      const double expected_weight = expected[order[j]] / sum * 1.5;
      check(std::isfinite(weights[row * selected + j]) &&
            std::abs(weights[row * selected + j] - expected_weight) < 2e-7);
    }
  }
}
}  // namespace

int main() {
  if (!ds4_gpu_init())
    return 77;
  for (unsigned rows : {1U, 4U, 31U, 32U, 129U})
    run(rows);
  ds4_gpu_cleanup();
  std::puts("Router scores, normalized weights and stable ties pass");
}

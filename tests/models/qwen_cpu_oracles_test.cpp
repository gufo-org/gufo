#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>

#include "src/models/qwen/qwen_oracles.hpp"

namespace {

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

void ExpectNear(float a, float b, float tol, std::string_view msg) {
  if (std::abs(a - b) > tol) {
    std::cerr << "Assertion failed: " << msg << " (got " << a << ", expected "
              << b << ", diff " << std::abs(a - b) << " > " << tol << ")\n";
    std::exit(1);
  }
}

void TestBf16Conversion() {
  const std::vector<float> test_vals = {0.0F,     -0.0F,  1.0F,     -1.0F,
                                        3.14159F, -42.5F, 65504.0F, 0.0001F};

  for (const float val : test_vals) {
    const std::uint16_t b = strix::models::qwen::FloatToBf16(val);
    const float back = strix::models::qwen::Bf16ToFloat(b);
    ExpectNear(back, val, std::abs(val) * 0.01F + 1e-5F, "BF16 roundtrip near");
  }
}

void TestRMSNorm() {
  const std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> w = {1.0F, 1.0F, 1.0F, 1.0F};
  std::vector<float> out(4, 0.0F);

  // mean(x^2) = (1 + 4 + 9 + 16)/4 = 30/4 = 7.5
  // rms = sqrt(7.5) = 2.7386127875
  const float expected_rms = std::sqrt(7.5F);

  strix::models::qwen::ReferenceRMSNorm(x, w, 0.0F, out);

  for (std::size_t i = 0; i < x.size(); ++i) {
    ExpectNear(out[i], x[i] / expected_rms, 1e-5F, "RMSNorm value match");
  }

  // Check scale invariance
  const float sum_sq =
      std::inner_product(out.begin(), out.end(), out.begin(), 0.0F);
  const float out_rms = std::sqrt(sum_sq / static_cast<float>(out.size()));
  ExpectNear(out_rms, 1.0F, 1e-5F, "Normalized RMS equals 1.0");
}

void TestRoPE() {
  // Test head of dimension 4
  const std::vector<float> head = {1.0F, 0.0F, 0.0F, 1.0F};
  std::vector<float> out(4, 0.0F);

  // At pos 0, RoPE must be pure identity
  strix::models::qwen::ReferenceRoPE(head, 0, 10000.0F, out);
  for (std::size_t i = 0; i < head.size(); ++i) {
    ExpectNear(out[i], head[i], 1e-6F, "RoPE pos 0 is identity");
  }

  // At pos > 0, L2 norm must be strictly preserved
  strix::models::qwen::ReferenceRoPE(head, 15, 10000.0F, out);
  const float norm_in =
      std::inner_product(head.begin(), head.end(), head.begin(), 0.0F);
  const float norm_out =
      std::inner_product(out.begin(), out.end(), out.begin(), 0.0F);
  ExpectNear(norm_in, norm_out, 1e-5F, "RoPE preserves L2 vector norm");
}

void TestSiLUAndSwiGLU() {
  // SiLU(0) = 0 * 0.5 = 0
  ExpectNear(strix::models::qwen::ReferenceSiLU(0.0F), 0.0F, 1e-6F,
             "SiLU(0) == 0");

  const std::vector<float> gate = {0.0F, 2.0F, -2.0F};
  const std::vector<float> up = {5.0F, 3.0F, 4.0F};
  std::vector<float> out(3, 0.0F);

  strix::models::qwen::ReferenceSwiGLU(gate, up, out);

  ExpectNear(out[0], 0.0F, 1e-6F, "SwiGLU at gate 0");
  const float expected_silu_2 = 2.0F / (1.0F + std::exp(-2.0F));
  ExpectNear(out[1], expected_silu_2 * 3.0F, 1e-5F, "SwiGLU at gate 2");
}

void TestSoftmax() {
  const std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> out(4, 0.0F);

  strix::models::qwen::ReferenceSoftmax(x, out);

  // Check sum is exactly 1.0
  const float sum = std::accumulate(out.begin(), out.end(), 0.0F);
  ExpectNear(sum, 1.0F, 1e-6F, "Softmax probabilities sum to 1.0");

  // Check monotonicity
  Expect(out[0] < out[1] && out[1] < out[2] && out[2] < out[3],
         "Softmax maintains strict monotonic ordering");
}

void TestGEMV() {
  // 2x3 matrix:
  // [1, 2, 3]
  // [4, 5, 6]
  const std::vector<float> mat = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const std::vector<float> x = {1.0F, 2.0F, 3.0F};
  std::vector<float> y(2, 0.0F);

  // y[0] = 1*1 + 2*2 + 3*3 = 14
  // y[1] = 4*1 + 5*2 + 6*3 = 32
  strix::models::qwen::ReferenceGEMV(mat, x, 2, 3, y);

  ExpectNear(y[0], 14.0F, 1e-5F, "GEMV row 0");
  ExpectNear(y[1], 32.0F, 1e-5F, "GEMV row 1");
}

}  // namespace

int main() {
  std::cout << "Running Qwen CPU reference oracles tests...\n";
  TestBf16Conversion();
  TestRMSNorm();
  TestRoPE();
  TestSiLUAndSwiGLU();
  TestSoftmax();
  TestGEMV();
  std::cout << "All Qwen CPU reference oracles tests passed successfully!\n";
  return 0;
}

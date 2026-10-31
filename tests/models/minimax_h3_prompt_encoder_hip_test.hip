#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/hip/arch.hpp"
#include "src/models/minimax_h3/prompt_encoder.hpp"
#include "src/models/minimax_h3/prompt_encoder_ops.hip.hpp"
#include "src/models/minimax_h3/sha256.hpp"
#include "src/models/minimax_h3/tokenizer.hpp"

namespace {

using gufo::minimax_h3::CancellationToken;
using gufo::minimax_h3::EncodePromptLayer50;
using gufo::minimax_h3::ModelInventory;
using gufo::minimax_h3::PromptEmbedding;
using gufo::minimax_h3::PromptEncoderLiveRegisteredHostBytes;
using gufo::minimax_h3::PromptEncoderOptions;
using gufo::minimax_h3::PromptEncoderTelemetry;
using gufo::minimax_h3::Sha256File;
using gufo::minimax_h3::Tokenizer;
namespace h3ops = gufo::minimax_h3::ops;

int failures = 0;

#define CHECK(condition)                                          \
  do {                                                            \
    if (!(condition)) {                                           \
      std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << ": " \
                << #condition << '\n';                            \
      ++failures;                                                 \
    }                                                             \
  } while (false)

std::uint16_t ToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

float ToFloat(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

template<typename T>
class DeviceArray {
public:
  explicit DeviceArray(std::size_t count) : count_(count) {
    if (hipMalloc(&data_, count * sizeof(T)) != hipSuccess) {
      data_ = nullptr;
    }
  }
  ~DeviceArray() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  bool CopyFrom(std::span<const T> input) {
    return input.size() == count_ &&
           hipMemcpy(data_, input.data(), input.size_bytes(),
                     hipMemcpyHostToDevice) == hipSuccess;
  }
  bool CopyTo(std::span<T> output) const {
    return output.size() == count_ &&
           hipMemcpy(output.data(), data_, output.size_bytes(),
                     hipMemcpyDeviceToHost) == hipSuccess;
  }
  T* get() { return data_; }
  const T* get() const { return data_; }
  explicit operator bool() const { return data_ != nullptr; }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

void CheckClose(std::span<const std::uint16_t> got,
                std::span<const float> expected, float tolerance) {
  CHECK(got.size() == expected.size());
  for (std::size_t index = 0; index < std::min(got.size(), expected.size());
       ++index) {
    CHECK(std::abs(ToFloat(got[index]) - expected[index]) <= tolerance);
  }
}

void CheckElementwiseKernels() {
  hipStream_t stream = nullptr;
  CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess);
  {
    const std::vector<std::uint16_t> left = {ToBf16(1.0F), ToBf16(-2.0F),
                                             ToBf16(0.5F), ToBf16(9.0F)};
    const std::vector<std::uint16_t> right = {ToBf16(2.0F), ToBf16(0.25F),
                                              ToBf16(-0.5F), ToBf16(-4.0F)};
    DeviceArray<std::uint16_t> d_left(left.size());
    DeviceArray<std::uint16_t> d_right(right.size());
    DeviceArray<std::uint16_t> d_output(left.size());
    CHECK(d_left && d_right && d_output);
    CHECK(d_left.CopyFrom(left));
    CHECK(d_right.CopyFrom(right));
    h3ops::LaunchAdd(d_left.get(), d_right.get(), d_output.get(), left.size(),
                     stream);
    CHECK(hipStreamSynchronize(stream) == hipSuccess);
    std::vector<std::uint16_t> got(left.size());
    CHECK(d_output.CopyTo(got));
    const std::vector<float> expected = {3.0F, -1.75F, 0.0F, 5.0F};
    CheckClose(got, expected, 0.01F);
  }
  {
    const std::vector<std::uint16_t> gate = {ToBf16(-1.0F), ToBf16(0.0F),
                                             ToBf16(2.0F)};
    const std::vector<std::uint16_t> up = {ToBf16(3.0F), ToBf16(4.0F),
                                           ToBf16(-0.5F)};
    DeviceArray<std::uint16_t> d_gate(gate.size());
    DeviceArray<std::uint16_t> d_up(up.size());
    DeviceArray<std::uint16_t> d_output(gate.size());
    CHECK(d_gate.CopyFrom(gate));
    CHECK(d_up.CopyFrom(up));
    h3ops::LaunchSwiGlu(d_gate.get(), d_up.get(), d_output.get(), gate.size(),
                        stream);
    CHECK(hipStreamSynchronize(stream) == hipSuccess);
    std::vector<std::uint16_t> got(gate.size());
    CHECK(d_output.CopyTo(got));
    const std::vector<float> expected = {
        -1.0F / (1.0F + std::exp(1.0F)) * 3.0F, 0.0F,
        2.0F / (1.0F + std::exp(-2.0F)) * -0.5F};
    CheckClose(got, expected, 0.015F);
  }
  (void)hipStreamDestroy(stream);
}

void CheckRmsNormKernel() {
  constexpr std::uint32_t kRows = 2;
  constexpr std::uint32_t kWidth = 4;
  const std::vector<std::uint16_t> input = {
      ToBf16(1.0F),  ToBf16(2.0F), ToBf16(3.0F), ToBf16(4.0F),
      ToBf16(-2.0F), ToBf16(1.0F), ToBf16(0.5F), ToBf16(-1.0F)};
  const std::vector<std::uint16_t> weight = {ToBf16(1.0F), ToBf16(0.5F),
                                             ToBf16(-1.0F), ToBf16(2.0F)};
  DeviceArray<std::uint16_t> d_input(input.size());
  DeviceArray<std::uint16_t> d_weight(weight.size());
  DeviceArray<std::uint16_t> d_output(input.size());
  CHECK(d_input.CopyFrom(input));
  CHECK(d_weight.CopyFrom(weight));
  h3ops::LaunchRmsNorm(d_input.get(), d_weight.get(), d_output.get(), kRows,
                       kWidth, 1.0e-6F, nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<std::uint16_t> got(input.size());
  CHECK(d_output.CopyTo(got));
  std::vector<float> expected(input.size());
  for (std::uint32_t row = 0; row < kRows; ++row) {
    float squares = 0.0F;
    for (std::uint32_t column = 0; column < kWidth; ++column) {
      const float value = ToFloat(input[row * kWidth + column]);
      squares = std::fma(value, value, squares);
    }
    const float inverse =
        1.0F / std::sqrt(squares / static_cast<float>(kWidth) + 1.0e-6F);
    for (std::uint32_t column = 0; column < kWidth; ++column) {
      expected[row * kWidth + column] = ToFloat(input[row * kWidth + column]) *
                                        inverse * ToFloat(weight[column]);
    }
  }
  CheckClose(got, expected, 0.015F);
}

void CheckHeadNormRopeKernels() {
  constexpr std::uint32_t kSequence = 2;
  constexpr std::uint32_t kQueryHeads = 2;
  constexpr std::uint32_t kKeyValueHeads = 1;
  constexpr std::uint32_t kHeadDimension = 4;
  std::vector<std::uint16_t> query(kSequence * kQueryHeads * kHeadDimension);
  std::vector<std::uint16_t> key(kSequence * kKeyValueHeads * kHeadDimension);
  for (std::size_t index = 0; index < query.size(); ++index) {
    query[index] = ToBf16(static_cast<float>(index + 1) / 8.0F);
  }
  for (std::size_t index = 0; index < key.size(); ++index) {
    key[index] = ToBf16(static_cast<float>(index + 2) / 7.0F);
  }
  const std::vector<std::uint16_t> weight(kHeadDimension, ToBf16(1.0F));
  DeviceArray<std::uint16_t> d_query(query.size());
  DeviceArray<std::uint16_t> d_key(key.size());
  DeviceArray<std::uint16_t> d_weight(weight.size());
  DeviceArray<float> d_cosine(kSequence * (kHeadDimension / 2U));
  DeviceArray<float> d_sine(kSequence * (kHeadDimension / 2U));
  CHECK(d_query.CopyFrom(query));
  CHECK(d_key.CopyFrom(key));
  CHECK(d_weight.CopyFrom(weight));
  h3ops::LaunchHeadRmsNorm(d_query.get(), d_weight.get(), kSequence,
                           kQueryHeads, kHeadDimension, 1.0e-6F, nullptr);
  h3ops::LaunchHeadRmsNorm(d_key.get(), d_weight.get(), kSequence,
                           kKeyValueHeads, kHeadDimension, 1.0e-6F, nullptr);
  h3ops::LaunchPrepareRope(d_cosine.get(), d_sine.get(), kSequence,
                           kHeadDimension, 5'000'000.0F, nullptr);
  h3ops::LaunchRope(d_query.get(), d_key.get(), d_cosine.get(), d_sine.get(),
                    kSequence, kQueryHeads, kKeyValueHeads, kHeadDimension,
                    nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  CHECK(d_query.CopyTo(query));
  CHECK(d_key.CopyTo(key));
  for (const std::uint16_t value : query) {
    CHECK(std::isfinite(ToFloat(value)));
  }
  for (const std::uint16_t value : key) {
    CHECK(std::isfinite(ToFloat(value)));
  }
  CHECK(std::abs(ToFloat(query[0]) - 0.365F) < 0.03F);
  CHECK(std::abs(ToFloat(query[kQueryHeads * kHeadDimension]) -
                 ToFloat(ToBf16(-0.416F))) < 0.04F);
}

void CheckAttentionKernel() {
  constexpr std::uint32_t kSequence = 2;
  constexpr std::uint32_t kQueryHeads = 2;
  constexpr std::uint32_t kKeyValueHeads = 1;
  constexpr std::uint32_t kHeadDimension = 4;
  std::vector<std::uint16_t> query = {
      ToBf16(1), ToBf16(0), ToBf16(0), ToBf16(0), ToBf16(0), ToBf16(1),
      ToBf16(0), ToBf16(0), ToBf16(1), ToBf16(1), ToBf16(0), ToBf16(0),
      ToBf16(0), ToBf16(1), ToBf16(1), ToBf16(0)};
  std::vector<std::uint16_t> key = {ToBf16(1), ToBf16(0), ToBf16(0), ToBf16(0),
                                    ToBf16(0), ToBf16(1), ToBf16(0), ToBf16(0)};
  std::vector<std::uint16_t> value = {ToBf16(1), ToBf16(2), ToBf16(3),
                                      ToBf16(4), ToBf16(5), ToBf16(6),
                                      ToBf16(7), ToBf16(8)};
  DeviceArray<std::uint16_t> d_query(query.size());
  DeviceArray<std::uint16_t> d_key(key.size());
  DeviceArray<std::uint16_t> d_value(value.size());
  DeviceArray<std::uint16_t> d_output(query.size());
  CHECK(d_query.CopyFrom(query));
  CHECK(d_key.CopyFrom(key));
  CHECK(d_value.CopyFrom(value));
  h3ops::LaunchCausalGqa(d_query.get(), d_key.get(), d_value.get(),
                         d_output.get(), kSequence, kQueryHeads, kKeyValueHeads,
                         kHeadDimension, 0.5F, nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<std::uint16_t> got(query.size());
  CHECK(d_output.CopyTo(got));
  const std::vector<float> first_row = {1, 2, 3, 4, 1, 2, 3, 4};
  CheckClose(std::span(got).first(first_row.size()), first_row, 0.02F);
  for (const std::uint16_t element : got) {
    CHECK(std::isfinite(ToFloat(element)));
  }
}

bool CheckAnalytic() {
  int device = 0;
  hipDeviceProp_t properties{};
  if (hipGetDevice(&device) != hipSuccess ||
      hipGetDeviceProperties(&properties, device) != hipSuccess ||
      !gufo::hip::IsRdna35(properties.gcnArchName)) {
    std::cout << "SKIP: RDNA 3.5 GPU is unavailable\n";
    return false;
  }
  CheckElementwiseKernels();
  CheckRmsNormKernel();
  CheckHeadNormRopeKernels();
  CheckAttentionKernel();
  CHECK(PromptEncoderLiveRegisteredHostBytes() == 0);
  return true;
}

int CheckReal() {
  const char* root_value = std::getenv("GUFO_H3_MODEL_ROOT");
  if (root_value == nullptr || *root_value == '\0') {
    std::cout << "SKIP: GUFO_H3_MODEL_ROOT unset\n";
    return 77;
  }
  const std::filesystem::path root(root_value);
  std::string error;
  auto inventory =
      ModelInventory::Inspect(root, GUFO_H3_SOURCE_MANIFEST, &error);
  CHECK(inventory.has_value());
  if (!inventory.has_value()) {
    std::cerr << error << '\n';
    return 1;
  }
  Tokenizer tokenizer;
  CHECK(Tokenizer::Load(root / "FL2VA/tokenizer/tokenizer.json", &tokenizer,
                        &error));
  const char* prompt_value = std::getenv("GUFO_H3_PROMPT_TEXT");
  const std::string prompt =
      prompt_value != nullptr && *prompt_value != '\0'
          ? prompt_value
          : "A red fox walking through snow";
  std::vector<std::uint32_t> ids;
  CHECK(tokenizer.EncodePrompt(prompt, &ids, &error));
  if (prompt_value == nullptr || *prompt_value == '\0') {
    CHECK(ids ==
          std::vector<std::uint32_t>({32, 2518, 38835, 11435, 1526, 11794}));
  }

  PromptEncoderOptions rejected;
  rejected.vision_inputs = true;
  PromptEmbedding rejected_output;
  PromptEncoderTelemetry rejected_telemetry;
  CHECK(!EncodePromptLayer50(*inventory, ids, rejected, nullptr,
                             &rejected_output, &rejected_telemetry, &error));
  CHECK(error.find("text-only") != std::string::npos);

  PromptEncoderOptions options;
  if (const char* layers = std::getenv("GUFO_H3_PROMPT_LAYERS");
      layers != nullptr && *layers != '\0') {
    options.layer_count =
        static_cast<std::size_t>(std::strtoull(layers, nullptr, 10));
  }
  const char* golden_path = std::getenv("GUFO_H3_PROMPT_GOLDEN");
  if (golden_path != nullptr && *golden_path != '\0') {
    const char* golden_layers = std::getenv("GUFO_H3_PROMPT_GOLDEN_LAYERS");
    options.layer_count = golden_layers != nullptr && *golden_layers != '\0'
                              ? static_cast<std::size_t>(
                                    std::strtoull(golden_layers, nullptr, 10))
                              : 50;
  }
  PromptEmbedding output;
  PromptEncoderTelemetry telemetry;
  const auto begin = std::chrono::steady_clock::now();
  CHECK(EncodePromptLayer50(*inventory, ids, options, nullptr, &output,
                            &telemetry, &error));
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count();
  if (output.values.empty()) {
    std::cerr << error << '\n';
    return 1;
  }
  CHECK(output.tokens == ids.size());
  CHECK(output.width == 5120);
  CHECK(telemetry.layers.size() == options.layer_count);
  CHECK(telemetry.total_dispatches == options.layer_count * 17);
  CHECK(telemetry.live_registered_host_bytes_after == 0);
  CHECK(PromptEncoderLiveRegisteredHostBytes() == 0);
  double layer_load_ms = 0.0;
  double layer_gpu_ms = 0.0;
  double layer_prefetch_wait_ms = 0.0;
  for (const auto& layer : telemetry.layers) {
    CHECK(layer.weight_bytes > 0);
    CHECK(layer.load_ms > 0.0);
    CHECK(layer.gpu_ms > 0.0);
    CHECK(layer.dispatches == 17);
    layer_load_ms += layer.load_ms;
    layer_gpu_ms += layer.gpu_ms;
    layer_prefetch_wait_ms += layer.prefetch_wait_ms;
  }
  for (const std::uint16_t value : output.values) {
    CHECK(std::isfinite(ToFloat(value)));
  }
  std::cout << "H3 prompt layers=" << options.layer_count
            << " seconds=" << seconds << " peak_device_gib="
            << static_cast<double>(telemetry.peak_device_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " dispatches=" << telemetry.total_dispatches
            << " tokens=" << output.tokens
            << " layer_load_ms=" << layer_load_ms
            << " layer_gpu_ms=" << layer_gpu_ms
            << " prefetch_wait_ms=" << layer_prefetch_wait_ms << '\n';

  if (const char* output_path = std::getenv("GUFO_H3_PROMPT_OUTPUT");
      output_path != nullptr && *output_path != '\0') {
    std::ofstream native_output(output_path, std::ios::binary);
    native_output.write(
        reinterpret_cast<const char*>(output.values.data()),
        static_cast<std::streamsize>(output.values.size() *
                                     sizeof(std::uint16_t)));
    CHECK(native_output.good());
  }

  if (golden_path != nullptr && *golden_path != '\0') {
    if (const char* expected_hash =
            std::getenv("GUFO_H3_PROMPT_GOLDEN_SHA256");
        expected_hash != nullptr && *expected_hash != '\0') {
      CHECK(Sha256File(golden_path) == expected_hash);
    }
    std::ifstream input(golden_path, std::ios::binary);
    std::vector<std::uint16_t> expected(output.values.size());
    input.read(
        reinterpret_cast<char*>(expected.data()),
        static_cast<std::streamsize>(expected.size() * sizeof(std::uint16_t)));
    CHECK(input.good() || input.eof());
    CHECK(input.gcount() == static_cast<std::streamsize>(
                                expected.size() * sizeof(std::uint16_t)));
    double squared_error = 0.0;
    double squared_expected = 0.0;
    double maximum_error = 0.0;
    double maximum_expected = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
      const double got = ToFloat(output.values[index]);
      const double want = ToFloat(expected[index]);
      const double delta = got - want;
      squared_error += delta * delta;
      squared_expected += want * want;
      maximum_error = std::max(maximum_error, std::abs(delta));
      maximum_expected = std::max(maximum_expected, std::abs(want));
    }
    const double relative_l2 =
        std::sqrt(squared_error / std::max(squared_expected, 1.0e-24));
    const double relative_max =
        maximum_error / std::max(maximum_expected, 1.0e-12);
    std::cout << std::setprecision(9) << "golden relative_l2=" << relative_l2
              << " relative_max=" << relative_max << '\n';
    CHECK(relative_l2 < 0.05);
    CHECK(relative_max < 0.1);
  }

  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || (std::string_view(argv[1]) != "--analytic" &&
                    std::string_view(argv[1]) != "--real")) {
    std::cerr << "usage: " << argv[0] << " --analytic|--real\n";
    return 2;
  }
  if (std::string_view(argv[1]) == "--real") {
    return CheckReal();
  }
  if (!CheckAnalytic()) {
    return 77;
  }
  if (failures != 0) {
    std::cerr << failures << " H3 prompt encoder analytic checks failed\n";
    return 1;
  }
  std::cout << "H3 prompt encoder analytic checks passed\n";
  return 0;
}

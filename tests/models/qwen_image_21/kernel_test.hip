#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include "src/models/qwen_image_21/hip/runtime.hpp"

namespace {
namespace image = gufo::models::qwen_image_21::hip;

template<typename T>
struct Device {
  T* data{};
  explicit Device(std::size_t count) {
    image::Check(hipMalloc(&data, count * sizeof(T)));
  }
  ~Device() { (void)hipFree(data); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};

std::uint16_t Bf16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  return (bits + 32767 + ((bits >> 16) & 1)) >> 16;
}

float Float(std::uint16_t value) {
  return std::bit_cast<float>(std::uint32_t(value) << 16);
}

void Normalization() {
  constexpr int rows = 9;
  for (int cols :
       {3, 31, 96, 128, 144, 192, 288, 384, 576, 768, 1024, 1152, 4096}) {
    std::vector<std::uint16_t> input(rows * cols), weight(cols), bias(cols),
        actual(input.size());
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        input[r * cols + c] =
            Bf16(r == 0 ? 0.0F : float((c * 17 + r * 13) % 97 - 48) / 19);
    for (int c = 0; c < cols; ++c) {
      weight[c] = Bf16(float(c % 11 + 1) / 13);
      bias[c] = Bf16(float(c % 7 - 3) / 11);
    }
    Device<std::uint16_t> x(input.size()), y(input.size()), w(cols), b(cols);
    image::Check(hipMemcpy(x.data, input.data(), input.size() * 2,
                           hipMemcpyHostToDevice));
    image::Check(
        hipMemcpy(w.data, weight.data(), cols * 2, hipMemcpyHostToDevice));
    image::Check(
        hipMemcpy(b.data, bias.data(), cols * 2, hipMemcpyHostToDevice));
    for (auto kind : {image::Norm::kLayer, image::Norm::kRms,
                      image::Norm::kZeroRms, image::Norm::kVae}) {
      image::NormalizeRows(x.data, y.data, w.data, b.data, rows, cols, kind,
                           1e-6F, nullptr);
      image::Check(hipMemcpy(actual.data(), y.data, actual.size() * 2,
                             hipMemcpyDeviceToHost));
      double error = 0, magnitude = 0;
      for (int r = 0; r < rows; ++r) {
        double mean = 0;
        if (kind == image::Norm::kLayer) {
          for (int c = 0; c < cols; ++c)
            mean += Float(input[r * cols + c]);
          mean /= cols;
        }
        double square = 0;
        for (int c = 0; c < cols; ++c) {
          const double v = Float(input[r * cols + c]) - mean;
          square += v * v;
        }
        const double inv = kind == image::Norm::kVae
                               ? 1 / std::max(std::sqrt(square), 1e-12)
                               : 1 / std::sqrt(square / cols + 1e-6);
        for (int c = 0; c < cols; ++c) {
          double v = (Float(input[r * cols + c]) - mean) * inv;
          if (kind == image::Norm::kRms)
            v = Float(Bf16(v)) * double(Float(weight[c]));
          else if (kind == image::Norm::kZeroRms)
            v *= 1 + double(Float(weight[c]));
          else if (kind == image::Norm::kVae) {
            v = Float(Bf16(Float(Bf16(v)) * std::sqrt(double(cols))));
            v = Float(Bf16(v * Float(weight[c])));
          } else
            v = v * Float(weight[c]) + Float(bias[c]);
          const double output = Float(actual[r * cols + c]);
          assert(std::isfinite(output));
          error += (output - v) * (output - v);
          magnitude += v * v;
        }
      }
      assert(std::sqrt(error / magnitude) < 0.005);
    }
  }
}

void Layout() {
  for (auto [rows, heads, kv] :
       {std::tuple{129, 32, 32}, {1024, 32, 8}, {4132, 32, 32}}) {
    for (bool inverse : {false, true}) {
      if (inverse && kv != heads)
        continue;
      const auto count = static_cast<std::size_t>(rows) * heads * 128;
      std::vector<std::uint16_t> input(static_cast<std::size_t>(rows) *
                                       (inverse ? heads : kv) * 128);
      std::vector<std::uint16_t> expected(count), actual(count);
      for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = (i * 7) % 65535;
      for (int row = 0; row < rows; ++row)
        for (int head = 0; head < heads; ++head)
          for (int d = 0; d < 128; ++d) {
            const auto head_major =
                (static_cast<std::size_t>(head) * rows + row) * 128 + d;
            const auto row_major =
                (static_cast<std::size_t>(row) * heads + head) * 128 + d;
            if (inverse)
              expected[row_major] = input[head_major];
            else
              expected[head_major] = input[(static_cast<std::size_t>(row) * kv +
                                            head / (heads / kv)) *
                                               128 +
                                           d];
          }
      Device<std::uint16_t> x(input.size()), y(count);
      image::Check(hipMemcpy(x.data, input.data(), input.size() * 2,
                             hipMemcpyHostToDevice));
      image::ReorderHeads(x.data, y.data, rows, heads, kv, 128, inverse,
                          nullptr);
      image::Check(hipGetLastError());
      image::Check(
          hipMemcpy(actual.data(), y.data, count * 2, hipMemcpyDeviceToHost));
      assert(actual == expected);
    }
  }
}

void Fused(bool masked) {
  std::mt19937 rng(43);
  constexpr int heads = 2, dim = 128, start = 128;
  constexpr std::uint16_t sentinel = 0x7fc1;
  for (int keys : {512, 513, 4132, 8230}) {
    const int rows = keys == 513 ? 253 : 384;
    const int count = std::min(128, rows - start);
    std::vector<std::uint16_t> queries(heads * rows * dim),
        key_values(heads * keys * dim), values(key_values.size());
    for (auto* input : {&queries, &key_values, &values})
      for (auto& x : *input)
        x = Bf16(static_cast<float>(int(rng() % 20001) - 10000) / 10000);
    // Alternate ordinary and sharp attention; exercise moving online maxima.
    if (keys == 513)
      for (auto& x : queries)
        x = Bf16(Float(x) * 100);
    std::vector<std::uint16_t> output(queries.size(), sentinel),
        replay(output.size());
    std::vector<int> limits(rows);
    for (int q = 0; q < rows; ++q)
      limits[q] = q % 3 == 0 ? 1 : q % 3 == 1 ? keys / 2 : keys;
    Device<int> device_limits(limits.size());
    image::Check(hipMemcpy(device_limits.data, limits.data(), limits.size() * 4,
                           hipMemcpyHostToDevice));
    const int* mask = masked ? device_limits.data : nullptr;
    Device<std::uint16_t> q(queries.size()), k(key_values.size()),
        v(values.size()), o(output.size());
    image::Check(hipMemcpy(q.data, queries.data(), queries.size() * 2,
                           hipMemcpyHostToDevice));
    image::Check(hipMemcpy(k.data, key_values.data(), key_values.size() * 2,
                           hipMemcpyHostToDevice));
    image::Check(hipMemcpy(v.data, values.data(), values.size() * 2,
                           hipMemcpyHostToDevice));
    image::Check(hipMemcpy(o.data, output.data(), output.size() * 2,
                           hipMemcpyHostToDevice));
    const auto launch = [&] {
      image::FusedAttention128(q.data, k.data, v.data, o.data, heads, rows,
                               keys, start, nullptr, count, mask);
      image::Check(hipGetLastError());
    };
    launch();
    image::Check(hipMemcpy(output.data(), o.data, output.size() * 2,
                           hipMemcpyDeviceToHost));
    launch();
    image::Check(hipMemcpy(replay.data(), o.data, replay.size() * 2,
                           hipMemcpyDeviceToHost));
    assert(output == replay);
    constexpr int tile = 128 * heads * dim;
    assert(std::all_of(output.begin(), output.begin() + tile,
                       [](auto x) { return x == sentinel; }));
    assert(std::all_of(output.begin() + (start + count) * heads * dim,
                       output.end(), [](auto x) { return x == sentinel; }));
    assert(std::all_of(output.begin() + tile,
                       output.begin() + (start + count) * heads * dim,
                       [](auto x) { return std::isfinite(Float(x)); }));
    for (int sample = 0; sample < 32; ++sample) {
      const int head = sample % heads, query = start + sample * 17 % count;
      const int channel = sample * 23 % dim;
      std::vector<double> logits(keys);
      double maximum = -INFINITY;
      const int visible = masked ? limits[query] : keys;
      for (int key = 0; key < visible; ++key) {
        double dot = 0;
        for (int d = 0; d < dim; ++d)
          dot += double(Float(queries[(head * rows + query) * dim + d])) *
                 Float(key_values[(head * keys + key) * dim + d]);
        logits[key] = dot / std::sqrt(double(dim));
        maximum = std::max(maximum, logits[key]);
      }
      double sum = 0, expected = 0, magnitude = 0;
      for (int key = 0; key < visible; ++key) {
        const double p = std::exp(logits[key] - maximum);
        const double value = Float(values[(head * keys + key) * dim + channel]);
        sum += p;
        expected += p * value;
        magnitude += p * std::abs(value);
      }
      expected /= sum;
      magnitude /= sum;
      const float actual =
          Float(output[(query * heads + head) * dim + channel]);
      assert(std::abs(actual - expected) <=
             0.00391 * std::abs(expected) + 1e-5 * magnitude);
    }
    image::FusedAttention128(q.data, k.data, v.data, o.data, heads, rows, keys,
                             0, nullptr, rows, mask);
    image::Check(hipMemcpy(output.data(), o.data, output.size() * 2,
                           hipMemcpyDeviceToHost));
    for (int begin = 0; begin < rows; begin += 128)
      image::FusedAttention128(q.data, k.data, v.data, o.data, heads, rows,
                               keys, begin, nullptr,
                               std::min(128, rows - begin), mask);
    image::Check(hipMemcpy(replay.data(), o.data, replay.size() * 2,
                           hipMemcpyDeviceToHost));
    assert(output == replay);
    if (masked) {
      constexpr int prefix = 37;
      for (int i = 0; i < rows; ++i)
        limits[i] = i < prefix ? i + 1 : keys;
      image::Check(hipMemcpy(device_limits.data, limits.data(),
                             limits.size() * sizeof(int),
                             hipMemcpyHostToDevice));
      image::FusedAttention128(q.data, k.data, v.data, o.data, heads, rows,
                               keys, 0, nullptr, rows, mask);
      image::Check(hipMemcpy(output.data(), o.data, output.size() * 2,
                             hipMemcpyDeviceToHost));
      image::FusedAttention128(q.data, k.data, v.data, o.data, heads, rows,
                               keys, 0, nullptr, prefix, mask);
      image::FusedAttention128(q.data, k.data, v.data, o.data, heads, rows,
                               keys, prefix, nullptr, rows - prefix);
      image::Check(hipMemcpy(replay.data(), o.data, replay.size() * 2,
                             hipMemcpyDeviceToHost));
      assert(output == replay);
    }
  }
}

void PackedLayout() {
  for (int rows : {1, 15, 16, 17, 129})
    for (int cols : {64, 128, 4096}) {
      std::vector<std::uint16_t> input(std::size_t(rows) * cols),
          output(input.size());
      for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = (i * 17) % 65535;
      Device<std::uint16_t> x(input.size()), y(output.size());
      image::Check(hipMemcpy(x.data, input.data(), input.size() * 2,
                             hipMemcpyHostToDevice));
      image::PackDense16(x.data, y.data, rows, cols, nullptr);
      image::Check(hipMemcpy(output.data(), y.data, output.size() * 2,
                             hipMemcpyDeviceToHost));
      for (int row = 0; row < rows; ++row)
        for (int k = 0; k < cols; ++k)
          assert(output[std::size_t(k / 16) * rows * 16 + row * 16 + k % 16] ==
                 input[std::size_t(row) * cols + k]);
    }
}

void Dense() {
  // Rejected shapes must leave their output untouched (and need no allocation).
  assert(!image::DenseBf16(nullptr, nullptr, nullptr, 0, 4096, 4096, nullptr));
  assert(
      !image::DenseBf16(nullptr, nullptr, nullptr, 4096, 1152, 4096, nullptr));
  assert(
      !image::DenseBf16(nullptr, nullptr, nullptr, 4096, 4096, 4095, nullptr));
  std::mt19937 rng(47);
  for (auto [rows, channels, inner] : {std::tuple{1, 4096, 4096},
                                       {36, 12288, 4096},
                                       {256, 4096, 12288},
                                       {4096, 4096, 4096},
                                       {4097, 12288, 4096},
                                       {4372, 4096, 12288}}) {
    std::vector<std::uint16_t> weights(std::size_t(channels) * inner),
        input(std::size_t(rows) * inner), output(std::size_t(rows) * channels),
        replay(output.size());
    for (auto* values : {&weights, &input})
      for (auto& x : *values)
        x = Bf16(float(int(rng() % 20001) - 10000) / 10000);
    Device<std::uint16_t> w(weights.size()), x(input.size()),
        packed_w(weights.size()), packed_x(input.size()), y(output.size());
    image::Check(hipMemcpy(w.data, weights.data(), weights.size() * 2,
                           hipMemcpyHostToDevice));
    image::Check(hipMemcpy(x.data, input.data(), input.size() * 2,
                           hipMemcpyHostToDevice));
    image::PackDense16(w.data, packed_w.data, channels, inner, nullptr);
    image::PackDense16(x.data, packed_x.data, rows, inner, nullptr);
    assert(image::DenseBf16(packed_w.data, packed_x.data, y.data, rows,
                            channels, inner, nullptr));
    image::Check(hipGetLastError());
    image::Check(hipMemcpy(output.data(), y.data, output.size() * 2,
                           hipMemcpyDeviceToHost));
    assert(image::DenseBf16(packed_w.data, packed_x.data, y.data, rows,
                            channels, inner, nullptr));
    image::Check(hipMemcpy(replay.data(), y.data, replay.size() * 2,
                           hipMemcpyDeviceToHost));
    assert(output == replay);
    assert(std::all_of(output.begin(), output.end(),
                       [](auto value) { return std::isfinite(Float(value)); }));
    double error = 0, magnitude = 0;
    for (int sample = 0; sample < 64; ++sample) {
      const int row =
          sample % 2 ? rows - 1 - (sample / 2 % rows) : sample * 73 % rows;
      const int channel =
          sample % 2 ? channels - 1 - sample / 2 : sample * 199 % channels;
      double expected = 0, absolute_sum = 0;
      for (int k = 0; k < inner; ++k) {
        const double product =
            double(Float(weights[std::size_t(channel) * inner + k])) *
            Float(input[std::size_t(row) * inner + k]);
        expected += product;
        absolute_sum += std::abs(product);
      }
      const double delta =
          Float(output[std::size_t(row) * channels + channel]) - expected;
      assert(std::abs(delta) <=
             0.00391 * std::abs(expected) + 1e-5 * absolute_sum);
      error += delta * delta;
      magnitude += expected * expected;
    }
    assert(std::sqrt(error / magnitude) < 0.004);
  }
}
void ConvolutionLayout() {
  assert(!image::ExpandConvolution3(nullptr, nullptr, 5, 7, 4, 1, 1, 1, 7, 0,
                                    35, nullptr));
  for (int channels : {8, 96, 144, 1152}) {
    for (auto [height, width, stride, upscale] :
         {std::tuple{5, 7, 1, 1}, {5, 7, 1, 2}, {5, 7, 2, 1}, {32, 64, 1, 1}}) {
      const int padding = stride == 2 ? 0 : 1;
      const int out_width = width * upscale / stride;
      const int out_height = height * upscale / stride;
      const int rows = out_width * out_height;
      std::vector<std::uint16_t> input(std::size_t(height) * width * channels);
      for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = (i * 17) % 65521;
      Device<std::uint16_t> x(input.size());
      image::Check(hipMemcpy(x.data, input.data(), input.size() * 2,
                             hipMemcpyHostToDevice));
      for (auto [start, count] : {std::pair{0, std::min(1024, rows)},
                                  {rows - 1, 1},
                                  {1, std::min(7, rows - 1)}}) {
        std::vector<std::uint16_t> output(std::size_t(count) * channels * 9);
        Device<std::uint16_t> y(output.size());
        assert(image::ExpandConvolution3(x.data, y.data, height, width,
                                         channels, stride, padding, upscale,
                                         out_width, start, count, nullptr));
        image::Check(hipGetLastError());
        image::Check(hipMemcpy(output.data(), y.data, output.size() * 2,
                               hipMemcpyDeviceToHost));
        for (int row = 0; row < count; ++row)
          for (int c = 0; c < channels; ++c)
            for (int k = 0; k < 9; ++k) {
              const int sy =
                  (start + row) / out_width * stride - padding + k / 3;
              const int sx =
                  (start + row) % out_width * stride - padding + k % 3;
              const std::uint16_t expected =
                  sy >= 0 && sx >= 0 && sy < height * upscale &&
                          sx < width * upscale
                      ? input[(std::size_t(sy / upscale) * width +
                               sx / upscale) *
                                  channels +
                              c]
                      : 0;
              assert(output[(std::size_t(row) * channels + c) * 9 + k] ==
                     expected);
            }
      }
    }
  }
}
}  // namespace

int main() {
  int devices = 0;
  image::Check(hipGetDeviceCount(&devices));
  if (!devices)
    return 77;
  Normalization();
  Layout();
  Fused(false);
  Fused(true);
  PackedLayout();
  Dense();
  ConvolutionLayout();
  std::cout << "Qwen-Image exact layouts and attention/projection FP64 oracles "
               "passed\n";
}

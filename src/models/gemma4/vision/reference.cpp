#include "src/models/gemma4/vision/reference.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>

#include "src/models/gemma4/vision/prompt.hpp"

namespace gufo::models::gemma4::vision {
namespace {

std::vector<float> Widen(const core::GgufTensorInfo& tensor) {
  std::vector<float> out(tensor.ElementCount());
  if (tensor.type == core::GgmlType::kF32) {
    const auto* data = static_cast<const float*>(tensor.data);
    std::copy(data, data + out.size(), out.begin());
  } else {
    const auto* data = static_cast<const std::uint16_t*>(tensor.data);
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = std::bit_cast<float>(std::uint32_t{data[i]} << 16);
    }
  }
  return out;
}

/// Runs fn(begin, end) over [0, count) on all hardware threads.
template<class Fn>
void Parallel(std::size_t count, Fn&& fn) {
  const std::size_t workers = std::min<std::size_t>(
      std::max(1U, std::thread::hardware_concurrency()), count);
  std::vector<std::thread> threads;
  for (std::size_t w = 0; w < workers; ++w) {
    threads.emplace_back(
        [&, w] { fn(count * w / workers, count * (w + 1) / workers); });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

/// out[r][o] = sum_i W[o][i] * x[r][i].
std::vector<float> MatMul(const core::GgufTensorInfo& tensor,
                          const std::vector<float>& x, std::size_t rows) {
  const auto w = Widen(tensor);
  const std::size_t in = tensor.dimensions[0];
  const std::size_t out_width = tensor.dimensions[1];
  std::vector<float> out(rows * out_width);
  Parallel(out_width, [&](std::size_t begin, std::size_t end) {
    for (std::size_t o = begin; o < end; ++o) {
      for (std::size_t r = 0; r < rows; ++r) {
        double sum = 0;
        for (std::size_t i = 0; i < in; ++i) {
          sum += double{w[o * in + i]} * x[r * in + i];
        }
        out[r * out_width + o] = static_cast<float>(sum);
      }
    }
  });
  return out;
}

/// RMS-normalizes each `width` group; `weight` may be empty.
void RmsNorm(float* x, std::size_t groups, std::size_t width,
             const std::vector<float>& weight) {
  for (std::size_t g = 0; g < groups; ++g) {
    float* row = x + g * width;
    double sum = 0;
    for (std::size_t i = 0; i < width; ++i) {
      sum += double{row[i]} * row[i];
    }
    const double scale = 1.0 / std::sqrt(sum / width + kEps);
    for (std::size_t i = 0; i < width; ++i) {
      row[i] = static_cast<float>(row[i] * scale *
                                  (weight.empty() ? 1.0 : weight[i]));
    }
  }
}

/// NEOX rotation of each half of every head: dims [0,36) by column,
/// [36,72) by row, each as pairs (i, i+18).
void Rope(float* x, std::size_t count, std::uint32_t columns) {
  constexpr std::uint32_t kHalf = kHeadDim / 2;
  constexpr std::uint32_t kPairs = kHalf / 2;
  for (std::size_t t = 0; t < count; ++t) {
    const double position[2] = {static_cast<double>(t % columns),
                                static_cast<double>(t / columns)};
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      for (std::uint32_t part = 0; part < 2; ++part) {
        float* base = x + t * kHidden + h * kHeadDim + part * kHalf;
        for (std::uint32_t i = 0; i < kPairs; ++i) {
          const double angle =
              position[part] *
              std::pow(double{kRopeTheta}, -2.0 * i / double{kHalf});
          const double c = std::cos(angle), s = std::sin(angle);
          const double a = base[i], b = base[i + kPairs];
          base[i] = static_cast<float>(a * c - b * s);
          base[i + kPairs] = static_cast<float>(b * c + a * s);
        }
      }
    }
  }
}

}  // namespace

std::vector<float> Reference::Encode(const core::Image& image,
                                     const Observer& observer) const {
  if (image.width == 0 || image.height == 0 ||
      image.width % kSideMultiple != 0 || image.height % kSideMultiple != 0 ||
      image.pixels.size() != std::size_t{image.width} * image.height * 3) {
    throw std::invalid_argument("vision reference needs a resized RGB image");
  }
  const std::uint32_t columns = image.width / kPatchSize;
  const std::uint32_t patch_rows = image.height / kPatchSize;
  const std::size_t count = std::size_t{columns} * patch_rows;
  if (columns > kPositions || patch_rows > kPositions) {
    throw std::invalid_argument("image exceeds the position tables");
  }
  const auto observe = [&](std::string_view stage,
                           const std::vector<float>& data) {
    if (observer) {
      observer(stage, data);
    }
  };

  // Patches in 2*(v/255)-1, ordered channel, row, column like the kernel.
  std::vector<float> patches(count * kPatchValues);
  for (std::size_t t = 0; t < count; ++t) {
    const std::size_t px = (t % columns) * kPatchSize;
    const std::size_t py = (t / columns) * kPatchSize;
    for (std::uint32_t c = 0; c < 3; ++c) {
      for (std::uint32_t y = 0; y < kPatchSize; ++y) {
        for (std::uint32_t x = 0; x < kPatchSize; ++x) {
          const auto value =
              image.pixels[((py + y) * image.width + px + x) * 3 + c];
          patches[t * kPatchValues + (c * kPatchSize + y) * kPatchSize + x] =
              2.0F * (static_cast<float>(value) / 255.0F) - 1.0F;
        }
      }
    }
  }
  auto h = MatMul(core::GgufTensorInfo{"",
                                       {kPatchValues, kHidden},
                                       core::GgmlType::kF32,
                                       0,
                                       weights_.patch->data,
                                       weights_.patch->size_bytes},
                  patches, count);
  const auto position = Widen(*weights_.position);
  for (std::size_t t = 0; t < count; ++t) {
    const std::size_t x = t % columns, y = t / columns;
    for (std::uint32_t d = 0; d < kHidden; ++d) {
      h[t * kHidden + d] +=
          position[x * kHidden + d] + position[(kPositions + y) * kHidden + d];
    }
  }
  observe("patch", h);

  for (std::uint32_t l = 0; l < kLayers; ++l) {
    const LayerTensors& L = weights_.layers[l];
    auto a = h;
    RmsNorm(a.data(), count, kHidden, Widen(*L.ln1));
    auto q = MatMul(*L.q, a, count);
    auto k = MatMul(*L.k, a, count);
    auto v = MatMul(*L.v, a, count);
    RmsNorm(q.data(), count * kHeads, kHeadDim, Widen(*L.q_norm));
    RmsNorm(k.data(), count * kHeads, kHeadDim, Widen(*L.k_norm));
    RmsNorm(v.data(), count * kHeads, kHeadDim, {});
    Rope(q.data(), count, columns);
    Rope(k.data(), count, columns);
    std::vector<float> attention(count * kHidden);
    Parallel(count * kHeads, [&](std::size_t begin, std::size_t end) {
      std::vector<double> scores(count);
      for (std::size_t job = begin; job < end; ++job) {
        const std::size_t t = job / kHeads, head = job % kHeads;
        const float* qt = &q[t * kHidden + head * kHeadDim];
        double maximum = -INFINITY;
        for (std::size_t s = 0; s < count; ++s) {
          const float* ks = &k[s * kHidden + head * kHeadDim];
          double dot = 0;
          for (std::uint32_t d = 0; d < kHeadDim; ++d) {
            dot += double{qt[d]} * ks[d];
          }
          scores[s] = dot;  // scale 1.0
          maximum = std::max(maximum, dot);
        }
        double sum = 0;
        for (auto& score : scores) {
          score = std::exp(score - maximum);
          sum += score;
        }
        for (std::uint32_t d = 0; d < kHeadDim; ++d) {
          double value = 0;
          for (std::size_t s = 0; s < count; ++s) {
            value += scores[s] * v[s * kHidden + head * kHeadDim + d];
          }
          attention[t * kHidden + head * kHeadDim + d] =
              static_cast<float>(value / sum);
        }
      }
    });
    auto o = MatMul(*L.out, attention, count);
    RmsNorm(o.data(), count, kHidden, Widen(*L.attn_post_norm));
    for (std::size_t i = 0; i < h.size(); ++i) {
      h[i] += o[i];
    }
    auto b = h;
    RmsNorm(b.data(), count, kHidden, Widen(*L.ln2));
    auto gate = MatMul(*L.gate, b, count);
    const auto up = MatMul(*L.up, b, count);
    for (std::size_t i = 0; i < gate.size(); ++i) {
      const double x = gate[i];
      const double gelu =
          0.5 * x *
          (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
      gate[i] = static_cast<float>(gelu * up[i]);
    }
    auto down = MatMul(*L.down, gate, count);
    RmsNorm(down.data(), count, kHidden, Widen(*L.ffn_post_norm));
    for (std::size_t i = 0; i < h.size(); ++i) {
      h[i] += down[i];
    }
    observe("layer" + std::to_string(l), h);
  }

  const std::uint32_t pooled_columns = columns / kPoolSize;
  const std::uint32_t pooled_rows = patch_rows / kPoolSize;
  const std::size_t rows = std::size_t{pooled_columns} * pooled_rows;
  std::vector<float> pooled(rows * kHidden);
  for (std::size_t r = 0; r < rows; ++r) {
    const std::size_t px = (r % pooled_columns) * kPoolSize;
    const std::size_t py = (r / pooled_columns) * kPoolSize;
    for (std::uint32_t d = 0; d < kHidden; ++d) {
      double sum = 0;
      for (std::uint32_t y = 0; y < kPoolSize; ++y) {
        for (std::uint32_t x = 0; x < kPoolSize; ++x) {
          sum += h[((py + y) * columns + px + x) * kHidden + d];
        }
      }
      pooled[r * kHidden + d] = static_cast<float>(
          sum / (kPoolSize * kPoolSize) * std::sqrt(double{kHidden}));
    }
  }
  observe("pooled", pooled);
  const auto bias = Widen(*weights_.std_bias);
  const auto scale = Widen(*weights_.std_scale);
  for (std::size_t i = 0; i < pooled.size(); ++i) {
    pooled[i] = (pooled[i] - bias[i % kHidden]) * scale[i % kHidden];
  }
  RmsNorm(pooled.data(), rows, kHidden, {});
  auto embedding = MatMul(*weights_.projection, pooled, rows);
  observe("embedding", embedding);
  return embedding;
}

}  // namespace gufo::models::gemma4::vision

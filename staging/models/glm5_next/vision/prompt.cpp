#include "staging/models/glm5_next/vision/prompt.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "staging/models/glm5_next/chat_template.hpp"

namespace gufo::models::glm5_next::vision {
namespace {

// Catmull-Rom cubic kernel (a = -0.5), matching the reference bicubic resize.
double CubicWeight(double t) {
  constexpr double kA = -0.5;
  const double x = std::fabs(t);
  if (x <= 1.0) {
    return ((kA + 2.0) * x - (kA + 3.0)) * x * x + 1.0;
  }
  if (x < 2.0) {
    return ((kA * x - 5.0 * kA) * x + 8.0 * kA) * x - 4.0 * kA;
  }
  return 0.0;
}

// One output pixel's taps along a single axis: the input indices to read and
// their normalized weights. Antialiasing widens the support by the downscale
// factor so minification does not alias.
struct AxisTaps {
  std::vector<int> index;
  std::vector<float> weight;
  // CSR offsets: pixel i reads [offset[i], offset[i + 1]).
  std::vector<std::size_t> offset;
};

AxisTaps BuildTaps(int in_size, int out_size) {
  AxisTaps taps;
  if (in_size <= 0 || out_size <= 0) {
    return taps;
  }
  const double scale = static_cast<double>(in_size) / out_size;
  const double filter_scale = std::max(1.0, scale);
  const double support = 2.0 * filter_scale;
  taps.index.reserve(static_cast<std::size_t>(2 * support) + 2);
  taps.weight.reserve(taps.index.capacity());
  taps.offset.reserve(static_cast<std::size_t>(out_size) + 1);
  for (int i = 0; i < out_size; ++i) {
    const double center = (i + 0.5) * scale - 0.5;
    const int left = static_cast<int>(std::floor(center - support));
    const int right = static_cast<int>(std::ceil(center + support));
    double sum = 0.0;
    const std::size_t base = taps.index.size();
    taps.offset.push_back(base);
    for (int j = left; j <= right; ++j) {
      const double w = CubicWeight((j - center) / filter_scale);
      if (w == 0.0) {
        continue;
      }
      taps.index.push_back(std::min(std::max(j, 0), in_size - 1));
      taps.weight.push_back(static_cast<float>(w));
      sum += w;
    }
    if (sum != 0.0) {
      const float inv = static_cast<float>(1.0 / sum);
      for (std::size_t k = base; k < taps.weight.size(); ++k) {
        taps.weight[k] *= inv;
      }
    }
  }
  taps.offset.push_back(taps.index.size());
  return taps;
}

}  // namespace

CanvasSize CanvasFor(std::uint32_t width, std::uint32_t height,
                     const Config& config) {
  if (width == 0 || height == 0 || config.min_pixels == 0 ||
      config.max_pixels == 0) {
    return {0, 0};
  }
  const std::int64_t factor = static_cast<std::int64_t>(config.AlignSize());
  const std::int64_t min_px = static_cast<std::int64_t>(config.min_pixels);
  const std::int64_t max_px = static_cast<std::int64_t>(config.max_pixels);
  const std::int64_t h = height;
  const std::int64_t w = width;
  const auto align = [factor](std::int64_t v) {
    return (v + factor - 1) / factor * factor;
  };

  std::int64_t canvas_h = align(h);
  std::int64_t canvas_w = align(w);
  if (canvas_h * canvas_w < min_px) {
    const double scale =
        std::sqrt(static_cast<double>(min_px) / static_cast<double>(h * w));
    canvas_h = align(std::max<std::int64_t>(1, static_cast<std::int64_t>(
                                                   std::ceil(h * scale))));
    canvas_w = align(std::max<std::int64_t>(1, static_cast<std::int64_t>(
                                                   std::ceil(w * scale))));
  }
  if (canvas_h * canvas_w > max_px) {
    // Largest content height whose aligned canvas fits the budget.
    std::int64_t lo = 1;
    std::int64_t hi = h;
    std::int64_t best_h = factor;
    std::int64_t best_w = factor;
    while (lo <= hi) {
      const std::int64_t ch = (lo + hi) / 2;
      const std::int64_t cw = std::max<std::int64_t>(1, w * ch / h);
      const std::int64_t ah = align(ch);
      const std::int64_t aw = align(cw);
      if (ah * aw <= max_px) {
        best_h = ah;
        best_w = aw;
        lo = ch + 1;
      } else {
        hi = ch - 1;
      }
    }
    canvas_h = best_h;
    canvas_w = best_w;
  }
  return {static_cast<std::uint32_t>(canvas_w),
          static_cast<std::uint32_t>(canvas_h)};
}

PreparedImage PrepareImage(const core::Image& image, const Config& config) {
  PreparedImage out;
  const CanvasSize canvas = CanvasFor(image.width, image.height, config);
  if (canvas.width == 0 || canvas.height == 0 || image.pixels.empty() ||
      image.pixels.size() !=
          static_cast<std::size_t>(image.width) * image.height * 3) {
    return out;
  }
  const std::int64_t height = image.height;
  const std::int64_t width = image.width;
  const std::int64_t canvas_h = canvas.height;
  const std::int64_t canvas_w = canvas.width;

  // Scaled to fit the canvas, and never upscaled, unless below the min budget.
  double scale =
      std::min(static_cast<double>(canvas_h) / height,
               static_cast<double>(canvas_w) / width);
  if (height * width >= static_cast<std::int64_t>(config.min_pixels)) {
    scale = std::min(1.0, scale);
  }
  const int content_h = static_cast<int>(std::max<std::int64_t>(
      1, std::min<std::int64_t>(canvas_h,
                                static_cast<std::int64_t>(
                                    std::floor(height * scale)))));
  const int content_w = static_cast<int>(std::max<std::int64_t>(
      1, std::min<std::int64_t>(canvas_w,
                                static_cast<std::int64_t>(
                                    std::floor(width * scale)))));

  // Bicubic resize to the content size, horizontally then vertically.
  const int sw = static_cast<int>(width);
  const int sh = static_cast<int>(height);
  const AxisTaps x_taps = BuildTaps(sw, content_w);
  const AxisTaps y_taps = BuildTaps(sh, content_h);
  // Horizontal pass: source [sh][sw][3] -> [sh][content_w][3] float.
  std::vector<float> hres(static_cast<std::size_t>(sh) * content_w * 3, 0.0F);
  for (int y = 0; y < sh; ++y) {
    const std::uint8_t* src_row =
        image.pixels.data() + static_cast<std::size_t>(y) * sw * 3;
    float* dst_row = hres.data() + static_cast<std::size_t>(y) * content_w * 3;
    for (int x = 0; x < content_w; ++x) {
      const std::size_t base = static_cast<std::size_t>(x) * 3;
      float acc[3] = {0.0F, 0.0F, 0.0F};
      for (std::size_t t = x_taps.offset[x]; t < x_taps.offset[x + 1]; ++t) {
        const float wgt = x_taps.weight[t];
        const std::uint8_t* s =
            src_row + static_cast<std::size_t>(x_taps.index[t]) * 3;
        acc[0] += wgt * s[0];
        acc[1] += wgt * s[1];
        acc[2] += wgt * s[2];
      }
      dst_row[base] = acc[0];
      dst_row[base + 1] = acc[1];
      dst_row[base + 2] = acc[2];
    }
  }

  // Black canvas of uint8-equivalent RGB values, content composited top-left.
  const std::size_t cw = static_cast<std::size_t>(canvas_w);
  const std::size_t ch = static_cast<std::size_t>(canvas_h);
  std::vector<float> canvas_rgb(cw * ch * 3, 0.0F);
  for (int y = 0; y < content_h; ++y) {
    // Vertical pass into an intermediate content plane.
    for (int x = 0; x < content_w; ++x) {
      float acc[3] = {0.0F, 0.0F, 0.0F};
      for (std::size_t t = y_taps.offset[y]; t < y_taps.offset[y + 1]; ++t) {
        const float wgt = y_taps.weight[t];
        const float* s = hres.data() +
                         static_cast<std::size_t>(y_taps.index[t]) * content_w *
                             3 +
                         static_cast<std::size_t>(x) * 3;
        acc[0] += wgt * s[0];
        acc[1] += wgt * s[1];
        acc[2] += wgt * s[2];
      }
      std::size_t dst = (static_cast<std::size_t>(y) * cw + x) * 3;
      canvas_rgb[dst] = acc[0];
      canvas_rgb[dst + 1] = acc[1];
      canvas_rgb[dst + 2] = acc[2];
    }
  }

  // CHW mean/std normalization: out[c][p] = (v / 255 - mean[c]) / std[c].
  const std::size_t plane = cw * ch;
  out.pixels.assign(3 * plane, 0.0F);
  for (std::size_t p = 0; p < plane; ++p) {
    for (int c = 0; c < 3; ++c) {
      const float v = canvas_rgb[p * 3 + c] / 255.0F;
      out.pixels[static_cast<std::size_t>(c) * plane + p] =
          (v - config.image_mean[c]) / config.image_std[c];
    }
  }
  out.nx = canvas.width;
  out.ny = canvas.height;
  const std::uint32_t align_size = config.AlignSize();
  out.tokens = (out.nx / align_size) * (out.ny / align_size);
  return out;
}

PreparedPrompt BuildPrompt(const GlmTokenizer& tokenizer,
                           std::string_view rendered,
                           const std::vector<core::Image>& images,
                           const Config& config, std::string* error_msg) {
  auto fail = [&](const std::string& message) {
    if (error_msg != nullptr) {
      *error_msg = message;
    }
    return PreparedPrompt{};
  };
  const std::optional<TokenId> marker =
      tokenizer.FindSpecialToken(kImageTag);
  if (!marker.has_value()) {
    return fail("image marker token missing from vocabulary");
  }

  const std::vector<TokenId> base =
      tokenizer.Encode(rendered, TokenizerOptions{});
  std::size_t marker_count = 0;
  for (const TokenId id : base) {
    marker_count += (id == *marker) ? 1 : 0;
  }
  if (marker_count != images.size()) {
    return fail("image marker count (" + std::to_string(marker_count) +
                ") does not match image count (" +
                std::to_string(images.size()) + ")");
  }

  PreparedPrompt out;
  out.tokens.reserve(base.size());
  out.images.reserve(images.size());
  std::size_t next_image = 0;
  for (const TokenId id : base) {
    if (id != *marker) {
      out.tokens.push_back(id);
      continue;
    }
    const PreparedImage prepared = PrepareImage(images[next_image++], config);
    if (prepared.tokens == 0) {
      return fail("image resized to zero tokens");
    }
    VisionSlot slot;
    slot.count = prepared.tokens;
    slot.pixels = std::move(prepared.pixels);
    slot.nx = prepared.nx;
    slot.ny = prepared.ny;
    slot.offset = static_cast<std::uint32_t>(out.tokens.size());
    out.tokens.insert(out.tokens.end(), prepared.tokens, *marker);
    out.images.push_back(std::move(slot));
  }
  return out;
}

}  // namespace gufo::models::glm5_next::vision

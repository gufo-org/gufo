#include "src/models/qwen/vision/prompt.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "src/core/crypto/sha256.hpp"
#include "src/models/qwen/control_tokens.hpp"

namespace gufo::models::qwen::vision {
namespace {
constexpr std::uint32_t kResizeFactor = kPatchSize * kMergeSize;
constexpr std::uint32_t kMinPixels = 65536;
constexpr std::uint32_t kMaxPixels = 16777216;

std::uint32_t RoundEven(double x) {
  const double floor = std::floor(x);
  const double part = x - floor;
  return static_cast<std::uint32_t>(
      floor +
      (part > 0.5 || (part == 0.5 && (static_cast<std::uint32_t>(floor) & 1))));
}

double Cubic(double x) {
  x = std::abs(x);
  if (x < 1)
    return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2)
    return (((-0.5 * x + 2.5) * x - 4) * x) + 2;
  return 0;
}

struct Filter {
  std::uint32_t begin;
  std::vector<std::int16_t> weights;
};

struct FilterBank {
  std::vector<Filter> rows;
  unsigned precision{0};
};

FilterBank Filters(std::uint32_t input, std::uint32_t output) {
  const double scale = static_cast<double>(input) / output;
  const double antialias = std::max(scale, 1.0);
  const double support = antialias * 2;
  FilterBank filters;
  filters.rows.reserve(output);
  std::vector<std::vector<double>> coefficients;
  coefficients.reserve(output);
  double maximum = 0;
  for (std::uint32_t i = 0; i < output; ++i) {
    const double center = (i + 0.5) * scale;
    const auto begin = std::max(0, static_cast<int>(center - support + 0.5));
    const auto end = std::min(static_cast<int>(input),
                              static_cast<int>(center + support + 0.5));
    Filter filter{static_cast<std::uint32_t>(begin), {}};
    auto& weights = coefficients.emplace_back();
    double sum = 0;
    for (int j = begin; j < end; ++j) {
      const double weight = Cubic((j + 0.5 - center) / antialias);
      weights.push_back(weight);
      sum += weight;
    }
    for (auto& weight : weights) {
      weight /= sum;
      maximum = std::max(maximum, weight);
    }
    filters.rows.push_back(std::move(filter));
  }
  // PyTorch's uint8 antialiased resize quantizes each axis' coefficients
  // to int16, choosing one shared precision for that axis.
  for (; filters.precision < 22; ++filters.precision) {
    if (static_cast<int>(0.5 + maximum * (1U << (filters.precision + 1))) >=
        (1 << 15))
      break;
  }
  for (std::size_t i = 0; i < coefficients.size(); ++i) {
    for (const auto weight : coefficients[i]) {
      const double scaled = weight * (1U << filters.precision);
      filters.rows[i].weights.push_back(
          static_cast<std::int16_t>(scaled < 0 ? scaled - 0.5 : scaled + 0.5));
    }
  }
  return filters;
}

std::uint8_t Pixel(std::int64_t value, unsigned precision) {
  return static_cast<std::uint8_t>(
      std::clamp<std::int64_t>(value >> precision, 0, 255));
}

void HashU32(crypto::Sha256Hasher& hash, std::uint32_t value) {
  const std::array<std::uint8_t, 4> bytes{
      static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8),
      static_cast<std::uint8_t>(value >> 16),
      static_cast<std::uint8_t>(value >> 24)};
  hash.Update(bytes);
}
}  // namespace

std::array<std::int32_t, 3> RopeLayout::Position(std::uint32_t physical) const {
  std::int64_t delta = 0;
  for (const auto& image : images) {
    if (physical < image.offset)
      break;
    const std::uint64_t count = std::uint64_t{image.height} * image.width;
    if (physical - image.offset < count) {
      const auto start = static_cast<std::int32_t>(image.offset + delta);
      const auto index = physical - image.offset;
      return {start, start + static_cast<std::int32_t>(index / image.width),
              start + static_cast<std::int32_t>(index % image.width)};
    }
    delta += static_cast<std::int64_t>(std::max(image.height, image.width)) -
             static_cast<std::int64_t>(count);
  }
  const auto value = static_cast<std::int32_t>(physical + delta);
  return {value, value, value};
}

std::int32_t RopeLayout::Delta() const {
  const auto end = PrefixLength();
  return Position(end)[0] - static_cast<std::int32_t>(end);
}

std::uint32_t RopeLayout::PrefixLength() const {
  if (images.empty())
    return 0;
  const auto& last = images.back();
  return last.offset + last.height * last.width;
}

RopeLayout RopeLayout::Prefix(std::uint32_t token_count) const {
  RopeLayout prefix;
  for (const auto& image : images) {
    if (image.offset >= token_count)
      break;
    prefix.images.push_back(image);
  }
  return prefix;
}

std::span<const std::uint8_t> Prompt::IdentityForPrefix(
    std::size_t token_count) const {
  for (auto image = images.rbegin(); image != images.rend(); ++image) {
    if (image->grid.offset < token_count)
      return image->prefix_identity;
  }
  return {};
}

void RopeLayout::Validate(std::uint32_t max_context) const {
  // Every non-overlapping image occupies at least one context position.
  if (images.size() > max_context)
    throw std::invalid_argument("image grids exceed model context");
  std::uint64_t previous_end = 0;
  for (const auto& image : images) {
    const std::uint64_t count = std::uint64_t{image.height} * image.width;
    if (image.height == 0 || image.width == 0 || image.offset < previous_end ||
        std::uint64_t{image.offset} + count > max_context ||
        max_context > static_cast<std::uint32_t>(
                          std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("invalid image position layout");
    }
    previous_end = image.offset + count;
  }
}

core::Image ResizeImage(const core::Image& image) {
  if (image.width == 0 || image.height == 0 ||
      std::uint64_t{image.width} * image.height > core::kMaxImagePixels ||
      image.pixels.size() != std::size_t{image.width} * image.height * 3 ||
      static_cast<double>(std::max(image.width, image.height)) /
              std::min(image.width, image.height) >
          200) {
    throw std::invalid_argument("invalid image dimensions or aspect ratio");
  }
  std::uint32_t height =
      RoundEven(static_cast<double>(image.height) / kResizeFactor) *
      kResizeFactor;
  std::uint32_t width =
      RoundEven(static_cast<double>(image.width) / kResizeFactor) *
      kResizeFactor;
  const auto rounded_pixels = std::uint64_t{height} * width;
  if (rounded_pixels > kMaxPixels) {
    const double beta =
        std::sqrt(static_cast<double>(image.width) * image.height / kMaxPixels);
    height = std::max(kResizeFactor, static_cast<std::uint32_t>(std::floor(
                                         image.height / beta / kResizeFactor)) *
                                         kResizeFactor);
    width = std::max(kResizeFactor, static_cast<std::uint32_t>(std::floor(
                                        image.width / beta / kResizeFactor)) *
                                        kResizeFactor);
  } else if (rounded_pixels < kMinPixels) {
    const double beta =
        std::sqrt(static_cast<double>(kMinPixels) /
                  (static_cast<double>(image.width) * image.height));
    height = static_cast<std::uint32_t>(
                 std::ceil(image.height * beta / kResizeFactor)) *
             kResizeFactor;
    width = static_cast<std::uint32_t>(
                std::ceil(image.width * beta / kResizeFactor)) *
            kResizeFactor;
  }
  if (width == image.width && height == image.height)
    return image;
  const auto horizontal = Filters(image.width, width);
  const auto vertical = Filters(image.height, height);
  std::vector<std::uint8_t> intermediate(std::size_t{width} * image.height * 3);
  for (std::uint32_t y = 0; y < image.height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      for (std::size_t c = 0; c < 3; ++c) {
        std::int64_t value = std::int64_t{1} << (horizontal.precision - 1);
        const auto& filter = horizontal.rows[x];
        for (std::size_t j = 0; j < filter.weights.size(); ++j) {
          value +=
              image.pixels[(std::size_t{y} * image.width + filter.begin + j) *
                               3 +
                           c] *
              filter.weights[j];
        }
        intermediate[(std::size_t{y} * width + x) * 3 + c] =
            Pixel(value, horizontal.precision);
      }
    }
  }
  core::Image resized{
      width, height,
      std::vector<std::uint8_t>(std::size_t{width} * height * 3)};
  for (std::uint32_t y = 0; y < height; ++y) {
    const auto& filter = vertical.rows[y];
    for (std::uint32_t x = 0; x < width; ++x) {
      for (std::size_t c = 0; c < 3; ++c) {
        std::int64_t value = std::int64_t{1} << (vertical.precision - 1);
        for (std::size_t j = 0; j < filter.weights.size(); ++j) {
          value += intermediate[((filter.begin + j) * width + x) * 3 + c] *
                   filter.weights[j];
        }
        resized.pixels[(std::size_t{y} * width + x) * 3 + c] =
            Pixel(value, vertical.precision);
      }
    }
  }
  return resized;
}

Prompt Prepare(const tokenization::QwenTokenizer& tokenizer,
               std::span<const tokenization::ChatMessage> messages,
               std::span<const tokenization::ChatTool> tools,
               const tokenization::ChatTemplateOptions& options,
               std::string_view encoder_identity, std::uint32_t max_context) {
  std::vector<std::size_t> offsets;
  std::vector<tokenization::ContentSpan> content_spans;
  std::size_t stable_prefix_bytes = 0;
  std::string error;
  const auto rendered = tokenization::QwenChatTemplate::Render(
      messages, tools, options, &error, &offsets, &stable_prefix_bytes,
      &content_spans);
  if (!rendered)
    throw std::invalid_argument(error);
  Prompt prompt;
  tokenization::TokenizerOptions tok_options;
  tok_options.add_bos = false;
  tok_options.add_eos = false;
  tok_options.parse_special_tokens = true;
  const auto append = [&](std::size_t begin, std::size_t end) {
    auto tokens = tokenization::QwenChatTemplate::EncodeRendered(
        tokenizer, *rendered, begin, end, content_spans, tok_options);
    if (tokens.size() > max_context - prompt.tokens.size()) {
      throw std::length_error(
          "prompt exceeds the " + std::to_string(max_context) +
          "-token context; increase --context or shorten the conversation");
    }
    prompt.tokens.insert(prompt.tokens.end(), tokens.begin(), tokens.end());
  };
  crypto::Sha256Hasher identity;
  constexpr std::string_view version = "qwen38-image-bicubic-rgb8-bf16-v1";
  identity.Update(
      {reinterpret_cast<const std::uint8_t*>(version.data()), version.size()});
  identity.Update(
      {reinterpret_cast<const std::uint8_t*>(encoder_identity.data()),
       encoder_identity.size()});
  std::size_t cursor = 0;
  std::size_t index = 0;
  for (const auto& message : messages) {
    for (const auto& image : message.images) {
      if (encoder_identity.empty()) {
        throw std::invalid_argument(
            "image input requires the model's matching mmproj");
      }
      if (index >= offsets.size())
        throw std::logic_error("image rendering lost a part");
      append(cursor, offsets[index]);
      auto pixels = ResizeImage(core::DecodeImage(*image.bytes));
      const ImageGrid grid{static_cast<std::uint32_t>(prompt.tokens.size()),
                           pixels.height / kResizeFactor,
                           pixels.width / kResizeFactor};
      const auto count = std::size_t{grid.height} * grid.width;
      if (count > max_context - prompt.tokens.size()) {
        throw std::length_error("image tokens exceed model context");
      }
      HashU32(identity, grid.offset);
      HashU32(identity, grid.height);
      HashU32(identity, grid.width);
      identity.Update(pixels.pixels);
      prompt.tokens.insert(prompt.tokens.end(), count, kImageToken);
      prompt.rope.images.push_back(grid);
      prompt.images.push_back({std::move(pixels), grid, identity.Digest()});
      cursor = offsets[index++] + tokenization::kImagePad.size();
    }
  }
  append(cursor, rendered->size());
  // The boundary starts a special token (the assistant turn, or a final user
  // turn an agent replaces each request), after every image.
  // Encode only the mutable suffix; never decode or resize images twice.
  const auto suffix = tokenization::QwenChatTemplate::EncodeRendered(
      tokenizer, *rendered, stable_prefix_bytes, rendered->size(),
      content_spans, tok_options);
  if (suffix.size() > prompt.tokens.size() ||
      !std::ranges::equal(suffix, std::span(prompt.tokens).last(suffix.size())))
    throw std::logic_error("Qwen stable prefix is not a token boundary");
  prompt.stable_prefix_tokens = prompt.tokens.size() - suffix.size();
  prompt.rope.Validate(max_context);
  if (!prompt.images.empty()) {
    const auto digest = identity.Finish();
    prompt.cache_identity.assign(digest.begin(), digest.end());
  }
  return prompt;
}
}  // namespace gufo::models::qwen::vision

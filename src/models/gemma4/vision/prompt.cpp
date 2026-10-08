#include "src/models/gemma4/vision/prompt.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "src/core/crypto/sha256.hpp"

namespace gufo::models::gemma4::vision {
namespace {

void HashU32(crypto::Sha256Hasher& hash, std::uint32_t value) {
  const std::array<std::uint8_t, 4> bytes{
      static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8),
      static_cast<std::uint8_t>(value >> 16),
      static_cast<std::uint8_t>(value >> 24)};
  hash.Update(bytes);
}

void HashText(crypto::Sha256Hasher& hash, std::string_view text) {
  hash.Update(
      {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
}

}  // namespace

ImageSize ResizedSize(std::uint32_t width, std::uint32_t height,
                      std::uint32_t soft_tokens) {
  if (width == 0 || height == 0 || soft_tokens == 0) {
    throw std::invalid_argument("invalid image size or token budget");
  }
  // transformers get_aspect_ratio_preserving_size, in the same double math.
  const double max_patches =
      static_cast<double>(soft_tokens) * kPoolSize * kPoolSize;
  const double target_pixels = max_patches * kPatchSize * kPatchSize;
  const double factor =
      std::sqrt(target_pixels / (static_cast<double>(height) * width));
  const double side = kSideMultiple;
  auto target_height =
      static_cast<std::uint64_t>(std::floor(factor * height / side)) *
      kSideMultiple;
  auto target_width =
      static_cast<std::uint64_t>(std::floor(factor * width / side)) *
      kSideMultiple;
  if (target_height == 0 && target_width == 0) {
    throw std::invalid_argument("image resizes to 0x0");
  }
  const std::uint64_t max_side = std::uint64_t{soft_tokens} * kSideMultiple;
  if (target_height == 0) {
    target_height = kSideMultiple;
    target_width = std::min<std::uint64_t>(
        std::uint64_t{width / height} * kSideMultiple, max_side);
  } else if (target_width == 0) {
    target_width = kSideMultiple;
    target_height = std::min<std::uint64_t>(
        std::uint64_t{height / width} * kSideMultiple, max_side);
  }
  if (static_cast<double>(target_height * target_width) > target_pixels) {
    throw std::invalid_argument("image resize exceeds its patch budget");
  }
  return {static_cast<std::uint32_t>(target_width),
          static_cast<std::uint32_t>(target_height)};
}

core::Image ResizeImage(const core::Image& image, std::uint32_t soft_tokens) {
  const auto size = ResizedSize(image.width, image.height, soft_tokens);
  return core::ResizeBicubic(image, size.width, size.height);
}

ImageTokens FindImageTokens(const Tokenizer& tokenizer) {
  const auto start = tokenizer.FindToken("<|image>");
  const auto soft = tokenizer.FindToken(kImagePlaceholder);
  const auto end = tokenizer.FindToken("<image|>");
  if (!start || !soft || !end) {
    throw std::invalid_argument("gemma4 vocabulary lacks image tokens");
  }
  return {*start, *soft, *end};
}

std::span<const std::uint8_t> Prompt::IdentityForPrefix(
    std::size_t token_count) const {
  for (auto image = images.rbegin(); image != images.rend(); ++image) {
    if (image->span.offset < token_count) {
      return image->prefix_identity;
    }
  }
  return {};
}

Prompt Prepare(const Tokenizer& tokenizer,
               std::span<const tokenization::ChatMessage> messages,
               std::span<const tokenization::ChatTool> tools,
               const ChatOptions& options, std::string_view encoder_identity,
               std::uint32_t max_context, std::uint32_t soft_tokens) {
  if (!IsSoftTokenBudget(soft_tokens)) {
    throw std::invalid_argument(
        "Gemma 4 image budget must be 70, 140, 280, 560 or 1120 tokens");
  }
  const bool has_images = std::ranges::any_of(
      messages, [](const auto& m) { return !m.images.empty(); });
  if (has_images && encoder_identity.empty()) {
    throw std::invalid_argument(
        "image input requires the model's matching --mmproj sidecar");
  }
  std::vector<std::size_t> offsets;
  std::string error;
  const auto rendered =
      ChatTemplate::Render(messages, tools, options, &error, &offsets);
  if (!rendered) {
    throw std::invalid_argument("Gemma 4 chat template: " + error);
  }
  const std::string_view text = rendered->text;
  Prompt prompt;
  const auto append = [&](std::string_view piece) {
    const auto tokens = tokenizer.Encode(piece, false, true);
    if (tokens.size() > max_context - prompt.tokens.size()) {
      throw std::length_error(
          "prompt exceeds the " + std::to_string(max_context) +
          "-token context; increase --context or shorten the conversation");
    }
    prompt.tokens.insert(prompt.tokens.end(), tokens.begin(), tokens.end());
  };

  crypto::Sha256Hasher identity;
  HashText(identity, "gemma4-image-bicubic-rgb8-bf16-v1");
  HashText(identity, encoder_identity);
  HashU32(identity, soft_tokens);
  const ImageTokens image_tokens =
      has_images ? FindImageTokens(tokenizer) : ImageTokens{};
  std::size_t cursor = 0;
  std::size_t index = 0;
  for (const auto& message : messages) {
    for (const auto& image : message.images) {
      if (index >= offsets.size()) {
        throw std::logic_error("image rendering lost a part");
      }
      append(text.substr(cursor, offsets[index] - cursor));
      auto pixels = ResizeImage(core::DecodeImage(*image.bytes), soft_tokens);
      const std::uint32_t rows =
          (pixels.width / kSideMultiple) * (pixels.height / kSideMultiple);
      if (std::size_t{rows} + 2 > max_context - prompt.tokens.size()) {
        throw std::length_error("image tokens exceed model context");
      }
      prompt.tokens.push_back(image_tokens.start);
      const ImageSpan span{static_cast<std::uint32_t>(prompt.tokens.size()),
                           rows};
      prompt.tokens.insert(prompt.tokens.end(), rows, image_tokens.soft);
      prompt.tokens.push_back(image_tokens.end);
      HashU32(identity, span.offset);
      HashU32(identity, pixels.width);
      HashU32(identity, pixels.height);
      identity.Update(pixels.pixels);
      prompt.images.push_back({std::move(pixels), span, identity.Digest()});
      cursor = offsets[index++] + kImagePlaceholder.size();
    }
  }
  if (index != offsets.size()) {
    throw std::logic_error("image rendering added a part");
  }
  append(text.substr(cursor));

  // The stable prefix ends before a special token, after every image.
  if (rendered->stable_prefix_offset < text.size()) {
    const auto suffix = tokenizer.Encode(
        text.substr(rendered->stable_prefix_offset), false, true);
    if (suffix.size() < prompt.tokens.size() &&
        std::ranges::equal(suffix,
                           std::span(prompt.tokens).last(suffix.size()))) {
      prompt.stable_prefix_tokens = prompt.tokens.size() - suffix.size();
    }
  }
  if (!prompt.images.empty()) {
    const auto digest = identity.Finish();
    prompt.cache_identity.assign(digest.begin(), digest.end());
  }
  return prompt;
}

}  // namespace gufo::models::gemma4::vision

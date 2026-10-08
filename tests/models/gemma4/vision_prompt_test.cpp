#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/gemma4/vision/prompt.hpp"
#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
namespace vision = gufo::models::gemma4::vision;
using gemma4_test::Require;
using gufo::tokenization::ChatMessage;
using gufo::tokenization::ChatRole;
using TokenType = g4::TokenType;

namespace {

/// Expected sizes come from transformers' get_aspect_ratio_preserving_size.
void CheckSizing() {
  struct Case {
    std::uint32_t width, height, budget, resized_width, resized_height;
  };
  for (const Case& c : std::array<Case, 12>{{
           {64, 64, 280, 768, 768},
           {1920, 1080, 280, 1056, 576},
           {1080, 1920, 280, 576, 1056},
           {10000, 10, 280, 13440, 48},
           {10, 5000, 280, 48, 13440},
           {48, 48, 70, 384, 384},
           {4032, 3024, 1120, 1824, 1344},
           {100, 1, 280, 8016, 48},
           {1, 1, 280, 768, 768},
           {333, 777, 140, 336, 864},
           {768, 768, 280, 768, 768},
           {3000, 2, 560, 26880, 48},
       }}) {
    const auto size = vision::ResizedSize(c.width, c.height, c.budget);
    Require(size == vision::ImageSize{c.resized_width, c.resized_height},
            "resize of " + std::to_string(c.width) + "x" +
                std::to_string(c.height) + " at " + std::to_string(c.budget) +
                " gave " + std::to_string(size.width) + "x" +
                std::to_string(size.height));
    Require((size.width / 48) * (size.height / 48) <= c.budget,
            "resize exceeds its token budget");
  }
  const gufo::core::Image flat{100, 60,
                               std::vector<std::uint8_t>(100 * 60 * 3, 77)};
  const auto resized = vision::ResizeImage(flat);
  Require(resized.width % 48 == 0 && resized.height % 48 == 0,
          "resized sides are not multiples of 48");
  Require(std::ranges::all_of(resized.pixels, [](auto v) { return v == 77; }),
          "bicubic resize changed a flat image");
}

std::unique_ptr<g4::Tokenizer> BuildTokenizer() {
  g4::Tokenizer::Vocabulary v;
  const auto add = [&](std::string text, TokenType type) {
    v.tokens.push_back(std::move(text));
    v.types.push_back(type);
  };
  add("<pad>", TokenType::kControl);
  add("<eos>", TokenType::kControl);
  add("<bos>", TokenType::kControl);
  add("<unk>", TokenType::kUnknown);
  for (const char* special : {"<|turn>", "<turn|>", "<|image>", "<|image|>",
                              "<image|>", "<|channel>", "<channel|>"}) {
    add(special, TokenType::kControl);
  }
  for (int b = 0; b < 256; ++b) {
    char text[8];
    std::snprintf(text, sizeof(text), "<0x%02X>", b);
    add(text, TokenType::kByte);
  }
  v.bos = 2;
  v.eos = 1;
  v.pad = 0;
  v.unknown = 3;
  std::string error;
  auto tokenizer = g4::Tokenizer::Create(std::move(v), &error);
  Require(tokenizer != nullptr, "synthetic vocabulary rejected: " + error);
  return tokenizer;
}

// A 64x64 PNG; the processor enlarges it to 768x768, 256 soft tokens.
std::shared_ptr<const std::vector<std::uint8_t>> Png() {
  return std::make_shared<const std::vector<std::uint8_t>>(
      gufo::core::ReadImageUrl(
          "data:image/png;base64,"
          "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAYklEQVR4nO3PMQ0AIADA"
          "MEAD/jUiAREcDcmqYJtn7/GzpQNeNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWg"
          "NaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaBdCLsBmEpLi1UAAAAASUVORK5C"
          "YII="));
}

void CheckPrompt() {
  const auto tokenizer = BuildTokenizer();
  const auto tokens = vision::FindImageTokens(*tokenizer);
  const auto png = Png();
  std::vector<ChatMessage> messages{{ChatRole::kUser, "ab"}};
  messages[0].images.push_back({1, png});
  messages[0].images.push_back({2, png});
  const auto prompt =
      vision::Prepare(*tokenizer, messages, {}, {}, "encoder", 4096);
  Require(prompt.images.size() == 2, "two images expected");
  for (const auto& image : prompt.images) {
    Require(image.pixels.width == 768 && image.pixels.height == 768,
            "64x64 image not resized to 768x768");
    Require(image.span.rows == 256, "768x768 image is 256 soft tokens");
    Require(
        prompt.tokens[image.span.offset - 1] == tokens.start &&
            prompt.tokens[image.span.offset + image.span.rows] == tokens.end,
        "soft tokens are not framed by <|image> and <image|>");
    Require(
        std::all_of(prompt.tokens.begin() + image.span.offset,
                    prompt.tokens.begin() + image.span.offset + image.span.rows,
                    [&](auto token) { return token == tokens.soft; }),
        "image span holds other tokens");
  }
  // "a" (byte), image, "b", image, then the turn close.
  const auto& first = prompt.images[0].span;
  const auto& second = prompt.images[1].span;
  Require(second.offset == first.offset + first.rows + 3,
          "one text token separates the images");
  Require(prompt.stable_prefix_tokens > second.offset + second.rows &&
              prompt.stable_prefix_tokens < prompt.tokens.size(),
          "stable prefix does not cover the images");
  Require(!prompt.cache_identity.empty(), "image prompt has no identity");
  Require(prompt.IdentityForPrefix(first.offset).empty() &&
              std::ranges::equal(prompt.IdentityForPrefix(first.offset + 1),
                                 prompt.images[0].prefix_identity) &&
              std::ranges::equal(prompt.IdentityForPrefix(prompt.tokens.size()),
                                 prompt.images[1].prefix_identity),
          "prefix identities do not follow the images");
  Require(prompt.images[0].prefix_identity != prompt.images[1].prefix_identity,
          "second image does not extend the identity");

  // Identity changes with the encoder; tokens do not.
  const auto other =
      vision::Prepare(*tokenizer, messages, {}, {}, "other", 4096);
  Require(other.tokens == prompt.tokens &&
              other.cache_identity != prompt.cache_identity,
          "encoder identity not part of the cache identity");

  bool rejected = false;
  try {
    (void)vision::Prepare(*tokenizer, messages, {}, {}, "", 4096);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "image accepted without an encoder");
  rejected = false;
  try {
    (void)vision::Prepare(*tokenizer, messages, {}, {}, "encoder", 300);
  } catch (const std::length_error&) {
    rejected = true;
  }
  Require(rejected, "image beyond the context accepted");

  // The largest budget keeps more detail; others are rejected.
  const auto large =
      vision::Prepare(*tokenizer, messages, {}, {}, "encoder", 4096, 1120);
  Require(large.images[0].span.rows == 1089 &&
              large.images[0].pixels.width == 1584 &&
              large.cache_identity != prompt.cache_identity,
          "a 1120-token budget must give 33x33 soft tokens");
  rejected = false;
  try {
    (void)vision::Prepare(*tokenizer, messages, {}, {}, "encoder", 4096, 100);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "unsupported image budget accepted");

  const auto text = vision::Prepare(
      *tokenizer, std::vector<ChatMessage>{{ChatRole::kUser, "ab"}}, {}, {}, "",
      4096);
  Require(text.images.empty() && text.cache_identity.empty(),
          "text prompt carries image state");
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckSizing();
    CheckPrompt();
  });
}

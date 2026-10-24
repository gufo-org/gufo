// Gemma 4 image prompts for model tests: a generated PNG and its prepared
// prompt, spans and embeddings.
#ifndef GUFO_TESTS_MODELS_GEMMA4_IMAGE_FIXTURE_HPP_
#define GUFO_TESTS_MODELS_GEMMA4_IMAGE_FIXTURE_HPP_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/models/gemma4/engine.hpp"
#include "src/models/gemma4/vision/encoder.hpp"
#include "src/models/gemma4/vision/prompt.hpp"

namespace gemma4_test {
namespace g4 = gufo::models::gemma4;
namespace vision = gufo::models::gemma4::vision;

/// The prompt builder decodes PNG/JPEG, so the fixture writes a small
/// stored-deflate PNG of a nearly solid colour.
inline std::vector<std::uint8_t> Png(std::uint32_t seed) {
  constexpr std::uint32_t kSide = 96;
  std::vector<std::uint8_t> raw;
  for (std::uint32_t y = 0; y < kSide; ++y) {
    raw.push_back(0);  // filter: none
    for (std::uint32_t x = 0; x < kSide; ++x) {
      // Seed 1 is red, seed 2 blue, with a faint texture.
      const auto texture = static_cast<std::uint8_t>((x * 7 + y * 3) % 16);
      raw.push_back(static_cast<std::uint8_t>(seed == 1 ? 220 : 20) + texture);
      raw.push_back(static_cast<std::uint8_t>(30 + texture));
      raw.push_back(static_cast<std::uint8_t>(seed == 1 ? 20 : 220) + texture);
    }
  }
  const auto crc = [](const std::uint8_t* data, std::size_t size) {
    std::uint32_t c = 0xffffffffU;
    for (std::size_t i = 0; i < size; ++i) {
      c ^= data[i];
      for (int k = 0; k < 8; ++k)
        c = (c >> 1) ^ (0xedb88320U & (0U - (c & 1U)));
    }
    return c ^ 0xffffffffU;
  };
  std::vector<std::uint8_t> png = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
  const auto be32 = [](std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int s = 24; s >= 0; s -= 8)
      out.push_back(static_cast<std::uint8_t>(v >> s));
  };
  const auto chunk = [&](const char* type,
                         const std::vector<std::uint8_t>& body) {
    be32(png, static_cast<std::uint32_t>(body.size()));
    std::vector<std::uint8_t> typed(type, type + 4);
    typed.insert(typed.end(), body.begin(), body.end());
    png.insert(png.end(), typed.begin(), typed.end());
    be32(png, crc(typed.data(), typed.size()));
  };
  std::vector<std::uint8_t> header;
  be32(header, kSide);
  be32(header, kSide);
  header.insert(header.end(), {8, 2, 0, 0, 0});
  chunk("IHDR", header);
  // zlib stream of stored deflate blocks with an Adler-32 trailer.
  std::vector<std::uint8_t> z = {0x78, 0x01};
  for (std::size_t at = 0; at < raw.size(); at += 65535) {
    const auto size = static_cast<std::uint16_t>(
        std::min<std::size_t>(65535, raw.size() - at));
    z.push_back(at + size == raw.size() ? 1 : 0);
    z.push_back(static_cast<std::uint8_t>(size));
    z.push_back(static_cast<std::uint8_t>(size >> 8));
    z.push_back(static_cast<std::uint8_t>(~size));
    z.push_back(static_cast<std::uint8_t>(~size >> 8));
    z.insert(z.end(), raw.begin() + at, raw.begin() + at + size);
  }
  std::uint32_t a = 1, b = 0;
  for (const auto byte : raw) {
    a = (a + byte) % 65521;
    b = (b + a) % 65521;
  }
  be32(z, (b << 16) | a);
  chunk("IDAT", z);
  chunk("IEND", {});
  return png;
}

struct Prepared {
  vision::Prompt prompt;
  std::vector<g4::ImageSpan> spans;
  std::vector<std::shared_ptr<const vision::Encoder::Embedding>> embeddings;
  g4::ImageEmbeddings embed() const {
    return [this](std::size_t i) { return embeddings.at(i)->data(); };
  }
};

inline Prepared Prepare(const g4::Model& model, vision::Encoder& encoder,
                        std::uint32_t seed, const std::string& question,
                        std::uint32_t context, std::uint32_t soft_tokens) {
  std::vector<gufo::tokenization::ChatMessage> messages(1);
  messages[0].content =
      "Here is a picture that I generated for a small experiment about "
      "colour perception in synthetic images: " +
      question;
  messages[0].images.push_back(
      {messages[0].content.find(':') + 1,
       std::make_shared<const std::vector<std::uint8_t>>(Png(seed))});
  Prepared p;
  p.prompt = vision::Prepare(model.tokenizer(), messages, {}, {},
                             encoder.identity(), context, soft_tokens);
  for (const auto& image : p.prompt.images) {
    p.embeddings.push_back(encoder.Encode(image.pixels));
    p.spans.push_back(
        {image.span.offset, image.span.rows, image.prefix_identity});
  }
  return p;
}

}  // namespace gemma4_test

#endif

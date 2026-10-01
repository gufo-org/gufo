#include <arpa/inet.h>
#include <webp/encode.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "src/models/qwen/vision/prompt.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/vision/encoder.hpp"
#endif

namespace {
using namespace gufo::models::qwen::vision;

void TestPositionLayout() {
  const RopeLayout layout{{{5, 2, 3}, {15, 3, 2}}};
  layout.Validate(64);
  assert((layout.Position(4) == std::array<std::int32_t, 3>{4, 4, 4}));
  assert((layout.Position(5) == std::array<std::int32_t, 3>{5, 5, 5}));
  assert((layout.Position(10) == std::array<std::int32_t, 3>{5, 6, 7}));
  assert((layout.Position(11) == std::array<std::int32_t, 3>{8, 8, 8}));
  assert((layout.Position(15) == std::array<std::int32_t, 3>{12, 12, 12}));
  assert((layout.Position(20) == std::array<std::int32_t, 3>{12, 14, 13}));
  assert((layout.Position(21) == std::array<std::int32_t, 3>{15, 15, 15}));
  assert(layout.Delta() == -6);
  assert(layout.PrefixLength() == 21);
  assert(layout.Prefix(5).images.empty());
  assert(layout.Prefix(6).images.size() == 1);
  assert(layout.Prefix(15).images.size() == 1);
  assert(layout.Prefix(16) == layout);
  for (std::uint32_t count = 0; count < 32; ++count) {
    const auto prefix = layout.Prefix(count);
    for (std::uint32_t token = 0; token < count; ++token)
      assert(prefix.Position(token) == layout.Position(token));
  }
  Prompt prompt;
  PreparedImage first, second;
  first.grid = layout.images[0];
  second.grid = layout.images[1];
  first.prefix_identity.fill(1);
  second.prefix_identity.fill(2);
  prompt.images = {first, second};
  assert(prompt.IdentityForPrefix(5).empty());
  assert(
      std::ranges::equal(prompt.IdentityForPrefix(6), first.prefix_identity));
  assert(
      std::ranges::equal(prompt.IdentityForPrefix(15), first.prefix_identity));
  assert(
      std::ranges::equal(prompt.IdentityForPrefix(16), second.prefix_identity));
  assert(
      (RopeLayout{}.Position(31) == std::array<std::int32_t, 3>{31, 31, 31}));
  bool rejected = false;
  try {
    RopeLayout{{{5, 2, 3}, {8, 2, 2}}}.Validate(64);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void TestPreprocessing() {
  gufo::core::Image input{48, 32, std::vector<std::uint8_t>(48 * 32 * 3, 123)};
  const auto result = ResizeImage(input);
  assert(result.width == 320 && result.height == 224);
  assert(std::all_of(result.pixels.begin(), result.pixels.end(),
                     [](auto value) { return value == 123; }));
  input = {256, 256, std::vector<std::uint8_t>(256 * 256 * 3, 42)};
  assert(ResizeImage(input).pixels == input.pixels);
  const std::vector<std::uint8_t> payload{0xff, 0xd8, 0, 1, 2};
  for (const auto* accepted :
       {"data:image/jpeg;base64,/9gAAQI=", "data:image/jpg;base64,/9gAAQI=",
        "data:IMAGE/JPEG;BASE64,/9gAAQI=",
        "data:image/jpeg;name=photo.jpg;base64,/9gAAQI="})
    assert(gufo::core::ReadImageUrl(accepted) == payload);
  assert(gufo::core::ReadImageUrl("data:image/webp;base64,/9gAAQI=") ==
         payload);
  for (const auto* invalid :
       {"data:image/png;base64,A===", "data:image/png;base64,AB==",
        "data:image/png;base64,AAAA=", "file:///tmp/image.png",
        "data:image/gif;base64,/9gAAQI=", "data:image/png,/9gAAQI=",
        "data:image/png;base64"}) {
    bool rejected = false;
    try {
      (void)gufo::core::ReadImageUrl(invalid);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
}

// Clients such as chat front ends often re-encode uploads as WebP.
void TestWebpDecoding() {
  const std::vector<std::uint8_t> rgb{255, 0,  0,  0,  255, 0,  0,  0,  255,
                                      10,  20, 30, 40, 50,  60, 70, 80, 90};
  std::uint8_t* encoded = nullptr;
  const auto size = WebPEncodeLosslessRGB(rgb.data(), 3, 2, 9, &encoded);
  assert(size > 0);
  const std::vector<std::uint8_t> bytes(encoded, encoded + size);
  WebPFree(encoded);
  const auto image = gufo::core::DecodeImage(bytes);
  assert(image.width == 3 && image.height == 2 && image.pixels == rgb);

  // Dropping alpha must retain RGB, not blend it onto a background.
  std::vector<std::uint8_t> rgba;
  for (std::size_t i = 0; i < rgb.size(); i += 3) {
    rgba.insert(rgba.end(), rgb.begin() + i, rgb.begin() + i + 3);
    rgba.push_back(128);
  }
  const auto alpha_size =
      WebPEncodeLosslessRGBA(rgba.data(), 3, 2, 12, &encoded);
  assert(alpha_size > 0);
  const std::vector<std::uint8_t> alpha_bytes(encoded, encoded + alpha_size);
  WebPFree(encoded);
  assert(gufo::core::DecodeImage(alpha_bytes).pixels == rgb);

  const auto rejects = [](const std::vector<std::uint8_t>& input,
                          std::string_view message) {
    bool rejected = false;
    try {
      (void)gufo::core::DecodeImage(input);
    } catch (const std::invalid_argument& error) {
      rejected = std::string_view(error.what()).find(message) !=
                 std::string_view::npos;
    }
    assert(rejected);
  };
  auto truncated = bytes;
  truncated.resize(24);
  rejects(truncated, "invalid WebP");
  // VP8L's packed dimensions: 8192x8192 exceeds the decoded pixel budget.
  auto oversized = bytes;
  assert(oversized[20] == 0x2f);
  const std::uint32_t dimensions = 8191U | (8191U << 14);
  for (std::size_t i = 0; i < 4; ++i)
    oversized[21 + i] = static_cast<std::uint8_t>(dimensions >> (8 * i));
  rejects(oversized, "decoded pixel limit");
}

void TestImageTransportLimits() {
  for (const auto* address :
       {"0.0.0.0", "10.1.2.3", "100.64.0.1", "127.0.0.1", "169.254.169.254",
        "172.16.1.2", "192.168.0.1", "198.18.0.1", "224.0.0.1", "::1",
        "::ffff:127.0.0.1", "64:ff9b::a00:1", "fc00::1", "fe80::1",
        "2002:7f00:1::", "2001::1", "2001:db8::1", "3fff::1"}) {
    std::array<std::uint8_t, 16> bytes{};
    const bool v4 =
        std::string_view(address).find(':') == std::string_view::npos;
    assert(inet_pton(v4 ? AF_INET : AF_INET6, address, bytes.data()) == 1);
    assert(!gufo::core::IsPublicImageAddress({bytes.data(), v4 ? 4U : 16U}));
  }
  for (const auto* address : {"1.1.1.1", "8.8.8.8", "2001:4860:4860::8888"}) {
    std::array<std::uint8_t, 16> bytes{};
    const bool v4 =
        std::string_view(address).find(':') == std::string_view::npos;
    assert(inet_pton(v4 ? AF_INET : AF_INET6, address, bytes.data()) == 1);
    assert(gufo::core::IsPublicImageAddress({bytes.data(), v4 ? 4U : 16U}));
  }
  const auto rejects = [](std::string_view url,
                          gufo::core::ImageReadBudget& budget) {
    bool failed = false;
    try {
      (void)gufo::core::ReadImageUrl(url, budget);
    } catch (const std::invalid_argument&) {
      failed = true;
    }
    assert(failed);
  };
  gufo::core::ImageReadBudget budget;
  for (const auto* url :
       {"https://127.0.0.1:1/image.png", "https://[::1]:1/image.png",
        "https://2130706433:1/image.png", "https://localhost:1/image.png"})
    rejects(url, budget);
  budget = {};
  budget.remaining_bytes = 3;
  assert(
      gufo::core::ReadImageUrl("data:image/png;base64,AQID", budget).size() ==
      3);
  rejects("data:image/png;base64,AQID", budget);
  budget = {};
  budget.deadline = std::chrono::steady_clock::now();
  rejects("data:image/png;base64,AQID", budget);
  budget = {};
  budget.remaining_images = 0;
  rejects("data:image/png;base64,AQID", budget);
}

void TestRendering() {
  using namespace gufo::tokenization;
  ChatMessage message{ChatRole::kUser, "beforeafter"};
  message.images.push_back(
      {6, std::make_shared<const std::vector<std::uint8_t>>(1, 0)});
  message.images.push_back({6, message.images.front().bytes});
  std::vector<std::size_t> offsets;
  const std::array messages{message};
  const auto text =
      QwenChatTemplate::Render(messages, {}, {}, nullptr, &offsets);
  assert(text && offsets.size() == 2);
  assert(text->find("before<|vision_start|><|image_pad|><|vision_end|>"
                    "<|vision_start|><|image_pad|><|vision_end|>after") !=
         std::string::npos);
  for (auto offset : offsets)
    assert(text->substr(offset, 13) == "<|image_pad|>");
}

void TestToolReasoningCheckpoint() {
  using namespace gufo::tokenization;
  std::vector<std::string> vocab;
  for (int i = 0; i < 256; ++i)
    vocab.emplace_back(1, static_cast<char>(i));
  std::unordered_map<std::string, TokenId> specials;
  for (const auto* token : {"<|im_start|>", "<|im_end|>", "<think>", "</think>",
                            "<|vision_start|>", "<|vision_end|>"}) {
    specials[token] = static_cast<TokenId>(vocab.size());
    vocab.emplace_back(token);
  }
  const auto tokenizer =
      QwenTokenizer::CreateFromVocabulary(vocab, {}, specials);
  assert(tokenizer);
  const auto pixels = std::make_shared<
      const std::vector<std::uint8_t>>(gufo::core::ReadImageUrl(
      "data:image/png;base64,"
      "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAYklEQVR4nO3PMQ0AIADAMEAD"
      "/jUiAREcDcmqYJtn7/GzpQNeNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWg"
      "NaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaBdCLsBmEpLi1UAAAAASUVORK5CYII="));
  for (const bool images : {false, true}) {
    for (const bool thinking : {false, true}) {
      for (const bool preserve : {false, true}) {
        for (const auto* thought : {"", "Read the fixture."}) {
          std::vector<ChatMessage> messages{
              {ChatRole::kUser, "Read the fixture."},
              {ChatRole::kAssistant, "", "", thought},
              {ChatRole::kTool, "The fixture is ready."}};
          messages[1].tool_calls.push_back({"call", "read_fixture", {}});
          if (images)
            messages[0].images.push_back({0, pixels});
          ChatTemplateOptions options;
          options.enable_thinking = thinking;
          options.preserve_thinking = preserve;
          const auto prompt =
              Prepare(*tokenizer, messages, {}, options, "fixture", 4096);
          messages.emplace_back(ChatRole::kUser, ".");
          const auto continued =
              Prepare(*tokenizer, messages, {}, options, "fixture", 4096);
          const auto stable =
              std::span(prompt.tokens).first(prompt.stable_prefix_tokens);
          assert(std::ranges::equal(
              stable, std::span(continued.tokens).first(stable.size())));
          if (images)
            assert(prompt.stable_prefix_tokens >= prompt.rope.PrefixLength());
          std::string error;
          std::size_t expected_bytes = 0;
          messages.pop_back();
          const auto rendered = QwenChatTemplate::Render(
              messages, {}, options, &error, nullptr, &expected_bytes);
          assert(rendered);
          if (!preserve) {
            const auto first_assistant =
                rendered->find("<|im_start|>assistant\n");
            assert(expected_bytes == first_assistant);
            assert(prompt.stable_prefix_tokens <
                   continued.stable_prefix_tokens);
          } else {
            assert(expected_bytes ==
                   rendered->size() - GenerationPrompt(thinking).size());
          }
        }
      }
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    TestPositionLayout();
    TestPreprocessing();
    TestRendering();
    TestToolReasoningCheckpoint();
    TestImageTransportLimits();
    TestWebpDecoding();
    if (argc == 1) {
      std::cout << "vision input, layout and rendering: passed\n";
      return 0;
    }
    const bool preprocess =
        argc == 4 && std::string_view(argv[1]) == "--preprocess";
    if (!preprocess && argc != 5)
      throw std::invalid_argument(
          "usage: qwen27b_vision_test MMPROJ IMAGE OUTPUT_DIR OUTPUT_WIDTH\n"
          "       qwen27b_vision_test --preprocess IMAGE OUTPUT_DIR");
    const auto image = ResizeImage(
        gufo::core::DecodeImage(gufo::core::ReadImageFile(argv[2])));
    const std::filesystem::path directory(argv[3]);
    std::filesystem::create_directories(directory);
    {
      std::ofstream rgb(directory / "resized.rgb", std::ios::binary);
      rgb.write(reinterpret_cast<const char*>(image.pixels.data()),
                image.pixels.size());
      std::ofstream shape(directory / "shape.txt");
      shape << image.width << ' ' << image.height << '\n';
    }
    if (preprocess)
      return 0;
#if defined(ENGINE_ENABLE_HIP)
    Encoder encoder(argv[1], std::stoul(argv[4]));
    {
      std::ofstream identity(directory / "projector.sha256");
      identity << encoder.identity() << '\n';
    }
    const auto result = encoder.Encode(
        image, [&](std::string_view stage, std::span<const float> values) {
          std::ofstream output(directory / (std::string(stage) + ".f32"),
                               std::ios::binary);
          output.write(reinterpret_cast<const char*>(values.data()),
                       values.size_bytes());
          if (!output)
            throw std::runtime_error("cannot write vision comparison tensor");
        });
    std::cout << "encoded " << image.width << 'x' << image.height << " -> "
              << result->rows() << 'x' << result->width() << '\n';
    return 0;
#else
    throw std::runtime_error("vision encoding requires HIP");
#endif
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

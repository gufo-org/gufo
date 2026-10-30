#include <iostream>
#include <stdexcept>

#include "src/models/lfm3/chat_template.hpp"
#include "src/models/lfm3_vl/vision.hpp"
namespace vl = gufo::models::lfm3_vl;
namespace l = gufo::models::lfm3;
using gufo::tokenization::ChatMessage;
using gufo::tokenization::ChatRole;
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
int main() {
  try {
    // Fixed bytes from LiquidAI's pinned Jinja template, independently of
    // production control constants. Equal offsets preserve image order.
    ChatMessage user(ChatRole::kUser, "AB");
    user.images = {{1, {}}, {1, {}}};
    const std::vector<l::ImageExpansion> images{{"<|image|>", 2},
                                                {"<|image|>", 1}};
    gufo::ReasoningOptions reasoning;
    reasoning.enabled = false;
    const std::vector<ChatMessage> messages{user};
    Require(l::RenderChat(messages, {}, reasoning, true, images) ==
                "<|startoftext|><|im_start|>user\nA<|vision_start|><|image|><|"
                "image|><|vision_end|>"
                "<|vision_start|><|image|><|vision_end|>B<|im_end|>\n<|im_"
                "start|>assistant\n<think></think>",
            "VL template differs from pinned Jinja bytes");
    bool rejected = false;
    try {
      (void)l::RenderChat(messages);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Require(rejected,
            "plain text runner admitted image markers without a projector");
    auto bad = messages;
    bad[0].images[0].offset = 3;
    rejected = false;
    try {
      (void)l::RenderChat(bad, {}, reasoning, true, images);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Require(rejected, "invalid image placement admitted");
    rejected = false;
    try {
      (void)l::RenderChat(messages, {}, reasoning, true,
                          std::span(images).first(1));
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Require(rejected, "image expansion count mismatch admitted");
    gufo::core::Image image{2048, 1024,
                            std::vector<std::uint8_t>(2048 * 1024 * 3, 255)};
    const auto native = vl::ProcessImage(image);
    Require(native.width == 128 && native.height == 64,
            "native VL resize used d1 pixel cap");
    const auto small = vl::ProcessImage(image, 131072);
    Require(small.width == 32 && small.height == 16,
            "low-detail pixel budget differs");
    Require(small.patches.front() == 1.0F && small.patches.back() == 1.0F,
            "white normalization differs");
    rejected = false;
    try {
      (void)vl::ProcessImage(image, 1);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Require(rejected, "invalid pixel budget admitted");
    std::cout
        << "LFM3-VL template, placement and preprocessing contracts passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

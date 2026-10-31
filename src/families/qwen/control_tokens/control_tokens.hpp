#ifndef GUFO_TOKENIZATION_QWEN_CONTROL_TOKENS_HPP_
#define GUFO_TOKENIZATION_QWEN_CONTROL_TOKENS_HPP_

#include <algorithm>
#include <array>
#include <string_view>

namespace gufo::tokenization {

/// Shared Qwen framing, also used by Qwen-Image, audio models and MiniMax H3.
/// Other token families remain with their models. See AGENTS.md for guidance.
inline constexpr std::string_view kEndOfText = "<|endoftext|>";
inline constexpr std::string_view kImStart = "<|im_start|>";
inline constexpr std::string_view kImEnd = "<|im_end|>";
inline constexpr std::string_view kObjectRefStart = "<|object_ref_start|>";
inline constexpr std::string_view kObjectRefEnd = "<|object_ref_end|>";
inline constexpr std::string_view kBoxStart = "<|box_start|>";
inline constexpr std::string_view kBoxEnd = "<|box_end|>";
inline constexpr std::string_view kQuadStart = "<|quad_start|>";
inline constexpr std::string_view kQuadEnd = "<|quad_end|>";
inline constexpr std::string_view kVisionStart = "<|vision_start|>";
inline constexpr std::string_view kVisionEnd = "<|vision_end|>";
inline constexpr std::string_view kVisionPad = "<|vision_pad|>";
inline constexpr std::string_view kImagePad = "<|image_pad|>";
inline constexpr std::string_view kVideoPad = "<|video_pad|>";

/// Compose fixed framing without allocation or dynamic initialization.
template<const std::string_view&... Parts>
consteval auto ConcatControlText() {
  std::array<char, (Parts.size() + ...)> text{};
  auto output = text.begin();
  ((output = std::copy(Parts.begin(), Parts.end(), output)), ...);
  return text;
}

}  // namespace gufo::tokenization

#endif  // GUFO_TOKENIZATION_QWEN_CONTROL_TOKENS_HPP_

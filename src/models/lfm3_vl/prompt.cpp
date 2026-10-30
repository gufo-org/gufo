#include "src/models/lfm3_vl/prompt.hpp"

#include <algorithm>
#include <stdexcept>
namespace gufo::models::lfm3_vl {
Prompt Prepare(const lfm3::Tokenizer& tokenizer,
               std::span<const tokenization::ChatMessage> messages,
               std::span<const tokenization::ChatTool> tools,
               const ReasoningOptions& reasoning, std::uint32_t context) {
  Prompt prompt;
  std::vector<lfm3::ImageExpansion> expansions;
  std::size_t visual_tokens = 0;
  for (const auto& message : messages)
    for (const auto& image : message.images) {
      if (!image.bytes || image.bytes->empty())
        throw std::invalid_argument("empty LFM3-VL image");
      const auto decoded = core::DecodeImage(*image.bytes);
      const auto low_pixels = unsigned(std::max<std::uint64_t>(
          65536, std::uint64_t(512) * 512 *
                     std::min(decoded.width, decoded.height) /
                     std::max(decoded.width, decoded.height)));
      auto picture = ProcessImage(
          decoded, image.detail == tokenization::ChatMessage::ImageDetail::kLow
                       ? low_pixels
                       : 16777216);
      const auto count = std::size_t(picture.height) * picture.width / 4;
      if (count > context - visual_tokens)
        throw std::length_error("LFM3-VL image tokens exceed model context");
      visual_tokens += count;
      expansions.push_back({std::string(kImageToken), count});
      prompt.images.push_back({0, std::move(picture)});
    }
  prompt.tokens =
      lfm3::EncodeChat(tokenizer, messages, tools, reasoning, true, expansions);
  if (prompt.tokens.size() > context)
    throw std::length_error("LFM3-VL prompt exceeds model context");
  const auto marker = tokenizer.Encode(kImageToken);
  if (marker.size() != 1)
    throw std::invalid_argument("LFM3-VL tokenizer lacks its image token");
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < expansions.size(); ++i) {
    auto begin = std::find(prompt.tokens.begin() + std::ptrdiff_t(cursor),
                           prompt.tokens.end(), marker.front());
    if (begin == prompt.tokens.end())
      throw std::logic_error("LFM3-VL image marker was lost");
    const auto offset = std::size_t(begin - prompt.tokens.begin());
    const auto count = expansions[i].count;
    if (count > prompt.tokens.size() - offset ||
        !std::all_of(begin, begin + std::ptrdiff_t(count),
                     [&](auto id) { return id == marker.front(); }))
      throw std::logic_error("LFM3-VL image marker count mismatch");
    prompt.images[i].offset = offset;
    cursor = offset + count;
  }
  const auto history = lfm3::EncodeChat(tokenizer, messages, tools, reasoning,
                                        false, expansions);
  if (history.size() > prompt.tokens.size() ||
      !std::equal(history.begin(), history.end(), prompt.tokens.begin()))
    throw std::logic_error("LFM3-VL history is not a prompt prefix");
  prompt.stable_prefix = history.size();
  return prompt;
}
}  // namespace gufo::models::lfm3_vl

#ifndef GUFO_MODELS_QWEN3_TTS_TEXT_STREAM_HPP_
#define GUFO_MODELS_QWEN3_TTS_TEXT_STREAM_HPP_

#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::qwen3_tts {

// Boundaries depend on text, not WebSocket packet sizes. Latin punctuation
// waits for whitespace so a decimal point does not force a split.
// This is opt-in: splitting changes the model's available future context.
inline std::vector<std::string> TakeSpeechSegments(std::string* pending,
                                                   std::string_view granularity,
                                                   bool flush) {
  std::vector<std::string> result;
  if (granularity != "none") {
    std::size_t start = 0;
    for (std::size_t i = 0; i < pending->size(); ++i) {
      const char c = (*pending)[i];
      const bool latin =
          c == '.' || c == '!' || c == '?' ||
          (granularity == "clause" && (c == ',' || c == ';' || c == ':'));
      std::size_t end = 0;
      if (c == '\n')
        end = i + 1;
      if (latin && i + 1 < pending->size() &&
          ((*pending)[i + 1] == ' ' || (*pending)[i + 1] == '\n' ||
           (*pending)[i + 1] == '\t'))
        end = i + 1;
      const auto tail = std::string_view(*pending).substr(i);
      for (const auto mark : {std::string_view("。"), std::string_view("！"),
                              std::string_view("？"), std::string_view("।")})
        if (tail.starts_with(mark))
          end = i + mark.size();
      if (granularity == "clause")
        for (const auto mark : {std::string_view("，"), std::string_view("；"),
                                std::string_view("："), std::string_view("،")})
          if (tail.starts_with(mark))
            end = i + mark.size();
      if (end != 0) {
        result.push_back(pending->substr(start, end - start));
        start = end;
        i = end - 1;
      }
    }
    pending->erase(0, start);
  }
  if (flush && !pending->empty()) {
    result.push_back(std::move(*pending));
    pending->clear();
  }
  return result;
}

}  // namespace gufo::models::qwen3_tts
#endif  // GUFO_MODELS_QWEN3_TTS_TEXT_STREAM_HPP_

#ifndef GUFO_MODELS_LFM3_AUDIO_GENERATION_HPP_
#define GUFO_MODELS_LFM3_AUDIO_GENERATION_HPP_
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/lfm3_audio/model.hpp"
namespace gufo::models::lfm3_audio {
// One piece of a prompt: template text (control tokens recognized), client
// text (taken literally) or a 16 kHz mono clip.
struct PromptPiece {
  enum class Kind { kTemplate, kLiteral, kAudio };
  Kind kind{Kind::kTemplate};
  std::string text;
  std::vector<float> audio;
};
struct ChatPart {
  bool audio{};
  std::string text;
  std::vector<float> samples;  // 16 kHz mono when audio
};
struct ChatMessage {
  std::string role;
  std::vector<ChatPart> parts;
};
// Official chat framing: <|startoftext|>, an optional system turn, each
// message as <|im_start|>role\n...<|im_end|>\n, then <|im_start|>assistant\n
// followed by assistant_prefix (for example <|audio_start|>, which starts
// speech, or <|mixed_start|>, which starts interleaved text and speech).
std::vector<PromptPiece> BuildPrompt(std::string_view system,
                                     std::span<const ChatMessage> messages,
                                     std::string_view assistant_prefix = {});
// System prompts of the model card's tasks.
std::string AsrSystemPrompt(std::string_view language = {});
// Language names or codes (en, en-gb, fr, ...) and female or male voices.
bool TtsSystemPrompt(std::string_view language, std::string_view voice,
                     std::string* prompt, std::string* error = nullptr);
inline constexpr std::string_view kInterleavedSystemPrompt =
    "Respond with interleaved text and audio.";
struct GenerateOptions {
  // Generated slots: a text token or a speech frame each.
  std::size_t max_new_tokens{1024};
  // Text sampling; temperature 0 is greedy (the model card default).
  float temperature{0.0F};
  int top_k{0};
  float top_p{1.0F};
  float repetition_penalty{1.0F};
  std::size_t repetition_window{32};
  Voice::Sampling audio;
  // Forbid <|mixed_start|> and <|audio_start|> in text mode.
  bool suppress_mixed{false}, suppress_audio_out{false};
  std::uint64_t seed{0};
  // Frames per codec decode once speech has started; the first chunk is
  // shorter so the first audio leaves sooner.
  std::size_t first_chunk_frames{1}, chunk_frames{4};
};
struct GenerateCallbacks {
  // Newly completed UTF-8 text; returning false cancels.
  std::function<bool(std::string_view)> on_text;
  // Newly decoded 24 kHz mono samples; returning false cancels.
  std::function<bool(std::span<const float>)> on_audio;
  std::function<bool()> cancelled;
};
struct GenerateTimings {
  double encode_ms{}, prefill_ms{}, first_text_ms{-1}, first_audio_ms{-1},
      backbone_ms{}, depth_ms{}, codec_ms{}, total_ms{};
};
struct GenerateResult {
  std::string text;  // generated text without control tokens
  std::vector<std::uint32_t> tokens;
  std::vector<std::int32_t> codes;  // frame-major [frames][12]
  std::vector<float> samples;       // 24 kHz
  std::size_t prompt_tokens{}, audio_tokens{}, frames{};
  bool cancelled{false};
  GenerateTimings timings;
};
class Generator {
public:
  // context bounds the prompt plus generated slots of one Generate call.
  static std::unique_ptr<Generator> Create(std::shared_ptr<Model> model,
                                           std::uint32_t context,
                                           std::string* error = nullptr);
  ~Generator();
  bool Generate(std::span<const PromptPiece> prompt,
                const GenerateOptions& options,
                const GenerateCallbacks& callbacks, GenerateResult* result,
                std::string* error = nullptr);
  Model& model() const;

private:
  Generator();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::models::lfm3_audio
#endif

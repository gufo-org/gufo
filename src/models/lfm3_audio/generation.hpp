#ifndef GUFO_MODELS_LFM3_AUDIO_GENERATION_HPP_
#define GUFO_MODELS_LFM3_AUDIO_GENERATION_HPP_
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
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
  // user, assistant or tool; tool content is wrapped in <tool_response>.
  std::string role;
  std::vector<ChatPart> parts;
};
// Official chat framing: <|startoftext|>, an optional system turn (with the
// tool definitions, one JSON object per line), each message as
// <|im_start|>role\n...<|im_end|>\n, then <|im_start|>assistant\n followed by
// assistant_prefix (for example <|audio_start|>, which starts speech,
// <|mixed_start|>, which starts interleaved text and speech, or <think>).
std::vector<PromptPiece> BuildPrompt(std::string_view system,
                                     std::span<const ChatMessage> messages,
                                     std::string_view assistant_prefix = {},
                                     std::span<const std::string> tools = {});
// Pieces of the system turn alone, with <|startoftext|> first.
std::vector<PromptPiece> SystemPieces(std::string_view system,
                                      std::span<const std::string> tools = {});
// Pieces of one message turn, closed with <|im_end|>\n.
std::vector<PromptPiece> MessagePieces(const ChatMessage& message);
// System prompts of the model card's tasks.
std::string AsrSystemPrompt(std::string_view language = {});
// Language names or codes (en, en-gb, fr, ...) and female or male voices.
bool TtsSystemPrompt(std::string_view language, std::string_view voice,
                     std::string* prompt, std::string* error = nullptr);
inline constexpr std::string_view kInterleavedSystemPrompt =
    "Respond with interleaved text and audio.";
// With tools, the model either calls them or answers: in interleaved text
// and speech, or (kToolsTextSystemPrompt) in text.
inline constexpr std::string_view kToolsInterleavedSystemPrompt =
    "Respond with function calls, and interleaved text and audio.\n\n"
    "List of tools:";
inline constexpr std::string_view kToolsTextSystemPrompt = "List of tools:";
inline constexpr std::string_view kLiveAsrSystemPrompt = "Perform live ASR.";
// A tool definition in the template's spelling: the function object (the
// "function" member of an OpenAI tool, or the tool itself) without
// defer_loading, as Jinja tojson writes it.
std::string ToolDefinition(std::string_view tool_json);

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
  // When set, the first generated token is one of these.
  std::vector<std::uint32_t> first_tokens;
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
  // Generated text without control tokens (<think>, </think> and tool-call
  // tags are ordinary text).
  std::string text;
  std::vector<std::uint32_t> tokens;
  std::vector<std::int32_t> codes;  // frame-major [frames][12]
  std::vector<float> samples;       // 24 kHz
  std::size_t prompt_tokens{}, audio_tokens{}, frames{};
  // The reply ended with <|im_end|> or another stop (not by the slot budget,
  // the context or cancellation).
  bool stopped{false};
  bool cancelled{false};
  GenerateTimings timings;
};
// A backbone context with one voice. Generate runs a prompt from scratch.
// Conversations append turns instead, including user audio as it arrives,
// and Continue the reply from the context, which then holds the reply.
class Generator {
public:
  // context bounds the prompt plus generated slots.
  static std::unique_ptr<Generator> Create(std::shared_ptr<Model> model,
                                           std::uint32_t context,
                                           std::string* error = nullptr);
  ~Generator();
  bool Generate(std::span<const PromptPiece> prompt,
                const GenerateOptions& options,
                const GenerateCallbacks& callbacks, GenerateResult* result,
                std::string* error = nullptr);
  // Empties the context.
  bool Reset(std::string* error = nullptr);
  // Prefills pieces (audio clips are encoded) after the context. Counters
  // and encode time accumulate into result when given.
  bool Append(std::span<const PromptPiece> pieces,
              GenerateResult* result = nullptr, std::string* error = nullptr);
  // Prefills rows of backbone-width embeddings, such as AudioInput output.
  bool AppendEmbeddings(std::span<const float> rows,
                        std::string* error = nullptr);
  // Generates from the context. The mode follows the last appended token:
  // <|audio_start|> speaks, <|mixed_start|> interleaves, anything else is
  // text (from which the model may switch). The context keeps the reply
  // without its final sampled token; close the turn with
  // Append(<|im_end|>\n) to continue the conversation.
  bool Continue(const GenerateOptions& options,
                const GenerateCallbacks& callbacks, GenerateResult* result,
                std::string* error = nullptr);
  std::uint32_t Position() const;
  std::uint32_t Context() const;
  // Replaces context rows [begin, end) with pieces (none erases them) and
  // recomputes the context from the recorded inputs (tokens, audio
  // embeddings and frame embeddings).
  bool Replace(std::uint32_t begin, std::uint32_t end,
               std::span<const PromptPiece> pieces = {},
               std::string* error = nullptr);
  Model& model() const;

private:
  friend class SpeechBatch;
  Generator();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Speech synthesis of several prompts at once: each step advances every
// stream by one frame, so the backbone and the depth decoder read their
// weights once per step for all streams. Streams join between steps (their
// prefill runs then) and leave when they end. Each stream samples with its
// own seed exactly as Generator would; batched arithmetic may round
// differently from a lone stream.
class SpeechBatch {
public:
  // At most lfm3::Session::kMaxBatch streams of context slots each.
  static std::unique_ptr<SpeechBatch> Create(std::shared_ptr<Model> model,
                                             std::uint32_t context,
                                             std::size_t streams,
                                             std::string* error = nullptr);
  ~SpeechBatch();
  // Called once per stream, from Step or Add, when it ends.
  using Done =
      std::function<void(bool ok, GenerateResult result, std::string error)>;
  // Starts a stream for a prompt ending in <|audio_start|>. False (error
  // set) when the batch is full or the prompt fails; done is not called.
  bool Add(std::span<const PromptPiece> prompt, const GenerateOptions& options,
           GenerateCallbacks callbacks, Done done,
           std::string* error = nullptr);
  // Advances every stream by one frame. A device failure ends every stream
  // with that error and returns false.
  bool Step(std::string* error = nullptr);
  std::size_t active() const;
  std::size_t capacity() const;

private:
  SpeechBatch();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Live speech recognition (English): the transcript of each 0.96 s of audio
// (12 audio tokens) is decoded as soon as that audio has arrived, inside one
// assistant turn: <prompt> [A x 12] text<|text_end|> [A x 12] text... The
// language context restarts every 30 s while the encoder keeps following
// the stream. Text is greedy, as the model card.
class LiveTranscriber {
public:
  static std::unique_ptr<LiveTranscriber> Create(std::shared_ptr<Model> model,
                                                 std::string* error = nullptr);
  ~LiveTranscriber();
  using TextCallback = std::function<bool(std::string_view)>;
  // Feeds 16 kHz samples; text of completed chunks goes to on_text, whose
  // false return stops the stream.
  bool Push(std::span<const float> samples, const TextCallback& on_text,
            std::string* error = nullptr);
  // Flushes the audio and ends the transcript. Reset starts a new stream.
  bool Finish(const TextCallback& on_text, std::string* error = nullptr);
  bool Reset(std::string* error = nullptr);
  const std::string& text() const;
  std::uint64_t samples() const;
  std::uint32_t chunks() const;

private:
  LiveTranscriber();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Splits streamed reply text into reasoning (up to </think>, when the reply
// started inside <think>), visible content and tool calls. Text after an
// opening <tool_call> is held back; Finish parses the calls.
struct ToolArgument {
  std::string name;
  std::string value;  // as written; JSON unless is_string
  bool is_string{true};
};
struct ToolCall {
  std::string name;
  std::vector<ToolArgument> arguments;
};
class ReplyParser {
public:
  explicit ReplyParser(bool thinking = false) : thinking_(thinking) {}
  struct Delta {
    std::string reasoning, content;
  };
  Delta Feed(std::string_view text);
  // Remaining deltas; tool calls written after the content.
  Delta Finish(std::vector<ToolCall>* calls);
  bool thinking() const { return thinking_; }
  // A tool call has started: content is held back from here on.
  bool calling() const { return calling_; }
  const std::string& reasoning() const { return reasoning_; }
  const std::string& content() const { return content_; }

private:
  Delta Drain(bool final);
  bool thinking_, calling_{false};
  std::string pending_, reasoning_, content_, calls_;
};
}  // namespace gufo::models::lfm3_audio
#endif

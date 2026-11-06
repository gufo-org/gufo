#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::ext {

struct TranscriptionTimings {
  double decode_audio_ms{0.0};
  double feature_extraction_ms{0.0};
  double audio_encoder_ms{0.0};
  double text_decoder_ms{0.0};
  double total_ms{0.0};
};

struct TranscriptionRequest {
  std::span<const std::byte> wav;
  std::string context;
  std::optional<std::string> language;
  std::size_t max_new_tokens{256};
  /// Alternative to WAV for live sessions: normalized mono audio at 16 kHz.
  /// Exactly one audio source must be supplied.
  std::span<const float> pcm16k;
  /// Cumulative UTF-8 transcript, called during generation. Returning false
  /// cancels the operation; the view is valid only during the callback.
  std::function<bool(std::string_view)> on_text;
};

struct TranscriptionResult {
  std::string language;
  std::string text;
  std::string decoded;
  std::vector<std::uint32_t> generated_ids;
  std::uint32_t sample_rate{16000};
  std::size_t audio_samples{0};
  std::size_t mel_frames{0};
  std::size_t audio_tokens{0};
  std::size_t prompt_tokens{0};
  std::size_t chunks{0};
  TranscriptionTimings timings;
};

using CancellationCheck = std::function<bool()>;

/// Incremental transcription of one audio stream: text is decoded while the
/// audio is still arriving. Samples are mono floats at the rate the stream
/// was created with.
class LiveTranscription {
public:
  using Text = std::function<bool(std::string_view)>;
  virtual ~LiveTranscription() = default;
  virtual bool Push(std::span<const float> samples, const Text& on_text,
                    std::string* error) = 0;
  /// Flushes the audio and ends the transcript.
  virtual bool Finish(const Text& on_text, std::string* error) = 0;
  virtual const std::string& text() const = 0;
};

/// One spoken conversation with a model that hears and speaks. User audio
/// is consumed as it arrives (prefilling the model while the user speaks),
/// replies stream text and speech, and the context keeps every turn.
class VoiceConversation {
public:
  struct Settings {
    std::string instructions;
    /// Function objects (name, description, parameters), as JSON.
    std::vector<std::string> tools;
    bool audio_output{true};
    /// Reason before a text answer.
    bool reasoning{false};
    std::size_t max_output_tokens{4096};
    std::optional<std::uint64_t> seed;
  };
  struct ToolCall {
    std::string name;
    std::string arguments;  ///< JSON object
  };
  struct Reply {
    std::string reasoning, text;
    std::vector<ToolCall> calls;
    std::size_t audio_samples{}, input_tokens{}, output_tokens{};
    bool cancelled{false};
    /// Ended by the output budget or the context, not by the model.
    bool incomplete{false};
    double first_text_ms{-1}, first_audio_ms{-1}, total_ms{};
  };
  struct Callbacks {
    std::function<bool(std::string_view)> on_reasoning, on_text;
    /// 24 kHz mono samples.
    std::function<bool(std::span<const float>)> on_audio;
    std::function<bool()> cancelled;
  };
  virtual ~VoiceConversation() = default;
  /// Applies to the next turn; instructions and tools rebuild the context.
  virtual bool Configure(const Settings& settings, std::string* error) = 0;
  /// Mono samples at the input rate of the conversation; the first append
  /// opens a user turn.
  virtual bool AppendAudio(std::span<const float> samples,
                           std::string* error) = 0;
  /// Closes the user audio turn; false (with no error) when it is empty.
  virtual bool CommitAudio(std::size_t* samples, std::string* error) = 0;
  virtual bool ClearAudio(std::string* error) = 0;
  virtual bool audio_open() const = 0;
  virtual bool AddUserText(std::string_view text, std::string* error) = 0;
  /// Results of the previous reply's calls, in order, as one tool turn.
  virtual bool AddToolOutputs(std::span<const std::string> outputs,
                              std::string* error) = 0;
  virtual bool Respond(const Settings& settings, const Callbacks& callbacks,
                       Reply* reply, std::string* error) = 0;
  /// Keeps only the first samples (24 kHz) of the last reply's speech in the
  /// context, as far as the user heard it; the turn stays closed.
  virtual bool TruncateReply(std::size_t samples, std::string* error) = 0;
};

/// Abstract capability runner for speech recognition models.
class AsrRunner {
public:
  virtual ~AsrRunner() = default;

  [[nodiscard]] virtual bool Transcribe(const TranscriptionRequest& request,
                                        const CancellationCheck& is_cancelled,
                                        TranscriptionResult* result,
                                        std::string* error = nullptr) = 0;

  /// Public model ID when the service names none; empty keeps the core
  /// default.
  [[nodiscard]] virtual std::string ModelId() const { return {}; }
  /// Live transcription of a stream at rate Hz; null with an error when the
  /// model has none.
  [[nodiscard]] virtual std::unique_ptr<LiveTranscription>
  CreateLiveTranscription(std::uint32_t rate, std::string* error) {
    (void)rate;
    if (error != nullptr)
      *error = "live transcription is not supported by this model";
    return nullptr;
  }
  [[nodiscard]] virtual bool SupportsLiveTranscription() const { return false; }
  /// A spoken conversation whose user audio arrives at rate Hz.
  [[nodiscard]] virtual std::unique_ptr<VoiceConversation>
  CreateVoiceConversation(std::uint32_t rate, std::string* error) {
    (void)rate;
    if (error != nullptr)
      *error = "spoken conversations are not supported by this model";
    return nullptr;
  }
  [[nodiscard]] virtual bool SupportsVoiceConversation() const { return false; }
};

/// Model artifacts and capacity for an ASR runner. projector_path is an
/// explicit sidecar (for example an mmproj GGUF); empty lets the provider
/// resolve its own.
struct AsrRunnerOptions {
  std::string model_root;
  std::size_t maximum_context_tokens{1024};
  std::string projector_path;
};

using AsrProbe = bool (*)(std::string_view model_root);
using AsrArtifactResolver = bool (*)(std::string_view model_root,
                                     std::string* error);
using AsrRunnerFactory = std::unique_ptr<AsrRunner> (*)(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error);
using AsrRunnerOptionsFactory = std::unique_ptr<AsrRunner> (*)(
    const AsrRunnerOptions& options, std::string* error);

struct AsrProvider {
  std::string id;
  AsrProbe probe = nullptr;
  AsrArtifactResolver resolve = nullptr;
  AsrRunnerFactory create = nullptr;
  /// Preferred over create when set; receives the projector path.
  AsrRunnerOptionsFactory create_with_options = nullptr;
};

void RegisterAsrProvider(AsrProvider provider);
const AsrProvider* FindAsrProvider(std::string_view id);
const AsrProvider* ProbeAsrProvider(std::string_view model_root);

}  // namespace gufo::ext

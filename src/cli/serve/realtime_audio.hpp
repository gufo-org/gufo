#ifndef GUFO_SERVER_REALTIME_AUDIO_HPP_
#define GUFO_SERVER_REALTIME_AUDIO_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::server {

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
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_REALTIME_AUDIO_HPP_

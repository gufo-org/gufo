#ifndef GUFO_SERVER_VOICE_ACTIVITY_HPP_
#define GUFO_SERVER_VOICE_ACTIVITY_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::server {

/// OpenAI Realtime "server_vad" settings.
struct VadSettings {
  /// 0 (most sensitive) to 1: the level above the noise floor that counts
  /// as speech, from 6 dB to 30 dB.
  double threshold{0.5};
  int prefix_padding_ms{300};
  int silence_duration_ms{500};
  bool create_response{true};
  bool interrupt_response{true};
};

/// Energy voice activity: 20 ms frames are voiced when their level is above
/// -50 dBFS and above the noise floor (the 10th percentile of the last three
/// seconds; speech always has pauses) by the threshold margin. Speech starts
/// after 100 ms of voiced frames and stops after silence_duration_ms without
/// one; a turn with less than kMinSpeechMs of voiced frames is noise. While
/// guarded (the reply is still playing on the client), a start needs 12 dB
/// more and -38 dBFS, so the reply's echo does not start a turn.
class VoiceActivity {
public:
  static constexpr int kMinSpeechMs = 200;
  struct Event {
    bool start;  ///< speech started (true) or stopped (false)
    /// Stream sample where speech started (its first voiced frame) or where
    /// the turn ends (after the trailing silence), and its offset in the
    /// pushed piece (0 when it lies in an earlier piece).
    std::uint64_t sample;
    std::size_t offset;
    /// Voiced audio of a stopped turn; below kMinSpeechMs it was noise.
    int voiced_ms{0};
  };
  void Guard(bool guarded) noexcept { guarded_ = guarded; }
  VoiceActivity(VadSettings settings, std::uint32_t rate)
      : settings_(settings),
        frame_(rate / 50),
        silence_frames_(std::max(1, settings.silence_duration_ms / 20)),
        margin_db_(6.0 + 24.0 * std::clamp(settings.threshold, 0.0, 1.0)) {}
  [[nodiscard]] bool speaking() const noexcept { return speaking_; }
  [[nodiscard]] const VadSettings& settings() const noexcept {
    return settings_;
  }
  std::vector<Event> Push(std::span<const float> samples) {
    std::vector<Event> events;
    for (std::size_t i = 0; i < samples.size(); ++i) {
      energy_ += double(samples[i]) * samples[i];
      if (++filled_ < frame_)
        continue;
      Frame(i + 1, &events);
      filled_ = 0;
      energy_ = 0;
    }
    position_ += samples.size();
    return events;
  }

private:
  static constexpr int kStartFrames = 5;
  static constexpr double kSilenceDb = -50.0, kGuardedDb = -38.0;
  static constexpr double kGuardMarginDb = 12.0;
  void Frame(std::size_t end, std::vector<Event>* events) {
    const double level =
        10.0 * std::log10(energy_ / double(frame_) + 1e-12);  // dBFS
    levels_[next_++ % levels_.size()] = level;
    const auto filled = std::min(next_, levels_.size());
    std::array<double, 150> sorted;
    std::copy_n(levels_.begin(), filled, sorted.begin());
    std::nth_element(sorted.begin(), sorted.begin() + filled / 10,
                     sorted.begin() + filled);
    const double floor = sorted[filled / 10];
    const bool voiced = level > kSilenceDb && level > floor + margin_db_;
    const std::uint64_t at = position_ + end;
    if (!speaking_) {
      const bool loud =
          !guarded_ ||
          (level > kGuardedDb && level > floor + margin_db_ + kGuardMarginDb);
      run_ = voiced && loud ? run_ + 1 : 0;
      if (run_ >= kStartFrames) {
        speaking_ = true;
        quiet_ = 0;
        voiced_ = run_;
        const std::uint64_t start = at - std::uint64_t(run_) * frame_;
        events->push_back(
            {true, start,
             start > position_ ? std::size_t(start - position_) : 0});
      }
      return;
    }
    quiet_ = voiced ? 0 : quiet_ + 1;
    voiced_ += voiced;
    if (quiet_ >= silence_frames_) {
      speaking_ = false;
      run_ = 0;
      events->push_back({false, at, end, voiced_ * 20});
    }
  }
  VadSettings settings_;
  std::size_t frame_;
  int silence_frames_;
  double margin_db_;
  double energy_{0};
  std::array<double, 150> levels_{};  // three seconds of frame levels
  std::size_t filled_{0}, next_{0};
  std::uint64_t position_{0};
  int run_{0}, quiet_{0}, voiced_{0};
  bool speaking_{false}, guarded_{false};
};

/// Reads a Realtime turn_detection value: null (the client commits turns) or
/// {"type": "server_vad", threshold, prefix_padding_ms, silence_duration_ms,
/// create_response, interrupt_response}. Throws std::invalid_argument.
inline std::optional<VadSettings> ParseTurnDetection(const json::Value& value) {
  if (value.is_null())
    return std::nullopt;
  if (!value.is_object())
    throw std::invalid_argument("turn_detection must be null or an object");
  VadSettings settings;
  for (const auto& [name, field] : value.members()) {
    if (name == "type") {
      if (!field.is_string() || field.str() != "server_vad")
        throw std::invalid_argument(
            "turn_detection.type must be server_vad (semantic_vad is not "
            "supported)");
    } else if (name == "threshold") {
      if (!field.is_number() || field.as_double() < 0 || field.as_double() > 1)
        throw std::invalid_argument("turn_detection.threshold is in [0, 1]");
      settings.threshold = field.as_double();
    } else if (name == "prefix_padding_ms" || name == "silence_duration_ms") {
      if (!field.is_number() || field.as_double() < 0 ||
          field.as_double() > 10000 ||
          field.as_double() != std::floor(field.as_double()))
        throw std::invalid_argument("turn_detection." + name +
                                    " is an integer in [0, 10000]");
      (name == "prefix_padding_ms" ? settings.prefix_padding_ms
                                   : settings.silence_duration_ms) =
          int(field.as_double());
    } else if (name == "create_response" || name == "interrupt_response") {
      if (!field.is_bool())
        throw std::invalid_argument("turn_detection." + name +
                                    " must be a boolean");
      (name == "create_response" ? settings.create_response
                                 : settings.interrupt_response) =
          field.as_bool();
    } else if (name != "idle_timeout_ms" || !field.is_null()) {
      throw std::invalid_argument("unsupported turn_detection field: " + name);
    }
  }
  if (!value.find("type"))
    throw std::invalid_argument("turn_detection needs type server_vad");
  return settings;
}

/// The session's view of turn detection settings.
inline json::Value TurnDetectionJson(const std::optional<VadSettings>& vad) {
  if (!vad)
    return json::Value{};
  auto value = json::Value::object();
  value["type"] = "server_vad";
  value["threshold"] = vad->threshold;
  value["prefix_padding_ms"] = vad->prefix_padding_ms;
  value["silence_duration_ms"] = vad->silence_duration_ms;
  value["create_response"] = vad->create_response;
  value["interrupt_response"] = vad->interrupt_response;
  return value;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_VOICE_ACTIVITY_HPP_

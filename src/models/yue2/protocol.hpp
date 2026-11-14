#ifndef GUFO_MODELS_YUE2_PROTOCOL_HPP_
#define GUFO_MODELS_YUE2_PROTOCOL_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/yue2/tokenizer.hpp"

namespace gufo::models::yue2 {

// Checkpoint-native vocabulary layout ("yue2-native-v1").
inline constexpr std::uint32_t kEndOfDocument = 151643;
inline constexpr std::uint32_t kAbcStart = 151847;
inline constexpr std::uint32_t kAbcEnd = 151848;
inline constexpr std::uint32_t kMusicStart = 151851;
inline constexpr std::uint32_t kMusicEnd = 151852;
inline constexpr std::uint32_t kCodecOffset = 151853;
inline constexpr std::uint32_t kCodecSize = 32768;
inline constexpr std::uint32_t kVocabularySize = 184704;
inline constexpr std::size_t kContext = 24576;
inline constexpr std::size_t kLatentDim = 64;
inline constexpr double kFrameRate = 25.0;
inline constexpr std::string_view kProtocolVersion = "yue2-native-v1";

/// Planning mode (upstream `cot`).
enum class Planning : std::uint8_t {
  kOff,
  kMelody,
  kFull,
};

[[nodiscard]] std::optional<Planning> ParsePlanning(std::string_view name);
[[nodiscard]] std::string_view PlanningName(Planning planning);

enum class Phase : std::uint8_t {
  kAbc,
  kSemantic,
};

/// One stage's sampling (upstream `Sampling`).
struct Sampling {
  float temperature{1.0F};
  float top_p{0.95F};
  std::uint32_t top_k{100};
  float repetition_penalty{1.2F};
  std::uint32_t penalty_window{50};
  std::uint32_t min_tokens{200};
  std::uint32_t max_tokens{9000};
};

[[nodiscard]] Sampling DefaultAbcSampling();
[[nodiscard]] Sampling DefaultSemanticSampling();
/// Upstream bounds; an error names the first violation.
[[nodiscard]] bool ValidateSampling(const Sampling& sampling,
                                    std::string* error);

struct SongPrompt {
  std::string style;
  std::string lyrics;
  Planning planning{Planning::kFull};
};

/// `"<instruction>\n[Tags]\n<style>\n[Lyrics]\n<lyrics>\n"`.
[[nodiscard]] std::string PromptText(const SongPrompt& prompt);

/// The planning prefix `[EOD] text <abc>`. With `planning == kOff` the
/// prefix already closes an empty score and opens the music.
[[nodiscard]] std::vector<std::uint32_t> PlanPrefix(
    const TextTokenizer& tokenizer, const SongPrompt& prompt);

/// The semantic prefix `[EOD] text <abc> abc </abc> <music>`; `abc_ids` is
/// ignored with `planning == kOff`.
[[nodiscard]] std::vector<std::uint32_t> SemanticPrefix(
    const TextTokenizer& tokenizer, const SongPrompt& prompt,
    std::span<const std::uint32_t> abc_ids);

/// The unconditional CFG branch: the instruction alone, plus the exact score.
[[nodiscard]] std::vector<std::uint32_t> NegativePrefix(
    const TextTokenizer& tokenizer, const SongPrompt& prompt,
    std::span<const std::uint32_t> abc_ids);

/// Default guidance: 1.01 without a score, otherwise 1 (one branch).
[[nodiscard]] float DefaultGuidance(Planning planning);

/// Codec frames per acoustic chunk; one chunk unless the prefix is huge.
[[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> ChunkRanges(
    std::size_t frames, std::size_t prefix_tokens);

}  // namespace gufo::models::yue2

#endif  // GUFO_MODELS_YUE2_PROTOCOL_HPP_

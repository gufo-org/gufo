#ifndef GUFO_CORE_TEXT_SAMPLING_DEFAULTS_HPP_
#define GUFO_CORE_TEXT_SAMPLING_DEFAULTS_HPP_

#include <optional>

#include "src/core/model_config.hpp"
#include "src/core/sampling.hpp"

namespace gufo::sampling {

enum class TextModelPreset { kUnspecified, kQwen38, kDeepSeekV4Flash };

// Presence is independent of value: an explicit zero must override a preset.
struct SamplingOverrides {
  bool temperature{false};
  bool top_k{false};
  bool top_p{false};
  bool min_p{false};
  bool min_keep{false};
  bool seed{false};
  bool repeat_penalty{false};
  bool repeat_last_n{false};
  bool frequency_penalty{false};
  bool presence_penalty{false};

  static constexpr SamplingOverrides All() noexcept {
    return {true, true, true, true, true, true, true, true, true, true};
  }
};

inline TextModelPreset TextPreset(const core::ModelConfig& config) {
  if (config.architecture == "deepseek4")
    return TextModelPreset::kDeepSeekV4Flash;
  if (config.architecture == "qwen4exp" ||
      (config.num_layers == 64 && config.hidden_size == 5120 &&
       config.vocab_size == 248320))
    return TextModelPreset::kQwen38;
  return TextModelPreset::kUnspecified;
}

inline SamplingConfig ResolveTextSampling(TextModelPreset model,
                                          std::optional<bool> thinking,
                                          const SamplingConfig& configured,
                                          const SamplingOverrides& supplied) {
  if (model == TextModelPreset::kUnspecified)
    return configured;
  SamplingConfig result;
  result.temperature = 1.0F;
  result.top_p = 0.95F;
  if (model == TextModelPreset::kQwen38) {
    result.top_k = 20;
    if (!thinking.value_or(true)) {
      result.temperature = 0.7F;
      result.top_p = 0.8F;
      result.presence_penalty = 1.5F;
    }
  }
  if (supplied.temperature)
    result.temperature = configured.temperature;
  if (supplied.top_k)
    result.top_k = configured.top_k;
  if (supplied.top_p)
    result.top_p = configured.top_p;
  if (supplied.min_p)
    result.min_p = configured.min_p;
  if (supplied.min_keep)
    result.min_keep = configured.min_keep;
  if (supplied.seed)
    result.seed = configured.seed;
  if (supplied.repeat_penalty)
    result.repeat_penalty = configured.repeat_penalty;
  if (supplied.repeat_last_n)
    result.repeat_last_n = configured.repeat_last_n;
  if (supplied.frequency_penalty)
    result.frequency_penalty = configured.frequency_penalty;
  if (supplied.presence_penalty)
    result.presence_penalty = configured.presence_penalty;
  return result;
}

}  // namespace gufo::sampling

#endif

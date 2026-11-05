#ifndef GUFO_CORE_TEXT_SAMPLING_DEFAULTS_HPP_
#define GUFO_CORE_TEXT_SAMPLING_DEFAULTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>

#include "src/core/model_config.hpp"
#include "src/core/sampling.hpp"

namespace gufo::sampling {

/// Core presets kept for existing extensions; new providers supply
/// `TextSamplingDefaults` instead.
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

/// Provider-defined values for the sampling controls. An unset field leaves
/// the control at the `SamplingConfig` default; set values, including zero,
/// are applied as written. Output constraints and the seed are never defaults.
struct TextSamplingValues {
  std::optional<float> temperature{};
  std::optional<std::int32_t> top_k{};
  std::optional<float> top_p{};
  std::optional<float> min_p{};
  std::optional<std::size_t> min_keep{};
  std::optional<float> repeat_penalty{};
  std::optional<std::size_t> repeat_last_n{};
  std::optional<float> frequency_penalty{};
  std::optional<float> presence_penalty{};

  bool operator==(const TextSamplingValues&) const = default;
};

/// Typed defaults a text provider recommends, selected by the request's
/// effective thinking mode (`std::nullopt`: the model decides).
struct TextSamplingDefaults {
  TextSamplingValues thinking_unspecified;
  TextSamplingValues thinking_enabled;
  TextSamplingValues thinking_disabled;

  [[nodiscard]] const TextSamplingValues& For(
      std::optional<bool> thinking) const noexcept {
    if (!thinking.has_value())
      return thinking_unspecified;
    return *thinking ? thinking_enabled : thinking_disabled;
  }

  bool operator==(const TextSamplingDefaults&) const = default;
};

/// Defaults of the in-tree presets, written as provider-supplied values.
inline std::optional<TextSamplingDefaults> PresetDefaults(
    TextModelPreset model) {
  if (model == TextModelPreset::kUnspecified)
    return std::nullopt;
  TextSamplingValues on;
  on.temperature = 1.0F;
  on.top_p = 0.95F;
  TextSamplingValues off = on;
  if (model == TextModelPreset::kQwen38) {
    on.top_k = 20;
    off.top_k = 20;
    off.temperature = 0.7F;
    off.top_p = 0.8F;
    off.presence_penalty = 1.5F;
  }
  // An unspecified thinking mode behaves as enabled.
  return TextSamplingDefaults{on, on, off};
}

/// Precedence: provider defaults < explicit server/CLI settings (`supplied`,
/// values in `configured`) < per-request overrides, which callers apply on top
/// of the result. Without provider defaults `configured` is returned as is.
inline SamplingConfig ResolveTextSampling(
    const std::optional<TextSamplingDefaults>& provider,
    std::optional<bool> thinking, const SamplingConfig& configured,
    const SamplingOverrides& supplied) {
  if (!provider.has_value())
    return configured;
  const TextSamplingValues& values = provider->For(thinking);
  SamplingConfig result;
  // Provider defaults change sampling controls, never the caller's output
  // grammar.
  result.constraint = configured.constraint;
  const auto pick = [](bool explicit_setting, const auto& configured_value,
                       const auto& provided, auto* out) {
    if (explicit_setting)
      *out = configured_value;
    else if (provided.has_value())
      *out = *provided;
  };
  pick(supplied.temperature, configured.temperature, values.temperature,
       &result.temperature);
  pick(supplied.top_k, configured.top_k, values.top_k, &result.top_k);
  pick(supplied.top_p, configured.top_p, values.top_p, &result.top_p);
  pick(supplied.min_p, configured.min_p, values.min_p, &result.min_p);
  pick(supplied.min_keep, configured.min_keep, values.min_keep,
       &result.min_keep);
  pick(supplied.repeat_penalty, configured.repeat_penalty,
       values.repeat_penalty, &result.repeat_penalty);
  pick(supplied.repeat_last_n, configured.repeat_last_n, values.repeat_last_n,
       &result.repeat_last_n);
  pick(supplied.frequency_penalty, configured.frequency_penalty,
       values.frequency_penalty, &result.frequency_penalty);
  pick(supplied.presence_penalty, configured.presence_penalty,
       values.presence_penalty, &result.presence_penalty);
  if (supplied.seed)
    result.seed = configured.seed;
  return result;
}

/// Throws `std::invalid_argument` when a thinking mode's values are invalid.
inline void ValidateTextSamplingDefaults(const TextSamplingDefaults& defaults) {
  for (const std::optional<bool> thinking :
       {std::optional<bool>{}, std::optional<bool>{true},
        std::optional<bool>{false}}) {
    ResolveTextSampling(defaults, thinking, {}, {}).Validate();
  }
}

inline SamplingConfig ResolveTextSampling(TextModelPreset model,
                                          std::optional<bool> thinking,
                                          const SamplingConfig& configured,
                                          const SamplingOverrides& supplied) {
  return ResolveTextSampling(PresetDefaults(model), thinking, configured,
                             supplied);
}

}  // namespace gufo::sampling

#endif

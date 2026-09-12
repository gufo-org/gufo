#ifndef GUFO_SERVER_SAMPLING_REQUEST_HPP_
#define GUFO_SERVER_SAMPLING_REQUEST_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include "src/cli/serve/json.hpp"
#include "src/core/sampling.hpp"

namespace gufo::server {

struct SamplingRequestError {
  std::string message;
  std::string code;
};

namespace detail {

inline constexpr std::int64_t kMaxExactJsonInteger = INT64_C(9007199254740991);
inline constexpr std::size_t kMaxSamplingWindow =
    std::numeric_limits<std::uint32_t>::max();

inline std::string SamplingErrorCode(std::string_view field) {
  std::string code = "invalid_";
  code.append(field);
  for (char& character : code) {
    if (character == '-') {
      character = '_';
    }
  }
  return code;
}

inline std::optional<SamplingRequestError> ReadSamplingFloat(
    const json::Value& body, std::string_view field, float* output) {
  const json::Value* value = body.find(std::string(field));
  if (value == nullptr) {
    return std::nullopt;
  }
  if (!value->is_number() || !std::isfinite(value->as_double()) ||
      value->as_double() <
          -static_cast<double>(std::numeric_limits<float>::max()) ||
      value->as_double() >
          static_cast<double>(std::numeric_limits<float>::max())) {
    return SamplingRequestError{
        .message = "'" + std::string(field) + "' must be a finite number",
        .code = SamplingErrorCode(field),
    };
  }
  *output = static_cast<float>(value->as_double());
  return std::nullopt;
}

template<typename Integer>
inline std::optional<SamplingRequestError> ReadSamplingInteger(
    const json::Value& body, std::string_view field, Integer minimum,
    Integer maximum, Integer* output) {
  const json::Value* value = body.find(std::string(field));
  if (value == nullptr) {
    return std::nullopt;
  }
  const double number = value->is_number() ? value->as_double() : 0.0;
  if (!value->is_number() || !std::isfinite(number) ||
      std::floor(number) != number || number < static_cast<double>(minimum) ||
      number > static_cast<double>(maximum)) {
    return SamplingRequestError{
        .message = "'" + std::string(field) + "' must be an integer between " +
                   std::to_string(minimum) + " and " + std::to_string(maximum),
        .code = SamplingErrorCode(field),
    };
  }
  *output = static_cast<Integer>(number);
  return std::nullopt;
}

}  // namespace detail

/// Reads all shared sampling controls, using defaults for omitted fields.
inline std::optional<SamplingRequestError> ParseSamplingConfig(
    const json::Value& body, const sampling::SamplingConfig& defaults,
    sampling::SamplingConfig* output) {
  if (output == nullptr) {
    return SamplingRequestError{
        .message = "sampling output must not be null",
        .code = "invalid_sampling",
    };
  }
  *output = defaults;

  // These controls alter proposal/target probabilities. Reject unsupported
  // spellings instead of accepting a request with a different distribution.
  for (const auto& [field, value] : body.members()) {
    const bool draft_control = field.starts_with("draft_") ||
                               field.ends_with("_draft") || field == "draft" ||
                               field == "speculative";
    if (draft_control || field == "samplers" || field == "typical_p" ||
        field == "tfs_z" || field == "mirostat" || field == "mirostat_eta" ||
        field == "mirostat_tau" || field == "dynatemp_range" ||
        field == "dynatemp_exponent") {
      return SamplingRequestError{
          .message =
              "request field '" + field + "' is not supported" +
              (draft_control ? "; DFlash2 uses the target temperature and its "
                               "trained draft selector"
                             : ""),
          .code = "unsupported_sampling",
      };
    }
  }

  if (auto error = detail::ReadSamplingFloat(body, "temperature",
                                             &output->temperature)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(
          body, "top_k", std::int32_t{0},
          std::numeric_limits<std::int32_t>::max(), &output->top_k)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(body, "top_p", &output->top_p)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(body, "min_p", &output->min_p)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(body, "min_keep", std::size_t{0},
                                               detail::kMaxSamplingWindow,
                                               &output->min_keep)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(body, "seed", std::int64_t{-1},
                                               detail::kMaxExactJsonInteger,
                                               &output->seed)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(body, "repeat_penalty",
                                             &output->repeat_penalty)) {
    return error;
  }
  if (auto error = detail::ReadSamplingInteger(
          body, "repeat_last_n", std::size_t{0}, detail::kMaxSamplingWindow,
          &output->repeat_last_n)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(body, "frequency_penalty",
                                             &output->frequency_penalty)) {
    return error;
  }
  if (auto error = detail::ReadSamplingFloat(body, "presence_penalty",
                                             &output->presence_penalty)) {
    return error;
  }

  try {
    output->Validate();
  } catch (const std::invalid_argument& exception) {
    return SamplingRequestError{
        .message = exception.what(),
        .code = "invalid_sampling",
    };
  }
  return std::nullopt;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_SAMPLING_REQUEST_HPP_

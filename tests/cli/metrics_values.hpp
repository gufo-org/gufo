#ifndef GUFO_TESTS_CLI_METRICS_VALUES_HPP_
#define GUFO_TESTS_CLI_METRICS_VALUES_HPP_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include "src/cli/serve/generation_metrics.hpp"

namespace gufo::server::testing {

inline std::string Sample(std::string_view name) {
  const std::string text = "\n" + Metrics().registry.Render();
  const std::string prefix = "\n" + std::string(name) + " ";
  auto start = text.find(prefix);
  if (start == std::string::npos)
    throw std::runtime_error("no sample " + std::string(name));
  start += prefix.size();
  return text.substr(start, text.find('\n', start) - start);
}

inline std::uint64_t Unsigned(std::string_view name) {
  return std::stoull(Sample(name));
}

inline std::uint64_t DeviceLostTotal() {
  return Unsigned("gufo_device_lost_total");
}
inline std::uint64_t TotalPromptTokens() {
  return Unsigned("llamacpp:prompt_tokens_total");
}
inline std::uint64_t TotalGenTokens() {
  return Unsigned("llamacpp:tokens_predicted_total");
}
inline std::uint64_t TotalCachedPromptTokens() {
  return Unsigned("llamacpp:prompt_tokens_cached_total");
}
inline double TotalPromptSeconds() {
  return std::stod(Sample("llamacpp:prompt_seconds_total"));
}
inline double TotalGenSeconds() {
  return std::stod(Sample("llamacpp:tokens_predicted_seconds_total"));
}
inline std::uint64_t MaxSequenceTokens() {
  return Unsigned("llamacpp:n_tokens_max");
}
inline std::uint64_t TotalDraftRounds() {
  return Unsigned("llamacpp:spec_decode_num_drafts_total");
}
inline std::uint64_t TotalDraftTokens() {
  return Unsigned("llamacpp:spec_decode_num_draft_tokens_total");
}
inline std::uint64_t TotalDraftAcceptedTokens() {
  return Unsigned("llamacpp:spec_decode_num_accepted_tokens_total");
}
inline std::int64_t RequestsProcessing() {
  return std::stoll(Sample("llamacpp:requests_processing"));
}
inline std::int64_t RequestsDeferred() {
  return std::stoll(Sample("llamacpp:requests_deferred"));
}
inline std::uint64_t HistogramCount(std::string_view name) {
  return Unsigned(std::string(name) + "_count");
}
inline double HistogramSum(std::string_view name) {
  return std::stod(Sample(std::string(name) + "_sum"));
}

}  // namespace gufo::server::testing

#endif  // GUFO_TESTS_CLI_METRICS_VALUES_HPP_

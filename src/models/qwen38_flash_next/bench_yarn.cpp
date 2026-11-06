#include "src/models/qwen38_flash_next/bench_yarn.hpp"

#include <algorithm>
#include <limits>

namespace gufo::models::qwen38_flash_next {

std::size_t FlashNextBenchContext(const ext::TextBenchOptions& options) {
  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  const std::size_t max_depth = max_or_zero(options.n_depths);
  return std::max(
      {std::size_t{4096}, max_depth + max_or_zero(options.n_prompts) + 1,
       std::max<std::size_t>(max_depth, 16) + max_or_zero(options.n_gens) + 1,
       options.validate_prefill_tokens + 1});
}

bool ResolveFlashNextBenchYarn(const ext::TextBenchOptions& options,
                               std::uint32_t native,
                               std::optional<RopeScaling>* scaling,
                               std::string* error) {
  const std::size_t context = FlashNextBenchContext(options);
  if (context > std::numeric_limits<std::uint32_t>::max()) {
    *scaling = std::nullopt;
    if (error != nullptr)
      *error = "benchmark workload exceeds the context range";
    return false;
  }
  return ResolveRopeScaling(static_cast<std::uint32_t>(context), native,
                            options.enable_yarn, scaling, error);
}

}  // namespace gufo::models::qwen38_flash_next

#ifndef GUFO_CLI_BENCH_HPP_
#define GUFO_CLI_BENCH_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen38_flash_next/rope_scaling.hpp"

namespace gufo::cli {

struct BenchOptions {
  std::string model_path;
  std::vector<std::size_t> n_prompts{2048};
  std::vector<std::size_t> n_gens{128};
  std::vector<std::size_t> n_depths{0};
  std::vector<std::size_t> concurrency{1};
  std::size_t repetitions{1};
  std::size_t validate_prefill_tokens{0};
  std::string speculative_backend{""};
  std::string mtp_model_path;
  std::string dflash_model_path;
  std::string draft_policy;
  std::string dspark_model_path;
  std::uint32_t draft_tokens{7};
  std::uint32_t min_draft_tokens{1};
  sampling::SamplingConfig sampling{.seed = 0};
  bool enable_yarn{false};
  bool verbose{false};
};

/// Context a Flash-Next benchmark reserves for its largest workload: depth
/// plus prompt, or depth plus generation, plus one (at least 4096).
[[nodiscard]] std::size_t FlashNextBenchContext(const BenchOptions& options);

/// Applies serve's --enable-yarn rule to FlashNextBenchContext: above
/// `native` the flag is required, at or below it the flag is rejected.
[[nodiscard]] bool ResolveFlashNextBenchYarn(
    const BenchOptions& options, std::uint32_t native,
    std::optional<models::qwen38_flash_next::RopeScaling>* scaling,
    std::string* error);

void PrintBenchHelp(std::string_view program_name);
std::optional<BenchOptions> ParseBenchOptions(std::span<const char* const> args,
                                              std::string* error_msg = nullptr);
int RunBench(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_BENCH_HPP_

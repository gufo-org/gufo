#include "src/cli/bench/bench.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gufo/ext/registry.hpp"
#include "gufo/ext/text.hpp"
#include "src/cli/arg_parser.hpp"
#include "src/cli/sampling_options.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"

namespace gufo::cli {

namespace {

void PrintModelLoadTime(std::chrono::steady_clock::time_point start,
                        bool success = true) {
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  auto& output = success ? std::cout : std::cerr;
  output << "[Model Load]: " << load_seconds << " s";
  if (!success) {
    output << " (failed)";
  }
  output << '\n';
}

std::optional<std::vector<std::size_t>> ParseCommaSeparatedSizes(
    std::string_view text, bool keep_zero = false) {
  if (text.empty() || text.back() == ',')
    return std::nullopt;
  std::vector<std::size_t> result;
  for (std::size_t start = 0; start < text.size();) {
    const auto comma = text.find(',', start);
    const auto end = comma == std::string_view::npos ? text.size() : comma;
    const auto part = text.substr(start, end - start);
    std::size_t value = 0;
    const auto [ptr, ec] =
        std::from_chars(part.data(), part.data() + part.size(), value);
    if (ec != std::errc{} || ptr != part.data() + part.size())
      return std::nullopt;
    if (value != 0 || keep_zero)
      result.push_back(value);
    start = end + 1;
  }
  return result;
}

void RegisterBenchOptions(ArgParser& parser, BenchOptions& opt,
                          bool& explicit_p, bool& explicit_n,
                          bool& speculative_explicit) {
  parser.AddOption("-m", "--model", "PATH",
                   "Path to GGUF model file (required)", "Model",
                   &opt.model_path);

  parser.AddCustomOption(
      "-p", "--n-prompt", "n,n,...",
      "Prompt token lengths to benchmark (default: 2048)", "Workload",
      [&opt, &explicit_p](std::string_view flag, std::string_view val,
                          std::string* error) -> bool {
        auto sizes = ParseCommaSeparatedSizes(val);
        if (!sizes) {
          if (error)
            *error = "Invalid token counts for " + std::string(flag);
          return false;
        }
        opt.n_prompts = std::move(*sizes);
        explicit_p = true;
        return true;
      });

  parser.AddCustomOption(
      "-n", "--n-gen", "n,n,...",
      "Generation token counts to benchmark (default: 128)", "Workload",
      [&opt, &explicit_n](std::string_view flag, std::string_view val,
                          std::string* error) -> bool {
        auto sizes = ParseCommaSeparatedSizes(val);
        if (!sizes) {
          if (error)
            *error = "Invalid token counts for " + std::string(flag);
          return false;
        }
        opt.n_gens = std::move(*sizes);
        explicit_n = true;
        return true;
      });

  parser.AddCustomOption(
      "-d", "--n-depth", "n,n,...",
      "Prefix context lengths to benchmark (default: 0)", "Workload",
      [&opt](std::string_view flag, std::string_view val,
             std::string* error) -> bool {
        auto sizes = ParseCommaSeparatedSizes(val, true);
        if (!sizes) {
          if (error)
            *error = "Invalid prefix lengths for " + std::string(flag);
          return false;
        }
        opt.n_depths = std::move(*sizes);
        return true;
      });

  parser.AddCustomOption(
      "-c", "--concurrency", "n,n,...",
      "DS4 simultaneous requests, 1..8 (default: 1); pp is aggregate, tg per "
      "user",
      "Workload",
      [&opt](std::string_view, std::string_view value, std::string* error) {
        auto sizes = ParseCommaSeparatedSizes(value, true);
        if (!sizes || sizes->empty() ||
            std::any_of(sizes->begin(), sizes->end(),
                        [](auto size) { return size < 1 || size > 8; })) {
          if (error)
            *error =
                "--concurrency requires DS4 request counts between 1 and 8";
          return false;
        }
        opt.concurrency = std::move(*sizes);
        return true;
      });

  parser.AddOption("-r", "--repetitions", "N",
                   "Number of repetitions per measurement (default: 1)",
                   "Workload", &opt.repetitions);

  parser.AddOption("", "--validate-prefill", "N",
                   "Validate prefill logits against reference for N tokens "
                   "(default: 0 = disabled)",
                   "Workload", &opt.validate_prefill_tokens);

  parser.AddCustomOption(
      "", "--speculative", "none|mtp|dflash2|dspark",
      "Speculative decoding backend (default: none)", "Speculative",
      [&opt, &speculative_explicit](std::string_view flag, std::string_view val,
                                    std::string* error) -> bool {
        speculative_explicit = true;
        if (val == "off" || val == "none") {
          opt.speculative_backend.clear();
          return true;
        }
        if (val == "mtp" || val == "dflash2" || val == "dspark") {
          opt.speculative_backend = std::string(val);
          return true;
        }
        if (error) {
          *error = "Unknown speculative backend: " + std::string(val);
        }
        return false;
      });

  parser.AddOption("", "--mtp-model", "PATH",
                   "Path to GGUF model containing MTP draft weights",
                   "Speculative", &opt.mtp_model_path);

  parser.AddOption("", "--dflash-model", "PATH",
                   "Path to GGUF model containing DFlash draft weights",
                   "Speculative", &opt.dflash_model_path);

  parser.AddOption("", "--draft-policy", "POLICY",
                   "DFlash2 draft length policy: fixed or adaptive "
                   "(default: adaptive)",
                   "Speculative", &opt.draft_policy);

  parser.AddOption("", "--dspark-model", "PATH",
                   "Path to GGUF model containing DSpark support weights",
                   "Speculative", &opt.dspark_model_path);

  parser.AddOption("", "--draft-tokens", "N",
                   "Maximum draft tokens per speculative step (default: 7)",
                   "Speculative", &opt.draft_tokens);

  parser.AddOption("", "--min-draft-tokens", "N",
                   "Minimum draft tokens per speculative step (default: 1)",
                   "Speculative", &opt.min_draft_tokens);

  RegisterSamplingOptions(parser, &opt.sampling, "Sampling", false);

  parser.AddFlag("", "--enable-yarn",
                 "Flash-Next: static YaRN, required when depth + prompt or "
                 "generation exceeds the native 262144",
                 "Model", &opt.enable_yarn);

  parser.AddFlag("-v", "--verbose",
                 "Print detailed timing, latency breakdown, and tok/s metrics",
                 "General", &opt.verbose);
}

}  // namespace

void PrintBenchHelp(std::string_view program_name) {
  BenchOptions dummy_opt;
  bool explicit_p = false;
  bool explicit_n = false;
  bool speculative_explicit = false;
  ArgParser parser(
      std::string(program_name) + " bench",
      "Benchmark prompt processing (pp) and token generation (tg) throughput.");
  RegisterBenchOptions(parser, dummy_opt, explicit_p, explicit_n,
                       speculative_explicit);
  parser.PrintHelp();
}

std::optional<BenchOptions> ParseBenchOptions(std::span<const char* const> args,
                                              std::string* error_msg) {
  BenchOptions opt;
  bool explicit_p = false;
  bool explicit_n = false;
  bool speculative_explicit = false;
  ArgParser parser(
      "gufo bench",
      "Benchmark prompt processing (pp) and token generation (tg) throughput.");
  RegisterBenchOptions(parser, opt, explicit_p, explicit_n,
                       speculative_explicit);

  if (!parser.Parse(args, error_msg)) {
    return std::nullopt;
  }
  if (parser.IsHelpRequested()) {
    return std::nullopt;
  }

  if (explicit_p && !explicit_n) {
    opt.n_gens.clear();
  } else if (!explicit_p && explicit_n) {
    opt.n_prompts.clear();
  }

  if (opt.repetitions == 0 || (opt.n_prompts.empty() && opt.n_gens.empty() &&
                               opt.validate_prefill_tokens == 0)) {
    if (error_msg)
      *error_msg =
          "Benchmark needs positive repetitions and at least one workload";
    return std::nullopt;
  }

  if (opt.draft_tokens == 0 || opt.min_draft_tokens == 0 ||
      opt.min_draft_tokens > opt.draft_tokens) {
    if (error_msg != nullptr) {
      *error_msg = "min-draft-tokens cannot exceed draft-tokens";
    }
    return std::nullopt;
  }

  try {
    opt.sampling.Validate();
    if (opt.sampling.seed < 0) {
      throw std::invalid_argument("benchmark seed must be non-negative");
    }
  } catch (const std::invalid_argument& error) {
    if (error_msg)
      *error_msg = error.what();
    return std::nullopt;
  }

  if (!speculative_explicit && !opt.dspark_model_path.empty()) {
    opt.speculative_backend = "dspark";
  }

  if (!opt.draft_policy.empty() &&
      ((opt.draft_policy != "fixed" && opt.draft_policy != "adaptive") ||
       opt.speculative_backend != "dflash2")) {
    if (error_msg != nullptr)
      *error_msg = "--draft-policy requires DFlash2 and fixed or adaptive";
    return std::nullopt;
  }
  if (opt.min_draft_tokens != 1 && opt.speculative_backend == "dflash2") {
    if (error_msg != nullptr)
      *error_msg =
          "DFlash2 requires --min-draft-tokens 1; bound blocks with "
          "--draft-tokens";
    return std::nullopt;
  }

  return opt;
}

int RunBench(std::span<const char* const> args) {
  std::string parse_err;
  const auto opt_res = ParseBenchOptions(args, &parse_err);
  if (!opt_res.has_value()) {
    if (!parse_err.empty()) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintBenchHelp("gufo");
      return 2;
    }
    PrintBenchHelp("gufo");
    return 0;
  }

  const auto& opt = *opt_res;
  if (opt.model_path.empty()) {
    std::cerr << "Error: --model <PATH> is required\n";
    return 2;
  }

  const auto model_load_start = std::chrono::steady_clock::now();
  std::string err;
  auto reader_owner = gufo::core::GgufReader::OpenFile(opt.model_path, &err);
  if (!reader_owner) {
    std::cerr << "Error loading GGUF model '" << opt.model_path << "': " << err
              << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  const std::shared_ptr<const gufo::core::GgufReader> reader(
      std::move(reader_owner));

  ext::Registry registry;
  ext::RegisterBuiltExtensions(registry);
  const auto* provider = ext::ProbeTextProvider(*reader);
  if (provider == nullptr || provider->bench_driver == nullptr) {
    std::cerr << "Error: unsupported model architecture for benchmark\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  return provider->bench_driver(opt, reader, model_load_start);
}

}  // namespace gufo::cli

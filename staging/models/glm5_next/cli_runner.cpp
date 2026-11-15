#include "staging/models/glm5_next/cli_runner.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/bench/bench.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "staging/models/glm5_next/engine.hpp"

namespace gufo::models::glm5_next {
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

struct BenchStats {
  double mean{0.0};
  double stddev{0.0};
};

BenchStats ComputeStats(const std::vector<double>& values) {
  if (values.empty()) {
    return {.mean = 0.0, .stddev = 0.0};
  }
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / static_cast<double>(values.size());
  if (values.size() <= 1) {
    return {.mean = mean, .stddev = 0.0};
  }
  double sq_sum = 0.0;
  for (const double v : values) {
    sq_sum += (v - mean) * (v - mean);
  }
  const double variance = sq_sum / static_cast<double>(values.size() - 1);
  return {.mean = mean, .stddev = std::sqrt(variance)};
}

std::string MakeTestName(std::string_view prefix, std::size_t count,
                         std::size_t depth) {
  std::string result = std::string(prefix) + std::to_string(count);
  if (depth > 0) {
    result += " @ d" + std::to_string(depth);
  }
  return result;
}

std::string TokenSha256(const std::vector<std::int32_t>& tokens) {
  return crypto::Sha256Hex(
      std::span(reinterpret_cast<const std::uint8_t*>(tokens.data()),
                tokens.size() * sizeof(std::int32_t)));
}

}  // namespace

bool IsGlm5Next(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "glm5-next";
}

int RunGlm5NextBenchmark(const cli::BenchOptions& options,
                         const std::shared_ptr<const core::GgufReader>& reader,
                         std::chrono::steady_clock::time_point model_load_start) {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: no HIP GPU is available for GLM-5.3-Flash\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  if (!options.speculative_backend.empty() &&
      options.speculative_backend != "mtp") {
    std::cerr << "Error: glm5-next bench supports only --speculative mtp\n";
    return 1;
  }
  if (options.concurrency != std::vector<std::size_t>{1}) {
    std::cerr << "Error: glm5-next bench supports C1; use the serving "
                 "benchmark for concurrent requests\n";
    return 1;
  }
  if (!options.sampling.can_use_unmodified_argmax()) {
    std::cerr << "Error: glm5-next bench supports greedy sampling only\n";
    return 1;
  }
  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  const std::size_t max_depth = max_or_zero(options.n_depths);
  const std::size_t max_prompt = max_or_zero(options.n_prompts);
  const std::size_t max_generation = max_or_zero(options.n_gens);
  const std::size_t required_context =
      std::max({max_depth + max_prompt + 1,
                std::max<std::size_t>(max_depth, 16) + max_generation + 1});
  if (required_context > std::numeric_limits<std::uint32_t>::max()) {
    std::cerr << "Error: glm5-next context is out of range\n";
    return 1;
  }

  std::string error;
  auto model = Model::Load(
      options.model_path,
      ModelOptions{.max_context = static_cast<std::uint32_t>(required_context)},
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating glm5-next model: " << error << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  auto session = model->CreateSession(
      static_cast<std::uint32_t>(required_context), &error);
  if (session == nullptr) {
    std::cerr << "Error creating glm5-next session: " << error << '\n';
    return 1;
  }
  const bool use_mtp = options.speculative_backend == "mtp";
  if (use_mtp) {
    if (!model->HasMtp()) {
      std::cerr << "Error: glm5-next model has no nextn block; omit "
                   "--speculative mtp\n";
      return 1;
    }
    // The draft cache is filled during prefill, so enable before the first
    // Sync.
    session->SetMtpEnabled(true);
    session->SetDraftLimits(options.min_draft_tokens, options.draft_tokens);
  }

  std::vector<std::int32_t> tokens;
  {
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. "
        "Strix Halo executes this deterministic benchmark sequence. ");
    if (pattern.empty()) {
      std::cerr << "Error: benchmark token pattern is empty\n";
      return 1;
    }
    tokens.resize(required_context);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      tokens[i] = pattern[i % pattern.size()];
    }
  }

  const double size_gib =
      static_cast<double>(reader->GetSize()) / (1024.0 * 1024.0 * 1024.0);
  const auto model_name = model->ModelName();
  std::cout << "| " << std::left << std::setw(32) << "model"
            << " | " << std::right << std::setw(10) << "size"
            << " | " << std::left << std::setw(10) << "backend"
            << " | " << std::right << std::setw(18) << "test"
            << " | " << std::right << std::setw(21) << "t/s"
            << " |\n"
            << "| " << std::string(32, '-') << " | " << std::string(10, '-')
            << " | " << std::string(10, '-') << " | " << std::string(18, '-')
            << " | " << std::string(21, '-') << " |\n";
  std::ostringstream size_text;
  size_text << std::fixed << std::setprecision(2) << size_gib << " GiB";
  const auto print_result = [&](std::string_view test_name,
                                const BenchStats& stats) {
    std::ostringstream throughput;
    throughput << std::fixed << std::setprecision(2) << stats.mean << " ± "
               << stats.stddev;
    std::cout << "| " << std::left << std::setw(32) << model_name << " | "
              << std::right << std::setw(10) << size_text.str() << " | "
              << std::left << std::setw(10) << "ROCm (HIP)"
              << " | " << std::right << std::setw(18) << test_name << " | "
              << std::right << std::setw(21) << throughput.str() << " |\n"
              << std::flush;
  };

  for (const std::size_t depth : options.n_depths) {
    for (const std::size_t prompt_length : options.n_prompts) {
      if (depth + prompt_length >= required_context) {
        std::cerr << "Error: prompt benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> runs;
      for (std::size_t repetition = 0; repetition <= options.repetitions;
           ++repetition) {
        session->Reset();
        if (depth > 0 &&
            !session->Sync(std::span(tokens).first(depth), &error)) {
          std::cerr << "Error preparing depth: " << error << '\n';
          return 1;
        }
        const auto start = std::chrono::steady_clock::now();
        if (!session->Sync(std::span(tokens).first(depth + prompt_length),
                           &error)) {
          std::cerr << "Error running prefill: " << error << '\n';
          return 1;
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        if (repetition > 0) {
          runs.push_back(static_cast<double>(prompt_length) / seconds);
        }
      }
      print_result(MakeTestName("pp", prompt_length, depth),
                   ComputeStats(runs));
    }

    for (const std::size_t generation_length : options.n_gens) {
      const std::size_t prefix_length = depth > 0 ? depth : 16;
      if (prefix_length + generation_length >= required_context) {
        std::cerr << "Error: generation benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> runs;
      for (std::size_t repetition = 0; repetition < options.repetitions;
           ++repetition) {
        session->Reset();
        if (!session->Sync(std::span(tokens).first(prefix_length), &error)) {
          std::cerr << "Error preparing generation: " << error << '\n';
          return 1;
        }
        std::vector<std::int32_t> generated;
        const auto start = std::chrono::steady_clock::now();
        if (use_mtp) {
          std::vector<sampling::TokenId> history;
          for (const std::int32_t token : session->Tokens()) {
            history.push_back(static_cast<sampling::TokenId>(token));
          }
          sampling::SamplerState sampler(options.sampling, history);
          std::string decode_error;
          while (generated.size() < generation_length) {
            Session::DecodeResult decoded;
            if (!session->DecodeStep(generation_length - generated.size(),
                                     sampler, &decoded, &decode_error)) {
              std::cerr << "Error running glm5-next MTP decode: "
                        << decode_error << '\n';
              return 1;
            }
            for (const std::int32_t next : decoded.tokens) {
              if (generated.size() < generation_length) {
                generated.push_back(next);
              }
            }
            if (decoded.stop) {
              break;
            }
          }
        } else {
          while (generated.size() < generation_length) {
            const auto logits = session->Logits();
            if (logits.empty()) {
              std::cerr << "Error running glm5-next decode: no logits\n";
              return 1;
            }
            const std::int32_t next =
                static_cast<std::int32_t>(std::max_element(logits.begin(),
                                                           logits.end()) -
                                          logits.begin());
            if (!session->Evaluate(next, &error)) {
              std::cerr << "Error running glm5-next decode: " << error << '\n';
              return 1;
            }
            generated.push_back(next);
          }
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        runs.push_back(static_cast<double>(generation_length) / seconds);
        if (options.verbose) {
          std::cerr << "glm5-next tg depth=" << depth
                    << " output_sha256=" << TokenSha256(generated) << " text="
                    << model->Decode(std::span(generated).first(
                           std::min<std::size_t>(generated.size(), 48)))
                    << '\n';
        }
      }
      print_result(MakeTestName("tg", generation_length, depth),
                   ComputeStats(runs));
    }
  }
  std::cout << '\n';
  return 0;
}

}  // namespace gufo::models::glm5_next

#endif  // defined(ENGINE_ENABLE_HIP)

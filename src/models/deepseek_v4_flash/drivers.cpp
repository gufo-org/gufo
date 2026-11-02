
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/crypto.hpp"
#include "gufo/ext/text.hpp"
#include "src/models/deepseek_v4_flash/chat_template.hpp"
#include "src/models/deepseek_v4_flash/dspark_sampler.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/testing/compare/logit_comparator.hpp"

namespace gufo::models::deepseek_v4_flash {
namespace {

constexpr std::size_t kDefaultContext = 4096;

#define HIP_CHECK(expr)                                               \
  do {                                                                \
    hipError_t err_ = (expr);                                         \
    if (err_ != hipSuccess) {                                         \
      std::cerr << "HIP error: " << hipGetErrorString(err_) << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;          \
      return 1;                                                       \
    }                                                                 \
  } while (0)

struct BenchStats {
  double mean = 0.0;
  double stddev = 0.0;
};

BenchStats ComputeStats(const std::vector<double>& values) {
  if (values.empty())
    return {};
  if (values.size() == 1)
    return {.mean = values.front(), .stddev = 0.0};
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / static_cast<double>(values.size());
  double sq_sum = 0.0;
  for (const double val : values) {
    sq_sum += (val - mean) * (val - mean);
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

std::vector<int> MakeDeepSeekBenchmarkTokens(
    const models::deepseek_v4_flash::Model& model, std::size_t count) {
  const auto pattern = model.Tokenize(
      "The quick brown fox jumps over the lazy dog. "
      "Strix Halo executes this deterministic benchmark sequence. ");
  if (pattern.empty()) {
    throw std::runtime_error("DeepSeek benchmark token pattern is empty");
  }
  std::vector<int> tokens(count);
  for (std::size_t index = 0; index < count; ++index) {
    tokens[index] = pattern[index % pattern.size()];
  }
  return tokens;
}

void PrintModelLoadTime(std::chrono::steady_clock::time_point start,
                        bool success = true) {
  if (start == std::chrono::steady_clock::time_point{})
    return;
  const auto end = std::chrono::steady_clock::now();
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cerr << "Model loaded in " << ms << " ms ("
            << (success ? "success" : "failed") << ").\n";
}

int RunDeepSeekBenchmarkImpl(
    const ext::TextBenchOptions& options,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start) {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: no HIP GPU is available for DeepSeek V4 Flash\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }

  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  const std::size_t max_depth = max_or_zero(options.n_depths);
  const std::size_t max_prompt = max_or_zero(options.n_prompts);
  const std::size_t max_generation = max_or_zero(options.n_gens);
  constexpr std::size_t context_limit = std::numeric_limits<int>::max() - 1;
  if (max_depth > context_limit || max_prompt > context_limit - max_depth ||
      max_generation > context_limit - std::max<std::size_t>(max_depth, 16) ||
      options.validate_prefill_tokens > context_limit) {
    std::cerr << "Error: requested DeepSeek benchmark context is too large\n";
    return 1;
  }
  const std::size_t required_context =
      std::max({std::size_t{4096}, max_depth + max_prompt + 1,
                std::max<std::size_t>(max_depth, 16) + max_generation + 1,
                options.validate_prefill_tokens + 1});
  const bool dspark = options.speculative_backend == "dspark";
  if (!options.speculative_backend.empty() && !dspark) {
    std::cerr << "Error: DeepSeek supports only --speculative dspark or off\n";
    return 1;
  }
  if (dspark && options.dspark_model_path.empty()) {
    std::cerr << "Error: --speculative dspark requires --dspark-model\n";
    return 1;
  }
  if (dspark && options.min_draft_tokens != 1) {
    std::cerr << "Error: DSpark uses model-owned adaptive drafting; custom "
                 "draft floors are unsupported\n";
    return 1;
  }

  // Both paths use the server's target distribution. Stochastic DSpark
  // preserves that distribution, with its own reproducible RNG trace.
  const bool sampled = !options.sampling.can_use_unmodified_argmax();
  const auto sampling_config = options.sampling;
  const auto sample_next = [&](models::deepseek_v4_flash::Session& session,
                               sampling::SamplerState& sampler,
                               std::string* error_msg) -> int {
    int token = session.TakePendingDsparkToken();
    if (token < 0) {
      const auto logits = session.Logits(error_msg);
      if (logits.empty())
        return -1;
      token = static_cast<int>(sampler.Sample(logits));
    }
    sampler.Accept(static_cast<sampling::TokenId>(token));
    return token;
  };

  std::string error;
  auto model = models::deepseek_v4_flash::Model::Load(
      options.model_path,
      models::deepseek_v4_flash::ModelOptions{
          .max_context = static_cast<std::uint32_t>(required_context),
          .dspark_model_path = dspark ? options.dspark_model_path : "",
      },
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating DeepSeek V4 Flash model: " << error << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  const auto tokens = MakeDeepSeekBenchmarkTokens(*model, required_context);
  if (options.validate_prefill_tokens != 0) {
    auto sequential =
        model->CreateSession(options.speculative_backend.empty()
                                 ? gufo::core::SessionMode::kAutoregressive
                                 : gufo::core::SessionMode::kSpeculative,
                             required_context, &error);
    auto batched =
        model->CreateSession(options.speculative_backend.empty()
                                 ? gufo::core::SessionMode::kAutoregressive
                                 : gufo::core::SessionMode::kSpeculative,
                             required_context, &error);
    const auto prefix =
        std::span(tokens).first(options.validate_prefill_tokens);
    if (!sequential || !batched || !sequential->Sync(prefix.first(1), &error) ||
        !batched->Sync(prefix, &error)) {
      std::cerr << "DeepSeek prefill validation failed: " << error << '\n';
      return 1;
    }
    for (const int token : prefix.subspan(1)) {
      if (!sequential->Evaluate(token, &error)) {
        std::cerr << "DeepSeek scalar reference failed: " << error << '\n';
        return 1;
      }
    }
    const auto reference = sequential->CopyLogits(&error);
    const auto candidate = batched->CopyLogits(&error);
    if (reference.empty() || reference.size() != candidate.size()) {
      std::cerr << "DeepSeek validation logit readback failed: " << error
                << '\n';
      return 1;
    }
    const auto comparison = testing::CompareLogits(reference, candidate);
    const std::size_t rank =
        1 + std::count_if(candidate.begin(), candidate.end(), [&](float value) {
          return value > candidate[comparison.reference_argmax];
        });
    std::cout << "[Prefill Validation] tokens=" << prefix.size()
              << " finite=" << comparison.finite
              << " scalar_winner_rank=" << rank
              << " rmse=" << comparison.root_mean_square_error
              << " cosine=" << comparison.cosine_similarity
              << " max_error=" << comparison.max_abs_diff << '\n';
    if (!comparison.finite || rank > 3 ||
        comparison.root_mean_square_error > 1.12F ||
        comparison.cosine_similarity < 0.979F ||
        comparison.max_abs_diff > 5.0F) {
      return 1;
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

  for (const std::size_t concurrency : options.concurrency) {
    if (concurrency > 1) {
      using models::deepseek_v4_flash::Session;
      for (const std::size_t depth : options.n_depths) {
        auto base = model->CreateSession(
            options.speculative_backend.empty()
                ? gufo::core::SessionMode::kAutoregressive
                : gufo::core::SessionMode::kSpeculative,
            static_cast<uint32_t>(required_context), &error);
        const std::size_t prefix = depth > 0 ? depth : 16;
        if (!base || !base->Sync(std::span(tokens).first(prefix), &error)) {
          std::cerr << "Error preparing concurrent depth: " << error << '\n';
          return 1;
        }
        auto snapshot = base->SaveSnapshot(&error);
        if (!snapshot) {
          std::cerr << "Error preparing concurrent snapshot: " << error << '\n';
          return 1;
        }
        base.reset();
        for (const bool generate : {false, true}) {
          for (const std::size_t count :
               generate ? options.n_gens : options.n_prompts) {
            std::vector<double> runs;
            for (std::size_t repeat = 0; repeat < options.repetitions;
                 ++repeat) {
              std::vector<std::unique_ptr<Session>> sessions;
              std::vector<std::vector<int>> generated(concurrency);
              const std::vector<sampling::TokenId> history(
                  tokens.begin(), tokens.begin() + prefix);
              std::vector<sampling::SamplerState> samplers(
                  concurrency,
                  sampling::SamplerState(sampling_config, history));
              for (std::size_t i = 0; i < concurrency; ++i) {
                auto session = model->CreateSession(
                    options.speculative_backend.empty()
                        ? gufo::core::SessionMode::kAutoregressive
                        : gufo::core::SessionMode::kSpeculative,
                    static_cast<uint32_t>(required_context), &error);
                if (!session ||
                    ((generate || depth > 0) &&
                     !session->RestoreSnapshot(*snapshot, &error))) {
                  std::cerr << "Error restoring concurrent depth: " << error
                            << '\n';
                  return 1;
                }
                sessions.push_back(std::move(session));
                generated[i].reserve(count);
              }
              const auto start = std::chrono::steady_clock::now();
              if (!generate) {
                for (auto& session : sessions) {
                  if (!session->Sync(std::span(tokens).first(depth + count),
                                     &error)) {
                    std::cerr << "Error in concurrent prefill: " << error
                              << '\n';
                    return 1;
                  }
                }
              } else if (dspark) {
                std::vector<std::vector<int>> emitted(concurrency);
                while (true) {
                  std::vector<models::deepseek_v4_flash::SessionDsparkBatchItem>
                      items;
                  std::vector<std::size_t> active;
                  std::vector<std::unique_ptr<
                      models::deepseek_v4_flash::DsparkSamplerBridge>>
                      bridges;
                  for (std::size_t i = 0; i < concurrency; ++i) {
                    if (generated[i].size() == count)
                      continue;
                    active.push_back(i);
                    const ds4_dspark_sampler* hook = nullptr;
                    if (sampled) {
                      bridges.push_back(
                          std::make_unique<
                              models::deepseek_v4_flash::DsparkSamplerBridge>(
                              samplers[i]));
                      hook = bridges.back()->hook();
                    }
                    items.push_back({.session = sessions[i].get(),
                                     .max_tokens = count - generated[i].size(),
                                     .max_draft_tokens = options.draft_tokens,
                                     .emitted = &emitted[i],
                                     .sampler = hook});
                  }
                  if (items.empty())
                    break;
                  if (!model->DsparkStepBatch(items, &error)) {
                    std::cerr << "Error in concurrent DSpark: " << error
                              << '\n';
                    return 1;
                  }
                  for (std::size_t slot = 0; slot < active.size(); ++slot) {
                    const auto i = active[slot];
                    if (emitted[i].empty() ||
                        emitted[i].size() > count - generated[i].size()) {
                      std::cerr << "Error: DSpark returned an invalid "
                                   "benchmark token count\n";
                      return 1;
                    }
                    if (sampled) {
                      samplers[i].SetRngState(bridges[slot]->rng_state());
                      for (const int token : emitted[i]) {
                        samplers[i].Accept(
                            static_cast<sampling::TokenId>(token));
                      }
                    }
                    generated[i].insert(generated[i].end(), emitted[i].begin(),
                                        emitted[i].end());
                  }
                }
              } else {
                for (std::size_t step = 0; step < count; ++step) {
                  std::vector<models::deepseek_v4_flash::SessionBatchItem>
                      items;
                  for (std::size_t i = 0; i < concurrency; ++i) {
                    const int token =
                        sampled ? sample_next(*sessions[i], samplers[i], &error)
                                : sessions[i]->SelectNext(0.0F, nullptr);
                    if (token < 0) {
                      std::cerr << "Error in concurrent sampling: " << error
                                << '\n';
                      return 1;
                    }
                    generated[i].push_back(token);
                    items.push_back(
                        {.session = sessions[i].get(), .token = token});
                  }
                  if (!model->EvaluateBatch(items, &error)) {
                    std::cerr << "Error in concurrent target decode: " << error
                              << '\n';
                    return 1;
                  }
                }
              }
              const double seconds =
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - start)
                      .count();
              runs.push_back(
                  static_cast<double>(count) *
                  (generate ? 1.0 : static_cast<double>(concurrency)) /
                  seconds);
              if (options.verbose && generate) {
                for (std::size_t i = 0; i < concurrency; ++i) {
                  const auto stats = sessions[i]->DsparkStatistics();
                  const auto* bytes =
                      reinterpret_cast<const uint8_t*>(generated[i].data());
                  std::cerr << "DeepSeek tg C=" << concurrency
                            << " depth=" << depth << " request=" << i
                            << " accepted=" << stats.support_accepted
                            << " drafted=" << stats.support_drafted
                            << " steps=" << stats.steps
                            << " skipped=" << stats.skipped << " output_sha256="
                            << crypto::Sha256Hex(std::span(
                                   bytes, generated[i].size() * sizeof(int)))
                            << '\n';
                }
              }
            }
            print_result(MakeTestName(generate ? "tg" : "pp", count, depth) +
                             " C" + std::to_string(concurrency),
                         ComputeStats(runs));
          }
        }
      }
      continue;
    }
    std::unique_ptr<models::deepseek_v4_flash::Session> prepared_session;
    std::size_t prepared_depth = 0;
    for (const std::size_t depth : options.n_depths) {
      std::unique_ptr<models::deepseek_v4_flash::SessionSnapshot> snapshot;
      if (depth > 0) {
        if (prepared_session == nullptr || depth < prepared_depth) {
          prepared_session = model->CreateSession(
              options.speculative_backend.empty()
                  ? gufo::core::SessionMode::kAutoregressive
                  : gufo::core::SessionMode::kSpeculative,
              static_cast<std::uint32_t>(required_context), &error);
          prepared_depth = 0;
        }
        if (prepared_session == nullptr ||
            !prepared_session->Sync(std::span(tokens.data(), depth), &error)) {
          std::cerr << "Error preparing DeepSeek depth " << depth << ": "
                    << error << '\n';
          return 1;
        }
        prepared_depth = depth;
        snapshot = prepared_session->SaveSnapshot(&error);
        if (snapshot == nullptr) {
          std::cerr << "Error snapshotting DeepSeek depth " << depth << ": "
                    << error << '\n';
          return 1;
        }
        if (options.verbose) {
          std::cerr << "Prepared DeepSeek depth " << depth
                    << " snapshot_bytes=" << snapshot->SizeBytes() << '\n';
        }
      }

      for (const std::size_t prompt_length : options.n_prompts) {
        if (depth + prompt_length >= required_context) {
          std::cerr << "Error: DeepSeek prompt benchmark exceeds context\n";
          return 1;
        }
        std::vector<double> runs;
        runs.reserve(options.repetitions);
        for (std::size_t repetition = 0; repetition < options.repetitions;
             ++repetition) {
          std::unique_ptr<models::deepseek_v4_flash::Session> local_session;
          models::deepseek_v4_flash::Session* session = prepared_session.get();
          if (snapshot == nullptr) {
            local_session = model->CreateSession(
                options.speculative_backend.empty()
                    ? gufo::core::SessionMode::kAutoregressive
                    : gufo::core::SessionMode::kSpeculative,
                static_cast<std::uint32_t>(required_context), &error);
            session = local_session.get();
            if (session != nullptr && depth > 0 &&
                !session->Sync(std::span(tokens).first(depth), &error)) {
              session = nullptr;
            }
          } else if (!prepared_session->RestoreSnapshot(*snapshot, &error)) {
            std::cerr << "Error restoring DeepSeek depth: " << error << '\n';
            return 1;
          }
          if (session == nullptr) {
            std::cerr << "Error creating DeepSeek session: " << error << '\n';
            return 1;
          }
          const auto start = std::chrono::steady_clock::now();
          if (!session->Sync(std::span(tokens.data(), depth + prompt_length),
                             &error)) {
            std::cerr << "Error running DeepSeek prefill: " << error << '\n';
            return 1;
          }
          const double seconds = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
          runs.push_back(static_cast<double>(prompt_length) / seconds);
          if (options.verbose) {
            std::cerr << "DeepSeek pp depth=" << depth
                      << " payload_bytes=" << session->PayloadBytes() << '\n';
          }
        }
        print_result(MakeTestName("pp", prompt_length, depth),
                     ComputeStats(runs));
      }

      for (const std::size_t generation_length : options.n_gens) {
        const std::size_t prefix_length = depth > 0 ? depth : 16;
        if (prefix_length + generation_length >= required_context) {
          std::cerr << "Error: DeepSeek generation benchmark exceeds context\n";
          return 1;
        }
        std::vector<double> runs;
        runs.reserve(options.repetitions);
        for (std::size_t repetition = 0; repetition < options.repetitions;
             ++repetition) {
          std::unique_ptr<models::deepseek_v4_flash::Session> local_session;
          models::deepseek_v4_flash::Session* session = prepared_session.get();
          if (snapshot == nullptr) {
            local_session = model->CreateSession(
                options.speculative_backend.empty()
                    ? gufo::core::SessionMode::kAutoregressive
                    : gufo::core::SessionMode::kSpeculative,
                static_cast<std::uint32_t>(required_context), &error);
            session = local_session.get();
            if (session != nullptr &&
                !session->Sync(std::span(tokens.data(), prefix_length),
                               &error)) {
              session = nullptr;
            }
          } else if (!prepared_session->RestoreSnapshot(*snapshot, &error)) {
            session = nullptr;
          }
          if (session == nullptr) {
            std::cerr << "Error preparing DeepSeek generation: " << error
                      << '\n';
            return 1;
          }

          std::vector<int> generated;
          generated.reserve(generation_length);
          const std::vector<sampling::TokenId> history(
              tokens.begin(), tokens.begin() + prefix_length);
          sampling::SamplerState sampler(sampling_config, history);
          const auto start = std::chrono::steady_clock::now();
          for (std::size_t step = 0; step < generation_length;) {
            if (dspark) {
              std::vector<int> emitted;
              std::optional<models::deepseek_v4_flash::DsparkSamplerBridge>
                  bridge;
              if (sampled)
                bridge.emplace(sampler);
              if (!session->DsparkStep(generation_length - step,
                                       options.draft_tokens, &emitted, &error,
                                       bridge ? bridge->hook() : nullptr) ||
                  emitted.empty()) {
                std::cerr << "Error running DSpark decode: " << error << '\n';
                return 1;
              }
              if (bridge) {
                sampler.SetRngState(bridge->rng_state());
                for (const int token : emitted) {
                  sampler.Accept(static_cast<sampling::TokenId>(token));
                }
              }
              generated.insert(generated.end(), emitted.begin(), emitted.end());
              step += emitted.size();
            } else {
              const int token = sampled ? sample_next(*session, sampler, &error)
                                        : session->SelectNext(0.0F, nullptr);
              if (token < 0 || !session->Evaluate(token, &error)) {
                std::cerr << "Error running DeepSeek decode: " << error << '\n';
                return 1;
              }
              generated.push_back(token);
              ++step;
            }
          }
          const double seconds = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
          runs.push_back(static_cast<double>(generation_length) / seconds);
          if (options.verbose) {
            const auto stats = session->DsparkStatistics();
            const auto* bytes =
                reinterpret_cast<const uint8_t*>(generated.data());
            std::cerr << "DeepSeek tg C=1 depth=" << depth
                      << " request=0 accepted=" << stats.support_accepted
                      << " drafted=" << stats.support_drafted
                      << " steps=" << stats.steps
                      << " skipped=" << stats.skipped << " output_sha256="
                      << crypto::Sha256Hex(
                             std::span(bytes, generated.size() * sizeof(int)))
                      << '\n';
          }
        }
        print_result(MakeTestName("tg", generation_length, depth),
                     ComputeStats(runs));
      }

      if (snapshot && !prepared_session->RestoreSnapshot(*snapshot, &error)) {
        std::cerr << "Error restoring final DeepSeek depth: " << error << '\n';
        return 1;
      }
    }
  }
  std::cout << '\n';
  return 0;
}

static void PrintTokenTrace(std::span<const int> tokens) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(tokens.size() * sizeof(std::uint32_t));
  for (const auto token : tokens) {
    const auto u32 = static_cast<std::uint32_t>(token);
    for (unsigned shift = 0; shift < 32; shift += 8) {
      bytes.push_back(static_cast<std::uint8_t>(u32 >> shift));
    }
  }
  std::cerr << "[TokenTrace]: count=" << tokens.size()
            << " sha256=" << crypto::Sha256Hex(bytes) << '\n';
}

static std::optional<gufo::ReasoningEffort> ParseReasoningEffort(
    std::string_view value) {
  if (value == "low") {
    return gufo::ReasoningEffort::kLow;
  }
  if (value == "medium") {
    return gufo::ReasoningEffort::kMedium;
  }
  if (value == "high") {
    return gufo::ReasoningEffort::kHigh;
  }
  if (value == "max") {
    return gufo::ReasoningEffort::kMax;
  }
  return std::nullopt;
}

static gufo::ReasoningOptions PromptReasoningOptions(
    const ext::TextPromptOptions& options) {
  gufo::ReasoningOptions reasoning;
  if (options.reasoning_mode == "on") {
    reasoning.enabled = true;
  } else if (options.reasoning_mode == "off") {
    reasoning.enabled = false;
  }
  if (options.reasoning_effort != "auto") {
    reasoning.effort = ParseReasoningEffort(options.reasoning_effort);
    reasoning.enabled = true;
  }
  if (options.preserve_thinking == "on") {
    reasoning.preserve_thinking = true;
  } else if (options.preserve_thinking == "off") {
    reasoning.preserve_thinking = false;
  }
  return reasoning;
}

static std::shared_ptr<models::deepseek_v4_flash::Model> LoadDeepSeekModel(
    const ext::TextPromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point load_start) {
  if (opt.force_cpu) {
    std::cerr << "DeepSeek V4 Flash is supported only by the ROCm backend\n";
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  std::string template_error;
  if (!models::deepseek_v4_flash::ValidateGgufTemplate(reader,
                                                       &template_error)) {
    std::cerr << "Unsupported DeepSeek chat template: " << template_error
              << '\n';
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  const bool dspark_requested = opt.speculative_backend == "dspark";
  if (!opt.speculative_backend.empty() && !dspark_requested) {
    std::cerr << "DeepSeek V4 Flash supports only --speculative dspark\n";
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  if (opt.speculative_backend == "dspark" && opt.dspark_model_path.empty()) {
    std::cerr << "--speculative dspark requires --dspark-model\n";
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  if (dspark_requested && opt.min_draft_tokens != 1) {
    std::cerr << "DSpark uses model-owned adaptive drafting; "
                 "--min-draft-tokens is unsupported\n";
    return nullptr;
  }

  std::string error;
  auto model = models::deepseek_v4_flash::Model::Load(
      opt.model_path,
      models::deepseek_v4_flash::ModelOptions{
          .max_context = static_cast<std::uint32_t>(kDefaultContext),
          .dspark_model_path = dspark_requested ? opt.dspark_model_path : "",
      },
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating DeepSeek V4 Flash model: " << error << '\n';
    PrintModelLoadTime(load_start, false);
    return nullptr;
  }
  PrintModelLoadTime(load_start);

  return model;
}

static int GenerateDeepSeekResponse(
    const ext::TextPromptOptions& opt,
    const std::shared_ptr<models::deepseek_v4_flash::Model>& model,
    models::deepseek_v4_flash::Session& session,
    std::span<const int> prompt_tokens, std::string* reply = nullptr) {
  std::string error;
  const auto emit = [&](int token) {
    const auto piece = model->DecodeToken(token);
    if (reply != nullptr)
      reply->append(piece);
    std::cout << piece << std::flush;
  };
  if (prompt_tokens.empty()) {
    std::cerr << "DeepSeek V4 Flash prompt produced no tokens\n";
    return 1;
  }
  if (prompt_tokens.size() >= kDefaultContext ||
      opt.max_tokens >= kDefaultContext - prompt_tokens.size()) {
    std::cerr << "DeepSeek V4 Flash prompt and output exceed the 4096-token "
                 "CLI context\n";
    return 1;
  }

  if (!session.Sync(prompt_tokens, &error)) {
    std::cerr << "DeepSeek V4 Flash prefill failed: " << error << '\n';
    return 1;
  }
  // Chat can extend an existing prefix. A new turn keeps its support KV but
  // must discard the previous turn's deferred draw and controller history.
  session.BeginRequest();

  if (opt.verbose) {
    std::cout << "[Engine]: DeepSeek V4 Flash ROCm (gfx1151)\n"
              << "Model: " << model->ModelName() << '\n'
              << "Prompt tokens: " << prompt_tokens.size() << '\n'
              << "Max tokens: " << opt.max_tokens << '\n'
              << "--- Generation Output ---\n";
  }

  std::vector<sampling::TokenId> sampling_history;
  sampling_history.reserve(prompt_tokens.size());
  for (const int token : prompt_tokens) {
    if (token < 0) {
      std::cerr << "DeepSeek V4 Flash produced an invalid prompt token\n";
      return 1;
    }
    sampling_history.push_back(static_cast<sampling::TokenId>(token));
  }
  sampling::SamplerState sampler(opt.sampling, sampling_history);

  const bool use_dspark = session.DsparkEnabled();

  const auto generation_start = std::chrono::steady_clock::now();
  std::size_t generated = 0;
  std::vector<int> generated_ids;
  if (use_dspark) {
    std::vector<int> emitted;
    bool stop = false;
    while (generated < opt.max_tokens && !stop) {
      std::optional<models::deepseek_v4_flash::DsparkSamplerBridge> bridge;
      if (!opt.sampling.can_use_unmodified_argmax())
        bridge.emplace(sampler);
      if (!session.DsparkStep(opt.max_tokens - generated, opt.draft_tokens,
                              &emitted, &error,
                              bridge ? bridge->hook() : nullptr, true)) {
        std::cerr << "\nDeepSeek V4 Flash speculative decode failed: " << error
                  << '\n';
        return 1;
      }
      if (emitted.empty())
        break;
      if (bridge)
        sampler.SetRngState(bridge->rng_state());
      for (const int token : emitted) {
        if (model->IsStopToken(token) || generated >= opt.max_tokens) {
          stop = true;
          break;
        }
        sampler.Accept(static_cast<sampling::TokenId>(token));
        emit(token);
        if (opt.verbose)
          generated_ids.push_back(token);
        ++generated;
      }
    }
  } else {
    for (; generated < opt.max_tokens; ++generated) {
      const auto logits = session.Logits(&error);
      if (logits.empty()) {
        std::cerr << "\nDeepSeek V4 Flash token selection failed: " << error
                  << '\n';
        return 1;
      }
      const int token = static_cast<int>(sampler.Sample(logits));
      if (model->IsStopToken(token)) {
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(token));
      emit(token);
      if (opt.verbose)
        generated_ids.push_back(token);
      // A reusable chat session must own the complete emitted prefix, like
      // the server and DSpark. Leaving its last token to the next bulk
      // prefill changes that call's boundary and numerical trajectory.
      if ((reply != nullptr || generated + 1 < opt.max_tokens) &&
          !session.Evaluate(token, &error)) {
        std::cerr << "\nDeepSeek V4 Flash decode failed: " << error << '\n';
        return 1;
      }
    }
  }
  std::cout << '\n';

  if (use_dspark) {
    const auto stats = session.DsparkStatistics();
    const double support_acceptance =
        stats.support_drafted != 0
            ? static_cast<double>(stats.support_accepted) /
                  static_cast<double>(stats.support_drafted)
            : 0.0;
    const double positional_acceptance =
        stats.support_drafted != 0
            ? static_cast<double>(stats.positional_accepted) /
                  static_cast<double>(stats.support_drafted)
            : 0.0;
    const double full_block_rate =
        stats.steps != 0 ? static_cast<double>(stats.full_blocks) /
                               static_cast<double>(stats.steps)
                         : 0.0;
    std::cerr << "[Speculative]: acceptance=" << support_acceptance
              << " drafted=" << stats.support_drafted
              << " accepted=" << stats.support_accepted
              << " verification_steps=" << stats.steps
              << " skipped=" << stats.skipped
              << " positional_acceptance=" << positional_acceptance
              << " positional_accepted=" << stats.positional_accepted
              << " full_block_rate=" << full_block_rate
              << " full_blocks=" << stats.full_blocks
              << " anchors=" << stats.anchors
              << " verifier_rows=" << stats.verifier_rows << '\n';
  }

  if (opt.verbose && generated > 0) {
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      generation_start)
            .count();
    std::cout << "Generated " << generated << " tokens on ROCm in " << seconds
              << "s (" << static_cast<double>(generated) / seconds
              << " tok/s)\n";
    PrintTokenTrace(generated_ids);
  }
  return 0;
}

static int RunDeepSeekPromptImpl(
    const ext::TextPromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point load_start) {
  auto model = LoadDeepSeekModel(opt, reader, load_start);
  if (!model)
    return 1;
  std::string error;
  auto session = model->CreateSession(
      opt.speculative_backend.empty() ? gufo::core::SessionMode::kAutoregressive
                                      : gufo::core::SessionMode::kSpeculative,
      static_cast<std::uint32_t>(kDefaultContext), &error);
  if (!session) {
    std::cerr << "DeepSeek session creation failed: " << error << '\n';
    return 1;
  }
  std::vector<int> prompt_tokens;
  if (opt.use_chat_template) {
    const auto reasoning = PromptReasoningOptions(opt);
    std::vector<models::deepseek_v4_flash::ChatMessage> messages;
    if (!opt.system_prompt.empty()) {
      messages.push_back({.role = "system",
                          .content = opt.system_prompt,
                          .reasoning_content = {},
                          .tool_calls = {},
                          .tool_call_id = {}});
    }
    messages.push_back({.role = "user",
                        .content = opt.prompt_text,
                        .reasoning_content = {},
                        .tool_calls = {},
                        .tool_call_id = {}});
    prompt_tokens = model->EncodeChat(
        messages,
        models::deepseek_v4_flash::ResolveDeepSeekChatOptions(reasoning));
  } else {
    prompt_tokens = model->Tokenize(opt.prompt_text);
  }
  return GenerateDeepSeekResponse(opt, model, *session, prompt_tokens);
}

static int RunDeepSeekChatImpl(
    const ext::TextPromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point load_start) {
  if (!opt.use_chat_template) {
    std::cerr << "DeepSeek interactive chat requires chat framing; use prompt "
                 "--raw for raw text\n";
    return 1;
  }
  auto model = LoadDeepSeekModel(opt, reader, load_start);
  if (!model)
    return 1;
  std::string error;
  auto session = model->CreateSession(
      opt.speculative_backend.empty() ? gufo::core::SessionMode::kAutoregressive
                                      : gufo::core::SessionMode::kSpeculative,
      static_cast<std::uint32_t>(kDefaultContext), &error);
  if (!session) {
    std::cerr << "DeepSeek session creation failed: " << error << '\n';
    return 1;
  }
  std::vector<models::deepseek_v4_flash::ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.push_back({.role = "system",
                       .content = opt.system_prompt,
                       .reasoning_content = {},
                       .tool_calls = {},
                       .tool_call_id = {}});
  }
  const auto reasoning = PromptReasoningOptions(opt);
  const auto chat_options =
      models::deepseek_v4_flash::ResolveDeepSeekChatOptions(reasoning);
  std::cout << "=== Gufo Interactive Chat (DeepSeek V4 Flash) ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";
  for (std::string input;;) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, input) || input == "exit" || input == "quit")
      break;
    if (input.empty())
      continue;
    history.push_back({.role = "user",
                       .content = input,
                       .reasoning_content = {},
                       .tool_calls = {},
                       .tool_call_id = {}});
    const auto tokens = model->EncodeChat(history, chat_options);
    std::cout << "<<< Assistant: ";
    std::string reply;
    if (GenerateDeepSeekResponse(opt, model, *session, tokens, &reply) != 0)
      return 1;
    models::deepseek_v4_flash::ChatMessage response{};
    response.role = "assistant";
    if (chat_options.enable_thinking) {
      constexpr std::string_view end = "</think>";
      const auto boundary = reply.find(end);
      response.reasoning_content = reply.substr(0, boundary);
      if (boundary != std::string::npos) {
        response.content = reply.substr(boundary + end.size());
      }
    } else {
      response.content = std::move(reply);
    }
    history.push_back(std::move(response));
    std::cout << '\n';
  }
  return 0;
}

}  // namespace

int RunDeepSeekBenchmark(
    const ext::TextBenchOptions& opt,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start) {
  return RunDeepSeekBenchmarkImpl(opt, reader, model_load_start);
}

int RunDeepSeekPrompt(const ext::TextPromptOptions& opt,
                      const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point model_load_start) {
  return RunDeepSeekPromptImpl(opt, reader, model_load_start);
}

int RunDeepSeekChat(const ext::TextPromptOptions& opt,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point model_load_start) {
  return RunDeepSeekChatImpl(opt, reader, model_load_start);
}

}  // namespace gufo::models::deepseek_v4_flash

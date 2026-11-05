
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
#include "src/families/qwen/control_tokens/control_tokens.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/families/qwen/vision/encoder.hpp"
#include "src/families/qwen/vision/prompt.hpp"
#include "src/models/qwen38_flash_next/bench_yarn.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/rope_scaling.hpp"
#include "src/testing/compare/logit_comparator.hpp"

namespace gufo::models::qwen38_flash_next {
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

std::vector<tokenization::TokenId> MakeBenchmarkTokens(
    std::size_t count, std::size_t start_pos = 0) {
  std::vector<tokenization::TokenId> tokens(count);
  for (std::size_t i = 0; i < count; ++i) {
    tokens[i] =
        static_cast<tokenization::TokenId>(((start_pos + i) % 1000) + 100);
  }
  return tokens;
}

std::string MakeTestName(std::string_view prefix, std::size_t count,
                         std::size_t depth) {
  std::string result = std::string(prefix) + std::to_string(count);
  if (depth > 0) {
    result += " @ d" + std::to_string(depth);
  }
  return result;
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

void PrintTokenTrace(std::span<const tokenization::TokenId> tokens) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(tokens.size() * sizeof(std::uint32_t));
  for (const auto token : tokens) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
      bytes.push_back(static_cast<std::uint8_t>(token >> shift));
    }
  }
  std::cerr << "[TokenTrace]: count=" << tokens.size()
            << " sha256=" << crypto::Sha256Hex(bytes) << '\n';
}

static std::optional<gufo::ReasoningEffort> ParseReasoningEffort(
    std::string_view value) {
  if (value == "minimal")
    return gufo::ReasoningEffort::kMinimal;
  if (value == "low")
    return gufo::ReasoningEffort::kLow;
  if (value == "medium")
    return gufo::ReasoningEffort::kMedium;
  if (value == "high")
    return gufo::ReasoningEffort::kHigh;
  if (value == "xhigh")
    return gufo::ReasoningEffort::kXHigh;
  if (value == "max")
    return gufo::ReasoningEffort::kMax;
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

void AttachImages(const ext::TextPromptOptions& opt,
                  tokenization::ChatMessage& user_msg) {
  for (const auto& path : opt.image_paths) {
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is) {
      throw std::runtime_error("Failed to read image: " + path);
    }
    const auto size = is.tellg();
    is.seekg(0, std::ios::beg);
    auto bytes = std::make_shared<std::vector<std::uint8_t>>(size);
    if (!is.read(reinterpret_cast<char*>(bytes->data()), size)) {
      throw std::runtime_error("Failed to read image: " + path);
    }
    user_msg.images.push_back(
        {0, std::move(bytes), tokenization::ChatMessage::ImageDetail::kAuto});
  }
}

std::shared_ptr<const models::qwen::vision::Prompt> PrepareVision(
    const ext::TextPromptOptions& opt,
    const tokenization::QwenTokenizer& tokenizer,
    std::span<const tokenization::ChatMessage> messages,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder) {
  if (!encoder)
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  const auto reasoning = PromptReasoningOptions(opt);
  return std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          tokenizer, messages, {},
          tokenization::ResolveQwenChatOptions(reasoning, opt.add_vision_id),
          encoder->identity(), kDefaultContext));
}

std::shared_ptr<models::qwen38_flash_next::Model> LoadFlashNextModel(
    const ext::TextPromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point load_start) {
  std::string error;
  if (opt.force_cpu ||
      (!opt.speculative_backend.empty() && opt.speculative_backend != "mtp") ||
      (opt.speculative_backend == "mtp" && opt.min_draft_tokens != 1)) {
    std::cerr << "Flash-Next requires ROCm and supports MTP with "
                 "--min-draft-tokens 1\n";
    return nullptr;
  }
  if (opt.use_chat_template &&
      !tokenization::QwenChatTemplate::ValidateGgufTemplate(reader, &error)) {
    std::cerr << "Unsupported Flash-Next chat template: " << error << '\n';
    return nullptr;
  }
  auto model = models::qwen38_flash_next::Model::Load(
      opt.model_path,
      {.max_context = kDefaultContext,
       .mtp_model_path =
           opt.speculative_backend == "mtp" ? opt.mtp_model_path : "",
       .max_draft_tokens = opt.draft_tokens,
       .vision_model_path = opt.vision_model_path},
      &error);
  PrintModelLoadTime(load_start, model != nullptr);
  if (!model)
    std::cerr << "Flash-Next load failed: " << error << '\n';
  return model;
}

int GenerateFlashNextResponse(const ext::TextPromptOptions& opt,
                              const models::qwen38_flash_next::Model& model,
                              models::qwen38_flash_next::Session& session,
                              std::span<const tokenization::TokenId> prompt,
                              std::string* reply = nullptr) {
  if (prompt.empty() || prompt.size() >= session.ContextSize() ||
      opt.max_tokens > session.ContextSize() - prompt.size()) {
    std::cerr
        << "Flash-Next prompt and output exceed the 4096-token CLI context\n";
    return 1;
  }
  const std::vector<std::int32_t> input(prompt.begin(), prompt.end());
  std::string error;
  if (!session.Sync(input, &error)) {
    std::cerr << "Flash-Next prefill failed: " << error << '\n';
    return 1;
  }
  sampling::SamplerState sampler(opt.sampling, prompt);
  std::vector<tokenization::TokenId> generated;
  const auto stats_before = session.Statistics();
  const auto start = std::chrono::steady_clock::now();
  while (generated.size() < opt.max_tokens) {
    models::qwen38_flash_next::Session::DecodeResult decoded;
    if (!session.DecodeStep(opt.max_tokens - generated.size(), sampler,
                            &decoded, &error)) {
      std::cerr << "Flash-Next decode failed: " << error << '\n';
      return 1;
    }
    for (const auto token : decoded.tokens) {
      const auto piece = model.TokenText(token);
      std::cout << piece << std::flush;
      if (reply)
        reply->append(piece);
      generated.push_back(static_cast<tokenization::TokenId>(token));
    }
    if (decoded.stop)
      break;
  }
  std::cout << '\n';
  if (opt.verbose) {
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::cerr << "Generated " << generated.size() << " tokens ("
              << generated.size() / seconds << " tok/s)\n";
    const auto& stats = session.Statistics();
    if (stats.cycles != stats_before.cycles) {
      std::cerr << "[Speculative]: cycles="
                << stats.cycles - stats_before.cycles
                << " drafted=" << stats.drafted - stats_before.drafted
                << " accepted=" << stats.accepted - stats_before.accepted
                << " lookup_proposed=" << stats.lookup - stats_before.lookup
                << " lookup_accepted="
                << stats.lookup_accepted - stats_before.lookup_accepted << '\n';
    }
    PrintTokenTrace(generated);
  }
  return 0;
}

int RunQwen38FlashNextBenchmarkImpl(
    const ext::TextBenchOptions& options,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start) {
  namespace qfn = models::qwen38_flash_next;
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: no HIP GPU is available for Qwen3.8-Flash-Next\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  if (std::max({max_or_zero(options.n_depths), max_or_zero(options.n_prompts),
                max_or_zero(options.n_gens), options.validate_prefill_tokens}) >
      std::numeric_limits<std::uint32_t>::max()) {
    std::cerr << "Error: benchmark workload exceeds the context range\n";
    return 1;
  }
  const std::size_t required_context = FlashNextBenchContext(options);
  const bool mtp = options.speculative_backend == "mtp";
  if (!options.speculative_backend.empty() && !mtp) {
    std::cerr << "Error: Qwen3.8-Flash-Next supports only --speculative mtp "
                 "or off\n";
    return 1;
  }
  if (mtp && options.mtp_model_path.empty()) {
    std::cerr << "Error: --speculative mtp requires --mtp-model\n";
    return 1;
  }
  if (options.concurrency != std::vector<std::size_t>{1}) {
    std::cerr << "Error: Flash-Next bench supports C1; use the serving "
                 "benchmark for concurrent requests\n";
    return 1;
  }
  if (mtp && options.min_draft_tokens != 1) {
    std::cerr << "Error: Flash-Next MTP requires --min-draft-tokens 1\n";
    return 1;
  }
  if (required_context > std::numeric_limits<std::uint32_t>::max()) {
    std::cerr << "Error: Flash-Next context is out of range\n";
    return 1;
  }

  // The same --enable-yarn rule as serve, applied to the context that sizes
  // the benchmark's own requests.
  const std::uint32_t qwen4exp_native_context =
      reader->GetMetadataUint32("qwen4exp.context_length").value_or(0);
  std::optional<qfn::RopeScaling> rope_scaling;
  std::string yarn_error;
  if (!ResolveFlashNextBenchYarn(options, qwen4exp_native_context,
                                 &rope_scaling, &yarn_error)) {
    std::cerr << "Error: Flash-Next benchmark: " << yarn_error << '\n';
    return 1;
  }

  std::string error;
  auto model = qfn::Model::Load(
      options.model_path,
      qfn::ModelOptions{
          .max_context = static_cast<std::uint32_t>(required_context),
          .mtp_model_path = mtp ? options.mtp_model_path : "",
          .max_draft_tokens = std::max<std::uint32_t>(1, options.draft_tokens),
          .rope_scaling = rope_scaling,
      },
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating Qwen3.8-Flash-Next model: " << error << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);
  if (rope_scaling) {
    std::cout << "[YaRN]: factor=" << rope_scaling->factor
              << " original_context=" << qwen4exp_native_context
              << " context=" << required_context << '\n';
  }

  // Repeat a fixed token pattern for reproducible timing. It uses far fewer
  // distinct PLE rows than varied requests.
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

  if (options.validate_prefill_tokens != 0) {
    // Batched prefill against one-token-at-a-time evaluation of the same
    // prefix; both run on the GPU, so this checks the batched kernels.
    auto sequential = model->CreateSession(
        options.speculative_backend.empty()
            ? gufo::core::SessionMode::kAutoregressive
            : gufo::core::SessionMode::kSpeculative,
        static_cast<std::uint32_t>(required_context), &error);
    auto batched = model->CreateSession(
        options.speculative_backend.empty()
            ? gufo::core::SessionMode::kAutoregressive
            : gufo::core::SessionMode::kSpeculative,
        static_cast<std::uint32_t>(required_context), &error);
    const auto prefix =
        std::span(tokens).first(options.validate_prefill_tokens);
    if (!sequential || !batched || !sequential->Sync(prefix.first(1), &error) ||
        !batched->Sync(prefix, &error)) {
      std::cerr << "Qwen3.8-Flash-Next prefill validation failed: " << error
                << '\n';
      return 1;
    }
    for (const std::int32_t token : prefix.subspan(1)) {
      if (!sequential->Evaluate(token, &error)) {
        std::cerr << "Qwen3.8-Flash-Next sequential reference failed: " << error
                  << '\n';
        return 1;
      }
    }
    const auto reference = sequential->Logits();
    const auto candidate = batched->Logits();
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

  for (const std::size_t depth : options.n_depths) {
    for (const std::size_t prompt_length : options.n_prompts) {
      if (depth + prompt_length >= required_context) {
        std::cerr << "Error: prompt benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> runs;
      // A full aligned prefix already warms the same chunk shapes and GEMM
      // plans. Other shapes still need an untimed pass before measurement.
      const auto chunk = model->PrefillCapacity();
      const bool warm_prefix = chunk > 0 && depth >= chunk &&
                               depth % chunk == 0 && prompt_length > 0 &&
                               prompt_length % chunk == 0;
      for (std::size_t repetition = warm_prefix ? 1 : 0;
           repetition <= options.repetitions; ++repetition) {
        auto session = model->CreateSession(
            options.speculative_backend.empty()
                ? gufo::core::SessionMode::kAutoregressive
                : gufo::core::SessionMode::kSpeculative,
            static_cast<std::uint32_t>(required_context), &error);
        if (!session ||
            (depth > 0 &&
             !session->Sync(std::span(tokens).first(depth), &error))) {
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
        auto session = model->CreateSession(
            options.speculative_backend.empty()
                ? gufo::core::SessionMode::kAutoregressive
                : gufo::core::SessionMode::kSpeculative,
            static_cast<std::uint32_t>(required_context), &error);
        if (!session ||
            !session->Sync(std::span(tokens).first(prefix_length), &error)) {
          std::cerr << "Error preparing generation: " << error << '\n';
          return 1;
        }
        std::vector<std::int32_t> generated;
        const std::vector<sampling::TokenId> history(
            tokens.begin(), tokens.begin() + prefix_length);
        sampling::SamplerState sampler(options.sampling, history);
        const auto start = std::chrono::steady_clock::now();
        while (generated.size() < generation_length) {
          qfn::Session::DecodeResult decoded;
          if (!session->DecodeStep(generation_length - generated.size(),
                                   sampler, &decoded, &error, false) ||
              decoded.tokens.empty()) {
            std::cerr << "Error running Flash-Next decode: " << error << '\n';
            return 1;
          }
          generated.insert(generated.end(), decoded.tokens.begin(),
                           decoded.tokens.end());
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        runs.push_back(static_cast<double>(generation_length) / seconds);
        if (options.verbose) {
          const auto stats = session->Statistics();
          std::cerr << "Qwen3.8-Flash-Next tg depth=" << depth
                    << " cycles=" << stats.cycles
                    << " drafted=" << stats.drafted
                    << " accepted=" << stats.accepted
                    << " lookup_proposed=" << stats.lookup
                    << " lookup_accepted=" << stats.lookup_accepted
                    << " output_sha256="
                    << crypto::Sha256Hex(
                           std::span(reinterpret_cast<const std::uint8_t*>(
                                         generated.data()),
                                     generated.size() * sizeof(std::int32_t)))
                    << " text="
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

int RunFlashNextPromptImpl(
    const ext::TextPromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point model_load_start) {
  std::string err;
  std::string rendered_prompt = opt.prompt_text;
  std::vector<tokenization::ContentSpan> content_spans;
  std::vector<tokenization::ChatMessage> messages;
  if (opt.use_chat_template) {
    if (!opt.system_prompt.empty()) {
      messages.push_back(
          {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
    }
    messages.push_back(
        {tokenization::ChatRole::kUser, opt.prompt_text, "", ""});
    const auto reasoning = PromptReasoningOptions(opt);
    const auto rendered = tokenization::QwenChatTemplate::Render(
        messages, {},
        tokenization::ResolveQwenChatOptions(reasoning, opt.add_vision_id),
        nullptr, nullptr, nullptr, &content_spans);
    if (rendered.has_value()) {
      rendered_prompt = *rendered;
    } else {
      std::cerr << "Error formatting chat template.\n";
      return 1;
    }
  }

  const auto encode_prompt = [&](const tokenization::QwenTokenizer& tokenizer) {
    return tokenization::QwenChatTemplate::EncodeRendered(
        tokenizer, rendered_prompt, 0, rendered_prompt.size(), content_spans,
        {});
  };

  auto model = LoadFlashNextModel(opt, reader, model_load_start);
  if (!model)
    return 1;
  auto session = model->CreateSession(
      opt.speculative_backend.empty() ? gufo::core::SessionMode::kAutoregressive
                                      : gufo::core::SessionMode::kSpeculative,
      kDefaultContext, &err);
  if (!session) {
    std::cerr << "Flash-Next session failed: " << err << '\n';
    return 1;
  }
  try {
    if (!opt.image_paths.empty()) {
      AttachImages(opt, messages.back());
      auto vision = PrepareVision(opt, model->tokenizer(), messages,
                                  model->VisionEncoder());
      session->ConfigureVision(vision);
      return GenerateFlashNextResponse(opt, *model, *session, vision->tokens);
    }
    const auto ids = encode_prompt(model->tokenizer());
    const std::vector<tokenization::TokenId> prompt(ids.begin(), ids.end());
    return GenerateFlashNextResponse(opt, *model, *session, prompt);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

}  // namespace

int RunQwen38FlashNextBenchmark(
    const ext::TextBenchOptions& opt,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start) {
  return RunQwen38FlashNextBenchmarkImpl(opt, reader, model_load_start);
}

int RunFlashNextPrompt(const ext::TextPromptOptions& opt,
                       const core::GgufReader& reader,
                       std::chrono::steady_clock::time_point model_load_start) {
  return RunFlashNextPromptImpl(opt, reader, model_load_start);
}

int RunFlashNextChat(const ext::TextPromptOptions& opt,
                     const core::GgufReader& reader,
                     std::chrono::steady_clock::time_point model_load_start) {
  auto model = LoadFlashNextModel(opt, reader, model_load_start);
  if (!model)
    return 1;
  std::string err;
  auto session = model->CreateSession(
      opt.speculative_backend.empty() ? gufo::core::SessionMode::kAutoregressive
                                      : gufo::core::SessionMode::kSpeculative,
      kDefaultContext, &err);
  if (!session) {
    std::cerr << "Flash-Next session failed: " << err << '\n';
    return 1;
  }
  const auto& tokenizer = model->tokenizer();
  auto vision_encoder = model->VisionEncoder();
  PrintModelLoadTime(model_load_start);

  std::cout << "=== Gufo Interactive Chat (qwen4exp) ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";

  std::vector<tokenization::ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.push_back(
        {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
  }

  bool first_turn = true;
  std::string user_input;
  while (true) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, user_input)) {
      break;
    }
    if (user_input == "exit" || user_input == "quit") {
      break;
    }
    if (user_input.empty()) {
      continue;
    }

    history.push_back({tokenization::ChatRole::kUser, user_input, "", ""});
    const auto reasoning = PromptReasoningOptions(opt);
    std::vector<tokenization::ContentSpan> content_spans;
    const auto rendered_prompt = tokenization::QwenChatTemplate::Render(
        history, {},
        tokenization::ResolveQwenChatOptions(reasoning, opt.add_vision_id),
        nullptr, nullptr, nullptr, &content_spans);
    if (!rendered_prompt.has_value()) {
      std::cerr << "Error formatting chat template.\n";
      return 1;
    }

    auto prompt_tokens = tokenization::QwenChatTemplate::EncodeRendered(
        tokenizer, *rendered_prompt, 0, rendered_prompt->size(), content_spans,
        {});

    std::cout << "<<< Assistant: ";
    std::string assistant_reply;

    try {
      if (first_turn)
        AttachImages(opt, history.back());
      first_turn = false;
      if (!opt.image_paths.empty()) {
        auto vision = PrepareVision(opt, tokenizer, history, vision_encoder);
        prompt_tokens = vision->tokens;
        session->ConfigureVision(vision);
      }
      if (GenerateFlashNextResponse(opt, *model, *session, prompt_tokens,
                                    &assistant_reply) != 0)
        return 1;
    } catch (const std::exception& exception) {
      std::cerr << "Generation failed: " << exception.what() << '\n';
      return 1;
    }

    std::cout << '\n';
    tokenization::ChatMessage reply{tokenization::ChatRole::kAssistant, ""};
    if (tokenization::ResolveQwenChatOptions(reasoning, opt.add_vision_id)
            .enable_thinking) {
      constexpr std::string_view end = "</think>";
      const auto boundary = assistant_reply.find(end);
      reply.thought = assistant_reply.substr(0, boundary);
      if (boundary != std::string::npos) {
        reply.content = assistant_reply.substr(boundary + end.size());
      }
    } else {
      reply.content = std::move(assistant_reply);
    }
    history.push_back(std::move(reply));
  }
  return 0;
}

}  // namespace gufo::models::qwen38_flash_next

#include <unistd.h>

#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "src/cli/serve/inference_backend.hpp"
#include "src/core/gguf_identity.hpp"
#include "src/core/image.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace {
using namespace gufo;
using Backend = server::InferenceBackend;

void Require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

std::string Lower(std::string value) {
  for (auto& character : value) {
    character =
        static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
  }
  return value;
}

server::ChatRequest ImageRequest(const std::filesystem::path& image) {
  server::ChatRequest request;
  // This 16-token recognition check evaluates the answer, not a thinking trace.
  request.reasoning.enabled = false;
  request.messages.emplace_back(tokenization::ChatRole::kUser,
                                "Describe the dominant color of this image, "
                                "then explain it in a full sentence.");
  request.messages.back().images.push_back(
      {0, std::make_shared<const std::vector<std::uint8_t>>(
              core::ReadImageFile(image))});
  return request;
}

void CheckConcurrent(Backend& backend,
                     std::span<const server::ChatRequest> requests,
                     std::span<const Backend::Result> expected,
                     const sampling::SamplingConfig& sampling,
                     bool timed_mtp = false) {
  std::vector<std::shared_ptr<Backend::GenerationRequest>> pending;
  for (const auto& request : requests)
    pending.push_back(backend.start_chat(request, 16, sampling));
  std::size_t width = 1;
  for (std::size_t i = 0; i < pending.size(); ++i) {
    const auto result = pending[i]->Wait();
    Require(result.tokens == expected[i].tokens,
            "concurrent image request changed token IDs");
    Require(result.draft_accepted_tokens <= result.draft_tokens,
            "image request accepted more tokens than it drafted");
    // Flash-Next's greedy controller measures batch cost and may select a
    // different depth at C4. Seeded sampling must replay both draws and depth.
    if (!timed_mtp || sampling.uses_random_sampling())
      Require(
          result.draft_tokens == expected[i].draft_tokens &&
              result.draft_accepted_tokens == expected[i].draft_accepted_tokens,
          "concurrent image request changed proposal/acceptance accounting");
    width = std::max(width, result.physical_execution_width);
  }
  Require(width > 1, "image test did not exercise a shared decode batch");
  std::cout << "image concurrency=" << requests.size()
            << " physical_width=" << width << '\n';
}

void CheckFlashIncrementalOracle(
    const std::shared_ptr<models::qwen38_flash_next::Model>& model,
    const server::ChatRequest& request, const Backend::Result& cached) {
  const auto prompt = std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          model->tokenizer(), request.messages, {},
          tokenization::ResolveQwenChatOptions(request.reasoning,
                                               request.add_vision_id),
          model->VisionEncoder()->identity(), model->MaxContext()));
  const std::vector<std::int32_t> tokens(prompt->tokens.begin(),
                                         prompt->tokens.end());
  std::string error;
  auto incremental = model->CreateSession(
      gufo::core::SessionMode::kAutoregressive, model->MaxContext(), &error);
  auto bulk = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                   model->MaxContext(), &error);
  Require(incremental && bulk, error);
  incremental->ConfigureVision(prompt);
  bulk->ConfigureVision(prompt);
  Require(incremental->Sync(
              std::span(tokens).first(cached.cached_prompt_tokens), &error),
          error);
  Require(incremental->Sync(tokens, &error) && bulk->Sync(tokens, &error),
          error);
  std::vector<tokenization::TokenId> generated;
  double max_kl = 0;
  float max_logit_error = 0;
  for (std::size_t i = 0; i < 16; ++i) {
    const auto left = incremental->Logits();
    const auto right = bulk->Logits();
    const float left_max = *std::max_element(left.begin(), left.end());
    const float right_max = *std::max_element(right.begin(), right.end());
    double left_sum = 0, right_sum = 0;
    for (std::size_t j = 0; j < left.size(); ++j) {
      Require(std::isfinite(left[j]) && std::isfinite(right[j]),
              "nonfinite Flash image logits");
      left_sum += std::exp(static_cast<double>(left[j] - left_max));
      right_sum += std::exp(static_cast<double>(right[j] - right_max));
      max_logit_error = std::max(max_logit_error, std::abs(left[j] - right[j]));
    }
    double kl = 0;
    for (std::size_t j = 0; j < left.size(); ++j) {
      const double log_p = left[j] - left_max - std::log(left_sum);
      const double log_q = right[j] - right_max - std::log(right_sum);
      kl += std::exp(log_p) * (log_p - log_q);
    }
    max_kl = std::max(max_kl, kl);
    const auto token = static_cast<std::int32_t>(
        std::max_element(left.begin(), left.end()) - left.begin());
    if (model->IsStopToken(token))
      break;
    generated.push_back(static_cast<tokenization::TokenId>(token));
    Require(
        incremental->Evaluate(token, &error) && bulk->Evaluate(token, &error),
        error);
  }
  std::cout << "Flash fresh incremental oracle exact="
            << (generated == cached.tokens)
            << " bulk_incremental_max_logit_error=" << max_logit_error
            << " max_KL_temperature1=" << max_kl << '\n';
  Require(generated == cached.tokens,
          "Flash snapshot changed incremental inference");
  Require(max_logit_error == 0,
          "Flash image continuation changed logits with prefill chunking");
}

void CheckQwenIncrementalOracle(
    const std::shared_ptr<const hip::QwenGpuModel>& model,
    const server::ChatRequest& request, const Backend::Result& previous,
    const Backend::Result& cached) {
  const auto prompt = std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          model->GetTokenizer(), request.messages, {},
          tokenization::ResolveQwenChatOptions(request.reasoning,
                                               request.add_vision_id),
          model->VisionEncoder()->identity(), 1024));
  hip::QwenGpuExecutor incremental(model, 1024);
  incremental.ConfigureVision(prompt, model->VisionEncoder());
  const auto tokens = std::span(prompt->tokens);
  // Replay the same execution history: the retained reply used decode
  // arithmetic, not matrix prefill. The separate cold comparison below
  // deliberately continues to check that stronger guarantee.
  (void)incremental.ForwardPromptBatch(tokens.first(previous.prompt_tokens));
  for (std::size_t i = previous.prompt_tokens; i < cached.cached_prompt_tokens;
       ++i)
    (void)incremental.ForwardToken(tokens[i], i);
  auto next = incremental.ForwardPromptBatch(
      tokens.subspan(cached.cached_prompt_tokens), cached.cached_prompt_tokens);
  std::vector<tokenization::TokenId> generated;
  for (std::size_t i = 0; i < cached.tokens.size(); ++i) {
    generated.push_back(next);
    next = incremental.ForwardToken(next, tokens.size() + i);
  }
  std::cout << "Qwen fresh live-history oracle exact="
            << (generated == cached.tokens)
            << " prefix=" << cached.cached_prompt_tokens
            << " total=" << tokens.size() << '\n';
  Require(generated == cached.tokens,
          "Qwen snapshot changed incremental inference");

  // Separate chunk invariance from replaying a generation history: BF16 BLAS
  // selected different reductions for the two batch shapes despite identical
  // normed rows, which amplified into different continuation tokens.
  incremental.Reset();
  (void)incremental.ForwardPromptBatch(tokens.first(previous.prompt_tokens));
  (void)incremental.ForwardPromptBatch(tokens.subspan(previous.prompt_tokens),
                                       previous.prompt_tokens);
  const auto split_logits = incremental.CopyLastLogits();
  const std::vector<float> expected(split_logits.begin(), split_logits.end());
  incremental.Reset();
  (void)incremental.ForwardPromptBatch(tokens);
  const auto cold_logits = incremental.CopyLastLogits();
  Require(std::ranges::equal(expected, cold_logits),
          "Qwen image prefill logits changed with chunk boundaries");
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 1)
    return 77;
  try {
    const bool disk_only =
        argc == 5 && std::string_view(argv[4]) == "--disk-only";
    Require(argc == 4 || disk_only,
            "usage: qwen_vision_serving_test MODEL DRAFT_OR_DASH "
            "IMAGE_DIRECTORY [--disk-only]");
    const std::string model_path(argv[1]);
    const std::string draft = std::string_view(argv[2]) == "-" ? "" : argv[2];
    const std::filesystem::path images(argv[3]);
    std::string error;
    auto reader = std::shared_ptr<const core::GgufReader>(
        core::GgufReader::OpenFile(model_path, &error));
    Require(reader != nullptr, error);
    constexpr std::uint32_t context = 1024;
    const bool flash =
        reader->GetMetadataString("general.architecture") == "qwen4exp";
    server::TextSpeculativeConfig speculative;
    speculative.backend = draft.empty()
                              ? server::TextSpeculativeBackend::kDisabled
                          : flash ? server::TextSpeculativeBackend::kMtp
                                  : server::TextSpeculativeBackend::kDFlash;
    speculative.draft_model_path = draft;
    std::shared_ptr<const hip::QwenGpuModel> qwen;
    std::shared_ptr<models::qwen38_flash_next::Model> qfn;
    if (flash) {
      qfn = models::qwen38_flash_next::Model::Load(
          model_path, {.max_context = context, .mtp_model_path = draft},
          &error);
      Require(qfn != nullptr && qfn->VisionEncoder() != nullptr, error);
    } else {
      auto encoder = models::qwen::vision::Encoder::Open(model_path, {}, 5120);
      Require(encoder != nullptr, "missing vision sidecar");
      qwen =
          hip::QwenGpuModel::CreateFromGguf(reader, &error, std::move(encoder));
      Require(qwen != nullptr, error);
    }
    const auto load = [&](bool use_spec,
                          const server::TextDiskCacheConfig& cache = {}) {
      auto backend = std::make_unique<Backend>();
      const auto mode =
          use_spec ? speculative : server::TextSpeculativeConfig{};
      auto mode_cache = cache;
      if (!use_spec)
        mode_cache.draft_model_artifact_fingerprint.clear();
      const bool loaded = flash ? backend->load(qfn, &error, context, 4, {}, {},
                                                mode, mode_cache)
                                : backend->load(qwen, &error, context, 4, {},
                                                {}, mode, mode_cache);
      Require(loaded, error);
      return backend;
    };
    std::array requests{
        ImageRequest(images / "red.png"), ImageRequest(images / "blue.png"),
        ImageRequest(images / "blue-wide.png"), server::ChatRequest{}};
    requests[3].messages.emplace_back(
        tokenization::ChatRole::kUser,
        "Continue counting from one, with commas between the numbers.");
    requests[3].reasoning.enabled = false;
    auto ar = load(false);
    sampling::SamplingConfig greedy;
    greedy.temperature = 0;
    sampling::SamplingConfig sampled;
    sampled.temperature = 0.8F;
    sampled.top_k = 30;
    sampled.top_p = 0.9F;
    sampled.min_p = 0.05F;
    sampled.seed = 47;
    std::array<Backend::Result, 4> references;
    std::unique_ptr<Backend> spec;
    if (!disk_only) {
      for (std::size_t i = 0; i < requests.size(); ++i) {
        references[i] = ar->chat(requests[i], 16, greedy);
        Require(!references[i].cache_hit,
                "different images shared a token-only cache entry");
        const auto replay = ar->chat(requests[i], 16, greedy);
        Require(replay.cache_hit && replay.prefill_tokens == 0 &&
                    replay.tokens == references[i].tokens,
                "image prefix replay is not exact");
        std::cout << "AR " << i << ": " << references[i].text << '\n';
      }
      Require(references[0].tokens != references[1].tokens &&
                  references[0].prompt_tokens == references[1].prompt_tokens,
              "same-size different images did not influence generation "
              "independently");
      Require(Lower(references[0].text).find("red") != std::string::npos &&
                  Lower(references[1].text).find("blue") != std::string::npos,
              "solid-color image recognition failed");
      ar.reset();
      ar = load(false);
      CheckConcurrent(*ar, requests, references, greedy);
      if (!draft.empty()) {
        spec = load(true);
        std::array<Backend::Result, 4> speculative_references;
        for (std::size_t i = 0; i < requests.size(); ++i) {
          speculative_references[i] = spec->chat(requests[i], 16, greedy);
          Require(speculative_references[i].tokens == references[i].tokens,
                  "greedy image speculation differs from autoregression");
          Require(speculative_references[i].draft_tokens > 0,
                  "image request bypassed speculative decoding");
        }
        spec.reset();
        spec = load(true);
        CheckConcurrent(*spec, requests, speculative_references, greedy, flash);
      }
      // Sampled replay and per-request RNG independence. Existing sampler
      // suites cover the distribution itself; this checks image-path
      // integration.
      for (auto* backend : {ar.get(), spec.get()}) {
        if (!backend)
          continue;
        std::array<Backend::Result, 4> expected;
        for (std::size_t i = 0; i < requests.size(); ++i) {
          expected[i] = backend->chat(requests[i], 16, sampled);
          Require(expected[i].tokens ==
                      backend->chat(requests[i], 16, sampled).tokens,
                  "seeded image generation is not reproducible");
        }
        CheckConcurrent(*backend, requests, expected, sampled,
                        flash && backend == spec.get());
        auto interrupted = requests[0];
        interrupted.messages.back().content =
            "Name the image color, then count from one to one hundred.";
        for (const auto& sampling : {greedy, sampled}) {
          const auto expected = backend->chat(interrupted, 8, sampling);
          std::size_t pieces = 0;
          const auto cancelled =
              backend->chat(interrupted, 128, sampling, {},
                            [&](std::string_view) { return ++pieces < 3; });
          Require(cancelled.cancelled,
                  "image cancellation fixture did not interrupt generation");
          const auto resumed = backend->chat(interrupted, 8, sampling);
          Require(resumed.cache_hit && resumed.prefill_tokens == 0 &&
                      resumed.tokens == expected.tokens,
                  "cancelled image request lost its exact reusable frontier");
        }
        std::cout << "image cancellation/resume exact, speculative="
                  << (backend == spec.get()) << '\n';
      }
    } else {
      references[0] = ar->chat(requests[0], 16, greedy);
    }
    // Cancellation above introduces a fifth independent prompt into four
    // slots. Re-establish the conversation before requiring live-prefix reuse.
    const auto root_replay = ar->chat(requests[0], 16, greedy);
    Require(root_replay.tokens == references[0].tokens,
            "image root changed after cancellation or cache eviction");
    references[0] = root_replay;
    auto continuation = requests[0];
    continuation.messages.emplace_back(tokenization::ChatRole::kAssistant,
                                       references[0].text);
    continuation.messages.emplace_back(
        tokenization::ChatRole::kUser,
        "What color did you see? Explain in one sentence.");
    auto multi_image = requests[0];
    multi_image.messages.back().content =
        "Reply with two color names only: first image, then second image.";
    multi_image.messages.back().images.push_back(
        requests[1].messages.back().images.front());
    const auto continued = ar->chat(continuation, 16, greedy);
    Require(continued.cache_hit && continued.cached_prompt_tokens > 0 &&
                continued.prefill_tokens > 0,
            "image continuation did not reuse its prefix");
    if (!disk_only) {
      const auto multiple = ar->chat(multi_image, 32, greedy);
      std::cout << "multiple images: " << multiple.text << '\n';
      const auto colors = Lower(multiple.text);
      Require(colors.find("red") != std::string::npos &&
                  colors.find("blue") != std::string::npos &&
                  colors.find("red") < colors.find("blue"),
              "multiple image order/content was lost");
    }
    ar.reset();
    spec.reset();
    // Fresh executors are independent cold oracles for a continued image chat.
    ar = load(false);
    const auto cold_continuation = ar->chat(continuation, 16, greedy);
    const bool continuation_exact =
        cold_continuation.tokens == continued.tokens;
    if (!continuation_exact) {
      std::cout << "continued cache: " << continued.text
                << "\ncontinued cold: " << cold_continuation.text << '\n';
      std::cout << "cached IDs:";
      for (auto token : continued.tokens)
        std::cout << ' ' << token;
      std::cout << "\ncold IDs:";
      for (auto token : cold_continuation.tokens)
        std::cout << ' ' << token;
      std::cout << '\n';
    }
    ar.reset();
    if (!flash)
      CheckQwenIncrementalOracle(qwen, continuation, references[0], continued);
    if (flash)
      CheckFlashIncrementalOracle(qfn, continuation, continued);

    std::array<char, 64> pattern{};
    const std::string temporary = "/tmp/gufo-vision-cache-XXXXXX";
    std::copy(temporary.begin(), temporary.end(), pattern.begin());
    Require(::mkdtemp(pattern.data()) != nullptr, "cannot create test cache");
    struct Cleanup {
      std::filesystem::path path;
      ~Cleanup() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
      }
    } cleanup{pattern.data()};
    server::TextDiskCacheConfig cache{
        .directory = cleanup.path,
        .capacity_bytes = std::size_t{2} << 30,
        .staging_capacity_bytes = std::size_t{512} << 20,
        .model_artifact_fingerprint = core::GgufIdentityHex(*reader)};
    if (!draft.empty()) {
      auto draft_reader = core::GgufReader::OpenFile(draft, &error);
      Require(draft_reader != nullptr, error);
      cache.draft_model_artifact_fingerprint =
          core::GgufIdentityHex(*draft_reader);
    }
    for (const bool use_spec : {false, true}) {
      if (use_spec && draft.empty())
        continue;
      cache.directory = cleanup.path / (use_spec ? "spec" : "ar");
      auto first = load(use_spec, cache);
      const auto red = first->chat(requests[0], 16, greedy);
      const auto sampled_red = first->chat(requests[0], 16, sampled);
      // Shutdown drains bounded persistence. A GPU-backed Qwen snapshot can
      // occupy most of the staging budget; a second concurrent enqueue is
      // allowed to miss. Qualify restoration independently of disk speed.
      first.reset();
      first = load(use_spec, cache);
      const auto blue = first->chat(requests[1], 16, greedy);
      const auto sampled_blue = first->chat(requests[1], 16, sampled);
      Require(
          red.cache_disk_queued_bytes > 0 && blue.cache_disk_queued_bytes > 0,
          "image snapshots did not reach disk");
      first.reset();
      {
        auto replay = load(use_spec, cache);
        for (const auto i : {0U, 1U}) {
          const auto result = replay->chat(requests[i], 16, sampled);
          Require(result.cache_disk_hit && result.prefill_tokens == 0 &&
                      result.tokens ==
                          (i == 0 ? sampled_red.tokens : sampled_blue.tokens),
                  "disk-restored sampled image replay differs");
        }
      }
      auto restored = load(use_spec, cache);
      for (const auto i : {0U, 1U}) {
        const auto result = restored->chat(requests[i], 16, greedy);
        std::cout << "disk image=" << i << " speculative=" << use_spec
                  << " hit=" << result.cache_disk_hit
                  << " prefill=" << result.prefill_tokens << " exact="
                  << (result.tokens == (i == 0 ? red.tokens : blue.tokens))
                  << '\n';
        Require(result.cache_disk_hit && result.prefill_tokens == 0 &&
                    result.tokens == (i == 0 ? red.tokens : blue.tokens),
                "disk-restored image state or identity differs");
      }
      const auto result = restored->chat(continuation, 16, greedy);
      Require(result.cached_prompt_tokens > red.prompt_tokens,
              "continued image check must reuse generated history");
      if (!flash)
        CheckQwenIncrementalOracle(qwen, continuation, red, result);
      if (flash || !disk_only)
        Require(result.tokens == continued.tokens,
                "disk-restored image continuation differs");
      std::cout << "disk image identity/layout replay exact, speculative="
                << use_spec << '\n';
    }
    if (!disk_only)
      Require(continuation_exact,
              "continued image cache differs from cold prefill");
    std::cout << (disk_only ? "image disk restoration: passed\n"
                            : "image AR/speculation, sampling, concurrency and "
                              "disk restoration: passed\n");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

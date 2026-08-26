#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {

using TokenId = strix::tokenization::TokenId;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

std::vector<TokenId> GenerateDirect(strix::hip::QwenGpuExecutor& executor,
                                    std::span<const TokenId> prompt_tokens,
                                    std::size_t max_tokens) {
  strix::models::GenerationOptions options;
  options.max_new_tokens = max_tokens;
  options.temperature = 0.0F;
  return executor.Generate(prompt_tokens, options);
}

void ExpectStableGpuMemory(std::size_t before, std::size_t after) {
  constexpr std::size_t tolerance = 16ULL * 1024ULL * 1024ULL;
  Expect(after + tolerance >= before,
         "request cleanup leaked more than 16 MiB of GPU memory");
}

}  // namespace

int main(int argc, const char* const* argv) {
  try {
    if (argc < 2) {
      std::cout << "SKIP: pass a Qwen GGUF path for the gfx1151 server test\n";
      return 77;
    }

    std::string error;
    auto reader_owner = strix::core::GgufReader::OpenFile(argv[1], &error);
    Expect(reader_owner != nullptr, error);
    const std::shared_ptr<const strix::core::GgufReader> reader(
        std::move(reader_owner));
    auto model = strix::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    Expect(model != nullptr, error);
    Expect(model->GetWeightRegionCount() == reader->GetMappedRegions().size(),
           "every mapped GGUF shard must have one shared GPU region");

    constexpr std::uint32_t context = 256;
    auto direct = strix::hip::QwenGpuExecutor::Create(model, &error, context);
    Expect(direct != nullptr, error);

    strix::server::InferenceBackend backend;
    Expect(backend.load(model, &error, context, 2), error);
    Expect(backend.model_id() == model->GetConfig().model_name,
           "HTTP model identifier");

    const std::string raw_prompt = "The capital of France is";
    const auto raw_prompt_tokens = model->GetTokenizer().Encode(raw_prompt);

    {
      auto snapshot_source =
          strix::hip::QwenGpuExecutor::Create(model, &error, context);
      Expect(snapshot_source != nullptr, error);
      const auto frontier =
          snapshot_source->ForwardPromptBatch(raw_prompt_tokens);
      auto snapshot = snapshot_source->SaveSnapshot(
          static_cast<std::uint32_t>(raw_prompt_tokens.size()));
      Expect(snapshot != nullptr && snapshot->PayloadBytes() > 0,
             "Qwen snapshot must own an accounted payload");
      const auto uninterrupted = snapshot_source->ForwardToken(
          frontier, static_cast<std::uint32_t>(raw_prompt_tokens.size()));

      for (int fork_index = 0; fork_index < 2; ++fork_index) {
        auto fork = strix::hip::QwenGpuExecutor::Create(model, &error, context);
        Expect(fork != nullptr, error);
        fork->RestoreSnapshot(*snapshot);
        const auto forked = fork->ForwardToken(
            frontier, static_cast<std::uint32_t>(raw_prompt_tokens.size()));
        Expect(forked == uninterrupted,
               "Qwen snapshot fork differs from uninterrupted execution");
      }
    }

    const auto direct_raw = GenerateDirect(*direct, raw_prompt_tokens, 2);
    const auto http_raw = backend.complete(raw_prompt, 2, 0.0F);
    Expect(http_raw.tokens == direct_raw,
           "raw HTTP and direct executor tokens differ");
    Expect(http_raw.text == model->GetTokenizer().Decode(direct_raw),
           "raw HTTP text must decode the exact generated tokens");
    Expect(http_raw.ttft_ms > 0.0, "raw HTTP TTFT must be reported");

    const std::vector<strix::tokenization::ChatMessage> messages = {
        {strix::tokenization::ChatRole::kSystem,
         "Answer with one short sentence.", "", ""},
        {strix::tokenization::ChatRole::kUser, "Name one primary color.", "",
         ""},
    };
    const auto rendered_chat =
        strix::tokenization::QwenChatTemplate::Render(messages);
    Expect(rendered_chat.has_value() && !rendered_chat->empty(),
           "CLI chat prompt rendering");
    const auto chat_prompt = model->GetTokenizer().Encode(*rendered_chat);
    const auto direct_chat = GenerateDirect(*direct, chat_prompt, 2);
    const auto http_chat = backend.chat(messages, 2, 0.0F);
    Expect(http_chat.tokens == direct_chat,
           "chat HTTP and direct executor tokens differ");
    Expect(!http_chat.cache_hit, "first chat request must be a cache miss");

    auto continued_messages = messages;
    continued_messages.emplace_back(strix::tokenization::ChatRole::kAssistant,
                                    http_chat.text);
    continued_messages.emplace_back(strix::tokenization::ChatRole::kUser,
                                    "Name a different primary color.");
    const auto rendered_continuation =
        strix::tokenization::QwenChatTemplate::Render(continued_messages);
    Expect(rendered_continuation.has_value(),
           "continued chat prompt rendering");
    const auto continuation_prompt =
        model->GetTokenizer().Encode(*rendered_continuation);
    const auto direct_continuation =
        GenerateDirect(*direct, continuation_prompt, 2);
    const auto http_continuation = backend.chat(continued_messages, 2, 0.0F);
    Expect(http_continuation.cache_hit,
           "continued chat must reuse the retained Qwen state");
    Expect(http_continuation.cached_prompt_tokens ==
               chat_prompt.size() + http_chat.tokens.size(),
           "Qwen cache reports the exact executed prefix");
    Expect(http_continuation.cached_prompt_tokens <
               http_continuation.prompt_tokens,
           "continued Qwen chat prefills only a suffix");
    Expect(http_continuation.tokens == direct_continuation,
           "cached Qwen continuation differs from cold full prefill");

    strix::server::ChatRequest concurrent_a({
        {strix::tokenization::ChatRole::kUser,
         "Continue this sequence with a few words: one, two, three,", "", ""},
    });
    concurrent_a.client_id = "batch-a";
    strix::server::ChatRequest concurrent_b({
        {strix::tokenization::ChatRole::kUser,
         "Complete this phrase with a few words: red, green, blue,", "", ""},
    });
    concurrent_b.client_id = "batch-b";
    const auto rendered_a =
        strix::tokenization::QwenChatTemplate::Render(concurrent_a.messages);
    const auto rendered_b =
        strix::tokenization::QwenChatTemplate::Render(concurrent_b.messages);
    Expect(rendered_a.has_value() && rendered_b.has_value(),
           "concurrent chat prompt rendering");
    const auto direct_a =
        GenerateDirect(*direct, model->GetTokenizer().Encode(*rendered_a), 4);
    const auto direct_b =
        GenerateDirect(*direct, model->GetTokenizer().Encode(*rendered_b), 4);

    auto pending_a = backend.start_chat(concurrent_a, 4, 0.0F);
    auto pending_b = backend.start_chat(concurrent_b, 4, 0.0F);
    const auto concurrent_result_a = pending_a->Wait();
    const auto concurrent_result_b = pending_b->Wait();
    Expect(concurrent_result_a.tokens == direct_a,
           "concurrent Qwen request A differs from isolated execution");
    Expect(concurrent_result_b.tokens == direct_b,
           "concurrent Qwen request B differs from isolated execution");
    Expect(concurrent_result_a.physical_execution_width == 2 &&
               concurrent_result_b.physical_execution_width == 2,
           "concurrent Qwen requests did not execute through W=2");
    Expect(concurrent_result_a.execution_plan == "batched-w2" &&
               concurrent_result_b.execution_plan == "batched-w2",
           "concurrent Qwen requests did not report the W=2 plan");

    std::size_t free_before = 0;
    std::size_t total_memory = 0;
    HIP_CHECK(hipMemGetInfo(&free_before, &total_memory));

    std::size_t cancel_checks = 0;
    const auto cancelled =
        backend.complete(raw_prompt, 4, 0.0F, [&cancel_checks] {
          ++cancel_checks;
          return cancel_checks > 2;
        });
    Expect(cancelled.cancelled, "request cancellation must be reported");
    Expect(cancelled.completion_tokens <= 1,
           "cancelled request emitted more than one token");

    bool callback_threw = false;
    std::size_t error_checks = 0;
    try {
      (void)backend.complete(raw_prompt, 4, 0.0F, [&error_checks] {
        ++error_checks;
        if (error_checks > 2) {
          throw std::runtime_error("injected request failure");
        }
        return false;
      });
    } catch (const std::runtime_error& exception) {
      callback_threw =
          std::string_view(exception.what()) == "injected request failure";
    }
    Expect(callback_threw, "injected request error must propagate");

    const auto recovered = backend.complete(raw_prompt, 2, 0.0F);
    Expect(recovered.tokens == direct_raw,
           "session was not reusable after cancellation and error");
    Expect(!recovered.cache_hit,
           "cancelled or failed Qwen state must not remain cached");

    HIP_CHECK(hipDeviceSynchronize());
    std::size_t free_after = 0;
    HIP_CHECK(hipMemGetInfo(&free_after, &total_memory));
    ExpectStableGpuMemory(free_before, free_after);

    std::cout << "HTTP HIP inference parity and cleanup tests passed.\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }
}

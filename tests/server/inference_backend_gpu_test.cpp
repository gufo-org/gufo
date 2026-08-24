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

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/server/inference_backend.hpp"
#include "src/models/qwen/chat_template.hpp"

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

    constexpr std::uint32_t context = 512;
    auto direct = strix::hip::QwenGpuExecutor::Create(model, &error, context);
    Expect(direct != nullptr, error);

    strix::server::InferenceBackend backend;
    Expect(backend.load(model, &error, context, 1), error);
    Expect(backend.model_id() == model->GetConfig().model_name,
           "HTTP model identifier");

    const std::string raw_prompt = "The capital of France is";
    const auto raw_prompt_tokens = model->GetTokenizer().Encode(raw_prompt);
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

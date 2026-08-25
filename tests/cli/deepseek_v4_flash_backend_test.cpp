#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace {

using Model = strix::models::deepseek_v4_flash::Model;
using ModelOptions = strix::models::deepseek_v4_flash::ModelOptions;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

std::vector<strix::tokenization::TokenId> GenerateDirect(
    const std::shared_ptr<Model>& model, std::span<const int> prompt,
    std::size_t max_tokens) {
  std::string error;
  auto session = model->CreateSession(512, &error);
  Expect(session != nullptr, error);
  Expect(session->Sync(prompt, &error), error);

  std::vector<strix::tokenization::TokenId> result;
  result.reserve(max_tokens);
  for (std::size_t index = 0; index < max_tokens; ++index) {
    const int token = session->SelectNext(0.0F, nullptr);
    Expect(token >= 0, "direct token selection");
    if (model->IsStopToken(token)) {
      break;
    }
    result.push_back(static_cast<strix::tokenization::TokenId>(token));
    if (index + 1 < max_tokens) {
      Expect(session->Evaluate(token, &error), error);
    }
  }
  return result;
}

}  // namespace

int main() {
  try {
    const char* model_path = std::getenv("STRIX_DEEPSEEK_V4_FLASH_MODEL");
    if (model_path == nullptr || model_path[0] == '\0') {
      std::cout << "SKIP: STRIX_DEEPSEEK_V4_FLASH_MODEL is not set\n";
      return 77;
    }

    std::string error;
    auto model = Model::Load(model_path,
                             ModelOptions{
                                 .max_context = 512,
                                 .prefill_chunk = 512,
                                 .power_percent = 100,
                             },
                             &error);
    Expect(model != nullptr, error);

    strix::server::InferenceBackend backend;
    Expect(backend.load(model, &error, 512, 1), error);
    Expect(backend.model_id() == model->ModelName(), "HTTP model identifier");

    const std::string raw_prompt = "The capital of France is";
    const auto direct_raw =
        GenerateDirect(model, model->Tokenize(raw_prompt), 2);
    const auto http_raw = backend.complete(raw_prompt, 2, 0.0F);
    Expect(http_raw.tokens == direct_raw, "raw direct/HTTP token parity");
    Expect(http_raw.ttft_ms > 0.0, "raw TTFT");

    const std::vector<strix::tokenization::ChatMessage> messages = {
        {strix::tokenization::ChatRole::kSystem,
         "Answer with one short sentence.", "", ""},
        {strix::tokenization::ChatRole::kUser, "Name one primary color.", "",
         ""},
    };
    const std::vector<strix::models::deepseek_v4_flash::ChatMessage>
        direct_messages = {
            {.role = "system", .content = "Answer with one short sentence."},
            {.role = "user", .content = "Name one primary color."},
        };
    const auto direct_chat =
        GenerateDirect(model, model->EncodeChat(direct_messages), 2);
    const auto http_chat = backend.chat(messages, 2, 0.0F);
    Expect(http_chat.tokens == direct_chat, "chat direct/HTTP token parity");

    std::size_t cancellation_checks = 0;
    const auto cancelled =
        backend.complete(raw_prompt, 4, 0.0F, [&cancellation_checks] {
          ++cancellation_checks;
          return cancellation_checks > 2;
        });
    Expect(cancelled.cancelled, "cancellation status");
    Expect(cancelled.completion_tokens <= 1, "cancellation token bound");

    const auto recovered = backend.complete(raw_prompt, 2, 0.0F);
    Expect(recovered.tokens == direct_raw, "session reuse after cancellation");

    std::cout << "DeepSeek V4 Flash HTTP parity test passed\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }
}

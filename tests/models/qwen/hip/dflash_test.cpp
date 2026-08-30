#include "src/models/qwen/hip/dflash.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/dflash_reference.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {

constexpr int kSkipped = 77;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

}  // namespace

int main(int argc, const char* const* argv) {
  try {
    if (argc < 3) {
      std::cout << "qwen_dflash_gpu_test: skipped "
                   "(pass base and DFlash GGUF paths)\n";
      return kSkipped;
    }

    std::string error;
    auto base_owner = gufo::core::GgufReader::OpenFile(argv[1], &error);
    Expect(base_owner != nullptr, error);
    auto dflash_owner = gufo::core::GgufReader::OpenFile(argv[2], &error);
    Expect(dflash_owner != nullptr, error);

    std::shared_ptr<const gufo::core::GgufReader> base_reader(
        std::move(base_owner));
    std::shared_ptr<const gufo::core::GgufReader> dflash_reader(
        std::move(dflash_owner));

    auto target_model =
        gufo::hip::QwenGpuModel::CreateFromGguf(base_reader, &error);
    Expect(target_model != nullptr, error);

    auto dflash_model = gufo::hip::QwenDFlashGpuModel::Create(
        dflash_reader, target_model, &error);
    Expect(dflash_model != nullptr, error);

    gufo::hip::QwenDFlashGpuDraftConfig config{
        .max_context = 512,
        .max_draft_tokens = 8,
    };
    auto backend = gufo::hip::QwenDFlashGpuDraftBackend::Create(dflash_model,
                                                                config, &error);
    Expect(backend != nullptr, error);

    Expect(backend->RequiresTargetHiddenStates(), "RequiresTargetHiddenStates");
    Expect(backend->Name() == "QwenDFlashGpuDraftBackend", "Name matches");

    const std::size_t feature_width =
        dflash_model->GetDFlashConfig().target_layer_ids.size() *
        target_model->GetConfig().hidden_size;
    Expect(feature_width > 0, "DFlash target feature width");
    const std::vector<gufo::tokenization::TokenId> prompt = {1, 2, 3};
    std::vector<float> prompt_features(prompt.size() * feature_width);
    for (std::size_t index = 0; index < prompt_features.size(); ++index) {
      prompt_features[index] =
          static_cast<float>(static_cast<int>(index % 31U) - 15) / 128.0F;
    }
    Expect(backend->PrimeTargetContext({
               .prompt_tokens = prompt,
               .prompt_hidden_states = prompt_features,
               .hidden_size = feature_width,
               .first_token = 4,
           }),
           "DFlash persistent source prime");
    std::vector<float> pending_features(feature_width);
    for (std::size_t index = 0; index < pending_features.size(); ++index) {
      pending_features[index] =
          static_cast<float>(static_cast<int>(index % 17U) - 8) / 64.0F;
    }
    backend->UpdateTargetHidden(pending_features);

    auto snapshot = backend->Snapshot();
    const std::size_t persistent_bytes = snapshot->PersistentPayloadBytes();
    std::vector<std::uint8_t> payload(persistent_bytes);
    Expect(snapshot->SerializePersistent(payload) == persistent_bytes,
           "DFlash persistent serializer byte count");

    auto corrupt_backend = gufo::hip::QwenDFlashGpuDraftBackend::Create(
        dflash_model, config, &error);
    Expect(corrupt_backend != nullptr, error);
    auto corrupt_payload = payload;
    corrupt_payload.front() ^= 0xFFU;
    bool rejected_corruption = false;
    try {
      corrupt_backend->RestorePersistentSnapshot(corrupt_payload);
    } catch (const std::invalid_argument&) {
      rejected_corruption = true;
    }
    Expect(rejected_corruption,
           "DFlash persistent restore rejects a malformed header");

    auto restored = gufo::hip::QwenDFlashGpuDraftBackend::Create(
        dflash_model, config, &error);
    Expect(restored != nullptr, error);
    restored->RestorePersistentSnapshot(payload);

    const std::vector<gufo::tokenization::TokenId> continued_prompt = {1, 2, 3,
                                                                       4};
    const auto uninterrupted =
        backend->Propose(continued_prompt, continued_prompt.size(), 4);
    const auto restarted =
        restored->Propose(continued_prompt, continued_prompt.size(), 4);
    Expect(restarted.tokens == uninterrupted.tokens,
           "DFlash persistent restore preserves exact draft proposals");
    backend->AcceptFeedback(
        {}, uninterrupted.tokens.empty() ? 0 : uninterrupted.tokens.front());
    restored->AcceptFeedback(
        {}, restarted.tokens.empty() ? 0 : restarted.tokens.front());

    std::cout << "qwen_dflash_gpu_test: ALL TESTS PASSED\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "DFlash GPU test exception: " << ex.what() << '\n';
    return 1;
  }
}

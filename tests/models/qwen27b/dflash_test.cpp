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
    const char* base_path =
        argc > 1 ? argv[1] : std::getenv("GUFO_QWEN27B_MODEL");
    const char* draft_path =
        argc > 2 ? argv[2] : std::getenv("GUFO_QWEN27B_DFLASH_MODEL");
    if (base_path == nullptr || draft_path == nullptr) {
      std::cout << "qwen_dflash_gpu_test: skipped "
                   "(pass base and DFlash GGUF paths)\n";
      return kSkipped;
    }

    std::string error;
    auto base_owner = gufo::core::GgufReader::OpenFile(base_path, &error);
    Expect(base_owner != nullptr, error);
    auto dflash_owner = gufo::core::GgufReader::OpenFile(draft_path, &error);
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

    // A large logical context must not reserve or serialize expired history.
    // Restore at the window boundary, overwrite wrapped slots, and replay.
    const auto window = dflash_model->GetDFlashConfig().sliding_window;
    const auto capacity = std::min<std::uint32_t>(
        262144, dflash_model->GetConfig().context_length);
    auto executor = gufo::hip::QwenDFlashGpuExecutor::Create(dflash_model,
                                                             capacity, &error);
    Expect(executor != nullptr, error);
    const auto& draft_config = dflash_model->GetConfig();
    const std::size_t expected_history_bytes =
        2ULL * dflash_model->GetDFlashConfig().num_layers * window *
        draft_config.num_key_value_heads * draft_config.head_dim *
        sizeof(float);
    Expect(executor->StateBytes() == expected_history_bytes,
           "draft history allocation is bounded by its attention window");
    std::vector<float> features((window + 7ULL) * feature_width);
    for (std::size_t index = 0; index < features.size(); ++index) {
      features[index] =
          static_cast<float>(static_cast<int>(index % 37U) - 18) / 128.0F;
    }
    const std::span<const float> rows(features);
    Expect(executor->InjectTargetContext(rows.first(window * feature_width), 0,
                                         window),
           "inject through window boundary");
    auto boundary = executor->SaveSnapshot();
    Expect(executor->InjectTargetContext(rows.subspan(window * feature_width),
                                         window, 7),
           "inject across ring wrap");
    auto wrapped = executor->SaveSnapshot();
    Expect(wrapped->PayloadBytes() == expected_history_bytes,
           "snapshot excludes expired history");
    std::vector<std::uint8_t> wrapped_payload(
        wrapped->PersistentPayloadBytes());
    Expect(
        wrapped->SerializePersistent(wrapped_payload) == wrapped_payload.size(),
        "serialize wrapped history");
    std::vector<float> confidences;
    const auto wrapped_proposal =
        executor->ForwardBlock(4, window + 7, 4, 0.0F, {}, &confidences);
    executor->RestoreSnapshot(*boundary);
    Expect(executor->InjectTargetContext(rows.subspan(window * feature_width),
                                         window, 7),
           "replay after in-memory restore");
    auto replay = executor->SaveSnapshot();
    std::vector<std::uint8_t> replay_payload(replay->PersistentPayloadBytes());
    Expect(replay->SerializePersistent(replay_payload) == replay_payload.size(),
           "serialize replayed history");
    Expect(replay_payload == wrapped_payload,
           "wrapped KV replay is byte exact");
    executor->Reset();
    executor->RestorePersistentSnapshot(wrapped_payload);
    std::vector<float> replay_confidences;
    Expect(executor->ForwardBlock(4, window + 7, 4, 0.0F, {},
                                  &replay_confidences) == wrapped_proposal &&
               confidences == replay_confidences,
           "persistent ring restore preserves proposals and confidence");
    Expect(!executor->InjectTargetContext({}, capacity + 1, 0),
           "reject out-of-range injection");
    std::cout << "draft context=" << capacity
              << " history_bytes=" << executor->StateBytes()
              << " snapshot_bytes=" << wrapped->PayloadBytes() << '\n';

    std::cout << "qwen_dflash_gpu_test: ALL TESTS PASSED\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "DFlash GPU test exception: " << ex.what() << '\n';
    return 1;
  }
}

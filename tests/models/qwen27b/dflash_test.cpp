#include "src/models/qwen/hip/dflash.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/json.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/dflash_weights.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {

constexpr int kSkipped = 77;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void TestLengthController() {
  using gufo::speculative::DFlashDraftPolicy;
  using gufo::speculative::DFlashLengthController;
  DFlashLengthController fixed(DFlashDraftPolicy::kFixed, 7);
  fixed.Observe(0, 7);
  Expect(fixed.Choose(100) == 7 && fixed.Choose(3) == 3 && fixed.Choose(0) == 0,
         "fixed blocks obey only the configured and remaining budgets");
  for (const bool q8_target : {false, true}) {
    DFlashLengthController adaptive(DFlashDraftPolicy::kAdaptive, 7, q8_target);
    for (int round = 0; round < 32; ++round)
      adaptive.Observe(0, adaptive.Choose(7));
    Expect(adaptive.Choose(7) == 1, "rejections reduce wasted verification");
    for (int round = 0; round < 32; ++round) {
      const auto drafted = adaptive.Choose(7);
      adaptive.Observe(drafted, drafted);
    }
    Expect(adaptive.Choose(7) == 7,
           "censored full acceptance probes upward instead of getting stuck");
    const auto saved = adaptive.State();
    adaptive.Reset();
    Expect(adaptive.State() != saved, "new requests reset learned acceptance");
    adaptive.Restore(saved);
    Expect(adaptive.Choose(7) == 7 && adaptive.Choose(2) == 2,
           "restored decisions retain history and obey the output budget");
    for (const float invalid : {-1.0F, 8.0F, INFINITY, NAN}) {
      bool rejected = false;
      try {
        adaptive.Restore(invalid);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      Expect(rejected && adaptive.State() == saved,
             "malformed controller state cannot mutate the decision history");
    }
  }
}

void TestConcurrentBlocks(
    const std::shared_ptr<const gufo::hip::QwenGpuModel>& target_model,
    const std::shared_ptr<const gufo::hip::QwenDFlashGpuModel>& draft_model) {
  using namespace gufo::hip;
  using gufo::speculative::DraftProposal;
  using Token = gufo::tokenization::TokenId;
  std::string error;
  auto target = QwenGpuExecutor::Create(target_model, &error, 128);
  Expect(target != nullptr, error);
  const auto& config = draft_model->GetDFlashConfig();
  target->SetPromptHiddenCapture(true, config.target_layer_ids);
  auto prompt = target->GetTokenizer().Encode(
      "Virtual memory gives every process a separate address space. The "
      "operating system maps virtual pages to physical memory, handles page "
      "faults, and keeps unrelated processes from changing each other's data.");
  Expect(prompt.size() >= 25, "concurrent draft reference prompt");
  prompt.resize(25);
  (void)target->ForwardPromptBatch(prompt);
  const auto captured = target->GetPromptHiddenStates();
  const std::vector<float> features(captured.begin(), captured.end());
  const auto feature_width =
      config.target_layer_ids.size() * target_model->GetConfig().hidden_size;
  target.reset();
  std::array<std::unique_ptr<QwenDFlashGpuExecutor>, 8> executors;
  std::array<std::uint32_t, 8> positions{};
  std::array<std::array<float, 7>, 8> uniforms{};
  const std::array<float, 8> temperatures{0.0F, 0.2F, 0.8F, 1.3F,
                                          2.0F, 0.0F, 0.7F, 1.0F};
  const auto serial_memory =
      QwenDFlashGpuExecutor::EstimateMemoryUsage(*draft_model, 128);
  const auto batch_memory =
      QwenDFlashGpuExecutor::EstimateMemoryUsage(*draft_model, 128, 8);
  for (std::size_t index = 0; index < executors.size(); ++index) {
    executors[index] = QwenDFlashGpuExecutor::Create(draft_model, 128, &error);
    Expect(executors[index] != nullptr, error);
    const auto memory = executors[index]->GetMemoryUsage();
    Expect(memory.request_state_bytes == serial_memory.request_state_bytes &&
               memory.temporary_scratch_bytes ==
                   serial_memory.temporary_scratch_bytes,
           "draft memory estimate matches allocated state and scalar scratch");
    positions[index] = 3U * static_cast<std::uint32_t>(index + 1U);
    Expect(executors[index]->InjectTargetContext(
               std::span(features).first(positions[index] * feature_width), 0,
               positions[index]),
           "independent concurrent draft context");
    for (std::size_t row = 0; row < uniforms[index].size(); ++row)
      uniforms[index][row] =
          (static_cast<float>((index * 13U + row * 7U) % 101U) + 0.5F) / 101.0F;
  }
  const auto exact = [](std::span<const float> actual,
                        std::span<const float> expected) {
    return actual.size() == expected.size() &&
           (actual.empty() || std::memcmp(actual.data(), expected.data(),
                                          actual.size_bytes()) == 0);
  };
  for (const bool full_block : {false, true}) {
    std::array<DraftProposal, 8> expected;
    std::array<std::map<std::string, std::vector<float>>, 8> traces;
    std::array<std::uint32_t, 8> counts{};
    for (std::size_t index = 0; index < executors.size(); ++index) {
      counts[index] =
          full_block ? 7U : 1U + static_cast<std::uint32_t>(index % 7U);
      auto& proposal = expected[index];
      proposal.start_pos = positions[index];
      proposal.candidates_per_token =
          temperatures[index] > 0.0F ? config.selector_top_k : 0;
      const DFlashTrace trace = [&](std::string_view name,
                                    std::span<const float> values) {
        traces[index].emplace(std::string(name),
                              std::vector<float>(values.begin(), values.end()));
      };
      proposal.tokens = executors[index]->ForwardBlock(
          prompt[positions[index]], positions[index], counts[index],
          temperatures[index], uniforms[index], nullptr,
          temperatures[index] > 0.0F ? &proposal.candidate_ids : nullptr,
          temperatures[index] > 0.0F ? &proposal.candidate_probabilities
                                     : nullptr,
          trace);
    }
    for (const std::size_t width : {2U, 4U, 6U, 8U}) {
      std::vector<QwenDFlashBlockRequest> requests;
      std::vector<std::size_t> order;
      std::array<std::set<std::string>, 8> visited;
      for (std::size_t row = 0; row < width; ++row) {
        const auto index = (row + width / 2U) % executors.size();
        order.push_back(index);
        requests.push_back(
            {executors[index].get(), prompt[positions[index]], positions[index],
             counts[index], temperatures[index], uniforms[index],
             [&, index](std::string_view name, std::span<const float> values) {
               const auto found = traces[index].find(std::string(name));
               Expect(
                   found != traces[index].end(),
                   "concurrent draft trace stage exists in scalar reference");
               Expect(exact(values, found->second),
                      "concurrent draft differs at " + std::string(name) +
                          " request " + std::to_string(index) + " width " +
                          std::to_string(width));
               visited[index].insert(std::string(name));
             }});
      }
      const auto actual = QwenDFlashGpuExecutor::ForwardBlockBatch(requests);
      Expect(actual.size() == width, "concurrent draft result count");
      const auto memory = requests.front().executor->GetMemoryUsage();
      Expect(
          memory.request_state_bytes == serial_memory.request_state_bytes &&
              memory.temporary_scratch_bytes >
                  serial_memory.temporary_scratch_bytes &&
              memory.temporary_scratch_bytes <=
                  batch_memory.temporary_scratch_bytes,
          "concurrent draft workspace is counted within its admission bound");
      if (full_block && width == 8)
        Expect(memory.TotalBytes() == batch_memory.TotalBytes(),
               "full concurrent draft memory matches its admission estimate");
      for (std::size_t row = 0; row < width; ++row) {
        const auto index = order[row];
        Expect(visited[index].size() == traces[index].size(),
               "every scalar draft stage was checked in the batch");
        const auto& reference = expected[index];
        Expect(actual[row].start_pos == reference.start_pos &&
                   actual[row].tokens == reference.tokens &&
                   actual[row].candidate_ids == reference.candidate_ids &&
                   actual[row].candidates_per_token ==
                       reference.candidates_per_token &&
                   exact(actual[row].candidate_probabilities,
                         reference.candidate_probabilities),
               "concurrent draft tokens and actual selector distributions");
        Expect(executors[index]->GetInjectedContextLength() == positions[index],
               "drafting cannot commit another request's context");
      }
    }
  }
  std::cout
      << "DFlash2 C2/C4/C6/C8: exact layers, logits, selector probabilities; "
         "ragged/full blocks, distinct positions/temperatures/draws\n";
  for (auto& executor : executors)
    executor.reset();
  std::array<std::unique_ptr<QwenDFlashGpuDraftBackend>, 8> backends;
  std::array<std::vector<Token>, 8> sequences;
  std::array<std::uint64_t, 8> rng{};
  for (std::size_t index = 0; index < backends.size(); ++index) {
    const QwenDFlashGpuDraftConfig policy{
        .max_context = 128,
        .max_draft_tokens = 7,
        .policy = index % 2 == 0
                      ? gufo::speculative::DFlashDraftPolicy::kAdaptive
                      : gufo::speculative::DFlashDraftPolicy::kFixed};
    backends[index] =
        QwenDFlashGpuDraftBackend::Create(draft_model, policy, &error);
    Expect(backends[index] != nullptr, error);
    Expect(backends[index]->PrimeTargetContext(
               {std::span(prompt).first(positions[index]),
                std::span(features).first(positions[index] * feature_width),
                feature_width, prompt[positions[index]]}),
           "concurrent backend prime");
    sequences[index].assign(prompt.begin(),
                            prompt.begin() + positions[index] + 1U);
    rng[index] = 191U + 31U * index;
  }
  const auto payload = [](const QwenDFlashGpuDraftBackend& backend) {
    auto snapshot = backend.Snapshot();
    std::vector<std::uint8_t> bytes(snapshot->PersistentPayloadBytes());
    Expect(snapshot->SerializePersistent(bytes) == bytes.size(),
           "concurrent backend snapshot size");
    return bytes;
  };
  for (std::size_t round = 0; round < 2; ++round) {
    std::array<DraftProposal, 8> expected;
    std::array<std::vector<std::uint8_t>, 8> expected_state;
    auto expected_rng = rng;
    std::vector<gufo::speculative::DraftProposalRequest> requests;
    std::array<std::size_t, 8> accepted{};
    for (std::size_t index = 0; index < backends.size(); ++index) {
      auto& backend = *backends[index];
      const auto saved = backend.Snapshot();
      const auto limit = 1U + static_cast<std::uint32_t>(index % 7U);
      expected[index] =
          temperatures[index] > 0.0F
              ? backend.ProposeSampled(sequences[index], positions[index],
                                       limit, temperatures[index],
                                       &expected_rng[index])
              : backend.Propose(sequences[index], positions[index], limit);
      accepted[index] = std::min(index % 4U, expected[index].tokens.size());
      backend.AcceptFeedback(
          std::span(expected[index].tokens).first(accepted[index]), 4);
      expected_state[index] = payload(backend);
      backend.RestoreSnapshot(*saved);
      requests.push_back({&backend, sequences[index], positions[index], limit,
                          temperatures[index],
                          temperatures[index] > 0.0F ? &rng[index] : nullptr});
    }
    const auto actual = backends.front()->ProposeBatch(requests);
    Expect(rng == expected_rng,
           "batched proposals preserve every request's RNG");
    Expect(actual.size() == backends.size(), "batched backend result count");
    for (std::size_t index = 0; index < actual.size(); ++index) {
      Expect(
          actual[index].start_pos == expected[index].start_pos &&
              actual[index].tokens == expected[index].tokens &&
              actual[index].candidate_ids == expected[index].candidate_ids &&
              actual[index].candidates_per_token ==
                  expected[index].candidates_per_token &&
              exact(actual[index].candidate_probabilities,
                    expected[index].candidate_probabilities),
          "batched backend preserves independent proposals and probabilities");
      auto& backend = *backends[index];
      backend.AcceptFeedback(
          std::span(actual[index].tokens).first(accepted[index]), 4);
      Expect(payload(backend) == expected_state[index],
             "batched feedback preserves each controller, cache and pending "
             "history");
      for (std::size_t row = 0; row <= accepted[index]; ++row) {
        const auto feature_row = (positions[index] + row) % prompt.size();
        backend.UpdateTargetHidden(std::span(features).subspan(
            feature_row * feature_width, feature_width));
      }
      sequences[index].insert(sequences[index].end(),
                              actual[index].tokens.begin(),
                              actual[index].tokens.begin() + accepted[index]);
      sequences[index].push_back(4);
      positions[index] += static_cast<std::uint32_t>(accepted[index] + 1U);
    }
  }
  std::cout
      << "DFlash2 batched backends: exact RNG, private fixed/adaptive policy, "
         "pending feature injection and persistent state\n";
}

void CaptureReferenceTrace(
    const std::shared_ptr<const gufo::hip::QwenGpuModel>& target_model,
    const std::shared_ptr<const gufo::hip::QwenDFlashGpuModel>& draft_model,
    const std::filesystem::path& directory) {
  Expect(std::filesystem::create_directories(directory),
         "trace destination must be new");
  const gufo::hip::DFlashTrace write = [&](std::string_view name,
                                           std::span<const float> data) {
    std::ofstream file(directory / (std::string(name) + ".f32"),
                       std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size_bytes()));
    Expect(file.good(), "write draft trace");
  };
  std::string error;
  auto target = gufo::hip::QwenGpuExecutor::Create(target_model, &error, 128);
  Expect(target != nullptr, error);
  const auto& config = draft_model->GetDFlashConfig();
  auto tokens = target->GetTokenizer().Encode(
      "Virtual memory gives each process its own address space. The operating "
      "system maps virtual pages to physical memory and handles page faults.");
  tokens.resize(std::min<std::size_t>(tokens.size(), 24));
  target->SetPromptHiddenCapture(true, config.target_layer_ids);
  const auto anchor = target->ForwardPromptBatch(tokens);
  const auto features = target->GetPromptHiddenStates();
  write("target_features", features);
  auto draft =
      gufo::hip::QwenDFlashGpuExecutor::Create(draft_model, 128, &error);
  Expect(draft != nullptr, error);
  Expect(draft->InjectTargetContext(features, 0, tokens.size(), write),
         "inject real target features");
  std::vector<float> probabilities, confidences;
  std::vector<gufo::tokenization::TokenId> candidates;
  const std::array<float, 7> uniforms{0.13F, 0.37F, 0.71F, 0.21F,
                                      0.59F, 0.83F, 0.43F};
  const auto proposed =
      draft->ForwardBlock(anchor, tokens.size(), 7, 0.8F, uniforms,
                          &confidences, &candidates, &probabilities, write);
  write("probabilities", probabilities);
  write("selected_probabilities", confidences);
  write("uniforms", uniforms);
  auto metadata = gufo::server::json::Value::object();
  const auto array = [&](const char* key, const auto& values) {
    auto& result = metadata[key];
    result = gufo::server::json::Value::array();
    for (const auto value : values)
      result.push_back(static_cast<double>(value));
  };
  array("prompt_tokens", tokens);
  array("target_layer_ids", config.target_layer_ids);
  array("proposed", proposed);
  array("candidates", candidates);
  metadata["anchor"] = static_cast<double>(anchor);
  metadata["position"] = tokens.size();
  metadata["draft_count"] = proposed.size();
  metadata["temperature"] = 0.8;
  metadata["block_activation_dtype"] = "fp32";
  metadata["head_type"] =
      static_cast<double>(draft_model->GetWeights().output.type);
  std::ofstream file(directory / "trace.json");
  file << metadata.dump() << '\n';
  Expect(file.good(), "write trace manifest");
  std::cout << "DFlash reference trace: " << directory << '\n';
}

void CaptureAcceptanceTrace(
    const std::shared_ptr<const gufo::hip::QwenGpuModel>& target_model,
    const std::shared_ptr<const gufo::hip::QwenDFlashGpuModel>& draft_model,
    const char* input_path, const char* output_path) {
  using gufo::server::json::Value;
  using Token = gufo::tokenization::TokenId;
  std::ifstream input(input_path);
  Expect(input.good(), "open acceptance trace request");
  const auto request = gufo::server::json::parse(
      std::string(std::istreambuf_iterator<char>(input), {}));
  const auto text = request.member_str("prompt");
  const auto limit = request.member_size("max_tokens", 128);
  Expect(!text.empty() && limit > 0 && limit <= 256,
         "acceptance trace requires a prompt and 1..256 output tokens");
  std::string rendered = text;
  if (const auto* raw = request.find("raw");
      raw == nullptr || !raw->as_bool()) {
    const std::vector<gufo::tokenization::ChatMessage> messages{
        {gufo::tokenization::ChatRole::kSystem,
         "You are a helpful, respectful, and honest assistant.", "", ""},
        {gufo::tokenization::ChatRole::kUser, text, "", ""},
    };
    const auto chat = gufo::tokenization::QwenChatTemplate::Render(
        messages, {.add_generation_prompt = true, .enable_thinking = false});
    Expect(chat.has_value(), "render acceptance trace prompt");
    rendered = *chat;
  }
  const auto& tokenizer = target_model->GetTokenizer();
  const auto prompt = tokenizer.Encode(rendered);
  Expect(prompt.size() + limit < 2048, "bounded acceptance trace context");
  std::string error;
  auto executor =
      gufo::hip::QwenGpuExecutor::Create(target_model, &error, 2048);
  Expect(executor != nullptr, error);
  gufo::models::GenerationOptions generation;
  generation.max_new_tokens = limit;
  const auto reference = executor->Generate(prompt, generation);
  Expect(!reference.empty(),
         "acceptance trace needs a nonempty AR continuation");
  auto backend = gufo::hip::QwenDFlashGpuDraftBackend::Create(
      draft_model, {.max_context = 2048, .max_draft_tokens = 7}, &error);
  Expect(backend != nullptr, error);
  auto* inspect = backend.get();
  gufo::speculative::SpeculativeVerifier verifier(
      *executor, std::move(backend),
      {.max_draft_tokens = 7,
       .initial_draft_tokens = 7,
       .enable_adaptive_draft_length = false,
       .use_batched_verification = true});
  Token current = verifier.Prime(prompt);
  auto sequence = prompt;
  sequence.push_back(current);
  std::vector<Token> output{current};
  auto rows = Value::array();
  while (output.size() < limit) {
    const auto budget = static_cast<std::uint32_t>(limit - output.size());
    // Observe the exact next proposal without advancing its logical state.
    const auto saved = inspect->Snapshot();
    const auto proposed = inspect->Propose(
        sequence, static_cast<std::uint32_t>(sequence.size() - 1),
        std::min(7U, budget - 1));
    inspect->RestoreSnapshot(*saved);
    const auto step = verifier.VerifyStep(
        sequence, static_cast<std::uint32_t>(sequence.size() - 1), current,
        tokenizer.GetEosTokenId(), budget);
    Expect(step.draft_count == proposed.tokens.size(), "trace proposal replay");
    auto row = Value::object();
    row["offset"] = output.size();
    row["drafted"] = step.draft_count;
    row["accepted"] = step.accepted_count;
    row["proposed"] = tokenizer.Decode(proposed.tokens);
    row["emitted"] = tokenizer.Decode(step.emitted_tokens);
    if (step.accepted_count < proposed.tokens.size()) {
      row["rejected"] = tokenizer.Decode(
          std::span(proposed.tokens).subspan(step.accepted_count, 1));
      row["correction"] = tokenizer.Decode(
          std::span(step.emitted_tokens).subspan(step.accepted_count, 1));
    }
    rows.push_back(std::move(row));
    for (std::size_t index = 0; index < step.emitted_tokens.size(); ++index) {
      if (step.hit_eos && index + 1 == step.emitted_tokens.size())
        break;
      const auto token = step.emitted_tokens[index];
      output.push_back(token);
      sequence.push_back(token);
    }
    if (step.hit_eos)
      break;
    current = step.next_token;
  }
  Expect(output == reference, "acceptance trace must reproduce every AR ID");
  auto document = Value::object();
  document["request"] = request;
  document["completion"] = tokenizer.Decode(output);
  document["tokens"] = output.size();
  document["exact_ar"] = true;
  document["steps"] = std::move(rows);
  std::ofstream file(output_path);
  file << document.dump() << '\n';
  Expect(file.good(), "write acceptance trace");
}

}  // namespace

int main(int argc, const char* const* argv) {
  try {
    TestLengthController();
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

    if (argc == 5 && std::string_view(argv[3]) == "--trace") {
      CaptureReferenceTrace(target_model, dflash_model, argv[4]);
      return 0;
    }
    if (argc == 6 && std::string_view(argv[3]) == "--acceptance-trace") {
      CaptureAcceptanceTrace(target_model, dflash_model, argv[4], argv[5]);
      return 0;
    }
    TestConcurrentBlocks(target_model, dflash_model);
    if (argc == 4 && std::string_view(argv[3]) == "--concurrency-only")
      return 0;

    gufo::hip::QwenDFlashGpuDraftConfig config{
        .max_context = 512,
        .max_draft_tokens = 8,
        .policy = gufo::speculative::DFlashDraftPolicy::kAdaptive,
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

    // Learn from another rejection, then compare both snapshot forms with a
    // sampled continuation. A reset-on-restore bug would change the block size,
    // the proposal distribution and the RNG frontier.
    backend->UpdateTargetHidden(pending_features);
    const std::vector<gufo::tokenization::TokenId> next_prompt{1, 2, 3, 4, 5};
    const auto rejected = backend->Propose(next_prompt, next_prompt.size(), 7);
    Expect(!rejected.tokens.empty(), "adaptive controller continues drafting");
    backend->AcceptFeedback({}, 6);
    backend->UpdateTargetHidden(pending_features);
    const auto learned = backend->Snapshot();
    std::vector<std::uint8_t> learned_payload(
        learned->PersistentPayloadBytes());
    Expect(
        learned->SerializePersistent(learned_payload) == learned_payload.size(),
        "serialize learned controller state");
    const auto persistent_bytes_for =
        [](const gufo::hip::QwenDFlashGpuDraftBackend& source) {
          const auto saved = source.Snapshot();
          std::vector<std::uint8_t> bytes(saved->PersistentPayloadBytes());
          Expect(saved->SerializePersistent(bytes) == bytes.size(),
                 "serialize complete controller snapshot");
          return bytes;
        };
    const std::vector<gufo::tokenization::TokenId> final_prompt{1, 2, 3,
                                                                4, 5, 6};
    std::uint64_t source_rng = 73;
    const auto expected = backend->ProposeSampled(
        final_prompt, final_prompt.size(), 7, 0.8F, &source_rng);
    Expect(!expected.tokens.empty(),
           "learned adaptive controller continues sampled drafting");
    for (const bool persistent : {false, true}) {
      if (persistent)
        restored->RestorePersistentSnapshot(learned_payload);
      else
        restored->RestoreSnapshot(*learned);
      Expect(persistent_bytes_for(*restored) == learned_payload,
             "both snapshot forms preserve every controller state byte");
      std::uint64_t replay_rng = 73;
      const auto replayed = restored->ProposeSampled(
          final_prompt, final_prompt.size(), 7, 0.8F, &replay_rng);
      Expect(replayed.tokens == expected.tokens &&
                 replayed.candidate_ids == expected.candidate_ids &&
                 replayed.candidate_probabilities ==
                     expected.candidate_probabilities &&
                 replay_rng == source_rng,
             "controller snapshots preserve sampled proposals and RNG state");
      restored->AcceptFeedback({}, 7);
    }

    restored->RestoreSnapshot(*learned);
    restored->BeginRequest();
    std::uint64_t reused_rng = 73;
    const auto reused = restored->ProposeSampled(
        final_prompt, final_prompt.size(), 7, 0.8F, &reused_rng);
    auto cold = gufo::hip::QwenDFlashGpuDraftBackend::Create(dflash_model,
                                                             config, &error);
    Expect(cold != nullptr, error);
    auto complete_features = prompt_features;
    for (int row = 0; row < 3; ++row)
      complete_features.insert(complete_features.end(),
                               pending_features.begin(),
                               pending_features.end());
    Expect(cold->PrimeTargetContext({
               .prompt_tokens = final_prompt,
               .prompt_hidden_states = complete_features,
               .hidden_size = feature_width,
               .first_token = 7,
           }),
           "prime cold reference for reused request");
    std::uint64_t cold_rng = 73;
    const auto cold_proposal = cold->ProposeSampled(
        final_prompt, final_prompt.size(), 7, 0.8F, &cold_rng);
    Expect(reused.tokens == cold_proposal.tokens &&
               reused.candidate_ids == cold_proposal.candidate_ids &&
               reused.candidate_probabilities ==
                   cold_proposal.candidate_probabilities &&
               reused_rng == cold_rng,
           "new cached requests reproduce cold proposals and RNG state");
    restored->AcceptFeedback({}, 7);
    cold->AcceptFeedback({}, 7);
    Expect(persistent_bytes_for(*restored) == persistent_bytes_for(*cold),
           "new cached requests retain the same future policy and model state");

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
    // A full scratch chunk expires, with an uneven final chunk.
    const auto tail_tokens = 263U;
    const auto context_tokens = window + tail_tokens;
    std::vector<float> features(context_tokens * feature_width);
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
                                         window, tail_tokens),
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
        executor->ForwardBlock(4, context_tokens, 4, 0.0F, {}, &confidences);
    executor->RestoreSnapshot(*boundary);
    Expect(executor->InjectTargetContext(rows.subspan(window * feature_width),
                                         window, tail_tokens),
           "replay after in-memory restore");
    auto replay = executor->SaveSnapshot();
    std::vector<std::uint8_t> replay_payload(replay->PersistentPayloadBytes());
    Expect(replay->SerializePersistent(replay_payload) == replay_payload.size(),
           "serialize replayed history");
    Expect(replay_payload == wrapped_payload,
           "wrapped KV replay is byte exact");
    executor->Reset();
    Expect(executor->InjectTargetContext(rows, 0, context_tokens),
           "inject a prompt with expired chunks");
    auto complete = executor->SaveSnapshot();
    std::vector<std::uint8_t> complete_payload(
        complete->PersistentPayloadBytes());
    Expect(
        complete->SerializePersistent(complete_payload) ==
                complete_payload.size() &&
            complete_payload == wrapped_payload,
        "skipping expired chunks preserves the complete history byte for byte");
    executor->Reset();
    executor->RestorePersistentSnapshot(wrapped_payload);
    std::vector<float> replay_confidences;
    Expect(executor->ForwardBlock(4, context_tokens, 4, 0.0F, {},
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

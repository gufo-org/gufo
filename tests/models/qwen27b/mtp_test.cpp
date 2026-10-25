#include "src/models/qwen/hip/mtp.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/mtp_reference.hpp"

namespace {

constexpr int kSkipped = 77;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

std::vector<float> MakeTargetHidden(std::size_t size) {
  std::vector<float> hidden(size);
  for (std::size_t index = 0; index < hidden.size(); ++index) {
    const auto centered = static_cast<int>(index % 257U) - 128;
    hidden[index] = static_cast<float>(centered) / 256.0F;
  }
  return hidden;
}

struct Comparison {
  double rmse{0.0};
  double cosine{0.0};
  float max_abs{0.0F};
};

Comparison Compare(std::span<const float> actual,
                   std::span<const float> expected) {
  Expect(actual.size() == expected.size(), "comparison size mismatch");
  double squared_error = 0.0;
  double actual_squared = 0.0;
  double expected_squared = 0.0;
  double dot = 0.0;
  float max_abs = 0.0F;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double lhs = actual[index];
    const double rhs = expected[index];
    const double difference = lhs - rhs;
    squared_error += difference * difference;
    actual_squared += lhs * lhs;
    expected_squared += rhs * rhs;
    dot += lhs * rhs;
    max_abs = std::max(max_abs, static_cast<float>(std::abs(difference)));
  }
  const double count = static_cast<double>(actual.size());
  return {
      .rmse = std::sqrt(squared_error / count),
      .cosine = dot / std::sqrt(actual_squared * expected_squared),
      .max_abs = max_abs,
  };
}

}  // namespace

int main(int argc, const char* const* argv) {
  try {
    const char* base_path =
        argc > 1 ? argv[1] : std::getenv("GUFO_QWEN27B_MODEL");
    const char* draft_path =
        argc > 2 ? argv[2] : std::getenv("GUFO_QWEN27B_MTP_MODEL");
    if (base_path == nullptr || draft_path == nullptr) {
      std::cout << "qwen_mtp_gpu_test: skipped "
                   "(pass base and MTP GGUF paths)\n";
      return kSkipped;
    }

    std::string error;
    auto base_owner = gufo::core::GgufReader::OpenFile(base_path, &error);
    Expect(base_owner != nullptr, error);
    auto mtp_owner = gufo::core::GgufReader::OpenFile(draft_path, &error);
    Expect(mtp_owner != nullptr, error);
    std::shared_ptr<const gufo::core::GgufReader> base_reader(
        std::move(base_owner));
    std::shared_ptr<const gufo::core::GgufReader> mtp_reader(
        std::move(mtp_owner));

    auto reference = gufo::speculative::QwenMtpReference::CreateWithTiedWeights(
        mtp_reader, base_reader, 8, &error);
    Expect(reference != nullptr, error);
    auto target_model =
        gufo::hip::QwenGpuModel::CreateFromGguf(base_reader, &error);
    Expect(target_model != nullptr, error);

    const auto pack_start = std::chrono::steady_clock::now();
    auto mtp_model =
        gufo::hip::QwenMtpGpuModel::Create(mtp_reader, target_model, &error);
    Expect(mtp_model != nullptr, error);
    auto executor = gufo::hip::QwenMtpGpuExecutor::Create(
        mtp_model, 8, gufo::hip::kMtpDefaultPrimeChunk, &error);
    Expect(executor != nullptr, error);
    const double setup_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      pack_start)
            .count();

    const auto target_hidden =
        MakeTargetHidden(reference->GetWeights().config.hidden_size);
    const auto reference_hidden =
        reference->ForwardHidden(17, target_hidden, 0);
    Expect(!reference_hidden.empty(), "reference position zero");
    const std::vector<float> reference_copy(reference_hidden.begin(),
                                            reference_hidden.end());

    const auto token =
        executor->ForwardTargetHidden(17, target_hidden, 0, true);
    const auto gpu_hidden = executor->CopyLastHidden();
    const auto hidden_comparison = Compare(gpu_hidden, reference_copy);
    Expect(std::isfinite(hidden_comparison.rmse), "hidden RMSE is finite");
    Expect(hidden_comparison.rmse < 0.15, "hidden RMSE");
    Expect(hidden_comparison.cosine > 0.999, "hidden cosine");
    Expect(hidden_comparison.max_abs < 1.0F, "hidden max absolute error");

    const auto logits = executor->CopyLastLogits();
    constexpr std::array<std::uint32_t, 6> kSelectedTokens{0,  1,   17,
                                                           42, 198, 248319};
    float max_logit_error = 0.0F;
    for (const auto selected : kSelectedTokens) {
      const float expected = reference->ComputeLogit(selected);
      max_logit_error =
          std::max(max_logit_error, std::abs(logits[selected] - expected));
    }
    Expect(max_logit_error < 2.0F, "selected logit error");

    const std::vector<float> first_gpu_hidden(gpu_hidden.begin(),
                                              gpu_hidden.end());
    executor->Reset();
    const auto repeated_token =
        executor->ForwardTargetHidden(17, target_hidden, 0, true);
    const auto repeated_hidden = executor->CopyLastHidden();
    const auto repeat_comparison = Compare(repeated_hidden, first_gpu_hidden);
    Expect(repeated_token == token, "reset token determinism");
    Expect(repeat_comparison.rmse == 0.0, "reset hidden determinism");

    gufo::hip::QwenMtpGpuDraftConfig draft_config{
        .max_context = 8,
        .max_draft_tokens = 2,
    };
    auto draft_backend = gufo::hip::QwenMtpGpuDraftBackend::Create(
        mtp_model, draft_config, &error);
    Expect(draft_backend != nullptr, error);
    const std::array<gufo::tokenization::TokenId, 2> prompt_tokens{5, 17};
    std::vector<float> prompt_hidden(target_hidden.size() * 2);
    std::ranges::copy(target_hidden, prompt_hidden.begin());
    for (std::size_t index = 0; index < target_hidden.size(); ++index) {
      prompt_hidden[target_hidden.size() + index] =
          target_hidden[index] * 0.75F;
    }
    const gufo::speculative::DraftTargetContext draft_context{
        .prompt_tokens = prompt_tokens,
        .prompt_hidden_states = prompt_hidden,
        .hidden_size = target_hidden.size(),
        .first_token = token,
    };
    Expect(draft_backend->PrimeTargetContext(draft_context),
           draft_backend->GetLastError());

    std::vector<gufo::tokenization::TokenId> sequence{prompt_tokens[0],
                                                      prompt_tokens[1], token};
    const auto first_proposal = draft_backend->Propose(sequence, 2, 2);
    Expect(first_proposal.tokens.size() == 2, "first GPU MTP proposal size");
    // Verification publishes committed target rows before feedback. Rebuilding
    // from a fresh teacher-forced prefix must produce the same next proposal.
    std::vector<float> committed_hidden(target_hidden);
    std::vector<float> anchor_hidden(target_hidden);
    for (auto& value : anchor_hidden)
      value *= -2.0F;
    for (auto& value : committed_hidden)
      value *= 0.5F;
    draft_backend->UpdateTargetHidden(anchor_hidden);
    draft_backend->UpdateTargetHidden(committed_hidden);
    draft_backend->AcceptFeedback(std::span<const gufo::tokenization::TokenId>(
                                      first_proposal.tokens.data(), 1),
                                  42);
    sequence.push_back(first_proposal.tokens.front());
    const auto committed_sequence = sequence;
    sequence.push_back(42);
    const auto second_proposal = draft_backend->Propose(sequence, 4, 2);
    auto teacher_forced = gufo::hip::QwenMtpGpuDraftBackend::Create(
        mtp_model, draft_config, &error);
    Expect(teacher_forced != nullptr, error);
    std::vector<float> committed_features(prompt_hidden);
    committed_features.insert(committed_features.end(), anchor_hidden.begin(),
                              anchor_hidden.end());
    committed_features.insert(committed_features.end(),
                              committed_hidden.begin(), committed_hidden.end());
    Expect(teacher_forced->PrimeTargetContext({
               .prompt_tokens = committed_sequence,
               .prompt_hidden_states = committed_features,
               .hidden_size = target_hidden.size(),
               .first_token = 42,
           }),
           "teacher-forced MTP replay");
    const auto expected_proposal = teacher_forced->Propose(sequence, 4, 2);
    Expect(
        second_proposal.tokens == expected_proposal.tokens,
        "feedback replay must match teacher-forced committed target features");

    // ── Abandoned batched chains (error-path drain) ─────────────────────────
    // A cohort member whose StartDraftChain throws must not strand another
    // session's in-flight chain: the rollback Rewind would bail on
    // RequireChainIdle, the executor would keep chain_active_ set, and the
    // next noexcept Reset() would terminate the process on the next
    // checkpoint. Abandoning every live chain first keeps the rollback clean.
    {
      auto chain_executor = gufo::hip::QwenMtpGpuExecutor::Create(
          mtp_model, 64, gufo::hip::kMtpDefaultPrimeChunk, &error);
      Expect(chain_executor != nullptr, error);
      chain_executor->StartDraftChain(17, target_hidden, 2);
      chain_executor->AbandonChain();
      bool collect_rejected = false;
      try {
        (void)chain_executor->CollectChainTokens();
      } catch (const std::exception&) {
        collect_rejected = true;
      }
      Expect(collect_rejected, "abandoned chain must not be collectable");
      chain_executor->Rewind(0);
      const auto recovered_token =
          chain_executor->ForwardTargetHidden(17, target_hidden, 0, true);
      Expect(recovered_token == token,
             "abandoned chain leaves the executor forward-deterministic");
      const auto recovered_comparison =
          Compare(chain_executor->CopyLastHidden(), first_gpu_hidden);
      Expect(recovered_comparison.rmse == 0.0,
             "abandoned chain leaves no hidden drift");
    }

    // ── Failed batch cohorts drain before rollback ─────────────────────────
    // One oversized depth (past the slot budget) makes its StartDraftChain
    // throw while the other session's chain is already in flight; the batch
    // must rethrow and both executors must stay usable.
    {
      constexpr std::uint32_t kOverSlotDepth =
          gufo::hip::kMtpMaxChainTokens + 4;
      gufo::hip::QwenMtpGpuDraftConfig batch_config{
          .max_context = 64,
          .max_draft_tokens = 2,
      };
      gufo::hip::QwenMtpGpuDraftConfig oversized_config{
          .max_context = 64,
          .max_draft_tokens = kOverSlotDepth,
      };
      auto rejected_victim = gufo::hip::QwenMtpGpuDraftBackend::Create(
          mtp_model, batch_config, &error);
      Expect(rejected_victim != nullptr, error);
      auto oversized_member = gufo::hip::QwenMtpGpuDraftBackend::Create(
          mtp_model, oversized_config, &error);
      Expect(oversized_member != nullptr, error);
      auto pristine_peer = gufo::hip::QwenMtpGpuDraftBackend::Create(
          mtp_model, batch_config, &error);
      Expect(pristine_peer != nullptr, error);
      for (auto* backend : {rejected_victim.get(), oversized_member.get(),
                            pristine_peer.get()}) {
        std::vector<float> prime_hidden(target_hidden.size() * 2);
        std::ranges::copy(target_hidden, prime_hidden.begin());
        for (std::size_t index = 0; index < target_hidden.size(); ++index) {
          prime_hidden[target_hidden.size() + index] =
              target_hidden[index] * 0.75F;
        }
        Expect(backend->PrimeTargetContext({
                   .prompt_tokens = prompt_tokens,
                   .prompt_hidden_states = prime_hidden,
                   .hidden_size = target_hidden.size(),
                   .first_token = token,
               }),
               "failed-batch cohorts must prime identically");
      }
      std::vector<gufo::tokenization::TokenId> request_tokens(
          prompt_tokens.begin(), prompt_tokens.end());
      request_tokens.push_back(token);
      const std::array<gufo::speculative::DraftProposalRequest, 2> batch{{
          {rejected_victim.get(), request_tokens, 2, 2, 0.0F, nullptr},
          {oversized_member.get(), request_tokens, 2, kOverSlotDepth, 0.0F,
           nullptr},
      }};
      bool batch_rejected = false;
      try {
        (void)rejected_victim->ProposeBatch(batch);
      } catch (const std::invalid_argument&) {
        batch_rejected = true;
      }
      Expect(batch_rejected, "an over-slot cohort member must abort the batch");
      Expect(rejected_victim->GetLastError().empty(),
             "a drained batch rollback must not surface chain errors");
      const auto pristine_proposal =
          pristine_peer->Propose(request_tokens, 2, 2);
      const auto recovered_proposal =
          rejected_victim->Propose(request_tokens, 2, 2);
      Expect(recovered_proposal.tokens == pristine_proposal.tokens,
             "an abandoned chain must leave the session recoverable");
    }

    // ── Snapshot gates ──────────────────────────────────────────────────────
    // Admission control sizes a session before it exists, so the estimate must
    // equal what the executor actually reserves.
    const auto estimate =
        gufo::hip::QwenMtpGpuExecutor::EstimateMemoryUsage(*mtp_model, 8);
    const auto actual = executor->GetMemoryUsage();
    Expect(estimate.request_state_bytes == actual.request_state_bytes,
           "MTP state estimate must match the allocation");
    Expect(estimate.temporary_scratch_bytes == actual.temporary_scratch_bytes,
           "MTP scratch estimate must match the allocation");

    // ── Batched prompt prime (stage A) ──────────────────────────────────────
    // The batched prime must reproduce the legacy per-token loop: the same KV
    // rows, the same cursor, and the same next token and logits.
    {
      const std::size_t prompt_width = mtp_model->GetConfig().hidden_size;
      constexpr std::size_t kPromptRows = 5;
      std::vector<gufo::tokenization::TokenId> prompt_tokens;
      std::vector<float> prompt_hidden;
      for (std::size_t row = 0; row <= kPromptRows; ++row) {
        prompt_tokens.push_back(
            static_cast<gufo::tokenization::TokenId>(31 + row));
        auto hidden = MakeTargetHidden(prompt_width);
        for (auto& value : hidden) {
          value += static_cast<float>(row);
        }
        prompt_hidden.insert(prompt_hidden.end(), hidden.begin(), hidden.end());
      }
      // Primed rows own positions [0, kPromptRows) and each contributes the
      // hidden of its own position; the final prompt row stays uncached and
      // seeds the next step, exactly as the backend's Propose does.
      const auto prime_rows = std::span<const float>(
          prompt_hidden.data(), kPromptRows * prompt_width);
      const auto successor_tokens =
          std::span<const gufo::tokenization::TokenId>(prompt_tokens.data() + 1,
                                                       kPromptRows);
      const auto last_hidden = std::span<const float>(
          prompt_hidden.data() + (kPromptRows * prompt_width), prompt_width);
      const auto decode = [&](gufo::hip::QwenMtpGpuExecutor& target,
                              gufo::tokenization::TokenId input) {
        const auto step = target.ForwardTargetHidden(
            input, last_hidden, static_cast<std::uint32_t>(kPromptRows), true);
        const auto logits = target.CopyLastLogits();
        return std::make_pair(step,
                              std::vector<float>(logits.begin(), logits.end()));
      };

      std::string prime_error;
      auto legacy =
          gufo::hip::QwenMtpGpuExecutor::Create(mtp_model, 8, 0, &prime_error);
      Expect(legacy != nullptr, prime_error);
      for (std::size_t row = 0; row < kPromptRows; ++row) {
        (void)legacy->ForwardTargetHidden(
            successor_tokens[row],
            prime_rows.subspan(row * prompt_width, prompt_width),
            static_cast<std::uint32_t>(row), false);
      }
      Expect(legacy->GetNextPosition() == kPromptRows, "legacy prime cursor");
      const auto legacy_step = decode(*legacy, prompt_tokens.front());

      // Chunk invariance: one row at a time, the prompt at once, and a chunk
      // wider than both. This is what catches an off-by-one in the slice
      // arithmetic.
      for (const std::uint32_t chunk : {1U, 5U, 8U}) {
        auto batched = gufo::hip::QwenMtpGpuExecutor::Create(
            mtp_model, 8, chunk, &prime_error);
        Expect(batched != nullptr, prime_error);
        batched->PrimeTargetHiddenBatch(0, successor_tokens, prime_rows);
        Expect(batched->GetNextPosition() == kPromptRows,
               "batched prime cursor");
        const auto batched_step = decode(*batched, prompt_tokens.front());
        Expect(batched_step.first == legacy_step.first,
               "batched prime must reproduce the legacy next token");
        const auto logit_comparison =
            Compare(batched_step.second, legacy_step.second);
        Expect(logit_comparison.cosine > 0.99999, "batched prime logit cosine");
        Expect(logit_comparison.max_abs < 1.0e-2F, "batched prime logit drift");
      }

      // The estimate must track the chunk it is sized for, including the
      // legacy (zero-chunk) and clamped configurations.
      for (const std::uint32_t chunk : {0U, 1U, 3U, 8U}) {
        auto sized = gufo::hip::QwenMtpGpuExecutor::Create(mtp_model, 8, chunk,
                                                           &prime_error);
        Expect(sized != nullptr, prime_error);
        const auto sized_estimate =
            gufo::hip::QwenMtpGpuExecutor::EstimateMemoryUsage(*mtp_model, 8,
                                                               chunk);
        const auto sized_actual = sized->GetMemoryUsage();
        Expect(sized_estimate.temporary_scratch_bytes ==
                   sized_actual.temporary_scratch_bytes,
               "MTP prime scratch estimate must match the allocation");
        Expect(sized_estimate.request_state_bytes ==
                   sized_actual.request_state_bytes,
               "MTP prime state estimate must match the allocation");
      }

      // Append identity: extending a primed prefix with more rows must
      // reproduce the per-row extension's cursor, next token and logits. This
      // is what the backend's batched AppendTargetContext rides on: the row at
      // draft position `pos + j` consumes the suffix token at `j` and the
      // trunk hidden of `j - 1`, with the primed prefix's own final hidden
      // reaching row `start_position`.
      constexpr std::size_t kExtendRows = 2;
      std::vector<gufo::tokenization::TokenId> extend_tokens;
      std::vector<float> extend_hidden;
      for (std::size_t row = 0; row < kExtendRows; ++row) {
        extend_tokens.push_back(
            static_cast<gufo::tokenization::TokenId>(41 + row));
        auto hidden = MakeTargetHidden(prompt_width);
        for (auto& value : hidden) {
          value -= 17.0F * static_cast<float>(row + 1);
        }
        extend_hidden.insert(extend_hidden.end(), hidden.begin(), hidden.end());
      }
      const auto extend_last_hidden = std::span<const float>(
          extend_hidden.data() + ((kExtendRows - 1) * prompt_width),
          prompt_width);
      const auto extend_step = [&](gufo::hip::QwenMtpGpuExecutor& target) {
        const auto token = target.ForwardTargetHidden(
            extend_tokens.front() + kExtendRows + 1, extend_last_hidden,
            static_cast<std::uint32_t>(kPromptRows + kExtendRows), true);
        const auto logits = target.CopyLastLogits();
        return std::make_pair(token,
                              std::vector<float>(logits.begin(), logits.end()));
      };
      std::vector<float> append_seed(last_hidden.begin(), last_hidden.end());
      auto extended_legacy =
          gufo::hip::QwenMtpGpuExecutor::Create(mtp_model, 8, 8, &prime_error);
      Expect(extended_legacy != nullptr, prime_error);
      extended_legacy->PrimeTargetHiddenBatch(0, successor_tokens, prime_rows);
      std::vector<float> extend_rolling(append_seed);
      for (std::size_t row = 0; row < kExtendRows; ++row) {
        (void)extended_legacy->ForwardTargetHidden(
            extend_tokens[row],
            std::span<const float>(extend_rolling.data(), prompt_width),
            static_cast<std::uint32_t>(kPromptRows + row), false);
        extend_rolling.assign(
            extend_hidden.begin() +
                static_cast<std::ptrdiff_t>(row * prompt_width),
            extend_hidden.begin() +
                static_cast<std::ptrdiff_t>((row + 1) * prompt_width));
      }
      Expect(extended_legacy->GetNextPosition() == kPromptRows + kExtendRows,
             "legacy append cursor");
      Expect(std::vector<float>(extend_rolling.begin(), extend_rolling.end()) ==
                 std::vector<float>(extend_last_hidden.begin(),
                                    extend_last_hidden.end()),
             "legacy append rolling hidden");
      const auto extended_legacy_step = extend_step(*extended_legacy);
      for (const std::uint32_t chunk : {1U, 5U, 8U}) {
        auto extended_batched = gufo::hip::QwenMtpGpuExecutor::Create(
            mtp_model, 8, chunk, &prime_error);
        Expect(extended_batched != nullptr, prime_error);
        extended_batched->PrimeTargetHiddenBatch(0, successor_tokens,
                                                 prime_rows);
        // The backend's append ladder: the primed prefix's rolling hidden,
        // then the suffix hiddens 0..kExtendRows-2.
        std::vector<float> append_rows(append_seed);
        append_rows.insert(
            append_rows.end(), extend_hidden.begin(),
            extend_hidden.begin() +
                static_cast<std::ptrdiff_t>((kExtendRows - 1) * prompt_width));
        extended_batched->PrimeTargetHiddenBatch(
            static_cast<std::uint32_t>(kPromptRows), extend_tokens,
            append_rows);
        Expect(extended_batched->GetNextPosition() == kPromptRows + kExtendRows,
               "batched append cursor");
        const auto extended_batched_step = extend_step(*extended_batched);
        Expect(extended_batched_step.first == extended_legacy_step.first,
               "batched append must reproduce the legacy next token");
        const auto extend_comparison =
            Compare(extended_batched_step.second, extended_legacy_step.second);
        Expect(extend_comparison.cosine > 0.99999,
               "batched append logit cosine");
        Expect(extend_comparison.max_abs < 1.0e-2F,
               "batched append logit drift");
      }
    }

    const auto& execution_config = mtp_model->GetConfig();
    const std::size_t kv_width =
        static_cast<std::size_t>(execution_config.num_key_value_heads) *
        execution_config.head_dim;

    // Executor level: a restored boundary must reproduce the next step's token
    // and logits bit for bit, which is what byte-identical output requires.
    executor->Reset();
    for (std::uint32_t position = 0; position < 3; ++position) {
      (void)executor->ForwardTargetHidden(
          static_cast<gufo::tokenization::TokenId>(11 + position),
          target_hidden, position, true);
    }
    Expect(executor->GetNextPosition() == 3, "MTP snapshot prefix cursor");
    auto snapshot = executor->SaveSnapshot();
    Expect(snapshot != nullptr, "MTP executor snapshot");
    Expect(snapshot->ValidContext() == 3, "MTP snapshot valid context");
    Expect(snapshot->PayloadBytes() == 2U * kv_width * 3U * sizeof(float),
           "MTP snapshot payload is the exact packed plane");

    const auto boundary_token =
        executor->ForwardTargetHidden(19, target_hidden, 3, true);
    const auto boundary_span = executor->CopyLastLogits();
    const std::vector<float> boundary_logits(boundary_span.begin(),
                                             boundary_span.end());

    for (std::uint32_t extra = 0; extra < 3; ++extra) {
      (void)executor->ForwardTargetHidden(
          static_cast<gufo::tokenization::TokenId>(23 + extra), target_hidden,
          4 + extra, true);
    }
    Expect(executor->GetNextPosition() == 7, "MTP dirty prefix cursor");
    executor->RestoreSnapshot(*snapshot);
    Expect(executor->GetNextPosition() == 3, "MTP restored cursor");
    const auto replayed_token =
        executor->ForwardTargetHidden(19, target_hidden, 3, true);
    const auto replayed_span = executor->CopyLastLogits();
    Expect(replayed_token == boundary_token, "MTP restored token determinism");
    Expect(std::vector<float>(replayed_span.begin(), replayed_span.end()) ==
               boundary_logits,
           "MTP restored logits must be bit-identical");

    // Persistent form: the same boundary after a device-memory round trip,
    // which is the disk-cache restore path.
    std::vector<std::uint8_t> persistent(snapshot->PersistentPayloadBytes());
    Expect(snapshot->SerializePersistent(persistent) == persistent.size(),
           "MTP persistent serialize size");
    executor->Reset();
    executor->RestorePersistentSnapshot(persistent);
    Expect(executor->GetNextPosition() == 3, "MTP persistent restored cursor");
    const auto persisted_token =
        executor->ForwardTargetHidden(19, target_hidden, 3, true);
    const auto persisted_span = executor->CopyLastLogits();
    Expect(persisted_token == boundary_token,
           "MTP persistent restored token determinism");
    Expect(std::vector<float>(persisted_span.begin(), persisted_span.end()) ==
               boundary_logits,
           "MTP persistent restored logits must be bit-identical");

    // Draft level: the boundary carries the last committed target hidden row as
    // well as the cache, so a restored draft must propose identically.
    auto snapshot_backend = gufo::hip::QwenMtpGpuDraftBackend::Create(
        mtp_model, draft_config, &error);
    Expect(snapshot_backend != nullptr, error);
    Expect(snapshot_backend->PrimeTargetContext(draft_context),
           snapshot_backend->GetLastError());
    const std::vector<gufo::tokenization::TokenId> anchored_sequence{
        prompt_tokens[0], prompt_tokens[1], token};
    const auto anchored_proposal =
        snapshot_backend->Propose(anchored_sequence, 2, 2);
    Expect(anchored_proposal.tokens.size() == 2, "MTP anchored proposal");
    snapshot_backend->UpdateTargetHidden(anchor_hidden);
    snapshot_backend->UpdateTargetHidden(committed_hidden);
    snapshot_backend->AcceptFeedback(
        std::span<const gufo::tokenization::TokenId>(
            anchored_proposal.tokens.data(), 1),
        42);
    std::vector<gufo::tokenization::TokenId> committed_anchored(
        anchored_sequence);
    committed_anchored.push_back(anchored_proposal.tokens.front());

    const auto committed_bytes = snapshot_backend->SnapshotPayloadBytes();
    auto draft_snapshot = snapshot_backend->Snapshot();
    Expect(draft_snapshot != nullptr, "MTP draft snapshot");
    Expect(draft_snapshot->PayloadBytes() == committed_bytes,
           "MTP draft snapshot payload must equal the reservation");
    const auto probe = snapshot_backend->Propose(committed_anchored, 4, 2);

    // Move the live backend past the captured boundary, then rewind.
    snapshot_backend->UpdateTargetHidden(anchor_hidden);
    snapshot_backend->UpdateTargetHidden(committed_hidden);
    snapshot_backend->AcceptFeedback(
        std::span<const gufo::tokenization::TokenId>(probe.tokens.data(), 1),
        7);
    snapshot_backend->RestoreSnapshot(*draft_snapshot);
    const auto restored_probe =
        snapshot_backend->Propose(committed_anchored, 4, 2);
    Expect(restored_probe.tokens == probe.tokens,
           "MTP draft snapshot restore must propose identically");

    std::vector<std::uint8_t> draft_persistent(
        draft_snapshot->PersistentPayloadBytes());
    Expect(draft_snapshot->SerializePersistent(draft_persistent) ==
               draft_persistent.size(),
           "MTP draft persistent serialize size");
    auto persisted_backend = gufo::hip::QwenMtpGpuDraftBackend::Create(
        mtp_model, draft_config, &error);
    Expect(persisted_backend != nullptr, error);
    persisted_backend->RestorePersistentSnapshot(draft_persistent);
    const auto persisted_probe =
        persisted_backend->Propose(committed_anchored, 4, 2);
    Expect(persisted_probe.tokens == probe.tokens,
           "MTP draft persistent restore must propose identically");

    // ── Small draft head (mtp_draft_vocab.hpp) ──────────────────────────────
    // Only Qwen3.6 35B-A3B drafts through it, so the block runs for that GGUF
    // alone. The head scores the prefix and the table rows of the full head,
    // so its proposal is the full head's best token among them, and a table
    // row comes back as its own token ID (see also qwen_sampling_hip_test).
    Expect(mtp_model->HasDraftVocab() ==
               (target_model->GetConfig().architecture ==
                    gufo::speculative::kQwenMtpDraftArchitecture &&
                execution_config.vocab_size ==
                    gufo::speculative::kQwenMtpDraftVocabRows),
           "MTP draft vocabulary gate");
    if (mtp_model->HasDraftVocab()) {
      const auto specials = mtp_model->GetDraftSpecials();
      const std::uint32_t prefix = gufo::speculative::kMtpDraftVocab;
      Expect(executor->GetDraftRows() == specials.size(),
             "MTP draft table opens with the special tokens");
      executor->Reset();
      (void)executor->ForwardTargetHidden(17, target_hidden, 0, true);
      const auto full_span = executor->CopyLastLogits();
      const std::vector<float> full_logits(full_span.begin(), full_span.end());
      // The best ordinary token above the prefix goes into the table.
      std::uint32_t rare = prefix;
      for (std::uint32_t id = prefix; id < full_logits.size(); ++id) {
        if (!std::ranges::binary_search(specials,
                                        static_cast<std::int32_t>(id)) &&
            full_logits[id] > full_logits[rare])
          rare = id;
      }
      const auto best_of = [&](bool with_rare) {
        std::vector<std::uint32_t> ids(prefix);
        std::iota(ids.begin(), ids.end(), 0U);
        ids.insert(ids.end(), specials.begin(), specials.end());
        if (with_rare)
          ids.push_back(rare);
        std::ranges::sort(ids, [&](std::uint32_t a, std::uint32_t b) {
          return full_logits[a] > full_logits[b];
        });
        // A near tie may resolve either way under the head's own row order.
        return full_logits[ids[0]] - full_logits[ids[1]] > 1.0e-3F
                   ? std::optional<std::uint32_t>(ids[0])
                   : std::nullopt;
      };
      const auto small_step = [&]() {
        executor->Reset();
        return executor->ForwardTargetHidden(17, target_hidden, 0, true, true);
      };
      if (const auto expected = best_of(false))
        Expect(small_step() == *expected,
               "MTP small head must propose the best prefix or special token");
      const std::array<std::int32_t, 1> rare_row{
          static_cast<std::int32_t>(rare)};
      executor->AppendDraftRows(rare_row);
      Expect(executor->GetDraftRows() == specials.size() + 1,
             "MTP draft table append");
      if (const auto expected = best_of(true))
        Expect(small_step() == *expected,
               "MTP small head must return a table row as its token ID");
      executor->ResetDraftRows();
      Expect(executor->GetDraftRows() == specials.size(),
             "MTP draft table reset keeps the special tokens");

      // Batched proposals take the serial head choice and match it exactly,
      // including a history that tables a rare token.
      const std::vector<gufo::tokenization::TokenId> rare_sequence{
          rare, prompt_tokens[1], token};
      const std::array<std::vector<gufo::tokenization::TokenId>, 2> sequences{
          anchored_sequence, rare_sequence};
      std::array<std::unique_ptr<gufo::hip::QwenMtpGpuDraftBackend>, 2> serial;
      std::array<std::unique_ptr<gufo::hip::QwenMtpGpuDraftBackend>, 2> batched;
      std::array<gufo::speculative::DraftProposalRequest, 2> requests;
      for (std::size_t index = 0; index < 2; ++index) {
        serial[index] = gufo::hip::QwenMtpGpuDraftBackend::Create(
            mtp_model, draft_config, &error);
        batched[index] = gufo::hip::QwenMtpGpuDraftBackend::Create(
            mtp_model, draft_config, &error);
        Expect(serial[index] != nullptr && batched[index] != nullptr, error);
        Expect(serial[index]->PrimeTargetContext(draft_context) &&
                   batched[index]->PrimeTargetContext(draft_context),
               "MTP small-head batch prime");
        requests[index] = {
            .backend = batched[index].get(),
            .tokens = sequences[index],
            .position = 2,
            .max_tokens = 2,
        };
      }
      const auto proposals = batched[0]->ProposeBatch(requests);
      Expect(proposals.size() == 2, "MTP small-head batch size");
      for (std::size_t index = 0; index < 2; ++index) {
        const auto expected = serial[index]->Propose(sequences[index], 2, 2);
        Expect(proposals[index].tokens == expected.tokens,
               "MTP batched small-head proposals must match the serial path");
      }
    }

    draft_backend->Reset();

    std::cout << "qwen_mtp_gpu_test: token=" << token
              << " packed_bytes=" << mtp_model->GetPackedWeightBytes()
              << " pack_seconds=" << mtp_model->GetPackTimeSeconds()
              << " setup_seconds=" << setup_seconds
              << " hidden_rmse=" << hidden_comparison.rmse
              << " hidden_cosine=" << hidden_comparison.cosine
              << " hidden_max_abs=" << hidden_comparison.max_abs
              << " selected_logit_max_abs=" << max_logit_error << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "qwen_mtp_gpu_test failed: " << exception.what() << '\n';
    return 1;
  }
}

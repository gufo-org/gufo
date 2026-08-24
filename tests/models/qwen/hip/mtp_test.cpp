#include "src/models/qwen/hip/mtp.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
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
    if (argc < 3) {
      std::cout << "qwen_mtp_gpu_test: skipped "
                   "(pass base and MTP GGUF paths)\n";
      return kSkipped;
    }

    std::string error;
    auto base_owner = strix::core::GgufReader::OpenFile(argv[1], &error);
    Expect(base_owner != nullptr, error);
    auto mtp_owner = strix::core::GgufReader::OpenFile(argv[2], &error);
    Expect(mtp_owner != nullptr, error);
    std::shared_ptr<const strix::core::GgufReader> base_reader(
        std::move(base_owner));
    std::shared_ptr<const strix::core::GgufReader> mtp_reader(
        std::move(mtp_owner));

    auto reference =
        strix::speculative::QwenMtpReference::CreateWithTiedWeights(
            mtp_reader, base_reader, 8, &error);
    Expect(reference != nullptr, error);
    auto target_model =
        strix::hip::QwenGpuModel::CreateFromGguf(base_reader, &error);
    Expect(target_model != nullptr, error);

    const auto pack_start = std::chrono::steady_clock::now();
    auto mtp_model =
        strix::hip::QwenMtpGpuModel::Create(mtp_reader, target_model, &error);
    Expect(mtp_model != nullptr, error);
    auto executor =
        strix::hip::QwenMtpGpuExecutor::Create(mtp_model, 8, &error);
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

#if defined(ENGINE_ENABLE_XRT)
    auto hybrid_executor = strix::hip::QwenMtpGpuExecutor::Create(
        mtp_model, 8, &error,
        strix::hip::QwenMtpExecutionMode::kHybridNpuEhProj);
    Expect(hybrid_executor != nullptr, error);
    const auto hybrid_token =
        hybrid_executor->ForwardTargetHidden(17, target_hidden, 0, true);
    const auto hybrid_hidden = hybrid_executor->CopyLastHidden();
    const auto hybrid_hidden_comparison = Compare(hybrid_hidden, gpu_hidden);
    Expect(std::isfinite(hybrid_hidden_comparison.rmse),
           "hybrid hidden RMSE is finite");
    Expect(hybrid_hidden_comparison.rmse < 0.15, "hybrid hidden RMSE");
    Expect(hybrid_hidden_comparison.cosine > 0.999, "hybrid hidden cosine");
    Expect(hybrid_hidden_comparison.max_abs < 1.0F,
           "hybrid hidden max absolute error");
    const auto hybrid_logits = hybrid_executor->CopyLastLogits();
    float hybrid_max_logit_error = 0.0F;
    for (const auto selected : kSelectedTokens) {
      hybrid_max_logit_error =
          std::max(hybrid_max_logit_error,
                   std::abs(hybrid_logits[selected] - logits[selected]));
    }
    Expect(hybrid_max_logit_error < 2.0F, "hybrid selected logit error");
    const auto hybrid_logit_comparison = Compare(hybrid_logits, logits);
    Expect(std::isfinite(hybrid_logit_comparison.rmse),
           "hybrid logit RMSE is finite");
    Expect(hybrid_logit_comparison.rmse < 0.2, "hybrid logit RMSE");
    Expect(hybrid_logit_comparison.cosine > 0.999, "hybrid logit cosine");
    Expect(hybrid_logit_comparison.max_abs < 2.0F,
           "hybrid logit max absolute error");
    Expect(hybrid_token == token, "hybrid draft token parity");
    const auto hybrid_metrics = hybrid_executor->GetHybridMetrics();
    Expect(hybrid_metrics.projection_count == 1, "hybrid projection count");
    Expect(hybrid_metrics.npu_command_us > 0.0, "hybrid NPU command timing");
#endif

    const std::vector<float> first_gpu_hidden(gpu_hidden.begin(),
                                              gpu_hidden.end());
    executor->Reset();
    const auto repeated_token =
        executor->ForwardTargetHidden(17, target_hidden, 0, true);
    const auto repeated_hidden = executor->CopyLastHidden();
    const auto repeat_comparison = Compare(repeated_hidden, first_gpu_hidden);
    Expect(repeated_token == token, "reset token determinism");
    Expect(repeat_comparison.rmse == 0.0, "reset hidden determinism");

    strix::hip::QwenMtpGpuDraftConfig draft_config{
        .max_context = 8,
        .max_draft_tokens = 2,
    };
    auto draft_backend = strix::hip::QwenMtpGpuDraftBackend::Create(
        mtp_model, draft_config, &error);
    Expect(draft_backend != nullptr, error);
    const std::array<strix::tokenization::TokenId, 2> prompt_tokens{5, 17};
    std::vector<float> prompt_hidden(target_hidden.size() * 2);
    std::ranges::copy(target_hidden, prompt_hidden.begin());
    for (std::size_t index = 0; index < target_hidden.size(); ++index) {
      prompt_hidden[target_hidden.size() + index] =
          target_hidden[index] * 0.75F;
    }
    const strix::speculative::DraftTargetContext draft_context{
        .prompt_tokens = prompt_tokens,
        .prompt_hidden_states = prompt_hidden,
        .hidden_size = target_hidden.size(),
        .first_token = token,
    };
    Expect(draft_backend->PrimeTargetContext(draft_context),
           draft_backend->GetLastError());

    std::vector<strix::tokenization::TokenId> sequence{prompt_tokens[0],
                                                       prompt_tokens[1], token};
    const auto first_proposal = draft_backend->Propose(sequence, 2, 2);
    Expect(first_proposal.tokens.size() == 2, "first GPU MTP proposal size");
    draft_backend->AcceptFeedback(std::span<const strix::tokenization::TokenId>(
                                      first_proposal.tokens.data(), 1),
                                  42);

    std::vector<float> committed_hidden(target_hidden);
    for (auto& value : committed_hidden) {
      value *= 0.5F;
    }
    draft_backend->UpdateTargetHidden(committed_hidden);
    sequence.push_back(first_proposal.tokens.front());
    sequence.push_back(42);
    const auto second_proposal = draft_backend->Propose(sequence, 4, 1);
    Expect(second_proposal.tokens.size() == 1, "second GPU MTP proposal size");
    draft_backend->AcceptFeedback({}, 7);
    draft_backend->Reset();

    std::cout << "qwen_mtp_gpu_test: token=" << token
              << " packed_bytes=" << mtp_model->GetPackedWeightBytes()
              << " pack_seconds=" << mtp_model->GetPackTimeSeconds()
              << " setup_seconds=" << setup_seconds
              << " hidden_rmse=" << hidden_comparison.rmse
              << " hidden_cosine=" << hidden_comparison.cosine
              << " hidden_max_abs=" << hidden_comparison.max_abs
              << " selected_logit_max_abs=" << max_logit_error
#if defined(ENGINE_ENABLE_XRT)
              << " hybrid_hidden_rmse=" << hybrid_hidden_comparison.rmse
              << " hybrid_hidden_cosine=" << hybrid_hidden_comparison.cosine
              << " hybrid_selected_logit_max_abs=" << hybrid_max_logit_error
              << " hybrid_logit_rmse=" << hybrid_logit_comparison.rmse
              << " hybrid_logit_cosine=" << hybrid_logit_comparison.cosine
              << " hybrid_logit_max_abs=" << hybrid_logit_comparison.max_abs
              << " hybrid_gpu_to_host_us=" << hybrid_metrics.gpu_to_host_us
              << " hybrid_activation_pack_us="
              << hybrid_metrics.activation_pack_us
              << " hybrid_npu_command_us=" << hybrid_metrics.npu_command_us
              << " hybrid_npu_end_to_end_us="
              << hybrid_metrics.npu_end_to_end_us
              << " hybrid_host_to_gpu_us=" << hybrid_metrics.host_to_gpu_us
#endif
              << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "qwen_mtp_gpu_test failed: " << exception.what() << '\n';
    return 1;
  }
}

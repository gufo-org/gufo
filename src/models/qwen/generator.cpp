#include "src/models/qwen/generator.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace gufo::models {

QwenGenerator::QwenGenerator(
    QwenModelWeights weights,
    std::unique_ptr<tokenization::QwenTokenizer> tokenizer,
    std::uint32_t max_context)
    : weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      kv_cache_(weights_.config.FullAttentionLayerCount(),
                weights_.config.num_key_value_heads, max_context,
                weights_.config.head_dim),
      ssm_cache_(
          weights_.config.num_layers, weights_.config.SsmQkvSize(),
          weights_.config.ssm_conv_kernel, weights_.config.ssm_time_step_rank,
          weights_.config.ssm_state_size, weights_.config.SsmValueSize()),
      arena_(weights_.config) {}

std::unique_ptr<QwenGenerator> QwenGenerator::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  auto weights_opt = QwenModelWeights::LoadFromGguf(reader, error_msg);
  if (!weights_opt.has_value()) {
    return nullptr;
  }

  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(reader, error_msg);
  if (!tokenizer || tokenizer->GetVocabSize() <= 256) {
    auto bin_tok = tokenization::QwenTokenizer::CreateFromBinaryFile(
        "models/qwen_vocab.bin");
    if (bin_tok) {
      tokenizer = std::move(bin_tok);
    } else if (!tokenizer) {
      return nullptr;
    }
  }

  const std::uint32_t context_len = weights_opt->config.context_length > 0
                                        ? weights_opt->config.context_length
                                        : 4096;
  return std::make_unique<QwenGenerator>(std::move(*weights_opt),
                                         std::move(tokenizer), context_len);
}

std::vector<tokenization::TokenId> QwenGenerator::Generate(
    std::span<const tokenization::TokenId> prompt_tokens,
    const GenerationOptions& options,
    const std::function<bool(tokenization::TokenId, std::string_view)>&
        on_token) {
  std::vector<tokenization::TokenId> output_tokens;
  if (prompt_tokens.empty()) {
    return output_tokens;
  }

  kv_cache_.Reset();
  ssm_cache_.Reset();

  // 1. Prefill prompt tokens
  for (std::size_t p = 0; p < prompt_tokens.size(); ++p) {
    ForwardModel(prompt_tokens[p], static_cast<std::uint32_t>(p), weights_,
                 kv_cache_, ssm_cache_, arena_, arena_.logits);
  }

  sampling::SamplerState sampler(options.sampling, prompt_tokens);

  // 2. Decode first generated token
  auto next_token = sampler.Sample(arena_.logits);
  std::size_t cur_pos = prompt_tokens.size();

  // 3. Auto-regressive decode loop
  while (output_tokens.size() < options.max_new_tokens) {
    if (next_token == options.eos_token_id ||
        next_token == options.endoftext_token_id) {
      break;
    }

    output_tokens.push_back(next_token);
    sampler.Accept(next_token);

    if (on_token != nullptr) {
      const std::array<tokenization::TokenId, 1> single_tok = {next_token};
      const std::string text_piece = tokenizer_->Decode(single_tok);
      const bool continue_gen = on_token(next_token, text_piece);
      if (!continue_gen) {
        break;
      }
    }

    ForwardModel(next_token, static_cast<std::uint32_t>(cur_pos), weights_,
                 kv_cache_, ssm_cache_, arena_, arena_.logits);
    next_token = sampler.Sample(arena_.logits);
    ++cur_pos;
  }

  return output_tokens;
}

std::string QwenGenerator::GenerateText(std::string_view prompt,
                                        const GenerationOptions& options) {
  const auto prompt_tokens = tokenizer_->Encode(prompt);
  const auto generated_tokens = Generate(prompt_tokens, options);
  return tokenizer_->Decode(generated_tokens);
}

}  // namespace gufo::models

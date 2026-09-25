#include "src/models/qwen/generator.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

void TestSyntheticGeneration() {
  gufo::core::ModelConfig config;
  config.architecture = "qwen35";
  config.num_layers = 1;
  config.hidden_size = 4;
  config.intermediate_size = 8;
  config.num_attention_heads = 2;
  config.num_key_value_heads = 2;
  config.head_dim = 2;
  config.vocab_size = 8;
  config.context_length = 32;
  config.full_attention_interval = 1;
  config.is_text_only = true;

  // Embedding table [vocab=8, hidden=4]
  std::vector<float> embd(config.vocab_size * config.hidden_size, 0.1F);
  // Layer weights
  std::vector<float> norm(config.hidden_size, 1.0F);
  std::vector<float> q_w(config.hidden_size * config.hidden_size, 0.05F);
  std::vector<float> k_w(config.hidden_size * config.hidden_size, 0.05F);
  std::vector<float> v_w(config.hidden_size * config.hidden_size, 0.05F);
  std::vector<float> o_w(config.hidden_size * config.hidden_size, 0.05F);
  std::vector<float> gate_w(config.intermediate_size * config.hidden_size,
                            0.05F);
  std::vector<float> up_w(config.intermediate_size * config.hidden_size, 0.05F);
  std::vector<float> down_w(config.hidden_size * config.intermediate_size,
                            0.05F);
  std::vector<float> out_w(config.vocab_size * config.hidden_size, 0.1F);

  auto make_ref =
      [](const std::vector<float>& v) -> gufo::models::QwenTensorRef {
    return {.data = v.data(),
            .type = gufo::core::GgmlType::kF32,
            .num_elements = v.size()};
  };

  gufo::models::QwenModelWeights weights;
  weights.config = config;
  weights.token_embd = make_ref(embd);
  weights.output_norm = make_ref(norm);
  weights.output = make_ref(out_w);
  weights.layers.resize(1);
  weights.layers[0].is_full_attention = true;
  weights.layers[0].attn_norm = make_ref(norm);
  weights.layers[0].attn_q = make_ref(q_w);
  weights.layers[0].attn_k = make_ref(k_w);
  weights.layers[0].attn_v = make_ref(v_w);
  weights.layers[0].attn_output = make_ref(o_w);
  weights.layers[0].ffn_norm = make_ref(norm);
  weights.layers[0].ffn_gate = make_ref(gate_w);
  weights.layers[0].ffn_up = make_ref(up_w);
  weights.layers[0].ffn_down = make_ref(down_w);

  const std::vector<std::string> vocab = {
      "<unk>", "hello", "world", "!", "a", "b", "c", "<|im_end|>"};
  const std::vector<std::string> merges = {};
  const std::unordered_map<std::string, gufo::tokenization::TokenId> specials =
      {
          {"<|im_end|>", 7},
      };

  auto tokenizer = gufo::tokenization::QwenTokenizer::CreateFromVocabulary(
      vocab, merges, specials);
  assert(tokenizer != nullptr);

  gufo::models::QwenGenerator generator(std::move(weights),
                                        std::move(tokenizer), 32);

  const std::vector<gufo::tokenization::TokenId> prompt = {1,
                                                           2};  // "hello world"
  gufo::models::GenerationOptions gen_opts;
  gen_opts.max_new_tokens = 4;

  std::vector<gufo::tokenization::TokenId> generated;
  auto out = generator.Generate(
      prompt, gen_opts, [&](gufo::tokenization::TokenId tok, std::string_view) {
        generated.push_back(tok);
        return true;
      });

  assert(out.size() <= 4);
  assert(generated.size() == out.size());
}

int main() {
  TestSyntheticGeneration();
  std::cout << "All Qwen generator tests passed.\n";
  return 0;
}

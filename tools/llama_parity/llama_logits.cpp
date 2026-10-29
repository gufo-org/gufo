// llama.cpp side of model-oracle logit comparisons: dumps float32 logits for
// every position of a token sequence, or prints the tokenization of a text.
// Gufo's per-model reference probes write the same layout (one vocabulary-wide
// row per position), which compare_logits.py reads.
//
//   llama_logits MODEL.gguf --tokens 1,2,3 --dump OUT.bin [--cpu] [--f32-kv]
//   llama_logits MODEL.gguf --text TEXT [--special]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "llama.h"

namespace {

std::vector<llama_token> ParseTokens(const std::string& text) {
  std::vector<llama_token> out;
  std::size_t start = 0;
  while (start < text.size()) {
    const std::size_t end = text.find(',', start);
    out.push_back(std::atoi(text.substr(start, end - start).c_str()));
    if (end == std::string::npos)
      break;
    start = end + 1;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s MODEL.gguf --tokens 1,2,3 --dump OUT.bin "
                 "[--cpu] [--f32-kv]\n"
                 "       %s MODEL.gguf --text TEXT [--special]\n",
                 argv[0], argv[0]);
    return 2;
  }
  std::string tokens_arg, text, dump;
  bool have_text = false, special = false, f32_kv = false, cpu = false;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--tokens" && i + 1 < argc) {
      tokens_arg = argv[++i];
    } else if (arg == "--text" && i + 1 < argc) {
      text = argv[++i];
      have_text = true;
    } else if (arg == "--dump" && i + 1 < argc) {
      dump = argv[++i];
    } else if (arg == "--cpu") {
      cpu = true;
    } else if (arg == "--special") {
      special = true;
    } else if (arg == "--f32-kv") {
      f32_kv = true;
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }

  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = cpu ? 0 : 999;
  mp.vocab_only = have_text;
  llama_model* model = llama_model_load_from_file(argv[1], mp);
  if (model == nullptr) {
    std::fprintf(stderr, "failed to load %s\n", argv[1]);
    return 1;
  }
  const llama_vocab* vocab = llama_model_get_vocab(model);

  if (have_text) {
    std::vector<llama_token> tokens(text.size() + 8);
    const int n = llama_tokenize(vocab, text.data(),
                                 static_cast<int32_t>(text.size()),
                                 tokens.data(),
                                 static_cast<int32_t>(tokens.size()),
                                 /*add_special=*/false, special);
    if (n < 0) {
      std::fprintf(stderr, "tokenization failed\n");
      return 1;
    }
    for (int i = 0; i < n; ++i)
      std::printf(i == 0 ? "%d" : ",%d", tokens[i]);
    std::printf("\n");
    llama_model_free(model);
    return 0;
  }

  const auto tokens = ParseTokens(tokens_arg);
  if (tokens.empty() || dump.empty()) {
    std::fprintf(stderr, "--tokens and --dump are required\n");
    return 2;
  }
  auto cp = llama_context_default_params();
  const auto n = static_cast<uint32_t>(tokens.size());
  cp.n_ctx = n < 256 ? 256 : n + 16;
  cp.n_batch = n;
  cp.n_ubatch = n;
  cp.n_seq_max = 1;
  // Without this, host-resident weights still run large matmuls on the GPU.
  cp.op_offload = !cpu;
  if (f32_kv) {
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
  }
  llama_context* ctx = llama_init_from_model(model, cp);
  if (ctx == nullptr) {
    std::fprintf(stderr, "failed to create context\n");
    return 1;
  }
  llama_batch batch = llama_batch_init(static_cast<int32_t>(n), 0, 1);
  for (uint32_t i = 0; i < n; ++i) {
    batch.token[i] = tokens[i];
    batch.pos[i] = static_cast<llama_pos>(i);
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = 1;
  }
  batch.n_tokens = static_cast<int32_t>(n);
  if (llama_decode(ctx, batch) != 0) {
    std::fprintf(stderr, "llama_decode failed\n");
    return 1;
  }
  const int vocab_size = llama_vocab_n_tokens(vocab);
  std::ofstream out(dump, std::ios::binary);
  for (uint32_t i = 0; i < n; ++i) {
    const float* row = llama_get_logits_ith(ctx, static_cast<int32_t>(i));
    out.write(reinterpret_cast<const char*>(row),
              static_cast<std::streamsize>(vocab_size * sizeof(float)));
  }
  std::fprintf(stderr, "wrote %u x %d logits to %s\n", n, vocab_size,
               dump.c_str());
  llama_batch_free(batch);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}

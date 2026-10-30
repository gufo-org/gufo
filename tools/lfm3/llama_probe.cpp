#include <algorithm>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

#include "llama.h"
int main(int argc, char** argv) {
  const bool gpu = argc > 1 && std::string_view(argv[1]) == "--gpu";
  const int first = gpu ? 2 : 1;
  if (argc < first + 3)
    return 2;
  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = gpu ? 999 : 0;
  auto* m = llama_model_load_from_file(argv[first], mp);
  if (!m)
    return 1;
  std::vector<llama_token> tokens;
  for (int i = first + 2; i < argc; ++i)
    tokens.push_back(std::stoi(argv[i]));
  auto cp = llama_context_default_params();
  cp.n_ctx = std::max<std::size_t>(2048, tokens.size() + 16);
  cp.type_k = GGML_TYPE_F32;
  cp.type_v = GGML_TYPE_F32;
  cp.n_batch = 512;
  cp.n_ubatch = 512;
  cp.n_threads = 8;
  cp.n_threads_batch = 8;
  cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  auto* c = llama_init_from_model(m, cp);
  if (!c)
    return 1;
  auto* v = llama_model_get_vocab(m);
  for (std::size_t offset = 0; offset < tokens.size(); offset += cp.n_batch) {
    auto count = std::min<std::size_t>(cp.n_batch, tokens.size() - offset);
    auto batch = llama_batch_get_one(tokens.data() + offset, count);
    if (llama_decode(c, batch))
      return 1;
  }
  auto* y = llama_get_logits_ith(c, -1);
  std::ofstream f(argv[first + 1], std::ios::binary);
  f.write((char*)y, llama_vocab_n_tokens(v) * 4);
  auto t = std::max_element(y, y + llama_vocab_n_tokens(v)) - y;
  std::cerr << "argmax=" << t << " logit=" << y[t] << "\n";
  llama_free(c);
  llama_model_free(m);
  llama_backend_free();
}

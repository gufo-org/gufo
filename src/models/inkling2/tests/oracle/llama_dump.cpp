// Oracle side of the M1 check: teacher-forces a token-id file through llama.cpp
// (libllama) and writes the same dump format as inkling2_oracle (see
// oracle.cpp). MIT; links libllama.
//   llama_dump MODEL.gguf IDS.bin OUT.bin [n_threads] [n_ubatch] [REF.bin|-]
//   [NGL]
// NGL = layers offloaded to the GPU backend (default 0: CPU, the exact oracle).
// With REF.bin the 4th column holds this run's logits at the reference dump's
// top-64 ids.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "llama.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: llama_dump MODEL IDS OUT [threads] [ubatch]\n");
    return 2;
  }
  const int threads = argc > 4 ? std::atoi(argv[4]) : 16;
  const int ub = argc > 5 ? std::atoi(argv[5]) : 512;
  std::ifstream f(argv[2], std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  const std::string raw = ss.str();
  std::vector<llama_token> ids(raw.size() / 4);
  std::memcpy(ids.data(), raw.data(), ids.size() * 4);

  std::vector<int32_t> ref_ids;
  if (argc > 6 && std::string(argv[6]) != "-") {
    std::ifstream rf(argv[6], std::ios::binary);
    std::ostringstream rs;
    rs << rf.rdbuf();
    const std::string rr = rs.str();
    const size_t rec = 4 + 64 * 12;
    const size_t n = (rr.size() - 16) / rec;
    ref_ids.resize(n * 64);
    for (size_t p = 0; p < n; ++p)
      std::memcpy(&ref_ids[p * 64], rr.data() + 16 + p * rec + 4, 256);
  }
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = argc > 7 ? std::atoi(argv[7]) : 0;
  llama_model* model = llama_model_load_from_file(argv[1], mp);
  if (!model) {
    std::fprintf(stderr, "load failed\n");
    return 1;
  }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int V = llama_vocab_n_tokens(vocab);
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = static_cast<uint32_t>(ids.size() + 16);
  cp.n_batch = ub;
  cp.n_ubatch = ub;
  cp.n_threads = threads;
  cp.n_threads_batch = threads;
  llama_context* ctx = llama_init_from_model(model, cp);
  if (!ctx) {
    std::fprintf(stderr, "context failed\n");
    return 1;
  }
  std::fprintf(stderr, "vocab %d, %zu tokens\n", V, ids.size());

  std::ofstream out(argv[3], std::ios::binary);
  const char magic[4] = {'I', '2', 'O', 'R'};
  const uint32_t version = 1, n_pos = static_cast<uint32_t>(ids.size()),
                 top = 64;
  out.write(magic, 4);
  out.write(reinterpret_cast<const char*>(&version), 4);
  out.write(reinterpret_cast<const char*>(&n_pos), 4);
  out.write(reinterpret_cast<const char*>(&top), 4);

  llama_batch batch = llama_batch_init(ub, 0, 1);
  std::vector<int> order(V);
  for (size_t start = 0; start < ids.size(); start += ub) {
    const int n = static_cast<int>(std::min<size_t>(ub, ids.size() - start));
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
      batch.token[i] = ids[start + i];
      batch.pos[i] = static_cast<llama_pos>(start + i);
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = 1;
    }
    if (llama_decode(ctx, batch) != 0) {
      std::fprintf(stderr, "decode failed at %zu\n", start);
      return 1;
    }
    for (int i = 0; i < n; ++i) {
      const float* l = llama_get_logits_ith(ctx, i);
      double mx = -1e30;
      for (int k = 0; k < V; ++k)
        mx = std::max<double>(mx, l[k]);
      double sum = 0;
      for (int k = 0; k < V; ++k)
        sum += std::exp(static_cast<double>(l[k]) - mx);
      const float lse = static_cast<float>(mx + std::log(sum));
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(
          order.begin(), order.begin() + 64, order.end(),
          [&](int x, int y) { return l[x] > l[y] || (l[x] == l[y] && x < y); });
      int32_t tid[64];
      float tl[64];
      for (int k = 0; k < 64; ++k) {
        tid[k] = order[k];
        tl[k] = l[order[k]];
      }
      out.write(reinterpret_cast<const char*>(&lse), 4);
      out.write(reinterpret_cast<const char*>(tid), sizeof tid);
      float at_ref[64];
      for (int k = 0; k < 64; ++k)
        at_ref[k] = ref_ids.empty() ? tl[k] : l[ref_ids[(start + i) * 64 + k]];
      out.write(reinterpret_cast<const char*>(tl), sizeof tl);
      out.write(reinterpret_cast<const char*>(at_ref), sizeof at_ref);
    }
    std::fprintf(stderr, "done %zu / %zu\n", start + n, ids.size());
  }
  return 0;
}

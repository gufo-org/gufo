// llama.cpp side of DFlash drafter comparisons. Decodes a token sequence with
// the target, injects its extracted layer inputs into the drafter as
// common/speculative.cpp does, and drafts one block at each anchor position
// from the features before it. Writes the target features, the drafter's
// projected context rows, and every anchor's block logits and LiLiCorr
// lattice rows, and prints the greedy lattice walk of each block.
//
//   llama_dflash TARGET.gguf DRAFT.gguf --tokens 1,2,3 --anchors 10,20
//                --dump OUT.bin [--f32-kv]
//
// Layout (little endian): "DFL1", then u32 n_tokens, n_target_layers,
// n_embd_target, n_anchors, block_size, vocab, n_embd_draft; float32
// features[n_tokens][n_target_layers * n_embd_target] (target layer order);
// float32 projected[n_tokens][n_embd_draft] (zero from the last anchor on,
// which is never injected); per anchor: i32 position,
// i32 anchor token, float32 logits[block_size][vocab], float32
// lattice[block_size][n_embd_draft].
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "common.h"
#include "llama-ext.h"
#include "llama.h"

namespace {

std::vector<int> ParseList(const std::string& text) {
  std::vector<int> out;
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

template<class T>
void Write(std::ofstream& out, const T* data, std::size_t count) {
  out.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(count * sizeof(T)));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s TARGET.gguf DRAFT.gguf --tokens 1,2,3 "
                 "--anchors 10,20 --dump OUT.bin [--f32-kv]\n",
                 argv[0]);
    return 2;
  }
  std::string tokens_arg, anchors_arg, dump;
  bool f32_kv = false;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--tokens" && i + 1 < argc) {
      tokens_arg = argv[++i];
    } else if (arg == "--anchors" && i + 1 < argc) {
      anchors_arg = argv[++i];
    } else if (arg == "--dump" && i + 1 < argc) {
      dump = argv[++i];
    } else if (arg == "--f32-kv") {
      f32_kv = true;
    } else {
      std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  const auto tokens = ParseList(tokens_arg);
  auto anchors = ParseList(anchors_arg);
  std::sort(anchors.begin(), anchors.end());
  const int n = static_cast<int>(tokens.size());
  if (n == 0 || anchors.empty() || dump.empty() || anchors.front() < 1 ||
      anchors.back() >= n) {
    std::fprintf(stderr, "need tokens, anchors in [1, n_tokens) and --dump\n");
    return 2;
  }

  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = 999;
  llama_model* target = llama_model_load_from_file(argv[1], mp);
  llama_model* draft = llama_model_load_from_file(argv[2], mp);
  if (target == nullptr || draft == nullptr) {
    std::fprintf(stderr, "failed to load the models\n");
    return 1;
  }
  const int n_layers = static_cast<int>(llama_model_target_layer_ids_n(draft));
  const int32_t* layer_ids = llama_model_target_layer_ids(draft);
  const int n_embd_tgt = llama_model_n_embd(target);
  const int n_embd_dec = llama_model_n_embd(draft);
  const int n_embd_enc = n_layers * n_embd_tgt;
  int block = 0;
  {
    char buf[32] = {};
    if (llama_model_meta_val_str(draft, "dflash.block_size", buf,
                                 sizeof(buf)) >= 0)
      block = std::atoi(buf);
  }
  const llama_vocab* vocab = llama_model_get_vocab(target);
  const int n_vocab = llama_vocab_n_tokens(vocab);
  const llama_token mask = llama_vocab_mask(llama_model_get_vocab(draft));
  if (block < 2) {
    std::fprintf(stderr, "draft has no block size\n");
    return 1;
  }

  auto cp = llama_context_default_params();
  cp.n_ctx = static_cast<uint32_t>(std::max(n + 64, 512));
  cp.n_batch = cp.n_ctx;
  cp.n_ubatch = cp.n_ctx;
  cp.n_seq_max = 1;
  if (f32_kv) {
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
  }
  llama_context* ctx_tgt = llama_init_from_model(target, cp);
  if (ctx_tgt == nullptr) {
    std::fprintf(stderr, "failed to create the target context\n");
    return 1;
  }
  for (int k = 0; k < n_layers; ++k)
    llama_set_embeddings_layer_inp(ctx_tgt, static_cast<uint32_t>(layer_ids[k]),
                                   true);
  auto cp_dft = cp;
  cp_dft.ctx_other = ctx_tgt;
  llama_context* ctx_dft = llama_init_from_model(draft, cp_dft);
  if (ctx_dft == nullptr) {
    std::fprintf(stderr, "failed to create the draft context\n");
    return 1;
  }
  llama_set_embeddings_nextn(ctx_dft, true, /*masked=*/false);
  llama_set_causal_attn(ctx_dft, false);

  // Target pass: one batch, so every row's layer inputs stay readable.
  llama_batch batch = llama_batch_init(n, 0, 1);
  for (int i = 0; i < n; ++i) {
    batch.token[i] = tokens[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = i == n - 1;
  }
  batch.n_tokens = n;
  if (llama_decode(ctx_tgt, batch) != 0) {
    std::fprintf(stderr, "target decode failed\n");
    return 1;
  }
  std::vector<float> features(static_cast<std::size_t>(n) * n_embd_enc);
  for (int k = 0; k < n_layers; ++k) {
    const float* layer = llama_get_embeddings_layer_inp(
        ctx_tgt, static_cast<uint32_t>(layer_ids[k]));
    if (layer == nullptr) {
      std::fprintf(stderr, "layer %d input not extracted\n", layer_ids[k]);
      return 1;
    }
    for (int i = 0; i < n; ++i)
      std::memcpy(features.data() + static_cast<std::size_t>(i) * n_embd_enc +
                      static_cast<std::size_t>(k) * n_embd_tgt,
                  layer + static_cast<std::size_t>(i) * n_embd_tgt,
                  static_cast<std::size_t>(n_embd_tgt) * sizeof(float));
  }

  std::ofstream out(dump, std::ios::binary);
  out.write("DFL1", 4);
  const std::uint32_t header[] = {
      static_cast<std::uint32_t>(n),         static_cast<std::uint32_t>(n_layers),
      static_cast<std::uint32_t>(n_embd_tgt), static_cast<std::uint32_t>(anchors.size()),
      static_cast<std::uint32_t>(block),     static_cast<std::uint32_t>(n_vocab),
      static_cast<std::uint32_t>(n_embd_dec)};
  Write(out, header, std::size(header));
  Write(out, features.data(), features.size());

  // Injection in anchor order: the drafter cache holds exactly the features
  // before each anchor when that anchor drafts.
  std::vector<float> projected(static_cast<std::size_t>(n) * n_embd_dec);
  std::vector<float> block_logits(static_cast<std::size_t>(block) * n_vocab);
  std::vector<float> lattice(static_cast<std::size_t>(block) * n_embd_dec);
  std::vector<std::vector<float>> anchor_logits, anchor_lattice;
  common_batch inject(ctx_dft);
  common_batch noise(ctx_dft);
  llama_memory_t mem = llama_get_memory(ctx_dft);
  int injected = 0;
  for (const int anchor : anchors) {
    if (anchor > injected) {
      inject.clear();
      for (int i = injected; i < anchor; ++i) {
        const llama_pos pos[4] = {i, i, i, 0};
        inject.add_embd({features.data() + static_cast<std::size_t>(i) * n_embd_enc,
                         1, static_cast<std::size_t>(n_embd_enc)},
                        pos, 0, false);
      }
      if (llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, inject.get()) != 0) {
        std::fprintf(stderr, "injection failed\n");
        return 1;
      }
      const float* rows = llama_get_embeddings_nextn(ctx_dft);
      if (rows == nullptr) {
        std::fprintf(stderr, "injection produced no projected rows\n");
        return 1;
      }
      std::memcpy(projected.data() + static_cast<std::size_t>(injected) * n_embd_dec,
                  rows,
                  static_cast<std::size_t>(anchor - injected) * n_embd_dec *
                      sizeof(float));
      injected = anchor;
    }
    noise.clear();
    for (int i = 0; i < block; ++i) {
      const int idx = noise.add(i == 0 ? tokens[anchor] : mask, anchor + i, 0,
                                true);
      noise.set_embd_state(
          idx, {projected.data() + static_cast<std::size_t>(anchor - 1) * n_embd_dec,
                1, static_cast<std::size_t>(n_embd_dec)});
    }
    if (llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, noise.get()) != 0) {
      std::fprintf(stderr, "draft decode failed\n");
      return 1;
    }
    for (int i = 0; i < block; ++i)
      std::memcpy(block_logits.data() + static_cast<std::size_t>(i) * n_vocab,
                  llama_get_logits_ith(ctx_dft, i),
                  static_cast<std::size_t>(n_vocab) * sizeof(float));
    std::memcpy(lattice.data(), llama_get_embeddings_nextn(ctx_dft),
                lattice.size() * sizeof(float));
    anchor_logits.push_back(block_logits);
    anchor_lattice.push_back(lattice);
    // Drop the noise block; the next anchor injects real features there.
    llama_memory_seq_rm(mem, 0, anchor, -1);
    injected = std::min(injected, anchor);

    // Greedy lattice walk as common/speculative.cpp does it.
    char top_k_buf[32] = {};
    llama_model_meta_val_str(draft, "lilicorr.candidate_topk", top_k_buf,
                             sizeof(top_k_buf));
    const int top_k = std::atoi(top_k_buf);
    int predecessor = 0;
    std::printf("anchor %d:", anchor);
    for (int i = 1; i < block; ++i) {
      const float* row = lattice.data() + static_cast<std::size_t>(i) * n_embd_dec;
      const float* scores = row + top_k + predecessor * top_k;
      predecessor = static_cast<int>(
          std::max_element(scores, scores + top_k) - scores);
      const int draft_token = static_cast<int>(row[predecessor]);
      const int truth = anchor + i < n ? tokens[anchor + i] : -1;
      std::printf(" %d%s", draft_token, draft_token == truth ? "" : "*");
    }
    std::printf("\n");
  }
  Write(out, projected.data(), projected.size());
  for (std::size_t a = 0; a < anchors.size(); ++a) {
    const std::int32_t meta[] = {anchors[a], tokens[anchors[a]]};
    Write(out, meta, 2);
    Write(out, anchor_logits[a].data(), anchor_logits[a].size());
    Write(out, anchor_lattice[a].data(), anchor_lattice[a].size());
  }
  std::fprintf(stderr, "wrote %d tokens and %zu anchors to %s\n", n,
               anchors.size(), dump.c_str());
  llama_batch_free(batch);
  llama_free(ctx_dft);
  llama_free(ctx_tgt);
  llama_model_free(draft);
  llama_model_free(target);
  llama_backend_free();
  return 0;
}

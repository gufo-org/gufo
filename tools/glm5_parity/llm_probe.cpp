// Greedy-continuation and top-k logit fixture probe for GLM-5.3-Flash parity.
//
// Loads a GGUF on the GPU, then for each embedded prompt feeds the tokens
// one per llama_decode call (sequential prefill, so every position's state
// matches a real streaming decode), recording top-k logits (fp32) at each
// position, then greedily samples n_gen tokens the same way. Output is a
// stable JSON document used as the wt semantic anchor; wt parity is
// assessed distributionally, not by a logit budget (see
// docs/models/glm-5.3-flash/BENCHMARKS.md#fixtures).
//
// Env: GUFO_PROBE_DUMP=<dir> writes per-stage f32 debug dumps;
//      GUFO_PROBE_STEPS=<file> writes the per-step top-k document.
//
// Build (in a llama.cpp checkout containing libllama for the target device):
//   g++ -std=c++17 -Iinclude -Iggml/include llm_probe.cpp -Lbin
//   -lllama -lggml -lggml-base -Wl,-rpath,$PWD/bin -o llm_probe
//
// Usage: llm_probe MODEL REVISION OUT_JSON [n_gpu_layers]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"

namespace {

constexpr int32_t n_ctx = 2048;
constexpr int32_t n_gen = 24;
constexpr int32_t k_top = 32;
constexpr int32_t max_prompt = 768;

struct Prompt {
  const char* name;
  std::string text;
};

std::vector<Prompt> prompts() {
  std::vector<Prompt> out;
  out.push_back({"fact", "The capital of France is"});
  out.push_back({"code", "def fibonacci(n):\n"});
  out.push_back(
      {"multi", "Translate to German: The machine learns very fast.\nGerman:"});

  std::string long_text;
  const char* sentences[] = {
      "The Strix Halo integrates a large unified memory architecture. ",
      "Sparse attention selects a small working set of keys for each query. ",
      "Recurrent state carries information between tokens without a cache. ",
      "Mixture of experts routes each token to a few specialized networks. ",
      "Quantized weights reduce bandwidth, which dominates decode speed. ",
      "Deterministic sampling makes numerical regressions visible. ",
  };
  for (int rep = 0; rep < (int)(sizeof(sentences) / sizeof(sentences[0]));
       ++rep) {
    long_text += sentences[rep];
  }
  out.push_back({"long", long_text});

  std::string rep;
  for (int i = 0; i < 30; ++i) {
    rep += "one two three four five six seven eight nine ten. ";
  }
  out.push_back({"repeat", rep});
  return out;
}

std::string json_escape(const char* s) {
  std::string out;
  for (const char* p = s; *p; ++p) {
    switch (*p) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\r':
        out += "\\r";
        break;
      default:
        if ((unsigned char)*p < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", *p);
          out += buf;
        } else {
          out += *p;
        }
    }
  }
  return out;
}

void write_top_k(FILE* f, const float* logits, int32_t n_vocab,
                 bool comma_before) {
  std::vector<int32_t> idx(n_vocab);
  for (int32_t i = 0; i < n_vocab; ++i) {
    idx[i] = i;
  }
  const int32_t k = std::min<int32_t>(k_top, n_vocab);
  std::partial_sort(
      idx.begin(), idx.begin() + k, idx.end(),
      [logits](int32_t a, int32_t b) { return logits[a] > logits[b]; });
  for (int32_t i = 0; i < k; ++i) {
    fprintf(f, "%s\n        [%d, %.9g]", comma_before || i > 0 ? "," : "",
            idx[i], (double)logits[idx[i]]);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s MODEL REVISION OUT_JSON [n_gpu_layers]\n",
            argv[0]);
    return 1;
  }
  const char* model_path = argv[1];
  const char* revision = argv[2];
  const char* out_path = argv[3];
  const int32_t ngl = argc > 4 ? atoi(argv[4]) : 99;
  // Optional side-output: top-k logits at EVERY prefill position and every
  // generation step (per-case, for executor bisection). The main fixture is
  // unaffected.
  const char* steps_path = std::getenv("GUFO_PROBE_STEPS");
  FILE* steps = nullptr;
  if (steps_path != nullptr && steps_path[0] != '\0') {
    steps = fopen(steps_path, "w");
    if (steps == nullptr) {
      fprintf(stderr, "failed to open %s\n", steps_path);
      return 1;
    }
    fprintf(steps, "{\"cases\":[\n");
  }

  llama_backend_init();

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = ngl;
  llama_model* model = llama_model_load_from_file(model_path, mparams);
  if (!model) {
    fprintf(stderr, "failed to load model: %s\n", model_path);
    return 1;
  }

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = n_ctx;
  cparams.n_batch = 512;
  cparams.n_ubatch = 512;
  cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

  llama_context* ctx = llama_init_from_model(model, cparams);
  if (!ctx) {
    fprintf(stderr, "failed to create context\n");
    return 1;
  }

  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int32_t n_vocab = llama_vocab_n_tokens(vocab);

  FILE* out = fopen(out_path, "w");
  if (!out) {
    fprintf(stderr, "failed to open %s\n", out_path);
    return 1;
  }

  fprintf(out, "{\n");
  fprintf(out, "  \"schema\": \"gufo.glm_next.llama_reference.v1\",\n");
  fprintf(out, "  \"engine\": \"llama.cpp\",\n");
  fprintf(out, "  \"revision\": \"%s\",\n", json_escape(revision).c_str());
  fprintf(out, "  \"model\": \"%s\",\n", json_escape(model_path).c_str());
  fprintf(out, "  \"n_gpu_layers\": %d,\n", ngl);
  fprintf(out, "  \"flash_attn\": true,\n");
  fprintf(out, "  \"n_ctx\": %d,\n", n_ctx);
  fprintf(out, "  \"sampling\": \"greedy argmax\",\n");
  fprintf(out, "  \"top_k\": %d,\n", k_top);
  fprintf(out, "  \"n_gen\": %d,\n", n_gen);
  fprintf(out, "  \"cases\": [\n");

  const auto prompts_list = prompts();
  for (size_t ci = 0; ci < prompts_list.size(); ++ci) {
    const Prompt& p = prompts_list[ci];

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

    std::vector<llama_token> toks(max_prompt);
    int32_t n_toks =
        llama_tokenize(vocab, p.text.c_str(), (int32_t)p.text.size(),
                       toks.data(), max_prompt, /* add_bos */ true,
                       /* parse_special */ true);
    if (n_toks <= 0) {
      fprintf(stderr, "tokenization failed for case %s\n", p.name);
      return 1;
    }
    toks.resize(n_toks);

    fprintf(out, "%s  {\n", ci > 0 ? ",\n" : "");
    fprintf(out, "    \"name\": \"%s\",\n", json_escape(p.name).c_str());
    fprintf(out, "    \"prompt\": \"%s\",\n",
            json_escape(p.text.c_str()).c_str());
    fprintf(out, "    \"prompt_tokens\": [%d", toks[0]);
    for (int32_t i = 1; i < n_toks; ++i) {
      fprintf(out, ", %d", toks[i]);
    }
    fprintf(out, "],\n");

    // Sequential one-token-per-decode prefill. The mHC/KDA layers are
    // stateful: a fused batched ubatch is not numerically identical to the
    // per-step recurrence the executor runs, so the fixture must capture
    // the per-step semantics that the forward test replays.
    std::vector<float> logits_buf(n_vocab);
    if (steps != nullptr) {
      fprintf(steps, "%s  {\"name\": \"%s\", \"prefill\": [\n", ci > 0 ? ",\n"
                                                                      : "",
              json_escape(p.name).c_str());
    }
    for (int32_t i = 0; i < n_toks; ++i) {
      llama_batch nb = llama_batch_get_one(&toks[i], 1);
      if (llama_decode(ctx, nb) != 0) {
        fprintf(stderr, "prompt decode failed for case %s at %d\n", p.name, i);
        return 1;
      }
      memcpy(logits_buf.data(), llama_get_logits(ctx),
             n_vocab * sizeof(float));
      if (steps != nullptr) {
        fprintf(steps, "%s      [", i > 0 ? "],\n" : "");
        write_top_k(steps, logits_buf.data(), n_vocab, false);
      }
    }
    if (steps != nullptr) {
      fprintf(steps, "]],\n      \"gen\": [\n");
    }

    fprintf(out, "    \"logits_after_prefill\": [");
    write_top_k(out, logits_buf.data(), n_vocab, false);
    fprintf(out, "],\n");

    fprintf(out, "    \"greedy_tokens\": [");
    std::vector<int32_t> gen;
    for (int step = 0; step < n_gen; ++step) {
      const float* logits = logits_buf.data();
      int32_t best = 0;
      float best_v = logits[0];
      for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
          best_v = logits[i];
          best = i;
        }
      }
      if (step == 0) {
        fprintf(out, "%d", best);
      } else {
        fprintf(out, ", %d", best);
      }
      gen.push_back(best);

      if (steps != nullptr) {
        fprintf(steps, "%s      [", step > 0 ? "],\n" : "");
        write_top_k(steps, logits, n_vocab, false);
      }

      llama_batch nb = llama_batch_get_one(&best, 1);
      if (llama_decode(ctx, nb) != 0) {
        fprintf(stderr, "decode failed for case %s step %d\n", p.name, step);
        return 1;
      }
      memcpy(logits_buf.data(), llama_get_logits(ctx),
             n_vocab * sizeof(float));
    }
    fprintf(out, "],\n");

    fprintf(out, "    \"logits_after_gen\": [");
    write_top_k(out, logits_buf.data(), n_vocab, false);
    fprintf(out, "]\n  }\n");
    if (steps != nullptr) {
      fprintf(steps, "]]\n  }\n");
    }
  }

  fprintf(out, "  ]\n}\n");
  fclose(out);
  if (steps != nullptr) {
    fprintf(steps, "]\n}\n");
    fclose(steps);
  }

  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  fprintf(stderr, "fixtures written to %s\n", out_path);
  return 0;
}

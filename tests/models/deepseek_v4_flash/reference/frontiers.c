/* Test-only adapter over the pinned upstream public API; no Gufo math. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4.h"

static void require(int condition, const char* message) {
  if (!condition) {
    fprintf(stderr, "ERROR: %s\n", message);
    exit(1);
  }
}

static char* read_text(const char* path) {
  FILE* f = fopen(path, "rb");
  require(f != NULL, "open prompt");
  require(fseek(f, 0, SEEK_END) == 0, "seek prompt");
  long size = ftell(f);
  require(size >= 0 && fseek(f, 0, SEEK_SET) == 0, "size prompt");
  char* text = malloc((size_t)size + 1);
  require(text != NULL, "allocate prompt");
  require(fread(text, 1, (size_t)size, f) == (size_t)size, "read prompt");
  text[size] = 0;
  fclose(f);
  return text;
}

static void dump(const char* directory, unsigned prefill_step,
                 unsigned long depth, const char* suffix, const float* logits,
                 int vocabulary) {
  char path[4096];
  require(snprintf(path, sizeof(path), "%s/%u-%lu-%s.f32", directory,
                   prefill_step, depth, suffix) < (int)sizeof(path),
          "output path");
  FILE* f = fopen(path, "wb");
  require(f != NULL, "open logits");
  require(fwrite(logits, sizeof(float), (size_t)vocabulary, f) ==
              (size_t)vocabulary,
          "write logits");
  require(fclose(f) == 0, "close logits");
}

int main(int argc, char** argv) {
  const int prefill_only =
      argc == 5 && strcmp(argv[4], "--prefill-only") == 0;
  require(argc == 4 || prefill_only,
          "usage: frontier_reference MODEL INPUT-DIR OUTPUT-DIR [--prefill-only]");
  const int decode_tokens = prefill_only ? 0 : 128;
  ds4_engine_options options = {.model_path = argv[1],
                                .backend = DS4_BACKEND_CUDA,
                                .context_size = 20480,
                                .placement_ctx_hint = 20480,
                                .placement_session_count_hint = 1,
                                .prefill_chunk = 4096,
                                .power_percent = 100,
                                .warm_weights = false,
                                .quality = false};
  ds4_engine* engine = NULL;
  require(ds4_engine_open(&engine, &options) == 0, "open engine");
  const int vocabulary = ds4_engine_vocab_size(engine);
  require(vocabulary > 0, "vocabulary");
  float* logits = malloc((size_t)vocabulary * sizeof(float));
  require(logits != NULL, "allocate logits");
  char path[4096];
  require(snprintf(path, sizeof(path), "%s/targets.txt", argv[2]) <
              (int)sizeof(path),
          "targets path");
  FILE* targets_file = fopen(path, "r");
  require(targets_file != NULL, "open targets");
  int targets[128];
  for (int i = 0; i < 128; ++i)
    require(fscanf(targets_file, "%d", &targets[i]) == 1 && targets[i] >= 0 &&
                targets[i] < vocabulary,
            "target token");
  fclose(targets_file);
  snprintf(path, sizeof(path), "%s/manifest.tsv", argv[2]);
  FILE* manifest = fopen(path, "r");
  require(manifest != NULL, "open manifest");
  snprintf(path, sizeof(path), "%s/steps.tsv", argv[3]);
  FILE* output = fopen(path, "w");
  require(output != NULL, "open steps");
  fprintf(output,
          "prefill_step\tprefill_capacity\tdepth\tprompt_"
          "tokens\tstep\ttarget\tgreedy\tlogprob\n");
  unsigned long depth, prefix;
  unsigned prefill_step, prefill_capacity;
  int points = 0;
  while (fscanf(manifest, "%u\t%lu\t%lu\t%u\t%4095[^\n]\n", &prefill_step,
                &depth, &prefix, &prefill_capacity, path) == 5) {
    require(prefill_step == 2048 || prefill_step == 4096,
            "supported prefill step");
    char* text = read_text(path);
    ds4_tokens prompt = {0};
    ds4_tokenize_rendered_chat(engine, text, &prompt);
    free(text);
    require(prompt.len == (int)prefix && prefix + 128 < 20480,
            "matched prompt token count");
    snprintf(path, sizeof(path), "%s/%lu.tokens.txt", argv[2], depth);
    FILE* expected_tokens = fopen(path, "r");
    require(expected_tokens != NULL, "open expected prompt tokens");
    for (int i = 0; i < prompt.len; ++i) {
      int expected;
      require(fscanf(expected_tokens, "%d", &expected) == 1 &&
                  expected == prompt.v[i],
              "identical prompt token IDs");
    }
    fclose(expected_tokens);
    ds4_session* session = NULL;
    require(ds4_session_create(&session, engine, 20480) == 0, "create session");
    require(ds4_session_prefill_cap(session) == (int)prefill_capacity,
            "matched allocated prefill capacity");
    char error[512] = {0};
    ds4_tokens partial = prompt;
    partial.len = 0;
    while (partial.len < prompt.len) {
      const int remaining = prompt.len - partial.len;
      partial.len +=
          remaining < (int)prefill_step ? remaining : (int)prefill_step;
      require(ds4_session_sync(session, &partial, error, sizeof(error)) == 0,
              error);
    }
    for (int step = 0; step <= 128; ++step) {
      require(
          ds4_session_copy_logits(session, logits, vocabulary) == vocabulary,
          "copy logits");
      int greedy = 0;
      for (int i = 0; i < vocabulary; ++i) {
        require(isfinite(logits[i]), "finite logits");
        if (logits[i] > logits[greedy])
          greedy = i;
      }
      if (step == 0)
        dump(argv[3], prefill_step, depth, "before", logits, vocabulary);
      if (step == decode_tokens) {
        if (!prefill_only)
          dump(argv[3], prefill_step, depth, "after", logits, vocabulary);
        break;
      }
      const double maximum = logits[greedy];
      double sum = 0;
      for (int i = 0; i < vocabulary; ++i)
        sum += exp((double)logits[i] - maximum);
      const double logprob = logits[targets[step]] - (maximum + log(sum));
      require(isfinite(logprob), "finite likelihood");
      fprintf(output, "%u\t%u\t%lu\t%lu\t%d\t%d\t%d\t%.17g\n", prefill_step,
              prefill_capacity, depth, prefix, step, targets[step], greedy,
              logprob);
      require(
          ds4_session_eval(session, targets[step], error, sizeof(error)) == 0,
          error);
    }
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ++points;
    fprintf(stderr,
            "Independent frontier prefill=%u depth=%lu tokens=%d complete\n",
            prefill_step, depth, decode_tokens);
    fflush(output);
  }
  require(points == 10, "complete frontier matrix");
  require(fclose(output) == 0, "close steps");
  fclose(manifest);
  free(logits);
  ds4_engine_close(engine);
  return 0;
}

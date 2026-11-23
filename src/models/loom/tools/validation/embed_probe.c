// On-GPU check of the logits readback through libloom_text_embed alone (no Gufo):
//   embed_probe PACKAGE_DIR [STEPS=24] [ROWS=3] > result.json
// Opens the package with logits on and ROWS resident rows, prefills a different prompt on each, then for STEPS
// steps checks, for every row, that the exported logit row's argmax is the token the device selected (the request
// is greedy: "temperature": 0), padding is not selected, and that feeding the host's token
// (decode_batch_tokens) and the device's own token (decode_batch) from the same state agree. Rows decode together
// in one epoch, so the export slot of each span is exercised. A request without "logits": true must leave no row.
// Exit status 0 only when every check held.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "loom_text_embed.h"

#define MAX_ROWS 8

static int failures = 0;
#define CHECK(cond, ...)                                              \
  do {                                                                \
    if (!(cond)) {                                                    \
      ++failures;                                                     \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);            \
      fprintf(stderr, __VA_ARGS__);                                   \
      fprintf(stderr, "\n");                                          \
    }                                                                 \
  } while (0)

static int argmax(const float* row, size_t n) {
  int best = 0;
  for (size_t i = 1; i < n; ++i)
    if (row[i] > row[best]) best = (int)i;
  return best;
}

static loom_text_embed_request_t* prepare(loom_text_embed_t* e, const char* model, const char* prompt,
                                          int logits, int32_t* tokens, size_t capacity, size_t* count) {
  char body[4096];
  snprintf(body, sizeof(body),
           "{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],\"temperature\":0%s}",
           model, prompt, logits ? ",\"logits\":true" : "");
  loom_text_embed_request_t* request = NULL;
  if (loom_text_embed_chat_prepare(e, body, strlen(body), tokens, capacity, count, &request)) {
    fprintf(stderr, "chat_prepare: %s\n", loom_text_embed_error(e));
    exit(2);
  }
  return request;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: embed_probe PACKAGE_DIR [STEPS] [ROWS] [PRESET]\n");
    return 2;
  }
  const int steps = argc > 2 ? atoi(argv[2]) : 24;
  int rows = argc > 3 ? atoi(argv[3]) : 3;
  if (rows < 1) rows = 1;
  if (rows > MAX_ROWS) rows = MAX_ROWS;
  static const char* prompts[MAX_ROWS] = {
      "What is the capital of France?", "Write a haiku about the sea.", "Name three primary colors.",
      "Explain why the sky is blue in one sentence.", "Count from one to five.", "Say hello in Spanish.",
      "What is 17 times 3?", "Name a river in Africa."};
  char error[2048] = {0};
  loom_text_embed_options_ex_t options;
  memset(&options, 0, sizeof(options));
  options.struct_size = sizeof(options);
  options.base.model_path = argv[1];
  options.base.rows = (uint32_t)rows;
  if (argc > 4 && argv[4][0]) options.base.preset = argv[4];   // a card with several presets needs one named (loom: "select a preset explicitly")
  options.logits = 1;
  loom_text_embed_t* e = NULL;
  if (loom_text_embed_open_ex(&options, &e, error, sizeof(error))) {
    fprintf(stderr, "open: %s\n", error);
    printf("{\"opened\": false, \"error\": \"%.200s\"}\n", error);
    return 3;
  }
  loom_text_embed_info_t info;
  loom_text_embed_info(e, &info);
  loom_text_embed_logits_info_t li;
  loom_text_embed_logits_info(e, &li);
  CHECK(li.supported && li.enabled && li.width > 0, "logits supported=%u enabled=%u width=%u", li.supported,
        li.enabled, li.width);
  CHECK(li.max_rows >= (uint32_t)rows, "slots %u < rows %d", li.max_rows, rows);
  const int32_t vocabulary = loom_text_embed_vocabulary_size(e);
  float* logits = (float*)malloc(sizeof(float) * (li.width ? li.width : 1));
  int32_t* prompt_tokens = (int32_t*)malloc(sizeof(int32_t) * info.context_capacity);

  // Each row: a logit request, its own prompt.
  uint32_t row_ids[MAX_ROWS];
  int32_t host_token[MAX_ROWS];
  for (int r = 0; r < rows; ++r) {
    size_t count = 0;
    loom_text_embed_request_t* request =
        prepare(e, info.model_id, prompts[r], 1, prompt_tokens, info.context_capacity, &count);
    CHECK(loom_text_embed_request_logits(request) == 1, "request %d lost its logits key", r);
    row_ids[r] = (uint32_t)r;
    if (loom_text_embed_row_begin(e, (uint32_t)r, request, count + (uint64_t)steps + 8, 1)) {
      fprintf(stderr, "row_begin: %s\n", loom_text_embed_error(e));
      return 2;
    }
    CHECK(loom_text_embed_row_is_logit(e, (uint32_t)r), "row %d is not a logit row", r);
    if (loom_text_embed_row_prefill(e, (uint32_t)r, prompt_tokens, count)) {
      fprintf(stderr, "prefill: %s\n", loom_text_embed_error(e));
      return 2;
    }
    loom_text_embed_request_free(request);
  }

  int checked = 0, host_equals_device = 0, padding_hits = 0;
  int first_failure_step = -1;
  for (int step = 0; step <= steps; ++step) {
    int before = failures;
    for (int r = 0; r < rows; ++r) {
      size_t n = 0;
      if (loom_text_embed_row_logits(e, (uint32_t)r, logits, li.width, &n)) {
        CHECK(0, "step %d row %d: row_logits: %s", step, r, loom_text_embed_error(e));
        continue;
      }
      const int top = argmax(logits, n);
      const int device = loom_text_embed_row_token(e, (uint32_t)r);
      ++checked;
      // A tie of the row's maximum (bf16 logits repeat values: Qwen2.5-32B step 10, both GPUs) is any of the tied tokens: the device and
      // argmax() may break it differently.
      const int tied = device >= 0 && (size_t)device < n && logits[device] == logits[top];
      if (top == device || tied) ++host_equals_device;
      CHECK(top == device || tied, "step %d row %d: argmax(row)=%d (%.3f) but the device selected %d (%.3f)", step, r, top,
            logits[top], device, device >= 0 && (size_t)device < n ? logits[device] : -1.0f);
      if (top >= vocabulary) ++padding_hits;
      CHECK(top < vocabulary || vocabulary <= 0, "step %d row %d: argmax %d is padding (vocabulary %d)", step, r, top,
            vocabulary);
      CHECK(isfinite(logits[top]) && logits[top] > -1e30f, "step %d row %d: top logit %f", step, r, logits[top]);
      host_token[r] = top;
    }
    if (failures != before && first_failure_step < 0) first_failure_step = step;
    if (step == steps) break;
    int done = 0;
    for (int r = 0; r < rows; ++r) done += loom_text_embed_row_is_eos(e, (uint32_t)r);
    if (done == rows) break;
    // Alternate the two ways of consuming a selection: the host's token and the device's own.
    int rc = (step & 1) ? loom_text_embed_decode_batch(e, row_ids, (size_t)rows)
                        : loom_text_embed_decode_batch_tokens(e, row_ids, (size_t)rows, host_token);
    if (rc) {
      CHECK(0, "step %d: epoch: %s", step, loom_text_embed_error(e));
      break;
    }
  }

  // A request without the key leaves no logit row, and the default path still works on the same load.
  size_t count = 0;
  loom_text_embed_request_t* plain = prepare(e, info.model_id, prompts[0], 0, prompt_tokens, info.context_capacity, &count);
  CHECK(loom_text_embed_request_logits(plain) == 0, "plain request carries logits");
  loom_text_embed_row_end(e, 0);
  loom_text_embed_row_reset(e, 0);
  CHECK(loom_text_embed_row_begin(e, 0, plain, count + 16, 1) == 0, "plain row_begin: %s", loom_text_embed_error(e));
  CHECK(!loom_text_embed_row_is_logit(e, 0), "plain row is a logit row");
  CHECK(loom_text_embed_row_prefill(e, 0, prompt_tokens, count) == 0, "plain prefill: %s", loom_text_embed_error(e));
  size_t n = 0;
  CHECK(loom_text_embed_row_logits(e, 0, logits, li.width, &n) != 0, "an unflagged row returned logits");
  loom_text_embed_request_free(plain);

  printf("{\"opened\": true, \"model\": \"%s\", \"rows\": %d, \"steps\": %d, \"width\": %u, \"vocabulary\": %d, "
         "\"checked\": %d, \"argmax_equals_device\": %d, \"padding_hits\": %d, \"failures\": %d, "
         "\"first_failure_step\": %d}\n",
         info.model_id, rows, steps, li.width, vocabulary, checked, host_equals_device, padding_hits, failures,
         first_failure_step);
  free(logits);
  free(prompt_tokens);
  loom_text_embed_close(e);
  return failures ? 1 : 0;
}

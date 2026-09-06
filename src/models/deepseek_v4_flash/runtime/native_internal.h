#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_NATIVE_INTERNAL_H_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_NATIVE_INTERNAL_H_

#include <cstddef>
#include <cstdint>

#include "model.h"

struct ds4_dspark_model;
struct ds4_model;
struct ds4_rocm_graph;
struct ds4_vocab;
struct ds4_weights;

struct ds4_string_view {
    const char *ptr;
    uint64_t len;
};

struct ds4_string_iterator {
    const ds4_model *model;
    uint64_t pos;
    uint64_t remaining;
};

struct ds4_rocm_batch_item {
    ds4_rocm_graph *graph;
    int token;
    uint32_t position;
    float *logits;
};

struct ds4_engine {
    ds4_model *model;
    ds4_vocab *vocab;
    ds4_weights *weights;
    ds4_dspark_model *dspark;
    ds4_rocm_graph *dspark_batch_workspace;
    int power_percent;
    uint32_t prefill_chunk;
    bool rocm_ready;
};

bool ds4_model_string_array_begin(const ds4_model *model,
                                  const char *key,
                                  ds4_string_iterator *iterator);
bool ds4_model_string_array_next(ds4_string_iterator *iterator,
                                 ds4_string_view *value);

ds4_vocab *ds4_vocab_create(const ds4_model *model);
void ds4_vocab_destroy(ds4_vocab *vocab);
void ds4_vocab_tokenize(const ds4_vocab *vocab,
                        const char *text,
                        ds4_tokens *tokens);
void ds4_vocab_tokenize_rendered_chat(const ds4_vocab *vocab,
                                      const char *text,
                                      ds4_tokens *tokens);
void ds4_vocab_encode_chat(const ds4_vocab *vocab,
                           const char *system,
                           const char *prompt,
                           ds4_tokens *tokens);
void ds4_vocab_chat_begin(const ds4_vocab *vocab, ds4_tokens *tokens);
void ds4_vocab_chat_append_message(const ds4_vocab *vocab,
                                   ds4_tokens *tokens,
                                   const char *role,
                                   const char *content);
void ds4_vocab_chat_append_assistant_prefix(const ds4_vocab *vocab,
                                            ds4_tokens *tokens);
char *ds4_vocab_token_text(const ds4_vocab *vocab,
                           int token,
                           size_t *length);
int ds4_vocab_size(const ds4_vocab *vocab);
int ds4_vocab_eos(const ds4_vocab *vocab);
bool ds4_vocab_is_stop(const ds4_vocab *vocab, int token);

void ds4_tokens_push(ds4_tokens *tokens, int token);
void ds4_tokens_copy(ds4_tokens *destination, const ds4_tokens *source);
bool ds4_tokens_starts_with(const ds4_tokens *tokens,
                            const ds4_tokens *prefix);

int ds4_sample_argmax(const float *logits, uint32_t vocabulary_size);
int ds4_sample_top_p_min_p(const float *logits,
                           uint32_t vocabulary_size,
                           float temperature,
                           int top_k,
                           float top_p,
                           float min_p,
                           uint64_t *rng_state);

uint32_t ds4_rocm_graph_prefill_capacity(int context_size);
uint32_t ds4_rocm_graph_resume_prefill_min_tokens(void);
ds4_rocm_graph *ds4_rocm_graph_create(ds4_engine *engine,
                                      int context_size,
                                      uint32_t prefill_capacity);
void ds4_rocm_graph_destroy(ds4_rocm_graph *graph);
bool ds4_rocm_graph_reset(ds4_rocm_graph *graph);
bool ds4_rocm_graph_prefill(ds4_rocm_graph *graph,
                            ds4_engine *engine,
                            const ds4_tokens *prompt,
                            float *logits);
bool ds4_rocm_graph_prefill_range(ds4_rocm_graph *graph,
                                  ds4_engine *engine,
                                  const ds4_tokens *prompt,
                                  uint32_t start,
                                  uint32_t token_count,
                                  float *logits);
bool ds4_rocm_graph_eval(ds4_rocm_graph *graph,
                         ds4_engine *engine,
                         int token,
                         uint32_t position,
                         float *logits);
bool ds4_rocm_graph_eval_batch(ds4_engine *engine,
                               const ds4_rocm_batch_item *items,
                               size_t item_count);
/* DSpark speculative verification.
 *
 * ds4_rocm_graph_verify_suffix scores `n_tokens` candidates in one batched pass
 * and returns the target's greedy continuation after each candidate row. The
 * caller decides how many rows to commit and must restore the saved frontier
 * whenever it commits fewer rows than it verified. */
bool ds4_rocm_graph_spec_prepare(ds4_rocm_graph *graph,
                                 ds4_engine *engine,
                                 uint32_t rows_cap);
uint32_t ds4_rocm_graph_spec_rows_cap(const ds4_rocm_graph *graph);
bool ds4_rocm_graph_spec_frontier_save(ds4_rocm_graph *graph);
bool ds4_rocm_graph_spec_frontier_restore(ds4_rocm_graph *graph);
bool ds4_rocm_graph_spec_frontier_commit_prefix(ds4_rocm_graph *graph,
                                                uint32_t prefix_len);
bool ds4_rocm_graph_verify_suffix(ds4_rocm_graph *graph,
                                  ds4_engine *engine,
                                  const ds4_tokens *tokens,
                                  uint32_t start,
                                  uint32_t n_tokens,
                                  int32_t *row_tops);
bool ds4_rocm_graph_read_spec_logits_row(const ds4_rocm_graph *graph,
                                         uint32_t row,
                                         float *logits);

/* DSpark drafting.
 *
 * Attaching a support model allocates the drafter's rings and enables target
 * feature capture. `_inject` turns captured features into ring rows for the given
 * absolute positions; `_draft` proposes one block for [pos0, pos0 + block_size).
 */
struct ds4_dspark_model;
bool ds4_rocm_graph_dspark_attach(ds4_rocm_graph *graph,
                                  ds4_engine *engine,
                                  const ds4_dspark_model *dspark);
uint32_t ds4_rocm_graph_dspark_block_size(const ds4_rocm_graph *graph);
void ds4_rocm_graph_dspark_set_capture_enabled(ds4_rocm_graph *graph,
                                               bool enabled);
bool ds4_rocm_graph_dspark_capture_ready(const ds4_rocm_graph *graph);
void ds4_rocm_graph_dspark_capture_reset(ds4_rocm_graph *graph);
uint32_t ds4_rocm_graph_dspark_batch_capture_rows(const ds4_rocm_graph *graph,
                                                 uint32_t *start);
bool ds4_rocm_graph_dspark_inject(ds4_rocm_graph *graph,
                                  uint32_t pos0,
                                  uint32_t n_rows);
uint32_t ds4_rocm_graph_dspark_context_len(const ds4_rocm_graph *graph);
void ds4_rocm_graph_dspark_truncate_context(ds4_rocm_graph *graph, uint32_t length);
/* Replaces row 0 with the target's known next token and re-traces the Markov
 * chain over the remaining rows from it. Must run before the verification pass,
 * which overwrites the draft's base logits. */
bool ds4_rocm_graph_dspark_reselect_tail(ds4_rocm_graph *graph,
                                         int known_first_token,
                                         uint32_t n_rows,
                                         int32_t *tokens_out);
bool ds4_rocm_graph_dspark_draft(ds4_rocm_graph *graph,
                                 ds4_engine *engine,
                                 int last_token,
                                 uint32_t pos0,
                                 int32_t *tokens_out,
                                 uint32_t *n_out);

uint64_t ds4_rocm_graph_snapshot_bytes(const ds4_rocm_graph *graph,
                                       const ds4_tokens *checkpoint);
int ds4_rocm_graph_save_snapshot(const ds4_rocm_graph *graph,
                                 const ds4_tokens *checkpoint,
                                 const float *logits,
                                 uint32_t prefill_capacity,
                                 int context_size,
                                 ds4_session_snapshot *snapshot,
                                 char *error,
                                 size_t error_capacity);
int ds4_rocm_graph_load_snapshot(ds4_rocm_graph *graph,
                                 ds4_tokens *checkpoint,
                                 float *logits,
                                 uint32_t prefill_capacity,
                                 int context_size,
                                 const ds4_session_snapshot *snapshot,
                                 char *error,
                                 size_t error_capacity);

#endif  // GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_NATIVE_INTERNAL_H_

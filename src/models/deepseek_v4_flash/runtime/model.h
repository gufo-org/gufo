#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_MODEL_H_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_MODEL_H_

#include <cstddef>
#include <cstdint>

struct ds4_tokens {
  int* v;
  int len;
  int cap;
};

struct ds4_engine;
struct ds4_session;

using ds4_session_cancel_fn = bool (*)(void* user_data);

struct ds4_engine_options {
  const char* model_path;
  /* Optional DSpark support model. NULL leaves speculative decoding off and
   * the engine byte-for-byte identical to a non-speculative build. */
  const char* dspark_model_path;
};

struct ds4_session_snapshot {
  uint8_t* ptr;
  uint64_t len;
  uint64_t cap;
};

struct ds4_session_batch_item {
  ds4_session* session;
  int token;
};

struct ds4_session_dspark_batch_item {
  ds4_session* session;
  int* emitted;
  int emitted_cap;
  uint32_t max_draft_tokens;
  int* n_emitted;
};

inline constexpr int DS4_SESSION_SYNC_INTERRUPTED = 2;
inline constexpr std::uint32_t DS4_SESSION_PAYLOAD_VERSION = 4;

int ds4_engine_open(ds4_engine** out, const ds4_engine_options* options);
void ds4_engine_close(ds4_engine* engine);
int ds4_engine_vocab_size(const ds4_engine* engine);
const char* ds4_engine_model_name(const ds4_engine* engine);

void ds4_tokens_free(ds4_tokens* tokens);
void ds4_tokenize_text(ds4_engine* engine, const char* text, ds4_tokens* out);
void ds4_tokenize_rendered_chat(ds4_engine* engine, const char* text,
                                ds4_tokens* out);
void ds4_encode_chat_prompt(ds4_engine* engine, const char* system,
                            const char* prompt, ds4_tokens* out);
void ds4_chat_begin(ds4_engine* engine, ds4_tokens* tokens);
void ds4_chat_append_message(ds4_engine* engine, ds4_tokens* tokens,
                             const char* role, const char* content);
void ds4_chat_append_assistant_prefix(ds4_engine* engine, ds4_tokens* tokens);
char* ds4_token_text(ds4_engine* engine, int token, size_t* length);
int ds4_token_eos(const ds4_engine* engine);
bool ds4_token_is_stop(const ds4_engine* engine, int token);

int ds4_session_create(ds4_session** out, ds4_engine* engine, int context_size);
void ds4_session_free(ds4_session* session);
int ds4_session_sync(ds4_session* session, const ds4_tokens* prompt,
                     char* error, size_t error_capacity);
int ds4_session_argmax(const ds4_session* session);
int ds4_session_sample(const ds4_session* session, float temperature, int top_k,
                       float top_p, float min_p, uint64_t* rng_state);
int ds4_session_eval(ds4_session* session, int token, char* error,
                     size_t error_capacity);
int ds4_sessions_eval_batch(const ds4_session_batch_item* items,
                            size_t item_count, char* error,
                            size_t error_capacity);
void ds4_session_begin_request(ds4_session* session);
void ds4_session_prepare_batch_execution(ds4_session* session);
bool ds4_engine_has_dspark(const ds4_engine* engine);
int ds4_sessions_dspark_step_batch(const ds4_session_dspark_batch_item* items,
                                   size_t item_count, char* error,
                                   size_t error_capacity);
void ds4_session_dspark_stats(const ds4_session* session, uint64_t* drafted,
                              uint64_t* accepted, uint64_t* support_drafted,
                              uint64_t* support_accepted,
                              uint64_t* positional_accepted, uint64_t* anchors,
                              uint64_t* full_blocks, uint64_t* steps,
                              uint64_t* skipped, uint32_t* context_tokens);
int ds4_session_copy_logits(const ds4_session* session, float* out,
                            int capacity);
int ds4_session_save_snapshot(const ds4_session* session,
                              ds4_session_snapshot* snapshot, char* error,
                              size_t error_capacity);
int ds4_session_load_snapshot(ds4_session* session,
                              const ds4_session_snapshot* snapshot, char* error,
                              size_t error_capacity);
void ds4_session_snapshot_free(ds4_session_snapshot* snapshot);
void ds4_session_set_cancel(ds4_session* session,
                            ds4_session_cancel_fn callback, void* user_data);
void ds4_session_invalidate(ds4_session* session);
int ds4_session_pos(const ds4_session* session);
int ds4_session_ctx(const ds4_session* session);
uint32_t ds4_session_prefill_capacity(const ds4_session* session);
uint64_t ds4_session_payload_bytes(const ds4_session* session);

#endif  // GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_MODEL_H_

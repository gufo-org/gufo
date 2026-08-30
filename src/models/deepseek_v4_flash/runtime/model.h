#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_MODEL_H_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_MODEL_H_

#include <cstddef>
#include <cstdint>

struct ds4_tokens {
    int *v;
    int len;
    int cap;
};

struct ds4_engine;
struct ds4_session;

using ds4_session_cancel_fn = bool (*)(void *user_data);

struct ds4_engine_options {
    const char *model_path;
    int context_size;
    uint32_t prefill_chunk;
    int power_percent;
};

struct ds4_session_snapshot {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
};

inline constexpr int DS4_SESSION_SYNC_INTERRUPTED = 2;
inline constexpr std::uint32_t DS4_SESSION_PAYLOAD_VERSION = 2;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *options);
void ds4_engine_close(ds4_engine *engine);
int ds4_engine_vocab_size(const ds4_engine *engine);
uint32_t ds4_engine_prefill_chunk(const ds4_engine *engine);
const char *ds4_engine_model_name(const ds4_engine *engine);

void ds4_tokens_free(ds4_tokens *tokens);
void ds4_tokenize_text(ds4_engine *engine, const char *text, ds4_tokens *out);
void ds4_encode_chat_prompt(ds4_engine *engine,
                            const char *system,
                            const char *prompt,
                            ds4_tokens *out);
void ds4_chat_begin(ds4_engine *engine, ds4_tokens *tokens);
void ds4_chat_append_message(ds4_engine *engine,
                             ds4_tokens *tokens,
                             const char *role,
                             const char *content);
void ds4_chat_append_assistant_prefix(ds4_engine *engine,
                                      ds4_tokens *tokens);
char *ds4_token_text(ds4_engine *engine, int token, size_t *length);
int ds4_token_eos(const ds4_engine *engine);
bool ds4_token_is_stop(const ds4_engine *engine, int token);

int ds4_session_create(ds4_session **out,
                       ds4_engine *engine,
                       int context_size);
void ds4_session_free(ds4_session *session);
int ds4_session_sync(ds4_session *session,
                     const ds4_tokens *prompt,
                     char *error,
                     size_t error_capacity);
int ds4_session_argmax(const ds4_session *session);
int ds4_session_argmax_excluding(const ds4_session *session,
                                 int excluded_token);
int ds4_session_sample(const ds4_session *session,
                       float temperature,
                       int top_k,
                       float top_p,
                       float min_p,
                       uint64_t *rng_state);
int ds4_session_eval(ds4_session *session,
                     int token,
                     char *error,
                     size_t error_capacity);
int ds4_session_copy_logits(const ds4_session *session,
                            float *out,
                            int capacity);
int ds4_session_save_snapshot(const ds4_session *session,
                              ds4_session_snapshot *snapshot,
                              char *error,
                              size_t error_capacity);
int ds4_session_load_snapshot(ds4_session *session,
                              const ds4_session_snapshot *snapshot,
                              char *error,
                              size_t error_capacity);
void ds4_session_snapshot_free(ds4_session_snapshot *snapshot);
void ds4_session_set_cancel(ds4_session *session,
                            ds4_session_cancel_fn callback,
                            void *user_data);
void ds4_session_invalidate(ds4_session *session);
int ds4_session_pos(const ds4_session *session);
int ds4_session_ctx(const ds4_session *session);
uint64_t ds4_session_payload_bytes(const ds4_session *session);

#endif  // GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_MODEL_H_

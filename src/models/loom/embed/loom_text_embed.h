// Loom text models embedded in another serving process.
//
// Loads a Loom model package (model.json, its programs and parameters) the
// way loom-serve registers it, and exposes the package's text model layer one
// step at a time: chat rendering and tokenization from the package's own
// programs, and resident rows advanced by prefill, decode and packed epochs on
// the package's kernels. The caller owns scheduling, admission and caching;
// no Loom service, session cache or HTTP is involved.
//
// Calls on one handle must be serialized by the caller. Functions return 0 on
// success; on failure they return nonzero and loom_text_embed_error() names
// the cause.

#ifndef LOOM_TEXT_EMBED_H_
#define LOOM_TEXT_EMBED_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_TEXT_EMBED_ABI_VERSION 2
// The Loom SDK release (hrx-loom, ROCm/hrx-system) this library is built from:
// packages must be built for it (their llm_config.json "sdk": {"hrx_loom":
// ...}). ../runtime.cmake (which checks this) and
// .devops/nix/hrx/loom-runtime.nix pin the same release.
#define LOOM_TEXT_EMBED_HRX_LOOM "0.2.0"

typedef struct loom_text_embed_t loom_text_embed_t;
typedef struct loom_text_embed_request_t loom_text_embed_request_t;

typedef struct loom_text_embed_options_t {
  // model.json of a fetched Loom package, or its directory.
  const char* model_path;
  // Named preset; NULL or "" selects the card's unambiguous preset.
  const char* preset;
  // Loom device URI; NULL selects amdgpu with an active completion wait.
  const char* device_uri;
  // Shared parameter/state budget in bytes (elastic backing); zero imposes no
  // limit.
  uint64_t memory_bytes;
  // Resident rows (concurrent sessions); zero keeps the preset's count.
  uint32_t rows;
  // Retained row checkpoints (shared device state); zero selects 64.
  uint32_t checkpoints;
} loom_text_embed_options_t;

typedef struct loom_text_embed_info_t {
  // The card's model identity, NUL-terminated.
  char model_id[256];
  uint32_t context_capacity;
  // Largest token count of one prefill call's packed shape.
  uint32_t prefill_capacity;
  uint32_t rows;
  // Proposed tokens per drafted step; zero when the package has no drafter.
  uint32_t draft_depth;
  // Widest packed decode epoch (rows advanced together).
  uint32_t max_epoch_spans;
  // Most images one request may carry; zero when the package has no vision.
  uint32_t max_images;
} loom_text_embed_info_t;

int loom_text_embed_open(const loom_text_embed_options_t* options,
                         loom_text_embed_t** out_embed, char* error,
                         size_t error_size);
void loom_text_embed_close(loom_text_embed_t* embed);
const char* loom_text_embed_error(const loom_text_embed_t* embed);
void loom_text_embed_info(const loom_text_embed_t* embed,
                          loom_text_embed_info_t* out_info);

// Renders an OpenAI chat request body with the package's chat program and
// returns its prompt tokens (tokens may be NULL to count only), plus the
// request's source-validated options (sampling, thinking) for row binding.
int loom_text_embed_chat_prepare(loom_text_embed_t* embed, const char* body,
                                 size_t body_size, int32_t* tokens,
                                 size_t capacity, size_t* out_count,
                                 loom_text_embed_request_t** out_request);
void loom_text_embed_request_free(loom_text_embed_request_t* request);

// Images of a prepared request (image_url parts in the body). media returns
// the input ranges [offset, offset + count) that each image's projected
// features replace, in order, and their number. project runs the package's
// image component on them; prefill_request then consumes request input
// [offset, offset + count) on a row, features included, selecting the next
// token when select is nonzero.
size_t loom_text_embed_request_media(const loom_text_embed_request_t* request,
                                     uint32_t* offsets, uint32_t* counts,
                                     size_t capacity);
int loom_text_embed_request_project(loom_text_embed_request_t* request);
int loom_text_embed_row_prefill_request(loom_text_embed_t* embed, uint32_t row,
                                        loom_text_embed_request_t* request,
                                        size_t offset, size_t count,
                                        int select);

// Output of one prepared chat request, read with the package's own chat
// program, as loom-serve reads it. feed appends generated text and returns
// what it settles, normalized: reasoning inside <think>...</think>, answer
// text as is, framing and tool syntax withheld. final releases the remainder
// and closes reasoning. tools then parses the withheld tool calls into a JSON
// array of {"id","type":"function","function":{"name","arguments"}}
// (truncated: the output hit its token limit; an unfinished call is dropped).
int loom_text_embed_output_feed(loom_text_embed_request_t* request,
                                const char* text, size_t text_size, int final,
                                char* out, size_t capacity, size_t* out_size);
// The raw framing the output reader consumed between the response's
// reasoning and its first answer text (the model's own reasoning close), once
// the response has reached its answer; empty before that or without
// reasoning.
int loom_text_embed_output_close(loom_text_embed_request_t* request, char* out,
                                 size_t capacity, size_t* out_size);
int loom_text_embed_output_tools(loom_text_embed_request_t* request,
                                 int truncated, char* json, size_t capacity,
                                 size_t* out_size);

// Tokenizes raw text; special-token spellings become special tokens when
// match_special is nonzero. Tokens may be NULL to count only.
int loom_text_embed_tokenize(loom_text_embed_t* embed, const char* text,
                             size_t text_size, int match_special,
                             int32_t* tokens, size_t capacity,
                             size_t* out_count);
// Decodes tokens to text, special tokens included. text may be NULL to size.
int loom_text_embed_detokenize(loom_text_embed_t* embed, const int32_t* tokens,
                               size_t count, char* text, size_t capacity,
                               size_t* out_size);
int32_t loom_text_embed_vocabulary_size(loom_text_embed_t* embed);

// Streaming detokenization: each feed returns exactly the text its tokens
// settle (a character split across tokens appears once it completes), so the
// concatenated output equals decoding all tokens at once. Special tokens are
// kept.
typedef struct loom_text_embed_decoder_t loom_text_embed_decoder_t;
int loom_text_embed_decoder_open(loom_text_embed_t* embed,
                                 loom_text_embed_decoder_t** out_decoder);
int loom_text_embed_decoder_feed(loom_text_embed_decoder_t* decoder,
                                 const int32_t* tokens, size_t count,
                                 char* text, size_t capacity, size_t* out_size);
int loom_text_embed_decoder_reset(loom_text_embed_decoder_t* decoder);
void loom_text_embed_decoder_free(loom_text_embed_decoder_t* decoder);

// Starts a request on a row: reserves state through `extent` consumed tokens
// (fresh replaces the row's history, otherwise it continues it) and binds the
// request's options. Returns 1 (and no error) when capacity is unavailable.
int loom_text_embed_row_begin(loom_text_embed_t* embed, uint32_t row,
                              const loom_text_embed_request_t* request,
                              uint64_t extent, int fresh);
// Grows the row's reservation, keeping its history. 1 means no capacity.
int loom_text_embed_row_extend(loom_text_embed_t* embed, uint32_t row,
                               uint64_t extent);
uint64_t loom_text_embed_row_extent(loom_text_embed_t* embed, uint32_t row);
// Rebinds request options (sampling) to a row between steps.
int loom_text_embed_row_bind(loom_text_embed_t* embed, uint32_t row,
                             const loom_text_embed_request_t* request);
void loom_text_embed_row_end(loom_text_embed_t* embed, uint32_t row);
int loom_text_embed_row_reset(loom_text_embed_t* embed, uint32_t row);

// Consumes known tokens (in prefill-capacity chunks) and selects the next.
int loom_text_embed_row_prefill(loom_text_embed_t* embed, uint32_t row,
                                const int32_t* tokens, size_t count);
// Consumes the pending selection and selects its successor.
int loom_text_embed_row_decode(loom_text_embed_t* embed, uint32_t row);
int32_t loom_text_embed_row_token(loom_text_embed_t* embed, uint32_t row);
int loom_text_embed_row_is_eos(loom_text_embed_t* embed, uint32_t row);
uint64_t loom_text_embed_row_position(loom_text_embed_t* embed, uint32_t row);

// One packed epoch: every row consumes its pending selection and selects its
// successor. Rows must be distinct and have pending, non-EOS selections.
int loom_text_embed_decode_batch(loom_text_embed_t* embed, const uint32_t* rows,
                                 size_t count);

// One drafted epoch (draft_depth > 0): each row proposes draft_depth tokens on
// device after its pending selection and verifies them greedily, accepting at
// most limits[i] new selections. tokens receives 16 IDs per row, of which
// counts[i] are valid: the accepted continuation, ending in the row's new
// pending selection (consumed[i] tokens entered state).
int loom_text_embed_verify_batch(loom_text_embed_t* embed, const uint32_t* rows,
                                 size_t count, const uint32_t* limits,
                                 int32_t* tokens, uint32_t* counts,
                                 uint32_t* consumed);

// Row checkpoints: the exact consumed state of a row, sharing its device
// pages (copy-on-write) until released. Restoring replaces a row's state.
// pin returns 1 when checkpoint capacity is exhausted; restore sets
// *restored to 0 when pool capacity is short.
typedef struct loom_text_embed_checkpoint_t loom_text_embed_checkpoint_t;
int loom_text_embed_row_pin(loom_text_embed_t* embed, uint32_t row,
                            loom_text_embed_checkpoint_t** out_checkpoint);
int loom_text_embed_row_restore(loom_text_embed_t* embed, uint32_t row,
                                const loom_text_embed_checkpoint_t* checkpoint,
                                int* restored);
void loom_text_embed_checkpoint_release(
    loom_text_embed_t* embed, loom_text_embed_checkpoint_t* checkpoint);

// Checkpoint images: a checkpoint's retained state as portable host bytes
// (no device addresses or page IDs), for caches that persist state. An image
// is fixed_bytes of fixed state (recurrence, carry), then one page per started
// block of block_size tokens, in token order, each page_bytes long; a page
// depends only on the tokens before its end. supported is zero when the
// preset's state is not elastic (images then cannot be restored).
typedef struct loom_text_embed_image_layout_t {
  uint32_t block_size;
  uint64_t fixed_bytes;
  uint64_t page_bytes;
  int supported;
} loom_text_embed_image_layout_t;
void loom_text_embed_image_layout(loom_text_embed_t* embed,
                                  loom_text_embed_image_layout_t* out_layout);

typedef struct loom_text_embed_checkpoint_info_t {
  uint32_t position;
  uint32_t source_position;
  int32_t token;
  int has_prediction;
} loom_text_embed_checkpoint_info_t;

// Writes a resident checkpoint's image (*out_size is set even when capacity
// is short, which fails).
int loom_text_embed_checkpoint_export(
    loom_text_embed_t* embed, const loom_text_embed_checkpoint_t* checkpoint,
    uint8_t* image, size_t capacity, size_t* out_size,
    loom_text_embed_checkpoint_info_t* out_info);
// Creates a resident checkpoint from an image (fresh pages, committed backing,
// contents uploaded). Returns 1 when checkpoint or page capacity is short.
int loom_text_embed_checkpoint_import(
    loom_text_embed_t* embed, const loom_text_embed_checkpoint_info_t* info,
    const uint8_t* image, size_t size,
    loom_text_embed_checkpoint_t** out_checkpoint);

//===----------------------------------------------------------------------===//
// ABI 2: logits readback (opt-in; ABI 1 callers are unchanged).
//===----------------------------------------------------------------------===//
//
// A package that declares the "llm.logits" capability can return the full
// logit row of any selecting span, so the caller can choose the token itself
// (grammar masks, logprobs, min-p, penalties). The capability is chosen at
// load time (loom_text_embed_open_ex, logits = 1); loading without it, or any
// load of a package that does not declare it, runs exactly as before.
//
// Per request: put "logits": true in the body given to chat_prepare (the
// package's request options read it). row_begin / row_bind with that request
// make the row a logit row: every epoch that selects for it also leaves its
// full logit row, which loom_text_embed_row_logits returns for the row's
// current pending selection. A logit row runs undrafted: use decode_batch /
// decode_batch_tokens for it, never verify_batch.

typedef struct loom_text_embed_options_ex_t {
  // sizeof(loom_text_embed_options_ex_t), for forward compatibility.
  uint32_t struct_size;
  // The ABI 1 options.
  loom_text_embed_options_t base;
  // Nonzero opens the package with the logits capability on. Fails when the
  // package does not declare "llm.logits" or its state cannot export rows.
  uint32_t logits;
  // Nonzero backs state with Loom's elastic pool even without memory_bytes
  // (checkpoint images need it). Zero: a fixed pool, committed at load, unless
  // memory_bytes sets a budget (which needs elastic backing).
  uint32_t elastic;
  // Nonzero sizes the shared KV pool for every row at this context (at most
  // the package's), so each row can reach it at once. Zero keeps memory_bytes'
  // share of an elastic pool, else one native context shared by all rows.
  uint64_t row_context;
} loom_text_embed_options_ex_t;

int loom_text_embed_open_ex(const loom_text_embed_options_ex_t* options,
                            loom_text_embed_t** out_embed, char* error,
                            size_t error_size);

typedef struct loom_text_embed_logits_info_t {
  // The package declares "llm.logits" and its state can return rows.
  uint32_t supported;
  // This handle was opened with logits on.
  uint32_t enabled;
  // Floats per row: token ids 0..width-1; ids the head does not have hold
  // a very negative value (<= -1e38).
  uint32_t width;
  // Most logit rows one epoch can return (its widest span count).
  uint32_t max_rows;
} loom_text_embed_logits_info_t;
void loom_text_embed_logits_info(const loom_text_embed_t* embed,
                                 loom_text_embed_logits_info_t* out_info);

// Whether a prepared request asked for logits ("logits": true).
int loom_text_embed_request_logits(const loom_text_embed_request_t* request);
// Whether the row is a logit row (its last begin/bind request asked).
int loom_text_embed_row_is_logit(loom_text_embed_t* embed, uint32_t row);

// Copies the full logit row of the row's pending selection: *out_count floats
// (at most capacity; capacity < width fails). Fails when the row is not a
// logit row or has no selection yet.
int loom_text_embed_row_logits(loom_text_embed_t* embed, uint32_t row,
                               float* out, size_t capacity, size_t* out_count);

// One packed epoch like decode_batch, but row i consumes tokens[i] instead of
// its device selection (the caller chose it from the logits). Rows must be
// distinct with a pending selection; they select their successors as usual.
int loom_text_embed_decode_batch_tokens(loom_text_embed_t* embed,
                                        const uint32_t* rows, size_t count,
                                        const int32_t* tokens);
// Whether a token ends the response (the package's stop token).
int loom_text_embed_is_stop_token(const loom_text_embed_t* embed,
                                  int32_t token);

// Device memory the open model holds. Its state (KV) is one paged pool that
// every row and checkpoint shares; reservations take pages from it as
// requests grow and give them back when rows reset.
typedef struct loom_text_embed_memory_t {
  // Device bytes the model's pool holds: weights, state and workspace.
  uint64_t device_bytes;
  // Token positions the state pool holds, page-rounded, and those in pages
  // nothing owns; zero for dense storage.
  uint64_t state_tokens;
  uint64_t free_tokens;
  // Bytes of one state page (every cache plane for block_size tokens).
  uint64_t page_bytes;
  uint32_t block_size;
} loom_text_embed_memory_t;
void loom_text_embed_memory(const loom_text_embed_t* embed,
                            loom_text_embed_memory_t* out_memory);

// The GPU a default device URI opens (the first AMD GPU endpoint), as the
// driver reports it on every platform, before any package is opened.
typedef struct loom_text_embed_gpu_t {
  // Compiler target, e.g. "gfx1151"; empty when there is no AMD GPU.
  char target[32];
  // Device-local memory in bytes (on an APU: system RAM on Windows, whose
  // local memory the driver reports as system-backed; the VRAM carve-out on
  // Linux); zero when it is not reported.
  uint64_t memory_bytes;
} loom_text_embed_gpu_t;
int loom_text_embed_query_gpu(loom_text_embed_gpu_t* out_gpu, char* error,
                              size_t error_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TEXT_EMBED_H_

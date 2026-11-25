#ifndef GUFO_CHAT_RESPONSE_H
#define GUFO_CHAT_RESPONSE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GufoChatStream GufoChatStream;
typedef struct GufoResponsesOutput GufoResponsesOutput;
typedef struct GufoAnthropicOutput GufoAnthropicOutput;

typedef enum GufoChatResponseStatus {
  GUFO_CHAT_RESPONSE_OK = 0,
  GUFO_CHAT_RESPONSE_CLOSED = 1,
  GUFO_CHAT_RESPONSE_INTERNAL = 2,
  /* The input is malformed; the reason went to the error writer. */
  GUFO_CHAT_RESPONSE_INVALID = 3,
} GufoChatResponseStatus;

typedef enum GufoChatFinishReason {
  GUFO_CHAT_FINISH_STOP = 0,
  GUFO_CHAT_FINISH_LENGTH = 1,
  GUFO_CHAT_FINISH_STOP_SEQUENCE = 2,
} GufoChatFinishReason;

typedef struct GufoChatBytes {
  const uint8_t* data;
  size_t len;
} GufoChatBytes;

typedef struct GufoChatToolCall {
  GufoChatBytes id;
  GufoChatBytes name;
  GufoChatBytes arguments;
} GufoChatToolCall;

typedef struct GufoChatMetrics {
  size_t prompt_tokens;
  size_t completion_tokens;
  size_t cached_prompt_tokens;
  size_t reasoning_tokens;
  size_t draft_tokens;
  size_t draft_accepted_tokens;
  size_t draft_rounds;
  double prompt_tokens_per_second;
  double decode_ms;
  bool cache_hit;
  GufoChatBytes cache_miss_reason;
  size_t cache_common_prefix_tokens;
  size_t cache_checkpoint_tokens;
  size_t cache_restore_bytes;
  size_t cache_snapshot_bytes;
  size_t cache_disk_queued_bytes;
  double cache_restore_ms;
  double cache_snapshot_ms;
  double cache_disk_enqueue_ms;
  bool cache_disk_hit;
  size_t prefill_tokens;
  size_t prefill_chunks;
  size_t active_decode_prefill_chunks;
  size_t max_prefill_chunk_tokens;
  size_t max_prefill_layer_groups;
  size_t prefill_yield_rounds;
  size_t queue_depth_at_submit;
  size_t client_queue_depth_at_submit;
  size_t resident_requests_at_admission;
  size_t requested_logical_concurrency;
  size_t physical_execution_width;
  double queue_ms;
  double shared_prefix_wait_ms;
  double prefill_ms;
  double ttft_ms;
  double mean_inter_token_ms;
  double max_inter_token_ms;
  GufoChatBytes execution_plan;
  GufoChatBytes component_cache_json;
  GufoChatBytes timings_json;
} GufoChatMetrics;

typedef struct GufoChatNonStreamingInput {
  GufoChatBytes id;
  int64_t created;
  GufoChatBytes model;
  GufoChatBytes reasoning;
  GufoChatBytes text;
  const GufoChatToolCall* calls;
  size_t calls_len;
  GufoChatBytes logprobs_json;
  GufoChatFinishReason finish_reason;
  bool cache_persist;
  const GufoChatMetrics* metrics;
} GufoChatNonStreamingInput;

typedef struct GufoChatNamespaceEntry {
  GufoChatBytes name;
  GufoChatBytes ns;
} GufoChatNamespaceEntry;

typedef struct GufoResponsesConfig {
  GufoChatBytes id;
  int64_t created_at;
  GufoChatBytes model;
  bool parallel_tool_calls;
  GufoChatBytes tool_choice;
  GufoChatBytes forced_tool_name;
  const GufoChatBytes* tool_definitions_json;
  size_t tool_definitions_len;
  const GufoChatNamespaceEntry* namespaces;
  size_t namespaces_len;
  bool cache_persist;
  bool has_top_logprobs;
  uint32_t top_logprobs;
} GufoResponsesConfig;

typedef struct GufoChatToolDelta {
  size_t index;
  bool begin;
  GufoChatBytes id;
  GufoChatBytes name;
  GufoChatBytes arguments;
  bool complete;
} GufoChatToolDelta;

typedef bool (*GufoChatWriteFn)(void* context, const uint8_t* data, size_t len);

/* One-shot JSON response encoders */

GufoChatResponseStatus gufo_chat_error_response(GufoChatBytes message,
                                                GufoChatBytes error_type,
                                                GufoChatBytes code,
                                                GufoChatWriteFn write_fn,
                                                void* write_ctx);

GufoChatResponseStatus gufo_chat_text_generation_error_response(
    GufoChatBytes message, GufoChatBytes code, bool has_required_bytes,
    size_t required_bytes, bool has_remaining_bytes, size_t remaining_bytes,
    GufoChatWriteFn write_fn, void* write_ctx);

GufoChatResponseStatus gufo_chat_input_tokens_response(size_t input_tokens,
                                                       GufoChatWriteFn write_fn,
                                                       void* write_ctx);

GufoChatResponseStatus gufo_chat_rendered_prompt_response(
    GufoChatBytes prompt, GufoChatWriteFn write_fn, void* write_ctx);

GufoChatResponseStatus gufo_chat_non_streaming_response(
    const GufoChatNonStreamingInput* input, GufoChatWriteFn write_fn,
    void* write_ctx);

/* OpenAI Chat Completions SSE stream */

GufoChatStream* gufo_chat_stream_new(GufoChatBytes id, int64_t created,
                                     GufoChatBytes model);
void gufo_chat_stream_free(GufoChatStream* stream);

GufoChatResponseStatus gufo_chat_stream_role(GufoChatStream* stream,
                                             GufoChatWriteFn write_fn,
                                             void* write_ctx);

GufoChatResponseStatus gufo_chat_stream_text_delta(
    GufoChatStream* stream, bool reasoning, GufoChatBytes text,
    GufoChatBytes logprobs_json, GufoChatWriteFn write_fn, void* write_ctx);

GufoChatResponseStatus gufo_chat_stream_empty_delta(GufoChatStream* stream,
                                                    GufoChatBytes logprobs_json,
                                                    GufoChatWriteFn write_fn,
                                                    void* write_ctx);

GufoChatResponseStatus gufo_chat_stream_tool_delta(
    GufoChatStream* stream, const GufoChatToolDelta* delta,
    GufoChatBytes logprobs_json, GufoChatWriteFn write_fn, void* write_ctx);

GufoChatResponseStatus gufo_chat_stream_progress(
    GufoChatStream* stream, GufoChatBytes prompt_progress_json,
    GufoChatWriteFn write_fn, void* write_ctx);

GufoChatResponseStatus gufo_chat_stream_finish(
    GufoChatStream* stream, GufoChatFinishReason finish_reason,
    bool has_tool_calls, bool include_usage, bool cache_persist,
    const GufoChatMetrics* metrics, GufoChatWriteFn write_fn, void* write_ctx);

GufoChatResponseStatus gufo_chat_stream_error(GufoChatBytes message,
                                              GufoChatBytes code,
                                              GufoChatWriteFn write_fn,
                                              void* write_ctx);

/* OpenAI Responses API (/v1/responses) */

GufoResponsesOutput* gufo_responses_output_new(
    const GufoResponsesConfig* config, bool streaming);
void gufo_responses_output_free(GufoResponsesOutput* output);

GufoChatResponseStatus gufo_responses_output_begin(GufoResponsesOutput* output,
                                                   GufoChatWriteFn write_fn,
                                                   void* write_ctx);

GufoChatResponseStatus gufo_responses_output_progress(
    GufoResponsesOutput* output, GufoChatBytes prompt_progress_json,
    GufoChatWriteFn write_fn, void* write_ctx);

GufoChatResponseStatus gufo_responses_output_append(
    GufoResponsesOutput* output, GufoChatBytes text, bool reasoning,
    GufoChatBytes item_id, GufoChatBytes logprobs_json,
    GufoChatBytes stream_logprobs_json, GufoChatWriteFn write_fn,
    void* write_ctx);

GufoChatResponseStatus gufo_responses_output_tool(
    GufoResponsesOutput* output, const GufoChatToolDelta* delta,
    GufoChatBytes fc_id, GufoChatWriteFn write_fn, void* write_ctx);

GufoChatResponseStatus gufo_responses_output_complete(
    GufoResponsesOutput* output, GufoChatFinishReason finish_reason,
    const GufoChatMetrics* metrics, GufoChatWriteFn write_fn, void* write_ctx,
    GufoChatWriteFn body_fn, void* body_ctx);

GufoChatResponseStatus gufo_responses_output_fail(GufoResponsesOutput* output,
                                                  GufoChatBytes message,
                                                  GufoChatWriteFn write_fn,
                                                  void* write_ctx);

/* Anthropic Messages API (/v1/messages) */

GufoAnthropicOutput* gufo_anthropic_output_new(GufoChatBytes id,
                                               GufoChatBytes model,
                                               bool streaming);
void gufo_anthropic_output_free(GufoAnthropicOutput* output);

GufoChatResponseStatus gufo_anthropic_output_begin(GufoAnthropicOutput* output,
                                                   GufoChatWriteFn write_fn,
                                                   void* write_ctx);

GufoChatResponseStatus gufo_anthropic_output_append(GufoAnthropicOutput* output,
                                                    GufoChatBytes text,
                                                    bool reasoning,
                                                    GufoChatWriteFn write_fn,
                                                    void* write_ctx);

/* GUFO_CHAT_RESPONSE_INVALID when a complete call's arguments are not JSON;
 * the parse error goes to error_fn. */
GufoChatResponseStatus gufo_anthropic_output_tool(
    GufoAnthropicOutput* output, const GufoChatToolDelta* delta,
    GufoChatWriteFn write_fn, void* write_ctx, GufoChatWriteFn error_fn,
    void* error_ctx);

GufoChatResponseStatus gufo_anthropic_output_complete(
    GufoAnthropicOutput* output, GufoChatFinishReason finish_reason,
    GufoChatBytes stop_sequence, const GufoChatMetrics* metrics,
    GufoChatWriteFn write_fn, void* write_ctx, GufoChatWriteFn body_fn,
    void* body_ctx);

GufoChatResponseStatus gufo_anthropic_output_fail(GufoAnthropicOutput* output,
                                                  GufoChatBytes message,
                                                  GufoChatWriteFn write_fn,
                                                  void* write_ctx);

#ifdef __cplusplus
}
#endif

#endif  // GUFO_CHAT_RESPONSE_H

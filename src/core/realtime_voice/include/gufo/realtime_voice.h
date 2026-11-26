#ifndef GUFO_REALTIME_VOICE_H
#define GUFO_REALTIME_VOICE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GufoRealtimeVoice GufoRealtimeVoice;

typedef enum GufoRealtimeVoiceStatus {
  GUFO_REALTIME_VOICE_OK = 0,
  GUFO_REALTIME_VOICE_CLOSED = 1,
  GUFO_REALTIME_VOICE_INTERNAL = 2,
} GufoRealtimeVoiceStatus;

typedef enum GufoRealtimeVoiceDecision {
  GUFO_REALTIME_VOICE_QUEUE = 0,
  GUFO_REALTIME_VOICE_CANCEL = 1,
  GUFO_REALTIME_VOICE_VAD_AUDIO = 2,
  GUFO_REALTIME_VOICE_HANDLED = 3,
} GufoRealtimeVoiceDecision;

typedef struct GufoRealtimeVoiceBytes {
  const uint8_t* data;
  size_t len;
} GufoRealtimeVoiceBytes;

typedef struct GufoRealtimeVoiceSettings {
  GufoRealtimeVoiceBytes instructions;
  bool audio_output;
  const GufoRealtimeVoiceBytes* tools;
  size_t tools_len;
  size_t max_output_tokens;
  bool reasoning;
  GufoRealtimeVoiceBytes voice;
  GufoRealtimeVoiceBytes response_instructions;
  bool call_tools;
  bool out_of_band;
} GufoRealtimeVoiceSettings;

typedef struct GufoRealtimeVoiceVadSettings {
  double threshold;
  int32_t prefix_padding_ms;
  int32_t silence_duration_ms;
  bool create_response;
  bool interrupt_response;
} GufoRealtimeVoiceVadSettings;

typedef struct GufoRealtimeVoiceFunctionCall {
  GufoRealtimeVoiceBytes name;
  GufoRealtimeVoiceBytes arguments;
} GufoRealtimeVoiceFunctionCall;

typedef struct GufoRealtimeVoiceResponseOutcome {
  bool ok;
  bool cancelled;
  bool incomplete;
  int64_t input_tokens;
  int64_t output_tokens;
  const GufoRealtimeVoiceFunctionCall* calls;
  size_t calls_len;
  GufoRealtimeVoiceBytes failure;
} GufoRealtimeVoiceResponseOutcome;

typedef bool (*GufoRealtimeVoiceSendFn)(void* context, const uint8_t* data,
                                        size_t len);
typedef void (*GufoRealtimeVoiceBytesFn)(void* context, const uint8_t* data,
                                         size_t len);
typedef void (*GufoRealtimeVoiceErrorSetFn)(void* error_ctx,
                                            const uint8_t* data, size_t len);
typedef void (*GufoRealtimeVoiceActiveResponseFn)(
    void* context, const GufoRealtimeVoiceSettings* settings,
    const uint8_t* response_id, size_t response_id_len);

/* Host callbacks invoked by worker-thread entry points. Callbacks must not
 * re-enter the same GufoRealtimeVoice handle. */
typedef struct GufoRealtimeVoiceHost {
  void* context;
  GufoRealtimeVoiceSendFn send_event;
  bool (*configure)(void* context, const GufoRealtimeVoiceSettings* settings,
                    GufoRealtimeVoiceErrorSetFn set_error, void* error_ctx);
  void (*on_vad_settings)(void* context, bool has_vad,
                          const GufoRealtimeVoiceVadSettings* vad);
  /* `record`: keep the turn's audio for `transcribe_turn`. */
  bool (*append_audio)(void* context, const uint8_t* pcm, size_t len,
                       bool record, GufoRealtimeVoiceErrorSetFn set_error,
                       void* error_ctx);
  bool (*commit_audio)(void* context, bool* committed,
                       GufoRealtimeVoiceErrorSetFn set_error, void* error_ctx);
  bool (*clear_audio)(void* context, GufoRealtimeVoiceErrorSetFn set_error,
                      void* error_ctx);
  bool (*truncate_reply)(void* context, size_t samples,
                         GufoRealtimeVoiceErrorSetFn set_error,
                         void* error_ctx);
  bool (*add_user_text)(void* context, const uint8_t* text, size_t len,
                        GufoRealtimeVoiceErrorSetFn set_error, void* error_ctx);
  bool (*add_tool_outputs)(void* context, const GufoRealtimeVoiceBytes* outputs,
                           size_t count, GufoRealtimeVoiceErrorSetFn set_error,
                           void* error_ctx);
  bool (*serves_model)(void* context, const uint8_t* name, size_t len);
  /* Transcribes the recorded audio of the user turn committed last;
     `language` is empty when the model detects it. The host calls
     `on_transcript` once with the whole transcript. */
  bool (*transcribe_turn)(void* context, GufoRealtimeVoiceBytes language,
                          GufoRealtimeVoiceBytes prompt,
                          GufoRealtimeVoiceBytesFn on_transcript,
                          void* transcript_ctx,
                          GufoRealtimeVoiceErrorSetFn set_error,
                          void* error_ctx);
  bool (*delete_user_turn)(void* context, GufoRealtimeVoiceErrorSetFn set_error,
                           void* error_ctx);
  bool (*supports_response_options)(void* context);
} GufoRealtimeVoiceHost;

GufoRealtimeVoice* gufo_realtime_voice_new(const uint8_t* id, size_t id_len,
                                           const uint8_t* model,
                                           size_t model_len);
void gufo_realtime_voice_free(GufoRealtimeVoice* session);

GufoRealtimeVoiceStatus gufo_realtime_voice_session_unavailable(
    const uint8_t* message, size_t message_len,
    GufoRealtimeVoiceSendFn send_event, void* send_ctx);

GufoRealtimeVoiceStatus gufo_realtime_voice_session_created(
    const GufoRealtimeVoice* session, GufoRealtimeVoiceSendFn send_event,
    void* send_ctx);

GufoRealtimeVoiceStatus gufo_realtime_voice_error(
    const GufoRealtimeVoice* session, const uint8_t* message,
    size_t message_len, const uint8_t* client_event, size_t client_event_len,
    const uint8_t* code, size_t code_len, GufoRealtimeVoiceSendFn send_event,
    void* send_ctx);

GufoRealtimeVoiceStatus gufo_realtime_voice_classify_client_event(
    const GufoRealtimeVoice* session, const uint8_t* raw, size_t raw_len,
    bool vad_active, bool responding, GufoRealtimeVoiceDecision* decision,
    GufoRealtimeVoiceBytesFn on_pcm, void* pcm_ctx,
    GufoRealtimeVoiceSendFn send_event, void* send_ctx);

GufoRealtimeVoiceStatus gufo_realtime_voice_vad_speech_started(
    const GufoRealtimeVoice* session, int64_t audio_start_ms,
    GufoRealtimeVoiceBytesFn on_item_id, void* item_ctx,
    GufoRealtimeVoiceSendFn send_event, void* send_ctx);

GufoRealtimeVoiceStatus gufo_realtime_voice_vad_speech_stopped(
    const GufoRealtimeVoice* session, const uint8_t* item_id,
    size_t item_id_len, int64_t audio_end_ms,
    GufoRealtimeVoiceSendFn send_event, void* send_ctx);

GufoRealtimeVoiceStatus gufo_realtime_voice_handle_client_event(
    const GufoRealtimeVoice* session, const uint8_t* raw, size_t raw_len,
    bool audio_open, const GufoRealtimeVoiceHost* host, bool* start_response);

GufoRealtimeVoiceStatus gufo_realtime_voice_handle_vad_audio(
    const GufoRealtimeVoice* session, const uint8_t* pcm, size_t pcm_len,
    bool audio_open, const GufoRealtimeVoiceHost* host);

GufoRealtimeVoiceStatus gufo_realtime_voice_handle_vad_clear(
    const GufoRealtimeVoice* session, const GufoRealtimeVoiceHost* host);

GufoRealtimeVoiceStatus gufo_realtime_voice_handle_vad_commit(
    const GufoRealtimeVoice* session, const uint8_t* item_id,
    size_t item_id_len, bool respond, bool interrupt_active,
    const GufoRealtimeVoiceHost* host, bool* start_response);

GufoRealtimeVoiceStatus gufo_realtime_voice_response_start(
    const GufoRealtimeVoice* session,
    GufoRealtimeVoiceActiveResponseFn on_active, void* active_ctx,
    GufoRealtimeVoiceSendFn send_event, void* send_ctx);

GufoRealtimeVoiceStatus gufo_realtime_voice_response_reasoning(
    const GufoRealtimeVoice* session, const uint8_t* text, size_t text_len,
    GufoRealtimeVoiceSendFn send_event, void* send_ctx);

GufoRealtimeVoiceStatus gufo_realtime_voice_response_text(
    const GufoRealtimeVoice* session, const uint8_t* text, size_t text_len,
    GufoRealtimeVoiceSendFn send_event, void* send_ctx);

GufoRealtimeVoiceStatus gufo_realtime_voice_response_audio(
    const GufoRealtimeVoice* session, const uint8_t* pcm, size_t pcm_len,
    GufoRealtimeVoiceSendFn send_event, void* send_ctx);

GufoRealtimeVoiceStatus gufo_realtime_voice_response_finish(
    const GufoRealtimeVoice* session,
    const GufoRealtimeVoiceResponseOutcome* outcome,
    GufoRealtimeVoiceSendFn send_event, void* send_ctx);

#ifdef __cplusplus
}
#endif

#endif  // GUFO_REALTIME_VOICE_H

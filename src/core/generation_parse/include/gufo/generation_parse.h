#ifndef GUFO_GENERATION_PARSE_H_
#define GUFO_GENERATION_PARSE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The C interface of the Rust generation parser: it splits generated text
   into reasoning, content and tool calls, streamed piece by piece or whole.
   It knows no marker of its own; src/cli/serve/generation_parse.cpp passes
   every one in, and keeps the Qwen call scanner, the registered dialect
   scanners and the tool grammar behind the callbacks. */
typedef struct GufoGeneration GufoGeneration;

typedef struct GufoGenerationBytes {
  const uint8_t* data;
  size_t len;
} GufoGenerationBytes;

typedef struct GufoGenerationTool {
  GufoGenerationBytes name;
  GufoGenerationBytes parameters_json;
} GufoGenerationTool;

/* [begin, end) bytes of the text a dialect scanner read. */
typedef struct GufoGenerationRange {
  size_t begin;
  size_t end;
} GufoGenerationRange;

/* [begin, begin + size) bytes of the decoded response (TextSourceSpan). */
typedef struct GufoGenerationSpan {
  size_t begin;
  size_t size;
} GufoGenerationSpan;

/* tokenization::QwenToolCallParameter. */
typedef struct GufoQwenParameter {
  GufoGenerationBytes name;
  GufoGenerationBytes value;
  uint8_t framed;
  size_t progress;
} GufoQwenParameter;

/* tokenization::QwenToolCallScan; the call fields count when has_call is
   set, and unfinished when has_unfinished is. */
typedef struct GufoQwenScan {
  uint8_t matched;
  uint8_t has_call;
  uint8_t native;
  uint8_t canonical_header;
  uint8_t complete;
  uint8_t has_unfinished;
  size_t progress;
  size_t end;
  GufoGenerationBytes name;
  GufoGenerationBytes json_payload;
  const GufoQwenParameter* parameters;
  size_t parameter_count;
  GufoQwenParameter unfinished;
} GufoQwenScan;

/* ext::ScannedToolArgument. */
typedef struct GufoScannedArgument {
  GufoGenerationBytes name;
  GufoGenerationBytes value;
  uint8_t is_string;
} GufoScannedArgument;

/* ext::ScannedToolCall; unfinished counts when has_unfinished is set. */
typedef struct GufoScannedCall {
  GufoGenerationBytes name;
  const GufoScannedArgument* arguments;
  size_t argument_count;
  uint8_t has_unfinished;
  uint8_t valid;
  uint8_t complete;
  GufoScannedArgument unfinished;
} GufoScannedCall;

typedef struct GufoDialectScan {
  const GufoScannedCall* calls;
  size_t call_count;
  const GufoGenerationRange* spans;
  size_t span_count;
} GufoDialectScan;

/* Each callback returns 0, or nonzero to stop the parse with
   GUFO_GENERATION_STOPPED. What it stores in its output stays valid until
   the next call of the same callback. */

/* ScanQwenToolCall of text[0, len). */
typedef int (*GufoGenerationQwenScanFn)(void* context, const uint8_t* text,
                                        size_t len, GufoQwenScan* scan);
/* The selected dialect's scan_tool_calls of text[0, len) with markers. */
typedef int (*GufoGenerationDialectScanFn)(void* context, const uint8_t* text,
                                           size_t len,
                                           const GufoGenerationBytes* markers,
                                           size_t marker_count, int partial,
                                           GufoDialectScan* scan);
/* Sets *accepted when the non-strict grammar of tools[tool] accepts the
   arguments object arguments[0, len); clear when it does not, including when
   the schema does not compile. */
typedef int (*GufoGenerationGrammarFn)(void* context, size_t tool,
                                       const uint8_t* arguments, size_t len,
                                       int* accepted);

typedef enum GufoGenerationInitial {
  GUFO_GENERATION_INITIAL_AUTO = 0,
  GUFO_GENERATION_INITIAL_REASONING = 1,
  GUFO_GENERATION_INITIAL_CONTENT = 2,
  GUFO_GENERATION_INITIAL_CONTENT_AFTER_REASONING = 3,
} GufoGenerationInitial;

typedef enum GufoGenerationMode {
  /* Unconstrained output. */
  GUFO_GENERATION_MODE_PLAIN = 0,
  /* Output under a response format. */
  GUFO_GENERATION_MODE_STRUCTURED = 1,
  /* Output under the tool grammar, without a response format. */
  GUFO_GENERATION_MODE_TOOL_ONLY = 2,
} GufoGenerationMode;

typedef enum GufoGenerationToolChoice {
  GUFO_GENERATION_TOOL_CHOICE_AUTO = 0,
  GUFO_GENERATION_TOOL_CHOICE_NONE = 1,
  GUFO_GENERATION_TOOL_CHOICE_REQUIRED = 2,
} GufoGenerationToolChoice;

/* Config flags. */
/* The host scans calls opened by qwen_open itself. */
#define GUFO_GENERATION_QWEN_CALLS 1u
/* A native grammar frames arguments early, so partial calls stream. */
#define GUFO_GENERATION_NATIVE 2u
/* dialect_scan reads the other markers' calls. */
#define GUFO_GENERATION_DIALECT 4u
/* Final content reports its source spans. */
#define GUFO_GENERATION_TRACK_SPANS 8u
/* Fed pieces are only buffered: tool deltas come from the final parse. */
#define GUFO_GENERATION_BUFFERED 16u

/* Every pointer is read during gufo_generation_new or gufo_generation_parse
   only; the callbacks and context stay in use until the handle is freed. */
typedef struct GufoGenerationConfig {
  uint32_t initial;     /* GufoGenerationInitial */
  uint32_t mode;        /* GufoGenerationMode */
  uint32_t tool_choice; /* GufoGenerationToolChoice */
  uint32_t flags;
  GufoGenerationBytes reasoning_open;
  GufoGenerationBytes reasoning_close;
  /* Whitespace after an explicit end of reasoning that is framing. */
  GufoGenerationBytes reasoning_separator;
  GufoGenerationBytes qwen_open;
  const GufoGenerationBytes* markers;
  size_t marker_count;
  const GufoGenerationBytes* reasoning_markers;
  size_t reasoning_marker_count;
  const GufoGenerationBytes* closers;
  size_t closer_count;
  const GufoGenerationBytes* envelope_heads;
  size_t envelope_head_count;
  const GufoGenerationBytes* envelope_parameters;
  size_t envelope_parameter_count;
  const GufoGenerationBytes* envelope_closers;
  size_t envelope_closer_count;
  const GufoGenerationTool* tools;
  size_t tool_count;
  GufoGenerationQwenScanFn qwen_scan;
  GufoGenerationDialectScanFn dialect_scan;
  GufoGenerationGrammarFn grammar_accepts;
  void* context;
} GufoGenerationConfig;

typedef enum GufoGenerationStatus {
  GUFO_GENERATION_COMPLETE = 0,
  /* tool_choice required was not met; a GUFO_GENERATION_UNSATISFIED event
     said how. */
  GUFO_GENERATION_TOOL_CHOICE_UNSATISFIED = 1,
  /* A streamed tool call did not complete as the final parse read it. */
  GUFO_GENERATION_UNSAFE_STREAM = 2,
  /* The event or a host callback returned nonzero. */
  GUFO_GENERATION_STOPPED = 3,
  GUFO_GENERATION_NO_MEMORY = 4,
  /* A caller bug (bad config, null pointer) or a parser bug. */
  GUFO_GENERATION_INTERNAL = 5,
} GufoGenerationStatus;

typedef enum GufoGenerationEventKind {
  /* Streamed text, with its source spans when known. */
  GUFO_GENERATION_REASONING_DELTA = 0,
  GUFO_GENERATION_CONTENT_DELTA = 1,
  /* The new argument suffix of call index; begin opens the call and carries
     name, complete closes it. */
  GUFO_GENERATION_TOOL_DELTA = 2,
  /* The final parse, after every delta: reasoning, content and its spans,
     each call with its arguments object, then the end with
     hide_tool_markup. */
  GUFO_GENERATION_FINAL_REASONING = 3,
  GUFO_GENERATION_FINAL_CONTENT = 4,
  GUFO_GENERATION_FINAL_CALL = 5,
  GUFO_GENERATION_FINAL_END = 6,
  /* Before GUFO_GENERATION_TOOL_CHOICE_UNSATISFIED: complete is set when a
     constrained turn did not complete a call, clear when none was
     produced. Its return value is ignored. */
  GUFO_GENERATION_UNSATISFIED = 7,
} GufoGenerationEventKind;

typedef struct GufoGenerationEvent {
  GufoGenerationEventKind kind;
  size_t index;
  GufoGenerationBytes text;
  GufoGenerationBytes name;
  GufoGenerationBytes arguments;
  const GufoGenerationSpan* spans;
  size_t span_count;
  uint8_t begin;
  uint8_t complete;
  uint8_t hide_tool_markup;
} GufoGenerationEvent;

/* Receives an event, valid during the call; returns 0 to continue. */
typedef int (*GufoGenerationEventFn)(void* context,
                                     const GufoGenerationEvent* event);

/* Makes a streaming handle in *handle, or stores NULL and fails. */
GufoGenerationStatus gufo_generation_new(const GufoGenerationConfig* config,
                                         GufoGeneration** handle);

void gufo_generation_free(GufoGeneration* handle);

/* Reads generated bytes, which may split UTF-8 scalars, and reports the
   deltas they release. After any status but COMPLETE the handle takes no
   more input. */
GufoGenerationStatus gufo_generation_feed(GufoGeneration* handle,
                                          const uint8_t* bytes, size_t len,
                                          GufoGenerationEventFn event,
                                          void* context);

/* Ends the response: the final tool and text deltas, then the final events.
   enforce_required fails an unmet tool_choice required; interrupted (a stop
   before the model's own end) lets unfinished streamed calls lapse. */
GufoGenerationStatus gufo_generation_finish(GufoGeneration* handle,
                                            int enforce_required,
                                            int interrupted,
                                            GufoGenerationEventFn event,
                                            void* context);

/* Parses a whole response and reports the final events. */
GufoGenerationStatus gufo_generation_parse(const GufoGenerationConfig* config,
                                           const uint8_t* bytes, size_t len,
                                           int enforce_required,
                                           GufoGenerationEventFn event,
                                           void* context);

#ifdef __cplusplus
}
#endif

#endif /* GUFO_GENERATION_PARSE_H_ */

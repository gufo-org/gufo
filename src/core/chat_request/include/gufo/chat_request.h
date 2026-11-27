#ifndef GUFO_CHAT_REQUEST_H_
#define GUFO_CHAT_REQUEST_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The C interface of the Rust chat request decoder. src/cli/serve/
   chat_request.cpp builds ChatRequest from the events it emits and runs the
   C++ helpers the completions endpoints share when the decoder asks. */

typedef enum GufoChatDecode {
  GUFO_CHAT_ACCEPTED = 0,
  /* The client sent a request the API rejects: the reject callback got it, or
     a host_field callback returned GUFO_CHAT_HOST_REJECTED. */
  GUFO_CHAT_REJECTED = 1,
  GUFO_CHAT_NO_MEMORY = 2,
  /* A callback failed; the caller rethrows what it recorded. */
  GUFO_CHAT_STOPPED = 3,
  /* A caller bug (null pointer, unknown API) or a decoder bug. */
  GUFO_CHAT_INTERNAL = 4,
} GufoChatDecode;

typedef enum GufoChatApi {
  /* Chat Completions (ParseRequest). */
  GUFO_CHAT_API_CHAT = 0,
  /* Responses (ReadOpenAiResponseChat). */
  GUFO_CHAT_API_RESPONSES = 1,
  /* Messages (ReadAnthropicChat). */
  GUFO_CHAT_API_MESSAGES = 2,
  /* ParseOpenAiResponseControls on the request in GufoChatDefaults. */
  GUFO_CHAT_FRAGMENT_RESPONSE_CONTROLS = 3,
  /* ParseOpenAiResponseMessage: one Responses item into one message. */
  GUFO_CHAT_FRAGMENT_RESPONSE_ITEM = 4,
  /* ParseAnthropicToolControls on the request in GufoChatDefaults. */
  GUFO_CHAT_FRAGMENT_ANTHROPIC_TOOLS = 5,
} GufoChatApi;

/* Values of an optional field: unset, false or true. */
#define GUFO_CHAT_UNSET (-1)

/* tokenization::ChatRole; src/cli/serve/chat_request.cpp checks each. */
typedef enum GufoChatRole {
  GUFO_CHAT_ROLE_SYSTEM = 0,
  GUFO_CHAT_ROLE_DEVELOPER = 1,
  GUFO_CHAT_ROLE_USER = 2,
  GUFO_CHAT_ROLE_ASSISTANT = 3,
  GUFO_CHAT_ROLE_TOOL = 4,
} GufoChatRole;

/* ReasoningEffort. */
typedef enum GufoChatEffort {
  GUFO_CHAT_EFFORT_MINIMAL = 0,
  GUFO_CHAT_EFFORT_LOW = 1,
  GUFO_CHAT_EFFORT_MEDIUM = 2,
  GUFO_CHAT_EFFORT_HIGH = 3,
  GUFO_CHAT_EFFORT_XHIGH = 4,
  GUFO_CHAT_EFFORT_MAX = 5,
} GufoChatEffort;

/* ChatRequest::ToolChoice. */
typedef enum GufoChatToolChoice {
  GUFO_CHAT_TOOL_CHOICE_AUTO = 0,
  GUFO_CHAT_TOOL_CHOICE_NONE = 1,
  GUFO_CHAT_TOOL_CHOICE_REQUIRED = 2,
} GufoChatToolChoice;

/* ReasoningOptions; each field is GUFO_CHAT_UNSET when absent. effort is a
   GufoChatEffort. budget_tokens counts only when has_budget is 1. */
typedef struct GufoChatReasoning {
  int32_t enabled;
  int32_t effort;
  int32_t preserve_thinking;
  uint8_t has_budget;
  uint64_t budget_tokens;
} GufoChatReasoning;

/* Bytes valid during the call that receives them. data is null only when len
   is 0. */
typedef struct GufoChatBytes {
  const uint8_t* data;
  size_t len;
} GufoChatBytes;

typedef struct GufoChatDefaults {
  /* Chat: the name a request without "model" selects unless require_model. */
  GufoChatBytes model_id;
  uint8_t require_model;
  /* Chat merges these server defaults after parsing; Responses and Messages
     start from them. The fragments start from the request's current values. */
  GufoChatReasoning reasoning;
  /* The fragments: names of the request's current tools in order, and its
     GufoChatToolChoice. The APIs pass none and GUFO_CHAT_TOOL_CHOICE_AUTO. */
  const GufoChatBytes* tool_names;
  size_t tool_name_count;
  int32_t tool_choice;
} GufoChatDefaults;

/* A rejection, valid during the reject callback only. status is 400 or 404;
   type is always invalid_request_error. */
typedef struct GufoChatRejection {
  int32_t status;
  const uint8_t* code;
  size_t code_len;
  const uint8_t* message;
  size_t message_len;
} GufoChatRejection;

/* One builder step. Steps marked [text] carry text/len, [int] integer,
   [index] index. A message, tool or tool call step adds one; the steps after
   it fill the newest. */
typedef enum GufoChatOp {
  /* ParsedChatRequest and the endpoint handlers. */
  GUFO_CHAT_OP_MODEL = 0,         /* [text] */
  GUFO_CHAT_OP_MAX_TOKENS = 1,    /* [int] */
  GUFO_CHAT_OP_STREAM = 2,        /* [int] 0 or 1 */
  GUFO_CHAT_OP_INCLUDE_USAGE = 3, /* [int] */
  /* ChatRequest fields, each assigned. */
  GUFO_CHAT_OP_CACHE_PROMPT = 10,    /* [int] */
  GUFO_CHAT_OP_CACHE_PERSIST = 11,   /* [int] */
  GUFO_CHAT_OP_RETURN_PROGRESS = 12, /* [int] */
  GUFO_CHAT_OP_ADD_VISION_ID = 13,   /* [int] */
  /* The whole ReasoningOptions, in these four steps. */
  GUFO_CHAT_OP_REASONING_ENABLED = 14,   /* [int] or GUFO_CHAT_UNSET */
  GUFO_CHAT_OP_REASONING_EFFORT = 15,    /* [int] GufoChatEffort or unset */
  GUFO_CHAT_OP_PRESERVE_THINKING = 16,   /* [int] or GUFO_CHAT_UNSET */
  GUFO_CHAT_OP_BUDGET_TOKENS = 17,       /* [int] or GUFO_CHAT_UNSET */
  GUFO_CHAT_OP_TOOL_CHOICE = 18,         /* [int] GufoChatToolChoice */
  GUFO_CHAT_OP_FORCED_TOOL_NAME = 19,    /* [text] */
  GUFO_CHAT_OP_PARALLEL_TOOL_CALLS = 20, /* [int] */
  GUFO_CHAT_OP_CONSTRAINED_TOOLS = 21,   /* [int] */
  /* tool_namespaces[key] = value. */
  GUFO_CHAT_OP_NAMESPACE_KEY = 22,   /* [text] */
  GUFO_CHAT_OP_NAMESPACE_VALUE = 23, /* [text] */
  /* Appends a ChatTool after the request's current ones. */
  GUFO_CHAT_OP_TOOL = 30,             /* [text] name */
  GUFO_CHAT_OP_TOOL_DESCRIPTION = 31, /* [text] */
  GUFO_CHAT_OP_TOOL_PARAMETERS = 32,  /* [text] parameters_json */
  GUFO_CHAT_OP_TOOL_DEFINITION = 33,  /* [text] definition_json */
  /* Keeps only the tool at [index] of all tools, current ones first. */
  GUFO_CHAT_OP_SELECT_TOOL = 34,
  /* Appends a ChatMessage. */
  GUFO_CHAT_OP_MESSAGE = 40,      /* [int] GufoChatRole */
  GUFO_CHAT_OP_CONTENT = 41,      /* [text] */
  GUFO_CHAT_OP_NAME = 42,         /* [text] */
  GUFO_CHAT_OP_THOUGHT = 43,      /* [text] */
  GUFO_CHAT_OP_TOOL_CALL_ID = 44, /* [text] */
  /* An image the read_image callback returned as [index], at content offset
     [int]; IMAGE_LOW has ImageDetail::kLow. */
  GUFO_CHAT_OP_IMAGE = 45,
  GUFO_CHAT_OP_IMAGE_LOW = 46,
  /* Appends a ToolCall to the newest message. */
  GUFO_CHAT_OP_CALL = 47,      /* [text] id */
  GUFO_CHAT_OP_CALL_NAME = 48, /* [text] */
  /* Appends a ToolArgument to the newest call. */
  GUFO_CHAT_OP_ARGUMENT = 49,       /* [text] name */
  GUFO_CHAT_OP_ARGUMENT_VALUE = 50, /* [text] value; [int] is_string */
  /* Appends a ChatMessage::Part to the newest message: a thought part when
     [index] is 1, else text, ending at [int] bytes of the thought or the
     content. */
  GUFO_CHAT_OP_PART = 51,
} GufoChatOp;

typedef struct GufoChatEvent {
  GufoChatOp op;
  uint32_t index;
  int64_t integer;
  const uint8_t* text;
  size_t len;
} GufoChatEvent;

/* The C++ helpers the decoder runs, each where its endpoint ran it. json is
   one JSON value (numbers spelled as parsed), or absent when len is 0. A body
   view is the request body with the values of "messages", "input", "tools",
   "system" and "instructions" replaced by null (the helpers read only their
   presence), or null when the body is not an object. */
typedef enum GufoChatHostField {
  /* Chat: ParseResponseFormat(response_format) and its description. */
  GUFO_CHAT_FIELD_RESPONSE_FORMAT = 0,
  /* Chat: ParseStopSequences(stop, kOpenAi). */
  GUFO_CHAT_FIELD_STOP = 1,
  /* Chat: ParseSamplingConfig on the body view; argument is the thinking mode
     (GUFO_CHAT_UNSET, 0 or 1). */
  GUFO_CHAT_FIELD_SAMPLING = 2,
  /* Chat: ParseLogprobs on the body view. */
  GUFO_CHAT_FIELD_LOGPROBS = 3,
  /* Responses: ReadCompatibilityOptions, then ParseResponsesLogprobs, on the
     body view; argument is the thinking mode. */
  GUFO_CHAT_FIELD_RESPONSES_OPTIONS = 4,
  /* Responses: ParseResponseFormat(text.format, responses) and its
     description. */
  GUFO_CHAT_FIELD_TEXT_FORMAT = 5,
  /* Messages: ReadCompatibilityOptions on the body view; argument is the
     thinking mode. */
  GUFO_CHAT_FIELD_MESSAGES_OPTIONS = 6,
  /* Messages: ParseStopSequences(stop_sequences, kAnthropic). */
  GUFO_CHAT_FIELD_STOP_SEQUENCES = 7,
} GufoChatHostField;

/* host_field results; any other value is a failure the caller recorded. */
#define GUFO_CHAT_HOST_OK 0
/* The helper rejected the request; the caller kept its response. */
#define GUFO_CHAT_HOST_REJECTED 1

/* Receives the message of a failed helper, valid during the call only. */
typedef struct GufoChatMessage {
  void* state;
  void (*set)(void* state, const uint8_t* data, size_t len);
} GufoChatMessage;

/* Every callback is noexcept. read_image and compile_schema return 0, 1 after
   passing their failure's message to error (a 400 rejection), or another
   value on a failure the caller recorded. */
typedef struct GufoChatHost {
  /* 1 when the process serves the model, 0 when not; other values fail. */
  int (*serves_model)(void* context, const uint8_t* name, size_t len);
  /* Reads an image URL within the request's budget; C++ keeps the bytes. */
  int (*read_image)(void* context, const uint8_t* url, size_t len,
                    uint32_t* image_id, const GufoChatMessage* error);
  /* Compiles a strict tool's parameters schema (one JSON value). */
  int (*compile_schema)(void* context, const uint8_t* json, size_t len,
                        const GufoChatMessage* error);
  int (*host_field)(void* context, GufoChatHostField field, int32_t argument,
                    const uint8_t* json, size_t len);
  /* Called only after the whole body is accepted; nonzero stops. */
  int (*emit)(void* context, const GufoChatEvent* event);
  void (*reject)(void* context, const GufoChatRejection* rejection);
} GufoChatHost;

/* Decodes body[0, len) as api reads it, in that endpoint's check order. On
   GUFO_CHAT_ACCEPTED emit got the request; otherwise it got nothing. */
GufoChatDecode gufo_chat_decode(GufoChatApi api, const uint8_t* body,
                                size_t len, const GufoChatDefaults* defaults,
                                const GufoChatHost* host, void* context);

#ifdef __cplusplus
}
#endif

#endif  // GUFO_CHAT_REQUEST_H_

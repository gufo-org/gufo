#ifndef GUFO_JSON_H_
#define GUFO_JSON_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The C interface of the Rust JSON parser. gufo::json::parse
   (src/core/json.cpp) builds its values from the events it reports. */
typedef enum GufoJsonParse {
  GUFO_JSON_COMPLETE = 0,
  /* The text is not JSON; the error sink got the reason. */
  GUFO_JSON_REJECTED = 1,
  GUFO_JSON_NO_MEMORY = 2,
  /* The event callback returned nonzero. */
  GUFO_JSON_STOPPED = 3,
  /* A caller bug (null pointer, missing callback) or a parser bug. */
  GUFO_JSON_INTERNAL = 4,
} GufoJsonParse;

typedef enum GufoJsonKind {
  GUFO_JSON_NULL = 0,
  GUFO_JSON_FALSE = 1,
  GUFO_JSON_TRUE = 2,
  /* number holds the value, text the token it was read from. */
  GUFO_JSON_NUMBER = 3,
  /* text holds the decoded UTF-8. */
  GUFO_JSON_STRING = 4,
  /* An object member's decoded key; its value follows. */
  GUFO_JSON_KEY = 5,
  GUFO_JSON_BEGIN_ARRAY = 6,
  GUFO_JSON_BEGIN_OBJECT = 7,
  /* Ends the innermost array or object. */
  GUFO_JSON_END = 8,
} GufoJsonKind;

/* One event, valid during the callback only. text is null when len is 0. */
typedef struct GufoJsonEvent {
  GufoJsonKind kind;
  double number;
  const uint8_t* text;
  size_t len;
} GufoJsonEvent;

/* Takes one event; returns nonzero to stop. */
typedef int (*GufoJsonEventFn)(void* context, const GufoJsonEvent* event);
/* Takes bytes valid during the call only. */
typedef void (*GufoJsonBytesFn)(void* context, const uint8_t* data, size_t len);

/* Parses the JSON text in data[0, len) as gufo::json::parse accepts it and
   reports its values to event in document order: scalars as they are read,
   containers between a begin and an end event, each member's value after its
   key. On GUFO_JSON_REJECTED, error got the reason (without the
   "JSON parse error: " prefix) unless it is null. Events taken before a
   failure are void. */
GufoJsonParse gufo_json_parse(const uint8_t* data, size_t len,
                              GufoJsonEventFn event, GufoJsonBytesFn error,
                              void* context);

/* gufo_json_dump writes numbers as parsed (json::Value::Numbers::kAsParsed)
   instead of canonically. */
#define GUFO_JSON_DUMP_AS_PARSED 1u
/* gufo_json_dump writes ", " and ": " as json::Value::tojson does, instead of
   "," and ":" as dump does. */
#define GUFO_JSON_DUMP_SPACED 2u

/* Parses data[0, len) as gufo_json_parse does into the Rust value tree and
   passes the text the Rust writer makes of it, in the style flags select, to
   out. On GUFO_JSON_REJECTED, error got the reason unless it is null. Unknown
   flags are GUFO_JSON_INTERNAL. json_write_test compares it with
   json::Value::dump; Gufo itself does not call it. */
GufoJsonParse gufo_json_dump(const uint8_t* data, size_t len, uint32_t flags,
                             GufoJsonBytesFn out, GufoJsonBytesFn error,
                             void* context);

#ifdef __cplusplus
}
#endif

#endif

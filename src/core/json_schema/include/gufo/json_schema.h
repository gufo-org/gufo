#ifndef GUFO_JSON_SCHEMA_H_
#define GUFO_JSON_SCHEMA_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The C interface of the Rust JSON Schema compiler. JsonSchemaRegex
   (src/core/json_schema_regex.cpp) steps over the compiled pattern tables;
   the lexemes (src/core/json_schema_lexeme.cpp) take the checked string
   keywords and the number lexeme from here. */
typedef enum GufoJsonSchemaStatus {
  GUFO_JSON_SCHEMA_COMPLETE = 0,
  /* The schema is invalid or unsupported; the error sink got the reason,
     without the "JSON Schema: " prefix. */
  GUFO_JSON_SCHEMA_REJECTED = 1,
  /* The schema is valid but no value satisfies it; the error sink got the
     reason, as for GUFO_JSON_SCHEMA_REJECTED. */
  GUFO_JSON_SCHEMA_EMPTY = 2,
  /* An allocation sized by the input failed. */
  GUFO_JSON_SCHEMA_NO_MEMORY = 3,
  /* The UnicodeSet callback failed. */
  GUFO_JSON_SCHEMA_STOPPED = 4,
  /* A caller bug (null pointer, unknown value kind) or a compiler bug. */
  GUFO_JSON_SCHEMA_INTERNAL = 5,
} GufoJsonSchemaStatus;

/* The kind of a GufoJsonSchemaValue. */
#define GUFO_JSON_SCHEMA_ABSENT 0
/* number holds the value; text holds its json::Value::dump() spelling when
   number is finite and is empty otherwise. */
#define GUFO_JSON_SCHEMA_NUMBER 1
/* text holds the UTF-8 string. */
#define GUFO_JSON_SCHEMA_STRING 2
/* Any other JSON value. */
#define GUFO_JSON_SCHEMA_OTHER 3

/* A schema keyword's value, valid during the call. text is null only when
   len is 0. */
typedef struct GufoJsonSchemaValue {
  int kind;
  double number;
  const uint8_t* text;
  size_t len;
} GufoJsonSchemaValue;

typedef struct GufoJsonSchemaBytes {
  const uint8_t* data;
  size_t len;
} GufoJsonSchemaBytes;

/* Takes bytes valid during the call only. */
typedef void (*GufoJsonSchemaBytesFn)(void* context, const uint8_t* data,
                                      size_t len);
/* Adds the codepoints [first, last] to the set being resolved. */
typedef void (*GufoJsonSchemaRangeFn)(void* sink, uint32_t first,
                                      uint32_t last);
/* Adds the string of len codepoints to the set being resolved: a property
   such as \p{RGI_Emoji_Flag_Sequence} holds multi-codepoint strings. */
typedef void (*GufoJsonSchemaStringFn)(void* sink, const uint32_t* codepoints,
                                       size_t len);
/* Resolves the ICU UnicodeSet pattern in pattern[0, len), such as
   "[\p{L}]", by calling add(sink, ...) for its ranges and
   add_string(sink, ...) for its strings. Returns
   GUFO_JSON_SCHEMA_SET_RESOLVED, GUFO_JSON_SCHEMA_SET_REJECTED when ICU
   rejects the pattern, or any other value (GUFO_JSON_SCHEMA_SET_FAILED) on
   failure; the compile then returns GUFO_JSON_SCHEMA_STOPPED. */
#define GUFO_JSON_SCHEMA_SET_RESOLVED 0
#define GUFO_JSON_SCHEMA_SET_REJECTED 1
#define GUFO_JSON_SCHEMA_SET_FAILED 2
typedef int (*GufoJsonSchemaSetFn)(void* context, const uint8_t* pattern,
                                   size_t len, GufoJsonSchemaRangeFn add,
                                   GufoJsonSchemaStringFn add_string,
                                   void* sink);

typedef struct GufoJsonSchemaRegex GufoJsonSchemaRegex;

/* The tables of a compiled pattern, valid until its handle is freed. State 0
   starts; UINT32_MAX is the dead state and marks no class. */
typedef struct GufoJsonSchemaRegexTables {
  uint32_t states;
  /* Alphabet classes: codepoints with the same transition from every state. */
  uint32_t classes;
  /* next[state * classes + class]: the successor, or UINT32_MAX where no
     accepted value continues. */
  const uint32_t* next;
  /* accepting[state] is 1 when the state accepts, 0 otherwise. */
  const uint8_t* accepting;
  /* distance[state]: the fewest codepoints to an accepting state, or
     UINT32_MAX. */
  const uint32_t* distance;
  /* The distinct successors of each state, dead ones included:
     successors[successor_offsets[s], successor_offsets[s + 1]). */
  const uint32_t* successor_offsets;
  const uint32_t* successors;
  /* The largest finite distance. */
  uint32_t maximum_suffix;
  /* Codepoint lookup: lookup_classes[i] is the class of
     [lookup_starts[i], lookup_starts[i + 1]), UINT32_MAX for surrogates.
     lookup_starts has lookup_len + 1 entries, starting at 0 and ending at
     0x110000. */
  size_t lookup_len;
  const uint32_t* lookup_starts;
  const uint32_t* lookup_classes;
  /* Each class's sorted, disjoint ranges as (first, last) pairs:
     class_ranges[2 * class_offsets[c], 2 * class_offsets[c + 1]). */
  const uint32_t* class_offsets;
  const uint32_t* class_ranges;
} GufoJsonSchemaRegexTables;

/* Compiles the intersection of the count patterns into a DFA whose
   transitions admit at most maximum codepoints, and writes its handle to out
   (null unless GUFO_JSON_SCHEMA_COMPLETE). sets resolves Unicode properties
   and \s. */
GufoJsonSchemaStatus gufo_json_schema_regex_compile(
    const GufoJsonSchemaBytes* patterns, size_t count, uint32_t maximum,
    GufoJsonSchemaSetFn sets, GufoJsonSchemaBytesFn error, void* context,
    GufoJsonSchemaRegex** out);
GufoJsonSchemaStatus gufo_json_schema_regex_tables(
    const GufoJsonSchemaRegex* regex, GufoJsonSchemaRegexTables* out);
/* Null is ignored. */
void gufo_json_schema_regex_free(GufoJsonSchemaRegex* regex);

/* Checks keywords[0..4): minLength, maxLength, pattern and format. Writes
   the length bounds (UINT32_MAX for no maximum, capped by the format) and
   reports, in that order, each pattern the decoded string must match. */
GufoJsonSchemaStatus gufo_json_schema_string(
    const GufoJsonSchemaValue* keywords, uint32_t* minimum, uint32_t* maximum,
    GufoJsonSchemaBytesFn pattern, GufoJsonSchemaBytesFn error, void* context);
/* Reports the pattern of the string format name[0, len) and writes its
   length cap (UINT32_MAX for none). */
GufoJsonSchemaStatus gufo_json_schema_format(const uint8_t* name, size_t len,
                                             uint32_t* maximum,
                                             GufoJsonSchemaBytesFn pattern,
                                             GufoJsonSchemaBytesFn error,
                                             void* context);

typedef struct GufoJsonSchemaNumber GufoJsonSchemaNumber;

/* Compiles the number lexeme from keywords[0..5): minimum, maximum,
   exclusiveMinimum, exclusiveMaximum and multipleOf, for integers when
   integer is nonzero. Numbers have up to digits integral and fraction
   digits, more where a schema number needs them. Writes the handle to out
   (null unless GUFO_JSON_SCHEMA_COMPLETE). */
GufoJsonSchemaStatus gufo_json_schema_number_new(
    const GufoJsonSchemaValue* keywords, int integer, uint32_t digits,
    GufoJsonSchemaBytesFn error, void* context, GufoJsonSchemaNumber** out);

#define GUFO_JSON_SCHEMA_PREFIX 1
#define GUFO_JSON_SCHEMA_MATCH 2
/* Writes to out GUFO_JSON_SCHEMA_PREFIX when bytes[0, len) can continue to
   an accepted number, and GUFO_JSON_SCHEMA_MATCH when it is one. */
GufoJsonSchemaStatus gufo_json_schema_number_check(
    const GufoJsonSchemaNumber* number, const uint8_t* bytes, size_t len,
    uint8_t* out);
/* Writes to accepted 1 when the JSON number spelled text[0, len) is
   accepted, 0 otherwise. */
GufoJsonSchemaStatus gufo_json_schema_number_accept(
    const GufoJsonSchemaNumber* number, const uint8_t* text, size_t len,
    uint8_t* accepted, GufoJsonSchemaBytesFn error, void* context);
/* Null is ignored. */
void gufo_json_schema_number_free(GufoJsonSchemaNumber* number);

/* Reports the exact least common multiple of two multipleOf values, spelled
   as digits, "e" and an exponent. */
GufoJsonSchemaStatus gufo_json_schema_multiple_of(const GufoJsonSchemaValue* a,
                                                  const GufoJsonSchemaValue* b,
                                                  GufoJsonSchemaBytesFn result,
                                                  GufoJsonSchemaBytesFn error,
                                                  void* context);
/* Writes to equal 1 when two JSON numbers have the same exact value. */
GufoJsonSchemaStatus gufo_json_schema_number_equal(
    const uint8_t* a, size_t a_len, const uint8_t* b, size_t b_len,
    uint8_t* equal, GufoJsonSchemaBytesFn error, void* context);

#ifdef __cplusplus
}
#endif

#endif

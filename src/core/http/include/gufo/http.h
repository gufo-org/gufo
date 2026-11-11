#ifndef GUFO_HTTP_H_
#define GUFO_HTTP_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The C interface of the Rust HTTP request parser: request heads as RFC 9112
   defines them (through httparse), body framing, query parameters and token
   lists in header fields. src/cli/serve/http_server.cpp reads its requests
   with it. */
typedef enum GufoHttpParse {
  GUFO_HTTP_COMPLETE = 0,
  /* No empty line has ended the head yet. */
  GUFO_HTTP_PARTIAL = 1,
  /* Not an origin-form HTTP/1.0 or HTTP/1.1 request head. */
  GUFO_HTTP_MALFORMED = 2,
  /* A caller bug (null pointer, a failing callback), memory exhausted, or a
     parser bug. */
  GUFO_HTTP_INTERNAL = 3,
  /* More than 100 header fields (lines after the request line). */
  GUFO_HTTP_TOO_MANY_FIELDS = 4,
} GufoHttpParse;

/* Bytes inside the caller's buffer. */
typedef struct GufoHttpSlice {
  const uint8_t* data;
  size_t len;
} GufoHttpSlice;

typedef struct GufoHttpHead {
  /* Bytes up to and including the empty line that ends the head. */
  size_t len;
  GufoHttpSlice method;
  /* The target before its first "?", and after it. */
  GufoHttpSlice path;
  GufoHttpSlice query;
  int http11;
  /* 0 for a Transfer-Encoding field, or Content-Length values that are not
     decimal digits, overflow or disagree. */
  int framing_valid;
  /* From Content-Length, 0 without one. */
  uint64_t content_length;
  /* An "Expect: 100-continue" field. */
  int expect_continue;
} GufoHttpHead;

/* Takes one header field, in order; returns nonzero to give up. */
typedef int (*GufoHttpFieldFn)(void* context, const uint8_t* name,
                               size_t name_len, const uint8_t* value,
                               size_t value_len);
/* Takes bytes valid during the call only. */
typedef void (*GufoHttpBytesFn)(void* context, const uint8_t* data, size_t len);

/* Parses the request head at the start of buffer. known_partial is a length
   of the same buffer that already parsed as GUFO_HTTP_PARTIAL (0 at first),
   so that only the new bytes are searched for the end of the head. On
   GUFO_HTTP_COMPLETE, *head describes it, its slices point into buffer, and
   field was called for each header field in order. */
GufoHttpParse gufo_http_parse_head(const uint8_t* buffer, size_t len,
                                   size_t known_partial, GufoHttpHead* head,
                                   GufoHttpFieldFn field, void* context);

/* Finds the first "&"-separated parameter of query whose decoded name is key
   ("+" as space, "%XX" as a byte), hands its decoded value to sink (empty
   without "=") and returns 1; 0 if there is none (an empty query and the end
   after a final "&" hold none); -1 on a bug or when memory runs out. */
int gufo_http_query_param(const uint8_t* query, size_t query_len,
                          const uint8_t* key, size_t key_len,
                          GufoHttpBytesFn sink, void* context);

/* 1 if the comma-separated value lists the lowercase token, ignoring ASCII
   case and the spaces and tabs around each item; 0 if not; -1 on a bug. */
int gufo_http_has_token(const uint8_t* value, size_t value_len,
                        const uint8_t* token, size_t token_len);

#ifdef __cplusplus
}
#endif

#endif

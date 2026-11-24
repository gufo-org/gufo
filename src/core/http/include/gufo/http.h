#ifndef GUFO_HTTP_H_
#define GUFO_HTTP_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The C interface of the Rust HTTP request parser: request heads as RFC 9112
   defines them (through httparse), body framing, query parameters, token
   lists in header fields and multipart/form-data bodies.
   src/cli/serve/http_server.cpp reads its requests with it. */
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

typedef enum GufoMultipartParse {
  GUFO_MULTIPART_COMPLETE = 0,
  /* The Content-Type is not multipart/form-data. */
  GUFO_MULTIPART_NOT_FORM_DATA = 1,
  /* Its boundary is missing, not 1 to 70 RFC 2046 characters, or in malformed
     parameters. */
  GUFO_MULTIPART_BAD_BOUNDARY = 2,
  /* The body breaks RFC 7578; the error sink got the message. */
  GUFO_MULTIPART_MALFORMED = 3,
  /* More than max_fields fields. */
  GUFO_MULTIPART_TOO_MANY_FIELDS = 4,
  /* With unique_names, a repeated field name; the error sink got it. */
  GUFO_MULTIPART_DUPLICATE_FIELD = 5,
  /* The field callback returned nonzero. */
  GUFO_MULTIPART_STOPPED = 6,
  /* A caller bug (null pointer) or a parser bug. */
  GUFO_MULTIPART_INTERNAL = 7,
} GufoMultipartParse;

/* One field of a form; its slices are valid during the callback only. */
typedef struct GufoMultipartField {
  GufoHttpSlice name;
  /* Its Content-Disposition has a filename (or filename*) parameter. */
  int has_filename;
  GufoHttpSlice content;
} GufoMultipartField;

/* Takes one field, in body order; returns nonzero to stop. */
typedef int (*GufoMultipartFieldFn)(void* context,
                                    const GufoMultipartField* field);

/* Reads a multipart/form-data body as RFC 7578 defines it (RFC 2046 framing
   from the first boundary at the body's start, with the epilogue ignored;
   RFC 9110 parameters), calling field for each field in order, each only
   after the checks that precede it. */
GufoMultipartParse gufo_multipart_parse(const uint8_t* content_type,
                                        size_t content_type_len,
                                        const uint8_t* body, size_t body_len,
                                        size_t max_fields, int unique_names,
                                        GufoMultipartFieldFn field,
                                        GufoHttpBytesFn error, void* context);

#ifdef __cplusplus
}
#endif

#endif

#ifndef GUFO_BASE64_H_
#define GUFO_BASE64_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The C interface of the Rust base64 codec: the standard alphabet (RFC 4648,
   section 4) with "=" padding. src/core/base64.hpp wraps it for C++; see
   there for the meaning of each status. */
typedef enum GufoBase64Status {
  GUFO_BASE64_OK = 0,
  GUFO_BASE64_LENGTH = 1,
  GUFO_BASE64_PADDING = 2,
  GUFO_BASE64_AFTER_PADDING = 3,
  GUFO_BASE64_CHARACTER = 4,
  GUFO_BASE64_NON_CANONICAL = 5,
  GUFO_BASE64_TOO_LONG = 6,
  /* A caller bug (null pointer, wrong output length) or a codec bug. */
  GUFO_BASE64_INTERNAL = 7,
} GufoBase64Status;

/* Encodes len bytes into out_len characters; out_len must be
   ceil(len / 3) * 4. */
GufoBase64Status gufo_base64_encode(const uint8_t* bytes, size_t len,
                                    uint8_t* out, size_t out_len);

/* Decodes len characters into out, which holds at least len / 4 * 3 bytes,
   and stores the decoded length in *written (0 on an error). With canonical,
   the bits that padding hides must be zero; more than max_decoded bytes is
   GUFO_BASE64_TOO_LONG. */
GufoBase64Status gufo_base64_decode(const uint8_t* text, size_t len,
                                    bool canonical, size_t max_decoded,
                                    uint8_t* out, size_t out_len,
                                    size_t* written);

#ifdef __cplusplus
}
#endif

#endif

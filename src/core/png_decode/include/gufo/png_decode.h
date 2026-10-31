#ifndef GUFO_PNG_DECODE_H_
#define GUFO_PNG_DECODE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum GufoPngStatus {
  GUFO_PNG_OK = 0,
  /* The bytes are not a valid PNG, or the image is wider or taller than
     1,000,000 pixels. */
  GUFO_PNG_INVALID = 1,
  /* A caller bug (null pointer, wrong output length) or a decoder bug. */
  GUFO_PNG_INTERNAL = 2,
} GufoPngStatus;

typedef struct GufoPngDimensions {
  uint32_t width;
  uint32_t height;
} GufoPngDimensions;

/* Reads the header and every chunk before the image data. */
GufoPngStatus gufo_png_dimensions(const uint8_t* bytes, size_t len,
                                  GufoPngDimensions* out);

/* Decodes into packed RGB8: alpha dropped without compositing, gamma and ICC
   ignored, 16-bit gray saturated at 255, other 16-bit samples reduced to the
   high byte. rgb_len must be 3 * width * height for these bytes. The output
   is unspecified unless the result is GUFO_PNG_OK. Neither buffer may change
   or overlap during the call; nothing is retained after it. */
GufoPngStatus gufo_png_decode_rgb8(const uint8_t* bytes, size_t len,
                                   uint8_t* rgb, size_t rgb_len);

#ifdef __cplusplus
}
#endif

#endif

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

typedef struct GufoPngDecoder GufoPngDecoder;

/* Reads the header and every chunk before the image data. On GUFO_PNG_OK,
   *decoder receives a decoder that reads the bytes until gufo_png_close, so
   they must stay unchanged until then; otherwise *decoder is null. */
GufoPngStatus gufo_png_open(const uint8_t* bytes, size_t len,
                            GufoPngDecoder** decoder, GufoPngDimensions* out);

/* Decodes the opened image into packed RGB8: alpha dropped without
   compositing, gamma and ICC ignored, 16-bit gray saturated at 255, other
   16-bit samples reduced to the high byte. rgb_len must be 3 * width * height,
   and rgb must not overlap the PNG bytes. A decoder decodes once; a second
   call returns GUFO_PNG_INTERNAL. The output is unspecified unless the result
   is GUFO_PNG_OK. */
GufoPngStatus gufo_png_decode_rgb8(GufoPngDecoder* decoder, uint8_t* rgb,
                                   size_t rgb_len);

/* Releases a decoder from gufo_png_open. A null decoder is ignored. */
void gufo_png_close(GufoPngDecoder* decoder);

#ifdef __cplusplus
}
#endif

#endif

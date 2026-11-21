#ifndef GUFO_JPEG_DECODE_H_
#define GUFO_JPEG_DECODE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum GufoJpegStatus {
  GUFO_JPEG_OK = 0,
  /* The bytes are not a valid, complete JPEG with 1, 3 or 4 components and
     8-bit samples. */
  GUFO_JPEG_INVALID = 1,
  /* A caller bug (null pointer, wrong output length) or a decoder bug. */
  GUFO_JPEG_INTERNAL = 2,
} GufoJpegStatus;

typedef struct GufoJpegInfo {
  uint32_t width;
  uint32_t height;
  /* The EXIF orientation, 1 to 8, from the first APP1 Exif segment before
     the first scan; 1 when it is absent or malformed. */
  uint32_t orientation;
} GufoJpegInfo;

typedef struct GufoJpegDecoder GufoJpegDecoder;

/* Reads the headers up to the first scan. On GUFO_JPEG_OK, *decoder receives
   a decoder that reads the bytes until gufo_jpeg_close, so they must stay
   unchanged until then; otherwise *decoder is null. */
GufoJpegStatus gufo_jpeg_open(const uint8_t* bytes, size_t len,
                              GufoJpegDecoder** decoder, GufoJpegInfo* out);

/* Decodes the opened image into packed RGB8, before orientation. Gray is
   replicated; Adobe CMYK and YCCK are read as inverted CMYK and multiplied by
   K, as Pillow converts them; ICC profiles are ignored. rgb_len must be
   3 * width * height, and rgb must not overlap the JPEG bytes. A decoder
   decodes once; a second call returns GUFO_JPEG_INTERNAL. The output is
   unspecified unless the result is GUFO_JPEG_OK. */
GufoJpegStatus gufo_jpeg_decode_rgb8(GufoJpegDecoder* decoder, uint8_t* rgb,
                                     size_t rgb_len);

/* Releases a decoder from gufo_jpeg_open. A null decoder is ignored. */
void gufo_jpeg_close(GufoJpegDecoder* decoder);

#ifdef __cplusplus
}
#endif

#endif

#ifndef GUFO_PNG_H_
#define GUFO_PNG_H_

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
  /* A caller bug (null pointer, wrong length, unknown format), a failed
     allocation or a codec bug. */
  GUFO_PNG_INTERNAL = 2,
} GufoPngStatus;

/* Values of the format argument of gufo_png_open. */
enum {
  /* Packed RGB8 as Pillow's convert("RGB"): alpha dropped without
     compositing, gamma and ICC ignored, 16-bit gray saturated at 255, other
     16-bit samples reduced to the high byte. */
  GUFO_PNG_RGB8 = 0,
  /* Packed, unpremultiplied RGBA8 as libpng's simplified API with
     PNG_FORMAT_RGBA: tRNS becomes alpha, 16-bit samples round to 8 bits, and
     color samples convert from the file gamma (sRGB or gAMA; otherwise sRGB
     for 8-bit or smaller and linear for 16-bit samples) to sRGB. Interlaced
     16-bit images decode correctly, unlike libpng 1.6.59. */
  GUFO_PNG_RGBA8 = 1,
};

typedef struct GufoPngDimensions {
  uint32_t width;
  uint32_t height;
} GufoPngDimensions;

typedef struct GufoPngDecoder GufoPngDecoder;

/* Reads the header and every chunk before the image data, to decode later in
   format. On GUFO_PNG_OK, *decoder receives a decoder that reads the bytes
   until gufo_png_close, so they must stay unchanged until then; otherwise
   *decoder is null. */
GufoPngStatus gufo_png_open(const uint8_t* bytes, size_t len, uint32_t format,
                            GufoPngDecoder** decoder, GufoPngDimensions* out);

/* Decodes the opened image in its format. len must be width * height times 3
   (RGB8) or 4 (RGBA8), and pixels must not overlap the PNG bytes. A decoder
   decodes once; a second call returns GUFO_PNG_INTERNAL. The output is
   unspecified unless the result is GUFO_PNG_OK. */
GufoPngStatus gufo_png_decode(GufoPngDecoder* decoder, uint8_t* pixels,
                              size_t len);

/* Releases a decoder from gufo_png_open. A null decoder is ignored. */
void gufo_png_close(GufoPngDecoder* decoder);

/* Receives the encoded PNG, valid only during the call. */
typedef void (*GufoPngBytesFn)(void* context, const uint8_t* data, size_t len);

/* Encodes packed RGBA8 rows (len = 4 * width * height; width and height from
   1 to 1,000,000) as an 8-bit RGBA PNG with an sRGB chunk, no interlacing,
   adaptive filters and fast deflate, and passes it to bytes once on
   GUFO_PNG_OK. bytes must not unwind. */
GufoPngStatus gufo_png_encode_rgba8(const uint8_t* rgba, size_t len,
                                    uint32_t width, uint32_t height,
                                    GufoPngBytesFn bytes, void* context);

#ifdef __cplusplus
}
#endif

#endif

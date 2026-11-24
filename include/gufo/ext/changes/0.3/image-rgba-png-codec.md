# RGBA image decoding and PNG encoding

Core image helpers (`src/core/image.hpp`, forwarded by `src/ext/image.hpp`):
- `RgbaImage` holds packed, unpremultiplied RGBA8 pixels with their width and
  height.
- `DecodeImageRgba` decodes PNG to RGBA8 with the colour handling of libpng's
  simplified API: tRNS becomes alpha, 16-bit samples round to 8 bits, color
  samples convert from the file gamma to sRGB, and EXIF orientation is
  ignored. JPEG and WebP decode as `DecodeImage` does, with opaque alpha.
- `EncodePngRgba` encodes RGBA8 pixels as an 8-bit RGBA PNG with an sRGB chunk.

Both are additions; existing extensions keep compiling. Image extensions can
use them instead of linking libpng, which Gufo no longer provides.

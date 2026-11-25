# RGBA image decoding and PNG encoding

Core image helpers (`src/core/image.hpp`, forwarded by `src/ext/image.hpp`):
- `RgbaImage` holds packed, unpremultiplied RGBA8 pixels with their width and
  height.
- `DecodeImageRgba` decodes PNG to RGBA8 with the color handling of libpng's
  simplified API: tRNS becomes alpha, 16-bit samples round to 8 bits, color
  samples convert from the file gamma to sRGB, and EXIF orientation is
  ignored. JPEG and WebP decode as `DecodeImage` does, with opaque alpha.
  Invalid input throws `std::invalid_argument`.
- `EncodePngRgba` encodes RGBA8 pixels as an 8-bit RGBA PNG with an sRGB
  chunk, and throws `std::runtime_error` for invalid geometry.

The declarations are additions. Gufo no longer finds libpng: the build drops
`find_package(PNG)`, and the extension SDK no longer offers or pins `PNG::PNG`.
An extension that links `PNG::PNG` must list libpng in its manifest
`packages`, or use these helpers instead.

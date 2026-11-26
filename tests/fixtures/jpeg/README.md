# JPEG decoder fixtures

`jpeg_decode_test` decodes these images and checks the size and pixel digest
of each, and libjpeg-turbo's pixels at three points within the documented
tolerance. It derives its metadata, EXIF orientation, framing, truncation and
bit-flip cases from them in memory. No encoder is needed to run the test.

Each file is named `<format>-<width>x<height>.jpg`. They were written once
with libjpeg-turbo 3.2.0 (the devshell's `pkgs.libjpeg`), with the encoder
settings and image of the libjpeg-turbo comparison the test replaced
(`tests/core/jpeg_decode_test.cpp` at `94b363816`), which also decoded the
recorded points. The image is a gradient with edges and a little noise,
seeded by its size, written at quality 90 unless the name says otherwise:

| Name | Coding |
|------|--------|
| `gray` | Grayscale |
| `ycc444`, `ycc422`, `ycc420`, `ycc440`, `ycc411`, `ycc410` | YCbCr with that chroma subsampling |
| `rgb` | RGB components |
| `cmyk`, `ycck` | Adobe CMYK, and YCCK with 4:2:0 luma sampling |
| `progressive-*`, `progressive444`, `progressive420`, `progressive422` | Progressive scans |
| `optimized420` | Optimized Huffman tables |
| `restart420`, `restart444` | A restart interval of one and two MCU rows |
| `progressive-restart420` | Progressive with a restart interval of one MCU row |
| `quality100-444`, `quality1-420` | Quality 100 and 1 |
| `arithmetic420`, `arithmetic-progressive420` | Arithmetic coding, which Gufo rejects |
| `lossless-rgb`, `lossless-gray` | Lossless JPEG, which Gufo rejects |

To add a fixture, write it with any encoder, record libjpeg-turbo's RGB
pixels at the test's three points (for YCbCr and gray, `djpeg -ppm`), and
check that DecodeImage stays within the tolerance before recording its digest.
A failing case prints the outcome it got.

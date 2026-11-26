// DecodeImage on the JPEG fixtures in tests/fixtures/jpeg, which
// libjpeg-turbo wrote once (see the README there), with expected results:
// each decodes to its size and pixel digest, within the documented tolerance
// of libjpeg-turbo's pixels at three recorded points; codings the Rust decoder
// does not support are rejected; and cases derived from the fixtures in
// memory (metadata, EXIF orientations, framing errors, truncations and bit
// flips) give their recorded outcomes. Every difference from the replaced
// libjpeg-turbo decoder is listed with its reason.
//
//   jpeg_decode_test                          run the cases
//   jpeg_decode_test --corpus DIR... [--repeat N]
//                                             decode and time real files

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <new>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/image.hpp"

namespace {

using Bytes = std::vector<std::uint8_t>;

// zune-jpeg's IDCT and chroma upsampling round differently from libjpeg-turbo's
// islow IDCT and fancy upsampling. With subsampled chroma, a width or height
// of 8 mod 16 ends the image inside a chroma block, and zune-jpeg upsamples
// the outermost row or column against the block's padding where libjpeg-turbo
// replicates the last sample. Only subsampled images get the border allowance.
constexpr int kMaxDifference = 6;
constexpr int kMaxSubsampledBorderDifference = 40;

// Differences from libjpeg-turbo.
constexpr std::string_view kArithmetic =
    "zune-jpeg does not decode arithmetic coding (SOF9, SOF10); libjpeg-turbo "
    "does";
constexpr std::string_view kLossless =
    "zune-jpeg does not decode lossless JPEG (SOF3); libjpeg-turbo does";
constexpr std::string_view kShortScan =
    "zune-jpeg pads a scan that reaches a marker before its last MCU, or an "
    "interval without its RST marker, with zeros; libjpeg-turbo warns, which "
    "the replaced decoder rejected";
constexpr std::string_view kDnl =
    "zune-jpeg's strict mode rejects a DNL marker; libjpeg-turbo skips one "
    "that follows a nonzero SOF height";
constexpr std::string_view kAcOverrun =
    "a flipped bit makes an AC run pass coefficient 63; libjpeg-turbo silently "
    "stores it in coefficient 63, zune-jpeg's slow Huffman path (pos & 63) "
    "wraps it into a low-frequency coefficient";
// A flipped entropy-coded or Huffman table bit that libjpeg-turbo warned
// about, which the replaced decoder rejected, may decode.

// FNV-1a over the RGB pixels.
std::uint64_t Digest(std::span<const std::uint8_t> pixels) {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  for (const auto byte : pixels)
    hash = (hash ^ byte) * 0x100000001b3ULL;
  return hash;
}

// The image DecodeImage returns, or that it threw std::invalid_argument.
struct Outcome {
  bool decoded{false};
  std::uint32_t width{0};
  std::uint32_t height{0};
  std::uint64_t digest{0};
  bool operator==(const Outcome&) const = default;
};
constexpr Outcome kInvalid{};

std::string Describe(const Outcome& outcome) {
  if (!outcome.decoded)
    return "std::invalid_argument";
  std::ostringstream text;
  text << outcome.width << 'x' << outcome.height << " digest 0x" << std::hex
       << outcome.digest;
  return text.str();
}

// Throws for anything but pixels or std::invalid_argument.
Outcome Decode(std::span<const std::uint8_t> jpeg,
               gufo::core::Image* image = nullptr) {
  try {
    auto decoded = gufo::core::DecodeImage(jpeg);
    const Outcome outcome{true, decoded.width, decoded.height,
                          Digest(decoded.pixels)};
    if (decoded.pixels.size() !=
        std::size_t{decoded.width} * decoded.height * 3)
      throw std::logic_error("pixels do not fill the image");
    if (image != nullptr)
      *image = std::move(decoded);
    return outcome;
  } catch (const std::invalid_argument&) {
    return kInvalid;
  }
}

class Failures {
public:
  void Check(bool condition, const std::string& message) {
    if (!condition) {
      std::cerr << "FAIL: " << message << '\n';
      ++count_;
    }
  }
  void Expect(const std::string& name, const Outcome& actual,
              const Outcome& expected) {
    Check(actual == expected,
          name + ": " + Describe(actual) + ", expected " + Describe(expected));
  }
  int count() const { return count_; }

private:
  int count_{0};
};

Bytes ReadFixture(std::string_view name) {
  const auto path =
      std::filesystem::path(GUFO_JPEG_FIXTURES) / (std::string(name) + ".jpg");
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot read " + path.string());
  return {std::istreambuf_iterator<char>(in), {}};
}

struct Sample {
  std::uint32_t x, y;
  std::array<int, 3> rgb;
};

// What libjpeg-turbo 3.2.0 decoded at three points, and DecodeImage's size
// and digest.
struct Decoded {
  std::string_view name;
  std::uint32_t width, height;
  std::uint64_t digest;
  bool subsampled;
  std::array<Sample, 3> reference;
};

// clang-format off
const Decoded kDecoded[] = {
    {"gray-17x9", 17, 9, 0x6f08cddc6c970ff8ULL, false, {{{0, 0, {68, 68, 68}}, {8, 4, {48, 48, 48}}, {16, 8, {88, 88, 88}}}}},
    {"gray-64x48", 64, 48, 0x50d74aef7088e360ULL, false, {{{0, 0, {66, 66, 66}}, {32, 24, {220, 220, 220}}, {63, 47, {236, 236, 236}}}}},
    {"ycc444-17x9", 17, 9, 0x6c16f14eaca252e1ULL, false, {{{0, 0, {76, 69, 149}}, {8, 4, {36, 214, 218}}, {16, 8, {90, 3, 45}}}}},
    {"ycc422-17x9", 17, 9, 0xbf4536af37864621ULL, true, {{{0, 0, {60, 78, 144}}, {8, 4, {54, 204, 221}}, {16, 8, {76, 12, 36}}}}},
    {"ycc422-8x8", 8, 8, 0xc0218977a504787cULL, true, {{{0, 0, {65, 70, 136}}, {4, 4, {12, 117, 61}}, {7, 7, {145, 145, 241}}}}},
    {"ycc420-33x31", 33, 31, 0xfab5482f48222123ULL, true, {{{0, 0, {65, 70, 151}}, {16, 15, {76, 86, 75}}, {32, 30, {223, 173, 124}}}}},
    {"ycc420-1x1", 1, 1, 0xfa076719a0521fd6ULL, true, {{{0, 0, {65, 78, 148}}, {0, 0, {65, 78, 148}}, {0, 0, {65, 78, 148}}}}},
    {"ycc420-8x8", 8, 8, 0xa8e76247ba0b7896ULL, true, {{{0, 0, {62, 67, 159}}, {4, 4, {6, 115, 86}}, {7, 7, {122, 154, 255}}}}},
    {"ycc420-24x8", 24, 8, 0x23002cbcf244e451ULL, true, {{{0, 0, {67, 80, 135}}, {12, 4, {112, 154, 236}}, {23, 7, {161, 222, 66}}}}},
    {"ycc420-250x3", 250, 3, 0x7c983c5da9941086ULL, true, {{{0, 0, {65, 72, 160}}, {125, 1, {148, 63, 96}}, {249, 2, {245, 58, 189}}}}},
    {"ycc440-17x9", 17, 9, 0x1a9d3188552670a0ULL, true, {{{0, 0, {62, 79, 131}}, {8, 4, {70, 196, 219}}, {16, 8, {54, 29, 7}}}}},
    {"ycc411-33x31", 33, 31, 0x1de7b7e783a01ff4ULL, true, {{{0, 0, {68, 71, 138}}, {16, 15, {116, 58, 116}}, {32, 30, {255, 156, 124}}}}},
    {"ycc410-33x31", 33, 31, 0x103b801013a894c8ULL, true, {{{0, 0, {61, 71, 160}}, {16, 15, {67, 92, 70}}, {32, 30, {251, 158, 124}}}}},
    {"rgb-17x9", 17, 9, 0xab9fd93c7d8b9518ULL, false, {{{0, 0, {69, 78, 144}}, {8, 4, {53, 197, 219}}, {16, 8, {90, 4, 45}}}}},
    {"cmyk-17x9", 17, 9, 0x9c9c19a609711190ULL, false, {{{0, 0, {8, 9, 16}}, {8, 4, {8, 38, 43}}, {16, 8, {50, 2, 24}}}}},
    {"ycck-17x9", 17, 9, 0xbd802b6f039f07ddULL, true, {{{0, 0, {9, 10, 20}}, {8, 4, {11, 38, 39}}, {16, 8, {15, 25, 0}}}}},
    {"progressive-gray-17x9", 17, 9, 0x6f08cddc6c970ff8ULL, false, {{{0, 0, {68, 68, 68}}, {8, 4, {48, 48, 48}}, {16, 8, {88, 88, 88}}}}},
    {"progressive444-17x9", 17, 9, 0x6c16f14eaca252e1ULL, false, {{{0, 0, {76, 69, 149}}, {8, 4, {36, 214, 218}}, {16, 8, {90, 3, 45}}}}},
    {"progressive422-17x9", 17, 9, 0xbf4536af37864621ULL, true, {{{0, 0, {60, 78, 144}}, {8, 4, {54, 204, 221}}, {16, 8, {76, 12, 36}}}}},
    {"progressive420-33x31", 33, 31, 0xfab5482f48222123ULL, true, {{{0, 0, {65, 70, 151}}, {16, 15, {76, 86, 75}}, {32, 30, {223, 173, 124}}}}},
    {"progressive420-64x48", 64, 48, 0x850b9a7f5085c9cdULL, true, {{{0, 0, {64, 73, 164}}, {32, 24, {227, 168, 32}}, {63, 47, {240, 149, 118}}}}},
    {"progressive-cmyk-17x9", 17, 9, 0x9c9c19a609711190ULL, false, {{{0, 0, {8, 9, 16}}, {8, 4, {8, 38, 43}}, {16, 8, {50, 2, 24}}}}},
    {"optimized420-17x9", 17, 9, 0x348837e61abadd81ULL, true, {{{0, 0, {63, 76, 144}}, {8, 4, {53, 205, 216}}, {16, 8, {45, 34, 6}}}}},
    {"restart420-64x48", 64, 48, 0x850b9a7f5085c9cdULL, true, {{{0, 0, {64, 73, 164}}, {32, 24, {227, 168, 32}}, {63, 47, {240, 149, 118}}}}},
    {"restart444-17x9", 17, 9, 0x6c16f14eaca252e1ULL, false, {{{0, 0, {76, 69, 149}}, {8, 4, {36, 214, 218}}, {16, 8, {90, 3, 45}}}}},
    {"progressive-restart420-33x31", 33, 31, 0xfab5482f48222123ULL, true, {{{0, 0, {65, 70, 151}}, {16, 15, {76, 86, 75}}, {32, 30, {223, 173, 124}}}}},
    {"quality100-444-17x9", 17, 9, 0x3de963d73a07410aULL, false, {{{0, 0, {69, 78, 145}}, {8, 4, {51, 203, 226}}, {16, 8, {90, 3, 45}}}}},
    {"quality1-420-17x9", 17, 9, 0xf0112921b25c2a5cULL, true, {{{0, 0, {39, 92, 162}}, {8, 4, {83, 152, 121}}, {16, 8, {39, 40, 0}}}}},
};
// clang-format on

void CheckDecoded(Failures& failures) {
  for (const auto& fixture : kDecoded) {
    const std::string name(fixture.name);
    gufo::core::Image image;
    failures.Expect(name, Decode(ReadFixture(fixture.name), &image),
                    {true, fixture.width, fixture.height, fixture.digest});
    if (image.width != fixture.width || image.height != fixture.height)
      continue;
    for (const auto& sample : fixture.reference) {
      const bool border = sample.x == 0 || sample.y == 0 ||
                          sample.x + 1 == image.width ||
                          sample.y + 1 == image.height;
      const int limit = border && fixture.subsampled
                            ? kMaxSubsampledBorderDifference
                            : kMaxDifference;
      const auto* pixel = image.pixels.data() +
                          (std::size_t{sample.y} * image.width + sample.x) * 3;
      for (int c = 0; c < 3; ++c) {
        failures.Check(std::abs(int{pixel[c]} - sample.rgb[c]) <= limit,
                       name + ": pixel " + std::to_string(sample.x) + "," +
                           std::to_string(sample.y) + " channel " +
                           std::to_string(c) + " is " +
                           std::to_string(pixel[c]) + ", libjpeg-turbo " +
                           std::to_string(sample.rgb[c]));
      }
    }
  }
  // Codings zune-jpeg does not decode. libjpeg-turbo cannot convert lossless
  // gray to RGB either, so it rejected that one too.
  const struct {
    std::string_view name, reason;
  } rejected[] = {
      {"arithmetic420-33x31", kArithmetic},
      {"arithmetic-progressive420-33x31", kArithmetic},
      {"lossless-rgb-33x31", kLossless},
      {"lossless-gray-33x31", kLossless},
  };
  for (const auto& fixture : rejected) {
    failures.Expect(
        std::string(fixture.name) + " (" + std::string(fixture.reason) + ")",
        Decode(ReadFixture(fixture.name)), kInvalid);
  }
}

std::size_t Find(const Bytes& jpeg, std::uint8_t marker) {
  for (std::size_t i = 2; i + 1 < jpeg.size(); ++i) {
    if (jpeg[i] == 0xff && jpeg[i + 1] == marker)
      return i;
  }
  throw std::logic_error("marker not found");
}

// Inserts a segment after the JFIF APP0 segment that follows SOI.
Bytes WithSegment(Bytes jpeg, std::uint8_t marker, const Bytes& data) {
  const std::size_t at = 4 + ((std::size_t{jpeg[4]} << 8) | jpeg[5]);
  const auto length = data.size() + 2;
  Bytes segment{0xff, marker, static_cast<std::uint8_t>(length >> 8),
                static_cast<std::uint8_t>(length)};
  segment.insert(segment.end(), data.begin(), data.end());
  jpeg.insert(jpeg.begin() + static_cast<std::ptrdiff_t>(at), segment.begin(),
              segment.end());
  return jpeg;
}

Bytes Exif(bool little, std::uint16_t orientation) {
  Bytes tiff;
  const auto put16 = [&](std::uint16_t value) {
    if (little) {
      tiff.push_back(static_cast<std::uint8_t>(value));
      tiff.push_back(static_cast<std::uint8_t>(value >> 8));
    } else {
      tiff.push_back(static_cast<std::uint8_t>(value >> 8));
      tiff.push_back(static_cast<std::uint8_t>(value));
    }
  };
  const auto put32 = [&](std::uint32_t value) {
    if (little) {
      put16(static_cast<std::uint16_t>(value));
      put16(static_cast<std::uint16_t>(value >> 16));
    } else {
      put16(static_cast<std::uint16_t>(value >> 16));
      put16(static_cast<std::uint16_t>(value));
    }
  };
  tiff.push_back(little ? 'I' : 'M');
  tiff.push_back(little ? 'I' : 'M');
  put16(42);
  put32(8);
  put16(1);
  put16(0x112);
  put16(3);
  put32(1);
  put16(orientation);
  put16(0);
  put32(0);
  Bytes segment{'E', 'x', 'i', 'f', 0, 0};
  segment.insert(segment.end(), tiff.begin(), tiff.end());
  return segment;
}

// The image as EXIF orientation 1 to 8 displays it.
gufo::core::Image Orient(const gufo::core::Image& image, unsigned orientation) {
  const bool transpose = orientation >= 5;
  gufo::core::Image result{transpose ? image.height : image.width,
                           transpose ? image.width : image.height,
                           Bytes(image.pixels.size())};
  for (std::uint32_t y = 0; y < result.height; ++y) {
    for (std::uint32_t x = 0; x < result.width; ++x) {
      const auto right = image.width - 1, bottom = image.height - 1;
      std::uint32_t sx = x, sy = y;
      switch (orientation) {
        case 2:
          sx = right - x;
          break;
        case 3:
          sx = right - x;
          sy = bottom - y;
          break;
        case 4:
          sy = bottom - y;
          break;
        case 5:
          sx = y;
          sy = x;
          break;
        case 6:
          sx = y;
          sy = bottom - x;
          break;
        case 7:
          sx = right - y;
          sy = bottom - x;
          break;
        case 8:
          sx = right - y;
          sy = x;
          break;
        default:
          break;
      }
      std::copy_n(
          image.pixels.data() + (std::size_t{sy} * image.width + sx) * 3, 3,
          result.pixels.data() + (std::size_t{y} * result.width + x) * 3);
    }
  }
  return result;
}

Outcome Of(const gufo::core::Image& image) {
  return {true, image.width, image.height, Digest(image.pixels)};
}

// Metadata leaves the pixels alone; EXIF orientation transforms them.
void CheckMetadata(Failures& failures) {
  const auto base = ReadFixture("ycc420-33x31");
  gufo::core::Image image;
  (void)Decode(base, &image);
  const auto plain = Of(image);
  Bytes icc(200, 0x41);
  const Bytes icc_header{'I', 'C', 'C', '_', 'P', 'R', 'O',
                         'F', 'I', 'L', 'E', 0,   1,   1};
  icc.insert(icc.begin(), icc_header.begin(), icc_header.end());
  failures.Expect("icc-profile", Decode(WithSegment(base, 0xe2, icc)), plain);
  failures.Expect("comment", Decode(WithSegment(base, 0xfe, Bytes(1000, 'c'))),
                  plain);
  const std::string_view xmp_name = "http://ns.adobe.com/xap/1.0/";
  // The namespace, its terminator and a packet.
  Bytes xmp(xmp_name.size() + 1 + 40, 'x');
  std::ranges::copy(xmp_name, xmp.begin());
  xmp[xmp_name.size()] = 0;
  failures.Expect("xmp", Decode(WithSegment(base, 0xe1, xmp)), plain);

  // Values outside 1 to 8 display the image as stored.
  for (std::uint16_t value = 0; value <= 9; ++value) {
    const auto expected =
        Of(Orient(image, value >= 1 && value <= 8 ? value : 1));
    for (const bool little : {true, false}) {
      failures.Expect(
          "orientation-" + std::to_string(value) + (little ? "-ii" : "-mm"),
          Decode(WithSegment(base, 0xe1, Exif(little, value))), expected);
    }
  }
  // The first Exif segment wins; one too short to hold its entry is ignored.
  failures.Expect("orientation-first-exif",
                  Decode(WithSegment(WithSegment(base, 0xe1, Exif(true, 3)),
                                     0xe1, Exif(true, 6))),
                  Of(Orient(image, 6)));
  auto short_tiff = Exif(true, 6);
  short_tiff.resize(13);
  failures.Expect("orientation-short-exif",
                  Decode(WithSegment(base, 0xe1, short_tiff)), plain);
}

Bytes WithSofSize(Bytes jpeg, unsigned width, unsigned height) {
  const auto sof = Find(jpeg, 0xc0);
  jpeg[sof + 5] = static_cast<std::uint8_t>(height >> 8);
  jpeg[sof + 6] = static_cast<std::uint8_t>(height);
  jpeg[sof + 7] = static_cast<std::uint8_t>(width >> 8);
  jpeg[sof + 8] = static_cast<std::uint8_t>(width);
  return jpeg;
}

void CheckFraming(Failures& failures) {
  const auto jpeg = ReadFixture("ycc420-33x31");
  const auto plain = Decode(jpeg);
  const auto sos = Find(jpeg, 0xda);

  auto trailing = jpeg;
  trailing.insert(trailing.end(), {0x00, 0x11, 0x22, 0xff, 0xd8});
  failures.Expect("trailing-bytes-after-eoi", Decode(trailing), plain);
  failures.Expect("missing-eoi", Decode(std::span(jpeg).first(jpeg.size() - 2)),
                  kInvalid);
  auto fill = jpeg;
  fill.insert(fill.begin() + static_cast<std::ptrdiff_t>(sos), {0xff, 0xff});
  failures.Expect("fill-bytes-before-sos", Decode(fill), plain);
  auto no_soi = jpeg;
  no_soi[1] = 0xd9;
  failures.Expect("bad-soi", Decode(no_soi), kInvalid);
  failures.Expect("sof-zero-width", Decode(WithSofSize(jpeg, 0, 31)), kInvalid);
  failures.Expect("sof-zero-height", Decode(WithSofSize(jpeg, 33, 0)),
                  kInvalid);
  failures.Expect("sof-over-pixel-limit",
                  Decode(WithSofSize(jpeg, 65535, 65535)), kInvalid);
  failures.Expect("empty", Decode({}), kInvalid);

  // A 2-component image: a 3-component YCbCr frame with its last component
  // dropped from SOF. libjpeg-turbo could not convert it to RGB either.
  auto two = ReadFixture("ycc444-17x9");
  const auto sof = Find(two, 0xc0);
  two[sof + 3] = static_cast<std::uint8_t>(two[sof + 3] - 3);
  two[sof + 9] = 2;
  two.erase(two.begin() + static_cast<std::ptrdiff_t>(sof) + 16,
            two.begin() + static_cast<std::ptrdiff_t>(sof) + 19);
  failures.Expect("two-components", Decode(two), kInvalid);

  // Differences from libjpeg-turbo, which rejected all three.
  failures.Expect("sof-wider-than-data (" + std::string(kShortScan) + ")",
                  Decode(WithSofSize(jpeg, 330, 31)),
                  {true, 330, 31, 0x839f191ca7fc9234ULL});
  // A restart interval without RST markers in the data.
  auto dri = jpeg;
  const Bytes dri_segment{0xff, 0xdd, 0x00, 0x04, 0x00, 0x01};
  dri.insert(dri.begin() + static_cast<std::ptrdiff_t>(sos),
             dri_segment.begin(), dri_segment.end());
  failures.Expect("dri-without-rst (" + std::string(kShortScan) + ")",
                  Decode(dri), {true, 33, 31, 0x6cd1a15800359bafULL});
  // A DNL segment that repeats the SOF height after the scan.
  auto dnl = jpeg;
  const Bytes dnl_segment{0xff, 0xdc, 0x00, 0x04, 0x00, 31};
  dnl.insert(dnl.end() - 2, dnl_segment.begin(), dnl_segment.end());
  failures.Expect("dnl-after-scan (" + std::string(kDnl) + ")", Decode(dnl),
                  kInvalid);
}

// One letter per outcome: p for pixels of the fixture's size, x for
// std::invalid_argument, o for pixels of another size.
char Letter(const Outcome& outcome, std::uint32_t width, std::uint32_t height) {
  if (!outcome.decoded)
    return 'x';
  return outcome.width == width && outcome.height == height ? 'p' : 'o';
}

// Truncations and single bit flips of the 64x48 fixtures.
void CheckDamage(Failures& failures) {
  const struct {
    std::string_view name, truncations;
  } truncated[] = {
      {"restart420-64x48", "xxxxxxxxxxx"},
      {"progressive420-64x48", "xxxxxxxxxxx"},
      {"gray-64x48", "xxxxxxxxxxx"},
  };
  for (const auto& fixture : truncated) {
    const auto jpeg = ReadFixture(fixture.name);
    const auto sos = Find(jpeg, 0xda);
    std::string letters;
    for (const auto length :
         {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{20}, sos,
          sos + 4, sos + 20, jpeg.size() / 2, jpeg.size() * 9 / 10,
          jpeg.size() - 3, jpeg.size() - 1}) {
      letters += Letter(Decode(std::span(jpeg).first(length)), 64, 48);
    }
    failures.Check(letters == fixture.truncations,
                   std::string(fixture.name) + " truncations: " + letters +
                       ", expected " + std::string(fixture.truncations));
  }

  // The positions and bits the libjpeg-turbo comparison drew. Flips in
  // entropy-coded data or Huffman tables that libjpeg-turbo warned about may
  // decode.
  const struct {
    std::string_view name, mutations;
  } mutated[] = {
      {"restart420-64x48", "ppxpppppxppppppppppppppppppxpppppppppppppppppppp"},
      {"progressive420-64x48",
       "ppppppxppppppppppppppppxppxppxxpppppppppxppxpxpp"},
  };
  for (const auto& fixture : mutated) {
    const auto jpeg = ReadFixture(fixture.name);
    std::mt19937 random(static_cast<unsigned>(jpeg.size()));
    std::string letters;
    for (int i = 0; i < 48; ++i) {
      auto changed = jpeg;
      const auto position = 2 + random() % (changed.size() - 2);
      changed[position] ^= static_cast<std::uint8_t>(1u << (random() % 8));
      const auto outcome = Decode(changed);
      letters += Letter(outcome, 64, 48);
      if (fixture.name == "restart420-64x48" && i == 7) {
        failures.Check(position == 2183, "mutation 7 moved");
        failures.Expect(
            "restart420-64x48 mutation 7 (" + std::string(kAcOverrun) + ")",
            outcome, {true, 64, 48, 0x7a39d3760cec0400ULL});
      }
    }
    failures.Check(letters == fixture.mutations,
                   std::string(fixture.name) + " mutations: " + letters +
                       ", expected " + std::string(fixture.mutations));
  }
}

int RunCases() {
  Failures failures;
  try {
    CheckDecoded(failures);
    CheckMetadata(failures);
    CheckFraming(failures);
    CheckDamage(failures);
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  if (failures.count() != 0) {
    std::cerr << failures.count() << " checks failed\n";
    return 1;
  }
  std::cout << "JPEG expected results match\n";
  return 0;
}

int RunCorpus(const std::vector<std::filesystem::path>& directories,
              int repeat) {
  std::vector<std::filesystem::path> files;
  for (const auto& directory : directories) {
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(directory)) {
      auto extension = entry.path().extension().string();
      std::transform(extension.begin(), extension.end(), extension.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (entry.is_regular_file() &&
          (extension == ".jpg" || extension == ".jpeg"))
        files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  std::cout << "file\tbytes\toutcome\tmedian_ms\n";
  std::size_t rejected = 0;
  for (const auto& file : files) {
    Bytes jpeg;
    try {
      jpeg = gufo::core::ReadImageFile(file);
    } catch (const std::exception& error) {
      std::cout << file.string() << "\t-\tskipped: " << error.what() << "\t-\n";
      continue;
    }
    std::vector<double> times;
    Outcome outcome;
    for (int i = 0; i < repeat; ++i) {
      const auto start = std::chrono::steady_clock::now();
      outcome = Decode(jpeg);
      const std::chrono::duration<double, std::milli> elapsed =
          std::chrono::steady_clock::now() - start;
      times.push_back(elapsed.count());
    }
    std::sort(times.begin(), times.end());
    rejected += !outcome.decoded;
    std::cout << file.string() << '\t' << jpeg.size() << '\t'
              << Describe(outcome) << '\t' << times[times.size() / 2] << '\n';
  }
  std::cout << "files " << files.size() << ", rejected " << rejected << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1)
    return RunCases();
  if (std::string_view(argv[1]) != "--corpus") {
    std::cerr << "usage: jpeg_decode_test [--corpus DIR... [--repeat N]]\n";
    return 2;
  }
  std::vector<std::filesystem::path> directories;
  int repeat = 5;
  for (int i = 2; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--repeat" && i + 1 < argc) {
      repeat = std::max(1, std::atoi(argv[++i]));
    } else {
      directories.emplace_back(argv[i]);
    }
  }
  return RunCorpus(directories, repeat);
}

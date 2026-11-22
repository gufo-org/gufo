// Compares DecodeImage on JPEG input with the frozen libjpeg-turbo reference
// in jpeg_reference.cpp. The cases are written with libjpeg-turbo's encoder,
// then truncated and mutated deterministically. Pixels may differ within the
// IDCT and upsampling tolerance; any other difference must be listed in
// kKnownDeviations with its reason.
//
//   jpeg_decode_test                          run the cases
//   jpeg_decode_test --corpus DIR... [--repeat N]
//                                             compare and time real files

#include <cstdio>
// jpeglib.h uses FILE but doesn't include <stdio.h>.
#include <jpeglib.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <csetjmp>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <new>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/image.hpp"

namespace gufo::core::libjpeg_reference {
Image DecodeImage(std::span<const std::uint8_t> bytes);
}

namespace {

using Bytes = std::vector<std::uint8_t>;

// zune-jpeg's IDCT and chroma upsampling round differently from libjpeg-turbo's
// islow IDCT and fancy upsampling; the run prints the largest values seen.
// With subsampled chroma, a width or height of 8 mod 16 ends the image inside
// a chroma block, and zune-jpeg upsamples the outermost row or column against
// the block's padding where libjpeg-turbo replicates the last sample. Only
// subsampled images get the border allowance.
constexpr int kMaxDifference = 6;
constexpr int kMaxSubsampledBorderDifference = 40;
constexpr double kMaxMeanDifference = 1.0;

constexpr std::string_view kArithmetic =
    "zune-jpeg does not decode arithmetic coding (SOF9, SOF10); libjpeg-turbo "
    "does";
constexpr std::string_view kLossless =
    "zune-jpeg does not decode lossless JPEG (SOF3); libjpeg-turbo does";
constexpr std::string_view kShortScan =
    "zune-jpeg pads a scan that reaches a marker before its last MCU, or an "
    "interval without its RST marker, with zeros; libjpeg-turbo warns, which "
    "the reference rejects";

constexpr std::string_view kDnl =
    "zune-jpeg's strict mode rejects a DNL marker; libjpeg-turbo skips one "
    "that follows a nonzero SOF height";

constexpr std::string_view kAcOverrun =
    "a flipped bit makes an AC run pass coefficient 63; libjpeg-turbo silently "
    "stores it in coefficient 63, zune-jpeg's slow Huffman path (pos & 63) "
    "wraps it into a low-frequency coefficient";

// What DecodeImage returns instead of the reference: pixels of this size, or
// std::invalid_argument when the size is zero.
struct Deviation {
  std::string_view reason;
  std::uint32_t width{0};
  std::uint32_t height{0};
};

const std::map<std::string_view, Deviation> kKnownDeviations{
    {"arithmetic420", {kArithmetic}},
    {"arithmetic-progressive420", {kArithmetic}},
    {"lossless-rgb", {kLossless}},
    {"dnl-after-scan", {kDnl}},
    {"sof-wider-than-data", {kShortScan, 330, 31}},
    {"dri-without-rst", {kShortScan, 33, 31}},
    {"ycc420-mutation-7-at-2183", {kAcOverrun, 64, 48}},
};

// A bit flip inside entropy-coded data or a Huffman table can make
// libjpeg-turbo warn about a bad Huffman code or bytes left before the next
// marker, which the reference rejects; zune-jpeg may still decode it. When the
// reference decodes, the result must match.
constexpr std::string_view kEntropyMutation =
    "a flipped entropy-coded or Huffman table bit that libjpeg-turbo warns "
    "about may decode";

struct Outcome {
  enum class Kind {
    kPixels,
    kInvalidArgument,
    kRuntimeError,
    kBadAlloc,
    kOther
  };
  Kind kind{Kind::kOther};
  std::uint32_t width{0};
  std::uint32_t height{0};
  Bytes pixels;
  std::string message;
};

std::string_view KindName(Outcome::Kind kind) {
  switch (kind) {
    case Outcome::Kind::kPixels:
      return "pixels";
    case Outcome::Kind::kInvalidArgument:
      return "std::invalid_argument";
    case Outcome::Kind::kRuntimeError:
      return "std::runtime_error";
    case Outcome::Kind::kBadAlloc:
      return "std::bad_alloc";
    case Outcome::Kind::kOther:
      return "other exception";
  }
  return "?";
}

std::string Describe(const Outcome& outcome) {
  std::ostringstream text;
  text << KindName(outcome.kind);
  if (outcome.kind == Outcome::Kind::kPixels) {
    text << ' ' << outcome.width << 'x' << outcome.height;
  } else {
    text << " \"" << outcome.message << '"';
  }
  return text.str();
}

struct PixelDifference {
  int max{0};
  // The largest difference away from the image border.
  int interior_max{0};
  double mean{0};
  std::size_t first{0};
};

// The border is the outermost row and column on each side, which EXIF
// orientation may move from the bottom or right.
PixelDifference Compare(std::uint32_t width, const Bytes& expected,
                        const Bytes& actual) {
  PixelDifference difference;
  const std::size_t height = width == 0 ? 0 : expected.size() / 3 / width;
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const int delta = std::abs(int{expected[i]} - int{actual[i]});
    if (delta > difference.max) {
      difference.max = delta;
      difference.first = i;
    }
    const std::size_t x = i / 3 % width;
    const std::size_t y = i / 3 / width;
    if (x != 0 && x + 1 != width && y != 0 && y + 1 != height)
      difference.interior_max = std::max(difference.interior_max, delta);
    total += static_cast<std::uint64_t>(delta);
  }
  if (!expected.empty())
    difference.mean = static_cast<double>(total) / expected.size();
  return difference;
}

PixelDifference Compare(const Outcome& expected, const Outcome& actual) {
  return Compare(expected.width, expected.pixels, actual.pixels);
}

std::string Difference(const Outcome& expected, const Outcome& actual) {
  std::ostringstream text;
  text << "    reference:   " << Describe(expected)
       << "\n    DecodeImage: " << Describe(actual) << '\n';
  if (expected.kind == Outcome::Kind::kPixels &&
      actual.kind == Outcome::Kind::kPixels &&
      expected.pixels.size() == actual.pixels.size()) {
    const auto difference = Compare(expected, actual);
    text << "    max channel difference " << difference.max << " at byte "
         << difference.first << " (pixel " << difference.first / 3
         << "), interior " << difference.interior_max << ", mean "
         << difference.mean << '\n';
  }
  return text.str();
}

Outcome Capture(const std::function<gufo::core::Image()>& decode) {
  Outcome outcome;
  try {
    auto image = decode();
    outcome.kind = Outcome::Kind::kPixels;
    outcome.width = image.width;
    outcome.height = image.height;
    outcome.pixels = std::move(image.pixels);
  } catch (const std::invalid_argument& error) {
    outcome.kind = Outcome::Kind::kInvalidArgument;
    outcome.message = error.what();
  } catch (const std::runtime_error& error) {
    outcome.kind = Outcome::Kind::kRuntimeError;
    outcome.message = error.what();
  } catch (const std::bad_alloc&) {
    outcome.kind = Outcome::Kind::kBadAlloc;
  } catch (const std::exception& error) {
    outcome.message = error.what();
  }
  return outcome;
}

Outcome DecodeReference(const Bytes& jpeg) {
  return Capture(
      [&] { return gufo::core::libjpeg_reference::DecodeImage(jpeg); });
}

Outcome DecodeCurrent(const Bytes& jpeg) {
  return Capture([&] { return gufo::core::DecodeImage(jpeg); });
}

// Outcomes match when both reject with the same exception type, or both
// decode to the same size within the pixel tolerance. Messages differ by
// design: the Rust decoder reports one "invalid JPEG image".
bool Matches(const Outcome& expected, const Outcome& actual, int border_limit) {
  if (expected.kind != actual.kind)
    return false;
  if (expected.kind != Outcome::Kind::kPixels)
    return true;
  if (expected.width != actual.width || expected.height != actual.height ||
      expected.pixels.size() != actual.pixels.size())
    return false;
  const auto difference = Compare(expected, actual);
  return difference.interior_max <= kMaxDifference &&
         difference.max <= border_limit &&
         difference.mean <= kMaxMeanDifference;
}

// --- Writing JPEGs with libjpeg-turbo -------------------------------------

struct Format {
  std::string name;
  J_COLOR_SPACE input{JCS_RGB};
  J_COLOR_SPACE stored{JCS_YCbCr};
  // Luma (or first component) sampling; the other components use 1x1.
  int h{2};
  int v{2};
  bool progressive{false};
  bool arithmetic{false};
  bool optimize{false};
  bool lossless{false};
  int restart_rows{0};
  int quality{90};
};

int InputComponents(J_COLOR_SPACE space) {
  switch (space) {
    case JCS_GRAYSCALE:
      return 1;
    case JCS_CMYK:
      return 4;
    default:
      return 3;
  }
}

struct Segment {
  int marker;
  Bytes data;
};

// Everything libjpeg modifies lives on the heap, so its values stay defined
// after its error handler longjmps back.
struct Writer {
  // First, so the error manager's address is the Writer's.
  jpeg_error_mgr manager{};
  jpeg_compress_struct encoder{};
  std::jmp_buf jump;
  unsigned char* buffer{nullptr};
  unsigned long size{0};
  bool created{false};
  std::vector<std::uint8_t> row;
  std::mt19937 noise;
  ~Writer() {
    if (created)
      jpeg_destroy_compress(&encoder);
    std::free(buffer);
  }
};

std::uint8_t Sample(unsigned x, unsigned y, unsigned c, std::mt19937& noise) {
  // Smooth gradients with edges and a little noise exercise both the DC and
  // AC paths and chroma upsampling.
  const unsigned base = (x * (3 + c) + y * (5 + 2 * c) + c * 71) & 0xff;
  const unsigned edge = ((x / 5 + y / 3 + c) % 3 == 0) ? 64 : 0;
  return static_cast<std::uint8_t>((base + edge + noise() % 9) & 0xff);
}

// Writes the JPEG into writer. libjpeg-turbo may longjmp out of it, so it
// keeps every object with a destructor in writer, and its caller holds no
// locals across the setjmp.
void Encode(Writer& writer, const Format& format, unsigned width,
            unsigned height, const std::vector<Segment>& segments) {
  auto& encoder = writer.encoder;
  auto& row = writer.row;
  auto& noise = writer.noise;
  const int channels = InputComponents(format.input);
  row.resize(std::size_t{width} * channels);
  noise.seed(width * 131 + height);
  jpeg_create_compress(&encoder);
  writer.created = true;
  jpeg_mem_dest(&encoder, &writer.buffer, &writer.size);
  encoder.image_width = width;
  encoder.image_height = height;
  encoder.input_components = channels;
  encoder.in_color_space = format.input;
  jpeg_set_defaults(&encoder);
  jpeg_set_colorspace(&encoder, format.stored);
  jpeg_set_quality(&encoder, format.quality, TRUE);
  if (format.lossless)
    jpeg_enable_lossless(&encoder, 1, 0);
  if (!format.lossless && encoder.num_components > 1) {
    encoder.comp_info[0].h_samp_factor = format.h;
    encoder.comp_info[0].v_samp_factor = format.v;
    for (int i = 1; i < encoder.num_components; ++i) {
      encoder.comp_info[i].h_samp_factor = 1;
      encoder.comp_info[i].v_samp_factor = 1;
    }
  }
  if (format.progressive)
    jpeg_simple_progression(&encoder);
  encoder.arith_code = format.arithmetic ? TRUE : FALSE;
  encoder.optimize_coding = format.optimize ? TRUE : FALSE;
  encoder.restart_in_rows = format.restart_rows;
  jpeg_start_compress(&encoder, TRUE);
  for (const auto& segment : segments) {
    jpeg_write_marker(&encoder, segment.marker, segment.data.data(),
                      static_cast<unsigned>(segment.data.size()));
  }
  while (encoder.next_scanline < encoder.image_height) {
    for (unsigned x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        row[std::size_t{x} * channels + c] =
            Sample(x, encoder.next_scanline, static_cast<unsigned>(c), noise);
      }
    }
    JSAMPROW pointer = row.data();
    jpeg_write_scanlines(&encoder, &pointer, 1);
  }
  jpeg_finish_compress(&encoder);
}

// Returns an empty buffer when this libjpeg-turbo cannot write the format.
Bytes WriteJpeg(const Format& format, unsigned width, unsigned height,
                const std::vector<Segment>& segments = {}) {
  const auto writer = std::make_unique<Writer>();
  writer->encoder.err = jpeg_std_error(&writer->manager);
  writer->manager.error_exit = [](j_common_ptr common) {
    std::longjmp(reinterpret_cast<Writer*>(common->err)->jump, 1);
  };
  writer->manager.output_message = [](j_common_ptr) {};
  if (setjmp(writer->jump) != 0)
    return {};
  Encode(*writer, format, width, height, segments);
  return Bytes(writer->buffer, writer->buffer + writer->size);
}

// --- Cases ----------------------------------------------------------------

struct Case {
  std::string name;
  Bytes jpeg;
  // Generated by the encoder unchanged; the reference must decode it.
  bool valid;
  // A mutation inside entropy-coded data or a DHT; see kEntropyMutation.
  bool entropy_mutation{false};
};

class Cases {
public:
  void Valid(std::string name, Bytes jpeg) {
    Add(std::move(name), std::move(jpeg), true);
  }
  void Mutated(std::string name, Bytes jpeg, bool entropy = false) {
    Add(std::move(name), std::move(jpeg), false);
    cases_.back().entropy_mutation = entropy;
  }
  const std::vector<Case>& all() const { return cases_; }
  std::vector<std::string> unsupported;

private:
  void Add(std::string name, Bytes jpeg, bool valid) {
    if (!names_.insert(name).second)
      throw std::logic_error("duplicate case " + name);
    cases_.push_back({std::move(name), std::move(jpeg), valid});
  }
  std::vector<Case> cases_;
  std::set<std::string> names_;
};

std::vector<Format> Formats() {
  std::vector<Format> formats{
      {.name = "gray", .input = JCS_GRAYSCALE, .stored = JCS_GRAYSCALE},
      {.name = "ycc444", .h = 1, .v = 1},
      {.name = "ycc422", .h = 2, .v = 1},
      {.name = "ycc420", .h = 2, .v = 2},
      {.name = "ycc440", .h = 1, .v = 2},
      {.name = "ycc411", .h = 4, .v = 1},
      {.name = "ycc410", .h = 4, .v = 2},
      {.name = "rgb", .stored = JCS_RGB, .h = 1, .v = 1},
      {.name = "cmyk", .input = JCS_CMYK, .stored = JCS_CMYK, .h = 1, .v = 1},
      {.name = "ycck", .input = JCS_CMYK, .stored = JCS_YCCK},
      {.name = "progressive-gray",
       .input = JCS_GRAYSCALE,
       .stored = JCS_GRAYSCALE,
       .progressive = true},
      {.name = "progressive444", .h = 1, .v = 1, .progressive = true},
      {.name = "progressive420", .progressive = true},
      {.name = "progressive422", .h = 2, .v = 1, .progressive = true},
      {.name = "progressive-cmyk",
       .input = JCS_CMYK,
       .stored = JCS_CMYK,
       .h = 1,
       .v = 1,
       .progressive = true},
      {.name = "optimized420", .optimize = true},
      {.name = "restart420", .restart_rows = 1},
      {.name = "restart444", .h = 1, .v = 1, .restart_rows = 2},
      {.name = "progressive-restart420",
       .progressive = true,
       .restart_rows = 1},
      {.name = "quality100-420", .quality = 100},
      {.name = "quality100-444", .h = 1, .v = 1, .quality = 100},
      {.name = "quality20-420", .quality = 20},
      {.name = "quality1-420", .quality = 1},
  };
  return formats;
}

constexpr std::array<std::array<unsigned, 2>, 15> kSizes{{{1, 1},
                                                          {24, 24},
                                                          {40, 16},
                                                          {8, 16},
                                                          {16, 8},
                                                          {1080, 16},
                                                          {24, 8},
                                                          {7, 5},
                                                          {8, 8},
                                                          {16, 16},
                                                          {17, 9},
                                                          {33, 31},
                                                          {64, 48},
                                                          {129, 65},
                                                          {250, 3}}};

std::string SizeName(unsigned width, unsigned height) {
  return std::to_string(width) + "x" + std::to_string(height);
}

// Codings the Rust decoder rejects, one size each; see kKnownDeviations.
void AddUnsupportedCases(Cases& cases) {
  for (const auto& format : {
           Format{.name = "arithmetic420", .arithmetic = true},
           Format{.name = "arithmetic-progressive420",
                  .progressive = true,
                  .arithmetic = true},
           Format{.name = "lossless-rgb", .stored = JCS_RGB, .lossless = true},
       }) {
    auto jpeg = WriteJpeg(format, 33, 31);
    if (jpeg.empty()) {
      cases.unsupported.push_back(format.name);
      continue;
    }
    cases.Valid(format.name, std::move(jpeg));
  }
  // libjpeg-turbo cannot convert lossless gray to RGB, so both reject it.
  cases.Mutated("lossless-gray", WriteJpeg({.name = "lossless-gray",
                                            .input = JCS_GRAYSCALE,
                                            .stored = JCS_GRAYSCALE,
                                            .lossless = true},
                                           33, 31));
}

void AddFormatCases(Cases& cases) {
  for (const auto& format : Formats()) {
    for (const auto& [width, height] : kSizes) {
      auto jpeg = WriteJpeg(format, width, height);
      const auto name = format.name + "-" + SizeName(width, height);
      if (jpeg.empty()) {
        cases.unsupported.push_back(name);
        continue;
      }
      cases.Valid(name, std::move(jpeg));
    }
  }
}

// Writes a 2-component image by declaring a 3-component YCbCr image and
// dropping its last component from SOF and SOS. libjpeg cannot convert it to
// RGB.
Bytes TwoComponents() {
  const Format format{.name = "two", .h = 1, .v = 1};
  auto jpeg = WriteJpeg(format, 8, 8);
  for (std::size_t i = 2; i + 4 < jpeg.size(); ++i) {
    if (jpeg[i] == 0xff && jpeg[i + 1] == 0xc0) {
      jpeg[i + 3] = static_cast<std::uint8_t>(jpeg[i + 3] - 3);
      jpeg[i + 9] = 2;
      jpeg.erase(jpeg.begin() + static_cast<std::ptrdiff_t>(i) + 16,
                 jpeg.begin() + static_cast<std::ptrdiff_t>(i) + 19);
      break;
    }
  }
  return jpeg;
}

std::size_t Find(const Bytes& jpeg, std::uint8_t marker) {
  for (std::size_t i = 2; i + 1 < jpeg.size(); ++i) {
    if (jpeg[i] == 0xff && jpeg[i + 1] == marker)
      return i;
  }
  throw std::logic_error("marker not found");
}

Bytes WithSofSize(Bytes jpeg, unsigned width, unsigned height) {
  const auto sof = Find(jpeg, 0xc0);
  jpeg[sof + 5] = static_cast<std::uint8_t>(height >> 8);
  jpeg[sof + 6] = static_cast<std::uint8_t>(height);
  jpeg[sof + 7] = static_cast<std::uint8_t>(width >> 8);
  jpeg[sof + 8] = static_cast<std::uint8_t>(width);
  return jpeg;
}

void AddFramingCases(Cases& cases) {
  const Format base{.name = "ycc420"};
  const auto jpeg = WriteJpeg(base, 33, 31);
  const auto sos = Find(jpeg, 0xda);

  auto trailing = jpeg;
  trailing.insert(trailing.end(), {0x00, 0x11, 0x22, 0xff, 0xd8});
  cases.Mutated("trailing-bytes-after-eoi", trailing);

  auto no_eoi = jpeg;
  no_eoi.resize(no_eoi.size() - 2);
  cases.Mutated("missing-eoi", no_eoi);

  auto fill = jpeg;
  fill.insert(fill.begin() + static_cast<std::ptrdiff_t>(sos), {0xff, 0xff});
  cases.Mutated("fill-bytes-before-sos", fill);

  auto no_soi = jpeg;
  no_soi[1] = 0xd9;
  cases.Mutated("bad-soi", no_soi);

  cases.Mutated("two-components", TwoComponents());
  cases.Mutated("sof-zero-width", WithSofSize(jpeg, 0, 31));
  cases.Mutated("sof-zero-height", WithSofSize(jpeg, 33, 0));
  cases.Mutated("sof-over-pixel-limit", WithSofSize(jpeg, 65535, 65535));
  cases.Mutated("sof-wider-than-data", WithSofSize(jpeg, 330, 31));

  // libjpeg accepts DRI and RST markers it can resynchronize; a restart
  // interval without RST markers in the data is corrupt.
  auto dri = jpeg;
  const Bytes dri_segment{0xff, 0xdd, 0x00, 0x04, 0x00, 0x01};
  dri.insert(dri.begin() + static_cast<std::ptrdiff_t>(sos),
             dri_segment.begin(), dri_segment.end());
  cases.Mutated("dri-without-rst", dri);

  // A DNL segment that repeats the SOF height after the scan.
  auto dnl = jpeg;
  const Bytes dnl_segment{0xff, 0xdc, 0x00, 0x04, 0x00, 31};
  dnl.insert(dnl.end() - 2, dnl_segment.begin(), dnl_segment.end());
  cases.Mutated("dnl-after-scan", dnl);
}

void AddMetadataCases(Cases& cases) {
  const Format base{.name = "ycc420"};
  Bytes icc(200, 0x41);
  const Bytes icc_header{'I', 'C', 'C', '_', 'P', 'R', 'O',
                         'F', 'I', 'L', 'E', 0,   1,   1};
  icc.insert(icc.begin(), icc_header.begin(), icc_header.end());
  cases.Valid("icc-profile", WriteJpeg(base, 17, 9, {{JPEG_APP0 + 2, icc}}));
  cases.Valid("comment",
              WriteJpeg(base, 17, 9, {{JPEG_COM, Bytes(1000, 'c')}}));
  cases.Valid("xmp",
              WriteJpeg(base, 17, 9, {{JPEG_APP0 + 1, [] {
                                         const std::string text =
                                             "http://ns.adobe.com/xap/1.0/";
                                         Bytes data(text.begin(), text.end());
                                         data.push_back(0);
                                         data.resize(data.size() + 40, 'x');
                                         return data;
                                       }()}}));
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

void AddOrientationCases(Cases& cases) {
  const Format base{.name = "ycc420"};
  for (std::uint16_t value = 0; value <= 9; ++value) {
    for (const bool little : {true, false}) {
      cases.Valid(
          "orientation-" + std::to_string(value) + (little ? "-ii" : "-mm"),
          WriteJpeg(base, 17, 9, {{JPEG_APP0 + 1, Exif(little, value)}}));
    }
  }
  // The first Exif segment wins.
  cases.Valid("orientation-first-exif",
              WriteJpeg(base, 17, 9,
                        {{JPEG_APP0 + 1, Exif(true, 6)},
                         {JPEG_APP0 + 1, Exif(true, 3)}}));
  auto short_tiff = Exif(true, 6);
  short_tiff.resize(13);
  cases.Valid("orientation-short-exif",
              WriteJpeg(base, 17, 9, {{JPEG_APP0 + 1, short_tiff}}));
}

void AddTruncationCases(Cases& cases) {
  for (const auto& format :
       {Format{.name = "ycc420", .restart_rows = 1},
        Format{.name = "progressive420", .progressive = true},
        Format{
            .name = "gray", .input = JCS_GRAYSCALE, .stored = JCS_GRAYSCALE}}) {
    const auto jpeg = WriteJpeg(format, 64, 48);
    const auto sos = Find(jpeg, 0xda);
    const std::vector<std::size_t> lengths{1,
                                           2,
                                           3,
                                           20,
                                           sos,
                                           sos + 4,
                                           sos + 20,
                                           jpeg.size() / 2,
                                           jpeg.size() * 9 / 10,
                                           jpeg.size() - 3,
                                           jpeg.size() - 1};
    for (const auto length : lengths) {
      Bytes truncated(jpeg.begin(),
                      jpeg.begin() + static_cast<std::ptrdiff_t>(length));
      cases.Mutated(format.name + "-truncated-" + std::to_string(length),
                    std::move(truncated));
    }
  }
}

// Returns whether each byte is entropy-coded data (including stuffed zeros
// and RST markers) or a DHT payload, from the segment structure the encoder
// wrote.
std::vector<bool> EntropyBytes(const Bytes& jpeg) {
  std::vector<bool> entropy(jpeg.size(), false);
  std::size_t offset = 2;
  while (offset + 3 < jpeg.size()) {
    const auto marker = jpeg[offset + 1];
    if (marker == 0xd9)
      break;
    const std::size_t length =
        (std::size_t{jpeg[offset + 2]} << 8) | jpeg[offset + 3];
    if (marker == 0xc4) {
      for (std::size_t i = offset + 4; i < offset + 2 + length; ++i)
        entropy[i] = true;
    }
    offset += 2 + length;
    if (marker != 0xda)
      continue;
    while (offset + 1 < jpeg.size() &&
           !(jpeg[offset] == 0xff && jpeg[offset + 1] != 0x00 &&
             (jpeg[offset + 1] < 0xd0 || jpeg[offset + 1] > 0xd7))) {
      entropy[offset++] = true;
    }
  }
  return entropy;
}

void AddMutationCases(Cases& cases) {
  for (const auto& format :
       {Format{.name = "ycc420", .restart_rows = 1},
        Format{.name = "progressive420", .progressive = true}}) {
    const auto jpeg = WriteJpeg(format, 64, 48);
    const auto entropy = EntropyBytes(jpeg);
    std::mt19937 random(static_cast<unsigned>(jpeg.size()));
    for (int i = 0; i < 48; ++i) {
      auto mutated = jpeg;
      const auto position = 2 + random() % (mutated.size() - 2);
      mutated[position] ^= static_cast<std::uint8_t>(1u << (random() % 8));
      cases.Mutated(format.name + "-mutation-" + std::to_string(i) + "-at-" +
                        std::to_string(position),
                    std::move(mutated), entropy[position]);
    }
  }
}

Cases BuildCases() {
  Cases cases;
  AddFormatCases(cases);
  AddUnsupportedCases(cases);
  AddFramingCases(cases);
  AddMetadataCases(cases);
  AddOrientationCases(cases);
  AddTruncationCases(cases);
  AddMutationCases(cases);
  return cases;
}

class Failures {
public:
  void Add(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    ++count_;
  }
  int count() const { return count_; }

private:
  int count_{0};
};

struct Worst {
  int interior{0};
  std::string interior_case;
  int max{0};
  double mean{0};
  std::string max_case;
  std::string mean_case;

  void Add(const std::string& name, const Outcome& expected,
           const Outcome& actual) {
    if (expected.kind != Outcome::Kind::kPixels ||
        actual.kind != Outcome::Kind::kPixels ||
        expected.pixels.size() != actual.pixels.size())
      return;
    const auto difference = Compare(expected, actual);
    if (difference.interior_max > interior) {
      interior = difference.interior_max;
      interior_case = name;
    }
    if (difference.max > max) {
      max = difference.max;
      max_case = name;
    }
    if (difference.mean > mean) {
      mean = difference.mean;
      mean_case = name;
    }
  }
};

// Whether the frame header samples its components at different rates.
bool Subsampled(const Bytes& jpeg) {
  for (std::size_t i = 2; i + 10 < jpeg.size(); ++i) {
    if (jpeg[i] != 0xff || jpeg[i + 1] < 0xc0 || jpeg[i + 1] > 0xc2)
      continue;
    const std::size_t components = jpeg[i + 9];
    if (i + 10 + components * 3 > jpeg.size())
      return false;
    for (std::size_t c = 1; c < components; ++c) {
      if (jpeg[i + 11 + c * 3] != jpeg[i + 11])
        return true;
    }
    return false;
  }
  return false;
}

int BorderLimit(const Bytes& jpeg) {
  return Subsampled(jpeg) ? kMaxSubsampledBorderDifference : kMaxDifference;
}

bool IsListedOutcome(const Deviation& deviation, const Outcome& actual) {
  if (deviation.width == 0)
    return actual.kind == Outcome::Kind::kInvalidArgument;
  return actual.kind == Outcome::Kind::kPixels &&
         actual.width == deviation.width && actual.height == deviation.height;
}

void CheckCase(const Case& test, Failures& failures, Worst& worst) {
  const auto expected = DecodeReference(test.jpeg);
  const auto actual = DecodeCurrent(test.jpeg);
  if (test.valid && expected.kind != Outcome::Kind::kPixels) {
    failures.Add(test.name +
                 ": the reference rejected a generated valid image\n" +
                 Difference(expected, actual));
    return;
  }
  if (test.valid)
    worst.Add(test.name, expected, actual);
  if (test.entropy_mutation &&
      expected.kind == Outcome::Kind::kInvalidArgument) {
    if (actual.kind != Outcome::Kind::kPixels &&
        actual.kind != Outcome::Kind::kInvalidArgument) {
      failures.Add(test.name + ": " + std::string(kEntropyMutation) +
                   ", or be rejected as invalid\n" +
                   Difference(expected, actual));
    }
    return;
  }
  const bool deviates = !Matches(expected, actual, BorderLimit(test.jpeg));
  const auto listed = kKnownDeviations.find(test.name);
  if (listed == kKnownDeviations.end()) {
    if (deviates) {
      failures.Add(test.name + ": DecodeImage differs from the reference\n" +
                   Difference(expected, actual));
    }
  } else if (!deviates) {
    failures.Add(
        test.name +
        " is in kKnownDeviations but matches the reference; remove the row");
  } else if (!IsListedOutcome(listed->second, actual)) {
    failures.Add(test.name +
                 ": DecodeImage differs from its kKnownDeviations "
                 "row\n" +
                 Difference(expected, actual));
  }
}

int RunCases() {
  const auto cases = BuildCases();
  Failures failures;
  Worst worst;
  std::set<std::string_view> names;
  for (const auto& test : cases.all()) {
    names.insert(test.name);
    CheckCase(test, failures, worst);
  }
  for (const auto& [name, reason] : kKnownDeviations) {
    if (!names.contains(name))
      failures.Add("kKnownDeviations names unknown case " + std::string(name));
  }
  for (const auto& name : cases.unsupported)
    std::cout << "skipped " << name << ": the encoder lacks the format\n";
  std::cout << "largest interior difference " << worst.interior << " ("
            << worst.interior_case << ")\n";
  std::cout << "largest channel difference " << worst.max << " ("
            << worst.max_case << "), largest mean " << worst.mean << " ("
            << worst.mean_case << ")\n";
  if (failures.count() != 0) {
    std::cerr << failures.count() << " of " << cases.all().size()
              << " cases failed\n";
    return 1;
  }
  std::cout << cases.all().size() << " cases match the reference, "
            << kKnownDeviations.size() << " known deviations\n";
  return 0;
}

struct Timing {
  double reference_ms;
  double decode_image_ms;
};

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

Timing MedianTimings(const Bytes& jpeg, int repeat) {
  const auto time = [&](auto&& decode) {
    const auto start = std::chrono::steady_clock::now();
    (void)decode(jpeg);
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - start;
    return elapsed.count();
  };
  std::vector<double> reference_times;
  std::vector<double> decode_image_times;
  for (int i = 0; i < repeat; ++i) {
    reference_times.push_back(time(DecodeReference));
    decode_image_times.push_back(time(DecodeCurrent));
  }
  return {Median(reference_times), Median(decode_image_times)};
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
  std::cout << "file\tbytes\tstatus\tmax_diff\tmean_diff\treference_ms\t"
               "decode_image_ms\treference_outcome\n";
  std::size_t matched = 0;
  std::size_t mismatched = 0;
  std::size_t skipped = 0;
  for (const auto& file : files) {
    Bytes jpeg;
    try {
      jpeg = gufo::core::ReadImageFile(file);
    } catch (const std::exception& error) {
      std::cout << file.string() << "\t-\tskipped\t-\t-\t-\t-\t" << error.what()
                << '\n';
      ++skipped;
      continue;
    }
    const auto expected = DecodeReference(jpeg);
    const auto actual = DecodeCurrent(jpeg);
    const bool same = Matches(expected, actual, BorderLimit(jpeg));
    PixelDifference difference;
    if (expected.kind == Outcome::Kind::kPixels &&
        actual.kind == Outcome::Kind::kPixels &&
        expected.pixels.size() == actual.pixels.size())
      difference = Compare(expected, actual);
    if (same) {
      ++matched;
    } else {
      ++mismatched;
      std::cerr << "MISMATCH: " << file.string() << '\n'
                << Difference(expected, actual);
    }
    const auto [reference_ms, decode_image_ms] = MedianTimings(jpeg, repeat);
    std::cout << file.string() << '\t' << jpeg.size() << '\t'
              << (same ? "match" : "MISMATCH") << '\t' << difference.max << '\t'
              << difference.mean << '\t' << reference_ms << '\t'
              << decode_image_ms << '\t' << Describe(expected) << '\n';
  }
  std::cout << "files " << files.size() << ", match " << matched
            << ", mismatches " << mismatched << ", skipped " << skipped << '\n';
  return mismatched == 0 ? 0 : 1;
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

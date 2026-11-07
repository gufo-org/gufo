#include <png.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <new>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/image.hpp"

namespace gufo::core::libpng_reference {
Image DecodeImage(std::span<const std::uint8_t> bytes);
}

namespace {

using Bytes = std::vector<std::uint8_t>;

constexpr std::string_view kNoAdler =
    "libpng rejects a zlib stream without its Adler-32 trailer; the png crate "
    "stops reading once the last row is complete";
constexpr std::string_view kChunkName =
    "libpng rejects a chunk type with a non-letter byte; the png crate "
    "treats it as an unknown chunk";
constexpr std::string_view kIend =
    "libpng ignores a bad IEND CRC or length; the png crate rejects it";
constexpr std::string_view kStrayIdat =
    "libpng accepts IDAT after another chunk once the image data is complete; "
    "the png crate rejects it";

const std::map<std::string_view, std::string_view> kKnownDeviations{
    {"missing-adler", kNoAdler},
    {"big-missing-adler", kNoAdler},
    {"bad-chunk-name", kChunkName},
    {"palette4-trns-mutation-32", kChunkName},
    {"palette4-trns-mutation-53", kChunkName},
    {"palette4-trns-mutation-72", kChunkName},
    {"iend-bad-crc", kIend},
    {"iend-nonempty", kIend},
    {"stray-idat-after-text", kStrayIdat},
};

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

  bool operator==(const Outcome&) const = default;
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

std::string Difference(const Outcome& expected, const Outcome& actual) {
  std::ostringstream text;
  text << "    reference:   " << Describe(expected)
       << "\n    DecodeImage: " << Describe(actual) << '\n';
  if (expected.kind == Outcome::Kind::kPixels &&
      actual.kind == Outcome::Kind::kPixels &&
      expected.pixels.size() == actual.pixels.size()) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < expected.pixels.size(); ++i) {
      if (expected.pixels[i] == actual.pixels[i])
        continue;
      if (++count == 1) {
        text << "    first differing byte " << i << " (pixel " << i / 3
             << ", channel " << i % 3 << "): reference "
             << static_cast<int>(expected.pixels[i]) << ", DecodeImage "
             << static_cast<int>(actual.pixels[i]) << '\n';
      }
    }
    text << "    differing bytes: " << count << " of " << expected.pixels.size()
         << '\n';
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

Outcome DecodeReference(const Bytes& png) {
  return Capture(
      [&] { return gufo::core::libpng_reference::DecodeImage(png); });
}

Outcome DecodeCurrent(const Bytes& png) {
  return Capture([&] { return gufo::core::DecodeImage(png); });
}

struct Chunk {
  std::string type;
  Bytes data;
  std::optional<std::uint32_t> crc;
};

constexpr std::array<std::uint8_t, 8> kSignature{0x89, 'P',  'N',  'G',
                                                 '\r', '\n', 0x1a, '\n'};

void AppendBe32(Bytes& out, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void AppendBe16(Bytes& out, std::uint32_t value) {
  out.push_back(static_cast<std::uint8_t>(value >> 8));
  out.push_back(static_cast<std::uint8_t>(value));
}

std::uint32_t ReadBe32(const std::uint8_t* p) {
  return (std::uint32_t{p[0]} << 24) | (std::uint32_t{p[1]} << 16) |
         (std::uint32_t{p[2]} << 8) | p[3];
}

std::uint32_t ChunkCrc(std::string_view type, const Bytes& data) {
  auto crc = crc32(0, reinterpret_cast<const Bytef*>(type.data()),
                   static_cast<uInt>(type.size()));
  // crc32 treats a null buffer as a reset, which an empty vector's data() is.
  if (!data.empty())
    crc = crc32(crc, data.data(), static_cast<uInt>(data.size()));
  return static_cast<std::uint32_t>(crc);
}

Bytes Serialize(const std::vector<Chunk>& chunks) {
  Bytes out(kSignature.begin(), kSignature.end());
  for (const auto& chunk : chunks) {
    AppendBe32(out, static_cast<std::uint32_t>(chunk.data.size()));
    out.insert(out.end(), chunk.type.begin(), chunk.type.end());
    out.insert(out.end(), chunk.data.begin(), chunk.data.end());
    AppendBe32(out, chunk.crc.value_or(ChunkCrc(chunk.type, chunk.data)));
  }
  return out;
}

std::vector<Chunk> Parse(const Bytes& png) {
  std::vector<Chunk> chunks;
  std::size_t offset = kSignature.size();
  while (offset + 12 <= png.size()) {
    const std::size_t length = ReadBe32(&png[offset]);
    if (length > png.size() - offset - 12) {
      std::cerr << "Parse: fixture chunk overruns the buffer\n";
      std::abort();
    }
    chunks.push_back(
        {std::string(reinterpret_cast<const char*>(&png[offset + 4]), 4),
         Bytes(png.begin() + offset + 8, png.begin() + offset + 8 + length),
         std::nullopt});
    offset += length + 12;
  }
  return chunks;
}

Bytes Compress(const Bytes& data, int level = Z_DEFAULT_COMPRESSION) {
  uLongf size = compressBound(static_cast<uLong>(data.size()));
  Bytes out(size);
  if (compress2(out.data(), &size, data.data(), static_cast<uLong>(data.size()),
                level) != Z_OK) {
    std::cerr << "compress2 failed\n";
    std::abort();
  }
  out.resize(size);
  return out;
}

Bytes Zeros(std::size_t size) {
  return Compress(Bytes(size, 0), Z_BEST_SPEED);
}

Bytes Text(std::string_view text) {
  return Bytes(text.begin(), text.end());
}

constexpr std::string_view kKeyValue{"k\0v", 3};

Chunk Make(std::string type, Bytes data = {}) {
  return {std::move(type), std::move(data), std::nullopt};
}

Chunk WithCrc(Chunk chunk, std::uint32_t crc) {
  chunk.crc = crc;
  return chunk;
}

Chunk Ihdr(std::uint32_t width, std::uint32_t height, int depth = 8,
           int color = 2, int interlace = 0, int compression = 0,
           int filter = 0) {
  Bytes data;
  AppendBe32(data, width);
  AppendBe32(data, height);
  for (const int field : {depth, color, compression, filter, interlace})
    data.push_back(static_cast<std::uint8_t>(field));
  return Make("IHDR", std::move(data));
}

Chunk Actl(std::uint32_t frames) {
  Bytes data;
  AppendBe32(data, frames);
  AppendBe32(data, 0);
  return Make("acTL", std::move(data));
}

Chunk Fctl(std::uint32_t sequence, std::uint32_t width, std::uint32_t height,
           std::uint32_t x, std::uint32_t y) {
  Bytes data;
  for (const auto value : {sequence, width, height, x, y})
    AppendBe32(data, value);
  AppendBe16(data, 1);
  AppendBe16(data, 10);
  data.push_back(0);
  data.push_back(0);
  return Make("fcTL", std::move(data));
}

// An fcTL with one byte changed: offset 24 is dispose_op, 25 blend_op.
Chunk FctlWith(std::size_t offset, std::uint8_t value) {
  Chunk chunk = Fctl(1, 8, 8, 0, 0);
  chunk.data[offset] = value;
  return chunk;
}

Chunk Fdat(std::uint32_t sequence, const Bytes& zlib_stream) {
  Bytes data;
  AppendBe32(data, sequence);
  data.insert(data.end(), zlib_stream.begin(), zlib_stream.end());
  return Make("fdAT", std::move(data));
}

using Fill = std::function<std::uint8_t(unsigned x, unsigned y, unsigned c)>;

std::uint8_t DefaultFill(unsigned x, unsigned y, unsigned c) {
  return static_cast<std::uint8_t>(x * 7 + y * 13 + c * 31);
}

Bytes RawRows(unsigned width, unsigned height, unsigned bytes_per_pixel,
              const Fill& fill = DefaultFill) {
  Bytes out;
  out.reserve((std::size_t{width} * bytes_per_pixel + 1) * height);
  for (unsigned y = 0; y < height; ++y) {
    out.push_back(0);
    for (unsigned x = 0; x < width; ++x) {
      for (unsigned c = 0; c < bytes_per_pixel; ++c)
        out.push_back(fill(x, y, c));
    }
  }
  return out;
}

std::vector<Chunk> Rgb8Chunks(Bytes idat, std::vector<Chunk> before = {},
                              std::vector<Chunk> after = {},
                              unsigned size = 8) {
  std::vector<Chunk> chunks{Ihdr(size, size)};
  chunks.insert(chunks.end(), before.begin(), before.end());
  chunks.push_back(Make("IDAT", std::move(idat)));
  chunks.insert(chunks.end(), after.begin(), after.end());
  chunks.push_back(Make("IEND"));
  return chunks;
}

Bytes GoodIdat() {
  return Compress(RawRows(8, 8, 3));
}

struct Format {
  int color;
  int depth;
};

constexpr std::array<Format, 15> kFormats{{
    {PNG_COLOR_TYPE_GRAY, 1},
    {PNG_COLOR_TYPE_GRAY, 2},
    {PNG_COLOR_TYPE_GRAY, 4},
    {PNG_COLOR_TYPE_GRAY, 8},
    {PNG_COLOR_TYPE_GRAY, 16},
    {PNG_COLOR_TYPE_PALETTE, 1},
    {PNG_COLOR_TYPE_PALETTE, 2},
    {PNG_COLOR_TYPE_PALETTE, 4},
    {PNG_COLOR_TYPE_PALETTE, 8},
    {PNG_COLOR_TYPE_GRAY_ALPHA, 8},
    {PNG_COLOR_TYPE_GRAY_ALPHA, 16},
    {PNG_COLOR_TYPE_RGB, 8},
    {PNG_COLOR_TYPE_RGB, 16},
    {PNG_COLOR_TYPE_RGB_ALPHA, 8},
    {PNG_COLOR_TYPE_RGB_ALPHA, 16},
}};

std::string_view ColorName(int color) {
  switch (color) {
    case PNG_COLOR_TYPE_GRAY:
      return "gray";
    case PNG_COLOR_TYPE_PALETTE:
      return "palette";
    case PNG_COLOR_TYPE_GRAY_ALPHA:
      return "graya";
    case PNG_COLOR_TYPE_RGB:
      return "rgb";
    default:
      return "rgba";
  }
}

unsigned Channels(int color) {
  switch (color) {
    case PNG_COLOR_TYPE_GRAY_ALPHA:
      return 2;
    case PNG_COLOR_TYPE_RGB:
      return 3;
    case PNG_COLOR_TYPE_RGB_ALPHA:
      return 4;
    default:
      return 1;
  }
}

std::size_t RowBytes(const Format& format, unsigned width) {
  return (std::size_t{width} * Channels(format.color) * format.depth + 7) / 8;
}

bool AllowsTrns(int color) {
  return color == PNG_COLOR_TYPE_GRAY || color == PNG_COLOR_TYPE_RGB ||
         color == PNG_COLOR_TYPE_PALETTE;
}

Bytes RandomRows(const Format& format, unsigned width, unsigned height) {
  std::mt19937 rng(width * 131 + height * 17 + format.color * 5 + format.depth);
  Bytes rows(RowBytes(format, width) * height);
  for (auto& byte : rows)
    byte = static_cast<std::uint8_t>(rng());
  return rows;
}

[[noreturn]] void LibpngFailure(png_structp, png_const_charp message) {
  std::cerr << "libpng fixture writer: " << message << '\n';
  std::abort();
}

// kStored writes uncompressed deflate blocks and filter type 0, so the bytes
// do not depend on the zlib version.
enum class Encoding { kAdaptive, kStored };

Bytes WritePng(const Format& format, unsigned width, unsigned height,
               const Bytes& rows, bool interlace, bool trns,
               Encoding encoding = Encoding::kAdaptive) {
  Bytes out;
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr,
                                            LibpngFailure, nullptr);
  png_infop info = png_create_info_struct(png);
  png_set_write_fn(
      png, &out,
      [](png_structp p, png_bytep data, size_t size) {
        auto& sink = *static_cast<Bytes*>(png_get_io_ptr(p));
        sink.insert(sink.end(), data, data + size);
      },
      [](png_structp) {});
  png_set_IHDR(png, info, width, height, format.depth, format.color,
               interlace ? PNG_INTERLACE_ADAM7 : PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  std::array<png_color, 256> palette{};
  std::array<png_byte, 256> alpha{};
  if (format.color == PNG_COLOR_TYPE_PALETTE) {
    const int entries = 1 << format.depth;
    for (int i = 0; i < entries; ++i) {
      palette[i] = {static_cast<png_byte>(i * 37),
                    static_cast<png_byte>(i * 91),
                    static_cast<png_byte>(255 - i)};
      alpha[i] = static_cast<png_byte>(i * 53);
    }
    png_set_PLTE(png, info, palette.data(), entries);
    if (trns)
      png_set_tRNS(png, info, alpha.data(), std::min(entries, 5), nullptr);
  } else if (trns) {
    png_color_16 key{};
    const auto sample = [&](std::size_t index) -> png_uint_16 {
      if (format.depth == 16)
        return static_cast<png_uint_16>((rows[index * 2] << 8) |
                                        rows[index * 2 + 1]);
      if (format.depth == 8)
        return rows[index];
      return rows[0] >> (8 - format.depth);
    };
    key.gray = key.red = sample(0);
    if (format.color == PNG_COLOR_TYPE_RGB) {
      key.green = sample(1);
      key.blue = sample(2);
    }
    png_set_tRNS(png, info, nullptr, 0, &key);
  }
  if (encoding == Encoding::kStored) {
    png_set_compression_level(png, 0);
    png_set_filter(png, PNG_FILTER_TYPE_BASE, PNG_FILTER_NONE);
  } else {
    png_set_filter(png, PNG_FILTER_TYPE_BASE, PNG_ALL_FILTERS);
  }
  png_write_info(png, info);
  const int passes = png_set_interlace_handling(png);
  const auto stride = RowBytes(format, width);
  for (int pass = 0; pass < passes; ++pass) {
    for (unsigned y = 0; y < height; ++y)
      png_write_row(png, rows.data() + y * stride);
  }
  png_write_end(png, info);
  png_destroy_write_struct(&png, &info);
  return out;
}

struct Case {
  std::string name;
  Bytes png;
  bool valid;
};

class Cases {
public:
  void AddValid(std::string name, Bytes png) {
    cases_.push_back({std::move(name), std::move(png), true});
  }
  void Add(std::string name, Bytes png) {
    cases_.push_back({std::move(name), std::move(png), false});
  }
  void Add(std::string name, const std::vector<Chunk>& chunks) {
    Add(std::move(name), Serialize(chunks));
  }
  const std::vector<Case>& all() const { return cases_; }

private:
  std::vector<Case> cases_;
};

void AddFormatCases(Cases& cases) {
  constexpr std::array<std::pair<unsigned, unsigned>, 5> kSizes{
      {{1, 1}, {1, 9}, {9, 1}, {17, 9}, {33, 65}}};
  for (const auto& format : kFormats) {
    for (const bool trns : {false, true}) {
      if (trns && !AllowsTrns(format.color))
        continue;
      for (const bool interlace : {false, true}) {
        for (const auto& [width, height] : kSizes) {
          std::ostringstream name;
          name << ColorName(format.color) << format.depth
               << (trns ? "-trns" : "") << (interlace ? "-adam7" : "") << '-'
               << width << 'x' << height;
          cases.AddValid(name.str(), WritePng(format, width, height,
                                              RandomRows(format, width, height),
                                              interlace, trns));
        }
      }
    }
  }
}

void AddGray16Cases(Cases& cases) {
  const Format format{PNG_COLOR_TYPE_GRAY, 16};
  const std::array<unsigned, 8> samples{0,    255,   256, 257,
                                        1000, 65535, 254, 512};
  Bytes rows;
  for (const auto sample : samples) {
    rows.push_back(static_cast<std::uint8_t>(sample >> 8));
    rows.push_back(static_cast<std::uint8_t>(sample));
  }
  for (const bool interlace : {false, true}) {
    cases.AddValid(interlace ? "gray16-saturation-adam7" : "gray16-saturation",
                   WritePng(format, 8, 1, rows, interlace, false));
  }
}

const Bytes& BigIdat() {
  static const Bytes idat = [] {
    std::mt19937 rng(7);
    Bytes rows;
    for (unsigned y = 0; y < 64; ++y) {
      rows.push_back(0);
      for (unsigned i = 0; i < 64 * 3; ++i)
        rows.push_back(static_cast<std::uint8_t>(rng()));
    }
    return Compress(rows, 6);
  }();
  return idat;
}

std::vector<Chunk> SplitIdat(const Bytes& stream, std::size_t piece) {
  std::vector<Chunk> chunks;
  for (std::size_t i = 0; i < stream.size(); i += piece) {
    chunks.push_back(Make(
        "IDAT", Bytes(stream.begin() + i,
                      stream.begin() + std::min(stream.size(), i + piece))));
  }
  return chunks;
}

Bytes FlipLast4(Bytes stream) {
  for (std::size_t i = stream.size() - 4; i < stream.size(); ++i)
    stream[i] ^= 0xff;
  return stream;
}

std::vector<Chunk> BigChunks(std::vector<Chunk> idats) {
  std::vector<Chunk> chunks{Ihdr(64, 64)};
  chunks.insert(chunks.end(), idats.begin(), idats.end());
  chunks.push_back(Make("IEND"));
  return chunks;
}

void AddAxisLimitCases(Cases& cases) {
  const Format format{PNG_COLOR_TYPE_RGB_ALPHA, 16};
  constexpr unsigned kWidth = 1'000'000;
  const auto rows = [&](unsigned height) {
    Bytes bytes(RowBytes(format, kWidth) * height);
    for (std::size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<std::uint8_t>(i % 251 + i / 4096);
    return bytes;
  };
  cases.AddValid("rgba16-wide-1000000x1",
                 WritePng(format, kWidth, 1, rows(1), false, false));
  cases.AddValid("rgba16-wide-1000000x2-adam7",
                 WritePng(format, kWidth, 2, rows(2), true, false));
}

void AddZlibAndFramingCases(Cases& cases) {
  const Bytes good = GoodIdat();
  const Bytes without_trailer(good.begin(), good.end() - 4);
  const Bytes big = BigIdat();

  cases.AddValid("baseline-rgb8", Serialize(Rgb8Chunks(good)));
  cases.Add("bad-adler", Serialize(Rgb8Chunks(FlipLast4(good))));
  cases.Add("missing-adler", Serialize(Rgb8Chunks(without_trailer)));
  cases.Add("extra-zlib-after-end", Serialize(Rgb8Chunks([&] {
              Bytes extra = good;
              extra.insert(extra.end(), 4, 0);
              return extra;
            }())));
  cases.Add(
      "extra-idat-after-end",
      std::vector<Chunk>{Ihdr(8, 8), Make("IDAT", good),
                         Make("IDAT", Compress(Bytes(64, 0))), Make("IEND")});
  cases.Add("too-much-image-data",
            Serialize(Rgb8Chunks(Compress(RawRows(8, 9, 3)))));
  cases.Add("not-enough-image-data",
            Serialize(Rgb8Chunks(Compress(RawRows(8, 7, 3)))));
  cases.Add(
      "empty-idat-split",
      std::vector<Chunk>{
          Ihdr(8, 8), Make("IDAT", Bytes(good.begin(), good.begin() + 10)),
          Make("IDAT"), Make("IDAT", Bytes(good.begin() + 10, good.end())),
          Make("IEND")});
  {
    Bytes trailing = Serialize(Rgb8Chunks(good));
    const auto text = Text("garbage after IEND");
    trailing.insert(trailing.end(), text.begin(), text.end());
    cases.Add("trailing-after-iend", std::move(trailing));
  }
  {
    auto chunks = Rgb8Chunks(good);
    chunks.pop_back();
    cases.Add("no-iend", chunks);
  }
  cases.AddValid("big-baseline", Serialize(BigChunks(SplitIdat(big, 256))));
  cases.Add("big-bad-adler", BigChunks(SplitIdat(FlipLast4(big), 256)));
  {
    auto idats = SplitIdat(Bytes(big.begin(), big.end() - 4), 256);
    auto own = idats;
    own.push_back(Make("IDAT", Bytes(big.end() - 4, big.end())));
    for (auto& byte : own.back().data)
      byte ^= 0xff;
    cases.Add("big-bad-adler-own-idat", BigChunks(own));
    cases.Add("big-missing-adler", BigChunks(idats));
  }
  cases.Add("stray-idat-after-text",
            std::vector<Chunk>{
                Ihdr(8, 8), Make("IDAT", good), Make("tEXt", Text(kKeyValue)),
                Make("IDAT", Compress(Bytes(64, 0))), Make("IEND")});
  cases.Add(
      "split-idat-run",
      std::vector<Chunk>{
          Ihdr(8, 8), Make("IDAT", Bytes(good.begin(), good.begin() + 20)),
          Make("tEXt", Text(kKeyValue)),
          Make("IDAT", Bytes(good.begin() + 20, good.end())), Make("IEND")});
  cases.Add("bad-filter-type", Serialize(Rgb8Chunks(Compress([] {
              Bytes rows;
              for (int y = 0; y < 8; ++y) {
                const auto row = RawRows(8, 1, 3);
                rows.push_back(5);
                rows.insert(rows.end(), row.begin() + 1, row.end());
              }
              return rows;
            }()))));
  cases.Add("zlib-bad-header", [&] {
    Bytes stream = good;
    stream[1] = 0x00;
    return Serialize(Rgb8Chunks(stream));
  }());
  cases.Add("zlib-preset-dict", [&] {
    Bytes stream{0x78, 0xbb, 0, 0, 0, 1};
    stream.insert(stream.end(), good.begin() + 2, good.end());
    return Serialize(Rgb8Chunks(stream));
  }());
  cases.Add("zlib-window-64k", [&] {
    const unsigned cmf = 0x88;
    const unsigned flg = 0x9c ^ ((cmf * 256 + 0x9c) % 31);
    Bytes stream{static_cast<std::uint8_t>(cmf),
                 static_cast<std::uint8_t>(flg)};
    stream.insert(stream.end(), good.begin() + 2, good.end());
    return Serialize(Rgb8Chunks(stream));
  }());
}

void AddDimensionCases(Cases& cases) {
  const auto gray = [](unsigned width, unsigned height) {
    return std::vector<Chunk>{
        Ihdr(width, height, 8, 0),
        Make("IDAT", Compress(RawRows(
                         width, height, 1,
                         [](auto, auto, auto) { return std::uint8_t{0x5a}; }))),
        Make("IEND")};
  };
  cases.Add("wide-1000000", gray(1'000'000, 1));
  cases.Add("wide-1000001", gray(1'000'001, 1));
  cases.Add("tall-1000001", gray(1, 1'000'001));
  const Bytes good = GoodIdat();
  cases.Add("ihdr-2pow31",
            std::vector<Chunk>{Ihdr(1U << 31, 1, 8, 0), Make("IDAT", good),
                               Make("IEND")});
  cases.Add("zero-width",
            std::vector<Chunk>{Ihdr(0, 8), Make("IDAT", good), Make("IEND")});
  cases.Add("over-pixel-limit",
            std::vector<Chunk>{Ihdr(30'000, 30'000), Make("IDAT", good),
                               Make("IEND")});
  cases.Add("over-pixel-limit-corrupt-chunk",
            std::vector<Chunk>{Ihdr(30'000, 30'000), Make("ABCD", Text("xyz")),
                               Make("IDAT", good), Make("IEND")});
  cases.Add(
      "over-pixel-limit-short-idat",
      std::vector<Chunk>{Ihdr(30'000, 30'000), Make("tEXt", Text(kKeyValue)),
                         Make("IDAT", Bytes{1, 2, 3}), Make("IEND")});
  cases.Add(
      "ihdr-bad-depth",
      std::vector<Chunk>{Ihdr(8, 8, 3, 2), Make("IDAT", good), Make("IEND")});
  cases.Add("ihdr-bad-interlace",
            std::vector<Chunk>{Ihdr(8, 8, 8, 2, 2), Make("IDAT", good),
                               Make("IEND")});
  cases.Add("ihdr-bad-compression",
            std::vector<Chunk>{Ihdr(8, 8, 8, 2, 0, 1), Make("IDAT", good),
                               Make("IEND")});
  cases.Add("ihdr-bad-filter-method",
            std::vector<Chunk>{Ihdr(8, 8, 8, 2, 0, 0, 1), Make("IDAT", good),
                               Make("IEND")});
  {
    auto ihdr = Ihdr(8, 8);
    ihdr.data.push_back(0);
    cases.Add("ihdr-long",
              std::vector<Chunk>{ihdr, Make("IDAT", good), Make("IEND")});
  }
  cases.Add("ihdr-not-first", [&] {
    auto chunks = Rgb8Chunks(good);
    chunks.insert(chunks.begin(), Make("tEXt", Text(kKeyValue)));
    return chunks;
  }());
  cases.Add("two-ihdr", std::vector<Chunk>{Ihdr(8, 8), Ihdr(8, 8),
                                           Make("IDAT", good), Make("IEND")});
}

void AddAncillaryCases(Cases& cases) {
  const Bytes good = GoodIdat();
  const auto with_before = [&](Chunk chunk) {
    return Rgb8Chunks(good, {std::move(chunk)});
  };
  const auto with_after = [&](Chunk chunk) {
    return Rgb8Chunks(good, {}, {std::move(chunk)});
  };
  Bytes gama_zero;
  AppendBe32(gama_zero, 0);
  Bytes gama;
  AppendBe32(gama, 45455);

  Bytes iccp = Text("profile");
  iccp.push_back(0);
  iccp.push_back(0);
  const Bytes nine_mib = Zeros(9 << 20);
  iccp.insert(iccp.end(), nine_mib.begin(), nine_mib.end());
  cases.Add("iccp-9mib", with_before(Make("iCCP", iccp)));
  cases.Add(
      "iccp-garbage",
      with_before(Make("iCCP", Text(std::string("profile\0\0not zlib", 17)))));
  Bytes ztxt = Text("Comment");
  ztxt.push_back(0);
  ztxt.push_back(0);
  ztxt.insert(ztxt.end(), nine_mib.begin(), nine_mib.end());
  cases.Add("ztxt-9mib", with_before(Make("zTXt", ztxt)));
  cases.Add(
      "ztxt-garbage",
      with_before(Make("zTXt", Text(std::string("Comment\0\0not zlib", 17)))));
  Bytes text = Text("Comment");
  text.push_back(0);
  text.insert(text.end(), 9 << 20, 'a');
  cases.Add("text-9mib", with_before(Make("tEXt", text)));

  cases.Add("gama-zero", with_before(Make("gAMA", gama_zero)));
  cases.Add("gama-short", with_before(Make("gAMA", Bytes{0, 1})));
  cases.Add("srgb-bad-intent", with_before(Make("sRGB", Bytes{9})));
  cases.Add("chrm-short", with_before(Make("cHRM", Bytes(7, 0))));
  cases.Add("sbit-zero", with_before(Make("sBIT", Bytes(3, 0))));
  cases.Add("phys-short", with_before(Make("pHYs", Bytes(3, 0))));
  cases.Add("time-bad", with_before(Make("tIME", Bytes(7, 0))));
  cases.Add("dup-gama",
            Rgb8Chunks(good, {Make("gAMA", gama), Make("gAMA", gama)}));
  cases.Add("gama-after-idat", with_after(Make("gAMA", gama)));
  cases.Add("unknown-ancillary", with_before(Make("abCd", Text("xyz"))));
  cases.Add("unknown-critical", with_before(Make("ABCD", Text("xyz"))));
  cases.Add("unknown-critical-after-idat",
            with_after(Make("ABCD", Text("xyz"))));
  cases.Add("bad-chunk-name", with_before(Make("a#c!", Text("xyz"))));
  cases.Add("ancillary-bad-crc",
            with_before(WithCrc(Make("tEXt", Text(kKeyValue)), 0)));
  cases.Add("ancillary-bad-crc-after-idat",
            with_after(WithCrc(Make("tEXt", Text(kKeyValue)), 0)));
  cases.Add("critical-bad-crc", [&] {
    auto chunks = Rgb8Chunks(good);
    chunks[1].crc = 0;
    return chunks;
  }());
  {
    auto chunks = Rgb8Chunks(good);
    chunks.back() = WithCrc(Make("IEND"), 0);
    cases.Add("iend-bad-crc", chunks);
    chunks.back() = Make("IEND", Text("x"));
    cases.Add("iend-nonempty", chunks);
  }
  cases.Add("bkgd-trns-rgb",
            Rgb8Chunks(good, {Make("bKGD", Bytes{0, 1, 0, 2, 0, 3}),
                              Make("tRNS", Bytes{0, 1, 0, 2, 0, 3})}));
  cases.Add("trns-rgb-short", with_before(Make("tRNS", Bytes{0, 1})));
  {
    Bytes plte(48);
    for (int i = 0; i < 48; ++i)
      plte[i] = static_cast<std::uint8_t>(i);
    cases.Add("plte-on-rgb", with_before(Make("PLTE", plte)));
    cases.Add("plte-on-gray",
              std::vector<Chunk>{Ihdr(8, 8, 8, 0), Make("PLTE", plte),
                                 Make("IDAT", Compress(RawRows(8, 8, 1))),
                                 Make("IEND")});
  }
  cases.Add("chunk-len-huge", [&] {
    Bytes bytes = Serialize({Ihdr(8, 8)});
    AppendBe32(bytes, 0x7fffffff);
    for (const char c : std::string_view("tEXtk\0v", 7))
      bytes.push_back(static_cast<std::uint8_t>(c));
    const auto rest = Serialize({Make("IDAT", good), Make("IEND")});
    bytes.insert(bytes.end(), rest.begin() + kSignature.size(), rest.end());
    return bytes;
  }());
  cases.Add("exif-bad",
            with_before(Make("eXIf", Bytes{0xff, 0xff, ' ', 'g', 'a'})));
  const Bytes exif = Text(std::string("MM\0*\0\0\0\x08\0\0", 10));
  cases.Add("exif-two",
            Rgb8Chunks(good, {Make("eXIf", exif), Make("eXIf", exif)}));
}

void AddPaletteCases(Cases& cases) {
  const auto zeros = Compress(
      RawRows(8, 8, 1, [](auto, auto, auto) { return std::uint8_t{0}; }));
  const auto indexed = [&](Bytes plte, Bytes idat,
                           std::vector<Chunk> middle = {}, int depth = 8) {
    std::vector<Chunk> chunks{Ihdr(8, 8, depth, 3)};
    if (!plte.empty())
      chunks.push_back(Make("PLTE", std::move(plte)));
    chunks.insert(chunks.end(), middle.begin(), middle.end());
    chunks.push_back(Make("IDAT", std::move(idat)));
    chunks.push_back(Make("IEND"));
    return chunks;
  };
  const auto range = [](int count) {
    Bytes bytes(count);
    for (int i = 0; i < count; ++i)
      bytes[i] = static_cast<std::uint8_t>(i);
    return bytes;
  };
  cases.Add(
      "palette-oob-index",
      indexed(range(6), Compress(RawRows(8, 8, 1, [](auto x, auto y, auto) {
                return static_cast<std::uint8_t>((x + 8 * y) * 4);
              }))));
  cases.Add(
      "palette-trns-long",
      indexed(range(6),
              Compress(RawRows(8, 8, 1,
                               [](auto x, auto y, auto) {
                                 return static_cast<std::uint8_t>((x + y) & 1);
                               })),
              {Make("tRNS", Bytes(10, 0x80))}));
  cases.Add("palette-257", indexed(Bytes(771, 3), zeros));
  cases.Add("palette-not-mult3", indexed(range(7), zeros));
  cases.Add("palette-missing", indexed({}, zeros));
  cases.Add("palette-1bit-oob",
            std::vector<Chunk>{
                Ihdr(8, 8, 1, 3), Make("PLTE", Bytes(3, 0)),
                Make("IDAT", Compress(RawRows(1, 8, 1,
                                              [](auto, auto, auto) {
                                                return std::uint8_t{0xaa};
                                              }))),
                Make("IEND")});
  cases.Add(
      "palette-4bit-oob",
      std::vector<Chunk>{
          Ihdr(8, 8, 4, 3), Make("PLTE", range(6)),
          Make("IDAT", Compress(RawRows(4, 8, 1,
                                        [](auto x, auto y, auto) {
                                          return static_cast<std::uint8_t>(
                                              x * 16 + y + 9);
                                        }))),
          Make("IEND")});
  cases.Add("idat-before-plte",
            std::vector<Chunk>{Ihdr(8, 8, 8, 3), Make("IDAT", zeros),
                               Make("PLTE", Bytes(6, 0)), Make("IEND")});
}

void AddApngCases(Cases& cases) {
  const Bytes good = GoodIdat();
  const Bytes frame = Compress(
      RawRows(8, 8, 3, [](auto, auto, auto) { return std::uint8_t{200}; }));
  cases.Add("apng-default-in-animation",
            std::vector<Chunk>{Ihdr(8, 8), Actl(1), Fctl(0, 8, 8, 0, 0),
                               Make("IDAT", good), Make("IEND")});
  cases.Add(
      "apng-default-separate",
      std::vector<Chunk>{Ihdr(8, 8), Actl(1), Make("IDAT", good),
                         Fctl(0, 8, 8, 0, 0), Fdat(1, frame), Make("IEND")});
  cases.Add(
      "apng-two-frames",
      std::vector<Chunk>{Ihdr(8, 8), Actl(2), Fctl(0, 8, 8, 0, 0),
                         Make("IDAT", good), Fctl(1, 4, 4, 2, 2),
                         Fdat(2, Compress(RawRows(4, 4, 3))), Make("IEND")});
  cases.Add("apng-bad-fdat-zlib",
            std::vector<Chunk>{Ihdr(8, 8), Actl(1), Make("IDAT", good),
                               Fctl(0, 8, 8, 0, 0),
                               Fdat(1, Text("not zlib at all")), Make("IEND")});
  cases.Add("apng-fdat-bad-crc",
            std::vector<Chunk>{Ihdr(8, 8), Actl(1), Make("IDAT", good),
                               Fctl(0, 8, 8, 0, 0), WithCrc(Fdat(1, frame), 0),
                               Make("IEND")});
  cases.Add("apng-first-fctl-narrow",
            std::vector<Chunk>{Ihdr(8, 8), Actl(1), Fctl(0, 7, 8, 0, 0),
                               Make("IDAT", good), Make("IEND")});
  cases.Add("apng-first-fctl-offset",
            std::vector<Chunk>{Ihdr(8, 8), Actl(1), Fctl(0, 4, 4, 2, 2),
                               Make("IDAT", good), Make("IEND")});
  cases.Add("apng-fctl-out-of-bounds",
            std::vector<Chunk>{Ihdr(8, 8), Actl(1), Fctl(0, 8, 8, 2, 2),
                               Make("IDAT", good), Make("IEND")});
  cases.Add("apng-later-fctl-out-of-bounds",
            std::vector<Chunk>{Ihdr(8, 8), Actl(2), Fctl(0, 8, 8, 0, 0),
                               Make("IDAT", good), Fctl(1, 8, 8, 4, 4),
                               Fdat(2, frame), Make("IEND")});
  // Malformed animation chunks after a complete default image. libpng skips
  // them all, with or without acTL.
  const auto after_image = [&](std::vector<Chunk> animation, bool actl) {
    std::vector<Chunk> chunks{Ihdr(8, 8)};
    if (actl)
      chunks.push_back(Actl(2));
    chunks.push_back(Make("IDAT", good));
    chunks.insert(chunks.end(), animation.begin(), animation.end());
    chunks.push_back(Make("IEND"));
    return chunks;
  };
  cases.Add("apng-fctl-bad-sequence",
            after_image({Fctl(7, 8, 8, 0, 0), Fdat(8, frame)}, true));
  cases.Add("apng-fctl-short",
            after_image({Make("fcTL", Bytes(25, 0)), Fdat(2, frame)}, true));
  cases.Add("apng-fctl-bad-dispose",
            after_image({FctlWith(24, 9), Fdat(2, frame)}, true));
  cases.Add("apng-fctl-bad-blend",
            after_image({FctlWith(25, 9), Fdat(2, frame)}, true));
  cases.Add("apng-fdat-without-fctl", after_image({Fdat(1, frame)}, true));
  cases.Add("apng-fdat-bad-sequence",
            after_image({Fctl(1, 8, 8, 0, 0), Fdat(5, frame)}, true));
  cases.Add(
      "apng-fdat-short",
      after_image({Fctl(1, 8, 8, 0, 0), Make("fdAT", Bytes(3, 0))}, true));
  cases.Add("apng-fctl-without-actl",
            after_image({Fctl(7, 8, 8, 4, 4), Fdat(8, frame)}, false));
  cases.Add("apng-actl-after-idat", after_image({Actl(1)}, false));
}

void AddOrientationCases(Cases& cases) {
  const Format format{PNG_COLOR_TYPE_RGB, 8};
  const auto plain =
      Parse(WritePng(format, 3, 2, RandomRows(format, 3, 2), false, false));
  for (std::uint8_t orientation = 1; orientation <= 8; ++orientation) {
    auto chunks = plain;
    // One IFD entry: tag 0x0112, type SHORT, count 1, value `orientation`.
    chunks.insert(chunks.begin() + 1,
                  Make("eXIf", Bytes{'M', 'M',         0,    '*', 0, 0, 0, 8, 0,
                                     1,   0x01,        0x12, 0,   3, 0, 0, 0, 1,
                                     0,   orientation, 0,    0,   0, 0, 0, 0}));
    cases.AddValid("exif-orientation-" + std::to_string(orientation),
                   Serialize(chunks));
  }
}

void RecomputeChunkCrcs(Bytes& png) {
  std::size_t position = kSignature.size();
  while (position + 12 <= png.size()) {
    const std::size_t length = ReadBe32(&png[position]);
    if (length > png.size() - position - 12)
      break;
    Bytes data(png.begin() + position + 8, png.begin() + position + 8 + length);
    const std::string type(reinterpret_cast<const char*>(&png[position + 4]),
                           4);
    const auto crc = ChunkCrc(type, data);
    for (int b = 0; b < 4; ++b)
      png[position + 8 + length + b] =
          static_cast<std::uint8_t>(crc >> (24 - 8 * b));
    position += length + 12;
  }
}

void AddMutationCases(Cases& cases) {
  const std::array<std::pair<std::string_view, Bytes>, 3> seeds{{
      {"rgb8", WritePng({PNG_COLOR_TYPE_RGB, 8}, 9, 5,
                        RandomRows({PNG_COLOR_TYPE_RGB, 8}, 9, 5), false, false,
                        Encoding::kStored)},
      {"rgba8-adam7", WritePng({PNG_COLOR_TYPE_RGB_ALPHA, 8}, 9, 5,
                               RandomRows({PNG_COLOR_TYPE_RGB_ALPHA, 8}, 9, 5),
                               true, false, Encoding::kStored)},
      {"palette4-trns", WritePng({PNG_COLOR_TYPE_PALETTE, 4}, 9, 5,
                                 RandomRows({PNG_COLOR_TYPE_PALETTE, 4}, 9, 5),
                                 false, true, Encoding::kStored)},
  }};
  std::mt19937 rng(20251031);
  for (const auto& [name, seed] : seeds) {
    for (std::size_t length = kSignature.size(); length < seed.size();
         ++length) {
      cases.Add(std::string(name) + "-truncated-" + std::to_string(length),
                Bytes(seed.begin(), seed.begin() + length));
    }
    for (int i = 0; i < 150; ++i) {
      Bytes mutated = seed;
      const auto offset =
          kSignature.size() + rng() % (seed.size() - kSignature.size());
      mutated[offset] = static_cast<std::uint8_t>(rng());
      RecomputeChunkCrcs(mutated);
      cases.Add(std::string(name) + "-mutation-" + std::to_string(i),
                std::move(mutated));
    }
  }
}

Cases BuildCases() {
  Cases cases;
  AddFormatCases(cases);
  AddGray16Cases(cases);
  AddAxisLimitCases(cases);
  AddZlibAndFramingCases(cases);
  AddDimensionCases(cases);
  AddAncillaryCases(cases);
  AddPaletteCases(cases);
  AddApngCases(cases);
  AddOrientationCases(cases);
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

void CheckCase(const std::string& name, const Bytes& png, bool valid,
               Failures& failures) {
  const auto expected = DecodeReference(png);
  const auto actual = DecodeCurrent(png);
  if (valid && expected.kind != Outcome::Kind::kPixels) {
    failures.Add(name + ": the reference rejected a generated valid image\n" +
                 Difference(expected, actual));
    return;
  }
  const bool deviates = expected != actual;
  const bool listed = kKnownDeviations.contains(name);
  if (valid && listed) {
    failures.Add(name + " is valid, so it cannot be in kKnownDeviations");
    return;
  }
  if (deviates && !listed) {
    failures.Add(name + ": DecodeImage differs from the reference\n" +
                 Difference(expected, actual));
  } else if (!deviates && listed) {
    failures.Add(
        name +
        " is in kKnownDeviations but matches the reference; remove the row");
  }
}

int RunCases() {
  const auto cases = BuildCases();
  Failures failures;
  std::set<std::string_view> names;
  for (const auto& test : cases.all()) {
    names.insert(test.name);
    CheckCase(test.name, test.png, test.valid, failures);
  }
  for (const auto& [name, reason] : kKnownDeviations) {
    if (!names.contains(name))
      failures.Add("kKnownDeviations names unknown case " + std::string(name));
  }
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

Timing MedianTimings(const Bytes& png, int repeat) {
  const auto time = [&](auto&& decode) {
    const auto start = std::chrono::steady_clock::now();
    (void)decode(png);
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
      if (entry.is_regular_file() && extension == ".png")
        files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  std::cout << "file\tbytes\tstatus\treference_ms\tdecode_image_ms\treference_"
               "outcome\n";
  std::size_t matched = 0;
  std::size_t mismatched = 0;
  std::size_t skipped = 0;
  for (const auto& file : files) {
    Bytes png;
    try {
      png = gufo::core::ReadImageFile(file);
    } catch (const std::exception& error) {
      std::cout << file.string() << "\t-\tskipped\t-\t-\t" << error.what()
                << '\n';
      ++skipped;
      continue;
    }
    const auto expected = DecodeReference(png);
    const auto actual = DecodeCurrent(png);
    const bool same = expected == actual;
    if (same) {
      ++matched;
    } else {
      ++mismatched;
      std::cerr << "MISMATCH: " << file.string() << '\n'
                << Difference(expected, actual);
    }
    const auto [reference_ms, decode_image_ms] = MedianTimings(png, repeat);
    std::cout << file.string() << '\t' << png.size() << '\t'
              << (same ? "match" : "MISMATCH") << '\t' << reference_ms << '\t'
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
    std::cerr << "usage: png_decode_test [--corpus DIR... [--repeat N]]\n";
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

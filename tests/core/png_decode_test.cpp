// Expected pixels and rejections at the C++ image caller; libpng only writes
// fixtures.
#include <png.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/image.hpp"
namespace {
using Bytes = std::vector<std::uint8_t>;
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

void Require(bool condition) {
  if (!condition)
    throw std::runtime_error("PNG expected-result check failed");
}

void Formats() {
  for (const auto format : kFormats) {
    for (const bool interlace : {false, true}) {
      for (const bool trns : {false, true}) {
        if (trns && !AllowsTrns(format.color))
          continue;
        constexpr unsigned width = 9, height = 7;
        const auto rows = RandomRows(format, width, height);
        const auto image = gufo::core::DecodeImage(
            WritePng(format, width, height, rows, interlace, trns));
        Require(image.width == width && image.height == height);
        Bytes expected;
        for (unsigned y = 0; y < height; ++y) {
          for (unsigned x = 0; x < width; ++x) {
            const auto sample = [&](unsigned c) -> unsigned {
              const auto index = x * Channels(format.color) + c;
              const auto offset = y * RowBytes(format, width);
              if (format.depth < 8)
                return (rows[offset + index * format.depth / 8] >>
                        (8 - format.depth - index * format.depth % 8)) &
                       ((1 << format.depth) - 1);
              if (format.depth == 16)
                return (rows[offset + index * 2] << 8) |
                       rows[offset + index * 2 + 1];
              return rows[offset + index];
            };
            if (format.color == PNG_COLOR_TYPE_PALETTE) {
              const auto index = sample(0);
              expected.push_back(static_cast<std::uint8_t>(index * 37));
              expected.push_back(static_cast<std::uint8_t>(index * 91));
              expected.push_back(static_cast<std::uint8_t>(255 - index));
            } else {
              for (unsigned c = 0; c < 3; ++c) {
                const auto value =
                    sample(format.color == PNG_COLOR_TYPE_RGB ||
                                   format.color == PNG_COLOR_TYPE_RGB_ALPHA
                               ? c
                               : 0);
                // Gray16 keeps libpng's saturating conversion; RGB16 strips the
                // low byte.
                const auto scaled =
                    format.depth < 8 ? value * 255 / ((1 << format.depth) - 1)
                    : format.depth == 16 ? (format.color == PNG_COLOR_TYPE_GRAY
                                                ? std::min(value, 255U)
                                                : value >> 8)
                                         : value;
                expected.push_back(static_cast<std::uint8_t>(scaled));
              }
            }
          }
        }
        if (image.pixels != expected) {
          std::cerr << "color " << format.color << " depth " << format.depth
                    << " interlace " << interlace << " trns " << trns << '\n';
          for (std::size_t i = 0; i < expected.size(); ++i)
            if (image.pixels[i] != expected[i]) {
              std::cerr << "byte " << i << " expected " << unsigned(expected[i])
                        << " actual " << unsigned(image.pixels[i]) << '\n';
              break;
            }
          Require(false);
        }
      }
    }
  }
}

void Framing() {
  const auto check = [](std::string_view name, const Bytes& png,
                        bool accepted) {
    try {
      const auto image = gufo::core::DecodeImage(png);
      if (!accepted || image.width != 8 || image.height != 8)
        throw std::runtime_error(std::string(name));
      Bytes expected;
      for (unsigned y = 0; y < 8; ++y)
        for (unsigned x = 0; x < 8; ++x)
          for (unsigned c = 0; c < 3; ++c)
            expected.push_back(DefaultFill(x, y, c));
      Require(image.pixels == expected);
    } catch (const std::invalid_argument& error) {
      if (accepted || std::string_view(error.what()) != "invalid PNG image")
        throw;
    }
  };
  auto stream = GoodIdat();
  check("valid", Serialize(Rgb8Chunks(stream)), true);
  stream.resize(stream.size() - 4);
  check("missing Adler trailer", Serialize(Rgb8Chunks(stream)), true);
  check("unknown non-letter chunk",
        Serialize(Rgb8Chunks(GoodIdat(), {Make("ab1d")})), true);
  auto chunks = Rgb8Chunks(GoodIdat());
  chunks.back() = WithCrc(Make("IEND"), 0);
  check("IEND CRC", Serialize(chunks), false);
  chunks.back() = Make("IEND", {0});
  check("nonempty IEND", Serialize(chunks), false);
  chunks = Rgb8Chunks(GoodIdat(), {},
                      {Make("tEXt", Text(kKeyValue)), Make("IDAT", {0})});
  check("stray IDAT after text", Serialize(chunks), false);
  chunks = Rgb8Chunks(GoodIdat());
  chunks[1] = WithCrc(chunks[1], 0);
  check("IDAT CRC", Serialize(chunks), false);
  check("truncated PNG", Bytes(kSignature.begin(), kSignature.end()), false);
  // The default image survives malformed animation metadata, including bad CRC.
  check(
      "APNG metadata ignored",
      Serialize(Rgb8Chunks(GoodIdat(), {Make("acTL", {0, 0, 0, 1, 0, 0, 0, 0})},
                           {Make("fcTL", {0}), WithCrc(Make("fdAT", {0}), 0)})),
      true);
  for (const auto dimensions :
       {std::pair{0U, 8U}, std::pair{8U, 0U}, std::pair{0x7fffffffU, 8U}}) {
    try {
      (void)gufo::core::DecodeImage(
          Serialize({Ihdr(dimensions.first, dimensions.second),
                     Make("IDAT", GoodIdat()), Make("IEND")}));
      throw std::runtime_error("invalid dimensions accepted");
    } catch (const std::invalid_argument&) {
    }
  }
}

void Orientation() {
  const Bytes rows{1,  2,  3,  4,  5,  6,  7,  8,  9,
                   10, 11, 12, 13, 14, 15, 16, 17, 18};
  const auto plain =
      Parse(WritePng({PNG_COLOR_TYPE_RGB, 8}, 3, 2, rows, false, false));
  // Pixel indices for EXIF orientations 1..8, laid out in output row order.
  const std::array<std::array<unsigned, 6>, 8> order{{{0, 1, 2, 3, 4, 5},
                                                      {2, 1, 0, 5, 4, 3},
                                                      {5, 4, 3, 2, 1, 0},
                                                      {3, 4, 5, 0, 1, 2},
                                                      {0, 3, 1, 4, 2, 5},
                                                      {3, 0, 4, 1, 5, 2},
                                                      {5, 2, 4, 1, 3, 0},
                                                      {2, 5, 1, 4, 0, 3}}};
  for (std::uint8_t orientation = 1; orientation <= 8; ++orientation) {
    auto chunks = plain;
    chunks.insert(chunks.begin() + 1,
                  Make("eXIf", {'M', 'M',         0,    '*', 0, 0, 0, 8, 0,
                                1,   1,           0x12, 0,   3, 0, 0, 0, 1,
                                0,   orientation, 0,    0,   0, 0, 0, 0}));
    const auto image = gufo::core::DecodeImage(Serialize(chunks));
    Require(image.width == (orientation < 5 ? 3U : 2U));
    Require(image.height == (orientation < 5 ? 2U : 3U));
    Bytes expected;
    for (const auto pixel : order[orientation - 1])
      expected.insert(expected.end(), rows.begin() + pixel * 3,
                      rows.begin() + pixel * 3 + 3);
    Require(image.pixels == expected);
  }
}
}  // namespace
int main() {
  try {
    Formats();
    Framing();
    Orientation();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

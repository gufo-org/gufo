// Expected pixels and rejections at the C++ image callers. The test writes
// its PNG cases itself, with stored deflate blocks and filter type 0.
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
#include <tuple>
#include <utility>
#include <vector>

#include "src/core/image.hpp"
namespace {
using gufo::core::DecodeImage;
using gufo::core::DecodeImageRgba;
using gufo::core::EncodePngRgba;
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

std::uint32_t Crc32(std::uint32_t crc, std::span<const std::uint8_t> bytes) {
  crc = ~crc;
  for (const auto byte : bytes) {
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit)
      crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1U)));
  }
  return ~crc;
}

std::uint32_t ChunkCrc(std::string_view type, const Bytes& data) {
  const auto* name = reinterpret_cast<const std::uint8_t*>(type.data());
  return Crc32(Crc32(0, {name, type.size()}), data);
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

// A zlib stream of stored deflate blocks.
Bytes Compress(const Bytes& data) {
  Bytes out{0x78, 0x01};
  std::size_t offset = 0;
  do {
    const auto size = std::min<std::size_t>(data.size() - offset, 65535);
    const bool last = offset + size == data.size();
    out.push_back(last ? 1 : 0);
    for (const auto field : {size, size ^ 0xffff}) {
      out.push_back(static_cast<std::uint8_t>(field));
      out.push_back(static_cast<std::uint8_t>(field >> 8));
    }
    out.insert(out.end(), data.begin() + offset, data.begin() + offset + size);
    offset += size;
  } while (offset < data.size());
  std::uint32_t a = 1, b = 0;
  for (const auto byte : data) {
    a = (a + byte) % 65521;
    b = (b + a) % 65521;
  }
  AppendBe32(out, (b << 16) | a);
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

constexpr int kGray = 0, kRgb = 2, kPalette = 3, kGrayAlpha = 4, kRgba = 6;

constexpr std::array<Format, 15> kFormats{{
    {kGray, 1},
    {kGray, 2},
    {kGray, 4},
    {kGray, 8},
    {kGray, 16},
    {kPalette, 1},
    {kPalette, 2},
    {kPalette, 4},
    {kPalette, 8},
    {kGrayAlpha, 8},
    {kGrayAlpha, 16},
    {kRgb, 8},
    {kRgb, 16},
    {kRgba, 8},
    {kRgba, 16},
}};

unsigned Channels(int color) {
  switch (color) {
    case kGrayAlpha:
      return 2;
    case kRgb:
      return 3;
    case kRgba:
      return 4;
    default:
      return 1;
  }
}

std::size_t RowBytes(const Format& format, unsigned width) {
  return (std::size_t{width} * Channels(format.color) * format.depth + 7) / 8;
}

bool AllowsTrns(int color) {
  return color == kGray || color == kRgb || color == kPalette;
}

Bytes RandomRows(const Format& format, unsigned width, unsigned height) {
  std::mt19937 rng(width * 131 + height * 17 + format.color * 5 + format.depth);
  Bytes rows(RowBytes(format, width) * height);
  for (auto& byte : rows)
    byte = static_cast<std::uint8_t>(rng());
  return rows;
}

// Scanlines in Adam7 pass order when interlaced.
Bytes Scanlines(const Format& format, unsigned width, unsigned height,
                const Bytes& rows, bool interlace) {
  const auto bits = Channels(format.color) * format.depth;
  const auto stride = RowBytes(format, width);
  struct Pass {
    unsigned x0, dx, y0, dy;
  };
  const std::vector<Pass> passes =
      interlace ? std::vector<Pass>{{0, 8, 0, 8}, {4, 8, 0, 8}, {0, 4, 4, 8},
                                    {2, 4, 0, 4}, {0, 2, 2, 4}, {1, 2, 0, 2},
                                    {0, 1, 1, 2}}
                : std::vector<Pass>{{0, 1, 0, 1}};
  Bytes out;
  // Every other scanline uses the Sub filter, the rest filter type 0.
  const std::size_t filter_step = std::max(bits / 8, 1U);
  bool sub = false;
  const auto filter = [&](std::size_t start) {
    if ((sub = !sub)) {
      out[start - 1] = 1;
      for (std::size_t i = out.size(); i-- > start + filter_step;)
        out[i] = static_cast<std::uint8_t>(out[i] - out[i - filter_step]);
    }
  };
  for (const auto& pass : passes) {
    if (pass.x0 >= width || pass.y0 >= height)
      continue;
    for (unsigned y = pass.y0; y < height; y += pass.dy) {
      out.push_back(0);
      const std::size_t start = out.size();
      const auto* row = rows.data() + y * stride;
      if (bits >= 8) {
        for (unsigned x = pass.x0; x < width; x += pass.dx)
          out.insert(out.end(), row + x * bits / 8, row + (x + 1) * bits / 8);
        filter(start);
        continue;
      }
      unsigned used = 0;
      for (unsigned x = pass.x0; x < width; x += pass.dx) {
        const unsigned value =
            (row[x * bits / 8] >> (8 - bits - x * bits % 8)) &
            ((1U << bits) - 1);
        if (used % 8 == 0)
          out.push_back(0);
        out.back() |= static_cast<std::uint8_t>(value << (8 - bits - used % 8));
        used += bits;
      }
      filter(start);
    }
  }
  return out;
}

constexpr std::uint8_t PaletteAlpha(unsigned index) {
  return static_cast<std::uint8_t>(index < 5 ? index * 53 : 255);
}

Bytes WritePng(const Format& format, unsigned width, unsigned height,
               const Bytes& rows, bool interlace, bool trns,
               std::vector<Chunk> color_chunks = {}) {
  std::vector<Chunk> chunks{
      Ihdr(width, height, format.depth, format.color, interlace ? 1 : 0)};
  chunks.insert(chunks.end(), color_chunks.begin(), color_chunks.end());
  if (format.color == kPalette) {
    const unsigned entries = 1U << format.depth;
    Bytes palette;
    for (unsigned i = 0; i < entries; ++i) {
      for (const auto value : {i * 37, i * 91, 255 - i})
        palette.push_back(static_cast<std::uint8_t>(value));
    }
    chunks.push_back(Make("PLTE", std::move(palette)));
    if (trns) {
      Bytes alpha;
      for (unsigned i = 0; i < std::min(entries, 5U); ++i)
        alpha.push_back(PaletteAlpha(i));
      chunks.push_back(Make("tRNS", std::move(alpha)));
    }
  } else if (trns) {
    // The first pixel's samples are the transparent key.
    Bytes key;
    for (unsigned c = 0; c < Channels(format.color); ++c) {
      if (format.depth == 16) {
        key.push_back(rows[c * 2]);
        key.push_back(rows[c * 2 + 1]);
      } else {
        key.push_back(0);
        key.push_back(format.depth == 8 ? rows[c]
                                        : rows[0] >> (8 - format.depth));
      }
    }
    chunks.push_back(Make("tRNS", std::move(key)));
  }
  chunks.push_back(Make(
      "IDAT", Compress(Scanlines(format, width, height, rows, interlace))));
  chunks.push_back(Make("IEND"));
  return Serialize(chunks);
}

void Require(bool condition) {
  if (!condition)
    throw std::runtime_error("PNG expected-result check failed");
}

unsigned Sample(const Format& format, unsigned width, const Bytes& rows,
                unsigned x, unsigned y, unsigned c) {
  const auto index = x * Channels(format.color) + c;
  const auto offset = y * RowBytes(format, width);
  if (format.depth < 8)
    return (rows[offset + index * format.depth / 8] >>
            (8 - format.depth - index * format.depth % 8)) &
           ((1U << format.depth) - 1);
  if (format.depth == 16)
    return (rows[offset + index * 2] << 8) | rows[offset + index * 2 + 1];
  return rows[offset + index];
}

std::uint8_t Round16(unsigned value) {
  return static_cast<std::uint8_t>((value * 255 + 32767) / 65535);
}

void RequireEqual(std::string_view name, const Bytes& actual,
                  const Bytes& expected) {
  if (actual == expected)
    return;
  std::cerr << name << '\n';
  for (std::size_t i = 0; i < std::min(actual.size(), expected.size()); ++i)
    if (actual[i] != expected[i]) {
      std::cerr << "byte " << i << " expected " << unsigned(expected[i])
                << " actual " << unsigned(actual[i]) << '\n';
      break;
    }
  Require(false);
}

// RGB8 follows Pillow's convert("RGB"); RGBA8 follows libpng's simplified
// API. 16-bit RGBA cases carry sRGB, so their samples only round.
void Formats() {
  for (const auto [width, height] :
       {std::pair{1U, 1U}, std::pair{3U, 3U}, std::pair{9U, 7U}}) {
    for (const auto format : kFormats) {
      for (const bool interlace : {false, true}) {
        for (const bool trns : {false, true}) {
          if (trns && !AllowsTrns(format.color))
            continue;
          const auto rows = RandomRows(format, width, height);
          const auto png =
              WritePng(format, width, height, rows, interlace, trns,
                       format.depth == 16 ? std::vector{Make("sRGB", {0})}
                                          : std::vector<Chunk>{});
          const auto rgb = DecodeImage(png);
          const auto rgba = DecodeImageRgba(png);
          Require(rgb.width == width && rgb.height == height);
          Require(rgba.width == width && rgba.height == height);
          const unsigned max = (1U << format.depth) - 1;
          const bool color = format.color == kRgb || format.color == kRgba;
          const bool alpha =
              format.color == kGrayAlpha || format.color == kRgba;
          Bytes expected_rgb, expected_rgba;
          for (unsigned y = 0; y < height; ++y) {
            for (unsigned x = 0; x < width; ++x) {
              const auto sample = [&](unsigned c) {
                return Sample(format, width, rows, x, y, c);
              };
              if (format.color == kPalette) {
                const auto index = sample(0);
                for (const auto value : {index * 37, index * 91, 255 - index}) {
                  expected_rgb.push_back(static_cast<std::uint8_t>(value));
                  expected_rgba.push_back(static_cast<std::uint8_t>(value));
                }
                expected_rgba.push_back(trns ? PaletteAlpha(index) : 255);
                continue;
              }
              bool keyed = trns;
              for (unsigned c = 0; c < 3; ++c) {
                const auto value = sample(color ? c : 0);
                keyed = keyed && value == Sample(format, width, rows, 0, 0,
                                                 color ? c : 0);
                // RGB8 saturates Gray16 and strips the low byte of other 16-bit
                // samples, as Pillow does; RGBA8 rounds, as libpng does.
                expected_rgb.push_back(static_cast<std::uint8_t>(
                    format.depth < 8 ? value * 255 / max
                    : format.depth == 16
                        ? (format.color == kGray ? std::min(value, 255U)
                                                 : value >> 8)
                        : value));
                expected_rgba.push_back(
                    format.depth < 8
                        ? static_cast<std::uint8_t>(value * 255 / max)
                    : format.depth == 16 ? Round16(value)
                                         : static_cast<std::uint8_t>(value));
              }
              if (alpha) {
                const auto value = sample(Channels(format.color) - 1);
                expected_rgba.push_back(format.depth == 16
                                            ? Round16(value)
                                            : static_cast<std::uint8_t>(value));
              } else {
                expected_rgba.push_back(keyed ? 0 : 255);
              }
            }
          }
          const auto name =
              std::to_string(width) + "x" + std::to_string(height) + " color " +
              std::to_string(format.color) + " depth " +
              std::to_string(format.depth) + " interlace " +
              std::to_string(interlace) + " trns " + std::to_string(trns);
          RequireEqual(name + " RGB", rgb.pixels, expected_rgb);
          RequireEqual(name + " RGBA", rgba.rgba, expected_rgba);
        }
      }
    }
  }
}

Chunk Gamma(std::uint32_t gamma) {
  Bytes data;
  AppendBe32(data, gamma);
  return Make("gAMA", std::move(data));
}

Bytes Be16(std::initializer_list<unsigned> values) {
  Bytes out;
  for (const auto value : values) {
    out.push_back(static_cast<std::uint8_t>(value >> 8));
    out.push_back(static_cast<std::uint8_t>(value));
  }
  return out;
}

// Expected values are libpng 1.6.59's simplified API output. RGB8 ignores
// the color chunks.
void Gamma() {
  const Bytes gray8{0, 1, 64, 127, 128, 200, 254, 255};
  const auto gray16 = Be16({0, 1, 255, 4096, 32768, 50000, 65534, 65535});
  const auto gray = [](std::initializer_list<unsigned> values) {
    Bytes out;
    for (const auto value : values) {
      for (int c = 0; c < 3; ++c)
        out.push_back(static_cast<std::uint8_t>(value));
      out.push_back(255);
    }
    return out;
  };
  const auto check = [](std::string_view name, const Format& format,
                        unsigned width, const Bytes& row,
                        std::vector<Chunk> color_chunks,
                        const Bytes& expected) {
    RequireEqual(name,
                 DecodeImageRgba(WritePng(format, width, 1, row, false, false,
                                          std::move(color_chunks)))
                     .rgba,
                 expected);
  };
  const auto linear8 = gray({0, 21, 136, 186, 186, 228, 255, 255});
  check("gray8 gAMA 1.0", {kGray, 8}, 8, gray8, {Gamma(100000)}, linear8);
  check("gray8 sRGB overrides gAMA", {kGray, 8}, 8, gray8,
        {Gamma(100000), Make("sRGB", {0})},
        gray({0, 1, 64, 127, 128, 200, 254, 255}));
  check("gray8 sRGB after gAMA and cICP keeps gAMA", {kGray, 8}, 8, gray8,
        {Make("cICP", {1, 13, 0, 1}), Gamma(100000), Make("sRGB", {0})},
        linear8);
  check("gray8 duplicate gAMA ignored", {kGray, 8}, 8, gray8,
        {Gamma(100000), Gamma(45455)}, linear8);
  check("gray16 linear by default", {kGray, 16}, 8, gray16, {},
        gray({0, 0, 19, 72, 186, 225, 255, 255}));
  check("gray16 sBIT 6", {kGray, 16}, 8, gray16, {Make("sBIT", {6})},
        gray({0, 0, 0, 71, 186, 225, 255, 255}));
  check("gray16 sRGB", {kGray, 16}, 8, gray16, {Make("sRGB", {0})},
        gray({0, 0, 1, 16, 128, 195, 255, 255}));
  check("rgba16 alpha only rounds", {kRgba, 16}, 1,
        Be16({32768, 4096, 65535, 32896}), {}, {186, 72, 255, 128});
  // gAMA after a suggested palette is out of place.
  check("gray8 sRGB intent 4 ignored", {kGray, 8}, 8, gray8,
        {Make("sRGB", {4}), Gamma(100000)}, linear8);
  check("gray8 gAMA above INT32_MAX ignored", {kGray, 8}, 8, gray8,
        {Gamma(0x80000000U), Gamma(100000)}, linear8);
  check("gray8 gAMA of bad length ignored", {kGray, 8}, 8, gray8,
        {Make("gAMA", {0, 1, 0x86, 0xa0, 0}), Gamma(100000)}, linear8);
  check("gray8 gAMA with bad CRC ignored", {kGray, 8}, 8, gray8,
        {WithCrc(Gamma(100000), 0)}, gray({0, 1, 64, 127, 128, 200, 254, 255}));
  check("gray16 sBIT 10", {kGray, 16}, 8, gray16, {Make("sBIT", {10})},
        gray({0, 0, 17, 72, 186, 225, 255, 255}));
  check("gray16 sBIT 9", {kGray, 16}, 8, gray16, {Make("sBIT", {9})},
        gray({0, 0, 11, 72, 186, 225, 255, 255}));
  // A tRNS key above the sample depth matches its low bits.
  check("gray2 tRNS 0x105", {kGray, 2}, 4, {0x1b}, {Make("tRNS", {1, 5})},
        {0, 0, 0, 255, 85, 85, 85, 0, 170, 170, 170, 255, 255, 255, 255, 255});
  // A suggested palette after tRNS is ignored, so gAMA still applies.
  check("rgb8 PLTE after tRNS ignored", {kRgb, 8}, 2, {10, 100, 200, 1, 2, 3},
        {Make("tRNS", {0, 1, 0, 2, 0, 3}), Make("PLTE", {0, 0, 0}),
         Gamma(100000)},
        {59, 167, 228, 255, 21, 28, 34, 0});
  check("rgb8 gAMA after PLTE ignored", {kRgb, 8}, 2,
        {10, 100, 200, 30, 60, 90}, {Make("PLTE", {0, 0, 0}), Gamma(100000)},
        {10, 100, 200, 255, 30, 60, 90, 255});
  // Palette entries are corrected, their tRNS alpha is not.
  const auto indexed =
      Serialize({Ihdr(3, 1, 8, kPalette), Gamma(100000),
                 Make("PLTE", {64, 128, 192, 1, 2, 3, 250, 251, 252}),
                 Make("tRNS", {100, 0}), Make("IDAT", Compress({0, 0, 1, 2})),
                 Make("IEND")});
  RequireEqual("indexed gAMA 1.0", DecodeImageRgba(indexed).rgba,
               {136, 186, 224, 100, 21, 28, 34, 0, 253, 253, 254, 255});
  RequireEqual("indexed gAMA 1.0 RGB", DecodeImage(indexed).pixels,
               {64, 128, 192, 1, 2, 3, 250, 251, 252});
}

void Framing() {
  const auto check = [](std::string_view name, const Bytes& png,
                        bool accepted) {
    try {
      const auto image = DecodeImage(png);
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
    // RGBA8 accepts and rejects the same files.
    try {
      const auto image = DecodeImageRgba(png);
      if (!accepted || image.width != 8 || image.height != 8)
        throw std::runtime_error(std::string(name) + " RGBA");
      for (std::size_t i = 0; i < 64; ++i)
        Require(image.rgba[i * 4 + 3] == 255 &&
                image.rgba[i * 4] == DefaultFill(i % 8, i / 8, 0));
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
      (void)DecodeImage(Serialize({Ihdr(dimensions.first, dimensions.second),
                                   Make("IDAT", GoodIdat()), Make("IEND")}));
      throw std::runtime_error("invalid dimensions accepted");
    } catch (const std::invalid_argument&) {
    }
  }
}

void Orientation() {
  const Bytes rows{1,  2,  3,  4,  5,  6,  7,  8,  9,
                   10, 11, 12, 13, 14, 15, 16, 17, 18};
  const auto plain = Parse(WritePng({kRgb, 8}, 3, 2, rows, false, false));
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
    const auto image = DecodeImage(Serialize(chunks));
    Require(image.width == (orientation < 5 ? 3U : 2U));
    Require(image.height == (orientation < 5 ? 2U : 3U));
    Bytes expected;
    for (const auto pixel : order[orientation - 1])
      expected.insert(expected.end(), rows.begin() + pixel * 3,
                      rows.begin() + pixel * 3 + 3);
    Require(image.pixels == expected);
    // RGBA8 ignores orientation, as libpng's simplified API did.
    const auto rgba = DecodeImageRgba(Serialize(chunks));
    Require(rgba.width == 3 && rgba.height == 2);
    for (unsigned pixel = 0; pixel < 6; ++pixel)
      Require(std::equal(rows.begin() + pixel * 3, rows.begin() + pixel * 3 + 3,
                         rgba.rgba.begin() + pixel * 4) &&
              rgba.rgba[pixel * 4 + 3] == 255);
  }
}

// Encoded pixels decode back unchanged in both formats.
void Encoding() {
  std::mt19937 rng(7);
  for (const auto [width, height] :
       {std::pair{1U, 1U}, std::pair{3U, 2U}, std::pair{17U, 5U},
        std::pair{64U, 64U}, std::pair{301U, 7U}}) {
    for (const bool noise : {false, true}) {
      Bytes rgba(std::size_t{width} * height * 4);
      for (std::size_t i = 0; i < rgba.size(); ++i)
        rgba[i] = static_cast<std::uint8_t>(
            noise ? rng() : i % 4 * 60 + i / 4 % width * 3 + i / 4 / width);
      const auto png = EncodePngRgba(width, height, rgba);
      const auto chunks = Parse(png);
      Require(chunks.size() >= 4 && chunks[0].type == "IHDR" &&
              chunks[0].data[8] == 8 && chunks[0].data[9] == kRgba &&
              chunks[0].data[10] == 0 && chunks[0].data[11] == 0 &&
              chunks[0].data[12] == 0 && chunks[1].type == "sRGB" &&
              chunks[1].data == Bytes{0} && chunks.back().type == "IEND");
      const auto decoded = DecodeImageRgba(png);
      Require(decoded.width == width && decoded.height == height);
      RequireEqual("encoded RGBA", decoded.rgba, rgba);
      Bytes rgb;
      for (std::size_t i = 0; i < rgba.size(); i += 4)
        rgb.insert(rgb.end(), rgba.begin() + i, rgba.begin() + i + 3);
      RequireEqual("encoded RGB", DecodeImage(png).pixels, rgb);
    }
  }
  for (const auto [width, height, size] :
       {std::tuple{2U, 2U, 15U}, std::tuple{0U, 1U, 0U}, std::tuple{1U, 0U, 0U},
        std::tuple{1'000'001U, 1U, 4'000'004U}}) {
    try {
      (void)EncodePngRgba(width, height, Bytes(size));
      throw std::runtime_error("invalid RGBA geometry encoded");
    } catch (const std::runtime_error& error) {
      if (std::string_view(error.what()) != "cannot encode output PNG")
        throw;
    }
  }
}
}  // namespace
int main() {
  try {
    Formats();
    Gamma();
    Framing();
    Orientation();
    Encoding();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

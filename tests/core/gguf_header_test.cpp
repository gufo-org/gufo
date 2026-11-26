// The Rust GGUF header parser through its C++ caller, GgufReader::OpenMemory:
// every metadata value type as the C++ variant holds it, with strings that
// point into the parsed bytes; arrays of every item type around the parser's
// blocks of 32768 items; versions, alignments and tensors; the exact message
// of every rejection; the descriptor budget filled to its last byte and one
// byte past it, which names the limit where 0.35.0 reported a malformed or
// over-long entry; and truncations and byte changes that must fail with one of
// those messages or open a header whose tensors lie inside the buffer. GGUF
// files named on the command line must open, and are timed. The crate's Rust
// tests cover the check order, the budget charges and the C ABI.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/platform/mapped_file.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::core::GgufMetadataValue;
using gufo::core::GgufOpenOptions;
using gufo::core::GgufReader;
using gufo::core::GgufTensorInfo;
using Bytes = std::vector<std::uint8_t>;

constexpr std::uint32_t kUint8 = 0;
constexpr std::uint32_t kInt8 = 1;
constexpr std::uint32_t kUint16 = 2;
constexpr std::uint32_t kInt16 = 3;
constexpr std::uint32_t kUint32 = 4;
constexpr std::uint32_t kInt32 = 5;
constexpr std::uint32_t kFloat32 = 6;
constexpr std::uint32_t kBool = 7;
constexpr std::uint32_t kString = 8;
constexpr std::uint32_t kArray = 9;
constexpr std::uint32_t kUint64 = 10;
constexpr std::uint32_t kInt64 = 11;
constexpr std::uint32_t kFloat64 = 12;
// Bytes of a scalar of each value type; strings and arrays have their own.
constexpr std::size_t kScalarBytes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
constexpr std::string_view kMalformed = "Malformed or truncated GGUF";
constexpr std::string_view kOverBudget =
    "GGUF metadata and tensor index exceed the header limit of 128 MiB";
// Every message OpenMemory fails with; names follow those ending in a space.
constexpr std::string_view kMessages[] = {
    "Invalid memory buffer for GGUF",
    "Invalid GGUF magic header",
    "Unsupported GGUF version: ",
    "GGUF descriptor counts exceed file or memory limits",
    kOverBudget,
    "Failed to read metadata key",
    "GGUF metadata key exceeds limits",
    "Failed to read metadata type for key ",
    "Invalid GGUF alignment",
    "Duplicate GGUF metadata key",
    "Failed to read tensor name",
    "Invalid tensor dimensions count",
    "Invalid GGUF tensor shape or unsupported storage type: ",
    "Duplicate GGUF tensor name",
    "GGUF data section is truncated before its alignment",
    "Tensor extent or alignment is invalid: ",
    kMalformed,
};

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::string Escape(std::string_view text) {
  std::string out;
  for (const char c : text) {
    const auto byte = static_cast<unsigned char>(c);
    if (byte >= 0x20 && byte < 0x7F && c != '\\') {
      out += c;
    } else {
      char hex[5];
      std::snprintf(hex, sizeof(hex), "\\x%02X", byte);
      out += hex;
    }
  }
  return out;
}

bool KnownMessage(std::string_view error) {
  return std::ranges::any_of(kMessages, [&](std::string_view message) {
    return message.ends_with(' ') ? error.starts_with(message)
                                  : error == message;
  });
}

class Writer {
public:
  // The fixed header of a version 3 file.
  static Writer Header(std::uint64_t tensors, std::uint64_t metadata,
                       std::uint32_t version = 3) {
    Writer out;
    out.bytes = {'G', 'G', 'U', 'F'};
    out.Pod(version);
    out.Pod(tensors);
    out.Pod(metadata);
    return out;
  }

  template<class T>
  Writer& Pod(T value) {
    const auto* data = reinterpret_cast<const std::uint8_t*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
    return *this;
  }

  Writer& Zeros(std::size_t count) {
    bytes.resize(bytes.size() + count, 0);
    return *this;
  }

  Writer& String(std::string_view text) {
    Pod<std::uint64_t>(text.size());
    bytes.insert(bytes.end(), text.begin(), text.end());
    return *this;
  }

  Writer& Key(std::string_view key, std::uint32_t type) {
    String(key);
    return Pod(type);
  }

  Writer& Tensor(std::string_view name,
                 std::initializer_list<std::uint64_t> dimensions,
                 std::uint32_t type, std::uint64_t offset) {
    String(name);
    Pod(static_cast<std::uint32_t>(dimensions.size()));
    for (const auto dimension : dimensions)
      Pod(dimension);
    Pod(type);
    return Pod(offset);
  }

  // Pads to `alignment` and appends `data` bytes of 0xAB.
  Writer& Data(std::size_t alignment, std::size_t data) {
    bytes.resize((bytes.size() + alignment - 1) / alignment * alignment, 0);
    bytes.resize(bytes.size() + data, 0xAB);
    return *this;
  }

  Bytes bytes;
};

struct Opened {
  std::unique_ptr<GgufReader> reader;
  std::string error;
};

Opened Open(std::span<const std::uint8_t> bytes, GgufOpenOptions options = {}) {
  Opened opened;
  opened.reader = GgufReader::OpenMemory(bytes.data(), bytes.size(),
                                         &opened.error, options);
  return opened;
}

// Fails with `message`, also with unsized tensors allowed unless `strict`,
// and without a message wanted.
void Rejects(std::span<const std::uint8_t> bytes, std::string_view message,
             const std::string& what, bool strict = false) {
  for (const bool unsized : {false, true}) {
    if (unsized && strict)
      continue;
    const auto opened = Open(bytes, {.allow_unsized_tensors = unsized});
    Require(!opened.reader && opened.error == message,
            what + ": fails with " + Escape(opened.error) + " instead of " +
                Escape(message));
  }
  Require(
      GgufReader::OpenMemory(bytes.data(), bytes.size(), nullptr) == nullptr,
      what + ": opens without an error message");
}

std::unique_ptr<GgufReader> Opens(std::span<const std::uint8_t> bytes,
                                  const std::string& what,
                                  GgufOpenOptions options = {}) {
  auto opened = Open(bytes, options);
  Require(opened.reader != nullptr,
          what + ": fails with " + Escape(opened.error));
  return std::move(opened.reader);
}

template<class T>
const T& Get(const GgufReader& reader, std::string_view key,
             std::uint32_t type) {
  const auto* value = reader.FindMetadata(key);
  Require(value != nullptr && static_cast<std::uint32_t>(value->type) == type &&
              std::holds_alternative<T>(value->value),
          "metadata " + std::string(key) + " is missing or of another type");
  return std::get<T>(value->value);
}

bool SameDouble(double a, double b) {
  return std::memcmp(&a, &b, sizeof(a)) == 0;
}

// Points at `text` where it sits in `bytes`, after its 8-byte length.
bool InPlace(std::string_view view, std::span<const std::uint8_t> bytes,
             std::string_view text) {
  const auto* start = reinterpret_cast<const char*>(bytes.data());
  if (view != text || view.data() < start + 8 ||
      view.data() + view.size() > start + bytes.size())
    return false;
  std::uint64_t length = 0;
  std::memcpy(&length, view.data() - 8, sizeof(length));
  return length == text.size();
}

void ValueTypes() {
  const auto bytes = Writer::Header(0, 15)
                         .Key("u8", kUint8)
                         .Pod<std::uint8_t>(200)
                         .Key("i8", kInt8)
                         .Pod<std::int8_t>(-128)
                         .Key("u16", kUint16)
                         .Pod<std::uint16_t>(65535)
                         .Key("i16", kInt16)
                         .Pod<std::int16_t>(-2)
                         .Key("u32", kUint32)
                         .Pod<std::uint32_t>(7)
                         .Key("i32", kInt32)
                         .Pod<std::int32_t>(-3)
                         .Key("f32", kFloat32)
                         .Pod(1.5F)
                         .Key("bool", kBool)
                         .Pod<std::uint8_t>(1)
                         .Key("str", kString)
                         .String(std::string_view("\xFF\0x", 3))
                         .Key("u64", kUint64)
                         .Pod(~std::uint64_t{0})
                         .Key("i64", kInt64)
                         .Pod(std::numeric_limits<std::int64_t>::min())
                         .Key("f64", kFloat64)
                         .Pod(-0.25)
                         .Key("strings", kArray)
                         .Pod(kString)
                         .Pod<std::uint64_t>(2)
                         .String("a")
                         .String("")
                         .Key("flags", kArray)
                         .Pod(kBool)
                         .Pod<std::uint64_t>(2)
                         .Pod<std::uint8_t>(0)
                         .Pod<std::uint8_t>(1)
                         .Key("empty", kArray)
                         .Pod(kFloat32)
                         .Pod<std::uint64_t>(0)
                         .bytes;
  const auto reader = Opens(bytes, "every value type");
  Require(reader->GetVersion() == 3 && reader->GetAlignment() == 32 &&
              reader->GetMetadataCount() == 15 && reader->GetTensors().empty(),
          "every value type: header differs");
  Require(
      Get<std::uint64_t>(*reader, "u8", kUint8) == 200 &&
          Get<std::int64_t>(*reader, "i8", kInt8) == -128 &&
          Get<std::uint64_t>(*reader, "u16", kUint16) == 65535 &&
          Get<std::int64_t>(*reader, "i16", kInt16) == -2 &&
          Get<std::uint64_t>(*reader, "u32", kUint32) == 7 &&
          Get<std::int64_t>(*reader, "i32", kInt32) == -3 &&
          SameDouble(Get<double>(*reader, "f32", kFloat32), 1.5) &&
          Get<bool>(*reader, "bool", kBool) &&
          Get<std::uint64_t>(*reader, "u64", kUint64) == ~std::uint64_t{0} &&
          Get<std::int64_t>(*reader, "i64", kInt64) ==
              std::numeric_limits<std::int64_t>::min() &&
          SameDouble(Get<double>(*reader, "f64", kFloat64), -0.25),
      "a scalar differs");
  const auto& text = Get<std::string_view>(*reader, "str", kString);
  Require(InPlace(text, bytes, std::string_view("\xFF\0x", 3)),
          "a string is not a view of the file");
  const auto& strings =
      Get<std::vector<std::string_view>>(*reader, "strings", kArray);
  Require(strings.size() == 2 && InPlace(strings[0], bytes, "a") &&
              strings[1].empty(),
          "a string array differs");
  // Boolean arrays read as numbers.
  Require(Get<std::vector<std::uint64_t>>(*reader, "flags", kArray) ==
              std::vector<std::uint64_t>{0, 1},
          "a boolean array differs");
  Require(Get<std::vector<double>>(*reader, "empty", kArray).empty(),
          "an empty array differs");
  Require(reader->FindMetadata("missing") == nullptr, "a missing key is found");
}

// Arrays of every item type around the parser's blocks of 32768 items,
// whole and cut off in their second block.
void LongArrays() {
  constexpr std::size_t kBlock = 32768;
  for (std::uint32_t item = 0; item < std::size(kScalarBytes); ++item) {
    if (item == kArray)
      continue;
    for (const std::size_t count :
         {kBlock - 1, kBlock, kBlock + 1, (3 * kBlock) + 5}) {
      auto out = Writer::Header(0, 2);
      out.Key("long", kArray).Pod(item).Pod<std::uint64_t>(count);
      std::vector<std::uint64_t> unsigneds;
      std::vector<std::int64_t> signeds;
      std::vector<double> floats;
      std::vector<std::string> strings;
      std::size_t cut = 0;
      for (std::size_t i = 0; i < count; ++i) {
        if (i == kBlock + 1)
          cut = out.bytes.size() + 1;
        // Every bit pattern of the narrow types, signs and fractions.
        const auto bits = i * 0x9E3779B97F4A7C15ULL;
        switch (item) {
          case kString:
            strings.push_back(std::to_string(i));
            out.String(strings.back());
            break;
          case kBool:
            unsigneds.push_back(i % 2);
            out.Pod<std::uint8_t>(i % 2);
            break;
          case kFloat32:
            floats.push_back((static_cast<double>(i) * 0.25) - 4096.0);
            out.Pod(static_cast<float>(floats.back()));
            break;
          case kFloat64:
            floats.push_back((static_cast<double>(i) / 3.0) - 1.0);
            out.Pod(floats.back());
            break;
          case kInt8:
            signeds.push_back(static_cast<std::int8_t>(bits));
            out.Pod(static_cast<std::int8_t>(bits));
            break;
          case kInt16:
            signeds.push_back(static_cast<std::int16_t>(bits));
            out.Pod(static_cast<std::int16_t>(bits));
            break;
          case kInt32:
            signeds.push_back(static_cast<std::int32_t>(bits));
            out.Pod(static_cast<std::int32_t>(bits));
            break;
          case kInt64:
            signeds.push_back(static_cast<std::int64_t>(bits));
            out.Pod(static_cast<std::int64_t>(bits));
            break;
          case kUint8:
            unsigneds.push_back(static_cast<std::uint8_t>(bits));
            out.Pod(static_cast<std::uint8_t>(bits));
            break;
          case kUint16:
            unsigneds.push_back(static_cast<std::uint16_t>(bits));
            out.Pod(static_cast<std::uint16_t>(bits));
            break;
          case kUint32:
            unsigneds.push_back(static_cast<std::uint32_t>(bits));
            out.Pod(static_cast<std::uint32_t>(bits));
            break;
          default:
            unsigneds.push_back(bits);
            out.Pod(bits);
            break;
        }
      }
      out.Key("after", kUint32).Pod<std::uint32_t>(7);
      const auto what = "array of " + std::to_string(count) +
                        " items of type " + std::to_string(item);
      const auto reader = Opens(out.bytes, what);
      Require(Get<std::uint64_t>(*reader, "after", kUint32) == 7,
              what + ": the entry after it differs");
      bool same = false;
      if (item == kString) {
        const auto& got =
            Get<std::vector<std::string_view>>(*reader, "long", kArray);
        same = std::ranges::equal(got, strings);
      } else if (item == kFloat32 || item == kFloat64) {
        same = std::ranges::equal(
            Get<std::vector<double>>(*reader, "long", kArray), floats,
            SameDouble);
      } else if (!signeds.empty()) {
        same =
            Get<std::vector<std::int64_t>>(*reader, "long", kArray) == signeds;
      } else {
        same = Get<std::vector<std::uint64_t>>(*reader, "long", kArray) ==
               unsigneds;
      }
      Require(same, what + ": items differ");
      if (cut == 0)
        cut = out.bytes.size() / 2;
      Rejects(std::span(out.bytes).first(cut), kMalformed, what + " cut off");
    }
  }
}

void VersionsAndAlignments() {
  const auto v2 = Writer::Header(0, 0, 2).bytes;
  Require(Opens(v2, "version 2")->GetVersion() == 2, "version 2 differs");
  for (const std::uint32_t version : {0U, 1U, 4U}) {
    Rejects(Writer::Header(0, 0, version).bytes,
            "Unsupported GGUF version: " + std::to_string(version),
            "version " + std::to_string(version));
  }
  // Any unsigned type holds a power of two.
  const auto aligned = [](std::uint32_t type, std::uint64_t value) {
    auto out = Writer::Header(1, 1);
    out.Key("general.alignment", type);
    out.bytes.insert(
        out.bytes.end(), reinterpret_cast<const std::uint8_t*>(&value),
        reinterpret_cast<const std::uint8_t*>(&value) + kScalarBytes[type]);
    out.Tensor("t", {8}, 0, 0);
    return out.Data(value, 32).bytes;
  };
  for (const auto& [type, value] :
       {std::pair{kUint8, 1ULL}, std::pair{kUint16, 256ULL},
        std::pair{kUint32, 64ULL}, std::pair{kUint64, 4096ULL}}) {
    const auto bytes = aligned(type, value);
    const auto reader = Opens(bytes, "alignment " + std::to_string(value));
    const auto& tensor = reader->GetTensors().front();
    Require(reader->GetAlignment() == value &&
                tensor.data == bytes.data() + bytes.size() - 32,
            "alignment " + std::to_string(value) + " differs");
  }
  for (const auto& [type, value] :
       {std::pair{kUint32, 0ULL}, std::pair{kUint32, 3ULL},
        std::pair{kUint32, 96ULL}, std::pair{kUint64, 1ULL << 32},
        std::pair{kInt32, 32ULL}, std::pair{kBool, 1ULL},
        std::pair{kFloat32, 0ULL}}) {
    auto out = Writer::Header(0, 1);
    out.Key("general.alignment", type);
    out.bytes.insert(
        out.bytes.end(), reinterpret_cast<const std::uint8_t*>(&value),
        reinterpret_cast<const std::uint8_t*>(&value) + kScalarBytes[type]);
    Rejects(out.bytes, "Invalid GGUF alignment",
            "alignment type " + std::to_string(type) + " value " +
                std::to_string(value));
  }
  auto array = Writer::Header(0, 1);
  array.Key("general.alignment", kArray)
      .Pod(kUint32)
      .Pod<std::uint64_t>(1)
      .Pod<std::uint32_t>(32);
  Rejects(array.bytes, "Invalid GGUF alignment", "alignment array");
}

// Core sizes the tensors; their data and index point into the file.
void Tensors() {
  // Q8_0 blocks of 32 take 34 bytes, Q4_K blocks of 256 take 144.
  const auto bytes = Writer::Header(3, 1)
                         .Key("general.alignment", kUint32)
                         .Pod<std::uint32_t>(64)
                         .Tensor("blk.0.attn_q.weight", {32, 2}, 8, 0)
                         .Tensor("b", {3}, 0, 128)
                         .Tensor("c", {256, 1, 1, 2}, 12, 192)
                         .Data(64, 192 + 288)
                         .bytes;
  const auto reader = Opens(bytes, "tensors");
  const auto* data = bytes.data() + bytes.size() - 192 - 288;
  const auto tensors = reader->GetTensors();
  Require(tensors.size() == 3, "tensor count differs");
  const struct {
    std::string_view name;
    std::vector<std::uint64_t> dimensions;
    GgmlType type;
    std::uint64_t offset;
    std::size_t size;
  } expected[] = {
      {"blk.0.attn_q.weight", {32, 2}, GgmlType::kQ8_0, 0, 68},
      {"b", {3}, GgmlType::kF32, 128, 12},
      {"c", {256, 1, 1, 2}, GgmlType::kQ4_K, 192, 288},
  };
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& got = tensors[i];
    const auto& want = expected[i];
    Require(InPlace(got.name, bytes, want.name) &&
                got.dimensions == want.dimensions && got.type == want.type &&
                got.offset == want.offset && got.data == data + want.offset &&
                got.size_bytes == want.size && got.sized &&
                reader->FindTensor(want.name) == &got,
            "tensor " + std::string(want.name) + " differs");
  }
  Require(reader->FindTensor("missing") == nullptr, "a missing tensor found");

  // A storage type core cannot size spans the bytes up to the end, and opens
  // only on request.
  const auto unknown = Writer::Header(2, 0)
                           .Tensor("plain", {8}, 0, 0)
                           .Tensor("unknown", {64}, 200, 32)
                           .Data(32, 32 + 96)
                           .bytes;
  Rejects(unknown,
          "Invalid GGUF tensor shape or unsupported storage type: unknown",
          "an unknown storage type", true);
  const auto lenient = Opens(unknown, "an unknown storage type",
                             {.allow_unsized_tensors = true});
  const auto& unsized = lenient->GetTensors()[1];
  Require(!unsized.sized && unsized.size_bytes == 96 &&
              lenient->GetTensors()[0].sized,
          "an unknown storage type differs");
}

void Rejections() {
  Require(GgufReader::OpenMemory(nullptr, 64, nullptr) == nullptr,
          "a null buffer opens");
  std::string error;
  Require(!GgufReader::OpenMemory(nullptr, 64, &error) &&
              error == "Invalid memory buffer for GGUF",
          "a null buffer fails with " + Escape(error));
  // OpenMemory rejects a buffer shorter than the fixed header itself.
  auto header = Writer::Header(0, 0).bytes;
  Rejects(std::span(header).first(0), "Invalid memory buffer for GGUF",
          "empty");
  Rejects(std::span(header).first(23), "Invalid memory buffer for GGUF",
          "23 bytes");
  header[0] = 'g';
  Rejects(header, "Invalid GGUF magic header", "magic");

  const struct {
    std::string_view what;
    Bytes bytes;
    std::string_view message;
    // Opens with unsized tensors allowed.
    bool strict = false;
  } cases[] = {
      {"counts", Writer::Header(0, 1).bytes,
       "GGUF descriptor counts exceed file or memory limits"},
      {"unreadable key",
       Writer::Header(0, 1).Pod<std::uint64_t>(100).Zeros(5).bytes,
       "Failed to read metadata key"},
      {"empty key", Writer::Header(0, 1).String("").Zeros(5).bytes,
       "GGUF metadata key exceeds limits"},
      {"unreadable type", Writer::Header(0, 1).String("key").Zeros(2).bytes,
       "Failed to read metadata type for key key"},
      {"value type 13", Writer::Header(0, 1).Key("k", 13).Zeros(8).bytes,
       kMalformed},
      // Not narrowed to a string.
      {"value type 0x108", Writer::Header(0, 1).Key("k", 0x108).Zeros(8).bytes,
       kMalformed},
      {"boolean 2",
       Writer::Header(0, 1).Key("k", kBool).Pod<std::uint8_t>(2).Zeros(4).bytes,
       kMalformed},
      {"array of arrays",
       Writer::Header(0, 1)
           .Key("k", kArray)
           .Pod(kArray)
           .Pod<std::uint64_t>(0)
           .bytes,
       kMalformed},
      {"short array",
       Writer::Header(0, 1)
           .Key("k", kArray)
           .Pod(kUint32)
           .Pod<std::uint64_t>(2)
           .Pod<std::uint32_t>(0)
           .bytes,
       kMalformed},
      {"duplicate key",
       Writer::Header(0, 2)
           .Key("k", kUint8)
           .Pod<std::uint8_t>(0)
           .Key("k", kUint8)
           .Pod<std::uint8_t>(0)
           .bytes,
       "Duplicate GGUF metadata key"},
      {"unreadable tensor name",
       Writer::Header(1, 0).Pod<std::uint64_t>(1ULL << 40).Zeros(24).bytes,
       "Failed to read tensor name"},
      {"empty tensor name",
       Writer::Header(1, 0).Tensor("", {32}, 0, 0).Data(32, 128).bytes,
       kMalformed},
      {"rank 0",
       Writer::Header(1, 0).String("t").Pod<std::uint32_t>(0).Zeros(32).bytes,
       "Invalid tensor dimensions count"},
      {"rank 9",
       Writer::Header(1, 0).String("t").Pod<std::uint32_t>(9).Zeros(80).bytes,
       "Invalid tensor dimensions count"},
      // Not narrowed to F32.
      {"storage type 0x10000",
       Writer::Header(1, 0).Tensor("t", {8}, 0x10000, 0).Data(32, 32).bytes,
       kMalformed},
      {"element overflow",
       Writer::Header(1, 0).Tensor("t", {~0ULL, 2}, 0, 0).Data(32, 32).bytes,
       "Invalid GGUF tensor shape or unsupported storage type: t"},
      {"zero dimension",
       Writer::Header(1, 0).Tensor("t", {8, 0}, 0, 0).Data(32, 32).bytes,
       "Invalid GGUF tensor shape or unsupported storage type: t"},
      // Q8_0 rows must be whole 32-element blocks.
      {"partial block",
       Writer::Header(1, 0).Tensor("t", {33}, 8, 0).Data(32, 64).bytes,
       "Invalid GGUF tensor shape or unsupported storage type: t", true},
      {"duplicate tensor",
       Writer::Header(2, 0)
           .Tensor("t", {8}, 0, 0)
           .Tensor("t", {8}, 0, 32)
           .Data(32, 64)
           .bytes,
       "Duplicate GGUF tensor name"},
      {"unpadded", Writer::Header(1, 0).Tensor("t", {8}, 0, 0).bytes,
       "GGUF data section is truncated before its alignment"},
      {"unaligned offset",
       Writer::Header(1, 0).Tensor("t", {8}, 0, 1).Data(32, 64).bytes,
       "Tensor extent or alignment is invalid: t"},
      {"past the end",
       Writer::Header(1, 0).Tensor("t", {8}, 0, 32).Data(32, 32).bytes,
       "Tensor extent or alignment is invalid: t"},
      {"offset overflow",
       Writer::Header(1, 0).Tensor("t", {8}, 0, ~0ULL).Data(32, 32).bytes,
       "Tensor extent or alignment is invalid: t"},
      {"one byte short",
       Writer::Header(1, 0).Tensor("t", {8}, 0, 0).Data(32, 31).bytes,
       "Tensor extent or alignment is invalid: t"},
  };
  for (const auto& test : cases)
    Rejects(test.bytes, test.message, std::string(test.what), test.strict);
  // Allowed, a shape core cannot size spans the data up to the end, but never
  // without elements.
  const auto partial =
      Writer::Header(1, 0).Tensor("t", {33}, 8, 0).Data(32, 64);
  const auto reader =
      Opens(partial.bytes, "partial block", {.allow_unsized_tensors = true});
  const auto& tensor = reader->GetTensors().front();
  Require(!tensor.sized && tensor.size_bytes == 64 &&
              reader->UnsizedTensorError() ==
                  "Invalid GGUF tensor shape or unsupported storage type: t",
          "an allowed partial block differs");
  // A file without tensors needs no padding.
  Opens(Writer::Header(0, 1).Key("k", kUint8).Pod<std::uint8_t>(1).bytes,
        "unpadded without tensors");
}

// Headers whose counts take the descriptor budget to its last bytes, so that
// each charge (the counts, a key, a string, the strings and numbers of an
// array, a tensor name) fits exactly or overruns it by one byte. A metadata
// entry is followed by the first tensor's one-byte name: the parse reaches
// that tensor's dimension count, all zero bytes, only if both fit. 0.35.0
// reported each overrun as a malformed or over-long entry, or as counts
// beyond its limits; the Rust parser names the budget.
void Budget() {
  constexpr std::size_t kBudget = std::size_t{128} << 20;
  constexpr std::size_t kNumberCharge = 8;
  constexpr std::string_view kFits = "Invalid tensor dimensions count";
  const auto metadata_charge = sizeof(GgufMetadataValue) + 64;
  const auto tensor_charge = sizeof(GgufTensorInfo) + 128;
  const auto string_charge = sizeof(std::string_view);
  const std::uint64_t most = (kBudget - metadata_charge) / tensor_charge;
  // What one metadata entry and one tensor fewer than the most leave.
  const std::size_t room =
      kBudget - metadata_charge - ((most - 1) * tensor_charge);
  // The counts must also fit the file: 32 bytes per tensor at least.
  Bytes bytes(24 + ((most + 1) * 32), 0);
  std::size_t cases = 0;
  const auto header = [&](std::uint64_t tensors, std::uint64_t metadata,
                          const Bytes& entry, bool fits) {
    auto out = Writer::Header(tensors, metadata);
    out.bytes.insert(out.bytes.end(), entry.begin(), entry.end());
    std::ranges::fill(bytes, 0);
    std::ranges::copy(out.bytes, bytes.begin());
    Rejects(bytes, fits ? kFits : kOverBudget,
            "budget case " + std::to_string(cases++));
  };
  // A key of `length` bytes and its value type.
  const auto key = [](std::size_t length, std::uint32_t type) {
    Writer out;
    out.Key(std::string(length, 'k'), type);
    return out;
  };
  // Ends a metadata entry with the first tensor's one-byte name.
  const auto named = [](Writer out) { return out.String("t").bytes; };
  // A key of `length` bytes with an empty string.
  const auto keyed = [&](std::size_t length) {
    auto out = key(length, kString);
    out.String("");
    return named(out);
  };
  // A string of `length` bytes.
  const auto string = [&](std::size_t length) {
    auto out = key(1, kString);
    out.String(std::string(length, 's'));
    return named(out);
  };
  // `count` strings, the first `first` bytes long and the others empty.
  const auto strings = [&](std::uint64_t count, std::size_t first) {
    auto out = key(1, kArray);
    out.Pod(kString).Pod(count).String(std::string(first, 's'));
    for (std::uint64_t i = 1; i < count; ++i)
      out.String("");
    return named(out);
  };
  // A key of `key_bytes` bytes with `count` 8-bit numbers, charged in steps
  // of kNumberCharge, so that the key takes up the rest.
  const auto numbers = [&](std::size_t key_bytes, std::uint64_t count) {
    auto out = key(key_bytes, kArray);
    out.Pod(kUint8).Pod(count);
    out.bytes.resize(out.bytes.size() + count, 7);
    return named(out);
  };
  // 0.35.0: descriptor counts exceed file or memory limits.
  header(most + 1, 1, string(0), false);
  header(most - 1, 1, keyed(room - 1), true);
  // 0.35.0: malformed or truncated.
  header(most - 1, 1, keyed(room), false);
  // 0.35.0: metadata key exceeds limits.
  header(most - 1, 1, keyed(room + 1), false);
  header(most - 1, 1, string(room - 2), true);
  header(most - 1, 1, string(room - 1), false);
  header(most - 1, 1, strings(1, room - 2 - string_charge), true);
  header(most - 1, 1, strings(1, room - 1 - string_charge), false);
  const auto count = (room - 2) / string_charge;
  const auto rest = (room - 2) % string_charge;
  header(most - 1, 1, strings(count, rest), true);
  header(most - 1, 1, strings(count, rest + 1), false);
  const auto numbers_count = (room - 2) / kNumberCharge;
  const auto key_bytes = room - 1 - (numbers_count * kNumberCharge);
  header(most - 1, 1, numbers(key_bytes, numbers_count), true);
  header(most - 1, 1, numbers(key_bytes + 1, numbers_count), false);
  // Without metadata, the most tensors leave room for one name.
  const auto name = [](std::size_t length) {
    Writer out;
    return out.String(std::string(length, 't')).bytes;
  };
  const auto name_room = kBudget - (most * tensor_charge);
  header(most, 0, name(name_room), true);
  header(most, 0, name(name_room + 1), false);
}

// Every truncation of a header fails, until only the tail of a tensor core
// cannot size is cut, which then spans what is left. Byte changes either fail
// with a known message or open tensors that lie inside the buffer.
void Damage() {
  const auto bytes = Writer::Header(3, 4)
                         .Key("general.architecture", kString)
                         .String("qwen35")
                         .Key("tokenizer.ggml.tokens", kArray)
                         .Pod(kString)
                         .Pod<std::uint64_t>(3)
                         .String("a")
                         .String("bc")
                         .String("")
                         .Key("general.alignment", kUint8)
                         .Pod<std::uint8_t>(16)
                         .Key("scores", kArray)
                         .Pod(kFloat32)
                         .Pod<std::uint64_t>(2)
                         .Pod(0.5F)
                         .Pod(-1.0F)
                         .Tensor("a", {32}, 8, 0)
                         .Tensor("b", {4}, 0, 48)
                         .Tensor("c", {64}, 200, 64)
                         .Data(16, 96)
                         .bytes;
  // The unsized tensor starts 64 bytes into the 96 bytes of data.
  const auto unsized_start = bytes.size() - 32;
  for (std::size_t len = 0; len <= bytes.size(); ++len) {
    const auto opened =
        Open(std::span(bytes).first(len), {.allow_unsized_tensors = true});
    const auto what =
        std::to_string(len) + " of " + std::to_string(bytes.size()) + " bytes";
    if (len < unsized_start) {
      Require(!opened.reader && KnownMessage(opened.error),
              what + " fail with " + Escape(opened.error));
    } else if (len > unsized_start) {
      Require(
          opened.reader != nullptr &&
              opened.reader->GetTensors()[2].size_bytes == len - unsized_start,
          what + " fail with " + Escape(opened.error));
    }
  }
  std::mt19937_64 random(0x47475546);
  const auto below = [&](std::size_t bound) {
    return std::uniform_int_distribution<std::size_t>(0, bound - 1)(random);
  };
  std::size_t opened_count = 0;
  for (int mutation = 0; mutation < 4000; ++mutation) {
    auto changed = bytes;
    const auto at = below(changed.size());
    switch (below(4)) {
      case 0:
        changed[at] ^= static_cast<std::uint8_t>(1U << below(8));
        break;
      case 1: {
        const std::uint64_t values[] = {0,       1,          9,    13,
                                        0x10000, 1ULL << 32, ~0ULL};
        const auto value = values[below(std::size(values))];
        std::memcpy(changed.data() + at, &value,
                    std::min(sizeof(value), changed.size() - at));
        break;
      }
      case 2:
        changed.insert(changed.begin() + static_cast<std::ptrdiff_t>(at),
                       static_cast<std::uint8_t>(below(256)));
        break;
      default:
        changed.erase(changed.begin() + static_cast<std::ptrdiff_t>(at));
        break;
    }
    for (const bool unsized : {false, true}) {
      const auto opened = Open(changed, {.allow_unsized_tensors = unsized});
      const auto what = "mutation " + std::to_string(mutation);
      if (!opened.reader) {
        Require(KnownMessage(opened.error),
                what + " fails with " + Escape(opened.error));
        continue;
      }
      ++opened_count;
      const auto* begin = changed.data();
      for (const auto& tensor : opened.reader->GetTensors()) {
        const auto* data = static_cast<const std::uint8_t*>(tensor.data);
        Require(data >= begin && tensor.size_bytes <= changed.size() &&
                    data - begin <= static_cast<std::ptrdiff_t>(
                                        changed.size() - tensor.size_bytes) &&
                    opened.reader->FindTensor(tensor.name) == &tensor &&
                    (unsized || tensor.sized),
                what + ": tensor " + Escape(tensor.name) + " is out of place");
      }
    }
  }
  Require(opened_count > 0, "no mutation opened");
}

// Only OpenFile parses a file shorter than the fixed header.
void FileTooSmall() {
  // Removed however the test ends; the name differs per run.
  struct Temporary {
    std::filesystem::path path;
    ~Temporary() {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
  } file{std::filesystem::temp_directory_path() /
         ("gufo-gguf-header-test-" + std::to_string(std::random_device{}()) +
          "-" +
          std::to_string(
              std::chrono::steady_clock::now().time_since_epoch().count()) +
          ".gguf")};
  {
    std::ofstream out(file.path, std::ios::binary);
    out.write("GGUF\3\0\0\0\0\0", 10);
  }
  std::string error;
  const auto reader = GgufReader::OpenFile(file.path, &error);
  Require(!reader && error == "File too small for GGUF header",
          "a 10-byte file fails with: " + Escape(error));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    ValueTypes();
    LongArrays();
    VersionsAndAlignments();
    Tensors();
    Rejections();
    Budget();
    Damage();
    FileTooSmall();

    for (int i = 1; i < argc; ++i) {
      std::string error;
      const auto mapping =
          gufo::platform::ReadOnlyMappedFile::Open(argv[i], &error);
      Require(mapping.valid(), std::string(argv[i]) + ": " + error);
      double best = std::numeric_limits<double>::max();
      for (int run = 0; run < 20; ++run) {
        const auto start = std::chrono::steady_clock::now();
        Opens(std::span(mapping.data(), mapping.size()), argv[i],
              {.allow_unsized_tensors = true});
        const std::chrono::duration<double, std::milli> elapsed =
            std::chrono::steady_clock::now() - start;
        best = std::min(best, elapsed.count());
      }
      std::printf("%s: %.3f ms\n", argv[i], best);
    }
    std::printf("GGUF header checks passed\n");
    return 0;
  } catch (const std::exception& failure) {
    std::fprintf(stderr, "gguf_header_test: %s\n", failure.what());
    return 1;
  }
}

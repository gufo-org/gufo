// Compares the Rust GGUF header parser, through GgufReader::OpenMemory, with
// a frozen copy of the 0.35.0 C++ parser
// (tests/core/gguf_header_reference.cpp). Generated headers cover every
// metadata value and array item type, alignments, tensor ranks, known and
// unknown storage types, unsized tensors and data layouts; deterministic
// mutations truncate, flip, overwrite, insert and erase bytes. Further headers
// hold arrays around the Rust parser's block size and fill the descriptor
// budget to its edge. Each buffer must open with the same version, alignment,
// metadata and tensors, whose strings and data point at the same bytes, or
// fail with the same message, with and without allow_unsized_tensors. Only a
// header over the descriptor budget fails with a message of its own, which
// names the limit, as 0.35.0 took it for a malformed one. Every rejection
// must be reached. GGUF files named on the command line are compared
// too, and both parsers are timed on them.
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
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "tests/core/gguf_header_reference.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::core::GgufMetadataValue;
using gufo::core::GgufOpenOptions;
using gufo::core::GgufReader;
using gufo::core::GgufTensorInfo;
using Bytes = std::vector<std::uint8_t>;
using Random = std::mt19937_64;
namespace reference = gufo::gguf_header_reference;

constexpr std::uint32_t kString = 8;
constexpr std::uint32_t kArray = 9;
// Bytes of a scalar of each value type; strings and arrays have their own.
constexpr std::size_t kScalarBytes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
constexpr std::uint32_t kKnownTypes[] = {0,  1,  2,  3,  6,  7,  8,  9,  10,
                                         11, 12, 13, 14, 15, 16, 17, 18, 19,
                                         20, 21, 22, 23, 26, 29, 30};
// Storage types core cannot size.
constexpr std::uint32_t kOtherTypes[] = {4, 5, 24, 25, 27, 28, 31, 200, 65535};
// What the Rust parser reports for a header over the descriptor budget,
// wherever 0.35.0 reported it.
constexpr std::string_view kOverBudget =
    "GGUF metadata and tensor index exceed the header limit of 128 MiB";
// Every outcome of 0.35.0's OpenMemory; names are cut off.
constexpr std::string_view kOutcomes[] = {
    "opened",
    "opened with unsized tensors",
    "Invalid memory buffer for GGUF",
    "Invalid GGUF magic header",
    "Unsupported GGUF version: ",
    "GGUF descriptor counts exceed file or memory limits",
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
    "Malformed or truncated GGUF",
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

bool SameDouble(double a, double b) {
  return std::memcmp(&a, &b, sizeof(a)) == 0;
}

// The same bytes, not merely equal ones.
bool SameView(std::string_view a, std::string_view b) {
  return a.data() == b.data() && a.size() == b.size();
}

bool SameValue(const GgufMetadataValue& got, const GgufMetadataValue& want) {
  if (got.type != want.type || got.value.index() != want.value.index())
    return false;
  return std::visit(
      [&](const auto& expected) {
        using T = std::decay_t<decltype(expected)>;
        const auto& actual = std::get<T>(got.value);
        if constexpr (std::is_same_v<T, double>) {
          return SameDouble(actual, expected);
        } else if constexpr (std::is_same_v<T, std::string_view>) {
          return SameView(actual, expected);
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
          return std::ranges::equal(actual, expected, SameDouble);
        } else if constexpr (std::is_same_v<T, std::vector<std::string_view>>) {
          return std::ranges::equal(actual, expected, SameView);
        } else {
          return actual == expected;
        }
      },
      want.value);
}

bool SameTensor(const GgufTensorInfo& a, const GgufTensorInfo& b) {
  return SameView(a.name, b.name) && a.dimensions == b.dimensions &&
         a.type == b.type && a.offset == b.offset && a.data == b.data &&
         a.size_bytes == b.size_bytes && a.sized == b.sized;
}

std::string Outcome(const reference::Result& result) {
  if (result.opened) {
    const bool unsized = std::ranges::any_of(
        result.tensors, [](const auto& tensor) { return !tensor.sized; });
    return unsized ? "opened with unsized tensors" : "opened";
  }
  for (const auto prefix : kOutcomes) {
    if (prefix.ends_with(' ') && result.error.starts_with(prefix))
      return std::string(prefix);
  }
  return result.error;
}

struct Totals {
  std::size_t compared = 0;
  std::map<std::string, std::size_t> outcomes;
};

// `over_budget` marks a header that 0.35.0 rejects for its descriptor
// budget.
void Compare(Totals& totals, std::span<const std::uint8_t> bytes,
             GgufOpenOptions options, const std::string& what,
             bool over_budget) {
  const auto expected =
      reference::OpenMemory(bytes.data(), bytes.size(), options);
  std::string error;
  const auto reader =
      GgufReader::OpenMemory(bytes.data(), bytes.size(), &error, options);
  ++totals.compared;
  ++totals.outcomes[Outcome(expected)];
  const auto fail = [&](const std::string& detail) {
    throw std::runtime_error(
        what + " (" + std::to_string(bytes.size()) + " bytes" +
        (options.allow_unsized_tensors ? ", unsized allowed" : "") +
        "): " + detail);
  };
  if (over_budget && expected.opened)
    fail("0.35.0 opened a header over the descriptor budget");
  if (!expected.opened) {
    if (reader)
      fail("opens; 0.35.0 failed with " + Escape(expected.error));
    const std::string want =
        over_budget ? std::string(kOverBudget) : expected.error;
    if (error != want)
      fail("fails with " + Escape(error) + " instead of " + Escape(want));
    return;
  }
  if (!reader)
    fail("fails with " + Escape(error) + "; 0.35.0 opened it");
  if (reader->GetVersion() != expected.version ||
      reader->GetAlignment() != expected.alignment)
    fail("version or alignment differs");
  if (reader->GetMetadataCount() != expected.metadata.size())
    fail("metadata count differs");
  for (const auto& [key, value] : expected.metadata) {
    const auto* got = reader->FindMetadata(key);
    if (got == nullptr || !SameValue(*got, value))
      fail("metadata " + Escape(key) + " differs");
  }
  const auto tensors = reader->GetTensors();
  if (tensors.size() != expected.tensors.size())
    fail("tensor count differs");
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (!SameTensor(tensors[i], expected.tensors[i]) ||
        reader->FindTensor(tensors[i].name) != &tensors[i])
      fail("tensor " + Escape(expected.tensors[i].name) + " differs");
  }
}

void CompareBoth(Totals& totals, std::span<const std::uint8_t> bytes,
                 const std::string& what, bool over_budget = false) {
  Compare(totals, bytes, {}, what, over_budget);
  Compare(totals, bytes, {.allow_unsized_tensors = true}, what, over_budget);
}

std::uint64_t Below(Random& random, std::uint64_t bound) {
  return std::uniform_int_distribution<std::uint64_t>(0, bound - 1)(random);
}

bool Chance(Random& random, std::uint64_t percent) {
  return Below(random, 100) < percent;
}

template<class T>
T Pick(Random& random, std::span<const T> items) {
  return items[Below(random, items.size())];
}

template<class T>
T Pick(Random& random, std::initializer_list<T> items) {
  return Pick(random, std::span<const T>(items.begin(), items.size()));
}

class Writer {
public:
  template<class T>
  void Pod(T value) {
    const auto* data = reinterpret_cast<const std::uint8_t*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
  }

  void String(std::string_view text) {
    Pod<std::uint64_t>(text.size());
    bytes.insert(bytes.end(), text.begin(), text.end());
  }

  void Noise(Random& random, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i)
      bytes.push_back(static_cast<std::uint8_t>(Below(random, 256)));
  }

  Bytes bytes;
};

// Mostly name-like text, sometimes any bytes.
std::string Text(Random& random, std::size_t most) {
  constexpr std::string_view kNameBytes = "abcdefghijklmnopqrstuvwxyz._0123";
  const bool raw = Chance(random, 10);
  std::string text(Below(random, most + 1), '\0');
  for (auto& c : text) {
    c = raw ? static_cast<char>(Below(random, 256))
            : kNameBytes[Below(random, kNameBytes.size())];
  }
  return text;
}

// Draws generated headers. A clean header has no errors injected, so that
// it opens and every value and tensor in it is compared.
struct Draw {
  Random& random;
  bool clean = false;

  // An injected error, never in a clean header.
  bool Odd(std::uint64_t percent) { return !clean && Chance(random, percent); }
};

// A value of one type that is not an array; booleans are sometimes 2.
void Scalar(Writer& out, Draw& draw, std::uint32_t type) {
  if (type == kString) {
    out.String(Text(draw.random, 24));
  } else if (type == 7) {
    out.Pod<std::uint8_t>(draw.Odd(5) ? 2 : Below(draw.random, 2));
  } else if (type < std::size(kScalarBytes)) {
    out.Noise(draw.random, kScalarBytes[type]);
  } else {
    out.Noise(draw.random, Below(draw.random, 9));
  }
}

void Value(Writer& out, Draw& draw, std::uint32_t type) {
  out.Pod(type);
  if (type != kArray) {
    Scalar(out, draw, type);
    return;
  }
  auto item = static_cast<std::uint32_t>(Below(draw.random, 13));
  if (item == kArray || draw.Odd(4))
    item = draw.clean ? kString : Pick(draw.random, {kArray, 13U, 0x108U});
  const auto count = Chance(draw.random, 10) ? 0 : Below(draw.random, 12);
  out.Pod(item);
  out.Pod<std::uint64_t>(count);
  for (std::uint64_t i = 0; i < count; ++i)
    Scalar(out, draw, item);
}

// A general.alignment entry, valid when clean; sets the alignment it
// declares.
void Alignment(Writer& out, Draw& draw, std::uint64_t& alignment) {
  out.String("general.alignment");
  auto& random = draw.random;
  const auto power = std::uint64_t{1} << Below(random, 9);
  switch (draw.clean ? Pick<std::uint64_t>(random, {0, 1, 2, 9})
                     : Below(random, 10)) {
    case 0:
      out.Pod<std::uint32_t>(0);
      out.Pod(static_cast<std::uint8_t>(power));
      alignment = power;
      return;
    case 1:
      out.Pod<std::uint32_t>(2);
      out.Pod(static_cast<std::uint16_t>(power));
      alignment = power;
      return;
    case 2:
      out.Pod<std::uint32_t>(10);
      out.Pod<std::uint64_t>(4096);
      alignment = 4096;
      return;
    case 3:
      out.Pod<std::uint32_t>(4);
      out.Pod(Pick<std::uint32_t>(random, {0, 3, 96, 0xFFFFFFFF}));
      return;
    case 4:
      out.Pod<std::uint32_t>(10);
      out.Pod(Pick<std::uint64_t>(
          random, {std::uint64_t{1} << 32, std::uint64_t{1} << 63}));
      return;
    case 5:
      Value(out, draw, Pick<std::uint32_t>(random, {1, 5, 6, 7, 8, 11}));
      return;
    case 6:
      out.Pod<std::uint32_t>(kArray);
      out.Pod<std::uint32_t>(4);
      out.Pod<std::uint64_t>(1);
      out.Pod<std::uint32_t>(32);
      return;
    default:
      out.Pod<std::uint32_t>(4);
      out.Pod(static_cast<std::uint32_t>(power));
      alignment = power;
      return;
  }
}

struct Tensor {
  std::string name;
  std::vector<std::uint64_t> dimensions;
  std::uint32_t type = 0;
  std::uint64_t size = 0;
  std::uint64_t offset = 0;
};

// Bytes as core sizes them, or a few for a layout it cannot size.
std::uint64_t PayloadBytes(Draw& draw, const Tensor& tensor) {
  std::uint64_t elements = 1;
  for (const auto dimension : tensor.dimensions) {
    if (dimension == 0 ||
        elements > std::numeric_limits<std::uint64_t>::max() / dimension) {
      elements = 0;
      break;
    }
    elements *= dimension;
  }
  const auto type = static_cast<GgmlType>(tensor.type);
  const auto block = gufo::quant::QuantizedBlockElements(type);
  const auto bytes =
      tensor.type > 0xFFFF || tensor.dimensions.empty() ||
              (block != 0 && tensor.dimensions.front() % block != 0)
          ? 0
          : gufo::quant::EncodedSizeBytes(type, elements);
  return bytes != 0 ? bytes : (draw.clean ? 1 : 0) + Below(draw.random, 80);
}

std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

Bytes Generate(Random& random) {
  Draw draw{random, Chance(random, 50)};
  Writer out;
  out.bytes = {'G', 'G', 'U', 'F'};
  out.Pod(draw.Odd(3)          ? Pick<std::uint32_t>(random, {1, 4})
          : Chance(random, 20) ? std::uint32_t{2}
                               : std::uint32_t{3});
  const auto metadata_count = Below(random, 10);
  const auto tensor_count = Below(random, 7);
  out.Pod(tensor_count);
  out.Pod(metadata_count);

  std::uint64_t alignment = 32;
  bool aligned = false;
  std::vector<std::string> keys;
  constexpr std::string_view kKeys[] = {"general.architecture", "general.name",
                                        "tokenizer.ggml.tokens",
                                        "qwen35.block_count", "split.count"};
  for (std::uint64_t i = 0; i < metadata_count; ++i) {
    if (Chance(random, 15) && !(draw.clean && aligned)) {
      Alignment(out, draw, alignment);
      aligned = true;
      continue;
    }
    std::string key =
        std::string(Pick(random, std::span<const std::string_view>(kKeys))) +
        "." + std::to_string(i);
    if (draw.Odd(3) && !keys.empty())
      key = keys[Below(random, keys.size())];
    else if (draw.Odd(3))
      key = "";
    else if (draw.Odd(3))
      key = Text(random, 12);
    keys.push_back(key);
    out.String(key);
    const auto type = draw.Odd(2) ? Pick<std::uint32_t>(random, {13, 0x108})
                      : Chance(random, 30)
                          ? kArray
                          : static_cast<std::uint32_t>(Below(random, 13));
    Value(out, draw, type);
  }

  std::vector<Tensor> tensors(tensor_count);
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    auto& tensor = tensors[i];
    tensor.name = "blk." + std::to_string(i) + ".weight";
    if (draw.Odd(3))
      tensor.name = "";
    else if (draw.Odd(3))
      tensor.name = Text(random, 10);
    else if (i > 0 && draw.Odd(3))
      tensor.name = tensors[i - 1].name;
    tensor.type =
        draw.Odd(3) ? Pick<std::uint32_t>(random, {65536, 0x10008})
        : Chance(random, 12)
            ? Pick(random, std::span<const std::uint32_t>(kOtherTypes))
            : Pick(random, std::span<const std::uint32_t>(kKnownTypes));
    const auto block =
        gufo::quant::QuantizedBlockElements(static_cast<GgmlType>(tensor.type));
    const auto rank = draw.Odd(3) ? Pick<std::size_t>(random, {0, 9, 12})
                                  : 1 + Below(random, 4);
    for (std::size_t d = 0; d < rank; ++d) {
      tensor.dimensions.push_back(
          draw.Odd(2) ? Pick<std::uint64_t>(
                            random, {0, std::uint64_t{1} << 32,
                                     std::uint64_t{1} << 63, ~std::uint64_t{0}})
          : d != 0 ? 1 + Below(random, 3)
          : block != 0 && Chance(random, 70)
              ? block * (1 + Below(random, 2))
              : Pick<std::uint64_t>(
                    random, {1, 2, 3, 7, 8, 31, 32, 33, 64, 100, 256, 512}));
    }
    tensor.size = PayloadBytes(draw, tensor);
  }
  // Lay the data out in a random order, now and then unaligned, overlapping
  // or far outside the file.
  std::vector<std::size_t> order(tensors.size());
  for (std::size_t i = 0; i < order.size(); ++i)
    order[i] = i;
  if (Chance(random, 30))
    std::shuffle(order.begin(), order.end(), random);
  std::uint64_t end = 0;
  std::uint64_t previous = 0;
  for (const auto index : order) {
    auto& tensor = tensors[index];
    tensor.offset = AlignUp(end, alignment);
    if (draw.Odd(3))
      tensor.offset += 1 + Below(random, 3);
    if (draw.Odd(3))
      tensor.offset = previous;
    previous = tensor.offset;
    if (draw.Odd(2)) {
      tensor.offset = Pick<std::uint64_t>(
          random, {std::uint64_t{1} << 40, ~std::uint64_t{0}});
    } else if (tensor.size < (std::uint64_t{1} << 20)) {
      end = std::max(end, tensor.offset + tensor.size);
    }
  }
  for (const auto& tensor : tensors) {
    out.String(tensor.name);
    out.Pod(static_cast<std::uint32_t>(tensor.dimensions.size()));
    for (const auto dimension : tensor.dimensions)
      out.Pod(dimension);
    out.Pod(tensor.type);
    out.Pod(tensor.offset);
  }
  if (!draw.Odd(4))
    out.bytes.resize(AlignUp(out.bytes.size(), alignment), 0);
  auto payload = end;
  if (Chance(random, 10))
    payload += Below(random, 40);
  else if (payload > 0 && draw.Odd(5))
    payload -= 1 + Below(random, std::min<std::uint64_t>(payload, 8));
  out.bytes.resize(out.bytes.size() + payload, 0xAB);
  return std::move(out.bytes);
}

Bytes Mutate(Random& random, Bytes bytes) {
  if (bytes.empty())
    return bytes;
  const auto at = Below(random, bytes.size());
  switch (Below(random, 5)) {
    case 0:
      bytes.resize(at);
      break;
    case 1:
      for (auto flips = 1 + Below(random, 3); flips > 0; --flips) {
        bytes[Below(random, bytes.size())] ^=
            static_cast<std::uint8_t>(1U << Below(random, 8));
      }
      break;
    case 2: {
      const auto value = Pick<std::uint64_t>(
          random, {0, 1, 2, 8, 9, 13, 32, 0xFFFF, 0x10000,
                   std::uint64_t{1} << 32, bytes.size(), bytes.size() / 13,
                   bytes.size() / 32, ~std::uint64_t{0}});
      std::memcpy(bytes.data() + at, &value,
                  std::min<std::size_t>(sizeof(value), bytes.size() - at));
      break;
    }
    case 3: {
      Bytes noise(1 + Below(random, 4));
      for (auto& byte : noise)
        byte = static_cast<std::uint8_t>(Below(random, 256));
      bytes.insert(bytes.begin() + static_cast<std::ptrdiff_t>(at),
                   noise.begin(), noise.end());
      break;
    }
    default: {
      const auto count =
          std::min<std::uint64_t>(1 + Below(random, 4), bytes.size() - at);
      bytes.erase(bytes.begin() + static_cast<std::ptrdiff_t>(at),
                  bytes.begin() + static_cast<std::ptrdiff_t>(at + count));
      break;
    }
  }
  return bytes;
}

// Headers whose counts take the descriptor budget to its last bytes, so that
// each charge (the counts, a key, a string, the strings and numbers of an
// array, a tensor name) fits exactly or overruns it by one byte. A metadata
// entry is followed by the first tensor's one-byte name: the parse reaches
// that tensor's dimension count, all zero bytes, only if both fit. 0.35.0
// reported each overrun as a malformed or over-long entry; the Rust parser
// names the budget.
void Budget(Totals& totals) {
  constexpr std::size_t kBudget = std::size_t{128} << 20;
  constexpr std::size_t kNumberCharge = 8;
  constexpr std::string_view kFits = "Invalid tensor dimensions count";
  constexpr std::string_view kOver = "Malformed or truncated GGUF";
  const auto metadata_charge = sizeof(GgufMetadataValue) + 64;
  const auto tensor_charge = sizeof(GgufTensorInfo) + 128;
  const auto string_charge = sizeof(std::string_view);
  const std::uint64_t most = (kBudget - metadata_charge) / tensor_charge;
  // What one metadata entry and one tensor fewer than the most leave.
  const std::size_t room =
      kBudget - metadata_charge - ((most - 1) * tensor_charge);
  // The counts must also fit the file: 32 bytes per tensor at least.
  Bytes bytes(24 + ((most + 1) * 32), 0);
  const auto header = [&](std::uint64_t tensors, std::uint64_t metadata,
                          const Bytes& entry, std::string_view outcome) {
    Writer out;
    out.bytes = {'G', 'G', 'U', 'F'};
    out.Pod<std::uint32_t>(3);
    out.Pod(tensors);
    out.Pod(metadata);
    out.bytes.insert(out.bytes.end(), entry.begin(), entry.end());
    std::ranges::fill(bytes, 0);
    std::ranges::copy(out.bytes, bytes.begin());
    // The case must end at the edge it was built for.
    const auto expected = reference::OpenMemory(bytes.data(), bytes.size(), {});
    Require(Outcome(expected) == outcome,
            "a budget case ends with " + Escape(Outcome(expected)) +
                " instead of " + std::string(outcome));
    CompareBoth(totals, bytes, "budget", outcome != kFits);
  };
  // A key of `length` bytes and its value type.
  const auto key = [](std::size_t length, std::uint32_t type) {
    Writer out;
    out.String(std::string(length, 'k'));
    out.Pod(type);
    return out;
  };
  // Ends a metadata entry with the first tensor's one-byte name.
  const auto named = [](Writer out) {
    out.String("t");
    return out.bytes;
  };
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
    out.Pod(kString);
    out.Pod(count);
    out.String(std::string(first, 's'));
    for (std::uint64_t i = 1; i < count; ++i)
      out.String("");
    return named(out);
  };
  // A key of `key_bytes` bytes with `count` 8-bit numbers, charged in steps
  // of kNumberCharge, so that the key takes up the rest.
  const auto numbers = [&](std::size_t key_bytes, std::uint64_t count) {
    auto out = key(key_bytes, kArray);
    out.Pod<std::uint32_t>(0);  // uint8
    out.Pod(count);
    out.bytes.resize(out.bytes.size() + count, 7);
    return named(out);
  };
  header(most + 1, 1, string(0),
         "GGUF descriptor counts exceed file or memory limits");
  header(most - 1, 1, keyed(room - 1), kFits);
  header(most - 1, 1, keyed(room), kOver);
  header(most - 1, 1, keyed(room + 1), "GGUF metadata key exceeds limits");
  header(most - 1, 1, string(room - 2), kFits);
  header(most - 1, 1, string(room - 1), kOver);
  header(most - 1, 1, strings(1, room - 2 - string_charge), kFits);
  header(most - 1, 1, strings(1, room - 1 - string_charge), kOver);
  const auto count = (room - 2) / string_charge;
  const auto rest = (room - 2) % string_charge;
  header(most - 1, 1, strings(count, rest), kFits);
  header(most - 1, 1, strings(count, rest + 1), kOver);
  const auto numbers_count = (room - 2) / kNumberCharge;
  const auto key_bytes = room - 1 - (numbers_count * kNumberCharge);
  header(most - 1, 1, numbers(key_bytes, numbers_count), kFits);
  header(most - 1, 1, numbers(key_bytes + 1, numbers_count), kOver);
  // Without metadata, the most tensors leave room for one name.
  const auto name = [](std::size_t length) {
    Writer out;
    out.String(std::string(length, 't'));
    return out.bytes;
  };
  const auto name_room = kBudget - (most * tensor_charge);
  header(most, 0, name(name_room), kFits);
  header(most, 0, name(name_room + 1), kOver);
}

// Arrays of every item type around the Rust parser's blocks of 32768 items,
// whole and cut off in their second block.
void LongArrays(Totals& totals) {
  constexpr std::size_t kBlock = 32768;
  for (std::uint32_t item = 0; item < std::size(kScalarBytes); ++item) {
    if (item == kArray)
      continue;
    for (const std::size_t count :
         {kBlock - 1, kBlock, kBlock + 1, (3 * kBlock) + 5}) {
      Writer out;
      out.bytes = {'G', 'G', 'U', 'F'};
      out.Pod<std::uint32_t>(3);
      out.Pod<std::uint64_t>(0);
      out.Pod<std::uint64_t>(2);
      out.String("long");
      out.Pod(kArray);
      out.Pod(item);
      out.Pod<std::uint64_t>(count);
      Random random(item);
      std::size_t cut = 0;
      for (std::size_t i = 0; i < count; ++i) {
        if (i == kBlock + 1)
          cut = out.bytes.size() + 1;
        if (item == kString) {
          out.String(std::to_string(i));
        } else if (item == 7) {
          out.Pod<std::uint8_t>(i % 2);
        } else {
          out.Noise(random, kScalarBytes[item]);
        }
      }
      out.String("after");
      out.Pod<std::uint32_t>(4);
      out.Pod<std::uint32_t>(7);
      const auto what = "array of " + std::to_string(count) +
                        " items of type " + std::to_string(item);
      CompareBoth(totals, out.bytes, what);
      if (cut == 0)
        cut = out.bytes.size() / 2;
      CompareBoth(totals, std::span(out.bytes).first(cut), what + " cut off");
    }
  }
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

template<class F>
double BestMilliseconds(F&& open) {
  double best = std::numeric_limits<double>::max();
  for (int run = 0; run < 20; ++run) {
    const auto start = std::chrono::steady_clock::now();
    open();
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - start;
    best = std::min(best, elapsed.count());
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Totals totals;
    Random random(0x47475546);
    for (int file = 0; file < 1500; ++file) {
      const auto bytes = Generate(random);
      const auto what = "file " + std::to_string(file);
      CompareBoth(totals, bytes, what);
      if (file < 100) {
        for (std::size_t len = 0; len < bytes.size(); ++len) {
          CompareBoth(totals, std::span(bytes).first(len), what + " truncated");
        }
      }
      for (int mutation = 0; mutation < 8; ++mutation)
        CompareBoth(totals, Mutate(random, bytes), what + " mutated");
    }
    Budget(totals);
    LongArrays(totals);
    FileTooSmall();
    for (const auto outcome : kOutcomes) {
      Require(totals.outcomes.contains(std::string(outcome)),
              "never reached: " + std::string(outcome));
    }

    for (int i = 1; i < argc; ++i) {
      std::string error;
      const auto mapping =
          gufo::platform::ReadOnlyMappedFile::Open(argv[i], &error);
      Require(mapping.valid(), std::string(argv[i]) + ": " + error);
      const std::span bytes(mapping.data(), mapping.size());
      CompareBoth(totals, bytes, argv[i]);
      const GgufOpenOptions unsized{.allow_unsized_tensors = true};
      const auto rust = BestMilliseconds([&] {
        Require(GgufReader::OpenMemory(bytes.data(), bytes.size(), nullptr,
                                       unsized) != nullptr,
                "open");
      });
      const auto frozen = BestMilliseconds([&] {
        Require(
            reference::OpenMemory(bytes.data(), bytes.size(), unsized).opened,
            "open");
      });
      std::printf("%s: Rust %.3f ms, 0.35.0 %.3f ms\n", argv[i], rust, frozen);
    }

    std::printf("GGUF header checks passed: %zu comparisons;", totals.compared);
    for (const auto& [outcome, count] : totals.outcomes)
      std::printf(" %zu %s;", count, Escape(outcome).c_str());
    std::printf("\n");
    return 0;
  } catch (const std::exception& failure) {
    std::fprintf(stderr, "gguf_header_test: %s\n", failure.what());
    return 1;
  }
}

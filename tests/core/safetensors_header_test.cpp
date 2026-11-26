// Compares the safetensors header reader (src/core/safetensors) with frozen
// copies of the six parsers it replaced (safetensors_reference.hpp), on
// generated headers, every truncation and deterministic mutations. A file
// the reader accepts must give each loader the tensors its old parser gave.
// The reader may reject a file an old parser accepted only for the stricter
// rules each loader takes on, listed in Stricter(); any other difference
// fails. It also checks the C++ interface itself. Safetensors files given as
// arguments are compared and timed too.
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "src/core/platform/mapped_file.hpp"
#include "src/core/safetensors.hpp"
#include "tests/core/safetensors_reference.hpp"

namespace {

namespace st = gufo::safetensors;
namespace ref = gufo::safetensors::reference;
using Bytes = std::vector<std::byte>;

int failures = 0;

void Fail(const std::string& message) {
  if (++failures <= 25)
    std::cerr << "FAIL: " << message << '\n';
}

void Check(bool condition, const std::string& what) {
  if (!condition)
    Fail(what);
}

Bytes File(std::uint64_t length, std::string_view header,
           std::uint64_t payload) {
  Bytes file(8 + header.size() + payload);
  for (unsigned index = 0; index < 8; ++index)
    file[index] = static_cast<std::byte>(length >> (index * 8U));
  std::memcpy(file.data() + 8, header.data(), header.size());
  for (std::uint64_t index = 0; index < payload; ++index)
    file[8 + header.size() + index] = static_cast<std::byte>(index * 7 + 1);
  return file;
}

Bytes File(std::string_view header, std::uint64_t payload) {
  return File(header.size(), header, payload);
}

std::string Entry(std::string_view dtype, std::string_view shape,
                  std::string_view offsets) {
  std::string entry = "{";
  for (auto [key, value] : {std::pair{"dtype", dtype},
                            {"shape", shape},
                            {"data_offsets", offsets}}) {
    if (value.empty())
      continue;
    if (entry.size() > 1)
      entry += ',';
    entry.append("\"").append(key).append("\":").append(value);
  }
  return entry + "}";
}

std::string Offsets(std::uint64_t begin, std::uint64_t end) {
  return "[" + std::to_string(begin) + "," + std::to_string(end) + "]";
}

// The reader's view, and what each loader makes of it.
struct Fresh {
  std::optional<st::Header> header;
  st::Reason reason{st::Reason::kFile};
  std::string name;
};

Fresh Run(std::span<const std::byte> file) {
  Fresh fresh;
  try {
    fresh.header = st::Parse(file);
  } catch (const st::Error& error) {
    fresh.reason = error.reason();
    fresh.name = error.name();
  }
  return fresh;
}

enum class Loader {
  kLora,
  kEmbeddingGemma2,
  kQwen3Asr,
  kQwen3Tts,
  kQwenImage,
  kMinimax
};
constexpr std::array kLoaders{Loader::kLora,      Loader::kEmbeddingGemma2,
                              Loader::kQwen3Asr,  Loader::kQwen3Tts,
                              Loader::kQwenImage, Loader::kMinimax};

std::string_view Label(Loader loader) {
  switch (loader) {
    case Loader::kLora:
      return "LoRA";
    case Loader::kEmbeddingGemma2:
      return "EmbeddingGemma 2";
    case Loader::kQwen3Asr:
      return "Qwen3-ASR";
    case Loader::kQwen3Tts:
      return "Qwen3-TTS";
    case Loader::kQwenImage:
      return "Qwen-Image";
    case Loader::kMinimax:
      return "MiniMax H3";
  }
  return "?";
}

ref::Result Reference(Loader loader, std::span<const std::byte> file) {
  switch (loader) {
    case Loader::kLora:
      return ref::lora::Parse(file);
    case Loader::kEmbeddingGemma2:
      return ref::embedding_gemma2::Parse(file);
    case Loader::kQwen3Asr:
      return ref::qwen3::Parse(file, false);
    case Loader::kQwen3Tts:
      return ref::qwen3::Parse(file, true);
    case Loader::kQwenImage:
      return ref::qwen_image::Parse(file);
    case Loader::kMinimax:
      return ref::minimax::Parse(file);
  }
  return {};
}

bool Among(std::string_view name,
           std::initializer_list<std::string_view> names) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

// What each ported loader does with the reader's tensors: the types it
// takes, the tensors it skips, and tensors without elements, which Qwen3-ASR
// and Qwen3-TTS skip, EmbeddingGemma 2 ignores outside its towers and the
// others reject.
ref::Result Policy(Loader loader, const st::Header& header) {
  ref::Result result;
  for (const auto& tensor : header.tensors) {
    const auto dtype = st::Name(tensor.dtype);
    ref::Tensor kept{.name = tensor.name,
                     .dtype = std::string(dtype),
                     .shape = tensor.shape,
                     .begin = tensor.begin,
                     .end = tensor.end};
    switch (loader) {
      case Loader::kLora:
        if (!Among(dtype, {"BF16", "F16", "F32"}))
          return ref::Failed("unsupported adapter dtype");
        if (tensor.begin == tensor.end)
          return ref::Failed("tensor has no elements");
        break;
      case Loader::kEmbeddingGemma2:
        if (!(kept.name.starts_with("language_model.") ||
              kept.name.starts_with("vision_tower.") ||
              kept.name.starts_with("embed_vision.") ||
              kept.name.starts_with("audio_tower.") ||
              kept.name.starts_with("embed_audio.")))
          continue;
        if (!Among(dtype, {"F32", "BF16", "F16"}))
          return ref::Failed("unsupported dtype");
        if (tensor.begin == tensor.end)
          return ref::Failed("tensor has no elements");
        break;
      case Loader::kQwen3Asr:
      case Loader::kQwen3Tts:
        if (!Among(dtype, {"F32", "BF16", "F16", "I64", "I32", "U8"}) ||
            tensor.begin == tensor.end)
          continue;
        break;
      case Loader::kQwenImage:
        if (!Among(dtype, {"BF16", "F32"}))
          return ref::Failed("unsupported Qwen-Image weight dtype");
        if (tensor.begin == tensor.end)
          return ref::Failed("tensor has no elements");
        break;
      case Loader::kMinimax:
        if (!Among(dtype, {"BF16", "F32", "I32"}))
          return ref::Failed("unsupported H3 tensor dtype");
        if (tensor.begin == tensor.end)
          return ref::Failed("tensor has no elements");
        break;
    }
    result.tensors.push_back(std::move(kept));
  }
  if ((loader == Loader::kQwen3Asr || loader == Loader::kQwen3Tts) &&
      result.tensors.empty())
    return ref::Failed("no usable tensor");
  if (loader == Loader::kMinimax)
    std::sort(result.tensors.begin(), result.tensors.end(),
              [](const auto& a, const auto& b) { return a.name < b.name; });
  if (loader == Loader::kLora)
    for (const auto& [key, value] : header.metadata)
      if (key == "lora_adapter_metadata")
        result.adapter_metadata = value;
  result.ok = true;
  return result;
}

// The rules each loader newly takes on, as reasons the reader gives.
const std::set<st::Reason>& Stricter(Loader loader) {
  using enum st::Reason;
  // Integer tokens only, string metadata, and LoRA's 2^40 bound is moot.
  static const std::set<st::Reason> lora{kMetadata, kDimension, kExtent};
  // Also no header without tensors.
  static const std::set<st::Reason> image{kMetadata, kDimension, kExtent,
                                          kEmpty};
  // Every entry is checked instead of skipped, the payload is tiled, with
  // tensors without elements between the others, and the header length
  // capped.
  static const std::set<st::Reason> lenient{
      kLength, kMetadata, kEntry, kDtype,    kLayout, kDimension,
      kExtent, kOverlap,  kGap,   kTrailing, kEmpty};
  // At most 5 dimensions of at most 2^31 - 1, no "-0", and the payload is
  // tiled, besides string metadata and a nonempty header.
  static const std::set<st::Reason> minimax{kLength,    kMetadata, kLayout,
                                            kDimension, kExtent,   kGap,
                                            kTrailing,  kEmpty};
  switch (loader) {
    case Loader::kLora:
      return lora;
    case Loader::kQwenImage:
      return image;
    case Loader::kMinimax:
      return minimax;
    default:
      return lenient;
  }
}

bool Same(const ref::Result& a, const ref::Result& b) {
  if (a.tensors.size() != b.tensors.size() ||
      a.adapter_metadata != b.adapter_metadata)
    return false;
  for (std::size_t index = 0; index < a.tensors.size(); ++index) {
    const auto& x = a.tensors[index];
    const auto& y = b.tensors[index];
    if (x.name != y.name || x.dtype != y.dtype || x.shape != y.shape ||
        x.begin != y.begin || x.end != y.end)
      return false;
  }
  return true;
}

// Every type the reader knows, in value order.
std::vector<st::Dtype> Dtypes() {
  std::vector<st::Dtype> dtypes;
  for (unsigned index = 0; !st::Name(static_cast<st::Dtype>(index)).empty();
       ++index)
    dtypes.push_back(static_cast<st::Dtype>(index));
  return dtypes;
}

// The bits a type token names, or 0 for any other token.
std::uint64_t TypeBits(std::string_view token) {
  for (const auto dtype : Dtypes()) {
    if (token == "\"" + std::string(st::Name(dtype)) + "\"")
      return st::Bits(dtype);
  }
  return 0;
}

std::uint64_t LengthOf(std::span<const std::byte> file) {
  std::uint64_t length = 0;
  for (unsigned index = 0; index < 8; ++index)
    length |=
        static_cast<std::uint64_t>(std::to_integer<unsigned char>(file[index]))
        << (index * 8U);
  return length;
}

gufo::json::Value HeaderOf(std::span<const std::byte> file) {
  return gufo::json::parse(
      std::string_view(reinterpret_cast<const char*>(file.data()) + 8,
                       static_cast<std::size_t>(LengthOf(file))));
}

// A number spelled with digits only that fits 64 bits.
std::optional<std::uint64_t> Plain(const gufo::json::Value& value) {
  if (!value.is_number())
    return std::nullopt;
  const auto token = value.dump(gufo::json::Value::Numbers::kAsParsed);
  std::uint64_t parsed = 0;
  const auto* last = token.data() + token.size();
  const auto result = std::from_chars(token.data(), last, parsed);
  if (token.find_first_not_of("0123456789") != std::string::npos ||
      result.ec != std::errc{} || result.ptr != last)
    return std::nullopt;
  return parsed;
}

bool KnownType(const gufo::json::Value* dtype) {
  return dtype && dtype->is_string() && TypeBits("\"" + dtype->str() + "\"");
}

// The bits of an entry's shape, saturated above 64 bits, unless a dimension
// is not a plain integer from 0 to 2^31 - 1 or the element count multiplied
// in order overflows 64 bits, as safetensors 0.8 checks it.
std::optional<std::uint64_t> ShapeBits(const gufo::json::Value& entry) {
  std::uint64_t bits = TypeBits("\"" + entry.member_str("dtype") + "\"");
  bool empty = false;
  std::optional<std::uint64_t> count = 1;
  for (const auto& item : entry.find("shape")->items()) {
    const auto dimension = Plain(item);
    if (!dimension || *dimension > 0x7fffffff)
      return std::nullopt;
    empty = empty || *dimension == 0;
    if (count && *dimension != 0 && *count > ~std::uint64_t{0} / *dimension)
      count.reset();
    else if (count)
      *count *= *dimension;
    if (*dimension != 0)
      bits = bits > ~std::uint64_t{0} / *dimension ? ~std::uint64_t{0}
                                                   : bits * *dimension;
  }
  if (!count)
    return std::nullopt;
  return empty ? 0 : bits;
}

// Whether the header shows the fault the reader reports, checked with the
// core JSON parser for files an old parser accepted. Those have a sound
// length prefix below 64 MiB and a JSON object for a header.
bool Explains(std::span<const std::byte> file, const Fresh& fresh) {
  using enum st::Reason;
  const auto length = LengthOf(file);
  if (fresh.reason == kLength)
    return length > st::kMaxHeaderBytes;
  if (fresh.reason == kFile || fresh.reason == kJson ||
      fresh.reason == kNotObject)
    return false;
  const auto root = HeaderOf(file);
  const auto payload = file.size() - 8 - length;
  if (fresh.reason == kMetadata) {
    const auto* metadata = root.find("__metadata__");
    if (!metadata)
      return false;
    if (!metadata->is_object())
      return fresh.name == "__metadata__";
    for (const auto& [key, value] : metadata->members())
      if (!value.is_string())
        return fresh.name == key;
    return false;
  }
  if (fresh.reason == kEmpty) {
    for (const auto& [key, value] : root.members())
      if (key != "__metadata__")
        return false;
    return true;
  }
  if (fresh.reason == kOverlap || fresh.reason == kGap ||
      fresh.reason == kTrailing) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> extents;
    std::vector<std::uint64_t> empty;
    for (const auto& [key, value] : root.members()) {
      if (key == "__metadata__")
        continue;
      // Tensors without elements, whose offsets are equal, hold no bytes.
      const auto& offsets = value.find("data_offsets")->items();
      const auto begin = *Plain(offsets[0]);
      const auto end = *Plain(offsets[1]);
      if (begin == end)
        empty.push_back(begin);
      else
        extents.emplace_back(begin, end);
    }
    std::sort(extents.begin(), extents.end());
    std::uint64_t cursor = 0;
    for (const auto& [begin, end] : extents) {
      if (begin != cursor)
        return fresh.reason == (begin < cursor ? kOverlap : kGap);
      cursor = end;
    }
    if (cursor != payload)
      return fresh.reason == kTrailing;
    // A tensor without elements inside another's data.
    return fresh.reason == kOverlap &&
           std::ranges::any_of(empty, [&](std::uint64_t at) {
             return std::ranges::any_of(extents, [at](const auto& extent) {
               return extent.first < at && at < extent.second;
             });
           });
  }
  const auto* entry = root.find(fresh.name);
  if (!entry)
    return false;
  const auto* dtype = entry->find("dtype");
  const auto* shape = entry->find("shape");
  const auto* offsets = entry->find("data_offsets");
  const bool layout = shape && shape->is_array() && shape->size() <= 5 &&
                      offsets && offsets->is_array() && offsets->size() == 2;
  switch (fresh.reason) {
    case kEntry:
      return !entry->is_object();
    case kDtype:
      return entry->is_object() && !KnownType(dtype);
    case kLayout:
      return KnownType(dtype) && !layout;
    case kDimension: {
      if (!KnownType(dtype) || !layout)
        return false;
      const auto bits = ShapeBits(*entry);
      return !bits || payload > ~std::uint64_t{0} / 8 || *bits > payload * 8;
    }
    case kExtent: {
      if (!KnownType(dtype) || !layout)
        return false;
      const auto bits = ShapeBits(*entry);
      const auto begin = Plain(offsets->items()[0]);
      const auto end = Plain(offsets->items()[1]);
      return bits && (!begin || !end || *begin > *end || *end > payload ||
                      *bits % 8 != 0 || (*end - *begin) != *bits / 8);
    }
    default:
      return false;
  }
}

// Checks an accepted header against the rules, independently of the reader.
void CheckAccepted(std::string_view label, std::span<const std::byte> file,
                   const st::Header& header) {
  Check(header.payload_offset == 8 + LengthOf(file),
        std::string(label) + ": offset");
  const auto payload = file.size() - header.payload_offset;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> extents;
  for (const auto& tensor : header.tensors) {
    // Accepted tensors fit the payload, so their bits fit 64 bits, unless
    // they have no elements.
    // For all, the element count multiplied in order fits 64 bits.
    const bool empty = std::ranges::find(tensor.shape, 0) != tensor.shape.end();
    std::uint64_t bits = empty ? 0 : st::Bits(tensor.dtype);
    std::uint64_t count = 1;
    for (const auto dimension : tensor.shape) {
      Check(dimension <= 0x7fffffff &&
                (dimension == 0 || count <= ~std::uint64_t{0} / dimension) &&
                (empty || bits <= ~std::uint64_t{0} / dimension),
            std::string(label) + ": dimension");
      count *= dimension;
      bits *= dimension;
    }
    Check(tensor.shape.size() <= 5 && tensor.begin <= tensor.end &&
              tensor.end <= payload && bits / 8 == tensor.end - tensor.begin &&
              bits % 8 == 0,
          std::string(label) + ": extent of " + tensor.name);
    if (!empty)
      extents.emplace_back(tensor.begin, tensor.end);
  }
  std::sort(extents.begin(), extents.end());
  std::uint64_t cursor = 0;
  for (const auto& [begin, end] : extents) {
    Check(begin == cursor, std::string(label) + ": tiling");
    cursor = end;
  }
  Check(!header.tensors.empty() && cursor == payload,
        std::string(label) + ": payload");
  // Those without elements lie between the others or at either end.
  for (const auto& tensor : header.tensors)
    if (tensor.begin == tensor.end)
      for (const auto& [begin, end] : extents)
        Check(tensor.begin <= begin || tensor.begin >= end,
              std::string(label) + ": " + tensor.name + " inside data");
  const auto root = HeaderOf(file);
  std::size_t members = 0;
  for (const auto& [key, value] : root.members()) {
    if (key == "__metadata__") {
      Check(value.is_object() || value.is_null(),
            std::string(label) + ": metadata object");
      for (const auto& [name, text] : value.members())
        Check(text.is_string(), std::string(label) + ": metadata " + name);
      continue;
    }
    ++members;
    for (const auto* numbers :
         {value.find("shape"), value.find("data_offsets")})
      for (const auto& item : numbers->items())
        Check(Plain(item).has_value(), std::string(label) + ": integer token");
  }
  Check(members == header.tensors.size(), std::string(label) + ": entries");
}

struct Tally {
  std::size_t files = 0;
  std::size_t accepted = 0;
  std::map<Loader, std::size_t> same;
  std::map<Loader, std::size_t> rejected;
  std::map<Loader, std::map<st::Reason, std::size_t>> stricter;
} tally;

void Compare(std::string_view label, std::span<const std::byte> file) {
  ++tally.files;
  const auto fresh = Run(file);
  if (fresh.header) {
    ++tally.accepted;
    CheckAccepted(label, file, *fresh.header);
  }
  for (const auto loader : kLoaders) {
    const auto old = Reference(loader, file);
    const auto now =
        fresh.header ? Policy(loader, *fresh.header) : ref::Failed("rejected");
    const auto where =
        std::string(label) + " (" + std::string(Label(loader)) + ")";
    if (now.ok && old.ok) {
      Check(Same(now, old), where + ": different tensors");
      ++tally.same[loader];
    } else if (now.ok) {
      Fail(where + ": accepted, but the old parser rejected it: " + old.error);
    } else if (old.ok) {
      if (fresh.header)
        Fail(where + ": the loader rejects the reader's tensors: " + now.error);
      else if (!Stricter(loader).contains(fresh.reason) ||
               !Explains(file, fresh))
        Fail(where + ": newly rejected for reason " +
             std::to_string(static_cast<int>(fresh.reason)) + " at '" +
             fresh.name + "'");
      else
        ++tally.stricter[loader][fresh.reason];
    } else {
      ++tally.rejected[loader];
    }
  }
}

void Compare(std::string_view label, std::string_view header,
             std::uint64_t payload) {
  const auto file = File(header, payload);
  Compare(label, file);
}

// Deterministic on every platform, unlike the standard distributions.
struct Random {
  std::uint64_t state;
  std::uint64_t Next() {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    return state;
  }
  std::size_t Below(std::size_t bound) { return Next() % bound; }
};

constexpr std::array<std::string_view, 22> kTypes{
    "\"BF16\"",    "\"F16\"",
    "\"F32\"",     "\"I32\"",
    "\"I64\"",     "\"U8\"",
    "\"BOOL\"",    "\"F4\"",
    "\"F6_E2M3\"", "\"F8_E4M3\"",
    "\"C64\"",     "\"F64\"",
    "\"U64\"",     "\"F8_E5M2FNUZ\"",
    "\"bf16\"",    "\"f32\"",
    "\"Q8_0\"",    "\"\"",
    "2",           "null",
    "[\"F32\"]",   ""};

constexpr std::array<std::string_view, 21> kShapes{"[]",
                                                   "[1]",
                                                   "[3]",
                                                   "[2,3]",
                                                   "[1,1,1,1,1]",
                                                   "[1,1,1,1,1,1]",
                                                   "[0]",
                                                   "[2,0]",
                                                   "[2147483647]",
                                                   "[2147483648]",
                                                   "[1.0]",
                                                   "[2e0]",
                                                   "[-1]",
                                                   "[-0]",
                                                   "[\"2\"]",
                                                   "[null]",
                                                   "{}",
                                                   "3",
                                                   "[4,2]",
                                                   "[18446744073709551616]",
                                                   ""};

// The elements a shape token holds, when it is a plain list of small ints.
std::optional<std::uint64_t> Elements(std::string_view shape) {
  if (shape.size() < 2 || shape.front() != '[' || shape.back() != ']')
    return std::nullopt;
  std::uint64_t elements = 1;
  std::string digits;
  for (const char c : shape.substr(1)) {
    if (c >= '0' && c <= '9') {
      digits += c;
    } else if (c == ',' || c == ']') {
      if (digits.empty() || digits.size() > 6)
        return c == ']' && digits.empty() && shape == "[]"
                   ? std::optional<std::uint64_t>(1)
                   : std::nullopt;
      elements *= std::stoull(digits);
      digits.clear();
    } else {
      return std::nullopt;
    }
  }
  return elements;
}

void SingleEntries() {
  constexpr std::array<std::string_view, 3> kNames{"language_model.w",
                                                   "vision_tower.w", "other.w"};
  std::size_t index = 0;
  for (const auto type : kTypes) {
    for (const auto shape : kShapes) {
      const auto bits = TypeBits(type);
      const auto elements = Elements(shape);
      std::uint64_t bytes = 4;
      if (bits && elements && *elements * bits % 8 == 0 &&
          *elements * bits / 8 <= 4096 && *elements != 0)
        bytes = *elements * bits / 8;
      const struct {
        std::string offsets;
        std::uint64_t payload;
      } layouts[] = {
          {Offsets(0, bytes), bytes},
          {Offsets(0, bytes), bytes + 1},
          {Offsets(1, bytes + 1), bytes + 1},
          {Offsets(0, bytes + 1), bytes + 1},
          {Offsets(0, bytes), bytes - 1},
          {"[0.0," + std::to_string(bytes) + "]", bytes},
          {"[0," + std::to_string(bytes) + "e0]", bytes},
          {"[0]", bytes},
          {Offsets(bytes, 0), bytes},
          {"", bytes},
          {"[0,\"" + std::to_string(bytes) + "\"]", bytes},
          {Offsets(0, 0), 0},
          {Offsets(1, 1), 1},
          {Offsets(2, 2), 1},
      };
      for (const auto& layout : layouts) {
        const auto name = kNames[index++ % kNames.size()];
        const auto header = "{\"" + std::string(name) +
                            "\":" + Entry(type, shape, layout.offsets) + "}";
        Compare(header, header, layout.payload);
      }
    }
  }
  // Entries that are not objects.
  for (const auto value : {"[]", "1", "\"x\"", "null", "true"}) {
    const auto header = std::string("{\"language_model.w\":") + value + "}";
    Compare(header, header, 0);
    const auto pair = std::string("{\"language_model.v\":") +
                      Entry("\"U8\"", "[2]", "[0,2]") +
                      ",\"other.w\":" + value + "}";
    Compare(pair, pair, 2);
  }
}

void Metadata() {
  const auto tensor =
      "\"language_model.w\":" + Entry("\"F32\"", "[1]", "[0,4]");
  for (const auto metadata :
       {"{}", "{\"format\":\"pt\"}",
        "{\"lora_adapter_metadata\":\"{\\\"lora_alpha\\\":8}\"}",
        "{\"format\":\"pt\",\"n\":1}", "{\"o\":{}}", "[]", "\"x\"", "null",
        "{\"lora_adapter_metadata\":7}"}) {
    for (const bool first : {true, false}) {
      const auto member = std::string("\"__metadata__\":") + metadata;
      const auto header =
          "{" + (first ? member + "," + tensor : tensor + "," + member) + "}";
      Compare(header, header, 4);
    }
  }
}

void Headers() {
  const auto valid =
      "{\"language_model.w\":" + Entry("\"U8\"", "[2]", "[0,2]") + "}";
  for (const auto header :
       {"", " ", "{", "[]", "{}", "null", "{\"__metadata__\":{}}", "1", "\"x\"",
        "{\"a\":1,\"a\":2}"})
    Compare(header, header, 0);
  Compare("padded", valid + "        ", 2);
  Compare("leading space", "  " + valid, 2);
  const auto twice =
      "{\"language_model.w\":" + Entry("\"U8\"", "[2]", "[0,2]") +
      ",\"language_model.w\":" + Entry("\"U8\"", "[2]", "[2,4]") + "}";
  Compare("duplicate", twice, 4);
  // Length fields against the file.
  for (const std::uint64_t length :
       {std::uint64_t{0}, std::uint64_t{1}, valid.size() - 1, valid.size() + 1,
        valid.size() + 2, valid.size() + 3, std::uint64_t{1} << 63,
        ~std::uint64_t{0}}) {
    const auto file = File(length, valid, 2);
    Compare("length " + std::to_string(length), file);
  }
  // A header above 32 MiB, which only some loaders accepted.
  const std::string large =
      "{\"language_model.w\":" + Entry("\"U8\"", "[1]", "[0,1]") + "}" +
      std::string((32U << 20U), ' ');
  Compare("large header", large, 1);
  for (std::size_t size = 0; size < 8; ++size)
    Compare("short file", Bytes(size, std::byte{1}));
}

void Layouts() {
  // Three U8 tensors of 2, 3 and 4 bytes in every order of every layout.
  const std::array<std::uint64_t, 3> sizes{2, 3, 4};
  const std::array<std::array<std::int64_t, 3>, 7> shifts{{
      {0, 0, 0},    // tiled
      {0, 1, 0},    // a gap before the second
      {0, -1, 0},   // the second overlaps the first
      {1, 0, 0},    // a gap at the start
      {0, 0, 1},    // a gap before the third
      {0, -2, -5},  // overlaps
      {0, 0, 0},    // tiled, then a trailing byte
  }};
  std::array<int, 3> order{0, 1, 2};
  do {
    for (std::size_t layout = 0; layout < shifts.size(); ++layout) {
      std::string header = "{";
      std::int64_t cursor = 0;
      for (std::size_t slot = 0; slot < 3; ++slot) {
        const auto tensor = order[slot];
        cursor += shifts[layout][slot];
        const auto begin = std::max<std::int64_t>(cursor, 0);
        const auto end = begin + static_cast<std::int64_t>(sizes[tensor]);
        cursor = end;
        if (slot)
          header += ',';
        header += "\"language_model.t" + std::to_string(tensor) + "\":" +
                  Entry("\"U8\"", "[" + std::to_string(sizes[tensor]) + "]",
                        Offsets(begin, end));
      }
      header += "}";
      Compare(header, header, cursor + (layout == 6 ? 1 : 0));
    }
  } while (std::next_permutation(order.begin(), order.end()));
}

// Tensors without elements before, inside, between, after and past three
// tensors with data, at every place in the header, and alone. Inside one's
// data the reader rejects them, as safetensors 0.8 does; the old Qwen3-ASR,
// Qwen3-TTS and EmbeddingGemma 2 parsers did not check the layout.
void EmptyTensors() {
  // The last overflows 64 bits before its 0, which safetensors 0.8 rejects.
  constexpr std::array<std::string_view, 5> kEmptyShapes{
      "[0]", "[3,0]", "[2147483647,2147483647,0]",
      "[0,2147483647,2147483647,2147483647,2147483647]",
      "[2147483647,2147483647,2147483647,0]"};
  // EmbeddingGemma 2 reads the first and ignores the second.
  constexpr std::array<std::string_view, 2> kEmptyNames{"language_model.e",
                                                        "other.e"};
  const std::array<std::string, 3> data{
      "\"language_model.a\":" + Entry("\"BF16\"", "[1]", "[0,2]"),
      "\"language_model.b\":" + Entry("\"F32\"", "[1]", "[2,6]"),
      "\"vision_tower.c\":" + Entry("\"BF16\"", "[2]", "[6,10]")};
  std::size_t index = 0;
  for (const std::uint64_t at : {0, 1, 2, 4, 6, 10, 11}) {
    for (const auto shape : kEmptyShapes) {
      for (const auto name : kEmptyNames) {
        const auto type = index++ % 2 ? "\"F32\"" : "\"BF16\"";
        const auto empty = "\"" + std::string(name) +
                           "\":" + Entry(type, shape, Offsets(at, at));
        for (std::size_t place = 0; place <= data.size(); ++place) {
          std::string header = "{";
          for (std::size_t slot = 0; slot <= data.size(); ++slot) {
            if (slot)
              header += ',';
            header +=
                slot == place ? empty : data[slot - (slot > place ? 1 : 0)];
          }
          header += "}";
          Compare(header, header, 10);
        }
      }
    }
  }
  const auto empty = [](std::string_view name, std::uint64_t at) {
    return "\"" + std::string(name) +
           "\":" + Entry("\"F32\"", "[0,8]", Offsets(at, at));
  };
  // Two at one offset.
  Compare("two at one offset",
          "{" + data[0] + "," + empty("language_model.e", 2) + "," +
              empty("other.f", 2) + "," + data[1] + "," + data[2] + "}",
          10);
  // Alone, with and without a payload.
  for (const std::uint64_t payload : {0, 1}) {
    for (const std::uint64_t at : {0, 1}) {
      Compare("alone", "{" + empty("other.e", at) + "}", payload);
      Compare(
          "two alone",
          "{" + empty("language_model.e", at) + "," + empty("other.f", 0) + "}",
          payload);
    }
  }
  // Beside a gap and an overlap among the others.
  Compare("gap",
          "{" + empty("other.e", 3) + "," + data[0] +
              ",\"language_model.b\":" + Entry("\"F32\"", "[1]", "[3,7]") + "}",
          7);
  Compare("overlap",
          "{" + empty("other.e", 1) + "," + data[0] +
              ",\"language_model.b\":" + Entry("\"F32\"", "[1]", "[1,5]") + "}",
          5);
}

// Members nothing reads, holding nested values, at every place in an entry
// and in the header.
void Extras() {
  constexpr std::array<std::string_view, 9> kExtras{"{}",
                                                    "{\"a\":[1,{\"b\":2}]}",
                                                    "[{}]",
                                                    "[[[]]]",
                                                    "\"s\"",
                                                    "null",
                                                    "{\"dtype\":1}",
                                                    "[{\"shape\":[]},[]]",
                                                    "{\"x\":{\"y\":{}}}"};
  const std::array<std::string_view, 3> kMembers{
      "\"dtype\":\"F32\"", "\"shape\":[1]", "\"data_offsets\":[0,4]"};
  for (const auto extra : kExtras) {
    for (std::size_t at = 0; at <= kMembers.size(); ++at) {
      std::string entry = "{";
      for (std::size_t index = 0; index <= kMembers.size(); ++index) {
        if (index)
          entry += ',';
        if (index == at)
          entry.append("\"x\":").append(extra);
        else
          entry.append(kMembers[index - (index > at ? 1 : 0)]);
      }
      entry += "}";
      const auto header =
          "{\"language_model.a\":" + entry +
          ",\"language_model.b\":" + Entry("\"F32\"", "[1]", "[4,8]") + "}";
      Compare(header, header, 8);
    }
    // A shape or offset item that is a container.
    for (const auto* field : {"shape", "data_offsets"}) {
      const auto header =
          std::string("{\"language_model.a\":{\"dtype\":\"F32\",") +
          (std::string_view(field) == "shape"
               ? "\"shape\":[1," + std::string(extra) +
                     "],\"data_offsets\":[0,4]"
               : "\"shape\":[1],\"data_offsets\":[0," + std::string(extra) +
                     "]") +
          "}}";
      Compare(header, header, 4);
    }
  }
}

Bytes Base(std::string& header, std::uint64_t& payload, int which) {
  switch (which) {
    case 0:
      header = "{\"__metadata__\":{\"format\":\"pt\"},\"language_model.a\":" +
               Entry("\"BF16\"", "[2,3]", "[0,12]") +
               ",\"vision_tower.b\":" + Entry("\"F32\"", "[2]", "[12,20]") +
               ",\"other.c\":" + Entry("\"I32\"", "[]", "[20,24]") + "}";
      payload = 24;
      break;
    case 1:
      header =
          "{\"language_model.x\":" + Entry("\"F16\"", "[1,1,1,1,2]", "[4,8]") +
          ",\"language_model.y\":" + Entry("\"U8\"", "[4]", "[0,4]") + "}";
      payload = 8;
      break;
    case 3:
      header = "{\"language_model.a\":" + Entry("\"BF16\"", "[1]", "[0,2]") +
               ",\"other.e\":" + Entry("\"F32\"", "[0,3]", "[2,2]") +
               ",\"language_model.b\":" + Entry("\"F32\"", "[1]", "[2,6]") +
               ",\"language_model.e\":" + Entry("\"BF16\"", "[2,0]", "[6,6]") +
               "}";
      payload = 6;
      break;
    default:
      header = "{\"lora.lora_A.weight\":" + Entry("\"F32\"", "[1]", "[0,4]") +
               ",\"__metadata__\":{\"lora_adapter_metadata\":\"{}\"}}";
      payload = 4;
      break;
  }
  return File(header, payload);
}

void Truncations() {
  for (int which = 0; which < 4; ++which) {
    std::string header;
    std::uint64_t payload = 0;
    const auto file = Base(header, payload, which);
    Compare("base " + std::to_string(which), file);
    // The file cut at every byte, its length field unchanged.
    for (std::size_t size = 0; size < file.size(); ++size)
      Compare("cut " + std::to_string(which) + " at " + std::to_string(size),
              std::span(file).first(size));
    // The header cut at every byte, with a matching length.
    for (std::size_t size = 0; size < header.size(); ++size)
      Compare("header cut", header.substr(0, size), payload);
  }
}

void Mutations() {
  constexpr std::string_view kAlphabet = "0123456789-.eE,:[]{}\" _FIUBabtrx";
  Random random{0x9e3779b97f4a7c15ULL};
  for (int which = 0; which < 4; ++which) {
    std::string base;
    std::uint64_t payload = 0;
    (void)Base(base, payload, which);
    for (int round = 0; round < 3000; ++round) {
      std::string header = base;
      std::uint64_t size = payload;
      const auto edits = 1 + random.Below(3);
      for (std::size_t edit = 0; edit < edits; ++edit) {
        const auto at = random.Below(header.size() + 1);
        const char c = kAlphabet[random.Below(kAlphabet.size())];
        switch (random.Below(5)) {
          case 0:
            if (at < header.size())
              header[at] = c;
            break;
          case 1:
            if (at < header.size())
              header.erase(at, 1);
            break;
          case 2:
            header.insert(at, 1, c);
            break;
          case 3: {
            const auto length = random.Below(8);
            header.insert(at, header.substr(at / 2, length));
            break;
          }
          default:
            size = size + random.Below(3) - 1;
            break;
        }
      }
      Compare(header, header, size);
    }
  }
}

void Interface() {
  const std::string header =
      "{\"b\":" + Entry("\"F32\"", "[2]", "[2,10]") +
      ",\"__metadata__\":{\"k\":\"v\",\"format\":\"pt\"},\"a\":" +
      Entry("\"U8\"", "[2]", "[0,2]") + "}";
  const auto file = File(header, 10);
  const auto parsed = st::Parse(file);
  Check(parsed.payload_offset == 8 + header.size(), "payload offset");
  Check(parsed.tensors.size() == 2 && parsed.tensors[0].name == "b" &&
            parsed.tensors[0].dtype == st::Dtype::kF32 &&
            parsed.tensors[0].shape == std::vector<std::uint64_t>{2} &&
            parsed.tensors[0].begin == 2 && parsed.tensors[0].end == 10 &&
            parsed.tensors[1].name == "a",
        "tensors in header order");
  Check(parsed.metadata ==
            std::vector<std::pair<std::string, std::string>>{{"k", "v"},
                                                             {"format", "pt"}},
        "metadata in header order");
  // The same through the two steps a reader without a mapping takes.
  const auto length = st::HeaderLength(std::span(file).first(8), file.size());
  const auto stepped =
      st::ParseHeader(std::span(file).subspan(8, length), file.size());
  Check(length == header.size() && stepped.tensors.size() == 2,
        "header length and parse");
  // A tensor without elements holds no bytes, here at the end of the
  // payload.
  try {
    const auto empty =
        st::Parse(File("{\"a\":" + Entry("\"U8\"", "[2]", "[0,2]") +
                           ",\"e\":" + Entry("\"F32\"", "[3,0]", "[2,2]") + "}",
                       2));
    Check(empty.tensors.size() == 2 &&
              empty.tensors[1].shape == std::vector<std::uint64_t>{3, 0} &&
              empty.tensors[1].begin == 2 && empty.tensors[1].end == 2,
          "tensor without elements");
  } catch (const st::Error& error) {
    Fail(std::string("tensor without elements: ") + error.what());
  }
  // Names and sizes as the Rust reader knows them, for all 22 types of
  // safetensors 0.8: 8 elements of each type take exactly its bits in bytes.
  const auto dtypes = Dtypes();
  Check(dtypes.size() == 22 && dtypes.back() == st::Dtype::kF8E5M2Fnuz,
        "22 types");
  for (const auto dtype : dtypes) {
    const auto bits = st::Bits(dtype);
    const auto typed =
        File("{\"t\":" +
                 Entry("\"" + std::string(st::Name(dtype)) + "\"", "[8]",
                       Offsets(0, bits)) +
                 "}",
             bits);
    try {
      const auto read = st::Parse(typed);
      Check(read.tensors.size() == 1 && read.tensors[0].dtype == dtype,
            "type " + std::string(st::Name(dtype)));
    } catch (const st::Error& error) {
      Fail("type " + std::string(st::Name(dtype)) + ": " + error.what());
    }
  }
  // A value that names no type has no name and no size.
  Check(st::Name(static_cast<st::Dtype>(255)).empty() &&
            st::Bits(static_cast<st::Dtype>(255)) == 0 &&
            st::Name(static_cast<st::Dtype>(dtypes.size())).empty(),
        "unknown type value");
  const auto expect = [](std::span<const std::byte> bad, st::Reason reason,
                         std::string_view name, std::string_view what) {
    try {
      (void)st::Parse(bad);
      Fail("accepted " + std::string(what));
    } catch (const st::Error& error) {
      Check(error.reason() == reason && error.name() == name &&
                error.what() == what,
            "error: " + std::string(error.what()));
    }
  };
  const auto unknown =
      File("{\"t\":" + Entry("\"Q4\"", "[1]", "[0,1]") + "}", 1);
  // The reader says what is wrong; the crate's tests cover each message.
  expect(unknown, st::Reason::kDtype, "t",
         "invalid safetensors header: t: unsupported dtype \"Q4\"");
  expect(File("{\"t\":" + Entry("\"U8\"", "[1]", "[0,1]") + "}", 2),
         st::Reason::kTrailing, "",
         "invalid safetensors header: payload continues after the last tensor");
  expect(File("{\"a\":" + Entry("\"U8\"", "[2]", "[0,2]") +
                  ",\"b\":" + Entry("\"U8\"", "[1]", "[1,2]") + "}",
              2),
         st::Reason::kOverlap, "b",
         "invalid safetensors header: b: data overlaps that of a");
  expect(File("{\"a\":" + Entry("\"U8\"", "[2]", "[0,2]") +
                  ",\"e\":" + Entry("\"F32\"", "[0]", "[1,1]") + "}",
              2),
         st::Reason::kOverlap, "e",
         "invalid safetensors header: e: offset lies inside the data of a");
  expect(File("[1]", 0), st::Reason::kNotObject, "",
         "invalid safetensors header: not a JSON object");
  expect(Bytes(4), st::Reason::kFile, "",
         "invalid safetensors file: shorter than the 8-byte header length");
  try {
    (void)st::Parse(File("{\"t\":", 0));
    Fail("accepted broken JSON");
  } catch (const st::Error& error) {
    Check(error.reason() == st::Reason::kJson &&
              std::string_view(error.what())
                  .starts_with("invalid safetensors header: not JSON: "),
          "JSON error: " + std::string(error.what()));
  }
}

// Compares real files given as arguments and times their headers: the best
// of `rounds` runs of the reader and of each old parser, alternating.
void Files(int argc, char** argv) {
  constexpr int kRounds = 30;
  std::cout << "file\theader KiB\ttensors\treader us";
  for (const auto loader : kLoaders)
    std::cout << '\t' << Label(loader) << " us";
  std::cout << '\n';
  for (int index = 1; index < argc; ++index) {
    gufo::platform::RawMappedFile mapped;
    if (!mapped.Open(argv[index])) {
      Fail(std::string("cannot map ") + argv[index]);
      continue;
    }
    const std::span file(static_cast<const std::byte*>(mapped.data),
                         mapped.size);
    Compare(argv[index], file);
    const auto time = [](auto&& call) {
      const auto start = std::chrono::steady_clock::now();
      call();
      return std::chrono::duration<double, std::micro>(
                 std::chrono::steady_clock::now() - start)
          .count();
    };
    double reader = 1e300;
    std::array<double, kLoaders.size()> old;
    old.fill(1e300);
    std::size_t tensors = 0;
    for (int round = 0; round < kRounds; ++round) {
      reader = std::min(reader, time([&] {
                          try {
                            tensors = st::Parse(file).tensors.size();
                          } catch (const st::Error&) {
                          }
                        }));
      for (std::size_t slot = 0; slot < kLoaders.size(); ++slot)
        old[slot] = std::min(
            old[slot], time([&] { (void)Reference(kLoaders[slot], file); }));
    }
    std::cout << argv[index] << '\t'
              << static_cast<double>(LengthOf(file)) / 1024 << '\t' << tensors
              << '\t' << reader;
    for (const auto value : old)
      std::cout << '\t' << value;
    std::cout << '\n';
    mapped.Close();
  }
}
}  // namespace

int main(int argc, char** argv) {
  Files(argc, argv);
  Interface();
  SingleEntries();
  Metadata();
  Headers();
  Layouts();
  EmptyTensors();
  Extras();
  Truncations();
  Mutations();
  if (failures != 0) {
    std::cerr << failures << " failures\n";
    return 1;
  }
  std::cout << tally.files << " files, " << tally.accepted
            << " accepted by the reader\n";
  for (const auto loader : kLoaders) {
    std::cout << "  " << Label(loader) << ": " << tally.same[loader]
              << " same, " << tally.rejected[loader] << " rejected by both";
    for (const auto& [reason, count] : tally.stricter[loader])
      std::cout << ", " << count << " newly rejected (reason "
                << static_cast<int>(reason) << ")";
    std::cout << '\n';
  }
  return 0;
}

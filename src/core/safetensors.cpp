#include "src/core/safetensors.hpp"

#include <gufo/safetensors.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/rust/callback.hpp"

namespace gufo::safetensors {
namespace {

// The C++ enums mirror the C ones value for value.
constexpr bool Mirrors() {
  constexpr std::pair<GufoSafetensorsDtype, Dtype> kTypes[] = {
      {GUFO_SAFETENSORS_BOOL, Dtype::kBool},
      {GUFO_SAFETENSORS_F4, Dtype::kF4},
      {GUFO_SAFETENSORS_F6_E2M3, Dtype::kF6E2M3},
      {GUFO_SAFETENSORS_F6_E3M2, Dtype::kF6E3M2},
      {GUFO_SAFETENSORS_U8, Dtype::kU8},
      {GUFO_SAFETENSORS_I8, Dtype::kI8},
      {GUFO_SAFETENSORS_F8_E5M2, Dtype::kF8E5M2},
      {GUFO_SAFETENSORS_F8_E4M3, Dtype::kF8E4M3},
      {GUFO_SAFETENSORS_F8_E8M0, Dtype::kF8E8M0},
      {GUFO_SAFETENSORS_I16, Dtype::kI16},
      {GUFO_SAFETENSORS_U16, Dtype::kU16},
      {GUFO_SAFETENSORS_F16, Dtype::kF16},
      {GUFO_SAFETENSORS_BF16, Dtype::kBF16},
      {GUFO_SAFETENSORS_I32, Dtype::kI32},
      {GUFO_SAFETENSORS_U32, Dtype::kU32},
      {GUFO_SAFETENSORS_F32, Dtype::kF32},
      {GUFO_SAFETENSORS_C64, Dtype::kC64},
      {GUFO_SAFETENSORS_F64, Dtype::kF64},
      {GUFO_SAFETENSORS_I64, Dtype::kI64},
      {GUFO_SAFETENSORS_U64, Dtype::kU64},
      {GUFO_SAFETENSORS_F8_E4M3FNUZ, Dtype::kF8E4M3Fnuz},
      {GUFO_SAFETENSORS_F8_E5M2FNUZ, Dtype::kF8E5M2Fnuz},
  };
  constexpr std::pair<GufoSafetensorsReason, Reason> kReasons[] = {
      {GUFO_SAFETENSORS_FILE, Reason::kFile},
      {GUFO_SAFETENSORS_LENGTH, Reason::kLength},
      {GUFO_SAFETENSORS_JSON, Reason::kJson},
      {GUFO_SAFETENSORS_NOT_OBJECT, Reason::kNotObject},
      {GUFO_SAFETENSORS_METADATA, Reason::kMetadata},
      {GUFO_SAFETENSORS_ENTRY, Reason::kEntry},
      {GUFO_SAFETENSORS_DTYPE, Reason::kDtype},
      {GUFO_SAFETENSORS_LAYOUT, Reason::kLayout},
      {GUFO_SAFETENSORS_DIMENSION, Reason::kDimension},
      {GUFO_SAFETENSORS_EXTENT, Reason::kExtent},
      {GUFO_SAFETENSORS_OVERLAP, Reason::kOverlap},
      {GUFO_SAFETENSORS_GAP, Reason::kGap},
      {GUFO_SAFETENSORS_TRAILING, Reason::kTrailing},
      {GUFO_SAFETENSORS_EMPTY, Reason::kEmpty},
  };
  for (const auto& [c, cpp] : kTypes)
    if (static_cast<int>(c) != static_cast<int>(cpp))
      return false;
  for (const auto& [c, cpp] : kReasons)
    if (static_cast<int>(c) != static_cast<int>(cpp))
      return false;
  return true;
}
static_assert(Mirrors());

std::string Message(Reason reason, const std::string& name,
                    std::string_view message) {
  std::string text = reason == Reason::kFile ? "invalid safetensors file: "
                                             : "invalid safetensors header: ";
  if (!name.empty())
    text.append(name).append(": ");
  text.append(message);
  return text;
}

std::string View(GufoSafetensorsSlice slice) {
  return {reinterpret_cast<const char*>(slice.data), slice.len};
}

const std::uint8_t* Bytes(std::span<const std::byte> bytes) {
  return reinterpret_cast<const std::uint8_t*>(bytes.data());
}

/// What the callbacks fill: the header, which only ParseHeader passes, and
/// the rejection, once there is one.
struct Context {
  Header* header{nullptr};
  std::optional<Error> error;
  core::ffi::CallbackFailure failure;
};

// No exception may cross the Rust frames; each is rethrown after the call.
void Reject(void* opaque, const GufoSafetensorsRejection* rejection) noexcept {
  auto& context = *static_cast<Context*>(opaque);
  context.failure.Guard([&] {
    context.error.emplace(static_cast<Reason>(rejection->reason),
                          View(rejection->name), View(rejection->message));
  });
}

void Finish(GufoSafetensorsParse status, Context& context) {
  context.failure.Rethrow();
  switch (status) {
    case GUFO_SAFETENSORS_COMPLETE:
      return;
    case GUFO_SAFETENSORS_REJECTED:
      if (context.error)
        throw *context.error;
      break;
    case GUFO_SAFETENSORS_NO_MEMORY:
      throw std::bad_alloc();
    case GUFO_SAFETENSORS_STOPPED:
    case GUFO_SAFETENSORS_INTERNAL:
      break;
  }
  throw std::logic_error("safetensors header reader failed");
}

}  // namespace

// The Rust reader holds the one table of names and sizes.
std::string_view Name(Dtype dtype) noexcept {
  const auto name =
      gufo_safetensors_dtype_name(static_cast<std::uint32_t>(dtype));
  return {reinterpret_cast<const char*>(name.data), name.len};
}

unsigned Bits(Dtype dtype) noexcept {
  return gufo_safetensors_dtype_bits(static_cast<std::uint32_t>(dtype));
}

Error::Error(Reason reason, std::string name, std::string_view message)
    : std::runtime_error(Message(reason, name, message)),
      reason_(reason),
      name_(std::move(name)) {}

std::uint64_t HeaderLength(std::span<const std::byte> prefix,
                           std::uint64_t file_size) {
  Context context;
  std::uint64_t length = 0;
  const auto status = gufo_safetensors_header_length(
      Bytes(prefix), prefix.size(), file_size, Reject, &context, &length);
  Finish(status, context);
  return length;
}

Header ParseHeader(std::span<const std::byte> header, std::uint64_t file_size) {
  Header parsed;
  parsed.payload_offset = kLengthBytes + header.size();
  Context context{&parsed, {}, {}};
  const auto metadata = [](void* opaque,
                           const GufoSafetensorsMetadata* entry) noexcept {
    auto& context = *static_cast<Context*>(opaque);
    return context.failure.Guard(1, [&] {
      context.header->metadata.emplace_back(View(entry->key),
                                            View(entry->value));
      return 0;
    });
  };
  const auto tensor = [](void* opaque,
                         const GufoSafetensorsTensor* entry) noexcept {
    auto& context = *static_cast<Context*>(opaque);
    return context.failure.Guard(1, [&] {
      context.header->tensors.push_back(
          {.name = View(entry->name),
           .dtype = static_cast<Dtype>(entry->dtype),
           .shape = {entry->shape, entry->shape + entry->rank},
           .begin = entry->begin,
           .end = entry->end});
      return 0;
    });
  };
  const auto status =
      gufo_safetensors_parse_header(Bytes(header), header.size(), file_size,
                                    metadata, tensor, Reject, &context);
  Finish(status, context);
  return parsed;
}

Header Parse(std::span<const std::byte> file) {
  const auto length = HeaderLength(file, file.size());
  return ParseHeader(file.subspan(kLengthBytes, length), file.size());
}

}  // namespace gufo::safetensors

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Reads safetensors headers. A file is an 8-byte little-endian header
/// length, a JSON header that names each tensor's type, shape and extent and
/// may hold a "__metadata__" object of strings, then the payload the tensors
/// tile. The Rust reader in src/core/safetensors checks the header; the
/// tensor data stays in the caller's file or mapping.
namespace gufo::safetensors {

/// A tensor's element type, as the format names it.
enum class Dtype : std::uint8_t {
  kBool,
  kF4,
  kF6E2M3,
  kF6E3M2,
  kU8,
  kI8,
  kF8E5M2,
  kF8E4M3,
  kF8E8M0,
  kI16,
  kU16,
  kF16,
  kBF16,
  kI32,
  kU32,
  kF32,
  kC64,
  kF64,
  kI64,
  kU64,
};

/// The name the format spells, such as "BF16".
[[nodiscard]] std::string_view Name(Dtype dtype) noexcept;
/// The size of one element in bits.
[[nodiscard]] unsigned Bits(Dtype dtype) noexcept;

/// Why a file is not a well-formed safetensors file.
enum class Reason {
  /// The file is shorter than the length prefix.
  kFile = 1,
  /// The header length is 0, above kMaxHeaderBytes or past the file.
  kLength,
  /// The header is not JSON.
  kJson,
  /// The header is not a JSON object.
  kNotObject,
  /// "__metadata__" is neither null nor an object of strings.
  kMetadata,
  /// A tensor entry is not an object.
  kEntry,
  /// A type is missing, not a string or not one the format names.
  kDtype,
  /// "shape" or "data_offsets" is missing or not an array, the shape has
  /// more than 5 dimensions or the offsets are not two.
  kLayout,
  /// A dimension is not an integer from 1 to 2^31 - 1, or the shape holds
  /// more bytes than the payload.
  kDimension,
  /// An offset is not an integer, or the offsets do not span exactly the
  /// tensor's bytes within the payload.
  kExtent,
  /// Two tensors share bytes.
  kOverlap,
  /// Payload bytes before a tensor belong to none.
  kGap,
  /// Payload bytes after the last tensor belong to none.
  kTrailing,
  /// The header holds no tensor.
  kEmpty,
};

/// A rejected file. what() reads "invalid safetensors file: ", the failing
/// member and ": " if there is one, and what is wrong, such as
/// "invalid safetensors file: w: unsupported dtype "bf16"".
class Error : public std::runtime_error {
public:
  Error(Reason reason, std::string name, std::string_view message);

  [[nodiscard]] Reason reason() const noexcept { return reason_; }
  /// The tensor or metadata key that fails; empty when the file or the
  /// header as a whole does.
  [[nodiscard]] const std::string& name() const noexcept { return name_; }

private:
  Reason reason_;
  std::string name_;
};

struct Tensor {
  std::string name;
  Dtype dtype{Dtype::kU8};
  std::vector<std::uint64_t> shape;
  /// The tensor's bytes are [begin, end) of the payload.
  std::uint64_t begin{0};
  std::uint64_t end{0};
};

struct Header {
  /// Where the payload starts in the file, after the length and header.
  std::uint64_t payload_offset{0};
  /// In header order. Together they tile the payload without gaps.
  std::vector<Tensor> tensors;
  /// The "__metadata__" entries in header order.
  std::vector<std::pair<std::string, std::string>> metadata;
};

/// The bytes of the header length that starts a file.
inline constexpr std::size_t kLengthBytes = 8;
/// The largest header accepted.
inline constexpr std::uint64_t kMaxHeaderBytes = std::uint64_t{32} << 20U;

/// The header length the first kLengthBytes of `prefix` give, checked against
/// a file of `file_size` bytes. Throws Error.
[[nodiscard]] std::uint64_t HeaderLength(std::span<const std::byte> prefix,
                                         std::uint64_t file_size);

/// Checks the header that follows the length of a file of `file_size`
/// bytes. Throws Error for a malformed file, std::bad_alloc when memory runs
/// out and std::logic_error on a reader bug.
[[nodiscard]] Header ParseHeader(std::span<const std::byte> header,
                                 std::uint64_t file_size);

/// Checks the header of a whole file held in memory, such as a mapping.
[[nodiscard]] Header Parse(std::span<const std::byte> file);

}  // namespace gufo::safetensors

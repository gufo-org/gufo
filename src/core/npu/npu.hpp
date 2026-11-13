#pragma once

// AMD XDNA NPU programs through XRT, for model runners and extensions.
//
// An NPU program is one xclbin (one hardware context and its MLIR-AIE kernel)
// plus instruction streams built for it, as mlir-aie's IRON flow emits them.
// A run passes an instruction stream and the program's buffers; the opcode-3
// calling convention of mlir-aie's NPU kernels is fixed here:
//
//   kernel(3, instructions, word_count, argument 3, argument 4, ...)
//
// The header names no XRT type, so code using it builds without XRT headers.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::npu {

class Error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// An NPU accelerator node the kernel driver exposes (`/sys/class/accel`). The
/// probe reads sysfs only and needs neither XRT nor device access.
struct DeviceInfo {
  std::string node;      ///< accel node name, such as "accel0"
  std::string driver;    ///< kernel driver, such as "amdxdna"
  std::string pci_id;    ///< "vendor:device", such as "1022:17f0"
  std::string firmware;  ///< firmware version when the driver reports one
};

/// XDNA NPUs present on this machine, in node order. Linux reads sysfs;
/// Windows asks XRT, so its entries carry only `node` and `driver`.
[[nodiscard]] std::vector<DeviceInfo> ProbeDevices();
/// The `amdxdna` nodes under a sysfs root.
[[nodiscard]] std::vector<DeviceInfo> ProbeDevices(
    const std::filesystem::path& sysfs_root);

/// Where a buffer lives. The NPU reads host memory either way.
enum class Memory {
  kHostOnly,   ///< weights, activations, state: XRT `host_only`
  kCacheable,  ///< instruction streams: XRT `cacheable`
};

class Device;
class Program;

/// A buffer object of one device, usable by every program on that device.
/// `ToDevice`/`FromDevice` flush or invalidate a byte range; the whole buffer
/// when `bytes` is `kWhole`.
class Buffer {
public:
  static constexpr std::size_t kWhole = static_cast<std::size_t>(-1);

  Buffer();
  ~Buffer();
  Buffer(Buffer&&) noexcept;
  Buffer& operator=(Buffer&&) noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return impl_ != nullptr;
  }
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::byte* data() const;
  template<typename T>
  [[nodiscard]] std::span<T> as() const {
    return {reinterpret_cast<T*>(data()), size() / sizeof(T)};
  }

  void ToDevice(std::size_t offset = 0, std::size_t bytes = kWhole);
  void FromDevice(std::size_t offset = 0, std::size_t bytes = kWhole);

  struct Impl;

private:
  friend class Program;
  std::unique_ptr<Impl> impl_;
};

/// An instruction stream: a cacheable buffer of 32-bit words, of which the
/// first `count()` words run. Streams that differ only by run-time values (a
/// token id, a position) can be patched in place instead of regenerated.
class Instructions {
public:
  /// One word that changes linearly with a run-time value v:
  /// word[index] = base[index] + v * delta.
  struct Patch {
    std::uint32_t index;
    std::uint32_t delta;
  };

  Instructions() = default;

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(buffer_);
  }
  [[nodiscard]] std::size_t capacity() const noexcept {
    return buffer_.size() / 4;
  }
  [[nodiscard]] std::size_t count() const noexcept { return count_; }
  [[nodiscard]] std::span<std::uint32_t> words() const {
    return buffer_.as<std::uint32_t>().first(count_);
  }

  /// Copies `words` in and flushes them. Throws Error beyond the capacity.
  void Assign(std::span<const std::uint32_t> words);
  /// Adds `value * delta` to each patched word and flushes those words.
  void Apply(std::span<const Patch> patches, std::uint32_t value);

  [[nodiscard]] const Buffer& buffer() const noexcept { return buffer_; }

private:
  friend class Program;
  Buffer buffer_;
  std::size_t count_{};
};

/// The patches that turn stream `at0` (built for value 0) into `at1` (built for
/// value 1). Throws Error when the streams differ in length.
[[nodiscard]] std::vector<Instructions::Patch> DiffInstructions(
    std::span<const std::uint32_t> at0, std::span<const std::uint32_t> at1);

/// A started run. `Wait` blocks until the NPU finishes and throws Error when
/// the run did not complete. Destroying a started run waits for it.
class Run {
public:
  Run();
  ~Run();
  Run(Run&&) noexcept;
  Run& operator=(Run&&) noexcept;
  Run(const Run&) = delete;
  Run& operator=(const Run&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return impl_ != nullptr;
  }
  void Wait();

  struct Impl;

private:
  friend class Program;
  std::unique_ptr<Impl> impl_;
};

/// One xclbin loaded on a device: its hardware context and NPU kernel.
/// Switching between hardware contexts costs milliseconds, so keep a phase of
/// work (a prefill, a decode step) inside one program.
class Program {
public:
  Program();
  ~Program();
  Program(Program&&) noexcept;
  Program& operator=(Program&&) noexcept;
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;

  /// Loads `xclbin` and opens its first kernel whose name starts with
  /// `kernel_prefix`.
  static Program Load(Device& device, const std::filesystem::path& xclbin,
                      std::string_view kernel_prefix = "MLIR_AIE");

  [[nodiscard]] const std::string& kernel_name() const noexcept;

  /// A buffer in the memory group of kernel argument `argument` (3 and up).
  [[nodiscard]] Buffer Allocate(std::size_t bytes, unsigned argument,
                                Memory memory = Memory::kHostOnly) const;
  /// As Allocate, filled from `file` (its whole size) and flushed.
  [[nodiscard]] Buffer LoadBuffer(const std::filesystem::path& file,
                                  unsigned argument,
                                  Memory memory = Memory::kHostOnly) const;

  /// An empty stream of `capacity` words.
  [[nodiscard]] Instructions AllocateInstructions(std::size_t capacity) const;
  /// A stream read from an mlir-aie `.bin`/`.insts` file (little-endian words).
  [[nodiscard]] Instructions LoadInstructions(
      const std::filesystem::path& file) const;

  /// Starts `instructions` with `arguments` as kernel arguments 3, 4, ...
  /// A null entry passes no buffer for that argument.
  [[nodiscard]] Run Start(const Instructions& instructions,
                          std::span<const Buffer* const> arguments) const;
  Run Start(const Instructions& instructions,
            std::initializer_list<const Buffer*> arguments) const {
    return Start(instructions, std::span<const Buffer* const>(
                                   arguments.begin(), arguments.size()));
  }

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

/// An opened NPU. Programs and buffers keep their device alive.
class Device {
public:
  /// Opens XRT device `index`. Throws Error without XRT or a usable device.
  static Device Open(unsigned index = 0);

  Device();
  ~Device();
  Device(Device&&) noexcept;
  Device& operator=(Device&&) noexcept;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  [[nodiscard]] std::string name() const;

  struct Impl;

private:
  friend class Program;
  std::shared_ptr<Impl> impl_;
};

/// A dispatch library of an IRON design with run-time scalars: mlir-aie's
/// `.prj` build exports `dispatch_generate(int32_t... scalars, uint32_t**
/// out)`, which writes a fresh instruction stream for those scalar values and
/// returns its word count. The words stay owned by the library until the next
/// call.
class DispatchLibrary {
public:
  static constexpr std::size_t kMaxScalars = 16;

  DispatchLibrary();
  ~DispatchLibrary();
  DispatchLibrary(DispatchLibrary&&) noexcept;
  DispatchLibrary& operator=(DispatchLibrary&&) noexcept;
  DispatchLibrary(const DispatchLibrary&) = delete;
  DispatchLibrary& operator=(const DispatchLibrary&) = delete;

  /// Loads `library`, whose `symbol` takes exactly `scalars` int32 values.
  static DispatchLibrary Open(const std::filesystem::path& library,
                              std::size_t scalars,
                              const char* symbol = "dispatch_generate");

  /// Generates the stream for `scalars` (exactly the count given to Open).
  /// Throws Error when the library reports no words.
  [[nodiscard]] std::span<const std::uint32_t> Generate(
      std::span<const std::int32_t> scalars) const;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::npu

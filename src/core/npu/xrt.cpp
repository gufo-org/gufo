// The NPU runtime on XRT.

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <cstdio>
#include <cstring>
#include <exception>

#include "src/core/npu/npu.hpp"
#include "src/core/npu/xrt_load.hpp"

namespace gufo::npu {
namespace {

// mlir-aie's NPU kernels: argument 0 is the opcode, 1 the instruction
// buffer, 2 its word count, 3 and up the data buffers.
constexpr unsigned kOpcode = 3;
constexpr int kInstructionArgument = 1;
constexpr unsigned kFirstDataArgument = 3;

template<typename F>
auto Wrap(const char* what, F&& f) -> decltype(f()) {
  try {
    return f();
  } catch (const Error&) {
    throw;
  } catch (const std::exception& e) {
    throw Error(std::string(what) + ": " + e.what());
  }
}

std::size_t FileBytes(const std::filesystem::path& file) {
  std::error_code ec;
  const auto bytes = std::filesystem::file_size(file, ec);
  if (ec)
    throw Error("cannot stat " + file.string() + ": " + ec.message());
  return static_cast<std::size_t>(bytes);
}

void ReadFile(const std::filesystem::path& file, std::byte* dst,
              std::size_t bytes) {
  std::FILE* f = std::fopen(file.string().c_str(), "rb");
  if (!f)
    throw Error("cannot open " + file.string());
  const std::size_t got = std::fread(dst, 1, bytes, f);
  std::fclose(f);
  if (got != bytes)
    throw Error(file.string() + ": short read");
}

}  // namespace

struct Device::Impl {
  xrt::device device;
};

struct Buffer::Impl {
  std::shared_ptr<Device::Impl> device;  // keeps the device open
  xrt::bo bo;
  std::byte* map{};
};

struct Run::Impl {
  std::shared_ptr<Device::Impl> device;
  xrt::run run;
  bool waited{};
};

struct Program::Impl {
  std::shared_ptr<Device::Impl> device;
  xrt::hw_context context;
  xrt::kernel kernel;
  std::string kernel_name;
};

Device::Device() = default;
Device::~Device() = default;
Device::Device(Device&&) noexcept = default;
Device& Device::operator=(Device&&) noexcept = default;

Device Device::Open(unsigned index) {
  if (!LoadXrt())
    throw Error(
        "cannot open the NPU: XRT (xrt_coreutil.dll from the AMD NPU "
        "driver) is not installed");
  return Wrap("cannot open the NPU", [&] {
    Device out;
    out.impl_ = std::make_shared<Impl>(Impl{xrt::device(index)});
    return out;
  });
}

std::string Device::name() const {
  if (!impl_)
    return {};
  return Wrap("NPU name", [&] {
    return impl_->device.get_info<xrt::info::device::name>();
  });
}

Buffer::Buffer() = default;
Buffer::~Buffer() = default;
Buffer::Buffer(Buffer&&) noexcept = default;
Buffer& Buffer::operator=(Buffer&&) noexcept = default;

std::size_t Buffer::size() const noexcept {
  return impl_ ? impl_->bo.size() : 0;
}
std::byte* Buffer::data() const {
  return impl_ ? impl_->map : nullptr;
}

namespace {

void Sync(Buffer::Impl* impl, xclBOSyncDirection dir, std::size_t offset,
          std::size_t bytes) {
  if (!impl)
    throw Error("sync of an empty NPU buffer");
  const std::size_t size = impl->bo.size();
  if (bytes == Buffer::kWhole)
    bytes = offset <= size ? size - offset : 0;
  if (offset > size || bytes > size - offset)
    throw Error("NPU buffer sync beyond its size");
  if (bytes == 0)
    return;
  Wrap("NPU buffer sync", [&] { impl->bo.sync(dir, bytes, offset); });
}

}  // namespace

void Buffer::ToDevice(std::size_t offset, std::size_t bytes) {
  Sync(impl_.get(), XCL_BO_SYNC_BO_TO_DEVICE, offset, bytes);
}
void Buffer::FromDevice(std::size_t offset, std::size_t bytes) {
  Sync(impl_.get(), XCL_BO_SYNC_BO_FROM_DEVICE, offset, bytes);
}

Run::Run() = default;
Run::~Run() {
  if (impl_ && !impl_->waited) {
    try {
      impl_->run.wait();
    } catch (...) {
    }
  }
}
Run::Run(Run&&) noexcept = default;
Run& Run::operator=(Run&& other) noexcept {
  if (this != &other) {
    Run discard(std::move(*this));  // waits for a started run
    impl_ = std::move(other.impl_);
  }
  return *this;
}

void Run::Wait() {
  if (!impl_ || impl_->waited)
    return;
  impl_->waited = true;
  const ert_cmd_state state =
      Wrap("NPU run", [&] { return impl_->run.wait(); });
  if (state != ERT_CMD_STATE_COMPLETED) {
    throw Error("NPU run ended in state " +
                std::to_string(static_cast<int>(state)));
  }
}

Program::Program() = default;
Program::~Program() = default;
Program::Program(Program&&) noexcept = default;
Program& Program::operator=(Program&&) noexcept = default;

Program Program::Load(Device& device, const std::filesystem::path& xclbin,
                      std::string_view kernel_prefix) {
  if (!device.impl_)
    throw Error("NPU device is not open");
  return Wrap(("cannot load " + xclbin.string()).c_str(), [&] {
    xrt::xclbin file(xclbin.string());
    device.impl_->device.register_xclbin(file);
    Program out;
    out.impl_ = std::make_unique<Impl>();
    out.impl_->device = device.impl_;
    out.impl_->context = xrt::hw_context(device.impl_->device, file.get_uuid());
    for (const auto& k : file.get_kernels()) {
      if (k.get_name().rfind(kernel_prefix, 0) == 0) {
        out.impl_->kernel_name = k.get_name();
        out.impl_->kernel =
            xrt::kernel(out.impl_->context, out.impl_->kernel_name);
        return out;
      }
    }
    throw Error(xclbin.string() + ": no kernel named " +
                std::string(kernel_prefix) + "*");
  });
}

const std::string& Program::kernel_name() const noexcept {
  static const std::string empty;
  return impl_ ? impl_->kernel_name : empty;
}

Buffer Program::Allocate(std::size_t bytes, unsigned argument,
                         Memory memory) const {
  if (!impl_)
    throw Error("NPU program is not loaded");
  if (bytes == 0)
    throw Error("NPU buffers need at least one byte");
  const auto flags = memory == Memory::kCacheable ? xrt::bo::flags::cacheable
                                                  : xrt::bo::flags::host_only;
  Buffer out;
  out.impl_ = std::make_unique<Buffer::Impl>();
  out.impl_->device = impl_->device;
  Wrap("NPU buffer allocation", [&] {
    out.impl_->bo = xrt::bo(impl_->device->device, bytes, flags,
                            impl_->kernel.group_id(static_cast<int>(argument)));
    out.impl_->map = out.impl_->bo.map<std::byte*>();
  });
  return out;
}

Buffer Program::LoadBuffer(const std::filesystem::path& file, unsigned argument,
                           Memory memory) const {
  Buffer out = Allocate(FileBytes(file), argument, memory);
  ReadFile(file, out.data(), out.size());
  out.ToDevice();
  return out;
}

Instructions Program::AllocateInstructions(std::size_t capacity) const {
  Instructions out;
  out.buffer_ =
      Allocate(capacity * 4, kInstructionArgument, Memory::kCacheable);
  return out;
}

Instructions Program::LoadInstructions(
    const std::filesystem::path& file) const {
  const std::size_t bytes = FileBytes(file);
  if (bytes == 0 || bytes % 4 != 0)
    throw Error(file.string() + ": not a stream of 32-bit words");
  Instructions out;
  out.buffer_ = LoadBuffer(file, kInstructionArgument, Memory::kCacheable);
  out.count_ = bytes / 4;
  return out;
}

Run Program::Start(const Instructions& instructions,
                   std::span<const Buffer* const> arguments) const {
  if (!impl_)
    throw Error("NPU program is not loaded");
  if (!instructions || instructions.count() == 0)
    throw Error("NPU run without instructions");
  Run out;
  out.impl_ = std::make_unique<Run::Impl>();
  out.impl_->device = impl_->device;
  Wrap("NPU run start", [&] {
    xrt::run run(impl_->kernel);
    run.set_arg(0, kOpcode);
    run.set_arg(kInstructionArgument, instructions.buffer().impl_->bo);
    run.set_arg(2, static_cast<std::uint32_t>(instructions.count()));
    for (std::size_t i = 0; i < arguments.size(); ++i) {
      if (arguments[i] == nullptr)
        continue;
      if (!*arguments[i])
        throw Error("NPU run argument " +
                    std::to_string(kFirstDataArgument + i) + " is empty");
      run.set_arg(static_cast<int>(kFirstDataArgument + i),
                  arguments[i]->impl_->bo);
    }
    run.start();
    out.impl_->run = std::move(run);
  });
  return out;
}

}  // namespace gufo::npu

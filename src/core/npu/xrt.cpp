// The NPU runtime on XRT, through the backend of xrt_load.hpp.

#include <cstdio>
#include <string>
#include <vector>

#include "src/core/npu/npu.hpp"
#include "src/core/npu/xrt_load.hpp"

namespace gufo::npu {
namespace {

// mlir-aie's NPU kernels: argument 0 is the opcode, 1 the instruction
// buffer, 2 its word count, 3 and up the data buffers.
constexpr unsigned kOpcode = 3;
constexpr int kInstructionArgument = 1;
constexpr unsigned kFirstDataArgument = 3;

void Check(const std::string& what, const char* error) {
  if (error)
    throw Error(what + ": " + error);
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

// Each Impl owns one backend handle. The backend keeps the XRT device open
// while any program, buffer or run made from it is alive.
struct Device::Impl {
  const GufoNpuXrt* xrt{};
  void* device{};
  ~Impl() {
    if (device)
      xrt->device_close(device);
  }
};

struct Buffer::Impl {
  const GufoNpuXrt* xrt{};
  void* bo{};
  std::byte* map{};
  std::size_t size{};
  ~Impl() {
    if (bo)
      xrt->buffer_close(bo);
  }
};

struct Run::Impl {
  const GufoNpuXrt* xrt{};
  void* run{};
  bool waited{};
  ~Impl() {
    if (run)
      xrt->run_close(run, waited ? 1 : 0);  // waits for a started run
  }
};

struct Program::Impl {
  const GufoNpuXrt* xrt{};
  void* program{};
  std::string kernel_name;
  ~Impl() {
    if (program)
      xrt->program_close(program);
  }
};

Device::Device() = default;
Device::~Device() = default;
Device::Device(Device&&) noexcept = default;
Device& Device::operator=(Device&&) noexcept = default;

Device Device::Open(unsigned index) {
  const GufoNpuXrt* xrt = nullptr;
  try {
    xrt = &Xrt();
  } catch (const Error& e) {
    throw Error(std::string("cannot open the NPU: ") + e.what());
  }
  Device out;
  out.impl_ = std::make_shared<Impl>();
  out.impl_->xrt = xrt;
  Check("cannot open the NPU", xrt->device_open(index, &out.impl_->device));
  return out;
}

std::string Device::name() const {
  if (!impl_)
    return {};
  const char* name = nullptr;
  Check("NPU name", impl_->xrt->device_name(impl_->device, &name));
  return name ? name : "";
}

Buffer::Buffer() = default;
Buffer::~Buffer() = default;
Buffer::Buffer(Buffer&&) noexcept = default;
Buffer& Buffer::operator=(Buffer&&) noexcept = default;

std::size_t Buffer::size() const noexcept {
  return impl_ ? impl_->size : 0;
}
std::byte* Buffer::data() const {
  return impl_ ? impl_->map : nullptr;
}

namespace {

void Sync(Buffer::Impl* impl, bool to_device, std::size_t offset,
          std::size_t bytes) {
  if (!impl)
    throw Error("sync of an empty NPU buffer");
  const std::size_t size = impl->size;
  if (bytes == Buffer::kWhole)
    bytes = offset <= size ? size - offset : 0;
  if (offset > size || bytes > size - offset)
    throw Error("NPU buffer sync beyond its size");
  if (bytes == 0)
    return;
  Check("NPU buffer sync",
        impl->xrt->buffer_sync(impl->bo, to_device ? 1 : 0, bytes, offset));
}

}  // namespace

void Buffer::ToDevice(std::size_t offset, std::size_t bytes) {
  Sync(impl_.get(), true, offset, bytes);
}
void Buffer::FromDevice(std::size_t offset, std::size_t bytes) {
  Sync(impl_.get(), false, offset, bytes);
}

Run::Run() = default;
Run::~Run() = default;
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
  int state = 0;
  Check("NPU run", impl_->xrt->run_wait(impl_->run, &state));
  if (state != impl_->xrt->completed_state)
    throw Error("NPU run ended in state " + std::to_string(state));
}

Program::Program() = default;
Program::~Program() = default;
Program::Program(Program&&) noexcept = default;
Program& Program::operator=(Program&&) noexcept = default;

Program Program::Load(Device& device, const std::filesystem::path& xclbin,
                      std::string_view kernel_prefix) {
  if (!device.impl_)
    throw Error("NPU device is not open");
  const GufoNpuXrt* xrt = device.impl_->xrt;
  Program out;
  out.impl_ = std::make_unique<Impl>();
  out.impl_->xrt = xrt;
  const std::string path = xclbin.string();  // as XRT's xclbin(std::string)
  const char* kernel_name = nullptr;
  Check("cannot load " + xclbin.string(),
        xrt->program_load(device.impl_->device, path.c_str(),
                          std::string(kernel_prefix).c_str(),
                          &out.impl_->program, &kernel_name));
  out.impl_->kernel_name = kernel_name ? kernel_name : "";
  return out;
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
  Buffer out;
  out.impl_ = std::make_unique<Buffer::Impl>();
  out.impl_->xrt = impl_->xrt;
  Check("NPU buffer allocation",
        impl_->xrt->buffer_allocate(impl_->program, bytes, argument,
                                    memory == Memory::kCacheable ? 1 : 0,
                                    &out.impl_->bo, &out.impl_->map));
  out.impl_->size = bytes;
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
  std::vector<void*> buffers(arguments.size());
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    if (arguments[i] == nullptr)
      continue;
    if (!*arguments[i])
      throw Error("NPU run argument " + std::to_string(kFirstDataArgument + i) +
                  " is empty");
    buffers[i] = arguments[i]->impl_->bo;
  }
  Run out;
  out.impl_ = std::make_unique<Run::Impl>();
  out.impl_->xrt = impl_->xrt;
  Check("NPU run start",
        impl_->xrt->run_start(impl_->program, kOpcode, kInstructionArgument,
                              instructions.buffer().impl_->bo,
                              static_cast<std::uint32_t>(instructions.count()),
                              kFirstDataArgument, buffers.data(),
                              buffers.size(), &out.impl_->run));
  return out;
}

}  // namespace gufo::npu

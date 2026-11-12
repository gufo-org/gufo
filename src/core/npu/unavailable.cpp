// The NPU runtime of builds without XRT (GUFO_ENABLE_XRT=OFF): the same API,
// with no device to open. No buffer, program or run can exist, so their
// members only report that.

#include "src/core/npu/npu.hpp"

namespace gufo::npu {
namespace {

[[noreturn]] void Unavailable() {
  throw Error("this Gufo build has no NPU runtime (built without XRT)");
}

}  // namespace

bool RuntimeAvailable() noexcept {
  return false;
}

struct Device::Impl {};
struct Buffer::Impl {};
struct Run::Impl {};
struct Program::Impl {};

Device::Device() = default;
Device::~Device() = default;
Device::Device(Device&&) noexcept = default;
Device& Device::operator=(Device&&) noexcept = default;
Device Device::Open(unsigned) {
  Unavailable();
}
std::string Device::name() const {
  return {};
}

Buffer::Buffer() = default;
Buffer::~Buffer() = default;
Buffer::Buffer(Buffer&&) noexcept = default;
Buffer& Buffer::operator=(Buffer&&) noexcept = default;
std::size_t Buffer::size() const noexcept {
  return 0;
}
std::byte* Buffer::data() const {
  return nullptr;
}
void Buffer::ToDevice(std::size_t, std::size_t) {
  Unavailable();
}
void Buffer::FromDevice(std::size_t, std::size_t) {
  Unavailable();
}

Run::Run() = default;
Run::~Run() = default;
Run::Run(Run&&) noexcept = default;
Run& Run::operator=(Run&&) noexcept = default;
void Run::Wait() {}

Program::Program() = default;
Program::~Program() = default;
Program::Program(Program&&) noexcept = default;
Program& Program::operator=(Program&&) noexcept = default;
Program Program::Load(Device&, const std::filesystem::path&, std::string_view) {
  Unavailable();
}
const std::string& Program::kernel_name() const noexcept {
  static const std::string empty;
  return empty;
}
Buffer Program::Allocate(std::size_t, unsigned, Memory) const {
  Unavailable();
}
Buffer Program::LoadBuffer(const std::filesystem::path&, unsigned,
                           Memory) const {
  Unavailable();
}
Instructions Program::AllocateInstructions(std::size_t) const {
  Unavailable();
}
Instructions Program::LoadInstructions(const std::filesystem::path&) const {
  Unavailable();
}
Run Program::Start(const Instructions&, std::span<const Buffer* const>) const {
  Unavailable();
}

}  // namespace gufo::npu

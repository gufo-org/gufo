// The only code that calls XRT (see xrt_backend.hpp). Nothing here names a
// Gufo symbol: on Linux this is a module beside an executable that exports
// Gufo's symbols to extensions, and its own code must not bind to them.

#include "src/core/npu/xrt_backend.hpp"

#include <xrt/experimental/xrt_system.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#define GUFO_NPU_XRT_EXPORT
#else
#define GUFO_NPU_XRT_EXPORT __attribute__((visibility("default")))
#endif

namespace {

thread_local std::string last_error;

template<typename F>
const char* Guard(F&& f) noexcept {
  try {
    f();
    return nullptr;
  } catch (const std::exception& e) {
    last_error = e.what();
  } catch (...) {
    last_error = "unknown XRT error";
  }
  return last_error.c_str();
}

struct Device {
  xrt::device device;
  std::string name;
};

struct Program {
  std::shared_ptr<Device> device;
  xrt::hw_context context;
  xrt::kernel kernel;
  std::string kernel_name;
};

struct Buffer {
  std::shared_ptr<Device> device;
  xrt::bo bo;
};

struct Run {
  std::shared_ptr<Device> device;
  xrt::run run;
};

// Handles are owning pointers to shared_ptr so that programs, buffers and
// runs share the device.
std::shared_ptr<Device>& DeviceOf(void* handle) {
  return *static_cast<std::shared_ptr<Device>*>(handle);
}

const char* DeviceCount(unsigned* count) {
  return Guard([&] { *count = xrt::system::enumerate_devices(); });
}

const char* DeviceOpen(unsigned index, void** device) {
  return Guard([&] {
    auto opened = std::make_shared<Device>(Device{xrt::device(index), {}});
    *device = new std::shared_ptr<Device>(std::move(opened));
  });
}

const char* DeviceName(void* device, const char** name) {
  return Guard([&] {
    Device& d = *DeviceOf(device);
    if (d.name.empty())
      d.name = d.device.get_info<xrt::info::device::name>();
    *name = d.name.c_str();
  });
}

void DeviceClose(void* device) {
  delete static_cast<std::shared_ptr<Device>*>(device);
}

const char* ProgramLoad(void* device, const char* path, const char* prefix,
                        void** program, const char** kernel_name) {
  return Guard([&] {
    const std::shared_ptr<Device>& d = DeviceOf(device);
    xrt::xclbin file{std::string(path)};
    d->device.register_xclbin(file);
    auto out = std::make_unique<Program>();
    out->device = d;
    out->context = xrt::hw_context(d->device, file.get_uuid());
    const std::string wanted(prefix);
    for (const auto& k : file.get_kernels()) {
      if (k.get_name().rfind(wanted, 0) == 0) {
        out->kernel_name = k.get_name();
        out->kernel = xrt::kernel(out->context, out->kernel_name);
        *kernel_name = out->kernel_name.c_str();
        *program = out.release();
        return;
      }
    }
    throw std::runtime_error(std::string(path) + ": no kernel named " + wanted +
                             "*");
  });
}

void ProgramClose(void* program) {
  delete static_cast<Program*>(program);
}

const char* BufferAllocate(void* program, std::size_t bytes, unsigned argument,
                           int cacheable, void** buffer, std::byte** map) {
  return Guard([&] {
    const Program& p = *static_cast<Program*>(program);
    const auto flags =
        cacheable ? xrt::bo::flags::cacheable : xrt::bo::flags::host_only;
    auto out = std::make_unique<Buffer>();
    out->device = p.device;
    out->bo = xrt::bo(p.device->device, bytes, flags,
                      p.kernel.group_id(static_cast<int>(argument)));
    *map = out->bo.map<std::byte*>();
    *buffer = out.release();
  });
}

const char* BufferSync(void* buffer, int to_device, std::size_t bytes,
                       std::size_t offset) {
  return Guard([&] {
    static_cast<Buffer*>(buffer)->bo.sync(
        to_device ? XCL_BO_SYNC_BO_TO_DEVICE : XCL_BO_SYNC_BO_FROM_DEVICE,
        bytes, offset);
  });
}

void BufferClose(void* buffer) {
  delete static_cast<Buffer*>(buffer);
}

const char* RunStart(void* program, std::uint32_t opcode,
                     int instruction_argument, void* instructions,
                     std::uint32_t words, unsigned first_argument,
                     void* const* buffers, std::size_t count, void** run) {
  return Guard([&] {
    const Program& p = *static_cast<Program*>(program);
    auto out = std::make_unique<Run>();
    out->device = p.device;
    xrt::run r(p.kernel);
    r.set_arg(0, opcode);
    r.set_arg(instruction_argument, static_cast<Buffer*>(instructions)->bo);
    r.set_arg(2, words);
    for (std::size_t i = 0; i < count; ++i) {
      if (buffers[i] != nullptr)
        r.set_arg(static_cast<int>(first_argument + i),
                  static_cast<Buffer*>(buffers[i])->bo);
    }
    r.start();
    out->run = std::move(r);
    *run = out.release();
  });
}

const char* RunWait(void* run, int* state) {
  return Guard(
      [&] { *state = static_cast<int>(static_cast<Run*>(run)->run.wait()); });
}

void RunClose(void* run, int waited) {
  std::unique_ptr<Run> owned(static_cast<Run*>(run));
  if (!waited) {
    try {
      owned->run.wait();
    } catch (...) {
    }
  }
}

constexpr GufoNpuXrt kApi{
    GUFO_NPU_XRT_VERSION,
    DeviceCount,
    DeviceOpen,
    DeviceName,
    DeviceClose,
    ProgramLoad,
    ProgramClose,
    BufferAllocate,
    BufferSync,
    BufferClose,
    RunStart,
    RunWait,
    RunClose,
    static_cast<int>(ERT_CMD_STATE_COMPLETED),
};

}  // namespace

extern "C" GUFO_NPU_XRT_EXPORT const GufoNpuXrt* gufo_npu_xrt(void) {
  return &kApi;
}

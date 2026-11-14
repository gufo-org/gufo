// The boundary between Gufo and the code that calls XRT (xrt_backend.cpp).
// Linux builds that code as its own module, loaded on first NPU use, so the
// executable neither needs XRT to start nor searches XRT's library directory
// for its other libraries. Windows links it into Gufo and delay-loads
// xrt_coreutil.dll instead. Only C types cross: the module has its own copy
// of the C++ runtime state and never throws across the boundary.
#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

/// Bumped whenever a field changes; Gufo refuses a module of another version.
#define GUFO_NPU_XRT_VERSION 1u
/// The one symbol the module exports: `const GufoNpuXrt* (void)`.
#define GUFO_NPU_XRT_SYMBOL "gufo_npu_xrt"

/// Handles own their XRT objects; programs, buffers and runs keep their
/// device open. A call that can fail returns null on success, or an error
/// message valid until the calling thread's next call.
struct GufoNpuXrt {
  std::uint32_t version;

  const char* (*device_count)(unsigned* count);
  const char* (*device_open)(unsigned index, void** device);
  /// `name` stays valid until the device closes.
  const char* (*device_name)(void* device, const char** name);
  void (*device_close)(void* device);

  /// Loads the xclbin at `path` (std::filesystem::path::string()) and opens its
  /// first kernel whose name starts with `prefix`; `kernel_name` stays valid
  /// until the program closes.
  const char* (*program_load)(void* device, const char* path,
                              const char* prefix, void** program,
                              const char** kernel_name);
  void (*program_close)(void* program);

  /// A buffer in the memory group of kernel argument `argument`, mapped.
  const char* (*buffer_allocate)(void* program, std::size_t bytes,
                                 unsigned argument, int cacheable,
                                 void** buffer, std::byte** map);
  const char* (*buffer_sync)(void* buffer, int to_device, std::size_t bytes,
                             std::size_t offset);
  void (*buffer_close)(void* buffer);

  /// Starts the kernel with opcode `opcode`, the instruction stream
  /// (`words` 32-bit words) at argument `instruction_argument`, and
  /// `count` data buffers from `first_argument` on; a null buffer passes none.
  const char* (*run_start)(void* program, std::uint32_t opcode,
                           int instruction_argument, void* instructions,
                           std::uint32_t words, unsigned first_argument,
                           void* const* buffers, std::size_t count, void** run);
  /// Waits and reports XRT's final command state.
  const char* (*run_wait)(void* run, int* state);
  /// Closes a run, first waiting for it unless it was waited.
  void (*run_close)(void* run, int waited);
  /// XRT's state of a completed command.
  int completed_state;
};

const GufoNpuXrt* gufo_npu_xrt(void);

}  // extern "C"

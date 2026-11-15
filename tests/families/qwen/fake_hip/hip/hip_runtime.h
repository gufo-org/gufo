#ifndef GUFO_TESTS_FAKE_HIP_RUNTIME_H_
#define GUFO_TESTS_FAKE_HIP_RUNTIME_H_

#include <cstddef>

// This header is visible only to the CPU DeviceInput lifecycle target. It
// provides synchronous logical allocations, not a GPU runtime or kernel test.
using hipStream_t = void*;
enum hipError_t { hipSuccess, hipErrorMemoryAllocation, hipErrorUnknown };
enum hipMemcpyKind { hipMemcpyHostToDevice };

namespace gufo::models::qwen::vision {
struct DeviceRope;
}

hipError_t hipMalloc(void** pointer, std::size_t bytes);
// Classify the stable text-base descriptor by its production pointer type,
// never by allocation size: an ordinary dynamic buffer can have that size.
hipError_t hipMalloc(gufo::models::qwen::vision::DeviceRope** pointer,
                     std::size_t bytes);
template<class T>
hipError_t hipMalloc(T** pointer, std::size_t bytes) {
  return hipMalloc(reinterpret_cast<void**>(pointer), bytes);
}
hipError_t hipFree(void* pointer);
hipError_t hipMemcpyAsync(void* destination, const void* source,
                          std::size_t bytes, hipMemcpyKind, hipStream_t);
hipError_t hipStreamSynchronize(hipStream_t);
const char* hipGetErrorString(hipError_t error);

#endif

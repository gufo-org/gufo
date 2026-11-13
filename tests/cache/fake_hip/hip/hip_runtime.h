#ifndef GUFO_TESTS_CACHE_FAKE_HIP_RUNTIME_H_
#define GUFO_TESTS_CACHE_FAKE_HIP_RUNTIME_H_

#include <cstddef>

// Only the CPU backing-admission target sees these declarations. Physical
// payloads are small host pages; registration, streams and events are
// simulated.
using hipStream_t = void*;
using hipEvent_t = void*;
enum hipError_t {
  hipSuccess,
  hipErrorInvalidValue,
  hipErrorOutOfMemory,
  hipErrorNotReady
};
enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost };
inline constexpr unsigned hipHostMallocDefault = 0;
inline constexpr unsigned hipHostMallocCoherent = 1;
inline constexpr unsigned hipHostRegisterDefault = 0;
inline constexpr unsigned hipStreamNonBlocking = 0;
inline constexpr unsigned hipEventDisableTiming = 0;
inline constexpr unsigned hipEventReleaseToSystem = 0x80000000;

hipError_t hipHostMalloc(void**, std::size_t, unsigned);
hipError_t hipHostFree(void*);
hipError_t hipHostRegister(void*, std::size_t, unsigned);
hipError_t hipHostUnregister(void*);
hipError_t hipMemGetInfo(std::size_t*, std::size_t*);
hipError_t hipStreamCreateWithFlags(hipStream_t*, unsigned);
hipError_t hipStreamDestroy(hipStream_t);
hipError_t hipStreamSynchronize(hipStream_t);
hipError_t hipEventCreateWithFlags(hipEvent_t*, unsigned);
hipError_t hipEventDestroy(hipEvent_t);
hipError_t hipEventRecord(hipEvent_t, hipStream_t);
hipError_t hipEventQuery(hipEvent_t);
hipError_t hipEventSynchronize(hipEvent_t);
hipError_t hipMemcpyAsync(void*, const void*, std::size_t, hipMemcpyKind,
                          hipStream_t);
const char* hipGetErrorString(hipError_t);

#endif

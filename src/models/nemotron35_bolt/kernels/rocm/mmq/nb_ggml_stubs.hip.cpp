// SPDX-License-Identifier: MIT
// Model-private HIP runtime helpers for the imported ggml matrix templates:
// the device-info singleton, device selection, the error hook and a
// per-stream scratch arena. See VENDOR.md.

#include "nb_mmq_prelude.h"

namespace nb_mmq {

#include "common.hip.hpp"  // pulls in nb_ggml_stubs.h via redirect headers

// Device properties are queried once and shared by the matrix dispatchers.
const nb_ggml_hip_device_info & nb_ggml_hip_info() {
    static nb_ggml_hip_device_info info;
    static std::once_flag once;
    std::call_once(once, []{
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        if (err != hipSuccess) {
            fprintf(stderr, "nb_ggml_hip_info: hipGetDeviceCount failed: %s\n", hipGetErrorString(err));
            count = 0;
        }
        if (count > NB_GGML_HIP_MAX_DEVICES) count = NB_GGML_HIP_MAX_DEVICES;
        info.device_count = count;
        for (int i = 0; i < count; i++) {
            hipDeviceProp_t p;
            NB_HIP_CHECK(hipGetDeviceProperties(&p, i));
            unsigned arch = 0;
            if (sscanf(p.gcnArchName, "gfx%x", &arch) != 1) {
                fprintf(stderr, "nb_ggml_hip_info: unsupported HIP architecture '%s'\n", p.gcnArchName);
            }
            info.devices[i].cc                          = NB_GGML_HIP_CC_OFFSET_AMD + (int)arch;
            info.devices[i].nsm                         = p.multiProcessorCount;
            info.devices[i].smpb                        = p.sharedMemPerBlock;
            info.devices[i].smpbo                       = p.sharedMemPerBlockOptin;
            info.devices[i].integrated                  = p.integrated != 0;
            info.devices[i].vmm                         = false;
            info.devices[i].vmm_granularity             = 0;
            info.devices[i].total_vram                  = p.totalGlobalMem;
            info.devices[i].warp_size                   = p.warpSize;
            info.devices[i].supports_cooperative_launch = p.cooperativeLaunch != 0;
        }
    });
    return info;
}

int nb_ggml_hip_get_device() {
    int dev = 0;
    (void)hipGetDevice(&dev);
    return dev;
}

void nb_ggml_hip_set_device(int device) {
    int cur = -1;
    (void)hipGetDevice(&cur);
    if (cur != device) {
        NB_HIP_CHECK(hipSetDevice(device));
    }
}

int64_t ggml_time_us() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
}

// Invoked by NB_HIP_CHECK / NB_HIPBLAS_CHECK on the error path; declared
// [[noreturn]] in common.hip.hpp.
[[noreturn]] void nb_ggml_hip_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "HIP error: %s\n  call: %s\n  in: %s at %s:%d\n", msg, stmt, func, file, line);
    fflush(stderr);
    abort();
}

namespace {

// Stream the scratch of the current call is ordered on; every public entry
// sets it before allocating.
thread_local hipStream_t t_nb_pool_stream = nullptr;

// Scratch arena per stream. A call allocates a few buffers, launches on its
// stream and releases them before returning, so a bump stack over one reserved
// buffer serves every call: the next call on the same stream is ordered after
// the previous one's kernels. This avoids stream-ordered allocation, which
// costs host time per call and which graph capture under ROCm 7.1 replayed
// with wrong contents. Distinct streams (sessions) get distinct arenas.
constexpr size_t kArenaBytes = size_t(256) << 20;
constexpr size_t kArenaAlign = 256;

struct nb_arena {
    char * base = nullptr;
    size_t top = 0;
    int live = 0;
};

struct nb_arena_pool : public nb_ggml_hip_pool {
    int device;
    std::mutex mutex;
    std::unordered_map<hipStream_t, nb_arena> arenas;

    explicit nb_arena_pool(int device) : device(device) {}

    ~nb_arena_pool() override {
        for (auto & [stream, arena] : arenas) {
            (void)stream;
            (void)hipFree(arena.base);
        }
    }

    // Allocates the arena of `stream` if it has none yet.
    hipError_t reserve(hipStream_t stream) {
        nb_ggml_hip_set_device(device);
        std::lock_guard<std::mutex> lock(mutex);
        nb_arena & arena = arenas[stream];
        return arena.base != nullptr ? hipSuccess : hipMalloc(&arena.base, kArenaBytes);
    }

    void release(hipStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = arenas.find(stream);
        if (it == arenas.end()) return;
        (void)hipFree(it->second.base);
        arenas.erase(it);
    }

    void * alloc(size_t size, size_t * actual_size) override {
        nb_ggml_hip_set_device(device);
        const size_t aligned = (size + kArenaAlign - 1) / kArenaAlign * kArenaAlign;
        std::lock_guard<std::mutex> lock(mutex);
        nb_arena & arena = arenas[t_nb_pool_stream];
        if (arena.base == nullptr) {
            NB_HIP_CHECK(hipMalloc(&arena.base, kArenaBytes));
        }
        if (aligned > kArenaBytes - arena.top) {
            nb_ggml_hip_error("arena alloc", __func__, __FILE__, __LINE__,
                              "matrix scratch exceeds the per-stream arena");
        }
        void * ptr = arena.base + arena.top;
        arena.top += aligned;
        ++arena.live;
        if (actual_size) *actual_size = size;
        return ptr;
    }

    void free(void * ptr, size_t /*size*/) override {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mutex);
        nb_arena & arena = arenas[t_nb_pool_stream];
        if (--arena.live == 0) {
            arena.top = 0;
        }
    }
};

} // anonymous namespace

void nb_pool_set_stream(hipStream_t stream) {
    t_nb_pool_stream = stream;
}

hipError_t nb_pool_reserve_stream(nb_ggml_hip_pool & pool, hipStream_t stream) {
    return static_cast<nb_arena_pool &>(pool).reserve(stream);
}

void nb_pool_release_stream(nb_ggml_hip_pool & pool, hipStream_t stream) {
    static_cast<nb_arena_pool &>(pool).release(stream);
}

std::unique_ptr<nb_ggml_hip_pool> nb_ggml_hip_context::new_pool_for_device(int device, int /*stream_no*/) {
    return std::unique_ptr<nb_ggml_hip_pool>(new nb_arena_pool(device));
}

nb_ggml_hip_context::~nb_ggml_hip_context() {
    if (copy_event) {
        (void)hipEventDestroy(copy_event);
        copy_event = nullptr;
    }
    // Streams and hipBLAS handles are never created by the adapter; the pools
    // destruct through unique_ptr.
}

}  // namespace nb_mmq

#include "qfn_mmq_prelude.h"
#include <gufo/ext/experimental/managed_alloc.hpp>
namespace qfn_mmq {
// SPDX-License-Identifier: MIT
// Implementations of the ggml-API stubs declared in qfn_ggml_stubs.h plus
// the bodies of ggml_backend_hip_context / ggml_hip_info /
// new_pool_for_device that the vendored common.hpp declares without
// defining.
//

#include "common.hpp"   // pulls in qfn_ggml_stubs.h via redirect headers

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <map>
#include <mutex>
#include <vector>

// ----------------------------------------------------------------------------
// Device info singleton.
//
// Common.hpp declares `const ggml_hip_device_info & ggml_hip_info();` -
// we provide the body. The struct layout (common.hpp:1091) is:
//   { int device_count; hip_device_info devices[GGML_HIP_MAX_DEVICES];
//     std::array<float, GGML_HIP_MAX_DEVICES> default_tensor_split; }
// where hip_device_info has { cc, nsm, smpb, smpbo, integrated, vmm,
// vmm_granularity, total_vram, warp_size, supports_cooperative_launch }.
// ----------------------------------------------------------------------------

const ggml_hip_device_info & ggml_hip_info() {
    static ggml_hip_device_info info;
    static std::once_flag once;
    std::call_once(once, []{
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        if (err != hipSuccess) {
            fprintf(stderr, "ggml_hip_info: hipGetDeviceCount failed: %s\n", hipGetErrorString(err));
            count = 0;
        }
        if (count > GGML_HIP_MAX_DEVICES) count = GGML_HIP_MAX_DEVICES;
        info.device_count = count;
        for (int i = 0; i < count; i++) {
            hipDeviceProp_t p;
            HIP_CHECK(hipGetDeviceProperties(&p, i));
            unsigned arch = 0;
            if (sscanf(p.gcnArchName, "gfx%x", &arch) != 1) {
                fprintf(stderr, "ggml_hip_info: unsupported HIP architecture '%s'\n", p.gcnArchName);
            }
            info.devices[i].cc                          = GGML_HIP_CC_OFFSET_AMD + (int)arch;
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

// ggml_hip_get_device / ggml_hip_set_device are declared (not defined) in
// common.hpp. We provide thin wrappers.

int ggml_hip_get_device() {
    int dev = 0;
    hipGetDevice(&dev);
    return dev;
}

void ggml_hip_set_device(int device) {
    int cur = -1;
    hipGetDevice(&cur);
    if (cur != device) {
        HIP_CHECK(hipSetDevice(device));
    }
}

int64_t ggml_time_us() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
}

// ----------------------------------------------------------------------------
// ggml_hip_error: invoked by the HIP_CHECK / HIPBLAS_CHECK macros defined
// in common.hpp on the error path. Marked [[noreturn]] in the declaration
// (common.hpp:155) - abort() satisfies that contract.
// ----------------------------------------------------------------------------

[[noreturn]] void ggml_hip_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "HIP error: %s\n  call: %s\n  in: %s at %s:%d\n", msg, stmt, func, file, line);
    fflush(stderr);
    abort();
}

// ----------------------------------------------------------------------------
// Scratch pool over one device arena.
// ----------------------------------------------------------------------------

namespace {

/* qwen38: every allocation the tier makes is scoped to one entry point and
 * released in reverse order, and each arena serves one stream (qfn_pool_for_stream),
 * so stream order makes a stack over device memory enough. It replaces
 * hipMallocAsync/hipFreeAsync, whose reuse of a just-freed block raced with
 * in-flight kernels on this ROCm and made prefill logits nondeterministic.
 *
 * The stack spans a list of blocks. An allocation that doesn't fit the current
 * block goes to the next one that fits (added if needed, at least twice the
 * last), so an arena can start small: the MTP and DFlash2 streams only quantize
 * a few rows. Blocks live as long as the pool: captured graphs bake in their
 * addresses and replay them after the arena has grown. */
struct qfn_stack_pool : public ggml_hip_pool {
    static constexpr size_t kFirstBlockBytes = (size_t) 16 << 20;

    struct block {
        char *base;
        size_t capacity;
        size_t top;
    };

    struct live_alloc {
        char *ptr;
        size_t bytes;
        size_t block;
    };

    int device;
    std::vector<block> blocks;
    size_t current = 0;
    std::vector<live_alloc> live;

    explicit qfn_stack_pool(int device) : device(device) {}

    ~qfn_stack_pool() override {
        for (const block &b : blocks) (void)hipFree(b.base);
    }

    void add_block(size_t bytes) {
        /* Drain the device first, as the single arena did. On ROCm 10.1 an
         * allocation made while kernels are in flight leaves later kernel
         * launches slower for seconds (+14 ms per short MTP request). Blocks
         * are only added by eager passes, never during capture. */
        (void)hipDeviceSynchronize();
        char *base = nullptr;
        HIP_CHECK(gufo::hip::AllocateDevice(&base, bytes));
        blocks.push_back({base, bytes, 0});
    }

    void * alloc(size_t size, size_t * actual_size) override {
        ggml_hip_set_device(device);
        const size_t aligned = (size + 255) & ~(size_t) 255;
        if (blocks.empty()) {
            size_t first = kFirstBlockBytes;
            while (first < aligned) first *= 2;
            add_block(first);
        }
        if (blocks[current].top + aligned > blocks[current].capacity) {
            // Blocks after `current` are empty: the stack only reaches them through it.
            size_t next = current + 1;
            while (next < blocks.size() && blocks[next].capacity < aligned) ++next;
            if (next == blocks.size()) {
                size_t bytes = blocks.back().capacity * 2;
                while (bytes < aligned) bytes *= 2;
                add_block(bytes);
            }
            current = next;
        }
        block &b = blocks[current];
        char *ptr = b.base + b.top;
        b.top += aligned;
        live.push_back({ptr, aligned, current});
        if (actual_size) *actual_size = aligned;
        return ptr;
    }

    void free(void * ptr, size_t /*size*/) override {
        if (!ptr) return;
        if (live.empty() || live.back().ptr != ptr) {
            fprintf(stderr, "qfn_stack_pool: non-LIFO free\n");
            abort();
        }
        const live_alloc a = live.back();
        live.pop_back();
        blocks[a.block].top -= a.bytes;
        while (current > 0 && blocks[current].top == 0) --current;
    }
};

} // anonymous namespace

std::unique_ptr<ggml_hip_pool> ggml_backend_hip_context::new_pool_for_device(int device, int /*stream_no*/) {
    return std::unique_ptr<ggml_hip_pool>(new qfn_stack_pool(device));
}

// One scratch arena per (device, stream). The stack pool above relies on stream order to
// reuse memory safely, so it must never be shared by two streams: the MTP drafter runs on its
// own stream and reached these kernels too, and reused the target's scratch while the target's
// kernels were still reading it.
namespace {
std::mutex g_stream_pools_mutex;
std::map<std::pair<int, hipStream_t>, std::unique_ptr<ggml_hip_pool>> g_stream_pools;
} // anonymous namespace

ggml_hip_pool & qfn_pool_for_stream(int device, hipStream_t stream) {
    const std::lock_guard<std::mutex> lock(g_stream_pools_mutex);
    auto & pool = g_stream_pools[{device, stream}];
    if (!pool) pool = ggml_backend_hip_context::new_pool_for_device(device, 0);
    return *pool;
}

// The caller must have stopped every host submission and allocation on `stream` first: the
// lock protects the map, not the pool references handed out by qfn_pool_for_stream.
extern "C" void qfn_mmq_release_stream(hipStream_t stream) {
    const std::lock_guard<std::mutex> lock(g_stream_pools_mutex);
    for (auto it = g_stream_pools.begin(); it != g_stream_pools.end();) {
        if (it->first.second != stream) {
            ++it;
            continue;
        }
        // The arena may still back queued work on this stream. If the wait fails, the work
        // may still be running, so keep (leak) the arena rather than free memory in use.
        const hipError_t status = hipStreamSynchronize(stream);
        if (status != hipSuccess) {
            fprintf(stderr, "qfn_mmq_release_stream: stream %p not synchronized (%s); keeping its scratch arena\n",
                    (void *) stream, hipGetErrorString(status));
            (void) it->second.release();
        }
        it = g_stream_pools.erase(it);
    }
}

ggml_backend_hip_context::~ggml_backend_hip_context() {
    if (copy_event) {
        hipEventDestroy(copy_event);
        copy_event = nullptr;
    }
    // streams[][], hipblas_handles[], and pools[][] are owned-by-value
    // (hipStream_t and hipblasHandle_t are opaque handles - destroying the
    // context "should" tear them down, but in our shim ds4 manages streams
    // externally and we leave them alone. The pools auto-destruct via
    // unique_ptr.).
}

}  // namespace qfn_mmq

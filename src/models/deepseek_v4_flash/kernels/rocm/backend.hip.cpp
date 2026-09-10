#include "backend.h"
#include <hipblaslt/hipblaslt.h>

/* Vendored llama.cpp quantized-matmul tier. It owns the routed IQ2 gate/up and
 * dense Q8 prefill kernels; see mmq/VENDOR.md. */
#include "mmq/ds4_mmq.h"

/* MMQ can reuse a producer-emitted Q8_1 activation instead of quantizing its
 * own. This backend keeps no such registry, so MMQ always takes its regular
 * activation-quantize prelude. */
extern "C" int ds4_cuda_q8_fold_take_q81(
        const void *src, uint64_t in_dim, const void **q81) {
    (void)src;
    (void)in_dim;
    if (q81) *q81 = NULL;
    return 0;
}

#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#define MASK_T uint64_t
#define DS4_GPU_BACKEND_NAME "ROCm"
#define DS4_GPU_LOG_PREFIX "ds4: ROCm "
#define DS4_GPU_BLAS_NAME "hipBLAS"

#include <stdint.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "resident_api.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_ROCM_ATTENTION_SCORE_CAP = 8192u,
    DS4_ROCM_ATTENTION_RAW_SCORE_CAP = 256u
};

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

#include "detail/ds4_rocm_runtime.hip.hpp"

#include "detail/ds4_rocm_common.hip.hpp"

#include "detail/ds4_rocm_q8.hip.hpp"

#include "detail/ds4_rocm_norm_rope.hip.hpp"

#include "detail/ds4_rocm_fp8_kv.hip.hpp"

#include "detail/ds4_rocm_attention.hip.hpp"

#include "detail/ds4_rocm_hc.hip.hpp"

#include "detail/ds4_rocm_output.hip.hpp"

#include "detail/ds4_rocm_indexer.hip.hpp"

#include "detail/ds4_rocm_embedding_launch.hip.hpp"

#include "detail/ds4_rocm_matmul.hip.hpp"

#include "detail/ds4_rocm_fp8_kv_launch.hip.hpp"

#include "detail/ds4_rocm_compressor.hip.hpp"

#include "detail/ds4_rocm_attention_launch.hip.hpp"

#include "detail/ds4_rocm_shared_expert.hip.hpp"

#include "detail/ds4_rocm_misc_launch.hip.hpp"
#include "detail/ds4_rocm_router.hip.hpp"

#include "detail/ds4_rocm_moe.hip.hpp"

#include "detail/ds4_rocm_moe_launch.hip.hpp"

#include "detail/ds4_rocm_hc_output_launch.hip.hpp"

#include "backend.h"
#include <hipblaslt/hipblaslt.h>

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
#include <strings.h>
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

#define ROCM_QK_K 256
#define DS4_ROCM_UNUSED __attribute__((unused))

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

typedef struct {
    uint8_t scales[ROCM_QK_K / 16];
    uint8_t qs[ROCM_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} hip_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[ROCM_QK_K / 2];
} hip_block_q4_K;

typedef struct {
    float d;
    int8_t qs[ROCM_QK_K];
    int16_t bsums[ROCM_QK_K / 16];
} hip_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[ROCM_QK_K / 8];
} hip_block_iq2_xxs;

#include "iq2_tables.inc"

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

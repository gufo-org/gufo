#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_MODEL_DATA_INTERNAL_H_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_MODEL_DATA_INTERNAL_H_

#include <cstddef>
#include <cstdint>

#include "model.h"
#include "tensor_types.h"

inline constexpr float DS4_NEG_INF = -1.0e30F;
inline constexpr float DS4_POS_INF = 1.0e30F;

inline constexpr uint32_t DS4_N_LAYER = 43;
inline constexpr uint32_t DS4_N_EMBD = 4096;
inline constexpr uint32_t DS4_N_VOCAB = 129280;
inline constexpr uint32_t DS4_N_HEAD = 64;
inline constexpr uint32_t DS4_N_HEAD_KV = 1;
inline constexpr uint32_t DS4_N_HEAD_DIM = 512;
inline constexpr uint32_t DS4_N_VALUE_DIM = 512;
inline constexpr uint32_t DS4_N_ROT = 64;
inline constexpr uint32_t DS4_N_OUT_GROUP = 8;
inline constexpr uint32_t DS4_N_LORA_Q = 1024;
inline constexpr uint32_t DS4_N_LORA_O = 1024;
inline constexpr uint32_t DS4_N_EXPERT = 256;
inline constexpr uint32_t DS4_N_EXPERT_USED = 6;
inline constexpr uint32_t DS4_N_EXPERT_SHARED = 1;
inline constexpr uint32_t DS4_N_FF_EXP = 2048;
inline constexpr uint32_t DS4_N_HASH_LAYER = 3;
inline constexpr uint32_t DS4_N_SWA = 128;
inline constexpr uint32_t DS4_N_INDEXER_HEAD = 64;
inline constexpr uint32_t DS4_N_INDEXER_HEAD_DIM = 128;
inline constexpr uint32_t DS4_N_INDEXER_TOP_K = 512;
inline constexpr uint32_t DS4_N_HC = 4;
inline constexpr uint32_t DS4_N_HC_SINKHORN_ITER = 20;
inline constexpr uint32_t DS4_MAX_DIMS = 8;
inline constexpr uint32_t DS4_MAX_LAYER = DS4_N_LAYER;

inline constexpr char DS4_MODEL_SHAPE_NAME[] = "DeepSeek V4 Flash";
inline constexpr float DS4_RMS_EPS = 1.0e-6F;
inline constexpr float DS4_HC_EPS = 1.0e-6F;
inline constexpr float DS4_EXPERT_WEIGHT_SCALE = 1.5F;
inline constexpr float DS4_SWIGLU_CLAMP_EXP = 10.0F;
inline constexpr float DS4_ROPE_FREQ_BASE = 10000.0F;
inline constexpr float DS4_ROPE_SCALE_FACTOR = 16.0F;
inline constexpr float DS4_ROPE_YARN_BETA_FAST = 32.0F;
inline constexpr float DS4_ROPE_YARN_BETA_SLOW = 1.0F;
inline constexpr float DS4_COMPRESS_ROPE_FREQ_BASE = 160000.0F;
inline constexpr uint64_t DS4_ROPE_ORIG_CTX = 65536;

struct ds4_str {
    const char *ptr;
    uint64_t len;
};

struct ds4_kv {
    ds4_str key;
    uint32_t type;
    uint64_t value_pos;
};

struct ds4_tensor {
    ds4_str name;
    uint32_t ndim;
    uint64_t dim[DS4_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
};

struct ds4_model {
    int fd;
    const uint8_t *map;
    uint64_t size;
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    uint64_t max_tensor_bytes;
    ds4_kv *kv;
    ds4_tensor *tensors;
};

struct ds4_layer_weights {
    ds4_tensor *hc_attn_fn;
    ds4_tensor *hc_attn_scale;
    ds4_tensor *hc_attn_base;
    ds4_tensor *attn_norm;
    ds4_tensor *attn_q_a;
    ds4_tensor *attn_q_a_norm;
    ds4_tensor *attn_q_b;
    ds4_tensor *attn_kv;
    ds4_tensor *attn_kv_a_norm;
    ds4_tensor *attn_sinks;
    ds4_tensor *attn_output_a;
    ds4_tensor *attn_output_b;
    ds4_tensor *attn_compressor_ape;
    ds4_tensor *attn_compressor_kv;
    ds4_tensor *attn_compressor_gate;
    ds4_tensor *attn_compressor_norm;
    ds4_tensor *indexer_attn_q_b;
    ds4_tensor *indexer_proj;
    ds4_tensor *indexer_compressor_ape;
    ds4_tensor *indexer_compressor_kv;
    ds4_tensor *indexer_compressor_gate;
    ds4_tensor *indexer_compressor_norm;
    ds4_tensor *hc_ffn_fn;
    ds4_tensor *hc_ffn_scale;
    ds4_tensor *hc_ffn_base;
    ds4_tensor *ffn_norm;
    ds4_tensor *ffn_gate_tid2eid;
    ds4_tensor *ffn_gate_inp;
    ds4_tensor *ffn_exp_probs_b;
    ds4_tensor *ffn_gate_exps;
    ds4_tensor *ffn_up_exps;
    ds4_tensor *ffn_down_exps;
    ds4_tensor *ffn_gate_shexp;
    ds4_tensor *ffn_up_shexp;
    ds4_tensor *ffn_down_shexp;
};

struct ds4_weights {
    ds4_tensor *token_embd;
    ds4_tensor *output_hc_base;
    ds4_tensor *output_hc_fn;
    ds4_tensor *output_hc_scale;
    ds4_tensor *output_norm;
    ds4_tensor *output;
    ds4_layer_weights layer[DS4_MAX_LAYER];
};

using token_vec = ds4_tokens;

uint32_t ds4_layer_compress_ratio(uint32_t layer);
void *ds4_xcalloc(size_t count, size_t size);
void *ds4_xmalloc(size_t size);
void *ds4_xrealloc(void *pointer, size_t size);
double ds4_now_seconds(void);
uint64_t ds4_align_up(uint64_t value, uint64_t alignment);
const char *ds4_tensor_type_name(uint32_t type);
uint64_t ds4_routed_expert_row_bytes(const ds4_tensor *tensor);

#endif  // GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_MODEL_DATA_INTERNAL_H_

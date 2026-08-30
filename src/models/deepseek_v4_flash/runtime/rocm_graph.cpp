#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "native_internal.h"
#include "model_data_internal.h"

#include "../kernels/rocm/resident_api.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float layer_rope_freq_base(uint32_t layer) {
    return ds4_layer_compress_ratio(layer) != 0 && DS4_COMPRESS_ROPE_FREQ_BASE > 0.0f
        ? DS4_COMPRESS_ROPE_FREQ_BASE
        : DS4_ROPE_FREQ_BASE;
}

static float layer_rope_freq_scale(uint32_t layer) {
    if (ds4_layer_compress_ratio(layer) == 0 || DS4_ROPE_SCALE_FACTOR <= 0.0f) {
        return 1.0f;
    }
    return 1.0f / DS4_ROPE_SCALE_FACTOR;
}

static uint32_t ds4_default_prefill_cap_for_prompt(int prompt_len) {
    if (prompt_len <= 0) return 1;
    uint32_t capacity = (uint32_t)prompt_len;

    const char *env = getenv("GUFO_DEEPSEEK_ROCM_PREFILL_CHUNK");
    if (env && env[0]) {
        char *end = NULL;
        const long value = strtol(env, &end, 10);
        if (end != env) {
            if (value <= 0) return capacity;
            capacity = (uint32_t)value;
        }
    } else if (prompt_len > 4096) {
        capacity = 4096u;
    }

    if (capacity == 0) capacity = 1;
    if (capacity > (uint32_t)prompt_len) capacity = (uint32_t)prompt_len;
    return capacity;
}

/* =========================================================================
 * ROCm Release Graph State.
 * =========================================================================
 *
 * The release ROCm executor owns one fixed set of tensors for single-token
 * decode and another for batched prefill.  The structure is DS4-specific:
 * tensor names follow the model stages rather than generic graph nodes.
 */

struct ds4_rocm_graph {
    /* One-token decode tensors.  These stay allocated for the life of a
     * session; a generated token enters as an embedding in cur_hc and leaves as
     * logits after all 43 layers update their raw/compressed/indexer caches. */
    ds4_gpu_tensor *cur_hc;
    ds4_gpu_tensor *flat_hc;
    ds4_gpu_tensor *hc_mix;
    ds4_gpu_tensor *hc_split;
    ds4_gpu_tensor *hc_pre;
    ds4_gpu_tensor *hc_post;
    ds4_gpu_tensor *hc_comb;
    ds4_gpu_tensor *attn_cur;
    ds4_gpu_tensor *attn_norm;
    ds4_gpu_tensor *qr;
    ds4_gpu_tensor *qr_norm;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *kv_raw;
    ds4_gpu_tensor *kv;

    /* Persistent KV state.  Raw KV is a sliding-window ring per layer.  Ratio-4
     * layers also keep an indexer-compressed cache; ratio-128 layers keep only
     * the attention-compressed cache.  The small state tensors are compressor
     * frontiers for the next compressed row, so they must be snapshotted with
     * the row counters whenever a checkpoint is saved or partially rewound. */
    ds4_gpu_tensor *layer_raw_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_comp_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_comp_cache_f16[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_comp_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_state_score[DS4_MAX_LAYER];

    uint32_t layer_n_comp[DS4_MAX_LAYER];
    uint32_t layer_n_index_comp[DS4_MAX_LAYER];
    uint32_t raw_cap;
    /* Maximum compressed-row capacity across layers.  Shared work buffers use
     * this worst-case size because ratio-4 indexer layers can still reach it. */
    uint32_t comp_cap;
    /* Persistent compressed caches are per layer, so size them from the actual
     * layer compression ratio instead of pessimistically using the ratio-4 cap
     * for every ratio-128 layer. */
    uint32_t layer_comp_cap[DS4_MAX_LAYER];
    /* Per-layer work tensors.  They are reused in place by every layer instead
     * of allocating a generic graph arena.  This is why the code is verbose but
     * predictable: each pointer names an actual DS4 stage. */
    ds4_gpu_tensor *comp_kv_cur;
    ds4_gpu_tensor *comp_sc_cur;
    ds4_gpu_tensor *indexer_q;
    ds4_gpu_tensor *indexer_weights;
    ds4_gpu_tensor *indexer_scores;
    ds4_gpu_tensor *comp_mask;
    ds4_gpu_tensor *comp_selected;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *attn_low;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *after_attn_hc;
    ds4_gpu_tensor *ffn_cur;
    ds4_gpu_tensor *ffn_norm;
    ds4_gpu_tensor *shared_gate;
    ds4_gpu_tensor *shared_up;
    ds4_gpu_tensor *shared_mid;
    ds4_gpu_tensor *shared_out;
    ds4_gpu_tensor *router_logits;
    ds4_gpu_tensor *router_probs;
    ds4_gpu_tensor *router_selected;
    ds4_gpu_tensor *router_weights;
    ds4_gpu_tensor *routed_gate;
    ds4_gpu_tensor *routed_up;
    ds4_gpu_tensor *routed_mid;
    ds4_gpu_tensor *routed_down;
    ds4_gpu_tensor *routed_out;
    ds4_gpu_tensor *after_ffn_hc;
    ds4_gpu_tensor *output_pre;
    ds4_gpu_tensor *output_weights;
    ds4_gpu_tensor *output_embd;
    ds4_gpu_tensor *output_norm;
    ds4_gpu_tensor *logits;

    uint32_t prefill_cap;
    uint32_t raw_window;

    /* Batched prefill tensors.  Prefill is layer-major: a chunk of prompt
     * tokens moves through layer 0, then layer 1, and so on, updating the same
     * persistent caches used by decode.  Keeping this separate from decode
     * avoids a slow loop of one-token graph steps for long prompts. */
    ds4_gpu_tensor *prefill_tokens;
    ds4_gpu_tensor *batch_cur_hc;
    ds4_gpu_tensor *batch_next_hc;
    ds4_gpu_tensor *batch_flat_hc;
    ds4_gpu_tensor *batch_hc_mix;
    ds4_gpu_tensor *batch_hc_split;
    ds4_gpu_tensor *batch_attn_cur;
    ds4_gpu_tensor *batch_attn_norm;
    ds4_gpu_tensor *batch_qr;
    ds4_gpu_tensor *batch_qr_norm;
    ds4_gpu_tensor *batch_q;
    ds4_gpu_tensor *batch_kv_raw;
    ds4_gpu_tensor *batch_kv;
    ds4_gpu_tensor *batch_comp_kv;
    ds4_gpu_tensor *batch_comp_sc;
    ds4_gpu_tensor *batch_indexer_q;
    ds4_gpu_tensor *batch_indexer_weights;
    ds4_gpu_tensor *batch_heads;
    ds4_gpu_tensor *batch_attn_low;
    ds4_gpu_tensor *batch_attn_out;
    ds4_gpu_tensor *batch_group_tmp;
    ds4_gpu_tensor *batch_low_tmp;
    ds4_gpu_tensor *batch_after_attn_hc;
    ds4_gpu_tensor *batch_ffn_cur;
    ds4_gpu_tensor *batch_ffn_norm;
    ds4_gpu_tensor *batch_shared_gate;
    ds4_gpu_tensor *batch_shared_up;
    ds4_gpu_tensor *batch_shared_mid;
    ds4_gpu_tensor *batch_shared_out;
    ds4_gpu_tensor *batch_router_logits;
    ds4_gpu_tensor *batch_router_probs;
    ds4_gpu_tensor *batch_router_selected;
    ds4_gpu_tensor *batch_router_weights;
    ds4_gpu_tensor *batch_routed_gate;
    ds4_gpu_tensor *batch_routed_up;
    ds4_gpu_tensor *batch_routed_mid;
    ds4_gpu_tensor *batch_routed_down;
    ds4_gpu_tensor *batch_routed_out;
    bool batch_routed_mid_is_f16;
    uint32_t power_percent;
    double prefill_layer_avg_sec[DS4_MAX_LAYER];
    double decode_token_avg_sec;
};

typedef struct ds4_rocm_graph ds4_gpu_graph;

static bool graph_power_throttle_enabled(const ds4_gpu_graph *g) {
    return g && g->power_percent > 0 && g->power_percent < 100;
}

static double graph_power_update_avg(double avg, double sample) {
    if (sample <= 0.0 || !isfinite(sample)) return avg;
    if (avg <= 0.0 || !isfinite(avg)) return sample;
    return avg * 0.875 + sample * 0.125;
}

static void graph_power_sleep(double work_sec, uint32_t power_percent) {
    if (power_percent == 0 || power_percent >= 100) return;
    /* Target duty cycle: work / (work + sleep) = power / 100.
     * At --power 50 this sleeps for one measured work interval; at 25 it
     * sleeps for three. */
    const double sleep = work_sec * (100.0 - (double)power_percent) /
                         (double)power_percent;
    ds4_sleep_seconds(sleep);
}

static void graph_power_note_prefill_layer(ds4_gpu_graph *g,
                                           uint32_t il,
                                           double elapsed_sec) {
    if (!graph_power_throttle_enabled(g)) return;
    if (il >= DS4_N_LAYER) return;
    g->prefill_layer_avg_sec[il] =
        graph_power_update_avg(g->prefill_layer_avg_sec[il], elapsed_sec);
    graph_power_sleep(g->prefill_layer_avg_sec[il], g->power_percent);
}

static void graph_power_note_decode_token(ds4_gpu_graph *g, double elapsed_sec) {
    if (!graph_power_throttle_enabled(g)) return;
    g->decode_token_avg_sec =
        graph_power_update_avg(g->decode_token_avg_sec, elapsed_sec);
    graph_power_sleep(g->decode_token_avg_sec, g->power_percent);
}

/* Release every ROCm tensor owned by the whole-model graph runtime. */
static void rocm_graph_free(ds4_gpu_graph *g) {
    ds4_gpu_tensor_free(g->batch_routed_out);
    ds4_gpu_tensor_free(g->batch_routed_down);
    ds4_gpu_tensor_free(g->batch_routed_mid);
    ds4_gpu_tensor_free(g->batch_routed_up);
    ds4_gpu_tensor_free(g->batch_routed_gate);
    ds4_gpu_tensor_free(g->batch_router_weights);
    ds4_gpu_tensor_free(g->batch_router_selected);
    ds4_gpu_tensor_free(g->batch_router_probs);
    ds4_gpu_tensor_free(g->batch_router_logits);
    ds4_gpu_tensor_free(g->batch_shared_out);
    ds4_gpu_tensor_free(g->batch_shared_mid);
    ds4_gpu_tensor_free(g->batch_shared_up);
    ds4_gpu_tensor_free(g->batch_shared_gate);
    ds4_gpu_tensor_free(g->batch_ffn_norm);
    ds4_gpu_tensor_free(g->batch_ffn_cur);
    ds4_gpu_tensor_free(g->batch_after_attn_hc);
    ds4_gpu_tensor_free(g->batch_low_tmp);
    ds4_gpu_tensor_free(g->batch_group_tmp);
    ds4_gpu_tensor_free(g->batch_attn_out);
    ds4_gpu_tensor_free(g->batch_attn_low);
    ds4_gpu_tensor_free(g->batch_heads);
    ds4_gpu_tensor_free(g->batch_indexer_weights);
    ds4_gpu_tensor_free(g->batch_indexer_q);
    ds4_gpu_tensor_free(g->batch_comp_sc);
    ds4_gpu_tensor_free(g->batch_comp_kv);
    ds4_gpu_tensor_free(g->batch_kv);
    ds4_gpu_tensor_free(g->batch_kv_raw);
    ds4_gpu_tensor_free(g->batch_q);
    ds4_gpu_tensor_free(g->batch_qr_norm);
    ds4_gpu_tensor_free(g->batch_qr);
    ds4_gpu_tensor_free(g->batch_attn_norm);
    ds4_gpu_tensor_free(g->batch_attn_cur);
    ds4_gpu_tensor_free(g->batch_hc_split);
    ds4_gpu_tensor_free(g->batch_hc_mix);
    ds4_gpu_tensor_free(g->batch_flat_hc);
    ds4_gpu_tensor_free(g->batch_next_hc);
    ds4_gpu_tensor_free(g->batch_cur_hc);
    ds4_gpu_tensor_free(g->prefill_tokens);
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->output_norm);
    ds4_gpu_tensor_free(g->output_embd);
    ds4_gpu_tensor_free(g->output_weights);
    ds4_gpu_tensor_free(g->output_pre);
    ds4_gpu_tensor_free(g->after_ffn_hc);
    ds4_gpu_tensor_free(g->routed_out);
    ds4_gpu_tensor_free(g->routed_down);
    ds4_gpu_tensor_free(g->routed_mid);
    ds4_gpu_tensor_free(g->routed_up);
    ds4_gpu_tensor_free(g->routed_gate);
    ds4_gpu_tensor_free(g->router_weights);
    ds4_gpu_tensor_free(g->router_selected);
    ds4_gpu_tensor_free(g->router_probs);
    ds4_gpu_tensor_free(g->router_logits);
    ds4_gpu_tensor_free(g->shared_out);
    ds4_gpu_tensor_free(g->shared_mid);
    ds4_gpu_tensor_free(g->shared_up);
    ds4_gpu_tensor_free(g->shared_gate);
    ds4_gpu_tensor_free(g->ffn_norm);
    ds4_gpu_tensor_free(g->ffn_cur);
    ds4_gpu_tensor_free(g->after_attn_hc);
    ds4_gpu_tensor_free(g->attn_out);
    ds4_gpu_tensor_free(g->attn_low);
    ds4_gpu_tensor_free(g->heads);
    ds4_gpu_tensor_free(g->comp_sc_cur);
    ds4_gpu_tensor_free(g->comp_kv_cur);
    ds4_gpu_tensor_free(g->comp_mask);
    ds4_gpu_tensor_free(g->comp_selected);
    ds4_gpu_tensor_free(g->indexer_scores);
    ds4_gpu_tensor_free(g->indexer_weights);
    ds4_gpu_tensor_free(g->indexer_q);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_raw_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_comp_cache[il]);
        ds4_gpu_tensor_free(g->layer_attn_comp_cache_f16[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_score[il]);
    }
    ds4_gpu_tensor_free(g->kv);
    ds4_gpu_tensor_free(g->kv_raw);
    ds4_gpu_tensor_free(g->q);
    ds4_gpu_tensor_free(g->qr_norm);
    ds4_gpu_tensor_free(g->qr);
    ds4_gpu_tensor_free(g->attn_norm);
    ds4_gpu_tensor_free(g->attn_cur);
    ds4_gpu_tensor_free(g->hc_comb);
    ds4_gpu_tensor_free(g->hc_post);
    ds4_gpu_tensor_free(g->hc_pre);
    ds4_gpu_tensor_free(g->hc_split);
    ds4_gpu_tensor_free(g->hc_mix);
    ds4_gpu_tensor_free(g->flat_hc);
    ds4_gpu_tensor_free(g->cur_hc);
    memset(g, 0, sizeof(*g));
}

static bool rocm_tensor_fill_f32(ds4_gpu_tensor *t, float v, uint64_t n) {
    return ds4_gpu_tensor_fill_f32(t, v, n) != 0;
}

static uint64_t rocm_graph_kv_cache_bytes_for_context(uint32_t ctx_size, uint32_t raw_cap) {
    uint64_t bytes = (uint64_t)DS4_N_LAYER *
                     raw_cap *
                     DS4_N_HEAD_DIM *
                     sizeof(float);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t comp_cap = (uint64_t)(ctx_size / ratio + 2u);
        bytes += comp_cap * DS4_N_HEAD_DIM * sizeof(float);
        if (ratio == 4) {
            bytes += comp_cap * DS4_N_HEAD_DIM * sizeof(uint16_t);
            bytes += comp_cap * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
        }
    }
    return bytes;
}

static uint64_t rocm_graph_context_bytes_for_kv_policy(
        uint32_t  ctx_size,
        uint32_t  raw_cap,
        uint32_t  prefill_cap,
        uint64_t *kv_cache_bytes_out) {
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    uint64_t comp_cap = (uint64_t)(ctx_size / min_ratio + 2u);
    if (comp_cap < 2u) comp_cap = 2u;
    const uint64_t kv_cache_bytes = rocm_graph_kv_cache_bytes_for_context(ctx_size, raw_cap);
    if (kv_cache_bytes_out) *kv_cache_bytes_out = kv_cache_bytes;
    return kv_cache_bytes + 2ull * comp_cap * prefill_cap * sizeof(float);
}

static ds4_gpu_tensor *rocm_graph_alloc_kv_cache_tensor(bool managed, uint64_t bytes) {
    return managed ? ds4_gpu_tensor_alloc_managed(bytes) : ds4_gpu_tensor_alloc(bytes);
}

/* =========================================================================
 * ROCm Release Graph Allocation.
 * ========================================================================= */

/* Allocate the ROCm graph state for a chosen raw-cache capacity.  The model
 * weights are not copied here; tensors reference the mapped GGUF. */
static bool rocm_graph_alloc_raw_cap(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap) {
    memset(g, 0, sizeof(*g));
    if (raw_cap == 0) raw_cap = 1;
    if (ctx_size == 0) ctx_size = raw_cap;
    if (prefill_cap == 0) prefill_cap = 1;
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > ctx_size) raw_window = ctx_size;
    if (raw_window == 0) raw_window = 1;
    if (raw_cap < raw_window) raw_cap = raw_window;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    g->raw_cap = raw_cap;
    g->raw_window = raw_window;
    g->prefill_cap = prefill_cap;
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    g->comp_cap = ctx_size / min_ratio + 2u;
    if (g->comp_cap < 2u) g->comp_cap = 2u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) {
            g->layer_comp_cap[il] = 0;
        } else {
            g->layer_comp_cap[il] = ctx_size / ratio + 2u;
            if (g->layer_comp_cap[il] < 2u) g->layer_comp_cap[il] = 2u;
        }
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t group_dim = (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t routed_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t vocab_dim = weights->output->dim[1];
    const uint64_t comp_width_max = 2ull * (DS4_N_HEAD_DIM > DS4_N_INDEXER_HEAD_DIM
        ? DS4_N_HEAD_DIM
        : DS4_N_INDEXER_HEAD_DIM);
    const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    const uint64_t pc = prefill_cap;
    uint64_t kv_cache_bytes = 0;
    const uint64_t context_bytes =
        rocm_graph_context_bytes_for_kv_policy(ctx_size, raw_cap, prefill_cap, &kv_cache_bytes);
    const bool managed_kv_cache =
        ds4_gpu_should_use_managed_kv_cache(kv_cache_bytes, context_bytes) != 0;
    if (managed_kv_cache) {
        /*
         * Device allocations are fastest, but very large contexts can exhaust
         * Strix Halo unified memory once resident weights and driver allocations
         * are present. Managed memory preserves a demand-paged fallback for this
         * long-lived cache class.
         */
        fprintf(stderr,
                "ds4: ROCm using managed KV cache for ctx=%u "
                "(kv cache %.2f GiB, context buffers %.2f GiB); "
                "this may degrade performance but is needed for very large contexts\n",
                ctx_size,
                (double)kv_cache_bytes / 1073741824.0,
                (double)context_bytes / 1073741824.0);
    }

    g->cur_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->flat_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->hc_mix = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_split = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_pre = ds4_gpu_tensor_view(g->hc_split, 0, (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_post = ds4_gpu_tensor_view(g->hc_split,
                                       (uint64_t)DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_comb = ds4_gpu_tensor_view(g->hc_split,
                                       2ull * DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    g->attn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->attn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->qr = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->qr_norm = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->q = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->kv_raw = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    g->kv = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    bool state_init_ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_raw_cache[il] = rocm_graph_alloc_kv_cache_tensor(
                managed_kv_cache,
                (uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            const uint64_t attn_rows = (uint64_t)coff * ratio;
            g->layer_attn_comp_cache[il] = rocm_graph_alloc_kv_cache_tensor(
                    managed_kv_cache,
                    (uint64_t)g->layer_comp_cap[il] * DS4_N_HEAD_DIM * sizeof(float));
            if (ratio == 4) {
                g->layer_attn_comp_cache_f16[il] =
                    rocm_graph_alloc_kv_cache_tensor(
                        managed_kv_cache,
                        (uint64_t)g->layer_comp_cap[il] *
                            DS4_N_HEAD_DIM * sizeof(uint16_t));
            }
            g->layer_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            g->layer_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            if (g->layer_attn_state_kv[il]) {
                state_init_ok = state_init_ok &&
                                rocm_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows);
            }
            if (g->layer_attn_state_score[il]) {
                state_init_ok = state_init_ok &&
                                rocm_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows);
            }

            if (ratio == 4) {
                const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
                const uint64_t index_rows = (uint64_t)coff * ratio;
                g->layer_index_comp_cache[il] = rocm_graph_alloc_kv_cache_tensor(
                        managed_kv_cache,
                        (uint64_t)g->layer_comp_cap[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                g->layer_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                g->layer_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                if (g->layer_index_state_kv[il]) {
                    state_init_ok = state_init_ok &&
                                    rocm_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows);
                }
                if (g->layer_index_state_score[il]) {
                    state_init_ok = state_init_ok &&
                                    rocm_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows);
                }
            }
        }
    }
    g->comp_kv_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->comp_sc_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->indexer_q = ds4_gpu_tensor_alloc(indexer_q_dim * sizeof(float));
    g->indexer_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_INDEXER_HEAD * sizeof(float));
    g->indexer_scores = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * pc * sizeof(float));
    g->comp_mask = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * pc * sizeof(float));
    g->comp_selected = ds4_gpu_tensor_alloc((uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) *
                                              pc * sizeof(uint32_t));
    g->heads = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->attn_low = ds4_gpu_tensor_alloc(low_dim * sizeof(float));
    g->attn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_attn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->ffn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->ffn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->shared_gate = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_up = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_mid = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->router_logits = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_probs = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_selected = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(int));
    g->router_weights = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(float));
    g->routed_gate = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_up = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_mid = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_down = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->routed_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_ffn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->output_pre = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_embd = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->output_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->logits = ds4_gpu_tensor_alloc(vocab_dim * sizeof(float));
    g->prefill_tokens = ds4_gpu_tensor_alloc(pc * sizeof(int32_t));
    g->batch_cur_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_next_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_flat_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_hc_mix = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_hc_split = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_attn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_attn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_qr = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_qr_norm = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_q = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_kv_raw = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_kv = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_comp_kv = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_comp_sc = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_indexer_q = ds4_gpu_tensor_alloc(pc * indexer_q_dim * sizeof(float));
    g->batch_indexer_weights = ds4_gpu_tensor_alloc(pc * DS4_N_INDEXER_HEAD * sizeof(float));
    g->batch_heads = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_attn_low = ds4_gpu_tensor_alloc(pc * low_dim * sizeof(float));
    g->batch_attn_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_group_tmp = ds4_gpu_tensor_alloc(pc * group_dim * sizeof(float));
    g->batch_low_tmp = ds4_gpu_tensor_alloc(pc * DS4_N_LORA_O * sizeof(float));
    g->batch_after_attn_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_ffn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_ffn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_shared_gate = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_up = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_mid = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_router_logits = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_probs = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_selected = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(int));
    g->batch_router_weights = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(float));
    g->batch_routed_gate = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_up = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_mid = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_down = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->batch_routed_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));

    bool layer_cache_ok = true;
    for (uint32_t il = 0; layer_cache_ok && il < DS4_N_LAYER; il++) {
        layer_cache_ok = g->layer_raw_cache[il] != NULL;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (layer_cache_ok && ratio != 0) {
            layer_cache_ok = g->layer_attn_comp_cache[il] != NULL &&
                             g->layer_attn_state_kv[il] != NULL &&
                             g->layer_attn_state_score[il] != NULL;
        }
        if (layer_cache_ok && ratio == 4) {
            layer_cache_ok = g->layer_attn_comp_cache_f16[il] != NULL &&
                             g->layer_index_comp_cache[il] != NULL &&
                             g->layer_index_state_kv[il] != NULL &&
                             g->layer_index_state_score[il] != NULL;
        }
    }

    const bool ok = state_init_ok && layer_cache_ok &&
                    g->cur_hc && g->flat_hc && g->hc_mix && g->hc_split &&
                    g->hc_pre && g->hc_post && g->hc_comb &&
                    g->attn_cur && g->attn_norm && g->qr && g->qr_norm &&
                    g->q && g->kv_raw && g->kv &&
                    g->comp_kv_cur && g->comp_sc_cur &&
                    g->indexer_q && g->indexer_weights && g->indexer_scores &&
                    g->comp_mask && g->comp_selected &&
                    g->heads && g->attn_low && g->attn_out &&
                    g->after_attn_hc && g->ffn_cur && g->ffn_norm &&
                    g->shared_gate && g->shared_up && g->shared_mid &&
                    g->shared_out &&
                    g->router_logits && g->router_probs && g->router_selected && g->router_weights &&
                    g->routed_gate && g->routed_up && g->routed_mid &&
                    g->routed_down && g->routed_out &&
                    g->after_ffn_hc &&
                    g->output_pre && g->output_weights && g->output_embd &&
                    g->output_norm && g->logits &&
                    g->prefill_tokens &&
                    g->batch_cur_hc && g->batch_next_hc && g->batch_flat_hc &&
                    g->batch_hc_mix && g->batch_hc_split &&
                    g->batch_attn_cur && g->batch_attn_norm &&
                    g->batch_qr && g->batch_qr_norm && g->batch_q &&
                    g->batch_kv_raw && g->batch_kv &&
                    g->batch_comp_kv && g->batch_comp_sc &&
                    g->batch_indexer_q && g->batch_indexer_weights &&
                    g->batch_heads && g->batch_attn_low && g->batch_attn_out &&
                    g->batch_group_tmp && g->batch_low_tmp && g->batch_after_attn_hc &&
                    g->batch_ffn_cur && g->batch_ffn_norm &&
                    g->batch_shared_gate && g->batch_shared_up &&
                    g->batch_shared_mid && g->batch_shared_out &&
                    g->batch_router_logits && g->batch_router_probs &&
                    g->batch_router_selected && g->batch_router_weights &&
                    g->batch_routed_gate && g->batch_routed_up &&
                    g->batch_routed_mid && g->batch_routed_down &&
                    g->batch_routed_out;
    if (!ok) rocm_graph_free(g);
    return ok;
}

static uint32_t rocm_graph_raw_span_for_batch(
        const ds4_gpu_graph *g,
        uint32_t               pos0,
        uint32_t               n_tokens) {
    if (!g || g->raw_cap == 0 || n_tokens == 0) return 0;

    const uint32_t window = g->raw_window ? g->raw_window : DS4_N_SWA;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    uint64_t needed = (uint64_t)n_tokens;
    if (window != 0) {
        needed += n_tokens == 1 ? (uint64_t)window - 1u : (uint64_t)window;
    }
    uint64_t available = (uint64_t)last_pos + 1u;
    if (needed > available) needed = available;
    if (needed > g->raw_cap) needed = g->raw_cap;
    return (uint32_t)needed;
}

static uint32_t rocm_graph_raw_start_for_span(
        const ds4_gpu_graph *g,
        uint32_t               last_pos,
        uint32_t               n_raw) {
    if (!g || g->raw_cap == 0 || n_raw == 0) return 0;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    return first_raw_pos % g->raw_cap;
}

static uint32_t rocm_graph_decode_indexer_sparse_threshold(const ds4_gpu_graph *g) {
    (void)g;
    static int parsed = -1;
    static uint32_t cached = 0;
    if (parsed < 0) {
        parsed = 0;
        const char *env = getenv("GUFO_DEEPSEEK_ROCM_DECODE_INDEXER_SPARSE_THRESHOLD");
        if (env && env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(env, &end, 10);
            while (end && isspace((unsigned char)*end)) end++;
            if (end != env && end && *end == '\0' &&
                (v == 64ul || v == 128ul || v == 256ul || v == 512ul ||
                 v == 1024ul || v == 2048ul || v == 4096ul)) {
                cached = (uint32_t)v;
                parsed = 1;
            } else {
                fprintf(stderr,
                        "ds4: invalid GUFO_DEEPSEEK_ROCM_DECODE_INDEXER_SPARSE_THRESHOLD=%s; "
                        "expected 64, 128, 256, 512, 1024, 2048, or 4096\n",
                        env);
            }
        }
    }
    if (parsed > 0) return cached;

    /* Keep dense attention longer than the legacy 512-row window by default.
     * Around the 2K frontier the sparse path's score/top-k setup dominates
     * the smaller attention scan, while larger contexts benefit from sparse
     * indexed attention.  This threshold changes only the implementation used
     * to consume the compressed rows; it must not lower the 512-row indexer
     * selection defined by DS4_N_INDEXER_TOP_K. */
    return 1024u;
}

/* =========================================================================
 * ROCm Decode Release Helpers and Reference Fallbacks.
 * =========================================================================
 *
 * The normal generation path uses the fused helpers below.  The older unfused
 * kernels remain available as diagnostic reference paths selected only by the
 * GUFO_DEEPSEEK_ROCM_DISABLE_*_FUSION environment switches.
 */

static bool rocm_graph_env_flag(const char *name, int *cache) {
    if (*cache == -1) {
        const char *env = getenv(name);
        *cache = env && env[0] && strcmp(env, "0") != 0;
    }
    return *cache != 0;
}

static bool rocm_graph_use_reference_hc_decode(void) {
    static int cache = -1;
    return rocm_graph_env_flag("GUFO_DEEPSEEK_ROCM_DISABLE_HC_FUSION", &cache);
}

static bool rocm_graph_use_reference_kv_decode(void) {
    static int cache = -1;
    return rocm_graph_env_flag("GUFO_DEEPSEEK_ROCM_DISABLE_KV_FUSION", &cache);
}

static bool rocm_graph_use_reference_qkv_norm(void) {
    static int cache = -1;
    return rocm_graph_env_flag("GUFO_DEEPSEEK_ROCM_DISABLE_QKV_NORM_FUSION", &cache);
}

static bool rocm_graph_use_reference_compressor_pair_proj(void) {
    static int cache = -1;
    return rocm_graph_env_flag("GUFO_DEEPSEEK_ROCM_DISABLE_COMPRESSOR_PAIR_PROJ", &cache);
}

static bool rocm_graph_use_reference_hc_norm_decode(void) {
    static int cache = -1;
    return rocm_graph_env_flag("GUFO_DEEPSEEK_ROCM_DISABLE_HC_NORM_FUSION", &cache);
}

static bool rocm_graph_use_reference_shared_down_hc(void) {
    static int cache = -1;
    return rocm_graph_env_flag("GUFO_DEEPSEEK_ROCM_DISABLE_SHARED_DOWN_HC_FUSION", &cache);
}

static bool rocm_graph_use_reference_attn_out_hc(void) {
    static int cache = -1;
    return rocm_graph_env_flag("GUFO_DEEPSEEK_ROCM_DISABLE_ATTN_OUT_HC_FUSION", &cache);
}

static bool rocm_graph_decode_hc_pre(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const ds4_model        *model,
        uint64_t                scale_offset,
        uint64_t                base_offset) {
    if (rocm_graph_use_reference_hc_decode()) {
        return ds4_gpu_hc_split_sinkhorn_tensor(split,
                                                  mix,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0 &&
               ds4_gpu_hc_weighted_sum_tensor(out,
                                                 residual_hc,
                                                 split,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    }

    return ds4_gpu_hc_split_weighted_sum_tensor(out,
                                                  split,
                                                  mix,
                                                  residual_hc,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0;
}

static bool rocm_graph_decode_kv_store(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row) {
    if (rocm_graph_use_reference_kv_decode()) {
        return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0 &&
               ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, DS4_N_HEAD_DIM) != 0;
    }

    return ds4_gpu_kv_fp8_store_raw_tensor(kv,
                                             raw_cache,
                                             raw_cap,
                                             raw_row,
                                             DS4_N_HEAD_DIM,
                                             DS4_N_ROT) != 0;
}

static uint32_t rocm_graph_attn_comp_cache_is_f16(void) {
    return 0;
}

static bool rocm_graph_use_attn_comp_mirror(
        const ds4_gpu_graph *g,
        uint32_t             il,
        uint32_t             n_tokens) {
    return n_tokens >= 128u && g->layer_attn_comp_cache_f16[il] != nullptr;
}

static ds4_gpu_tensor *rocm_graph_prefill_attn_comp_cache(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       n_tokens) {
    return rocm_graph_use_attn_comp_mirror(g, il, n_tokens)
        ? g->layer_attn_comp_cache_f16[il]
        : g->layer_attn_comp_cache[il];
}

static uint32_t rocm_graph_prefill_attn_comp_cache_is_f16(
        const ds4_gpu_graph *g,
        uint32_t             il,
        uint32_t             n_tokens) {
    return rocm_graph_use_attn_comp_mirror(g, il, n_tokens) ? 1u : 0u;
}

static ds4_gpu_tensor *rocm_graph_attn_comp_update_target(
        ds4_gpu_graph *g,
        uint32_t       il) {
    return g->layer_attn_comp_cache[il];
}

static uint32_t rocm_graph_attn_comp_update_row(uint32_t row) {
    return row;
}

static ds4_gpu_tensor *rocm_graph_attn_comp_row_view(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row) {
    return ds4_gpu_tensor_view(g->layer_attn_comp_cache[il],
                               (uint64_t)row * DS4_N_HEAD_DIM * sizeof(float),
                               (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
}

static ds4_gpu_tensor *rocm_graph_attn_comp_mirror_row_view(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row) {
    if (!g->layer_attn_comp_cache_f16[il]) return nullptr;
    return ds4_gpu_tensor_view(
        g->layer_attn_comp_cache_f16[il],
        (uint64_t)row * DS4_N_HEAD_DIM * sizeof(uint16_t),
        (uint64_t)DS4_N_HEAD_DIM * sizeof(uint16_t));
}

static ds4_gpu_tensor *rocm_graph_attn_comp_prefill_target(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    const uint32_t view_rows = rows ? rows : 1u;
    return ds4_gpu_tensor_view(g->layer_attn_comp_cache[il],
                               (uint64_t)first_row * DS4_N_HEAD_DIM * sizeof(float),
                               (uint64_t)view_rows * DS4_N_HEAD_DIM * sizeof(float));
}

static ds4_gpu_tensor *rocm_graph_attn_comp_prefill_mirror(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (!g->layer_attn_comp_cache_f16[il]) return nullptr;
    const uint32_t view_rows = rows ? rows : 1u;
    return ds4_gpu_tensor_view(
        g->layer_attn_comp_cache_f16[il],
        (uint64_t)first_row * DS4_N_HEAD_DIM * sizeof(uint16_t),
        (uint64_t)view_rows * DS4_N_HEAD_DIM * sizeof(uint16_t));
}

static void rocm_graph_attn_comp_prefill_target_free(ds4_gpu_tensor *t) {
    ds4_gpu_tensor_free(t);
}

static bool rocm_graph_quantize_attn_comp_row(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row,
        ds4_gpu_tensor *comp_row_f32) {
    ds4_gpu_tensor *mirror_row =
        rocm_graph_attn_comp_mirror_row_view(g, il, row);
    const bool ok = mirror_row
        ? ds4_gpu_dsv4_fp8_kv_quantize_mirror_f16_tensor(
              comp_row_f32, mirror_row, 1u, DS4_N_HEAD_DIM, DS4_N_ROT) != 0
        : ds4_gpu_dsv4_fp8_kv_quantize_tensor(
              comp_row_f32, 1u, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
    ds4_gpu_tensor_free(mirror_row);
    return ok;
}

static bool rocm_graph_rebuild_attn_comp_mirror(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       rows) {
    if (!g->layer_attn_comp_cache_f16[il] || rows == 0u) return true;
    return ds4_gpu_tensor_convert_f32_to_f16(
               g->layer_attn_comp_cache_f16[il],
               g->layer_attn_comp_cache[il],
               (uint64_t)rows * DS4_N_HEAD_DIM) != 0;
}

/* Encode one DS4 decode layer on ROCm.  This is the release single-token
 * layer path; diagnostics reuse it so they compare exactly what generation
 * runs. */
static bool rocm_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0);
static bool rocm_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0);
static bool rocm_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

static bool rocm_graph_encode_decode_layer(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        ds4_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t shared_dim = (uint32_t)layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const bool compressed = ds4_layer_compress_ratio(il) != 0;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const bool qkv_rms_fused = !rocm_graph_use_reference_qkv_norm();

    bool ok = true;
    const bool decode_stage_profile = getenv("GUFO_DEEPSEEK_ROCM_DECODE_STAGE_PROFILE") != NULL;
    double decode_stage_t0 = decode_stage_profile ? ds4_now_seconds() : 0.0;
#define GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE(name) do { \
        if (ok && decode_stage_profile) { \
            ok = rocm_graph_layer_stage_profile_boundary("decode", (name), il, pos, 1, &decode_stage_t0); \
        } \
    } while (0)
    if (ok) ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = rocm_graph_matmul_plain_tensor(g->hc_mix, model, layer->hc_attn_fn,
                                                 hc_dim, mix_hc, g->flat_hc, 1);
    const bool fuse_hc_norm =
        !rocm_graph_use_reference_hc_decode() &&
        !rocm_graph_use_reference_hc_norm_decode();
    if (ok && fuse_hc_norm) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(g->attn_cur,
                                                         g->attn_norm,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->cur_hc,
                                                         model->map,
                                                         model->size,
                                                         layer->hc_attn_scale->abs_offset,
                                                         layer->hc_attn_base->abs_offset,
                                                         layer->attn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_HC_SINKHORN_ITER,
                                                         DS4_HC_EPS,
                                                         DS4_RMS_EPS) != 0;
    } else if (ok) {
        ok = rocm_graph_decode_hc_pre(g->attn_cur,
                                       g->hc_split,
                                       g->hc_mix,
                                       g->cur_hc,
                                       model,
                                       layer->hc_attn_scale->abs_offset,
                                       layer->hc_attn_base->abs_offset);
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("attn_hc_pre");
    if (ok) {
    }
    if (ok) {
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(g->attn_norm, g->attn_cur,
                                                                   model->map, model->size,
                                                                   layer->attn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("attn_norm");
    if (ok) {
    }
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->qr, model->map, model->size,
                                              layer->attn_q_a->abs_offset,
                                              DS4_N_EMBD, q_rank,
                                              g->attn_norm, 1) != 0;
    if (ok) {
    }
    if (qkv_rms_fused) {
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->kv_raw, model->map, model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD, DS4_N_HEAD_DIM,
                                                  g->attn_norm, 1) != 0;
        if (ok) {
        }
        if (ok) ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->qr_norm,
                                                             g->qr,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             g->kv,
                                                             g->kv_raw,
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             DS4_N_HEAD_DIM,
                                                             1,
                                                             DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->qr_norm, g->qr,
                                                      model->map, model->size,
                                                      layer->attn_q_a_norm->abs_offset,
                                                      (uint32_t)q_rank, DS4_RMS_EPS) != 0;
    }
    if (ok) {
    }
    if (qkv_rms_fused && ok) {
    }
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->q, model->map, model->size,
                                              layer->attn_q_b->abs_offset,
                                              q_rank, q_dim,
                                              g->qr_norm, 1) != 0;
    if (ok) {
    }
    if (ok) ok = ds4_gpu_head_rms_norm_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
    if (ok) {
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false, freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("q_path");
    if (ok) {
    }
    if (!qkv_rms_fused) {
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->kv_raw, model->map, model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD, DS4_N_HEAD_DIM,
                                                  g->attn_norm, 1) != 0;
        if (ok) {
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->kv, g->kv_raw,
                                                      model->map, model->size,
                                                      layer->attn_kv_a_norm->abs_offset,
                                                      DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
        if (ok) {
        }
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->kv, 1, DS4_N_HEAD_KV, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false, freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
    }
    /* RoPE stays as the exact standalone kernel above.  The decode fusion
     * starts after that, where FP8 KV quantization and raw-cache storage can
     * share one pass without changing the trigonometric path. */
    if (ok) ok = rocm_graph_decode_kv_store(g->kv, raw_cache, raw_cap, raw_row);
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("kv_path");
    if (ok) {
    }

    uint32_t n_comp = 0;
    ds4_gpu_tensor *comp_cache = NULL;
    ds4_gpu_tensor *comp_selected = NULL;
    uint32_t n_selected = 0;
    double decode_index_stage_t0 = 0.0;
    const bool decode_index_stage_profile = getenv("GUFO_DEEPSEEK_ROCM_INDEXER_STAGE_PROFILE") != NULL;
    if (ok && compressed) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool emit = ((pos + 1u) % ratio) == 0u;
        if (!layer->attn_compressor_kv || !layer->attn_compressor_gate ||
            !layer->attn_compressor_ape || !layer->attn_compressor_norm ||
            layer->attn_compressor_kv->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_gate->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_kv->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_gate->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_kv->dim[1] != comp_width ||
            layer->attn_compressor_gate->dim[1] != comp_width) {
            fprintf(stderr, "ds4: ROCm graph compressor expects paired F16 compressor projections\n");
            ok = false;
        }
        if (ok && emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
            fprintf(stderr, "ds4: ROCm graph compressed KV cache capacity exceeded at layer %u\n", il);
            ok = false;
        }
        if (ok && !rocm_graph_use_reference_compressor_pair_proj()) {
            ok = ds4_gpu_matmul_f16_pair_tensor(g->comp_kv_cur,
                                                  g->comp_sc_cur,
                                                  model->map,
                                                  model->size,
                                                  layer->attn_compressor_kv->abs_offset,
                                                  layer->attn_compressor_gate->abs_offset,
                                                  DS4_N_EMBD,
                                                  comp_width,
                                                  g->attn_norm,
                                                  1) != 0;
        } else {
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_kv_cur, model->map, model->size,
                                                     layer->attn_compressor_kv->abs_offset,
                                                     DS4_N_EMBD, comp_width,
                                                     g->attn_norm, 1) != 0;
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_sc_cur, model->map, model->size,
                                                     layer->attn_compressor_gate->abs_offset,
                                                     DS4_N_EMBD, comp_width,
                                                     g->attn_norm, 1) != 0;
        }
        const uint32_t comp_row = g->layer_n_comp[il];
        if (ok) ok = ds4_gpu_compressor_update_tensor(g->comp_kv_cur,
                                                        g->comp_sc_cur,
                                                        g->layer_attn_state_kv[il],
                                                        g->layer_attn_state_score[il],
                                                        rocm_graph_attn_comp_update_target(g, il),
                                                        model->map,
                                                        model->size,
                                                        layer->attn_compressor_ape->abs_offset,
                                                        layer->attn_compressor_ape->type,
                                                        layer->attn_compressor_norm->abs_offset,
                                                        layer->attn_compressor_norm->type,
                                                        DS4_N_HEAD_DIM,
                                                        ratio,
                                                        pos,
                                                        rocm_graph_attn_comp_update_row(comp_row),
                                                        DS4_N_ROT,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW,
                                                        DS4_RMS_EPS,
                                                        false,
                                                        true,
                                                        false) != 0;
        if (ok && emit) {
            ds4_gpu_tensor *comp_row_view = rocm_graph_attn_comp_row_view(g, il, comp_row);
            if (!comp_row_view) {
                ok = false;
            } else {
                ok = rocm_graph_quantize_attn_comp_row(
                    g, il, comp_row, comp_row_view);
                if (ok) {
                }
                ds4_gpu_tensor_free(comp_row_view);
            }
        }
        if (ok && emit) g->layer_n_comp[il]++;

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                layer->indexer_compressor_kv->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_gate->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_kv->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_gate->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_kv->dim[1] != index_width ||
                layer->indexer_compressor_gate->dim[1] != index_width) {
                fprintf(stderr, "ds4: ROCm graph indexer compressor expects paired F16 projections\n");
                ok = false;
            }
            if (ok && emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: ROCm graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok && !rocm_graph_use_reference_compressor_pair_proj()) {
                ok = ds4_gpu_matmul_f16_pair_tensor(g->comp_kv_cur,
                                                      g->comp_sc_cur,
                                                      model->map,
                                                      model->size,
                                                      layer->indexer_compressor_kv->abs_offset,
                                                      layer->indexer_compressor_gate->abs_offset,
                                                      DS4_N_EMBD,
                                                      index_width,
                                                      g->attn_norm,
                                                      1) != 0;
            } else {
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_kv_cur, model->map, model->size,
                                                         layer->indexer_compressor_kv->abs_offset,
                                                         DS4_N_EMBD, index_width,
                                                         g->attn_norm, 1) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_sc_cur, model->map, model->size,
                                                         layer->indexer_compressor_gate->abs_offset,
                                                         DS4_N_EMBD, index_width,
                                                         g->attn_norm, 1) != 0;
            }
            const uint32_t index_row = g->layer_n_index_comp[il];
            if (ok) ok = ds4_gpu_compressor_update_tensor(g->comp_kv_cur,
                                                            g->comp_sc_cur,
                                                            g->layer_index_state_kv[il],
                                                            g->layer_index_state_score[il],
                                                            g->layer_index_comp_cache[il],
                                                            model->map,
                                                            model->size,
                                                            layer->indexer_compressor_ape->abs_offset,
                                                            layer->indexer_compressor_ape->type,
                                                            layer->indexer_compressor_norm->abs_offset,
                                                            layer->indexer_compressor_norm->type,
                                                            DS4_N_INDEXER_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            index_row,
                                                            DS4_N_ROT,
                                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            DS4_ROPE_YARN_BETA_FAST,
                                                            DS4_ROPE_YARN_BETA_SLOW,
                                                            DS4_RMS_EPS,
                                                            false,
                                                            true,
                                                            false) != 0;
            if (ok && emit) {
                ds4_gpu_tensor *index_row_view = ds4_gpu_tensor_view(
                        g->layer_index_comp_cache[il],
                        (uint64_t)index_row * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                        (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                if (!index_row_view) {
                    ok = false;
                } else {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(index_row_view,
                                                          1,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
                    ds4_gpu_tensor_free(index_row_view);
                }
            }
            if (ok && emit) g->layer_n_index_comp[il]++;
            const uint32_t decode_sparse_threshold =
                rocm_graph_decode_indexer_sparse_threshold(g);
            if (ok &&
                g->layer_n_comp[il] > decode_sparse_threshold &&
                g->layer_n_index_comp[il] > DS4_N_INDEXER_TOP_K) {
                const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
                if (!layer->indexer_attn_q_b ||
                    layer->indexer_attn_q_b->type != DS4_TENSOR_F16 ||
                    layer->indexer_attn_q_b->dim[0] != q_rank ||
                    layer->indexer_attn_q_b->dim[1] != indexer_q_dim) {
                    fprintf(stderr, "ds4: ROCm graph indexer q projection expects F16 weights\n");
                    ok = false;
                }
                if (ok && (!layer->indexer_proj ||
                           layer->indexer_proj->type != DS4_TENSOR_F16 ||
                           layer->indexer_proj->dim[0] != DS4_N_EMBD ||
                           layer->indexer_proj->dim[1] != DS4_N_INDEXER_HEAD)) {
                    fprintf(stderr, "ds4: ROCm graph indexer weight projection expects F16 weights\n");
                    ok = false;
                }
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->indexer_q, model->map, model->size,
                                                         layer->indexer_attn_q_b->abs_offset,
                                                         q_rank, indexer_q_dim,
                                                         g->qr_norm, 1) != 0;
                if (ok) ok = ds4_gpu_rope_tail_tensor(g->indexer_q, 1,
                                                        DS4_N_INDEXER_HEAD,
                                                        DS4_N_INDEXER_HEAD_DIM,
                                                        DS4_N_ROT,
                                                        pos,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        false,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW) != 0;
                if (ok) ok = ds4_gpu_dsv4_indexer_qat_tensor(g->indexer_q,
                                                              DS4_N_INDEXER_HEAD,
                                                              DS4_N_INDEXER_HEAD_DIM) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->indexer_weights, model->map, model->size,
                                                         layer->indexer_proj->abs_offset,
                                                         DS4_N_EMBD, DS4_N_INDEXER_HEAD,
                                                         g->attn_norm, 1) != 0;
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (ok && decode_index_stage_profile) {
                    ok = rocm_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_gpu_indexer_score_one_tensor(g->indexer_scores,
                                                                g->indexer_q,
                                                                g->indexer_weights,
                                                                g->layer_index_comp_cache[il],
                                                                g->layer_n_index_comp[il],
                                                                DS4_N_INDEXER_HEAD,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                index_scale) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = rocm_graph_indexer_stage_profile_boundary("decode_score",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                           g->indexer_scores,
                                                           g->layer_n_index_comp[il],
                                                           1,
                                                           DS4_N_INDEXER_TOP_K) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = rocm_graph_indexer_stage_profile_boundary("decode_topk",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                /* Decode used to materialize a dense compressed-row mask and
                 * call the generic gathered FlashAttention wrapper below.
                 * That wrapper scans every compressed row and rejects long
                 * contexts once raw+compressed rows exceed 8192.  Ratio-4 DS4
                 * attention is sparse after indexer top-k, so use the private
                 * indexed attention kernel instead: it scans only SWA raw rows
                 * plus the selected compressed rows, matching prefill and
                 * avoiding the long-context decode failure. */
                if (ok) {
                    comp_selected = g->comp_selected;
                    /*
                     * Contract: the indexer top-k is fixed by the model config
                     * and must remain the full 512 rows.  Do not reduce this for
                     * throughput benchmarks.
                     *
                     * Why: the indexer is not just an implementation detail.  It
                     * decides which compressed memory rows are visible to the
                     * attention kernel.  If we keep only 128/256 rows, the later
                     * indexed-attention math may be perfectly computed, but it is
                     * computed over the wrong candidate set: rows ranked 257-512
                     * are removed before softmax/PV can use them.  Those rows may
                     * carry weak-but-necessary evidence for retrieval, name/number
                     * recall, or long-context disambiguation.  The error is
                     * therefore semantic/algorithmic, not the acceptable kind of
                     * local numerical drift caused by a different reduction order
                     * or Tensor/NAX precision.
                     *
                     * Short prompt tests, first-token agreement, or even a small
                     * official-vector set can miss this because many prompts do
                     * not need the tail of the 512 selected compressed rows.  The
                     * failure appears only when the model needs information that
                     * fell below the reduced cutoff.  Optimizations belong inside
                     * the score/top-k/attention implementation while preserving
                     * DS4_N_INDEXER_TOP_K.
                     */
                    n_selected = DS4_N_INDEXER_TOP_K < g->layer_n_index_comp[il]
                        ? DS4_N_INDEXER_TOP_K
                        : g->layer_n_index_comp[il];
                }
            }
        }

        n_comp = g->layer_n_comp[il];
        comp_cache = g->layer_attn_comp_cache[il];
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("compressor_indexer");

    if (ok) {
        const uint32_t raw_start = rocm_graph_raw_start_for_span(g, pos, n_raw);
        if (n_comp != 0 && comp_selected != NULL && n_selected != 0) {
            ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                    g->heads,
                    model->map,
                    model->size,
                    layer->attn_sinks->abs_offset,
                    g->q,
                    raw_cache,
                    g->layer_attn_comp_cache[il],
                    rocm_graph_attn_comp_cache_is_f16(),
                    comp_selected,
                    1,
                    pos,
                    n_raw,
                    raw_cap,
                    raw_start,
                    n_comp,
                    n_selected,
                    g->raw_window,
                    ds4_layer_compress_ratio(il),
                    DS4_N_HEAD,
                    DS4_N_HEAD_DIM) != 0;
            if (ok && decode_index_stage_profile) {
                ok = rocm_graph_indexer_stage_profile_boundary("decode_attention",
                                                                il,
                                                                pos,
                                                                1,
                                                                n_comp,
                                                                &decode_index_stage_t0);
            }
        } else {
            ok = ds4_gpu_attention_decode_heads_tensor(g->heads,
                                                         model->map, model->size,
                                                         layer->attn_sinks->abs_offset,
                                                         g->q, raw_cache, n_raw,
                                                         raw_cap,
                                                         raw_start,
                                                         n_comp ? comp_cache : NULL,
                                                         rocm_graph_attn_comp_cache_is_f16(),
                                                         n_comp,
                                                         NULL,
                                                         0,
                                                         DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
        }
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("attention");
    if (ok) {
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->heads,
                                            1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
    }
    const bool fuse_attn_out_hc =
        !rocm_graph_use_reference_attn_out_hc();
    if (ok && fuse_attn_out_hc) {
        ok = ds4_gpu_attention_output_low_q8_tensor(g->attn_low,
                                                      model->map,
                                                      model->size,
                                                      layer->attn_output_a->abs_offset,
                                                      group_dim,
                                                      rank,
                                                      n_groups,
                                                      g->heads) != 0;
        if (ok) {
            ok = ds4_gpu_matmul_q8_0_hc_expand_tensor(g->after_attn_hc,
                                                        g->attn_out,
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_b->abs_offset,
                                                        (uint64_t)n_groups * rank,
                                                        DS4_N_EMBD,
                                                        g->attn_low,
                                                        g->cur_hc,
                                                        g->hc_split,
                                                        DS4_N_EMBD,
                                                        DS4_N_HC) != 0;
        }
    } else if (ok) {
        ok = ds4_gpu_attention_output_q8_batch_tensor(g->attn_out,
                                                        g->attn_low,
                                                        g->batch_group_tmp,
                                                        g->batch_low_tmp,
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_a->abs_offset,
                                                        layer->attn_output_b->abs_offset,
                                                        group_dim, rank,
                                                        n_groups, DS4_N_EMBD,
                                                        g->heads, 1) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("attn_output");
    if (ok) {
    }
    if (ok) {
    }
    if (ok && !fuse_attn_out_hc) {
        ok = ds4_gpu_hc_expand_tensor(g->after_attn_hc, g->attn_out, g->cur_hc,
                                        g->hc_post, g->hc_comb, DS4_N_EMBD, DS4_N_HC) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("attn_hc_post");
    if (ok) {
    }
    if (ok) ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->after_attn_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = rocm_graph_matmul_plain_tensor(g->hc_mix, model, layer->hc_ffn_fn,
                                                 hc_dim, mix_hc, g->flat_hc, 1);
    if (ok && fuse_hc_norm) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(g->ffn_cur,
                                                         g->ffn_norm,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->after_attn_hc,
                                                         model->map,
                                                         model->size,
                                                         layer->hc_ffn_scale->abs_offset,
                                                         layer->hc_ffn_base->abs_offset,
                                                         layer->ffn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_HC_SINKHORN_ITER,
                                                         DS4_HC_EPS,
                                                         DS4_RMS_EPS) != 0;
    } else if (ok) {
        ok = rocm_graph_decode_hc_pre(g->ffn_cur,
                                       g->hc_split,
                                       g->hc_mix,
                                       g->after_attn_hc,
                                       model,
                                       layer->hc_ffn_scale->abs_offset,
                                       layer->hc_ffn_base->abs_offset);
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("ffn_hc_pre");
    if (ok) {
    }
    if (ok) {
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(g->ffn_norm, g->ffn_cur,
                                                                   model->map, model->size,
                                                                   layer->ffn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("ffn_norm");
    if (ok) {
    }
    const uint64_t gate_row_bytes = ds4_routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = ds4_routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    if (ok) ok = rocm_graph_matmul_plain_tensor(g->router_logits, model, layer->ffn_gate_inp,
                                                 DS4_N_EMBD, DS4_N_EXPERT, g->ffn_norm, 1);
    if (ok) ok = ds4_gpu_router_select_tensor(g->router_selected, g->router_weights, g->router_probs,
                                                model->map, model->size,
                                                layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                (uint32_t)token,
                                                DS4_N_EXPERT,
                                                DS4_N_EXPERT_USED,
                                                DS4_EXPERT_WEIGHT_SCALE,
                                                0,
                                                0,
                                                layer->ffn_exp_probs_b != NULL,
                                                layer->ffn_gate_tid2eid != NULL,
                                                g->router_logits) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("router");
    if (ok) {
    }
    if (ok) ok = ds4_gpu_routed_moe_one_tensor(g->routed_out,
                                                 g->routed_gate,
                                                 g->routed_up,
                                                 g->routed_mid,
                                                 g->routed_down,
                                                 model->map, model->size,
                                                 layer->ffn_gate_exps->abs_offset,
                                                 layer->ffn_up_exps->abs_offset,
                                                 layer->ffn_down_exps->abs_offset,
                                                 layer->ffn_gate_exps->type,
                                                 layer->ffn_down_exps->type,
                                                 gate_expert_bytes, gate_row_bytes,
                                                 down_expert_bytes, down_row_bytes,
                                                 (uint32_t)expert_in_dim,
                                                 (uint32_t)down_in_dim,
                                                 (uint32_t)routed_out_dim,
                                                 g->router_selected, g->router_weights,
                                                 DS4_N_EXPERT,
                                                 DS4_N_EXPERT_USED,
                                                 DS4_SWIGLU_CLAMP_EXP,
                                                 g->ffn_norm,
                                                 NULL,
                                                 il,
                                                 false) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("routed_moe");
    const bool fuse_shared_gate_up =
        getenv("GUFO_DEEPSEEK_ROCM_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION") == NULL;
    if (ok && fuse_shared_gate_up) {
        ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(g->shared_gate,
                                                         g->shared_up,
                                                         g->shared_mid,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_gate_shexp->abs_offset,
                                                         layer->ffn_up_shexp->abs_offset,
                                                         DS4_N_EMBD,
                                                         shared_dim,
                                                         g->ffn_norm,
                                                         DS4_SWIGLU_CLAMP_EXP) != 0;
    } else {
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->shared_gate, model->map, model->size,
                                                  layer->ffn_gate_shexp->abs_offset,
                                                  DS4_N_EMBD, shared_dim,
                                                  g->ffn_norm, 1) != 0;
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->shared_up, model->map, model->size,
                                                  layer->ffn_up_shexp->abs_offset,
                                                  DS4_N_EMBD, shared_dim,
                                                  g->ffn_norm, 1) != 0;
        if (ok) ok = ds4_gpu_swiglu_tensor(g->shared_mid, g->shared_gate, g->shared_up,
                                           shared_dim, DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("shared_gate_up");
    const bool fuse_shared_down_hc =
        !rocm_graph_use_reference_shared_down_hc();
    if (ok && fuse_shared_down_hc) {
        ok = ds4_gpu_shared_down_hc_expand_q8_0_tensor(g->after_ffn_hc,
                                                         g->shared_out,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_down_shexp->abs_offset,
                                                         shared_dim,
                                                         DS4_N_EMBD,
                                                         g->shared_mid,
                                                         g->routed_out,
                                                         g->after_attn_hc,
                                                         g->hc_split,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC) != 0;
    } else if (ok) {
        ok = ds4_gpu_matmul_q8_0_tensor(g->shared_out, model->map, model->size,
                                          layer->ffn_down_shexp->abs_offset,
                                          shared_dim, DS4_N_EMBD,
                                          g->shared_mid, 1) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("shared_down");
    if (ok) {
    }
    if (ok && !fuse_shared_down_hc) {
        ok = ds4_gpu_hc_expand_add_split_tensor(g->after_ffn_hc,
                                                  g->routed_out,
                                                  g->shared_out,
                                                  g->after_attn_hc,
                                                  g->hc_split,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("ffn_hc_post");
#undef GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE
    if (ok) {
    }
    return ok;
}

/* Encode the final HC collapse, output norm, and vocab projection on ROCm. */
static bool rocm_graph_encode_output_head(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(g->output_pre,
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             g->flat_hc,
                                             1) != 0;
    if (ok) {
    }
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(g->output_weights,
                                                    g->output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) {
    }
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(g->output_embd,
                                                  g->cur_hc,
                                                  g->output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) {
    }
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                                  g->output_embd,
                                                  model->map,
                                                  model->size,
                                                  weights->output_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) {
    }
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->logits,
                                              model->map,
                                              model->size,
                                              weights->output->abs_offset,
                                              DS4_N_EMBD,
                                              vocab_dim,
                                              g->output_norm,
                                              1) != 0;
    if (ok) {
    }
    return ok;
}

/* Shared dense projection helpers used by decode and batched prefill. */
static bool rocm_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (w->type == DS4_TENSOR_F16) {
        return ds4_gpu_matmul_f16_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_F32) {
        return ds4_gpu_matmul_f32_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    fprintf(stderr, "ds4: ROCm plain matmul does not support %s\n", ds4_tensor_type_name(w->type));
    return false;
}

static bool rocm_graph_matmul_q8_0_named_tensor(
        const char             *module,
        uint32_t                il,
        uint32_t                pos0,
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    (void)module;
    (void)il;
    (void)pos0;
    const bool ok = ds4_gpu_matmul_q8_0_tensor(out,
                                                 model->map,
                                                 model->size,
                                                 w->abs_offset,
                                                 in_dim,
                                                 out_dim,
                                                 x,
                                                 n_tok) != 0;
    return ok;
}

/* =========================================================================
 * ROCm Release Decode and Prefill.
 * =========================================================================
 *
 * Everything below is the user-facing ROCm backend.
 */

/* Encode a full single-token decode step on ROCm.  This is the generation
 * hot path: update caches, run all layers, then produce logits. */
static bool rocm_graph_encode_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        bool                   need_logits,
        bool                   allow_split_flush) {
    if (g->raw_cap == 0) {
        fprintf(stderr, "ds4: ROCm graph raw KV cache is not allocated\n");
        return false;
    }
    const uint32_t raw_row = pos % g->raw_cap;
    const uint32_t n_raw = rocm_graph_raw_span_for_batch(g, pos, 1);

    bool ok = ds4_gpu_embed_token_hc_tensor(g->cur_hc,
                                              model->map,
                                              model->size,
                                              weights->token_embd->abs_offset,
                                              (uint32_t)weights->token_embd->dim[1],
                                              (uint32_t)token,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;

    /*
     * Start executing the prefix of the decode graph while the CPU is still
     * encoding the rest. The split point is layer-based because this executor is
     * a fixed DS4 tape, not a dynamic node graph; four layers is the measured
     * point where the prefix is large enough to hide useful work without
     * starving the second command buffer.
     */
    uint32_t split_after_layers = 4;
    const char *split_env = getenv("GUFO_DEEPSEEK_ROCM_GRAPH_TOKEN_SPLIT_LAYERS");
    if (split_env && split_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(split_env, &end, 10);
        if (end != split_env && v <= DS4_N_LAYER) split_after_layers = (uint32_t)v;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ok = rocm_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             raw_row,
                                             n_raw,
                                             token);
        ds4_gpu_tensor *tmp = g->cur_hc;
        g->cur_hc = g->after_ffn_hc;
        g->after_ffn_hc = tmp;
        if (ok && allow_split_flush && split_after_layers != 0 && il + 1u == split_after_layers) {
            ok = ds4_gpu_flush_commands() != 0;
        }
    }

    if (ok && need_logits) {
        ok = rocm_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    }
    return ok;
}

static ds4_gpu_tensor *rocm_graph_tensor_row_view(
        ds4_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values) {
    return ds4_gpu_tensor_view(base,
                                 (uint64_t)row * row_values * sizeof(float),
                                 row_values * sizeof(float));
}

/* Upload prompt token ids for kernels that need token-aware hash routing. */
static bool rocm_graph_upload_prompt_tokens(
        ds4_gpu_tensor *out_tokens,
        const token_vec  *prompt,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (!out_tokens || pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) {
        return false;
    }

    auto *tokens = static_cast<int32_t *>(
        ds4_xmalloc(static_cast<size_t>(n_tokens) * sizeof(int32_t)));
    for (uint32_t i = 0; i < n_tokens; i++) tokens[i] = prompt->v[pos0 + i];

    const bool ok = ds4_gpu_tensor_write(out_tokens,
                                           0,
                                           tokens,
                                           (uint64_t)n_tokens * sizeof(tokens[0])) != 0;
    free(tokens);
    return ok;
}

/* Rebuild ratio-4 compressor state after chunked prefill so a following decode
 * token sees the same rolling compression window. */
static bool rocm_graph_refresh_ratio4_compressor_state(
        ds4_gpu_graph  *g,
        const ds4_model  *model,
        ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score,
        const ds4_tensor *kv_weight,
        const ds4_tensor *score_weight,
        const ds4_tensor *ape,
        uint32_t          head_dim,
        uint32_t          width,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (n_tokens < 4) {
        return true;
    }
    if (!g || !model || !state_kv || !state_score || !kv_weight || !score_weight || !ape ||
        head_dim == 0 || width == 0) {
        return false;
    }

    /*
     * The recurrent ratio-4 state is intentionally rebuilt from the last
     * four tokens using the small-batch projection kernel. The full-chunk
     * projection is already available, but it uses the matrix-matrix path;
     * mixing those two accumulation orders changes a few FP8 rounding
     * decisions in later chunks.
     */
    ds4_gpu_tensor *tail_hc = ds4_gpu_tensor_view(
            g->batch_attn_norm,
            (uint64_t)(n_tokens - 4u) * DS4_N_EMBD * sizeof(float),
            4ull * DS4_N_EMBD * sizeof(float));
    bool ok = tail_hc != NULL;
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_kv,
                                         model->map,
                                         model->size,
                                         kv_weight->abs_offset,
                                         DS4_N_EMBD,
                                         width,
                                         tail_hc,
                                         4) != 0;
    }
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_sc,
                                         model->map,
                                         model->size,
                                         score_weight->abs_offset,
                                         DS4_N_EMBD,
                                         width,
                                         tail_hc,
                                         4) != 0;
    }
    if (ok) {
        ok = ds4_gpu_compressor_prefill_state_ratio4_tensor(state_kv,
                                                              state_score,
                                                              g->batch_comp_kv,
                                                              g->batch_comp_sc,
                                                              model->map,
                                                              model->size,
                                                              ape->abs_offset,
                                                              ape->type,
                                                              head_dim,
                                                              pos0 + n_tokens - 4u) != 0;
    }
    ds4_gpu_tensor_free(tail_hc);
    return ok;
}

/* Seed the batched HC state from token ids: every HC stream starts as the same
 * 4096-wide embedding.  The model-owned ROCm kernel handles every batch size;
 * model data never dequantizes or uploads embeddings through a host fallback. */
static bool rocm_graph_upload_prompt_embeddings_hc(
        ds4_gpu_tensor   *out_hc,
        ds4_gpu_tensor   *tokens,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;
    if (!tokens || n_tokens == 0) return false;
    return ds4_gpu_embed_tokens_hc_tensor(out_hc,
                                           tokens,
                                           model->map,
                                           model->size,
                                           weights->token_embd->abs_offset,
                                           (uint32_t)weights->token_embd->dim[1],
                                           n_tokens,
                                           DS4_N_EMBD,
                                           DS4_N_HC) != 0;
}

static bool rocm_graph_warmup_prefill_kernels(
        ds4_gpu_graph   *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           n_tokens) {
    static bool warmed = false;
    if (warmed || getenv("GUFO_DEEPSEEK_ROCM_NO_PREFILL_KERNEL_WARMUP") != NULL) return true;

    /*
     * The first batched F16 matmul can pay ROCm's one-time pipeline execution
     * cost. Run the same HC attention projection on scratch storage before the
     * measured prefill. The output is overwritten by the real graph.
     */
    if (n_tokens <= 8) return true;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(g->batch_hc_mix,
                                         model->map,
                                         model->size,
                                         weights->layer[0].hc_attn_fn->abs_offset,
                                         hc_dim,
                                         mix_hc,
                                         g->batch_flat_hc,
                                         n_tokens) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) {
        fprintf(stderr, "ds4: ROCm prefill kernel warmup failed\n");
        return false;
    }

    warmed = true;
    return true;
}

/* Encode the batched prefill attention half for one layer.  It mirrors the CPU
 * layer-major path: HC pre/norm, Q/KV, cache/compression, prefix attention. */
static bool rocm_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = ds4_now_seconds();
    if (stage != NULL) {
        fprintf(stderr,
                    "ds4: ROCm indexer stage layer=%u pos=%u tokens=%u comp=%u %s=%.3f ms\n",
                il,
                pos0,
                n_tokens,
                n_comp,
                stage,
                (now - *stage_t0) * 1000.0);
    }
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

/* Optional prefill stage profiler. It intentionally ends the current ROCm
 * command buffer and waits, so the printed number includes encoding plus GPU
 * execution for the stage just emitted. This is disabled by default because it
 * adds synchronization points and changes scheduling. */
static bool rocm_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = ds4_now_seconds();
    fprintf(stderr,
            "ds4: ROCm layer stage part=%s layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            part,
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static bool rocm_graph_q_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = ds4_now_seconds();
    fprintf(stderr,
            "ds4: ROCm Q path stage layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static bool rocm_graph_encode_layer_attention_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const bool compressed = ratio != 0;
    const bool zero_prefix = pos0 == 0;
    const bool index_stage_profile = getenv("GUFO_DEEPSEEK_ROCM_INDEXER_STAGE_PROFILE") != NULL;
    const bool layer_stage_profile = getenv("GUFO_DEEPSEEK_ROCM_LAYER_STAGE_PROFILE") != NULL;
    const bool q_stage_profile = getenv("GUFO_DEEPSEEK_ROCM_Q_STAGE_PROFILE") != NULL;
    double layer_stage_t0 = layer_stage_profile ? ds4_now_seconds() : 0.0;
    double q_stage_t0 = q_stage_profile ? ds4_now_seconds() : 0.0;
#define GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = rocm_graph_layer_stage_profile_boundary("attn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)
#define GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE(name) do { \
        if (ok && q_stage_profile) { \
            ok = rocm_graph_q_stage_profile_boundary((name), il, pos0, n_tokens, &q_stage_t0); \
        } \
    } while (0)
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    auto *comp_counts = compressed
                            ? static_cast<uint32_t *>(
                                  ds4_xcalloc(n_tokens, sizeof(uint32_t)))
                            : nullptr;
    auto *index_counts = ratio == 4
                             ? static_cast<uint32_t *>(
                                   ds4_xcalloc(n_tokens, sizeof(uint32_t)))
                             : nullptr;
    const bool qkv_rms_fused = !rocm_graph_use_reference_qkv_norm();
    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
            g->batch_attn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(
            g->batch_after_attn_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(hc_mix_view,
                                             model->map,
                                             model->size,
                                             layer->hc_attn_fn->abs_offset,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (rocm_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_attn_scale->abs_offset,
                                                        layer->hc_attn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(attn_cur_view,
                                                            g->batch_cur_hc,
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(attn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            g->batch_cur_hc,
                                                            model->map,
                                                            model->size,
                                                            layer->hc_attn_scale->abs_offset,
                                                            layer->hc_attn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
    }
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("hc_pre");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm,
                                                       g->batch_attn_cur,
                                                       model->map,
                                                       model->size,
                                                       layer->attn_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("norm");
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("pre_q");
    if (ok) ok = rocm_graph_matmul_q8_0_named_tensor("attn_q_a",
                                                      il,
                                                      pos0,
                                                      g->batch_qr,
                                                      model,
                                                      layer->attn_q_a,
                                                      DS4_N_EMBD,
                                                      q_rank,
                                                      g->batch_attn_norm,
                                                      n_tokens);
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("q_a");
    if (qkv_rms_fused) {
        if (ok) ok = rocm_graph_matmul_q8_0_named_tensor("attn_kv",
                                                          il,
                                                          pos0,
                                                          g->batch_kv_raw,
                                                          model,
                                                          layer->attn_kv,
                                                          DS4_N_EMBD,
                                                          DS4_N_HEAD_DIM,
                                                          g->batch_attn_norm,
                                                          n_tokens);
        if (ok) {
        }
        if (ok) ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->batch_qr_norm,
                                                             g->batch_qr,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             g->batch_kv,
                                                             g->batch_kv_raw,
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             DS4_N_HEAD_DIM,
                                                             n_tokens,
                                                             DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_qr_norm,
                                                           g->batch_qr,
                                                           model->map,
                                                           model->size,
                                                           layer->attn_q_a_norm->abs_offset,
                                                           (uint32_t)q_rank,
                                                           n_tokens,
                                                           DS4_RMS_EPS) != 0;
    }
    if (ok) {
    }
    if (qkv_rms_fused && ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("q_a_norm");
    if (ok) ok = rocm_graph_matmul_q8_0_named_tensor("attn_q_b",
                                                      il,
                                                      pos0,
                                                      g->batch_q,
                                                      model,
                                                      layer->attn_q_b,
                                                      q_rank,
                                                      q_dim,
                                                      g->batch_qr_norm,
                                                      n_tokens);
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("q_b");
    if (ok) ok = ds4_gpu_head_rms_norm_tensor(g->batch_q,
                                                n_tokens,
                                                DS4_N_HEAD,
                                                DS4_N_HEAD_DIM,
                                                DS4_RMS_EPS) != 0;
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("head_norm");
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_q,
                                            n_tokens,
                                            DS4_N_HEAD,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("rope");
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("q_path");
    if (!qkv_rms_fused) {
        if (ok) ok = rocm_graph_matmul_q8_0_named_tensor("attn_kv",
                                                          il,
                                                          pos0,
                                                          g->batch_kv_raw,
                                                          model,
                                                          layer->attn_kv,
                                                          DS4_N_EMBD,
                                                          DS4_N_HEAD_DIM,
                                                          g->batch_attn_norm,
                                                          n_tokens);
        if (ok) {
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_kv,
                                                           g->batch_kv_raw,
                                                           model->map,
                                                           model->size,
                                                           layer->attn_kv_a_norm->abs_offset,
                                                           DS4_N_HEAD_DIM,
                                                           n_tokens,
                                                           DS4_RMS_EPS) != 0;
        if (ok) {
        }
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_kv,
                                            n_tokens,
                                            DS4_N_HEAD_KV,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
    }
    if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->batch_kv,
                                                       n_tokens,
                                                       DS4_N_HEAD_DIM,
                                                       DS4_N_ROT) != 0;
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("kv_path");
    /*
     * Static graph order is q, kv, cpy_k(raw SWA), then attention. For a
     * zero-prefix batch it is safe to store the whole batch at once: attention
     * reads the contiguous batch KV, and the ring only has to end with the last
     * SWA rows for later chunks/decode. For nonzero chunks the physical ring is
     * sized to hold the current chunk plus the previous SWA window, while the
     * attention mask still enforces the 128-token logical window.
     */
    if (ok && zero_prefix) ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                                    g->batch_kv,
                                                                    g->raw_cap,
                                                                    pos0,
                                                                    n_tokens,
                                                                    DS4_N_HEAD_DIM) != 0;
    const bool raw_batch_attention = zero_prefix && ratio == 0;
    bool batch_attention_done = false;

    if (ok && raw_batch_attention) {
        ok = ds4_gpu_attention_prefill_raw_heads_tensor(g->batch_heads,
                                                          model->map,
                                                          model->size,
                                                          layer->attn_sinks->abs_offset,
                                                          g->batch_q,
                                                          g->batch_kv,
                                                          n_tokens,
                                                          g->raw_window,
                                                          DS4_N_HEAD,
                                                          DS4_N_HEAD_DIM) != 0;
        if (ok) batch_attention_done = true;
    } else if (ok && !zero_prefix && ratio == 0 && n_tokens <= g->raw_cap) {
        /*
         * The ubatch path stores the whole batch in the SWA cache, then runs
         * one batched attention kernel with an absolute-position causal/window
         * mask.  This avoids mixing prefill with the different single-token
         * attention path.
         */
        const uint32_t n_raw = rocm_graph_raw_span_for_batch(g, pos0, n_tokens);
        /* Nonzero prompt chunks read the SWA cache as a ring.  FlashAttention
         * receives a linearized window starting at raw_start, not physical row
         * zero; otherwise wrapped chunks silently miss recent raw keys. */
        const uint32_t raw_start = rocm_graph_raw_start_for_span(g,
                                                                  pos0 + n_tokens - 1u,
                                                                  n_raw);
        ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                 g->batch_kv,
                                                 g->raw_cap,
                                                 pos0,
                                                 n_tokens,
                                                 DS4_N_HEAD_DIM) != 0;
        if (ok) {
        }
        if (ok) {
            ok = ds4_gpu_attention_decode_raw_batch_heads_tensor(g->batch_heads,
                                                                   model->map,
                                                                   model->size,
                                                                   layer->attn_sinks->abs_offset,
                                                                   g->batch_q,
                                                                   g->layer_raw_cache[il],
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   g->raw_cap,
                                                                   raw_start,
                                                                   g->raw_window,
                                                                   DS4_N_HEAD,
                                                                   DS4_N_HEAD_DIM) != 0;
        }
        if (ok) batch_attention_done = true;
    } else if (ok && ratio != 0) {
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool have_attn_comp = layer->attn_compressor_kv && layer->attn_compressor_gate &&
                                    layer->attn_compressor_ape && layer->attn_compressor_norm;
        if (!have_attn_comp) {
            fprintf(stderr, "ds4: ROCm layer-major prefill needs attention compressor weights\n");
            ok = false;
        }
        if (ok) {
            ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_kv,
                                             model->map,
                                             model->size,
                                             layer->attn_compressor_kv->abs_offset,
                                             DS4_N_EMBD,
                                             comp_width,
                                             g->batch_attn_norm,
                                             n_tokens) != 0;
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_sc,
                                                     model->map,
                                                     model->size,
                                                     layer->attn_compressor_gate->abs_offset,
                                                     DS4_N_EMBD,
                                                     comp_width,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
        }
        uint32_t n_comp = g->layer_n_comp[il];
        if (zero_prefix) {
            n_comp = n_tokens / ratio;
            if (ok && n_comp > g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: ROCm layer-major compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            ds4_gpu_tensor *attn_comp_target = nullptr;
            ds4_gpu_tensor *attn_comp_mirror = nullptr;
            if (ok) {
                attn_comp_target = rocm_graph_attn_comp_prefill_target(g, il, 0, n_comp);
                if (ratio == 4) {
                    attn_comp_mirror =
                        rocm_graph_attn_comp_prefill_mirror(g, il, 0, n_comp);
                }
                ok = attn_comp_target != nullptr &&
                     (ratio != 4 || attn_comp_mirror != nullptr) &&
                     ds4_gpu_compressor_prefill_tensor(attn_comp_target,
                                                         g->layer_attn_state_kv[il],
                                                         g->layer_attn_state_score[il],
                                                         g->batch_comp_kv,
                                                         g->batch_comp_sc,
                                                         model->map,
                                                         model->size,
                                                         layer->attn_compressor_ape->abs_offset,
                                                         layer->attn_compressor_ape->type,
                                                         layer->attn_compressor_norm->abs_offset,
                                                         layer->attn_compressor_norm->type,
                                                         DS4_N_HEAD_DIM,
                                                         ratio,
                                                         pos0,
                                                         n_tokens,
                                                         DS4_N_ROT,
                                                         compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                         true,
                                                         freq_base,
                                                         freq_scale,
                                                         ext_factor,
                                                         attn_factor,
                                                         DS4_ROPE_YARN_BETA_FAST,
                                                         DS4_ROPE_YARN_BETA_SLOW,
                                                         DS4_RMS_EPS,
                                                         attn_comp_mirror) != 0;
                if (ok && ratio == 4) {
                    ok = rocm_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
            }
            if (ok) {
                g->layer_n_comp[il] = n_comp;
                for (uint32_t t = 0; t < n_tokens; t++) {
                    comp_counts[t] = (pos0 + t + 1u) / ratio;
                }
                if (n_comp != 0) {
                }
            }
            rocm_graph_attn_comp_prefill_target_free(attn_comp_mirror);
            rocm_graph_attn_comp_prefill_target_free(attn_comp_target);
        } else {
            const bool aligned_chunk = (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
            if (aligned_chunk) {
                const uint32_t comp_before = g->layer_n_comp[il];
                const uint32_t comp_chunk = n_tokens / ratio;
                if (comp_before + comp_chunk > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: ROCm graph compressed KV cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                ds4_gpu_tensor *attn_comp_target = ok
                    ? rocm_graph_attn_comp_prefill_target(
                          g, il, comp_before, comp_chunk)
                    : nullptr;
                ds4_gpu_tensor *attn_comp_mirror = ok && ratio == 4
                    ? rocm_graph_attn_comp_prefill_mirror(
                          g, il, comp_before, comp_chunk)
                    : nullptr;
                if (ok && (!attn_comp_target ||
                           (ratio == 4 && !attn_comp_mirror))) {
                    ok = false;
                }
                if (ok && ratio == 4) {
                    ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                            attn_comp_target,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            g->batch_comp_kv,
                            g->batch_comp_sc,
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            true,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS,
                            attn_comp_mirror) != 0;
                } else if (ok) {
                    ok = ds4_gpu_compressor_prefill_tensor(
                            attn_comp_target,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            g->batch_comp_kv,
                            g->batch_comp_sc,
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            ratio,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            true,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS,
                            attn_comp_mirror) != 0;
                }
                if (ok && ratio == 4) {
                    ok = rocm_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_comp[il] = comp_before + comp_chunk;
                    if (comp_counts) {
                        for (uint32_t t = 0; t < n_tokens; t++) {
                            comp_counts[t] = (pos0 + t + 1u) / ratio;
                        }
                    }
                }
                rocm_graph_attn_comp_prefill_target_free(attn_comp_mirror);
                rocm_graph_attn_comp_prefill_target_free(attn_comp_target);
            } else {
                for (uint32_t t = 0; ok && t < n_tokens; t++) {
                    const uint32_t pos = pos0 + t;
                    const bool emit = ((pos + 1u) % ratio) == 0u;
                    if (emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: ROCm graph compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                        break;
                    }
                    ds4_gpu_tensor *kv_view = rocm_graph_tensor_row_view(g->batch_comp_kv, t, comp_width);
                    ds4_gpu_tensor *sc_view = rocm_graph_tensor_row_view(g->batch_comp_sc, t, comp_width);
                    const uint32_t comp_row = g->layer_n_comp[il];
                    ok = kv_view && sc_view &&
                         ds4_gpu_compressor_update_tensor(kv_view,
                                                            sc_view,
                                                            g->layer_attn_state_kv[il],
                                                            g->layer_attn_state_score[il],
                                                            rocm_graph_attn_comp_update_target(g, il),
                                                            model->map,
                                                            model->size,
                                                            layer->attn_compressor_ape->abs_offset,
                                                            layer->attn_compressor_ape->type,
                                                            layer->attn_compressor_norm->abs_offset,
                                                            layer->attn_compressor_norm->type,
                                                            DS4_N_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            rocm_graph_attn_comp_update_row(comp_row),
                                                            DS4_N_ROT,
                                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            DS4_ROPE_YARN_BETA_FAST,
                                                            DS4_ROPE_YARN_BETA_SLOW,
                                                            DS4_RMS_EPS,
                                                            false,
                                                            false,
                                                            false) != 0;
                    if (ok && emit) {
                        ds4_gpu_tensor *comp_row_view = rocm_graph_attn_comp_row_view(g, il, comp_row);
                        ok = comp_row_view && rocm_graph_quantize_attn_comp_row(
                            g, il, comp_row, comp_row_view);
                        if (ok) {
                        }
                        ds4_gpu_tensor_free(comp_row_view);
                    }
                    if (ok && emit) g->layer_n_comp[il]++;
                    if (comp_counts) comp_counts[t] = g->layer_n_comp[il];
                    ds4_gpu_tensor_free(sc_view);
                    ds4_gpu_tensor_free(kv_view);
                }
            }
            n_comp = g->layer_n_comp[il];
        }
        GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("compressor");

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                !layer->indexer_attn_q_b || !layer->indexer_proj) {
                fprintf(stderr, "ds4: ROCm layer-major prefill needs indexer weights\n");
                ok = false;
            }
            if (ok) {
                ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_kv,
                                                 model->map,
                                                 model->size,
                                                 layer->indexer_compressor_kv->abs_offset,
                                                 DS4_N_EMBD,
                                                 index_width,
                                                 g->batch_attn_norm,
                                                 n_tokens) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_sc,
                                                         model->map,
                                                         model->size,
                                                         layer->indexer_compressor_gate->abs_offset,
                                                         DS4_N_EMBD,
                                                         index_width,
                                                         g->batch_attn_norm,
                                                         n_tokens) != 0;
            }
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_indexer_q,
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_attn_q_b->abs_offset,
                                                     q_rank,
                                                     (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                     g->batch_qr_norm,
                                                     n_tokens) != 0;
            if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_indexer_q,
                                                    n_tokens,
                                                    DS4_N_INDEXER_HEAD,
                                                    DS4_N_INDEXER_HEAD_DIM,
                                                    DS4_N_ROT,
                                                    pos0,
                                                    compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                    false,
                                                    freq_base,
                                                    freq_scale,
                                                    ext_factor,
                                                    attn_factor,
                                                    DS4_ROPE_YARN_BETA_FAST,
                                                    DS4_ROPE_YARN_BETA_SLOW) != 0;
            if (ok) ok = ds4_gpu_dsv4_indexer_qat_tensor(g->batch_indexer_q,
                                                          n_tokens * DS4_N_INDEXER_HEAD,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_indexer_weights,
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_proj->abs_offset,
                                                     DS4_N_EMBD,
                                                     DS4_N_INDEXER_HEAD,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
            if (zero_prefix) {
                if (ok && n_comp > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: ROCm layer-major indexer cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok) {
                    ok = ds4_gpu_compressor_prefill_tensor(g->layer_index_comp_cache[il],
                                                             g->layer_index_state_kv[il],
                                                             g->layer_index_state_score[il],
                                                             g->batch_comp_kv,
                                                             g->batch_comp_sc,
                                                             model->map,
                                                             model->size,
                                                             layer->indexer_compressor_ape->abs_offset,
                                                             layer->indexer_compressor_ape->type,
                                                             layer->indexer_compressor_norm->abs_offset,
                                                             layer->indexer_compressor_norm->type,
                                                             DS4_N_INDEXER_HEAD_DIM,
                                                             ratio,
                                                             pos0,
                                                             n_tokens,
                                                             DS4_N_ROT,
                                                             compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                             false,
                                                             freq_base,
                                                             freq_scale,
                                                             ext_factor,
                                                             attn_factor,
                                                             DS4_ROPE_YARN_BETA_FAST,
                                                             DS4_ROPE_YARN_BETA_SLOW,
                                                             DS4_RMS_EPS,
                                                             nullptr) != 0;
                }
                if (ok && n_comp != 0) {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(g->layer_index_comp_cache[il],
                                                          n_comp,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
                }
                if (ok) {
                    ok = rocm_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_index_state_kv[il],
                                                                     g->layer_index_state_score[il],
                                                                     layer->indexer_compressor_kv,
                                                                     layer->indexer_compressor_gate,
                                                                     layer->indexer_compressor_ape,
                                                                     DS4_N_INDEXER_HEAD_DIM,
                                                                     index_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_index_comp[il] = n_comp;
                    for (uint32_t t = 0; t < n_tokens; t++) {
                        index_counts[t] = (pos0 + t + 1u) / ratio;
                    }
                    if (n_comp != 0) {
                    }
                }
            } else {
                const bool aligned_chunk = (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
                if (aligned_chunk) {
                    const uint32_t index_before = g->layer_n_index_comp[il];
                    const uint32_t index_chunk = n_tokens / ratio;
                    if (index_before + index_chunk > g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: ROCm graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                    }
                    ds4_gpu_tensor *index_view = NULL;
                    if (ok) {
                        index_view = ds4_gpu_tensor_view(
                                g->layer_index_comp_cache[il],
                                (uint64_t)index_before * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                (uint64_t)index_chunk * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                        ok = index_view != NULL;
                    }
                    if (ok) {
                        ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                                index_view,
                                g->layer_index_state_kv[il],
                                g->layer_index_state_score[il],
                                g->batch_comp_kv,
                                g->batch_comp_sc,
                                model->map,
                                model->size,
                                layer->indexer_compressor_ape->abs_offset,
                                layer->indexer_compressor_ape->type,
                                layer->indexer_compressor_norm->abs_offset,
                                layer->indexer_compressor_norm->type,
                                DS4_N_INDEXER_HEAD_DIM,
                                pos0,
                                n_tokens,
                                DS4_N_ROT,
                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                false,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                DS4_ROPE_YARN_BETA_FAST,
                                DS4_ROPE_YARN_BETA_SLOW,
                                DS4_RMS_EPS,
                                nullptr) != 0;
                    }
                    if (ok && index_chunk != 0) {
                        ok = ds4_gpu_dsv4_indexer_qat_tensor(index_view,
                                                              index_chunk,
                                                              DS4_N_INDEXER_HEAD_DIM) != 0;
                    }
                    if (ok) {
                        ok = rocm_graph_refresh_ratio4_compressor_state(g,
                                                                         model,
                                                                         g->layer_index_state_kv[il],
                                                                         g->layer_index_state_score[il],
                                                                         layer->indexer_compressor_kv,
                                                                         layer->indexer_compressor_gate,
                                                                         layer->indexer_compressor_ape,
                                                                         DS4_N_INDEXER_HEAD_DIM,
                                                                         index_width,
                                                                         pos0,
                                                                         n_tokens);
                    }
                    if (ok) {
                        g->layer_n_index_comp[il] = index_before + index_chunk;
                        if (index_counts) {
                            for (uint32_t t = 0; t < n_tokens; t++) {
                                index_counts[t] = (pos0 + t + 1u) / ratio;
                            }
                        }
                    }
                    ds4_gpu_tensor_free(index_view);
                } else {
                    for (uint32_t t = 0; ok && t < n_tokens; t++) {
                        const uint32_t pos = pos0 + t;
                        const bool emit = ((pos + 1u) % ratio) == 0u;
                        if (emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                            fprintf(stderr, "ds4: ROCm graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                            ok = false;
                            break;
                        }
                        ds4_gpu_tensor *kv_view = rocm_graph_tensor_row_view(g->batch_comp_kv, t, index_width);
                        ds4_gpu_tensor *sc_view = rocm_graph_tensor_row_view(g->batch_comp_sc, t, index_width);
                        const uint32_t index_row = g->layer_n_index_comp[il];
                        ok = kv_view && sc_view &&
                             ds4_gpu_compressor_update_tensor(kv_view,
                                                                sc_view,
                                                                g->layer_index_state_kv[il],
                                                                g->layer_index_state_score[il],
                                                                g->layer_index_comp_cache[il],
                                                                model->map,
                                                                model->size,
                                                                layer->indexer_compressor_ape->abs_offset,
                                                                layer->indexer_compressor_ape->type,
                                                                layer->indexer_compressor_norm->abs_offset,
                                                                layer->indexer_compressor_norm->type,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                ratio,
                                                                pos,
                                                                index_row,
                                                                DS4_N_ROT,
                                                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                                freq_base,
                                                                freq_scale,
                                                                ext_factor,
                                                                attn_factor,
                                                                DS4_ROPE_YARN_BETA_FAST,
                                                                DS4_ROPE_YARN_BETA_SLOW,
                                                                DS4_RMS_EPS,
                                                                false,
                                                                false,
                                                                false) != 0;
                        if (ok && emit) {
                            ds4_gpu_tensor *index_row_view = ds4_gpu_tensor_view(
                                    g->layer_index_comp_cache[il],
                                    (uint64_t)index_row * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                    (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                            if (!index_row_view) {
                                ok = false;
                            } else {
                                ok = ds4_gpu_dsv4_indexer_qat_tensor(index_row_view,
                                                                      1,
                                                                      DS4_N_INDEXER_HEAD_DIM) != 0;
                                ds4_gpu_tensor_free(index_row_view);
                            }
                        }
                        if (ok && emit) g->layer_n_index_comp[il]++;
                        if (index_counts) index_counts[t] = g->layer_n_index_comp[il];
                        ds4_gpu_tensor_free(sc_view);
                        ds4_gpu_tensor_free(kv_view);
                    }
                }
            }
        }
        if (ratio == 4) GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("indexer_setup");

        if (ok && !zero_prefix && n_tokens <= g->raw_cap) {
            const uint32_t n_raw = rocm_graph_raw_span_for_batch(g, pos0, n_tokens);
            /* See the raw-only branch above: batched mixed attention also
             * consumes a logical raw window, linearized out of the ring. */
            const uint32_t raw_start = rocm_graph_raw_start_for_span(g,
                                                                      pos0 + n_tokens - 1u,
                                                                      n_raw);
            uint32_t use_comp_mask = 0;
            bool use_indexed_comp = false;
            double index_stage_t0 = 0.0;

            ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                     g->batch_kv,
                                                     g->raw_cap,
                                                     pos0,
                                                     n_tokens,
                                                     DS4_N_HEAD_DIM) != 0;
            if (ok && ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K) {
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (index_stage_profile) {
                    ok = rocm_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                ok = ds4_gpu_indexer_scores_decode_batch_tensor(g->indexer_scores,
                                                                  g->batch_indexer_q,
                                                                  g->batch_indexer_weights,
                                                                  g->layer_index_comp_cache[il],
                                                                  n_comp,
                                                                  n_tokens,
                                                                  pos0,
                                                                  DS4_N_INDEXER_HEAD,
                                                                  DS4_N_INDEXER_HEAD_DIM,
                                                                  ratio,
                                                                  index_scale) != 0;
                if (ok && index_stage_profile) {
                    ok = rocm_graph_indexer_stage_profile_boundary("score",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                }
                if (ok) {
                    ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                       g->indexer_scores,
                                                       n_comp,
                                                       n_tokens,
                                                       DS4_N_INDEXER_TOP_K) != 0;
                    if (ok && index_stage_profile) {
                        ok = rocm_graph_indexer_stage_profile_boundary("topk",
                                                                        il,
                                                                        pos0,
                                                                        n_tokens,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                    if (ok) {
                    }
                }
                if (ok) {
                    use_indexed_comp = true;
                }
                use_comp_mask = 1;
            }
            if (ok) {
                if (use_indexed_comp) {
                    ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(g->batch_heads,
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              g->batch_q,
                                                                              g->layer_raw_cache[il],
                                                                              rocm_graph_prefill_attn_comp_cache(
                                                                                  g, il, n_tokens),
                                                                              rocm_graph_prefill_attn_comp_cache_is_f16(
                                                                                  g, il, n_tokens),
                                                                              g->comp_selected,
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              DS4_N_INDEXER_TOP_K,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              DS4_N_HEAD,
                                                                              DS4_N_HEAD_DIM) != 0;
                    if (ok && index_stage_profile) {
                        ok = rocm_graph_indexer_stage_profile_boundary("attention",
                                                                        il,
                                                                        pos0,
                                                                        n_tokens,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                } else {
                    ok = ds4_gpu_attention_decode_mixed_batch_heads_tensor(g->batch_heads,
                                                                             model->map,
                                                                             model->size,
                                                                             layer->attn_sinks->abs_offset,
                                                                             g->batch_q,
                                                                             g->layer_raw_cache[il],
                                                                             g->layer_attn_comp_cache[il],
                                                                             rocm_graph_attn_comp_cache_is_f16(),
                                                                             use_comp_mask ? g->comp_mask : NULL,
                                                                             use_comp_mask,
                                                                             n_tokens,
                                                                             pos0,
                                                                             n_raw,
                                                                             g->raw_cap,
                                                                             raw_start,
                                                                             n_comp,
                                                                             g->raw_window,
                                                                             ratio,
                                                                             DS4_N_HEAD,
                                                                             DS4_N_HEAD_DIM) != 0;
                }
            }
            if (ok) batch_attention_done = true;
        }

        const bool topk_prefill_needed = ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K;
        if (ok && zero_prefix && topk_prefill_needed && n_comp != 0) {
            const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
            double index_stage_t0 = 0.0;
            if (index_stage_profile) {
                ok = rocm_graph_indexer_stage_profile_boundary(NULL,
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            ok = ds4_gpu_indexer_scores_prefill_tensor(g->indexer_scores,
                                                         g->batch_indexer_q,
                                                         g->batch_indexer_weights,
                                                         g->layer_index_comp_cache[il],
                                                         n_comp,
                                                         n_tokens,
                                                         DS4_N_INDEXER_HEAD,
                                                         DS4_N_INDEXER_HEAD_DIM,
                                                         ratio,
                                                         index_scale) != 0;
            if (ok && index_stage_profile) {
                ok = rocm_graph_indexer_stage_profile_boundary("score",
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            if (ok) {
            }
            if (ok) {
                ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                   g->indexer_scores,
                                                   n_comp,
                                                   n_tokens,
                                                   DS4_N_INDEXER_TOP_K) != 0;
                if (ok && index_stage_profile) {
                    ok = rocm_graph_indexer_stage_profile_boundary("topk",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                }
            }
            if (ok) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(g->batch_heads,
                                                                          model->map,
                                                                          model->size,
                                                                          layer->attn_sinks->abs_offset,
                                                                          g->batch_q,
                                                                          g->layer_raw_cache[il],
                                                                          rocm_graph_prefill_attn_comp_cache(
                                                                              g, il, n_tokens),
                                                                          rocm_graph_prefill_attn_comp_cache_is_f16(
                                                                              g, il, n_tokens),
                                                                          g->comp_selected,
                                                                          n_tokens,
                                                                          pos0,
                                                                          n_tokens,
                                                                          g->raw_cap,
                                                                          0,
                                                                          n_comp,
                                                                          DS4_N_INDEXER_TOP_K,
                                                                          g->raw_window,
                                                                          ratio,
                                                                          DS4_N_HEAD,
                                                                          DS4_N_HEAD_DIM) != 0;
                if (ok && index_stage_profile) {
                    ok = rocm_graph_indexer_stage_profile_boundary("attention",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
            }
            if (ok) batch_attention_done = true;
        }
        if (ok && zero_prefix && !topk_prefill_needed && n_comp != 0) {
            ok = ds4_gpu_attention_prefill_static_mixed_heads_tensor(g->batch_heads,
                                                                       model->map,
                                                                       model->size,
                                                                       layer->attn_sinks->abs_offset,
                                                                       g->batch_q,
                                                                       g->batch_kv,
                                                                       g->layer_attn_comp_cache[il],
                                                                       rocm_graph_attn_comp_cache_is_f16(),
                                                                       n_tokens,
                                                                       n_comp,
                                                                       g->raw_window,
                                                                       ratio,
                                                                       DS4_N_HEAD,
                                                                       DS4_N_HEAD_DIM) != 0;
            if (ok) batch_attention_done = true;
        }
    }

    if (ok && !raw_batch_attention && !batch_attention_done) {
        uint32_t raw_prefix_tokens = 0;
        if (zero_prefix && ratio != 0 && n_tokens <= g->raw_cap && comp_counts != NULL) {
            while (raw_prefix_tokens < n_tokens && comp_counts[raw_prefix_tokens] == 0u) {
                raw_prefix_tokens++;
            }
        }

        if (raw_prefix_tokens != 0) {
            ok = ds4_gpu_attention_prefill_raw_heads_tensor(g->batch_heads,
                                                              model->map,
                                                              model->size,
                                                              layer->attn_sinks->abs_offset,
                                                              g->batch_q,
                                                              g->batch_kv,
                                                              raw_prefix_tokens,
                                                              g->raw_window,
                                                              DS4_N_HEAD,
                                                              DS4_N_HEAD_DIM) != 0;
        }
        if (raw_prefix_tokens < n_tokens) {
            for (uint32_t t = raw_prefix_tokens; ok && t < n_tokens; t++) {
                const uint32_t pos = pos0 + t;
                const uint32_t n_raw = rocm_graph_raw_span_for_batch(g, pos, 1);
                const uint32_t raw_start = rocm_graph_raw_start_for_span(g, pos, n_raw);
                const uint32_t cur_comp = comp_counts ? comp_counts[t] : 0u;
                const uint32_t cur_index = index_counts ? index_counts[t] : 0u;
                uint32_t n_selected = 0;
                ds4_gpu_tensor *comp_mask = NULL;

                if (ratio == 4 && cur_comp > DS4_N_INDEXER_TOP_K) {
                    const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    ds4_gpu_tensor *indexer_q_view = rocm_graph_tensor_row_view(
                            g->batch_indexer_q, t, (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM);
                    ds4_gpu_tensor *indexer_w_view = rocm_graph_tensor_row_view(
                            g->batch_indexer_weights, t, DS4_N_INDEXER_HEAD);
                    ok = indexer_q_view && indexer_w_view &&
                         ds4_gpu_indexer_score_one_tensor(g->indexer_scores,
                                                            indexer_q_view,
                                                            indexer_w_view,
                                                            g->layer_index_comp_cache[il],
                                                            cur_index,
                                                            DS4_N_INDEXER_HEAD,
                                                            DS4_N_INDEXER_HEAD_DIM,
                                                            index_scale) != 0 &&
                         ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                       g->indexer_scores,
                                                       cur_index,
                                                       1,
                                                       DS4_N_INDEXER_TOP_K) != 0 &&
                         ds4_gpu_dsv4_topk_mask_tensor(g->comp_mask,
                                                         g->comp_selected,
                                                         cur_index,
                                                         1,
                                                         DS4_N_INDEXER_TOP_K) != 0;
                    ds4_gpu_tensor_free(indexer_w_view);
                    ds4_gpu_tensor_free(indexer_q_view);
                    if (ok) {
                        comp_mask = g->comp_mask;
                        n_selected = DS4_N_INDEXER_TOP_K < cur_index
                            ? DS4_N_INDEXER_TOP_K
                            : cur_index;
                    }
                }

                ds4_gpu_tensor *q_view = rocm_graph_tensor_row_view(g->batch_q, t, q_dim);
                ds4_gpu_tensor *kv_cache_view = rocm_graph_tensor_row_view(g->batch_kv, t, DS4_N_HEAD_DIM);
                ds4_gpu_tensor *heads_view = rocm_graph_tensor_row_view(g->batch_heads, t, q_dim);
                ok = ok && q_view && kv_cache_view && heads_view;
                if (ok && !zero_prefix) {
                    ok = ds4_gpu_store_raw_kv_tensor(g->layer_raw_cache[il],
                                                       kv_cache_view,
                                                       g->raw_cap,
                                                       pos % g->raw_cap,
                                                       DS4_N_HEAD_DIM) != 0;
                }
                if (ok && comp_mask != NULL && n_selected != 0) {
                    ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(heads_view,
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              q_view,
                                                                              g->layer_raw_cache[il],
                                                                              g->layer_attn_comp_cache[il],
                                                                              rocm_graph_attn_comp_cache_is_f16(),
                                                                              g->comp_selected,
                                                                              1,
                                                                              pos,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              cur_comp,
                                                                              n_selected,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              DS4_N_HEAD,
                                                                              DS4_N_HEAD_DIM) != 0;
                } else if (ok) {
                    ok = ds4_gpu_attention_decode_heads_tensor(heads_view,
                                                                 model->map,
                                                                 model->size,
                                                                 layer->attn_sinks->abs_offset,
                                                                 q_view,
                                                                 g->layer_raw_cache[il],
                                                                 n_raw,
                                                                 g->raw_cap,
                                                                 raw_start,
                                                                 cur_comp ? g->layer_attn_comp_cache[il] : NULL,
                                                                 rocm_graph_attn_comp_cache_is_f16(),
                                                                 cur_comp,
                                                                 comp_mask,
                                                                 n_selected,
                                                                 DS4_N_HEAD,
                                                                 DS4_N_HEAD_DIM) != 0;
                }
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(kv_cache_view);
                ds4_gpu_tensor_free(q_view);
            }
        }
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("attention");

    if (ok) {
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_heads,
                                            n_tokens,
                                            DS4_N_HEAD,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("inv_rope");
    if (ok) {
        ok = ds4_gpu_attention_output_q8_batch_tensor(g->batch_attn_out,
                                                        g->batch_attn_low,
                                                        g->batch_group_tmp,
                                                        g->batch_low_tmp,
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_a->abs_offset,
                                                        layer->attn_output_b->abs_offset,
                                                        group_dim,
                                                        rank,
                                                        n_groups,
                                                        DS4_N_EMBD,
                                                        g->batch_heads,
                                                        n_tokens) != 0;
    }
    if (ok) {
    }
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("output_proj");
    if (ok) ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                                  g->batch_attn_out,
                                                  g->batch_cur_hc,
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("hc_post");
    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    free(index_counts);
    free(comp_counts);
#undef GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE
#undef GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE
    return ok;
}

/* Encode the batched prefill FFN half: HC pre/norm, shared expert, routed
 * experts, sum, and HC post. */
static bool rocm_graph_encode_layer_ffn_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const uint64_t gate_row_bytes = ds4_routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = ds4_routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    const bool layer_stage_profile = getenv("GUFO_DEEPSEEK_ROCM_LAYER_STAGE_PROFILE") != NULL;
    double layer_stage_t0 = layer_stage_profile ? ds4_now_seconds() : 0.0;
#define GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = rocm_graph_layer_stage_profile_boundary("ffn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)

    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *ffn_cur_view = ds4_gpu_tensor_view(
            g->batch_ffn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *next_hc_view = ds4_gpu_tensor_view(
            g->batch_next_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && ffn_cur_view && next_hc_view;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_after_attn_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(hc_mix_view,
                                             model->map,
                                             model->size,
                                             layer->hc_ffn_fn->abs_offset,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (rocm_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_ffn_scale->abs_offset,
                                                        layer->hc_ffn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(ffn_cur_view,
                                                            g->batch_after_attn_hc,
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(ffn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            g->batch_after_attn_hc,
                                                            model->map,
                                                            model->size,
                                                            layer->hc_ffn_scale->abs_offset,
                                                            layer->hc_ffn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
    }
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("hc_pre");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_ffn_norm,
                                                       g->batch_ffn_cur,
                                                       model->map,
                                                       model->size,
                                                       layer->ffn_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("norm");
    if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_router_logits,
                                             model->map,
                                             model->size,
                                             layer->ffn_gate_inp->abs_offset,
                                             DS4_N_EMBD,
                                             DS4_N_EXPERT,
                                             g->batch_ffn_norm,
                                             n_tokens) != 0;

    if (ok) ok = ds4_gpu_router_select_batch_tensor(g->batch_router_selected,
                                                      g->batch_router_weights,
                                                      g->batch_router_probs,
                                                      model->map,
                                                      model->size,
                                                      layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                      0,
                                                      0,
                                                      layer->ffn_exp_probs_b != NULL,
                                                      layer->ffn_gate_tid2eid != NULL,
                                                      g->batch_router_logits,
                                                      g->prefill_tokens,
                                                      DS4_N_EXPERT,
                                                      DS4_N_EXPERT_USED,
                                                      DS4_EXPERT_WEIGHT_SCALE,
                                                      n_tokens) != 0;
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("router");

    if (ok) {
        ok = ds4_gpu_routed_moe_batch_tensor(g->batch_routed_out,
                                               g->batch_routed_gate,
                                               g->batch_routed_up,
                                               g->batch_routed_mid,
                                               g->batch_routed_down,
                                               model->map,
                                               model->size,
                                               layer->ffn_gate_exps->abs_offset,
                                               layer->ffn_up_exps->abs_offset,
                                               layer->ffn_down_exps->abs_offset,
                                               layer->ffn_gate_exps->type,
                                               layer->ffn_down_exps->type,
                                               gate_expert_bytes,
                                               gate_row_bytes,
                                               down_expert_bytes,
                                               down_row_bytes,
                                               (uint32_t)expert_in_dim,
                                               (uint32_t)down_in_dim,
                                               (uint32_t)routed_out_dim,
                                               g->batch_router_selected,
                                               g->batch_router_weights,
                                               DS4_N_EXPERT,
                                               DS4_N_EXPERT_USED,
                                               DS4_SWIGLU_CLAMP_EXP,
                                               g->batch_ffn_norm,
                                               il,
                                               n_tokens,
                                               &g->batch_routed_mid_is_f16,
                                               false) != 0;
    }
    if (ok) {
    }
    if (ok) {
        const uint64_t routed_mid_elems = (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim;
        if (g->batch_routed_mid_is_f16) {
        } else {
        }
    }
    if (ok) {
    }
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("routed_moe");
    if (ok) ok = rocm_graph_matmul_q8_0_named_tensor("shared_gate",
                                                      il,
                                                      pos0,
                                                      g->batch_shared_gate,
                                                      model,
                                                      layer->ffn_gate_shexp,
                                                      DS4_N_EMBD,
                                                      shared_dim,
                                                      g->batch_ffn_norm,
                                                      n_tokens);
    if (ok) ok = rocm_graph_matmul_q8_0_named_tensor("shared_up",
                                                      il,
                                                      pos0,
                                                      g->batch_shared_up,
                                                      model,
                                                      layer->ffn_up_shexp,
                                                      DS4_N_EMBD,
                                                      shared_dim,
                                                      g->batch_ffn_norm,
                                                      n_tokens);
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("shared_gate_up");
    if (ok) ok = ds4_gpu_swiglu_tensor(g->batch_shared_mid,
                                         g->batch_shared_gate,
                                         g->batch_shared_up,
                                         (uint32_t)((uint64_t)n_tokens * shared_dim),
                                         DS4_SWIGLU_CLAMP_EXP,
                                         1.0f) != 0;
    if (ok) ok = rocm_graph_matmul_q8_0_named_tensor("shared_down",
                                                      il,
                                                      pos0,
                                                      g->batch_shared_out,
                                                      model,
                                                      layer->ffn_down_shexp,
                                                      shared_dim,
                                                      DS4_N_EMBD,
                                                      g->batch_shared_mid,
                                                      n_tokens);
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("shared_down");
    if (ok) {
    }

    if (ok) {
        ok = ds4_gpu_hc_expand_add_split_tensor(next_hc_view,
                                                  g->batch_routed_out,
                                                  g->batch_shared_out,
                                                  g->batch_after_attn_hc,
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("hc_post");
    ds4_gpu_tensor_free(next_hc_view);
    ds4_gpu_tensor_free(ffn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
#undef GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE
    return ok;
}

/* Encode one complete layer for prefill by chaining attention and FFN batches. */
static bool rocm_graph_encode_layer_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    bool ok = rocm_graph_encode_layer_attention_batch(g, model, layer, il, pos0, n_tokens);
    if (ok) ok = rocm_graph_encode_layer_ffn_batch(g, model, layer, il, pos0, n_tokens);
    if (ok) {
        ds4_gpu_tensor *tmp = g->batch_cur_hc;
        g->batch_cur_hc = g->batch_next_hc;
        g->batch_next_hc = tmp;
    }
    return ok;
}

/* Execute one ROCm decode token and read back logits. */
static bool rocm_graph_eval_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits) {
    const bool profile = getenv("GUFO_DEEPSEEK_ROCM_GRAPH_TOKEN_PROFILE") != NULL;
    const bool throttle = graph_power_throttle_enabled(g);
    const double t0 = (profile || throttle) ? ds4_now_seconds() : 0.0;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = rocm_graph_encode_token_raw_swa(g, model, weights, token, pos, logits != NULL, true);
    const double t_encoded = (profile || throttle) ? ds4_now_seconds() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    const double t_done = (profile || throttle) ? ds4_now_seconds() : 0.0;

    if (ok && logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    const double t_read = (profile || throttle) ? ds4_now_seconds() : 0.0;
    if (profile) {
        fprintf(stderr,
                "ds4: ROCm graph token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                pos,
                (t_encoded - t0) * 1000.0,
                (t_done - t_encoded) * 1000.0,
                (t_read - t_done) * 1000.0,
                (t_read - t0) * 1000.0,
                logits != NULL);
    }
    if (ok) graph_power_note_decode_token(g, t_read - t0);
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: ROCm synchronize after graph eval failure also failed\n");
        }
    }
    return ok;
}

/* Greedy verifier helper.  Speculative decoding only needs the target model's
 * top token after most accepted draft rows; the full vocabulary row is needed
 * once, for the final committed state that normal sampling will continue from.
 * Keeping intermediate rows device-resident avoids turning verification into a
 * sequence of large CPU readbacks. */
static bool rocm_graph_reset_prefill_state(ds4_gpu_graph *g) {
    memset(g->layer_n_comp, 0, sizeof(g->layer_n_comp));
    memset(g->layer_n_index_comp, 0, sizeof(g->layer_n_index_comp));
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
        const uint64_t attn_rows = (uint64_t)coff * ratio;
        if (!rocm_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows)) return false;
        if (!rocm_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows)) return false;
        if (ratio == 4) {
            const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_rows = (uint64_t)coff * ratio;
            if (!rocm_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows)) return false;
            if (!rocm_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows)) return false;
        }
    }
    return true;
}

/* Execute ROCm prefill in layer-major order so intermediate activations stay
 * on the GPU and cache state is built exactly once. */
static bool rocm_graph_prefill_layer_major(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;

    bool ok = rocm_graph_upload_prompt_tokens(g->prefill_tokens, prompt, start, n_tokens);
    if (!ok) return false;

    if (!rocm_graph_warmup_prefill_kernels(g, model, weights, n_tokens)) return false;

    const bool split_profile = getenv("GUFO_DEEPSEEK_ROCM_GRAPH_PREFILL_SPLIT_PROFILE") != NULL;
    /*
     * A full long-prompt prefill can keep the GPU busy long enough for macOS
     * to watchdog the desktop. Split long prefills into completed command
     * buffers that also serve as scheduling/keepalive points.
     */
    const bool throttle = graph_power_throttle_enabled(g);
    const bool split_commands = split_profile || throttle || n_tokens > 2048;
    const bool profile = getenv("GUFO_DEEPSEEK_ROCM_GRAPH_PREFILL_PROFILE") != NULL || split_profile;
    const double t0 = profile ? ds4_now_seconds() : 0.0;
    double encode_s = 0.0;
    double execute_s = 0.0;

    if (!split_commands) {
        ok = rocm_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                     g->prefill_tokens,
                                                     model,
                                                     weights,
                                                     prompt,
                                                     start,
                                                     n_tokens);
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = rocm_graph_encode_layer_batch(g,
                                                model,
                                                &weights->layer[il],
                                                il,
                                                start,
                                                n_tokens);
            if (show_progress) {
                fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
                fflush(stderr);
            }
        }
        if (show_progress) fputc('\n', stderr);
        const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
        uint32_t output_row = (uint32_t)n_tokens - 1u;
        const char *output_row_env = getenv("GUFO_DEEPSEEK_ROCM_GRAPH_OUTPUT_ROW");
        if (output_row_env && output_row_env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(output_row_env, &end, 10);
            if (end != output_row_env && v < (unsigned long)n_tokens) {
                output_row = (uint32_t)v;
            }
        }
        ds4_gpu_tensor *saved_cur = g->cur_hc;
        ds4_gpu_tensor *last_hc = NULL;
        if (ok && logits) {
            last_hc = rocm_graph_tensor_row_view(g->batch_cur_hc, output_row, hc_dim);
            ok = last_hc != NULL;
        }
        if (ok && logits) {
            g->cur_hc = last_hc;
            ok = rocm_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
            g->cur_hc = saved_cur;
        }

        const double t_encoded = profile ? ds4_now_seconds() : 0.0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double t_done = profile ? ds4_now_seconds() : 0.0;
        g->cur_hc = saved_cur;
        if (last_hc) ds4_gpu_tensor_free(last_hc);
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: ROCm synchronize after whole-prefill graph failure also failed\n");
            }
            return false;
        }

        const double t_before_read = profile ? ds4_now_seconds() : 0.0;
        if (logits) {
            ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        if (profile) {
            const double t_read = ds4_now_seconds();
            fprintf(stderr,
                    "ds4: gpu graph prefill total tokens=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                    n_tokens,
                    (t_encoded - t0) * 1000.0,
                    (t_done - t_encoded) * 1000.0,
                    (t_read - t_before_read) * 1000.0,
                    (t_read - t0) * 1000.0);
        }
        return ok;
    }

    double t_layer0 = (profile || throttle) ? ds4_now_seconds() : 0.0;
    ok = rocm_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                 g->prefill_tokens,
                                                 model,
                                                 weights,
                                                 prompt,
                                                 start,
                                                 n_tokens);
    const double t_embed_encoded = (profile || throttle) ? ds4_now_seconds() : 0.0;
    const double t_embed_done = (profile || throttle) ? ds4_now_seconds() : 0.0;
    if (profile) {
        encode_s += t_embed_encoded - t_layer0;
        execute_s += t_embed_done - t_embed_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "ds4: ROCm layer-major prefill embed encode=%.3f ms execute=%.3f ms\n",
                    (t_embed_encoded - t_layer0) * 1000.0,
                    (t_embed_done - t_embed_encoded) * 1000.0);
        }
    }
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: ROCm synchronize after layer-major prefill embed failure also failed\n");
        }
        return false;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        double layer_elapsed = 0.0;
        if (split_profile) {
            const double t_attn0 = ds4_now_seconds();
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = rocm_graph_encode_layer_attention_batch(g,
                                                                  model,
                                                                  &weights->layer[il],
                                                                  il,
                                                                  start,
                                                                  n_tokens);
            const double t_attn_encoded = ds4_now_seconds();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_attn_done = ds4_now_seconds();

            const double t_ffn0 = ds4_now_seconds();
            if (ok) ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = rocm_graph_encode_layer_ffn_batch(g,
                                                            model,
                                                            &weights->layer[il],
                                                            il,
                                                            start,
                                                            n_tokens);
            if (ok) {
                ds4_gpu_tensor *tmp = g->batch_cur_hc;
                g->batch_cur_hc = g->batch_next_hc;
                g->batch_next_hc = tmp;
            }
            const double t_ffn_encoded = ds4_now_seconds();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_ffn_done = ds4_now_seconds();
            layer_elapsed = (t_attn_done - t_attn0) + (t_ffn_done - t_ffn0);

            encode_s += (t_attn_encoded - t_attn0) + (t_ffn_encoded - t_ffn0);
            execute_s += (t_attn_done - t_attn_encoded) + (t_ffn_done - t_ffn_encoded);
            fprintf(stderr,
                    "ds4: ROCm layer-major prefill layer %u attn encode=%.3f execute=%.3f ms ffn encode=%.3f execute=%.3f ms\n",
                    il,
                    (t_attn_encoded - t_attn0) * 1000.0,
                    (t_attn_done - t_attn_encoded) * 1000.0,
                    (t_ffn_encoded - t_ffn0) * 1000.0,
                    (t_ffn_done - t_ffn_encoded) * 1000.0);
        } else {
            const double t_chunk0 = (profile || throttle) ? ds4_now_seconds() : 0.0;
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = rocm_graph_encode_layer_batch(g,
                                                        model,
                                                        &weights->layer[il],
                                                        il,
                                                        start,
                                                        n_tokens);
            const double t_encoded = (profile || throttle) ? ds4_now_seconds() : 0.0;
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_done = (profile || throttle) ? ds4_now_seconds() : 0.0;
            layer_elapsed = t_done - t_chunk0;
            if (profile) {
                encode_s += t_encoded - t_chunk0;
                execute_s += t_done - t_encoded;
                fprintf(stderr,
                        "ds4: gpu layer-major prefill layer %u encode=%.3f ms execute=%.3f ms\n",
                        il,
                        (t_encoded - t_chunk0) * 1000.0,
                        (t_done - t_encoded) * 1000.0);
            }
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: ROCm synchronize after layer-major prefill failure also failed\n");
            }
            return false;
        }
        graph_power_note_prefill_layer(g, il, layer_elapsed);
        if (show_progress) {
            fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
            fflush(stderr);
        }
    }
    if (show_progress) fputc('\n', stderr);

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    uint32_t output_row = (uint32_t)n_tokens - 1u;
    const char *output_row_env = getenv("GUFO_DEEPSEEK_ROCM_GRAPH_OUTPUT_ROW");
    if (output_row_env && output_row_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(output_row_env, &end, 10);
        if (end != output_row_env && v < (unsigned long)n_tokens) {
            output_row = (uint32_t)v;
        }
    }
    ds4_gpu_tensor *saved_cur = g->cur_hc;
    ds4_gpu_tensor *last_hc = NULL;

    const double t_head0 = profile ? ds4_now_seconds() : 0.0;
    if (logits) {
        last_hc = rocm_graph_tensor_row_view(g->batch_cur_hc,
                                              output_row,
                                              hc_dim);
        ok = last_hc != NULL;
    }
    if (ok && logits) {
        g->cur_hc = last_hc;
        ok = ds4_gpu_begin_commands() != 0;
    }
    if (ok && logits) ok = rocm_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    const double t_head_encoded = profile ? ds4_now_seconds() : 0.0;
    if (ok && logits) ok = ds4_gpu_end_commands() != 0;
    const double t_head_done = profile ? ds4_now_seconds() : 0.0;
    g->cur_hc = saved_cur;
    if (last_hc) ds4_gpu_tensor_free(last_hc);
    if (!ok) return false;

    const double t_before_read = profile ? ds4_now_seconds() : 0.0;
    if (logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (profile) {
        const double t_read = ds4_now_seconds();
        encode_s += t_head_encoded - t_head0;
        execute_s += t_head_done - t_head_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "ds4: gpu layer-major prefill head encode=%.3f ms execute=%.3f ms\n",
                    (t_head_encoded - t_head0) * 1000.0,
                    (t_head_done - t_head_encoded) * 1000.0);
        }
        fprintf(stderr,
                "ds4: gpu layer-major prefill total tokens=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                n_tokens,
                encode_s * 1000.0,
                execute_s * 1000.0,
                (t_read - t_before_read) * 1000.0,
                (t_read - t0) * 1000.0);
    }
    return ok;
}

static bool rocm_graph_prefill_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress) {
    if (n_tokens <= 0 || n_tokens > prompt->len) return false;
    if ((uint32_t)n_tokens > g->prefill_cap) return false;
    return rocm_graph_prefill_layer_major(g,
                                           model,
                                           weights,
                                           prompt,
                                           0,
                                           (uint32_t)n_tokens,
                                           logits,
                                           show_progress);
}

/* Prefill a contiguous token range in fixed-size chunks.
 *
 * The common case starts at token zero, but server sessions also use this to
 * extend an existing KV cache with a long suffix.  Resumed chunks are aligned
 * to the same absolute prefill-cap boundaries used by a cold full prompt, so
 * compression windows and row finalization follow the same schedule after the
 * cached prefix.
 */
static bool rocm_graph_prefill_chunked_range(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress) {
    if (n_tokens == 0 || g->prefill_cap == 0) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;

    uint32_t chunk_cap = g->prefill_cap;
    if (start != 0 && chunk_cap > g->raw_cap) chunk_cap = g->raw_cap;
    if (chunk_cap == 0) return false;

    const bool profile = getenv("GUFO_DEEPSEEK_ROCM_GRAPH_PREFILL_PROFILE") != NULL;
    const double t0 = profile ? ds4_now_seconds() : 0.0;
    const uint32_t end = start + n_tokens;

    for (uint32_t pos0 = start; pos0 < end; ) {
        const uint32_t remaining = end - pos0;
        uint32_t local_cap = chunk_cap;
        if (start != 0 && g->prefill_cap != 0) {
            const uint32_t mod = pos0 % g->prefill_cap;
            if (mod != 0) {
                const uint32_t to_boundary = g->prefill_cap - mod;
                if (to_boundary < local_cap) local_cap = to_boundary;
            }
        }
        const uint32_t chunk = remaining < local_cap ? remaining : local_cap;
        const uint32_t chunk_end = pos0 + chunk;
        float *chunk_logits = chunk_end == end ? logits : nullptr;
        bool ok = rocm_graph_prefill_layer_major(g,
                                                  model,
                                                  weights,
                                                  prompt,
                                                  pos0,
                                                  chunk,
                                                  chunk_logits,
                                                  show_progress);
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: ROCm synchronize after chunked prefill failure also failed\n");
            }
            return false;
        }
        pos0 = chunk_end;
    }
    if (show_progress) fputc('\n', stderr);
    if (profile) {
        const double t_read = ds4_now_seconds();
        fprintf(stderr,
                "ds4: gpu chunked prefill start=%u tokens=%u chunk=%u total=%.3f ms\n",
                start,
                n_tokens,
                chunk_cap,
                (t_read - t0) * 1000.0);
    }
    return true;
}

/* Long prompts are prefetched in fixed-size chunks.  Chunks bound transient
 * attention buffers while preserving the same final KV/cache state. */
static bool rocm_graph_prefill_chunked(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress) {
    if (n_tokens <= 0) return false;
    return rocm_graph_prefill_chunked_range(g,
                                             model,
                                             weights,
                                             prompt,
                                             0,
                                             (uint32_t)n_tokens,
                                             logits,
                                             show_progress);
}

/* Pick a raw SWA cache size for ROCm.  During batched prefill it must cover
 * the previous window plus the current ubatch. */
static uint32_t rocm_graph_raw_cap_for_context(int ctx_size, uint32_t prefill_cap) {
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > (uint32_t)ctx_size) raw_window = (uint32_t)ctx_size;
    if (raw_window == 0) raw_window = 1;

    /*
     * During batched prefill the SWA cache must hold the current ubatch plus
     * the previous logical window. The cache is padded to a 256-row multiple
     * so the physical row order and FlashAttention block grouping match the
     * model path we compare against.
     */
    uint64_t wanted = (uint64_t)raw_window + prefill_cap;
    if (wanted > (uint32_t)ctx_size) wanted = (uint32_t)ctx_size;
    if (wanted == 0) wanted = 1;
    wanted = ds4_align_up(wanted, 256u);
    if (wanted > 8192u) wanted = 8192u;
    uint32_t raw_cap = (uint32_t)wanted;
    if (raw_cap < raw_window) raw_cap = raw_window;

    const char *env = getenv("GUFO_DEEPSEEK_ROCM_GRAPH_RAW_CAP");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env && v > 0) {
            raw_cap = (uint32_t)v;
            if (raw_cap > (uint32_t)ctx_size) raw_cap = (uint32_t)ctx_size;
            if (raw_cap > 8192u) raw_cap = 8192u;
            if (raw_cap < raw_window) raw_cap = raw_window;
        }
    }

    return raw_cap;
}

/* Choose the prefill ubatch size.  Whole-batch is fastest for normal prompts;
 * long prompts default to 4096-token chunks. */
static uint32_t rocm_graph_prefill_cap_for_prompt(int prompt_len) {
    return ds4_default_prefill_cap_for_prompt(prompt_len);
}

/* Extend shared prefixes with batched prefill once the suffix is large enough
 * to amortize batch setup. */
static uint32_t rocm_graph_resume_prefill_min_tokens(void) {
    const char *env = getenv("GUFO_DEEPSEEK_ROCM_RESUME_PREFILL_MIN");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env) {
            if (v <= 0) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
    return 4u;
}

uint32_t ds4_rocm_graph_prefill_capacity(int context_size) {
    return rocm_graph_prefill_cap_for_prompt(context_size);
}

uint32_t ds4_rocm_graph_resume_prefill_min_tokens(void) {
    return rocm_graph_resume_prefill_min_tokens();
}

ds4_rocm_graph *ds4_rocm_graph_create(ds4_engine *engine,
                                      int context_size,
                                      uint32_t prefill_capacity) {
    if (!engine || !engine->weights || context_size <= 0 ||
        prefill_capacity == 0) {
        return NULL;
    }

    auto *graph =
        static_cast<ds4_rocm_graph *>(ds4_xcalloc(1, sizeof(ds4_rocm_graph)));
    const uint32_t raw_capacity =
        rocm_graph_raw_cap_for_context(context_size, prefill_capacity);
    if (!rocm_graph_alloc_raw_cap(graph,
                                   engine->weights,
                                   &engine->weights->layer[0],
                                   raw_capacity,
                                   (uint32_t)context_size,
                                   prefill_capacity)) {
        free(graph);
        return NULL;
    }
    graph->power_percent = (uint32_t)engine->power_percent;
    return graph;
}

void ds4_rocm_graph_destroy(ds4_rocm_graph *graph) {
    if (!graph) return;
    rocm_graph_free(graph);
    free(graph);
}

bool ds4_rocm_graph_reset(ds4_rocm_graph *graph) {
    return graph && rocm_graph_reset_prefill_state(graph);
}

bool ds4_rocm_graph_prefill(ds4_rocm_graph *graph,
                            ds4_engine *engine,
                            const ds4_tokens *prompt,
                            float *logits) {
    if (!graph || !engine || !prompt || prompt->len <= 0) return false;
    if (graph->prefill_cap < (uint32_t)prompt->len) {
        return rocm_graph_prefill_chunked(graph,
                                           engine->model,
                                           engine->weights,
                                           prompt,
                                           prompt->len,
                                           logits,
                                           false);
    }
    return rocm_graph_prefill_raw_swa(graph,
                                       engine->model,
                                       engine->weights,
                                       prompt,
                                       prompt->len,
                                       logits,
                                       false);
}

bool ds4_rocm_graph_prefill_range(ds4_rocm_graph *graph,
                                  ds4_engine *engine,
                                  const ds4_tokens *prompt,
                                  uint32_t start,
                                  uint32_t token_count,
                                  float *logits) {
    return graph && engine && prompt &&
           rocm_graph_prefill_chunked_range(graph,
                                             engine->model,
                                             engine->weights,
                                             prompt,
                                             start,
                                             token_count,
                                             logits,
                                             false);
}

bool ds4_rocm_graph_eval(ds4_rocm_graph *graph,
                         ds4_engine *engine,
                         int token,
                         uint32_t position,
                         float *logits) {
    return graph && engine &&
           rocm_graph_eval_token_raw_swa(graph,
                                          engine->model,
                                          engine->weights,
                                          token,
                                          position,
                                          logits);
}

/* =========================================================================
 * Session Snapshot Payloads.
 * =========================================================================
 *
 * The server disk cache stores a high-level file header, then delegates the
 * graph-specific payload below to the engine.  This payload is intentionally
 * not mmaped: restoring a checkpoint copies bytes back into the already
 * allocated ROCm tensors, preserving the same live graph buffers used by
 * normal prefill/decode.  The raw SWA cache is serialized as the last logical
 * window only; suffix prefill writes its own raw rows before attention.  The
 * compressed caches are serialized up to their live row counts because sparse
 * attention may select rows from the whole prefix.
 *
 * The payload is model-specific rather than self-describing.  The fixed header
 * records enough shape information to reject a file written for a different
 * DS4 runtime, then the body writes: checkpoint tokens, last logits, per-layer
 * compressed row counts, raw SWA rows in logical order, compressed attention
 * rows, and the compressor/indexer frontiers.  That is the minimum state needed
 * for the next token to match a session that had just prefetched the prefix.
 */

#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u
#define DS4_SESSION_IO_CHUNK (8u * 1024u * 1024u)

static void payload_set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}

static void payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)(v);
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}

static uint32_t payload_get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static int payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const auto *p = static_cast<const uint8_t *>(ptr);
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to write session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}

static int payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    const uint64_t original = bytes;
    auto *p = static_cast<uint8_t *>(ptr);
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to read session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}

static int payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4];
    payload_put_u32(b, v);
    return payload_write_bytes(fp, b, sizeof(b), err, errlen);
}

static int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (remaining && *remaining < sizeof(b)) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        payload_set_err(err, errlen, "failed to read session payload");
        return 1;
    }
    if (remaining) *remaining -= sizeof(b);
    *v = payload_get_u32(b);
    return 0;
}

static uint64_t layer_attn_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
}

static uint64_t layer_index_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
}

/* Only the last logical sliding-window rows are needed from the raw cache.
 * The physical ROCm tensor is a ring sized for ubatches, but after restore
 * the next suffix chunk will write its own raw rows before any attention read.
 * Compressed rows are different: sparse attention can select any row from the
 * prefix, so those are persisted up to their live row counts. */
static uint32_t session_raw_live_rows(const ds4_gpu_graph *g, uint32_t checkpoint_len) {
    uint32_t rows = g->raw_window ? g->raw_window : DS4_N_SWA;
    if (rows > g->raw_cap) rows = g->raw_cap;
    if (rows > checkpoint_len) rows = checkpoint_len;
    return rows;
}

/* Return the exact engine-owned payload size, excluding the server's KVC file
 * header and observability text.  This is deliberately based on live row counts
 * rather than capacities so the disk cache scales with saved tokens, not with
 * the maximum context size used to allocate the graph. */
static uint64_t session_payload_live_tensor_bytes(const ds4_gpu_graph *g, uint32_t checkpoint_len) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_raw_live_rows(g, checkpoint_len);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float);
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}

/* Accelerator tensors are copied through a fixed-size CPU buffer.  We do not mmap the
 * cache file and we do not allocate a second graph-sized blob just to serialize
 * it; both would be poor fits for this very large model. */
static int payload_write_tensor_span(FILE *fp, const ds4_gpu_tensor *tensor,
                                     uint64_t offset, uint64_t bytes,
                                     uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (ds4_gpu_tensor_read(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to read accelerator session tensor");
            return 1;
        }
        if (payload_write_bytes(fp, buf, n, err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}

static int payload_read_tensor_span(FILE *fp, ds4_gpu_tensor *tensor,
                                    uint64_t offset, uint64_t bytes,
                                    uint8_t *buf, size_t cap, uint64_t *remaining,
                                    char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (payload_read_bytes(fp, buf, n, remaining, err, errlen) != 0) return 1;
        if (ds4_gpu_tensor_write(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to restore accelerator session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}

uint64_t ds4_rocm_graph_snapshot_bytes(const ds4_rocm_graph *graph,
                                       const ds4_tokens *checkpoint) {
    if (!graph || !checkpoint || checkpoint->len <= 0) return 0;
    uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)checkpoint->len * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += session_payload_live_tensor_bytes(graph, (uint32_t)checkpoint->len);
    return bytes;
}

static int rocm_graph_save_payload(const ds4_rocm_graph *graph,
                                   const ds4_tokens *checkpoint,
                                   const float *logits,
                                   uint32_t prefill_capacity,
                                   int context_size,
                                   FILE *fp,
                                   char *err,
                                   size_t errlen) {
    if (!graph || !checkpoint || checkpoint->len <= 0 || !logits || !fp) {
        payload_set_err(err, errlen, "invalid graph snapshot save");
        return 1;
    }
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before snapshot");
        return 1;
    }

    const uint32_t raw_live =
        session_raw_live_rows(graph, (uint32_t)checkpoint->len);
    /* Header fields:
     *   0 magic, 1 version, 2 ctx, 3 prefill chunk, 4 raw cap,
     *   5 raw window, 6 compressed cap, 7 token count,
     *   8 layers, 9 raw head dim, 10 indexer head dim, 11 vocab,
     *   12 live raw rows serialized below.
     */
    uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC,
        DS4_SESSION_PAYLOAD_VERSION,
        (uint32_t)context_size,
        prefill_capacity,
        graph->raw_cap,
        graph->raw_window,
        graph->comp_cap,
        (uint32_t)checkpoint->len,
        DS4_N_LAYER,
        DS4_N_HEAD_DIM,
        DS4_N_INDEXER_HEAD_DIM,
        DS4_N_VOCAB,
        raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < checkpoint->len; i++) {
        if (payload_write_u32(fp,
                              (uint32_t)checkpoint->v[i],
                              err,
                              errlen) != 0) {
            return 1;
        }
    }
    if (payload_write_bytes(fp,
                            logits,
                            (uint64_t)DS4_N_VOCAB * sizeof(float),
                            err,
                            errlen) != 0) {
        return 1;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp,
                              graph->layer_n_comp[il],
                              err,
                              errlen) != 0) {
            return 1;
        }
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp,
                              graph->layer_n_index_comp[il],
                              err,
                              errlen) != 0) {
            return 1;
        }
    }

    auto *buf =
        static_cast<uint8_t *>(ds4_xmalloc(DS4_SESSION_IO_CHUNK));
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        /* Write the raw ring in logical position order.  The file does not care
         * where the rows happened to live physically in the source graph. */
        const uint32_t raw_first = (uint32_t)checkpoint->len - raw_live;
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % graph->raw_cap;
            rc = payload_write_tensor_span(fp,
                                           graph->layer_raw_cache[il],
                                           (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                           (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        /* Compressed rows are append-only from row zero, so the live prefix is
         * contiguous.  The two compressor state tensors hold the partial window
         * that will become the next compressed row. */
        rc = payload_write_tensor_span(fp,
                                       graph->layer_attn_comp_cache[il],
                                       0,
                                       (uint64_t)graph->layer_n_comp[il] *
                                           DS4_N_HEAD_DIM * sizeof(float),
                                       buf,
                                       DS4_SESSION_IO_CHUNK,
                                       err,
                                       errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    graph->layer_attn_state_kv[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    graph->layer_attn_state_score[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_write_tensor_span(fp,
                                           graph->layer_index_comp_cache[il],
                                           0,
                                           (uint64_t)graph->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        graph->layer_index_state_kv[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        graph->layer_index_state_score[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
        }
    }
    free(buf);
    return rc;
}

static int rocm_graph_load_payload(ds4_rocm_graph *graph,
                                   ds4_tokens *checkpoint,
                                   float *logits,
                                   uint32_t prefill_capacity,
                                   int context_size,
                                   FILE *fp,
                                   uint64_t payload_bytes,
                                   char *err,
                                   size_t errlen) {
    if (!graph || !checkpoint || !logits || !fp) {
        payload_set_err(err, errlen, "invalid graph snapshot load");
        return 1;
    }
    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != DS4_SESSION_PAYLOAD_MAGIC || h[1] != DS4_SESSION_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session payload version");
        return 1;
    }
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_raw_live = h[12];
    if (saved_ctx > (uint32_t)context_size ||
        saved_tokens >= (uint32_t)context_size) {
        payload_set_err(err, errlen, "KV checkpoint does not fit current context");
        return 1;
    }
    if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
        h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB)
    {
        payload_set_err(err, errlen, "KV checkpoint was written for a different DS4 layout");
        return 1;
    }
    /* prefill_cap is scratch scheduling capacity, not durable KV layout.
     * Old checkpoints remain valid as long as the raw KV window matches. */
    (void)saved_prefill_cap;
    (void)prefill_capacity;
    if (saved_raw_window != graph->raw_window) {
        payload_set_err(err, errlen, "KV checkpoint graph chunk layout does not match current runtime");
        return 1;
    }
    /* The raw rows in the file are logical rows.  We can restore them into any
     * current ring with enough capacity, but the saved live count must be exactly
     * the last window implied by the saved token count. */
    const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > graph->raw_cap)
    {
        payload_set_err(err, errlen, "KV checkpoint raw ring layout does not match current context");
        return 1;
    }
    if (saved_comp_cap > graph->comp_cap) {
        payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
        return 1;
    }

    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < saved_tokens; i++) {
        uint32_t tok = 0;
        if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
            ds4_tokens_free(&new_checkpoint);
            return 1;
        }
        ds4_tokens_push(&new_checkpoint, (int)tok);
    }
    if (payload_read_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                           &remaining, err, errlen) != 0)
    {
        ds4_tokens_free(&new_checkpoint);
        return 1;
    }
    uint32_t n_comp[DS4_MAX_LAYER];
    uint32_t n_index_comp[DS4_MAX_LAYER];
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
            ds4_tokens_free(&new_checkpoint);
            return 1;
        }
        if (n_comp[il] > saved_comp_cap ||
            n_comp[il] > graph->layer_comp_cap[il]) {
            ds4_tokens_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
            ds4_tokens_free(&new_checkpoint);
            return 1;
        }
        if (n_index_comp[il] > saved_comp_cap ||
            n_index_comp[il] > graph->layer_comp_cap[il]) {
            ds4_tokens_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid indexer row count");
            return 1;
        }
    }

    if (ds4_gpu_synchronize() == 0) {
        ds4_tokens_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator before KV restore");
        return 1;
    }
    auto *buf =
        static_cast<uint8_t *>(ds4_xmalloc(DS4_SESSION_IO_CHUNK));
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        /* Rebuild the physical raw ring expected by the current graph.  This is
         * why the file stores rows in logical order instead of dumping bytes from
         * the old ring layout. */
        const uint32_t raw_first = saved_tokens - saved_raw_live;
        for (uint32_t r = 0; rc == 0 && r < saved_raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % graph->raw_cap;
            rc = payload_read_tensor_span(fp,
                                          graph->layer_raw_cache[il],
                                          (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                          (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        rc = payload_read_tensor_span(fp,
                                      graph->layer_attn_comp_cache[il],
                                      0,
                                      (uint64_t)n_comp[il] *
                                          DS4_N_HEAD_DIM * sizeof(float),
                                      buf,
                                      DS4_SESSION_IO_CHUNK,
                                      &remaining,
                                      err,
                                      errlen);
        if (rc == 0 && ratio == 4 &&
            !rocm_graph_rebuild_attn_comp_mirror(
                graph, il, n_comp[il])) {
            payload_set_err(
                err, errlen, "failed to rebuild FP16 attention cache mirror");
            rc = 1;
        }
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   graph->layer_attn_state_kv[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   graph->layer_attn_state_score[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_read_tensor_span(fp,
                                          graph->layer_index_comp_cache[il],
                                          0,
                                          (uint64_t)n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       graph->layer_index_state_kv[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       graph->layer_index_state_score[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
        }
    }
    free(buf);
    if (rc != 0) {
        ds4_tokens_free(&new_checkpoint);
        return 1;
    }
    if (remaining != 0) {
        ds4_tokens_free(&new_checkpoint);
        payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
        return 1;
    }
    if (ds4_gpu_synchronize() == 0) {
        ds4_tokens_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator after KV restore");
        return 1;
    }

    ds4_tokens_free(checkpoint);
    *checkpoint = new_checkpoint;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        graph->layer_n_comp[il] = n_comp[il];
        graph->layer_n_index_comp[il] = n_index_comp[il];
    }
    return 0;
}

int ds4_rocm_graph_save_snapshot(const ds4_rocm_graph *graph,
                                 const ds4_tokens *checkpoint,
                                 const float *logits,
                                 uint32_t prefill_capacity,
                                 int context_size,
                                 ds4_session_snapshot *snap,
                                 char *err,
                                 size_t errlen) {
    if (!graph || !checkpoint || !logits || !snap) {
        payload_set_err(err, errlen, "invalid graph snapshot save");
        return 1;
    }
    const uint64_t bytes =
        ds4_rocm_graph_snapshot_bytes(graph, checkpoint);
    if (bytes == 0) {
        payload_set_err(err, errlen, "session has no valid checkpoint to snapshot");
        return 1;
    }
    if (bytes > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }
    if (snap->cap < bytes) {
        auto *p = static_cast<uint8_t *>(
            realloc(snap->ptr, static_cast<size_t>(bytes)));
        if (!p) {
            payload_set_err(err, errlen, "out of memory while allocating session snapshot");
            return 1;
        }
        snap->ptr = p;
        snap->cap = bytes;
    }

    FILE *fp = fmemopen(snap->ptr, (size_t)bytes, "wb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot");
        return 1;
    }
    const int rc = rocm_graph_save_payload(graph,
                                           checkpoint,
                                           logits,
                                           prefill_capacity,
                                           context_size,
                                           fp,
                                           err,
                                           errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to finalize memory session snapshot");
        return 1;
    }
    if (rc != 0) return 1;
    snap->len = bytes;
    return 0;
}

int ds4_rocm_graph_load_snapshot(ds4_rocm_graph *graph,
                                 ds4_tokens *checkpoint,
                                 float *logits,
                                 uint32_t prefill_capacity,
                                 int context_size,
                                 const ds4_session_snapshot *snap,
                                 char *err,
                                 size_t errlen) {
    if (!graph || !checkpoint || !logits || !snap || !snap->ptr ||
        snap->len == 0) {
        payload_set_err(err, errlen, "invalid graph snapshot load");
        return 1;
    }
    if (snap->len > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }

    FILE *fp = fmemopen((void *)snap->ptr, (size_t)snap->len, "rb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot restore");
        return 1;
    }
    const int rc = rocm_graph_load_payload(graph,
                                           checkpoint,
                                           logits,
                                           prefill_capacity,
                                           context_size,
                                           fp,
                                           snap->len,
                                           err,
                                           errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close memory session snapshot");
        return 1;
    }
    return rc;
}

#include <vector>
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dspark_internal.h"
#include "native_internal.h"
#include "model_data_internal.h"

#include "../kernels/rocm/resident_api.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DS4_SPEC_PREFIX_SLOTS 4u

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

    /* Long prompts run in 4,096-token chunks. An 8K capacity regressed the 8K
     * prompt (201.80 against 210.21 tok/s) and was noise in a 16K replay, so
     * this is the policy rather than a tunable. */
    if (prompt_len > 4096) {
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
    ds4_gpu_tensor *batch_flat_hc_h;
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
    ds4_gpu_tensor* batch_dspark_verify_cur_hc;
    ds4_gpu_tensor* batch_dspark_verify_after_attn_hc;
    uint32_t batch_dspark_verify_rows_cap;
    bool borrows_batch_workspace;
    bool batch_routed_mid_is_f16;
    uint32_t power_percent;
    double prefill_layer_avg_sec[DS4_MAX_LAYER];
    double decode_token_avg_sec;

    /* =====================================================================
     * DSpark speculative verification state.
     * =====================================================================
     *
     * A verification block runs the drafted suffix through the batched layer
     * path in one command stream, then reads back only the target's top token
     * per row.  Everything here is allocated on first use so a session that
     * never enables DSpark keeps exactly the residency it had before.
     */
    uint32_t spec_rows_cap;
    ds4_gpu_tensor *spec_logits;
    ds4_gpu_tensor *spec_row_tops;

    /*
     * Rollback frontier.
     *
     * Raw KV rows and compressed cache rows are append-only: a rejected block
     * leaves stale rows that the next write overwrites, and the row counters
     * say how much is live.  The compressor frontier tensors are the exception
     * because they are updated in place, so they are the only cache class a
     * rejected block has to restore.
     */
    ds4_gpu_tensor *spec_saved_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_saved_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_saved_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_saved_index_state_score[DS4_MAX_LAYER];
    uint32_t spec_saved_n_comp[DS4_MAX_LAYER];
    uint32_t spec_saved_n_index_comp[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix_index_state_score[DS4_MAX_LAYER];
    uint32_t spec_prefix_n_comp[DS4_SPEC_PREFIX_SLOTS][DS4_MAX_LAYER];
    uint32_t spec_prefix_n_index_comp[DS4_SPEC_PREFIX_SLOTS][DS4_MAX_LAYER];
    bool spec_frontier_valid;
    bool spec_capture_prefixes;

    /* =====================================================================
     * DSpark drafting state.
     * =====================================================================
     *
     * The drafter needs two things the target does not keep: the target's
     * hidden states at the sampled layers, and one KV ring per DSpark stage
     * holding those hidden states projected into that stage's key/value space.
     * Block work tensors are borrowed from the batch_* set because drafting,
     * prefill, and verification never overlap within a session.
     */
    const ds4_dspark_model *dspark;
    uint32_t dspark_cache_cap;
    uint32_t dspark_capture_rows_cap;
    bool dspark_capture_enabled;
    /* Absolute position one past the last injected context row. */
    uint32_t dspark_context_len;
    ds4_gpu_tensor *dspark_kv_cache[DS4_DSPARK_MAX_STAGES];
    ds4_gpu_tensor *dspark_hc_mean;
    ds4_gpu_tensor *dspark_hc_mean_rows;
    ds4_gpu_tensor *dspark_features;
    ds4_gpu_tensor *dspark_features_batch;
    ds4_gpu_tensor *dspark_fused;
    /* Fused feature of the last committed position. It becomes row 0 of every
     * stage's input, which is how the drafter sees the target's current state
     * on the query side rather than only as injected history. */
    ds4_gpu_tensor *dspark_fused_last;
    ds4_gpu_tensor *dspark_block_tokens;
    ds4_gpu_tensor* dspark_confidence;
    ds4_gpu_tensor *dspark_markov_key;
    ds4_gpu_tensor *dspark_markov_index;
    /* Bitmask of sampled target layers captured for the current position. */
    uint32_t dspark_capture_mask;
    uint32_t dspark_capture_batch_mask;
    uint32_t dspark_capture_batch_start;
    uint32_t dspark_capture_batch_tokens;
};

typedef struct ds4_rocm_graph ds4_gpu_graph;

/* DSpark feature capture, defined with the drafting code below. Both are no-ops
 * until a support model is attached, so the target paths only pay a branch. */
static bool rocm_graph_dspark_capture_decode_layer(ds4_gpu_graph *g, uint32_t il);
static bool rocm_graph_dspark_capture_rows(ds4_gpu_graph* g, uint32_t il,
                                           const ds4_gpu_tensor* hc,
                                           uint32_t n_rows);
static bool rocm_graph_dspark_capture_batch_layer(ds4_gpu_graph *g,
                                                  uint32_t il,
                                                  uint32_t start,
                                                  uint32_t n_tokens);
static bool rocm_graph_dspark_inject(ds4_gpu_graph *g,
                                     uint32_t pos0,
                                     uint32_t n_rows);
static bool rocm_graph_dspark_capture_complete(const ds4_gpu_graph *g);
static bool rocm_graph_capture_prefix_attn_state(ds4_gpu_graph *g,
                                                 uint32_t il,
                                                 uint32_t slot);
static bool rocm_graph_capture_prefix_index_state(ds4_gpu_graph *g,
                                                  uint32_t il,
                                                  uint32_t slot);

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
static void rocm_graph_spec_free(ds4_gpu_graph *g) {
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->spec_saved_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_saved_attn_state_score[il]);
        ds4_gpu_tensor_free(g->spec_saved_index_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_saved_index_state_score[il]);
        ds4_gpu_tensor_free(g->spec_prefix_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_prefix_attn_state_score[il]);
        ds4_gpu_tensor_free(g->spec_prefix_index_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_prefix_index_state_score[il]);
        g->spec_saved_attn_state_kv[il] = NULL;
        g->spec_saved_attn_state_score[il] = NULL;
        g->spec_saved_index_state_kv[il] = NULL;
        g->spec_saved_index_state_score[il] = NULL;
        g->spec_prefix_attn_state_kv[il] = NULL;
        g->spec_prefix_attn_state_score[il] = NULL;
        g->spec_prefix_index_state_kv[il] = NULL;
        g->spec_prefix_index_state_score[il] = NULL;
    }
    ds4_gpu_tensor_free(g->spec_row_tops);
    ds4_gpu_tensor_free(g->spec_logits);
    g->spec_row_tops = NULL;
    g->spec_logits = NULL;
    g->spec_rows_cap = 0;
    g->spec_frontier_valid = false;
    g->spec_capture_prefixes = false;
}

static void rocm_graph_dspark_free(ds4_gpu_graph *g) {
    for (uint32_t stage = 0; stage < DS4_DSPARK_MAX_STAGES; stage++) {
        ds4_gpu_tensor_free(g->dspark_kv_cache[stage]);
        g->dspark_kv_cache[stage] = NULL;
    }
    ds4_gpu_tensor_free(g->dspark_markov_index);
    ds4_gpu_tensor_free(g->dspark_markov_key);
    ds4_gpu_tensor_free(g->dspark_confidence);
    ds4_gpu_tensor_free(g->dspark_block_tokens);
    ds4_gpu_tensor_free(g->dspark_fused_last);
    ds4_gpu_tensor_free(g->dspark_fused);
    ds4_gpu_tensor_free(g->dspark_features_batch);
    ds4_gpu_tensor_free(g->dspark_features);
    ds4_gpu_tensor_free(g->dspark_hc_mean_rows);
    ds4_gpu_tensor_free(g->dspark_hc_mean);
    g->dspark_markov_index = NULL;
    g->dspark_markov_key = NULL;
    g->dspark_confidence = NULL;
    g->dspark_block_tokens = NULL;
    g->dspark_fused_last = NULL;
    g->dspark_fused = NULL;
    g->dspark_features_batch = NULL;
    g->dspark_features = NULL;
    g->dspark_hc_mean_rows = NULL;
    g->dspark_hc_mean = NULL;
    g->dspark = NULL;
    g->dspark_cache_cap = 0;
    g->dspark_capture_rows_cap = 0;
    g->dspark_capture_enabled = false;
    g->dspark_context_len = 0;
    g->dspark_capture_mask = 0;
    g->dspark_capture_batch_mask = 0;
    g->dspark_capture_batch_start = 0;
    g->dspark_capture_batch_tokens = 0;
}

/*
 * Prompt capture can be as wide as a full prefill chunk, while decode and
 * verification need only block_size + 1 rows. Keep the large buffers only
 * while the owning session is actively prefilling so idle HTTP sessions do
 * not each pin hundreds of MiB of unified memory.
 */
static bool rocm_graph_dspark_resize_capture(ds4_gpu_graph *g,
                                             uint32_t rows_cap) {
    if (!g || !g->dspark || !g->dspark_capture_enabled) return true;
    if (rows_cap == 0 || rows_cap > g->prefill_cap) return false;
    if (rows_cap == g->dspark_capture_rows_cap &&
        g->dspark_features && g->dspark_fused) {
        return true;
    }

    const uint32_t feature_width =
        g->dspark->n_target_layers * DS4_N_EMBD;
    ds4_gpu_tensor *features = ds4_gpu_tensor_alloc(
            (uint64_t)rows_cap * feature_width * sizeof(float));
    ds4_gpu_tensor *fused = ds4_gpu_tensor_alloc(
            (uint64_t)rows_cap * DS4_N_EMBD * sizeof(float));
    if (!features || !fused) {
        ds4_gpu_tensor_free(fused);
        ds4_gpu_tensor_free(features);
        return false;
    }

    ds4_gpu_tensor_free(g->dspark_fused);
    ds4_gpu_tensor_free(g->dspark_features);
    g->dspark_features = features;
    g->dspark_fused = fused;
    g->dspark_capture_rows_cap = rows_cap;
    g->dspark_capture_mask = 0;
    g->dspark_capture_batch_mask = 0;
    g->dspark_capture_batch_start = 0;
    g->dspark_capture_batch_tokens = 0;
    return true;
}

static void rocm_graph_bind_batch_workspace(
        ds4_gpu_graph *g,
        const ds4_gpu_graph *workspace) {
    g->prefill_tokens = workspace->prefill_tokens;
    g->batch_cur_hc = workspace->batch_cur_hc;
    g->batch_next_hc = workspace->batch_next_hc;
    g->batch_flat_hc = workspace->batch_flat_hc;
    g->batch_flat_hc_h = workspace->batch_flat_hc_h;
    g->batch_hc_mix = workspace->batch_hc_mix;
    g->batch_hc_split = workspace->batch_hc_split;
    g->batch_attn_cur = workspace->batch_attn_cur;
    g->batch_attn_norm = workspace->batch_attn_norm;
    g->batch_qr = workspace->batch_qr;
    g->batch_qr_norm = workspace->batch_qr_norm;
    g->batch_q = workspace->batch_q;
    g->batch_kv_raw = workspace->batch_kv_raw;
    g->batch_kv = workspace->batch_kv;
    g->batch_comp_kv = workspace->batch_comp_kv;
    g->batch_comp_sc = workspace->batch_comp_sc;
    g->batch_indexer_q = workspace->batch_indexer_q;
    g->batch_indexer_weights = workspace->batch_indexer_weights;
    g->batch_heads = workspace->batch_heads;
    g->batch_attn_low = workspace->batch_attn_low;
    g->batch_attn_out = workspace->batch_attn_out;
    g->batch_group_tmp = workspace->batch_group_tmp;
    g->batch_low_tmp = workspace->batch_low_tmp;
    g->batch_after_attn_hc = workspace->batch_after_attn_hc;
    g->batch_ffn_cur = workspace->batch_ffn_cur;
    g->batch_ffn_norm = workspace->batch_ffn_norm;
    g->batch_shared_gate = workspace->batch_shared_gate;
    g->batch_shared_up = workspace->batch_shared_up;
    g->batch_shared_mid = workspace->batch_shared_mid;
    g->batch_shared_out = workspace->batch_shared_out;
    g->batch_router_logits = workspace->batch_router_logits;
    g->batch_router_probs = workspace->batch_router_probs;
    g->batch_router_selected = workspace->batch_router_selected;
    g->batch_router_weights = workspace->batch_router_weights;
    g->batch_routed_gate = workspace->batch_routed_gate;
    g->batch_routed_up = workspace->batch_routed_up;
    g->batch_routed_mid = workspace->batch_routed_mid;
    g->batch_routed_down = workspace->batch_routed_down;
    g->batch_routed_out = workspace->batch_routed_out;
    g->batch_dspark_verify_cur_hc = workspace->batch_dspark_verify_cur_hc;
    g->batch_dspark_verify_after_attn_hc =
        workspace->batch_dspark_verify_after_attn_hc;
    g->batch_dspark_verify_rows_cap = workspace->batch_dspark_verify_rows_cap;
    g->batch_routed_mid_is_f16 = workspace->batch_routed_mid_is_f16;
    g->borrows_batch_workspace = true;
}

static void rocm_graph_free_batch_workspace(ds4_gpu_graph *g) {
  ds4_gpu_tensor_free(g->batch_dspark_verify_after_attn_hc);
  ds4_gpu_tensor_free(g->batch_dspark_verify_cur_hc);
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
  ds4_gpu_tensor_free(g->batch_flat_hc_h);
  ds4_gpu_tensor_free(g->batch_next_hc);
  ds4_gpu_tensor_free(g->batch_cur_hc);
  ds4_gpu_tensor_free(g->prefill_tokens);
}

static void rocm_graph_free(ds4_gpu_graph *g) {
    rocm_graph_dspark_free(g);
    rocm_graph_spec_free(g);
    if (!g->borrows_batch_workspace) {
        rocm_graph_free_batch_workspace(g);
    }
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
        uint32_t                prefill_cap,
        const ds4_gpu_graph    *batch_workspace) {
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
    if (batch_workspace) {
        if (batch_workspace->prefill_cap < prefill_cap ||
            batch_workspace->comp_cap < g->comp_cap) {
            rocm_graph_free(g);
            return false;
        }
        rocm_graph_bind_batch_workspace(g, batch_workspace);
    } else {
        g->prefill_tokens = ds4_gpu_tensor_alloc(pc * sizeof(int32_t));
        g->batch_cur_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
        g->batch_next_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
        g->batch_flat_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
        /* F16 mirror so the hyper-connection projection can consume the norm
         * directly; optional, the F32 path stands if this allocation fails. */
        g->batch_flat_hc_h =
            ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(uint16_t));
        g->batch_hc_mix = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
        g->batch_hc_split = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
        g->batch_attn_cur =
            ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
        g->batch_attn_norm =
            ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
        g->batch_qr = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
        g->batch_qr_norm = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
        g->batch_q = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
        g->batch_kv_raw =
            ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
        g->batch_kv =
            ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
        g->batch_comp_kv =
            ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
        g->batch_comp_sc =
            ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
        g->batch_indexer_q =
            ds4_gpu_tensor_alloc(pc * indexer_q_dim * sizeof(float));
        g->batch_indexer_weights =
            ds4_gpu_tensor_alloc(pc * DS4_N_INDEXER_HEAD * sizeof(float));
        g->batch_heads = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
        g->batch_attn_low =
            ds4_gpu_tensor_alloc(pc * low_dim * sizeof(float));
        g->batch_attn_out =
            ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
        g->batch_group_tmp =
            ds4_gpu_tensor_alloc(pc * group_dim * sizeof(float));
        g->batch_low_tmp =
            ds4_gpu_tensor_alloc(pc * DS4_N_LORA_O * sizeof(float));
        g->batch_after_attn_hc =
            ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
        g->batch_ffn_cur =
            ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
        g->batch_ffn_norm =
            ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
        g->batch_shared_gate =
            ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
        g->batch_shared_up =
            ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
        g->batch_shared_mid =
            ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
        g->batch_shared_out =
            ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
        g->batch_router_logits =
            ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
        g->batch_router_probs =
            ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
        g->batch_router_selected =
            ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(int));
        g->batch_router_weights =
            ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(float));
        g->batch_routed_gate = ds4_gpu_tensor_alloc(
            pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
        g->batch_routed_up = ds4_gpu_tensor_alloc(
            pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
        g->batch_routed_mid = ds4_gpu_tensor_alloc(
            pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
        g->batch_routed_down = ds4_gpu_tensor_alloc(
            pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
        g->batch_routed_out =
            ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    }

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
    /* Keep dense attention longer than the legacy 512-row window. Around the 2K
     * frontier the sparse path's score/top-k setup dominates the smaller
     * attention scan, while larger contexts benefit from sparse indexed
     * attention. This threshold changes only the implementation used to consume
     * the compressed rows; it must not lower the 512-row indexer selection
     * defined by DS4_N_INDEXER_TOP_K. */
    return 1024u;
}

/* =========================================================================
 * ROCm Decode Release Helpers and Reference Fallbacks.
 * =========================================================================
 *
 * The generation path uses the fused helpers below. The older unfused kernels
 * are retained as reference implementations for numerics debugging but are no
 * longer selectable at run time -- the fused route is unconditional.
 */

static bool rocm_graph_decode_hc_pre(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const ds4_model        *model,
        uint64_t                scale_offset,
        uint64_t                base_offset) {
    if (false) {
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
    if (false) {
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
    const bool qkv_rms_fused = !false;

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
        !false &&
        !false;
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
        if (ok && !false) {
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
            if (ok && !false) {
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
        !false;
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
    if (ok) {
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
        !false;
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
    if (w->type == DS4_TENSOR_Q8_0) {
        /* The target's dense projections are F16 or F32; the DSpark support
         * model ships a Q8_0 router projection. */
        return ds4_gpu_matmul_q8_0_tensor(out, model->map, model->size,
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
    const uint32_t split_after_layers = 4;

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
        /* Capture after the swap so cur_hc holds this layer's output. */
        if (ok) ok = rocm_graph_dspark_capture_decode_layer(g, il);
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
    if (warmed) return true;

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

/*
 * Stateful middle of one decode attention row.
 *
 * Multi-session decode batches share the dense HC/Q/KV projections, but each
 * row owns a different raw/compressed cache and absolute position. This helper
 * keeps that stateful middle identical to ordinary decode while accepting row
 * views produced by the shared front end.
 */
static bool rocm_graph_encode_session_attention_core(
        ds4_gpu_graph       *g,
        const ds4_model     *model,
        const ds4_layer_weights *layer,
        uint32_t             il,
        uint32_t             pos,
        const ds4_gpu_tensor *attn_norm,
        const ds4_gpu_tensor *qr_norm,
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *kv,
        ds4_gpu_tensor       *heads) {
    if (!g || !model || !layer || !attn_norm || !qr_norm || !q || !kv ||
        !heads || g->raw_cap == 0) {
        return false;
    }

    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const bool compressed = ds4_layer_compress_ratio(il) != 0;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor =
        compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    const uint32_t raw_row = pos % g->raw_cap;
    const uint32_t n_raw = rocm_graph_raw_span_for_batch(g, pos, 1);
    ds4_gpu_tensor *raw_cache = g->layer_raw_cache[il];
    bool ok = ds4_gpu_rope_tail_tensor(
                  q,
                  1,
                  DS4_N_HEAD,
                  DS4_N_HEAD_DIM,
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
    if (ok) {
        ok = ds4_gpu_rope_tail_tensor(
                 kv,
                 1,
                 DS4_N_HEAD_KV,
                 DS4_N_HEAD_DIM,
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
    }
    if (ok) {
        ok = rocm_graph_decode_kv_store(kv, raw_cache, g->raw_cap, raw_row);
    }

    uint32_t n_comp = 0;
    ds4_gpu_tensor *comp_cache = NULL;
    ds4_gpu_tensor *comp_selected = NULL;
    uint32_t n_selected = 0;
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
            fprintf(stderr,
                    "ds4: ROCm graph compressor expects paired F16 "
                    "compressor projections\n");
            ok = false;
        }
        if (ok && emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
            fprintf(stderr,
                    "ds4: ROCm graph compressed KV cache capacity exceeded "
                    "at layer %u\n",
                    il);
            ok = false;
        }
        if (ok) {
            ok = ds4_gpu_matmul_f16_pair_tensor(
                     g->comp_kv_cur,
                     g->comp_sc_cur,
                     model->map,
                     model->size,
                     layer->attn_compressor_kv->abs_offset,
                     layer->attn_compressor_gate->abs_offset,
                     DS4_N_EMBD,
                     comp_width,
                     attn_norm,
                     1) != 0;
        }
        const uint32_t comp_row = g->layer_n_comp[il];
        if (ok) {
            ok = ds4_gpu_compressor_update_tensor(
                     g->comp_kv_cur,
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
        }
        if (ok && emit) {
            ds4_gpu_tensor *comp_row_view =
                rocm_graph_attn_comp_row_view(g, il, comp_row);
            if (!comp_row_view) {
                ok = false;
            } else {
                ok = rocm_graph_quantize_attn_comp_row(
                    g, il, comp_row, comp_row_view);
                ds4_gpu_tensor_free(comp_row_view);
            }
        }
        if (ok && emit) {
            g->layer_n_comp[il]++;
        }

        if (ok && ratio == 4) {
            const uint32_t index_width =
                coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv ||
                !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape ||
                !layer->indexer_compressor_norm ||
                layer->indexer_compressor_kv->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_gate->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_kv->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_gate->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_kv->dim[1] != index_width ||
                layer->indexer_compressor_gate->dim[1] != index_width) {
                fprintf(stderr,
                        "ds4: ROCm graph indexer compressor expects paired "
                        "F16 compressor projections\n");
                ok = false;
            }
            if (ok && emit &&
                g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                fprintf(stderr,
                        "ds4: ROCm graph indexer compressed KV cache capacity "
                        "exceeded at layer %u\n",
                        il);
                ok = false;
            }
            if (ok) {
                ok = ds4_gpu_matmul_f16_pair_tensor(
                         g->comp_kv_cur,
                         g->comp_sc_cur,
                         model->map,
                         model->size,
                         layer->indexer_compressor_kv->abs_offset,
                         layer->indexer_compressor_gate->abs_offset,
                         DS4_N_EMBD,
                         index_width,
                         attn_norm,
                         1) != 0;
            }
            const uint32_t index_row = g->layer_n_index_comp[il];
            if (ok) {
                ok = ds4_gpu_compressor_update_tensor(
                         g->comp_kv_cur,
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
            }
            if (ok && emit) {
                ds4_gpu_tensor *index_row_view = ds4_gpu_tensor_view(
                    g->layer_index_comp_cache[il],
                    (uint64_t)index_row * DS4_N_INDEXER_HEAD_DIM *
                        sizeof(float),
                    (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                if (!index_row_view) {
                    ok = false;
                } else {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(
                             index_row_view,
                             1,
                             DS4_N_INDEXER_HEAD_DIM) != 0;
                    ds4_gpu_tensor_free(index_row_view);
                }
            }
            if (ok && emit) {
                g->layer_n_index_comp[il]++;
            }
            const uint32_t sparse_threshold =
                rocm_graph_decode_indexer_sparse_threshold(g);
            if (ok && g->layer_n_comp[il] > sparse_threshold &&
                g->layer_n_index_comp[il] > DS4_N_INDEXER_TOP_K) {
                const uint64_t indexer_q_dim =
                    (uint64_t)DS4_N_INDEXER_HEAD *
                    DS4_N_INDEXER_HEAD_DIM;
                if (!layer->indexer_attn_q_b ||
                    layer->indexer_attn_q_b->type != DS4_TENSOR_F16 ||
                    layer->indexer_attn_q_b->dim[0] != q_rank ||
                    layer->indexer_attn_q_b->dim[1] != indexer_q_dim ||
                    !layer->indexer_proj ||
                    layer->indexer_proj->type != DS4_TENSOR_F16 ||
                    layer->indexer_proj->dim[0] != DS4_N_EMBD ||
                    layer->indexer_proj->dim[1] !=
                        DS4_N_INDEXER_HEAD) {
                    fprintf(stderr,
                            "ds4: ROCm graph indexer projections have "
                            "unexpected shapes\n");
                    ok = false;
                }
                if (ok) {
                    ok = ds4_gpu_matmul_f16_tensor(
                             g->indexer_q,
                             model->map,
                             model->size,
                             layer->indexer_attn_q_b->abs_offset,
                             q_rank,
                             indexer_q_dim,
                             qr_norm,
                             1) != 0;
                }
                if (ok) {
                    ok = ds4_gpu_rope_tail_tensor(
                             g->indexer_q,
                             1,
                             DS4_N_INDEXER_HEAD,
                             DS4_N_INDEXER_HEAD_DIM,
                             DS4_N_ROT,
                             pos,
                             compressed
                                 ? (uint32_t)DS4_ROPE_ORIG_CTX
                                 : 0,
                             false,
                             freq_base,
                             freq_scale,
                             ext_factor,
                             attn_factor,
                             DS4_ROPE_YARN_BETA_FAST,
                             DS4_ROPE_YARN_BETA_SLOW) != 0;
                }
                if (ok) {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(
                             g->indexer_q,
                             DS4_N_INDEXER_HEAD,
                             DS4_N_INDEXER_HEAD_DIM) != 0;
                }
                if (ok) {
                    ok = ds4_gpu_matmul_f16_tensor(
                             g->indexer_weights,
                             model->map,
                             model->size,
                             layer->indexer_proj->abs_offset,
                             DS4_N_EMBD,
                             DS4_N_INDEXER_HEAD,
                             attn_norm,
                             1) != 0;
                }
                const float index_scale =
                    1.0f /
                    sqrtf((float)(DS4_N_INDEXER_HEAD_DIM *
                                  DS4_N_INDEXER_HEAD));
                if (ok) {
                    ok = ds4_gpu_indexer_score_one_tensor(
                             g->indexer_scores,
                             g->indexer_q,
                             g->indexer_weights,
                             g->layer_index_comp_cache[il],
                             g->layer_n_index_comp[il],
                             DS4_N_INDEXER_HEAD,
                             DS4_N_INDEXER_HEAD_DIM,
                             index_scale) != 0;
                }
                if (ok) {
                    ok = ds4_gpu_indexer_topk_tensor(
                             g->comp_selected,
                             g->indexer_scores,
                             g->layer_n_index_comp[il],
                             1,
                             DS4_N_INDEXER_TOP_K) != 0;
                }
                if (ok) {
                    comp_selected = g->comp_selected;
                    n_selected =
                        DS4_N_INDEXER_TOP_K <
                                g->layer_n_index_comp[il]
                            ? DS4_N_INDEXER_TOP_K
                            : g->layer_n_index_comp[il];
                }
            }
        }

        n_comp = g->layer_n_comp[il];
        comp_cache = g->layer_attn_comp_cache[il];
    }

    if (ok) {
        const uint32_t raw_start =
            rocm_graph_raw_start_for_span(g, pos, n_raw);
        if (n_comp != 0 && comp_selected != NULL &&
            n_selected != 0) {
            ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                     heads,
                     model->map,
                     model->size,
                     layer->attn_sinks->abs_offset,
                     q,
                     raw_cache,
                     g->layer_attn_comp_cache[il],
                     rocm_graph_attn_comp_cache_is_f16(),
                     comp_selected,
                     1,
                     pos,
                     n_raw,
                     g->raw_cap,
                     raw_start,
                     n_comp,
                     n_selected,
                     g->raw_window,
                     ds4_layer_compress_ratio(il),
                     DS4_N_HEAD,
                     DS4_N_HEAD_DIM) != 0;
        } else {
            ok = ds4_gpu_attention_decode_heads_tensor(
                     heads,
                     model->map,
                     model->size,
                     layer->attn_sinks->abs_offset,
                     q,
                     raw_cache,
                     n_raw,
                     g->raw_cap,
                     raw_start,
                     n_comp ? comp_cache : NULL,
                     rocm_graph_attn_comp_cache_is_f16(),
                     n_comp,
                     NULL,
                     0,
                     DS4_N_HEAD,
                     DS4_N_HEAD_DIM) != 0;
        }
    }
    if (ok) {
        ok = ds4_gpu_rope_tail_tensor(
                 heads,
                 1,
                 DS4_N_HEAD,
                 DS4_N_HEAD_DIM,
                 DS4_N_ROT,
                 pos,
                 compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                 true,
                 freq_base,
                 freq_scale,
                 ext_factor,
                 attn_factor,
                 DS4_ROPE_YARN_BETA_FAST,
                 DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    return ok;
}

static bool rocm_graph_encode_layer_attention_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    /* Any published F16 activation mirror belongs to the previous layer, whose
     * buffers this layer reuses. Drop it before anything can overwrite them. */
    ds4_gpu_clear_f16_input();

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
    const bool qkv_rms_fused = !false;
    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
            g->batch_attn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(
            g->batch_after_attn_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;
    /* Normalize straight to F16 when the projection can take it: the F32 store
     * and the conversion pass that followed both disappear. Falls back whole. */
    bool hc_norm_f16 = false;
    bool hc_pre_fused = false;
    if (ok && n_tokens >= 128u) {
        hc_pre_fused = ds4_gpu_hc_norm_mix_split_weighted_sum_tensor(
                           attn_cur_view, hc_mix_view, hc_split_view,
                           g->batch_cur_hc, model->map, model->size,
                           layer->hc_attn_fn->abs_offset,
                           layer->hc_attn_scale->abs_offset,
                           layer->hc_attn_base->abs_offset,
                           DS4_N_EMBD, DS4_N_HC, n_tokens,
                           DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS,
                           DS4_RMS_EPS) != 0;
    }
    if (ok && !hc_pre_fused && n_tokens >= 128u && g->batch_flat_hc_h) {
        hc_norm_f16 = ds4_gpu_rms_norm_plain_rows_f16_tensor(
                          g->batch_flat_hc_h, g->batch_cur_hc,
                          (uint32_t)hc_dim, n_tokens, DS4_RMS_EPS) != 0 &&
                      ds4_gpu_matmul_f16_f16_input_tensor(
                          hc_mix_view, model->map, model->size,
                          layer->hc_attn_fn->abs_offset, hc_dim, mix_hc,
                          g->batch_flat_hc_h, n_tokens) != 0;
    }
    if (ok && !hc_pre_fused && !hc_norm_f16) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok && !hc_pre_fused && !hc_norm_f16) ok = ds4_gpu_matmul_f16_tensor(hc_mix_view,
                                             model->map,
                                             model->size,
                                             layer->hc_attn_fn->abs_offset,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (false) {
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
    } else if (!hc_pre_fused) {
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
    /* Eight projections in a ratio-4 layer read these rows, and each F16 route
     * among them was converting them again. One mirror serves all of them; the
     * conversion is row-local, so every consumer sees the bytes it would have
     * produced itself. Declining is not an error. */
    if (ok && n_tokens >= 128u) {  /* DS4_ROCM_WIDE_PREFILL_ROWS */
        (void)ds4_gpu_publish_f16_input_tensor(g->batch_attn_norm,
                                                (uint64_t)n_tokens * DS4_N_EMBD);
    }
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
    /* One pass over the query rows instead of two. The fused kernel stages each
     * row in LDS, so the norm's write-back is no longer read again by the rope
     * pass; it is bit-identical and falls back when the row does not fit. */
    bool q_norm_rope_fused =
        ok && ds4_gpu_head_rms_norm_rope_tail_tensor(g->batch_q,
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
                                                     DS4_ROPE_YARN_BETA_SLOW,
                                                     DS4_RMS_EPS) != 0;
    if (ok && !q_norm_rope_fused) ok = ds4_gpu_head_rms_norm_tensor(g->batch_q,
                                                n_tokens,
                                                DS4_N_HEAD,
                                                DS4_N_HEAD_DIM,
                                                DS4_RMS_EPS) != 0;
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("head_norm");
    if (ok && !q_norm_rope_fused) ok = ds4_gpu_rope_tail_tensor(g->batch_q,
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
            const bool aligned_chunk =
                (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
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
                    if (ok && t + 1u < n_tokens &&
                        t < DS4_SPEC_PREFIX_SLOTS) {
                        ok = rocm_graph_capture_prefix_attn_state(g, il, t);
                    }
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
                const bool aligned_chunk =
                    (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
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
                        if (ok && t + 1u < n_tokens &&
                            t < DS4_SPEC_PREFIX_SLOTS) {
                            ok = rocm_graph_capture_prefix_index_state(g, il, t);
                        }
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
    /*
     * A one-row batch is used by the multi-session route only for the
     * cache-owning attention middle. Keep its output projection on the release
     * decode fusion: the generic batch projection is sized for prompt rows and
     * costs about 2 ms per layer at W=1, dwarfing the attention itself.
     */
    bool output_hc_fused = false;
    if (ok && n_tokens == 1u) {
        ok = ds4_gpu_rope_tail_tensor(g->batch_heads,
                                     1,
                                     DS4_N_HEAD,
                                     DS4_N_HEAD_DIM,
                                     DS4_N_ROT,
                                     pos0,
                                     compressed
                                         ? (uint32_t)DS4_ROPE_ORIG_CTX
                                         : 0,
                                     true,
                                     freq_base,
                                     freq_scale,
                                     ext_factor,
                                     attn_factor,
                                     DS4_ROPE_YARN_BETA_FAST,
                                     DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok) {
            ok = ds4_gpu_attention_output_low_q8_tensor(
                     g->attn_low,
                     model->map,
                     model->size,
                     layer->attn_output_a->abs_offset,
                     group_dim,
                     rank,
                     n_groups,
                     g->batch_heads) != 0;
        }
        if (ok) {
            ok = ds4_gpu_matmul_q8_0_hc_expand_tensor(
                     after_attn_hc_view,
                     g->attn_out,
                     model->map,
                     model->size,
                     layer->attn_output_b->abs_offset,
                     (uint64_t)n_groups * rank,
                     DS4_N_EMBD,
                     g->attn_low,
                     g->batch_cur_hc,
                     hc_split_view,
                     DS4_N_EMBD,
                     DS4_N_HC) != 0;
        }
        output_hc_fused = ok;
    }

    /* The output projection can fold the inverse rotated tail into its F16 group
     * pack; nothing else reads the roped F32 heads. It declines when the shape
     * or width does not qualify, in which case the separate pass runs. */
    bool inv_rope_fused = false;
    if (ok && !output_hc_fused) {
        inv_rope_fused = ds4_gpu_attention_output_q8_batch_inv_rope_tensor(
                             g->batch_attn_out,
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
                             n_tokens,
                             DS4_N_HEAD_DIM,
                             DS4_N_ROT,
                             pos0,
                             compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                             freq_base,
                             freq_scale,
                             ext_factor,
                             attn_factor,
                             DS4_ROPE_YARN_BETA_FAST,
                             DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    if (ok && !output_hc_fused && !inv_rope_fused) ok = ds4_gpu_rope_tail_tensor(g->batch_heads,
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
    if (ok && !output_hc_fused && !inv_rope_fused) {
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
    if (ok && !output_hc_fused) ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
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

/*
 * Multi-session decode attention.
 *
 * HC/Q/KV and output projections are position independent, so run them once
 * over all active rows. Cache mutation, compressor recurrence, indexer lookup,
 * and attention itself remain on each session graph.
 */
static bool rocm_graph_encode_sessions_attention_batch(
        ds4_gpu_graph             *g,
        const ds4_model           *model,
        const ds4_layer_weights   *layer,
        uint32_t                   il,
        const ds4_rocm_batch_item *items,
        uint32_t                   n_tokens) {
    if (!g || !model || !layer || !items || n_tokens < 2u ||
        n_tokens > g->prefill_cap) {
        return false;
    }
    for (uint32_t row = 0; row < n_tokens; ++row) {
        if (!items[row].graph || items[row].graph->raw_cap == 0) {
            return false;
        }
    }

    ds4_gpu_clear_f16_input();
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc =
        2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim =
        (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const bool stage_profile =
        getenv("GUFO_DEEPSEEK_ROCM_LAYER_STAGE_PROFILE") != NULL;
    double stage_t0 = stage_profile ? ds4_now_seconds() : 0.0;
#define GUFO_DEEPSEEK_ROCM_PROFILE_SESSION_ATTN(name) do { \
        if (ok && stage_profile) { \
            ok = rocm_graph_layer_stage_profile_boundary( \
                "session_attn", (name), il, items[0].position, n_tokens, \
                &stage_t0); \
        } \
    } while (0)

    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
        g->batch_hc_mix,
        0,
        (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
        g->batch_hc_split,
        0,
        (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
        g->batch_attn_cur,
        0,
        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(
        g->batch_after_attn_hc,
        0,
        (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && attn_cur_view &&
              after_attn_hc_view;
    if (ok) {
        ok = ds4_gpu_rms_norm_plain_rows_tensor(
                 g->batch_flat_hc,
                 g->batch_cur_hc,
                 (uint32_t)hc_dim,
                 n_tokens,
                 DS4_RMS_EPS) != 0;
    }
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(
                 hc_mix_view,
                 model->map,
                 model->size,
                 layer->hc_attn_fn->abs_offset,
                 hc_dim,
                 mix_hc,
                 g->batch_flat_hc,
                 n_tokens) != 0;
    }
    if (ok) {
        /*
         * Use the release decode fusion for every row. The kernel maps one
         * independent block to each session, so widening the grid preserves
         * the one-row Sinkhorn, weighted-sum, and RMS reduction order exactly.
         */
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(
                 attn_cur_view,
                 g->batch_attn_norm,
                 hc_split_view,
                 hc_mix_view,
                 g->batch_cur_hc,
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
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_SESSION_ATTN("hc_pre");

    if (ok) {
        ok = rocm_graph_matmul_q8_0_named_tensor(
            "attn_q_a",
            il,
            items[0].position,
            g->batch_qr,
            model,
            layer->attn_q_a,
            DS4_N_EMBD,
            q_rank,
            g->batch_attn_norm,
            n_tokens);
    }
    if (ok) {
        ok = rocm_graph_matmul_q8_0_named_tensor(
            "attn_kv",
            il,
            items[0].position,
            g->batch_kv_raw,
            model,
            layer->attn_kv,
            DS4_N_EMBD,
            DS4_N_HEAD_DIM,
            g->batch_attn_norm,
            n_tokens);
    }
    if (ok) {
        ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                 g->batch_qr_norm,
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
    }
    if (ok) {
        ok = rocm_graph_matmul_q8_0_named_tensor(
            "attn_q_b",
            il,
            items[0].position,
            g->batch_q,
            model,
            layer->attn_q_b,
            q_rank,
            q_dim,
            g->batch_qr_norm,
            n_tokens);
    }
    if (ok) {
        ok = ds4_gpu_head_rms_norm_tensor(
                 g->batch_q,
                 n_tokens,
                 DS4_N_HEAD,
                 DS4_N_HEAD_DIM,
                 DS4_RMS_EPS) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_SESSION_ATTN("qkv");

    for (uint32_t row = 0; ok && row < n_tokens; ++row) {
        ds4_gpu_tensor *attn_norm = rocm_graph_tensor_row_view(
            g->batch_attn_norm, row, DS4_N_EMBD);
        ds4_gpu_tensor *qr_norm = rocm_graph_tensor_row_view(
            g->batch_qr_norm, row, q_rank);
        ds4_gpu_tensor *q =
            rocm_graph_tensor_row_view(g->batch_q, row, q_dim);
        ds4_gpu_tensor *kv = rocm_graph_tensor_row_view(
            g->batch_kv, row, DS4_N_HEAD_DIM);
        ds4_gpu_tensor *heads =
            rocm_graph_tensor_row_view(g->batch_heads, row, q_dim);
        ok = attn_norm && qr_norm && q && kv && heads;
        if (ok) {
            ok = rocm_graph_encode_session_attention_core(
                items[row].graph,
                model,
                layer,
                il,
                items[row].position,
                attn_norm,
                qr_norm,
                q,
                kv,
                heads);
        }
        ds4_gpu_tensor_free(heads);
        ds4_gpu_tensor_free(kv);
        ds4_gpu_tensor_free(q);
        ds4_gpu_tensor_free(qr_norm);
        ds4_gpu_tensor_free(attn_norm);
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_SESSION_ATTN("stateful_core");

    if (ok) {
        ok = ds4_gpu_attention_output_q8_batch_tensor(
                 g->batch_attn_out,
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
        ok = ds4_gpu_hc_expand_split_tensor(
                 after_attn_hc_view,
                 g->batch_attn_out,
                 g->batch_cur_hc,
                 hc_split_view,
                 DS4_N_EMBD,
                 DS4_N_HC) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_SESSION_ATTN("output");

    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
#undef GUFO_DEEPSEEK_ROCM_PROFILE_SESSION_ATTN
    return ok;
}

/* Encode the batched prefill FFN half: HC pre/norm, shared expert, routed
 * experts, sum, and HC post. */
static bool rocm_graph_encode_layer_ffn_batch(
    ds4_gpu_graph* g, const ds4_model* model, const ds4_layer_weights* layer,
    uint32_t il, uint32_t pos0, uint32_t n_tokens,
    const ds4_gpu_tensor* tokens = nullptr, uint32_t router_group_rows = 0) {
  if (n_tokens == 0 || n_tokens > g->prefill_cap)
    return false;
  if (!tokens)
    tokens = g->prefill_tokens;
  if (router_group_rows > n_tokens ||
      (router_group_rows != 0u && n_tokens % router_group_rows != 0u)) {
    return false;
  }

  const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
  const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
  const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
  const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
  const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
  const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
  const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
  const uint64_t gate_row_bytes =
      ds4_routed_expert_row_bytes(layer->ffn_gate_exps);
  const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
  const uint64_t down_row_bytes =
      ds4_routed_expert_row_bytes(layer->ffn_down_exps);
  const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
  const bool layer_stage_profile =
      getenv("GUFO_DEEPSEEK_ROCM_LAYER_STAGE_PROFILE") != NULL;
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
    /* Normalize straight to F16 when the projection can take it: the F32 store
     * and the conversion pass that followed both disappear. Falls back whole. */
    bool hc_norm_f16 = false;
    bool hc_pre_fused = false;
    if (ok && n_tokens >= 128u) {
        hc_pre_fused = ds4_gpu_hc_norm_mix_split_weighted_sum_tensor(
                           ffn_cur_view, hc_mix_view, hc_split_view,
                           g->batch_after_attn_hc, model->map, model->size,
                           layer->hc_ffn_fn->abs_offset,
                           layer->hc_ffn_scale->abs_offset,
                           layer->hc_ffn_base->abs_offset,
                           DS4_N_EMBD, DS4_N_HC, n_tokens,
                           DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS,
                           DS4_RMS_EPS) != 0;
    }
    if (ok && !hc_pre_fused && n_tokens >= 128u && g->batch_flat_hc_h) {
        hc_norm_f16 = ds4_gpu_rms_norm_plain_rows_f16_tensor(
                          g->batch_flat_hc_h, g->batch_after_attn_hc,
                          (uint32_t)hc_dim, n_tokens, DS4_RMS_EPS) != 0 &&
                      ds4_gpu_matmul_f16_f16_input_tensor(
                          hc_mix_view, model->map, model->size,
                          layer->hc_ffn_fn->abs_offset, hc_dim, mix_hc,
                          g->batch_flat_hc_h, n_tokens) != 0;
    }
    if (ok && !hc_pre_fused && !hc_norm_f16) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_after_attn_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok && !hc_pre_fused && !hc_norm_f16) ok = ds4_gpu_matmul_f16_tensor(hc_mix_view,
                                             model->map,
                                             model->size,
                                             layer->hc_ffn_fn->abs_offset,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (ok && !hc_pre_fused) {
        /*
         * The decode fusion is row-independent, so a W2-W8 grid is exact to
         * launching the C1 kernel once per session and also avoids a separate
         * normalization pass.
         */
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(
                 ffn_cur_view,
                 g->batch_ffn_norm,
                 hc_split_view,
                 hc_mix_view,
                 g->batch_after_attn_hc,
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
    }
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("hc_pre");
    if (ok && hc_pre_fused) ok = ds4_gpu_rms_norm_weight_rows_tensor(
                                      g->batch_ffn_norm,
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
    if (ok && router_group_rows != 0u && router_group_rows < n_tokens) {
      const uint64_t logits_row_bytes = (uint64_t)DS4_N_EXPERT * sizeof(float);
      const uint64_t selected_row_bytes =
          (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t);
      const uint64_t weights_row_bytes =
          (uint64_t)DS4_N_EXPERT_USED * sizeof(float);
      for (uint32_t row0 = 0; ok && row0 < n_tokens;
           row0 += router_group_rows) {
        ds4_gpu_tensor* logits_view = ds4_gpu_tensor_view(
            g->batch_router_logits, (uint64_t)row0 * logits_row_bytes,
            (uint64_t)router_group_rows * logits_row_bytes);
        ds4_gpu_tensor* probs_view = ds4_gpu_tensor_view(
            g->batch_router_probs, (uint64_t)row0 * logits_row_bytes,
            (uint64_t)router_group_rows * logits_row_bytes);
        ds4_gpu_tensor* selected_view = ds4_gpu_tensor_view(
            g->batch_router_selected, (uint64_t)row0 * selected_row_bytes,
            (uint64_t)router_group_rows * selected_row_bytes);
        ds4_gpu_tensor* weights_view = ds4_gpu_tensor_view(
            g->batch_router_weights, (uint64_t)row0 * weights_row_bytes,
            (uint64_t)router_group_rows * weights_row_bytes);
        ds4_gpu_tensor* norm_view = ds4_gpu_tensor_view(
            g->batch_ffn_norm, (uint64_t)row0 * DS4_N_EMBD * sizeof(float),
            (uint64_t)router_group_rows * DS4_N_EMBD * sizeof(float));
        ds4_gpu_tensor* tokens_view =
            ds4_gpu_tensor_view(tokens, (uint64_t)row0 * sizeof(int32_t),
                                (uint64_t)router_group_rows * sizeof(int32_t));
        ok = logits_view && probs_view && selected_view && weights_view &&
             norm_view && tokens_view;
        if (ok) {
          ok = rocm_graph_matmul_plain_tensor(
              logits_view, model, layer->ffn_gate_inp, DS4_N_EMBD, DS4_N_EXPERT,
              norm_view, router_group_rows);
        }
        if (ok) {
          ok = ds4_gpu_router_select_batch_tensor(
                   selected_view, weights_view, probs_view, model->map,
                   model->size,
                   layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset
                                          : 0,
                   layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset
                                           : 0,
                   layer->ffn_gate_tid2eid
                       ? static_cast<uint32_t>(layer->ffn_gate_tid2eid->dim[1])
                       : 0,
                   0, 0, layer->ffn_exp_probs_b != nullptr,
                   layer->ffn_gate_tid2eid != nullptr, logits_view, tokens_view,
                   DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_EXPERT_WEIGHT_SCALE,
                   router_group_rows) != 0;
        }
        ds4_gpu_tensor_free(tokens_view);
        ds4_gpu_tensor_free(norm_view);
        ds4_gpu_tensor_free(weights_view);
        ds4_gpu_tensor_free(selected_view);
        ds4_gpu_tensor_free(probs_view);
        ds4_gpu_tensor_free(logits_view);
      }
    } else {
      if (ok) {
        ok = rocm_graph_matmul_plain_tensor(
            g->batch_router_logits, model, layer->ffn_gate_inp, DS4_N_EMBD,
            DS4_N_EXPERT, g->batch_ffn_norm, n_tokens);
      }
      if (ok) {
        ok =
            ds4_gpu_router_select_batch_tensor(
                g->batch_router_selected, g->batch_router_weights,
                g->batch_router_probs, model->map, model->size,
                layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset
                                        : 0,
                layer->ffn_gate_tid2eid
                    ? static_cast<uint32_t>(layer->ffn_gate_tid2eid->dim[1])
                    : 0,
                0, 0, layer->ffn_exp_probs_b != nullptr,
                layer->ffn_gate_tid2eid != nullptr, g->batch_router_logits,
                tokens, DS4_N_EXPERT, DS4_N_EXPERT_USED,
                DS4_EXPERT_WEIGHT_SCALE, n_tokens) != 0;
      }
    }
    if (ok) {
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("router");

    /* Ask the routed MoE to leave its per-expert F16 rows unsummed so the
     * hyper-connection expansion below can fold the 6-way sum in, saving one
     * round trip through batch_routed_out. Advisory: only the F16 down route can
     * do it, so the outcome is read back after the call. */
    if (ok) ds4_gpu_set_routed_defer_sum(n_tokens >= 128u ? 1 : 0);
    if (ok && router_group_rows != 0u && router_group_rows < n_tokens) {
      const uint64_t gate_row_elems =
          (uint64_t)DS4_N_EXPERT_USED * expert_mid_dim;
      const uint64_t down_row_elems =
          (uint64_t)DS4_N_EXPERT_USED * routed_out_dim;
      const uint64_t selected_row_bytes =
          (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t);
      const uint64_t weights_row_bytes =
          (uint64_t)DS4_N_EXPERT_USED * sizeof(float);
      bool all_mid_f16 = true;
      for (uint32_t row0 = 0; ok && row0 < n_tokens;
           row0 += router_group_rows) {
        ds4_gpu_tensor* out_view = ds4_gpu_tensor_view(
            g->batch_routed_out,
            (uint64_t)row0 * routed_out_dim * sizeof(float),
            (uint64_t)router_group_rows * routed_out_dim * sizeof(float));
        ds4_gpu_tensor* gate_view = ds4_gpu_tensor_view(
            g->batch_routed_gate,
            (uint64_t)row0 * gate_row_elems * sizeof(float),
            (uint64_t)router_group_rows * gate_row_elems * sizeof(float));
        ds4_gpu_tensor* up_view = ds4_gpu_tensor_view(
            g->batch_routed_up, (uint64_t)row0 * gate_row_elems * sizeof(float),
            (uint64_t)router_group_rows * gate_row_elems * sizeof(float));
        ds4_gpu_tensor* mid_view = ds4_gpu_tensor_view(
            g->batch_routed_mid,
            (uint64_t)row0 * gate_row_elems * sizeof(float),
            (uint64_t)router_group_rows * gate_row_elems * sizeof(float));
        ds4_gpu_tensor* down_view = ds4_gpu_tensor_view(
            g->batch_routed_down,
            (uint64_t)row0 * down_row_elems * sizeof(float),
            (uint64_t)router_group_rows * down_row_elems * sizeof(float));
        ds4_gpu_tensor* selected_view = ds4_gpu_tensor_view(
            g->batch_router_selected, (uint64_t)row0 * selected_row_bytes,
            (uint64_t)router_group_rows * selected_row_bytes);
        ds4_gpu_tensor* weights_view = ds4_gpu_tensor_view(
            g->batch_router_weights, (uint64_t)row0 * weights_row_bytes,
            (uint64_t)router_group_rows * weights_row_bytes);
        ds4_gpu_tensor* norm_view = ds4_gpu_tensor_view(
            g->batch_ffn_norm, (uint64_t)row0 * DS4_N_EMBD * sizeof(float),
            (uint64_t)router_group_rows * DS4_N_EMBD * sizeof(float));
        ok = out_view && gate_view && up_view && mid_view && down_view &&
             selected_view && weights_view && norm_view;
        bool group_mid_f16 = false;
        if (ok) {
          ok = ds4_gpu_routed_moe_batch_tensor(
                   out_view, gate_view, up_view, mid_view, down_view,
                   model->map, model->size, layer->ffn_gate_exps->abs_offset,
                   layer->ffn_up_exps->abs_offset,
                   layer->ffn_down_exps->abs_offset, layer->ffn_gate_exps->type,
                   layer->ffn_down_exps->type, gate_expert_bytes,
                   gate_row_bytes, down_expert_bytes, down_row_bytes,
                   static_cast<uint32_t>(expert_in_dim),
                   static_cast<uint32_t>(down_in_dim),
                   static_cast<uint32_t>(routed_out_dim), selected_view,
                   weights_view, DS4_N_EXPERT, DS4_N_EXPERT_USED,
                   DS4_SWIGLU_CLAMP_EXP, norm_view, il, router_group_rows,
                   &group_mid_f16, false) != 0;
        }
        all_mid_f16 = all_mid_f16 && group_mid_f16;
        ds4_gpu_tensor_free(norm_view);
        ds4_gpu_tensor_free(weights_view);
        ds4_gpu_tensor_free(selected_view);
        ds4_gpu_tensor_free(down_view);
        ds4_gpu_tensor_free(mid_view);
        ds4_gpu_tensor_free(up_view);
        ds4_gpu_tensor_free(gate_view);
        ds4_gpu_tensor_free(out_view);
      }
      g->batch_routed_mid_is_f16 = all_mid_f16;
    } else if (ok) {
      ok = ds4_gpu_routed_moe_batch_tensor(
               g->batch_routed_out, g->batch_routed_gate, g->batch_routed_up,
               g->batch_routed_mid, g->batch_routed_down, model->map,
               model->size, layer->ffn_gate_exps->abs_offset,
               layer->ffn_up_exps->abs_offset, layer->ffn_down_exps->abs_offset,
               layer->ffn_gate_exps->type, layer->ffn_down_exps->type,
               gate_expert_bytes, gate_row_bytes, down_expert_bytes,
               down_row_bytes, static_cast<uint32_t>(expert_in_dim),
               static_cast<uint32_t>(down_in_dim),
               static_cast<uint32_t>(routed_out_dim), g->batch_router_selected,
               g->batch_router_weights, DS4_N_EXPERT, DS4_N_EXPERT_USED,
               DS4_SWIGLU_CLAMP_EXP, g->batch_ffn_norm, il, n_tokens,
               &g->batch_routed_mid_is_f16, false) != 0;
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

    const bool routed_sum_deferred = ds4_gpu_routed_sum_deferred() != 0;
    ds4_gpu_set_routed_defer_sum(0);
    if (ok && routed_sum_deferred) {
        ok = ds4_gpu_hc_expand_add_split_moesum_tensor(next_hc_view,
                                                         g->batch_routed_down,
                                                         g->batch_shared_out,
                                                         g->batch_after_attn_hc,
                                                         hc_split_view,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_EXPERT_USED,
                                                         n_tokens) != 0;
        if (!ok) {
            fprintf(stderr,
                    "ds4: fused routed sum expansion rejected the shape; "
                    "set GUFO_DEEPSEEK_ROCM_FUSED_MOE_SUM=0\n");
        }
    } else if (ok) {
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
    /* Capture after the swap so batch_cur_hc holds this layer's output. */
    if (ok) ok = rocm_graph_dspark_capture_batch_layer(g, il, pos0, n_tokens);
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
    g->dspark_context_len = 0;
    g->dspark_capture_mask = 0;
    g->dspark_capture_batch_mask = 0;
    g->dspark_capture_batch_start = 0;
    g->dspark_capture_batch_tokens = 0;
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
        const uint32_t output_row = (uint32_t)n_tokens - 1u;
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
    const uint32_t output_row = (uint32_t)n_tokens - 1u;
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
    const uint32_t rows = (uint32_t)n_tokens;
    if (!rocm_graph_dspark_resize_capture(g, rows)) return false;
    bool ok = rocm_graph_prefill_layer_major(g,
                                             model,
                                             weights,
                                             prompt,
                                             0,
                                             rows,
                                             logits,
                                             show_progress);
    if (ok && g->dspark && g->dspark_capture_enabled) {
        ok = rocm_graph_dspark_capture_complete(g) &&
             rocm_graph_dspark_inject(g, 0, rows);
    }
    if (g->dspark && g->dspark_capture_enabled) {
        const uint32_t steady_rows = g->dspark->block_size + 1u;
        if (!rocm_graph_dspark_resize_capture(g, steady_rows)) ok = false;
    }
    return ok;
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
        bool ok = (!g->dspark_capture_enabled ||
                   rocm_graph_dspark_resize_capture(g, chunk)) &&
                  rocm_graph_prefill_layer_major(g,
                                                 model,
                                                 weights,
                                                 prompt,
                                                 pos0,
                                                 chunk,
                                                 chunk_logits,
                                                 show_progress);
        if (ok && g->dspark && g->dspark_capture_enabled) {
            ok = rocm_graph_dspark_capture_complete(g) &&
                 rocm_graph_dspark_inject(g, pos0, chunk);
        }
        if (!ok) {
            if (g->dspark && g->dspark_capture_enabled) {
                (void)rocm_graph_dspark_resize_capture(
                    g, g->dspark->block_size + 1u);
            }
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: ROCm synchronize after chunked prefill failure also failed\n");
            }
            return false;
        }
        pos0 = chunk_end;
    }
    if (g->dspark && g->dspark_capture_enabled &&
        !rocm_graph_dspark_resize_capture(
            g, g->dspark->block_size + 1u)) {
        return false;
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

/* =========================================================================
 * DSpark Speculative Verification.
 * =========================================================================
 *
 * Speculative decoding only pays off if the target can score a whole drafted
 * suffix for roughly the cost of one token.  DS4 decode is dominated by
 * streaming weights: the dense attention projections, shared expert, and LM
 * head are read once per step regardless of how many rows are in flight, and
 * only the routed experts grow with the row count.  Verification therefore
 * reuses the batched layer path rather than looping the one-token decode path.
 *
 * The verifier writes into the same persistent caches as prefill, so a rejected
 * block must be undone.  That is cheap here: raw and compressed rows are
 * append-only and get overwritten, so only the row counters and the in-place
 * compressor frontier need saving.
 */

static uint32_t rocm_graph_spec_rows_cap(const ds4_gpu_graph *g) {
    return g ? g->spec_rows_cap : 0u;
}

static bool rocm_graph_spec_prepare(ds4_gpu_graph *g,
                                    const ds4_weights *weights,
                                    uint32_t rows_cap) {
    if (!g || !weights || rows_cap == 0 || rows_cap > DS4_SPEC_MAX_ROWS) return false;
    if (rows_cap > g->prefill_cap) return false;
    if (g->spec_rows_cap >= rows_cap && g->spec_logits && g->spec_row_tops) return true;

    rocm_graph_spec_free(g);
    const uint64_t vocab_dim = weights->output->dim[1];
    g->spec_logits =
        ds4_gpu_tensor_alloc((uint64_t)rows_cap * vocab_dim * sizeof(float));
    g->spec_row_tops =
        ds4_gpu_tensor_alloc((uint64_t)rows_cap * sizeof(int32_t));
    bool ok = g->spec_logits != NULL && g->spec_row_tops != NULL;

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_bytes =
            (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
        g->spec_saved_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_bytes);
        g->spec_saved_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_bytes);
        g->spec_prefix_attn_state_kv[il] =
            ds4_gpu_tensor_alloc(DS4_SPEC_PREFIX_SLOTS * attn_bytes);
        g->spec_prefix_attn_state_score[il] =
            ds4_gpu_tensor_alloc(DS4_SPEC_PREFIX_SLOTS * attn_bytes);
        ok = g->spec_saved_attn_state_kv[il] != NULL &&
             g->spec_saved_attn_state_score[il] != NULL &&
             g->spec_prefix_attn_state_kv[il] != NULL &&
             g->spec_prefix_attn_state_score[il] != NULL;
        if (ok && ratio == 4) {
            const uint64_t index_bytes =
                (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
            g->spec_saved_index_state_kv[il] = ds4_gpu_tensor_alloc(index_bytes);
            g->spec_saved_index_state_score[il] = ds4_gpu_tensor_alloc(index_bytes);
            g->spec_prefix_index_state_kv[il] =
                ds4_gpu_tensor_alloc(DS4_SPEC_PREFIX_SLOTS * index_bytes);
            g->spec_prefix_index_state_score[il] =
                ds4_gpu_tensor_alloc(DS4_SPEC_PREFIX_SLOTS * index_bytes);
            ok = g->spec_saved_index_state_kv[il] != NULL &&
                 g->spec_saved_index_state_score[il] != NULL &&
                 g->spec_prefix_index_state_kv[il] != NULL &&
                 g->spec_prefix_index_state_score[il] != NULL;
        }
    }

    if (!ok) {
        fprintf(stderr, "ds4: ROCm failed to allocate DSpark verification state\n");
        rocm_graph_spec_free(g);
        return false;
    }
    g->spec_rows_cap = rows_cap;
    return true;
}

static bool rocm_graph_capture_prefix_attn_state(ds4_gpu_graph *g,
                                                 uint32_t il,
                                                 uint32_t slot) {
    if (!g->spec_capture_prefixes) return true;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (ratio == 0) return true;
    if (slot >= DS4_SPEC_PREFIX_SLOTS ||
        !g->spec_prefix_attn_state_kv[il] ||
        !g->spec_prefix_attn_state_score[il]) {
        return false;
    }
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    const uint64_t bytes =
        (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
    const uint64_t offset = (uint64_t)slot * bytes;
    g->spec_prefix_n_comp[slot][il] = g->layer_n_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix_attn_state_kv[il],
                               offset,
                               g->layer_attn_state_kv[il],
                               0,
                               bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix_attn_state_score[il],
                               offset,
                               g->layer_attn_state_score[il],
                               0,
                               bytes) != 0;
}

static bool rocm_graph_capture_prefix_index_state(ds4_gpu_graph *g,
                                                  uint32_t il,
                                                  uint32_t slot) {
    if (!g->spec_capture_prefixes) return true;
    if (ds4_layer_compress_ratio(il) != 4) return true;
    if (slot >= DS4_SPEC_PREFIX_SLOTS ||
        !g->spec_prefix_index_state_kv[il] ||
        !g->spec_prefix_index_state_score[il]) {
        return false;
    }
    constexpr uint32_t kCoff = 2u;
    constexpr uint32_t kRatio = 4u;
    const uint64_t bytes =
        (uint64_t)kCoff * DS4_N_INDEXER_HEAD_DIM * kCoff * kRatio *
        sizeof(float);
    const uint64_t offset = (uint64_t)slot * bytes;
    g->spec_prefix_n_index_comp[slot][il] =
        g->layer_n_index_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix_index_state_kv[il],
                               offset,
                               g->layer_index_state_kv[il],
                               0,
                               bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix_index_state_score[il],
                               offset,
                               g->layer_index_state_score[il],
                               0,
                               bytes) != 0;
}

/*
 * Attach a DSpark support model to this session's graph.
 *
 * Everything here is allocated only when a drafter is attached. The steady
 * capture buffers cover one verification block; prompt prefill grows them
 * temporarily and shrinks them again after seeding the support KV ring.
 */
static bool rocm_graph_dspark_attach(ds4_gpu_graph *g,
                                     const ds4_weights *weights,
                                     const ds4_dspark_model *dspark) {
    if (!g || !weights || !dspark) return false;
    if (g->dspark == dspark) return true;
    if (dspark->n_stages == 0 || dspark->n_stages > DS4_DSPARK_MAX_STAGES) return false;
    if (dspark->block_size == 0 || dspark->block_size > DS4_DSPARK_MAX_BLOCK) return false;

    rocm_graph_dspark_free(g);
    /* The verifier's row storage doubles as the draft block's logits. */
    if (!rocm_graph_spec_prepare(g, weights, dspark->block_size + 1u)) return false;

    g->dspark = dspark;
    g->dspark_cache_cap = g->raw_cap;
    g->dspark_capture_rows_cap = dspark->block_size + 1u;
    g->dspark_capture_enabled = true;
    g->dspark_context_len = 0;
    const uint32_t feature_width = dspark->n_target_layers * DS4_N_EMBD;

    bool ok = true;
    for (uint32_t stage = 0; ok && stage < dspark->n_stages; stage++) {
        g->dspark_kv_cache[stage] = ds4_gpu_tensor_alloc(
                (uint64_t)g->dspark_cache_cap * DS4_N_HEAD_DIM * sizeof(float));
        ok = g->dspark_kv_cache[stage] != NULL;
    }
    if (ok) {
        g->dspark_features = ds4_gpu_tensor_alloc(
                (uint64_t)g->dspark_capture_rows_cap * feature_width *
                    sizeof(float));
        g->dspark_fused = ds4_gpu_tensor_alloc(
                (uint64_t)g->dspark_capture_rows_cap * DS4_N_EMBD *
                    sizeof(float));
        g->dspark_fused_last =
                ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->dspark_block_tokens = ds4_gpu_tensor_alloc(
                (uint64_t)(DS4_DSPARK_MAX_BLOCK + 1u) * sizeof(int32_t));
        g->dspark_confidence = ds4_gpu_tensor_alloc(sizeof(float));
        g->dspark_markov_key = ds4_gpu_tensor_alloc(sizeof(unsigned long long));
        g->dspark_markov_index = ds4_gpu_tensor_alloc(sizeof(int32_t));
        ok = g->dspark_features && g->dspark_fused && g->dspark_fused_last &&
             g->dspark_block_tokens && g->dspark_confidence &&
             g->dspark_markov_key && g->dspark_markov_index;
    }
    if (!ok) {
        fprintf(stderr, "ds4: ROCm failed to allocate DSpark drafting state\n");
        rocm_graph_dspark_free(g);
        return false;
    }
    fprintf(stderr,
            "ds4: DSpark drafting attached stages=%u block=%u ring=%u rows "
            "features=%.1f MiB\n",
            dspark->n_stages,
            dspark->block_size,
            g->dspark_cache_cap,
            (double)((uint64_t)g->dspark_capture_rows_cap * feature_width *
                     sizeof(float)) /
                1048576.0);
    return true;
}

static bool rocm_graph_spec_copy_frontier(ds4_gpu_graph *g, bool save) {
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_bytes =
            (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
        ds4_gpu_tensor *live_kv = g->layer_attn_state_kv[il];
        ds4_gpu_tensor *live_score = g->layer_attn_state_score[il];
        ds4_gpu_tensor *saved_kv = g->spec_saved_attn_state_kv[il];
        ds4_gpu_tensor *saved_score = g->spec_saved_attn_state_score[il];
        if (!live_kv || !live_score || !saved_kv || !saved_score) return false;
        ds4_gpu_tensor *dst_kv = save ? saved_kv : live_kv;
        ds4_gpu_tensor *dst_score = save ? saved_score : live_score;
        const ds4_gpu_tensor *src_kv = save ? live_kv : saved_kv;
        const ds4_gpu_tensor *src_score = save ? live_score : saved_score;
        if (ds4_gpu_tensor_copy(dst_kv, 0, src_kv, 0, attn_bytes) == 0) return false;
        if (ds4_gpu_tensor_copy(dst_score, 0, src_score, 0, attn_bytes) == 0) return false;
        if (ratio != 4) continue;
        const uint64_t index_bytes =
            (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
        ds4_gpu_tensor *live_index_kv = g->layer_index_state_kv[il];
        ds4_gpu_tensor *live_index_score = g->layer_index_state_score[il];
        ds4_gpu_tensor *saved_index_kv = g->spec_saved_index_state_kv[il];
        ds4_gpu_tensor *saved_index_score = g->spec_saved_index_state_score[il];
        if (!live_index_kv || !live_index_score || !saved_index_kv ||
            !saved_index_score) {
            return false;
        }
        if (ds4_gpu_tensor_copy(save ? saved_index_kv : live_index_kv,
                                0,
                                save ? live_index_kv : saved_index_kv,
                                0,
                                index_bytes) == 0) {
            return false;
        }
        if (ds4_gpu_tensor_copy(save ? saved_index_score : live_index_score,
                                0,
                                save ? live_index_score : saved_index_score,
                                0,
                                index_bytes) == 0) {
            return false;
        }
    }
    return true;
}

static bool rocm_graph_spec_frontier_save(ds4_gpu_graph *g) {
    if (!g || g->spec_rows_cap == 0) return false;
    memcpy(g->spec_saved_n_comp, g->layer_n_comp, sizeof(g->spec_saved_n_comp));
    memcpy(g->spec_saved_n_index_comp,
           g->layer_n_index_comp,
           sizeof(g->spec_saved_n_index_comp));
    g->spec_frontier_valid = rocm_graph_spec_copy_frontier(g, true);
    return g->spec_frontier_valid;
}

static bool rocm_graph_spec_frontier_restore(ds4_gpu_graph *g) {
    if (!g || !g->spec_frontier_valid) return false;
    memcpy(g->layer_n_comp, g->spec_saved_n_comp, sizeof(g->layer_n_comp));
    memcpy(g->layer_n_index_comp,
           g->spec_saved_n_index_comp,
           sizeof(g->layer_n_index_comp));
    return rocm_graph_spec_copy_frontier(g, false);
}

static bool rocm_graph_spec_frontier_commit_prefix(ds4_gpu_graph *g,
                                                   uint32_t prefix_len) {
    if (!g || prefix_len == 0 || prefix_len > DS4_SPEC_PREFIX_SLOTS) {
        return false;
    }
    const uint32_t slot = prefix_len - 1u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_bytes =
            (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
        const uint64_t attn_offset = (uint64_t)slot * attn_bytes;
        g->layer_n_comp[il] = g->spec_prefix_n_comp[slot][il];
        if (ds4_gpu_tensor_copy(g->layer_attn_state_kv[il],
                                0,
                                g->spec_prefix_attn_state_kv[il],
                                attn_offset,
                                attn_bytes) == 0 ||
            ds4_gpu_tensor_copy(g->layer_attn_state_score[il],
                                0,
                                g->spec_prefix_attn_state_score[il],
                                attn_offset,
                                attn_bytes) == 0) {
            return false;
        }
        if (ratio != 4) continue;
        const uint64_t index_bytes =
            (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio *
            sizeof(float);
        const uint64_t index_offset = (uint64_t)slot * index_bytes;
        g->layer_n_index_comp[il] =
            g->spec_prefix_n_index_comp[slot][il];
        if (ds4_gpu_tensor_copy(g->layer_index_state_kv[il],
                                0,
                                g->spec_prefix_index_state_kv[il],
                                index_offset,
                                index_bytes) == 0 ||
            ds4_gpu_tensor_copy(g->layer_index_state_score[il],
                                0,
                                g->spec_prefix_index_state_score[il],
                                index_offset,
                                index_bytes) == 0) {
            return false;
        }
    }
    return true;
}

/* Batched LM head: one logits row per verification row.  This mirrors
 * rocm_graph_encode_output_head exactly; only the row count differs, and the
 * HC helpers derive their row count from the destination tensor size. */
static bool rocm_graph_encode_output_head_batch(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;
    if (n_tokens > g->spec_rows_cap) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *output_pre =
        ds4_gpu_tensor_view(g->batch_hc_mix, 0,
                            (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_weights =
        ds4_gpu_tensor_view(g->batch_hc_split, 0,
                            (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_embd =
        ds4_gpu_tensor_view(g->batch_ffn_cur, 0,
                            (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *output_norm =
        ds4_gpu_tensor_view(g->batch_ffn_norm, 0,
                            (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));

    bool ok = output_pre && output_weights && output_embd && output_norm;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                     g->batch_cur_hc,
                                                     (uint32_t)hc_dim,
                                                     n_tokens,
                                                     DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(output_pre,
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(output_weights,
                                                   output_pre,
                                                   model->map,
                                                   model->size,
                                                   weights->output_hc_scale->abs_offset,
                                                   weights->output_hc_base->abs_offset,
                                                   DS4_N_HC,
                                                   DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                                  g->batch_cur_hc,
                                                  output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                       output_embd,
                                                       model->map,
                                                       model->size,
                                                       weights->output_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->spec_logits,
                                              model->map,
                                              model->size,
                                              weights->output->abs_offset,
                                              DS4_N_EMBD,
                                              vocab_dim,
                                              output_norm,
                                              n_tokens) != 0;

    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}

/*
 * Advance independent sessions through one layer-synchronous narrow-row pass.
 *
 * Dense HC/Q/KV projections, attention output, FFN/MoE, and the LM head share
 * the coordinator's batch arena. The stateful attention middle remains
 * session-owned because each row has independent caches and may have a
 * different absolute position. C=1 never enters this function.
 */
static bool rocm_graph_eval_sessions_batch(
        const ds4_model       *model,
        const ds4_weights     *weights,
        const ds4_rocm_batch_item *items,
        size_t                 item_count) {
    if (!model || !weights || !items || item_count < 2u || item_count > 8u) {
        return false;
    }

    ds4_gpu_graph *coordinator = items[0].graph;
    if (!coordinator || item_count > coordinator->prefill_cap) return false;
    for (size_t row = 0; row < item_count; ++row) {
        if (!items[row].graph || !items[row].logits ||
            items[row].graph->raw_cap == 0) {
            return false;
        }
    }
    if (!rocm_graph_spec_prepare(
            coordinator, weights, static_cast<uint32_t>(item_count))) {
        return false;
    }
    for (size_t row = 0; row < item_count; ++row) {
      ds4_rocm_graph_dspark_capture_reset(items[row].graph);
    }

    int32_t tokens[8] = {};
    for (size_t row = 0; row < item_count; ++row) {
        tokens[row] = items[row].token;
    }
    if (ds4_gpu_tensor_write(coordinator->prefill_tokens,
                             0,
                             tokens,
                             item_count * sizeof(tokens[0])) == 0) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_row_bytes = hc_dim * sizeof(float);
    ds4_gpu_set_small_batch_mode(1);
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_embed_tokens_hc_tensor(
                 coordinator->batch_cur_hc,
                 coordinator->prefill_tokens,
                 model->map,
                 model->size,
                 weights->token_embd->abs_offset,
                 (uint32_t)weights->token_embd->dim[1],
                 static_cast<uint32_t>(item_count),
                 DS4_N_EMBD,
                 DS4_N_HC) != 0;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; ++il) {
        const ds4_layer_weights *layer = &weights->layer[il];
        ok = rocm_graph_encode_sessions_attention_batch(
                coordinator,
                model,
                layer,
                il,
                items,
                static_cast<uint32_t>(item_count));
        if (ok) {
            ok = rocm_graph_encode_layer_ffn_batch(
                    coordinator,
                    model,
                    layer,
                    il,
                    items[0].position,
                    static_cast<uint32_t>(item_count));
        }
        if (ok) {
            ds4_gpu_tensor *tmp = coordinator->batch_cur_hc;
            coordinator->batch_cur_hc = coordinator->batch_next_hc;
            coordinator->batch_next_hc = tmp;
        }
        for (size_t row = 0; ok && row < item_count; ++row) {
          ds4_gpu_tensor* row_hc = ds4_gpu_tensor_view(
              coordinator->batch_cur_hc, row * hc_row_bytes, hc_row_bytes);
          ok = row_hc != nullptr;
          if (ok) {
            ok = rocm_graph_dspark_capture_rows(items[row].graph, il, row_hc,
                                                1u);
          }
          ds4_gpu_tensor_free(row_hc);
        }
    }

    if (ok) {
        ok = rocm_graph_encode_output_head_batch(
                coordinator,
                model,
                weights,
                static_cast<uint32_t>(item_count),
                weights->output->dim[1]);
    }
    for (size_t row = 0; ok && row < item_count; ++row) {
        ok = ds4_gpu_tensor_copy(items[row].graph->cur_hc,
                                 0,
                                 coordinator->batch_cur_hc,
                                 row * hc_row_bytes,
                                 hc_row_bytes) != 0;
    }
    if (ok) {
        ok = ds4_gpu_end_commands() != 0;
    } else {
        (void)ds4_gpu_synchronize();
    }
    ds4_gpu_set_small_batch_mode(0);
    if (!ok) return false;

    const uint64_t logits_row_bytes =
        weights->output->dim[1] * sizeof(float);
    for (size_t row = 0; row < item_count; ++row) {
        if (ds4_gpu_tensor_read(coordinator->spec_logits,
                                row * logits_row_bytes,
                                items[row].logits,
                                logits_row_bytes) == 0) {
            return false;
        }
    }
    return true;
}

/*
 * Score `n_tokens` candidate tokens starting at `start` in one batched pass.
 *
 * On success `row_tops[i]` is the target's greedy continuation after candidate
 * row i, which is what the caller compares against candidate i+1.  The caller
 * owns the accept decision and must call rocm_graph_spec_rollback() when it
 * commits fewer rows than it verified.
 */
static bool rocm_graph_verify_suffix(ds4_gpu_graph* g, const ds4_model* model,
                                     const ds4_weights* weights,
                                     const token_vec* tokens, uint32_t start,
                                     uint32_t n_tokens, int32_t* row_tops,
                                     bool run_output_head = true) {
  if (!g || !model || !weights || !tokens || n_tokens == 0)
    return false;
  if (n_tokens > g->prefill_cap || n_tokens > g->spec_rows_cap)
    return false;
  if (!g->spec_logits || !g->spec_row_tops)
    return false;
  if (start > (uint32_t)tokens->len ||
      n_tokens > (uint32_t)tokens->len - start) {
    return false;
  }
  if (run_output_head && !row_tops)
    return false;

  if (!rocm_graph_upload_prompt_tokens(g->prefill_tokens, tokens, start,
                                       n_tokens)) {
    return false;
  }
  if (!rocm_graph_warmup_prefill_kernels(g, model, weights, n_tokens))
    return false;
  if (!rocm_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                              g->prefill_tokens, model, weights,
                                              tokens, start, n_tokens)) {
    return false;
  }

  const uint64_t vocab_dim = weights->output->dim[1];
  /*
   * Verification-block kernel selection is scoped to this pass. Prefill and
   * decode keep the routes they were tuned and baselined with, so attaching a
   * DSpark drafter cannot move their numbers.
   */
  const bool saved_capture_prefixes = g->spec_capture_prefixes;
  g->spec_capture_prefixes =
      n_tokens > 1u && n_tokens <= DS4_SPEC_PREFIX_SLOTS + 2u;
  ds4_gpu_set_small_batch_mode(1);
  bool ok = ds4_gpu_begin_commands() != 0;
  for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
    ok = rocm_graph_encode_layer_batch(g, model, &weights->layer[il], il, start,
                                       n_tokens);
  }
  if (ok && run_output_head) {
    ok = rocm_graph_encode_output_head_batch(g, model, weights, n_tokens,
                                             vocab_dim);
  }
  if (ok && run_output_head) {
    ok = ds4_gpu_spec_row_argmax_tensor(g->spec_row_tops, g->spec_logits,
                                        (uint32_t)vocab_dim, n_tokens) != 0;
  }
  if (ok) {
    ok = ds4_gpu_end_commands() != 0;
  } else {
    (void)ds4_gpu_synchronize();
  }
  ds4_gpu_set_small_batch_mode(0);
  g->spec_capture_prefixes = saved_capture_prefixes;
  if (!ok)
    return false;
  if (!run_output_head)
    return true;

  return ds4_gpu_tensor_read(g->spec_row_tops, 0, row_tops,
                             (uint64_t)n_tokens * sizeof(row_tops[0])) != 0;
}

/*
 * Verify one equally sized draft block per session.
 *
 * Attention remains session-local because it mutates independent raw,
 * compressed, and indexer caches. The much larger FFN/MoE and output-head
 * weights are shared across every verification row, so flatten the blocks and
 * stream those weights once per layer.
 */
static bool rocm_graph_verify_sessions_batch(ds4_engine* engine,
                                             const ds4_rocm_verify_item* items,
                                             size_t item_count) {
  if (!engine || !engine->model || !engine->weights || !items ||
      item_count < 2u || item_count > 8u) {
    return false;
  }

  ds4_gpu_graph* coordinator = items[0].graph;
  if (!coordinator || !coordinator->batch_dspark_verify_cur_hc ||
      !coordinator->batch_dspark_verify_after_attn_hc) {
    return false;
  }
  const uint32_t rows_per_session = items[0].n_tokens;
  if (rows_per_session == 0u)
    return false;
  const uint32_t total_rows =
      static_cast<uint32_t>(item_count) * rows_per_session;
  if (total_rows > coordinator->prefill_cap ||
      total_rows > coordinator->batch_dspark_verify_rows_cap) {
    return false;
  }

  std::vector<int32_t> flat_tokens(total_rows);
  std::vector<int32_t> flat_tops(total_rows);
  bool saved_capture_prefixes[8] = {};
  for (size_t index = 0; index < item_count; ++index) {
    const ds4_rocm_verify_item& item = items[index];
    if (!item.graph || !item.tokens || !item.row_tops ||
        item.n_tokens != rows_per_session ||
        item.start > static_cast<uint32_t>(item.tokens->len) ||
        item.n_tokens > static_cast<uint32_t>(item.tokens->len) - item.start) {
      return false;
    }
    saved_capture_prefixes[index] = item.graph->spec_capture_prefixes;
  }

  size_t prepared_count = 0;
  for (; prepared_count < item_count; ++prepared_count) {
    const ds4_rocm_verify_item& item = items[prepared_count];
    const uint32_t rows_cap =
        prepared_count == 0u ? total_rows : rows_per_session;
    if (!rocm_graph_spec_prepare(item.graph, engine->weights, rows_cap) ||
        !rocm_graph_spec_frontier_save(item.graph)) {
      break;
    }
    item.graph->spec_capture_prefixes =
        rows_per_session > 1u && rows_per_session <= DS4_SPEC_PREFIX_SLOTS + 2u;
    ds4_rocm_graph_dspark_capture_reset(item.graph);
    for (uint32_t row = 0; row < rows_per_session; ++row) {
      flat_tokens[prepared_count * rows_per_session + row] =
          item.tokens->v[item.start + row];
    }
  }
  if (prepared_count != item_count) {
    for (size_t index = 0; index < prepared_count; ++index) {
      items[index].graph->spec_capture_prefixes = saved_capture_prefixes[index];
    }
    return false;
  }

  const bool serial_layers =
      getenv("GUFO_DEEPSEEK_DSPARK_BATCH_SERIAL_LAYERS") != NULL;
  const bool combine_output_head =
      getenv("GUFO_DEEPSEEK_DSPARK_BATCH_HEAD") != NULL;
  const bool interleave =
      getenv("GUFO_DEEPSEEK_DSPARK_BATCH_INTERLEAVE") != NULL ||
      (item_count == 2u && getenv("GUFO_DEEPSEEK_DSPARK_BATCH_DIRECT") == NULL);
  if ((!interleave && !serial_layers) ||
      (serial_layers && !combine_output_head)) {
    bool ok = true;
    for (size_t index = 0; ok && index < item_count; ++index) {
      const ds4_rocm_verify_item& item = items[index];
      ok = rocm_graph_verify_suffix(item.graph, engine->model, engine->weights,
                                    item.tokens, item.start, item.n_tokens,
                                    item.row_tops);
    }
    for (size_t index = 0; index < item_count; ++index) {
      items[index].graph->spec_capture_prefixes = saved_capture_prefixes[index];
    }
    return ok;
  }

  const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
  const uint64_t hc_row_bytes = hc_dim * sizeof(float);
  if (serial_layers) {
    bool ok = true;
    for (size_t index = 0; ok && index < item_count; ++index) {
      const ds4_rocm_verify_item& item = items[index];
      ok = rocm_graph_verify_suffix(item.graph, engine->model, engine->weights,
                                    item.tokens, item.start, item.n_tokens,
                                    nullptr, false);
      if (ok) {
        ok = ds4_gpu_tensor_copy(
                 coordinator->batch_dspark_verify_cur_hc,
                 (uint64_t)index * rows_per_session * hc_row_bytes,
                 item.graph->batch_cur_hc, 0,
                 (uint64_t)rows_per_session * hc_row_bytes) != 0;
      }
    }
    ds4_gpu_set_small_batch_mode(1);
    if (ok)
      ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
      ok = ds4_gpu_tensor_copy(coordinator->batch_cur_hc, 0,
                               coordinator->batch_dspark_verify_cur_hc, 0,
                               (uint64_t)total_rows * hc_row_bytes) != 0;
    }
    if (ok) {
      ok = rocm_graph_encode_output_head_batch(coordinator, engine->model,
                                               engine->weights, total_rows,
                                               engine->weights->output->dim[1]);
    }
    if (ok) {
      ok = ds4_gpu_spec_row_argmax_tensor(
               coordinator->spec_row_tops, coordinator->spec_logits,
               static_cast<uint32_t>(engine->weights->output->dim[1]),
               total_rows) != 0;
    }
    if (ok) {
      ok = ds4_gpu_end_commands() != 0;
    } else {
      (void)ds4_gpu_synchronize();
    }
    ds4_gpu_set_small_batch_mode(0);
    for (size_t index = 0; index < item_count; ++index) {
      items[index].graph->spec_capture_prefixes = saved_capture_prefixes[index];
    }
    if (!ok ||
        ds4_gpu_tensor_read(coordinator->spec_row_tops, 0, flat_tops.data(),
                            (uint64_t)total_rows * sizeof(flat_tops[0])) == 0) {
      return false;
    }

    const uint64_t logits_row_bytes =
        engine->weights->output->dim[1] * sizeof(float);
    for (size_t index = 0; index < item_count; ++index) {
      if (index != 0u &&
          ds4_gpu_tensor_copy(
              items[index].graph->spec_logits, 0, coordinator->spec_logits,
              (uint64_t)index * rows_per_session * logits_row_bytes,
              (uint64_t)rows_per_session * logits_row_bytes) == 0) {
        return false;
      }
      memcpy(items[index].row_tops, flat_tops.data() + index * rows_per_session,
             (size_t)rows_per_session * sizeof(items[index].row_tops[0]));
    }
    return true;
  }

  const bool combine_ffn = getenv("GUFO_DEEPSEEK_DSPARK_BATCH_FFN") != NULL ||
                           (item_count == 2u && !serial_layers);
  const bool sync_layers =
      getenv("GUFO_DEEPSEEK_DSPARK_BATCH_SYNC_LAYERS") != NULL;
  bool ok = true;
  if (ok) {
    ok = ds4_gpu_tensor_write(
             coordinator->prefill_tokens, 0, flat_tokens.data(),
             (uint64_t)total_rows * sizeof(flat_tokens[0])) != 0;
  }
  if (ok && !serial_layers) {
    ok = ds4_gpu_embed_tokens_hc_tensor(
             coordinator->batch_dspark_verify_cur_hc,
             coordinator->prefill_tokens, engine->model->map,
             engine->model->size, engine->weights->token_embd->abs_offset,
             static_cast<uint32_t>(engine->weights->token_embd->dim[1]),
             total_rows, DS4_N_EMBD, DS4_N_HC) != 0;
  }

  ds4_gpu_set_small_batch_mode(1);
  if (ok)
    ok = ds4_gpu_begin_commands() != 0;
  if (serial_layers) {
    for (size_t index = 0; ok && index < item_count; ++index) {
      ds4_gpu_graph* graph = items[index].graph;
      ok = ds4_gpu_embed_tokens_hc_tensor(
               graph->batch_cur_hc, graph->prefill_tokens, engine->model->map,
               engine->model->size, engine->weights->token_embd->abs_offset,
               static_cast<uint32_t>(engine->weights->token_embd->dim[1]),
               rows_per_session, DS4_N_EMBD, DS4_N_HC) != 0;
      for (uint32_t il = 0; ok && il < DS4_N_LAYER; ++il) {
        ok = rocm_graph_encode_layer_batch(
            graph, engine->model, &engine->weights->layer[il], il,
            items[index].start, rows_per_session);
      }
      if (ok && combine_output_head) {
        ok = ds4_gpu_tensor_copy(
                 coordinator->batch_dspark_verify_cur_hc,
                 (uint64_t)index * rows_per_session * hc_row_bytes,
                 graph->batch_cur_hc, 0,
                 (uint64_t)rows_per_session * hc_row_bytes) != 0;
      }
      if (ok && !combine_output_head) {
        ok = rocm_graph_encode_output_head_batch(
            graph, engine->model, engine->weights, rows_per_session,
            engine->weights->output->dim[1]);
      }
      if (ok && !combine_output_head) {
        ok = ds4_gpu_spec_row_argmax_tensor(
                 graph->spec_row_tops, graph->spec_logits,
                 static_cast<uint32_t>(engine->weights->output->dim[1]),
                 rows_per_session) != 0;
      }
      if (ok) {
        ok = ds4_gpu_flush_commands() != 0;
      }
    }
  }
  for (uint32_t il = 0; ok && !serial_layers && il < DS4_N_LAYER; ++il) {
    const ds4_layer_weights* layer = &engine->weights->layer[il];
    for (size_t index = 0; ok && index < item_count; ++index) {
      ds4_gpu_graph* graph = items[index].graph;
      const uint64_t block_offset =
          (uint64_t)index * rows_per_session * hc_row_bytes;
      const uint64_t block_bytes = (uint64_t)rows_per_session * hc_row_bytes;
      ok = ds4_gpu_tensor_copy(graph->batch_cur_hc, 0,
                               coordinator->batch_dspark_verify_cur_hc,
                               block_offset, block_bytes) != 0;
      if (ok) {
        ok = rocm_graph_encode_layer_attention_batch(
            graph, engine->model, layer, il, items[index].start,
            rows_per_session);
      }
      if (ok) {
        if (combine_ffn) {
          ok = ds4_gpu_tensor_copy(
                   coordinator->batch_dspark_verify_after_attn_hc, block_offset,
                   graph->batch_after_attn_hc, 0, block_bytes) != 0;
        } else {
          ds4_gpu_tensor* session_tokens = ds4_gpu_tensor_view(
              coordinator->prefill_tokens,
              (uint64_t)index * rows_per_session * sizeof(flat_tokens[0]),
              (uint64_t)rows_per_session * sizeof(flat_tokens[0]));
          ok = session_tokens != nullptr;
          if (ok) {
            ok = rocm_graph_encode_layer_ffn_batch(
                graph, engine->model, layer, il, items[index].start,
                rows_per_session, session_tokens);
          }
          ds4_gpu_tensor_free(session_tokens);
        }
      }
      if (ok && !combine_ffn) {
        ds4_gpu_tensor* tmp = graph->batch_cur_hc;
        graph->batch_cur_hc = graph->batch_next_hc;
        graph->batch_next_hc = tmp;
        ok = rocm_graph_dspark_capture_rows(graph, il, graph->batch_cur_hc,
                                            rows_per_session);
      }
      if (ok && !combine_ffn) {
        ok = ds4_gpu_tensor_copy(coordinator->batch_dspark_verify_after_attn_hc,
                                 block_offset, graph->batch_cur_hc, 0,
                                 block_bytes) != 0;
      }
      if (ok && sync_layers) {
        ok = ds4_gpu_flush_commands() != 0;
      }
    }
    if (ok && combine_ffn) {
      ok = ds4_gpu_tensor_copy(coordinator->batch_after_attn_hc, 0,
                               coordinator->batch_dspark_verify_after_attn_hc,
                               0, (uint64_t)total_rows * hc_row_bytes) != 0;
    }
    if (ok && combine_ffn) {
      ok = rocm_graph_encode_layer_ffn_batch(
          coordinator, engine->model, layer, il, items[0].start, total_rows,
          coordinator->prefill_tokens, rows_per_session);
    }
    for (size_t index = 0; ok && combine_ffn && index < item_count; ++index) {
      const uint64_t block_offset =
          (uint64_t)index * rows_per_session * hc_row_bytes;
      ds4_gpu_tensor* block_hc =
          ds4_gpu_tensor_view(coordinator->batch_next_hc, block_offset,
                              (uint64_t)rows_per_session * hc_row_bytes);
      ok = block_hc != NULL;
      if (ok) {
        ok = rocm_graph_dspark_capture_rows(items[index].graph, il, block_hc,
                                            rows_per_session);
      }
      ds4_gpu_tensor_free(block_hc);
    }
    if (ok) {
      ok = ds4_gpu_tensor_copy(
               coordinator->batch_dspark_verify_cur_hc, 0,
               combine_ffn ? coordinator->batch_next_hc
                           : coordinator->batch_dspark_verify_after_attn_hc,
               0, (uint64_t)total_rows * hc_row_bytes) != 0;
    }
    if (ok && sync_layers) {
      ok = ds4_gpu_flush_commands() != 0;
    }
  }
  if (ok && combine_output_head) {
    ok = ds4_gpu_tensor_copy(coordinator->batch_cur_hc, 0,
                             coordinator->batch_dspark_verify_cur_hc, 0,
                             (uint64_t)total_rows * hc_row_bytes) != 0;
  }
  if (ok && combine_output_head) {
    ok = rocm_graph_encode_output_head_batch(coordinator, engine->model,
                                             engine->weights, total_rows,
                                             engine->weights->output->dim[1]);
  }
  if (ok && combine_output_head) {
    ok = ds4_gpu_spec_row_argmax_tensor(
             coordinator->spec_row_tops, coordinator->spec_logits,
             static_cast<uint32_t>(engine->weights->output->dim[1]),
             total_rows) != 0;
  }
  for (size_t index = 0;
       ok && !combine_output_head && !serial_layers && index < item_count;
       ++index) {
    ds4_gpu_graph* graph = items[index].graph;
    const uint64_t block_offset =
        (uint64_t)index * rows_per_session * hc_row_bytes;
    const uint64_t block_bytes = (uint64_t)rows_per_session * hc_row_bytes;
    ok = ds4_gpu_tensor_copy(graph->batch_cur_hc, 0,
                             coordinator->batch_dspark_verify_cur_hc,
                             block_offset, block_bytes) != 0;
    if (ok) {
      ok = rocm_graph_encode_output_head_batch(
          graph, engine->model, engine->weights, rows_per_session,
          engine->weights->output->dim[1]);
    }
    if (ok) {
      ok = ds4_gpu_spec_row_argmax_tensor(
               graph->spec_row_tops, graph->spec_logits,
               static_cast<uint32_t>(engine->weights->output->dim[1]),
               rows_per_session) != 0;
    }
  }
  if (ok) {
    ok = ds4_gpu_end_commands() != 0;
  } else {
    (void)ds4_gpu_synchronize();
  }
  ds4_gpu_set_small_batch_mode(0);
  for (size_t index = 0; index < item_count; ++index) {
    items[index].graph->spec_capture_prefixes = saved_capture_prefixes[index];
  }
  if (!ok)
    return false;

  const uint64_t logits_row_bytes =
      engine->weights->output->dim[1] * sizeof(float);
  if (combine_output_head &&
      ds4_gpu_tensor_read(coordinator->spec_row_tops, 0, flat_tops.data(),
                          (uint64_t)total_rows * sizeof(flat_tops[0])) == 0) {
    return false;
  }
  for (size_t index = 0; index < item_count; ++index) {
    if (combine_output_head && index != 0u &&
        ds4_gpu_tensor_copy(
            items[index].graph->spec_logits, 0, coordinator->spec_logits,
            (uint64_t)index * rows_per_session * logits_row_bytes,
            (uint64_t)rows_per_session * logits_row_bytes) == 0) {
      return false;
    }
    if (combine_output_head) {
      memcpy(items[index].row_tops, flat_tops.data() + index * rows_per_session,
             (size_t)rows_per_session * sizeof(items[index].row_tops[0]));
    } else if (ds4_gpu_tensor_read(items[index].graph->spec_row_tops, 0,
                                   items[index].row_tops,
                                   (uint64_t)rows_per_session *
                                       sizeof(items[index].row_tops[0])) == 0) {
      return false;
    }
  }
  return true;
}

static bool rocm_graph_read_spec_logits_row(const ds4_gpu_graph *g,
                                            uint32_t row,
                                            float *logits) {
    if (!g || !g->spec_logits || !logits || row >= g->spec_rows_cap) return false;
    const uint64_t row_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    return ds4_gpu_tensor_read(g->spec_logits,
                                (uint64_t)row * row_bytes,
                                logits,
                                row_bytes) != 0;
}

/* =========================================================================
 * DSpark Drafting.
 * =========================================================================
 *
 * One draft pass proposes a whole block. The target's hidden states at the
 * sampled layers are projected into each stage's key/value space and appended to
 * that stage's ring, so the three-block drafter tracks the forty-three-block
 * target without re-running it. The block itself enters as
 * [last_committed_token, MASK, ...] and every block row attends to the injected
 * ring *and* to all other block rows, which is what lets one forward pass
 * produce `block_size` candidates instead of `block_size` sequential steps.
 */


static uint32_t rocm_graph_dspark_feature_row_stride(const ds4_gpu_graph *g) {
    return g->dspark ? g->dspark->n_target_layers * DS4_N_EMBD : 0u;
}

/* Feature slot for a target layer, or -1 when this layer is not sampled. */
static int rocm_graph_dspark_slot(const ds4_gpu_graph *g, uint32_t il) {
    if (!g->dspark) return -1;
    for (uint32_t i = 0; i < g->dspark->n_target_layers; i++) {
        if (g->dspark->target_layer_ids[i] == il) return (int)i;
    }
    return -1;
}

static uint32_t rocm_graph_dspark_complete_mask(const ds4_gpu_graph *g) {
    if (!g->dspark || g->dspark->n_target_layers >= 32u) return 0u;
    return (1u << g->dspark->n_target_layers) - 1u;
}

static bool rocm_graph_dspark_capture_rows(ds4_gpu_graph *g,
                                           uint32_t il,
                                           const ds4_gpu_tensor *hc,
                                           uint32_t n_rows) {
    int slot = rocm_graph_dspark_slot(g, il);
    if (slot < 0) return true;
    if (!g->dspark_features || n_rows == 0 ||
        n_rows > g->dspark_capture_rows_cap) {
        return false;
    }
    const uint32_t stride = rocm_graph_dspark_feature_row_stride(g);
    if (!ds4_gpu_dspark_capture_features_tensor(g->dspark_features,
                                                hc,
                                                stride,
                                                (uint32_t)slot * DS4_N_EMBD,
                                                DS4_N_EMBD,
                                                DS4_N_HC,
                                                n_rows)) {
        return false;
    }
    g->dspark_capture_mask |= 1u << (uint32_t)slot;
    return true;
}

/* Called from the single-token decode path once layer `il` has updated cur_hc. */
static bool rocm_graph_dspark_capture_decode_layer(ds4_gpu_graph *g, uint32_t il) {
    if (!g->dspark || !g->dspark_capture_enabled) return true;
    return rocm_graph_dspark_capture_rows(g, il, g->cur_hc, 1u);
}

/* Called from the batched path (prefill and verification) per layer. */
static bool rocm_graph_dspark_capture_batch_layer(ds4_gpu_graph *g,
                                                  uint32_t il,
                                                  uint32_t start,
                                                  uint32_t n_tokens) {
    if (!g->dspark || !g->dspark_capture_enabled) return true;
    const int slot = rocm_graph_dspark_slot(g, il);
    if (slot < 0) return true;
    /* A different (start, count) means a new batch has begun - a verification
     * block after a prompt chunk, say - so start a fresh capture window rather
     * than treating it as an inconsistency. */
    if (g->dspark_capture_batch_mask == 0u ||
        g->dspark_capture_batch_start != start ||
        g->dspark_capture_batch_tokens != n_tokens) {
        g->dspark_capture_batch_mask = 0u;
        g->dspark_capture_mask = 0u;
        g->dspark_capture_batch_start = start;
        g->dspark_capture_batch_tokens = n_tokens;
    }
    if (!rocm_graph_dspark_capture_rows(g, il, g->batch_cur_hc, n_tokens)) return false;
    g->dspark_capture_batch_mask |= 1u << (uint32_t)slot;
    return true;
}

static bool rocm_graph_dspark_capture_complete(const ds4_gpu_graph *g) {
    if (!g || !g->dspark_capture_enabled) return false;
    const uint32_t want = rocm_graph_dspark_complete_mask(g);
    return want != 0u && g->dspark_capture_mask == want;
}

/*
 * Project captured target features into every stage's ring at absolute positions
 * [pos0, pos0 + n_rows). Row r of dspark_features must hold position pos0 + r.
 */
static bool rocm_graph_dspark_inject(ds4_gpu_graph *g,
                                     uint32_t pos0,
                                     uint32_t n_rows) {
    const ds4_dspark_model *d = g->dspark;
    if (!d || n_rows == 0 || n_rows > g->dspark_capture_rows_cap) return false;
    if (!g->dspark_features || !g->dspark_fused) return false;
    const ds4_model *sm = d->model;
    const ds4_dspark_stage_weights *first = &d->stage[0];
    const uint32_t feature_width = rocm_graph_dspark_feature_row_stride(g);

    ds4_gpu_tensor *fused = ds4_gpu_tensor_view(
            g->dspark_fused, 0, (uint64_t)n_rows * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *kv_raw = ds4_gpu_tensor_view(
            g->batch_kv_raw, 0, (uint64_t)n_rows * DS4_N_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *kv = ds4_gpu_tensor_view(
            g->batch_kv, 0, (uint64_t)n_rows * DS4_N_HEAD_DIM * sizeof(float));
    bool ok = fused && kv_raw && kv;

    /* Fuse the sampled layers once; every stage reads the same fused feature. */
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(fused,
                                             sm->map,
                                             sm->size,
                                             first->main_proj->abs_offset,
                                             feature_width,
                                             DS4_N_EMBD,
                                             g->dspark_features,
                                             n_rows) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->dspark_fused,
                                                      fused,
                                                      sm->map,
                                                      sm->size,
                                                      first->main_norm->abs_offset,
                                                      DS4_N_EMBD,
                                                      n_rows,
                                                      DS4_RMS_EPS) != 0;

    for (uint32_t stage = 0; ok && stage < d->n_stages; stage++) {
        const ds4_layer_weights *layer = &d->stage[stage].block;
        ok = ds4_gpu_matmul_q8_0_tensor(kv_raw,
                                         sm->map,
                                         sm->size,
                                         layer->attn_kv->abs_offset,
                                         DS4_N_EMBD,
                                         DS4_N_HEAD_DIM,
                                         fused,
                                         n_rows) != 0;
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(kv,
                                                          kv_raw,
                                                          sm->map,
                                                          sm->size,
                                                          layer->attn_kv_a_norm->abs_offset,
                                                          DS4_N_HEAD_DIM,
                                                          n_rows,
                                                          DS4_RMS_EPS) != 0;
        if (ok) ok = ds4_gpu_rope_tail_tensor(kv,
                                               n_rows,
                                               DS4_N_HEAD_KV,
                                               DS4_N_HEAD_DIM,
                                               DS4_N_ROT,
                                               pos0,
                                               0,
                                               false,
                                               DS4_ROPE_FREQ_BASE,
                                               1.0f,
                                               0.0f,
                                               1.0f,
                                               DS4_ROPE_YARN_BETA_FAST,
                                               DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv,
                                                          n_rows,
                                                          DS4_N_HEAD_DIM,
                                                          DS4_N_ROT) != 0;
        if (ok) ok = ds4_gpu_store_raw_kv_batch_tensor(g->dspark_kv_cache[stage],
                                                        kv,
                                                        g->dspark_cache_cap,
                                                        pos0,
                                                        n_rows,
                                                        DS4_N_HEAD_DIM) != 0;
    }

    /* Keep the newest row: the next draft uses it as its stage-input row 0. */
    if (ok) {
        ok = ds4_gpu_tensor_copy(g->dspark_fused_last,
                                 0,
                                 g->dspark_fused,
                                 (uint64_t)(n_rows - 1u) * DS4_N_EMBD * sizeof(float),
                                 (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
    }
    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(kv_raw);
    ds4_gpu_tensor_free(fused);
    /*
     * Context coverage is contiguous from position zero. A snapshot restores
     * only the target cache, so a later one-row injection must not claim that
     * the absent support prefix is initialized.
     */
    if (ok && pos0 <= g->dspark_context_len &&
        pos0 + n_rows > g->dspark_context_len) {
        g->dspark_context_len = pos0 + n_rows;
    }
    return ok;
}

/*
 * One DSpark stage over the draft block: the DS4 block with the causal window
 * replaced by non-causal attention over the injected ring plus the block itself.
 * The FFN half is shared with the target's batched path, which is safe because
 * that half is driven entirely by the supplied weights.
 */
[[maybe_unused]] static bool rocm_graph_encode_dspark_stage_legacy(
        ds4_gpu_graph *g,
        uint32_t stage,
        uint32_t pos0,
        uint32_t n_rows) {
    const ds4_dspark_model *d = g->dspark;
    if (!d || stage >= d->n_stages || n_rows == 0) return false;
    const ds4_model *sm = d->model;
    const ds4_layer_weights *layer = &d->stage[stage].block;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t group_heads = DS4_N_HEAD / DS4_N_OUT_GROUP;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;

    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_rows * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_rows * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
            g->batch_attn_cur, 0, (uint64_t)n_rows * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(
            g->batch_after_attn_hc, 0, (uint64_t)n_rows * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;

    /*
     * Row 0 is the encoder row: the fused target feature, re-supplied at every
     * stage rather than carried from the previous stage's output. It contributes
     * keys and values so the drafted rows can attend to the target's current
     * state directly.
     */
    if (ok) ok = ds4_gpu_dspark_repeat_hc_tensor(g->batch_cur_hc,
                                                  g->dspark_fused_last,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC,
                                                  1u) != 0;

    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                     g->batch_cur_hc,
                                                     (uint32_t)hc_dim,
                                                     n_rows,
                                                     DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(hc_mix_view,
                                            sm->map,
                                            sm->size,
                                            layer->hc_attn_fn->abs_offset,
                                            hc_dim,
                                            mix_hc,
                                            g->batch_flat_hc,
                                            n_rows) != 0;
    if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(attn_cur_view,
                                                       hc_split_view,
                                                       hc_mix_view,
                                                       g->batch_cur_hc,
                                                       sm->map,
                                                       sm->size,
                                                       layer->hc_attn_scale->abs_offset,
                                                       layer->hc_attn_base->abs_offset,
                                                       DS4_N_EMBD,
                                                       DS4_N_HC,
                                                       DS4_N_HC_SINKHORN_ITER,
                                                       DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm,
                                                      g->batch_attn_cur,
                                                      sm->map,
                                                      sm->size,
                                                      layer->attn_norm->abs_offset,
                                                      DS4_N_EMBD,
                                                      n_rows,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_qr,
                                             sm->map,
                                             sm->size,
                                             layer->attn_q_a->abs_offset,
                                             DS4_N_EMBD,
                                             q_rank,
                                             g->batch_attn_norm,
                                             n_rows) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_kv_raw,
                                             sm->map,
                                             sm->size,
                                             layer->attn_kv->abs_offset,
                                             DS4_N_EMBD,
                                             DS4_N_HEAD_DIM,
                                             g->batch_attn_norm,
                                             n_rows) != 0;
    if (ok) ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->batch_qr_norm,
                                                        g->batch_qr,
                                                        sm->map,
                                                        sm->size,
                                                        layer->attn_q_a_norm->abs_offset,
                                                        (uint32_t)q_rank,
                                                        g->batch_kv,
                                                        g->batch_kv_raw,
                                                        layer->attn_kv_a_norm->abs_offset,
                                                        DS4_N_HEAD_DIM,
                                                        n_rows,
                                                        DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_q,
                                             sm->map,
                                             sm->size,
                                             layer->attn_q_b->abs_offset,
                                             q_rank,
                                             q_dim,
                                             g->batch_qr_norm,
                                             n_rows) != 0;
    if (ok) ok = ds4_gpu_head_rms_norm_tensor(g->batch_q,
                                               n_rows,
                                               DS4_N_HEAD,
                                               DS4_N_HEAD_DIM,
                                               DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_q,
                                           n_rows,
                                           DS4_N_HEAD,
                                           DS4_N_HEAD_DIM,
                                           DS4_N_ROT,
                                           pos0,
                                           0,
                                           false,
                                           DS4_ROPE_FREQ_BASE,
                                           1.0f,
                                           0.0f,
                                           1.0f,
                                           DS4_ROPE_YARN_BETA_FAST,
                                           DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_kv,
                                           n_rows,
                                           DS4_N_HEAD_KV,
                                           DS4_N_HEAD_DIM,
                                           DS4_N_ROT,
                                           pos0,
                                           0,
                                           false,
                                           DS4_ROPE_FREQ_BASE,
                                           1.0f,
                                           0.0f,
                                           1.0f,
                                           DS4_ROPE_YARN_BETA_FAST,
                                           DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->batch_kv,
                                                      n_rows,
                                                      DS4_N_HEAD_DIM,
                                                      DS4_N_ROT) != 0;
    /* The block's own keys and values occupy the ring rows the block will be
     * verified at; ring maintenance overwrites them with injected features once
     * the target commits. */
    if (ok) ok = ds4_gpu_store_raw_kv_batch_tensor(g->dspark_kv_cache[stage],
                                                    g->batch_kv,
                                                    g->dspark_cache_cap,
                                                    pos0,
                                                    n_rows,
                                                    DS4_N_HEAD_DIM) != 0;

    if (ok) {
        const uint32_t visible = pos0 + n_rows;
        const uint32_t n_raw =
            visible < g->dspark_cache_cap ? visible : g->dspark_cache_cap;
        const uint32_t raw_start = (visible - n_raw) % g->dspark_cache_cap;
        ok = ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
                g->batch_heads,
                sm->map,
                sm->size,
                layer->attn_sinks->abs_offset,
                g->batch_q,
                g->dspark_kv_cache[stage],
                n_rows,
                n_raw,
                g->dspark_cache_cap,
                raw_start,
                DS4_N_HEAD,
                DS4_N_HEAD_DIM) != 0;
    }
    if (ok) ok = ds4_gpu_attention_output_q8_batch_tensor(g->batch_attn_out,
                                                           g->batch_attn_low,
                                                           g->batch_group_tmp,
                                                           g->batch_low_tmp,
                                                           sm->map,
                                                           sm->size,
                                                           layer->attn_output_a->abs_offset,
                                                           layer->attn_output_b->abs_offset,
                                                           group_dim,
                                                           DS4_N_LORA_O,
                                                           DS4_N_OUT_GROUP,
                                                           DS4_N_EMBD,
                                                           g->batch_heads,
                                                           n_rows) != 0;
    if (ok) ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                                 g->batch_attn_out,
                                                 g->batch_cur_hc,
                                                 hc_split_view,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;

    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    if (!ok) return false;

    /* The FFN half is weight-driven, so the target's batched implementation
     * runs the support model's experts without modification. */
    if (!rocm_graph_encode_layer_ffn_batch(g, sm, layer, stage, pos0, n_rows)) {
        return false;
    }
    ds4_gpu_tensor *tmp = g->batch_cur_hc;
    g->batch_cur_hc = g->batch_next_hc;
    g->batch_next_hc = tmp;
    return true;
}

/*
 * DSpark's trained block has one encoder row plus `block_size` draft rows.
 *
 * The encoder contributes K/V at the same logical position as draft row zero,
 * but it is not itself queried or sent through the FFN. Keeping the row
 * contract explicit matters: treating all block_size + 1 rows as ordinary
 * token rows shifts every draft's RoPE position and changes the support model
 * being evaluated.
 */
static bool rocm_graph_encode_dspark_stage_reference(ds4_gpu_graph *g,
                                                     uint32_t stage,
                                                     uint32_t pos0,
                                                     uint32_t n_rows) {
    const ds4_dspark_model *d = g->dspark;
    if (!d || stage >= d->n_stages || n_rows < 2u) return false;
    const uint32_t draft_rows = n_rows - 1u;
    const ds4_model *sm = d->model;
    const ds4_layer_weights *layer = &d->stage[stage].block;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc =
        2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t group_heads = DS4_N_HEAD / DS4_N_OUT_GROUP;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;

    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
        g->batch_hc_mix, 0, (uint64_t)n_rows * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
        g->batch_hc_split, 0, (uint64_t)n_rows * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
        g->batch_attn_cur,
        0,
        (uint64_t)n_rows * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *draft_attn_norm_view = ds4_gpu_tensor_view(
        g->batch_attn_norm,
        (uint64_t)DS4_N_EMBD * sizeof(float),
        (uint64_t)draft_rows * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *draft_hc_view = ds4_gpu_tensor_view(
        g->batch_cur_hc,
        hc_dim * sizeof(float),
        (uint64_t)draft_rows * hc_dim * sizeof(float));
    ds4_gpu_tensor *draft_hc_split_view = ds4_gpu_tensor_view(
        g->batch_hc_split,
        mix_hc * sizeof(float),
        (uint64_t)draft_rows * mix_hc * sizeof(float));
    ds4_gpu_tensor *kv_target_view = ds4_gpu_tensor_view(
        g->batch_kv, 0, (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *kv_draft_view = ds4_gpu_tensor_view(
        g->batch_kv,
        (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
        (uint64_t)draft_rows * DS4_N_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(
        g->batch_after_attn_hc,
        0,
        (uint64_t)draft_rows * hc_dim * sizeof(float));
    bool ok =
        hc_mix_view && hc_split_view && attn_cur_view &&
        draft_attn_norm_view && draft_hc_view && draft_hc_split_view &&
        kv_target_view && kv_draft_view && after_attn_hc_view;

    if (ok) {
        ok = ds4_gpu_dspark_repeat_hc_tensor(g->batch_cur_hc,
                                             g->dspark_fused_last,
                                             DS4_N_EMBD,
                                             DS4_N_HC,
                                             1u) != 0;
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                g->batch_cur_hc,
                                                (uint32_t)hc_dim,
                                                n_rows,
                                                DS4_RMS_EPS) != 0;
    }
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(hc_mix_view,
                                       sm->map,
                                       sm->size,
                                       layer->hc_attn_fn->abs_offset,
                                       hc_dim,
                                       mix_hc,
                                       g->batch_flat_hc,
                                       n_rows) != 0;
    }
    if (ok) {
        ok = ds4_gpu_hc_split_weighted_sum_tensor(
                 attn_cur_view,
                 hc_split_view,
                 hc_mix_view,
                 g->batch_cur_hc,
                 sm->map,
                 sm->size,
                 layer->hc_attn_scale->abs_offset,
                 layer->hc_attn_base->abs_offset,
                 DS4_N_EMBD,
                 DS4_N_HC,
                 DS4_N_HC_SINKHORN_ITER,
                 DS4_HC_EPS) != 0;
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_weight_rows_tensor(
                 g->batch_attn_norm,
                 g->batch_attn_cur,
                 sm->map,
                 sm->size,
                 layer->attn_norm->abs_offset,
                 DS4_N_EMBD,
                 n_rows,
                 DS4_RMS_EPS) != 0;
    }

    if (ok) {
        ok = ds4_gpu_matmul_q8_0_tensor(g->batch_qr,
                                        sm->map,
                                        sm->size,
                                        layer->attn_q_a->abs_offset,
                                        DS4_N_EMBD,
                                        q_rank,
                                        draft_attn_norm_view,
                                        draft_rows) != 0;
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_weight_rows_tensor(
                 g->batch_qr_norm,
                 g->batch_qr,
                 sm->map,
                 sm->size,
                 layer->attn_q_a_norm->abs_offset,
                 (uint32_t)q_rank,
                 draft_rows,
                 DS4_RMS_EPS) != 0;
    }
    if (ok) {
        ok = ds4_gpu_matmul_q8_0_tensor(g->batch_q,
                                        sm->map,
                                        sm->size,
                                        layer->attn_q_b->abs_offset,
                                        q_rank,
                                        q_dim,
                                        g->batch_qr_norm,
                                        draft_rows) != 0;
    }
    if (ok) {
        ok = ds4_gpu_head_rms_norm_tensor(g->batch_q,
                                          draft_rows,
                                          DS4_N_HEAD,
                                          DS4_N_HEAD_DIM,
                                          DS4_RMS_EPS) != 0;
    }
    if (ok) {
        ok = ds4_gpu_rope_tail_tensor(g->batch_q,
                                      draft_rows,
                                      DS4_N_HEAD,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_ROT,
                                      pos0,
                                      0,
                                      false,
                                      DS4_ROPE_FREQ_BASE,
                                      1.0f,
                                      0.0f,
                                      1.0f,
                                      DS4_ROPE_YARN_BETA_FAST,
                                      DS4_ROPE_YARN_BETA_SLOW) != 0;
    }

    if (ok) {
        ok = ds4_gpu_matmul_q8_0_tensor(g->batch_kv_raw,
                                        sm->map,
                                        sm->size,
                                        layer->attn_kv->abs_offset,
                                        DS4_N_EMBD,
                                        DS4_N_HEAD_DIM,
                                        g->batch_attn_norm,
                                        n_rows) != 0;
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_weight_rows_tensor(
                 g->batch_kv,
                 g->batch_kv_raw,
                 sm->map,
                 sm->size,
                 layer->attn_kv_a_norm->abs_offset,
                 DS4_N_HEAD_DIM,
                 n_rows,
                 DS4_RMS_EPS) != 0;
    }
    if (ok) {
        ok = ds4_gpu_rope_tail_tensor(kv_target_view,
                                      1u,
                                      DS4_N_HEAD_KV,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_ROT,
                                      pos0,
                                      0,
                                      false,
                                      DS4_ROPE_FREQ_BASE,
                                      1.0f,
                                      0.0f,
                                      1.0f,
                                      DS4_ROPE_YARN_BETA_FAST,
                                      DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    if (ok) {
        ok = ds4_gpu_rope_tail_tensor(kv_draft_view,
                                      draft_rows,
                                      DS4_N_HEAD_KV,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_ROT,
                                      pos0,
                                      0,
                                      false,
                                      DS4_ROPE_FREQ_BASE,
                                      1.0f,
                                      0.0f,
                                      1.0f,
                                      DS4_ROPE_YARN_BETA_FAST,
                                      DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    if (ok) {
        ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->batch_kv,
                                                  n_rows,
                                                  DS4_N_HEAD_DIM,
                                                  DS4_N_ROT) != 0;
    }
    if (ok) {
        ok = ds4_gpu_store_raw_kv_batch_tensor(g->dspark_kv_cache[stage],
                                               g->batch_kv,
                                               g->dspark_cache_cap,
                                               pos0,
                                               n_rows,
                                               DS4_N_HEAD_DIM) != 0;
    }

    if (ok) {
        const uint32_t visible = pos0 + n_rows;
        const uint32_t n_raw =
            visible < g->dspark_cache_cap ? visible : g->dspark_cache_cap;
        const uint32_t raw_start =
            (visible - n_raw) % g->dspark_cache_cap;
        ok = ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
                 g->batch_heads,
                 sm->map,
                 sm->size,
                 layer->attn_sinks->abs_offset,
                 g->batch_q,
                 g->dspark_kv_cache[stage],
                 draft_rows,
                 n_raw,
                 g->dspark_cache_cap,
                 raw_start,
                 DS4_N_HEAD,
                 DS4_N_HEAD_DIM) != 0;
    }
    if (ok) {
        ok = ds4_gpu_rope_tail_tensor(g->batch_heads,
                                      draft_rows,
                                      DS4_N_HEAD,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_ROT,
                                      pos0,
                                      0,
                                      true,
                                      DS4_ROPE_FREQ_BASE,
                                      1.0f,
                                      0.0f,
                                      1.0f,
                                      DS4_ROPE_YARN_BETA_FAST,
                                      DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    if (ok) {
        ok = ds4_gpu_attention_output_q8_batch_tensor(
                 g->batch_attn_out,
                 g->batch_attn_low,
                 g->batch_group_tmp,
                 g->batch_low_tmp,
                 sm->map,
                 sm->size,
                 layer->attn_output_a->abs_offset,
                 layer->attn_output_b->abs_offset,
                 group_dim,
                 DS4_N_LORA_O,
                 DS4_N_OUT_GROUP,
                 DS4_N_EMBD,
                 g->batch_heads,
                 draft_rows) != 0;
    }
    if (ok) {
        ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                             g->batch_attn_out,
                                             draft_hc_view,
                                             draft_hc_split_view,
                                             DS4_N_EMBD,
                                             DS4_N_HC) != 0;
    }

    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(kv_draft_view);
    ds4_gpu_tensor_free(kv_target_view);
    ds4_gpu_tensor_free(draft_hc_split_view);
    ds4_gpu_tensor_free(draft_hc_view);
    ds4_gpu_tensor_free(draft_attn_norm_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    if (!ok)
      return false;
    if (!rocm_graph_encode_layer_ffn_batch(
            g, sm, layer, stage, pos0, draft_rows)) {
        return false;
    }

    /*
     * The next stage consumes [encoder, draft_0, ..., draft_N]. Keep the
     * encoder in row zero and shift this stage's draft outputs behind it.
     */
    return ds4_gpu_tensor_copy(g->batch_cur_hc,
                               hc_dim * sizeof(float),
                               g->batch_next_hc,
                               0,
                               (uint64_t)draft_rows * hc_dim *
                                   sizeof(float)) != 0;
}

static bool rocm_graph_encode_dspark_stage(ds4_gpu_graph *g,
                                           uint32_t stage,
                                           uint32_t pos0,
                                           uint32_t n_rows) {
    return rocm_graph_encode_dspark_stage_reference(
        g, stage, pos0, n_rows);
}

/* Final support hidden rows, consumed by the tied target LM head. */
static bool rocm_graph_encode_dspark_final_hidden(ds4_gpu_graph *g,
                                                  uint32_t row_offset,
                                                  uint32_t n_rows,
                                                  ds4_gpu_tensor *hidden_out,
                                                  uint32_t hidden_row_offset) {
    const ds4_dspark_model *d = g->dspark;
    if (!d || !hidden_out || n_rows == 0 || n_rows > g->spec_rows_cap) {
        return false;
    }
    const ds4_model *sm = d->model;
    const ds4_dspark_stage_weights *last = &d->stage[d->n_stages - 1u];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;

    /* Skip the encoder row: only the drafted rows feed the final head. */
    ds4_gpu_tensor *rows_hc = ds4_gpu_tensor_view(
            g->batch_cur_hc,
            (uint64_t)row_offset * hc_dim * sizeof(float),
            (uint64_t)n_rows * hc_dim * sizeof(float));
    if (!rows_hc) return false;

    ds4_gpu_tensor *head_pre = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_rows * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *head_weights = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_rows * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *head_embd = ds4_gpu_tensor_view(
            g->batch_ffn_cur, 0, (uint64_t)n_rows * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *head_norm = ds4_gpu_tensor_view(
            hidden_out,
            (uint64_t)hidden_row_offset * DS4_N_EMBD * sizeof(float),
            (uint64_t)n_rows * DS4_N_EMBD * sizeof(float));
    bool ok = head_pre && head_weights && head_embd && head_norm;

    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                     rows_hc,
                                                     (uint32_t)hc_dim,
                                                     n_rows,
                                                     DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(head_pre,
                                            sm->map,
                                            sm->size,
                                            last->hc_head_fn->abs_offset,
                                            hc_dim,
                                            DS4_N_HC,
                                            g->batch_flat_hc,
                                            n_rows) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(head_weights,
                                                   head_pre,
                                                   sm->map,
                                                   sm->size,
                                                   last->hc_head_scale->abs_offset,
                                                   last->hc_head_base->abs_offset,
                                                   DS4_N_HC,
                                                   DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(head_embd,
                                                 rows_hc,
                                                 head_weights,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(head_norm,
                                                      head_embd,
                                                      sm->map,
                                                      sm->size,
                                                      last->norm->abs_offset,
                                                      DS4_N_EMBD,
                                                      n_rows,
                                                      DS4_RMS_EPS) != 0;

    ds4_gpu_tensor_free(head_norm);
    ds4_gpu_tensor_free(head_embd);
    ds4_gpu_tensor_free(head_weights);
    ds4_gpu_tensor_free(head_pre);
    ds4_gpu_tensor_free(rows_hc);
    return ok;
}

static bool rocm_graph_encode_dspark_logits(
        ds4_gpu_graph *g,
        const ds4_model *target_model,
        const ds4_weights *target_weights,
        const ds4_gpu_tensor *hidden_rows,
        uint32_t n_rows) {
    if (!g || !target_model || !target_weights || !hidden_rows ||
        n_rows == 0u ||
        n_rows > g->spec_rows_cap) {
        return false;
    }
    const uint64_t vocab_dim = target_weights->output->dim[1];
    return ds4_gpu_matmul_q8_0_tensor(g->spec_logits,
                                      target_model->map,
                                      target_model->size,
                                      target_weights->output->abs_offset,
                                      DS4_N_EMBD,
                                      vocab_dim,
                                      hidden_rows,
                                      n_rows) != 0;
}

static float rocm_graph_dspark_confidence_threshold(void) {
  const char* setting = getenv("GUFO_DEEPSEEK_DSPARK_CONFIDENCE_THRESHOLD");
  if (!setting || !setting[0])
    return 0.70f;
  char* end = NULL;
  const double value = strtod(setting, &end);
  if (end == setting || *end != '\0' || !isfinite(value) || value < 0.0 ||
      value > 1.0) {
    return -1.0f;
  }
  return (float)value;
}

/*
 * Evaluate the support artifact's learned confidence head for one proposed
 * position. DeepSeek trains it over [normalized draft hidden, W1[prev token]].
 * The fused kernel decodes both Q8_0 inputs in registers and returns the
 * sigmoid probability directly.
 */
static bool rocm_graph_dspark_confidence(ds4_gpu_graph* g,
                                         const ds4_gpu_tensor* hidden_rows,
                                         uint32_t source_row,
                                         uint32_t previous_token,
                                         float* probability) {
  if (!g || !g->dspark || !probability || !g->dspark_confidence ||
      !hidden_rows) {
    return false;
  }
  const ds4_dspark_model* d = g->dspark;
  const ds4_model* sm = d->model;
  const ds4_dspark_stage_weights* last = &d->stage[d->n_stages - 1u];
  if (!last->confidence_proj || !last->markov_w1 ||
      previous_token >= DS4_N_VOCAB) {
    return false;
  }

  const uint64_t hidden_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
  ds4_gpu_tensor* hidden = ds4_gpu_tensor_view(
      hidden_rows, (uint64_t)source_row * hidden_bytes, hidden_bytes);
  bool ok = hidden != NULL;
  if (ok) {
    ok = ds4_gpu_dspark_confidence_tensor(
             g->dspark_confidence, hidden, sm->map, sm->size,
             last->confidence_proj->abs_offset, last->markov_w1->abs_offset,
             DS4_N_EMBD, d->markov_rank, previous_token) != 0;
  }
  if (ok) {
    ok = ds4_gpu_tensor_read(g->dspark_confidence, 0u, probability,
                             sizeof(*probability)) != 0;
  }
  ds4_gpu_tensor_free(hidden);
  if (!ok)
    return false;
  return isfinite(*probability);
}

/*
 * Trace the block's trajectory with the Markov path selector.
 *
 * Selecting each position by its own argmax lets the block drift: the best token
 * at position t given the block's shared context is not the best token given what
 * position t-1 actually emitted. The rank-256 bilinear term restores that
 * coupling. Positions are resolved in order because each one conditions the next,
 * but each step stays on the device and only the chosen index is read back.
 */
static bool rocm_graph_dspark_select_block(
    ds4_gpu_graph* g, const ds4_gpu_tensor* hidden_rows,
    const ds4_gpu_tensor* logits_rows, uint32_t source_row_offset,
    int last_token, uint32_t first_row, uint32_t n_rows, int32_t* tokens_out,
    bool schedule_confidence, uint32_t* selected_end) {
  const ds4_dspark_model* d = g->dspark;
  if (!d || !hidden_rows || !logits_rows || n_rows == 0 ||
      first_row >= n_rows || !tokens_out) {
    return false;
  }
  if (!g->dspark_markov_index || !g->dspark_markov_key)
    return false;
  const ds4_model* sm = d->model;
  const ds4_dspark_stage_weights* last = &d->stage[d->n_stages - 1u];
  const float confidence_threshold =
      schedule_confidence ? rocm_graph_dspark_confidence_threshold() : -1.0f;
  const bool trace_confidence =
      getenv("GUFO_DEEPSEEK_DSPARK_CONFIDENCE_TRACE") != NULL;
  int previous = last_token;
  uint32_t end = first_row;
  for (uint32_t row = first_row; row < n_rows; row++) {
    if (previous < 0 || (uint32_t)previous >= DS4_N_VOCAB)
      return false;
    if (confidence_threshold >= 0.0f || trace_confidence) {
      float confidence = 0.0f;
      if (!rocm_graph_dspark_confidence(g, hidden_rows, source_row_offset + row,
                                        (uint32_t)previous, &confidence)) {
        return false;
      }
      if (trace_confidence) {
        fprintf(stderr,
                "ds4: DSpark confidence row=%u probability=%.6f "
                "threshold=%.6f\n",
                row, confidence, confidence_threshold);
      }
      if (confidence_threshold > 0.0f && confidence < confidence_threshold) {
        break;
      }
    }
    ds4_gpu_tensor* logits_row = ds4_gpu_tensor_view(
        logits_rows,
        (uint64_t)(source_row_offset + row) * DS4_N_VOCAB * sizeof(float),
        (uint64_t)DS4_N_VOCAB * sizeof(float));
    if (!logits_row)
      return false;
    const bool ok =
        ds4_gpu_dspark_markov_argmax_tensor(
            g->dspark_markov_index, g->dspark_markov_key, logits_row, sm->map,
            sm->size, last->markov_w1->abs_offset, last->markov_w2->abs_offset,
            DS4_N_VOCAB, d->markov_rank, (uint32_t)previous) != 0;
    ds4_gpu_tensor_free(logits_row);
    if (!ok)
      return false;
    int32_t selected = -1;
    if (ds4_gpu_tensor_read(g->dspark_markov_index, 0, &selected,
                            sizeof(selected)) == 0) {
      return false;
    }
    if (selected < 0 || (uint32_t)selected >= DS4_N_VOCAB)
      return false;
    tokens_out[row] = selected;
    previous = selected;
    end = row + 1u;
  }
  if (selected_end)
    *selected_end = end;
  return true;
}

/*
 * Propose a block of draft tokens for positions [pos0, pos0 + block_size).
 *
 * `last_token` is the most recently committed token; it seeds both the block's
 * first embedding slot and the Markov chain. Requires that the target has already
 * captured its sampled hidden states for position pos0 - 1 and that those have
 * been injected into the rings.
 */
static bool rocm_graph_dspark_prepare_hidden(
    ds4_gpu_graph* g, const ds4_model* target_model,
    const ds4_weights* target_weights, int last_token, uint32_t pos0,
    ds4_gpu_tensor* hidden_out, uint32_t hidden_row_offset) {
  const ds4_dspark_model* d = g->dspark;
  if (!d)
    return false;
  const uint32_t block = d->block_size;
  /* One encoder row plus one row per drafted token. */
  const uint32_t rows = block + 1u;
  if (block == 0 || block > g->spec_rows_cap || rows > g->prefill_cap)
    return false;
  if (pos0 == 0 || g->dspark_context_len < pos0)
    return false;
  if (rows > g->dspark_cache_cap)
    return false;

  /*
   * Row 0's token is unused: the encoder row's hidden state is overwritten with
   * the fused target feature. Rows 1.. carry the committed token followed by
   * MASK placeholders.
   */
  int32_t block_tokens[DS4_DSPARK_MAX_BLOCK + 1u];
  block_tokens[0] = last_token;
  block_tokens[1] = last_token;
  for (uint32_t i = 2; i < rows; i++) {
    block_tokens[i] = (int32_t)d->noise_token_id;
  }
  if (ds4_gpu_tensor_write(g->dspark_block_tokens, 0, block_tokens,
                           (uint64_t)rows * sizeof(block_tokens[0])) == 0) {
    return false;
  }
  if (ds4_gpu_tensor_write(g->prefill_tokens, 0, block_tokens + 1,
                           (uint64_t)block * sizeof(block_tokens[0])) == 0) {
    return false;
  }

  bool ok = ds4_gpu_begin_commands() != 0;
  /* Tied target embeddings, expanded across the hyper-connection streams. */
  if (ok)
    ok = ds4_gpu_embed_tokens_hc_tensor(
             g->batch_cur_hc, g->dspark_block_tokens, target_model->map,
             target_model->size, target_weights->token_embd->abs_offset,
             DS4_N_VOCAB, rows, DS4_N_EMBD, DS4_N_HC) != 0;
  for (uint32_t stage = 0; ok && stage < d->n_stages; stage++) {
    ok = rocm_graph_encode_dspark_stage(g, stage, pos0, rows);
  }
  if (ok) {
    ok = rocm_graph_encode_dspark_final_hidden(
        g, 1u, block, hidden_out, hidden_row_offset);
  }
  if (ok) {
    ok = ds4_gpu_end_commands() != 0;
  } else {
    (void)ds4_gpu_synchronize();
  }
  return ok;
}

static bool rocm_graph_dspark_draft(ds4_gpu_graph* g,
                                    const ds4_model* target_model,
                                    const ds4_weights* target_weights,
                                    int last_token, uint32_t pos0,
                                    int32_t* tokens_out, uint32_t* n_out,
                                    bool schedule_confidence) {
  if (!g || !tokens_out || !n_out)
    return false;
  *n_out = 0;
  const ds4_dspark_model* d = g->dspark;
  if (!d)
    return false;
  const uint32_t block = d->block_size;

  /* Mode 2 keeps generic narrow-batch routing available to the support
   * model without enabling verifier-only arithmetic candidates. */
  ds4_gpu_set_small_batch_mode(2);
  bool ok = rocm_graph_dspark_prepare_hidden(
      g, target_model, target_weights, last_token, pos0, g->batch_ffn_norm,
      0u);
  if (!ok) {
    ds4_gpu_set_small_batch_mode(0);
    return false;
  }
  ok = ds4_gpu_begin_commands() != 0;
  if (ok) {
    ok = rocm_graph_encode_dspark_logits(
        g, target_model, target_weights, g->batch_ffn_norm, block);
  }
  if (ok) {
    ok = ds4_gpu_end_commands() != 0;
  } else {
    (void)ds4_gpu_synchronize();
  }
  if (!ok) {
    ds4_gpu_set_small_batch_mode(0);
    return false;
  }

  uint32_t selected_rows = 0u;
  const bool selected = rocm_graph_dspark_select_block(
      g, g->batch_ffn_norm, g->spec_logits, 0u, last_token, 0u, block,
      tokens_out, schedule_confidence, &selected_rows);
  ds4_gpu_set_small_batch_mode(0);
  if (!selected) {
    return false;
  }
  *n_out = selected_rows;
  return true;
}

/*
 * Draft each request with its session-local support attention and KV state,
 * then stream the tied target vocabulary projection once across all hidden
 * rows. This preserves the support model's arithmetic through final norm and
 * changes only the row count of the shared Q8 output projection.
 */
static bool rocm_graph_dspark_draft_head_batch(
    ds4_engine* engine, const ds4_rocm_dspark_draft_item* items,
    size_t item_count, bool schedule_confidence) {
  if (!engine || !engine->model || !engine->weights || !items ||
      item_count != 2u) {
    return false;
  }
  ds4_gpu_graph* coordinator = items[0].graph;
  if (!coordinator || !coordinator->dspark ||
      !coordinator->batch_dspark_verify_after_attn_hc) {
    return false;
  }
  const uint32_t block = coordinator->dspark->block_size;
  const uint32_t total_rows = static_cast<uint32_t>(item_count) * block;
  if (block == 0u || total_rows > coordinator->prefill_cap ||
      total_rows > coordinator->batch_dspark_verify_rows_cap ||
      total_rows > DS4_SPEC_MAX_ROWS ||
      !rocm_graph_spec_prepare(coordinator, engine->weights, total_rows)) {
    return false;
  }

  bool ok = true;
  ds4_gpu_set_small_batch_mode(2);
  for (size_t index = 0; ok && index < item_count; ++index) {
    const ds4_rocm_dspark_draft_item& item = items[index];
    if (!item.graph || item.graph->dspark != coordinator->dspark ||
        !item.tokens || !item.n_tokens) {
      ok = false;
      break;
    }
    *item.n_tokens = 0u;
    ok = rocm_graph_dspark_prepare_hidden(
        item.graph, engine->model, engine->weights, item.last_token,
        item.position, coordinator->batch_dspark_verify_after_attn_hc,
        static_cast<uint32_t>(index) * block);
  }

  if (ok)
    ok = ds4_gpu_begin_commands() != 0;
  if (ok) {
    ok = rocm_graph_encode_dspark_logits(
        coordinator, engine->model, engine->weights,
        coordinator->batch_dspark_verify_after_attn_hc, total_rows);
  }
  if (ok) {
    ok = ds4_gpu_end_commands() != 0;
  } else {
    (void)ds4_gpu_synchronize();
  }
  if (!ok) {
    ds4_gpu_set_small_batch_mode(0);
    return false;
  }

  for (size_t index = 0; ok && index < item_count; ++index) {
    const ds4_rocm_dspark_draft_item& item = items[index];
    uint32_t selected_rows = 0u;
    ok = rocm_graph_dspark_select_block(
        item.graph, coordinator->batch_dspark_verify_after_attn_hc,
        coordinator->spec_logits, static_cast<uint32_t>(index) * block,
        item.last_token, 0u, block, item.tokens, schedule_confidence,
        &selected_rows);
    if (ok)
      *item.n_tokens = selected_rows;
  }
  ds4_gpu_set_small_batch_mode(0);
  return ok;
}

/*
 * Re-select the block's tail once the target's own next token is known.
 *
 * Position zero of a block is never worth drafting: the target's logits at the
 * current frontier already name that token exactly. Drafting it instead makes a
 * whole cycle a total loss whenever the guess differs, and it seeds the Markov
 * chain from a wrong token so the rest of the block is conditioned on fiction.
 * Overriding row zero with the known token and re-tracing the chain from it costs
 * one vocabulary pass per remaining row and reuses the base logits already on the
 * device.
 */
bool ds4_rocm_graph_dspark_reselect_tail(ds4_rocm_graph *graph,
                                         int known_first_token,
                                         uint32_t n_rows,
                                         int32_t *tokens_out) {
    if (!graph || n_rows < 2u || !tokens_out) return false;
    return rocm_graph_dspark_select_block(
        graph, graph->batch_ffn_norm, graph->spec_logits, 0u, known_first_token,
        1u, n_rows, tokens_out, false, NULL);
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
                                   prefill_capacity,
                                   engine->dspark
                                       ? engine->dspark_batch_workspace
                                       : NULL)) {
        free(graph);
        return NULL;
    }
    if (engine->dspark && !engine->dspark_batch_workspace) {
      const uint32_t verify_rows_cap = 8u * (engine->dspark->block_size + 1u);
      const uint64_t verify_hc_elems =
          (uint64_t)verify_rows_cap * DS4_N_HC * DS4_N_EMBD;
      graph->batch_dspark_verify_cur_hc =
          ds4_gpu_tensor_alloc(verify_hc_elems * sizeof(float));
      graph->batch_dspark_verify_after_attn_hc =
          ds4_gpu_tensor_alloc(verify_hc_elems * sizeof(float));
      if (!graph->batch_dspark_verify_cur_hc ||
          !graph->batch_dspark_verify_after_attn_hc) {
        rocm_graph_free(graph);
        free(graph);
        return NULL;
      }
      graph->batch_dspark_verify_rows_cap = verify_rows_cap;
      auto* workspace =
          static_cast<ds4_rocm_graph*>(ds4_xcalloc(1, sizeof(ds4_rocm_graph)));
      workspace->prefill_cap = graph->prefill_cap;
      workspace->comp_cap = graph->comp_cap;
      rocm_graph_bind_batch_workspace(workspace, graph);
      workspace->borrows_batch_workspace = false;
      graph->borrows_batch_workspace = true;
      engine->dspark_batch_workspace = workspace;
      fprintf(stderr, "ds4: DSpark sessions share one %u-row batch workspace\n",
              graph->prefill_cap);
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

bool ds4_rocm_graph_eval_batch(ds4_engine *engine,
                               const ds4_rocm_batch_item *items,
                               size_t item_count) {
    return engine && rocm_graph_eval_sessions_batch(engine->model,
                                                    engine->weights,
                                                    items,
                                                    item_count);
}

bool ds4_rocm_graph_spec_prepare(ds4_rocm_graph *graph,
                                 ds4_engine *engine,
                                 uint32_t rows_cap) {
    return graph && engine &&
           rocm_graph_spec_prepare(graph, engine->weights, rows_cap);
}

uint32_t ds4_rocm_graph_spec_rows_cap(const ds4_rocm_graph *graph) {
    return rocm_graph_spec_rows_cap(graph);
}

bool ds4_rocm_graph_spec_frontier_save(ds4_rocm_graph *graph) {
    return rocm_graph_spec_frontier_save(graph);
}

bool ds4_rocm_graph_spec_frontier_restore(ds4_rocm_graph *graph) {
    return rocm_graph_spec_frontier_restore(graph);
}

bool ds4_rocm_graph_spec_frontier_commit_prefix(ds4_rocm_graph *graph,
                                                uint32_t prefix_len) {
    return rocm_graph_spec_frontier_commit_prefix(graph, prefix_len);
}

bool ds4_rocm_graph_verify_suffix(ds4_rocm_graph *graph,
                                  ds4_engine *engine,
                                  const ds4_tokens *tokens,
                                  uint32_t start,
                                  uint32_t n_tokens,
                                  int32_t *row_tops) {
    return graph && engine &&
           rocm_graph_verify_suffix(graph,
                                     engine->model,
                                     engine->weights,
                                     tokens,
                                     start,
                                     n_tokens,
                                     row_tops);
}

bool ds4_rocm_graph_verify_batch(ds4_engine* engine,
                                 const ds4_rocm_verify_item* items,
                                 size_t item_count) {
  return rocm_graph_verify_sessions_batch(engine, items, item_count);
}

bool ds4_rocm_graph_read_spec_logits_row(const ds4_rocm_graph *graph,
                                         uint32_t row,
                                         float *logits) {
    return rocm_graph_read_spec_logits_row(graph, row, logits);
}

bool ds4_rocm_graph_dspark_attach(ds4_rocm_graph *graph,
                                  ds4_engine *engine,
                                  const ds4_dspark_model *dspark) {
    return graph && engine &&
           rocm_graph_dspark_attach(graph, engine->weights, dspark);
}

uint32_t ds4_rocm_graph_dspark_block_size(const ds4_rocm_graph *graph) {
    return graph && graph->dspark ? graph->dspark->block_size : 0u;
}

void ds4_rocm_graph_dspark_set_capture_enabled(ds4_rocm_graph *graph,
                                               bool enabled) {
    if (!graph || !graph->dspark) return;
    graph->dspark_capture_enabled = enabled;
    if (!enabled) {
        graph->dspark_context_len = 0u;
        graph->dspark_capture_mask = 0u;
        graph->dspark_capture_batch_mask = 0u;
        graph->dspark_capture_batch_start = 0u;
        graph->dspark_capture_batch_tokens = 0u;
    }
}

bool ds4_rocm_graph_dspark_capture_ready(const ds4_rocm_graph *graph) {
    return graph && rocm_graph_dspark_capture_complete(graph);
}

void ds4_rocm_graph_dspark_capture_reset(ds4_rocm_graph *graph) {
    if (!graph) return;
    graph->dspark_capture_mask = 0u;
    graph->dspark_capture_batch_mask = 0u;
    graph->dspark_capture_batch_start = 0u;
    graph->dspark_capture_batch_tokens = 0u;
}

uint32_t ds4_rocm_graph_dspark_batch_capture_rows(const ds4_rocm_graph *graph,
                                                 uint32_t *start) {
    if (!graph || !graph->dspark) return 0u;
    const uint32_t want = rocm_graph_dspark_complete_mask(graph);
    if (want == 0u || graph->dspark_capture_batch_mask != want) return 0u;
    if (start) *start = graph->dspark_capture_batch_start;
    return graph->dspark_capture_batch_tokens;
}

bool ds4_rocm_graph_dspark_inject(ds4_rocm_graph *graph,
                                  uint32_t pos0,
                                  uint32_t n_rows) {
    if (!graph) return false;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = rocm_graph_dspark_inject(graph, pos0, n_rows);
    if (ok) {
        ok = ds4_gpu_end_commands() != 0;
    } else {
        (void)ds4_gpu_synchronize();
    }
    return ok;
}

uint32_t ds4_rocm_graph_dspark_context_len(const ds4_rocm_graph *graph) {
    return graph ? graph->dspark_context_len : 0u;
}

void ds4_rocm_graph_dspark_truncate_context(ds4_rocm_graph *graph, uint32_t length) {
    if (!graph) return;
    if (length < graph->dspark_context_len) graph->dspark_context_len = length;
}

bool ds4_rocm_graph_dspark_draft(ds4_rocm_graph* graph, ds4_engine* engine,
                                 int last_token, uint32_t pos0,
                                 int32_t* tokens_out, uint32_t* n_out,
                                 bool schedule_confidence) {
  return graph && engine &&
         rocm_graph_dspark_draft(graph, engine->model, engine->weights,
                                 last_token, pos0, tokens_out, n_out,
                                 schedule_confidence);
}

bool ds4_rocm_graph_dspark_draft_head_batch(
    ds4_engine* engine, const ds4_rocm_dspark_draft_item* items,
    size_t item_count, bool schedule_confidence) {
  return rocm_graph_dspark_draft_head_batch(
      engine, items, item_count, schedule_confidence);
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
    graph->dspark_context_len = 0;
    graph->dspark_capture_mask = 0;
    graph->dspark_capture_batch_mask = 0;
    graph->dspark_capture_batch_start = 0;
    graph->dspark_capture_batch_tokens = 0;
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

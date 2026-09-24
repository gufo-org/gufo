#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "../kernels/rocm/resident_api.h"
#include "dspark_internal.h"
#include "dspark_policy.h"
#include "model_data_internal.h"
#include "native_internal.h"

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
    bool managed_kv_cache;
    /* Per-layer work tensors.  They are reused in place by every layer instead
     * of allocating a generic graph arena.  This is why the code is verbose but
     * predictable: each pointer names an actual DS4 stage. */
    ds4_gpu_tensor *comp_kv_cur;
    ds4_gpu_tensor *comp_sc_cur;
    ds4_gpu_tensor *indexer_q;
    ds4_gpu_tensor *indexer_weights;
    ds4_gpu_tensor* indexer_scores;
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
    ds4_gpu_tensor* batch_stage_scratch;
    ds4_gpu_tensor *batch_cur_hc;
    ds4_gpu_tensor *batch_next_hc;
    ds4_gpu_tensor* batch_flat_hc;
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
    ds4_rocm_graph* batch_workspace_owner;
    uint32_t batch_workspace_borrowers;
    // Keep the largest scratch arena cached. Older arenas remain alive only
    // while sessions still borrow them and are pruned on session creation.
    ds4_rocm_graph* next_batch_workspace;

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
    ds4_gpu_tensor *spec_frontier_logits;

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
    ds4_gpu_tensor *dspark_markov_key;
    ds4_gpu_tensor* dspark_candidates;
    ds4_gpu_tensor* dspark_candidate_scratch;
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
    ds4_gpu_tensor_free(g->spec_frontier_logits);
    g->spec_row_tops = NULL;
    g->spec_logits = NULL;
    g->spec_frontier_logits = NULL;
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
    ds4_gpu_tensor_free(g->dspark_candidates);
    ds4_gpu_tensor_free(g->dspark_candidate_scratch);
    ds4_gpu_tensor_free(g->dspark_fused);
    ds4_gpu_tensor_free(g->dspark_features_batch);
    ds4_gpu_tensor_free(g->dspark_features);
    ds4_gpu_tensor_free(g->dspark_hc_mean_rows);
    ds4_gpu_tensor_free(g->dspark_hc_mean);
    g->dspark_markov_index = NULL;
    g->dspark_markov_key = NULL;
    g->dspark_candidates = nullptr;
    g->dspark_candidate_scratch = nullptr;
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
    if (rows_cap == g->dspark_capture_rows_cap && g->dspark_features &&
        g->dspark_fused) {
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
    g->batch_stage_scratch = workspace->batch_stage_scratch;
    g->batch_cur_hc = workspace->batch_cur_hc;
    g->batch_next_hc = workspace->batch_next_hc;
    g->batch_flat_hc = workspace->batch_flat_hc;
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
  ds4_gpu_tensor_free(g->batch_next_hc);
  ds4_gpu_tensor_free(g->batch_cur_hc);
  ds4_gpu_tensor_free(g->prefill_tokens);
  ds4_gpu_tensor_free(g->batch_stage_scratch);
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
    return kv_cache_bytes + comp_cap * prefill_cap * sizeof(float);
}

static ds4_gpu_tensor *rocm_graph_alloc_kv_cache_tensor(bool managed, uint64_t bytes) {
    return managed ? ds4_gpu_tensor_alloc_managed(bytes) : ds4_gpu_tensor_alloc(bytes);
}

static bool rocm_graph_reserve_compressed_rows(ds4_gpu_graph* g, uint32_t il,
                                               uint32_t required) {
  if (required > g->layer_comp_cap[il])
    return false;
  const uint64_t row_bytes = DS4_N_HEAD_DIM * sizeof(float);
  const uint64_t capacity =
      ds4_gpu_tensor_bytes(g->layer_attn_comp_cache[il]) / row_bytes;
  if (required <= capacity)
    return true;

  const uint64_t next_capacity = std::min<uint64_t>(
      g->layer_comp_cap[il], std::max<uint64_t>(required, capacity * 2));
  using Tensor =
      std::unique_ptr<ds4_gpu_tensor, decltype(&ds4_gpu_tensor_free)>;
  Tensor attention(rocm_graph_alloc_kv_cache_tensor(g->managed_kv_cache,
                                                    next_capacity * row_bytes),
                   ds4_gpu_tensor_free);
  Tensor mirror(nullptr, ds4_gpu_tensor_free);
  Tensor indexer(nullptr, ds4_gpu_tensor_free);
  const bool indexed = ds4_layer_compress_ratio(il) == 4;
  if (indexed) {
    mirror.reset(rocm_graph_alloc_kv_cache_tensor(
        g->managed_kv_cache,
        next_capacity * DS4_N_HEAD_DIM * sizeof(uint16_t)));
    indexer.reset(rocm_graph_alloc_kv_cache_tensor(
        g->managed_kv_cache,
        next_capacity * DS4_N_INDEXER_HEAD_DIM * sizeof(float)));
  }
  if (!attention || (indexed && (!mirror || !indexer)))
    return false;
  const auto copy_rows = [](ds4_gpu_tensor* to, const ds4_gpu_tensor* from,
                            uint64_t bytes) {
    return bytes == 0 || ds4_gpu_tensor_copy(to, 0, from, 0, bytes) != 0;
  };
  if (!copy_rows(attention.get(), g->layer_attn_comp_cache[il],
                 static_cast<uint64_t>(g->layer_n_comp[il]) * row_bytes) ||
      (indexed && (!copy_rows(mirror.get(), g->layer_attn_comp_cache_f16[il],
                              static_cast<uint64_t>(g->layer_n_comp[il]) *
                                  DS4_N_HEAD_DIM * sizeof(uint16_t)) ||
                   !copy_rows(indexer.get(), g->layer_index_comp_cache[il],
                              static_cast<uint64_t>(g->layer_n_index_comp[il]) *
                                  DS4_N_INDEXER_HEAD_DIM * sizeof(float))))) {
    return false;
  }
  // All replacements and byte-exact copies succeed before releasing any old
  // cache. These caches have no persistent views; snapshots store live rows.
  ds4_gpu_tensor_free(g->layer_attn_comp_cache[il]);
  ds4_gpu_tensor_free(g->layer_attn_comp_cache_f16[il]);
  ds4_gpu_tensor_free(g->layer_index_comp_cache[il]);
  g->layer_attn_comp_cache[il] = attention.release();
  g->layer_attn_comp_cache_f16[il] = mirror.release();
  g->layer_index_comp_cache[il] = indexer.release();
  return true;
}

static bool rocm_graph_reserve_compressed_position(ds4_gpu_graph* g,
                                                   uint32_t il, uint64_t end) {
  const uint32_t ratio = ds4_layer_compress_ratio(il);
  if (ratio == 0)
    return true;
  return rocm_graph_reserve_compressed_rows(
      g, il,
      static_cast<uint32_t>(
          std::min<uint64_t>(g->layer_comp_cap[il], end / ratio + 2)));
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
    g->managed_kv_cache = managed_kv_cache;
    if (managed_kv_cache) {
        /*
         * Device allocations are fastest, but very large contexts can exhaust
         * Strix Halo unified memory once resident weights and driver allocations
         * are present. Managed memory preserves a demand-paged fallback for this
         * long-lived cache class.
         */
        fprintf(stderr,
                "ds4: ROCm using managed KV cache for ctx=%u "
                "(maximum KV %.2f GiB, context estimate %.2f GiB); "
                "this may degrade performance but is needed for very large "
                "contexts\n",
                ctx_size, (double)kv_cache_bytes / 1073741824.0,
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
          // Reserve the first prompt chunk plus one raw window of decode
          // headroom. The configured limits stay independent of storage.
          const uint64_t initial_comp_cap = std::min<uint64_t>(
              g->layer_comp_cap[il], (pc + DS4_N_SWA) / ratio + 2);
          const uint32_t coff = ratio == 4 ? 2u : 1u;
          const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
          const uint64_t attn_rows = (uint64_t)coff * ratio;
          g->layer_attn_comp_cache[il] = rocm_graph_alloc_kv_cache_tensor(
              managed_kv_cache,
              initial_comp_cap * DS4_N_HEAD_DIM * sizeof(float));
          if (ratio == 4) {
            g->layer_attn_comp_cache_f16[il] = rocm_graph_alloc_kv_cache_tensor(
                managed_kv_cache,
                initial_comp_cap * DS4_N_HEAD_DIM * sizeof(uint16_t));
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
                g->layer_index_comp_cache[il] =
                    rocm_graph_alloc_kv_cache_tensor(
                        managed_kv_cache, initial_comp_cap *
                                              DS4_N_INDEXER_HEAD_DIM *
                                              sizeof(float));
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
    // Cover the first prefill chunk and single-token decoding at any depth.
    // Later chunks grow this disposable scratch to the actual score shape.
    // A large configured window must not reserve a full-window prefill matrix.
    const uint64_t initial_score_count = std::max<uint64_t>(
        g->comp_cap, std::min<uint64_t>(g->comp_cap, pc / min_ratio + 2) * pc);
    g->indexer_scores =
        ds4_gpu_tensor_alloc(initial_score_count * sizeof(float));
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
      if (batch_workspace->prefill_cap < prefill_cap) {
        rocm_graph_free(g);
        return false;
      }
        rocm_graph_bind_batch_workspace(g, batch_workspace);
    } else {
      // These stages run in order, including across grouped requests:
      // HC normalization -> attention Q/heads -> routed MoE -> shared expert.
      // Only routed down survives into the shared-expert stage. Keep it
      // beyond both sets of FFN intermediates, then reuse the other ranges.
      const uint64_t q_bytes = pc * q_dim * sizeof(float);
      const uint64_t hc_bytes = pc * hc_dim * sizeof(float);
      const uint64_t routed_mid_bytes =
          pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float);
      const uint64_t routed_down_bytes =
          pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float);
      const uint64_t shared_mid_bytes = pc * shared_dim * sizeof(float);
      const uint64_t shared_out_bytes = pc * DS4_N_EMBD * sizeof(float);
      const uint64_t down_offset = std::max(
          3 * routed_mid_bytes, 3 * shared_mid_bytes + shared_out_bytes);
      g->batch_stage_scratch = ds4_gpu_tensor_alloc(
          std::max({2 * q_bytes, hc_bytes, down_offset + routed_down_bytes}));
      g->prefill_tokens = ds4_gpu_tensor_alloc(pc * sizeof(int32_t));
      g->batch_cur_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
      g->batch_next_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
      g->batch_flat_hc =
          ds4_gpu_tensor_view(g->batch_stage_scratch, 0, hc_bytes);
      g->batch_hc_mix = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
      g->batch_hc_split = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
      g->batch_attn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
      g->batch_attn_norm =
          ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
      g->batch_qr = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
      g->batch_qr_norm = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
      g->batch_q = ds4_gpu_tensor_view(g->batch_stage_scratch, 0, q_bytes);
      g->batch_kv_raw =
          ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
      g->batch_kv = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
      g->batch_comp_kv =
          ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
      g->batch_comp_sc =
          ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
      g->batch_indexer_q =
          ds4_gpu_tensor_alloc(pc * indexer_q_dim * sizeof(float));
      g->batch_indexer_weights =
          ds4_gpu_tensor_alloc(pc * DS4_N_INDEXER_HEAD * sizeof(float));
      // Indexer scoring may borrow this range for F16 keys and queries before
      // attention writes its heads. It is disjoint from live attention Q.
      g->batch_heads =
          ds4_gpu_tensor_view(g->batch_stage_scratch, q_bytes, q_bytes);
      g->batch_attn_low = ds4_gpu_tensor_alloc(pc * low_dim * sizeof(float));
      g->batch_attn_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
      g->batch_group_tmp = ds4_gpu_tensor_alloc(pc * group_dim * sizeof(float));
      g->batch_low_tmp =
          ds4_gpu_tensor_alloc(pc * DS4_N_LORA_O * sizeof(float));
      g->batch_after_attn_hc =
          ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
      g->batch_ffn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
      g->batch_ffn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
      g->batch_shared_gate =
          ds4_gpu_tensor_view(g->batch_stage_scratch, 0, shared_mid_bytes);
      g->batch_shared_up = ds4_gpu_tensor_view(
          g->batch_stage_scratch, shared_mid_bytes, shared_mid_bytes);
      g->batch_shared_mid = ds4_gpu_tensor_view(
          g->batch_stage_scratch, 2 * shared_mid_bytes, shared_mid_bytes);
      g->batch_shared_out = ds4_gpu_tensor_view(
          g->batch_stage_scratch, 3 * shared_mid_bytes, shared_out_bytes);
      g->batch_router_logits =
          ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
      g->batch_router_probs =
          ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
      g->batch_router_selected =
          ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(int));
      g->batch_router_weights =
          ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(float));
      g->batch_routed_gate =
          ds4_gpu_tensor_view(g->batch_stage_scratch, 0, routed_mid_bytes);
      g->batch_routed_up = ds4_gpu_tensor_view(
          g->batch_stage_scratch, routed_mid_bytes, routed_mid_bytes);
      g->batch_routed_mid = ds4_gpu_tensor_view(
          g->batch_stage_scratch, 2 * routed_mid_bytes, routed_mid_bytes);
      g->batch_routed_down = ds4_gpu_tensor_view(
          g->batch_stage_scratch, down_offset, routed_down_bytes);
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

    const bool ok =
        state_init_ok && layer_cache_ok && g->cur_hc && g->flat_hc &&
        g->hc_mix && g->hc_split && g->hc_pre && g->hc_post && g->hc_comb &&
        g->attn_cur && g->attn_norm && g->qr && g->qr_norm && g->q &&
        g->kv_raw && g->kv && g->comp_kv_cur && g->comp_sc_cur &&
        g->indexer_q && g->indexer_weights && g->indexer_scores &&
        g->comp_selected && g->heads && g->attn_low && g->attn_out &&
        g->after_attn_hc && g->ffn_cur && g->ffn_norm && g->shared_gate &&
        g->shared_up && g->shared_mid && g->shared_out && g->router_logits &&
        g->router_probs && g->router_selected && g->router_weights &&
        g->routed_gate && g->routed_up && g->routed_mid && g->routed_down &&
        g->routed_out && g->after_ffn_hc && g->output_pre &&
        g->output_weights && g->output_embd && g->output_norm && g->logits &&
        g->prefill_tokens && g->batch_stage_scratch && g->batch_cur_hc &&
        g->batch_next_hc && g->batch_flat_hc && g->batch_hc_mix &&
        g->batch_hc_split && g->batch_attn_cur && g->batch_attn_norm &&
        g->batch_qr && g->batch_qr_norm && g->batch_q && g->batch_kv_raw &&
        g->batch_kv && g->batch_comp_kv && g->batch_comp_sc &&
        g->batch_indexer_q && g->batch_indexer_weights && g->batch_heads &&
        g->batch_attn_low && g->batch_attn_out && g->batch_group_tmp &&
        g->batch_low_tmp && g->batch_after_attn_hc && g->batch_ffn_cur &&
        g->batch_ffn_norm && g->batch_shared_gate && g->batch_shared_up &&
        g->batch_shared_mid && g->batch_shared_out && g->batch_router_logits &&
        g->batch_router_probs && g->batch_router_selected &&
        g->batch_router_weights && g->batch_routed_gate && g->batch_routed_up &&
        g->batch_routed_mid && g->batch_routed_down && g->batch_routed_out;
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

/* =========================================================================
 * ROCm Decode Helpers.
 * =========================================================================
 *
 * The generation path uses the qualified fused helpers below.
 */

static bool rocm_graph_decode_kv_store(ds4_gpu_tensor* kv,
                                       ds4_gpu_tensor* raw_cache,
                                       uint32_t raw_cap, uint32_t raw_row) {
  return ds4_gpu_kv_fp8_store_raw_tensor(kv, raw_cache, raw_cap, raw_row,
                                         DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
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

static bool rocm_graph_reserve_indexer_scores(ds4_gpu_graph* g, uint32_t n_comp,
                                              uint32_t n_tokens) {
  if (n_comp == 0 || n_comp > g->comp_cap || n_tokens == 0 ||
      n_tokens > g->prefill_cap) {
    return false;
  }
  const uint64_t required =
      static_cast<uint64_t>(n_comp) * n_tokens * sizeof(float);
  const uint64_t capacity = ds4_gpu_tensor_bytes(g->indexer_scores);
  if (required <= capacity)
    return true;
  const uint64_t maximum =
      static_cast<uint64_t>(g->comp_cap) * g->prefill_cap * sizeof(float);
  auto* replacement =
      ds4_gpu_tensor_alloc(std::min(maximum, std::max(required, capacity * 2)));
  if (replacement == nullptr)
    return false;
  // Scores are overwritten before use; KV state and snapshots never refer
  // to this allocation. Geometric growth avoids reallocating each chunk.
  ds4_gpu_tensor_free(g->indexer_scores);
  g->indexer_scores = replacement;
  return true;
}

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
  if (!rocm_graph_reserve_compressed_position(g, il,
                                              static_cast<uint64_t>(pos) + 1))
    return false;
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
  const float ext_factor =
      compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
  float attn_factor = 1.0f;
  if (ext_factor != 0.0f && freq_scale > 0.0f) {
    attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
  }

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
    if (ok) {
      ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(
               g->attn_cur, g->attn_norm, g->hc_split, g->hc_mix, g->cur_hc,
               model->map, model->size, layer->hc_attn_scale->abs_offset,
               layer->hc_attn_base->abs_offset, layer->attn_norm->abs_offset,
               DS4_N_EMBD, DS4_N_HC, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS,
               DS4_RMS_EPS) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("attn_hc_pre");
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("attn_norm");
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->qr, model->map, model->size,
                                              layer->attn_q_a->abs_offset,
                                              DS4_N_EMBD, q_rank,
                                              g->attn_norm, 1) != 0;
    {
      if (ok)
        ok = ds4_gpu_matmul_q8_0_tensor(g->kv_raw, model->map, model->size,
                                        layer->attn_kv->abs_offset, DS4_N_EMBD,
                                        DS4_N_HEAD_DIM, g->attn_norm, 1) != 0;
      if (ok)
        ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                 g->qr_norm, g->qr, model->map, model->size,
                 layer->attn_q_a_norm->abs_offset, (uint32_t)q_rank, g->kv,
                 g->kv_raw, layer->attn_kv_a_norm->abs_offset, DS4_N_HEAD_DIM,
                 1, DS4_RMS_EPS) != 0;
    }
    if (ok)
      ok = ds4_gpu_matmul_q8_0_tensor(g->q, model->map, model->size,
                                      layer->attn_q_b->abs_offset, q_rank,
                                      q_dim, g->qr_norm, 1) != 0;
    if (ok)
      ok = ds4_gpu_head_rms_norm_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                        DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false, freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("q_path");

    if (ok)
      ok = ds4_gpu_rope_tail_tensor(
               g->kv, 1, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos,
               compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false, freq_base,
               freq_scale, ext_factor, attn_factor, DS4_ROPE_YARN_BETA_FAST,
               DS4_ROPE_YARN_BETA_SLOW) != 0;
    /* RoPE stays as the exact standalone kernel above.  The decode fusion
     * starts after that, where FP8 KV quantization and raw-cache storage can
     * share one pass without changing the trigonometric path. */
    if (ok) ok = rocm_graph_decode_kv_store(g->kv, raw_cache, raw_cap, raw_row);
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("kv_path");

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
          fprintf(stderr,
                  "ds4: ROCm graph compressor expects paired F16 compressor "
                  "projections\n");
          ok = false;
        }
        if (ok && emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
            fprintf(stderr, "ds4: ROCm graph compressed KV cache capacity exceeded at layer %u\n", il);
            ok = false;
        }
        if (ok) {
          ok = ds4_gpu_matmul_f16_pair_tensor(
                   g->comp_kv_cur, g->comp_sc_cur, model->map, model->size,
                   layer->attn_compressor_kv->abs_offset,
                   layer->attn_compressor_gate->abs_offset, DS4_N_EMBD,
                   comp_width, g->attn_norm, 1) != 0;
        } else {
          if (ok)
            ok = ds4_gpu_matmul_f16_tensor(
                     g->comp_kv_cur, model->map, model->size,
                     layer->attn_compressor_kv->abs_offset, DS4_N_EMBD,
                     comp_width, g->attn_norm, 1) != 0;
          if (ok)
            ok = ds4_gpu_matmul_f16_tensor(
                     g->comp_sc_cur, model->map, model->size,
                     layer->attn_compressor_gate->abs_offset, DS4_N_EMBD,
                     comp_width, g->attn_norm, 1) != 0;
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
              ok = rocm_graph_quantize_attn_comp_row(g, il, comp_row,
                                                     comp_row_view);
              ds4_gpu_tensor_free(comp_row_view);
            }
        }
        if (ok && emit) g->layer_n_comp[il]++;

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
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
                      "ds4: ROCm graph indexer compressor expects paired F16 "
                      "projections\n");
              ok = false;
            }
            if (ok && emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: ROCm graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok) {
              ok = ds4_gpu_matmul_f16_pair_tensor(
                       g->comp_kv_cur, g->comp_sc_cur, model->map, model->size,
                       layer->indexer_compressor_kv->abs_offset,
                       layer->indexer_compressor_gate->abs_offset, DS4_N_EMBD,
                       index_width, g->attn_norm, 1) != 0;
            } else {
              if (ok)
                ok = ds4_gpu_matmul_f16_tensor(
                         g->comp_kv_cur, model->map, model->size,
                         layer->indexer_compressor_kv->abs_offset, DS4_N_EMBD,
                         index_width, g->attn_norm, 1) != 0;
              if (ok)
                ok = ds4_gpu_matmul_f16_tensor(
                         g->comp_sc_cur, model->map, model->size,
                         layer->indexer_compressor_gate->abs_offset, DS4_N_EMBD,
                         index_width, g->attn_norm, 1) != 0;
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
            if (ok && g->layer_n_comp[il] > DS4_N_INDEXER_TOP_K &&
                g->layer_n_index_comp[il] > DS4_N_INDEXER_TOP_K) {
              const uint64_t indexer_q_dim =
                  (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
              if (!layer->indexer_attn_q_b ||
                  layer->indexer_attn_q_b->type != DS4_TENSOR_F16 ||
                  layer->indexer_attn_q_b->dim[0] != q_rank ||
                  layer->indexer_attn_q_b->dim[1] != indexer_q_dim) {
                fprintf(stderr,
                        "ds4: ROCm graph indexer q projection expects F16 "
                        "weights\n");
                ok = false;
              }
              if (ok && (!layer->indexer_proj ||
                         layer->indexer_proj->type != DS4_TENSOR_F16 ||
                         layer->indexer_proj->dim[0] != DS4_N_EMBD ||
                         layer->indexer_proj->dim[1] != DS4_N_INDEXER_HEAD)) {
                fprintf(stderr,
                        "ds4: ROCm graph indexer weight projection expects F16 "
                        "weights\n");
                ok = false;
              }
              if (ok)
                ok = ds4_gpu_matmul_f16_tensor(
                         g->indexer_q, model->map, model->size,
                         layer->indexer_attn_q_b->abs_offset, q_rank,
                         indexer_q_dim, g->qr_norm, 1) != 0;
              if (ok)
                ok = ds4_gpu_rope_tail_tensor(
                         g->indexer_q, 1, DS4_N_INDEXER_HEAD,
                         DS4_N_INDEXER_HEAD_DIM, DS4_N_ROT, pos,
                         compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false,
                         freq_base, freq_scale, ext_factor, attn_factor,
                         DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
              if (ok)
                ok = ds4_gpu_dsv4_indexer_qat_tensor(
                         g->indexer_q, DS4_N_INDEXER_HEAD,
                         DS4_N_INDEXER_HEAD_DIM) != 0;
              if (ok)
                ok = ds4_gpu_matmul_f16_tensor(
                         g->indexer_weights, model->map, model->size,
                         layer->indexer_proj->abs_offset, DS4_N_EMBD,
                         DS4_N_INDEXER_HEAD, g->attn_norm, 1) != 0;
              const float index_scale =
                  1.0f /
                  sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
              if (ok && decode_index_stage_profile) {
                ok = rocm_graph_indexer_stage_profile_boundary(
                    NULL, il, pos, 1, g->layer_n_index_comp[il],
                    &decode_index_stage_t0);
              }
              if (ok)
                ok = rocm_graph_reserve_indexer_scores(
                         g, g->layer_n_index_comp[il], 1) &&
                     ds4_gpu_indexer_score_one_tensor(
                         g->indexer_scores, g->indexer_q, g->indexer_weights,
                         g->layer_index_comp_cache[il],
                         g->layer_n_index_comp[il], DS4_N_INDEXER_HEAD,
                         DS4_N_INDEXER_HEAD_DIM, index_scale) != 0;
              if (ok && decode_index_stage_profile) {
                ok = rocm_graph_indexer_stage_profile_boundary(
                    "decode_score", il, pos, 1, g->layer_n_index_comp[il],
                    &decode_index_stage_t0);
              }
              if (ok)
                ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                 g->indexer_scores,
                                                 g->layer_n_index_comp[il], 1,
                                                 DS4_N_INDEXER_TOP_K) != 0;
              if (ok && decode_index_stage_profile) {
                ok = rocm_graph_indexer_stage_profile_boundary(
                    "decode_topk", il, pos, 1, g->layer_n_index_comp[il],
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
      ok = ds4_gpu_attention_output_low_q8_tensor(
               g->attn_low, model->map, model->size,
               layer->attn_output_a->abs_offset, group_dim, rank, n_groups,
               g->heads) != 0;
      if (ok) {
        ok = ds4_gpu_matmul_q8_0_hc_expand_tensor(
                 g->after_attn_hc, g->attn_out, model->map, model->size,
                 layer->attn_output_b->abs_offset, (uint64_t)n_groups * rank,
                 DS4_N_EMBD, g->attn_low, g->cur_hc, g->hc_split, DS4_N_EMBD,
                 DS4_N_HC) != 0;
      }
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("attn_output");

    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("attn_hc_post");
    if (ok) ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->after_attn_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = rocm_graph_matmul_plain_tensor(g->hc_mix, model, layer->hc_ffn_fn,
                                                 hc_dim, mix_hc, g->flat_hc, 1);
    if (ok) {
      ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(
               g->ffn_cur, g->ffn_norm, g->hc_split, g->hc_mix,
               g->after_attn_hc, model->map, model->size,
               layer->hc_ffn_scale->abs_offset, layer->hc_ffn_base->abs_offset,
               layer->ffn_norm->abs_offset, DS4_N_EMBD, DS4_N_HC,
               DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, DS4_RMS_EPS) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("ffn_hc_pre");
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("ffn_norm");
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
    if (ok)
      ok = ds4_gpu_routed_moe_one_tensor(
               g->routed_out, g->routed_gate, g->routed_up, g->routed_mid,
               g->routed_down, model->map, model->size,
               layer->ffn_gate_exps->abs_offset, layer->ffn_up_exps->abs_offset,
               layer->ffn_down_exps->abs_offset, layer->ffn_gate_exps->type,
               layer->ffn_down_exps->type, gate_expert_bytes, gate_row_bytes,
               down_expert_bytes, down_row_bytes, (uint32_t)expert_in_dim,
               (uint32_t)down_in_dim, (uint32_t)routed_out_dim,
               g->router_selected, g->router_weights, DS4_N_EXPERT,
               DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, g->ffn_norm) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("routed_moe");
    if (ok) {
      ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
               g->shared_gate, g->shared_up, g->shared_mid, model->map,
               model->size, layer->ffn_gate_shexp->abs_offset,
               layer->ffn_up_shexp->abs_offset, DS4_N_EMBD, shared_dim,
               g->ffn_norm, DS4_SWIGLU_CLAMP_EXP, 1u) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("shared_gate_up");
    if (ok) {
      ok = ds4_gpu_shared_down_hc_expand_q8_0_tensor(
               g->after_ffn_hc, g->shared_out, model->map, model->size,
               layer->ffn_down_shexp->abs_offset, shared_dim, DS4_N_EMBD,
               g->shared_mid, g->routed_out, g->after_attn_hc, g->hc_split,
               DS4_N_EMBD, DS4_N_HC) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("shared_down");

    GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE("ffn_hc_post");
#undef GUFO_DEEPSEEK_ROCM_PROFILE_DECODE_STAGE
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
    if (ok)
      ok = ds4_gpu_matmul_f16_tensor(g->output_pre, model->map, model->size,
                                     weights->output_hc_fn->abs_offset, hc_dim,
                                     DS4_N_HC, g->flat_hc, 1) != 0;
    if (ok)
      ok = ds4_gpu_output_hc_weights_tensor(
               g->output_weights, g->output_pre, model->map, model->size,
               weights->output_hc_scale->abs_offset,
               weights->output_hc_base->abs_offset, DS4_N_HC, DS4_HC_EPS) != 0;
    if (ok)
      ok = ds4_gpu_hc_weighted_sum_tensor(g->output_embd, g->cur_hc,
                                          g->output_weights, DS4_N_EMBD,
                                          DS4_N_HC) != 0;
    if (ok)
      ok = ds4_gpu_rms_norm_weight_tensor(
               g->output_norm, g->output_embd, model->map, model->size,
               weights->output_norm->abs_offset, DS4_N_EMBD, DS4_RMS_EPS) != 0;
    if (ok)
      ok = ds4_gpu_matmul_q8_0_tensor(g->logits, model->map, model->size,
                                      weights->output->abs_offset, DS4_N_EMBD,
                                      vocab_dim, g->output_norm, 1) != 0;
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
    if (!g || !model || !state_kv || !state_score || !kv_weight ||
        !score_weight || !ape || head_dim == 0 || width == 0) {
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
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *precomputed_comp_kv,
        const ds4_gpu_tensor *precomputed_comp_sc) {
  if (!g || !model || !layer || !attn_norm || !qr_norm || !q || !kv || !heads ||
      g->raw_cap == 0 ||
      ((precomputed_comp_kv == nullptr) != (precomputed_comp_sc == nullptr))) {
    return false;
  }
  if (!rocm_graph_reserve_compressed_position(g, il,
                                              static_cast<uint64_t>(pos) + 1))
    return false;

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
        if (ok && !precomputed_comp_kv) {
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
        const ds4_gpu_tensor *attn_comp_kv =
            precomputed_comp_kv ? precomputed_comp_kv : g->comp_kv_cur;
        const ds4_gpu_tensor *attn_comp_sc =
            precomputed_comp_sc ? precomputed_comp_sc : g->comp_sc_cur;
        const uint32_t comp_row = g->layer_n_comp[il];
        if (ok) {
            ok = ds4_gpu_compressor_update_tensor(
                     attn_comp_kv,
                     attn_comp_sc,
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
            if (ok && g->layer_n_comp[il] > DS4_N_INDEXER_TOP_K &&
                g->layer_n_index_comp[il] > DS4_N_INDEXER_TOP_K) {
              const uint64_t indexer_q_dim =
                  (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
              if (!layer->indexer_attn_q_b ||
                  layer->indexer_attn_q_b->type != DS4_TENSOR_F16 ||
                  layer->indexer_attn_q_b->dim[0] != q_rank ||
                  layer->indexer_attn_q_b->dim[1] != indexer_q_dim ||
                  !layer->indexer_proj ||
                  layer->indexer_proj->type != DS4_TENSOR_F16 ||
                  layer->indexer_proj->dim[0] != DS4_N_EMBD ||
                  layer->indexer_proj->dim[1] != DS4_N_INDEXER_HEAD) {
                fprintf(stderr,
                        "ds4: ROCm graph indexer projections have "
                        "unexpected shapes\n");
                ok = false;
              }
              if (ok) {
                ok = ds4_gpu_matmul_f16_tensor(
                         g->indexer_q, model->map, model->size,
                         layer->indexer_attn_q_b->abs_offset, q_rank,
                         indexer_q_dim, qr_norm, 1) != 0;
              }
              if (ok) {
                ok = ds4_gpu_rope_tail_tensor(
                         g->indexer_q, 1, DS4_N_INDEXER_HEAD,
                         DS4_N_INDEXER_HEAD_DIM, DS4_N_ROT, pos,
                         compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false,
                         freq_base, freq_scale, ext_factor, attn_factor,
                         DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
              }
              if (ok) {
                ok = ds4_gpu_dsv4_indexer_qat_tensor(
                         g->indexer_q, DS4_N_INDEXER_HEAD,
                         DS4_N_INDEXER_HEAD_DIM) != 0;
              }
              if (ok) {
                ok = ds4_gpu_matmul_f16_tensor(
                         g->indexer_weights, model->map, model->size,
                         layer->indexer_proj->abs_offset, DS4_N_EMBD,
                         DS4_N_INDEXER_HEAD, attn_norm, 1) != 0;
              }
              const float index_scale =
                  1.0f /
                  sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
              if (ok) {
                ok = rocm_graph_reserve_indexer_scores(
                         g, g->layer_n_index_comp[il], 1) &&
                     ds4_gpu_indexer_score_one_tensor(
                         g->indexer_scores, g->indexer_q, g->indexer_weights,
                         g->layer_index_comp_cache[il],
                         g->layer_n_index_comp[il], DS4_N_INDEXER_HEAD,
                         DS4_N_INDEXER_HEAD_DIM, index_scale) != 0;
              }
              if (ok) {
                ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                 g->indexer_scores,
                                                 g->layer_n_index_comp[il], 1,
                                                 DS4_N_INDEXER_TOP_K) != 0;
              }
              if (ok) {
                comp_selected = g->comp_selected;
                n_selected = DS4_N_INDEXER_TOP_K < g->layer_n_index_comp[il]
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
        if (n_comp != 0 && comp_selected != NULL && n_selected != 0) {
          ok =
              ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                  heads, model->map, model->size, layer->attn_sinks->abs_offset,
                  q, raw_cache, g->layer_attn_comp_cache[il],
                  rocm_graph_attn_comp_cache_is_f16(), comp_selected, 1, pos,
                  n_raw, g->raw_cap, raw_start, n_comp, n_selected,
                  g->raw_window, ds4_layer_compress_ratio(il), DS4_N_HEAD,
                  DS4_N_HEAD_DIM) != 0;
        } else {
          ok = ds4_gpu_attention_decode_heads_tensor(
                   heads, model->map, model->size,
                   layer->attn_sinks->abs_offset, q, raw_cache, n_raw,
                   g->raw_cap, raw_start, n_comp ? comp_cache : NULL,
                   rocm_graph_attn_comp_cache_is_f16(), n_comp, NULL, 0,
                   DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
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
        uint32_t                n_tokens,
        bool                    front_ready = false,
        bool                    front_only = false,
        bool                    project_qb = true,
        bool                    project_q_a_kv = true,
        bool                    compressor_projection_ready = false,
        bool                    indexer_query_projection_ready = false,
        bool                    project_output = true) {
  if (n_tokens == 0 || n_tokens > g->prefill_cap ||
      (front_ready && front_only)) {
    return false;
  }
  if (!rocm_graph_reserve_compressed_position(
          g, il, static_cast<uint64_t>(pos0) + n_tokens))
    return false;

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
  const bool index_stage_profile =
      getenv("GUFO_DEEPSEEK_ROCM_INDEXER_STAGE_PROFILE") != NULL;
  const bool layer_stage_profile =
      getenv("GUFO_DEEPSEEK_ROCM_LAYER_STAGE_PROFILE") != NULL;
  const bool q_stage_profile =
      getenv("GUFO_DEEPSEEK_ROCM_Q_STAGE_PROFILE") != NULL;
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
    auto* index_counts =
        ratio == 4
            ? static_cast<uint32_t*>(ds4_xcalloc(n_tokens, sizeof(uint32_t)))
            : nullptr;
    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
            g->batch_attn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(
            g->batch_after_attn_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;
    if (!front_ready) {
      bool hc_pre_fused = false;
      if (ok && n_tokens >= 128u) {
        hc_pre_fused = true;
        ok =
            ds4_gpu_hc_norm_mix_split_weighted_sum_tensor(
                attn_cur_view, hc_mix_view, hc_split_view, g->batch_cur_hc,
                model->map, model->size, layer->hc_attn_fn->abs_offset,
                layer->hc_attn_scale->abs_offset,
                layer->hc_attn_base->abs_offset, DS4_N_EMBD, DS4_N_HC, n_tokens,
                DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, DS4_RMS_EPS) != 0;
      }
      if (ok && !hc_pre_fused)
        ok = ds4_gpu_rms_norm_plain_rows_tensor(
                 g->batch_flat_hc, g->batch_cur_hc, (uint32_t)hc_dim, n_tokens,
                 DS4_RMS_EPS) != 0;
      if (ok && !hc_pre_fused)
        ok = ds4_gpu_matmul_f16_tensor(hc_mix_view, model->map, model->size,
                                       layer->hc_attn_fn->abs_offset, hc_dim,
                                       mix_hc, g->batch_flat_hc, n_tokens) != 0;
      if (!hc_pre_fused) {
        if (ok)
          ok = ds4_gpu_hc_split_weighted_sum_tensor(
                   attn_cur_view, hc_split_view, hc_mix_view, g->batch_cur_hc,
                   model->map, model->size, layer->hc_attn_scale->abs_offset,
                   layer->hc_attn_base->abs_offset, DS4_N_EMBD, DS4_N_HC,
                   DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS) != 0;
      }
      GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("hc_pre");
      if (ok)
        ok = ds4_gpu_rms_norm_weight_rows_tensor(
                 g->batch_attn_norm, g->batch_attn_cur, model->map, model->size,
                 layer->attn_norm->abs_offset, DS4_N_EMBD, n_tokens,
                 DS4_RMS_EPS) != 0;
      /* Eight projections in a ratio-4 layer read these rows, and each F16
       * route among them was converting them again. One mirror serves all of
       * them; the conversion is row-local, so every consumer sees the bytes it
       * would have produced itself. Declining is not an error. */
      if (ok && n_tokens >= 128u) { /* DS4_ROCM_WIDE_PREFILL_ROWS */
        (void)ds4_gpu_publish_f16_input_tensor(g->batch_attn_norm,
                                                (uint64_t)n_tokens * DS4_N_EMBD);
      }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("norm");
    if (project_q_a_kv) {
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("pre_q");
    if (ok)
      ok = rocm_graph_matmul_q8_0_named_tensor(
          "attn_q_a", il, pos0, g->batch_qr, model, layer->attn_q_a, DS4_N_EMBD,
          q_rank, g->batch_attn_norm, n_tokens);
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("q_a");
    {
      if (ok)
        ok = rocm_graph_matmul_q8_0_named_tensor(
            "attn_kv", il, pos0, g->batch_kv_raw, model, layer->attn_kv,
            DS4_N_EMBD, DS4_N_HEAD_DIM, g->batch_attn_norm, n_tokens);
      if (ok)
        ok =
            ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                g->batch_qr_norm, g->batch_qr, model->map, model->size,
                layer->attn_q_a_norm->abs_offset, (uint32_t)q_rank, g->batch_kv,
                g->batch_kv_raw, layer->attn_kv_a_norm->abs_offset,
                DS4_N_HEAD_DIM, n_tokens, DS4_RMS_EPS) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("q_a_norm");
    }
    if (ok && project_qb) {
        ok = rocm_graph_matmul_q8_0_named_tensor("attn_q_b",
                                                 il,
                                                 pos0,
                                                 g->batch_q,
                                                 model,
                                                 layer->attn_q_b,
                                                 q_rank,
                                                 q_dim,
                                                 g->batch_qr_norm,
                                                 n_tokens);
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("q_b");
    }
    if (front_only) {
        ds4_gpu_tensor_free(after_attn_hc_view);
        ds4_gpu_tensor_free(attn_cur_view);
        ds4_gpu_tensor_free(hc_split_view);
        ds4_gpu_tensor_free(hc_mix_view);
        free(index_counts);
        free(comp_counts);
        return ok;
    }
    /* One pass over the query rows instead of two. The fused kernel stages each
     * row in LDS, so the norm's write-back is no longer read again by the rope
     * pass; it is bit-identical and falls back when the row does not fit. */
    bool q_norm_rope_fused =
        ok && n_tokens > 6u &&
        ds4_gpu_head_rms_norm_rope_tail_tensor(
            g->batch_q, n_tokens, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos0,
            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false, freq_base,
            freq_scale, ext_factor, attn_factor, DS4_ROPE_YARN_BETA_FAST,
            DS4_ROPE_YARN_BETA_SLOW, DS4_RMS_EPS) != 0;
    if (ok && !q_norm_rope_fused)
      ok = ds4_gpu_head_rms_norm_tensor(g->batch_q, n_tokens, DS4_N_HEAD,
                                        DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("head_norm");
    if (ok && !q_norm_rope_fused)
      ok = ds4_gpu_rope_tail_tensor(
               g->batch_q, n_tokens, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT,
               pos0, compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false,
               freq_base, freq_scale, ext_factor, attn_factor,
               DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_Q_STAGE("rope");
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("q_path");

    if (ok)
      ok = ds4_gpu_rope_tail_tensor(
               g->batch_kv, n_tokens, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT,
               pos0, compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, false,
               freq_base, freq_scale, ext_factor, attn_factor,
               DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok)
      ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->batch_kv, n_tokens,
                                               DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
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
          ok = ds4_gpu_attention_decode_raw_batch_heads_tensor(
                   g->batch_heads, model->map, model->size,
                   layer->attn_sinks->abs_offset, g->batch_q,
                   g->layer_raw_cache[il], n_tokens, pos0, n_raw, g->raw_cap,
                   raw_start, g->raw_window, DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
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
        if (ok && !compressor_projection_ready) {
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
            // Verification may reject an aligned block partway through.
            // The bulk compressor only saves its final state, so use the
            // row loop when rollback needs intermediate prefix snapshots.
            const bool aligned_chunk =
                !g->spec_capture_prefixes &&
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
                if (ok &&
                    (!attn_comp_target || (ratio == 4 && !attn_comp_mirror))) {
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
                        ok = comp_row_view &&
                             rocm_graph_quantize_attn_comp_row(g, il, comp_row,
                                                               comp_row_view);
                        ds4_gpu_tensor_free(comp_row_view);
                    }
                    if (ok && emit) g->layer_n_comp[il]++;
                    if (comp_counts) comp_counts[t] = g->layer_n_comp[il];
                    if (ok && t + 1u < n_tokens && t < DS4_SPEC_PREFIX_SLOTS) {
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
            if (!layer->indexer_compressor_kv ||
                !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape ||
                !layer->indexer_compressor_norm || !layer->indexer_attn_q_b ||
                !layer->indexer_proj) {
              fprintf(stderr,
                      "ds4: ROCm layer-major prefill needs indexer weights\n");
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
            if (ok && !indexer_query_projection_ready) {
                ok = ds4_gpu_matmul_f16_tensor(g->batch_indexer_q,
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_attn_q_b->abs_offset,
                                                     q_rank,
                                                     (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                     g->batch_qr_norm,
                                                     n_tokens) != 0;
            }
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
            if (ok && !indexer_query_projection_ready) {
                ok = ds4_gpu_matmul_f16_tensor(g->batch_indexer_weights,
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_proj->abs_offset,
                                                     DS4_N_EMBD,
                                                     DS4_N_INDEXER_HEAD,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
            }
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
                    !g->spec_capture_prefixes &&
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

        // A verifier block may cross the first sparse-indexer boundary. Rows
        // before it still use scalar's dense compressed attention order.
        const bool crosses_indexer_boundary =
            g->spec_capture_prefixes && ratio == 4 &&
            n_comp > DS4_N_INDEXER_TOP_K &&
            (pos0 + 1u) / ratio <= DS4_N_INDEXER_TOP_K;
        if (ok && !zero_prefix && n_tokens <= g->raw_cap &&
            !crosses_indexer_boundary) {
            const uint32_t n_raw = rocm_graph_raw_span_for_batch(g, pos0, n_tokens);
            /* See the raw-only branch above: batched mixed attention also
             * consumes a logical raw window, linearized out of the ring. */
            const uint32_t raw_start =
                rocm_graph_raw_start_for_span(g, pos0 + n_tokens - 1u, n_raw);
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
                ok = ok &&
                     rocm_graph_reserve_indexer_scores(g, n_comp, n_tokens) &&
                     ds4_gpu_indexer_scores_decode_batch_tensor(
                         g->indexer_scores, g->batch_indexer_q,
                         g->batch_indexer_weights,
                         g->layer_index_comp_cache[il], n_comp, n_tokens, pos0,
                         DS4_N_INDEXER_HEAD, DS4_N_INDEXER_HEAD_DIM, ratio,
                         index_scale, g->batch_heads) != 0;
                if (ok && index_stage_profile) {
                    ok = rocm_graph_indexer_stage_profile_boundary("score",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                  ok = ds4_gpu_indexer_topk_tensor(
                           g->comp_selected, g->indexer_scores, n_comp,
                           n_tokens, DS4_N_INDEXER_TOP_K) != 0;
                  if (ok && index_stage_profile) {
                    ok = rocm_graph_indexer_stage_profile_boundary(
                        "topk", il, pos0, n_tokens, n_comp, &index_stage_t0);
                  }
                }
                if (ok) {
                    use_indexed_comp = true;
                }
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
                  ok = ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                           g->batch_heads, model->map, model->size,
                           layer->attn_sinks->abs_offset, g->batch_q,
                           g->layer_raw_cache[il], g->layer_attn_comp_cache[il],
                           rocm_graph_attn_comp_cache_is_f16(), nullptr, 0,
                           n_tokens, pos0, n_raw, g->raw_cap, raw_start, n_comp,
                           g->raw_window, ratio, DS4_N_HEAD,
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
            ok = ok && rocm_graph_reserve_indexer_scores(g, n_comp, n_tokens) &&
                 ds4_gpu_indexer_scores_prefill_tensor(
                     g->indexer_scores, g->batch_indexer_q,
                     g->batch_indexer_weights, g->layer_index_comp_cache[il],
                     n_comp, n_tokens, DS4_N_INDEXER_HEAD,
                     DS4_N_INDEXER_HEAD_DIM, ratio, index_scale,
                     g->batch_heads) != 0;
            if (ok && index_stage_profile) {
                ok = rocm_graph_indexer_stage_profile_boundary("score",
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            if (ok) {
              ok = ds4_gpu_indexer_topk_tensor(
                       g->comp_selected, g->indexer_scores, n_comp, n_tokens,
                       DS4_N_INDEXER_TOP_K) != 0;
              if (ok && index_stage_profile) {
                ok = rocm_graph_indexer_stage_profile_boundary(
                    "topk", il, pos0, n_tokens, n_comp, &index_stage_t0);
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

                if (ratio == 4 && cur_comp > DS4_N_INDEXER_TOP_K) {
                    const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    ds4_gpu_tensor *indexer_q_view = rocm_graph_tensor_row_view(
                            g->batch_indexer_q, t, (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM);
                    ds4_gpu_tensor *indexer_w_view = rocm_graph_tensor_row_view(
                            g->batch_indexer_weights, t, DS4_N_INDEXER_HEAD);
                    ok = indexer_q_view && indexer_w_view &&
                         rocm_graph_reserve_indexer_scores(g, cur_index, 1) &&
                         ds4_gpu_indexer_score_one_tensor(
                             g->indexer_scores, indexer_q_view, indexer_w_view,
                             g->layer_index_comp_cache[il], cur_index,
                             DS4_N_INDEXER_HEAD, DS4_N_INDEXER_HEAD_DIM,
                             index_scale) != 0 &&
                         ds4_gpu_indexer_topk_tensor(
                             g->comp_selected, g->indexer_scores, cur_index, 1,
                             DS4_N_INDEXER_TOP_K) != 0;
                    ds4_gpu_tensor_free(indexer_w_view);
                    ds4_gpu_tensor_free(indexer_q_view);
                    if (ok) {
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
                if (ok && n_selected != 0) {
                  ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                           heads_view, model->map, model->size,
                           layer->attn_sinks->abs_offset, q_view,
                           g->layer_raw_cache[il], g->layer_attn_comp_cache[il],
                           rocm_graph_attn_comp_cache_is_f16(),
                           g->comp_selected, 1, pos, n_raw, g->raw_cap,
                           raw_start, cur_comp, n_selected, g->raw_window,
                           ratio, DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
                } else if (ok) {
                  ok = ds4_gpu_attention_decode_heads_tensor(
                           heads_view, model->map, model->size,
                           layer->attn_sinks->abs_offset, q_view,
                           g->layer_raw_cache[il], n_raw, g->raw_cap, raw_start,
                           cur_comp ? g->layer_attn_comp_cache[il] : NULL,
                           rocm_graph_attn_comp_cache_is_f16(), cur_comp,
                           nullptr, 0, DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
                }
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(kv_cache_view);
                ds4_gpu_tensor_free(q_view);
            }
        }
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("attention");

    if (ok && !project_output) {
        ok = ds4_gpu_rope_tail_tensor(
                 g->batch_heads, n_tokens, DS4_N_HEAD, DS4_N_HEAD_DIM,
                 DS4_N_ROT, pos0,
                 compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, true,
                 freq_base, freq_scale, ext_factor, attn_factor,
                 DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
        GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("inv_rope");
        ds4_gpu_tensor_free(after_attn_hc_view);
        ds4_gpu_tensor_free(attn_cur_view);
        ds4_gpu_tensor_free(hc_split_view);
        ds4_gpu_tensor_free(hc_mix_view);
        free(index_counts);
        free(comp_counts);
        return ok;
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
    if (ok && !output_hc_fused && !inv_rope_fused) {
      ok = ds4_gpu_rope_tail_tensor(
               g->batch_heads, n_tokens, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT,
               pos0, compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0, true,
               freq_base, freq_scale, ext_factor, attn_factor,
               DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("inv_rope");
    if (ok && !output_hc_fused && !inv_rope_fused) {
      ok = ds4_gpu_attention_output_q8_batch_tensor(
               g->batch_attn_out, g->batch_attn_low, g->batch_group_tmp,
               g->batch_low_tmp, model->map, model->size,
               layer->attn_output_a->abs_offset,
               layer->attn_output_b->abs_offset, group_dim, rank, n_groups,
               DS4_N_EMBD, g->batch_heads, n_tokens) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_ATTN_STAGE("output_proj");
    if (ok && !output_hc_fused) {
        ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                                  g->batch_attn_out,
                                                  g->batch_cur_hc,
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
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

    /*
     * The compressor projections use the same ordered per-row reduction as
     * serial decode while sharing both F16 weight streams across C2-C8.
     */
    const bool use_compressor_batch = ds4_layer_compress_ratio(il) != 0u;
    uint32_t compressor_width = 0u;
    if (ok && use_compressor_batch) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t coff = ratio == 4u ? 2u : 1u;
        compressor_width = coff * DS4_N_HEAD_DIM;
        if (!layer->attn_compressor_kv || !layer->attn_compressor_gate ||
            layer->attn_compressor_kv->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_gate->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_kv->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_gate->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_kv->dim[1] != compressor_width ||
            layer->attn_compressor_gate->dim[1] != compressor_width) {
          ok = false;
        } else {
          ok = ds4_gpu_matmul_f16_pair_narrow_tensor(
                   g->batch_comp_kv, g->batch_comp_sc, model->map, model->size,
                   layer->attn_compressor_kv->abs_offset,
                   layer->attn_compressor_gate->abs_offset, DS4_N_EMBD,
                   compressor_width, g->batch_attn_norm, n_tokens) != 0;
        }
    }
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
        ds4_gpu_tensor *comp_kv = use_compressor_batch
            ? rocm_graph_tensor_row_view(
                  g->batch_comp_kv, row, compressor_width)
            : nullptr;
        ds4_gpu_tensor *comp_sc = use_compressor_batch
            ? rocm_graph_tensor_row_view(
                  g->batch_comp_sc, row, compressor_width)
            : nullptr;
        ok = attn_norm && qr_norm && q && kv && heads &&
             (!use_compressor_batch || (comp_kv && comp_sc));
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
                heads,
                comp_kv,
                comp_sc);
        }
        ds4_gpu_tensor_free(comp_sc);
        ds4_gpu_tensor_free(comp_kv);
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
    const ds4_gpu_tensor* tokens = nullptr,
    const uint32_t* router_group_offsets = nullptr,
    uint32_t router_group_count = 0) {
  if (n_tokens == 0 || n_tokens > g->prefill_cap)
    return false;
  if (!tokens)
    tokens = g->prefill_tokens;
  if ((router_group_offsets != nullptr) != (router_group_count != 0u) ||
      router_group_count > 8u) {
    return false;
  }
  std::array<uint32_t, 9> group_offsets{};
  const uint32_t group_count = router_group_count;
  if (router_group_offsets != nullptr) {
    if (router_group_offsets[0] != 0u ||
        router_group_offsets[router_group_count] != n_tokens) {
      return false;
    }
    for (uint32_t group = 0; group <= router_group_count; ++group) {
      group_offsets[group] = router_group_offsets[group];
      if (group != 0u && group_offsets[group] <= group_offsets[group - 1u]) {
        return false;
      }
    }
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
    bool hc_pre_fused = false;
    if (ok && n_tokens >= 128u) {
      hc_pre_fused = true;
      ok = ds4_gpu_hc_norm_mix_split_weighted_sum_tensor(
               ffn_cur_view, hc_mix_view, hc_split_view, g->batch_after_attn_hc,
               model->map, model->size, layer->hc_ffn_fn->abs_offset,
               layer->hc_ffn_scale->abs_offset, layer->hc_ffn_base->abs_offset,
               DS4_N_EMBD, DS4_N_HC, n_tokens, DS4_N_HC_SINKHORN_ITER,
               DS4_HC_EPS, DS4_RMS_EPS) != 0;
    }
    if (ok && !hc_pre_fused)
      ok = ds4_gpu_rms_norm_plain_rows_tensor(
               g->batch_flat_hc, g->batch_after_attn_hc, (uint32_t)hc_dim,
               n_tokens, DS4_RMS_EPS) != 0;
    if (ok && !hc_pre_fused) {
      ok = ds4_gpu_matmul_f16_tensor(hc_mix_view, model->map, model->size,
                                     layer->hc_ffn_fn->abs_offset, hc_dim,
                                     mix_hc, g->batch_flat_hc, n_tokens) != 0;
    }
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
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("hc_pre");
    if (ok && hc_pre_fused)
      ok = ds4_gpu_rms_norm_weight_rows_tensor(
               g->batch_ffn_norm, g->batch_ffn_cur, model->map, model->size,
               layer->ffn_norm->abs_offset, DS4_N_EMBD, n_tokens,
               DS4_RMS_EPS) != 0;
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("norm");
    if (ok && group_count != 0u) {
      const uint64_t logits_row_bytes = (uint64_t)DS4_N_EXPERT * sizeof(float);
      const uint64_t selected_row_bytes =
          (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t);
      const uint64_t weights_row_bytes =
          (uint64_t)DS4_N_EXPERT_USED * sizeof(float);
      for (uint32_t group = 0; ok && group < group_count; ++group) {
        const uint32_t row0 = group_offsets[group];
        const uint32_t group_rows =
            group_offsets[group + 1u] - row0;
        ds4_gpu_tensor* logits_view = ds4_gpu_tensor_view(
            g->batch_router_logits, (uint64_t)row0 * logits_row_bytes,
            (uint64_t)group_rows * logits_row_bytes);
        ds4_gpu_tensor* probs_view = ds4_gpu_tensor_view(
            g->batch_router_probs, (uint64_t)row0 * logits_row_bytes,
            (uint64_t)group_rows * logits_row_bytes);
        ds4_gpu_tensor* selected_view = ds4_gpu_tensor_view(
            g->batch_router_selected, (uint64_t)row0 * selected_row_bytes,
            (uint64_t)group_rows * selected_row_bytes);
        ds4_gpu_tensor* weights_view = ds4_gpu_tensor_view(
            g->batch_router_weights, (uint64_t)row0 * weights_row_bytes,
            (uint64_t)group_rows * weights_row_bytes);
        ds4_gpu_tensor* norm_view = ds4_gpu_tensor_view(
            g->batch_ffn_norm, (uint64_t)row0 * DS4_N_EMBD * sizeof(float),
            (uint64_t)group_rows * DS4_N_EMBD * sizeof(float));
        ds4_gpu_tensor* tokens_view =
            ds4_gpu_tensor_view(tokens, (uint64_t)row0 * sizeof(int32_t),
                                (uint64_t)group_rows * sizeof(int32_t));
        ok = logits_view && probs_view && selected_view && weights_view &&
             norm_view && tokens_view;
        if (ok) {
          ok = rocm_graph_matmul_plain_tensor(
              logits_view, model, layer->ffn_gate_inp, DS4_N_EMBD, DS4_N_EXPERT,
              norm_view, group_rows);
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
                   group_rows) != 0;
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
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("router");

    /* Ask the routed MoE to leave its per-expert F16 rows unsummed so the
     * hyper-connection expansion below can fold the 6-way sum in, saving one
     * round trip through batch_routed_out. Advisory: only the F16 down route can
     * do it, so the outcome is read back after the call. */
    if (ok) ds4_gpu_set_routed_defer_sum(n_tokens >= 128u ? 1 : 0);
    if (ok) {
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
               DS4_SWIGLU_CLAMP_EXP, g->batch_ffn_norm, n_tokens) != 0;
    }
    GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("routed_moe");
    if (ok && n_tokens <= 136u && (n_tokens <= 16u || group_count != 0u)) {
      ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
               g->batch_shared_gate, g->batch_shared_up, g->batch_shared_mid,
               model->map, model->size, layer->ffn_gate_shexp->abs_offset,
               layer->ffn_up_shexp->abs_offset, DS4_N_EMBD, shared_dim,
               g->batch_ffn_norm, DS4_SWIGLU_CLAMP_EXP, n_tokens) != 0;
    } else {
      if (ok)
        ok = rocm_graph_matmul_q8_0_named_tensor(
            "shared_gate", il, pos0, g->batch_shared_gate, model,
            layer->ffn_gate_shexp, DS4_N_EMBD, shared_dim, g->batch_ffn_norm,
            n_tokens);
      if (ok)
        ok = rocm_graph_matmul_q8_0_named_tensor(
            "shared_up", il, pos0, g->batch_shared_up, model,
            layer->ffn_up_shexp, DS4_N_EMBD, shared_dim, g->batch_ffn_norm,
            n_tokens);
      GUFO_DEEPSEEK_ROCM_PROFILE_FFN_STAGE("shared_gate_up");
      if (ok)
        ok = ds4_gpu_swiglu_tensor(g->batch_shared_mid, g->batch_shared_gate,
                                   g->batch_shared_up,
                                   (uint32_t)((uint64_t)n_tokens * shared_dim),
                                   DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
    }
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
                  "ds4: fused routed sum expansion rejected the shape\n");
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
    const double t0 = profile ? ds4_now_seconds() : 0.0;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = rocm_graph_encode_token_raw_swa(g, model, weights, token, pos, logits != NULL, true);
    const double t_encoded = profile ? ds4_now_seconds() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    const double t_done = profile ? ds4_now_seconds() : 0.0;

    if (ok && logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    const double t_read = profile ? ds4_now_seconds() : 0.0;
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

    // Grow before prefill dispatch and leave one raw window for decoding.
    // Normal decode only grows when its actual rows exceed the allocation.
    for (uint32_t il = 0; il < DS4_N_LAYER; ++il) {
      if (!rocm_graph_reserve_compressed_position(
              g, il, static_cast<uint64_t>(start) + n_tokens + DS4_N_SWA))
        return false;
    }

    bool ok = rocm_graph_upload_prompt_tokens(g->prefill_tokens, prompt, start, n_tokens);
    if (!ok) return false;

    const bool profile =
        getenv("GUFO_DEEPSEEK_ROCM_GRAPH_PREFILL_PROFILE") != NULL;
    const double t0 = profile ? ds4_now_seconds() : 0.0;

    ok = rocm_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                 g->prefill_tokens,
                                                 model,
                                                 weights,
                                                 prompt,
                                                 start,
                                                 n_tokens);
    if (ok)
      ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
      // Keep feature capture and the HC swap in the common layer encoder.
      ok = rocm_graph_encode_layer_batch(g, model, &weights->layer[il], il,
                                         start, n_tokens);
      if (show_progress) {
        fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1,
                (uint32_t)DS4_N_LAYER);
        fflush(stderr);
      }
    }
    if (show_progress) fputc('\n', stderr);

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint32_t output_row = n_tokens - 1u;
    ds4_gpu_tensor *saved_cur = g->cur_hc;
    ds4_gpu_tensor *last_hc = NULL;
    if (ok && logits) {
      last_hc = rocm_graph_tensor_row_view(g->batch_cur_hc, output_row, hc_dim);
      ok = last_hc != NULL;
    }
    if (ok && logits) {
        g->cur_hc = last_hc;
        ok = rocm_graph_encode_output_head(g, model, weights,
                                           weights->output->dim[1]);
        g->cur_hc = saved_cur;
    }

    const double t_encoded = profile ? ds4_now_seconds() : 0.0;
    if (ok)
      ok = ds4_gpu_end_commands() != 0;
    const double t_done = profile ? ds4_now_seconds() : 0.0;
    if (last_hc) ds4_gpu_tensor_free(last_hc);
    if (!ok) {
      if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr,
                "ds4: ROCm synchronize after prefill failure also failed\n");
      }
      return false;
    }

    const double t_before_read = profile ? ds4_now_seconds() : 0.0;
    if (logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (profile) {
        const double t_read = ds4_now_seconds();
        const double encode_s = t_encoded - t0;
        const double execute_s = t_done - t_encoded;
        fprintf(stderr,
                "ds4: gpu graph prefill total tokens=%u encode=%.3f ms "
                "execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                n_tokens, encode_s * 1000.0, execute_s * 1000.0,
                (t_read - t_before_read) * 1000.0, (t_read - t0) * 1000.0);
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
        !rocm_graph_dspark_resize_capture(g, g->dspark->block_size + 1u)) {
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
    if (g->spec_rows_cap >= rows_cap && g->spec_logits && g->spec_row_tops &&
        g->spec_frontier_logits) {
      return true;
    }

    rocm_graph_spec_free(g);
    const uint64_t vocab_dim = weights->output->dim[1];
    g->spec_logits =
        ds4_gpu_tensor_alloc((uint64_t)rows_cap * vocab_dim * sizeof(float));
    g->spec_row_tops =
        ds4_gpu_tensor_alloc((uint64_t)rows_cap * sizeof(int32_t));
    g->spec_frontier_logits = ds4_gpu_tensor_alloc(
        (uint64_t)(rows_cap < 8u ? rows_cap : 8u) *
        vocab_dim * sizeof(float));
    bool ok = g->spec_logits != NULL && g->spec_row_tops != NULL &&
              g->spec_frontier_logits != NULL;

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
    if (slot >= DS4_SPEC_PREFIX_SLOTS || !g->spec_prefix_attn_state_kv[il] ||
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
    if (slot >= DS4_SPEC_PREFIX_SLOTS || !g->spec_prefix_index_state_kv[il] ||
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
            (uint64_t)g->dspark_capture_rows_cap * DS4_N_EMBD * sizeof(float));
        g->dspark_markov_key = ds4_gpu_tensor_alloc(
                8u * sizeof(unsigned long long));
        g->dspark_markov_index =
                ds4_gpu_tensor_alloc(8u * sizeof(int32_t));
        ok = g->dspark_features && g->dspark_fused && g->dspark_markov_key &&
             g->dspark_markov_index;
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
        if (ds4_gpu_tensor_copy(g->layer_attn_state_kv[il], 0,
                                g->spec_prefix_attn_state_kv[il], attn_offset,
                                attn_bytes) == 0 ||
            ds4_gpu_tensor_copy(g->layer_attn_state_score[il], 0,
                                g->spec_prefix_attn_state_score[il],
                                attn_offset, attn_bytes) == 0) {
          return false;
        }
        if (ratio != 4) continue;
        const uint64_t index_bytes =
            (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio *
            sizeof(float);
        const uint64_t index_offset = (uint64_t)slot * index_bytes;
        g->layer_n_index_comp[il] =
            g->spec_prefix_n_index_comp[slot][il];
        if (ds4_gpu_tensor_copy(g->layer_index_state_kv[il], 0,
                                g->spec_prefix_index_state_kv[il], index_offset,
                                index_bytes) == 0 ||
            ds4_gpu_tensor_copy(g->layer_index_state_score[il], 0,
                                g->spec_prefix_index_state_score[il],
                                index_offset, index_bytes) == 0) {
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

static bool rocm_graph_distribute_verify_results(
    ds4_gpu_graph* coordinator, const ds4_rocm_verify_item* items,
    size_t item_count, const std::array<uint32_t, 9>& row_offsets,
    const std::vector<int32_t>& flat_tops, uint64_t vocab_dim) {
  if (!coordinator || !items || item_count == 0u ||
      flat_tops.size() < row_offsets[item_count]) {
    return false;
  }

  bool gather_frontiers = coordinator->spec_frontier_logits != nullptr;
  for (size_t index = 0; index < item_count; ++index) {
    const ds4_rocm_verify_item& item = items[index];
    memcpy(item.row_tops, flat_tops.data() + row_offsets[index],
           (size_t)item.n_tokens * sizeof(item.row_tops[0]));
    gather_frontiers =
        gather_frontiers && item.frontier_logits != nullptr &&
        item.logical_n_tokens != 0u &&
        item.logical_n_tokens <= item.n_tokens;
  }

  const uint64_t logits_row_bytes = vocab_dim * sizeof(float);
  if (gather_frontiers) {
    // Sampled items decide acceptance on the CPU, so they take every verified
    // row instead of the argmax-selected frontier.
    for (size_t index = 0; index < item_count; ++index) {
      const ds4_rocm_verify_item& item = items[index];
      if (item.row_logits == nullptr) continue;
      if (ds4_gpu_tensor_read(coordinator->spec_logits,
                              (uint64_t)row_offsets[index] * logits_row_bytes,
                              item.row_logits,
                              (uint64_t)item.n_tokens * logits_row_bytes) == 0) {
        return false;
      }
    }
    bool ok = ds4_gpu_begin_commands() != 0;
    for (size_t index = 0; ok && index < item_count; ++index) {
      const ds4_rocm_verify_item& item = items[index];
      if (item.row_logits != nullptr) continue;
      uint32_t accepted = 1u;
      while (accepted < item.logical_n_tokens &&
             item.row_tops[accepted - 1u] ==
                 item.tokens->v[item.start + accepted]) {
        ++accepted;
      }
      const uint32_t source_row =
          row_offsets[index] + accepted - 1u;
      ok = ds4_gpu_tensor_copy(
               coordinator->spec_frontier_logits,
               (uint64_t)index * logits_row_bytes,
               coordinator->spec_logits,
               (uint64_t)source_row * logits_row_bytes,
               logits_row_bytes) != 0;
    }
    if (ok) {
      ok = ds4_gpu_end_commands() != 0;
    } else {
      (void)ds4_gpu_synchronize();
    }
    if (!ok) return false;

    std::vector<float> frontier_logits(item_count * vocab_dim);
    if (ds4_gpu_tensor_read(
            coordinator->spec_frontier_logits, 0, frontier_logits.data(),
            (uint64_t)item_count * logits_row_bytes) == 0) {
      return false;
    }
    for (size_t index = 0; index < item_count; ++index) {
      if (items[index].row_logits != nullptr) continue;
      memcpy(items[index].frontier_logits,
             frontier_logits.data() + index * vocab_dim,
             (size_t)logits_row_bytes);
    }
    return true;
  }

  for (size_t index = 0; index < item_count; ++index) {
    if (index != 0u &&
        ds4_gpu_tensor_copy(
            items[index].graph->spec_logits, 0, coordinator->spec_logits,
            (uint64_t)row_offsets[index] * logits_row_bytes,
            (uint64_t)items[index].n_tokens * logits_row_bytes) == 0) {
      return false;
    }
  }
  return true;
}

/*
 * Verify one independently sized draft block per session.
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
  std::array<uint32_t, 9> row_offsets{};
  bool saved_capture_prefixes[8] = {};
  for (size_t index = 0; index < item_count; ++index) {
    const ds4_rocm_verify_item& item = items[index];
    if (!item.graph || !item.tokens || !item.row_tops || item.n_tokens == 0u ||
        item.logical_n_tokens == 0u || item.logical_n_tokens > item.n_tokens ||
        item.start > static_cast<uint32_t>(item.tokens->len) ||
        item.n_tokens > static_cast<uint32_t>(item.tokens->len) - item.start) {
      return false;
    }
    row_offsets[index + 1u] = row_offsets[index] + item.n_tokens;
    saved_capture_prefixes[index] = item.graph->spec_capture_prefixes;
  }
  const uint32_t total_rows = row_offsets[item_count];
  if (total_rows > coordinator->prefill_cap ||
      total_rows > coordinator->batch_dspark_verify_rows_cap) {
    return false;
  }

  std::vector<int32_t> flat_tokens(total_rows);
  std::vector<int32_t> flat_tops(total_rows);

  size_t prepared_count = 0;
  for (; prepared_count < item_count; ++prepared_count) {
    const ds4_rocm_verify_item& item = items[prepared_count];
    const uint32_t rows_cap =
        prepared_count == 0u ? total_rows : item.n_tokens;
    if (!rocm_graph_spec_prepare(item.graph, engine->weights, rows_cap) ||
        !rocm_graph_spec_frontier_save(item.graph)) {
      break;
    }
    item.graph->spec_capture_prefixes =
        item.n_tokens > 1u &&
        item.n_tokens <= DS4_SPEC_PREFIX_SLOTS + 2u;
    ds4_rocm_graph_dspark_capture_reset(item.graph);
    for (uint32_t row = 0; row < item.n_tokens; ++row) {
      flat_tokens[row_offsets[prepared_count] + row] =
          item.tokens->v[item.start + row];
    }
  }
  if (prepared_count != item_count) {
    for (size_t index = 0; index < prepared_count; ++index) {
      items[index].graph->spec_capture_prefixes = saved_capture_prefixes[index];
    }
    return false;
  }

  const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
  const uint64_t hc_row_bytes = hc_dim * sizeof(float);

  bool attn_group_pairs_compatible = (item_count & 1u) == 0u;
  for (size_t index = 0;
       attn_group_pairs_compatible && index < item_count; index += 2u) {
    attn_group_pairs_compatible =
        items[index].n_tokens != 0u &&
        items[index].n_tokens <= 6u &&
        items[index + 1u].n_tokens != 0u &&
        items[index + 1u].n_tokens <= 6u;
  }
  const bool combine_attn_front = attn_group_pairs_compatible;
  const bool exact_attn_output_rows =
      (total_rows >= 4u && total_rows <= 8u) ||
      total_rows == 10u || total_rows == 12u || total_rows == 16u ||
      total_rows == 18u || total_rows == 24u ||
      total_rows == 32u;
  const bool combine_attn_output = combine_attn_front && exact_attn_output_rows;
  bool ok = true;
  if (ok) {
    ok = ds4_gpu_tensor_write(
             coordinator->prefill_tokens, 0, flat_tokens.data(),
             (uint64_t)total_rows * sizeof(flat_tokens[0])) != 0;
  }
  if (ok) {
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

  for (uint32_t il = 0; ok && il < DS4_N_LAYER; ++il) {
    const ds4_layer_weights* layer = &engine->weights->layer[il];
    const uint64_t mix_hc =
        2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim =
        (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const uint32_t comp_width =
        (ratio == 4u ? 2u : 1u) * DS4_N_HEAD_DIM;
    const uint64_t indexer_q_dim =
        (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    const bool combine_attn_f16 = combine_attn_front && ratio != 0u;
    const bool combine_indexer_f16 =
        combine_attn_f16 && ratio == 4u;
    auto with_attention_slice =
        [&](ds4_gpu_graph* graph, uint32_t row0, uint32_t rows,
            auto&& operation) {
          // Use the coordinator's projected rows even when sessions borrow
          // different arenas. Views also confine compressor and indexer
          // scratch writes to each request's possibly ragged slice.
          std::array<ds4_gpu_tensor**, 17> slots{{
              &graph->batch_cur_hc,
              &graph->batch_flat_hc,
              &graph->batch_hc_mix,
              &graph->batch_hc_split,
              &graph->batch_attn_cur,
              &graph->batch_attn_norm,
              &graph->batch_qr,
              &graph->batch_qr_norm,
              &graph->batch_q,
              &graph->batch_kv_raw,
              &graph->batch_kv,
              &graph->batch_after_attn_hc,
              &graph->batch_indexer_q,
              &graph->batch_indexer_weights,
              &graph->batch_heads,
              &graph->batch_comp_kv,
              &graph->batch_comp_sc,
          }};
          std::array<ds4_gpu_tensor*, 17> bases{{
              coordinator->batch_cur_hc,
              coordinator->batch_flat_hc,
              coordinator->batch_hc_mix,
              coordinator->batch_hc_split,
              coordinator->batch_attn_cur,
              coordinator->batch_attn_norm,
              coordinator->batch_qr,
              coordinator->batch_qr_norm,
              coordinator->batch_q,
              coordinator->batch_kv_raw,
              coordinator->batch_kv,
              coordinator->batch_after_attn_hc,
              coordinator->batch_indexer_q,
              coordinator->batch_indexer_weights,
              coordinator->batch_heads,
              coordinator->batch_comp_kv,
              coordinator->batch_comp_sc,
          }};
          const std::array<uint64_t, 17> row_bytes{{
              hc_row_bytes,
              hc_row_bytes,
              mix_hc * sizeof(float),
              mix_hc * sizeof(float),
              (uint64_t)DS4_N_EMBD * sizeof(float),
              (uint64_t)DS4_N_EMBD * sizeof(float),
              q_rank * sizeof(float),
              q_rank * sizeof(float),
              q_dim * sizeof(float),
              (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
              (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
              hc_row_bytes,
              indexer_q_dim * sizeof(float),
              (uint64_t)DS4_N_INDEXER_HEAD * sizeof(float),
              q_dim * sizeof(float),
              (uint64_t)comp_width * sizeof(float),
              (uint64_t)comp_width * sizeof(float),
          }};
          std::array<ds4_gpu_tensor*, 17> saved{};
          std::array<ds4_gpu_tensor*, 17> views{};
          bool slice_ok = true;
          for (size_t field = 0; field < slots.size(); ++field) {
            saved[field] = *slots[field];
            views[field] = ds4_gpu_tensor_view(
                bases[field], (uint64_t)row0 * row_bytes[field],
                (uint64_t)rows * row_bytes[field]);
            slice_ok = slice_ok && views[field] != nullptr;
          }
          if (slice_ok) {
            for (size_t field = 0; field < slots.size(); ++field) {
              *slots[field] = views[field];
            }
            slice_ok = operation();
            for (size_t field = 0; field < slots.size(); ++field) {
              *slots[field] = saved[field];
            }
          }
          for (ds4_gpu_tensor* view : views) {
            ds4_gpu_tensor_free(view);
          }
          return slice_ok;
        };
    if (combine_attn_front) {
      for (size_t index = 0; ok && index < item_count; ++index) {
        ds4_gpu_graph* graph = items[index].graph;
        const uint32_t row0 = row_offsets[index];
        const uint32_t rows = items[index].n_tokens;
        ok = ds4_gpu_tensor_copy(
                 coordinator->batch_cur_hc,
                 (uint64_t)row0 * hc_row_bytes,
                 coordinator->batch_dspark_verify_cur_hc,
                 (uint64_t)row0 * hc_row_bytes,
                 (uint64_t)rows * hc_row_bytes) != 0;
        if (ok) {
          ok = with_attention_slice(graph, row0, rows, [&]() {
            return rocm_graph_encode_layer_attention_batch(
                graph, engine->model, layer, il, items[index].start, rows,
                false, true, !combine_attn_front, !combine_attn_front);
          });
        }
      }
      if (ok && combine_attn_front) {
        ok = ds4_gpu_matmul_q8_0_group_pairs_tensor(
                 coordinator->batch_qr, engine->model->map,
                 engine->model->size, layer->attn_q_a->abs_offset,
                 DS4_N_EMBD, q_rank, coordinator->batch_attn_norm,
                 total_rows, row_offsets.data(),
                 static_cast<uint32_t>(item_count)) != 0;
      }
      if (ok && combine_attn_front) {
        ok = ds4_gpu_matmul_q8_0_group_pairs_tensor(
                 coordinator->batch_kv_raw, engine->model->map,
                 engine->model->size, layer->attn_kv->abs_offset,
                 DS4_N_EMBD, DS4_N_HEAD_DIM,
                 coordinator->batch_attn_norm, total_rows,
                 row_offsets.data(), static_cast<uint32_t>(item_count)) != 0;
      }
      if (ok && combine_attn_front) {
        ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                 coordinator->batch_qr_norm, coordinator->batch_qr,
                 engine->model->map, engine->model->size,
                 layer->attn_q_a_norm->abs_offset,
                 static_cast<uint32_t>(q_rank), coordinator->batch_kv,
                 coordinator->batch_kv_raw,
                 layer->attn_kv_a_norm->abs_offset, DS4_N_HEAD_DIM,
                 total_rows, DS4_RMS_EPS) != 0;
      }

      if (ok && combine_attn_front) {
        ok = ds4_gpu_matmul_q8_0_group_pairs_tensor(
                 coordinator->batch_q, engine->model->map,
                 engine->model->size, layer->attn_q_b->abs_offset,
                 q_rank, q_dim, coordinator->batch_qr_norm, total_rows,
                 row_offsets.data(), static_cast<uint32_t>(item_count)) != 0;
      }

      if (ok && combine_attn_f16) {
        ok = ds4_gpu_matmul_f16_group_pairs_tensor(
                 coordinator->batch_comp_kv, engine->model->map,
                 engine->model->size,
                 layer->attn_compressor_kv->abs_offset, DS4_N_EMBD,
                 comp_width, coordinator->batch_attn_norm, total_rows,
                 row_offsets.data(),
                 static_cast<uint32_t>(item_count)) != 0;
      }
      if (ok && combine_attn_f16) {
        ok = ds4_gpu_matmul_f16_group_pairs_tensor(
                 coordinator->batch_comp_sc, engine->model->map,
                 engine->model->size,
                 layer->attn_compressor_gate->abs_offset, DS4_N_EMBD,
                 comp_width, coordinator->batch_attn_norm, total_rows,
                 row_offsets.data(),
                 static_cast<uint32_t>(item_count)) != 0;
      }
      if (ok && combine_indexer_f16) {
        ok = ds4_gpu_matmul_f16_group_pairs_tensor(
                 coordinator->batch_indexer_q, engine->model->map,
                 engine->model->size,
                 layer->indexer_attn_q_b->abs_offset, q_rank,
                 indexer_q_dim, coordinator->batch_qr_norm, total_rows,
                 row_offsets.data(),
                 static_cast<uint32_t>(item_count)) != 0;
      }
      if (ok && combine_indexer_f16) {
        ok = ds4_gpu_matmul_f16_group_pairs_tensor(
                 coordinator->batch_indexer_weights, engine->model->map,
                 engine->model->size, layer->indexer_proj->abs_offset,
                 DS4_N_EMBD, DS4_N_INDEXER_HEAD,
                 coordinator->batch_attn_norm, total_rows,
                 row_offsets.data(),
                 static_cast<uint32_t>(item_count)) != 0;
      }
    }
    for (size_t index = 0; ok && index < item_count; ++index) {
      ds4_gpu_graph* graph = items[index].graph;
      const uint64_t block_offset =
          (uint64_t)row_offsets[index] * hc_row_bytes;
      const uint64_t block_bytes =
          (uint64_t)items[index].n_tokens * hc_row_bytes;
      if (!combine_attn_front) {
        ok = ds4_gpu_tensor_copy(graph->batch_cur_hc, 0,
                                 coordinator->batch_dspark_verify_cur_hc,
                                 block_offset, block_bytes) != 0;
      }
      if (ok) {
        if (combine_attn_front) {
          ok = with_attention_slice(
              graph, row_offsets[index], items[index].n_tokens, [&]() {
                return rocm_graph_encode_layer_attention_batch(
                    graph, engine->model, layer, il, items[index].start,
                    items[index].n_tokens, true, false, true, true,
                    combine_attn_f16, combine_indexer_f16,
                    !combine_attn_output);
              });
        } else {
          ok = rocm_graph_encode_layer_attention_batch(
              graph, engine->model, layer, il, items[index].start,
              items[index].n_tokens);
        }
      }
      if (ok && !combine_attn_output) {
        ok = ds4_gpu_tensor_copy(
                 coordinator->batch_dspark_verify_after_attn_hc, block_offset,
                 combine_attn_front ? coordinator->batch_after_attn_hc
                                    : graph->batch_after_attn_hc,
                 combine_attn_front ? block_offset : 0, block_bytes) != 0;
      }
    }
    if (ok && combine_attn_output) {
      ok = ds4_gpu_attention_output_q8_exact_batch_tensor(
               coordinator->batch_attn_out, coordinator->batch_attn_low,
               engine->model->map, engine->model->size,
               layer->attn_output_a->abs_offset,
               layer->attn_output_b->abs_offset,
               DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP),
               DS4_N_LORA_O, DS4_N_OUT_GROUP, DS4_N_EMBD,
               coordinator->batch_heads, total_rows) != 0;
    }
    if (ok && combine_attn_output) {
      ds4_gpu_tensor* attn_out = ds4_gpu_tensor_view(
          coordinator->batch_attn_out, 0,
          (uint64_t)total_rows * DS4_N_EMBD * sizeof(float));
      ds4_gpu_tensor* cur_hc = ds4_gpu_tensor_view(
          coordinator->batch_cur_hc, 0,
          (uint64_t)total_rows * hc_row_bytes);
      ds4_gpu_tensor* hc_split = ds4_gpu_tensor_view(
          coordinator->batch_hc_split, 0,
          (uint64_t)total_rows * mix_hc * sizeof(float));
      ds4_gpu_tensor* after_attn_hc = ds4_gpu_tensor_view(
          coordinator->batch_dspark_verify_after_attn_hc, 0,
          (uint64_t)total_rows * hc_row_bytes);
      ok = attn_out && cur_hc && hc_split && after_attn_hc &&
           ds4_gpu_hc_expand_split_tensor(
               after_attn_hc, attn_out, cur_hc, hc_split, DS4_N_EMBD,
               DS4_N_HC) != 0;
      ds4_gpu_tensor_free(after_attn_hc);
      ds4_gpu_tensor_free(hc_split);
      ds4_gpu_tensor_free(cur_hc);
      ds4_gpu_tensor_free(attn_out);
    }
    if (ok) {
      ok = ds4_gpu_tensor_copy(coordinator->batch_after_attn_hc, 0,
                               coordinator->batch_dspark_verify_after_attn_hc,
                               0, (uint64_t)total_rows * hc_row_bytes) != 0;
    }
    if (ok) {
      ok = rocm_graph_encode_layer_ffn_batch(
          coordinator, engine->model, layer, il, items[0].start, total_rows,
          coordinator->prefill_tokens, row_offsets.data(),
          static_cast<uint32_t>(item_count));
    }
    for (size_t index = 0; ok && index < item_count; ++index) {
      const uint64_t block_offset =
          (uint64_t)row_offsets[index] * hc_row_bytes;
      ds4_gpu_tensor* block_hc =
          ds4_gpu_tensor_view(coordinator->batch_next_hc, block_offset,
                              (uint64_t)items[index].n_tokens * hc_row_bytes);
      ok = block_hc != NULL;
      if (ok) {
        ok = rocm_graph_dspark_capture_rows(items[index].graph, il, block_hc,
                                            items[index].n_tokens);
      }
      ds4_gpu_tensor_free(block_hc);
    }
    if (ok) {
      ok = ds4_gpu_tensor_copy(coordinator->batch_dspark_verify_cur_hc, 0,
                               coordinator->batch_next_hc, 0,
                               (uint64_t)total_rows * hc_row_bytes) != 0;
    }
  }
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
  if (!ok)
    return false;

  if (ds4_gpu_tensor_read(coordinator->spec_row_tops, 0, flat_tops.data(),
                          (uint64_t)total_rows * sizeof(flat_tops[0])) == 0) {
    return false;
  }
  {
    return rocm_graph_distribute_verify_results(
        coordinator, items, item_count, row_offsets, flat_tops,
        engine->weights->output->dim[1]);
  }
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

    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(kv_raw);
    ds4_gpu_tensor_free(fused);
    /*
     * Context coverage is contiguous from position zero. A later injection
     * cannot fill a gap left while support capture was disabled.
     */
    if (ok && pos0 <= g->dspark_context_len &&
        pos0 + n_rows > g->dspark_context_len) {
      g->dspark_context_len = pos0 + n_rows;
    }
    return ok;
}

/*
 * One DSpark stage over the draft block: the DS4 block with the causal window
 * replaced by non-causal attention over the injected ring plus the block
 * itself. The FFN half is shared with the target's batched path, which is safe
 * because that half is driven entirely by the supplied weights.
 */
static bool rocm_graph_encode_dspark_stage_attention(
    ds4_gpu_graph* g, uint32_t stage, uint32_t pos0, uint32_t n_rows,
    bool front_ready = false, bool front_only = false, bool project_qb = true,
    bool project_q_a_kv = true) {
  const ds4_dspark_model* d = g->dspark;
  if (!d || stage >= d->n_stages || n_rows == 0u)
    return false;
  const uint32_t draft_rows = n_rows;
  const auto span =
      ds4_dspark_visible_span(pos0, draft_rows, g->dspark_cache_cap);
  if (span.count == 0u)
    return false;
  const ds4_model* sm = d->model;
  const ds4_layer_weights* layer = &d->stage[stage].block;
  const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
  const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
  const uint64_t q_rank = layer->attn_q_a->dim[1];
  const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
  const uint32_t group_heads = DS4_N_HEAD / DS4_N_OUT_GROUP;
  const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;

  ds4_gpu_tensor* hc_mix_view = ds4_gpu_tensor_view(
      g->batch_hc_mix, 0, (uint64_t)n_rows * mix_hc * sizeof(float));
  ds4_gpu_tensor* hc_split_view = ds4_gpu_tensor_view(
      g->batch_hc_split, 0, (uint64_t)n_rows * mix_hc * sizeof(float));
  ds4_gpu_tensor* attn_cur_view = ds4_gpu_tensor_view(
      g->batch_attn_cur, 0, (uint64_t)n_rows * DS4_N_EMBD * sizeof(float));
  ds4_gpu_tensor* after_attn_hc_view = ds4_gpu_tensor_view(
      g->batch_after_attn_hc, 0, (uint64_t)draft_rows * hc_dim * sizeof(float));
  bool ok = hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;

  if (ok && !front_ready) {
    ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc, g->batch_cur_hc,
                                            (uint32_t)hc_dim, n_rows,
                                            DS4_RMS_EPS) != 0;
  }
  if (ok && !front_ready) {
    ok = ds4_gpu_matmul_f16_tensor(hc_mix_view, sm->map, sm->size,
                                   layer->hc_attn_fn->abs_offset, hc_dim,
                                   mix_hc, g->batch_flat_hc, n_rows) != 0;
  }
  if (ok && !front_ready) {
    ok = ds4_gpu_hc_split_weighted_sum_tensor(
             attn_cur_view, hc_split_view, hc_mix_view, g->batch_cur_hc,
             sm->map, sm->size, layer->hc_attn_scale->abs_offset,
             layer->hc_attn_base->abs_offset, DS4_N_EMBD, DS4_N_HC,
             DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS) != 0;
  }
  if (ok && !front_ready) {
    ok =
        ds4_gpu_rms_norm_weight_rows_tensor(
            g->batch_attn_norm, g->batch_attn_cur, sm->map, sm->size,
            layer->attn_norm->abs_offset, DS4_N_EMBD, n_rows, DS4_RMS_EPS) != 0;
  }

  if (ok && project_q_a_kv) {
    ok = ds4_gpu_matmul_q8_0_tensor(
             g->batch_qr, sm->map, sm->size, layer->attn_q_a->abs_offset,
             DS4_N_EMBD, q_rank, g->batch_attn_norm, draft_rows) != 0;
  }
  if (ok && project_q_a_kv) {
    ok = ds4_gpu_rms_norm_weight_rows_tensor(
             g->batch_qr_norm, g->batch_qr, sm->map, sm->size,
             layer->attn_q_a_norm->abs_offset, (uint32_t)q_rank, draft_rows,
             DS4_RMS_EPS) != 0;
  }
  if (ok && project_qb) {
    ok = ds4_gpu_matmul_q8_0_tensor(g->batch_q, sm->map, sm->size,
                                    layer->attn_q_b->abs_offset, q_rank, q_dim,
                                    g->batch_qr_norm, draft_rows) != 0;
  }
  if (front_only) {
    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    return ok;
  }
  if (ok) {
    ok = ds4_gpu_head_rms_norm_tensor(g->batch_q, draft_rows, DS4_N_HEAD,
                                      DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
  }
  if (ok) {
    ok = ds4_gpu_rope_tail_tensor(
             g->batch_q, draft_rows, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT,
             pos0, 0, false, DS4_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
             DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
  }

  if (ok && project_q_a_kv) {
    ok = ds4_gpu_matmul_q8_0_tensor(
             g->batch_kv_raw, sm->map, sm->size, layer->attn_kv->abs_offset,
             DS4_N_EMBD, DS4_N_HEAD_DIM, g->batch_attn_norm, n_rows) != 0;
  }
  if (ok && project_q_a_kv) {
    ok = ds4_gpu_rms_norm_weight_rows_tensor(
             g->batch_kv, g->batch_kv_raw, sm->map, sm->size,
             layer->attn_kv_a_norm->abs_offset, DS4_N_HEAD_DIM, n_rows,
             DS4_RMS_EPS) != 0;
  }
  if (ok) {
    ok = ds4_gpu_rope_tail_tensor(
             g->batch_kv, draft_rows, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT,
             pos0, 0, false, DS4_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
             DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
  }
  if (ok) {
    ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->batch_kv, n_rows,
                                             DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
  }
  if (ok) {
    ok = ds4_gpu_store_raw_kv_batch_tensor(g->dspark_kv_cache[stage],
                                           g->batch_kv, g->dspark_cache_cap,
                                           pos0, n_rows, DS4_N_HEAD_DIM) != 0;
  }

  if (ok) {
    ok = ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
             g->batch_heads, sm->map, sm->size, layer->attn_sinks->abs_offset,
             g->batch_q, g->dspark_kv_cache[stage], draft_rows, span.count,
             g->dspark_cache_cap, span.start, DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
  }
  if (ok) {
    ok = ds4_gpu_rope_tail_tensor(
             g->batch_heads, draft_rows, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT,
             pos0, 0, true, DS4_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
             DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
  }
  if (ok) {
    ok = ds4_gpu_attention_output_q8_batch_tensor(
             g->batch_attn_out, g->batch_attn_low, g->batch_group_tmp,
             g->batch_low_tmp, sm->map, sm->size,
             layer->attn_output_a->abs_offset, layer->attn_output_b->abs_offset,
             group_dim, DS4_N_LORA_O, DS4_N_OUT_GROUP, DS4_N_EMBD,
             g->batch_heads, draft_rows) != 0;
  }
  if (ok) {
    ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view, g->batch_attn_out,
                                        g->batch_cur_hc, hc_split_view,
                                        DS4_N_EMBD, DS4_N_HC) != 0;
  }

  ds4_gpu_tensor_free(after_attn_hc_view);
  ds4_gpu_tensor_free(attn_cur_view);
  ds4_gpu_tensor_free(hc_split_view);
  ds4_gpu_tensor_free(hc_mix_view);
  return ok;
}

static bool rocm_graph_encode_dspark_stage(ds4_gpu_graph* g, uint32_t stage,
                                           uint32_t pos0, uint32_t n_rows) {
  if (!rocm_graph_encode_dspark_stage_attention(g, stage, pos0, n_rows)) {
    return false;
  }
  const ds4_dspark_model* d = g->dspark;
  const uint32_t draft_rows = n_rows;
  const ds4_model* sm = d->model;
  const ds4_layer_weights* layer = &d->stage[stage].block;
  const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
  if (!rocm_graph_encode_layer_ffn_batch(g, sm, layer, stage, pos0,
                                         draft_rows)) {
    return false;
  }

  return ds4_gpu_tensor_copy(g->batch_cur_hc, 0, g->batch_next_hc, 0,
                             (uint64_t)draft_rows * hc_dim * sizeof(float)) !=
         0;
}

/* Final support hidden rows, consumed by the tied target LM head. */
static bool rocm_graph_encode_dspark_final_hidden(ds4_gpu_graph* g,
                                                  uint32_t n_rows,
                                                  ds4_gpu_tensor* hidden_out,
                                                  uint32_t hidden_row_offset) {
  const ds4_dspark_model* d = g->dspark;
  if (!d || !hidden_out || n_rows == 0 || n_rows > g->spec_rows_cap) {
    return false;
  }
  const ds4_model* sm = d->model;
  const ds4_dspark_stage_weights* last = &d->stage[d->n_stages - 1u];
  const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;

  /* Only the draft block feeds the final head. */
  ds4_gpu_tensor* rows_hc = ds4_gpu_tensor_view(
      g->batch_cur_hc, 0, (uint64_t)n_rows * hc_dim * sizeof(float));
  if (!rows_hc)
    return false;

  ds4_gpu_tensor* head_pre = ds4_gpu_tensor_view(
      g->batch_hc_mix, 0, (uint64_t)n_rows * DS4_N_HC * sizeof(float));
  ds4_gpu_tensor* head_weights = ds4_gpu_tensor_view(
      g->batch_hc_split, 0, (uint64_t)n_rows * DS4_N_HC * sizeof(float));
  ds4_gpu_tensor* head_embd = ds4_gpu_tensor_view(
      g->batch_ffn_cur, 0, (uint64_t)n_rows * DS4_N_EMBD * sizeof(float));
  ds4_gpu_tensor* head_norm = ds4_gpu_tensor_view(
      hidden_out, (uint64_t)hidden_row_offset * DS4_N_EMBD * sizeof(float),
      (uint64_t)n_rows * DS4_N_EMBD * sizeof(float));
  bool ok = head_pre && head_weights && head_embd && head_norm;

  if (ok)
    ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc, rows_hc,
                                            (uint32_t)hc_dim, n_rows,
                                            DS4_RMS_EPS) != 0;
  if (ok)
    ok = ds4_gpu_matmul_f16_tensor(head_pre, sm->map, sm->size,
                                   last->hc_head_fn->abs_offset, hc_dim,
                                   DS4_N_HC, g->batch_flat_hc, n_rows) != 0;
  if (ok)
    ok = ds4_gpu_output_hc_weights_tensor(
             head_weights, head_pre, sm->map, sm->size,
             last->hc_head_scale->abs_offset, last->hc_head_base->abs_offset,
             DS4_N_HC, DS4_HC_EPS) != 0;
  if (ok)
    ok = ds4_gpu_hc_weighted_sum_tensor(head_embd, rows_hc, head_weights,
                                        DS4_N_EMBD, DS4_N_HC) != 0;
  if (ok)
    ok = ds4_gpu_rms_norm_weight_rows_tensor(
             head_norm, head_embd, sm->map, sm->size, last->norm->abs_offset,
             DS4_N_EMBD, n_rows, DS4_RMS_EPS) != 0;

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
  if (!g || !target_model || !target_weights || !hidden_rows || n_rows == 0u ||
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

/*
 * Trace the block's trajectory with the Markov path selector.
 *
 * Selecting each position by its own argmax lets the block drift: the best
 * token at position t given the block's shared context is not the best token
 * given what position t-1 actually emitted. The rank-256 bilinear term restores
 * that coupling. Positions resolve in order because each conditions the next.
 * Greedy reads back the chosen index; sampled cohorts read a compact shortlist
 * and confidence for each request.
 */
static bool rocm_graph_dspark_select_sampled(
    ds4_gpu_graph* g, const ds4_rocm_dspark_draft_item* items, size_t count,
    uint32_t rows_per_item) {
  constexpr uint32_t blocks = (DS4_N_VOCAB + 255u) / 256u;
  if (!g->dspark_candidates)
    g->dspark_candidates =
        ds4_gpu_tensor_alloc(8u * sizeof(ds4_dspark_candidates));
  if (!g->dspark_candidate_scratch)
    g->dspark_candidate_scratch = ds4_gpu_tensor_alloc(
        8u * blocks * DS4_DSPARK_CANDIDATES * sizeof(uint64_t));
  if (!g->dspark_candidates || !g->dspark_candidate_scratch)
    return false;
  const auto* d = g->dspark;
  const auto& last = d->stage[d->n_stages - 1];
  std::array<int32_t, 8> previous{};
  std::array<ds4_dspark_candidates, 8> candidates{};
  std::array<ds4_dspark_confidence_policy, 8> policies{};
  std::array<bool, 8> stopped{};
  uint32_t max_rows = 0;
  for (size_t i = 0; i < count; ++i) {
    previous[i] = items[i].target_next_token;
    if (previous[i] < 0 || (uint32_t)previous[i] >= DS4_N_VOCAB)
      return false;
    policies[i] = {
        .depth = items[i].position,
        .concurrency = items[i].concurrency,
        .maximum = std::min(items[i].max_draft_tokens, rows_per_item)};
    *items[i].n_tokens = 0;
    max_rows =
        std::max(max_rows, std::min(items[i].max_draft_tokens, rows_per_item));
  }
  for (uint32_t row = 0; row < max_rows; ++row) {
    if (!ds4_gpu_tensor_write(g->prefill_tokens, 0, previous.data(),
                              count * sizeof(previous[0])))
      return false;
    const uint64_t span_rows = (count - 1) * rows_per_item + 1;
    auto* logits = ds4_gpu_tensor_view(
        g->spec_logits, (uint64_t)row * DS4_N_VOCAB * sizeof(float),
        span_rows * DS4_N_VOCAB * sizeof(float));
    // DS4 confidence uses hc_head(x), before norm; this buffer survives the
    // tied LM-head projection and holds independent rows for every request.
    auto* hidden = ds4_gpu_tensor_view(
        g->batch_ffn_cur, (uint64_t)row * DS4_N_EMBD * sizeof(float),
        span_rows * DS4_N_EMBD * sizeof(float));
    const bool ok =
        logits && hidden &&
        ds4_gpu_dspark_candidates_tensor(
            g->dspark_candidates, g->dspark_candidate_scratch, logits,
            (uint64_t)rows_per_item * DS4_N_VOCAB, hidden,
            (uint64_t)rows_per_item * DS4_N_EMBD, g->prefill_tokens,
            d->model->map, d->model->size, last.markov_w1->abs_offset,
            last.markov_w2->abs_offset, last.confidence->abs_offset,
            DS4_N_VOCAB, d->markov_rank, DS4_N_EMBD, count);
    ds4_gpu_tensor_free(hidden);
    ds4_gpu_tensor_free(logits);
    if (!ok || !ds4_gpu_tensor_read(g->dspark_candidates, 0, candidates.data(),
                                    count * sizeof(candidates[0])))
      return false;
    for (size_t i = 0; i < count; ++i) {
      if (stopped[i] || row >= items[i].max_draft_tokens)
        continue;
      const auto* sampler = items[i].sampler;
      if (sampler && sampler->propose &&
          !policies[i].Include(candidates[i].confidence)) {
        stopped[i] = true;
        continue;
      }
      const int token =
          sampler && sampler->propose
              ? sampler->propose(sampler->ctx, row, &candidates[i])
              : candidates[i].ids[0];
      if (token < 0 || (uint32_t)token >= DS4_N_VOCAB)
        return false;
      items[i].tokens[row] = token;
      previous[i] = token;
      *items[i].n_tokens = row + 1;
    }
    bool more = false;
    for (size_t i = 0; i < count; ++i)
      more |= !stopped[i] && row + 1 < items[i].max_draft_tokens;
    if (!more)
      break;
  }
  return true;
}

static bool rocm_graph_dspark_select_block(ds4_gpu_graph* g,
                                           const ds4_gpu_tensor* logits_rows,
                                           int target_next_token,
                                           uint32_t n_rows,
                                           int32_t* tokens_out) {
  const ds4_dspark_model* d = g->dspark;
  if (!d || !logits_rows || n_rows == 0 || !tokens_out) {
    return false;
  }
  if (!g->dspark_markov_index || !g->dspark_markov_key)
    return false;
  const ds4_model* sm = d->model;
  const ds4_dspark_stage_weights* last = &d->stage[d->n_stages - 1u];
  int previous = target_next_token;
  for (uint32_t row = 0; row < n_rows; row++) {
    if (previous < 0 || (uint32_t)previous >= DS4_N_VOCAB)
      return false;

    ds4_gpu_tensor* logits_row = ds4_gpu_tensor_view(
        logits_rows, (uint64_t)row * DS4_N_VOCAB * sizeof(float),
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
  }
  return true;
}

static bool rocm_graph_dspark_select_block_batch(
    ds4_gpu_graph* coordinator, const ds4_rocm_dspark_draft_item* items,
    size_t item_count, const ds4_gpu_tensor* logits_rows,
    uint32_t rows_per_item) {
  if (!coordinator || !coordinator->dspark || !items || !logits_rows ||
      rows_per_item == 0u ||
      (item_count != 2u && item_count != 4u && item_count != 6u &&
       item_count != 8u) ||
      !coordinator->dspark_markov_index || !coordinator->dspark_markov_key) {
    return false;
  }
  const ds4_dspark_model* d = coordinator->dspark;
  const ds4_model* sm = d->model;
  const ds4_dspark_stage_weights* last =
      &d->stage[d->n_stages - 1u];
  if (!last->markov_w1 || !last->markov_w2) {
    return false;
  }

  std::array<int32_t, 8> previous{};
  std::array<int32_t, 8> selected{};
  std::array<bool, 8> active{};
  std::array<uint32_t, 8> row_limits{};
  std::array<uint32_t, 8> selected_rows{};
  for (size_t index = 0; index < item_count; ++index) {
    if (!items[index].tokens || !items[index].n_tokens ||
        items[index].max_draft_tokens == 0u ||
        items[index].target_next_token < 0 ||
        (uint32_t)items[index].target_next_token >= DS4_N_VOCAB) {
      return false;
    }
    previous[index] = items[index].target_next_token;
    row_limits[index] =
        std::min(items[index].max_draft_tokens, rows_per_item);
    active[index] = true;
  }

  const uint64_t vocab_row_bytes =
      (uint64_t)DS4_N_VOCAB * sizeof(float);
  const uint64_t logits_row_stride =
      (uint64_t)rows_per_item * DS4_N_VOCAB;
  for (uint32_t row = 0u; row < rows_per_item; ++row) {
    if (ds4_gpu_tensor_write(
            coordinator->prefill_tokens, 0u, previous.data(),
            (uint64_t)item_count * sizeof(previous[0])) == 0) {
      return false;
    }

    const uint64_t logits_bytes =
        ((uint64_t)(item_count - 1u) * rows_per_item + 1u) *
        vocab_row_bytes;
    ds4_gpu_tensor* logits_view = ds4_gpu_tensor_view(
        logits_rows, (uint64_t)row * vocab_row_bytes, logits_bytes);
    if (!logits_view) {
      return false;
    }
    const bool ok =
        ds4_gpu_dspark_markov_argmax_batch_tensor(
            coordinator->dspark_markov_index,
            coordinator->dspark_markov_key, logits_view,
            coordinator->prefill_tokens, sm->map, sm->size,
            last->markov_w1->abs_offset, last->markov_w2->abs_offset,
            DS4_N_VOCAB, d->markov_rank,
            static_cast<uint32_t>(item_count), logits_row_stride) != 0;
    ds4_gpu_tensor_free(logits_view);
    if (!ok || ds4_gpu_tensor_read(
                   coordinator->dspark_markov_index, 0u, selected.data(),
                   (uint64_t)item_count * sizeof(selected[0])) == 0) {
      return false;
    }

    for (size_t index = 0; index < item_count; ++index) {
      if (selected[index] < 0 || (uint32_t)selected[index] >= DS4_N_VOCAB) {
        return false;
      }
      if (!active[index] || row >= row_limits[index]) {
        active[index] = false;
        continue;
      }

      items[index].tokens[row] = selected[index];
      previous[index] = selected[index];
      selected_rows[index] = row + 1u;
    }
  }
  for (size_t index = 0; index < item_count; ++index) {
    *items[index].n_tokens = selected_rows[index];
  }
  return true;
}

/*
 * Propose the tokens following the target-selected token at position pos0.
 *
 * `target_next_token` is the target's selected next token; it seeds both the
 * block's first embedding slot and the Markov chain. Requires that the target
 * has already captured its sampled hidden states for position pos0 - 1 and that
 * those have been injected into the rings.
 */
static bool rocm_graph_dspark_prepare_hidden(
    ds4_gpu_graph* g, const ds4_model* target_model,
    const ds4_weights* target_weights, int target_next_token, uint32_t pos0,
    ds4_gpu_tensor* hidden_out, uint32_t hidden_row_offset) {
  const ds4_dspark_model* d = g->dspark;
  if (!d)
    return false;
  const uint32_t block = d->block_size;
  const uint32_t rows = block;
  if (block == 0 || block > g->spec_rows_cap || rows > g->prefill_cap)
    return false;
  if (pos0 == 0 || g->dspark_context_len < pos0)
    return false;
  if (rows > g->dspark_cache_cap)
    return false;

  // The target's selected next token seeds draft row zero and the Markov head.
  int32_t block_tokens[DS4_DSPARK_MAX_BLOCK];
  block_tokens[0] = target_next_token;
  for (uint32_t i = 1; i < rows; i++) {
    block_tokens[i] = (int32_t)d->noise_token_id;
  }
  if (ds4_gpu_tensor_write(g->prefill_tokens, 0, block_tokens,
                           (uint64_t)block * sizeof(block_tokens[0])) == 0) {
    return false;
  }

  bool ok = ds4_gpu_begin_commands() != 0;
  /* Tied target embeddings, expanded across the hyper-connection streams. */
  if (ok)
    ok = ds4_gpu_embed_tokens_hc_tensor(
             g->batch_cur_hc, g->prefill_tokens, target_model->map,
             target_model->size, target_weights->token_embd->abs_offset,
             DS4_N_VOCAB, rows, DS4_N_EMBD, DS4_N_HC) != 0;
  for (uint32_t stage = 0; ok && stage < d->n_stages; stage++) {
    ok = rocm_graph_encode_dspark_stage(g, stage, pos0, rows);
  }
  if (ok) {
    ok = rocm_graph_encode_dspark_final_hidden(g, block, hidden_out,
                                               hidden_row_offset);
  }
  if (ok) {
    ok = ds4_gpu_end_commands() != 0;
  } else {
    (void)ds4_gpu_synchronize();
  }
  return ok;
}

/*
 * Run one support-model block for every active request.
 *
 * All sessions intentionally borrow one large batch scratch arena. Keep each
 * request's persistent draft rows in the dedicated flattened buffers and stage
 * only the request currently running its stateful attention into shared
 * scratch. The support FFN/MoE is stateless, so it can consume all flattened
 * rows in one pass while retaining one router group per request.
 */
static bool rocm_graph_dspark_prepare_hidden_batch(
    ds4_engine* engine, const ds4_rocm_dspark_draft_item* items,
    size_t item_count, uint32_t block, ds4_gpu_tensor* hidden_out) {
  if (!engine || !engine->model || !engine->weights || !items ||
      item_count < 2u || item_count > 8u || !hidden_out) {
    return false;
  }
  ds4_gpu_graph* coordinator = items[0].graph;
  const ds4_dspark_model* d = coordinator ? coordinator->dspark : nullptr;
  if (!coordinator || !d || !coordinator->batch_dspark_verify_cur_hc ||
      !coordinator->batch_dspark_verify_after_attn_hc) {
    return false;
  }
  const uint32_t rows = block;
  const uint32_t total_draft_rows = static_cast<uint32_t>(item_count) * block;
  if (block == 0u || block > d->block_size ||
      total_draft_rows > coordinator->prefill_cap ||
      total_draft_rows > coordinator->batch_dspark_verify_rows_cap ||
      total_draft_rows > DS4_SPEC_MAX_ROWS) {
    return false;
  }

  std::vector<int32_t> flat_tokens(total_draft_rows);
  for (size_t index = 0; index < item_count; ++index) {
    const ds4_rocm_dspark_draft_item& item = items[index];
    ds4_gpu_graph* graph = item.graph;
    if (!graph || graph->dspark != d || item.position == 0u ||
        graph->dspark_context_len < item.position ||
        rows > graph->prefill_cap || rows > graph->dspark_cache_cap) {
      return false;
    }
    const uint32_t row0 = static_cast<uint32_t>(index) * block;
    flat_tokens[row0] = item.target_next_token;
    for (uint32_t row = 1u; row < block; ++row) {
      flat_tokens[row0 + row] =
          static_cast<int32_t>(d->noise_token_id);
    }
  }
  if (ds4_gpu_tensor_write(
          coordinator->prefill_tokens, 0, flat_tokens.data(),
          static_cast<uint64_t>(total_draft_rows) *
              sizeof(flat_tokens[0])) == 0) {
    return false;
  }

  const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
  const uint64_t hc_row_bytes = hc_dim * sizeof(float);
  const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
  const bool combine_attn_q8 =
      item_count >= 4u && (item_count & 1u) == 0u && rows <= 6u;
  std::array<uint32_t, 9> front_offsets{};
  for (size_t index = 0; index <= item_count; ++index) {
    front_offsets[index] = static_cast<uint32_t>(index) * rows;
  }

  auto with_front_slice =
      [&](ds4_gpu_graph* graph, uint32_t row0, auto&& operation) {
        std::array<ds4_gpu_tensor**, 6> slots{{
            &graph->batch_cur_hc,
            &graph->batch_flat_hc,
            &graph->batch_hc_mix,
            &graph->batch_hc_split,
            &graph->batch_attn_cur,
            &graph->batch_attn_norm,
        }};
        std::array<ds4_gpu_tensor*, 6> bases{{
            coordinator->batch_cur_hc,
            coordinator->batch_flat_hc,
            coordinator->batch_hc_mix,
            coordinator->batch_hc_split,
            coordinator->batch_attn_cur,
            coordinator->batch_attn_norm,
        }};
        const std::array<uint64_t, 6> row_bytes{{
            hc_row_bytes,
            hc_row_bytes,
            mix_hc * sizeof(float),
            mix_hc * sizeof(float),
            (uint64_t)DS4_N_EMBD * sizeof(float),
            (uint64_t)DS4_N_EMBD * sizeof(float),
        }};
        std::array<ds4_gpu_tensor*, 6> saved{};
        std::array<ds4_gpu_tensor*, 6> views{};
        bool slice_ok = true;
        for (size_t field = 0; field < slots.size(); ++field) {
          saved[field] = *slots[field];
          views[field] = ds4_gpu_tensor_view(
              bases[field], (uint64_t)row0 * row_bytes[field],
              (uint64_t)rows * row_bytes[field]);
          slice_ok = slice_ok && views[field] != nullptr;
        }
        if (slice_ok) {
          for (size_t field = 0; field < slots.size(); ++field) {
            *slots[field] = views[field];
          }
          slice_ok = operation();
          for (size_t field = 0; field < slots.size(); ++field) {
            *slots[field] = saved[field];
          }
        }
        for (ds4_gpu_tensor* view : views) {
          ds4_gpu_tensor_free(view);
        }
        return slice_ok;
      };

  auto with_projected_slice =
      [&](ds4_gpu_graph* graph, uint32_t front_row0, auto&& operation) {
        ds4_gpu_tensor* saved_cur_hc = graph->batch_cur_hc;
        ds4_gpu_tensor* saved_hc_split = graph->batch_hc_split;
        ds4_gpu_tensor* saved_attn_norm = graph->batch_attn_norm;
        ds4_gpu_tensor* saved_q = graph->batch_q;
        ds4_gpu_tensor* saved_kv = graph->batch_kv;
        ds4_gpu_tensor* cur_hc = ds4_gpu_tensor_view(
            coordinator->batch_cur_hc,
            (uint64_t)front_row0 * hc_row_bytes,
            (uint64_t)rows * hc_row_bytes);
        ds4_gpu_tensor* hc_split = ds4_gpu_tensor_view(
            coordinator->batch_hc_split,
            (uint64_t)front_row0 * mix_hc * sizeof(float),
            (uint64_t)rows * mix_hc * sizeof(float));
        ds4_gpu_tensor* attn_norm = ds4_gpu_tensor_view(
            coordinator->batch_attn_norm,
            (uint64_t)front_row0 * DS4_N_EMBD * sizeof(float),
            (uint64_t)rows * DS4_N_EMBD * sizeof(float));
        ds4_gpu_tensor* q = ds4_gpu_tensor_view(
            coordinator->batch_q,
            (uint64_t)front_row0 * DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float),
            (uint64_t)block * DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(float));
        ds4_gpu_tensor* kv = ds4_gpu_tensor_view(
            coordinator->batch_kv,
            (uint64_t)front_row0 * DS4_N_HEAD_DIM * sizeof(float),
            (uint64_t)rows * DS4_N_HEAD_DIM * sizeof(float));
        bool slice_ok =
            cur_hc && hc_split && attn_norm && q && kv;
        if (slice_ok) {
          graph->batch_cur_hc = cur_hc;
          graph->batch_hc_split = hc_split;
          graph->batch_attn_norm = attn_norm;
          graph->batch_q = q;
          graph->batch_kv = kv;
          slice_ok = operation();
          graph->batch_cur_hc = saved_cur_hc;
          graph->batch_hc_split = saved_hc_split;
          graph->batch_attn_norm = saved_attn_norm;
          graph->batch_q = saved_q;
          graph->batch_kv = saved_kv;
        }
        ds4_gpu_tensor_free(kv);
        ds4_gpu_tensor_free(q);
        ds4_gpu_tensor_free(attn_norm);
        ds4_gpu_tensor_free(hc_split);
        ds4_gpu_tensor_free(cur_hc);
        return slice_ok;
      };

  bool ok = ds4_gpu_begin_commands() != 0;
  if (ok) {
    ok = ds4_gpu_embed_tokens_hc_tensor(
             coordinator->batch_dspark_verify_cur_hc,
             coordinator->prefill_tokens, engine->model->map,
             engine->model->size,
             engine->weights->token_embd->abs_offset, DS4_N_VOCAB,
             total_draft_rows, DS4_N_EMBD, DS4_N_HC) != 0;
  }

  for (uint32_t stage = 0; ok && stage < d->n_stages; ++stage) {
    const ds4_layer_weights* layer = &d->stage[stage].block;
    if (combine_attn_q8) {
      const uint64_t q_rank = layer->attn_q_a->dim[1];
      const uint64_t q_dim =
          (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
      for (size_t index = 0; ok && index < item_count; ++index) {
        ds4_gpu_graph* graph = items[index].graph;
        const uint32_t front_row0 = front_offsets[index];
        const uint64_t block_offset =
            static_cast<uint64_t>(index) * block * hc_row_bytes;
        ok = ds4_gpu_tensor_copy(
                 coordinator->batch_cur_hc, (uint64_t)front_row0 * hc_row_bytes,
                 coordinator->batch_dspark_verify_cur_hc, block_offset,
                 static_cast<uint64_t>(block) * hc_row_bytes) != 0;
        if (ok) {
          ok = with_front_slice(graph, front_row0, [&]() {
            return rocm_graph_encode_dspark_stage_attention(
                graph, stage, items[index].position, rows, false, true,
                false, false);
          });
        }
      }
      if (ok) {
        ok = ds4_gpu_matmul_q8_0_group_pairs_tensor(
                 coordinator->batch_qr, d->model->map, d->model->size,
                 layer->attn_q_a->abs_offset, DS4_N_EMBD, q_rank,
                 coordinator->batch_attn_norm, total_draft_rows,
                 front_offsets.data(), static_cast<uint32_t>(item_count)) != 0;
      }
      if (ok) {
        ok = ds4_gpu_matmul_q8_0_group_pairs_tensor(
                 coordinator->batch_kv_raw, d->model->map, d->model->size,
                 layer->attn_kv->abs_offset, DS4_N_EMBD, DS4_N_HEAD_DIM,
                 coordinator->batch_attn_norm, total_draft_rows,
                 front_offsets.data(), static_cast<uint32_t>(item_count)) != 0;
      }
      if (ok) {
        ok =
            ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                coordinator->batch_qr_norm, coordinator->batch_qr,
                d->model->map, d->model->size, layer->attn_q_a_norm->abs_offset,
                static_cast<uint32_t>(q_rank), coordinator->batch_kv,
                coordinator->batch_kv_raw, layer->attn_kv_a_norm->abs_offset,
                DS4_N_HEAD_DIM, total_draft_rows, DS4_RMS_EPS) != 0;
      }
      if (ok) {
        ok = ds4_gpu_matmul_q8_0_group_pairs_tensor(
                 coordinator->batch_q, d->model->map, d->model->size,
                 layer->attn_q_b->abs_offset, q_rank, q_dim,
                 coordinator->batch_qr_norm, total_draft_rows,
                 front_offsets.data(), static_cast<uint32_t>(item_count)) != 0;
      }
      for (size_t index = 0; ok && index < item_count; ++index) {
        ds4_gpu_graph* graph = items[index].graph;
        const uint32_t front_row0 = front_offsets[index];
        const uint64_t block_offset =
            static_cast<uint64_t>(index) * block * hc_row_bytes;
        ok = with_projected_slice(graph, front_row0, [&]() {
          return rocm_graph_encode_dspark_stage_attention(
              graph, stage, items[index].position, rows, true, false,
              false, false);
        });
        if (ok) {
          ok = ds4_gpu_tensor_copy(
                   coordinator->batch_dspark_verify_after_attn_hc,
                   block_offset, graph->batch_after_attn_hc, 0,
                   static_cast<uint64_t>(block) * hc_row_bytes) != 0;
        }
      }
    } else {
      for (size_t index = 0; ok && index < item_count; ++index) {
        ds4_gpu_graph* graph = items[index].graph;
        const uint64_t block_offset =
            static_cast<uint64_t>(index) * block * hc_row_bytes;
        ok =
            ds4_gpu_tensor_copy(
                graph->batch_cur_hc, 0, coordinator->batch_dspark_verify_cur_hc,
                block_offset, static_cast<uint64_t>(block) * hc_row_bytes) != 0;
        if (ok) {
          ok = rocm_graph_encode_dspark_stage_attention(
              graph, stage, items[index].position, rows);
        }
        if (ok) {
          ok = ds4_gpu_tensor_copy(
                   coordinator->batch_dspark_verify_after_attn_hc,
                   block_offset, graph->batch_after_attn_hc, 0,
                   static_cast<uint64_t>(block) * hc_row_bytes) != 0;
        }
      }
    }
    if (ok) {
      ok = ds4_gpu_tensor_copy(
               coordinator->batch_after_attn_hc, 0,
               coordinator->batch_dspark_verify_after_attn_hc, 0,
               static_cast<uint64_t>(total_draft_rows) * hc_row_bytes) != 0;
    }
    if (ok) {
      ok = rocm_graph_encode_layer_ffn_batch(
          coordinator, d->model, layer, stage, items[0].position,
          total_draft_rows, coordinator->prefill_tokens);
    }
    if (ok) {
      ok = ds4_gpu_tensor_copy(
               coordinator->batch_dspark_verify_cur_hc, 0,
               coordinator->batch_next_hc, 0,
               static_cast<uint64_t>(total_draft_rows) * hc_row_bytes) != 0;
    }
  }

  if (ok) {
    ok = ds4_gpu_tensor_copy(
             coordinator->batch_cur_hc, 0,
             coordinator->batch_dspark_verify_cur_hc, 0,
             static_cast<uint64_t>(total_draft_rows) * hc_row_bytes) != 0;
  }
  if (ok) {
    ok = rocm_graph_encode_dspark_final_hidden(coordinator, total_draft_rows,
                                               hidden_out, 0u);
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
                                    int target_next_token, uint32_t pos0,
                                    int32_t* tokens_out, uint32_t* n_out,
                                    const ds4_dspark_sampler* sampler,
                                    uint32_t max_tokens, size_t concurrency) {
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
  bool ok = rocm_graph_dspark_prepare_hidden(g, target_model, target_weights,
                                             target_next_token, pos0,
                                             g->batch_ffn_norm, 0u);
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

  const ds4_rocm_dspark_draft_item item{.graph = g,
                                        .target_next_token = target_next_token,
                                        .position = pos0,
                                        .max_draft_tokens = max_tokens,
                                        .tokens = tokens_out,
                                        .n_tokens = n_out,
                                        .sampler = sampler,
                                        .concurrency = concurrency};
  const bool sampled = sampler && sampler->propose;
  const bool selected =
      sampled ? rocm_graph_dspark_select_sampled(g, &item, 1, block)
              : rocm_graph_dspark_select_block(
                    g, g->spec_logits, target_next_token, block, tokens_out);
  ds4_gpu_set_small_batch_mode(0);
  if (!selected) {
    return false;
  }
  if (!sampled)
    *n_out = block;
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
    size_t item_count) {
  if (!engine || !engine->model || !engine->weights || !items ||
      item_count < 2u || item_count > 8u) {
    return false;
  }
  ds4_gpu_graph* coordinator = items[0].graph;
  if (!coordinator || !coordinator->dspark ||
      !coordinator->batch_dspark_verify_after_attn_hc) {
    return false;
  }
  const uint32_t block = coordinator->dspark->block_size;
  uint32_t requested_block = 0u;
  for (size_t index = 0; index < item_count; ++index) {
    requested_block = std::max(
        requested_block,
        std::min(block, items[index].max_draft_tokens));
  }
  const uint32_t total_rows =
      static_cast<uint32_t>(item_count) * requested_block;
  if (requested_block == 0u || total_rows > coordinator->prefill_cap ||
      total_rows > coordinator->batch_dspark_verify_rows_cap ||
      total_rows > DS4_SPEC_MAX_ROWS ||
      !rocm_graph_spec_prepare(coordinator, engine->weights, total_rows)) {
    return false;
  }

  bool ok = true;
  for (size_t index = 0; index < item_count; ++index) {
    const ds4_rocm_dspark_draft_item& item = items[index];
    if (!item.graph || item.graph->dspark != coordinator->dspark ||
        !item.tokens || !item.n_tokens) {
      return false;
    }
    *item.n_tokens = 0u;
  }

  ds4_gpu_set_small_batch_mode(2);

  ok = rocm_graph_dspark_prepare_hidden_batch(
      engine, items, item_count, requested_block,
      coordinator->batch_dspark_verify_after_attn_hc);

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

  const bool sampled = std::any_of(
      items, items + item_count,
      [](const auto& item) { return item.sampler && item.sampler->propose; });
  ok = sampled ? rocm_graph_dspark_select_sampled(coordinator, items,
                                                  item_count, requested_block)
               : rocm_graph_dspark_select_block_batch(
                     coordinator, items, item_count, coordinator->spec_logits,
                     requested_block);

  ds4_gpu_set_small_batch_mode(0);
  return ok;
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

  // The head is the largest arena. Drop idle older arenas, and drop an idle
  // head before growing it, so increasing session capacities do not retain
  // every previous allocation. Active borrowers keep their original buffers.
  for (auto** link = &engine->batch_workspace; *link;) {
    auto* workspace = *link;
    const bool reuse_head = workspace == engine->batch_workspace &&
                            workspace->prefill_cap >= prefill_capacity;
    if (workspace->batch_workspace_borrowers != 0 || reuse_head) {
      link = &workspace->next_batch_workspace;
    } else {
      *link = workspace->next_batch_workspace;
      rocm_graph_free(workspace);
      free(workspace);
    }
  }
    auto *graph =
        static_cast<ds4_rocm_graph *>(ds4_xcalloc(1, sizeof(ds4_rocm_graph)));
    const uint32_t raw_capacity =
        rocm_graph_raw_cap_for_context(context_size, prefill_capacity);
    ds4_rocm_graph* batch_workspace = engine->batch_workspace;
    while (batch_workspace && batch_workspace->prefill_cap < prefill_capacity) {
      batch_workspace = batch_workspace->next_batch_workspace;
    }
    if (!rocm_graph_alloc_raw_cap(
            graph, engine->weights, &engine->weights->layer[0], raw_capacity,
            (uint32_t)context_size, prefill_capacity, batch_workspace)) {
      free(graph);
      return NULL;
    }
    if (!batch_workspace) {
      if (engine->dspark) {
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
      }
      auto* workspace =
          static_cast<ds4_rocm_graph*>(ds4_xcalloc(1, sizeof(ds4_rocm_graph)));
      workspace->prefill_cap = graph->prefill_cap;
      rocm_graph_bind_batch_workspace(workspace, graph);
      workspace->borrows_batch_workspace = false;
      graph->borrows_batch_workspace = true;
      workspace->next_batch_workspace = engine->batch_workspace;
      engine->batch_workspace = workspace;
      batch_workspace = workspace;
      fprintf(stderr, "ds4: sessions share one %u-row batch workspace\n",
              graph->prefill_cap);
    }
    graph->batch_workspace_owner = batch_workspace;
    ++batch_workspace->batch_workspace_borrowers;
    return graph;
}

void ds4_rocm_graph_destroy(ds4_rocm_graph *graph) {
  while (graph) {
    ds4_rocm_graph* next = graph->next_batch_workspace;
    if (graph->batch_workspace_owner)
      --graph->batch_workspace_owner->batch_workspace_borrowers;
    rocm_graph_free(graph);
    free(graph);
    graph = next;
  }
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
                                 int target_next_token, uint32_t pos0,
                                 int32_t* tokens_out, uint32_t* n_out,
                                 const ds4_dspark_sampler* sampler,
                                 uint32_t max_tokens, size_t concurrency) {
  return graph && engine &&
         rocm_graph_dspark_draft(graph, engine->model, engine->weights,
                                 target_next_token, pos0, tokens_out, n_out,
                                 sampler, max_tokens, concurrency);
}

bool ds4_rocm_graph_dspark_draft_head_batch(
    ds4_engine* engine, const ds4_rocm_dspark_draft_item* items,
    size_t item_count) {
  return rocm_graph_dspark_draft_head_batch(engine, items, item_count);
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
 * rows, and the compressor/indexer frontiers. DSpark additionally persists its
 * injected support KV and deterministic request policy.
 * Temporary captures are unnecessary once support coverage reaches the
 * checkpoint.
 */

#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_U32_FIELDS (19u + DS4_DSPARK_MAX_TARGET_LAYERS)
#define DS4_SESSION_IO_CHUNK (8u * 1024u * 1024u)

static constexpr std::array<uint64_t ds4_dspark_request_state::*, 12>
    dspark_snapshot_counters{
        &ds4_dspark_request_state::drafted,
        &ds4_dspark_request_state::accepted,
        &ds4_dspark_request_state::support_drafted,
        &ds4_dspark_request_state::support_accepted,
        &ds4_dspark_request_state::positional_accepted,
        &ds4_dspark_request_state::anchors,
        &ds4_dspark_request_state::full_blocks,
        &ds4_dspark_request_state::steps,
        &ds4_dspark_request_state::skipped,
        &ds4_dspark_request_state::policy_start_drafted,
        &ds4_dspark_request_state::policy_start_accepted,
        &ds4_dspark_request_state::policy_start_steps,
    };

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

static int payload_write_dspark_state(FILE* fp,
                                      const ds4_dspark_request_state& state,
                                      char* err, size_t errlen) {
  for (const auto field : dspark_snapshot_counters) {
    const uint64_t value = state.*field;
    if (payload_write_u32(fp, static_cast<uint32_t>(value), err, errlen) ||
        payload_write_u32(fp, static_cast<uint32_t>(value >> 32), err, errlen))
      return 1;
  }
  for (const uint32_t value :
       {state.skip_remaining, state.concurrent_width,
        uint32_t(state.plain_only), uint32_t(state.force_plain_request),
        state.policy_concurrency, state.probe_cycles, state.probe_emitted,
        state.probe_cost}) {
    if (payload_write_u32(fp, value, err, errlen))
      return 1;
  }
  return 0;
}

static int payload_read_dspark_state(FILE* fp, ds4_dspark_request_state& state,
                                     uint64_t* remaining, char* err,
                                     size_t errlen) {
  for (const auto field : dspark_snapshot_counters) {
    uint32_t low = 0, high = 0;
    if (payload_read_u32(fp, &low, remaining, err, errlen) ||
        payload_read_u32(fp, &high, remaining, err, errlen))
      return 1;
    state.*field = uint64_t(low) | (uint64_t(high) << 32);
  }
  uint32_t flags[8]{};
  for (auto& value : flags) {
    if (payload_read_u32(fp, &value, remaining, err, errlen))
      return 1;
  }
  if (flags[0] > DS4_DSPARK_RETRY_TOKENS || flags[1] == 0 ||
      flags[1] > DS4_DSPARK_MAX_BLOCK || flags[2] > 1 || flags[3] > flags[2] ||
      state.accepted > state.drafted ||
      state.support_accepted > state.support_drafted ||
      state.positional_accepted > state.support_drafted ||
      state.full_blocks > state.steps || state.anchors != state.steps ||
      state.policy_start_drafted > state.support_drafted ||
      state.policy_start_accepted > state.support_accepted ||
      state.policy_start_steps > state.steps || flags[4] < 1 || flags[4] > 8 ||
      flags[5] > 3 || flags[6] > flags[5] * (DS4_DSPARK_MAX_BLOCK + 1u) ||
      flags[7] > flags[5] * 1000u) {
    payload_set_err(err, errlen, "invalid DSpark controller state");
    return 1;
  }
  state.skip_remaining = flags[0];
  state.concurrent_width = flags[1];
  state.plain_only = flags[2] != 0;
  state.force_plain_request = flags[3] != 0;
  state.policy_concurrency = flags[4];
  state.probe_cycles = flags[5];
  state.probe_emitted = flags[6];
  state.probe_cost = flags[7];
  return 0;
}

static uint32_t session_dspark_live_rows(const ds4_gpu_graph* graph) {
  return std::min(graph->dspark_context_len, DS4_N_SWA);
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
      bytes > ds4_gpu_tensor_bytes(tensor) - offset) {
    payload_set_err(err, errlen, "session tensor is smaller than the payload");
    return 1;
  }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (ds4_gpu_tensor_snapshot_read(tensor, offset + done, buf, n) == 0) {
          payload_set_err(err, errlen,
                          "failed to read accelerator session tensor");
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
      bytes > ds4_gpu_tensor_bytes(tensor) - offset) {
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

// A logical ring window has at most two contiguous spans. Transfer each span
// in bulk; copying thousands of individual rows dominates long-prefix reuse.
static int payload_ring_tensor(FILE* fp, ds4_gpu_tensor* tensor,
                               uint32_t capacity, uint32_t context,
                               uint32_t live, bool writing, uint8_t* buffer,
                               uint64_t* remaining, char* err, size_t errlen) {
  const uint32_t first = (context - live) % capacity;
  const uint32_t before_wrap = std::min(live, capacity - first);
  constexpr uint64_t row_bytes = DS4_N_HEAD_DIM * sizeof(float);
  const auto transfer = [&](uint64_t offset, uint64_t bytes) {
    return writing
               ? payload_write_tensor_span(fp, tensor, offset, bytes, buffer,
                                           DS4_SESSION_IO_CHUNK, err, errlen)
               : payload_read_tensor_span(fp, tensor, offset, bytes, buffer,
                                          DS4_SESSION_IO_CHUNK, remaining, err,
                                          errlen);
  };
  if (transfer(first * row_bytes, before_wrap * row_bytes))
    return 1;
  return live > before_wrap ? transfer(0, (live - before_wrap) * row_bytes) : 0;
}

// Only the semantic target window is visible to the next draft. Physical
// slots used for old context or uncommitted draft rows are not continuation
// state.
static int payload_dspark_tensors(FILE* fp, const ds4_gpu_graph* graph,
                                  uint32_t context, bool writing,
                                  uint8_t* buffer, uint64_t* remaining,
                                  char* err, size_t errlen) {
  if (!graph->dspark || context == 0)
    return 0;
  const uint32_t live = std::min(context, DS4_N_SWA);
  for (uint32_t stage = 0; stage < graph->dspark->n_stages; ++stage) {
    if (payload_ring_tensor(fp, graph->dspark_kv_cache[stage],
                            graph->dspark_cache_cap, context, live, writing,
                            buffer, remaining, err, errlen))
      return 1;
  }
  return 0;
}

uint64_t ds4_rocm_graph_snapshot_bytes(const ds4_rocm_graph *graph,
                                       const ds4_tokens *checkpoint) {
    if (!graph || !checkpoint || checkpoint->len <= 0) return 0;
    if (graph->dspark && graph->dspark_capture_enabled &&
        graph->dspark_context_len != static_cast<uint32_t>(checkpoint->len)) {
      return 0;  // A partial capture is not a complete continuation.
    }
    uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += dspark_snapshot_counters.size() * sizeof(uint64_t) +
             8 * sizeof(uint32_t);
    if (graph->dspark && graph->dspark_context_len != 0) {
      bytes += uint64_t(graph->dspark->n_stages) *
               session_dspark_live_rows(graph) * DS4_N_HEAD_DIM * sizeof(float);
    }
    bytes += (uint64_t)checkpoint->len * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += session_payload_live_tensor_bytes(graph, (uint32_t)checkpoint->len);
    return bytes;
}

static int rocm_graph_save_payload(const ds4_rocm_graph* graph,
                                   const ds4_tokens* checkpoint,
                                   const float* logits,
                                   uint32_t prefill_capacity, int context_size,
                                   const ds4_dspark_request_state* state,
                                   FILE* fp, char* err, size_t errlen) {
  if (!graph || !checkpoint || checkpoint->len <= 0 || !logits || !fp) {
    payload_set_err(err, errlen, "invalid graph snapshot save");
    return 1;
  }
  // Successful session operations have already completed device writes.
  // The capture worker holds this session frozen while other sessions run.

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
      graph->dspark ? graph->dspark->n_stages : 0u,
      graph->dspark ? graph->dspark->block_size : 0u,
      graph->dspark ? graph->dspark_cache_cap : 0u,
      graph->dspark_context_len,
      graph->dspark && graph->dspark_capture_enabled ? 1u : 0u,
      graph->dspark ? graph->dspark->n_target_layers : 0u,
  };
  if (graph->dspark) {
    std::copy_n(graph->dspark->target_layer_ids, graph->dspark->n_target_layers,
                header + 19);
  }
  for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
    if (payload_write_u32(fp, header[i], err, errlen) != 0)
      return 1;
  }
  if (payload_write_dspark_state(fp, *state, err, errlen))
    return 1;
  for (int i = 0; i < checkpoint->len; i++) {
    if (payload_write_u32(fp, (uint32_t)checkpoint->v[i], err, errlen) != 0) {
      return 1;
    }
  }
  if (payload_write_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                          err, errlen) != 0) {
    return 1;
  }
  for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
    if (payload_write_u32(fp, graph->layer_n_comp[il], err, errlen) != 0) {
      return 1;
    }
  }
  for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
    if (payload_write_u32(fp, graph->layer_n_index_comp[il], err, errlen) !=
        0) {
      return 1;
    }
  }

  auto* buf = static_cast<uint8_t*>(ds4_xmalloc(DS4_SESSION_IO_CHUNK));
  int rc = 0;
  for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
    /* Write the raw ring in logical position order.  The file does not care
     * where the rows happened to live physically in the source graph. */
    rc = payload_ring_tensor(fp, graph->layer_raw_cache[il], graph->raw_cap,
                             static_cast<uint32_t>(checkpoint->len), raw_live,
                             true, buf, nullptr, err, errlen);
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (rc != 0 || ratio == 0)
      continue;
    /* Compressed rows are append-only from row zero, so the live prefix is
     * contiguous.  The two compressor state tensors hold the partial window
     * that will become the next compressed row. */
    rc = payload_write_tensor_span(
        fp, graph->layer_attn_comp_cache[il], 0,
        (uint64_t)graph->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float), buf,
        DS4_SESSION_IO_CHUNK, err, errlen);
    if (rc == 0)
      rc = payload_write_tensor_span(fp, graph->layer_attn_state_kv[il], 0,
                                     layer_attn_state_bytes(ratio), buf,
                                     DS4_SESSION_IO_CHUNK, err, errlen);
    if (rc == 0)
      rc = payload_write_tensor_span(fp, graph->layer_attn_state_score[il], 0,
                                     layer_attn_state_bytes(ratio), buf,
                                     DS4_SESSION_IO_CHUNK, err, errlen);
    if (rc == 0 && ratio == 4) {
      rc = payload_write_tensor_span(fp, graph->layer_index_comp_cache[il], 0,
                                     (uint64_t)graph->layer_n_index_comp[il] *
                                         DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                     buf, DS4_SESSION_IO_CHUNK, err, errlen);
      if (rc == 0)
        rc = payload_write_tensor_span(fp, graph->layer_index_state_kv[il], 0,
                                       layer_index_state_bytes(ratio), buf,
                                       DS4_SESSION_IO_CHUNK, err, errlen);
      if (rc == 0)
        rc = payload_write_tensor_span(fp, graph->layer_index_state_score[il],
                                       0, layer_index_state_bytes(ratio), buf,
                                       DS4_SESSION_IO_CHUNK, err, errlen);
    }
  }
  if (rc == 0)
    rc = payload_dspark_tensors(fp, graph, graph->dspark_context_len, true, buf,
                                nullptr, err, errlen);
  free(buf);
  return rc;
}

static int rocm_graph_load_payload(ds4_rocm_graph* graph,
                                   ds4_tokens* checkpoint, float* logits,
                                   uint32_t prefill_capacity, int context_size,
                                   ds4_dspark_request_state* state, FILE* fp,
                                   uint64_t payload_bytes, char* err,
                                   size_t errlen) {
  if (!graph || !checkpoint || !logits || !fp) {
    payload_set_err(err, errlen, "invalid graph snapshot load");
    return 1;
  }
  uint64_t remaining = payload_bytes;
  uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
  for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
    if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0)
      return 1;
  }
  if (h[0] != DS4_SESSION_PAYLOAD_MAGIC ||
      h[1] != DS4_SESSION_PAYLOAD_VERSION) {
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
      h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB) {
    payload_set_err(err, errlen,
                    "KV checkpoint was written for a different DS4 layout");
    return 1;
  }
  /* prefill_cap is scratch scheduling capacity, not durable KV layout.
   * Old checkpoints remain valid as long as the raw KV window matches. */
  (void)saved_prefill_cap;
  (void)prefill_capacity;
  if (saved_raw_window != graph->raw_window) {
    payload_set_err(
        err, errlen,
        "KV checkpoint graph chunk layout does not match current runtime");
    return 1;
  }
  /* The raw rows in the file are logical rows.  We can restore them into any
   * current ring with enough capacity, but the saved live count must be exactly
   * the last window implied by the saved token count. */
  const uint32_t expected_raw_live =
      saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
  if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
      saved_raw_live > saved_raw_cap || saved_raw_live > graph->raw_cap) {
    payload_set_err(
        err, errlen,
        "KV checkpoint raw ring layout does not match current context");
    return 1;
  }
  if (saved_comp_cap > graph->comp_cap) {
    payload_set_err(
        err, errlen,
        "KV checkpoint compressed cache is larger than current context");
    return 1;
  }

  const uint32_t stages = graph->dspark ? graph->dspark->n_stages : 0u;
  const uint32_t block = graph->dspark ? graph->dspark->block_size : 0u;
  const uint32_t targets = graph->dspark ? graph->dspark->n_target_layers : 0u;
  if (saved_tokens == 0 || h[13] != stages || h[14] != block ||
      h[15] != (graph->dspark ? graph->dspark_cache_cap : 0u) ||
      h[16] > saved_tokens || h[17] > 1u || h[18] != targets ||
      (h[17] != 0u && h[16] != saved_tokens) || (h[17] == 0u && h[16] != 0u)) {
    payload_set_err(err, errlen,
                    "incompatible DSpark snapshot layout or coverage");
    return 1;
  }
  for (uint32_t i = 0; i < DS4_DSPARK_MAX_TARGET_LAYERS; ++i) {
    const uint32_t layer =
        i < targets ? graph->dspark->target_layer_ids[i] : 0u;
    if (h[19 + i] != layer) {
      payload_set_err(err, errlen, "incompatible DSpark target feature layout");
      return 1;
    }
  }
  ds4_dspark_request_state restored_state{};
  if (payload_read_dspark_state(fp, restored_state, &remaining, err, errlen))
    return 1;
  if (graph->dspark && ((h[17] != 0u) == restored_state.force_plain_request)) {
    payload_set_err(err, errlen,
                    "inconsistent DSpark capture and request state");
    return 1;
  }

  token_vec new_checkpoint = {0};
  for (uint32_t i = 0; i < saved_tokens; i++) {
    uint32_t tok = 0;
    if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
      ds4_tokens_free(&new_checkpoint);
      return 1;
    }
    if (tok >= DS4_N_VOCAB) {
      ds4_tokens_free(&new_checkpoint);
      payload_set_err(err, errlen, "invalid checkpoint token ID");
      return 1;
    }
    ds4_tokens_push(&new_checkpoint, (int)tok);
  }
  if (payload_read_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                         &remaining, err, errlen) != 0) {
    ds4_tokens_free(&new_checkpoint);
    return 1;
  }
  if (!std::all_of(logits, logits + DS4_N_VOCAB,
                   [](float value) { return std::isfinite(value); })) {
    ds4_tokens_free(&new_checkpoint);
    payload_set_err(err, errlen, "checkpoint contains non-finite logits");
    return 1;
  }
  uint32_t n_comp[DS4_MAX_LAYER];
  uint32_t n_index_comp[DS4_MAX_LAYER];
  for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
    if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
      ds4_tokens_free(&new_checkpoint);
      return 1;
    }
    if (n_comp[il] > saved_comp_cap || n_comp[il] > graph->layer_comp_cap[il]) {
      ds4_tokens_free(&new_checkpoint);
      payload_set_err(err, errlen,
                      "KV checkpoint has invalid compressed row count");
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
      payload_set_err(err, errlen,
                      "KV checkpoint has invalid indexer row count");
      return 1;
    }
  }

  if (ds4_gpu_synchronize() == 0) {
    ds4_tokens_free(&new_checkpoint);
    payload_set_err(err, errlen,
                    "failed to synchronize accelerator before KV restore");
    return 1;
  }
  for (uint32_t il = 0; il < DS4_N_LAYER; ++il) {
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (ratio == 0)
      continue;
    const uint64_t required =
        static_cast<uint64_t>(std::max(n_comp[il], n_index_comp[il])) +
        (DS4_N_SWA + ratio - 1) / ratio + 2;
    if (!rocm_graph_reserve_compressed_rows(
            graph, il,
            static_cast<uint32_t>(
                std::min<uint64_t>(graph->layer_comp_cap[il], required)))) {
      ds4_tokens_free(&new_checkpoint);
      payload_set_err(err, errlen,
                      "failed to reserve compressed cache for KV restore");
      return 1;
    }
  }
  auto* buf = static_cast<uint8_t*>(ds4_xmalloc(DS4_SESSION_IO_CHUNK));
  int rc = 0;
  for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
    /* Rebuild the physical raw ring expected by the current graph.  This is
     * why the file stores rows in logical order instead of dumping bytes from
     * the old ring layout. */
    rc = payload_ring_tensor(fp, graph->layer_raw_cache[il], graph->raw_cap,
                             saved_tokens, saved_raw_live, false, buf,
                             &remaining, err, errlen);
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (rc != 0 || ratio == 0)
      continue;
    rc = payload_read_tensor_span(
        fp, graph->layer_attn_comp_cache[il], 0,
        (uint64_t)n_comp[il] * DS4_N_HEAD_DIM * sizeof(float), buf,
        DS4_SESSION_IO_CHUNK, &remaining, err, errlen);
    if (rc == 0 && ratio == 4 &&
        !rocm_graph_rebuild_attn_comp_mirror(graph, il, n_comp[il])) {
      payload_set_err(err, errlen,
                      "failed to rebuild FP16 attention cache mirror");
      rc = 1;
    }
    if (rc == 0)
      rc = payload_read_tensor_span(
          fp, graph->layer_attn_state_kv[il], 0, layer_attn_state_bytes(ratio),
          buf, DS4_SESSION_IO_CHUNK, &remaining, err, errlen);
    if (rc == 0)
      rc = payload_read_tensor_span(fp, graph->layer_attn_state_score[il], 0,
                                    layer_attn_state_bytes(ratio), buf,
                                    DS4_SESSION_IO_CHUNK, &remaining, err,
                                    errlen);
    if (rc == 0 && ratio == 4) {
      rc = payload_read_tensor_span(
          fp, graph->layer_index_comp_cache[il], 0,
          (uint64_t)n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
          buf, DS4_SESSION_IO_CHUNK, &remaining, err, errlen);
      if (rc == 0)
        rc = payload_read_tensor_span(fp, graph->layer_index_state_kv[il], 0,
                                      layer_index_state_bytes(ratio), buf,
                                      DS4_SESSION_IO_CHUNK, &remaining, err,
                                      errlen);
      if (rc == 0)
        rc = payload_read_tensor_span(fp, graph->layer_index_state_score[il], 0,
                                      layer_index_state_bytes(ratio), buf,
                                      DS4_SESSION_IO_CHUNK, &remaining, err,
                                      errlen);
    }
  }
  if (rc == 0)
    rc = payload_dspark_tensors(fp, graph, h[16], false, buf, &remaining, err,
                                errlen);
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
    payload_set_err(err, errlen,
                    "failed to synchronize accelerator after KV restore");
    return 1;
  }

  ds4_tokens_free(checkpoint);
  *checkpoint = new_checkpoint;
  for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
    graph->layer_n_comp[il] = n_comp[il];
    graph->layer_n_index_comp[il] = n_index_comp[il];
  }
  *state = restored_state;
  graph->dspark_capture_enabled = h[17] != 0u;
  graph->dspark_context_len = h[16];
  graph->dspark_capture_mask = 0;
  graph->dspark_capture_batch_mask = 0;
  graph->dspark_capture_batch_start = 0;
  graph->dspark_capture_batch_tokens = 0;
  return 0;
}

int ds4_rocm_graph_save_snapshot(const ds4_rocm_graph* graph,
                                 const ds4_tokens* checkpoint,
                                 const float* logits, uint32_t prefill_capacity,
                                 int context_size,
                                 const ds4_dspark_request_state* state,
                                 ds4_session_snapshot* snap, char* err,
                                 size_t errlen) {
  if (!graph || !checkpoint || !logits || !snap) {
    payload_set_err(err, errlen, "invalid graph snapshot save");
    return 1;
  }
  const uint64_t bytes = ds4_rocm_graph_snapshot_bytes(graph, checkpoint);
  if (bytes == 0) {
    payload_set_err(err, errlen, "session has no valid checkpoint to snapshot");
    return 1;
  }
  if (bytes >= (uint64_t)SIZE_MAX) {
    payload_set_err(err, errlen,
                    "session snapshot is too large for this platform");
    return 1;
  }
  // fmemopen appends a terminator on close, including in binary mode.
  // Keep that byte outside the serialized payload.
  if (snap->cap < bytes + 1u) {
    auto* p = static_cast<uint8_t*>(
        realloc(snap->ptr, static_cast<size_t>(bytes + 1u)));
    if (!p) {
      payload_set_err(err, errlen,
                      "out of memory while allocating session snapshot");
      return 1;
    }
    snap->ptr = p;
    snap->cap = bytes + 1u;
  }

  FILE* fp = fmemopen(snap->ptr, static_cast<size_t>(bytes + 1u), "wb");
  if (!fp) {
    payload_set_err(err, errlen,
                    "failed to open memory stream for session snapshot");
    return 1;
  }
  const int rc =
      rocm_graph_save_payload(graph, checkpoint, logits, prefill_capacity,
                              context_size, state, fp, err, errlen);
  if (fclose(fp) != 0 && rc == 0) {
    payload_set_err(err, errlen, "failed to finalize memory session snapshot");
    return 1;
  }
  if (rc != 0)
    return 1;
  snap->len = bytes;
  return 0;
}

int ds4_rocm_graph_load_snapshot(ds4_rocm_graph* graph, ds4_tokens* checkpoint,
                                 float* logits, uint32_t prefill_capacity,
                                 int context_size,
                                 ds4_dspark_request_state* state,
                                 const ds4_session_snapshot* snap, char* err,
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
    const int rc = rocm_graph_load_payload(graph, checkpoint, logits,
                                           prefill_capacity, context_size,
                                           state, fp, snap->len, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close memory session snapshot");
        return 1;
    }
    return rc;
}

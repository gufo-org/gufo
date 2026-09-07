#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_DSPARK_INTERNAL_H_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_DSPARK_INTERNAL_H_

#include <cstddef>
#include <cstdint>

#include "model_data_internal.h"

/*
 * DeepSeek V4 Flash DSpark support model.
 *
 * DSpark is DeepSeek's own speculative drafter for this checkpoint. It is a
 * separate GGUF holding a short stack of full DS4 blocks (MLA attention, routed
 * MoE, hyper-connections) plus a low-rank Markov path selector. The support
 * artifact also carries a confidence head for format compatibility, but the
 * retained fixed-block scheduler does not evaluate it. DSpark is unrelated to
 * the Qwen DFlash and DFlash-2 drafters, which use block diffusion over a
 * single fused draft layer.
 *
 * One draft pass proposes a whole block of `block_size` tokens:
 *
 *   1. Feature injection - the target's hidden states at `target_layer_ids` are
 *      concatenated, projected by `main_proj`, normalized by `main_norm`, and
 *      turned into per-stage KV rows. This is what lets a three-block drafter
 *      track a forty-three-block target.
 *   2. Non-causal block attention - the block enters as
 *      [last_committed_token, MASK, MASK, ...] embedded with the target's tied
 *      token embeddings. Every block row attends to the injected KV *and* to
 *      all other block rows, so the whole block is produced by one forward
 *      pass instead of `block_size` sequential steps.
 *   3. Path selection - base logits come from the target LM head; the Markov
 *      head then adds W2 * W1[previous token] so the block follows one
 *      coherent trajectory rather than independent per-position argmaxes.
 */

inline constexpr uint32_t DS4_DSPARK_MAX_STAGES = 8;
inline constexpr uint32_t DS4_DSPARK_MAX_BLOCK = 16;
inline constexpr uint32_t DS4_DSPARK_MAX_TARGET_LAYERS = 8;

/* One request verifies its drafted block plus the target correction row.
 * A coordinator lazily grows to eight such independent request blocks. */
inline constexpr uint32_t DS4_SPEC_SESSION_MAX_ROWS =
    DS4_DSPARK_MAX_BLOCK + 1u;
inline constexpr uint32_t DS4_SPEC_MAX_ROWS =
    8u * DS4_SPEC_SESSION_MAX_ROWS;

struct ds4_dspark_stage_weights {
    /* Present on the first stage only: fuses the target's sampled layers. */
    ds4_tensor *main_proj;
    ds4_tensor *main_norm;

    /* Present on the last stage only. */
    ds4_tensor *norm;
    ds4_tensor *hc_head_base;
    ds4_tensor *hc_head_fn;
    ds4_tensor *hc_head_scale;
    ds4_tensor *markov_w1;
    ds4_tensor *markov_w2;
    /* Validated and cached because it is part of the support artifact, but not
     * evaluated by the retained fixed-width scheduler. */
    ds4_tensor *confidence_proj;

    /* Every stage is a DS4 block with no compressor and no indexer, so it
     * binds into the same layout the target's ratio-0 layers use. */
    ds4_layer_weights block;
};

struct ds4_dspark_model {
    ds4_model *model;
    uint32_t n_stages;
    uint32_t block_size;
    uint32_t markov_rank;
    uint32_t noise_token_id;
    uint32_t n_target_layers;
    uint32_t target_layer_ids[DS4_DSPARK_MAX_TARGET_LAYERS];
    ds4_dspark_stage_weights stage[DS4_DSPARK_MAX_STAGES];
};

/* Loads and strictly validates a DSpark support GGUF, then makes its tensor
 * spans GPU-visible through the selected residency policy. Returns 0 on
 * success. */
int ds4_dspark_open(ds4_dspark_model **out, const char *path);
void ds4_dspark_close(ds4_dspark_model *dspark);

/* Total concatenated target feature width consumed by main_proj. */
uint32_t ds4_dspark_feature_width(const ds4_dspark_model *dspark);

#endif  // GUFO_MODELS_DEEPSEEK_V4_FLASH_RUNTIME_DSPARK_INTERNAL_H_

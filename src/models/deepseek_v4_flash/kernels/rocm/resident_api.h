#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_ROCM_RESIDENT_API_H_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_ROCM_RESIDENT_API_H_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

struct ds4_gpu_tensor;

int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a,
                       const ds4_gpu_tensor *b, uint32_t n);
int ds4_gpu_attention_decode_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, const ds4_gpu_tensor *comp_kv, uint32_t comp_kv_f16, uint32_t n_comp, const ds4_gpu_tensor *comp_mask, uint32_t use_mask, uint32_t n_head, uint32_t head_dim);
int ds4_gpu_attention_decode_mixed_batch_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv, uint32_t comp_kv_f16, const ds4_gpu_tensor *comp_mask, uint32_t use_comp_mask, uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim);
int ds4_gpu_attention_decode_raw_batch_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t window, uint32_t n_head, uint32_t head_dim);
int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv, uint32_t comp_kv_f16, const ds4_gpu_tensor *topk, uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp, uint32_t top_k, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim);
int ds4_gpu_attention_output_low_q8_tensor(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size, uint64_t out_a_offset, uint64_t group_dim, uint64_t rank, uint32_t n_groups, const ds4_gpu_tensor *heads);
int ds4_gpu_attention_output_q8_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *low, ds4_gpu_tensor *group_tmp, ds4_gpu_tensor *low_tmp, const void *model_map, uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset, uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint64_t out_dim, const ds4_gpu_tensor *heads, uint32_t n_tokens);
/* Same projection with the inverse rotated tail folded into the F16 group pack,
 * which removes a read-modify-write pass over the whole heads tensor. Returns 0
 * when the shape, width, or output route cannot absorb it, so the caller keeps
 * its separate ds4_gpu_rope_tail_tensor launch. */
int ds4_gpu_attention_output_q8_batch_inv_rope_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *low, ds4_gpu_tensor *group_tmp, ds4_gpu_tensor *low_tmp, const void *model_map, uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset, uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint64_t out_dim, const ds4_gpu_tensor *heads, uint32_t n_tokens, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
int ds4_gpu_attention_prefill_raw_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim);
int ds4_gpu_attention_prefill_static_mixed_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv, uint32_t comp_kv_f16, uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim);
int ds4_gpu_begin_commands(void);
int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
int ds4_gpu_reserve_support_map(const void *support_map, uint64_t support_size, uint64_t arena_bytes);
int ds4_gpu_cache_support_range(const void *support_map, uint64_t support_size, uint64_t offset, uint64_t bytes, const char *label);
void ds4_gpu_release_support_map(void);
/* Enables verification-block kernel selection. Off for prefill and decode so
 * their kernel choice, output, and throughput stay exactly as baselined. */
void ds4_gpu_set_small_batch_mode(int enabled);
int ds4_gpu_dspark_repeat_hc_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *x, uint32_t n_embd, uint32_t n_hc, uint32_t n_rows);
int ds4_gpu_dspark_capture_features_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *hc, uint32_t out_row_stride, uint32_t slot_offset, uint32_t n_embd, uint32_t n_hc, uint32_t n_rows);
void ds4_gpu_cleanup(void);
int ds4_gpu_compressor_prefill_ratio4_replay_tensor(ds4_gpu_tensor *comp_cache, ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *sc, const void *model_map, uint64_t model_size, uint64_t ape_offset, uint32_t ape_type, uint64_t norm_offset, uint32_t norm_type, uint32_t head_dim, uint32_t pos0, uint32_t n_tokens, uint32_t n_rot, uint32_t n_ctx_orig, bool quantize_fp8, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float rms_eps, ds4_gpu_tensor *comp_mirror_f16);
int ds4_gpu_compressor_prefill_state_ratio4_tensor(ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, const ds4_gpu_tensor *kv_tail, const ds4_gpu_tensor *sc_tail, const void *model_map, uint64_t model_size, uint64_t ape_offset, uint32_t ape_type, uint32_t head_dim, uint32_t pos0);
int ds4_gpu_compressor_prefill_tensor(ds4_gpu_tensor *comp_cache, ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *sc, const void *model_map, uint64_t model_size, uint64_t ape_offset, uint32_t ape_type, uint64_t norm_offset, uint32_t norm_type, uint32_t head_dim, uint32_t ratio, uint32_t pos0, uint32_t n_tokens, uint32_t n_rot, uint32_t n_ctx_orig, bool quantize_fp8, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float rms_eps, ds4_gpu_tensor *comp_mirror_f16);
int ds4_gpu_compressor_update_tensor(const ds4_gpu_tensor *kv_cur, const ds4_gpu_tensor *sc_cur, ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, ds4_gpu_tensor *comp_cache, const void *model_map, uint64_t model_size, uint64_t ape_offset, uint32_t ape_type, uint64_t norm_offset, uint32_t norm_type, uint32_t head_dim, uint32_t ratio, uint32_t pos, uint32_t comp_row, uint32_t n_rot, uint32_t n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float rms_eps, bool state_already_stored, bool decode_one_token, bool defer_finalize);
int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot);
int ds4_gpu_dsv4_fp8_kv_quantize_mirror_f16_tensor(ds4_gpu_tensor *x, ds4_gpu_tensor *mirror_f16, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot);
int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim);
int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(ds4_gpu_tensor *q_out, const ds4_gpu_tensor *q, const void *model_map, uint64_t model_size, uint64_t q_weight_offset, uint32_t q_n, ds4_gpu_tensor *kv_out, const ds4_gpu_tensor *kv, uint64_t kv_weight_offset, uint32_t kv_n, uint32_t rows, float eps);
int ds4_gpu_dsv4_topk_mask_tensor(ds4_gpu_tensor *mask, const ds4_gpu_tensor *topk, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k);
int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc);
int ds4_gpu_embed_tokens_hc_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *tokens, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc);
int ds4_gpu_end_commands(void);
int ds4_gpu_flush_commands(void);
int ds4_gpu_hc_expand_add_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc);
int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc);
int ds4_gpu_hc_expand_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc);
int ds4_gpu_hc_split_sinkhorn_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t sinkhorn_iters, float eps);
int ds4_gpu_hc_split_weighted_sum_norm_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *norm_out, ds4_gpu_tensor *split, const ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual_hc, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint64_t norm_weight_offset, uint32_t n_embd, uint32_t n_hc, uint32_t sinkhorn_iters, float eps, float norm_eps);
int ds4_gpu_hc_split_weighted_sum_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *split, const ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual_hc, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_embd, uint32_t n_hc, uint32_t sinkhorn_iters, float eps);
int ds4_gpu_hc_weighted_sum_split_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc);
int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc);
int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps);
/* Fused head RMS norm plus rotated tail. Returns 0 when the shape is not
 * supported so the caller keeps the separate norm and rope launches. */
int ds4_gpu_head_rms_norm_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float eps);
int ds4_gpu_indexer_score_one_tensor(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp, uint32_t n_comp, uint32_t n_head, uint32_t head_dim, float scale);
int ds4_gpu_indexer_scores_decode_batch_tensor(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp, uint32_t n_comp, uint32_t n_tokens, uint32_t pos0, uint32_t n_head, uint32_t head_dim, uint32_t ratio, float scale);
int ds4_gpu_indexer_scores_prefill_tensor(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp, uint32_t n_comp, uint32_t n_tokens, uint32_t n_head, uint32_t head_dim, uint32_t ratio, float scale);
int ds4_gpu_indexer_topk_tensor(ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k);
int ds4_gpu_init(void);
int ds4_gpu_kv_fp8_store_raw_tensor(ds4_gpu_tensor *kv, ds4_gpu_tensor *raw_cache, uint32_t raw_cap, uint32_t row, uint32_t head_dim, uint32_t n_rot);
int ds4_gpu_matmul_f16_pair_tensor(ds4_gpu_tensor *out_a, ds4_gpu_tensor *out_b, const void *model_map, uint64_t model_size, uint64_t weight_a_offset, uint64_t weight_b_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok);
int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok);
int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok);
int ds4_gpu_matmul_q8_0_hc_expand_tensor(ds4_gpu_tensor *out_hc, ds4_gpu_tensor *block_out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc);
int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok);
int ds4_gpu_output_hc_weights_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *pre, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, float eps);
int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, uint32_t rows, float eps);
int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, float eps);
int ds4_gpu_rms_norm_weight_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps);
int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps);
int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *experts, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, bool *mid_is_f16, bool force_resident);
int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *experts, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, const ds4_gpu_tensor *add_in, uint32_t layer_index, bool force_resident);
int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_tokens);
int ds4_gpu_router_select_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t token, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits);
int ds4_gpu_set_model_fd(int fd);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
int ds4_gpu_shared_down_hc_expand_q8_0_tensor(ds4_gpu_tensor *out_hc, ds4_gpu_tensor *shared_out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *shared_mid, const ds4_gpu_tensor *routed_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc);
int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, float clamp);
int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
int ds4_gpu_spec_row_argmax_tensor(ds4_gpu_tensor *out_index, const ds4_gpu_tensor *logits, uint32_t vocab, uint32_t n_rows);
int ds4_gpu_attention_noncausal_raw_batch_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t n_head, uint32_t head_dim);
int ds4_gpu_dspark_markov_w1_row_tensor(ds4_gpu_tensor *out_state, const void *model_map, uint64_t model_size, uint64_t w1_offset, uint32_t markov_rank, uint32_t token);
int ds4_gpu_dspark_markov_argmax_tensor(ds4_gpu_tensor *out_index, ds4_gpu_tensor *scratch_key, const ds4_gpu_tensor *logits_row, const void *model_map, uint64_t model_size, uint64_t w1_offset, uint64_t w2_offset, uint32_t vocab, uint32_t markov_rank, uint32_t previous_token);
int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim);
int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim);
int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight);
int ds4_gpu_synchronize(void);
ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes);
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset, const ds4_gpu_tensor *src, uint64_t src_offset, uint64_t bytes);
int ds4_gpu_tensor_convert_f32_to_f16(ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t count);
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count);
void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GUFO_MODELS_DEEPSEEK_V4_FLASH_ROCM_RESIDENT_API_H_

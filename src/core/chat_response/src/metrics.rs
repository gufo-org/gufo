//! Generation metrics and `usage` JSON encoding.

use gufo_json::{Object, Value};

use crate::format::{num_f64, num_usize, parse_json, push_member};

#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_prompt_tokens: usize,
    pub reasoning_tokens: usize,
    pub draft_tokens: usize,
    pub draft_accepted_tokens: usize,
    pub draft_rounds: usize,
    pub prompt_tokens_per_second: f64,
    pub decode_ms: f64,
    pub cache_hit: bool,
    pub cache_miss_reason: String,
    pub cache_common_prefix_tokens: usize,
    pub cache_checkpoint_tokens: usize,
    pub cache_restore_bytes: usize,
    pub cache_snapshot_bytes: usize,
    pub cache_disk_queued_bytes: usize,
    pub cache_restore_ms: f64,
    pub cache_snapshot_ms: f64,
    pub cache_disk_enqueue_ms: f64,
    pub cache_disk_hit: bool,
    pub prefill_tokens: usize,
    pub prefill_chunks: usize,
    pub active_decode_prefill_chunks: usize,
    pub max_prefill_chunk_tokens: usize,
    pub max_prefill_layer_groups: usize,
    pub prefill_yield_rounds: usize,
    pub queue_depth_at_submit: usize,
    pub client_queue_depth_at_submit: usize,
    pub resident_requests_at_admission: usize,
    pub requested_logical_concurrency: usize,
    pub physical_execution_width: usize,
    pub queue_ms: f64,
    pub shared_prefix_wait_ms: f64,
    pub prefill_ms: f64,
    pub ttft_ms: f64,
    pub mean_inter_token_ms: f64,
    pub max_inter_token_ms: f64,
    pub execution_plan: String,
    pub component_cache_json: String,
    pub timings_json: String,
}

impl Metrics {
    pub fn timings_value(&self) -> Value {
        parse_json(&self.timings_json).unwrap_or_else(|| Value::Object(Object::new()))
    }
}

pub fn usage_json(m: &Metrics) -> Value {
    let mut usage = Object::new();
    push_member(&mut usage, "prompt_tokens", num_usize(m.prompt_tokens));
    push_member(
        &mut usage,
        "completion_tokens",
        num_usize(m.completion_tokens),
    );
    push_member(
        &mut usage,
        "total_tokens",
        num_usize(m.prompt_tokens.saturating_add(m.completion_tokens)),
    );

    let mut prompt_details = Object::new();
    push_member(
        &mut prompt_details,
        "cached_tokens",
        num_usize(m.cached_prompt_tokens),
    );
    push_member(
        &mut usage,
        "prompt_tokens_details",
        Value::Object(prompt_details),
    );

    let mut completion_details = Object::new();
    push_member(
        &mut completion_details,
        "reasoning_tokens",
        num_usize(m.reasoning_tokens),
    );
    push_member(
        &mut usage,
        "completion_tokens_details",
        Value::Object(completion_details),
    );

    let predicted_per_second = if m.decode_ms > 0.0 && m.completion_tokens > 0 {
        (m.completion_tokens as f64) / (m.decode_ms / 1000.0)
    } else {
        0.0
    };

    push_member(
        &mut usage,
        "cached_tokens",
        num_usize(m.cached_prompt_tokens),
    );
    push_member(
        &mut usage,
        "prompt_tokens_per_second",
        num_f64(m.prompt_tokens_per_second),
    );
    push_member(
        &mut usage,
        "completion_tokens_per_second",
        num_f64(predicted_per_second),
    );
    push_member(&mut usage, "draft_tokens", num_usize(m.draft_tokens));
    push_member(
        &mut usage,
        "draft_tokens_accepted",
        num_usize(m.draft_accepted_tokens),
    );

    let mut metrics = Object::new();
    if let Some(component_cache) = parse_json(&m.component_cache_json) {
        push_member(&mut metrics, "component_cache", component_cache);
    }
    push_member(&mut metrics, "cache_hit", Value::Bool(m.cache_hit));
    if !m.cache_miss_reason.is_empty() {
        push_member(
            &mut metrics,
            "cache_miss_reason",
            Value::String(m.cache_miss_reason.clone()),
        );
        push_member(
            &mut metrics,
            "cache_common_prefix_tokens",
            num_usize(m.cache_common_prefix_tokens),
        );
        push_member(
            &mut metrics,
            "cache_checkpoint_tokens",
            num_usize(m.cache_checkpoint_tokens),
        );
    }
    push_member(
        &mut metrics,
        "cache_restore_bytes",
        num_usize(m.cache_restore_bytes),
    );
    push_member(
        &mut metrics,
        "cache_snapshot_bytes",
        num_usize(m.cache_snapshot_bytes),
    );
    push_member(
        &mut metrics,
        "cache_disk_queued_bytes",
        num_usize(m.cache_disk_queued_bytes),
    );
    push_member(&mut metrics, "cache_shared_bytes", num_usize(0));
    push_member(
        &mut metrics,
        "cache_restore_ms",
        num_f64(m.cache_restore_ms),
    );
    push_member(
        &mut metrics,
        "cache_snapshot_ms",
        num_f64(m.cache_snapshot_ms),
    );
    push_member(
        &mut metrics,
        "cache_disk_enqueue_ms",
        num_f64(m.cache_disk_enqueue_ms),
    );
    push_member(
        &mut metrics,
        "cache_disk_hit",
        Value::Bool(m.cache_disk_hit),
    );
    push_member(&mut metrics, "draft_rounds", num_usize(m.draft_rounds));
    push_member(&mut metrics, "prefill_tokens", num_usize(m.prefill_tokens));
    push_member(&mut metrics, "prefill_chunks", num_usize(m.prefill_chunks));
    push_member(
        &mut metrics,
        "active_decode_prefill_chunks",
        num_usize(m.active_decode_prefill_chunks),
    );
    push_member(
        &mut metrics,
        "max_prefill_chunk_tokens",
        num_usize(m.max_prefill_chunk_tokens),
    );
    push_member(
        &mut metrics,
        "max_prefill_layer_groups",
        num_usize(m.max_prefill_layer_groups),
    );
    push_member(
        &mut metrics,
        "prefill_yield_rounds",
        num_usize(m.prefill_yield_rounds),
    );
    push_member(
        &mut metrics,
        "queue_depth_at_submit",
        num_usize(m.queue_depth_at_submit),
    );
    push_member(
        &mut metrics,
        "client_queue_depth_at_submit",
        num_usize(m.client_queue_depth_at_submit),
    );
    push_member(
        &mut metrics,
        "resident_requests_at_admission",
        num_usize(m.resident_requests_at_admission),
    );
    push_member(
        &mut metrics,
        "requested_logical_concurrency",
        num_usize(m.requested_logical_concurrency),
    );
    push_member(
        &mut metrics,
        "physical_execution_width",
        num_usize(m.physical_execution_width),
    );
    push_member(&mut metrics, "queue_ms", num_f64(m.queue_ms));
    push_member(
        &mut metrics,
        "shared_prefix_wait_ms",
        num_f64(m.shared_prefix_wait_ms),
    );
    push_member(&mut metrics, "prefill_ms", num_f64(m.prefill_ms));
    push_member(&mut metrics, "decode_ms", num_f64(m.decode_ms));
    push_member(&mut metrics, "ttft_ms", num_f64(m.ttft_ms));
    push_member(
        &mut metrics,
        "mean_inter_token_ms",
        num_f64(m.mean_inter_token_ms),
    );
    push_member(
        &mut metrics,
        "max_inter_token_ms",
        num_f64(m.max_inter_token_ms),
    );
    push_member(
        &mut metrics,
        "execution_plan",
        Value::String(m.execution_plan.clone()),
    );
    push_member(&mut usage, "gufo", Value::Object(metrics));
    Value::Object(usage)
}

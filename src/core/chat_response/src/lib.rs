//! C ABI for OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages
//! response and SSE encoding.
#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod anthropic;
mod chat;
mod format;
mod metrics;
mod responses;

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ffi::c_void;

use anthropic::AnthropicOutput;
use chat::{ChatStream, FinishReason, NonStreamingInput, ToolCall, ToolDelta};
use gufo_ffi::{guard, input};
use metrics::Metrics;
use responses::{ResponsesConfig, ResponsesOutput};

pub struct GufoChatStream {
    inner: ChatStream,
}

pub struct GufoResponsesOutput {
    inner: ResponsesOutput,
}

pub struct GufoAnthropicOutput {
    inner: AnthropicOutput,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoChatResponseStatus {
    Ok = 0,
    Closed = 1,
    Internal = 2,
    /// The input is malformed; the reason went to the error writer.
    Invalid = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoChatFinishReason {
    Stop = 0,
    Length = 1,
    StopSequence = 2,
}

impl From<GufoChatFinishReason> for FinishReason {
    fn from(value: GufoChatFinishReason) -> Self {
        match value {
            GufoChatFinishReason::Stop => Self::Stop,
            GufoChatFinishReason::Length => Self::Length,
            GufoChatFinishReason::StopSequence => Self::StopSequence,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoChatBytes {
    pub data: *const u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoChatToolCall {
    pub id: GufoChatBytes,
    pub name: GufoChatBytes,
    pub arguments: GufoChatBytes,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoChatMetrics {
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
    pub cache_miss_reason: GufoChatBytes,
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
    pub execution_plan: GufoChatBytes,
    pub component_cache_json: GufoChatBytes,
    pub timings_json: GufoChatBytes,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoChatNonStreamingInput {
    pub id: GufoChatBytes,
    pub created: i64,
    pub model: GufoChatBytes,
    pub reasoning: GufoChatBytes,
    pub text: GufoChatBytes,
    pub calls: *const GufoChatToolCall,
    pub calls_len: usize,
    pub logprobs_json: GufoChatBytes,
    pub finish_reason: GufoChatFinishReason,
    pub cache_persist: bool,
    pub metrics: *const GufoChatMetrics,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoChatNamespaceEntry {
    pub name: GufoChatBytes,
    pub ns: GufoChatBytes,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoResponsesConfig {
    pub id: GufoChatBytes,
    pub created_at: i64,
    pub model: GufoChatBytes,
    pub parallel_tool_calls: bool,
    pub tool_choice: GufoChatBytes,
    pub forced_tool_name: GufoChatBytes,
    pub tool_definitions_json: *const GufoChatBytes,
    pub tool_definitions_len: usize,
    pub namespaces: *const GufoChatNamespaceEntry,
    pub namespaces_len: usize,
    pub cache_persist: bool,
    pub has_top_logprobs: bool,
    pub top_logprobs: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoChatToolDelta {
    pub index: usize,
    pub begin: bool,
    pub id: GufoChatBytes,
    pub name: GufoChatBytes,
    pub arguments: GufoChatBytes,
    pub complete: bool,
}

pub type GufoChatWriteFn =
    unsafe extern "C" fn(context: *mut c_void, data: *const u8, len: usize) -> bool;

/// # Safety
/// `bytes.data` must point to `bytes.len` readable bytes (or be null when `len == 0`).
#[allow(unsafe_code)]
unsafe fn read_str<'a>(bytes: GufoChatBytes) -> Option<Cow<'a, str>> {
    // SAFETY: caller upholds `bytes` contract.
    let slice = unsafe { input(bytes.data, bytes.len) }?;
    Some(String::from_utf8_lossy(slice))
}

/// # Safety
/// `write_fn` must be valid to invoke with `write_ctx` for the duration of the call.
#[allow(unsafe_code)]
unsafe fn call_write(write_fn: GufoChatWriteFn, write_ctx: *mut c_void, text: &str) -> bool {
    let ptr = if text.is_empty() {
        std::ptr::null()
    } else {
        text.as_ptr()
    };
    // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
    unsafe { write_fn(write_ctx, ptr, text.len()) }
}

/// # Safety
/// `raw` must point to a valid `GufoChatMetrics` with readable byte slices.
#[allow(unsafe_code)]
unsafe fn read_metrics(raw: *const GufoChatMetrics) -> Option<Metrics> {
    // SAFETY: caller guarantees `raw` is readable or null.
    let m = unsafe { raw.as_ref() }?;
    // SAFETY: caller guarantees readable byte slices.
    let cache_miss_reason = unsafe { read_str(m.cache_miss_reason) }?.into_owned();
    // SAFETY: caller guarantees readable byte slices.
    let execution_plan = unsafe { read_str(m.execution_plan) }?.into_owned();
    // SAFETY: caller guarantees readable byte slices.
    let component_cache_json = unsafe { read_str(m.component_cache_json) }?.into_owned();
    // SAFETY: caller guarantees readable byte slices.
    let timings_json = unsafe { read_str(m.timings_json) }?.into_owned();
    Some(Metrics {
        prompt_tokens: m.prompt_tokens,
        completion_tokens: m.completion_tokens,
        cached_prompt_tokens: m.cached_prompt_tokens,
        reasoning_tokens: m.reasoning_tokens,
        draft_tokens: m.draft_tokens,
        draft_accepted_tokens: m.draft_accepted_tokens,
        draft_rounds: m.draft_rounds,
        prompt_tokens_per_second: m.prompt_tokens_per_second,
        decode_ms: m.decode_ms,
        cache_hit: m.cache_hit,
        cache_miss_reason,
        cache_common_prefix_tokens: m.cache_common_prefix_tokens,
        cache_checkpoint_tokens: m.cache_checkpoint_tokens,
        cache_restore_bytes: m.cache_restore_bytes,
        cache_snapshot_bytes: m.cache_snapshot_bytes,
        cache_disk_queued_bytes: m.cache_disk_queued_bytes,
        cache_restore_ms: m.cache_restore_ms,
        cache_snapshot_ms: m.cache_snapshot_ms,
        cache_disk_enqueue_ms: m.cache_disk_enqueue_ms,
        cache_disk_hit: m.cache_disk_hit,
        prefill_tokens: m.prefill_tokens,
        prefill_chunks: m.prefill_chunks,
        active_decode_prefill_chunks: m.active_decode_prefill_chunks,
        max_prefill_chunk_tokens: m.max_prefill_chunk_tokens,
        max_prefill_layer_groups: m.max_prefill_layer_groups,
        prefill_yield_rounds: m.prefill_yield_rounds,
        queue_depth_at_submit: m.queue_depth_at_submit,
        client_queue_depth_at_submit: m.client_queue_depth_at_submit,
        resident_requests_at_admission: m.resident_requests_at_admission,
        requested_logical_concurrency: m.requested_logical_concurrency,
        physical_execution_width: m.physical_execution_width,
        queue_ms: m.queue_ms,
        shared_prefix_wait_ms: m.shared_prefix_wait_ms,
        prefill_ms: m.prefill_ms,
        ttft_ms: m.ttft_ms,
        mean_inter_token_ms: m.mean_inter_token_ms,
        max_inter_token_ms: m.max_inter_token_ms,
        execution_plan,
        component_cache_json,
        timings_json,
    })
}

/// # Safety
/// `message`, `error_type`, `code` must have readable slices; `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_chat_error_response(
    message: GufoChatBytes,
    error_type: GufoChatBytes,
    code: GufoChatBytes,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        let Some(write_fn) = write_fn else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slices.
        let (Some(msg), Some(ty), Some(cd)) =
            (unsafe { (read_str(message), read_str(error_type), read_str(code)) })
        else {
            return GufoChatResponseStatus::Internal;
        };
        let body = format::error_response_json(&msg, &ty, &cd);
        // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
        if unsafe { call_write(write_fn, write_ctx, &body) } {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `message` and `code` must have readable slices; `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_chat_text_generation_error_response(
    message: GufoChatBytes,
    code: GufoChatBytes,
    has_required_bytes: bool,
    required_bytes: usize,
    has_remaining_bytes: bool,
    remaining_bytes: usize,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        let Some(write_fn) = write_fn else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slices.
        let (Some(msg), Some(cd)) = (unsafe { (read_str(message), read_str(code)) }) else {
            return GufoChatResponseStatus::Internal;
        };
        let body = format::text_generation_error_json(
            &msg,
            &cd,
            has_required_bytes.then_some(required_bytes),
            has_remaining_bytes.then_some(remaining_bytes),
        );
        // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
        if unsafe { call_write(write_fn, write_ctx, &body) } {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `write_fn` must be valid with `write_ctx`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_chat_input_tokens_response(
    input_tokens: usize,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        let Some(write_fn) = write_fn else {
            return GufoChatResponseStatus::Internal;
        };
        let body = format::input_tokens_json(input_tokens);
        // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
        if unsafe { call_write(write_fn, write_ctx, &body) } {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `prompt` must have a readable byte slice; `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_chat_rendered_prompt_response(
    prompt: GufoChatBytes,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        let Some(write_fn) = write_fn else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slice.
        let Some(prompt_str) = (unsafe { read_str(prompt) }) else {
            return GufoChatResponseStatus::Internal;
        };
        let body = format::rendered_prompt_json(&prompt_str);
        // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
        if unsafe { call_write(write_fn, write_ctx, &body) } {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `raw` and all pointers inside it must be readable; `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_chat_non_streaming_response(
    raw: *const GufoChatNonStreamingInput,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        let Some(write_fn) = write_fn else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees `raw` is readable.
        let Some(inp) = (unsafe { raw.as_ref() }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable slices and metrics pointer.
        let (Some(id), Some(model), Some(reasoning), Some(text), Some(logprobs), Some(metrics)) =
            (unsafe {
                (
                    read_str(inp.id),
                    read_str(inp.model),
                    read_str(inp.reasoning),
                    read_str(inp.text),
                    read_str(inp.logprobs_json),
                    read_metrics(inp.metrics),
                )
            })
        else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees `calls` has `calls_len` readable entries.
        let Some(call_slice) = (unsafe { input(inp.calls, inp.calls_len) }) else {
            return GufoChatResponseStatus::Internal;
        };
        let mut calls = Vec::new();
        if calls.try_reserve_exact(call_slice.len()).is_err() {
            return GufoChatResponseStatus::Internal;
        }
        for c in call_slice {
            // SAFETY: caller guarantees readable byte slices.
            let (Some(cid), Some(name), Some(args)) =
                (unsafe { (read_str(c.id), read_str(c.name), read_str(c.arguments)) })
            else {
                return GufoChatResponseStatus::Internal;
            };
            calls.push(ToolCall {
                id: cid.into_owned(),
                name: name.into_owned(),
                arguments: args.into_owned(),
            });
        }
        let body = chat::non_streaming_response(&NonStreamingInput {
            id: &id,
            created: inp.created,
            model: &model,
            reasoning: &reasoning,
            text: &text,
            calls: &calls,
            logprobs_json: &logprobs,
            finish_reason: inp.finish_reason.into(),
            cache_persist: inp.cache_persist,
            metrics: &metrics,
        });
        // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
        if unsafe { call_write(write_fn, write_ctx, &body) } {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `id` and `model` must have readable byte slices.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_chat_stream_new(
    id: GufoChatBytes,
    created: i64,
    model: GufoChatBytes,
) -> *mut GufoChatStream {
    guard(std::ptr::null_mut(), || {
        // SAFETY: caller guarantees readable byte slices.
        let (Some(id_s), Some(model_s)) = (unsafe { (read_str(id), read_str(model)) }) else {
            return std::ptr::null_mut();
        };
        Box::into_raw(Box::new(GufoChatStream {
            inner: ChatStream::new(id_s.into_owned(), created, model_s.into_owned()),
        }))
    })
}

/// # Safety
/// `stream` must be null or a live handle returned by `gufo_chat_stream_new`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_chat_stream_free(stream: *mut GufoChatStream) {
    guard((), || {
        if !stream.is_null() {
            // SAFETY: caller transfers ownership of a live handle.
            drop(unsafe { Box::from_raw(stream) });
        }
    });
}

/// # Safety
/// `stream` must be live; `write_fn` must be valid with `write_ctx`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_chat_stream_role(
    stream: *mut GufoChatStream,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `stream`.
        let Some(stream) = (unsafe { stream.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        let Some(write_fn) = write_fn else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
        let mut write = |s: &str| unsafe { call_write(write_fn, write_ctx, s) };
        if stream.inner.emit_role(&mut write) {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `stream` must be live; byte slices and `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_chat_stream_text_delta(
    stream: *mut GufoChatStream,
    reasoning: bool,
    text: GufoChatBytes,
    logprobs_json: GufoChatBytes,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `stream`.
        let Some(stream) = (unsafe { stream.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        let Some(write_fn) = write_fn else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slices.
        let (Some(text_s), Some(lp_s)) = (unsafe { (read_str(text), read_str(logprobs_json)) })
        else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
        let mut write = |s: &str| unsafe { call_write(write_fn, write_ctx, s) };
        if stream
            .inner
            .emit_text_delta(reasoning, &text_s, &lp_s, &mut write)
        {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `stream` must be live; byte slices and `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_chat_stream_empty_delta(
    stream: *mut GufoChatStream,
    logprobs_json: GufoChatBytes,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `stream`.
        let Some(stream) = (unsafe { stream.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        let Some(write_fn) = write_fn else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slice.
        let Some(lp_s) = (unsafe { read_str(logprobs_json) }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
        let mut write = |s: &str| unsafe { call_write(write_fn, write_ctx, s) };
        if stream.inner.emit_empty_delta(&lp_s, &mut write) {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `stream` must be live; `delta`, byte slices, and `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_chat_stream_tool_delta(
    stream: *mut GufoChatStream,
    delta: *const GufoChatToolDelta,
    logprobs_json: GufoChatBytes,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `stream`.
        let Some(stream) = (unsafe { stream.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees `delta` is readable.
        let Some(d) = (unsafe { delta.as_ref() }) else {
            return GufoChatResponseStatus::Internal;
        };
        let Some(write_fn) = write_fn else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slices.
        let (Some(id), Some(name), Some(args), Some(lp)) = (unsafe {
            (
                read_str(d.id),
                read_str(d.name),
                read_str(d.arguments),
                read_str(logprobs_json),
            )
        }) else {
            return GufoChatResponseStatus::Internal;
        };
        let td = ToolDelta {
            index: d.index,
            begin: d.begin,
            id: &id,
            name: &name,
            arguments: &args,
            complete: d.complete,
        };
        // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
        let mut write = |s: &str| unsafe { call_write(write_fn, write_ctx, s) };
        if stream.inner.emit_tool_delta(&td, &lp, &mut write) {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `stream` must be live; byte slice and `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_chat_stream_progress(
    stream: *mut GufoChatStream,
    prompt_progress_json: GufoChatBytes,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `stream`.
        let Some(stream) = (unsafe { stream.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        let Some(write_fn) = write_fn else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slice.
        let Some(pp) = (unsafe { read_str(prompt_progress_json) }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
        let mut write = |s: &str| unsafe { call_write(write_fn, write_ctx, s) };
        if stream.inner.emit_progress(&pp, &mut write) {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `stream` must be live; `metrics` and `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_chat_stream_finish(
    stream: *mut GufoChatStream,
    finish_reason: GufoChatFinishReason,
    has_tool_calls: bool,
    include_usage: bool,
    cache_persist: bool,
    metrics: *const GufoChatMetrics,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `stream`.
        let Some(stream) = (unsafe { stream.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        let Some(write_fn) = write_fn else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees `metrics` is readable.
        let Some(m) = (unsafe { read_metrics(metrics) }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
        let mut write = |s: &str| unsafe { call_write(write_fn, write_ctx, s) };
        if stream.inner.finish(
            finish_reason.into(),
            has_tool_calls,
            include_usage,
            cache_persist,
            &m,
            &mut write,
        ) {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `message`, `code`, and `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_chat_stream_error(
    message: GufoChatBytes,
    code: GufoChatBytes,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        let Some(write_fn) = write_fn else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slices.
        let (Some(msg), Some(cd)) = (unsafe { (read_str(message), read_str(code)) }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
        let mut write = |s: &str| unsafe { call_write(write_fn, write_ctx, s) };
        if chat::stream_error_sse(&msg, &cd, &mut write) {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `config` and all slices inside it must be readable.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_responses_output_new(
    config: *const GufoResponsesConfig,
    streaming: bool,
) -> *mut GufoResponsesOutput {
    guard(std::ptr::null_mut(), || {
        // SAFETY: caller guarantees `config` is readable.
        let Some(c) = (unsafe { config.as_ref() }) else {
            return std::ptr::null_mut();
        };
        // SAFETY: caller guarantees readable byte slices.
        let (Some(id), Some(model), Some(tc), Some(forced)) = (unsafe {
            (
                read_str(c.id),
                read_str(c.model),
                read_str(c.tool_choice),
                read_str(c.forced_tool_name),
            )
        }) else {
            return std::ptr::null_mut();
        };
        // SAFETY: caller guarantees readable tool definitions slice.
        let Some(def_slice) = (unsafe { input(c.tool_definitions_json, c.tool_definitions_len) })
        else {
            return std::ptr::null_mut();
        };
        let mut tool_definitions_json = Vec::new();
        if tool_definitions_json
            .try_reserve_exact(def_slice.len())
            .is_err()
        {
            return std::ptr::null_mut();
        }
        for d in def_slice {
            // SAFETY: caller guarantees readable byte slice.
            let Some(s) = (unsafe { read_str(*d) }) else {
                return std::ptr::null_mut();
            };
            tool_definitions_json.push(s.into_owned());
        }
        // SAFETY: caller guarantees readable namespaces slice.
        let Some(ns_slice) = (unsafe { input(c.namespaces, c.namespaces_len) }) else {
            return std::ptr::null_mut();
        };
        let mut namespaces = BTreeMap::new();
        for entry in ns_slice {
            // SAFETY: caller guarantees readable byte slices.
            let (Some(name), Some(ns)) = (unsafe { (read_str(entry.name), read_str(entry.ns)) })
            else {
                return std::ptr::null_mut();
            };
            namespaces.insert(name.into_owned(), ns.into_owned());
        }
        let cfg = ResponsesConfig {
            id: id.into_owned(),
            created_at: c.created_at,
            model: model.into_owned(),
            parallel_tool_calls: c.parallel_tool_calls,
            tool_choice: tc.into_owned(),
            forced_tool_name: forced.into_owned(),
            tool_definitions_json,
            namespaces,
            cache_persist: c.cache_persist,
            top_logprobs: c.has_top_logprobs.then_some(c.top_logprobs),
        };
        Box::into_raw(Box::new(GufoResponsesOutput {
            inner: ResponsesOutput::new(cfg, streaming),
        }))
    })
}

/// # Safety
/// `output` must be null or a live handle returned by `gufo_responses_output_new`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_responses_output_free(output: *mut GufoResponsesOutput) {
    guard((), || {
        if !output.is_null() {
            // SAFETY: caller transfers ownership of a live handle.
            drop(unsafe { Box::from_raw(output) });
        }
    });
}

/// # Safety
/// `output` must be live; `write_fn` must be valid if streaming.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_responses_output_begin(
    output: *mut GufoResponsesOutput,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `output`.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        let mut write = |s: &str| match write_fn {
            // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
            Some(f) => unsafe { call_write(f, write_ctx, s) },
            None => true,
        };
        if output.inner.begin(&mut write) {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `output` must be live; byte slice and `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_responses_output_progress(
    output: *mut GufoResponsesOutput,
    prompt_progress_json: GufoChatBytes,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `output`.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slice.
        let Some(pp) = (unsafe { read_str(prompt_progress_json) }) else {
            return GufoChatResponseStatus::Internal;
        };
        let mut write = |s: &str| match write_fn {
            // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
            Some(f) => unsafe { call_write(f, write_ctx, s) },
            None => true,
        };
        if output.inner.progress(&pp, &mut write) {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `output` must be live; byte slices and `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_responses_output_append(
    output: *mut GufoResponsesOutput,
    text: GufoChatBytes,
    reasoning: bool,
    item_id: GufoChatBytes,
    logprobs_json: GufoChatBytes,
    stream_logprobs_json: GufoChatBytes,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `output`.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slices.
        let (Some(text_s), Some(id_s), Some(lp_s), Some(slp_s)) = (unsafe {
            (
                read_str(text),
                read_str(item_id),
                read_str(logprobs_json),
                read_str(stream_logprobs_json),
            )
        }) else {
            return GufoChatResponseStatus::Internal;
        };
        let mut write = |s: &str| match write_fn {
            // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
            Some(f) => unsafe { call_write(f, write_ctx, s) },
            None => true,
        };
        if output
            .inner
            .append(&text_s, reasoning, &id_s, &lp_s, &slp_s, &mut write)
        {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `output` must be live; `delta`, `fc_id`, and `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_responses_output_tool(
    output: *mut GufoResponsesOutput,
    delta: *const GufoChatToolDelta,
    fc_id: GufoChatBytes,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `output`.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees `delta` is readable.
        let Some(d) = (unsafe { delta.as_ref() }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slices.
        let (Some(id), Some(name), Some(args), Some(fc)) = (unsafe {
            (
                read_str(d.id),
                read_str(d.name),
                read_str(d.arguments),
                read_str(fc_id),
            )
        }) else {
            return GufoChatResponseStatus::Internal;
        };
        let td = ToolDelta {
            index: d.index,
            begin: d.begin,
            id: &id,
            name: &name,
            arguments: &args,
            complete: d.complete,
        };
        let mut write = |s: &str| match write_fn {
            // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
            Some(f) => unsafe { call_write(f, write_ctx, s) },
            None => true,
        };
        if output.inner.tool(&td, &fc, &mut write) {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `output` must be live; `metrics` and callbacks must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_responses_output_complete(
    output: *mut GufoResponsesOutput,
    finish_reason: GufoChatFinishReason,
    metrics: *const GufoChatMetrics,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
    body_fn: Option<GufoChatWriteFn>,
    body_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `output`.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees `metrics` is readable.
        let Some(m) = (unsafe { read_metrics(metrics) }) else {
            return GufoChatResponseStatus::Internal;
        };
        let mut write = |s: &str| match write_fn {
            // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
            Some(f) => unsafe { call_write(f, write_ctx, s) },
            None => true,
        };
        let (body, ok) = output.inner.complete(finish_reason.into(), &m, &mut write);
        if let Some(bf) = body_fn {
            // SAFETY: caller guarantees `body_fn` and `body_ctx` are valid.
            if !unsafe { call_write(bf, body_ctx, &body) } {
                return GufoChatResponseStatus::Closed;
            }
        }
        if ok {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `output` must be live; `message` and `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_responses_output_fail(
    output: *mut GufoResponsesOutput,
    message: GufoChatBytes,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `output`.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slice.
        let Some(msg) = (unsafe { read_str(message) }) else {
            return GufoChatResponseStatus::Internal;
        };
        let mut write = |s: &str| match write_fn {
            // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
            Some(f) => unsafe { call_write(f, write_ctx, s) },
            None => true,
        };
        if output.inner.fail(&msg, &mut write) {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `id` and `model` must have readable byte slices.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_anthropic_output_new(
    id: GufoChatBytes,
    model: GufoChatBytes,
    streaming: bool,
) -> *mut GufoAnthropicOutput {
    guard(std::ptr::null_mut(), || {
        // SAFETY: caller guarantees readable byte slices.
        let (Some(id_s), Some(model_s)) = (unsafe { (read_str(id), read_str(model)) }) else {
            return std::ptr::null_mut();
        };
        Box::into_raw(Box::new(GufoAnthropicOutput {
            inner: AnthropicOutput::new(id_s.into_owned(), model_s.into_owned(), streaming),
        }))
    })
}

/// # Safety
/// `output` must be null or a live handle returned by `gufo_anthropic_output_new`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_anthropic_output_free(output: *mut GufoAnthropicOutput) {
    guard((), || {
        if !output.is_null() {
            // SAFETY: caller transfers ownership of a live handle.
            drop(unsafe { Box::from_raw(output) });
        }
    });
}

/// # Safety
/// `output` must be live; `write_fn` must be valid if streaming.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_anthropic_output_begin(
    output: *mut GufoAnthropicOutput,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `output`.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        let mut write = |s: &str| match write_fn {
            // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
            Some(f) => unsafe { call_write(f, write_ctx, s) },
            None => true,
        };
        if output.inner.begin(&mut write) {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `output` must be live; `text` and `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_anthropic_output_append(
    output: *mut GufoAnthropicOutput,
    text: GufoChatBytes,
    reasoning: bool,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `output`.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slice.
        let Some(text_s) = (unsafe { read_str(text) }) else {
            return GufoChatResponseStatus::Internal;
        };
        let mut write = |s: &str| match write_fn {
            // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
            Some(f) => unsafe { call_write(f, write_ctx, s) },
            None => true,
        };
        if output.inner.append(&text_s, reasoning, &mut write) {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `output` must be live; `delta` and `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_anthropic_output_tool(
    output: *mut GufoAnthropicOutput,
    delta: *const GufoChatToolDelta,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
    error_fn: Option<GufoChatWriteFn>,
    error_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `output`.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees `delta` is readable.
        let Some(d) = (unsafe { delta.as_ref() }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slices.
        let (Some(id), Some(name), Some(args)) =
            (unsafe { (read_str(d.id), read_str(d.name), read_str(d.arguments)) })
        else {
            return GufoChatResponseStatus::Internal;
        };
        let td = ToolDelta {
            index: d.index,
            begin: d.begin,
            id: &id,
            name: &name,
            arguments: &args,
            complete: d.complete,
        };
        let mut write = |s: &str| match write_fn {
            // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
            Some(f) => unsafe { call_write(f, write_ctx, s) },
            None => true,
        };
        if output.inner.tool(&td, &mut write) {
            return GufoChatResponseStatus::Ok;
        }
        let Some(reason) = output.inner.take_invalid_arguments() else {
            return GufoChatResponseStatus::Closed;
        };
        if let Some(f) = error_fn {
            // SAFETY: caller guarantees `error_fn` and `error_ctx` are valid.
            let _ = unsafe { call_write(f, error_ctx, &reason) };
        }
        GufoChatResponseStatus::Invalid
    })
}

/// # Safety
/// `output` must be live; `stop_sequence`, `metrics`, and callbacks must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_anthropic_output_complete(
    output: *mut GufoAnthropicOutput,
    finish_reason: GufoChatFinishReason,
    stop_sequence: GufoChatBytes,
    metrics: *const GufoChatMetrics,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
    body_fn: Option<GufoChatWriteFn>,
    body_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `output`.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable `stop_sequence` and `metrics`.
        let (Some(stop_seq), Some(m)) =
            (unsafe { (read_str(stop_sequence), read_metrics(metrics)) })
        else {
            return GufoChatResponseStatus::Internal;
        };
        let mut write = |s: &str| match write_fn {
            // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
            Some(f) => unsafe { call_write(f, write_ctx, s) },
            None => true,
        };
        let (body, ok) = output
            .inner
            .complete(finish_reason.into(), &stop_seq, &m, &mut write);
        if let Some(bf) = body_fn {
            // SAFETY: caller guarantees `body_fn` and `body_ctx` are valid.
            if !unsafe { call_write(bf, body_ctx, &body) } {
                return GufoChatResponseStatus::Closed;
            }
        }
        if ok {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

/// # Safety
/// `output` must be live; `message` and `write_fn` must be valid.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_anthropic_output_fail(
    output: *mut GufoAnthropicOutput,
    message: GufoChatBytes,
    write_fn: Option<GufoChatWriteFn>,
    write_ctx: *mut c_void,
) -> GufoChatResponseStatus {
    guard(GufoChatResponseStatus::Internal, || {
        // SAFETY: caller guarantees exclusive access to live `output`.
        let Some(output) = (unsafe { output.as_mut() }) else {
            return GufoChatResponseStatus::Internal;
        };
        // SAFETY: caller guarantees readable byte slice.
        let Some(msg) = (unsafe { read_str(message) }) else {
            return GufoChatResponseStatus::Internal;
        };
        let mut write = |s: &str| match write_fn {
            // SAFETY: caller guarantees `write_fn` and `write_ctx` are valid.
            Some(f) => unsafe { call_write(f, write_ctx, s) },
            None => true,
        };
        if output.inner.fail(&msg, &mut write) {
            GufoChatResponseStatus::Ok
        } else {
            GufoChatResponseStatus::Closed
        }
    })
}

//! C ABI for parsing Hugging Face `tokenizer.json` files.
#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod parser;

use std::ffi::c_void;

use gufo_ffi::{guard, input};
use parser::{ParseError, TokenizerJson};

pub struct GufoTokenizerJson {
    inner: TokenizerJson,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoTokenizerJsonStatus {
    Ok = 0,
    Invalid = 1,
    Internal = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoTokenizerJsonBytes {
    pub data: *const u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoTokenizerJsonVocabEntry {
    pub token: GufoTokenizerJsonBytes,
    pub id: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoTokenizerJsonMergeEntry {
    pub left: GufoTokenizerJsonBytes,
    pub right: GufoTokenizerJsonBytes,
    pub was_string: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoTokenizerJsonAddedToken {
    pub content: GufoTokenizerJsonBytes,
    pub id: u32,
    pub has_single_word: bool,
    pub single_word: bool,
    pub has_lstrip: bool,
    pub lstrip: bool,
    pub has_rstrip: bool,
    pub rstrip: bool,
    pub has_normalized: bool,
    pub normalized: bool,
    pub flags_valid: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoTokenizerJsonMeta {
    pub truncation_is_null: bool,
    pub padding_is_null: bool,

    pub has_model: bool,
    pub has_vocab: bool,
    pub has_merges: bool,
    pub has_added_tokens: bool,

    pub model_type: GufoTokenizerJsonBytes,
    pub has_dropout: bool,
    pub dropout_is_null: bool,
    pub continuing_subword_prefix: GufoTokenizerJsonBytes,
    pub end_of_word_suffix: GufoTokenizerJsonBytes,
    pub has_byte_fallback: bool,
    pub byte_fallback: bool,
    pub has_ignore_merges: bool,
    pub ignore_merges: bool,
    pub has_unk_token: bool,
    pub unk_token_is_null: bool,
    pub unk_token: GufoTokenizerJsonBytes,

    pub has_normalizer: bool,
    pub normalizer_type: GufoTokenizerJsonBytes,
    pub normalizer_pattern_string: GufoTokenizerJsonBytes,
    pub normalizer_content: GufoTokenizerJsonBytes,

    pub has_pre_tokenizer: bool,
    pub pre_tokenizer_type: GufoTokenizerJsonBytes,
    pub pre_tokenizer_pattern_string: GufoTokenizerJsonBytes,
    pub pre_tokenizer_behavior: GufoTokenizerJsonBytes,
    pub has_pre_tokenizer_invert: bool,
    pub pre_tokenizer_invert: bool,

    pub pre_tokenizer_sub_count: usize,
    pub split_type: GufoTokenizerJsonBytes,
    pub split_behavior: GufoTokenizerJsonBytes,
    pub has_split_invert: bool,
    pub split_invert: bool,
    pub split_regex: GufoTokenizerJsonBytes,
    pub byte_level_type: GufoTokenizerJsonBytes,
    pub has_add_prefix_space: bool,
    pub add_prefix_space: bool,
    pub has_use_regex: bool,
    pub use_regex: bool,

    pub has_post_processor: bool,
    pub post_processor_valid: bool,
    pub post_processor_error: GufoTokenizerJsonBytes,
}

pub type GufoTokenizerJsonErrorFn =
    unsafe extern "C" fn(ctx: *mut c_void, data: *const u8, len: usize);

fn as_bytes(s: &str) -> GufoTokenizerJsonBytes {
    GufoTokenizerJsonBytes {
        data: if s.is_empty() {
            std::ptr::null()
        } else {
            s.as_ptr()
        },
        len: s.len(),
    }
}

/// Parses a `tokenizer.json` byte buffer into an opaque `GufoTokenizerJson` handle.
///
/// # Safety
/// `data` must point to `len` readable bytes (or null at length 0); `out` must
/// be writable; `on_error` (if non-null) must be valid with `error_ctx`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_tokenizer_json_parse(
    data: *const u8,
    len: usize,
    max_token_id: u32,
    out: *mut *mut GufoTokenizerJson,
    on_error: Option<GufoTokenizerJsonErrorFn>,
    error_ctx: *mut c_void,
) -> GufoTokenizerJsonStatus {
    guard(GufoTokenizerJsonStatus::Internal, || {
        if out.is_null() {
            return GufoTokenizerJsonStatus::Internal;
        }
        // SAFETY: caller guarantees `out` is a valid writable pointer.
        unsafe { out.write(std::ptr::null_mut()) };
        // SAFETY: caller guarantees `data` has `len` readable bytes.
        let Some(slice) = (unsafe { input(data, len) }) else {
            return GufoTokenizerJsonStatus::Internal;
        };
        match parser::parse(slice, max_token_id) {
            Ok(inner) => {
                // SAFETY: caller guarantees `out` is a valid writable pointer.
                unsafe { out.write(Box::into_raw(Box::new(GufoTokenizerJson { inner }))) };
                GufoTokenizerJsonStatus::Ok
            }
            Err(err) => {
                let (status, msg) = match err {
                    ParseError::Invalid(m) => (GufoTokenizerJsonStatus::Invalid, m),
                    ParseError::Internal(m) => (GufoTokenizerJsonStatus::Internal, m),
                };
                if let Some(cb) = on_error {
                    let b = as_bytes(&msg);
                    // SAFETY: caller guarantees `on_error` and `error_ctx` are valid.
                    unsafe { cb(error_ctx, b.data, b.len) };
                }
                status
            }
        }
    })
}

/// Frees a `GufoTokenizerJson` handle. Null is a no-op.
///
/// # Safety
/// `handle` must be null or a live pointer returned by `gufo_tokenizer_json_parse`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_tokenizer_json_free(handle: *mut GufoTokenizerJson) {
    guard((), || {
        if !handle.is_null() {
            // SAFETY: caller transfers ownership of a live handle.
            drop(unsafe { Box::from_raw(handle) });
        }
    });
}

/// Fills `out` with borrowed metadata views valid for the lifetime of `handle`.
///
/// # Safety
/// `handle` must be live and `out` must be writable.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_tokenizer_json_meta(
    handle: *const GufoTokenizerJson,
    out: *mut GufoTokenizerJsonMeta,
) -> GufoTokenizerJsonStatus {
    guard(GufoTokenizerJsonStatus::Internal, || {
        if out.is_null() {
            return GufoTokenizerJsonStatus::Internal;
        }
        // SAFETY: caller guarantees `handle` is live.
        let Some(h) = (unsafe { handle.as_ref() }) else {
            return GufoTokenizerJsonStatus::Internal;
        };
        let m = &h.inner.meta;
        let value = GufoTokenizerJsonMeta {
            truncation_is_null: m.truncation_is_null,
            padding_is_null: m.padding_is_null,
            has_model: m.has_model,
            has_vocab: m.has_vocab,
            has_merges: m.has_merges,
            has_added_tokens: m.has_added_tokens,
            model_type: as_bytes(&m.model_type),
            has_dropout: m.has_dropout,
            dropout_is_null: m.dropout_is_null,
            continuing_subword_prefix: as_bytes(&m.continuing_subword_prefix),
            end_of_word_suffix: as_bytes(&m.end_of_word_suffix),
            has_byte_fallback: m.has_byte_fallback,
            byte_fallback: m.byte_fallback,
            has_ignore_merges: m.has_ignore_merges,
            ignore_merges: m.ignore_merges,
            has_unk_token: m.has_unk_token,
            unk_token_is_null: m.unk_token_is_null,
            unk_token: as_bytes(&m.unk_token),
            has_normalizer: m.has_normalizer,
            normalizer_type: as_bytes(&m.normalizer_type),
            normalizer_pattern_string: as_bytes(&m.normalizer_pattern_string),
            normalizer_content: as_bytes(&m.normalizer_content),
            has_pre_tokenizer: m.has_pre_tokenizer,
            pre_tokenizer_type: as_bytes(&m.pre_tokenizer_type),
            pre_tokenizer_pattern_string: as_bytes(&m.pre_tokenizer_pattern_string),
            pre_tokenizer_behavior: as_bytes(&m.pre_tokenizer_behavior),
            has_pre_tokenizer_invert: m.has_pre_tokenizer_invert,
            pre_tokenizer_invert: m.pre_tokenizer_invert,
            pre_tokenizer_sub_count: m.pre_tokenizer_sub_count,
            split_type: as_bytes(&m.split_type),
            split_behavior: as_bytes(&m.split_behavior),
            has_split_invert: m.has_split_invert,
            split_invert: m.split_invert,
            split_regex: as_bytes(&m.split_regex),
            byte_level_type: as_bytes(&m.byte_level_type),
            has_add_prefix_space: m.has_add_prefix_space,
            add_prefix_space: m.add_prefix_space,
            has_use_regex: m.has_use_regex,
            use_regex: m.use_regex,
            has_post_processor: m.has_post_processor,
            post_processor_valid: m.post_processor_valid,
            post_processor_error: as_bytes(&m.post_processor_error),
        };
        // SAFETY: caller guarantees `out` is a valid writable pointer.
        unsafe { out.write(value) };
        GufoTokenizerJsonStatus::Ok
    })
}

/// # Safety
/// `handle` must be null or a live handle.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_tokenizer_json_vocab_len(handle: *const GufoTokenizerJson) -> usize {
    guard(0, || {
        // SAFETY: caller guarantees `handle` is live or null.
        unsafe { handle.as_ref() }.map_or(0, |h| h.inner.vocab.len())
    })
}

/// # Safety
/// `handle` must be live and `out` must be writable.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_tokenizer_json_vocab_at(
    handle: *const GufoTokenizerJson,
    index: usize,
    out: *mut GufoTokenizerJsonVocabEntry,
) -> GufoTokenizerJsonStatus {
    guard(GufoTokenizerJsonStatus::Internal, || {
        if out.is_null() {
            return GufoTokenizerJsonStatus::Internal;
        }
        // SAFETY: caller guarantees `handle` is live.
        let Some(h) = (unsafe { handle.as_ref() }) else {
            return GufoTokenizerJsonStatus::Internal;
        };
        let Some(entry) = h.inner.vocab.get(index) else {
            return GufoTokenizerJsonStatus::Invalid;
        };
        // SAFETY: caller guarantees `out` is a valid writable pointer.
        unsafe {
            out.write(GufoTokenizerJsonVocabEntry {
                token: as_bytes(&entry.token),
                id: entry.id,
            });
        }
        GufoTokenizerJsonStatus::Ok
    })
}

/// # Safety
/// `handle` must be null or a live handle.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_tokenizer_json_merges_len(handle: *const GufoTokenizerJson) -> usize {
    guard(0, || {
        // SAFETY: caller guarantees `handle` is live or null.
        unsafe { handle.as_ref() }.map_or(0, |h| h.inner.merges.len())
    })
}

/// # Safety
/// `handle` must be live and `out` must be writable.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_tokenizer_json_merge_at(
    handle: *const GufoTokenizerJson,
    index: usize,
    out: *mut GufoTokenizerJsonMergeEntry,
) -> GufoTokenizerJsonStatus {
    guard(GufoTokenizerJsonStatus::Internal, || {
        if out.is_null() {
            return GufoTokenizerJsonStatus::Internal;
        }
        // SAFETY: caller guarantees `handle` is live.
        let Some(h) = (unsafe { handle.as_ref() }) else {
            return GufoTokenizerJsonStatus::Internal;
        };
        let Some(entry) = h.inner.merges.get(index) else {
            return GufoTokenizerJsonStatus::Invalid;
        };
        // SAFETY: caller guarantees `out` is a valid writable pointer.
        unsafe {
            out.write(GufoTokenizerJsonMergeEntry {
                left: as_bytes(&entry.left),
                right: as_bytes(&entry.right),
                was_string: entry.was_string,
            });
        }
        GufoTokenizerJsonStatus::Ok
    })
}

/// # Safety
/// `handle` must be null or a live handle.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_tokenizer_json_added_len(handle: *const GufoTokenizerJson) -> usize {
    guard(0, || {
        // SAFETY: caller guarantees `handle` is live or null.
        unsafe { handle.as_ref() }.map_or(0, |h| h.inner.added_tokens.len())
    })
}

/// # Safety
/// `handle` must be live and `out` must be writable.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_tokenizer_json_added_at(
    handle: *const GufoTokenizerJson,
    index: usize,
    out: *mut GufoTokenizerJsonAddedToken,
) -> GufoTokenizerJsonStatus {
    guard(GufoTokenizerJsonStatus::Internal, || {
        if out.is_null() {
            return GufoTokenizerJsonStatus::Internal;
        }
        // SAFETY: caller guarantees `handle` is live.
        let Some(h) = (unsafe { handle.as_ref() }) else {
            return GufoTokenizerJsonStatus::Internal;
        };
        let Some(entry) = h.inner.added_tokens.get(index) else {
            return GufoTokenizerJsonStatus::Invalid;
        };
        // SAFETY: caller guarantees `out` is a valid writable pointer.
        unsafe {
            out.write(GufoTokenizerJsonAddedToken {
                content: as_bytes(&entry.content),
                id: entry.id,
                has_single_word: entry.has_single_word,
                single_word: entry.single_word,
                has_lstrip: entry.has_lstrip,
                lstrip: entry.lstrip,
                has_rstrip: entry.has_rstrip,
                rstrip: entry.rstrip,
                has_normalized: entry.has_normalized,
                normalized: entry.normalized,
                flags_valid: entry.flags_valid,
            });
        }
        GufoTokenizerJsonStatus::Ok
    })
}

/// # Safety
/// `handle` must be live; `data` and `len` must be writable.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_tokenizer_json_prefix_ids(
    handle: *const GufoTokenizerJson,
    data: *mut *const u32,
    len: *mut usize,
) -> GufoTokenizerJsonStatus {
    guard(GufoTokenizerJsonStatus::Internal, || {
        if data.is_null() || len.is_null() {
            return GufoTokenizerJsonStatus::Internal;
        }
        // SAFETY: caller guarantees `handle` is live.
        let Some(h) = (unsafe { handle.as_ref() }) else {
            return GufoTokenizerJsonStatus::Internal;
        };
        let ptr = if h.inner.prefix_ids.is_empty() {
            std::ptr::null()
        } else {
            h.inner.prefix_ids.as_ptr()
        };
        // SAFETY: caller guarantees `data` and `len` are valid writable pointers.
        unsafe {
            len.write(h.inner.prefix_ids.len());
            data.write(ptr);
        }
        GufoTokenizerJsonStatus::Ok
    })
}

/// # Safety
/// `handle` must be live; `data` and `len` must be writable.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_tokenizer_json_suffix_ids(
    handle: *const GufoTokenizerJson,
    data: *mut *const u32,
    len: *mut usize,
) -> GufoTokenizerJsonStatus {
    guard(GufoTokenizerJsonStatus::Internal, || {
        if data.is_null() || len.is_null() {
            return GufoTokenizerJsonStatus::Internal;
        }
        // SAFETY: caller guarantees `handle` is live.
        let Some(h) = (unsafe { handle.as_ref() }) else {
            return GufoTokenizerJsonStatus::Internal;
        };
        let ptr = if h.inner.suffix_ids.is_empty() {
            std::ptr::null()
        } else {
            h.inner.suffix_ids.as_ptr()
        };
        // SAFETY: caller guarantees `data` and `len` are valid writable pointers.
        unsafe {
            len.write(h.inner.suffix_ids.len());
            data.write(ptr);
        }
        GufoTokenizerJsonStatus::Ok
    })
}

//! The C interface of the disk manifest codec and its CRC-64. `DiskStore`
//! (src/cache/disk.cpp) encodes, decodes and checksums with it; files,
//! directories, recovery, durability and retention stay in C++.
#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

pub mod crc;
mod manifest;

use gufo_ffi::guard;
use std::ffi::{c_int, c_void};

use manifest::{Component, Failure, Header, Payload, Sink, Stopped};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoCacheStatus {
    Complete = 0,
    Rejected = 1,
    NoMemory = 2,
    Stopped = 3,
    Internal = 4,
}

impl From<Failure> for GufoCacheStatus {
    fn from(failure: Failure) -> Self {
        match failure {
            Failure::Rejected => Self::Rejected,
            Failure::NoMemory => Self::NoMemory,
            Failure::Stopped => Self::Stopped,
        }
    }
}

fn status(result: Result<(), Failure>) -> GufoCacheStatus {
    result.map_or_else(Into::into, |()| GufoCacheStatus::Complete)
}

pub type GufoCachePayload = Payload;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoCacheComponent {
    pub id: u32,
    pub layout_version: u32,
    pub kind: u8,
    pub row_bytes: u64,
    pub rows_per_chunk: u64,
    pub state_bytes: u64,
    pub window_rows: u64,
    pub position_id: u32,
    pub valid_rows: u64,
    pub chunks: *const GufoCachePayload,
    pub chunk_count: usize,
    pub has_tail: c_int,
    pub tail: GufoCachePayload,
    pub has_private_state: c_int,
    pub private_state: GufoCachePayload,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoCacheManifest {
    pub checkpoint: u64,
    pub lineage: u64,
    pub compatibility: [u8; 32],
    pub purpose: u8,
    pub rank: u32,
    pub input: *const u8,
    pub input_len: usize,
    pub tokens: *const u32,
    pub token_count: usize,
    pub components: *const GufoCacheComponent,
    pub component_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GufoCacheManifestPlan {
    pub input_bytes: usize,
    pub tokens: usize,
    pub components: usize,
    pub chunks: usize,
    pub references: usize,
}

pub type GufoCacheBeginFn =
    unsafe extern "C" fn(context: *mut c_void, manifest: *const GufoCacheManifest) -> c_int;
pub type GufoCacheTokensFn =
    unsafe extern "C" fn(context: *mut c_void, tokens: *const u32, count: usize) -> c_int;
pub type GufoCacheComponentFn =
    unsafe extern "C" fn(context: *mut c_void, component: *const GufoCacheComponent) -> c_int;
pub type GufoCachePayloadsFn = unsafe extern "C" fn(
    context: *mut c_void,
    payloads: *const GufoCachePayload,
    count: usize,
) -> c_int;
pub type GufoCacheBufferFn = unsafe extern "C" fn(context: *mut c_void, len: usize) -> *mut u8;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoCacheManifestSink {
    pub context: *mut c_void,
    pub begin: Option<GufoCacheBeginFn>,
    pub tokens: Option<GufoCacheTokensFn>,
    pub component: Option<GufoCacheComponentFn>,
    pub chunks: Option<GufoCachePayloadsFn>,
}

fn optional(present: c_int, payload: GufoCachePayload) -> Option<Payload> {
    (present != 0).then_some(payload)
}

fn present(payload: &Option<Payload>) -> (c_int, GufoCachePayload) {
    (payload.is_some().into(), payload.unwrap_or_default())
}

fn stopped(result: c_int) -> Result<(), Stopped> {
    match result {
        0 => Ok(()),
        _ => Err(Stopped),
    }
}

/// The caller's callbacks, which it allows to be called with `context` until
/// the decoding returns.
struct Callbacks {
    context: *mut c_void,
    begin: GufoCacheBeginFn,
    tokens: GufoCacheTokensFn,
    component: GufoCacheComponentFn,
    chunks: GufoCachePayloadsFn,
}

impl Sink for Callbacks {
    #[allow(unsafe_code)]
    fn begin(
        &mut self,
        header: &Header,
        input: &[u8],
        tokens: usize,
        components: usize,
    ) -> Result<(), Stopped> {
        let manifest = GufoCacheManifest {
            checkpoint: header.checkpoint,
            lineage: header.lineage,
            compatibility: header.compatibility,
            purpose: header.purpose,
            rank: header.rank,
            input: input.as_ptr(),
            input_len: input.len(),
            tokens: std::ptr::null(),
            token_count: tokens,
            components: std::ptr::null(),
            component_count: components,
        };
        // SAFETY: the caller allows this call (see `Callbacks`); `manifest`
        // lives until it returns, and `input` points into the caller's bytes.
        stopped(unsafe { (self.begin)(self.context, &manifest) })
    }

    #[allow(unsafe_code)]
    fn tokens(&mut self, tokens: &[u32]) -> Result<(), Stopped> {
        // SAFETY: the caller allows this call (see `Callbacks`); the block
        // lives until it returns.
        stopped(unsafe { (self.tokens)(self.context, tokens.as_ptr(), tokens.len()) })
    }

    #[allow(unsafe_code)]
    fn component(&mut self, c: &Component<usize>) -> Result<(), Stopped> {
        let (has_tail, tail) = present(&c.tail);
        let (has_private_state, private_state) = present(&c.private_state);
        let component = GufoCacheComponent {
            id: c.id,
            layout_version: c.layout_version,
            kind: c.kind,
            row_bytes: c.row_bytes,
            rows_per_chunk: c.rows_per_chunk,
            state_bytes: c.state_bytes,
            window_rows: c.window_rows,
            position_id: c.position_id,
            valid_rows: c.valid_rows,
            chunks: std::ptr::null(),
            chunk_count: c.chunks,
            has_tail,
            tail,
            has_private_state,
            private_state,
        };
        // SAFETY: the caller allows this call (see `Callbacks`); `component`
        // lives until it returns.
        stopped(unsafe { (self.component)(self.context, &component) })
    }

    #[allow(unsafe_code)]
    fn chunks(&mut self, chunks: &[Payload]) -> Result<(), Stopped> {
        // SAFETY: the caller allows this call (see `Callbacks`); the block
        // lives until it returns.
        stopped(unsafe { (self.chunks)(self.context, chunks.as_ptr(), chunks.len()) })
    }
}

/// Writes the CRC-64/ECMA-182 of `data` after bytes whose CRC is `seed` to
/// `*checksum`.
///
/// # Safety
/// `data` points to `len` readable bytes that stay unchanged during the
/// call, or `len` is 0; `checksum` points to a writable value.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_cache_checksum(
    data: *const u8,
    len: usize,
    seed: u64,
    checksum: *mut u64,
) -> GufoCacheStatus {
    guard(GufoCacheStatus::Internal, || {
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(bytes) = (unsafe { gufo_ffi::input(data, len) }) else {
            return GufoCacheStatus::Internal;
        };
        // SAFETY: the caller guarantees a writable value or passes null.
        let Some(checksum) = (unsafe { checksum.as_mut() }) else {
            return GufoCacheStatus::Internal;
        };
        *checksum = crc::checksum(bytes, seed);
        GufoCacheStatus::Complete
    })
}

/// Reads the counts of the manifest in `data` into `*plan`.
///
/// # Safety
/// `data` points to `len` readable bytes that stay unchanged during the
/// call, or `len` is 0; `plan` points to a writable value.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_cache_manifest_plan(
    data: *const u8,
    len: usize,
    plan: *mut GufoCacheManifestPlan,
) -> GufoCacheStatus {
    guard(GufoCacheStatus::Internal, || {
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(bytes) = (unsafe { gufo_ffi::input(data, len) }) else {
            return GufoCacheStatus::Internal;
        };
        // SAFETY: the caller guarantees a writable value or passes null.
        let Some(out) = (unsafe { plan.as_mut() }) else {
            return GufoCacheStatus::Internal;
        };
        match manifest::plan(bytes) {
            Ok(p) => {
                *out = GufoCacheManifestPlan {
                    input_bytes: p.input_bytes,
                    tokens: p.tokens,
                    components: p.components,
                    chunks: p.chunks,
                    references: p.references,
                };
                GufoCacheStatus::Complete
            }
            Err(failure) => failure.into(),
        }
    })
}

/// Decodes and validates the manifest in `data`, then reports it to `sink`.
///
/// # Safety
/// `data` points to `len` readable bytes that stay unchanged during the
/// call, or `len` is 0; `sink` points to a readable value whose callbacks
/// may be called with its context until this returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_cache_manifest_decode(
    data: *const u8,
    len: usize,
    sink: *const GufoCacheManifestSink,
) -> GufoCacheStatus {
    guard(GufoCacheStatus::Internal, || {
        // SAFETY: the caller guarantees `len` readable bytes.
        let Some(bytes) = (unsafe { gufo_ffi::input(data, len) }) else {
            return GufoCacheStatus::Internal;
        };
        // SAFETY: the caller guarantees a readable sink or passes null.
        let Some(sink) = (unsafe { sink.as_ref() }) else {
            return GufoCacheStatus::Internal;
        };
        let (Some(begin), Some(tokens), Some(component), Some(chunks)) =
            (sink.begin, sink.tokens, sink.component, sink.chunks)
        else {
            return GufoCacheStatus::Internal;
        };
        status(manifest::decode(
            bytes,
            &mut Callbacks {
                context: sink.context,
                begin,
                tokens,
                component,
                chunks,
            },
        ))
    })
}

/// Validates and encodes `manifest` into the buffer `buffer` returns.
///
/// # Safety
/// `manifest` points to a readable value whose input, tokens, components and
/// their chunks point to as many readable values as their counts, or have a
/// count of 0, all unchanged during the call; `buffer` may be called with
/// `context` until this returns, and returns null or `len` writable bytes
/// that alias nothing else here.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_cache_manifest_encode(
    manifest: *const GufoCacheManifest,
    buffer: Option<GufoCacheBufferFn>,
    context: *mut c_void,
) -> GufoCacheStatus {
    guard(GufoCacheStatus::Internal, || {
        // SAFETY: the caller guarantees a readable manifest or passes null.
        let (Some(m), Some(buffer)) = (unsafe { manifest.as_ref() }, buffer) else {
            return GufoCacheStatus::Internal;
        };
        // Rejected before the components are borrowed, so no slice outlives
        // what the caller allocated.
        if m.component_count > manifest::MAX_COMPONENTS {
            return GufoCacheStatus::Rejected;
        }
        // SAFETY: the caller guarantees the counts of readable values.
        let (Some(input), Some(tokens), Some(components)) = (unsafe {
            (
                gufo_ffi::input(m.input, m.input_len),
                gufo_ffi::input(m.tokens, m.token_count),
                gufo_ffi::input(m.components, m.component_count),
            )
        }) else {
            return GufoCacheStatus::Internal;
        };
        let mut views = Vec::new();
        if views.try_reserve_exact(components.len()).is_err() {
            return GufoCacheStatus::NoMemory;
        }
        for c in components {
            // SAFETY: the caller guarantees `chunk_count` readable chunks.
            let Some(chunks) = (unsafe { gufo_ffi::input(c.chunks, c.chunk_count) }) else {
                return GufoCacheStatus::Internal;
            };
            views.push(Component {
                id: c.id,
                layout_version: c.layout_version,
                kind: c.kind,
                row_bytes: c.row_bytes,
                rows_per_chunk: c.rows_per_chunk,
                state_bytes: c.state_bytes,
                window_rows: c.window_rows,
                position_id: c.position_id,
                valid_rows: c.valid_rows,
                chunks,
                tail: optional(c.has_tail, c.tail),
                private_state: optional(c.has_private_state, c.private_state),
            });
        }
        let header = Header {
            checkpoint: m.checkpoint,
            lineage: m.lineage,
            compatibility: m.compatibility,
            purpose: m.purpose,
            rank: m.rank,
        };
        status(manifest::encode(&header, input, tokens, &views, |len| {
            // SAFETY: the caller allows this call and returns null or `len`
            // exclusively writable bytes that live until the encoding returns.
            let data = unsafe { buffer(context, len) };
            // SAFETY: as above; null stops the encoding.
            match unsafe { gufo_ffi::output(data, len) } {
                Some(bytes) if !data.is_null() => Ok(bytes),
                _ => Err(Stopped),
            }
        }))
    })
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Seen {
        tokens: Vec<u32>,
        components: Vec<GufoCacheComponent>,
        chunks: Vec<Payload>,
        input: Vec<u8>,
        checkpoint: u64,
        out: Vec<u8>,
        stop: bool,
    }

    fn seen<'a>(context: *mut c_void) -> &'a mut Seen {
        // SAFETY: every test passes a live `Seen` as the context.
        unsafe { &mut *context.cast::<Seen>() }
    }

    unsafe extern "C" fn begin(context: *mut c_void, m: *const GufoCacheManifest) -> c_int {
        let seen = seen(context);
        // SAFETY: the codec passes a live manifest whose input is readable.
        let m = unsafe { &*m };
        assert!(m.tokens.is_null() && m.components.is_null());
        seen.checkpoint = m.checkpoint;
        // SAFETY: as above.
        seen.input = unsafe { std::slice::from_raw_parts(m.input, m.input_len) }.to_vec();
        c_int::from(seen.stop)
    }

    unsafe extern "C" fn tokens(context: *mut c_void, data: *const u32, count: usize) -> c_int {
        // SAFETY: the codec passes a readable block.
        let block = unsafe { std::slice::from_raw_parts(data, count) };
        seen(context).tokens.extend_from_slice(block);
        0
    }

    unsafe extern "C" fn component(context: *mut c_void, c: *const GufoCacheComponent) -> c_int {
        // SAFETY: the codec passes a live component.
        seen(context).components.push(unsafe { *c });
        0
    }

    unsafe extern "C" fn chunks(context: *mut c_void, data: *const Payload, count: usize) -> c_int {
        // SAFETY: the codec passes a readable block.
        let block = unsafe { std::slice::from_raw_parts(data, count) };
        seen(context).chunks.extend_from_slice(block);
        0
    }

    unsafe extern "C" fn buffer(context: *mut c_void, len: usize) -> *mut u8 {
        let seen = seen(context);
        if seen.stop {
            return std::ptr::null_mut();
        }
        seen.out.resize(len, 0);
        seen.out.as_mut_ptr()
    }

    fn payload(n: u8, bytes: u64) -> Payload {
        Payload {
            file: [n; 16],
            bytes,
            checksum: u64::from(n),
        }
    }

    #[test]
    fn manifests_cross_the_boundary() {
        let chunk_list = [payload(1, 64), payload(2, 64)];
        let token_list: Vec<u32> = (0..40).collect();
        let none = payload(0, 0);
        let components = [
            GufoCacheComponent {
                id: 1,
                layout_version: 1,
                kind: manifest::APPEND_ROWS,
                row_bytes: 4,
                rows_per_chunk: 16,
                state_bytes: 0,
                window_rows: 0,
                position_id: 1,
                valid_rows: 40,
                chunks: chunk_list.as_ptr(),
                chunk_count: 2,
                has_tail: 1,
                tail: payload(3, 32),
                has_private_state: 0,
                private_state: none,
            },
            GufoCacheComponent {
                id: 2,
                layout_version: 1,
                kind: manifest::PRIVATE_STATE,
                row_bytes: 0,
                rows_per_chunk: 0,
                state_bytes: 8,
                window_rows: 0,
                position_id: 2,
                valid_rows: 40,
                chunks: std::ptr::null(),
                chunk_count: 0,
                has_tail: 0,
                tail: none,
                has_private_state: 1,
                private_state: payload(4, 8),
            },
        ];
        let input = b"identity";
        let m = GufoCacheManifest {
            checkpoint: 5,
            lineage: 6,
            compatibility: [7; 32],
            purpose: 1,
            rank: 2,
            input: input.as_ptr(),
            input_len: input.len(),
            tokens: token_list.as_ptr(),
            token_count: token_list.len(),
            components: components.as_ptr(),
            component_count: components.len(),
        };
        let mut seen = Seen::default();
        let context: *mut c_void = std::ptr::from_mut(&mut seen).cast();
        // SAFETY: every pointer above lives throughout.
        let encoded = unsafe { gufo_cache_manifest_encode(&m, Some(buffer), context) };
        assert_eq!(encoded, GufoCacheStatus::Complete);
        let bytes = std::mem::take(&mut seen.out);
        let sink = GufoCacheManifestSink {
            context,
            begin: Some(begin),
            tokens: Some(tokens),
            component: Some(component),
            chunks: Some(chunks),
        };
        let mut plan = GufoCacheManifestPlan::default();
        // SAFETY: `bytes`, `sink`, `plan` and `seen` live throughout.
        unsafe {
            assert_eq!(
                gufo_cache_manifest_plan(bytes.as_ptr(), bytes.len(), &mut plan),
                GufoCacheStatus::Complete
            );
            assert_eq!(
                gufo_cache_manifest_decode(bytes.as_ptr(), bytes.len(), &sink),
                GufoCacheStatus::Complete
            );
        }
        assert_eq!(
            plan,
            GufoCacheManifestPlan {
                input_bytes: 8,
                tokens: 40,
                components: 2,
                chunks: 2,
                references: 4,
            }
        );
        assert_eq!(seen.checkpoint, 5);
        assert_eq!(seen.input, input);
        assert_eq!(seen.tokens, token_list);
        assert_eq!(seen.chunks, chunk_list);
        assert_eq!(seen.components.len(), 2);
        assert_eq!(seen.components[0].chunk_count, 2);
        assert_eq!(seen.components[0].has_tail, 1);
        assert_eq!(seen.components[0].tail, payload(3, 32));
        assert_eq!(seen.components[1].has_private_state, 1);
        assert_eq!(seen.components[1].private_state, payload(4, 8));

        seen.stop = true;
        // SAFETY: as above.
        unsafe {
            assert_eq!(
                gufo_cache_manifest_decode(bytes.as_ptr(), bytes.len(), &sink),
                GufoCacheStatus::Stopped
            );
            assert_eq!(
                gufo_cache_manifest_encode(&m, Some(buffer), context),
                GufoCacheStatus::Stopped
            );
            assert_eq!(
                gufo_cache_manifest_decode(bytes.as_ptr(), bytes.len() - 1, &sink),
                GufoCacheStatus::Rejected
            );
        }
        let mut wide = m;
        wide.component_count = manifest::MAX_COMPONENTS + 1;
        // SAFETY: the count is rejected before the components are read.
        let encoded = unsafe { gufo_cache_manifest_encode(&wide, Some(buffer), context) };
        assert_eq!(encoded, GufoCacheStatus::Rejected);
    }

    #[test]
    fn checksum_and_caller_errors() {
        let mut checksum = 0;
        let mut seen = Seen::default();
        let context: *mut c_void = std::ptr::from_mut(&mut seen).cast();
        // SAFETY: null is passed only where the function rejects it before
        // reading, or with a zero length.
        unsafe {
            assert_eq!(
                gufo_cache_checksum(b"123456789".as_ptr(), 9, 0, &mut checksum),
                GufoCacheStatus::Complete
            );
            assert_eq!(checksum, 0x6c40_df5f_0b49_7347);
            assert_eq!(
                gufo_cache_checksum(std::ptr::null(), 0, 3, &mut checksum),
                GufoCacheStatus::Complete
            );
            assert_eq!(checksum, 3);
            assert_eq!(
                gufo_cache_checksum(std::ptr::null(), 1, 0, &mut checksum),
                GufoCacheStatus::Internal
            );
            assert_eq!(
                gufo_cache_checksum(b"1".as_ptr(), 1, 0, std::ptr::null_mut()),
                GufoCacheStatus::Internal
            );
            assert_eq!(
                gufo_cache_manifest_plan(b"1".as_ptr(), 1, std::ptr::null_mut()),
                GufoCacheStatus::Internal
            );
            assert_eq!(
                gufo_cache_manifest_decode(b"1".as_ptr(), 1, std::ptr::null()),
                GufoCacheStatus::Internal
            );
            let sink = GufoCacheManifestSink {
                context,
                begin: Some(begin),
                tokens: None,
                component: Some(component),
                chunks: Some(chunks),
            };
            assert_eq!(
                gufo_cache_manifest_decode(b"1".as_ptr(), 1, &sink),
                GufoCacheStatus::Internal
            );
            assert_eq!(
                gufo_cache_manifest_encode(std::ptr::null(), Some(buffer), context),
                GufoCacheStatus::Internal
            );
        }
    }
}

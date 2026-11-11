#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

#[cfg(panic = "abort")]
compile_error!("the C functions catch panics at the boundary, so build with panic = \"unwind\"");

mod frames;

use std::ffi::{c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use frames::{Event, Reader};

/// The frame reader of one connection.
pub struct GufoWsReader(Reader);

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GufoWsEventKind {
    End = 0,
    Message = 1,
    Ping = 2,
    Close = 3,
    Error = 4,
    Internal = 5,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GufoWsEvent {
    pub kind: GufoWsEventKind,
    /// The close code of an `Error`.
    pub code: u16,
    /// The text of a `Message`, or the payload of a `Ping` or `Close`;
    /// valid until the next call with the same reader.
    pub data: *const u8,
    pub len: usize,
}

/// Fills `len` bytes at `data` and returns nonzero, or returns 0 once the
/// connection is over.
pub type GufoWsReadFn =
    unsafe extern "C" fn(context: *mut c_void, data: *mut u8, len: usize) -> c_int;

/// A reader for messages of at most `max_message` bytes, or null if it could
/// not be allocated.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn gufo_ws_reader_new(max_message: usize) -> *mut GufoWsReader {
    catch_unwind(|| try_box(GufoWsReader(Reader::new(max_message)))).unwrap_or(std::ptr::null_mut())
}

/// Moves `value` to the heap like `Box::new`, but returns null when memory
/// runs out instead of aborting the process.
fn try_box<T>(value: T) -> *mut T {
    let mut slot = Vec::new();
    if slot.try_reserve_exact(1).is_err() {
        return std::ptr::null_mut();
    }
    slot.push(value);
    // One element fills the reserved capacity, so no reallocation happens,
    // and `[T; 1]` has the layout of `T`: `Box::<T>::from_raw` frees it.
    Box::<[T; 1]>::try_from(slot.into_boxed_slice())
        .map_or(std::ptr::null_mut(), |boxed| Box::into_raw(boxed).cast())
}

/// # Safety
/// `reader` is null or comes from `gufo_ws_reader_new` and is not used
/// afterwards.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_ws_reader_free(reader: *mut GufoWsReader) {
    if !reader.is_null() {
        // SAFETY: the caller passes a reader from `gufo_ws_reader_new` once,
        // allocated by the global allocator with the layout of the reader.
        drop(unsafe { Box::from_raw(reader) });
    }
}

/// Reads the next event through `read` into `*event`. Pongs are skipped and
/// fragments joined. After `Close`, `Error`, `End` or `Internal`, the
/// connection is over.
///
/// # Safety
/// `reader` comes from `gufo_ws_reader_new`; `event` points to a writable
/// value. `read` may be called with `context` from this thread until this
/// function returns.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_ws_reader_next(
    reader: *mut GufoWsReader,
    read: Option<GufoWsReadFn>,
    context: *mut c_void,
    event: *mut GufoWsEvent,
) {
    // SAFETY: the caller upholds the contract above; `as_mut` rejects null.
    let Some(event) = (unsafe { event.as_mut() }) else {
        return;
    };
    *event = GufoWsEvent {
        kind: GufoWsEventKind::Internal,
        code: 0,
        data: std::ptr::null(),
        len: 0,
    };
    // SAFETY: the caller passes a live reader; `as_mut` rejects null.
    let (Some(reader), Some(read)) = (unsafe { reader.as_mut() }, read) else {
        return;
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the caller allows calling `read` with `context` here, and
        // the buffer is writable for its length.
        let mut read =
            |buffer: &mut [u8]| unsafe { read(context, buffer.as_mut_ptr(), buffer.len()) } != 0;
        let (kind, code, bytes): (_, _, &[u8]) = match reader.0.next(&mut read) {
            Event::End => (GufoWsEventKind::End, 0, &[]),
            Event::Message(bytes) => (GufoWsEventKind::Message, 0, bytes),
            Event::Ping(bytes) => (GufoWsEventKind::Ping, 0, bytes),
            Event::Close(bytes) => (GufoWsEventKind::Close, 0, bytes),
            Event::Error(code) => (GufoWsEventKind::Error, code, &[]),
        };
        GufoWsEvent {
            kind,
            code,
            data: bytes.as_ptr(),
            len: bytes.len(),
        }
    }));
    if let Ok(next) = result {
        *event = next;
    }
}

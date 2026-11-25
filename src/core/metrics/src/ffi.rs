use std::ffi::{c_int, c_void};
use std::sync::Arc;

use gufo_ffi::guard;

use crate::registry::{Error, Family, Kind, Registry, Series};

const OK: u32 = 0;
const INVALID: u32 = 1;
const DUPLICATE: u32 = 2;
const CALLBACK: u32 = 3;
const INTERNAL: u32 = 4;

fn status(error: Error) -> u32 {
    match error {
        Error::Invalid => INVALID,
        Error::Duplicate => DUPLICATE,
    }
}

fn kind(value: u32) -> Option<Kind> {
    Some(match value {
        0 => Kind::CounterU64,
        1 => Kind::CounterF64,
        2 => Kind::GaugeU64,
        3 => Kind::GaugeF64,
        4 => Kind::GaugeI64,
        5 => Kind::Histogram,
        _ => return None,
    })
}

#[repr(C)]
pub struct GufoMetricsLabel {
    pub name: *const u8,
    pub name_len: usize,
    pub value: *const u8,
    pub value_len: usize,
}

type WriteFn = extern "C" fn(*mut c_void, *const u8, usize) -> c_int;

/// # Safety
/// `data` and `len` satisfy `gufo_ffi::input`.
#[allow(unsafe_code)]
unsafe fn text<'a>(data: *const u8, len: usize) -> Option<&'a str> {
    // SAFETY: forwarded from the caller.
    let bytes = unsafe { gufo_ffi::input(data, len) }?;
    std::str::from_utf8(bytes).ok()
}

#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn gufo_metrics_registry_new() -> *mut c_void {
    guard(std::ptr::null_mut(), || {
        Box::into_raw(Box::new(Registry::new())).cast()
    })
}

/// # Safety
/// `registry` is null or from `gufo_metrics_registry_new`, not freed before,
/// and no other thread uses it or any handle taken from it during or after.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_metrics_registry_free(registry: *mut c_void) {
    guard((), || {
        if !registry.is_null() {
            // SAFETY: the caller owns the registry and gives it up here.
            drop(unsafe { Box::from_raw(registry.cast::<Registry>()) });
        }
    })
}

/// # Safety
/// `registry` is live; `name`, `help` and `bounds` satisfy `gufo_ffi::input`
/// with their lengths; `out` is writable.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_metrics_family_new(
    registry: *const c_void,
    kind_value: u32,
    name: *const u8,
    name_len: usize,
    help: *const u8,
    help_len: usize,
    bounds: *const f64,
    bounds_len: usize,
    out: *mut *const c_void,
) -> u32 {
    guard(INTERNAL, || {
        // SAFETY: the caller supplies a live registry and a writable `out`.
        let (Some(registry), Some(out)) =
            (unsafe { registry.cast::<Registry>().as_ref() }, unsafe {
                out.as_mut()
            })
        else {
            return INTERNAL;
        };
        // SAFETY: the caller supplies readable buffers of these lengths.
        let (Some(name), Some(help), Some(bounds)) = (
            unsafe { text(name, name_len) },
            unsafe { text(help, help_len) },
            unsafe { gufo_ffi::input(bounds, bounds_len) },
        ) else {
            return INVALID;
        };
        let Some(kind) = kind(kind_value) else {
            return INVALID;
        };
        match registry.add_family(kind, name, help, bounds) {
            Ok(family) => {
                *out = Arc::as_ptr(&family).cast();
                OK
            }
            Err(error) => status(error),
        }
    })
}

/// # Safety
/// `family` is from `gufo_metrics_family_new` on a live registry; `labels`
/// holds `labels_len` entries whose strings satisfy `gufo_ffi::input`; `out`
/// is writable.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_metrics_series_new(
    family: *const c_void,
    labels: *const GufoMetricsLabel,
    labels_len: usize,
    out: *mut *const c_void,
) -> u32 {
    guard(INTERNAL, || {
        // SAFETY: the caller supplies a live family and a writable `out`.
        let (Some(family), Some(out)) = (unsafe { family.cast::<Family>().as_ref() }, unsafe {
            out.as_mut()
        }) else {
            return INTERNAL;
        };
        // SAFETY: the caller supplies `labels_len` readable entries.
        let Some(entries) = (unsafe { gufo_ffi::input(labels, labels_len) }) else {
            return INTERNAL;
        };
        let mut pairs = Vec::with_capacity(entries.len());
        for entry in entries {
            // SAFETY: the caller supplies readable strings of these lengths.
            let (Some(name), Some(value)) = (unsafe { text(entry.name, entry.name_len) }, unsafe {
                text(entry.value, entry.value_len)
            }) else {
                return INVALID;
            };
            pairs.push((name, value));
        }
        match family.add_series(&pairs) {
            Ok(series) => {
                *out = Arc::as_ptr(&series).cast();
                OK
            }
            Err(error) => status(error),
        }
    })
}

/// # Safety
/// `family` is from `gufo_metrics_family_new` on a live registry.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_metrics_family_set_visible(family: *const c_void, visible: bool) {
    guard((), || {
        // SAFETY: the caller supplies a live family or null.
        if let Some(family) = unsafe { family.cast::<Family>().as_ref() } {
            family.set_visible(visible);
        }
    })
}

/// # Safety
/// `registry` is live; `write` is callable with `context`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_metrics_render(
    registry: *const c_void,
    write: Option<WriteFn>,
    context: *mut c_void,
) -> u32 {
    guard(INTERNAL, || {
        // SAFETY: the caller supplies a live registry.
        let (Some(registry), Some(write)) =
            (unsafe { registry.cast::<Registry>().as_ref() }, write)
        else {
            return INTERNAL;
        };
        let text = registry.render();
        if write(context, text.as_ptr(), text.len()) != 0 {
            CALLBACK
        } else {
            OK
        }
    })
}

/// # Safety
/// `cell` is null or from `gufo_metrics_series_new` on a live registry.
#[allow(unsafe_code)]
unsafe fn with_cell(cell: *const c_void, update: impl FnOnce(&Series)) {
    guard((), || {
        // SAFETY: the caller supplies a live cell or null.
        if let Some(series) = unsafe { cell.cast::<Series>().as_ref() } {
            update(series);
        }
    })
}

macro_rules! cell_update {
    ($name:ident, $arg:ty, $method:ident) => {
        /// # Safety
        /// `cell` is null or from `gufo_metrics_series_new` on a live registry.
        #[allow(unsafe_code)]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(cell: *const c_void, value: $arg) {
            // SAFETY: forwarded from the caller.
            unsafe { with_cell(cell, |series| series.value.$method(value)) }
        }
    };
}

cell_update!(gufo_metrics_u64_add, u64, add_u64);
cell_update!(gufo_metrics_u64_set, u64, set_u64);
cell_update!(gufo_metrics_u64_max, u64, max_u64);
cell_update!(gufo_metrics_f64_add, f64, add_f64);
cell_update!(gufo_metrics_f64_set, f64, set_f64);
cell_update!(gufo_metrics_i64_add, i64, add_i64);
cell_update!(gufo_metrics_i64_set, i64, set_i64);
cell_update!(gufo_metrics_histogram_observe, f64, observe);

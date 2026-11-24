//! Shared contracts at the C boundary. No Rust panic may cross it.
#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

#[cfg(panic = "abort")]
compile_error!("C boundary guards require panic = \"unwind\"");

/// Catch an unexpected panic and report the caller's internal error value.
/// Captured state must be discarded or remain valid after a panic.
pub fn guard<T>(fallback: T, call: impl FnOnce() -> T) -> T {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)).unwrap_or(fallback)
}

/// Borrow a C input, accepting null only for an empty slice.
///
/// # Safety
/// For nonzero `len`, a non-null `data` must be aligned and point to `len`
/// initialized values in one allocation, readable for the returned lifetime.
/// The size must not exceed `isize::MAX`, and no mutation may alias the borrow.
#[allow(unsafe_code)]
pub unsafe fn input<'a, T>(data: *const T, len: usize) -> Option<&'a [T]> {
    match (data.is_null(), len) {
        (_, 0) => Some(&[]),
        (true, _) => None,
        // SAFETY: the caller supplies a valid, readable allocation.
        (false, _) => Some(unsafe { std::slice::from_raw_parts(data, len) }),
    }
}

/// Borrow a C output, accepting null only for an empty slice.
///
/// # Safety
/// For nonzero `len`, a non-null `data` must be aligned and point to `len`
/// initialized values in one allocation, writable for the returned lifetime.
/// The size must not exceed `isize::MAX`, and the borrow must be exclusive.
#[allow(unsafe_code)]
pub unsafe fn output<'a, T>(data: *mut T, len: usize) -> Option<&'a mut [T]> {
    match (data.is_null(), len) {
        (_, 0) => Some(&mut []),
        (true, _) => None,
        // SAFETY: the caller supplies a valid, exclusively writable allocation.
        (false, _) => Some(unsafe { std::slice::from_raw_parts_mut(data, len) }),
    }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;

    #[test]
    fn panic_reports_fallback() {
        assert_eq!(guard(7, || 3), 3);
        assert_eq!(guard(7, || panic!("boundary failure")), 7);
    }

    #[test]
    fn null_and_nonempty_buffers() {
        // SAFETY: null is explicitly supported; the arrays live throughout.
        unsafe {
            assert_eq!(input::<u8>(std::ptr::null(), 0), Some(&[][..]));
            assert!(input::<u8>(std::ptr::null(), 1).is_none());
            assert_eq!(output::<u8>(std::ptr::null_mut(), 0), Some(&mut [][..]));
            assert!(output::<u8>(std::ptr::null_mut(), 1).is_none());
            let mut values = [1u32, 2];
            assert_eq!(input(values.as_ptr(), 2), Some(&values[..]));
            output(values.as_mut_ptr(), 2).unwrap()[1] = 9;
            assert_eq!(values, [1, 9]);
        }
    }
}

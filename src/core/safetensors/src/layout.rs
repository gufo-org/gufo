//! The file layout: the length prefix and the payload the tensors tile.

use crate::{Problem, Reason};

/// The bytes of the little-endian header length that starts a file.
pub const LENGTH_BYTES: u64 = 8;
/// The largest header accepted. Real headers take well under a megabyte.
pub const MAX_HEADER_BYTES: u64 = 32 << 20;

const SHORT: Problem = (Reason::File, "shorter than the 8-byte header length");

/// Checks a header length against the size of its file: at least one byte,
/// at most `MAX_HEADER_BYTES`, and within the file.
pub fn check_length(header: u64, file_size: u64) -> Result<(), Problem> {
    if file_size < LENGTH_BYTES {
        return Err(SHORT);
    }
    if header == 0 {
        return Err((Reason::Length, "length is 0"));
    }
    if header > MAX_HEADER_BYTES {
        return Err((Reason::Length, "length is above 32 MiB"));
    }
    if header > file_size - LENGTH_BYTES {
        return Err((Reason::Length, "runs past the end of the file"));
    }
    Ok(())
}

/// Reads the header length from the first bytes of a file of `file_size`
/// bytes and checks it.
pub fn header_length(prefix: &[u8], file_size: u64) -> Result<u64, Problem> {
    let Some(bytes) = prefix.first_chunk::<8>() else {
        return Err(SHORT);
    };
    let header = u64::from_le_bytes(*bytes);
    check_length(header, file_size)?;
    Ok(header)
}

/// Why extents do not tile the payload, and the tensors involved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Untiled {
    /// `tensor` begins inside `other`, the tensor before it.
    Overlap { tensor: usize, other: usize },
    /// Payload bytes before `tensor` belong to none.
    Gap(usize),
    /// Payload bytes after the last tensor belong to none.
    Trailing,
}

/// Checks that the extents of the tensors with data, each `(begin, end,
/// tensor)`, tile a payload of `payload` bytes exactly: sorted by offset,
/// each starts where the one before ended and the last ends with the
/// payload, which is empty when no tensor has data. Leaves them sorted.
pub fn tile(extents: &mut [(u64, u64, usize)], payload: u64) -> Result<(), Untiled> {
    extents.sort_unstable();
    let mut cursor = 0;
    let mut previous = None;
    for &(begin, end, tensor) in extents.iter() {
        if begin < cursor {
            // The extents before tile [0, cursor), so the last one holds
            // `begin`.
            let other = previous.unwrap_or(tensor);
            return Err(Untiled::Overlap { tensor, other });
        }
        if begin > cursor {
            return Err(Untiled::Gap(tensor));
        }
        cursor = end;
        previous = Some(tensor);
    }
    if cursor != payload {
        return Err(Untiled::Trailing);
    }
    Ok(())
}

/// The tensor whose data holds `at`, an offset within the payload, if any:
/// `extents` are those `tile` accepted. Offsets between two tensors and at
/// either end of the payload lie in none.
pub fn holder(extents: &[(u64, u64, usize)], at: u64) -> Option<usize> {
    // The last extent that begins at or before `at`.
    let last = extents
        .partition_point(|&(begin, _, _)| begin <= at)
        .checked_sub(1)?;
    let (begin, end, tensor) = extents[last];
    (begin < at && at < end).then_some(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths() {
        let prefix = |length: u64| length.to_le_bytes();
        assert_eq!(header_length(&prefix(2), 10), Ok(2));
        assert_eq!(header_length(&prefix(1), 9), Ok(1));
        let reason =
            |prefix: &[u8], size| header_length(prefix, size).map_err(|(reason, _)| reason);
        assert_eq!(
            header_length(&prefix(2), 9),
            Err((Reason::Length, "runs past the end of the file"))
        );
        assert_eq!(
            header_length(&prefix(0), 100),
            Err((Reason::Length, "length is 0"))
        );
        assert_eq!(
            header_length(&prefix(u64::MAX), u64::MAX),
            Err((Reason::Length, "length is above 32 MiB"))
        );
        let largest = MAX_HEADER_BYTES + LENGTH_BYTES;
        assert_eq!(
            header_length(&prefix(MAX_HEADER_BYTES), largest),
            Ok(MAX_HEADER_BYTES)
        );
        assert_eq!(
            reason(&prefix(MAX_HEADER_BYTES + 1), largest + 1),
            Err(Reason::Length)
        );
        // Short files, and prefixes shorter than the length.
        assert_eq!(header_length(&prefix(1), 7), Err(SHORT));
        assert_eq!(header_length(&[1, 0, 0, 0, 0, 0, 0], 100), Err(SHORT));
        assert_eq!(header_length(&[], 0), Err(SHORT));
        // Only the first eight bytes count.
        assert_eq!(header_length(&[3, 0, 0, 0, 0, 0, 0, 0, 9], 11), Ok(3));
    }

    #[test]
    fn tiles() {
        assert_eq!(tile(&mut [], 0), Ok(()));
        assert_eq!(tile(&mut [], 1), Err(Untiled::Trailing));
        assert_eq!(tile(&mut [(4, 6, 0), (0, 4, 1)], 6), Ok(()));
        assert_eq!(tile(&mut [(0, 4, 0), (5, 6, 1)], 6), Err(Untiled::Gap(1)));
        assert_eq!(tile(&mut [(1, 6, 0)], 6), Err(Untiled::Gap(0)));
        assert_eq!(
            tile(&mut [(4, 6, 2), (0, 4, 0), (3, 6, 1)], 6),
            Err(Untiled::Overlap {
                tensor: 1,
                other: 0
            })
        );
        assert_eq!(
            tile(&mut [(0, 4, 0), (0, 4, 1)], 4),
            Err(Untiled::Overlap {
                tensor: 1,
                other: 0
            })
        );
        assert_eq!(tile(&mut [(0, 4, 0), (4, 5, 1)], 6), Err(Untiled::Trailing));
    }

    #[test]
    fn holders() {
        let mut extents = [(4, 6, 1), (0, 4, 0)];
        assert_eq!(tile(&mut extents, 6), Ok(()));
        let holders: Vec<_> = (0..=6).map(|at| holder(&extents, at)).collect();
        let expected = [None, Some(0), Some(0), Some(0), None, Some(1), None];
        assert_eq!(holders, expected);
        assert_eq!(holder(&[], 0), None);
    }
}

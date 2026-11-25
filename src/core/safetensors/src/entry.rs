//! One tensor entry of the header: its type, shape and extent in the payload.

use crate::dtype::Dtype;
use crate::{Problem, Reason};
use std::ops::Range;

/// The most dimensions a tensor may have.
pub const MAX_RANK: usize = 5;
/// The largest dimension, which C++ callers keep in an `int`.
pub const MAX_DIMENSION: u64 = i32::MAX as u64;

/// A checked tensor entry. `begin` and `end` are offsets in the payload,
/// which starts right after the header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    pub dtype: Dtype,
    dimensions: [u64; MAX_RANK],
    rank: usize,
    pub begin: u64,
    pub end: u64,
}

impl Entry {
    pub fn shape(&self) -> &[u64] {
        &self.dimensions[..self.rank]
    }
}

/// The items of a shape or offset array as read: each a plain integer or
/// `None`. Only the first `MAX_RANK + 1` are kept, but all are counted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Numbers {
    values: [Option<u64>; MAX_RANK + 1],
    len: usize,
}

impl Numbers {
    pub fn push(&mut self, value: Option<u64>) {
        if let Some(slot) = self.values.get_mut(self.len) {
            *slot = value;
        }
        self.len = self.len.saturating_add(1);
    }
}

/// A member of an entry as read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Member<T> {
    #[default]
    Missing,
    /// A value of the wrong kind.
    Wrong,
    Read(T),
}

/// The members of a tensor entry that the reader checks. A type the format
/// does not name keeps where its text is, for the rejection.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Raw {
    pub dtype: Member<Result<Dtype, Range<usize>>>,
    pub shape: Member<Numbers>,
    pub offsets: Member<Numbers>,
}

/// A JSON integer read exactly from its token: digits only, no sign,
/// fraction or exponent.
pub fn integer(token: &[u8]) -> Option<u64> {
    if token.is_empty() {
        return None;
    }
    token.iter().try_fold(0u64, |value, &byte| {
        let digit = byte.checked_sub(b'0').filter(|digit| *digit < 10)?;
        value.checked_mul(10)?.checked_add(u64::from(digit))
    })
}

/// Checks one tensor entry of a header whose payload holds `payload` bytes:
/// its type, then the shape and offset arrays, each dimension, and finally
/// the extent, which must span exactly the tensor's bytes. A tensor without
/// elements spans none: its offsets are equal and within the payload, and
/// its element count multiplied in order fits 64 bits. An unknown type is
/// reported without its name, which the caller adds.
pub fn check(raw: &Raw, payload: u64) -> Result<Entry, Problem> {
    let dtype = match raw.dtype {
        Member::Read(Ok(dtype)) => dtype,
        Member::Read(Err(_)) => return Err((Reason::Dtype, "unsupported dtype")),
        Member::Missing => return Err((Reason::Dtype, "missing dtype")),
        Member::Wrong => return Err((Reason::Dtype, "dtype is not a string")),
    };
    let shape = match &raw.shape {
        Member::Read(shape) => shape,
        Member::Missing => return Err((Reason::Layout, "missing shape")),
        Member::Wrong => return Err((Reason::Layout, "shape is not an array")),
    };
    let offsets = match &raw.offsets {
        Member::Read(offsets) => offsets,
        Member::Missing => return Err((Reason::Layout, "missing data_offsets")),
        Member::Wrong => return Err((Reason::Layout, "data_offsets is not an array")),
    };
    if shape.len > MAX_RANK {
        return Err((Reason::Layout, "shape has more than 5 dimensions"));
    }
    if offsets.len != 2 {
        return Err((Reason::Layout, "data_offsets does not hold 2 values"));
    }
    let mut dimensions = [0; MAX_RANK];
    for (slot, dimension) in dimensions.iter_mut().zip(&shape.values[..shape.len]) {
        *slot = dimension
            .filter(|dimension| *dimension <= MAX_DIMENSION)
            .ok_or((
                Reason::Dimension,
                "shape holds a dimension that is not an integer from 0 to 2147483647",
            ))?;
    }
    let shape = &dimensions[..shape.len];
    // The bits of the tensor, none without elements. A shape larger than the
    // payload is rejected as soon as it is, which also keeps the product in
    // range. Without elements, the element count multiplied in order must
    // still fit 64 bits before the first 0, as in safetensors 0.8.
    let room = u128::from(payload) * 8;
    let mut bits = 0;
    if !shape.contains(&0) {
        bits = u128::from(dtype.bits());
        for &dimension in shape {
            bits *= u128::from(dimension);
            if bits > room {
                return Err((Reason::Dimension, "shape holds more bytes than the payload"));
            }
        }
    } else if shape
        .iter()
        .try_fold(1u64, |count, &dimension| count.checked_mul(dimension))
        .is_none()
    {
        return Err((
            Reason::Dimension,
            "shape's element count overflows 64 bits before its 0 dimension",
        ));
    }
    let [Some(begin), Some(end)] = [offsets.values[0], offsets.values[1]] else {
        return Err((Reason::Extent, "data_offsets are not unsigned integers"));
    };
    if bits % 8 != 0 {
        return Err((Reason::Extent, "shape does not fill whole bytes"));
    }
    if begin > end {
        return Err((Reason::Extent, "data_offsets decrease"));
    }
    if end > payload {
        return Err((Reason::Extent, "data_offsets end past the payload"));
    }
    // `bits` is at most `room`, so the bytes fit a u64.
    if u128::from(end - begin) * 8 != bits {
        return Err((
            Reason::Extent,
            "data_offsets do not span exactly the shape's bytes",
        ));
    }
    Ok(Entry {
        dtype,
        dimensions,
        rank: shape.len(),
        begin,
        end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gufo_json::{Limits, Value, parse_value};

    type Checked = Result<(Dtype, Vec<u64>, u64, u64), Problem>;

    /// Reads an entry from a value tree, as the header walk reads it from
    /// the parser's events.
    fn raw(value: &Value) -> Raw {
        let numbers = |value: Option<&Value>| match value {
            None => Member::Missing,
            Some(Value::Array(items)) => {
                let mut numbers = Numbers::default();
                for item in items {
                    let token = item.as_number().and_then(|number| number.token());
                    numbers.push(token.and_then(|token| integer(token.as_bytes())));
                }
                Member::Read(numbers)
            }
            Some(_) => Member::Wrong,
        };
        Raw {
            dtype: match value.get("dtype") {
                None => Member::Missing,
                Some(Value::String(name)) => Member::Read(Dtype::parse(name).ok_or(0..name.len())),
                Some(_) => Member::Wrong,
            },
            shape: numbers(value.get("shape")),
            offsets: numbers(value.get("data_offsets")),
        }
    }

    fn check_text(text: &str, payload: u64) -> Checked {
        let value = parse_value(text.as_bytes(), Limits::NONE).unwrap();
        check(&raw(&value), payload)
            .map(|entry| (entry.dtype, entry.shape().to_vec(), entry.begin, entry.end))
    }

    fn reason(text: &str, payload: u64) -> Reason {
        check_text(text, payload).unwrap_err().0
    }

    fn message(text: &str, payload: u64) -> &'static str {
        check_text(text, payload).unwrap_err().1
    }

    #[test]
    fn integers() {
        assert_eq!(integer(b"0"), Some(0));
        assert_eq!(integer(b"18446744073709551615"), Some(u64::MAX));
        for token in [
            &b""[..],
            b"18446744073709551616",
            b"-0",
            b"1.0",
            b"1e0",
            b"+1",
            b" 1",
        ] {
            assert_eq!(integer(token), None, "{token:?}");
        }
    }

    #[test]
    fn reads_an_entry() {
        assert_eq!(
            check_text(
                r#"{"dtype":"BF16","shape":[2,3],"data_offsets":[4,16]}"#,
                16
            ),
            Ok((Dtype::BF16, vec![2, 3], 4, 16))
        );
        // A scalar, extra members, and members in any order.
        assert_eq!(
            check_text(
                r#"{"data_offsets":[0,4],"x":[],"shape":[],"dtype":"F32"}"#,
                4
            ),
            Ok((Dtype::F32, vec![], 0, 4))
        );
        // Sub-byte types fill whole bytes.
        assert_eq!(
            check_text(r#"{"dtype":"F4","shape":[6],"data_offsets":[0,3]}"#, 3),
            Ok((Dtype::F4, vec![6], 0, 3))
        );
    }

    #[test]
    fn reads_tensors_without_elements() {
        // Equal offsets anywhere in the payload, up to its end.
        for (shape, offset) in [("[0]", 0), ("[2,0]", 3), ("[0,4]", 8)] {
            let text =
                format!(r#"{{"dtype":"F32","shape":{shape},"data_offsets":[{offset},{offset}]}}"#);
            assert!(check_text(&text, 8).is_ok(), "{text}");
        }
        // The other dimensions keep their bound, but not the payload's.
        let wide = r#"{"dtype":"C64","shape":[2147483647,2147483647,0,2147483647,2147483647],"data_offsets":[0,0]}"#;
        assert_eq!(
            check_text(wide, 0),
            Ok((
                Dtype::C64,
                vec![2147483647, 2147483647, 0, 2147483647, 2147483647],
                0,
                0
            ))
        );
        // Multiplied in order, they must fit 64 bits up to the first 0.
        let first =
            r#"{"dtype":"U8","shape":[0,2147483647,2147483647,2147483647],"data_offsets":[0,0]}"#;
        assert!(check_text(first, 0).is_ok());
        let late =
            r#"{"dtype":"U8","shape":[2147483647,2147483647,2147483647,0],"data_offsets":[0,0]}"#;
        assert_eq!(
            check_text(late, 0),
            Err((
                Reason::Dimension,
                "shape's element count overflows 64 bits before its 0 dimension"
            ))
        );
        for (text, expected) in [
            (
                r#"{"dtype":"U8","shape":[0,2147483648],"data_offsets":[0,0]}"#,
                Reason::Dimension,
            ),
            (
                r#"{"dtype":"U8","shape":[0,1,1,1,1,1],"data_offsets":[0,0]}"#,
                Reason::Layout,
            ),
            (
                r#"{"dtype":"U8","shape":[0],"data_offsets":[9,9]}"#,
                Reason::Extent,
            ),
            (
                r#"{"dtype":"U8","shape":[0],"data_offsets":[0,1]}"#,
                Reason::Extent,
            ),
            (
                r#"{"dtype":"U8","shape":[0],"data_offsets":[1,0]}"#,
                Reason::Extent,
            ),
        ] {
            assert_eq!(reason(text, 8), expected, "{text}");
        }
    }

    #[test]
    fn rejects_bad_types() {
        for text in [
            r#"{"dtype":"bf16","shape":[1],"data_offsets":[0,2]}"#,
            r#"{"dtype":2,"shape":[1],"data_offsets":[0,2]}"#,
            r#"{"shape":[1],"data_offsets":[0,2]}"#,
        ] {
            assert_eq!(reason(text, 2), Reason::Dtype, "{text}");
        }
    }

    #[test]
    fn rejects_bad_layouts() {
        for text in [
            r#"{"dtype":"U8","data_offsets":[0,1]}"#,
            r#"{"dtype":"U8","shape":[1]}"#,
            r#"{"dtype":"U8","shape":1,"data_offsets":[0,1]}"#,
            r#"{"dtype":"U8","shape":[1],"data_offsets":[0]}"#,
            r#"{"dtype":"U8","shape":[1],"data_offsets":[0,1,1]}"#,
            r#"{"dtype":"U8","shape":[1],"data_offsets":{"0":1}}"#,
            r#"{"dtype":"U8","shape":[1,1,1,1,1,1],"data_offsets":[0,1]}"#,
            r#"{"dtype":"U8","shape":[1,1,1,1,1,1,1,1],"data_offsets":[0,1]}"#,
        ] {
            assert_eq!(reason(text, 1), Reason::Layout, "{text}");
        }
        assert!(
            check_text(
                r#"{"dtype":"U8","shape":[1,1,1,1,1],"data_offsets":[0,1]}"#,
                1
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_bad_dimensions() {
        for shape in [
            "-1",
            "1.0",
            "1e0",
            "\"1\"",
            "null",
            "[1]",
            "2147483648",
            "18446744073709551616",
        ] {
            let text = format!(r#"{{"dtype":"U8","shape":[{shape}],"data_offsets":[0,1]}}"#);
            assert_eq!(reason(&text, u64::MAX), Reason::Dimension, "{shape}");
        }
        let largest = r#"{"dtype":"U8","shape":[2147483647],"data_offsets":[0,2147483647]}"#;
        assert!(check_text(largest, 1 << 31).is_ok());
        // A shape larger than the payload, even when the product overflows.
        assert_eq!(
            reason(r#"{"dtype":"F32","shape":[2],"data_offsets":[0,8]}"#, 7),
            Reason::Dimension
        );
        let huge = r#"{"dtype":"C64","shape":[2147483647,2147483647,2147483647,2147483647,2147483647],"data_offsets":[0,1]}"#;
        assert_eq!(reason(huge, u64::MAX), Reason::Dimension);
    }

    #[test]
    fn rejects_bad_extents() {
        for offsets in [
            "[0,3]",
            "[1,4]",
            "[0,1]",
            "[2,2]",
            "[2,0]",
            "[0.0,2]",
            "[0,2e0]",
            "[-0,2]",
            "[0,\"2\"]",
            "[0,[2]]",
            "[0,18446744073709551616]",
        ] {
            let text = format!(r#"{{"dtype":"U16","shape":[1],"data_offsets":{offsets}}}"#);
            assert_eq!(reason(&text, 4), Reason::Extent, "{offsets}");
        }
        // Beyond the payload.
        assert_eq!(
            reason(r#"{"dtype":"U16","shape":[1],"data_offsets":[4,6]}"#, 5),
            Reason::Extent
        );
        assert!(check_text(r#"{"dtype":"U16","shape":[1],"data_offsets":[4,6]}"#, 6).is_ok());
        // Bits that do not fill whole bytes.
        assert_eq!(
            reason(r#"{"dtype":"F4","shape":[3],"data_offsets":[0,2]}"#, 2),
            Reason::Extent
        );
        assert_eq!(
            reason(r#"{"dtype":"F6_E2M3","shape":[2],"data_offsets":[0,2]}"#, 2),
            Reason::Extent
        );
    }

    #[test]
    fn says_what_is_wrong() {
        for (text, payload, expected) in [
            (r#"{"shape":[1],"data_offsets":[0,1]}"#, 1, "missing dtype"),
            (
                r#"{"dtype":1,"shape":[1],"data_offsets":[0,1]}"#,
                1,
                "dtype is not a string",
            ),
            (
                r#"{"dtype":"Q4","shape":[1],"data_offsets":[0,1]}"#,
                1,
                "unsupported dtype",
            ),
            (r#"{"dtype":"U8","data_offsets":[0,1]}"#, 1, "missing shape"),
            (
                r#"{"dtype":"U8","shape":1,"data_offsets":[0,1]}"#,
                1,
                "shape is not an array",
            ),
            (r#"{"dtype":"U8","shape":[1]}"#, 1, "missing data_offsets"),
            (
                r#"{"dtype":"U8","shape":[1],"data_offsets":{}}"#,
                1,
                "data_offsets is not an array",
            ),
            (
                r#"{"dtype":"U8","shape":[1,1,1,1,1,1],"data_offsets":[0,1]}"#,
                1,
                "shape has more than 5 dimensions",
            ),
            (
                r#"{"dtype":"U8","shape":[1],"data_offsets":[0]}"#,
                1,
                "data_offsets does not hold 2 values",
            ),
            (
                r#"{"dtype":"U8","shape":[-1],"data_offsets":[0,1]}"#,
                1,
                "shape holds a dimension that is not an integer from 0 to 2147483647",
            ),
            (
                r#"{"dtype":"U8","shape":[2],"data_offsets":[0,2]}"#,
                1,
                "shape holds more bytes than the payload",
            ),
            (
                r#"{"dtype":"U8","shape":[1],"data_offsets":[0,1.0]}"#,
                1,
                "data_offsets are not unsigned integers",
            ),
            (
                r#"{"dtype":"F4","shape":[1],"data_offsets":[0,1]}"#,
                1,
                "shape does not fill whole bytes",
            ),
            (
                r#"{"dtype":"U8","shape":[1],"data_offsets":[1,0]}"#,
                1,
                "data_offsets decrease",
            ),
            (
                r#"{"dtype":"U8","shape":[1],"data_offsets":[1,2]}"#,
                1,
                "data_offsets end past the payload",
            ),
            (
                r#"{"dtype":"U8","shape":[1],"data_offsets":[0,2]}"#,
                2,
                "data_offsets do not span exactly the shape's bytes",
            ),
        ] {
            assert_eq!(message(text, payload), expected, "{text}");
        }
        // The message names the largest dimension.
        let large = r#"{"dtype":"U8","shape":[2147483648],"data_offsets":[0,1]}"#;
        assert!(message(large, 1).ends_with(&MAX_DIMENSION.to_string()));
        // Equal offsets span no bytes.
        assert_eq!(
            message(r#"{"dtype":"U8","shape":[1],"data_offsets":[1,1]}"#, 1),
            "data_offsets do not span exactly the shape's bytes"
        );
    }

    #[test]
    fn checks_type_first() {
        // An entry with several faults reports the first in check order.
        assert_eq!(reason(r#"{"dtype":"X","shape":[-1]}"#, 8), Reason::Dtype);
        assert_eq!(reason(r#"{"dtype":"U8","shape":[-1]}"#, 8), Reason::Layout);
        assert_eq!(
            reason(r#"{"dtype":"U8","shape":[-1],"data_offsets":[1,0]}"#, 8),
            Reason::Dimension
        );
    }
}

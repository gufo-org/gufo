//! The element types the safetensors format names, and their sizes in bits.

/// A tensor's element type, as in safetensors 0.8. The values are those of
/// `GufoSafetensorsDtype` in include/gufo/safetensors.h; new types take the
/// next values, so existing ones never change.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bool = 0,
    F4 = 1,
    F6E2M3 = 2,
    F6E3M2 = 3,
    U8 = 4,
    I8 = 5,
    F8E5M2 = 6,
    F8E4M3 = 7,
    F8E8M0 = 8,
    I16 = 9,
    U16 = 10,
    F16 = 11,
    BF16 = 12,
    I32 = 13,
    U32 = 14,
    F32 = 15,
    C64 = 16,
    F64 = 17,
    I64 = 18,
    U64 = 19,
    F8E4M3Fnuz = 20,
    F8E5M2Fnuz = 21,
}

/// Each type with its name in the header and its size in bits.
const TYPES: [(Dtype, &str, u64); 22] = [
    (Dtype::Bool, "BOOL", 8),
    (Dtype::F4, "F4", 4),
    (Dtype::F6E2M3, "F6_E2M3", 6),
    (Dtype::F6E3M2, "F6_E3M2", 6),
    (Dtype::U8, "U8", 8),
    (Dtype::I8, "I8", 8),
    (Dtype::F8E5M2, "F8_E5M2", 8),
    (Dtype::F8E4M3, "F8_E4M3", 8),
    (Dtype::F8E8M0, "F8_E8M0", 8),
    (Dtype::I16, "I16", 16),
    (Dtype::U16, "U16", 16),
    (Dtype::F16, "F16", 16),
    (Dtype::BF16, "BF16", 16),
    (Dtype::I32, "I32", 32),
    (Dtype::U32, "U32", 32),
    (Dtype::F32, "F32", 32),
    (Dtype::C64, "C64", 64),
    (Dtype::F64, "F64", 64),
    (Dtype::I64, "I64", 64),
    (Dtype::U64, "U64", 64),
    (Dtype::F8E4M3Fnuz, "F8_E4M3FNUZ", 8),
    (Dtype::F8E5M2Fnuz, "F8_E5M2FNUZ", 8),
];

impl Dtype {
    /// The type a header names; names are case-sensitive.
    pub fn parse(name: &str) -> Option<Self> {
        TYPES
            .iter()
            .find(|(_, spelled, _)| *spelled == name)
            .map(|(dtype, _, _)| *dtype)
    }

    /// The type whose value is `value`, if any.
    pub fn from_value(value: u32) -> Option<Self> {
        let index = usize::try_from(value).ok()?;
        TYPES.get(index).map(|(dtype, _, _)| *dtype)
    }

    pub fn name(self) -> &'static str {
        TYPES[self as usize].1
    }

    pub fn bits(self) -> u64 {
        TYPES[self as usize].2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_in_value_order() {
        for (index, (dtype, name, _)) in TYPES.iter().enumerate() {
            assert_eq!(*dtype as usize, index);
            assert_eq!(Dtype::parse(name), Some(*dtype));
            assert_eq!(dtype.name(), *name);
            assert_eq!(Dtype::from_value(index as u32), Some(*dtype));
        }
        assert_eq!(Dtype::from_value(TYPES.len() as u32), None);
        assert_eq!(Dtype::from_value(u32::MAX), None);
    }

    #[test]
    fn names_are_exact() {
        for name in ["bf16", "Bf16", "BF16 ", "", "F8", "Q8_0", "F8_E4M3FN"] {
            assert_eq!(Dtype::parse(name), None, "{name:?}");
        }
    }

    #[test]
    fn sizes() {
        assert_eq!(Dtype::F4.bits(), 4);
        assert_eq!(Dtype::F6E3M2.bits(), 6);
        assert_eq!(Dtype::BF16.bits(), 16);
        assert_eq!(Dtype::C64.bits(), 64);
        assert_eq!(Dtype::F8E4M3Fnuz.bits(), 8);
        assert_eq!(Dtype::F8E5M2Fnuz.bits(), 8);
    }
}

//! Token pieces can split a Unicode scalar or contain malformed byte
//! fallback output, as C++ `core::Utf8Decoder` reads them: incomplete valid
//! prefixes wait for the next piece, while malformed subsequences become
//! U+FFFD.

use crate::config::Error;

const REPLACEMENT: &[u8] = "\u{fffd}".as_bytes();

#[derive(Debug, Clone, Default)]
pub struct Decoder {
    pending: Vec<u8>,
}

impl Decoder {
    /// Appends the complete scalars of `pending + bytes` to `output`; with
    /// `last`, an incomplete tail becomes U+FFFD.
    pub fn push(&mut self, bytes: &[u8], last: bool, output: &mut Vec<u8>) -> Result<(), Error> {
        self.pending.try_reserve(bytes.len())?;
        self.pending.extend_from_slice(bytes);
        output.try_reserve(self.pending.len() + self.pending.len() / 2 + 3)?;
        let pending = &self.pending;
        let mut offset = 0;
        while offset < pending.len() {
            let lead = pending[offset];
            if lead < 0x80 {
                output.push(lead);
                offset += 1;
                continue;
            }
            let length = match lead {
                0xC2..=0xDF => 2,
                0xE0..=0xEF => 3,
                0xF0..=0xF4 => 4,
                _ => 0,
            };
            if length == 0 {
                output.try_reserve(REPLACEMENT.len())?;
                output.extend_from_slice(REPLACEMENT);
                offset += 1;
                continue;
            }
            let mut prefix = 1;
            while prefix < length && offset + prefix < pending.len() {
                let byte = pending[offset + prefix];
                if !(0x80..=0xBF).contains(&byte)
                    || (prefix == 1
                        && ((lead == 0xE0 && byte < 0xA0)
                            || (lead == 0xED && byte > 0x9F)
                            || (lead == 0xF0 && byte < 0x90)
                            || (lead == 0xF4 && byte > 0x8F)))
                {
                    break;
                }
                prefix += 1;
            }
            if prefix == length {
                output.try_reserve(length)?;
                output.extend_from_slice(&pending[offset..offset + length]);
                offset += length;
            } else if offset + prefix == pending.len() && !last {
                break;
            } else {
                output.try_reserve(REPLACEMENT.len())?;
                output.extend_from_slice(REPLACEMENT);
                offset += prefix;
            }
        }
        self.pending.drain(..offset);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Decoder;

    fn decode(pieces: &[&[u8]]) -> Vec<u8> {
        let mut decoder = Decoder::default();
        let mut out = Vec::new();
        for piece in pieces {
            decoder.push(piece, false, &mut out).unwrap();
        }
        decoder.push(&[], true, &mut out).unwrap();
        out
    }

    #[test]
    fn holds_split_scalars_and_replaces_malformed_bytes() {
        assert_eq!(decode(&[b"\xE2\x82", b"\xAC"]), "€".as_bytes());
        assert_eq!(decode(&[b"a\xFFb"]), "a\u{fffd}b".as_bytes());
        assert_eq!(decode(&[b"\xE2\x82"]), "\u{fffd}".as_bytes());
        assert_eq!(
            decode(&[b"\xED\xA0\x80"]),
            "\u{fffd}\u{fffd}\u{fffd}".as_bytes()
        );
        assert_eq!(decode(&[b"\xF0\x9F", b"\x98", b"\x80!"]), "😀!".as_bytes());
    }
}

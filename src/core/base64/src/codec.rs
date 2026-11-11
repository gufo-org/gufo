//! Standard base64 (RFC 4648, section 4): the alphabet `A-Z a-z 0-9 + /`
//! with `=` padding.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Not a digit; `=` included.
const INVALID: u8 = 0xff;

/// The value of each byte as a digit, or `INVALID`.
const DIGITS: [u8; 256] = {
    let mut table = [INVALID; 256];
    let mut i = 0;
    while i < ALPHABET.len() {
        table[ALPHABET[i] as usize] = i as u8;
        i += 1;
    }
    table
};

/// Why `decode` rejected a text. Quads are read in order and each quad's
/// characters in order; the first problem is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Not a whole number of four-character quads.
    Length,
    /// `=` among a quad's first two characters, or in a quad before the last.
    Padding,
    /// A character after `=`.
    AfterPadding,
    /// A byte outside the alphabet.
    Character,
    /// `canonical` only: the padding hides nonzero bits.
    NonCanonical,
    /// More than `max_decoded` bytes, reported once every quad is read.
    TooLong,
}

/// The encoded length of `len` bytes, or `None` if it overflows.
pub fn encoded_len(len: usize) -> Option<usize> {
    len.div_ceil(3).checked_mul(4)
}

/// Encodes `bytes` into `out`, which holds exactly `encoded_len` characters.
pub fn encode(bytes: &[u8], out: &mut [u8]) {
    let mut quads = out.chunks_exact_mut(4);
    for (group, quad) in bytes.chunks(3).zip(&mut quads) {
        let at = |i: usize| group.get(i).copied().map_or(0, u32::from);
        let word = at(0) << 16 | at(1) << 8 | at(2);
        let digit = |shift: u32| ALPHABET[(word >> shift) as usize & 63];
        quad[0] = digit(18);
        quad[1] = digit(12);
        quad[2] = if group.len() > 1 { digit(6) } else { b'=' };
        quad[3] = if group.len() > 2 { digit(0) } else { b'=' };
    }
}

/// The most bytes `decode` writes for a text of `len` characters.
pub fn decoded_capacity(len: usize) -> usize {
    len / 4 * 3
}

/// Decodes `text` into the front of `out`, which holds at least
/// `decoded_capacity(text.len())` bytes, and returns the decoded length. An
/// empty text decodes to nothing. With `canonical`, the bits that padding
/// hides must be zero.
pub fn decode(
    text: &[u8],
    canonical: bool,
    max_decoded: usize,
    out: &mut [u8],
) -> Result<usize, Error> {
    if !text.len().is_multiple_of(4) {
        return Err(Error::Length);
    }
    let Some(last_start) = text.len().checked_sub(4) else {
        return Ok(0);
    };
    let (body, last) = text.split_at(last_start);
    let mut written = 0;
    for (quad, bytes) in body.chunks_exact(4).zip(out.chunks_exact_mut(3)) {
        let digits = [quad[0], quad[1], quad[2], quad[3]].map(|c| DIGITS[usize::from(c)]);
        // Digits are below 64; `INVALID` sets the top bits.
        if (digits[0] | digits[1] | digits[2] | digits[3]) & 0xc0 != 0 {
            return Err(read_quad(quad, false).err().unwrap_or(Error::Character));
        }
        let word = u32::from(digits[0]) << 18
            | u32::from(digits[1]) << 12
            | u32::from(digits[2]) << 6
            | u32::from(digits[3]);
        bytes.copy_from_slice(&[(word >> 16) as u8, (word >> 8) as u8, word as u8]);
        written += 3;
    }
    let (word, padding) = read_quad(last, true)?;
    if canonical && word & ((1 << (8 * padding)) - 1) != 0 {
        return Err(Error::NonCanonical);
    }
    let count = 3 - padding;
    let tail = [(word >> 16) as u8, (word >> 8) as u8, word as u8];
    out[written..written + count].copy_from_slice(&tail[..count]);
    written += count;
    if written > max_decoded {
        return Err(Error::TooLong);
    }
    Ok(written)
}

/// One quad, character by character; `last` allows padding. Returns the
/// 24-bit word, with zeros in place of padding, and the padding count.
fn read_quad(quad: &[u8], last: bool) -> Result<(u32, usize), Error> {
    let mut word = 0;
    let mut padding = 0;
    for (index, &c) in quad.iter().enumerate() {
        if c == b'=' {
            if index < 2 || !last {
                return Err(Error::Padding);
            }
            padding += 1;
            word <<= 6;
        } else if padding != 0 {
            return Err(Error::AfterPadding);
        } else {
            let digit = DIGITS[usize::from(c)];
            if digit == INVALID {
                return Err(Error::Character);
            }
            word = word << 6 | u32::from(digit);
        }
    }
    Ok((word, padding))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded(bytes: &[u8]) -> Vec<u8> {
        let mut out = vec![0; encoded_len(bytes.len()).unwrap()];
        encode(bytes, &mut out);
        out
    }

    fn decoded(text: &[u8], canonical: bool, max: usize) -> Result<Vec<u8>, Error> {
        let mut out = vec![0; decoded_capacity(text.len())];
        let len = decode(text, canonical, max, &mut out)?;
        out.truncate(len);
        Ok(out)
    }

    #[test]
    fn rfc4648_vectors() {
        for (plain, text) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(encoded(plain.as_bytes()), text.as_bytes());
            assert_eq!(
                decoded(text.as_bytes(), true, usize::MAX).unwrap(),
                plain.as_bytes()
            );
        }
    }

    #[test]
    fn every_length_round_trips() {
        let bytes: Vec<u8> = (0..=255).collect();
        for len in 0..bytes.len() {
            let text = encoded(&bytes[..len]);
            assert_eq!(decoded(&text, true, len).unwrap(), &bytes[..len]);
            if len > 0 {
                assert_eq!(decoded(&text, true, len - 1), Err(Error::TooLong));
            }
        }
    }

    #[test]
    fn errors_in_reading_order() {
        let error = |text: &str, canonical| decoded(text.as_bytes(), canonical, usize::MAX).err();
        assert_eq!(error("Zm9", true), Some(Error::Length));
        assert_eq!(error("=AAA", true), Some(Error::Padding));
        assert_eq!(error("A=AA", true), Some(Error::Padding));
        assert_eq!(error("Zg==Zm9v", true), Some(Error::Padding));
        assert_eq!(error("Zg=v", true), Some(Error::AfterPadding));
        assert_eq!(error("Zg=!", true), Some(Error::AfterPadding));
        assert_eq!(error("Zm-v", true), Some(Error::Character));
        assert_eq!(error("Zm9v!AAA", true), Some(Error::Character));
        // The first problem in the quad wins.
        assert_eq!(error("A!=A", true), Some(Error::Character));
        assert_eq!(error("!=AA", true), Some(Error::Character));
        assert_eq!(error("Zh==", true), Some(Error::NonCanonical));
        assert_eq!(error("Zm9=", true), Some(Error::NonCanonical));
        assert_eq!(decoded(b"Zh==", false, usize::MAX).unwrap(), b"f");
        // A later error outranks the length limit.
        assert_eq!(
            decoded(b"Zm9vZm9v!AAA", true, 2).err(),
            Some(Error::Character)
        );
    }
}

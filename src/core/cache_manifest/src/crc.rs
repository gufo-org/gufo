//! CRC-64/ECMA-182: polynomial 0x42f0e1eba9ea3693, no reflection and no
//! final xor. The seed is the CRC of the preceding bytes (zero at the start),
//! so a payload can be checked piece by piece.

const POLYNOMIAL: u64 = 0x42f0_e1eb_a9ea_3693;

/// Inputs from this length on, which fill the four folding lanes, fold with
/// carry-less multiplication when the CPU has it. With the final Barrett
/// reduction, folding 64 bytes is already about twice as fast as the table.
pub const FOLD_MIN_BYTES: usize = 64;

/// Slicing-by-8: `TABLE[0]` advances the CRC by one byte, `TABLE[k]` by a
/// byte followed by `k` zero bytes.
static TABLE: [[u64; 256]; 8] = table();

const fn table() -> [[u64; 256]; 8] {
    let mut t = [[0u64; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut n = (i as u64) << 56;
        let mut bit = 0;
        while bit < 8 {
            n = (n << 1) ^ if n >> 63 != 0 { POLYNOMIAL } else { 0 };
            bit += 1;
        }
        t[0][i] = n;
        i += 1;
    }
    let mut slice = 1;
    while slice < 8 {
        let mut i = 0;
        while i < 256 {
            let previous = t[slice - 1][i];
            t[slice][i] = (previous << 8) ^ t[0][(previous >> 56) as usize];
            i += 1;
        }
        slice += 1;
    }
    t
}

/// The CRC of `bytes` after the bytes whose CRC is `crc`.
pub fn checksum(bytes: &[u8], crc: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if let Some(crc) = fold::checksum(bytes, crc) {
        return crc;
    }
    table_checksum(bytes, crc)
}

/// The table path alone, for inputs of any length.
pub fn table_checksum(bytes: &[u8], mut crc: u64) -> u64 {
    // Eight bytes at a time break the per-byte dependency chain.
    let mut words = bytes.chunks_exact(8);
    for word in &mut words {
        let word: [u8; 8] = word.try_into().expect("eight-byte chunk");
        crc ^= u64::from_be_bytes(word);
        // The bytes at the ends need the fewest instructions to look up, so
        // their lookups start the XOR chain.
        crc = TABLE[7][(crc >> 56) as usize]
            ^ TABLE[0][(crc & 255) as usize]
            ^ TABLE[6][((crc >> 48) & 255) as usize]
            ^ TABLE[5][((crc >> 40) & 255) as usize]
            ^ TABLE[4][((crc >> 32) & 255) as usize]
            ^ TABLE[3][((crc >> 24) & 255) as usize]
            ^ TABLE[2][((crc >> 16) & 255) as usize]
            ^ TABLE[1][((crc >> 8) & 255) as usize];
    }
    for &b in words.remainder() {
        crc = (crc << 8) ^ TABLE[0][((crc >> 56) ^ u64::from(b)) as usize];
    }
    crc
}

/// x^exponent modulo the polynomial.
#[cfg(target_arch = "x86_64")]
const fn power(mut exponent: u32) -> u64 {
    let mut result = 1u64;
    while exponent > 0 {
        result = (result << 1) ^ if result >> 63 != 0 { POLYNOMIAL } else { 0 };
        exponent -= 1;
    }
    result
}

/// The low 64 bits of floor(x^128 / P), whose x^64 term is implied: the
/// Barrett constant.
#[cfg(target_arch = "x86_64")]
const fn barrett() -> u64 {
    // Long division: x^128 minus x^64 * P leaves POLYNOMIAL * x^64; each
    // further quotient bit cancels the remainder's x^(64 + i) term.
    let p = (1u128 << 64) | POLYNOMIAL as u128;
    let mut remainder = (POLYNOMIAL as u128) << 64;
    let mut quotient = 0u64;
    let mut i = 64;
    while i > 0 {
        i -= 1;
        if (remainder >> (64 + i)) & 1 != 0 {
            quotient |= 1 << i;
            remainder ^= p << i;
        }
    }
    quotient
}

#[cfg(target_arch = "x86_64")]
pub mod fold {
    //! Folding with PCLMULQDQ in GF(2), modulo the same polynomial. A
    //! 128-bit lane's low limb advances by x^128 (or x^512 across four
    //! lanes) and its high limb another 64 bits. Four independent lanes hide
    //! the multiply latency. Their merge is reduced to 64 bits by Barrett
    //! reduction; the table takes the partial suffix.

    use super::{FOLD_MIN_BYTES, POLYNOMIAL, barrett, power, table_checksum};
    use std::arch::x86_64::{
        __m128i, _MM_HINT_T0, _mm_clmulepi64_si128, _mm_cvtsi128_si64, _mm_loadu_si128,
        _mm_prefetch, _mm_set_epi64x, _mm_setr_epi8, _mm_shuffle_epi8, _mm_slli_si128,
        _mm_srli_si128, _mm_xor_si128,
    };
    use std::sync::atomic::{AtomicU8, Ordering};

    const POWERS: (u64, u64) = (power(192), power(128));
    const WIDE_POWERS: (u64, u64) = (power(576), power(512));
    /// x^128 mod P and the Barrett constant, for the final reduction.
    const REDUCE: (u64, u64) = (power(128), barrett());

    /// 0 until detected, then 1 without and 2 with the CPU features. A local
    /// flag keeps the check out of the standard library's code and data,
    /// which a checksum of streamed payloads would otherwise often miss.
    static FOLDING: AtomicU8 = AtomicU8::new(0);

    /// Whether this CPU can fold.
    pub fn available() -> bool {
        match FOLDING.load(Ordering::Relaxed) {
            0 => {
                let available = std::arch::is_x86_feature_detected!("pclmulqdq")
                    && std::arch::is_x86_feature_detected!("ssse3");
                FOLDING.store(1 + u8::from(available), Ordering::Relaxed);
                available
            }
            state => state == 2,
        }
    }

    /// The folding path, or `None` when the CPU cannot fold or `bytes` is
    /// shorter than [`FOLD_MIN_BYTES`].
    #[allow(unsafe_code)]
    pub fn checksum(bytes: &[u8], crc: u64) -> Option<u64> {
        if bytes.len() < FOLD_MIN_BYTES || !available() {
            return None;
        }
        // SAFETY: the CPU has the target features `fold` enables.
        Some(unsafe { fold(bytes, crc) })
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq,ssse3")]
    fn reverse(value: __m128i) -> __m128i {
        _mm_shuffle_epi8(
            value,
            _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
        )
    }

    /// Loads 16 bytes as a polynomial whose low limb is in the low lane: the
    /// big-endian byte order reversed.
    #[allow(unsafe_code)]
    #[inline]
    #[target_feature(enable = "pclmulqdq,ssse3")]
    fn load(bytes: &[u8; 16]) -> __m128i {
        // SAFETY: `bytes` holds the 16 bytes the unaligned load reads.
        reverse(unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) })
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq,ssse3")]
    fn advance(previous: __m128i, powers: __m128i, next: __m128i) -> __m128i {
        _mm_xor_si128(
            _mm_xor_si128(
                _mm_clmulepi64_si128::<0x00>(previous, powers),
                _mm_clmulepi64_si128::<0x11>(previous, powers),
            ),
            next,
        )
    }

    /// The CRC of the 16 bytes that `value` holds as a polynomial, with a
    /// zero seed: value * x^64 mod P, by multiplication alone, so no table
    /// lookup waits on memory.
    #[inline]
    #[target_feature(enable = "pclmulqdq,ssse3")]
    fn reduce(value: __m128i) -> u64 {
        let constants = _mm_set_epi64x(REDUCE.1 as i64, REDUCE.0 as i64);
        // high * (x^128 mod P) + low * x^64 keeps the value modulo P in 128
        // bits.
        let t = _mm_xor_si128(
            _mm_clmulepi64_si128::<0x01>(value, constants),
            _mm_slli_si128::<8>(value),
        );
        let high = _mm_srli_si128::<8>(t);
        // Barrett: the quotient of high * x^64 by P, then its remainder.
        let quotient = _mm_xor_si128(
            _mm_srli_si128::<8>(_mm_clmulepi64_si128::<0x10>(high, constants)),
            high,
        );
        let product = _mm_clmulepi64_si128::<0x00>(quotient, _mm_set_epi64x(0, POLYNOMIAL as i64));
        (_mm_cvtsi128_si64(t) ^ _mm_cvtsi128_si64(product)) as u64
    }

    fn block(bytes: &[u8], at: usize) -> &[u8; 16] {
        bytes[at..at + 16].try_into().expect("16-byte block")
    }

    #[allow(unsafe_code)]
    #[target_feature(enable = "pclmulqdq,ssse3")]
    fn fold(mut bytes: &[u8], crc: u64) -> u64 {
        let powers = _mm_set_epi64x(POWERS.0 as i64, POWERS.1 as i64);
        let wide_powers = _mm_set_epi64x(WIDE_POWERS.0 as i64, WIDE_POWERS.1 as i64);
        let mut a = _mm_xor_si128(load(block(bytes, 0)), _mm_set_epi64x(crc as i64, 0));
        let mut b = load(block(bytes, 16));
        let mut c = load(block(bytes, 32));
        let mut d = load(block(bytes, 48));
        bytes = &bytes[64..];
        while bytes.len() >= 64 {
            // Payloads stream from memory; asking four steps ahead keeps short
            // pieces from waiting on the hardware prefetcher.
            // SAFETY: a prefetch never faults, wherever it points.
            unsafe { _mm_prefetch::<_MM_HINT_T0>(bytes.as_ptr().wrapping_add(256).cast()) };
            // All multiplies first: the scheduler keeps this order, which
            // lets them start before the loads and shuffles of the step.
            // Written out, they stay inline in unoptimized test builds too.
            let (a0, a1) = (
                _mm_clmulepi64_si128::<0x00>(a, wide_powers),
                _mm_clmulepi64_si128::<0x11>(a, wide_powers),
            );
            let (b0, b1) = (
                _mm_clmulepi64_si128::<0x00>(b, wide_powers),
                _mm_clmulepi64_si128::<0x11>(b, wide_powers),
            );
            let (c0, c1) = (
                _mm_clmulepi64_si128::<0x00>(c, wide_powers),
                _mm_clmulepi64_si128::<0x11>(c, wide_powers),
            );
            let (d0, d1) = (
                _mm_clmulepi64_si128::<0x00>(d, wide_powers),
                _mm_clmulepi64_si128::<0x11>(d, wide_powers),
            );
            a = _mm_xor_si128(_mm_xor_si128(a0, a1), load(block(bytes, 0)));
            b = _mm_xor_si128(_mm_xor_si128(b0, b1), load(block(bytes, 16)));
            c = _mm_xor_si128(_mm_xor_si128(c0, c1), load(block(bytes, 32)));
            d = _mm_xor_si128(_mm_xor_si128(d0, d1), load(block(bytes, 48)));
            bytes = &bytes[64..];
        }
        a = advance(advance(advance(a, powers, b), powers, c), powers, d);
        while bytes.len() >= 16 {
            a = advance(a, powers, load(block(bytes, 0)));
            bytes = &bytes[16..];
        }
        table_checksum(bytes, reduce(a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One bit at a time, straight from the definition.
    fn bitwise(bytes: &[u8], mut crc: u64) -> u64 {
        for &b in bytes {
            crc ^= u64::from(b) << 56;
            for _ in 0..8 {
                crc = (crc << 1) ^ if crc >> 63 != 0 { POLYNOMIAL } else { 0 };
            }
        }
        crc
    }

    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u64 {
            // SplitMix64.
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn barrett_constant() {
        // floor(x^128 / P) * P differs from x^128 by less than P.
        let mu = (1u128 << 64) | u128::from(barrett());
        let p = (1u128 << 64) | u128::from(POLYNOMIAL);
        // Carry-less product of two 65-bit polynomials, as 130 bits.
        let (mut high, mut low) = (0u128, 0u128);
        for i in 0..65 {
            if (mu >> i) & 1 != 0 {
                low ^= p << i;
                if i > 0 {
                    high ^= p >> (128 - i);
                }
            }
        }
        // x^128 + product has degree below 64.
        assert_eq!(high, 1);
        assert_eq!(low >> 64, 0);
    }

    #[test]
    fn check_value() {
        // The catalogue check value of CRC-64/ECMA-182.
        assert_eq!(checksum(b"123456789", 0), 0x6c40_df5f_0b49_7347);
        assert_eq!(checksum(&[], 0), 0);
        assert_eq!(checksum(&[], 0x1234), 0x1234);
    }

    #[test]
    fn paths_agree_over_lengths_seeds_and_offsets() {
        let mut random = Random(7);
        let data: Vec<u8> = (0..9000).map(|_| random.next() as u8).collect();
        for len in (0..600).chain([1023, 1024, 1025, 4096, 8191]) {
            let offset = (random.next() % 9) as usize;
            let bytes = &data[offset..offset + len];
            let seed = if len % 3 == 0 { 0 } else { random.next() };
            let expected = bitwise(bytes, seed);
            assert_eq!(table_checksum(bytes, seed), expected, "table {len}");
            assert_eq!(checksum(bytes, seed), expected, "dispatch {len}");
            #[cfg(target_arch = "x86_64")]
            if len >= 64 && fold::available() {
                assert_eq!(fold::checksum(bytes, seed), Some(expected), "fold {len}");
            }
        }
    }

    #[test]
    fn pieces_chain_through_the_seed() {
        let mut random = Random(11);
        let data: Vec<u8> = (0..20000).map(|_| random.next() as u8).collect();
        let whole = checksum(&data, 0);
        for _ in 0..50 {
            let mut crc = 0;
            let mut rest = &data[..];
            while !rest.is_empty() {
                let n = ((random.next() % 1500) as usize + 1).min(rest.len());
                crc = checksum(&rest[..n], crc);
                rest = &rest[n..];
            }
            assert_eq!(crc, whole);
        }
    }
}

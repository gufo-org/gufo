//! Converts PNG samples to RGBA8 as libpng's simplified API does when asked
//! for `PNG_FORMAT_RGBA`. This decoder replaced libpng 1.6, so every rule
//! below mirrors one of its read transforms:
//!
//! - Palette, low-depth gray and tRNS expand to RGBA; gray copies to RGB.
//! - 8-bit and smaller samples assume sRGB without gAMA or sRGB chunks, and
//!   16-bit samples assume linear light. Color channels convert from that file
//!   gamma to sRGB unless the two are within 5%; alpha never changes.
//! - 16-bit samples round to 8 bits; gamma-corrected ones go through libpng's
//!   16-to-8 table, whose precision follows the sBIT chunk.
//!
//! The png crate decodes the pixels with no transformations. The color chunks
//! are read here, because libpng ignores malformed or misplaced ones that the
//! png crate may still record. iCCP, cHRM and cICP do not change the pixels.

/// `PNG_GAMMA_sRGB_INVERSE`, the file gamma that sRGB and 8-bit input imply.
const SRGB_FILE_GAMMA: i32 = 45455;
/// `PNG_GAMMA_LINEAR`, the default file gamma of 16-bit input.
const LINEAR_GAMMA: i32 = 100_000;
/// `PNG_GAMMA_sRGB`, the screen gamma of 8-bit sRGB output.
const SRGB_SCREEN_GAMMA: i32 = 220_000;
const FP_1: i32 = 100_000;
const GAMMA_THRESHOLD: i32 = 5000;
/// `PNG_MAX_GAMMA_8`: 16-to-8 tables index at most this many high bits.
const MAX_GAMMA_8: u32 = 11;

const IHDR: [u8; 4] = *b"IHDR";
const PLTE: [u8; 4] = *b"PLTE";
const IDAT: [u8; 4] = *b"IDAT";
const TRNS: [u8; 4] = *b"tRNS";
const BKGD: [u8; 4] = *b"bKGD";
const GAMA: [u8; 4] = *b"gAMA";
const SRGB: [u8; 4] = *b"sRGB";
const SBIT: [u8; 4] = *b"sBIT";
const CICP: [u8; 4] = *b"cICP";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Gray,
    Rgb,
    Indexed,
    GrayAlpha,
    Rgba,
}

impl Color {
    fn channels(self) -> usize {
        match self {
            Self::Gray | Self::Indexed => 1,
            Self::GrayAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    fn has_color(self) -> bool {
        matches!(self, Self::Rgb | Self::Indexed | Self::Rgba)
    }
}

/// The libpng state that the color chunks before IDAT leave behind.
#[derive(Debug, Default)]
struct Chunks {
    /// Palette entries kept from PLTE, at most 2^depth for indexed images.
    palette_len: usize,
    /// Indexed: alpha per palette entry. Gray and RGB: the 16-bit key.
    trns: Option<Vec<u8>>,
    /// `png_struct::chunk_gamma`; 0 when no chunk sets it.
    chunk_gamma: i32,
    /// The largest color (or the gray) sBIT value; 0 without sBIT.
    sig_bit: u8,
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

/// Reads the ancillary chunks that change RGBA output with libpng's position,
/// duplicate, length and CRC rules: a rejected chunk is skipped and does not
/// count as seen. The png crate has already accepted the stream, so a
/// malformed chunk structure only stops the walk.
fn read_chunks(bytes: &[u8], color: Color, depth: u8) -> Chunks {
    let mut chunks = Chunks::default();
    let (mut have_plte, mut seen_trns, mut seen_bkgd) = (false, false, false);
    let (mut seen_gama, mut seen_srgb, mut seen_sbit, mut seen_cicp) = (false, false, false, false);
    let mut offset = 8;
    while let Some(&[l0, l1, l2, l3, t0, t1, t2, t3]) = bytes.get(offset..offset + 8) {
        let length = u32::from_be_bytes([l0, l1, l2, l3]) as usize;
        let kind = [t0, t1, t2, t3];
        let Some(end) = length
            .checked_add(12)
            .and_then(|size| offset.checked_add(size))
            .filter(|&end| end <= bytes.len())
        else {
            break;
        };
        if kind == IDAT {
            break;
        }
        let data = &bytes[offset + 8..offset + 8 + length];
        let crc = u32::from_be_bytes([
            bytes[end - 4],
            bytes[end - 3],
            bytes[end - 2],
            bytes[end - 1],
        ]);
        let crc_ok = || crc32(&bytes[offset + 4..end - 4]) == crc;
        // Chunks that must precede PLTE (and IDAT, which ends the walk).
        let before_plte = !have_plte;
        match kind {
            IHDR => {}
            PLTE => {
                // An indexed image's PLTE is critical: the png crate rejects
                // the ones libpng rejects. Elsewhere libpng ignores bad ones;
                // the png crate rejects some of those, and a bad CRC, first.
                if color == Color::Indexed {
                    have_plte = true;
                    chunks.palette_len = (length / 3).min(1 << depth);
                } else if !have_plte
                    && color.has_color()
                    && length <= 768
                    && length % 3 == 0
                    && !seen_trns
                    && !seen_bkgd
                {
                    have_plte = true;
                }
            }
            TRNS if !seen_trns && length <= 256 => {
                let valid = match color {
                    Color::Gray => length == 2,
                    Color::Rgb => length == 6,
                    Color::Indexed => have_plte && length > 0 && length <= chunks.palette_len,
                    Color::GrayAlpha | Color::Rgba => false,
                };
                if valid && crc_ok() {
                    seen_trns = true;
                    chunks.trns = Some(data.to_vec());
                }
            }
            // Only gates a later PLTE, which indexed images never ignore.
            BKGD if color != Color::Indexed && !seen_bkgd && (1..=6).contains(&length) => {
                let valid = match color {
                    Color::Indexed => false,
                    Color::Gray | Color::GrayAlpha => {
                        length == 2
                            && crc_ok()
                            && (depth == 16 || (data[0] == 0 && u32::from(data[1]) < 1 << depth))
                    }
                    Color::Rgb | Color::Rgba => {
                        length == 6
                            && crc_ok()
                            && (depth == 16 || (data[0] == 0 && data[2] == 0 && data[4] == 0))
                    }
                };
                seen_bkgd |= valid;
            }
            GAMA if before_plte && !seen_gama && length == 4 && crc_ok() => {
                let gamma = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
                if let Ok(gamma) = i32::try_from(gamma) {
                    seen_gama = true;
                    if chunks.chunk_gamma == 0 {
                        chunks.chunk_gamma = gamma;
                    }
                }
            }
            SRGB if before_plte && !seen_srgb && length == 1 && crc_ok() && data[0] <= 3 => {
                seen_srgb = true;
                if !seen_cicp || chunks.chunk_gamma == 0 {
                    chunks.chunk_gamma = SRGB_FILE_GAMMA;
                }
            }
            SBIT if before_plte && !seen_sbit && (1..=4).contains(&length) => {
                let (expected, sample_depth) = match color {
                    Color::Indexed => (3, 8),
                    _ => (color.channels(), depth),
                };
                if length == expected
                    && crc_ok()
                    && data.iter().all(|&bits| bits != 0 && bits <= sample_depth)
                {
                    seen_sbit = true;
                    chunks.sig_bit = if color.has_color() {
                        data[0].max(data[1]).max(data[2])
                    } else {
                        data[0]
                    };
                }
            }
            CICP if before_plte && !seen_cicp && length == 4 && crc_ok() => seen_cicp = true,
            _ => {}
        }
        offset = end;
    }
    chunks
}

/// `png_muldiv` with floating-point arithmetic; None on overflow.
fn muldiv(a: i32, times: i32, divisor: i32) -> Option<i32> {
    if divisor == 0 {
        return None;
    }
    if a == 0 || times == 0 {
        return Some(0);
    }
    let r = (f64::from(a) * f64::from(times) / f64::from(divisor) + 0.5).floor();
    (-2_147_483_648.0..=2_147_483_647.0)
        .contains(&r)
        .then_some(r as i32)
}

fn reciprocal(a: i32) -> i32 {
    let r = (1e10 / f64::from(a) + 0.5).floor();
    if (-2_147_483_648.0..=2_147_483_647.0).contains(&r) {
        r as i32
    } else {
        0
    }
}

fn reciprocal2(a: i32, b: i32) -> i32 {
    if a == 0 || b == 0 {
        return 0;
    }
    let r = (1e15 / f64::from(a) / f64::from(b) + 0.5).floor();
    if (-2_147_483_648.0..=2_147_483_647.0).contains(&r) {
        r as i32
    } else {
        0
    }
}

fn gamma_significant(gamma: i32) -> bool {
    !(FP_1 - GAMMA_THRESHOLD..=FP_1 + GAMMA_THRESHOLD).contains(&gamma)
}

fn gamma_8bit_correct(value: u32, gamma: i32) -> u8 {
    if value > 0 && value < 255 {
        let r = (255.0 * (f64::from(value) / 255.0).powf(f64::from(gamma) * 0.00001) + 0.5).floor();
        r as u8
    } else {
        value as u8
    }
}

fn gamma_16bit_correct(value: u32, gamma: i32) -> u16 {
    if value > 0 && value < 65535 {
        let r =
            (65535.0 * (f64::from(value) / 65535.0).powf(f64::from(gamma) * 0.00001) + 0.5).floor();
        r as u16
    } else {
        value as u16
    }
}

/// `png_do_scale_16_to_8`, exact rounding of a 16-bit sample.
fn scale_16(value: u16) -> u8 {
    let [high, low] = value.to_be_bytes();
    let high = i32::from(high);
    (high + (((i32::from(low) - high + 128) * 65535) >> 24)) as u8
}

/// Maps file samples to RGBA8 output samples.
enum Gamma {
    None,
    /// Color samples of 8 bits or fewer, after expansion to 8 bits.
    Table8(Box<[u8; 256]>),
    /// 16-bit color samples shifted right by `shift`, already reduced to 8
    /// bits as libpng's 16-to-8 table and the scale that follows do.
    Table16 {
        shift: u32,
        table: Vec<u8>,
    },
}

impl Gamma {
    fn new(depth: u8, chunks: &Chunks) -> Self {
        let file = match chunks.chunk_gamma {
            0 if depth == 16 => LINEAR_GAMMA,
            0 => SRGB_FILE_GAMMA,
            gamma => gamma,
        };
        let screen = SRGB_SCREEN_GAMMA;
        if muldiv(file, screen, FP_1).is_some_and(|product| !gamma_significant(product)) {
            return Self::None;
        }
        let correction = reciprocal2(screen, file);
        if depth <= 8 {
            let mut table = Box::new([0; 256]);
            for (i, out) in (0u32..).zip(table.iter_mut()) {
                *out = if gamma_significant(correction) {
                    gamma_8bit_correct(i, correction)
                } else {
                    i as u8
                };
            }
            return Self::Table8(table);
        }
        let sig_bit = u32::from(chunks.sig_bit);
        let shift = if sig_bit > 0 && sig_bit < 16 {
            16 - sig_bit
        } else {
            0
        };
        let shift = shift.clamp(16 - MAX_GAMMA_8, 8);
        // png_build_16to8_table: each 8-bit output covers the inputs below
        // the gamma-encoded midpoint to the next output.
        let gamma = reciprocal(correction);
        let max = (1u32 << (16 - shift)) - 1;
        let mut table = vec![255; 1 << (16 - shift)];
        let mut last = 0;
        for i in 0..255u32 {
            let bound = u32::from(gamma_16bit_correct(i * 257 + 128, gamma));
            let bound = ((bound * max + 32768) / 65535 + 1) as usize;
            while last < bound {
                table[last] = i as u8;
                last += 1;
            }
        }
        Self::Table16 { shift, table }
    }

    fn color8(&self, value: u8) -> u8 {
        match self {
            Self::Table8(table) => table[usize::from(value)],
            _ => value,
        }
    }

    fn color16(&self, value: u16) -> u8 {
        match self {
            Self::Table16 { shift, table } => table[usize::from(value >> shift)],
            _ => scale_16(value),
        }
    }
}

/// Converts rows of one image to RGBA8.
pub struct Converter {
    color: Color,
    depth: u8,
    gamma: Gamma,
    /// Indexed images: the gamma-corrected RGBA of each index.
    palette: Box<[[u8; 4]; 256]>,
    /// Gray and RGB: the 16-bit tRNS key, masked to the sample depth.
    key: Option<[u16; 3]>,
}

impl Converter {
    /// `bytes` is the whole PNG and `palette` the PLTE the png crate read.
    pub fn new(bytes: &[u8], color: Color, depth: u8, palette: Option<&[u8]>) -> Self {
        let chunks = read_chunks(bytes, color, depth);
        let gamma = Gamma::new(depth, &chunks);
        let mut rgba = Box::new([[0, 0, 0, 255]; 256]);
        if color == Color::Indexed {
            let entries = palette.unwrap_or_default().as_chunks::<3>().0;
            for (out, rgb) in rgba.iter_mut().zip(entries).take(chunks.palette_len) {
                *out = [
                    gamma.color8(rgb[0]),
                    gamma.color8(rgb[1]),
                    gamma.color8(rgb[2]),
                    255,
                ];
            }
            if let Some(alpha) = &chunks.trns {
                for (out, &a) in rgba.iter_mut().zip(alpha) {
                    out[3] = a;
                }
            }
        }
        let mask = if depth == 16 {
            0xffff
        } else {
            (1u16 << depth) - 1
        };
        let key = match (color, &chunks.trns) {
            (Color::Gray, Some(t)) => {
                let gray = u16::from_be_bytes([t[0], t[1]]) & mask;
                Some([gray; 3])
            }
            (Color::Rgb, Some(t)) => {
                Some([0, 2, 4].map(|i| u16::from_be_bytes([t[i], t[i + 1]]) & mask))
            }
            _ => None,
        };
        Self {
            color,
            depth,
            gamma,
            palette: rgba,
            key,
        }
    }

    /// Converts `out.len() / 4` pixels from a raw row with no filter byte.
    pub fn row(&self, row: &[u8], out: &mut [u8]) {
        let out = out.as_chunks_mut::<4>().0;
        let opaque = |samples: [u16; 3]| self.key != Some(samples);
        match (self.color, self.depth) {
            (Color::Indexed, depth) => {
                for (x, px) in out.iter_mut().enumerate() {
                    *px = self.palette[usize::from(packed(row, x, depth))];
                }
            }
            (Color::Gray, 16) => {
                for (px, s) in out.iter_mut().zip(row.as_chunks::<2>().0) {
                    let v = u16::from_be_bytes(*s);
                    let g = self.gamma.color16(v);
                    *px = [g, g, g, if opaque([v; 3]) { 255 } else { 0 }];
                }
            }
            (Color::Gray, depth) => {
                let scale = match depth {
                    1 => 0xff,
                    2 => 0x55,
                    4 => 0x11,
                    _ => 1,
                };
                for (x, px) in out.iter_mut().enumerate() {
                    let v = packed(row, x, depth);
                    let g = self.gamma.color8(v * scale);
                    *px = [g, g, g, if opaque([u16::from(v); 3]) { 255 } else { 0 }];
                }
            }
            (Color::GrayAlpha, 16) => {
                for (px, s) in out.iter_mut().zip(row.as_chunks::<4>().0) {
                    let g = self.gamma.color16(u16::from_be_bytes([s[0], s[1]]));
                    *px = [g, g, g, scale_16(u16::from_be_bytes([s[2], s[3]]))];
                }
            }
            (Color::GrayAlpha, _) => {
                for (px, s) in out.iter_mut().zip(row.as_chunks::<2>().0) {
                    let g = self.gamma.color8(s[0]);
                    *px = [g, g, g, s[1]];
                }
            }
            (Color::Rgb, 16) => {
                for (px, s) in out.iter_mut().zip(row.as_chunks::<6>().0) {
                    let v = [0, 2, 4].map(|i| u16::from_be_bytes([s[i], s[i + 1]]));
                    let [r, g, b] = v.map(|v| self.gamma.color16(v));
                    *px = [r, g, b, if opaque(v) { 255 } else { 0 }];
                }
            }
            (Color::Rgb, _) => {
                for (px, s) in out.iter_mut().zip(row.as_chunks::<3>().0) {
                    let [r, g, b] = s.map(|v| self.gamma.color8(v));
                    *px = [r, g, b, if opaque(s.map(u16::from)) { 255 } else { 0 }];
                }
            }
            (Color::Rgba, 16) => {
                for (px, s) in out.iter_mut().zip(row.as_chunks::<8>().0) {
                    let [r, g, b] =
                        [0, 2, 4].map(|i| self.gamma.color16(u16::from_be_bytes([s[i], s[i + 1]])));
                    *px = [r, g, b, scale_16(u16::from_be_bytes([s[6], s[7]]))];
                }
            }
            (Color::Rgba, _) => {
                for (px, s) in out.iter_mut().zip(row.as_chunks::<4>().0) {
                    *px = [
                        self.gamma.color8(s[0]),
                        self.gamma.color8(s[1]),
                        self.gamma.color8(s[2]),
                        s[3],
                    ];
                }
            }
        }
    }
}

/// The `x`th sample of a row packed MSB-first at `depth` bits (8 or fewer).
fn packed(row: &[u8], x: usize, depth: u8) -> u8 {
    match depth {
        8 => row[x],
        _ => {
            let bit = x * usize::from(depth);
            let shift = 8 - usize::from(depth) - bit % 8;
            (row[bit / 8] >> shift) & ((1 << depth) - 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_16_rounds_exactly() {
        for v in 0..=u16::MAX {
            let exact = (u32::from(v) * 255 + 32767) / 65535;
            assert_eq!(u32::from(scale_16(v)), exact, "{v}");
        }
    }

    #[test]
    fn crc32_matches_known_value() {
        assert_eq!(crc32(b"IEND"), 0xae42_6082);
    }

    #[test]
    fn srgb_gamma_is_not_significant() {
        let chunks = Chunks::default();
        assert!(matches!(Gamma::new(8, &chunks), Gamma::None));
        let chunks = Chunks {
            chunk_gamma: SRGB_FILE_GAMMA,
            ..Chunks::default()
        };
        assert!(matches!(Gamma::new(16, &chunks), Gamma::None));
        assert!(matches!(
            Gamma::new(16, &Chunks::default()),
            Gamma::Table16 { shift: 5, .. }
        ));
    }
}

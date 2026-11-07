//! Decodes PNG bytes to packed RGB8 as Pillow's `convert("RGB")` does. It
//! ignores gamma and ICC profiles and drops alpha without compositing.

use std::borrow::Cow;
use std::io::Cursor;

use png::{BitDepth, ColorType, Decoder, InterlaceInfo, Reader, Transformations};

/// Wider or taller images are invalid, even when they fit the pixel limit.
const MAX_DIMENSION: u32 = 1_000_000;

const RGB8_BITS_PER_PIXEL: u8 = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Invalid,
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    Gray8,
    GrayAlpha8,
    Rgb8,
    Rgba8,
    /// Pillow converts 16-bit gray by saturating at 255, unlike 16-bit RGB,
    /// which keeps the high byte.
    Gray16,
}

impl Layout {
    fn of((color, depth): (ColorType, BitDepth)) -> Option<Self> {
        match (color, depth) {
            (ColorType::Grayscale, BitDepth::Eight) => Some(Self::Gray8),
            (ColorType::GrayscaleAlpha, BitDepth::Eight) => Some(Self::GrayAlpha8),
            (ColorType::Rgb, BitDepth::Eight) => Some(Self::Rgb8),
            (ColorType::Rgba, BitDepth::Eight) => Some(Self::Rgba8),
            (ColorType::Grayscale, BitDepth::Sixteen) => Some(Self::Gray16),
            _ => None,
        }
    }

    fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray8 => 1,
            Self::GrayAlpha8 | Self::Gray16 => 2,
            Self::Rgb8 => 3,
            Self::Rgba8 => 4,
        }
    }

    fn to_rgb8(self, row: &[u8], rgb: &mut [u8]) {
        let rgb = rgb.as_chunks_mut::<3>().0;
        match self {
            Self::Gray8 => row.iter().zip(rgb).for_each(|(&g, out)| *out = [g; 3]),
            Self::GrayAlpha8 => row
                .as_chunks::<2>()
                .0
                .iter()
                .zip(rgb)
                .for_each(|(ga, out)| *out = [ga[0]; 3]),
            Self::Rgb8 => row
                .as_chunks::<3>()
                .0
                .iter()
                .zip(rgb)
                .for_each(|(px, out)| *out = *px),
            Self::Rgba8 => row
                .as_chunks::<4>()
                .0
                .iter()
                .zip(rgb)
                .for_each(|(px, out)| *out = [px[0], px[1], px[2]]),
            Self::Gray16 => row
                .as_chunks::<2>()
                .0
                .iter()
                .zip(rgb)
                .for_each(|(sample, out)| {
                    *out = [u16::from_be_bytes(*sample).min(255) as u8; 3];
                }),
        }
    }
}

fn transformations(color: ColorType, depth: BitDepth) -> Transformations {
    match (color, depth) {
        (ColorType::Grayscale, BitDepth::Sixteen) => Transformations::IDENTITY,
        _ => Transformations::EXPAND | Transformations::STRIP_16,
    }
}

/// Returns the bytes without acTL, fcTL and fdAT chunks. libpng, which this
/// decoder replaced, does not know APNG and skips them as unknown ancillary
/// chunks, while the png crate validates them even after the default image.
/// Only animated files pay for the copy. A malformed chunk structure stops the
/// walk and is left for the decoder to reject.
fn without_animation(bytes: &[u8]) -> Cow<'_, [u8]> {
    const SIGNATURE_LEN: usize = 8;
    let mut kept: Option<Vec<u8>> = None;
    let mut copied = 0;
    let mut offset = SIGNATURE_LEN;
    while let Some(&[l0, l1, l2, l3, t0, t1, t2, t3]) = bytes.get(offset..offset + 8) {
        let length = u32::from_be_bytes([l0, l1, l2, l3]) as usize;
        let Some(end) = length
            .checked_add(12)
            .and_then(|size| offset.checked_add(size))
            .filter(|&end| end <= bytes.len())
        else {
            break;
        };
        match &[t0, t1, t2, t3] {
            b"acTL" | b"fcTL" | b"fdAT" => {
                kept.get_or_insert_with(|| Vec::with_capacity(bytes.len()))
                    .extend_from_slice(&bytes[copied..offset]);
                copied = end;
            }
            b"IEND" => break,
            _ => {}
        }
        offset = end;
    }
    match kept {
        None => Cow::Borrowed(bytes),
        Some(mut kept) => {
            kept.extend_from_slice(&bytes[copied..]);
            Cow::Owned(kept)
        }
    }
}

fn reader(bytes: &[u8]) -> Result<Reader<Cursor<Cow<'_, [u8]>>>, Error> {
    let mut decoder = Decoder::new(Cursor::new(without_animation(bytes)));
    // The png crate skips the zlib Adler-32 check by default.
    decoder.ignore_checksums(false);
    decoder.set_ignore_text_chunk(true);
    decoder.set_ignore_iccp_chunk(true);
    let header = decoder.read_header_info().map_err(|_| Error::Invalid)?;
    let (width, height) = header.size();
    let (color, depth) = (header.color_type, header.bit_depth);
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(Error::Invalid);
    }
    decoder.set_transformations(transformations(color, depth));
    let reader = decoder.read_info().map_err(|_| Error::Invalid)?;
    // png 0.18.1 panics while expanding such a palette; libpng rejects it too.
    // Check before the caller allocates the output.
    if color == ColorType::Indexed
        && reader
            .info()
            .palette
            .as_ref()
            .is_some_and(|palette| palette.len() % 3 != 0)
    {
        return Err(Error::Invalid);
    }
    Ok(reader)
}

/// A PNG whose header and chunks before the image data have been read.
pub struct Png<'a> {
    reader: Reader<Cursor<Cow<'a, [u8]>>>,
}

pub fn open(bytes: &[u8]) -> Result<Png<'_>, Error> {
    Ok(Png {
        reader: reader(bytes)?,
    })
}

impl Png<'_> {
    pub fn size(&self) -> (u32, u32) {
        self.reader.info().size()
    }

    pub fn decode_rgb8(self, rgb: &mut [u8]) -> Result<(), Error> {
        decode_rgb8(self.reader, rgb)
    }
}

fn decode_rgb8(mut reader: Reader<Cursor<Cow<'_, [u8]>>>, rgb: &mut [u8]) -> Result<(), Error> {
    let (width, height) = reader.info().size();
    let stride = width as usize * 3;
    if stride.checked_mul(height as usize) != Some(rgb.len()) {
        return Err(Error::Internal);
    }
    let layout = Layout::of(reader.output_color_type()).ok_or(Error::Internal)?;
    if reader.info().interlaced {
        let mut pass_rgb = vec![0; stride];
        while let Some(row) = reader.next_interlaced_row().map_err(|_| Error::Invalid)? {
            let InterlaceInfo::Adam7(pass) = row.interlace() else {
                return Err(Error::Internal);
            };
            let pixels = row.data().len() / layout.bytes_per_pixel();
            let pass_rgb = &mut pass_rgb[..pixels * 3];
            layout.to_rgb8(row.data(), pass_rgb);
            png::expand_interlaced_row(rgb, stride, pass_rgb, pass, RGB8_BITS_PER_PIXEL);
        }
    } else {
        for out in rgb.chunks_exact_mut(stride) {
            if layout == Layout::Rgb8 {
                reader
                    .read_row(out)
                    .map_err(|_| Error::Invalid)?
                    .ok_or(Error::Internal)?;
            } else {
                let row = reader
                    .next_row()
                    .map_err(|_| Error::Invalid)?
                    .ok_or(Error::Internal)?;
                layout.to_rgb8(row.data(), out);
            }
        }
    }
    reader.finish().map_err(|_| Error::Invalid)
}

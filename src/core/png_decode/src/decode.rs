//! Decodes PNG bytes to packed RGB8 as Pillow's `convert("RGB")` does. It
//! ignores gamma and ICC profiles and drops alpha without compositing.

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

fn reader(bytes: &[u8]) -> Result<Reader<Cursor<&[u8]>>, Error> {
    let mut decoder = Decoder::new(Cursor::new(bytes));
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
    decoder.read_info().map_err(|_| Error::Invalid)
}

pub fn dimensions(bytes: &[u8]) -> Result<(u32, u32), Error> {
    Ok(reader(bytes)?.info().size())
}

pub fn decode_rgb8(bytes: &[u8], rgb: &mut [u8]) -> Result<(), Error> {
    let mut reader = reader(bytes)?;
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

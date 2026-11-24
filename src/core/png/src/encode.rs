//! Encodes RGBA8 pixels as an sRGB PNG: no interlacing, adaptive filtering,
//! and fast deflate so that encoding does not hold up the response.

use png::{BitDepth, ColorType, DeflateCompression, Encoder, Filter, SrgbRenderingIntent};

/// miniz level 2. Its level 1 trades more size for speed than zlib's level 1,
/// which libpng used here; level 2 produced smaller files than that, faster.
const DEFLATE_LEVEL: u8 = 2;

use crate::decode::{Error, MAX_DIMENSION};

/// Returns the PNG for `rgba`, packed rows of `width` pixels. Wider or taller
/// images are rejected, as libpng and this crate's decoder reject them. The
/// png crate grows its compressed buffer infallibly; callers bound the input
/// by their pixel limits.
pub fn encode_rgba8(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, Error> {
    let len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4));
    if width == 0
        || height == 0
        || width > MAX_DIMENSION
        || height > MAX_DIMENSION
        || len != Some(rgba.len())
    {
        return Err(Error::Internal);
    }
    let mut out = Vec::new();
    let mut encoder = Encoder::new(&mut out, width, height);
    encoder.set_color(ColorType::Rgba);
    encoder.set_depth(BitDepth::Eight);
    encoder.set_source_srgb(SrgbRenderingIntent::Perceptual);
    encoder.set_deflate_compression(DeflateCompression::Level(DEFLATE_LEVEL));
    encoder.set_filter(Filter::Adaptive);
    let mut writer = encoder.write_header().map_err(|_| Error::Internal)?;
    writer.write_image_data(rgba).map_err(|_| Error::Internal)?;
    writer.finish().map_err(|_| Error::Internal)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{Format, open};

    fn round_trip(width: u32, height: u32, rgba: &[u8]) {
        let png = encode_rgba8(rgba, width, height).unwrap();
        let decoder = open(&png, Format::Rgba8).unwrap();
        assert_eq!(decoder.size(), (width, height));
        let mut decoded = vec![0; rgba.len()];
        decoder.decode(&mut decoded).unwrap();
        assert_eq!(decoded, rgba);
    }

    #[test]
    fn round_trips_pixels() {
        let mut state = 0x2545_f491_u32;
        for (width, height) in [(1, 1), (3, 2), (17, 5), (64, 64), (257, 3)] {
            let noise: Vec<u8> = (0..width * height * 4)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state as u8
                })
                .collect();
            round_trip(width, height, &noise);
            let gradient: Vec<u8> = (0..width * height * 4)
                .map(|i| (i % 4 * 60 + i / 4 % width * 3 + i / 4 / width) as u8)
                .collect();
            round_trip(width, height, &gradient);
        }
        round_trip(2, 2, &[0; 16]);
        round_trip(2, 2, &[255; 16]);
    }

    #[test]
    fn writes_srgb_without_gamma() {
        let png = encode_rgba8(&[1, 2, 3, 4], 1, 1).unwrap();
        let has = |kind: &[u8]| png.windows(4).any(|w| w == kind);
        assert!(has(b"sRGB"));
        assert!(!has(b"gAMA"));
        assert!(!has(b"cHRM"));
    }

    #[test]
    fn rejects_mismatched_lengths() {
        assert_eq!(encode_rgba8(&[0; 15], 2, 2), Err(Error::Internal));
        assert_eq!(encode_rgba8(&[], 0, 1), Err(Error::Internal));
        assert_eq!(encode_rgba8(&[], 1, 0), Err(Error::Internal));
        assert_eq!(
            encode_rgba8(&[0; 4 * 1_000_001], 1_000_001, 1),
            Err(Error::Internal)
        );
    }
}

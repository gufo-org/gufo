//! Decodes JPEG bytes to packed RGB8 as Pillow's `convert("RGB")` does with
//! libjpeg-turbo, apart from IDCT and chroma upsampling rounding. It ignores
//! ICC profiles. zune-jpeg pads missing entropy-coded data with zeros, so a
//! JPEG without EOI, which libjpeg-turbo rejected through its warnings, is
//! rejected here before decoding.

use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

/// The JPEG frame header limit; zune-jpeg defaults to 16,384.
const MAX_DIMENSION: usize = 65_535;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Invalid,
    Internal,
}

/// A JPEG whose headers up to the first scan have been read.
pub struct Jpeg<'a> {
    decoder: JpegDecoder<ZCursor<&'a [u8]>>,
    width: u32,
    height: u32,
}

/// Walks the segments after SOI and the entropy-coded data after each SOS.
/// True when EOI ends the walk; false when the bytes end first, or when a
/// marker is missing where a segment ends, which libjpeg-turbo warns about.
/// Between segments, `FF 00` reads as a marker with a length, where zune-jpeg
/// treats it as fill; the image is rejected either way.
fn reaches_eoi(bytes: &[u8]) -> bool {
    let mut offset = 2;
    loop {
        if bytes.get(offset) != Some(&0xff) {
            return false;
        }
        while bytes.get(offset) == Some(&0xff) {
            offset += 1;
        }
        let Some(&marker) = bytes.get(offset) else {
            return false;
        };
        offset += 1;
        match marker {
            0xd9 => return true,
            0x01 | 0xd0..=0xd7 => continue,
            _ => {}
        }
        let Some(&[high, low]) = bytes.get(offset..offset + 2) else {
            return false;
        };
        offset += usize::from(u16::from_be_bytes([high, low]));
        if marker != 0xda {
            continue;
        }
        // Entropy-coded data ends at the first marker other than a stuffed
        // zero or a restart marker.
        loop {
            let Some(found) = bytes
                .get(offset..)
                .and_then(|rest| rest.iter().position(|&b| b == 0xff))
            else {
                return false;
            };
            offset += found;
            let mut next = offset + 1;
            while bytes.get(next) == Some(&0xff) {
                next += 1;
            }
            match bytes.get(next) {
                Some(0x00 | 0xd0..=0xd7) => offset = next + 1,
                Some(_) => break,
                None => return false,
            }
        }
    }
}

pub fn open(bytes: &[u8]) -> Result<Jpeg<'_>, Error> {
    let options = DecoderOptions::default()
        .set_strict_mode(true)
        .set_max_width(MAX_DIMENSION)
        .set_max_height(MAX_DIMENSION)
        .jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    decoder.decode_headers().map_err(|_| Error::Invalid)?;
    if !reaches_eoi(bytes) {
        return Err(Error::Invalid);
    }
    let info = decoder.info().ok_or(Error::Internal)?;
    // libjpeg converts only gray, YCbCr, RGB, CMYK and YCCK to RGB.
    if !matches!(info.components, 1 | 3 | 4) {
        return Err(Error::Invalid);
    }
    if decoder.output_colorspace() != Some(ColorSpace::RGB) {
        return Err(Error::Internal);
    }
    Ok(Jpeg {
        decoder,
        width: u32::from(info.width),
        height: u32::from(info.height),
    })
}

impl Jpeg<'_> {
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn decode_rgb8(mut self, rgb: &mut [u8]) -> Result<(), Error> {
        let expected = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|pixels| pixels.checked_mul(3));
        if expected != Some(rgb.len()) || self.decoder.output_buffer_size() != expected {
            return Err(Error::Internal);
        }
        self.decoder.decode_into(rgb).map_err(|_| Error::Invalid)
    }
}

#[cfg(test)]
mod tests {
    use super::reaches_eoi;

    const SCAN: &[u8] = &[0xff, 0xda, 0x00, 0x03, 0x01, 0x12, 0xff, 0x00, 0x34];

    fn jpeg(parts: &[&[u8]]) -> Vec<u8> {
        let mut bytes = vec![0xff, 0xd8];
        parts.iter().for_each(|part| bytes.extend_from_slice(part));
        bytes
    }

    #[test]
    fn finds_eoi_after_scans_and_restarts() {
        let restart: &[u8] = &[0xff, 0xd0, 0x56];
        assert!(reaches_eoi(&jpeg(&[SCAN, restart, &[0xff, 0xd9]])));
        assert!(reaches_eoi(&jpeg(&[SCAN, &[0xff, 0xff, 0xd9]])));
        // A progressive image has segments between scans.
        let dht: &[u8] = &[0xff, 0xc4, 0x00, 0x03, 0xd9];
        assert!(reaches_eoi(&jpeg(&[SCAN, dht, SCAN, &[0xff, 0xd9, 0x00]])));
    }

    #[test]
    fn rejects_every_truncation() {
        let full = jpeg(&[&[0xff, 0xe0, 0x00, 0x04, 0xff, 0xd9], SCAN, &[0xff, 0xd9]]);
        assert!(reaches_eoi(&full));
        for len in 0..full.len() - 1 {
            assert!(!reaches_eoi(&full[..len]), "length {len}");
        }
    }

    #[test]
    fn rejects_bytes_between_segments() {
        let app: &[u8] = &[0xff, 0xe0, 0x00, 0x02, 0x00];
        assert!(!reaches_eoi(&jpeg(&[app, SCAN, &[0xff, 0xd9]])));
    }
}

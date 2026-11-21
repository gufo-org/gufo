//! Reads the EXIF orientation from the first APP1 Exif segment before the
//! first scan. Malformed or missing data reads as 1, the identity.

const IDENTITY: u32 = 1;

/// Walks the markers after SOI. A fill byte run, a standalone marker or a
/// segment that does not fit ends nothing but itself; a non-marker byte, SOS,
/// EOI or a truncated segment ends the walk.
pub fn of(bytes: &[u8]) -> u32 {
    let mut offset = 2;
    while offset < bytes.len() {
        if bytes[offset] != 0xff {
            break;
        }
        // JPEG permits repeated FF fill bytes before a marker.
        while bytes.get(offset) == Some(&0xff) {
            offset += 1;
        }
        let Some(&marker) = bytes.get(offset) else {
            break;
        };
        offset += 1;
        match marker {
            0xda | 0xd9 => break,
            0x01 | 0xd0..=0xd8 => continue,
            _ => {}
        }
        let Some(&[high, low]) = bytes.get(offset..offset + 2) else {
            break;
        };
        let length = usize::from(u16::from_be_bytes([high, low]));
        if length < 2 || length > bytes.len() - offset {
            break;
        }
        let data = &bytes[offset + 2..offset + length];
        if marker == 0xe1 && data.len() >= 14 && data.starts_with(b"Exif\0\0") {
            return tiff(&data[6..]);
        }
        offset += length;
    }
    IDENTITY
}

/// Reads tag 0x112 (one SHORT) from IFD0 of a TIFF header.
fn tiff(data: &[u8]) -> u32 {
    let little = match data.get(..2) {
        Some(b"II") => true,
        Some(b"MM") => false,
        _ => return IDENTITY,
    };
    // An out-of-range field reads as 0, which no check below accepts.
    let read = |pos: usize, size: usize| -> usize {
        let Some(field) = pos.checked_add(size).and_then(|end| data.get(pos..end)) else {
            return 0;
        };
        let fold = |value: usize, &byte: &u8| value << 8 | usize::from(byte);
        if little {
            field.iter().rev().fold(0, fold)
        } else {
            field.iter().fold(0, fold)
        }
    };
    if data.len() < 8 || read(2, 2) != 42 {
        return IDENTITY;
    }
    let ifd = read(4, 4);
    if ifd > data.len() || data.len() - ifd < 2 {
        return IDENTITY;
    }
    let count = read(ifd, 2);
    if count > (data.len() - ifd - 2) / 12 {
        return IDENTITY;
    }
    (0..count)
        .map(|i| ifd + 2 + i * 12)
        .find(|&p| read(p, 2) == 0x112 && read(p + 2, 2) == 3 && read(p + 4, 4) == 1)
        .map_or(IDENTITY, |p| match read(p + 8, 2) {
            value @ 1..=8 => value as u32,
            _ => IDENTITY,
        })
}

#[cfg(test)]
mod tests {
    use super::of;

    fn exif(little: bool, orientation: u16) -> Vec<u8> {
        let u16b = |value: u16| {
            if little {
                value.to_le_bytes()
            } else {
                value.to_be_bytes()
            }
        };
        let u32b = |value: u32| {
            if little {
                value.to_le_bytes()
            } else {
                value.to_be_bytes()
            }
        };
        let mut tiff = Vec::new();
        tiff.extend_from_slice(if little { b"II" } else { b"MM" });
        tiff.extend_from_slice(&u16b(42));
        tiff.extend_from_slice(&u32b(8));
        tiff.extend_from_slice(&u16b(1));
        tiff.extend_from_slice(&u16b(0x112));
        tiff.extend_from_slice(&u16b(3));
        tiff.extend_from_slice(&u32b(1));
        tiff.extend_from_slice(&u16b(orientation));
        tiff.extend_from_slice(&[0, 0]);
        let mut segment = b"Exif\0\0".to_vec();
        segment.extend_from_slice(&tiff);
        segment
    }

    fn jpeg(segments: &[(u8, &[u8])]) -> Vec<u8> {
        let mut bytes = vec![0xff, 0xd8];
        for &(marker, data) in segments {
            bytes.extend_from_slice(&[0xff, marker]);
            bytes.extend_from_slice(&u16::try_from(data.len() + 2).unwrap().to_be_bytes());
            bytes.extend_from_slice(data);
        }
        bytes.extend_from_slice(&[0xff, 0xda, 0, 2, 0xff, 0xd9]);
        bytes
    }

    #[test]
    fn reads_both_byte_orders() {
        for value in 1..=8 {
            assert_eq!(of(&jpeg(&[(0xe1, &exif(true, value))])), u32::from(value));
            assert_eq!(of(&jpeg(&[(0xe1, &exif(false, value))])), u32::from(value));
        }
    }

    #[test]
    fn rejects_out_of_range_values() {
        assert_eq!(of(&jpeg(&[(0xe1, &exif(true, 0))])), 1);
        assert_eq!(of(&jpeg(&[(0xe1, &exif(true, 9))])), 1);
    }

    #[test]
    fn uses_the_first_exif_segment() {
        let first = exif(true, 6);
        let second = exif(true, 3);
        assert_eq!(
            of(&jpeg(&[(0xe0, b"JFIF\0"), (0xe1, &first), (0xe1, &second)])),
            6
        );
    }

    #[test]
    fn stops_at_the_first_scan() {
        let mut bytes = jpeg(&[]);
        bytes.truncate(bytes.len() - 2);
        let segment = exif(true, 6);
        bytes.extend_from_slice(&[0xff, 0xe1]);
        bytes.extend_from_slice(&u16::try_from(segment.len() + 2).unwrap().to_be_bytes());
        bytes.extend_from_slice(&segment);
        assert_eq!(of(&bytes), 1);
    }

    #[test]
    fn skips_fill_bytes_and_standalone_markers() {
        let segment = exif(false, 8);
        let mut bytes = vec![0xff, 0xd8, 0xff, 0xff, 0xff, 0x01, 0xff, 0xd3, 0xff, 0xe1];
        bytes.extend_from_slice(&u16::try_from(segment.len() + 2).unwrap().to_be_bytes());
        bytes.extend_from_slice(&segment);
        assert_eq!(of(&bytes), 8);
    }

    #[test]
    fn truncation_reads_as_identity() {
        let full = jpeg(&[(0xe1, &exif(true, 6))]);
        let segment_end = 2 + 4 + exif(true, 6).len();
        for len in 0..segment_end {
            assert_eq!(of(&full[..len]), 1, "length {len}");
        }
        assert_eq!(of(&full[..segment_end]), 6);
    }

    #[test]
    fn bounds_the_ifd() {
        let mut segment = exif(true, 6);
        // IFD offset past the end, then an entry count past the end.
        segment[10..14].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(of(&jpeg(&[(0xe1, &segment)])), 1);
        let mut segment = exif(true, 6);
        segment[14..16].copy_from_slice(&2u16.to_le_bytes());
        assert_eq!(of(&jpeg(&[(0xe1, &segment)])), 1);
    }
}

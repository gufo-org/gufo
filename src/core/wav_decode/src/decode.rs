//! RIFF WAV with interleaved integer PCM or 32-bit float samples, decoded to
//! `f32` in [-1, 1]. Chunks are walked like the C++ decoders this replaced:
//! unknown chunks are skipped, the last `fmt ` and `data` chunks win, and a
//! chunk that claims more bytes than follow is an error.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Under 44 bytes, or no RIFF/WAVE header.
    NotRiff,
    /// A chunk claims more bytes than follow.
    Truncated,
    /// A `fmt ` chunk under 16 bytes.
    InvalidFmt,
    /// A format tag other than integer PCM or IEEE float.
    UnsupportedFormat,
    /// Channels or sample rate outside the limits, or zero.
    OutOfBounds,
    /// No `data` chunk, or an empty one.
    NoData,
    /// A sample width the limits do not accept, or a block alignment that
    /// does not match it.
    UnsupportedLayout,
    TooLong,
    NonFinite,
}

/// What a caller accepts. Integer PCM is decoded from 16 bits up to
/// `max_pcm_bits` (16, 24 or 32); 32-bit float always. A sample rate of 0 is
/// rejected whatever `min_sample_rate` says.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_channels: u32,
    pub min_sample_rate: u32,
    pub max_sample_rate: u32,
    pub max_seconds: u64,
    pub max_pcm_bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Encoding {
    Pcm16,
    Pcm24,
    Pcm32,
    Float32,
}

const FORMAT_PCM: u16 = 1;
const FORMAT_FLOAT: u16 = 3;

#[derive(Debug)]
pub struct Wav<'a> {
    pub sample_rate: u32,
    pub channels: u32,
    pub frames: u64,
    encoding: Encoding,
    data: &'a [u8],
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Reads the chunks and checks the format against `limits`.
pub fn parse<'a>(bytes: &'a [u8], limits: &Limits) -> Result<Wav<'a>, Error> {
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(Error::NotRiff);
    }
    let (mut format, mut channels, mut sample_rate) = (0u16, 0u16, 0u32);
    let (mut block_align, mut bits_per_sample) = (0u16, 0u16);
    let mut data: &[u8] = &[];
    let mut offset = 12usize;
    while offset + 8 <= bytes.len() {
        let chunk_size = u32_at(bytes, offset + 4) as usize;
        let body = offset + 8;
        let rest = bytes.len() - body;
        if chunk_size > rest {
            return Err(Error::Truncated);
        }
        match &bytes[offset..offset + 4] {
            b"fmt " => {
                if chunk_size < 16 {
                    return Err(Error::InvalidFmt);
                }
                format = u16_at(bytes, body);
                channels = u16_at(bytes, body + 2);
                sample_rate = u32_at(bytes, body + 4);
                block_align = u16_at(bytes, body + 12);
                bits_per_sample = u16_at(bytes, body + 14);
            }
            b"data" => data = &bytes[body..body + chunk_size],
            _ => {}
        }
        // A chunk of odd size is followed by a pad byte, which the last chunk
        // may omit.
        let padded = chunk_size + (chunk_size & 1);
        if padded > rest {
            break;
        }
        offset = body + padded;
    }

    if format != FORMAT_PCM && format != FORMAT_FLOAT {
        return Err(Error::UnsupportedFormat);
    }
    if channels == 0
        || u32::from(channels) > limits.max_channels
        || sample_rate < limits.min_sample_rate.max(1)
        || sample_rate > limits.max_sample_rate
    {
        return Err(Error::OutOfBounds);
    }
    if data.is_empty() {
        return Err(Error::NoData);
    }
    let encoding = match (format, u32::from(bits_per_sample)) {
        (FORMAT_PCM, 16) => Encoding::Pcm16,
        (FORMAT_PCM, 24) if limits.max_pcm_bits >= 24 => Encoding::Pcm24,
        (FORMAT_PCM, 32) if limits.max_pcm_bits >= 32 => Encoding::Pcm32,
        (FORMAT_FLOAT, 32) => Encoding::Float32,
        _ => return Err(Error::UnsupportedLayout),
    };
    let block_align = usize::from(block_align);
    if block_align != usize::from(channels) * usize::from(bits_per_sample / 8)
        || !data.len().is_multiple_of(block_align)
    {
        return Err(Error::UnsupportedLayout);
    }
    // `data` holds at least one whole frame here.
    let frames = (data.len() / block_align) as u64;
    if frames > limits.max_seconds.saturating_mul(u64::from(sample_rate)) {
        return Err(Error::TooLong);
    }
    Ok(Wav {
        sample_rate,
        channels: u32::from(channels),
        frames,
        encoding,
        data,
    })
}

impl Wav<'_> {
    /// Interleaved samples: `frames * channels`.
    pub fn samples(&self) -> usize {
        self.data.len() / self.bytes_per_sample()
    }

    fn bytes_per_sample(&self) -> usize {
        match self.encoding {
            Encoding::Pcm16 => 2,
            Encoding::Pcm24 => 3,
            Encoding::Pcm32 | Encoding::Float32 => 4,
        }
    }

    /// Decodes the interleaved samples from `first` on into `out`. Integer
    /// PCM is divided by 2^(bits - 1); floats must be finite and are clamped
    /// to [-1, 1]. The contents of `out` are unspecified on error.
    pub fn decode(&self, first: usize, out: &mut [f32]) -> Result<(), Error> {
        let width = self.bytes_per_sample();
        assert!(first + out.len() <= self.samples(), "sample range");
        let data = &self.data[first * width..(first + out.len()) * width];
        let samples = out.iter_mut().zip(data.chunks_exact(width));
        match self.encoding {
            Encoding::Pcm16 => {
                for (out, b) in samples {
                    *out = f32::from(i16::from_le_bytes([b[0], b[1]])) / 32_768.0;
                }
            }
            Encoding::Pcm24 => {
                for (out, b) in samples {
                    // The arithmetic shift sign-extends the 24-bit value.
                    let pcm = i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8;
                    *out = pcm as f32 / 8_388_608.0;
                }
            }
            Encoding::Pcm32 => {
                for (out, b) in samples {
                    *out = i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0;
                }
            }
            Encoding::Float32 => {
                for (out, b) in samples {
                    let value = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    if !value.is_finite() {
                        return Err(Error::NonFinite);
                    }
                    *out = value.clamp(-1.0, 1.0);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHORT: Limits = Limits {
        max_channels: 8,
        min_sample_rate: 1,
        max_sample_rate: 192_000,
        max_seconds: 30,
        max_pcm_bits: 16,
    };
    const WIDE: Limits = Limits {
        max_seconds: 30 * 60,
        max_pcm_bits: 32,
        ..SHORT
    };

    fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut bytes = id.to_vec();
        bytes.extend_from_slice(&(body.len() as u32).to_le_bytes());
        bytes.extend_from_slice(body);
        if body.len() % 2 == 1 {
            bytes.push(0);
        }
        bytes
    }

    fn fmt(format: u16, channels: u16, rate: u32, bits: u16) -> Vec<u8> {
        let block = channels * bits / 8;
        let mut body = format.to_le_bytes().to_vec();
        body.extend_from_slice(&channels.to_le_bytes());
        body.extend_from_slice(&rate.to_le_bytes());
        body.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
        body.extend_from_slice(&block.to_le_bytes());
        body.extend_from_slice(&bits.to_le_bytes());
        chunk(b"fmt ", &body)
    }

    fn riff(chunks: &[Vec<u8>]) -> Vec<u8> {
        let body: Vec<u8> = chunks.concat();
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(4 + body.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(&body);
        bytes
    }

    fn wav(format: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        riff(&[fmt(format, channels, rate, bits), chunk(b"data", data)])
    }

    fn decode(bytes: &[u8], limits: &Limits) -> Result<Vec<f32>, Error> {
        let wav = parse(bytes, limits)?;
        let mut out = vec![0.0; wav.samples()];
        // In two parts, as gufo-audio-decode decodes in blocks.
        let (head, tail) = out.split_at_mut(wav.samples() / 2);
        wav.decode(0, head)?;
        wav.decode(head.len(), tail)?;
        Ok(out)
    }

    #[test]
    fn integer_pcm_scales_by_its_width() {
        let pcm16 = wav(1, 1, 16_000, 16, &[0x00, 0x80, 0xFF, 0x7F]);
        assert_eq!(decode(&pcm16, &SHORT).unwrap(), [-1.0, 32_767.0 / 32_768.0]);
        let pcm24 = wav(1, 2, 48_000, 24, &[0, 0, 0x80, 0xFF, 0xFF, 0x7F]);
        assert_eq!(
            decode(&pcm24, &WIDE).unwrap(),
            [-1.0, 8_388_607.0f32 / 8_388_608.0]
        );
        let pcm32 = wav(1, 1, 8_000, 32, &i32::MIN.to_le_bytes());
        assert_eq!(decode(&pcm32, &WIDE).unwrap(), [-1.0]);
    }

    #[test]
    fn wide_pcm_needs_the_limit() {
        let pcm24 = wav(1, 1, 8_000, 24, &[1, 2, 3]);
        assert_eq!(decode(&pcm24, &SHORT), Err(Error::UnsupportedLayout));
        let pcm8 = wav(1, 1, 8_000, 8, &[1, 2]);
        assert_eq!(decode(&pcm8, &WIDE), Err(Error::UnsupportedLayout));
    }

    #[test]
    fn floats_are_clamped_and_must_be_finite() {
        let mut data = 2.5f32.to_le_bytes().to_vec();
        data.extend_from_slice(&(-0.0f32).to_le_bytes());
        let values = decode(&wav(3, 1, 8_000, 32, &data), &SHORT).unwrap();
        assert_eq!(values[0], 1.0);
        assert!(values[1] == 0.0 && values[1].is_sign_negative());
        let nan = wav(3, 1, 8_000, 32, &f32::NAN.to_le_bytes());
        assert_eq!(decode(&nan, &SHORT), Err(Error::NonFinite));
    }

    #[test]
    fn chunks_are_walked_like_the_cpp_decoders() {
        let samples = [0x00, 0x40];
        let list = chunk(b"LIST", b"odd");
        let data_first = riff(&[chunk(b"data", &samples), list.clone(), fmt(1, 1, 8_000, 16)]);
        assert_eq!(decode(&data_first, &SHORT).unwrap(), [0.5]);
        let mut unpadded = riff(&[fmt(1, 1, 8_000, 16), chunk(b"data", &samples), list]);
        unpadded.pop();
        assert_eq!(decode(&unpadded, &SHORT).unwrap(), [0.5]);

        let mut truncated = wav(1, 1, 8_000, 16, &[0; 4]);
        truncated.pop();
        assert_eq!(decode(&truncated, &SHORT), Err(Error::Truncated));
        let short_fmt = riff(&[chunk(b"fmt ", &[1, 0, 1, 0]), chunk(b"data", &[0; 32])]);
        assert_eq!(decode(&short_fmt, &SHORT), Err(Error::InvalidFmt));
        assert_eq!(decode(b"RIFF", &SHORT), Err(Error::NotRiff));
    }

    #[test]
    fn formats_and_lengths_respect_the_limits() {
        let alaw = wav(6, 1, 8_000, 8, &[0; 2]);
        assert_eq!(decode(&alaw, &SHORT), Err(Error::UnsupportedFormat));
        let nine = wav(1, 9, 8_000, 16, &[0; 18]);
        assert_eq!(decode(&nine, &SHORT), Err(Error::OutOfBounds));
        let fast = wav(1, 1, 192_001, 16, &[0; 2]);
        assert_eq!(decode(&fast, &SHORT), Err(Error::OutOfBounds));
        let slow = wav(1, 1, 7_999, 16, &[0; 2]);
        let eight_khz = Limits {
            min_sample_rate: 8_000,
            ..SHORT
        };
        assert_eq!(decode(&slow, &eight_khz), Err(Error::OutOfBounds));
        assert_eq!(decode(&slow, &SHORT).unwrap().len(), 1);
        let silent = riff(&[fmt(1, 1, 8_000, 16), chunk(b"data", &[])]);
        assert_eq!(decode(&silent, &SHORT), Err(Error::NoData));
        let mut misaligned = wav(1, 2, 8_000, 16, &[0; 6]);
        assert_eq!(decode(&misaligned, &SHORT), Err(Error::UnsupportedLayout));
        misaligned[32] = 3; // block_align
        assert_eq!(decode(&misaligned, &SHORT), Err(Error::UnsupportedLayout));

        // At one sample per second, the limits are a few frames long.
        let thirty = wav(1, 1, 1, 16, &[0; 60]);
        assert_eq!(decode(&thirty, &SHORT).unwrap().len(), 30);
        let longer = wav(1, 1, 1, 16, &[0; 62]);
        assert_eq!(decode(&longer, &SHORT), Err(Error::TooLong));
        assert_eq!(decode(&longer, &WIDE).unwrap().len(), 31);
    }
}

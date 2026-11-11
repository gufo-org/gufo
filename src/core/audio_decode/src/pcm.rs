//! WAV sample codecs converted straight from symphonia's packets. symphonia's
//! PCM decoder writes each packet into a planar buffer, which
//! `copy_to_slice_interleaved` interleaves again; converting the interleaved
//! packet bytes gives the same floats in one pass.

use symphonia::core::codecs::audio::AudioCodecParameters;
use symphonia::core::codecs::audio::well_known::{
    CODEC_ID_PCM_ALAW, CODEC_ID_PCM_F32LE, CODEC_ID_PCM_MULAW, CODEC_ID_PCM_S16LE,
    CODEC_ID_PCM_S24LE, CODEC_ID_PCM_S32LE,
};

/// A codec whose packets `convert` turns into the floats symphonia decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pcm {
    S16,
    S24,
    S32,
    F32,
    ALaw,
    MuLaw,
}

impl Pcm {
    /// The codec of `params`, or `None` for other codecs and for samples
    /// coded in fewer bits than declared, which symphonia shifts up.
    pub fn of(params: &AudioCodecParameters) -> Option<Self> {
        let pcm = match params.codec {
            CODEC_ID_PCM_S16LE => Self::S16,
            CODEC_ID_PCM_S24LE => Self::S24,
            CODEC_ID_PCM_S32LE => Self::S32,
            CODEC_ID_PCM_F32LE => Self::F32,
            CODEC_ID_PCM_ALAW => Self::ALaw,
            CODEC_ID_PCM_MULAW => Self::MuLaw,
            _ => return None,
        };
        let unshifted = params.bits_per_coded_sample.is_none()
            || params.bits_per_coded_sample == params.bits_per_sample;
        unshifted.then_some(pcm)
    }

    fn width(self) -> usize {
        match self {
            Self::ALaw | Self::MuLaw => 1,
            Self::S16 => 2,
            Self::S24 => 3,
            Self::S32 | Self::F32 => 4,
        }
    }

    /// Replaces `out` with the samples of the whole frames in the interleaved
    /// `data`, scaled as symphonia scales them, and returns the frame count.
    /// `channels` is not 0.
    pub fn convert(self, data: &[u8], channels: usize, out: &mut Vec<f32>) -> usize {
        let frame = self.width() * channels;
        let frames = data.len() / frame;
        let data = &data[..frames * frame];
        out.clear();
        match self {
            Self::S16 => out.extend(
                data.chunks_exact(2)
                    .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32_768.0),
            ),
            // The arithmetic shift sign-extends the 24-bit value.
            Self::S24 => out
                .extend(data.chunks_exact(3).map(|b| {
                    (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.0
                })),
            Self::S32 => out.extend(data.chunks_exact(4).map(|b| {
                (f64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])) / 2_147_483_648.0) as f32
            })),
            Self::F32 => out.extend(
                data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            ),
            Self::ALaw => out.extend(
                data.iter()
                    .map(|&b| f32::from(ALAW[usize::from(b)]) / 32_768.0),
            ),
            Self::MuLaw => out.extend(
                data.iter()
                    .map(|&b| f32::from(MULAW[usize::from(b)]) / 32_768.0),
            ),
        }
        frames
    }
}

/// G.711 A-law to 16-bit linear PCM: magnitudes up to 32256.
const fn alaw(byte: u8) -> i16 {
    let value = byte ^ 0x55;
    let exponent = (value >> 4) & 7;
    let mantissa = (value & 0x0f) as i16;
    let magnitude = if exponent == 0 {
        (mantissa << 4) + 8
    } else {
        ((mantissa << 4) + 0x108) << (exponent - 1)
    };
    if value & 0x80 != 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// G.711 mu-law to 16-bit linear PCM: magnitudes up to 32124.
const fn mulaw(byte: u8) -> i16 {
    let value = !byte;
    let exponent = (value >> 4) & 7;
    let mantissa = (value & 0x0f) as i16;
    let magnitude = (((mantissa << 3) + 0x84) << exponent) - 0x84;
    if value & 0x80 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

const ALAW: [i16; 256] = {
    let mut table = [0; 256];
    let mut byte = 0;
    while byte < 256 {
        table[byte] = alaw(byte as u8);
        byte += 1;
    }
    table
};

const MULAW: [i16; 256] = {
    let mut table = [0; 256];
    let mut byte = 0;
    while byte < 256 {
        table[byte] = mulaw(byte as u8);
        byte += 1;
    }
    table
};

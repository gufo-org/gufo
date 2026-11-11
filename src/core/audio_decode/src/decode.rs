//! Encoded audio to interleaved float samples. `Formats::PcmWav` reads WAV
//! strictly through gufo-wav-decode; `Formats::All` reads WAV (integer PCM,
//! float, A-law, µ-law, IMA and Microsoft ADPCM), FLAC, MPEG audio layers I
//! to III and Ogg Vorbis through symphonia, which converts integer samples to
//! floats with the same power-of-two scale as libsndfile, and Ogg Opus through
//! symphonia's demuxer and libopus. WAV integer PCM, float, A-law and mu-law
//! packets are converted to symphonia's floats here, without its PCM decoder.

use std::io::Cursor;

use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::codecs::audio::well_known::{
    CODEC_ID_MP1, CODEC_ID_MP2, CODEC_ID_MP3, CODEC_ID_OPUS,
};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
use symphonia::core::packet::Packet;

use crate::opus;
use crate::pcm::Pcm;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// Why an input was rejected; the C++ `AudioStatus` names the same cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Not a container the formats read, or no audio track in it.
    Unrecognized,
    /// A malformed format description: a WAV `fmt ` chunk under 16 bytes.
    MalformedHeader,
    /// A codec, or WAV format tag, the formats do not decode.
    UnsupportedCodec,
    /// `PcmWav`: a sample width the limits do not accept, or a block
    /// alignment that does not match it.
    UnsupportedLayout,
    /// Channels or sample rate outside the limits, or zero.
    OutOfBounds,
    /// No samples: a WAV without data, or a stream that decodes to none.
    NoAudio,
    /// Longer than the limits allow, as declared or as decoded.
    TooLong,
    /// The data ends early, a packet fails to decode, or fewer frames decode
    /// than declared.
    Truncated,
    NonFinite,
    /// `stop` returned true, or the sink refused samples.
    Stopped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Formats {
    PcmWav,
    All,
}

/// What a caller accepts.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub formats: Formats,
    pub max_channels: u32,
    pub min_sample_rate: u32,
    pub max_sample_rate: u32,
    pub max_seconds: u64,
    /// `PcmWav` only: the widest integer PCM, 16, 24 or 32 bits.
    pub max_pcm_bits: u32,
}

/// The stream being decoded. While the sink receives samples, `frames` is
/// the declared length when the container states one within the limits (0
/// otherwise), for reserving; a forged header can make a caller reserve that
/// much address space but not fill it. Once decoding succeeds, it is the
/// decoded length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Info {
    pub sample_rate: u32,
    pub channels: u32,
    pub frames: u64,
}

/// Frames per block on the `PcmWav` path, between two calls to `stop`.
const WAV_BLOCK_FRAMES: usize = 16_384;

/// Decodes `bytes` and hands the interleaved samples to `sink` in order;
/// `sink` returns false to abort. `stop` is polled before each packet or
/// block. On error, the samples already handed over are to be discarded.
pub fn decode(
    bytes: &[u8],
    limits: &Limits,
    stop: &mut dyn FnMut() -> bool,
    sink: &mut dyn FnMut(&Info, &[f32]) -> bool,
) -> Result<Info, Error> {
    if stop() {
        return Err(Error::Stopped);
    }
    match limits.formats {
        Formats::PcmWav => decode_pcm_wav(bytes, limits, stop, sink),
        Formats::All => decode_all(bytes, limits, stop, sink, PcmPath::Direct),
    }
}

/// How `decode_all` turns the packets of a codec `Pcm` handles into floats.
/// The tests compare the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PcmPath {
    Direct,
    #[cfg_attr(not(test), allow(dead_code))]
    Symphonia,
}

fn decode_pcm_wav(
    bytes: &[u8],
    limits: &Limits,
    stop: &mut dyn FnMut() -> bool,
    sink: &mut dyn FnMut(&Info, &[f32]) -> bool,
) -> Result<Info, Error> {
    let wav_limits = gufo_wav_decode::Limits {
        max_channels: limits.max_channels,
        min_sample_rate: limits.min_sample_rate,
        max_sample_rate: limits.max_sample_rate,
        max_seconds: limits.max_seconds,
        max_pcm_bits: limits.max_pcm_bits,
    };
    let wav = gufo_wav_decode::parse(bytes, &wav_limits).map_err(wav_error)?;
    let info = Info {
        sample_rate: wav.sample_rate,
        channels: wav.channels,
        frames: wav.frames,
    };
    let total = wav.samples();
    let mut block = vec![0.0f32; total.min(WAV_BLOCK_FRAMES * wav.channels as usize)];
    let mut first = 0;
    while first < total {
        if stop() {
            return Err(Error::Stopped);
        }
        let len = block.len().min(total - first);
        wav.decode(first, &mut block[..len]).map_err(wav_error)?;
        if !sink(&info, &block[..len]) {
            return Err(Error::Stopped);
        }
        first += len;
    }
    Ok(info)
}

fn wav_error(error: gufo_wav_decode::Error) -> Error {
    use gufo_wav_decode::Error as Wav;
    match error {
        Wav::NotRiff => Error::Unrecognized,
        Wav::Truncated => Error::Truncated,
        Wav::InvalidFmt => Error::MalformedHeader,
        Wav::UnsupportedFormat => Error::UnsupportedCodec,
        Wav::OutOfBounds => Error::OutOfBounds,
        Wav::NoData => Error::NoAudio,
        Wav::UnsupportedLayout => Error::UnsupportedLayout,
        Wav::TooLong => Error::TooLong,
        Wav::NonFinite => Error::NonFinite,
    }
}

fn decode_all(
    bytes: &[u8],
    limits: &Limits,
    stop: &mut dyn FnMut() -> bool,
    sink: &mut dyn FnMut(&Info, &[f32]) -> bool,
    path: PcmPath,
) -> Result<Info, Error> {
    let source = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(
            &Hint::new(),
            source,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|_| Error::Unrecognized)?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or(Error::Unrecognized)?;
    let (id, declared) = (track.id, track.num_frames);
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or(Error::Unrecognized)?;
    if params.codec == CODEC_ID_OPUS {
        let head = params.extra_data.as_deref().and_then(opus::Head::parse);
        return decode_opus(&mut *format, id, head, declared, limits, stop, sink);
    }
    // Made for every codec, so that a codec `Pcm` converts is rejected as
    // symphonia rejects it.
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(|_| Error::UnsupportedCodec)?;
    let pcm = match path {
        PcmPath::Direct => Pcm::of(params),
        PcmPath::Symphonia => None,
    };
    let channels = params
        .channels
        .as_ref()
        .map(|c| c.count())
        .ok_or(Error::Unrecognized)?;
    let rate = params.sample_rate.ok_or(Error::Unrecognized)?;
    // An MPEG stream without a Xing/Info header only has an estimated length;
    // keep the samples that decode, as soundfile under librosa does.
    let estimated = [CODEC_ID_MP1, CODEC_ID_MP2, CODEC_ID_MP3].contains(&params.codec);
    let declared = if estimated { None } else { declared };
    if channels == 0
        || channels > limits.max_channels as usize
        || !(limits.min_sample_rate.max(1)..=limits.max_sample_rate).contains(&rate)
    {
        return Err(Error::OutOfBounds);
    }
    let max_frames = u64::from(rate).saturating_mul(limits.max_seconds);
    match declared {
        Some(0) => return Err(Error::NoAudio),
        Some(frames) if frames > max_frames => return Err(Error::TooLong),
        _ => {}
    }
    let mut info = Info {
        sample_rate: rate,
        channels: channels as u32,
        frames: declared.unwrap_or(0),
    };
    let mut frames = 0u64;
    let mut block = Vec::new();
    loop {
        if stop() {
            return Err(Error::Stopped);
        }
        let Some(packet) = next_packet(&mut *format, id)? else {
            break;
        };
        let (decoded, float) = match pcm {
            Some(pcm) => (
                pcm.convert(&packet.data, channels, &mut block) as u64,
                pcm == Pcm::F32,
            ),
            None => {
                let decoded = match decoder.decode(&packet) {
                    Ok(decoded) => decoded,
                    // An MPEG decoder resynchronises at the next frame, as
                    // mpg123 does under libsndfile.
                    Err(SymphoniaError::DecodeError(_)) if estimated => continue,
                    Err(_) => return Err(Error::Truncated),
                };
                let spec = decoded.spec();
                if spec.rate() != rate || spec.channels().count() != channels {
                    return Err(Error::Truncated);
                }
                block.resize(decoded.samples_interleaved(), 0.0f32);
                decoded.copy_to_slice_interleaved(&mut block[..]);
                let float = matches!(
                    decoded,
                    GenericAudioBufferRef::F32(_) | GenericAudioBufferRef::F64(_)
                );
                (decoded.frames() as u64, float)
            }
        };
        if frames + decoded > max_frames {
            return Err(Error::TooLong);
        }
        // Integer samples convert to finite floats.
        if float && !block.iter().all(|value| value.is_finite()) {
            return Err(Error::NonFinite);
        }
        if !block.is_empty() && !sink(&info, &block) {
            return Err(Error::Stopped);
        }
        frames += decoded;
    }
    if frames == 0 {
        return Err(Error::NoAudio);
    }
    if declared.is_some_and(|declared| frames != declared) {
        return Err(Error::Truncated);
    }
    info.frames = frames;
    Ok(info)
}

/// The next packet of track `id`, or `None` at the end. A WAV, FLAC or Ogg
/// stream that ends early ends in an IO error; the frame count reports it.
/// Like libsndfile, stop at the first physical stream of a chained Ogg file.
fn next_packet(format: &mut dyn FormatReader, id: u32) -> Result<Option<Packet>, Error> {
    loop {
        match format.next_packet() {
            Ok(Some(packet)) if packet.track_id == id => return Ok(Some(packet)),
            Ok(Some(_)) => {}
            Ok(None) | Err(SymphoniaError::IoError(_)) => return Ok(None),
            // Ogg's demuxer signals the next physical stream with a reset.
            // The caller still checks the first stream's declared length.
            Err(SymphoniaError::ResetRequired) => return Ok(None),
            Err(_) => return Err(Error::Truncated),
        }
    }
}

/// Ogg Opus as libsndfile decodes it: at the rate the header's input rate
/// selects, without the header's pre-skip and the end padding the last
/// granule position marks, and no further than the declared length.
fn decode_opus(
    format: &mut dyn FormatReader,
    id: u32,
    head: Option<opus::Head>,
    declared: Option<u64>,
    limits: &Limits,
    stop: &mut dyn FnMut() -> bool,
    sink: &mut dyn FnMut(&Info, &[f32]) -> bool,
) -> Result<Info, Error> {
    let head = head.ok_or(Error::MalformedHeader)?;
    let factor = head.rate_factor();
    let rate = 48_000 / factor;
    if u32::from(head.channels) > limits.max_channels
        || !(limits.min_sample_rate.max(1)..=limits.max_sample_rate).contains(&rate)
    {
        return Err(Error::OutOfBounds);
    }
    let max_frames = u64::from(rate).saturating_mul(limits.max_seconds);
    // The declared length counts 48 kHz samples from the stream's start to
    // its last granule position, pre-skip included.
    let pre_skip = u64::from(head.pre_skip);
    let expected = declared.map(|frames| frames.saturating_sub(pre_skip) / u64::from(factor));
    match expected {
        Some(0) => return Err(Error::NoAudio),
        Some(frames) if frames > max_frames => return Err(Error::TooLong),
        _ => {}
    }
    let mut decoder = opus::Decoder::new(&head, rate).ok_or(Error::MalformedHeader)?;
    let channels = usize::from(head.channels);
    let mut info = Info {
        sample_rate: rate,
        channels: u32::from(head.channels),
        frames: expected.unwrap_or(0),
    };
    let mut pcm = vec![0.0f32; opus::MAX_PACKET_FRAMES * channels];
    let mut frames = 0u64;
    // The 48 kHz position of the next packet, from the stream's start.
    let mut position = 0u64;
    let factor = u64::from(factor);
    while expected != Some(frames) {
        if stop() {
            return Err(Error::Stopped);
        }
        let Some(packet) = next_packet(&mut *format, id)? else {
            break;
        };
        let decoded = decoder
            .decode(&packet.data, &mut pcm)
            .ok_or(Error::Truncated)?;
        // The pre-skip and the end padding at the decoding rate, rounded as
        // libsndfile rounds them.
        let end = (decoded as u64 * factor).saturating_sub(packet.trim_end.get()) / factor;
        let start = (pre_skip.saturating_sub(position) / factor).min(end);
        position += decoded as u64 * factor;
        let end = expected.map_or(end, |expected| end.min(start + expected - frames));
        let kept = end - start;
        if frames + kept > max_frames {
            return Err(Error::TooLong);
        }
        let block = &pcm[start as usize * channels..end as usize * channels];
        if !block.iter().all(|value| value.is_finite()) {
            return Err(Error::NonFinite);
        }
        if !block.is_empty() && !sink(&info, block) {
            return Err(Error::Stopped);
        }
        frames += kept;
    }
    if frames == 0 {
        return Err(Error::NoAudio);
    }
    if expected.is_some_and(|expected| frames != expected) {
        return Err(Error::Truncated);
    }
    info.frames = frames;
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: Limits = Limits {
        formats: Formats::All,
        max_channels: 8,
        min_sample_rate: 1,
        max_sample_rate: 192_000,
        max_seconds: 30,
        max_pcm_bits: 32,
    };

    const PCM: u16 = 1;
    const FLOAT: u16 = 3;
    const ALAW: u16 = 6;
    const MULAW: u16 = 7;
    const EXTENSIBLE: u16 = 0xfffe;

    /// A `fmt ` chunk body at 16 kHz, with `cbSize` for tags other than PCM.
    fn fmt(tag: u16, channels: u16, bits: u16) -> Vec<u8> {
        let align = channels * bits.div_ceil(8);
        let mut fmt = Vec::new();
        fmt.extend(tag.to_le_bytes());
        fmt.extend(channels.to_le_bytes());
        fmt.extend(16_000u32.to_le_bytes());
        fmt.extend((16_000 * u32::from(align)).to_le_bytes());
        fmt.extend(align.to_le_bytes());
        fmt.extend(bits.to_le_bytes());
        if tag != PCM {
            fmt.extend(0u16.to_le_bytes());
        }
        fmt
    }

    /// A `WAVE_FORMAT_EXTENSIBLE` chunk body with `valid` of `bits` bits.
    fn extensible(subformat: u16, channels: u16, bits: u16, valid: u16) -> Vec<u8> {
        let mut fmt = fmt(EXTENSIBLE, channels, bits);
        fmt.truncate(16);
        fmt.extend(22u16.to_le_bytes());
        fmt.extend(valid.to_le_bytes());
        fmt.extend(0u32.to_le_bytes());
        fmt.extend(subformat.to_le_bytes());
        fmt.extend([0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71]);
        fmt
    }

    /// A RIFF WAV of one `fmt ` chunk and one `data` chunk declaring `size`.
    fn wav_sized(fmt: &[u8], data: &[u8], size: usize) -> Vec<u8> {
        let mut wav = b"RIFF".to_vec();
        wav.extend(((4 + 8 + fmt.len() + 8 + size) as u32).to_le_bytes());
        wav.extend(b"WAVEfmt ");
        wav.extend((fmt.len() as u32).to_le_bytes());
        wav.extend(fmt);
        wav.extend(b"data");
        wav.extend((size as u32).to_le_bytes());
        wav.extend(data);
        wav
    }

    fn wav(fmt: &[u8], data: &[u8]) -> Vec<u8> {
        wav_sized(fmt, data, data.len())
    }

    /// `len` bytes from a fixed linear congruential generator.
    fn noise(len: usize) -> Vec<u8> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 56) as u8
            })
            .collect()
    }

    fn floats(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn decode_with(bytes: &[u8], path: PcmPath) -> Result<(Info, Vec<f32>), Error> {
        let mut samples = Vec::new();
        let info = decode_all(
            bytes,
            &LIMITS,
            &mut || false,
            &mut |_, block| {
                samples.extend_from_slice(block);
                true
            },
            path,
        )?;
        Ok((info, samples))
    }

    /// The codec the direct path converts in `bytes`, if any.
    fn direct_codec(bytes: &[u8]) -> Option<Pcm> {
        let source = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
        let format = symphonia::default::get_probe()
            .probe(
                &Hint::new(),
                source,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .ok()?;
        let track = format.default_track(TrackType::Audio)?;
        Pcm::of(track.codec_params.as_ref()?.audio()?)
    }

    /// The direct path takes `codec` and gives symphonia's decoder's result:
    /// the same info and bit-identical samples, or the same error.
    fn assert_same(name: &str, bytes: &[u8], codec: Option<Pcm>) {
        assert_eq!(direct_codec(bytes), codec, "{name}: codec");
        let direct = decode_with(bytes, PcmPath::Direct);
        let symphonia = decode_with(bytes, PcmPath::Symphonia);
        match (&direct, &symphonia) {
            (Ok((info, samples)), Ok((expected_info, expected))) => {
                assert_eq!(info, expected_info, "{name}: info");
                assert_eq!(samples.len(), expected.len(), "{name}: length");
                let same = samples
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "{name}: samples differ");
            }
            _ => assert_eq!(
                direct.as_ref().err(),
                symphonia.as_ref().err(),
                "{name}: status"
            ),
        }
    }

    #[test]
    fn direct_pcm_matches_symphonia() {
        // Every 16-bit value, every A-law and mu-law byte; 3,000 frames span
        // packets of symphonia's 1,152 frames.
        let every_pcm16: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
        let every_byte: Vec<u8> = (0..=u8::MAX).collect();
        let s = Some;
        assert_same(
            "PCM16 every value",
            &wav(&fmt(PCM, 1, 16), &every_pcm16),
            s(Pcm::S16),
        );
        assert_same("A-law", &wav(&fmt(ALAW, 1, 8), &every_byte), s(Pcm::ALaw));
        assert_same(
            "mu-law",
            &wav(&fmt(MULAW, 1, 8), &every_byte),
            s(Pcm::MuLaw),
        );
        for channels in [1, 2, 3, 8] {
            let frames = 3_000;
            for (bits, codec) in [(16, Pcm::S16), (24, Pcm::S24), (32, Pcm::S32)] {
                let data = noise(frames * usize::from(channels * bits / 8));
                let name = format!("PCM{bits}, {channels} channels");
                assert_same(&name, &wav(&fmt(PCM, channels, bits), &data), s(codec));
            }
            let values: Vec<f32> = noise(frames * usize::from(channels))
                .iter()
                .map(|&b| (f32::from(b) - 128.0) / 37.0)
                .collect();
            let name = format!("float32, {channels} channels");
            assert_same(
                &name,
                &wav(&fmt(FLOAT, channels, 32), &floats(&values)),
                s(Pcm::F32),
            );
        }
        let special = floats(&[0.0, -0.0, f32::MIN_POSITIVE / 2.0, f32::MAX, -1e-30]);
        assert_same(
            "float32 extremes",
            &wav(&fmt(FLOAT, 1, 32), &special),
            s(Pcm::F32),
        );
        for bad in [f32::NAN, f32::INFINITY] {
            let data = floats(&[0.5, bad]);
            assert_same(
                "float32 non-finite",
                &wav(&fmt(FLOAT, 1, 32), &data),
                s(Pcm::F32),
            );
        }

        // WAVE_FORMAT_EXTENSIBLE, also with fewer valid bits than written.
        let stereo24 = noise(3_000 * 6);
        for (bits, valid, codec) in [
            (16, 16, Pcm::S16),
            (16, 12, Pcm::S16),
            (24, 24, Pcm::S24),
            (24, 20, Pcm::S24),
            (32, 24, Pcm::S32),
        ] {
            let data = noise(3_000 * 2 * usize::from(bits / 8));
            let name = format!("extensible PCM{bits}, {valid} valid bits");
            assert_same(
                &name,
                &wav(&extensible(PCM, 2, bits, valid), &data),
                s(codec),
            );
        }
        let data = floats(&[0.25, -3.0, 1.5, 0.0]);
        assert_same(
            "extensible float32",
            &wav(&extensible(FLOAT, 2, 32, 32), &data),
            s(Pcm::F32),
        );

        // A partial last frame, and a data chunk that ends early.
        assert_same(
            "partial frame",
            &wav(&fmt(PCM, 2, 24), &stereo24[..6 * 100 + 4]),
            s(Pcm::S24),
        );
        let cut = wav_sized(&fmt(PCM, 2, 24), &stereo24[..600], 1_200);
        assert_same("cut data chunk", &cut, s(Pcm::S24));

        // Codecs left to symphonia's decoder.
        assert_same("PCM8", &wav(&fmt(PCM, 1, 8), &every_byte), None);
        let float64: Vec<u8> = (0..100)
            .flat_map(|i| (f64::from(i) / 37.0).to_le_bytes())
            .collect();
        assert_same("float64", &wav(&fmt(FLOAT, 1, 64), &float64), None);
        assert_same(
            "extensible PCM8",
            &wav(&extensible(PCM, 1, 8, 8), &every_byte),
            None,
        );
    }
}

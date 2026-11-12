//! Ogg Opus through libopus, as libsndfile decodes it: the Ogg pages come from
//! symphonia's demuxer, and each packet goes to libopus's multistream decoder
//! at the lowest of 8, 12, 16, 24 or 48 kHz that is at least the header's
//! input rate, with the header's output gain.

use std::ffi::c_int;
use std::ptr::NonNull;

/// The most samples per channel one Opus packet decodes to: 120 ms at 48 kHz.
pub const MAX_PACKET_FRAMES: usize = 5760;

/// The identification header (RFC 7845, section 5.1), checked as libsndfile
/// checks it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Head {
    pub channels: u8,
    /// Samples at 48 kHz to discard from the start.
    pub pre_skip: u16,
    pub input_rate: i32,
    /// Q7.8 dB.
    pub gain: i16,
    pub streams: u8,
    pub coupled: u8,
    pub mapping: Vec<u8>,
}

impl Head {
    pub fn parse(packet: &[u8]) -> Option<Self> {
        if !(19..=276).contains(&packet.len()) || &packet[..8] != b"OpusHead" {
            return None;
        }
        if packet[8] > 1 {
            return None;
        }
        let channels = packet[9];
        let pre_skip = u16::from_le_bytes([packet[10], packet[11]]);
        let input_rate = i32::from_le_bytes([packet[12], packet[13], packet[14], packet[15]]);
        let gain = i16::from_le_bytes([packet[16], packet[17]]);
        let family = packet[18];
        let (streams, coupled, mapping, size) = if family == 0 {
            if !(1..=2).contains(&channels) {
                return None;
            }
            (1, channels - 1, (0..channels).collect(), 19)
        } else {
            let size = 21 + usize::from(channels);
            if packet.len() < size || (family == 1 && channels > 8) {
                return None;
            }
            let (streams, coupled) = (packet[19], packet[20]);
            if streams < 1 || coupled > streams || u16::from(streams) + u16::from(coupled) > 255 {
                return None;
            }
            let mapping = packet[21..size].to_vec();
            if mapping
                .iter()
                .any(|&m| m != 255 && u16::from(m) > u16::from(streams) + u16::from(coupled))
            {
                return None;
            }
            (streams, coupled, mapping, size)
        };
        // The specification makes extra bytes after the header an error.
        if packet.len() != size {
            return None;
        }
        Some(Head {
            channels,
            pre_skip,
            input_rate,
            gain,
            streams,
            coupled,
            mapping,
        })
    }

    /// 48 kHz samples per decoded sample: libsndfile decodes at the lowest
    /// Opus rate that is at least the input rate.
    pub fn rate_factor(&self) -> u32 {
        match self.input_rate {
            r if r > 24_000 => 1,
            r if r > 16_000 => 2,
            r if r > 12_000 => 3,
            r if r > 8_000 => 4,
            _ => 6,
        }
    }
}

#[repr(C)]
struct OpusMsDecoder {
    _private: [u8; 0],
}

const OPUS_OK: c_int = 0;
const OPUS_SET_GAIN_REQUEST: c_int = 4034;

// CMake links libopus into every binary through pkg-config. The attribute
// links it into Cargo's own test binaries; MSVC would embed it as a
// /DEFAULTLIB directive that the linker resolves by name only, so it is left
// to CMake there.
#[allow(unsafe_code)]
#[cfg_attr(not(target_env = "msvc"), link(name = "opus"))]
unsafe extern "C" {
    fn opus_multistream_decoder_create(
        fs: i32,
        channels: c_int,
        streams: c_int,
        coupled_streams: c_int,
        mapping: *const u8,
        error: *mut c_int,
    ) -> *mut OpusMsDecoder;
    fn opus_multistream_decode_float(
        st: *mut OpusMsDecoder,
        data: *const u8,
        len: i32,
        pcm: *mut f32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    fn opus_multistream_decoder_ctl(st: *mut OpusMsDecoder, request: c_int, ...) -> c_int;
    fn opus_multistream_decoder_destroy(st: *mut OpusMsDecoder);
}

/// A libopus multistream decoder; the only owner of its state.
pub struct Decoder {
    state: NonNull<OpusMsDecoder>,
    channels: usize,
}

impl Decoder {
    /// A decoder at `rate` Hz, one of 8, 12, 16, 24 or 48 kHz, with the
    /// header's channel mapping and gain.
    #[allow(unsafe_code)]
    pub fn new(head: &Head, rate: u32) -> Option<Self> {
        let mut error: c_int = OPUS_OK;
        // SAFETY: `mapping` holds `channels` bytes and `error` is writable;
        // libopus reads and writes nothing else.
        let state = unsafe {
            opus_multistream_decoder_create(
                rate as i32,
                c_int::from(head.channels),
                c_int::from(head.streams),
                c_int::from(head.coupled),
                head.mapping.as_ptr(),
                &mut error,
            )
        };
        let state = NonNull::new(state).filter(|_| error == OPUS_OK)?;
        let decoder = Decoder {
            state,
            channels: usize::from(head.channels),
        };
        // SAFETY: `state` is a live decoder, and OPUS_SET_GAIN takes one
        // opus_int32 argument.
        let gained = unsafe {
            opus_multistream_decoder_ctl(
                decoder.state.as_ptr(),
                OPUS_SET_GAIN_REQUEST,
                i32::from(head.gain),
            )
        };
        (gained == OPUS_OK).then_some(decoder)
    }

    /// Decodes one packet into `pcm`, interleaved, and returns the samples
    /// per channel, or `None` when libopus rejects the packet.
    #[allow(unsafe_code)]
    pub fn decode(&mut self, packet: &[u8], pcm: &mut [f32]) -> Option<usize> {
        let len = i32::try_from(packet.len()).ok()?;
        let frames = c_int::try_from(pcm.len() / self.channels).ok()?;
        // SAFETY: `state` is a live decoder, `packet` holds `len` bytes and
        // `pcm` holds `frames * channels` floats.
        let decoded = unsafe {
            opus_multistream_decode_float(
                self.state.as_ptr(),
                packet.as_ptr(),
                len,
                pcm.as_mut_ptr(),
                frames,
                0,
            )
        };
        usize::try_from(decoded).ok()
    }
}

impl Drop for Decoder {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        // SAFETY: `state` came from opus_multistream_decoder_create and is
        // destroyed once, here.
        unsafe { opus_multistream_decoder_destroy(self.state.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head(channels: u8, family: u8, tail: &[u8]) -> Vec<u8> {
        let mut packet = b"OpusHead".to_vec();
        packet.extend_from_slice(&[1, channels]);
        packet.extend_from_slice(&312u16.to_le_bytes());
        packet.extend_from_slice(&16_000i32.to_le_bytes());
        packet.extend_from_slice(&(-256i16).to_le_bytes());
        packet.push(family);
        packet.extend_from_slice(tail);
        packet
    }

    #[test]
    fn heads_are_checked_like_libsndfile() {
        let mono = Head::parse(&head(1, 0, &[])).unwrap();
        assert_eq!(
            (mono.pre_skip, mono.input_rate, mono.gain),
            (312, 16_000, -256)
        );
        assert_eq!(
            (mono.streams, mono.coupled, &mono.mapping[..]),
            (1, 0, &[0][..])
        );
        assert_eq!(mono.rate_factor(), 3);
        assert!(Head::parse(&head(3, 0, &[])).is_none());
        assert!(Head::parse(&head(1, 0, &[0])).is_none());
        let surround = Head::parse(&head(6, 1, &[4, 2, 0, 4, 1, 2, 3, 5])).unwrap();
        assert_eq!((surround.streams, surround.coupled), (4, 2));
        assert!(Head::parse(&head(9, 1, &[5, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8])).is_none());
        assert!(Head::parse(&head(2, 1, &[0, 0, 0, 1])).is_none());
        let mut version = head(1, 0, &[]);
        version[8] = 2;
        assert!(Head::parse(&version).is_none());
    }

    #[test]
    fn decodes_silence_at_each_rate() {
        let head = Head::parse(&head(1, 0, &[])).unwrap();
        for rate in [8_000, 12_000, 16_000, 24_000, 48_000] {
            let mut decoder = Decoder::new(&head, rate).unwrap();
            let mut pcm = vec![1.0f32; MAX_PACKET_FRAMES];
            // A 20 ms SILK packet of digital silence (TOC 0x08, one frame).
            let frames = decoder.decode(&[0x08], &mut pcm).unwrap();
            assert_eq!(frames, rate as usize / 50);
            assert!(pcm[..frames].iter().all(|&v| v == 0.0));
        }
    }
}

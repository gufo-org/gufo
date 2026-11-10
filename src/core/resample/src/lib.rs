//! One-shot resampling with the response of soxr's high-quality ("HQ")
//! preset: linear phase, flat to 0.913 of the lower Nyquist frequency,
//! stopband from that Nyquist frequency on, 130 dB of attenuation.
//!
//! The steep filter that defines the response runs as an FFT convolution at
//! the cheapest rate that can carry it; where the ratio needs more, a short
//! polyphase filter covers the rest:
//!
//! - ratios l/m with l <= 2 whose rate times l is at most three times the
//!   output rate (16 kHz from 8, 24, 32 or 48 kHz): the steep filter alone,
//!   at that rate, keeping every m-th sample;
//! - other inputs up to three times the output rate: the steep filter at the
//!   input rate, or twice it below twice the output rate, then a short
//!   polyphase filter to the output rate;
//! - higher inputs: a polyphase filter to twice the output rate, then the
//!   steep filter there, keeping every second sample.
//!
//! The plan follows soxr's; the code shares none of it.

#![forbid(unsafe_code)]

mod convolve;
mod fft;
mod filter;

use convolve::convolve;
use filter::{Polyphase, gcd, lowpass};

/// The largest ratio between the input and output rates, either way.
pub const MAX_RATIO: u32 = 64;

/// The passband ends at this fraction of the lower Nyquist frequency.
const PASSBAND_END: f64 = 0.913;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// A rate is zero, or one rate is more than MAX_RATIO times the other.
    Rates,
}

/// `input` at `rate` Hz resampled to `out_rate` Hz, aligned so that output
/// sample n is the input at time n / out_rate, with zeros before and after
/// the input. Like `soxr_oneshot`, it returns `len * out_rate / rate` samples
/// rounded half up. Equal rates return the input unchanged.
pub fn resample(input: &[f32], rate: u32, out_rate: u32) -> Result<Vec<f32>, Error> {
    let (low, high) = (u64::from(rate.min(out_rate)), u64::from(rate.max(out_rate)));
    if low == 0 || high > u64::from(MAX_RATIO) * low {
        return Err(Error::Rates);
    }
    let (rate, out_rate) = (rate as usize, out_rate as usize);
    if rate == out_rate {
        return Ok(input.to_vec());
    }
    // soxr's flush computes the length in double precision.
    let count = (input.len() as f64 / (rate as f64 / out_rate as f64) + 0.5) as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    let edge = rate.min(out_rate) as f64 / 2.0;
    let pass = PASSBAND_END * edge;
    let g = gcd(rate, out_rate);
    let (l, m) = (out_rate / g, rate / g);
    if l <= 2 && rate * l <= 3 * out_rate {
        let h = lowpass(rate * l, pass, edge, l as f32);
        return Ok(convolve(input, l, &h, m, h.len() / 2, count));
    }
    if rate <= 3 * out_rate {
        let up = if rate < 2 * out_rate { 2 } else { 1 };
        let fast = rate * up;
        let h = lowpass(fast, pass, edge, up as f32);
        // The steep filter leaves nothing between edge and fast - edge, so
        // the polyphase filter may let that band through.
        let poly = Polyphase::new(fast, out_rate, edge, fast as f64 - edge);
        let steep = convolve(input, up, &h, 1, h.len() / 2, poly.needed(count));
        return Ok(poly.run(&steep, count));
    }
    let middle = 2 * out_rate;
    let h = lowpass(middle, pass, edge, 1.0);
    let poly = Polyphase::new(rate, middle, edge, middle as f64 - edge);
    let y = poly.run(input, 2 * (count - 1) + h.len() / 2 + 1);
    Ok(convolve(&y, 1, &h, 2, h.len() / 2, count))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATES: [u32; 12] = [
        8000, 8001, 11025, 16000, 22050, 24000, 32000, 44056, 44100, 48000, 96000, 192000,
    ];

    fn sine(rate: u32, freq: f64, seconds: f64) -> Vec<f32> {
        let len = (f64::from(rate) * seconds) as usize;
        (0..len)
            .map(|n| (2.0 * std::f64::consts::PI * freq * n as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    /// The largest deviation from `sin(2 pi freq n / rate)` away from the
    /// ends, where the filters see the zeros around the input.
    fn sine_error(out: &[f32], rate: u32, freq: f64) -> f64 {
        let margin = out.len() / 8;
        (margin..out.len() - margin)
            .map(|n| {
                let expected =
                    (2.0 * std::f64::consts::PI * freq * n as f64 / f64::from(rate)).sin();
                (f64::from(out[n]) - expected).abs()
            })
            .fold(0.0, f64::max)
    }

    fn rms(out: &[f32]) -> f64 {
        let margin = out.len() / 8;
        let inner = &out[margin..out.len() - margin];
        (inner
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / inner.len() as f64)
            .sqrt()
    }

    #[test]
    fn rejects_zero_and_extreme_rates() {
        assert_eq!(resample(&[0.0], 0, 16000), Err(Error::Rates));
        assert_eq!(resample(&[0.0], 16000, 0), Err(Error::Rates));
        assert_eq!(resample(&[0.0], 16000 * 65, 16000), Err(Error::Rates));
        assert_eq!(resample(&[0.0], 16000, 16000 * 65), Err(Error::Rates));
        assert!(resample(&[0.0], 16000 * 64, 16000).is_ok());
        assert!(resample(&[0.0], 250, 16000).is_ok());
    }

    #[test]
    fn length_rounds_half_up_and_equal_rates_copy() {
        for len in [0usize, 1, 2, 3, 160, 161, 4410, 44101] {
            let x: Vec<f32> = (0..len).map(|i| i as f32 / 7.0).collect();
            // len * 16000 / 44100 + 1/2 = (320 len + 441) / 882.
            assert_eq!(
                resample(&x, 44100, 16000).unwrap().len(),
                (320 * len + 441) / 882
            );
            assert_eq!(resample(&x, 32000, 16000).unwrap().len(), len.div_ceil(2));
            assert_eq!(resample(&x, 8000, 16000).unwrap().len(), 2 * len);
            assert_eq!(resample(&x, 16000, 16000).unwrap(), x);
        }
    }

    #[test]
    fn passband_sines_keep_amplitude_and_phase() {
        for rate in RATES {
            for out in [8000, 16000, 22050, 24000, 44100, 48000] {
                if rate == out {
                    continue;
                }
                let edge = f64::from(rate.min(out)) / 2.0;
                for fraction in [0.05, 0.4, 0.85] {
                    let freq = fraction * edge;
                    let y = resample(&sine(rate, freq, 0.25), rate, out).unwrap();
                    let error = sine_error(&y, out, freq);
                    assert!(error < 1e-4, "{rate} -> {out} Hz, {freq} Hz: error {error}");
                }
            }
        }
    }

    #[test]
    fn stopband_sines_vanish() {
        for rate in RATES {
            for out in [8000, 16000, 22050, 44100] {
                if rate <= out {
                    continue;
                }
                // Between the output's Nyquist frequency and the input's.
                let edge = f64::from(out) / 2.0;
                for freq in [1.02 * edge, 1.5 * edge, 0.98 * f64::from(rate) / 2.0] {
                    if freq <= edge || freq >= f64::from(rate) / 2.0 {
                        continue;
                    }
                    let y = resample(&sine(rate, freq, 0.25), rate, out).unwrap();
                    let level = rms(&y);
                    assert!(level < 1e-5, "{rate} -> {out} Hz, {freq} Hz: rms {level}");
                }
            }
        }
    }

    #[test]
    fn upsampling_leaves_no_images() {
        for rate in [8000, 16000, 22050, 32000, 44100] {
            for out in [24000, 48000, 96000] {
                if rate >= out {
                    continue;
                }
                // An input sine at f leaves images near rate - f after
                // stuffing; the output must hold the sine alone.
                let freq = 0.85 * f64::from(rate) / 2.0;
                let y = resample(&sine(rate, freq, 0.25), rate, out).unwrap();
                let error = sine_error(&y, out, freq);
                assert!(error < 1e-4, "{rate} -> {out} Hz: error {error}");
            }
        }
    }
}

//! Kaiser-windowed sinc low-pass filters: the steep filter as plain taps for
//! the FFT convolution, and short filters in polyphase form.

use std::f64::consts::PI;

/// Stopband attenuation in dB.
const ATTENUATION: f64 = 130.0;
/// Ratios with up to this many output phases get one exact row per phase;
/// others interpolate linearly between INTERPOLATED_PHASES + 1 rows.
const EXACT_PHASES: usize = 1024;
const INTERPOLATED_PHASES: usize = 512;
/// Rows are padded with zero taps to whole chunks of this many lanes.
const LANES: usize = 16;

pub(crate) fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// I0 by Abramowitz & Stegun 9.8.1 and 9.8.2 (relative error below 2e-7).
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        1.0 + y
            * (3.5156229
                + y * (3.0899424
                    + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        ax.exp() / ax.sqrt()
            * (0.39894228
                + y * (0.01328592
                    + y * (0.00225319
                        + y * (-0.00157565
                            + y * (0.00916281
                                + y * (-0.02057706
                                    + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
    }
}

/// A low-pass filter at `rate` that passes up to `pass` Hz and stops from
/// `stop` Hz, by Kaiser's formulas for the window and its length.
struct Kaiser {
    /// The cutoff in cycles per sample.
    cutoff: f64,
    /// The window's half length in samples.
    half: f64,
    beta: f64,
    norm: f64,
    /// Taps per side: the filter spans 2k - 1 samples, or 2k at fractional
    /// phases.
    k: usize,
}

impl Kaiser {
    fn new(rate: usize, pass: f64, stop: f64) -> Self {
        let cutoff = (pass + stop) / 2.0 / rate as f64;
        let width = (stop - pass) / rate as f64;
        let beta = 0.1102 * (ATTENUATION - 8.7);
        let half = (ATTENUATION - 7.95) / (2.285 * 2.0 * PI * width) / 2.0;
        Self {
            cutoff,
            half,
            beta,
            norm: bessel_i0(beta),
            k: half.ceil() as usize,
        }
    }

    /// The impulse response `d` samples from its center.
    fn tap(&self, d: f64) -> f32 {
        let r = d / self.half;
        if r.abs() >= 1.0 {
            return 0.0;
        }
        let x = 2.0 * self.cutoff * d;
        let sinc = if x == 0.0 {
            1.0
        } else {
            (PI * x).sin() / (PI * x)
        };
        let window = bessel_i0(self.beta * (1.0 - r * r).sqrt()) / self.norm;
        (2.0 * self.cutoff * sinc * window) as f32
    }
}

/// The taps of a low-pass filter at `rate` passing up to `pass` Hz and
/// stopping from `stop` Hz, scaled by `gain`: an odd, symmetric filter whose
/// center is tap (len - 1) / 2.
pub(crate) fn lowpass(rate: usize, pass: f64, stop: f64, gain: f32) -> Vec<f32> {
    let kaiser = Kaiser::new(rate, pass, stop);
    let k = kaiser.k as isize;
    (1 - k..k).map(|d| kaiser.tap(d as f64) * gain).collect()
}

/// A filter from `rate` to `out_rate` = `rate` * l / m, applied in
/// polyphase form: output n is the input at time n m / l through the filter.
pub(crate) struct Polyphase {
    l: usize,
    m: usize,
    k: usize,
    /// Taps per row, a multiple of LANES.
    taps: usize,
    interpolated: bool,
    /// Row p holds the filter at phase p / l, or p / INTERPOLATED_PHASES
    /// when interpolated; tap j weighs the input j - k + 1 samples from the
    /// output's integer input position.
    table: Vec<f32>,
}

impl Polyphase {
    pub(crate) fn new(rate: usize, out_rate: usize, pass: f64, stop: f64) -> Self {
        let g = gcd(rate, out_rate);
        let (l, m) = (out_rate / g, rate / g);
        let kaiser = Kaiser::new(rate, pass, stop);
        let k = kaiser.k;
        let taps = (2 * k).next_multiple_of(LANES);
        let interpolated = l > EXACT_PHASES;
        let (rows, step) = if interpolated {
            (INTERPOLATED_PHASES + 1, INTERPOLATED_PHASES)
        } else {
            (l, l)
        };
        let mut table = vec![0.0f32; rows * taps];
        for (p, row) in table.chunks_exact_mut(taps).enumerate() {
            let frac = p as f64 / step as f64;
            for (j, tap) in row.iter_mut().enumerate() {
                *tap = kaiser.tap((j as isize - k as isize + 1) as f64 - frac);
            }
        }
        Self {
            l,
            m,
            k,
            taps,
            interpolated,
            table,
        }
    }

    /// How many input samples `run(x, count)` reads, counting zeros past the
    /// end of `x`.
    pub(crate) fn needed(&self, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        let last = (count as u128 - 1) * self.m as u128 / self.l as u128;
        usize::try_from(last).expect("an input position fits in usize") + self.k + 1
    }

    fn row(&self, p: usize) -> &[f32] {
        &self.table[p * self.taps..(p + 1) * self.taps]
    }

    /// `count` output samples from `x`, with zeros outside `x`.
    pub(crate) fn run(&self, x: &[f32], count: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(count);
        // Output n reads from input position i = n m / l, at phase rem / l.
        let (step, carry) = (self.m / self.l, self.m % self.l);
        let (mut i, mut rem) = (0usize, 0usize);
        for _ in 0..count {
            let first = i as isize - self.k as isize + 1;
            out.push(if self.interpolated {
                let f = (rem * INTERPOLATED_PHASES) as f64 / self.l as f64;
                let p = f as usize;
                let t = (f - p as f64) as f32;
                let (a, b) = (
                    dot_padded(x, first, self.row(p)),
                    dot_padded(x, first, self.row(p + 1)),
                );
                a + (b - a) * t
            } else {
                dot_padded(x, first, self.row(rem))
            });
            i += step;
            rem += carry;
            if rem >= self.l {
                rem -= self.l;
                i += 1;
            }
        }
        out
    }
}

/// `row` against `x[first..]`, with zeros outside `x`.
fn dot_padded(x: &[f32], first: isize, row: &[f32]) -> f32 {
    if first >= 0 && first as usize + row.len() <= x.len() {
        let first = first as usize;
        return dot(&x[first..first + row.len()], row);
    }
    let start = first.max(0) as usize;
    let skip = (start as isize - first) as usize;
    let end = ((first + row.len() as isize).max(0) as usize).min(x.len());
    if start >= end {
        return 0.0;
    }
    dot(&x[start..end], &row[skip..skip + (end - start)])
}

/// A dot product in LANES independent lanes, which the compiler vectorizes.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; LANES];
    let (ca, cb) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..LANES {
            lanes[i] += x[i] * y[i];
        }
    }
    let mut sum = lanes.iter().sum::<f32>();
    for (x, y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

//! A complex FFT for power-of-two sizes on separate real and imaginary
//! arrays. The forward transform leaves the spectrum in bit-reversed order
//! and the inverse takes it in that order, which is all a convolution needs,
//! so neither transform reorders. Radix-4 stages, with one radix-2 stage when
//! log2(n) is odd, work on fixed blocks of four lanes, which the compiler
//! turns into packed SIMD instructions without `unsafe` code.

use std::f64::consts::PI;

const W: usize = 4;
type V = [f32; W];

#[inline(always)]
fn ld(s: &[f32], j: usize) -> V {
    let mut v = [0.0; W];
    v.copy_from_slice(&s[j..j + W]);
    v
}

#[inline(always)]
fn st(s: &mut [f32], j: usize, v: V) {
    s[j..j + W].copy_from_slice(&v);
}

#[inline(always)]
fn add(a: V, b: V) -> V {
    std::array::from_fn(|l| a[l] + b[l])
}

#[inline(always)]
fn sub(a: V, b: V) -> V {
    std::array::from_fn(|l| a[l] - b[l])
}

/// (a + ib)(c + id), lane by lane.
#[inline(always)]
fn cmul(ar: V, ai: V, cr: V, ci: V) -> (V, V) {
    (
        std::array::from_fn(|l| ar[l] * cr[l] - ai[l] * ci[l]),
        std::array::from_fn(|l| ar[l] * ci[l] + ai[l] * cr[l]),
    )
}

/// (a + ib) times the conjugate of (c + id), lane by lane.
#[inline(always)]
fn cmulc(ar: V, ai: V, cr: V, ci: V) -> (V, V) {
    (
        std::array::from_fn(|l| ar[l] * cr[l] + ai[l] * ci[l]),
        std::array::from_fn(|l| ai[l] * cr[l] - ar[l] * ci[l]),
    )
}

/// The first `count` powers `w^(power j)` of `w = exp(-2 pi i / len)`, as
/// real and imaginary parts.
fn twiddles(count: usize, len: usize, power: usize) -> (Vec<f32>, Vec<f32>) {
    (0..count)
        .map(|j| {
            let a = -2.0 * PI * (power * j) as f64 / len as f64;
            (a.cos() as f32, a.sin() as f32)
        })
        .unzip()
}

/// One radix-4 stage over blocks of 4q values.
struct Radix4 {
    q: usize,
    /// w^j, w^2j and w^3j for j < q, real and imaginary parts, with
    /// w = exp(-2 pi i / 4q).
    w: [Vec<f32>; 6],
}

pub(crate) struct Fft {
    n: usize,
    /// The twiddles of the first, radix-2 stage when log2(n) is odd.
    half: Option<(Vec<f32>, Vec<f32>)>,
    /// The radix-4 stages with q >= 4, largest first; the q = 1 stage
    /// follows them.
    stages: Vec<Radix4>,
}

impl Fft {
    /// `n` is a power of two of at least 64.
    pub(crate) fn new(n: usize) -> Self {
        assert!(n >= 64 && n.is_power_of_two());
        let odd = n.trailing_zeros() % 2 == 1;
        let half = odd.then(|| twiddles(n / 2, n, 1));
        let mut stages = Vec::new();
        let mut q = if odd { n / 8 } else { n / 4 };
        while q >= W {
            let (w1r, w1i) = twiddles(q, 4 * q, 1);
            let (w2r, w2i) = twiddles(q, 4 * q, 2);
            let (w3r, w3i) = twiddles(q, 4 * q, 3);
            stages.push(Radix4 {
                q,
                w: [w1r, w1i, w2r, w2i, w3r, w3i],
            });
            q /= 4;
        }
        Self { n, half, stages }
    }

    /// The DFT of `re + i im`, in bit-reversed order (decimation in
    /// frequency).
    pub(crate) fn forward(&self, re: &mut [f32], im: &mut [f32]) {
        assert!(re.len() == self.n && im.len() == self.n);
        if let Some((wr, wi)) = &self.half {
            let s = self.n / 2;
            let (ar, br) = re.split_at_mut(s);
            let (ai, bi) = im.split_at_mut(s);
            for j in (0..s).step_by(W) {
                let (xr, xi, yr, yi) = (ld(ar, j), ld(ai, j), ld(br, j), ld(bi, j));
                st(ar, j, add(xr, yr));
                st(ai, j, add(xi, yi));
                let (pr, pi) = cmul(sub(xr, yr), sub(xi, yi), ld(wr, j), ld(wi, j));
                st(br, j, pr);
                st(bi, j, pi);
            }
        }
        for stage in &self.stages {
            let q = stage.q;
            let [w1r, w1i, w2r, w2i, w3r, w3i] = &stage.w;
            for (re, im) in re.chunks_exact_mut(4 * q).zip(im.chunks_exact_mut(4 * q)) {
                for j in (0..q).step_by(W) {
                    let (ar, ai, br, bi) = (ld(re, j), ld(im, j), ld(re, j + q), ld(im, j + q));
                    let (cr, ci) = (ld(re, j + 2 * q), ld(im, j + 2 * q));
                    let (dr, di) = (ld(re, j + 3 * q), ld(im, j + 3 * q));
                    let (t0r, t0i, t1r, t1i) = (add(ar, cr), add(ai, ci), sub(ar, cr), sub(ai, ci));
                    // t3 = (b - d) times -i.
                    let (t2r, t2i, t3r, t3i) = (add(br, dr), add(bi, di), sub(bi, di), sub(dr, br));
                    st(re, j, add(t0r, t2r));
                    st(im, j, add(t0i, t2i));
                    let (pr, pi) = cmul(sub(t0r, t2r), sub(t0i, t2i), ld(w2r, j), ld(w2i, j));
                    st(re, j + q, pr);
                    st(im, j + q, pi);
                    let (pr, pi) = cmul(add(t1r, t3r), add(t1i, t3i), ld(w1r, j), ld(w1i, j));
                    st(re, j + 2 * q, pr);
                    st(im, j + 2 * q, pi);
                    let (pr, pi) = cmul(sub(t1r, t3r), sub(t1i, t3i), ld(w3r, j), ld(w3i, j));
                    st(re, j + 3 * q, pr);
                    st(im, j + 3 * q, pi);
                }
            }
        }
        // q = 1: the four butterflies of each 16 values, with the four
        // inputs of one butterfly gathered into the lanes.
        for (r, i) in re.chunks_exact_mut(16).zip(im.chunks_exact_mut(16)) {
            let g = |s: &[f32], k: usize| -> V { std::array::from_fn(|l| s[4 * l + k]) };
            let (ar, br, cr, dr) = (g(r, 0), g(r, 1), g(r, 2), g(r, 3));
            let (ai, bi, ci, di) = (g(i, 0), g(i, 1), g(i, 2), g(i, 3));
            let (t0r, t0i, t1r, t1i) = (add(ar, cr), add(ai, ci), sub(ar, cr), sub(ai, ci));
            let (t2r, t2i, t3r, t3i) = (add(br, dr), add(bi, di), sub(bi, di), sub(dr, br));
            let out = [
                (add(t0r, t2r), add(t0i, t2i)),
                (sub(t0r, t2r), sub(t0i, t2i)),
                (add(t1r, t3r), add(t1i, t3i)),
                (sub(t1r, t3r), sub(t1i, t3i)),
            ];
            for (k, (vr, vi)) in out.into_iter().enumerate() {
                for l in 0..W {
                    r[4 * l + k] = vr[l];
                    i[4 * l + k] = vi[l];
                }
            }
        }
    }

    /// n times the inverse DFT of a spectrum in bit-reversed order
    /// (decimation in time), in natural order.
    pub(crate) fn inverse(&self, re: &mut [f32], im: &mut [f32]) {
        assert!(re.len() == self.n && im.len() == self.n);
        for (r, i) in re.chunks_exact_mut(16).zip(im.chunks_exact_mut(16)) {
            let g = |s: &[f32], k: usize| -> V { std::array::from_fn(|l| s[4 * l + k]) };
            let (v0r, v1r, v2r, v3r) = (g(r, 0), g(r, 1), g(r, 2), g(r, 3));
            let (v0i, v1i, v2i, v3i) = (g(i, 0), g(i, 1), g(i, 2), g(i, 3));
            let (p0r, p0i, p1r, p1i) = (add(v0r, v1r), add(v0i, v1i), sub(v0r, v1r), sub(v0i, v1i));
            let (sr, si, er, ei) = (add(v2r, v3r), add(v2i, v3i), sub(v2r, v3r), sub(v2i, v3i));
            // Positions 0 and 2: p0 +- s; 1 and 3: p1 +- i e.
            let out = [
                (add(p0r, sr), add(p0i, si)),
                (sub(p1r, ei), add(p1i, er)),
                (sub(p0r, sr), sub(p0i, si)),
                (add(p1r, ei), sub(p1i, er)),
            ];
            for (k, (vr, vi)) in out.into_iter().enumerate() {
                for l in 0..W {
                    r[4 * l + k] = vr[l];
                    i[4 * l + k] = vi[l];
                }
            }
        }
        for stage in self.stages.iter().rev() {
            let q = stage.q;
            let [w1r, w1i, w2r, w2i, w3r, w3i] = &stage.w;
            for (re, im) in re.chunks_exact_mut(4 * q).zip(im.chunks_exact_mut(4 * q)) {
                for j in (0..q).step_by(W) {
                    let (v0r, v0i) = (ld(re, j), ld(im, j));
                    let (v1r, v1i) = cmulc(ld(re, j + q), ld(im, j + q), ld(w2r, j), ld(w2i, j));
                    let (v2r, v2i) =
                        cmulc(ld(re, j + 2 * q), ld(im, j + 2 * q), ld(w1r, j), ld(w1i, j));
                    let (v3r, v3i) =
                        cmulc(ld(re, j + 3 * q), ld(im, j + 3 * q), ld(w3r, j), ld(w3i, j));
                    let (p0r, p0i, p1r, p1i) =
                        (add(v0r, v1r), add(v0i, v1i), sub(v0r, v1r), sub(v0i, v1i));
                    let (sr, si, er, ei) =
                        (add(v2r, v3r), add(v2i, v3i), sub(v2r, v3r), sub(v2i, v3i));
                    st(re, j, add(p0r, sr));
                    st(im, j, add(p0i, si));
                    st(re, j + 2 * q, sub(p0r, sr));
                    st(im, j + 2 * q, sub(p0i, si));
                    st(re, j + q, sub(p1r, ei));
                    st(im, j + q, add(p1i, er));
                    st(re, j + 3 * q, add(p1r, ei));
                    st(im, j + 3 * q, sub(p1i, er));
                }
            }
        }
        if let Some((wr, wi)) = &self.half {
            let s = self.n / 2;
            let (ar, br) = re.split_at_mut(s);
            let (ai, bi) = im.split_at_mut(s);
            for j in (0..s).step_by(W) {
                let (xr, xi) = (ld(ar, j), ld(ai, j));
                let (tr, ti) = cmulc(ld(br, j), ld(bi, j), ld(wr, j), ld(wi, j));
                st(ar, j, add(xr, tr));
                st(ai, j, add(xi, ti));
                st(br, j, sub(xr, tr));
                st(bi, j, sub(xi, ti));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bit_reverse(i: usize, bits: u32) -> usize {
        i.reverse_bits() >> (usize::BITS - bits)
    }

    /// Deterministic values in [-1, 1).
    fn signal(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    #[test]
    fn forward_is_the_dft_in_bit_reversed_order() {
        for n in [64, 128, 256, 512] {
            let (xr, xi) = (signal(n, 1), signal(n, 2));
            let (mut re, mut im) = (xr.clone(), xi.clone());
            Fft::new(n).forward(&mut re, &mut im);
            let bits = n.trailing_zeros();
            for k in 0..n {
                let (mut sr, mut si) = (0.0f64, 0.0f64);
                for t in 0..n {
                    let a = -2.0 * PI * (k * t % n) as f64 / n as f64;
                    let (c, s) = (a.cos(), a.sin());
                    sr += f64::from(xr[t]) * c - f64::from(xi[t]) * s;
                    si += f64::from(xr[t]) * s + f64::from(xi[t]) * c;
                }
                let j = bit_reverse(k, bits);
                assert!(
                    (f64::from(re[j]) - sr).abs() < 1e-4 * n as f64,
                    "n {n} bin {k}"
                );
                assert!(
                    (f64::from(im[j]) - si).abs() < 1e-4 * n as f64,
                    "n {n} bin {k}"
                );
            }
        }
    }

    #[test]
    fn inverse_undoes_forward() {
        for n in [64, 128, 1024, 2048, 4096, 8192] {
            let (xr, xi) = (signal(n, 3), signal(n, 4));
            let (mut re, mut im) = (xr.clone(), xi.clone());
            let fft = Fft::new(n);
            fft.forward(&mut re, &mut im);
            fft.inverse(&mut re, &mut im);
            for t in 0..n {
                assert!((re[t] / n as f32 - xr[t]).abs() < 1e-5, "n {n} sample {t}");
                assert!((im[t] / n as f32 - xi[t]).abs() < 1e-5, "n {n} sample {t}");
            }
        }
    }
}

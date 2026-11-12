//! Long filters as FFT convolution by overlap-save.

use crate::fft::Fft;

/// FFT size in multiples of the filter length: 4 balances the transform
/// against the samples each block yields.
const SIZE_PER_TAP: usize = 4;
const MIN_SIZE: usize = 1024;

/// `x` with `l - 1` zeros after each sample, filtered by `h`, keeping
/// `count` samples: output n is filtered sample `factor n + offset`.
pub(crate) fn convolve(
    x: &[f32],
    l: usize,
    h: &[f32],
    factor: usize,
    offset: usize,
    count: usize,
) -> Vec<f32> {
    let taps = h.len();
    let size = (SIZE_PER_TAP * taps).next_power_of_two().max(MIN_SIZE);
    // Each block yields the filtered samples that need no input before it.
    let valid = size - (taps - 1);
    let fft = Fft::new(size);
    // The filter's spectrum, scaled by 1 / size for the inverse transform.
    let (mut hr, mut hi) = (vec![0.0f32; size], vec![0.0f32; size]);
    hr[..taps].copy_from_slice(h);
    fft.forward(&mut hr, &mut hi);
    let scale = 1.0 / size as f32;
    for (r, i) in hr.iter_mut().zip(hi.iter_mut()) {
        *r *= scale;
        *i *= scale;
    }
    let (mut re, mut im) = (vec![0.0f32; size], vec![0.0f32; size]);
    let mut out = Vec::with_capacity(count);
    let mut start = 0usize;
    // Two real blocks share one complex transform: the filtered samples from
    // `start` in the real part and from `start + valid` in the imaginary part.
    while out.len() < count {
        let second = start + valid;
        fill(&mut re, x, l, start as isize - (taps as isize - 1));
        fill(&mut im, x, l, second as isize - (taps as isize - 1));
        fft.forward(&mut re, &mut im);
        for ((r, i), (a, b)) in re.iter_mut().zip(im.iter_mut()).zip(hr.iter().zip(&hi)) {
            let (x, z) = (*r, *i);
            *r = x * a - z * b;
            *i = x * b + z * a;
        }
        fft.inverse(&mut re, &mut im);
        // Filtered sample base + t is block[taps - 1 + t] for t < valid.
        for (block, base) in [(&re, start), (&im, second)] {
            if out.len() == count {
                break;
            }
            if factor == 1 {
                let from = taps - 1 + (out.len() + offset - base);
                let n = (count - out.len()).min(base + valid - (out.len() + offset));
                out.extend_from_slice(&block[from..from + n]);
                continue;
            }
            while out.len() < count {
                let t = factor * out.len() + offset;
                if t >= base + valid {
                    break;
                }
                out.push(block[taps - 1 + t - base]);
            }
        }
        start += 2 * valid;
    }
    out
}

/// `dst[i]` = sample `start + i` of `x` with `l - 1` zeros after each sample,
/// and zeros outside `x`.
fn fill(dst: &mut [f32], x: &[f32], l: usize, start: isize) {
    dst.fill(0.0);
    // Input samples j with start <= j l < start + len.
    let first = if start <= 0 {
        0
    } else {
        (start as usize).div_ceil(l)
    };
    let end = ((start + dst.len() as isize).max(0) as usize)
        .div_ceil(l)
        .min(x.len());
    if first >= end {
        return;
    }
    if l == 1 {
        let at = (first as isize - start) as usize;
        dst[at..at + end - first].copy_from_slice(&x[first..end]);
        return;
    }
    for (j, &v) in x.iter().enumerate().take(end).skip(first) {
        dst[(j as isize * l as isize - start) as usize] = v;
    }
}

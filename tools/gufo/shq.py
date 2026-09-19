"""SHQ-T16 quantization and packing.

Implements the SHQ4-T16 and SHQ8-T16 encodings used by the offline tools.
Byte layouts are byte-exact by design and checked by gufo.conformance:

SHQ4 U4Z G64:
  qweight[n_tile][k_group][k16_subtile][output_lane=16][packed_k=8]
  scales [n_tile][k_group][output_lane=16]  BF16
  zeros  [n_tile][k_group][output_lane=16]  packed UINT4

Weight byte offset:
  (((n_tile * k_group_count + k_group) * k16_per_group + k16_subtile)
       * 16 + output_lane) * 8 + k_pair

Scale/zero index:
  scale_index = (n_tile * k_group_count + k_group) * 16 + output_lane

Quantizer rounding:
  q_real = w / s + z      (U4Z)
  q = clamp(round_to_nearest_ties_to_even(q_real), code_min, code_max)
  scale rounded to BF16 RNE-ties-to-even before final code selection.

This is the deterministic v1 scale search: per-channel-per-group min/max range
for U4Z, max-abs for S4/SHQ8. When an importance vector (per-input-channel
E[x^2] imatrix) is supplied, scales/zeros are searched GPTQ/imatrix style:
per (tile, group), all 16 output lanes at once, enumerate zero candidates
{round(-min/s_r) + d, d in -1..1}, refine the scale by importance-weighted
least squares against the integer codes (s = sum(h*w*d)/sum(h*d^2),
d = q-z), fix s to BF16 RNE, and keep the best (s, z, q) by weighted error.
Keeping the min/max zero candidate makes imatrix never worse than the range
baseline per block. The packed planes are byte-identical in layout; only the
s/z/q values differ. Same output for same input (deterministic).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np

U4_MAX = 15
S4_MIN, S4_MAX = -8, 7
S8_MAX = 127
S6_MAX = 31  # signed 6-bit -32..31 (SHQ6, 6.56bpw; Q6_K-class)


def f32_to_bf16_uint16(arr: np.ndarray) -> np.ndarray:
    """Round FP32 to BF16 (round-to-nearest, ties-to-even). Returns uint16 array."""
    a = arr.astype(np.float32, copy=False)
    u = a.view(np.uint32).astype(np.uint64)
    lsb = np.right_shift(u, 16) & np.uint64(1)
    rounding_bias = np.uint64(0x7FFF) + lsb
    u = u + rounding_bias
    out = np.right_shift(u, np.uint64(16)).astype(np.uint32).astype(np.uint16)
    return out


def bf16_uint16_to_f32(ui: np.ndarray) -> np.ndarray:
    """Decode BF16 uint16 -> float32."""
    u32 = ui.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def _clamp(v, lo, hi):
    return np.minimum(np.maximum(v, lo), hi)


def _round_even(v):
    # numpy round is banker's rounding (ties-to-even) already.
    return np.round(v)


def _bf16r(a):
    """Round float32 array/scalar to BF16 RNE-ties-to-even (vectorized)."""
    return bf16_uint16_to_f32(f32_to_bf16_uint16(np.asarray(a, dtype=np.float32)))


def _imatrix_block_search_batched(Wb: np.ndarray, h: np.ndarray, G: int):
    """Batched importance-weighted scale/zero search over many (tile, group)
    blocks at once. Wb: [B,16,G] weights, h: [B,1,G] importance (broadcast over
    lanes). For each block/lane enumerate zero candidates {round(-min/s_r)+d,
    d in -1..1}, refine scale by weighted least squares (s = sum(h*w*d) /
    sum(h*d^2), d = q-z) for 2 rounds, fix s to BF16 RNE, keep best by weighted
    error. Constant rows keep the all-zero convention. Returns (s_bf [B,16],
    z [B,16] int, q [B,16,G] codes) with s_bf <= 0 for constant rows.
    """
    B = Wb.shape[0]
    wmin = Wb.min(axis=2)
    wmax = Wb.max(axis=2)
    rng = (wmax - wmin) / 15.0
    const = rng <= 0.0
    s_safe = np.where(const, 1.0, rng)
    z_rng = _clamp(_round_even(np.where(const, 0.0, -wmin / np.where(const, 1.0, rng))),
                   0, U4_MAX).astype(np.int64)

    best_err = np.full((B, 16), np.inf)
    best_s = np.zeros((B, 16), dtype=np.float32)
    best_z = np.zeros((B, 16), dtype=np.int64)
    best_q = np.zeros((B, 16, G), dtype=np.float32)
    for dz in (-1, 0, 1):
        z = np.clip(z_rng + dz, 0, 15)  # [B,16]
        s = s_safe.copy()
        q = _clamp(_round_even(Wb / s[..., None] + z[..., None]), 0, U4_MAX)
        for _ in range(2):
            d = (q - z[..., None]).astype(np.float32)
            den = (h * d * d).sum(axis=2)
            num = (h * Wb * d).sum(axis=2)
            ok = den > 1e-18
            s_new = np.where(ok, num / np.where(ok, den, 1.0), s)
            s = np.where(s_new > 0.0, s_new, s)
            q = _clamp(_round_even(Wb / s[..., None] + z[..., None]), 0, U4_MAX)
        s_bf = _bf16r(s)
        s_bf = np.where((s_bf > 0.0) | const, s_bf, s_safe)
        q = _clamp(_round_even(Wb / s_bf[..., None] + z[..., None]), 0, U4_MAX)
        d = (q - z[..., None]).astype(np.float32)
        err = (h * (Wb - s_bf[..., None] * d) ** 2).sum(axis=2)
        upd = err < best_err
        best_err = np.where(upd, err, best_err)
        best_s = np.where(upd, s_bf, best_s)
        best_z = np.where(upd, z, best_z)
        best_q = np.where(upd[..., None], q, best_q)
    best_s = np.where(const, 0.0, best_s)
    best_z = np.where(const, 0, best_z)
    best_q = np.where(const[..., None], 0.0, best_q)
    return best_s, best_z, best_q


def _quantize_chunk(Wc, hc, symmetric, G, k16_per):
    """Scale/zero/code selection + packing for one chunk of [Bc,16,G] blocks.

    Returns (weight_bytes, scale_bytes, zero_bytes) for the chunk, in packing
    order (gidx = nt*k_group_count+g). Runs one chunk's worth of vectorized
    numpy; callers fan chunks out across threads (numpy releases the GIL
    inside its C loops, so elementwise/bandwidth-bound work parallelizes).
    """
    Bc = Wc.shape[0]
    wmin = Wc.min(axis=2)
    wmax = Wc.max(axis=2)
    const = wmax == wmin

    if symmetric:
        s = np.where(const, 0.0, np.maximum(np.abs(wmax), np.abs(wmin)) / 7.0)
    else:
        s = np.where(const, 0.0, (wmax - wmin) / 15.0)
    s_bf = _bf16r(s)  # [Bc,16] scale, BF16 RNE
    valid = (const == False) & (s_bf > 0.0)

    if symmetric:
        qx = np.where((Wc == 0), 0,
                      _clamp(_round_even(Wc / np.where(valid, s_bf, 1.0)[..., None]),
                             S4_MIN, S4_MAX))
        q = np.where(valid[..., None], qx, 0.0)
        z = np.zeros((Bc, 16))
    elif hc is not None:
        s_bf, z, q = _imatrix_block_search_batched(Wc, hc, G)
    else:
        s_eff = np.where(valid, s_bf, 1.0)
        z = np.where(valid, _clamp(_round_even(-wmin / s_eff), 0, U4_MAX), 0.0)
        q = _clamp(_round_even(Wc / s_eff[..., None] + z[..., None]), 0, U4_MAX)
        q = np.where(valid[..., None], q, 0.0)

    sbytes = f32_to_bf16_uint16(np.where(valid, s_bf, 0.0)).astype(np.uint16).tobytes()
    if not symmetric:
        zz = np.clip(np.round(z), 0, U4_MAX).reshape(Bc, 16).astype(np.uint8)
        zz = zz.reshape(Bc, 8, 2)
        zbytes = (zz[:, :, 0] | (zz[:, :, 1] << 4)).tobytes()
    else:
        zbytes = b""
    if symmetric:
        qn = (q.astype(np.int8).astype(np.uint8)) & 0x0F
    else:
        qn = np.clip(np.round(q), 0, U4_MAX).astype(np.uint8)
    qr = qn.reshape(Bc, 16, k16_per, 16).transpose(0, 2, 1, 3)
    wbytes = (qr[:, :, :, 0::2] | (qr[:, :, :, 1::2] << 4)).tobytes()
    return wbytes, sbytes, zbytes


def _quant_workers() -> int:
    """Thread count for chunk fan-out: env override, else min(8, cpus)."""
    env = os.environ.get("GUFO_QUANT_THREADS")
    if env:
        return max(1, int(env))
    return min(4, os.cpu_count() or 1)


def quantize_shq4(W: np.ndarray, group_size: int, symmetric: bool = False,
                  importance: np.ndarray | None = None):
    """Quantize logical W[N,K] (float32) to SHQ4-T16 planes.

    importance: optional per-input-channel imatrix vector of length K
    (E[x_j^2] over calibration). When given, scales/zeros come from the
    importance-weighted least-squares search instead of the min/max range.
    Layout is identical; only s/z/q values differ.

    Fully batched over all (tile, group) blocks at once: scale/zero/code
    selection and packing are vectorized numpy ops on one [B,16,G] block
    array, processed in chunks to bound peak memory (see CHUNK). Deterministic.

    Returns dict with byte planes: 'weight', 'scale', 'zero' and metadata.
    """
    W = W.astype(np.float32, copy=False)
    N, K = W.shape
    Np = (N + 15) // 16 * 16
    Kp = (K + group_size - 1) // group_size * group_size
    n_tiles = Np // 16
    k_groups = Kp // group_size
    k16_per = group_size // 16
    k_group_count = k_groups
    G = group_size
    B = n_tiles * k_group_count
    if importance is not None:
        if importance.shape[0] != K:
            raise ValueError(f"importance length {importance.shape[0]} != K {K}")
        importance = np.ascontiguousarray(importance.astype(np.float32))
        # [k_groups, G] -> [B, G]: block (nt,g) gets input-channel group g
        hbig = np.broadcast_to(importance.reshape(k_groups, G),
                               (n_tiles, k_groups, G)).reshape(B, G)

    # Pad W to Np x Kp with q=z (U4Z) / 0 (S4), then reblock to [B,16,G].
    # Block order is the packing order: gidx = nt*k_group_count+g.
    Wp = np.zeros((Np, Kp), dtype=np.float32)
    Wp[:N, :K] = W
    Wb = np.ascontiguousarray(Wp.reshape(n_tiles, 16, k_groups, G)
                              .transpose(0, 2, 1, 3)).reshape(B, 16, G)

    n_workers = _quant_workers()
    # auto chunk so there are ~4x workers worth of chunks (parallel granularity);
    # env override GUFO_QUANT_CHUNK forces a fixed block count.
    env_chunk = os.environ.get("GUFO_QUANT_CHUNK")
    CHUNK = int(env_chunk) if env_chunk else max(1024, B // max(1, n_workers * 4))
    bounds = [(b0, min(B, b0 + CHUNK)) for b0 in range(0, B, CHUNK)]

    def run(bound):
        b0, b1 = bound
        hc = hbig[b0:b1, None, :] if importance is not None else None
        return _quantize_chunk(Wb[b0:b1], hc, symmetric, G, k16_per)

    if len(bounds) < 2 or n_workers < 2:
        parts = [run(b) for b in bounds]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            parts = list(ex.map(run, bounds))
    weight = b"".join(p[0] for p in parts)
    scale = b"".join(p[1] for p in parts)
    zero = b"".join(p[2] for p in parts)

    return {
        "weight": bytes(weight),
        "scale": bytes(scale),
        "zero": bytes(zero),
        "N": N, "K": K, "Np": Np, "Kp": Kp,
        "group_size": group_size,
        "symmetric": symmetric,
        "format": f"SHQ4_T16_V1_{'S4' if symmetric else 'U4Z'}_G{group_size}",
    }


def _quantize_shq8_batched(Wb: np.ndarray):
    """Batched SHQ8 scale/code selection over [B,16,G] blocks.

    Per lane: max-abs scale, codes = clamp(round(w/s), -127,127). Returns
    (s_bf [B,16] float, q [B,16,G] float codes). Constant rows -> zero scale.
    """
    B = Wb.shape[0]
    amax = np.maximum(np.abs(Wb).min(axis=2), np.abs(Wb).max(axis=2))  # use max abs
    amax = np.max(np.abs(Wb), axis=2)
    const = amax <= 0.0
    s_safe = np.where(const, 1.0, amax / S8_MAX)
    s_bf = _bf16r(s_safe)
    valid = (const == False) & (s_bf > 0.0)
    q = _clamp(_round_even(Wb / np.where(valid, s_bf, 1.0)[..., None]), -S8_MAX, S8_MAX)
    q = np.where(valid[..., None], q, 0.0)
    s = np.where(const, 0.0, s_bf)
    return s, q


def quantize_shq8(W: np.ndarray, group_size: int):
    """Quantize W[N,K] to SHQ8-T16 (signed INT8, no zero point). Batched."""
    W = W.astype(np.float32, copy=False)
    N, K = W.shape
    Np = (N + 15) // 16 * 16
    Kp = (K + group_size - 1) // group_size * group_size
    n_tiles = Np // 16
    k_groups = Kp // group_size
    k16_per = group_size // 16
    B = n_tiles * k_groups
    G = group_size

    Wp = np.zeros((Np, Kp), dtype=np.float32)
    Wp[:N, :K] = W
    Wb = np.ascontiguousarray(Wp.reshape(n_tiles, 16, k_groups, G)
                              .transpose(0, 2, 1, 3)).reshape(B, 16, G)

    s, q = _quantize_shq8_batched(Wb)  # [B,16], [B,16,G]
    sbytes = f32_to_bf16_uint16(s).astype(np.uint16).tobytes()
    # pack: [B,16,G] -> [B, k16_per, 16, 16] microtile order, byte per value
    qn = q.astype(np.int8).astype(np.uint8)
    qr = qn.reshape(B, 16, k16_per, 16).transpose(0, 2, 1, 3)  # [B,k16,lane,kp16]
    weight = qr.tobytes()
    return {
        "weight": bytes(weight),
        "scale": bytes(sbytes),
        "zero": b"",
        "N": N, "K": K, "Np": Np, "Kp": Kp,
        "group_size": group_size,
        "symmetric": False,
        "format": f"SHQ8_T16_V1_G{group_size}",
    }


def _quantize_shq6_batched(Wb: np.ndarray):
    """Batched SHQ6 scale/code selection over [B,16,G] blocks.

    Per lane: max-abs scale, signed 6-bit codes -32..31, dequant = s*q
    (symmetric, no zero point; Q6_K-class). Constant rows -> zero scale.
    """
    B = Wb.shape[0]
    amax = np.max(np.abs(Wb), axis=2)
    const = amax <= 0.0
    s_safe = np.where(const, 1.0, amax / S6_MAX)
    s_bf = _bf16r(s_safe)
    valid = (const == False) & (s_bf > 0.0)
    q = _clamp(_round_even(Wb / np.where(valid, s_bf, 1.0)[..., None]), -S6_MAX, S6_MAX)
    q = np.where(valid[..., None], q, 0.0)
    s = np.where(const, 0.0, s_bf)
    return s, q


def quantize_shq6(W: np.ndarray, group_size: int):
    """Quantize W[N,K] to SHQ6-T16 (signed 6-bit, Q6_K-class).

    Same T16 tile layout as SHQ4/SHQ8: microtile [k16][lane][kp16]. Codes are
    6-bit signed -32..31 packed 4-per-3-bytes (little-endian bit stream along
    kp). Scale plane: BF16 per output lane per K group (like SHQ8). Batched.
    """
    W = W.astype(np.float32, copy=False)
    N, K = W.shape
    Np = (N + 15) // 16 * 16
    Kp = (K + group_size - 1) // group_size * group_size
    n_tiles = Np // 16
    k_groups = Kp // group_size
    k16_per = group_size // 16
    B = n_tiles * k_groups
    G = group_size

    Wp = np.zeros((Np, Kp), dtype=np.float32)
    Wp[:N, :K] = W
    Wb = np.ascontiguousarray(Wp.reshape(n_tiles, 16, k_groups, G)
                              .transpose(0, 2, 1, 3)).reshape(B, 16, G)

    s, q = _quantize_shq6_batched(Wb)  # [B,16], [B,16,G]
    sbytes = f32_to_bf16_uint16(s).astype(np.uint16).tobytes()
    qn = (q.astype(np.int8).astype(np.uint8)) & 0x3F  # 6-bit codes
    qr = qn.reshape(B, 16, k16_per, 16).transpose(0, 2, 1, 3)  # [B,k16,lane,kp16]
    # pack 4 consecutive kp codes -> 3 bytes (little-endian 24-bit stream)
    qg = qr.reshape(B, k16_per, 16, 4, 4)  # [B,k16,lane,g4,kp4]
    v0, v1, v2, v3 = qg[..., 0], qg[..., 1], qg[..., 2], qg[..., 3]
    b0 = v0 | ((v1 & 3) << 6)
    b1 = (v1 >> 2) | ((v2 & 15) << 4)
    b2 = (v2 >> 4) | (v3 << 2)
    pack = np.stack([b0, b1, b2], axis=-1).astype(np.uint8)  # [B,k16,lane,g4,3]
    weight = pack.tobytes()
    return {
        "weight": bytes(weight),
        "scale": bytes(sbytes),
        "zero": b"",
        "N": N, "K": K, "Np": Np, "Kp": Kp,
        "group_size": group_size,
        "symmetric": False,
        "format": f"SHQ6_T16_V1_G{group_size}",
    }


def dequant_shq6(planes: dict) -> np.ndarray:
    """Decode SHQ6-T16 planes back to float32 W[Np,Kp]. CPU reference oracle."""
    weight = np.frombuffer(planes["weight"], dtype=np.uint8)
    scale = np.frombuffer(planes["scale"], dtype=np.uint16)
    Np, Kp = planes["Np"], planes["Kp"]
    G = planes["group_size"]
    n_tiles = Np // 16
    k_groups = Kp // G
    k16_per = G // 16
    B = n_tiles * k_groups

    wg = weight.reshape(B, k16_per, 16, 4, 3)
    b0 = wg[..., 0]; b1 = wg[..., 1]; b2 = wg[..., 2]
    v0 = b0 & 0x3F
    v1 = (b0 >> 6) | ((b1 & 0x0F) << 2)
    v2 = (b1 >> 4) | ((b2 & 0x03) << 4)
    v3 = (b2 >> 2) & 0x3F
    codes = np.stack([v0, v1, v2, v3], axis=-1).reshape(B, k16_per, 16, 4, 4)
    codes = codes.reshape(B, k16_per, 16, 16).transpose(0, 2, 1, 3).reshape(B, 16, G).astype(np.float32)
    codes = np.where(codes < 32, codes, codes - 64)  # signed -32..31
    s = bf16_uint16_to_f32(scale).reshape(B, 16)
    W = s[..., None] * codes
    W = W.reshape(n_tiles, k_groups, 16, G).transpose(0, 2, 1, 3).reshape(Np, Kp)
    return W


def dequant_shq4(planes: dict) -> np.ndarray:
    """Decode SHQ4-T16 planes back to float32 W[Np,Kp]. CPU reference oracle.

    Fully batched: unpack all block codes, scales and zeros into [B,16,G]
    arrays, reconstruct s*(q-z), then reblock to [Np,Kp]. Byte layout and
    semantics identical to the reference loop.
    """
    weight = np.frombuffer(planes["weight"], dtype=np.uint8)
    scale = np.frombuffer(planes["scale"], dtype=np.uint16)
    Np, Kp = planes["Np"], planes["Kp"]
    G = planes["group_size"]
    n_tiles = Np // 16
    k_groups = Kp // G
    k16_per = G // 16
    B = n_tiles * k_groups

    # codes: weight bytes [B][k16][lane][kp8] -> lo/hi nibbles interleaved
    wb = weight.reshape(B, k16_per, 16, 8)
    lo = (wb & 0x0F)
    hi = ((wb >> 4) & 0x0F)
    codes = np.stack([lo, hi], axis=-1).reshape(B, k16_per, 16, 16)  # [B,k16,lane,16]
    codes = codes.transpose(0, 2, 1, 3).reshape(B, 16, G).astype(np.float32)

    s = bf16_uint16_to_f32(scale).reshape(B, 16)
    if planes["symmetric"]:
        codes = np.where(codes < 8, codes, codes - 16)  # signed S4
        W = s[..., None] * codes
    else:
        zero = np.frombuffer(planes["zero"], dtype=np.uint8).reshape(B, 8)
        zi = np.stack([zero & 0x0F, (zero >> 4) & 0x0F], axis=-1).reshape(B, 16).astype(np.float32)
        W = s[..., None] * (codes - zi[..., None])
    # reblock [B(nt,g), lane, k] -> [nt, lane, g, k] -> [Np,Kp]
    W = W.reshape(n_tiles, k_groups, 16, G).transpose(0, 2, 1, 3).reshape(Np, Kp)
    return W


def dequant_shq8(planes: dict) -> np.ndarray:
    weight = np.frombuffer(planes["weight"], dtype=np.uint8)
    scale = np.frombuffer(planes["scale"], dtype=np.uint16)
    Np, Kp = planes["Np"], planes["Kp"]
    G = planes["group_size"]
    n_tiles = Np // 16
    k_groups = Kp // G
    k16_per = G // 16
    W = np.zeros((Np, Kp), dtype=np.float32)
    for nt in range(n_tiles):
        for g in range(k_groups):
            gidx = nt * k_groups + g
            for lane in range(16):
                s = float(bf16_uint16_to_f32(np.array([scale[gidx * 16 + lane]]))[0])
                for k16 in range(k16_per):
                    micro = (gidx * k16_per + k16) * 16 * 16 + lane * 16
                    for kp in range(16):
                        v = int(weight[micro + kp])
                        if v >= 128:
                            v -= 256
                        W[nt * 16 + lane, g * G + k16 * 16 + kp] = s * v
    return W

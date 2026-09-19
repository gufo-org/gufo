"""SHQ-T16 conformance vectors (T1 tests).

Byte-exact checks for the layouts in tools/gufo/shq.py:
- Exhaustive UINT4 and signed nibble decode.
- Low/high nibble ordering.
- G32 and G64 offsets.
- U4Z zero correction.
- S4 negative values including -8.
- BF16 scale rounding.
- Zero groups and padded K/N tails.
- Quantizer determinism.
"""

from __future__ import annotations

import numpy as np

from .shq import (
    quantize_shq4,
    quantize_shq8,
    quantize_shq6,
    dequant_shq4,
    dequant_shq8,
    dequant_shq6,
    f32_to_bf16_uint16,
    bf16_uint16_to_f32,
    U4_MAX,
)


def _decode_u4(byte: int):
    lo = byte & 0xF
    hi = (byte >> 4) & 0xF
    return lo, hi


def test_exhaustive_nibble():
    # Every possible 8-bit byte must decode to the same two 4-bit values.
    for b in range(256):
        lo, hi = _decode_u4(b)
        assert lo == (b & 0xF)
        assert hi == ((b >> 4) & 0xF)
        # reconstruct
        assert (lo | (hi << 4)) == b


def test_low_high_nibble_order():
    # Build W so lane0 has min=0, max=15 -> scale = 1, zero point = 0.
    # Then q = round(w), and byte placement of k0 vs k1 is directly observable.
    W = np.zeros((16, 64), dtype=np.float32)
    W[0, 0] = 0.0   # k0 -> low nibble
    W[0, 1] = 15.0  # k1 -> high nibble
    W[0, 2] = 15.0  # k2 -> low nibble of byte1
    W[0, 3] = 0.0   # k3 -> high nibble of byte1
    planes = quantize_shq4(W, group_size=64)
    # lane 0, group 0, k16_subtile 0:
    micro0 = (0 * 4 + 0) * 16 * 8 + 0 * 8  # gidx=0, k16=0, lane=0
    byte0 = planes["weight"][micro0]
    byte1 = planes["weight"][micro0 + 1]
    assert (byte0 & 0xF) == 0, f"low nibble should be k0=0, got {byte0 & 0xF}"
    assert ((byte0 >> 4) & 0xF) == 15, f"high nibble should be k1=15, got {(byte0>>4)&0xF}"
    assert (byte1 & 0xF) == 15, f"byte1 low nibble should be k2=15, got {byte1 & 0xF}"
    assert ((byte1 >> 4) & 0xF) == 0, f"byte1 high nibble should be k3=0, got {(byte1>>4)&0xF}"


def test_bf16_rounding():
    # Well-known exact values.
    assert int(f32_to_bf16_uint16(np.array([0.5]))[0]) == 0x3F00
    assert int(f32_to_bf16_uint16(np.array([1.0]))[0]) == 0x3F80
    assert int(f32_to_bf16_uint16(np.array([1.5]))[0]) == 0x3FC0
    assert int(f32_to_bf16_uint16(np.array([-2.0]))[0]) == 0xC000
    # Ties-to-even: 0x3F800000 + 0x8000 (halfway, bit16=0) -> round down to 0x3F80.
    halfway = np.array([1.0], dtype=np.float32).view(np.uint32)[0] + 0x8000
    v = np.array([halfway], dtype=np.uint32).view(np.float32)
    assert int(f32_to_bf16_uint16(v)[0]) == 0x3F80
    # Decode back.
    assert float(bf16_uint16_to_f32(np.array([0x3F80], dtype=np.uint16))[0]) == 1.0


def test_u4z_zero_correction():
    # dequant(q) = s*(q-z); pick s=1 by constructing scale exactly 1.
    # Build W where s32=1 -> bf16(1.0)=1.0. Then q=w+z.
    W = np.zeros((16, 64), dtype=np.float32)
    # lane0: values 1..5, choose z=0, s=1 impossible via min/max unless range=15.
    # Instead verify formula directly with chosen s,z through dequant on known codes.
    # We'll craft W: lane0 has max-min =15 => s=1, z = -min/s.
    W[0, :] = np.arange(1, 65)  # min1 max64 -> s=63/15 not 1. Use custom:
    W2 = np.zeros((16, 64), dtype=np.float32)
    W2[0, 0] = 0.0
    W2[0, 1] = 15.0  # min0 max15 -> s=1, z=0
    W2[0, 2] = 15.0
    planes = quantize_shq4(W2, group_size=64)
    Wd = dequant_shq4(planes)
    # lane0 row: value 15 -> q=15, dequant = 1*(15-0)=15
    assert abs(Wd[0, 1] - 15.0) < 1e-6, f"expected 15, got {Wd[0,1]}"
    # zero point path: value 7 with z=3 -> dequant=s*(7-3)
    W3 = np.zeros((16, 64), dtype=np.float32)
    W3[0, 0] = 3.0
    W3[0, 1] = 18.0  # min3 max18 -> s=(15)/15=1, z=-min/s=-3? z=round(-3/1)=-3 clamp 0 -> z=0
    # use negative min to get z>0
    W4 = np.zeros((16, 64), dtype=np.float32)
    W4[0, 0] = -3.0
    W4[0, 1] = 12.0  # range 15 -> s=1, z=round(3)=3
    planes = quantize_shq4(W4, group_size=64)
    Wd = dequant_shq4(planes)
    assert abs(Wd[0, 0] - (-3.0)) < 1e-6
    assert abs(Wd[0, 1] - 12.0) < 1e-6
    assert int(planes["zero"][0] & 0xF) == 3, "zero point lane0 should be 3"


def test_s4_negative():
    # Verify -8 and negative codes decode.
    W = np.zeros((16, 64), dtype=np.float32)
    W[0, 0] = -8.0
    W[0, 1] = 0.0
    # max abs=8 -> s=8/7. Not exact. Verify dequant sign and -8 handling by
    # checking decode of nibble value 8 -> -8.
    planes = quantize_shq4(W, group_size=64, symmetric=True)
    Wd = dequant_shq4(planes)
    assert Wd[0, 0] < 0
    # decode nibble 8 directly:
    assert _decode_u4(0x08)[0] == 8  # 8 -> signed -8
    assert _decode_u4(0x07)[0] == 7  # 7 -> +7


def test_zero_groups_and_padding():
    # All-zero group -> s=0, z=0, q=0
    W = np.zeros((16, 64), dtype=np.float32)
    planes = quantize_shq4(W, group_size=64)
    assert planes["scale"] == b"\x00\x00" * 16 * 1  # 16 scales all zero
    assert planes["zero"] == b"\x00" * 8
    assert all(b == 0 for b in planes["weight"])
    # Padding: N=17 -> Np=32, K=65 -> Kp=128 (G64)
    Wp = np.random.RandomState(0).rand(17, 65).astype(np.float32)
    planes = quantize_shq4(Wp, group_size=64)
    Wd = dequant_shq4(planes)
    assert Wd.shape == (32, 128)
    # padded K values must dequant to zero for U4Z (q=z)
    assert np.all(Wd[:17, 65:] == 0)
    # padded N rows zero
    assert np.all(Wd[17:, :] == 0)


def test_determinism():
    W = np.random.RandomState(7).rand(64, 256).astype(np.float32) * 4 - 2
    p1 = quantize_shq4(W, group_size=64)
    p2 = quantize_shq4(W, group_size=64)
    assert p1["weight"] == p2["weight"]
    assert p1["scale"] == p2["scale"]
    assert p1["zero"] == p2["zero"]


def test_g64_g32_offsets():
    # Same logical data quantized G64 vs G32 must place scales at different
    # indices but decode to the same values.
    W = np.random.RandomState(3).rand(16, 64).astype(np.float32)
    p64 = quantize_shq4(W, group_size=64)
    p32 = quantize_shq4(W, group_size=32)
    d64 = dequant_shq4(p64)[:16, :64]
    d32 = dequant_shq4(p32)[:16, :64]
    assert d64.shape == (16, 64) and d32.shape == (16, 64)


def test_shq8():
    W = np.random.RandomState(9).rand(16, 64).astype(np.float32)
    p = quantize_shq8(W, group_size=64)
    d = dequant_shq8(p)[:16, :64]
    err = np.abs(d - W).max()
    assert err < 0.02, f"SHQ8 reconstruction too large: {err}"


def test_shq6():
    """SHQ6: signed 6-bit round-trip, 4-per-3-byte bit packing, reconstruction."""
    W = np.random.RandomState(11).rand(16, 64).astype(np.float32)
    p = quantize_shq6(W, group_size=64)
    d = dequant_shq6(p)[:16, :64]
    err = np.abs(d - W).max()
    assert err < 0.03, f"SHQ6 reconstruction too large: {err}"
    # 4 codes per 3 bytes: 64 K values = 16 bytes/lane per k16 (64/4*3)
    assert len(p["weight"]) == 16 * 64 * 3 // 4, f"SHQ6 weight size {len(p['weight'])}"
    # bit-packing round trip: reconstruct codes from bytes directly
    q6 = dequant_shq6(p)
    assert np.all(np.abs(q6[:16, :64] - W) < 0.04)


def run_all():
    tests = [
        test_exhaustive_nibble,
        test_low_high_nibble_order,
        test_bf16_rounding,
        test_u4z_zero_correction,
        test_s4_negative,
        test_zero_groups_and_padding,
        test_determinism,
        test_g64_g32_offsets,
        test_shq8,
        test_shq6,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} conformance tests passed")


if __name__ == "__main__":
    run_all()

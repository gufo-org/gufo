"""Small representation checks, not kernel or model qualification."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gufo.shq import quantize_shq6


def main():
    for group, expected in ((32, 832), (64, 800)):
        planes = quantize_shq6(np.ones((16, 64), np.float32), group)
        size = sum(len(planes[key]) for key in ("weight", "scale", "zero"))
        assert size == expected, (group, size)
        print(f"SHQ6 G{group}: {size} bytes/1024 weights = {size/128} bpw")

    # Proposed microtile: low nibbles in 128 bytes, high bits in 32 bytes.
    # This tests an explicit logical lane-major packing, not WMMA lane mapping.
    signed = np.tile(np.arange(-16, 16, dtype=np.int32), 8)
    codes = signed & 31
    low = bytes(int(codes[i] & 15) | (int(codes[i+1] & 15) << 4)
                for i in range(0, 256, 2))
    high = bytes(sum(int((codes[i+j] >> 4) & 1) << j for j in range(8))
                 for i in range(0, 256, 8))
    restored = [(low[i//2] >> (4*(i%2)) & 15)
                - 16*(high[i//8] >> (i%8) & 1) for i in range(256)]
    assert restored == signed.tolist()
    assert len(low) + len(high) == 160
    assert (4*160+32)*8/1024 == 5.25
    assert (4*160+32+10)*8/1024 == 5.328125
    print("Proposed SHQ5: all 32 signed codes round-trip; S5=5.25, U5Z=5.328125 bpw")

    # An official E2M1 value 6*2^-8, grid factor 0.15 from the archived
    # fitter: the resulting scale cannot be encoded by an E8M0 exponent.
    scale = np.float32(6 * 2**-8) * np.float32(0.15)
    mantissa, exponent = np.frexp(scale)
    assert mantissa != 0.5
    print(f"Allowed floating-search scale example: {scale:.10g}, "
          f"frexp=({mantissa}, {exponent}); not an E8M0 power of two")

    # Stronger counterexample: run the archived q2_sym grid on a constructed
    # block of legal official MXFP4 values. This is not a sampled model block.
    block = np.tile(np.array([-6, -4, -3, -2, -1.5, -1, -.5, 0,
                              0, .5, 1, 1.5, 2, 3, 4, 6], np.float32), 2) * 2**-8
    fits = []
    for factor in np.linspace(0.15, 0.5, 29):
        candidate_scale = float(np.abs(block).max()) * float(factor)
        q = np.clip(np.round((block/candidate_scale-1)/2), -2, 1)*2+1
        fits.append((float(np.square(q*candidate_scale-block).sum()), candidate_scale))
    error, best_scale = min(fits)
    assert np.frexp(best_scale)[0] != 0.5
    print(f"Constructed legal MXFP4 block: best grid scale={best_scale:.10g}, "
          f"squared error={error:.10g}; selected scale is not a power of two")
    print("PASS (representation checks only)")


if __name__ == "__main__":
    main()

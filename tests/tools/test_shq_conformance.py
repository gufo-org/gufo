"""Unit test runner for SHQ4/SHQ6/SHQ8 quantization and packing conformance."""

from __future__ import annotations

import unittest
from tools.gufo import conformance


class TestShqConformance(unittest.TestCase):

    def test_exhaustive_nibble(self):
        conformance.test_exhaustive_nibble()

    def test_low_high_nibble_order(self):
        conformance.test_low_high_nibble_order()

    def test_bf16_rounding(self):
        conformance.test_bf16_rounding()

    def test_u4z_zero_correction(self):
        conformance.test_u4z_zero_correction()

    def test_s4_negative(self):
        conformance.test_s4_negative()

    def test_zero_groups_and_padding(self):
        conformance.test_zero_groups_and_padding()

    def test_determinism(self):
        conformance.test_determinism()

    def test_g64_g32_offsets(self):
        conformance.test_g64_g32_offsets()

    def test_shq8(self):
        conformance.test_shq8()

    def test_shq6(self):
        conformance.test_shq6()


if __name__ == "__main__":
    unittest.main()

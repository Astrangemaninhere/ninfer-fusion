"""FP8 row-scaled quantization: the rule recovered by inverting a release."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from tools.artifact.layouts import dequantize_fp8_row_scaled, encoded_size, row_scale_geometry
from tools.convert.common import quantize as Q


class Fp8RowQuantizeTest(unittest.TestCase):
    def test_scale_is_amax_over_448_rounded_through_bf16(self) -> None:
        weight = torch.randn(16, 128, dtype=torch.float32)
        quantized = Q.quantize_fp8_row_matrix(weight)
        amax = weight.abs().amax(dim=1)
        expected = (
            torch.from_numpy((amax.numpy().astype(np.float64) / 448.0).astype(np.float32))
            .to(torch.bfloat16)
        )
        self.assertTrue(torch.equal(quantized.row_scales, expected))

    def test_codes_reconstruct_within_the_e4m3fn_step(self) -> None:
        weight = torch.randn(8, 256, dtype=torch.float32)
        payload = Q.quantize_and_encode_fp8_row(weight)
        geometry = row_scale_geometry("FP8_E4M3FN_ROW_BF16S", tuple(weight.shape))
        self.assertEqual(len(payload), encoded_size("row-scale-v1", "FP8_E4M3FN_ROW_BF16S", tuple(weight.shape)))
        self.assertEqual(len(payload), geometry.payload_bytes)
        restored = dequantize_fp8_row_scaled(payload, tuple(weight.shape), dtype=torch.float32)
        quantized = Q.quantize_fp8_row_matrix(weight)
        # E4M3FN carries three mantissa bits, so the step at magnitude m is m/8 and
        # the nearest-code error cannot exceed m/16. The coarsest row step is
        # 448 * row_scale, hence the absolute bound below.
        bound = (quantized.row_scales.float() * (448.0 / 16.0)).unsqueeze(1)
        error = (restored - weight).abs()
        self.assertTrue(bool((error <= bound + 1e-6).all()))

    def test_every_nonzero_row_pins_its_largest_code_to_448(self) -> None:
        """``scale == bf16(amax/448)`` forces the row maximum to E4M3FN's 448.

        BF16 rounding moves the scale by at most 0.4%, so ``amax/scale`` stays
        inside the interval that rounds to 448; this is the structural invariant
        the released artifacts satisfy.
        """

        weight = torch.randn(32, 128, dtype=torch.float32) * 4.0
        quantized = Q.quantize_fp8_row_matrix(weight)
        # magnitude, not the raw byte: -448 encodes as 0xFE, which is the largest byte
        magnitude = quantized.codes & 0x7F
        self.assertTrue(bool((magnitude.amax(dim=1) == 0x7E).all()))

    def test_saturating_row_reconstructs_within_the_scale_rounding(self) -> None:
        weight = torch.full((4, 64), 1.0, dtype=torch.float32)
        weight[0, 0] = 7.5
        payload = Q.quantize_and_encode_fp8_row(weight)
        restored = dequantize_fp8_row_scaled(payload, (4, 64), dtype=torch.float32)
        # the residual is the BF16 rounding of the row multiplier, not a code error
        self.assertLess(abs(float(restored[0, 0]) - 7.5) / 7.5, 5e-3)
        self.assertAlmostEqual(float(restored[0, 1]), 1.0, delta=0.01)

    def test_zero_row_keeps_a_zero_multiplier_and_signed_zero_codes(self) -> None:
        weight = torch.randn(4, 64, dtype=torch.float32)
        weight[2] = 0.0
        payload = Q.quantize_and_encode_fp8_row(weight)
        # encode_fp8_row_scaled validates zero-scale rows itself; decoding proves it
        restored = dequantize_fp8_row_scaled(payload, (4, 64), dtype=torch.float32)
        self.assertTrue(torch.equal(restored[2], torch.zeros(64)))

    def test_non_finite_input_is_refused(self) -> None:
        weight = torch.randn(4, 64, dtype=torch.float32)
        weight[1, 3] = float("inf")
        with self.assertRaises(ValueError):
            Q.quantize_fp8_row_matrix(weight)

    def test_rank_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            Q.quantize_fp8_row_matrix(torch.randn(8, dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()

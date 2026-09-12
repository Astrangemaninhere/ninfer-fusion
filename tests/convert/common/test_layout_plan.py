#!/usr/bin/env python3
"""Structural layout selection: the contract that must not drift.

The point of these tests is that ``layout_plan`` decides from STRUCTURE.  Each
case below is a shape/dtype relationship, so a regression here means the import
would start keying on something else (a name, a family, a hardcoded list).

Run: python3 -m unittest tests.convert.common.test_layout_plan
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.convert.common.layout_plan import (  # noqa: E402
    ENCODER_DIRECT,
    ENCODER_FP8_ROW,
    ENCODER_NVFP4,
    LAYOUT_BLOCKSCALE,
    LAYOUT_CONTIGUOUS,
    LAYOUT_ROW_SCALE,
    Choice,
    Refusal,
    TensorMeta,
    group_by_module,
    plan_census,
    select,
)


def meta(dtype: str, shape) -> TensorMeta:
    return TensorMeta(dtype, tuple(shape))


class TestDispatchOrder(unittest.TestCase):
    """Codes dtype decides first; a plain tensor is never an unrecognised group."""

    def test_plain_bf16_weight_is_contiguous(self):
        d = select({"weight": meta("BF16", (248320, 5120))})
        self.assertIsInstance(d, Choice)
        self.assertEqual(d.layout, LAYOUT_CONTIGUOUS)
        self.assertEqual(d.encoder, ENCODER_DIRECT)

    def test_plain_vector_is_contiguous(self):
        d = select({"weight": meta("BF16", (5120,))})
        self.assertIsInstance(d, Choice)
        self.assertEqual(d.layout, LAYOUT_CONTIGUOUS)

    def test_group_without_codes_is_contiguous(self):
        d = select({"A_log": meta("F32", (128,))})
        self.assertIsInstance(d, Choice)
        self.assertEqual(d.layout, LAYOUT_CONTIGUOUS)

    def test_non_direct_codes_without_companion_is_refused(self):
        # I8 is neither directly representable nor a codec the layouts know, and
        # there is no scale companion to hint at one -> unrecognised, and the
        # reason must name BOTH facts so the operator knows what is missing.
        d = select({"weight": meta("I8", (4, 4))})
        self.assertIsInstance(d, Refusal)
        self.assertEqual(d.code, "F-UNRECOGNISED-GROUP")
        self.assertIn("no quantisation companion", d.reason)
        self.assertIn("I8", d.missing_mechanism)

    def test_unsupported_dtype_among_direct_members_is_refused(self):
        # A directly representable code tensor WITH a stray undecodable companion
        # is the case the direct branch must refuse rather than pass through.
        d = select({"weight": meta("BF16", (8, 8)), "odd": meta("I8", (2, 2))})
        self.assertIsInstance(d, Refusal)
        self.assertEqual(d.code, "F-UNSUPPORTED-DTYPE")


class TestNvfp4Structure(unittest.TestCase):
    """U8 codes [N,K/2] + F8_E4M3 scales [N,K/16] is NVFP4, by ratio not by name."""

    def test_ratio_8_is_nvfp4(self):
        d = select({"weight": meta("U8", (17408, 2560)),
                    "weight_scale": meta("F8_E4M3", (17408, 320)),
                    "weight_scale_2": meta("F32", ())})
        self.assertIsInstance(d, Choice)
        self.assertEqual(d.layout, LAYOUT_BLOCKSCALE)
        self.assertEqual(d.encoder, ENCODER_NVFP4)
        self.assertTrue(any("weight_divisor" in x for x in d.deviations))

    def test_wrong_ratio_is_refused(self):
        # 2560 code cols with 640 scale cols is ratio 4, i.e. group 32, not 16.
        d = select({"weight": meta("U8", (17408, 2560)),
                    "weight_scale": meta("F8_E4M3", (17408, 640))})
        self.assertIsInstance(d, Refusal)

    def test_layout_constraint_is_refused_and_named(self):
        # n=100 is not divisible by 128 -> the artifact layout cannot hold it.
        d = select({"weight": meta("U8", (100, 32)),
                    "weight_scale": meta("F8_E4M3", (100, 4))})
        self.assertIsInstance(d, Refusal)
        self.assertEqual(d.code, "F-LAYOUT-CONSTRAINT")
        self.assertIn("128", d.missing_mechanism + d.reason)


class TestFp8RowScale(unittest.TestCase):
    """Scalar scale is the degenerate per-row case: broadcast + record the narrowing."""

    def test_scalar_scale_broadcasts_with_recorded_deviation(self):
        d = select({"weight": meta("F8_E4M3", (5120, 6144)),
                    "weight_scale": meta("F32", ())})
        self.assertIsInstance(d, Choice)
        self.assertEqual(d.layout, LAYOUT_ROW_SCALE)
        self.assertEqual(d.encoder, ENCODER_FP8_ROW)
        self.assertTrue(any("broadcast" in x for x in d.deviations))

    def test_per_row_scale_taken_directly(self):
        d = select({"weight": meta("F8_E4M3", (5120, 6144)),
                    "weight_scale": meta("F32", (5120,))})
        self.assertIsInstance(d, Choice)
        self.assertEqual(d.layout, LAYOUT_ROW_SCALE)
        self.assertTrue(any("directly" in x for x in d.deviations))

    def test_scale_row_count_mismatch_is_refused(self):
        d = select({"weight": meta("F8_E4M3", (5120, 6144)),
                    "weight_scale": meta("F32", (4096,))})
        self.assertIsInstance(d, Refusal)
        self.assertEqual(d.code, "F-SCALE-SHAPE")


class TestDeclarationCrossCheck(unittest.TestCase):
    """The declaration is a cross-check: disagreement is recorded, not obeyed."""

    def test_declaration_disagreeing_with_structure_is_recorded(self):
        d = select({"weight": meta("U8", (17408, 2560)),
                    "weight_scale": meta("F8_E4M3", (17408, 320))},
                   declared_algo="FP8")
        self.assertIsInstance(d, Choice)
        self.assertTrue(any("disagrees" in x for x in d.deviations))

    def test_matching_declaration_adds_no_deviation(self):
        d = select({"weight": meta("U8", (17408, 2560)),
                    "weight_scale": meta("F8_E4M3", (17408, 320))},
                   declared_algo="NVFP4")
        self.assertFalse(any("disagrees" in x for x in d.deviations))


class TestGrouping(unittest.TestCase):
    def test_group_by_module_splits_on_last_dot(self):
        groups = group_by_module({
            "model.layers.0.mlp.gate_proj.weight": meta("U8", (2, 2)),
            "model.layers.0.mlp.gate_proj.weight_scale": meta("F8_E4M3", (2, 1)),
            "lm_head.weight": meta("BF16", (8, 8)),
        })
        self.assertIn("model.layers.0.mlp.gate_proj", groups)
        self.assertEqual(set(groups["model.layers.0.mlp.gate_proj"]), {"weight", "weight_scale"})
        self.assertIn("lm_head", groups)


class TestPlanCensus(unittest.TestCase):
    def test_histogram_and_no_refusals(self):
        census = {
            "a.weight": meta("U8", (128, 32)),
            "a.weight_scale": meta("F8_E4M3", (128, 4)),
            "b.weight": meta("F8_E4M3", (10, 100)),
            "b.weight_scale": meta("F32", ()),
            "c.weight": meta("BF16", (4, 4)),
        }
        hist, refusals = plan_census(census, {"a": "NVFP4", "b": "FP8"})
        self.assertEqual(refusals, [])
        self.assertEqual(hist[LAYOUT_BLOCKSCALE], 1)
        self.assertEqual(hist[LAYOUT_ROW_SCALE], 1)
        self.assertEqual(hist[LAYOUT_CONTIGUOUS], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

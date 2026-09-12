"""MTP assembly: the byte-exact rebuild of the 12 ``mtp/*`` objects.

The assembly reproduces the registered group-scale rule host-side and streams row
segments, so the tests here pin two independent properties: the rule matches the
repository's own grouped quantizer bit-for-bit (including the K-padding and
scale-underflow paths the released artifacts never exercise), and - when the two
real files are present - the produced payloads equal the released ones byte for
byte.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.artifact.layouts import row_split_geometry
from tools.artifact.numeric import get_format
from tools.convert.common import quantize as repo_quantize
from tools.convert.qwen3_8_27b import mtp


MTP_SOURCE = Path(os.environ.get(
    "NINFER_MTP_SOURCE",
    "/home/user/models/q38_abl_huihui_nvfp4/model-mtp-bf16.safetensors",
))
REFERENCE = Path(os.environ.get(
    "NINFER_QWEN3_8_27B_NVFP4_WEIGHTS",
    "/home/user/models/qwen3_8_27b_nvfp4.ninfer",
))

W8 = "W8G32_F16S"


def _bits(array: np.ndarray) -> np.ndarray:
    """Bit pattern of a float16/float32 array, for exact comparison."""

    view = np.uint16 if array.dtype == np.float16 else np.uint32
    return array.view(view)


class ScaleRuleTest(unittest.TestCase):
    def test_matches_the_repository_quantizer_on_random_input(self) -> None:
        rng = np.random.default_rng(20260912)
        for shape in ((7, 3), (1, 1), (64, 17)):
            with self.subTest(shape=shape):
                max_abs = (rng.random(shape, dtype=np.float32) * 12.0).astype(np.float32)
                scales, reciprocals = mtp.canonical_scale_words(max_abs, 127)
                expected, expected_reciprocals = repo_quantize._canonical_scale_words(
                    torch.from_numpy(max_abs), 127
                )
                np.testing.assert_array_equal(
                    _bits(scales), _bits(expected.numpy())
                )
                np.testing.assert_array_equal(reciprocals, expected_reciprocals.numpy())

    def test_zero_groups_keep_a_zero_scale_and_reciprocal(self) -> None:
        scales, reciprocals = mtp.canonical_scale_words(np.zeros((4, 4), np.float32), 127)
        self.assertTrue(bool((scales == 0).all()))
        self.assertTrue(bool((reciprocals == 0).all()))

    def test_positive_underflow_floors_at_the_smallest_subnormal(self) -> None:
        tiny = np.full((3, 3), 1e-30, dtype=np.float32)
        scales, _ = mtp.canonical_scale_words(tiny, 127)
        self.assertTrue(bool((scales == np.float16(2.0**-24)).all()))

    def test_scale_overflow_raises_like_the_repository(self) -> None:
        huge = np.full((2, 2), 1e30, dtype=np.float32)
        with self.assertRaises(ValueError):
            mtp.canonical_scale_words(huge, 127)
        with self.assertRaises(ValueError):
            repo_quantize._canonical_scale_words(torch.from_numpy(huge), 127)

    def test_round_half_ties_agree(self) -> None:
        # a group maximum that lands exactly between two float16 values
        boundary = np.float32(2.0**-14 * 1.5)
        max_abs = np.full((2, 5), boundary, dtype=np.float32)
        scales, _ = mtp.canonical_scale_words(max_abs, 127)
        expected, _ = repo_quantize._canonical_scale_words(torch.from_numpy(max_abs), 127)
        np.testing.assert_array_equal(_bits(scales), _bits(expected.numpy()))


class SegmentEncodingTest(unittest.TestCase):
    def _compare(self, rows: int, k: int) -> None:
        rng = np.random.default_rng(rows * 1000 + k)
        logical = rng.standard_normal((rows, k), dtype=np.float32)
        spec = get_format(W8)
        geometry = row_split_geometry(spec, (rows, k))

        base, scales = mtp.quantize_segment(logical, spec, geometry.k_pad)

        quantized = repo_quantize.quantize_matrix(torch.from_numpy(logical), W8)
        np.testing.assert_array_equal(
            base.reshape(rows, geometry.groups_per_row, spec.group_size).astype(np.int8),
            quantized.codes.numpy(),
        )
        np.testing.assert_array_equal(_bits(scales), _bits(quantized.scales.numpy()))

    def test_aligned_k_matches_the_repository(self) -> None:
        self._compare(rows=64, k=128)

    def test_unaligned_k_padding_matches_the_repository(self) -> None:
        # K is padded to 128, so the padding groups exist only on this path
        self._compare(rows=48, k=96)
        self._compare(rows=16, k=320)

    def test_segment_size_never_drops_below_one_group(self) -> None:
        for k in (128, 5120, 17408):
            self.assertGreaterEqual(mtp.segment_geometry(k), mtp._SEGMENT_ROW_QUANTUM)


class FusedRowOrderTest(unittest.TestCase):
    def test_attention_rows_are_head_interleaved_not_contiguous(self) -> None:
        """The fused attention matrix takes the first/second 256 rows of each head.

        A contiguous split of ``q_proj`` would be the natural thing to write and it
        is wrong; this pins the interleaving (per-head stride of 512 rows, query at
        the head base and output-gate at base + 256) so a later "simplification"
        fails here instead of silently changing the model.
        """

        runs = mtp.QKGV_CANDIDATES[mtp.QKGV_ORDER]
        q_runs = [run for run in runs if run.source_key.endswith("q_proj.weight")]
        self.assertEqual(len(q_runs), 2 * mtp.HEADS)
        query_runs, gate_runs = q_runs[: mtp.HEADS], q_runs[mtp.HEADS :]
        for head, run in enumerate(query_runs):
            self.assertEqual(
                (run.source_row, run.rows), (head * mtp.HEAD_ROWS, mtp.PART_ROWS)
            )
        for head, run in enumerate(gate_runs):
            self.assertEqual(
                (run.source_row, run.rows),
                (head * mtp.HEAD_ROWS + mtp.PART_ROWS, mtp.PART_ROWS),
            )

    def test_gate_up_is_gate_then_up(self) -> None:
        runs = mtp.GATE_UP_CANDIDATES[mtp.GATE_UP_ORDER]
        self.assertTrue(runs[0].source_key.endswith("gate_proj.weight"))
        self.assertTrue(runs[1].source_key.endswith("up_proj.weight"))

    def test_object_plan_covers_the_twelve_registered_names(self) -> None:
        names = [spec.object_name for spec in mtp.MTP_OBJECTS]
        self.assertEqual(len(names), 12)
        self.assertEqual(sorted(names), sorted(set(names)))
        self.assertIn("mtp/input_projection", names)
        self.assertIn("mtp/layer/attention/query_key_gate_value", names)

    def test_rows_of_each_fused_object_match_its_declared_shape(self) -> None:
        for spec in mtp.MTP_OBJECTS:
            if not isinstance(spec, mtp.MatrixObject):
                continue
            rows = sum(run.rows for run in spec.runs)
            self.assertEqual(rows, spec.shape[0], spec.object_name)


@unittest.skipUnless(MTP_SOURCE.is_file(), f"missing {MTP_SOURCE}")
class RealMaterialTest(unittest.TestCase):
    def test_twelve_objects_are_byte_exact_against_the_reference(self) -> None:
        if not REFERENCE.is_file():
            self.skipTest(f"missing {REFERENCE}")
        from tools.artifact.container import Artifact

        produced = mtp.build_mtp_objects(MTP_SOURCE)
        self.assertEqual(len(produced), 12)
        with Artifact.open(REFERENCE) as artifact:
            by_name = {obj.name: obj for obj in artifact.objects}
            for name, payload in produced.items():
                obj = by_name[name]
                with open(artifact.path, "rb") as handle:
                    handle.seek(artifact.payload_offset + obj.offset)
                    stored = handle.read(obj.bytes)
                self.assertEqual(len(payload), len(stored), name)
                self.assertEqual(payload, stored, name)


if __name__ == "__main__":
    unittest.main()

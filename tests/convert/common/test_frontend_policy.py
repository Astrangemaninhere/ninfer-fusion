"""Frontend resource resolution: the pin is a revision pin, not the requirement."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.convert.qwen3_6.common import frontend_policy as policy


def _write_tokenizer(path: Path, vocab: dict[str, int] | None = None) -> None:
    path.write_text(json.dumps({"model": {"vocab": vocab or {"a": 0, "b": 1}}}), encoding="utf-8")


QWEN_CONFIG = {
    "model_type": "qwen3_5",
    "vision_start_token_id": 5,
    "image_token_id": 6,
    "text_config": {"vocab_size": 8},
}
MISSING_PIN = "0" * 64


class FrontendPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.source = self.tmp / "source"
        self.source.mkdir()

    def resolve(self, pins: dict[str, str], roots: list[Path] | None = None) -> policy.FrontendProfile:
        return policy.resolve_frontend_profile(
            self.source, QWEN_CONFIG, pins=pins, roots=roots or [],
        )

    def test_pinned_resource_is_accepted_without_evidence(self) -> None:
        _write_tokenizer(self.source / "tokenizer.json")
        digest = policy.sha256_file(self.source / "tokenizer.json")

        profile = self.resolve({"frontend/tokenizer.json": digest})

        resolution = profile.by_name["frontend/tokenizer.json"]
        self.assertEqual(resolution.status, "pinned")
        self.assertIn("钉死值一致", resolution.evidence)
        self.assertIsNone(policy.acceptability_error(profile, allow_unproven=False))

    def test_drifted_but_consistent_tokenizer_is_accepted(self) -> None:
        _write_tokenizer(self.source / "tokenizer.json", {"a": 0, "b": 5, "c": 6, "d": 7})

        profile = self.resolve({"frontend/tokenizer.json": MISSING_PIN})

        resolution = profile.by_name["frontend/tokenizer.json"]
        self.assertEqual(resolution.status, "consistent")
        self.assertIn("特殊 token id 全部对得上", resolution.evidence)
        self.assertIsNone(policy.acceptability_error(profile, allow_unproven=False))

    def test_missing_special_token_makes_the_tokenizer_unproven(self) -> None:
        _write_tokenizer(self.source / "tokenizer.json", {"a": 0, "b": 1})

        profile = self.resolve({"frontend/tokenizer.json": MISSING_PIN})

        self.assertEqual(profile.by_name["frontend/tokenizer.json"].status, "unproven")
        error = policy.acceptability_error(profile, allow_unproven=False)
        self.assertIsNotNone(error)
        self.assertIn("无法证明等价", error)
        self.assertIsNone(policy.acceptability_error(profile, allow_unproven=True))

    def test_vocab_larger_than_declared_is_unproven(self) -> None:
        _write_tokenizer(self.source / "tokenizer.json", {"a": 0, "b": 5, "c": 6, "d": 99})

        profile = self.resolve({"frontend/tokenizer.json": MISSING_PIN})

        resolution = profile.by_name["frontend/tokenizer.json"]
        self.assertEqual(resolution.status, "unproven")
        self.assertIn("超出 vocab_size", resolution.evidence)

    def test_absent_resource_is_fatal_and_names_the_file(self) -> None:
        profile = self.resolve({"frontend/tokenizer.json": MISSING_PIN})

        self.assertEqual(profile.missing, ("frontend/tokenizer.json",))
        error = policy.acceptability_error(profile, allow_unproven=True)
        self.assertIsNotNone(error)
        self.assertIn("tokenizer.json", error)
        self.assertIn("revision", error)

    def test_pinned_copy_elsewhere_beats_a_drifted_local_file(self) -> None:
        official = self.tmp / "official"
        official.mkdir()
        _write_tokenizer(self.source / "tokenizer.json", {"a": 0, "b": 5, "c": 6, "d": 7})
        _write_tokenizer(official / "tokenizer.json", {"a": 0})
        digest = policy.sha256_file(official / "tokenizer.json")

        profile = self.resolve({"frontend/tokenizer.json": digest}, roots=[self.tmp])

        resolution = profile.by_name["frontend/tokenizer.json"]
        self.assertEqual(resolution.status, "pinned")
        self.assertEqual(resolution.provider, official / "tokenizer.json")
        self.assertIn("本地钉死副本", resolution.evidence)

    def test_candidate_from_a_different_family_is_refused(self) -> None:
        stranger = self.tmp / "stranger"
        stranger.mkdir()
        _write_tokenizer(stranger / "tokenizer.json", {"a": 0, "b": 5, "c": 6, "d": 7})
        (stranger / "config.json").write_text(
            json.dumps({"model_type": "xlm-roberta", "text_config": {"vocab_size": 250002}}),
            encoding="utf-8",
        )

        profile = self.resolve({"frontend/tokenizer.json": MISSING_PIN}, roots=[self.tmp])

        resolution = profile.by_name["frontend/tokenizer.json"]
        self.assertEqual(resolution.status, "missing")
        self.assertIn("词表不符", resolution.evidence)

    def test_second_local_copy_corroborates_the_id_map(self) -> None:
        peer = self.tmp / "peer"
        peer.mkdir()
        vocab = {"a": 0, "b": 5, "c": 6, "d": 7}
        _write_tokenizer(self.source / "tokenizer.json", dict(vocab))
        (peer / "tokenizer.json").write_text(
            json.dumps({"model": {"vocab": dict(vocab)}, "added_tokens": []}), encoding="utf-8",
        )
        self.assertNotEqual(
            policy.sha256_file(self.source / "tokenizer.json"),
            policy.sha256_file(peer / "tokenizer.json"),
        )

        profile = self.resolve({"frontend/tokenizer.json": MISSING_PIN}, roots=[self.tmp])

        resolution = profile.by_name["frontend/tokenizer.json"]
        self.assertEqual(resolution.status, "consistent")
        self.assertIn("逐项相同", resolution.evidence)

    def test_unproven_chat_template_needs_the_explicit_opt_in(self) -> None:
        (self.source / "chat_template.jinja").write_text("{{ prompt }}", encoding="utf-8")

        profile = self.resolve({"frontend/chat_template.jinja": MISSING_PIN})

        self.assertEqual(profile.by_name["frontend/chat_template.jinja"].status, "unproven")
        self.assertIsNotNone(policy.acceptability_error(profile, allow_unproven=False))
        self.assertIsNone(policy.acceptability_error(profile, allow_unproven=True))

    def test_report_records_every_deviation(self) -> None:
        _write_tokenizer(self.source / "tokenizer.json", {"a": 0, "b": 5, "c": 6, "d": 7})

        profile = self.resolve({"frontend/tokenizer.json": MISSING_PIN})
        report = profile.report()

        entry = report["resources"]["frontend/tokenizer.json"]
        self.assertEqual(entry["status"], "consistent")
        self.assertEqual(entry["provider"], str(self.source / "tokenizer.json"))
        self.assertEqual(entry["sha256"], policy.sha256_file(self.source / "tokenizer.json"))


if __name__ == "__main__":
    unittest.main()

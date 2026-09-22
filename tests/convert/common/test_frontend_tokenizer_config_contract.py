"""The engine validates tokenizer_config.json, so resource provenance must obey THAT contract.

Regression for the 2026-09-14 ornith artifact (``/home/user/scratch/p29b/art/qwen3_5_9b.ninfer``):

``tools/convert/qwen3_6/common/frontend_policy.py`` admitted a ``tokenizer_config.json`` on
``vocab_size`` alone -- the test that belongs to ``tokenizer.json``, whose id space must match --
so ``models/Qwen3.8-27B-NVFP4/tokenizer_config.json`` (a *multimodal* config: ``pad_token``
``'<|im_end|>'`` and **no** ``chat_template`` key at all) was embedded into a ``qwen3.5-9b``
artifact.  The two checkpoints share ``vocab_size``, so the proxy could not see the difference,
and the engine then refused the artifact at
``src/targets/qwen3_6/impl/frontend/frontend.cpp:205``:

    tokenizer_config.json does not match Qwen3.6 tokenizer prefix semantics

Two same-vocabulary models are not interchangeable for ``tokenizer_config.json``.  The fix is a
*tightening*: the added predicate can only remove candidates, never add one, and it makes the
engine's validator the authority instead of a weaker restatement of it.

Self-locating, so it runs from any cwd (the repo has no pytest and no convert test is registered
with ctest):

    python3 tests/convert/common/test_frontend_tokenizer_config_contract.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.convert.qwen3_6.common import frontend_policy as policy  # noqa: E402

FRONTEND_CPP = REPO / "src/targets/qwen3_6/impl/frontend/frontend.cpp"
CHAT_TEMPLATE_CPP = REPO / "src/targets/qwen3_6/impl/frontend/chat_template.cpp"
ARTIFACT = Path("/home/user/scratch/p29b/art/qwen3_5_9b.ninfer")

MISSING_PIN = "0" * 64
QWEN_CONFIG = {"model_type": "qwen3_5", "text_config": {"vocab_size": 8}}

#: What the artifact actually ships in frontend/tokenizer_config.json (1121 B, sha256
#: e5d078b0…): no `add_bos_token`, pad_token '<|im_end|>', no chat_template.  Reproduced so the
#: regression runs without the artifact, and cross-checked against the real bytes when present.
POISON_CONFIG = {
    "add_prefix_space": False,
    "bos_token": "<|endoftext|>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "<|im_end|>",
    "model_max_length": 262144,
    "pad_token": "<|im_end|>",
    "split_special_tokens": False,
    "tokenizer_class": "Qwen2Tokenizer",
}

CONFORMING = {"add_bos_token": False, "add_prefix_space": False,
              "pad_token": "<|endoftext|>"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def engine_template_digests() -> dict[str, str]:
    """The digests chat_template.cpp:414-424 accepts, assembled from the engine's own bytes."""

    source = CHAT_TEMPLATE_CPP.read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for name, body in re.findall(
            r"constexpr Sha256Digest k(\w+?)TemplateDigest\{(.*?)\};", source, re.S):
        found[name] = bytes(int(b, 16) for b in re.findall(r"0x([0-9a-fA-F]{2})", body)).hex()
    return found


class TokenizerConfigProvenanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.source = self.tmp / "source"
        self.source.mkdir()
        (self.source / "config.json").write_text(json.dumps(QWEN_CONFIG), encoding="utf-8")
        (self.source / "tokenizer.json").write_text(
            json.dumps({"model": {"vocab": {"a": 0, "b": 1}}}), encoding="utf-8")

    def provider(self, name: str, config: dict, template: bytes | None) -> Path:
        """A sibling checkpoint sharing vocab_size, so the old vocab proxy could not refuse it."""

        directory = self.tmp / name
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps({"model_type": "qwen3_8", "text_config": {"vocab_size": 8}}),
            encoding="utf-8")
        (directory / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
        if template is not None:
            (directory / "chat_template.jinja").write_bytes(template)
        return directory

    def resolve(self) -> "policy.FrontendProfile":
        return policy.resolve_frontend_profile(
            self.source, QWEN_CONFIG,
            pins={"frontend/tokenizer_config.json": MISSING_PIN,
                  "frontend/tokenizer.json": MISSING_PIN},
            roots=[self.tmp])

    # ---- the defect ----------------------------------------------------------------
    def test_multimodal_tokenizer_config_is_not_admitted_on_vocab_alone(self) -> None:
        """A same-vocab but contract-violating config must not be offered at all."""

        self.provider("Qwen3.8-27B-NVFP4", POISON_CONFIG, b"{{ prompt }}  # that provider\n")

        resolution = self.resolve().by_name["frontend/tokenizer_config.json"]
        self.assertEqual(
            resolution.status, "missing",
            "a tokenizer_config.json the engine refuses at frontend.cpp:205 was offered as %r "
            "with evidence %r" % (resolution.status, resolution.evidence))
        self.assertIn("tokenizer_config", resolution.evidence)

    def test_the_artifacts_own_bytes_are_refused_as_a_candidate(self) -> None:
        """Feed the real poisoned bytes (and the real template) through the resolver."""

        if not ARTIFACT.is_file():
            self.skipTest("SKIP: no artifact at %s, real bytes unavailable" % ARTIFACT)
        if str(REPO / "tools") not in sys.path:
            sys.path.insert(0, str(REPO / "tools"))
        from artifact.container import Artifact  # noqa: E402

        with Artifact.open(ARTIFACT) as art:
            poison = bytes(art.payload(art.find("frontend/tokenizer_config.json")))
            template = bytes(art.payload(art.find("frontend/chat_template.jinja")))
        self.assertEqual(len(poison), 1121)
        self.assertEqual(sha(poison)[:16], "e5d078b00e6c1223")

        directory = self.tmp / "as-shipped"
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5", "text_config": {"vocab_size": 8}}),
            encoding="utf-8")
        (directory / "tokenizer_config.json").write_bytes(poison)
        (directory / "chat_template.jinja").write_bytes(template)

        resolution = self.resolve().by_name["frontend/tokenizer_config.json"]
        self.assertEqual(resolution.status, "missing",
                         "the shipped artifact's own tokenizer_config.json was re-admitted: %r"
                         % (resolution.evidence,))

    # ---- the fix must not over-refuse ----------------------------------------------
    def test_a_contract_passing_config_is_still_offered(self) -> None:
        """The hunt is narrowed, not closed."""

        template = b"{{ prompt }}  # engine-accepted template\n"
        self.provider("contract-passing",
                      dict(CONFORMING, chat_template=template.decode("utf-8")), template)

        resolution = self.resolve().by_name["frontend/tokenizer_config.json"]
        self.assertNotEqual(resolution.status, "missing",
                            "a contract-passing provider was refused: %r" % (resolution.evidence,))

    # ---- the authority is the engine, not a transcription --------------------------
    def test_pad_token_constant_equals_the_engines_own_literal(self) -> None:
        source = FRONTEND_CPP.read_text(encoding="utf-8")
        block = source[source.index("void validate_tokenizer_config"):]
        block = block[:block.index("fi::CompiledChatTemplate compile_chat_template")]
        literals = re.findall(r'"(<\|[a-z_]+\|>)"', block)
        self.assertIn(policy.ENGINE_TOKENIZER_CONFIG_PAD_TOKEN, literals,
                      "frontend.cpp validate_tokenizer_config no longer compares against %r "
                      "(literals seen: %r)"
                      % (policy.ENGINE_TOKENIZER_CONFIG_PAD_TOKEN, literals))

    def test_every_engine_condition_is_one_the_predicate_refuses(self) -> None:
        cases = {
            "absent add_bos_token defaults to true": POISON_CONFIG,
            "add_prefix_space true": dict(POISON_CONFIG, add_prefix_space=True),
            "pad_token is not <|endoftext|>": dict(POISON_CONFIG, pad_token="<|im_end|>"),
            "no chat_template key": POISON_CONFIG,
        }
        for label, document in cases.items():
            with self.subTest(label):
                self.assertTrue(
                    policy.engine_tokenizer_config_reasons(json.dumps(document).encode("utf-8")),
                    "the predicate accepted a config the engine refuses: %s" % label)

    def test_a_conforming_config_has_no_reasons(self) -> None:
        template = b"{{ prompt }}  # accepted\n"
        self.assertEqual(
            policy.engine_tokenizer_config_reasons(
                json.dumps(dict(CONFORMING, chat_template=template.decode("utf-8")))
                .encode("utf-8"), template), ())

    def test_chat_template_digests_are_parsed_from_the_engine(self) -> None:
        found = engine_template_digests()
        self.assertEqual(sorted(found), ["ReasoningEffort", "ThinkingToggle"],
                         "chat_template.cpp no longer pins exactly the two known templates: %r"
                         % sorted(found))
        self.assertEqual(found["ReasoningEffort"],
                         "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041")
        self.assertEqual(found["ThinkingToggle"],
                         "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259")

    def test_the_one_contract_passing_root_provides_a_conforming_config(self) -> None:
        """The box has exactly one six-resource root the engine accepts; use it as the fix's target."""

        root = Path("/mnt/c/Users/User/Documents/ziqinzhang/models/"
                    "Qwen3.8-Flash-Next-ABLITERATED-NVFP4")
        config = root / "tokenizer_config.json"
        template = root / "chat_template.jinja"
        if not (config.is_file() and template.is_file()):
            self.skipTest("SKIP: the contract-passing root is absent from this box")
        self.assertEqual(
            policy.engine_tokenizer_config_reasons(config.read_bytes(), template.read_bytes()), (),
            "%s no longer satisfies the engine's tokenizer_config contract" % config)
        from tools.convert.qwen3_6.common.official_resources import REGISTERED_RESOURCE_SHA256
        self.assertIn(sha(config.read_bytes()),
                      REGISTERED_RESOURCE_SHA256.get("frontend/tokenizer_config.json", {}),
                      "the contract-passing config is not a registered revision, so the resolver "
                      "cannot prefer it by digest")


if __name__ == "__main__":
    unittest.main(verbosity=2)

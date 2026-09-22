# Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
"""Reference MTP (multi-token prediction) block for Qwen3.8-Flash-Next.

Faithful to vLLM's ``Qwen4ExpMultiTokenPredictor`` (vllm/models/qwen4_exp/nvidia/mtp.py),
which is the only public implementation: transformers discards ``mtp.*`` weights
(``_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]``) and ships no MTP module.

The stem, verbatim from vLLM::

    inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)      # GemmaRMSNorm(H)
    inputs_embeds = self.fc_embedding(inputs_embeds)
    hidden_states = hidden_states.view(T, hc_count, H)
    hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(T, hc_count, H)
    hidden_states = self.fc_hidden(hidden_states)                  # per stream, shared weight
    prev_block_output = inputs_embeds                              # added to EVERY stream, unit weight

then ONE decoder layer on the four DISTINCT streams, then ONE mixer at the end::

    multi_hidden, sample_hidden_states, _ = self.hyper_connection_mixer.combine_and_mix(...)
    return sample_hidden_states, multi_hidden

There is no stem mixer. The earlier reconstruction in this directory collapsed the
four streams through ``hyper_connection_mixer`` in the stem and then repeated the
result four times, so every stream entering the layer was identical, and it
omitted ``pre_fc_norm_hidden`` entirely although the checkpoint ships it
(``mtp.pre_fc_norm_hidden.weight``, shape 10240). It also built a second
``final_mixer`` the checkpoint has no weights for. All three are corrected here.

Two things that are easy to get wrong and are pinned by tests:

* ``pre_fc_norm_hidden`` is a SINGLE GemmaRMSNorm over all hc*H = 10240 features
  (``GemmaRMSNorm(self.hidden_size * self.hc_count)`` in vLLM). It is NOT the
  per-branch grouped norm the mixers use for their own ``hc_norm``.
* Draft step k+1 consumes the previous step's ``multi_hidden`` -- the post-combine
  multi-stream residual BEFORE the final mix -- not the target's hidden and not
  the single-stream ``final_hidden``.

Token/position contract (vLLM ``SpecDecodeBaseProposer.set_inputs_first_pass``):
draft slot t pairs the target's multi-stream hidden at position t with the
embedding of token t+1, at position t. Prefill runs this over every prompt slot
to seed the draft KV; position 0 is not skipped. Subsequent draft steps advance
all three MRoPE coordinates by one.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# The transformers classes are only needed for the full reference model. The pure
# tensor functions below import nothing from transformers so the stem semantics can
# be unit-tested on a machine whose transformers has no qwen4_exp.
try:  # pragma: no cover - exercised only where transformers>=5.16 is installed
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextDecoderLayer,
        Qwen4ExpTextGatedResidual,
        Qwen4ExpTextRMSNorm,
        Qwen4ExpTextRotaryEmbedding,
    )

    HAVE_QWEN4_EXP = True
except ImportError:
    HAVE_QWEN4_EXP = False


# Canonical stage names. These are what the C++ side should emit after its stem is
# corrected; ``mtp_hidden_mix`` and ``mtp_trunk_input`` no longer exist because
# there is no stem mixer and the trunk IS the four-stream hyper_init.
STEM_STAGES = (
    "mtp_embedding_norm",
    "mtp_embedding_proj",
    "mtp_hidden_norm",
    "mtp_hidden_proj",
    "mtp_hyper_init",
)
LAYER_STAGES = (
    "mtp_attn_block_input",
    "mtp_attn_block_output",
    "mtp_hyper_after_attn",
    "mtp_mlp_block_input",
    "mtp_mlp_block_output",
    "mtp_hyper_after_mlp",
)
HEAD_STAGES = (
    "mtp_multi_hidden",
    "mtp_final_hidden",
    "mtp_draft_logits",
    "mtp_draft_tokens",
)
ALL_STAGES = STEM_STAGES + LAYER_STAGES + HEAD_STAGES


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with the unit offset: ``normalize(x) * (1 + w)``. Single group over
    the last dimension. Matches transformers ``Qwen4ExpTextRMSNorm`` with
    ``group_size=None`` and vLLM ``GemmaRMSNorm``."""
    x32 = x.float()
    out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.float())).type_as(x)


def mtp_stem(
    input_embedding: torch.Tensor,
    backbone_hyper_hidden: torch.Tensor,
    *,
    embedding_norm_weight: torch.Tensor,
    fc_embedding_weight: torch.Tensor,
    hidden_norm_weight: torch.Tensor,
    fc_hidden_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    eps: float = 1e-6,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """The vLLM stem as pure tensor ops.

    ``input_embedding``: ``[..., H]``. ``backbone_hyper_hidden``: ``[..., hc*H]``.
    Returns the stem stages and ``hyper_init`` of shape ``[..., hc*H]`` holding
    ``hc`` DISTINCT streams, each ``fc_hidden(norm(h))_i + fc_embedding(norm(e))``.
    """
    stages: Dict[str, torch.Tensor] = {}
    emb_norm = gemma_rmsnorm(input_embedding, embedding_norm_weight, eps)
    stages["mtp_embedding_norm"] = emb_norm
    emb_proj = F.linear(emb_norm, fc_embedding_weight)
    stages["mtp_embedding_proj"] = emb_proj

    # One norm across every stream at once: 10240 features, one RMS.
    hid_norm = gemma_rmsnorm(backbone_hyper_hidden, hidden_norm_weight, eps)
    stages["mtp_hidden_norm"] = hid_norm
    # Then the SAME 2560x2560 projection applied to each stream separately.
    per_stream = hid_norm.unflatten(-1, (hc_count, hidden_size))
    hid_proj = F.linear(per_stream, fc_hidden_weight)
    stages["mtp_hidden_proj"] = hid_proj.flatten(-2)
    # The embedding joins every stream with unit weight -- vLLM's NVIDIA path
    # expresses this as prev_block_output with a missing injection; the AMD path
    # as an explicit broadcast add. Same numbers.
    hyper_init = (hid_proj + emb_proj.unsqueeze(-2)).flatten(-2)
    stages["mtp_hyper_init"] = hyper_init
    return stages, hyper_init


def stem_streams_are_distinct(hyper_init: torch.Tensor, hc_count: int, hidden_size: int) -> bool:
    """True when the streams differ -- the property the old stem violated."""
    s = hyper_init.unflatten(-1, (hc_count, hidden_size))
    return bool(any(not torch.allclose(s[..., 0, :], s[..., i, :]) for i in range(1, hc_count)))


if HAVE_QWEN4_EXP:

    class Qwen4ExpMTPReference(nn.Module):
        """vLLM-faithful MTP block built from the transformers decoder-layer
        classes, so attention (including the QSA indexer) is the real thing."""

        def __init__(self, config: "Qwen4ExpTextConfig"):
            super().__init__()
            self.config = config
            self.hc_count = config.hc_count
            self.hidden_size = config.hidden_size
            hc_hidden = self.hc_count * self.hidden_size
            self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.fc_embedding = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            # Single group over hc*H. group_size stays None on purpose.
            self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(hc_hidden, eps=config.rms_norm_eps)
            self.fc_hidden = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.rotary = Qwen4ExpTextRotaryEmbedding(config=config)
            self.layer = Qwen4ExpTextDecoderLayer(config, layer_idx=0)
            # The ONE mixer the checkpoint ships, used ONCE, at the end.
            self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        def forward(
            self,
            input_embedding: torch.Tensor,
            backbone_hyper_hidden: torch.Tensor,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values=None,
        ) -> Dict[str, torch.Tensor]:
            """``input_embedding`` ``[B, T, H]``; ``backbone_hyper_hidden`` ``[B, T, hc*H]``;
            ``position_ids`` ``[3, B, T]`` (MRoPE rows; all equal for text);
            ``attention_mask`` boolean ``[B, 1, T, S]`` or None for a single token with no
            history; ``past_key_values`` a transformers Cache to seed and extend."""
            if input_embedding.ndim == 2:
                input_embedding = input_embedding.unsqueeze(1)
            if backbone_hyper_hidden.ndim == 2:
                backbone_hyper_hidden = backbone_hyper_hidden.unsqueeze(1)
            bsz, seq, _ = input_embedding.shape
            if position_ids is None:
                position_ids = torch.zeros((3, bsz, seq), dtype=torch.long)
            if attention_mask is None:
                # Causal over whatever is visible; with no cache that is just T.
                visible = seq
                attention_mask = torch.ones((bsz, 1, seq, visible), dtype=torch.bool).tril(visible - seq)

            stages, hyper_init = mtp_stem(
                input_embedding,
                backbone_hyper_hidden,
                embedding_norm_weight=self.pre_fc_norm_embedding.weight,
                fc_embedding_weight=self.fc_embedding.weight,
                hidden_norm_weight=self.pre_fc_norm_hidden.weight,
                fc_hidden_weight=self.fc_hidden.weight,
                hc_count=self.hc_count,
                hidden_size=self.hidden_size,
                eps=self.config.rms_norm_eps,
            )

            # Rotary needs cos/sin for the FULL visible span (the indexer scores
            # every cached key); the attention slices the current positions itself.
            if past_key_values is not None and past_key_values.get_seq_length() > 0:
                past = past_key_values.get_seq_length()
                full_pos = torch.cat(
                    [torch.arange(past, dtype=torch.long).view(1, 1, past).expand(3, bsz, past), position_ids],
                    dim=-1,
                )
            else:
                full_pos = position_ids
            pos_emb = self.rotary(hyper_init, position_ids=full_pos)

            def out0(name):
                def hook(_m, _i, out):
                    stages[name] = out[0] if isinstance(out, (tuple, list)) else out
                return hook

            def in0(name):
                def hook(_m, inp):
                    stages[name] = inp[0]
                return hook

            handles = [
                self.layer.attn_hyper_connection.register_forward_hook(out0("mtp_attn_block_input")),
                self.layer.self_attn.register_forward_hook(out0("mtp_attn_block_output")),
                self.layer.mlp_hyper_connection.register_forward_pre_hook(in0("mtp_hyper_after_attn")),
                self.layer.mlp_hyper_connection.register_forward_hook(out0("mtp_mlp_block_input")),
                self.layer.mlp.register_forward_hook(out0("mtp_mlp_block_output")),
            ]
            try:
                hyper_after_mlp = self.layer(
                    hyper_init,
                    position_embeddings=pos_emb,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
            finally:
                for h in handles:
                    h.remove()
            stages["mtp_hyper_after_mlp"] = hyper_after_mlp

            # multi_hidden is the post-combine multi-stream residual BEFORE the final
            # mix. The transformers layer applies its injections eagerly, so its
            # output already is that quantity. This is what the next draft step eats.
            stages["mtp_multi_hidden"] = hyper_after_mlp
            final_hidden = self.hyper_connection_mixer(hyper_after_mlp)
            stages["mtp_final_hidden"] = final_hidden
            logits = self.lm_head(final_hidden)
            stages["mtp_draft_logits"] = logits
            stages["mtp_draft_tokens"] = logits.argmax(dim=-1)
            return stages

    def make_mtp_config(text_cfg_dict: dict) -> "Qwen4ExpTextConfig":
        """The MTP block is one full-attention layer with no PLE."""
        d = dict(text_cfg_dict)
        d["layer_types"] = ["full_attention"]
        d["num_hidden_layers"] = 1
        d["ple_layer_ids"] = []
        return Qwen4ExpTextConfig(**d)

    def load_mtp_weights(mtp: "Qwen4ExpMTPReference", model_dir: str) -> None:
        """Load the 31 ``mtp.*`` tensors plus the shared ``lm_head`` from the source
        checkpoint's safetensors index. MTP experts are BF16 fused 3-D tensors
        in the checkpoint, which is exactly the transformers ``Qwen4ExpTextExperts``
        parameter layout, so they load directly."""
        import safetensors

        with open(os.path.join(model_dir, "model.safetensors.index.json"), "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        handles: Dict[str, object] = {}

        def raw(name: str) -> torch.Tensor:
            fn = weight_map[name]
            if fn not in handles:
                handles[fn] = safetensors.safe_open(os.path.join(model_dir, fn), framework="pt", device="cpu")
            return handles[fn].get_tensor(name).float()

        sd: Dict[str, torch.Tensor] = {}
        for key in weight_map:
            if key.startswith("mtp.layers.0."):
                sd["layer." + key[len("mtp.layers.0."):]] = raw(key)
            elif key.startswith("mtp."):
                sd[key[len("mtp."):]] = raw(key)
        sd["lm_head.weight"] = raw("lm_head.weight")
        missing, unexpected = mtp.load_state_dict(sd, strict=False)
        # Rotary has no weights; anything else missing is a real mapping error.
        missing = [m for m in missing if not m.startswith("rotary.")]
        if missing or unexpected:
            raise RuntimeError(f"MTP weight mapping incomplete: missing={missing} unexpected={unexpected}")


def dump_stages(dump_root: str, position: int, token_id: int, mrope: Tuple[int, int, int],
                stages: Dict[str, torch.Tensor], manifest: dict) -> None:
    """Same on-disk format the main oracle and the C++ StateDumper use."""
    import numpy as np

    pos_dir = os.path.join(dump_root, f"pos{position:04d}")
    os.makedirs(pos_dir, exist_ok=True)
    rec = {"position": position, "token_id": token_id, "mrope_position": list(mrope), "tensors": []}
    for name, tensor in stages.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        with open(os.path.join(pos_dir, f"{name}.bin"), "wb") as f:
            f.write(arr.tobytes())
        rec["tensors"].append({"name": name, "dtype": "FP32", "shape": list(arr.shape),
                               "file": f"pos{position:04d}/{name}.bin", "bytes": arr.nbytes})
    manifest["positions"].append(rec)

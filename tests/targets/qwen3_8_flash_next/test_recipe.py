# Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
from __future__ import annotations

from tools.convert.qwen3_8_flash_next import inventory, recipe


def test_recipe_closes_every_artifact_tensor_and_mixed_source_entry():
    recipe.validate_recipe()

    assert len(recipe.DIRECT_RECIPES) == 1_240
    assert len(recipe.FP8_RECIPES) == 96
    assert len(recipe.EXPERT_BANK_RECIPES) == 96
    assert len(recipe.PLE_RECIPES) == 128
    assert len(recipe.RECIPES_BY_OBJECT) == len(inventory.TENSOR_SPECS)

    assert len(recipe.MIXED_DIRECT_SOURCES) == 1_278
    assert len(recipe.MIXED_FP8_SOURCES) == 312
    assert len(recipe.MIXED_EXPERT_SOURCES) == 294_912
    assert len(recipe.MIXED_SOURCES) == 296_502


def test_fused_recipes_preserve_semantic_source_order():
    gdn = recipe.RECIPES_BY_OBJECT["text/layers/0/gdn/query_key_value_z"]
    assert isinstance(gdn, recipe.Fp8Recipe)
    assert tuple(part.name.rsplit(".", 1)[0] for part in gdn.matrices) == (
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
    )

    attention = recipe.RECIPES_BY_OBJECT[
        "text/layers/3/attention/query_gate_key_value"
    ]
    assert isinstance(attention, recipe.Fp8Recipe)
    assert tuple(part.name.rsplit(".", 1)[0] for part in attention.matrices) == (
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj",
        "model.language_model.layers.3.self_attn.v_proj",
    )

    gate_up = recipe.RECIPES_BY_OBJECT["text/layers/47/mlp/experts/gate_up"]
    assert isinstance(gate_up, recipe.ExpertBankRecipe)
    assert gate_up.projections == ("gate_proj", "up_proj")

    multipliers = recipe.RECIPES_BY_OBJECT[
        "text/layers/1/ple/embedding/layer_multipliers"
    ]
    assert isinstance(multipliers, recipe.DirectRecipe)
    assert multipliers.sources[0].dtype == "I64"
    assert multipliers.transform == "copy"


def test_real_pinned_bundle_matches_recipe_when_available():
    mixed = recipe.DEFAULT_MIXED_DIR
    if not mixed.is_dir():
        return
    result = recipe.validate_mixed_source(mixed)
    assert result.index_tensors == 296_630
    assert result.consumed_tensors == 296_502
    assert result.replaced_ple_tensors == 128

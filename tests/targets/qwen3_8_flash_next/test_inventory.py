# Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
from __future__ import annotations

from tools.convert.qwen3_8_flash_next import inventory


def test_inventory_is_closed_and_uses_bank_objects():
    inventory.validate_inventory()

    assert len(inventory.RESOURCE_SPECS) == 6
    assert len(inventory.TENSOR_SPECS) == 1_560
    assert len(inventory.OBJECT_SPECS) == 1_566
    assert inventory.FORMAT_COUNTS == {
        "BF16": 1_237,
        "I64": 3,
        "NVFP4": 96,
        "FP8_E4M3FN_ROW_F32S": 96,
        "U4Z8G16_F16S": 128,
    }

    names = {spec.name for spec in inventory.TENSOR_SPECS}
    assert "text/layers/0/mlp/experts/gate_up" in names
    assert "text/layers/47/mlp/experts/down" in names
    assert not any("experts/0/" in name for name in names)


def test_projection_and_ple_shapes_match_runtime_objects():
    by_name = {spec.name: spec for spec in inventory.TENSOR_SPECS}

    assert by_name["text/layers/0/gdn/query_key_value_z"].shape == (16_384, 2_560)
    assert by_name["text/layers/3/attention/query_gate_key_value"].shape == (
        13_312,
        2_560,
    )
    assert by_name["text/layers/0/mlp/experts/gate_up"].shape == (
        512,
        1_280,
        2_560,
    )
    assert by_name["text/layers/0/mlp/experts/down"].shape == (512, 2_560, 640)
    assert by_name["text/layers/1/ple/embedding/shards/127"].shape == (
        2_500_012,
        160,
    )

    assert by_name["text/layers/0/gdn/query_key_value_z"].layout == (
        "row-scale-f32-v1"
    )
    assert by_name["text/layers/0/mlp/experts/gate_up"].layout == (
        "expert-blockscale-k16-m128x4-v1"
    )
    assert by_name["text/layers/1/ple/embedding/shards/127"].layout == (
        "packed-u4-g16-v1"
    )


def test_layer_type_partition_and_mtp_projection_fusion_are_exact():
    assert inventory.FULL_ATTENTION_LAYERS == tuple(range(3, 48, 4))
    assert len(inventory.GDN_LAYERS) == 36

    by_name = {spec.name: spec for spec in inventory.TENSOR_SPECS}
    assert by_name["mtp/layer/attention/query_gate_key_value"].shape == (
        13_312,
        2_560,
    )
    assert "mtp/layer/attention/query" not in by_name
    assert "mtp/layer/attention/key" not in by_name
    assert "mtp/layer/attention/value" not in by_name

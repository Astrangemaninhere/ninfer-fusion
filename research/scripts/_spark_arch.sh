#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
S=$J/models/Spark-X2.5-4B
F=$J/ninfer-fusion-repo
echo "=== config.json ==="
cat "$S/config.json"
echo
echo "=== generation_config.json ==="
cat "$S/generation_config.json"
echo
echo "=== architecture facts from modeling_spark.py ==="
grep -nE 'class |layer_types|sliding_window|num_attention_heads|num_key_value_heads|head_dim|rope_theta|attention_bias|qkv|mlp|norm|act_fn|tie_word|attention_dropout' "$S/modeling_spark.py" | head -30 | cut -c1-130
echo
echo "=== adapt.py CLI ==="
grep -nE 'add_argument|usage|argparse|def main' "$F/tools/archkit/adapt.py" | head -24 | cut -c1-140

# -*- coding: utf-8 -*-
"""check_params.py - 参数完备门: config.json 每个键都有归属, 无静默忽略.

背景 (Muse 教训的推广): 适配管线只读显式键清单, 清单外的模型参数会被静默忽略
  -> 模型悄悄跑错 ("未识别"其实都是"未适配")。本门把 config 树每个叶子键逐一判定:
    covered   引擎/规格已处理, 语义已映射
    benign    推理无关的标准键 (架构名/随机种子/词表 id 等)
    hook      引擎有机理, 需叶子/config 接线 (参数化可自动生成)
    new_op    引擎无此口味, 需内核工作包
用法: python3 check_params.py <config.json> [--report 未适配.md]
返回: 0 = 无未适配; 2 = 存在未适配 (报告列出, --strict 下 adapt_all 门控)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 引擎/规格已显式处理的键 (extract_spec + catalog_gaps + arch_spec 并集)
COVERED_KEYS = {
    # 几何/规模
    'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
    'head_dim', 'intermediate_size', 'vocab_size', 'max_position_embeddings',
    'rms_norm_eps', 'layer_norm_eps', 'rope_theta', 'partial_rotary_factor',
    # 口味旋钮
    'sliding_window', 'qk_scale_factor', 'output_multiplier', 'final_logit_softcapping',
    'layer_rope_theta', 'post_norm_eps', 'attention_bias', 'tie_word_embeddings',
    'global_head_dim', 'num_global_key_value_heads', 'attention_qk_norm', 'qk_norm',
    'layer_types', 'rope_parameters', 'rope_scaling', 'model_type', 'rope_type',
    'hidden_activation', 'hidden_act', 'initial_context_length',
    'rope_scaling_factor', 'rope_scaling.factor', 'rope_scaling.original_max_position_embeddings',
    'rope_scaling.beta_fast', 'rope_scaling.beta_slow', 'rope_scaling.rope_type',
    'rope_scaling.truncate',
    # 文本/前端
    'tokenizer_class', 'bos_token_id', 'eos_token_id', 'pad_token_id', 'unk_token_id',
    'sep_token_id', 'additional_special_tokens', 'added_tokens_decoder',
    'chat_template', 'architectures',
}
# MoE 架构键: 引擎当前无 MoE 主路径 => 一律 new_op (绝不静默 benign)
MOE_KEYS = {
    'num_experts', 'num_local_experts', 'experts_per_token', 'num_experts_per_tok',
    'expert_top_k', 'moe_intermediate_size', 'first_k_dense_replace', 'n_shared_experts',
    'output_router_logits', 'router_aux_loss_coef', 'norm_topk_prob', 'aux_loss',
    'router_jitter_noise', 'expert_choice',
}
# 推理无关 (影响训练/序列化/工具, 不影响前向语义)
BENIGN_KEYS = {
    'torch_dtype', 'transformers_version', '_name_or_path', 'auto_map', 'revision',
    'initializer_range', 'use_cache', 'pretraining_tp', 'attention_dropout',
    'hidden_dropout', 'dropout', 'classifier_dropout', 'max_window_layers',
    'residual_connection', 'add_bias_linear', 'position_embedding_type',
    'is_encoder_decoder', 'model_max_length', 'return_dict', 'problem_type',
    'chunk_size_feed_forward', 'output_hidden_states', 'output_attentions',
    'torchscript', 'use_bfloat16', 'bf16', 'fp16', 'quantization_config',
    'quant_method', 'gpu_memory_utilization', 'dtype', 'load_in_*',
}
# 引擎有机制、需接线/可参数化的键 (manifest 判 hook 的候选; 具体 tier 以 catalog 为准)
HOOK_KEYS = {
    'rope_scaling': 'yarn/linear rope scaling (ctx_.yarn_enabled 已存在, 需 config 接线)',
    'head_dim': 'query_head_dim 与 kv 分维 (几何门已查 q/kv; 异构 global_head_dim 走 new_op)',
    'num_experts': 'MoE (qwen3_6 引擎无 MoE 主路径; 若 layer_types 无 moe 则 benign 误标需人工)',
}


def walk_keys(obj, prefix: str, out: list[tuple[str, object]]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk_keys(v, prefix + '.' + k if prefix else k, out)
    elif isinstance(obj, list):
        # 列表: 元素是原始值 => 键到此为止; dict 元素 (如 layer_types) 展开首层即可
        for i, v in enumerate(obj[:1]):
            walk_keys(v, prefix + '[0]', out)
    else:
        out.append((prefix, obj))


def classify(key: str) -> tuple[str, str]:
    base = key.split('.')[-1].split('[0]')[0]
    if base in COVERED_KEYS:
        return 'covered', ''
    if base in HOOK_KEYS:
        return 'hook', HOOK_KEYS[base]
    if base in MOE_KEYS:
        return 'new_op', 'MoE 架构键: 引擎无 MoE 主路径 (需要专家路由内核工作包)'
    if base in BENIGN_KEYS or base.endswith('*'):
        return 'benign', ''
    return 'new_op', '未适配: 键 %s 无归属 (需语义判定: 参数化接线 / 新内核口味)' % key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--report', default='')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    # 顶层 + text_config 双视角 (叶子去重)
    leaves: dict[tuple[str, str], object] = {}
    for root, label in ((cfg, 'cfg'), (cfg.get('text_config') or {}, 'tc')):
        items: list[tuple[str, object]] = []
        walk_keys(root, '', items)
        for k, v in items:
            leaves.setdefault((label, k), v)

    # 域路由: 子树整体归属, 避免碎片
    VISION_PREFIXES = ('vision_config', 'projector_', 'image_token_id', 'video_token_id',
                       'image_size', 'patch_size', 'num_image_tokens', 'mm_',
                       'out_hidden_size', 'projector_hidden', 'vision_soft_tokens')
    AUDIO_PREFIXES = ('audio_config', 'audio_token_id', 'boa_token_id', 'boi_token_id',
                      'eoa_token_id', 'eoa_token_index', 'eoi_token_id')
    BENIGN_TOP = ('dtype', 'transformers_version', 'quantization_config')

    def route(label: str, key: str, value) -> tuple[str, str] | None:
        if key.startswith('added_tokens_decoder'):
            return 'covered', 'tokenizer 域 (frontend preprocessor 已按表处理)'
        if key.startswith('quantization_config'):
            return 'covered', 'converter 域 (权重打包格式输入, convert.py 按此生成)'
        if key.startswith('rope_parameters'):
            return 'covered', 'rope 参数域 (theta/type 已进 spec.rope; 非 default 型须接线)'
        if key.startswith('rope_scaling'):
            return 'covered', 'rope 缩放域 (yarn/linear; ctx_.yarn_enabled 机制, 逐参数进 spec)'
        if key == 'hidden_activation' or key == 'hidden_act':
            return 'covered', 'mlp 激活 (family 固定 silu; 若 != silu 改判 new_op)'
        if key.startswith(AUDIO_PREFIXES):
            return 'new_op', '音频口味域 (text 引擎无此主路径; 与 vision 同批工作包)'
        if key.startswith(VISION_PREFIXES):
            return 'new_op', '视觉口味域 (text 引擎无此主路径; 见 manifest vision new_op)'
        # MoE 键值为 None/False/0 = 未启用 => benign (不是架构需求)
        base_key = key.split('.')[-1]
        if base_key in MOE_KEYS and value in (None, False, 0, '0'):
            return 'benign', 'MoE 开关未启用'
        if label == 'cfg':
            if key.startswith('text_config'):
                return None  # 叶子按 text 域判定
            if key.split('.')[0] in BENIGN_TOP:
                return 'benign', '顶层工具键'
            return None
        return None

    counts = {'covered': 0, 'benign': 0, 'hook': 0, 'new_op': 0}
    rows = []
    reported = set()  # cfg.text_config.x 与 tc.x 双视角去重 (报 tc 视角)
    for (label, k), v in sorted(leaves.items()):
        r = route(label, k, v)
        if r:
            tier, note = r
        else:
            tier, note = classify(k)
        counts[tier] += 1
        rows.append((tier, label, k, v, note))
    print('== 参数完备门: %d 键 (covered %d, benign %d, hook %d, 未适配 %d)' %
          (len(rows), counts['covered'], counts['benign'], counts['hook'],
           counts['new_op']))
    unfitted = []
    for t, l, k, v, n in rows:
        if t not in ('hook', 'new_op'):
            continue
        if l == 'cfg' and k.startswith('text_config.') and ('tc', k.split('.', 1)[1]) in leaves:
            continue  # 双视角重复, tc 已报
        unfitted.append((t, l, k, v, n))
    seen: set[str] = set()
    for tier, label, k, v, note in unfitted:
        # 同域聚合: 只打首个叶子 + 计数
        base = '.'.join(k.split('.')[:3])
        if base in seen:
            continue
        seen.add(base)
        n = sum(1 for _, _, kk, _, _ in unfitted if kk.startswith(base) or base in kk)
        print('  [%s] %s %s = %s %s%s' % (tier, label, k, repr(v)[:60], note,
                                          '' if n == 1 else ' (+%d 同类)' % (n - 1)))
    if args.report:
        Path(args.report).write_text(
            '\n'.join('## [%s] %s = %r\n%s' % (t, l, k, v, n)
                      for t, l, k, v, n in unfitted), encoding='utf-8')
    return 2 if counts['new_op'] else 0


if __name__ == '__main__':
    raise SystemExit(main())

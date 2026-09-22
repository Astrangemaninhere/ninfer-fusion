# -*- coding: utf-8 -*-
"""gguf_spec.py - GGUF -> archkit spec (qwen35 等家族), 打通 GGUF 模型的自动适配管线.
GGUF 没有 config.json; 本工具从 GGUF kv 元数据 (qwen35.*, llama.* 通用键) 构造
与 adapt.extract_spec 同构的 spec json, 可直接喂 adapt_all (几何门/参数门全走).
Usage: python3 gguf_spec.py <model.gguf> [--out spec.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_tensors import scan  # noqa: E402  (dims 为 u64 的成熟解析)


def read_kv(path: str) -> dict:
    """全 kv 元数据 (字符串/数值; 数组只取前几个)."""
    import struct
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF'
    version, n_tensors, n_kv = struct.unpack('<IQQ', f.read(20))

    def read_str():
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n).decode('utf-8', 'replace')

    def read_val(t):
        if t == 0:
            return f.read(1)[0]
        if t == 1:
            return int.from_bytes(f.read(1), 'little', signed=True)
        if t in (2, 4, 10):
            size, fmt = (2, '<H') if t == 2 else ((4, '<I') if t == 4 else (8, '<Q'))
            return struct.unpack(fmt, f.read(size))[0]
        if t in (3, 5, 11):
            size, fmt = (2, '<h') if t == 3 else ((4, '<i') if t == 5 else (8, '<q'))
            return struct.unpack(fmt, f.read(size))[0]
        if t == 6:
            return struct.unpack('<f', f.read(4))[0]
        if t == 7:
            return f.read(1)[0] != 0
        if t == 8:
            return read_str()
        if t == 9:
            atype, count = struct.unpack('<IQ', f.read(12))
            return [read_val(atype) for _ in range(count)]
        if t == 12:
            return struct.unpack('<d', f.read(8))[0]
        raise SystemExit('unknown kv type %d' % t)

    meta = {}
    for _ in range(n_kv):
        key = read_str()
        t = struct.unpack('<I', f.read(4))[0]
        meta[key] = read_val(t)
    f.close()
    return meta


# GGUF arch 前缀 -> archkit family
ARCH_FAMILY = {
    'qwen35': 'qwen38',      # llama.cpp 命名 qwen35 覆盖 qwen3.5/3.6/3.8 文本族
    'qwen3': 'qwen38',
    'qwen2': 'qwen2',
    'llama': 'qwen2',
}


def _split_nextn(meta: dict, arch: str) -> tuple[int, int]:
    """(main_layers, nextn_layers) -- delegate to the one implementation that owns
    this rule: tools/convert/gguf_names.layer_split.

    `block_count` counts the nextn/draft blocks too.  Ornith-1.5-9B-Q4_K_M declares
    `qwen35.block_count = 33` and `qwen35.nextn_predict_layers = 1`, and block 32
    carries `blk.32.nextn.eh_proj|enorm|hnorm|shared_head_norm` next to a full set of
    attn/ffn weights -- so the decoder stack is 32 layers (full attention at
    3,7,...,31 = 8 of them) and one block is the draft/MTP block.  This module used to
    use block_count as `layers` (33) and let the draft block fall through to
    "full attention": a TextConfig one layer too deep, with a 9th full-attention layer
    that the decoder does not have, and `mtp` never recorded.  Measured before the fix:
    the real file and a synthetic file with the same shape both produced
    `33 layers / full=9 / gdn=24`, rc=0.

    tools/convert/gguf_names.py (its "The MTP block" note) states the rule and used to
    name this file as the place that got it wrong -- this file now imports the rule, and
    so does tools/gui/model_import.py (`_gguf_main_layers`), so no line number is quoted
    here: the reference is to the rule's owner, not to a line that will move.
    The rule is imported rather than copied: two spellings of "how
    many layers does this file have" is exactly the writer/reader split this tree has
    paid for before.  A checkout where that module cannot be imported gets a named
    refusal, not a silent fallback to the wrong arithmetic.
    """
    try:
        from tools.convert.gguf_names import layer_split
    except ImportError as exc:              # run-by-path / partial checkout
        root = str(Path(__file__).resolve().parents[2])
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from tools.convert.gguf_names import layer_split
        except ImportError:
            raise SystemExit(
                '层数记账无法进行: 需要 tools/convert/gguf_names.layer_split 这条规则的'
                '唯一实现 (block_count 把 nextn/draft 块也算进去了, 不能直接当层数), '
                '但导入失败: %s。缺了它就只能按 block_count 记账, 那正是本文件修掉的'
                '错 (把草稿块当成第 33 层 full attention)。' % exc)
    # The kv keys are the *GGUF* architecture's (`qwen35.block_count`), not the family
    # this spec is filed under (`qwen38`), so `arch` has to be the declared one.
    blocks = int(meta.get('%s.block_count' % arch) or 0)
    nextn = int(meta.get('%s.nextn_predict_layers' % arch) or 0)
    if nextn > blocks:
        raise SystemExit(
            '层数记账不一致: %s.nextn_predict_layers=%d 但 %s.block_count=%d —— 草稿块'
            '比总块数还多。这两个数来自文件自己, 对不上时不猜。'
            % (arch, nextn, arch, blocks))
    try:
        return layer_split(meta, arch)
    except ValueError as exc:
        raise SystemExit('层数记账失败: %s' % exc)


def build_spec(path: str) -> dict:
    meta = read_kv(path)
    arch = str(meta.get('general.architecture', ''))
    if arch not in ARCH_FAMILY:
        raise SystemExit('未适配的 GGUF 架构: %s (已支持: %s)' % (arch, list(ARCH_FAMILY)))
    p = lambda k: meta.get('%s.%s' % (arch, k))
    tensors = scan(path)
    # 层记账: 元数据说的层数, 与张量表量出来的层数, 必须对得上; 每个层索引还必须
    # 能被认出是 full attention (有 .attn_output.weight) 还是 linear/gdn (有 .ssm_*)。
    #
    # 这里原来写的是 `n_layers = int(p('block_count') or 0)` 加上一个"既不在
    # full_layers 也不在 gdn_layers 就算 full_attention"的兜底, 于是有两个静默
    # 记账错误: (a) block_count 缺失/为 0 时, 产出 `layers: 0` + `layer_types: []`
    # 而 rc=0 —— 一份声称 0 层的 spec; (b) 任何没认出来的层被记成 full attention,
    # 也就是记成"引擎已覆盖", 于是混合模型里真正的 GDN 层缺口 (new_op) 会凭空消失。
    # 两个方向都表现为"账错了但不响", 正是这个项目最怕的那种结果, 所以现在两个都
    # 拒绝, 且把两个数都打出来。
    #
    # 第三个 (同一次修复漏掉的): block_count 里**含 nextn/draft 块**。33 块的
    # Ornith 主栈只有 32 层, 第 33 个块是草稿/MTP 块 (张量表里带着 blk.32.nextn.*)。
    # 把它当普通 full attention 层, 会得到"33 层 / 9 层 full"的记账, 而
    # check_geometry 只按 (query_heads, kv_heads, head_dim) 判定 —— 实测同一个
    # 几何下 layers=32/33/64/64000 的门都是 rc=0 —— 所以这个错位没有任何门看得见。
    n_meta = int(p('block_count') or 0)
    main_layers_meta, nextn_layers = _split_nextn(meta, arch)
    full_layers = set()
    gdn_layers = set()
    seen_layers = set()
    nextn_blocks = set()
    for name, _t, _d in tensors:
        parts = name.split('.')
        if len(parts) >= 3 and parts[0] == 'blk' and parts[1].isdigit():
            seen_layers.add(int(parts[1]))
        if '.attn_output.weight' in name:
            full_layers.add(int(parts[1]))
        if '.ssm_' in name:
            gdn_layers.add(int(parts[1]))
        if len(parts) >= 3 and parts[0] == 'blk' and parts[1].isdigit() \
                and parts[2] == 'nextn':
            nextn_blocks.add(int(parts[1]))
    n_tensor = (max(seen_layers) + 1) if seen_layers else 0
    # ---- nextn/draft block accounting ------------------------------------------
    # Two independent pieces of evidence have to agree, and each of them is checked
    # against the other rather than trusted:
    #   * the metadata: block_count - nextn_predict_layers  -> main stack depth
    #   * the tensor table: the blocks that actually carry `.nextn.` tensors
    #     (Ornith: exactly one, blk.32 = the LAST block)
    # A file where they disagree is not something to average out: either the declared
    # count is wrong or the draft tensors are somewhere unexpected, and both cases
    # change which block is a decoder layer.
    n_obs_nextn = len(nextn_blocks)
    top_nextn = set(range(n_tensor - n_obs_nextn, n_tensor)) if n_obs_nextn else set()
    if n_obs_nextn != nextn_layers:
        raise SystemExit(
            '层数记账不一致: %s.nextn_predict_layers=%d (元数据说 %d 个草稿块), 但张量'
            '表里有 %d 个块带着 .nextn.* 张量 (blk.%s)。草稿块既是被排除的那部分, '
            '也是 block_count 虚高的那部分 —— 两个数对不上时不能猜: 少算会把草稿块'
            '当成一层解码层 (几何多一层), 多算会把一层解码层当成草稿块 (几何少一层)。'
            % (arch, nextn_layers, nextn_layers, n_obs_nextn,
               ', blk.'.join(str(i) for i in sorted(nextn_blocks)) or '(无)'))
    if nextn_blocks and nextn_blocks != top_nextn:
        raise SystemExit(
            '层数记账不一致: .nextn.* 张量在 blk.%s, 而草稿块只能是最后 %d 个块 '
            '(blk.%s)。把草稿块从层数里扣掉, 前提就是它在主栈之后 —— 位置不对时'
            '这条规则不成立, 拒绝猜。'
            % (', blk.'.join(str(i) for i in sorted(nextn_blocks)), n_obs_nextn,
               ', blk.'.join(str(i) for i in sorted(top_nextn)) or '(无)'))
    n_main_tensor = n_tensor - n_obs_nextn
    if n_meta and n_tensor and n_meta != n_tensor:
        raise SystemExit(
            '层数记账不一致: %s.block_count=%d, 但张量表里有 %d 个块 (blk.0..blk.%d)。'
            '两个数必须对得上 —— 元数据说几块就按几块记账会让几何/层型整体错位, '
            '而错位是不响的。请先确认真实层数 (张量表是文件自己的证据)。'
            % (arch, n_meta, n_tensor, n_tensor - 1))
    n_layers = n_main_tensor or main_layers_meta
    if not n_layers:
        raise SystemExit(
            '层数记账失败: %s.block_count 缺失/为 0, 张量表里也没有 blk.N.* 名字, '
            '所以层数未知。未知不是 0 —— 一份声称 0 层的 spec 会让下游按 0 层记账。'
            % arch)
    # 草稿块不参与"层型认不认得出"的判定: 它不是解码层。它仍必须能被张量表看见
    # (否则上面的 nextn 计数就抓不到它), 但把它当 full attention 记进 layer_types
    # 正是这次要修掉的错。
    decoder_layers = seen_layers - nextn_blocks
    unclassified = sorted(decoder_layers - full_layers - gdn_layers)
    if unclassified:
        raise SystemExit(
            '有 %d 层的层型认不出来: blk.%s —— 既没有 .attn_output.weight (full '
            'attention), 也没有 .ssm_* (linear/gdn)。不能默认记成 full attention: '
            '那等于记成"引擎已覆盖", 会把真正的 new_op 缺口藏起来。'
            % (len(unclassified), ', blk.'.join(str(i) for i in unclassified)))
    spec = {
        'model_id': Path(path).stem.lower().replace('_', '-'),
        'family': ARCH_FAMILY[arch],
        'hf': {'source': 'gguf', 'file': path},
        'geometry': {
            'hidden': int(p('embedding_length') or 0),
            'layers': n_layers,
            'query_heads': int(p('attention.head_count') or 0),
            'kv_heads': int(p('attention.head_count_kv') or 0),
            'head_dim': int(p('attention.key_length') or 0),
            'vocab': int(p('vocab_size') or meta.get('tokenizer.ggml.tokens') and
                         len(meta['tokenizer.ggml.tokens']) or 0),
            'max_ctx': int(p('context_length') or 0),
            'intermediate': int(p('feed_forward_length') or 0),
            'rms_eps': float(p('attention.layer_norm_rms_epsilon') or 1e-6),
        },
        # 每个索引走到这里都已经认出来了 (上面 unclassified 为空), 所以不再有
        # "认不出就记成 full_attention" 那条兜底路径。
        'layer_types': ['linear_attention' if i in gdn_layers else 'full_attention'
                        for i in range(n_layers)],
        'knobs': {
            'rope_theta': float(p('rope.freq_base') or 0),
            'tie_word_embeddings': not any(n == 'output.weight' for n, _, _ in tensors),
        },
        'rope': {'rope_theta': float(p('rope.freq_base') or 0), 'rope_type': 'default'},
        'multimodal': False,
        'notes': ['spec 由 GGUF 元数据生成 (gguf_spec.py); 权重转换需 qwen35 GGUF recipe'],
    }
    if n_obs_nextn:
        # The draft block is EXCLUDED from geometry.layers, so its existence has to be
        # recorded somewhere a machine reads -- burying "one layer was dropped" in a
        # note string is how a fact stops being checked. `mtp` is the documented
        # archkit field for it (arch_spec.OPTIONAL) and gen_target.py reads it.
        spec['mtp'] = {'mtp_num_hidden_layers': n_obs_nextn}
        spec['notes'].append(
            'nextn/draft 块: blk.%s 被排除在主栈之外 (block_count=%d 含草稿块, 主栈 '
            '%d 层); 该块的张量属于 MTP 头, 不是第 %d 层解码层。'
            % (', blk.'.join(str(i) for i in sorted(nextn_blocks)), n_meta or n_tensor,
               n_layers, n_layers + 1))
    # 量化格式统计 -> converter 域提示
    from collections import Counter
    qhist = Counter(t for _, t, _ in tensors)
    spec['quant_formats'] = dict(qhist)
    return spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('gguf')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    spec = build_spec(args.gguf)
    out = args.out or str(Path(__file__).resolve().parent / 'specs' /
                          ('%s_spec.json' % spec['model_id']))
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
    print('spec ->', out)
    g = spec['geometry']
    print('geometry: %d q, %d kv, hd %d, %d layers, vocab %d, ctx %d'
          % (g['query_heads'], g['kv_heads'], g['head_dim'], g['layers'], g['vocab'],
             g['max_ctx']))
    print('layer_types: full=%d gdn=%d' %
          (spec['layer_types'].count('full_attention'),
           spec['layer_types'].count('linear_attention')))
    print('quant:', spec['quant_formats'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

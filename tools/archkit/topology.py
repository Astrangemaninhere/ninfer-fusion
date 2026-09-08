# -*- coding: utf-8 -*-
"""topology.py — 拓扑自动判别: dense | moe | moe_ngram (以及 dense+ngram 备注)。

输入两种形态:
  - HF config.json 形态: 顶层含 text_config/architectures (或本身就是 text_config)
  - archkit spec 形态: 顶层含 geometry/layer_types/hf + moe/ple/gdn/mtp 段
输出统一 manifest['topology']: kind + evidence + 关键事实 + 处理指引 + draft 建议。

"不同情况不同处理"落点:
  dense    -> 每层独立稠密 proj, 显存全量驻留, draft 按能力注册
  moe      -> MLP 层 = router + N 专家(页粒度换入/换出), 共享专家可选,
              router 需要校准数据; 权重布局 = 专家页
  moe_ngram-> 在 moe 之上叠加 ngram 查表草稿 (PLE sidecar, SSD 外挂),
              draft 注册 ngram 后端; KV/专家页策略同 moe
"""
from __future__ import annotations

MOE_KEY_HINTS = (
    'num_local_experts', 'num_experts', 'num_experts_per_tok',
    'moe_intermediate_size', 'router_aux_loss_coef', 'router_top_k',
    'shared_expert_intermediate_size', 'moe_router_top_k',
)
MOE_FACTS = (
    'num_local_experts', 'num_experts', 'num_experts_per_tok',
    'moe_intermediate_size', 'shared_expert_intermediate_size',
    'router_aux_loss_coef', 'router_top_k',
)
NGRAM_FACTS = ('ngram_size', 'ngram_vocab_size_base', 'heads_per_ngram',
               'ngram_vocab_size', 'ple_conv_kernel_size', 'ple_embed_dim',
               'ple_layer_ids', 'make_ngram_vocab_size_divisible_by')


def _ngram_hits(tc: dict) -> list:
    hits = []
    for k in tc:
        kl = k.lower()
        if 'ngram' in kl or kl.startswith('ple') or 'n_gram' in kl:
            hits.append(k)
    return hits


def _positive(tc: dict, k: str) -> bool:
    """值语义: None/0/False 不算信号 (Gemma-4 带 enable_moe_block=false 的
    MoE 配置残留即此类)."""
    v = tc.get(k)
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    try:
        return float(v) > 0.0
    except (TypeError, ValueError):
        return False


def classify_config_text(tc: dict, architectures=None) -> dict:
    """HF text_config 形态判别 (键存在 + 数值为正才算信号)."""
    strong = ('num_local_experts', 'num_experts', 'num_experts_per_tok',
              'moe_intermediate_size', 'top_k_experts', 'enable_moe_block',
              'moe_router_top_k')
    moe_hits = [k for k in strong if k in tc and _positive(tc, k)]
    arch = ' '.join(architectures or []).lower()
    if 'moe' in arch and ('router_aux_loss_coef' in tc or moe_hits):
        moe_hits.append('architectures:%s' % arch[:60])
    ngram_hits = _ngram_hits(tc)
    has_moe = bool(moe_hits)
    has_ngram = bool(ngram_hits)
    kind = 'moe_ngram' if (has_moe and has_ngram) else 'moe' if has_moe else 'dense'
    return {
        'kind': kind,
        'evidence': {'moe_keys': moe_hits, 'ngram_keys': ngram_hits},
        'moe': {k: tc[k] for k in MOE_FACTS if k in tc and _positive(tc, k)},
        'ngram': {k: tc[k] for k in NGRAM_FACTS if k in tc},
    }


def classify_spec(spec: dict) -> dict:
    """archkit spec 形态判别 (moe/ple/gdn 段)."""
    has_moe = 'moe' in spec and bool(spec.get('moe'))
    has_ngram = 'ple' in spec and bool(spec.get('ple'))
    kind = 'moe_ngram' if (has_moe and has_ngram) else 'moe' if has_moe else 'dense'
    return {
        'kind': kind,
        'evidence': {'spec_sections': [s for s in ('moe', 'ple', 'gdn', 'mtp')
                                       if s in spec and spec.get(s)]},
        'moe': dict(spec.get('moe') or {}),
        'ngram': dict(spec.get('ple') or {}),
    }


def classify(cfg: dict, spec: dict | None = None) -> dict:
    """统一入口: 优先 config 字段, spec 段补充事实."""
    if cfg and ('text_config' in cfg or 'architectures' in cfg or
                any(k in cfg for k in MOE_KEY_HINTS) or _ngram_hits(cfg)):
        tc = cfg.get('text_config') or cfg
        res = classify_config_text(tc, cfg.get('architectures'))
    elif spec is not None and 'geometry' in spec:
        res = classify_spec(spec)
    else:
        res = {'kind': 'dense', 'evidence': {'note': 'no signals'}, 'moe': {}, 'ngram': {}}
    # spec 段补充 (config 无 ngram 键但 spec 有 ple 段时)
    if spec is not None and res['kind'] == 'dense' and 'ple' in spec and spec.get('ple'):
        res['kind'] = 'moe_ngram' if res.get('moe') else 'dense'
        res['ngram'].update(dict(spec['ple']))
    return finalize(res)


GUIDANCE = {
    'dense': ('稠密族: 层图按 layer_types(full/swa/gdn), MLP=稠密 swiglu; '
              '显存全量驻留; draft 按能力注册 (mtp/查表需 ngram 数据侧车)'),
    'moe': ('MoE 族: MLP 层 = router + 专家 (页粒度按需换入, MoE 权重卸载友好); '
            '布局 = 专家页 + 共享专家(如有); router 需校准数据; draft 同族可选'),
    'moe_ngram': ('MoE+ngram 族: 在 MoE 之上叠加 ngram 查表草稿 (PLE sidecar, '
                  'SSD 外挂); draft 注册 ngram 后端; KV/专家页策略同 moe'),
}


def finalize(res: dict) -> dict:
    res['guidance'] = GUIDANCE.get(res['kind'], '')
    res['draft_recommendation'] = ('ngram_ple' if res['kind'] == 'moe_ngram'
                                   else 'mtp_or_existing')
    return res

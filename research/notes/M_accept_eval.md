# 接受率判定 (预注册规则, _TODO.md 112)

ckpt=data\dflash2_ckpts\step_001200.pt anchors=300
- chain  source=None hit@1=None hit@4-hit@1=None
- beam   source=teacher hit@1=0.2719 hit@4-hit@1=0.1357
- tree   source=None hit@1=None hit@4-hit@1=None
**VERDICT: EVAL_FAILED** (无 DECISION 行 — 缺失数据不得当结论)

### chain 尾部
    teacher: hit@8  (oracle tree acceptance, 8 leaves): 0.4943
    teacher: hit@16 (oracle tree acceptance, 16 leaves): 0.5238
    teacher: mean rank of teacher token: 9771.54
    teacher: chain headroom (1 - hit@1): 0.7281
    teacher-recall ceiling ids16[t,0]==tokens[t+1] (sampled files, by src):
      unknown    0.2594  (12771 positions)
      TOTAL      0.2594  (12771 positions)
      (src labels unavailable: set DF2_CORPUS or place hs_corpus.jsonl by the cache)

### beam 尾部
    hit@1    0.0476      0.0476
    hit@2    0.0524      0.0600
    hit@4    0.0605      0.0767
    hit@8    0.0662      0.0938
    DECISION teacher row-wise : hit@1=0.2719 hit@4-hit@1=+0.1357
    DECISION teacher path-wise: hit@1=0.2719 hit@4-hit@1=+0.0671
    ORACLE-GAP row_hit4 - path_hit4 = +0.0686
    GATE (path-wise, engine decision, targets=teacher): hit@1>=0.35 AND hit@4-hit@1>=0.12 -> NO-TREE (keep training / abandon trees)

### tree 尾部
    tree: blocks=300  EAL chain=0.6200  EAL beam4=0.7333  delta=+0.1133 tok/block
    tree: CAVEAT -- offline rows have no pair codebooks (predecessor-independent); this is BUILDER COVERAGE, not draft acceptance.
    tree: re-measure hit@1/hit@4 and EAL in-engine before believing gains.

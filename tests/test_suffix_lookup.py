# -*- coding: utf-8 -*-
"""suffix_lookup 语义验证 (离线 CPU 版, 内核算法镜像)。

GPU 被训练占用时先用纯 Python 把算法语义钉死; GPU 空闲后跑同一批
用例对照真内核。镜像实现逐行对应 suffix_lookup_kernel。
"""
import random

VOCAB = 20


def kernel_mirror(ids, start, length, query, min_len, cont_tokens):
    """逐行镜像 GPU 内核: 每个候选窗口 o 算尾部对齐长度, 取 (最长, 最新)。
    只搜严格更早的非重叠窗口: o+query <= start (2026-09-03 修正, 防自匹配)。"""
    limit = min(length - query - cont_tokens, start - query)
    best_l, best_o = 0, -1
    for o in range(max(0, limit)):
        l = 0
        for q in range(query):
            if ids[start + query - 1 - q] == ids[o + query - 1 - q]:
                l = q + 1
            else:
                break
        if l >= min_len and (l > best_l or (l == best_l and o > best_o)):
            best_l, best_o = l, o
    if best_l < min_len:
        return 0, -1, [-1] * cont_tokens
    return best_l, best_o, [ids[best_o + query + k] for k in range(cont_tokens)]


def brute(ids, start, length, query, min_len, cont_tokens):
    """暴力语义: 与镜像同一规则但直接枚举所有 o(校验镜像无笔误)。"""
    limit = min(length - query - cont_tokens, start - query)
    cands = []
    for o in range(max(0, limit)):
        l = 0
        for q in range(query):
            if ids[start + query - 1 - q] == ids[o + query - 1 - q]:
                l = q + 1
            else:
                break
        if l >= min_len:
            cands.append((l, o))
    if not cands:
        return 0, -1, [-1] * cont_tokens
    l, o = max(cands, key=lambda t: (t[0], t[1]))
    return l, o, [ids[o + query + k] for k in range(cont_tokens)]


def main():
    rng = random.Random(7)
    fails = 0
    for trial in range(4000):
        query = rng.randint(2, 8)
        cont = rng.randint(1, 4)
        min_len = rng.randint(1, query)
        length = rng.randint(query * 2 + 4, 60)     # 保证 start 有合法区间
        ids = [rng.randrange(VOCAB) for _ in range(length)]
        start = rng.randint(query, length - query)  # 查询尾必须完整落在历史内
        m = kernel_mirror(ids, start, length, query, min_len, cont)
        b = brute(ids, start, length, query, min_len, cont)
        if m != b:
            fails += 1
            if fails < 4:
                print('MISMATCH', ids, start, query, min_len, cont, m, b)
    print('suffix_lookup semantic trials=4000 fails=%d' % fails)
    # 手写样例: 历史重复段应命中并续写 (查询 = ids[start .. start+query))
    ids = [1, 2, 3, 9, 9, 1, 2, 3, 7, 8]           # [1,2,3] @0 与 @5
    l, o, c = kernel_mirror(ids, start=5, length=10, query=3, min_len=2, cont_tokens=3)
    print('dup sample: len=%d off=%d cont=%s (expect len=3 off=0 cont=[9,9,1])' % (l, o, c))
    assert (l, o, c) == (3, 0, [9, 9, 1])
    # 部分匹配: 窗口 [1,2,6,8] 的尾 2 token 对齐查询 [3,5,6,8] (第 3 位 2 != 5)
    ids = [1, 2, 6, 8, 0, 0, 3, 5, 6, 8]
    l, o, c = kernel_mirror(ids, start=6, length=10, query=4, min_len=2, cont_tokens=2)
    print('partial: len=%d off=%d cont=%s (expect len=2 off=0 cont=[0,0])' % (l, o, c))
    assert l == 2 and o == 0 and c == [0, 0]        # 续写 = 历史在匹配窗口后 2 token
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())

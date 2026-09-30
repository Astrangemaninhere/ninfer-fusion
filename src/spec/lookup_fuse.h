#pragma once
// lookup_fuse.h — LABD/1Cat "历史整段填充" 链式查表 (vLLM DFLASH2_CHAIN /
// PR366 fuse_draft 的语义移植, v1 纯宿主逻辑; GPU 侧是 suffix_lookup 内核 +
// 本策略的宿主调度)。
//
// 策略 (v1, 与 1Cat "verify block the context fills" 对齐):
//   - 窗口全匹配 (L == Q): 直接链式提出 K 个历史续写 token
//     (上下文整段来自历史, 草稿器不参与 —— 1Cat 25k ctx 15-token/步的来源)
//   - 部分匹配 (min_len <= L < Q): 提出 1 个试探 token (历史在匹配段后的首 token),
//     下轮把该 token 并入查询再查 (渐进扩展); 引擎侧每轮调用一次本函数。
//   - L < min_len: 不出草案 (回退 drafter/MTP)。
#include "ninfer/ops/suffix_lookup.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ninfer::spec {

struct SuffixHit {
    std::int32_t length = 0;   // 匹配长度 L (0 = 无命中)
    std::int32_t offset = 0;   // 命中窗口起点 o (ids[o..o+Q) 对齐查询)
};

// 纯宿主后缀匹配 (与 suffix_lookup 内核同语义; 单行, 供策略测试/宿主回退)。
// 扫描边界由 include/ninfer/ops/suffix_lookup.h 的 suffix_lookup_scan_limit 单点给出:
// 严格更早、不重叠、且给 K 个续写留出空间 —— 与内核逐字同一判据。
// continuation_tokens 必须与将要填入草稿块的 K 相同, 否则选出的 offset 会与内核不同。
inline SuffixHit suffix_best(const std::vector<std::int32_t>& ids, std::int32_t query_start,
                             std::int32_t query_len, std::int32_t min_len,
                             std::int32_t continuation_tokens = 0) {
    SuffixHit best;
    const std::int32_t limit = ninfer::ops::suffix_lookup_scan_limit(
        query_start, static_cast<std::int32_t>(ids.size()), query_len, continuation_tokens);
    for (std::int32_t o = 0; o < limit; ++o) {
        std::int32_t l = 0;
        for (std::int32_t q = 0; q < query_len; ++q) {
            if (ids[query_start + query_len - 1 - q] == ids[o + query_len - 1 - q]) {
                l = q + 1;
            } else {
                break;
            }
        }
        if (l >= min_len && (l > best.length ||
                             (l == best.length && o > best.offset))) {
            best.length = l;
            best.offset = o;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// THE COPIED-FROM SOURCE, EXPOSED -- leg 3's input. Nothing else changes.
//
// WHY. `fuse_chain` returns only a COUNT. The offset it copied from is computed inside
// `suffix_best` (`SuffixHit::offset`, the history window start o), consumed inside `fill()` as
// `pos = hit.offset + query_len + j`, and THROWN AWAY when the function returns. So the one
// datum that would let a retention policy be keyed on WHICH PAGES OF HISTORY the drafts come
// from -- the sequence position each filled slot was copied from -- exists for the duration of
// this function and nowhere afterwards. A per-PAGE access histogram is therefore not merely
// unprinted: it is never formed.
//
// THIS IS ADDITIVE AND BEHAVIOUR-PRESERVING. `fuse_chain` keeps its exact signature and its
// exact behaviour by DELEGATING to the new function with a null source pointer, so every
// existing call site (program_impl.h:15760 and both CPU tests) compiles and behaves
// identically, and there is ONE implementation of the chain rather than two.
//
// `source_pos_out` may be null. When it is not, `source_pos_out[j]` receives the HISTORY
// position the j-th filled slot was copied from, for j < the returned count. With
// kPagedKVPageSize == 64 (core/paged_kv_cache.h:17, == kColdHostPageTokens,
// cold_host_tier.h:68-69) the page is `source_pos_out[j] / 64`, which is the cross-layer unit
// sum_dir.h:20 describes ("one logical page ACROSS ALL TEXT LAYERS ... a page moves whole").
// ---------------------------------------------------------------------------
inline std::int32_t fuse_chain_with_source(const std::vector<std::int32_t>& ids,
                                           std::int32_t query_start, std::int32_t query_len,
                                           std::int32_t min_len, std::int32_t k,
                                           std::int32_t* out,
                                           std::int32_t* source_pos_out) {
    const SuffixHit hit = suffix_best(ids, query_start, query_len, min_len, k);
    if (hit.length < min_len) { return 0; }
    const auto fill = [&](std::int32_t n) {
        std::int32_t filled = 0;
        for (std::int32_t j = 0; j < n; ++j) {
            const std::int32_t pos = hit.offset + query_len + j;
            if (pos >= static_cast<std::int32_t>(ids.size())) { break; }
            out[j] = ids[pos];
            if (source_pos_out != nullptr) { source_pos_out[j] = pos; }
            ++filled;
        }
        return filled;
    };
    if (hit.length == query_len) {
        // 窗口全匹配: 直接链 K 个历史续写 (上下文整段来自历史, 草稿器不参与)
        return fill(k);
    }
    // 部分匹配: 试探 1 个 (历史在匹配窗后的首 token), 下轮并入查询再查
    return fill(1);
}

// 链式决策: 返回应填入草稿块的 token 数 (0 = 回退 drafter); out 前 n 项为草案。
// The v1 chain, UNCHANGED, now delegating: one implementation, and this signature is
// what the live call site and both CPU tests already use.
inline std::int32_t fuse_chain(const std::vector<std::int32_t>& ids,
                               std::int32_t query_start, std::int32_t query_len,
                               std::int32_t min_len, std::int32_t k,
                               std::int32_t* out) {
    return fuse_chain_with_source(ids, query_start, query_len, min_len, k, out, nullptr);
}


} // namespace ninfer::spec

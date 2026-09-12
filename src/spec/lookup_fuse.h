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
#include <algorithm>
#include <cstdint>
#include <vector>

namespace ninfer::spec {

struct SuffixHit {
    std::int32_t length = 0;   // 匹配长度 L (0 = 无命中)
    std::int32_t offset = 0;   // 命中窗口起点 o (ids[o..o+Q) 对齐查询)
};

// 纯宿主后缀匹配 (与 suffix_lookup 内核同语义; 单行, 供策略测试/宿主回退)。
// 只搜严格更早的非重叠位置: o + query_len <= query_start。
inline SuffixHit suffix_best(const std::vector<std::int32_t>& ids, std::int32_t query_start,
                             std::int32_t query_len, std::int32_t min_len) {
    SuffixHit best;
    const std::int32_t limit =
        std::min(static_cast<std::int32_t>(ids.size()) - query_len, query_start);
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

// 链式决策: 返回应填入草稿块的 token 数 (0 = 回退 drafter); out 前 n 项为草案。
// Q = 查询窗长 (通常等于草稿块窗), K = 想提的 token 数。
// 语义: 匹配段后历史首 token = ids[o+Q] (窗口起点 o 的续写位, 全/部分匹配同式)。
inline std::int32_t fuse_chain(const std::vector<std::int32_t>& ids,
                               std::int32_t query_start, std::int32_t query_len,
                               std::int32_t min_len, std::int32_t k,
                               std::int32_t* out) {
    const SuffixHit hit = suffix_best(ids, query_start, query_len, min_len);
    if (hit.length < min_len) { return 0; }
    const auto fill = [&](std::int32_t n) {
        std::int32_t filled = 0;
        for (std::int32_t j = 0; j < n; ++j) {
            const std::int32_t pos = hit.offset + query_len + j;
            if (pos >= static_cast<std::int32_t>(ids.size())) { break; }
            out[j] = ids[pos];
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

} // namespace ninfer::spec

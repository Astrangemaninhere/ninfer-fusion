// lookup_fuse_test — fuse_chain 语义 CPU 测试 (与 suffix_lookup 测试同精神).
#include "spec/lookup_fuse.h"

#include <cstdio>
#include <vector>

int main() {
    using ninfer::spec::fuse_chain;
    int ok = 0;
    const int total = 6;

    // 1) 全匹配链 K: 历史 [1,2,3]@0 与查询 [1,2,3]@5 -> 续写 [9,9,1]
    {
        const std::vector<std::int32_t> ids = {1, 2, 3, 9, 9, 1, 2, 3, 7, 8};
        std::int32_t out[8];
        const int n = fuse_chain(ids, 5, 3, 2, 3, out);
        const bool pass = n == 3 && out[0] == 9 && out[1] == 9 && out[2] == 1;
        ok += pass;
        std::printf("%s full-chain n=%d tok=%d,%d,%d\n", pass ? "PASS" : "FAIL", n,
                    out[0], out[1], out[2]);
    }
    // 2) 部分匹配试探 1: 窗 [1,2,6,8]@0 尾 2 对齐查询 [3,5,6,8]@6 -> 试探 ids[4]=0
    {
        const std::vector<std::int32_t> ids = {1, 2, 6, 8, 0, 0, 3, 5, 6, 8};
        std::int32_t out[8];
        const int n = fuse_chain(ids, 6, 4, 2, 3, out);
        const bool pass = n == 1 && out[0] == 0;
        ok += pass;
        std::printf("%s partial-probe n=%d tok=%d\n", pass ? "PASS" : "FAIL", n, out[0]);
    }
    // 3) 无命中回退: 随机历史无重复
    {
        const std::vector<std::int32_t> ids = {1, 5, 2, 6, 3, 7, 4, 8, 9, 0};
        std::int32_t out[8];
        const int n = fuse_chain(ids, 6, 4, 3, 3, out);
        const bool pass = n == 0;
        ok += pass;
        std::printf("%s no-hit fallback n=%d\n", pass ? "PASS" : "FAIL", n);
    }
    // 4) 自匹配必须被拒绝: 唯一"匹配"位置与查询窗重叠, 且 K 装不下 -> 无草案
    {
        // ids[3..6) = [1,2,3] 与查询 ids[6..9) = [1,2,3] 相同, 但该窗 o=3 的续写就是查询
        // 自身 (o+Q=6=start, 重叠), 且 K=4 在 len=9 处放不下: 两个条件都由
        // suffix_lookup_scan_limit 排除 (limit = min(9-3-4, 6-3) = 2, o=0..1 皆不匹配)。
        // 旧期望 n=3 / tok=1,2,3 正是 2026-09-03 内核修掉的"自匹配"缺陷。
        const std::vector<std::int32_t> ids = {7, 8, 9, 1, 2, 3, 1, 2, 3};
        std::int32_t out[8];
        const int n = fuse_chain(ids, 6, 3, 2, 4, out);
        const bool pass = n == 0;
        ok += pass;
        std::printf("%s self-match rejected n=%d\n", pass ? "PASS" : "FAIL", n);
    }
    // 5) 同长度并列必须取【最晚】出现 (src/spec/lookup_fuse.h:44-45 明文;
    //    include/ninfer/ops/suffix_lookup.h:16-17 的内核契约也是 "ties prefer the LATEST
    //    occurrence (largest o)"). 这一格此前【没有任何测试覆盖】: 上面 4 例每一个都只有
    //    一个 L>=min_len 的候选位 (L16b 的探针 t5 实测: #1 只有 o=0 L=3, #2 只有 o=0 L=2,
    //    #3/#4 无候选), 所以把 `o > best.offset` 写成 `o < best.offset` 曾经 4/4 全绿。
    //    令 o=0 与 o=1 的 L 相等, 并让两个候选位给出不同 token:
    //      ids={5,5,5,5,2,0,1,5,5,0} qs=6 Q=3 K=2 min_len=2 -> scan_limit=3,
    //      o=0:L=2, o=1:L=2, o=2:L=0  => 并列
    //      取最晚(o=1) -> 试探 1 个: ids[1+3]=ids[4]=2
    //      取最早(o=0) -> 试探 1 个: ids[0+3]=ids[3]=5   <-- 注入形状
    {
        const std::vector<std::int32_t> ids = {5, 5, 5, 5, 2, 0, 1, 5, 5, 0};
        std::int32_t out[8];
        const int n = fuse_chain(ids, 6, 3, 2, 2, out);
        const bool pass = n == 1 && out[0] == 2;
        ok += pass;
        std::printf("%s tie-break partial picks LATEST n=%d tok=%d (earliest would give 5)\n",
                    pass ? "PASS" : "FAIL", n, out[0]);
    }
    // 6) 并列 + 全窗口匹配 (fill(K) 分支也必须取最晚):
    //      ids={5,5,5,5,2,0,5,5,5,0} qs=6 Q=3 K=2 min_len=2 -> scan_limit=3,
    //      o=0:L=3, o=1:L=3, o=2:L=0  => 并列且 L==Q
    //      取最晚(o=1) -> 链 2 个: ids[4],ids[5] = 2,0
    //      取最早(o=0) -> 链 2 个: ids[3],ids[4] = 5,2   <-- 注入形状
    {
        const std::vector<std::int32_t> ids = {5, 5, 5, 5, 2, 0, 5, 5, 5, 0};
        std::int32_t out[8];
        const int n = fuse_chain(ids, 6, 3, 2, 2, out);
        const bool pass = n == 2 && out[0] == 2 && out[1] == 0;
        ok += pass;
        std::printf("%s tie-break full-match picks LATEST n=%d tok=%d,%d (earliest would give 5,2)\n",
                    pass ? "PASS" : "FAIL", n, out[0], out[1]);
    }
    std::printf("== %d/%d\n", ok, total);
    return ok == total ? 0 : 1;
}

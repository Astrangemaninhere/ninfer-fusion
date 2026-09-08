// lookup_fuse_test — fuse_chain 语义 CPU 测试 (与 suffix_lookup 测试同精神).
#include "spec/lookup_fuse.h"

#include <cstdio>
#include <vector>

int main() {
    using ninfer::spec::fuse_chain;
    int ok = 0;
    const int total = 4;

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
    // 4) 匹配窗口的续写跨过历史末尾: 安全截断, 不越界不崩
    {
        // [1,2,3]@3 全匹配查询 @6? 查询 = ids[6..9) = [1,2,3]; K=4 需 pos 6..10,
        // 只有 3 个可用 (6,7,8) -> n=3, tok = 查询自身重复段
        const std::vector<std::int32_t> ids = {7, 8, 9, 1, 2, 3, 1, 2, 3};
        std::int32_t out[8];
        const int n = fuse_chain(ids, 6, 3, 2, 4, out);
        const bool pass = n == 3 && out[0] == 1 && out[1] == 2 && out[2] == 3;
        ok += pass;
        std::printf("%s boundary n=%d tok=%d,%d,%d\n", pass ? "PASS" : "FAIL", n,
                    out[0], out[1], out[2]);
    }
    std::printf("== %d/%d\n", ok, total);
    return ok == total ? 0 : 1;
}

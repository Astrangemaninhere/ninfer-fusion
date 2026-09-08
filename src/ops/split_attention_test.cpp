// split_attention_test — CPU 对拍: 分片+合并 (split_verify) 必须与整段
// softmax (attention_mono) 一致 (浮点误差内)。GPU 批用同用例对照真算子。
#include "ops/split_attention_ref.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const int batch = 2, query_heads = 4, kv_heads = 2, kv_group = 2;
    const int t = 16, dim = 8;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    std::vector<float> q(batch * query_heads * t * dim), k(batch * kv_heads * t * dim),
        v(batch * kv_heads * t * dim);
    for (auto& x : q) { x = u(rng); }
    for (auto& x : k) { x = u(rng); }
    for (auto& x : v) { x = u(rng); }

    int ok = 0;
    for (int shards : {1, 2, 4}) {
        std::vector<float> mono, split;
        ninfer::ops::ref::attention_mono(q, k, v, batch, query_heads, kv_heads, kv_group,
                                         t, dim, scale, mono);
        ninfer::ops::ref::split_verify(q, k, v, batch, query_heads, kv_heads, kv_group, t,
                                       dim, scale, shards, split);
        float maxerr = 0.0f;
        for (std::size_t i = 0; i < mono.size(); ++i) {
            maxerr = std::max(maxerr, std::fabs(mono[i] - split[i]));
        }
        const bool pass = maxerr < 1e-4f;
        ok += pass;
        std::printf("%s shards=%d maxerr=%.2e\n", pass ? "PASS" : "FAIL", shards, maxerr);
    }
    std::printf("== %d/3\n", ok);
    return ok == 3 ? 0 : 1;
}

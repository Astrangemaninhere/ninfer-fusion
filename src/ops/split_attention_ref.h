// split_attention_ref.h — split-KV verify 数学的 CPU 参考 (对拍用)。
// 语义: 把 QK^T 行按局部片切分, 每片做在线 softmax 出 (max, sum, pv),
// 合并后与"整段一次 softmax"逐位近似相等 (浮点误差内)。
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace ninfer::ops::ref {

// 整段注意力 (monolithic): out[b,hq,tq] = softmax_tk(scale*q·k) · v
inline void attention_mono(const std::vector<float>& q, const std::vector<float>& k,
                           const std::vector<float>& v, int batch, int query_heads,
                           int kv_heads, int kv_group, int t, int dim, float scale,
                           std::vector<float>& out) {
    out.assign(static_cast<std::size_t>(batch) * query_heads * t * dim, 0.0f);
    for (int b = 0; b < batch; ++b) {
        for (int hq = 0; hq < query_heads; ++hq) {
            const int g = hq / kv_group;
            for (int tq = 0; tq < t; ++tq) {
                std::vector<float> scores(t);
                float m = -1e30f;
                for (int tk = 0; tk <= tq; ++tk) {
                    float acc = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        acc += q[((b * query_heads + hq) * t + tq) * dim + d] *
                               k[((b * kv_heads + g) * t + tk) * dim + d];
                    }
                    scores[tk] = acc * scale;
                    m = std::max(m, scores[tk]);
                }
                float s = 0.0f;
                for (int tk = 0; tk <= tq; ++tk) { s += std::exp(scores[tk] - m); }
                for (int d = 0; d < dim; ++d) {
                    float acc = 0.0f;
                    for (int tk = 0; tk <= tq; ++tk) {
                        acc += std::exp(scores[tk] - m) *
                               v[((b * kv_heads + g) * t + tk) * dim + d];
                    }
                    out[((b * query_heads + hq) * t + tq) * dim + d] = s > 0 ? acc / s : 0.0f;
                }
            }
        }
    }
}

// 分片局部 partials + 合并 (镜像 split_attention.cu 语义; shards 等分位置,
// 位置分片: 片 r 覆盖 [r*ts, (r+1)*ts), ts = t/shards, t 须整除)。
inline void split_verify(const std::vector<float>& q, const std::vector<float>& k,
                         const std::vector<float>& v, int batch, int query_heads,
                         int kv_heads, int kv_group, int t, int dim, float scale,
                         int shards, std::vector<float>& out) {
    const int ts = t / shards;
    out.assign(static_cast<std::size_t>(batch) * query_heads * t * dim, 0.0f);
    for (int b = 0; b < batch; ++b) {
        for (int hq = 0; hq < query_heads; ++hq) {
            const int g = hq / kv_group;
            for (int tq = 0; tq < t; ++tq) {
                // 每片在线 softmax
                float m = -1e30f, s = 0.0f;
                std::vector<float> pv(dim, 0.0f);
                for (int r = 0; r < shards; ++r) {
                    float lm = -1e30f, ls = 0.0f;
                    std::vector<float> lp(dim, 0.0f);
                    for (int tk = r * ts; tk < (r + 1) * ts; ++tk) {
                        if (tk > tq) { break; }
                        float acc = 0.0f;
                        for (int d = 0; d < dim; ++d) {
                            acc += q[((b * query_heads + hq) * t + tq) * dim + d] *
                                   k[((b * kv_heads + g) * t + tk) * dim + d];
                        }
                        acc *= scale;
                        const float mn = acc > lm ? acc : lm;
                        const float corr = std::exp(lm - mn);
                        ls = ls * corr + std::exp(acc - mn);
                        for (int d = 0; d < dim; ++d) {
                            lp[d] = lp[d] * corr +
                                    std::exp(acc - mn) *
                                        v[((b * kv_heads + g) * t + tk) * dim + d];
                        }
                        lm = mn;
                    }
                    // 合并片 r
                    const float mn = lm > m ? lm : m;
                    const float corr = std::exp(m - mn);
                    s = s * corr + std::exp(lm - mn) * ls;
                    for (int d = 0; d < dim; ++d) {
                        pv[d] = pv[d] * corr + std::exp(lm - mn) * lp[d];
                    }
                    m = mn;
                }
                for (int d = 0; d < dim; ++d) {
                    out[((b * query_heads + hq) * t + tq) * dim + d] =
                        s > 0 ? pv[d] / s : 0.0f;
                }
            }
        }
    }
}

} // namespace ninfer::ops::ref

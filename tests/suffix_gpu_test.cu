// suffix_gpu_test.cu — suffix_lookup 真核 vs CPU 镜像 (GPU 空闲对照)。
// 语义与 tests/test_suffix_lookup.py 镜像一致: 只搜 o+query<=start。
#include "ninfer/ops/suffix_lookup.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int mirror_best(const std::vector<int>& ids, int start, int len, int query,
                       int cont, int* out) {
    int limit = std::min(len - query - cont, start - query);
    int best_l = 0, best_o = -1;
    for (int o = 0; o < std::max(0, limit); ++o) {
        int l = 0;
        for (int q = 0; q < query; ++q) {
            if (ids[start + query - 1 - q] == ids[o + query - 1 - q]) { l = q + 1; }
            else { break; }
        }
        if (l >= 2 && (l > best_l || (l == best_l && o > best_o))) { best_l = l; best_o = o; }
    }
    if (best_l < 2) { return 0; }
    for (int k = 0; k < cont; ++k) {
        out[k] = ids[best_o + query + k];
    }
    return best_l;
}

int main() {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> tok(0, 19);
    std::vector<int> hist;
    std::vector<int> starts(1), lengths(1), best_len(1), best_off(1), contd(8);
    int fails = 0, runs = 0;
    for (int trial = 0; trial < 300; ++trial) {
        int query = 2 + rng() % 6;
        int len = query * 2 + 4 + rng() % 50;
        hist.resize(len);
        for (auto& x : hist) { x = tok(rng); }
        int start = query + rng() % max(1, len - query - 10 + 1);
        if (len - query - 8 < 0) { continue; }
        starts[0] = start;
        lengths[0] = len;
        int *d_hist, *d_st, *d_le, *d_bl, *d_bo, *d_co;
        cudaMalloc(&d_hist, len * 4);
        cudaMalloc(&d_st, 4); cudaMalloc(&d_le, 4);
        cudaMalloc(&d_bl, 4); cudaMalloc(&d_bo, 4); cudaMalloc(&d_co, 32);
        cudaMemcpy(d_hist, hist.data(), len * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_st, starts.data(), 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_le, lengths.data(), 4, cudaMemcpyHostToDevice);
        ninfer::ops::suffix_lookup(d_hist, len, d_st, d_le, 1, query, 2, 8, d_bl, d_bo,
                                   d_co, 0);
        cudaMemcpy(best_len.data(), d_bl, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(best_off.data(), d_bo, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(contd.data(), d_co, 32, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        int mout[8];
        int ml = mirror_best(hist, start, len, query, 8, mout);
        int bl = best_len[0];
        ++runs;
        bool ok = (bl == 0 && ml == 0) || (bl >= 2 && ml >= 2 && bl == ml);
        if (bl >= 2 && ml >= 2) {
            for (int k = 0; k < 8; ++k) { if (contd[k] != mout[k]) { ok = false; } }
        }
        if (!ok) {
            ++fails;
            if (fails < 4) {
                printf("FAIL trial %d q=%d start=%d len=%d gpu_l=%d mirror_l=%d\n",
                       trial, query, start, len, bl, ml);
            }
        }
        cudaFree(d_hist); cudaFree(d_st); cudaFree(d_le);
        cudaFree(d_bl); cudaFree(d_bo); cudaFree(d_co);
    }
    printf("runs=%d fails=%d\n", runs, fails);
    return fails == 0 ? 0 : 1;
}

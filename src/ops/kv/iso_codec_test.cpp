// iso_codec_test — CPU 对拍: 固定向量 -> iso4/iso3 编解码 -> 打印与误差,
// 供 python 参考 (kv_iso_ref.py) 与引擎 CUDA 版对照。
#include "ops/kv/iso_codec.h"

#include <cstdio>
#include <vector>

int main() {
    // 固定向量: 覆盖正负/零/饱和 (组大小 16 / 8 整除)
    const float v4[16] = {0.5f, -0.5f, 1.0f, -1.0f, 3.0f, -3.0f, 7.0f, -7.0f,
                          6.5f, 0.0f, 2.0f, -2.0f, 4.5f, -4.5f, 1.5f, -1.5f};
    const float v3[8] = {0.5f, -0.5f, 1.0f, -1.0f, 3.0f, -3.0f, 0.0f, 2.5f};

    std::uint8_t c4[8];
    std::uint16_t s4[1];
    ninfer::ops::kv::iso4_encode(v4, 16, c4, s4);
    float d4[16];
    ninfer::ops::kv::iso4_decode(c4, s4, 16, d4);
    std::printf("iso4 codes:");
    for (int i = 0; i < 8; ++i) { std::printf(" %02x", c4[i]); }
    std::printf(" scale=%04x\n", s4[0]);
    float maxe4 = 0.0f;
    for (int i = 0; i < 16; ++i) {
        maxe4 = std::max(maxe4, std::fabs(d4[i] - v4[i]));
        std::printf("  d4[%d] %.4f -> %.4f\n", i, v4[i], d4[i]);
    }

    std::uint8_t c3[3];
    std::uint16_t s3[1];
    ninfer::ops::kv::iso3_encode(v3, 8, c3, s3);
    float d3[8];
    ninfer::ops::kv::iso3_decode(c3, s3, 8, d3);
    std::printf("iso3 codes:");
    for (int i = 0; i < 3; ++i) { std::printf(" %02x", c3[i]); }
    std::printf(" scale=%04x\n", s3[0]);
    float maxe3 = 0.0f;
    for (int i = 0; i < 8; ++i) {
        maxe3 = std::max(maxe3, std::fabs(d3[i] - v3[i]));
        std::printf("  d3[%d] %.4f -> %.4f\n", i, v3[i], d3[i]);
    }
    std::printf("maxerr iso4=%.5f iso3=%.5f\n", maxe4, maxe3);
    return 0;
}

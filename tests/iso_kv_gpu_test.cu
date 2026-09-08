// iso_kv_gpu_test.cu — iso4/iso3 GPU 编解码 vs CPU 参考 (逐字节)。
#include "ops/kv/iso_codec.h"

#include <cstdio>
#include <random>
#include <vector>

using namespace ninfer::ops::kv;

__global__ void iso4e(const float* v, unsigned char* codes, unsigned short* scales,
                      int groups) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) { return; }
    const float* x = v + g * kIso4Group;
    float mx = 0.0f;
    for (int i = 0; i < kIso4Group; ++i) { mx = fmaxf(mx, fabsf(x[i])); }
    float s = mx / 7.0f;
    scales[g] = fp16_bits(s > 0.0f ? s : 1.0f);
    float ss = s > 0.0f ? s : 1.0f;
    for (int i = 0; i < kIso4Group; i += 2) {
        int c0 = (int)lrintf(x[i] / ss);
        int c1 = (int)lrintf(x[i + 1] / ss);
        c0 = (c0 < -7 ? -7 : (c0 > 7 ? 7 : c0)) & 0xf;
        c1 = (c1 < -7 ? -7 : (c1 > 7 ? 7 : c1)) & 0xf;
        codes[(g * kIso4Group + i) / 2] = (unsigned char)(c0 | (c1 << 4));
    }
}

__global__ void iso3e(const float* v, unsigned char* codes, unsigned short* scales,
                      int groups) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) { return; }
    const float* x = v + g * kIso3Group;
    float mx = 0.0f;
    for (int i = 0; i < kIso3Group; ++i) { mx = fmaxf(mx, fabsf(x[i])); }
    float s = mx / 3.0f;
    scales[g] = fp16_bits(s > 0.0f ? s : 1.0f);
    float ss = s > 0.0f ? s : 1.0f;
    unsigned packed = 0;
    for (int i = 0; i < kIso3Group; ++i) {
        int mag = (int)lrintf(fabsf(x[i]) / ss);
        mag = mag < 0 ? 0 : (mag > 3 ? 3 : mag);
        packed |= ((x[i] < 0 ? 1u : 0u) << 2 | (unsigned)mag) << (3 * i);
    }
    unsigned char* c = codes + g * 3;
    c[0] = packed & 0xff;
    c[1] = (packed >> 8) & 0xff;
    c[2] = (packed >> 16) & 0xff;
}

int main() {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-3.0f, 3.0f);
    const int G4 = 4096, G3 = 4096;
    std::vector<float> v4(G4 * kIso4Group), v3(G3 * kIso3Group);
    for (auto& x : v4) { x = u(rng); }
    for (auto& x : v3) { x = u(rng); }
    std::vector<unsigned char> c4(iso4_codes_bytes(v4.size())),
        c3(iso3_codes_bytes(v3.size()));
    std::vector<unsigned short> s4(G4), s3(G3);
    float *dv4, *dv3;
    unsigned char *dc4, *dc3;
    unsigned short *ds4, *ds3;
    cudaMalloc(&dv4, v4.size() * 4);
    cudaMalloc(&dv3, v3.size() * 4);
    cudaMalloc(&dc4, c4.size());
    cudaMalloc(&dc3, c3.size());
    cudaMalloc(&ds4, G4 * 2);
    cudaMalloc(&ds3, G3 * 2);
    cudaMemcpy(dv4, v4.data(), v4.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv3, v3.data(), v3.size() * 4, cudaMemcpyHostToDevice);
    iso4e<<<(G4 + 255) / 256, 256>>>(dv4, dc4, ds4, G4);
    iso3e<<<(G3 + 255) / 256, 256>>>(dv3, dc3, ds3, G3);
    cudaMemcpy(c4.data(), dc4, c4.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(c3.data(), dc3, c3.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(s4.data(), ds4, G4 * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(s3.data(), ds3, G3 * 2, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::vector<unsigned char> r4(c4.size()), r3(c3.size());
    std::vector<unsigned short> q4(G4), q3(G3);
    iso4_encode(v4.data(), v4.size(), r4.data(), q4.data());
    iso3_encode(v3.data(), v3.size(), r3.data(), q3.data());
    int bad = 0;
    for (size_t i = 0; i < c4.size(); ++i) { bad += c4[i] != r4[i]; }
    for (size_t i = 0; i < c3.size(); ++i) { bad += c3[i] != r3[i]; }
    for (int i = 0; i < G4; ++i) { bad += s4[i] != q4[i]; }
    for (int i = 0; i < G3; ++i) { bad += s3[i] != q3[i]; }
    printf("iso gpu-vs-cpu: iso4 bytes=%zu iso3 bytes=%zu scales=%d mismatches=%d\n",
           c4.size(), c3.size(), G4 + G3, bad);
    return bad == 0 ? 0 : 1;
}

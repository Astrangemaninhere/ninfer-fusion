// ple_gather_test.cu — PLE n-gram 真表 gather 的 GPU/宿主双通道对拍 (W2①)
//
// 目的: 把 tools/archkit/ple_gather_check.py 在宿主侧验证过的「行号推导 + 真表 gather」
// 搬到 GPU 上跑一遍, 逐字节对拍。两个通道都必须独立实现同一公式 (不共享代码路径):
//   host : CPU 推导行号 + 宿主随机读文件
//   device: CUDA 核推导行号 + 设备端 gather (从上传的唯一行表里二分查找)
//
// 公式 (HF 源码实证, transformers.models.qwen4_exp.modeling_qwen4_exp):
//   mixed = tok0*M0  XOR  tok1*M1  [XOR tok2*M2]      // bigram / trigram 各自算, uint64 回绕
//   row[h] = mixed % per_head_vocabulary_sizes[h] + per_head_offsets[h]
//   头 0..7 = bigram, 8..15 = trigram; 上下文遇 EOS 截断 (后续位填 EOS)
//
// 规格文件由 ple_gather_check.py --emit-spec 生成 (扁平文本, 避免在 .cu 里写 JSON 解析)。
//
// 编译 (任意 GPU 窗口, 无引擎依赖):
//   nvcc -O2 -std=c++17 tools/ple_gather_test.cu -o ple_gather_test
// 用法:
//   ./ple_gather_test --spec ple_spec.txt --data-dir <分片目录> [--expect-fnv 0x...]
//   --expect-fnv 取自 ple_gather_check.py 打印的「载荷 fnv1a64」
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#define CUDA_OK(expr)                                                                    \
    do {                                                                                 \
        cudaError_t err__ = (expr);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "%s:%d CUDA error %s: %s\n", __FILE__, __LINE__,        \
                         #expr, cudaGetErrorString(err__));                              \
            std::exit(2);                                                                \
        }                                                                                \
    } while (0)

namespace {

constexpr uint64_t kFnvOffset = 0xCBF29CE484222325ULL;
constexpr uint64_t kFnvPrime  = 0x100000001B3ULL;

uint64_t fnv1a64(const unsigned char* data, size_t bytes, uint64_t seed = kFnvOffset) {
    uint64_t value = seed;
    for (size_t i = 0; i < bytes; ++i) {
        value ^= data[i];
        value *= kFnvPrime;
    }
    return value;
}

struct Part {
    uint64_t global_row_start = 0;
    uint64_t rows             = 0;
    uint64_t file_offset      = 0;
    int      file_index       = 0;
};

struct Spec {
    int ngram = 3;
    int heads_per_ngram = 8;
    int n_heads = 16;
    int row_stride = 320;
    int row_dim = 160;
    uint64_t usable_rows = 0;
    int eos = 0;
    std::vector<uint64_t> mult, sizes, offsets;
    std::vector<int> tokens;
    std::vector<std::string> files;  // index -> path
    std::vector<Part> parts;
};

bool load_spec(const std::string& path, Spec& spec) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open spec %s\n", path.c_str()); return false; }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream is(line);
        std::string key;
        is >> key;
        if (key == "ngram") is >> spec.ngram;
        else if (key == "heads_per_ngram") is >> spec.heads_per_ngram;
        else if (key == "n_heads") is >> spec.n_heads;
        else if (key == "row_stride") is >> spec.row_stride;
        else if (key == "row_dim") is >> spec.row_dim;
        else if (key == "usable_rows") is >> spec.usable_rows;
        else if (key == "eos") is >> spec.eos;
        else if (key == "mult" || key == "sizes" || key == "offsets") {
            uint64_t value = 0;
            std::vector<uint64_t> values;
            while (is >> value) values.push_back(value);
            if (key == "mult") spec.mult = values;
            else if (key == "sizes") spec.sizes = values;
            else spec.offsets = values;
        } else if (key == "tokens") {
            int value = 0;
            while (is >> value) spec.tokens.push_back(value);
        } else if (key == "file") {
            int index = 0;
            std::string name;
            is >> index >> name;
            if (static_cast<int>(spec.files.size()) <= index) spec.files.resize(index + 1);
            spec.files[index] = name;
        } else if (key == "part") {
            Part part;
            is >> part.global_row_start >> part.rows >> part.file_offset >> part.file_index;
            spec.parts.push_back(part);
        }
    }
    std::sort(spec.parts.begin(), spec.parts.end(),
              [](const Part& a, const Part& b) { return a.global_row_start < b.global_row_start; });
    return !spec.mult.empty() && !spec.sizes.empty() && !spec.tokens.empty() && !spec.parts.empty();
}

// ---------------- host 通道 ----------------
std::vector<uint64_t> host_derive_rows(const Spec& spec, const std::vector<int>& prevs) {
    const int n_prev = spec.ngram - 1;
    std::vector<uint64_t> rows(static_cast<size_t>(spec.tokens.size()) * spec.n_heads, 0);
    for (size_t i = 0; i < spec.tokens.size(); ++i) {
        uint64_t ctx[3] = {static_cast<uint64_t>(spec.tokens[i]),
                           static_cast<uint64_t>(spec.eos), static_cast<uint64_t>(spec.eos)};
        bool cut = false;
        for (int s = 1; s < spec.ngram; ++s) {
            const int prev = prevs[i * n_prev + (s - 1)];
            cut = cut || prev < 0 || prev == spec.eos;
            ctx[s] = cut ? static_cast<uint64_t>(spec.eos) : static_cast<uint64_t>(prev);
        }
        for (int n = 2; n <= spec.ngram; ++n) {
            uint64_t mixed = ctx[0] * spec.mult[0];
            for (int j = 1; j < n; ++j) mixed ^= ctx[j] * spec.mult[j];
            const int base = (n - 2) * spec.heads_per_ngram;
            for (int g = 0; g < spec.heads_per_ngram; ++g) {
                const int h = base + g;
                rows[i * spec.n_heads + h] = (mixed % spec.sizes[h]) + spec.offsets[h];
            }
        }
    }
    return rows;
}

bool host_gather_row(const Spec& spec, const std::string& dir, uint64_t row,
                     std::vector<unsigned char>& out) {
    for (const Part& part : spec.parts) {
        if (row < part.global_row_start || row >= part.global_row_start + part.rows) continue;
        const uint64_t offset = part.file_offset + (row - part.global_row_start) * spec.row_stride;
        const std::string path = dir + "/" + spec.files[part.file_index];
        std::ifstream in(path, std::ios::binary);
        if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
        in.seekg(static_cast<std::streamoff>(offset));
        out.assign(spec.row_stride, 0);
        in.read(reinterpret_cast<char*>(out.data()), spec.row_stride);
        return static_cast<bool>(in);
    }
    std::fprintf(stderr, "row %llu out of logical range\n",
                 static_cast<unsigned long long>(row));
    return false;
}

// ---------------- device 通道 ----------------
__global__ void derive_rows_kernel(const int* tokens, const int* prevs, int n_tokens, int eos,
                                   int ngram, int heads_per_ngram, int n_heads,
                                   const uint64_t* mult, const uint64_t* sizes,
                                   const uint64_t* offsets, uint64_t* rows_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tokens) return;
    const int n_prev = ngram - 1;
    uint64_t ctx[3];
    ctx[0] = static_cast<uint64_t>(tokens[i]);
    bool cut = false;
    for (int s = 1; s < ngram; ++s) {
        const int prev = prevs[i * n_prev + (s - 1)];
        cut = cut || prev < 0 || prev == eos;
        ctx[s] = cut ? static_cast<uint64_t>(eos) : static_cast<uint64_t>(prev);
    }
    for (int n = 2; n <= ngram; ++n) {
        uint64_t mixed = ctx[0] * mult[0];
        for (int j = 1; j < n; ++j) mixed ^= ctx[j] * mult[j];
        const int base = (n - 2) * heads_per_ngram;
        for (int g = 0; g < heads_per_ngram; ++g) {
            const int h = base + g;
            rows_out[i * n_heads + h] = (mixed % sizes[h]) + offsets[h];
        }
    }
}

// 唯一行表按 row id 升序; 每个 (token, head) 用二分查找定位后拷贝 row_stride 字节。
__global__ void gather_kernel(const uint64_t* rows, int n_rows_per_token, int n_tokens,
                              const uint64_t* row_ids, const unsigned char* row_data,
                              int unique_rows, int row_stride, unsigned char* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_tokens * n_rows_per_token;
    if (idx >= total) return;
    const uint64_t target = rows[idx];
    int lo = 0, hi = unique_rows - 1, found = -1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        if (row_ids[mid] == target) { found = mid; break; }
        if (row_ids[mid] < target) lo = mid + 1; else hi = mid - 1;
    }
    if (found < 0) return;  // 由 host 侧保证存在
    const unsigned char* src = row_data + static_cast<size_t>(found) * row_stride;
    unsigned char* dst = out + static_cast<size_t>(idx) * row_stride;
    for (int b = 0; b < row_stride; ++b) dst[b] = src[b];
}

__global__ void fnv_kernel(const unsigned char* data, size_t bytes, uint64_t* acc) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;  // 单线程串行算 FNV (测试规模小, 简单即可)
    uint64_t value = kFnvOffset;
    for (size_t i = 0; i < bytes; ++i) { value ^= data[i]; value *= kFnvPrime; }
    *acc = value;
}

}  // namespace

int main(int argc, char** argv) {
    std::string spec_path, data_dir, expect_fnv;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--spec" && i + 1 < argc) spec_path = argv[++i];
        else if (arg == "--data-dir" && i + 1 < argc) data_dir = argv[++i];
        else if (arg == "--expect-fnv" && i + 1 < argc) expect_fnv = argv[++i];
        else { std::fprintf(stderr, "unknown arg %s\n", arg.c_str()); return 2; }
    }
    if (spec_path.empty() || data_dir.empty()) {
        std::fprintf(stderr, "usage: %s --spec <ple_spec.txt> --data-dir <dir> [--expect-fnv 0x...]\n",
                     argv[0]);
        return 2;
    }
    Spec spec;
    if (!load_spec(spec_path, spec)) return 2;
    std::printf("spec: tokens=%zu n_heads=%d row_stride=%d ngram=%d\n",
                spec.tokens.size(), spec.n_heads, spec.row_stride, spec.ngram);

    // 1) host 通道: 推导 + gather
    std::vector<int> prevs(spec.tokens.size() * (spec.ngram - 1), spec.eos);
    const std::vector<uint64_t> host_rows = host_derive_rows(spec, prevs);

    std::vector<uint64_t> unique_rows = host_rows;
    std::sort(unique_rows.begin(), unique_rows.end());
    unique_rows.erase(std::unique(unique_rows.begin(), unique_rows.end()), unique_rows.end());

    std::vector<unsigned char> host_payload;
    host_payload.reserve(host_rows.size() * spec.row_stride);
    for (uint64_t row : host_rows) {
        std::vector<unsigned char> vec;
        if (!host_gather_row(spec, data_dir, row, vec)) return 1;
        host_payload.insert(host_payload.end(), vec.begin(), vec.end());
    }
    const uint64_t host_fnv = fnv1a64(host_payload.data(), host_payload.size());
    std::printf("host : rows=%zu unique=%zu payload=%zuB fnv1a64=0x%016llx\n",
                host_rows.size(), unique_rows.size(), host_payload.size(),
                static_cast<unsigned long long>(host_fnv));

    // 2) device 通道: 上传唯一行数据 + 推导核 + gather 核
    const size_t row_bytes = static_cast<size_t>(unique_rows.size()) * spec.row_stride;
    std::vector<unsigned char> unique_payload(row_bytes);
    {
        std::unordered_map<uint64_t, size_t> slot;
        for (size_t i = 0; i < unique_rows.size(); ++i) slot[unique_rows[i]] = i;
        for (const auto& kv : slot) {
            std::vector<unsigned char> vec;
            if (!host_gather_row(spec, data_dir, kv.first, vec)) return 1;
            std::memcpy(unique_payload.data() + kv.second * spec.row_stride, vec.data(),
                        spec.row_stride);
        }
    }

    int* d_tokens = nullptr; int* d_prevs = nullptr;
    uint64_t* d_mult = nullptr; uint64_t* d_sizes = nullptr; uint64_t* d_offsets = nullptr;
    uint64_t* d_rows = nullptr; uint64_t* d_row_ids = nullptr;
    unsigned char* d_row_data = nullptr; unsigned char* d_payload = nullptr; uint64_t* d_fnv = nullptr;
    CUDA_OK(cudaMalloc(&d_tokens, spec.tokens.size() * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_prevs, prevs.size() * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_mult, spec.mult.size() * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&d_sizes, spec.sizes.size() * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&d_offsets, spec.offsets.size() * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&d_rows, host_rows.size() * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&d_row_ids, unique_rows.size() * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&d_row_data, row_bytes));
    CUDA_OK(cudaMalloc(&d_payload, host_payload.size()));
    CUDA_OK(cudaMalloc(&d_fnv, sizeof(uint64_t)));

    CUDA_OK(cudaMemcpy(d_tokens, spec.tokens.data(), spec.tokens.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_prevs, prevs.data(), prevs.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_mult, spec.mult.data(), spec.mult.size() * sizeof(uint64_t),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_sizes, spec.sizes.data(), spec.sizes.size() * sizeof(uint64_t),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_offsets, spec.offsets.data(), spec.offsets.size() * sizeof(uint64_t),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_row_ids, unique_rows.data(), unique_rows.size() * sizeof(uint64_t),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_row_data, unique_payload.data(), row_bytes, cudaMemcpyHostToDevice));

    const int n_tokens = static_cast<int>(spec.tokens.size());
    const int block = 128;
    derive_rows_kernel<<<(n_tokens + block - 1) / block, block>>>(
        d_tokens, d_prevs, n_tokens, spec.eos, spec.ngram, spec.heads_per_ngram, spec.n_heads,
        d_mult, d_sizes, d_offsets, d_rows);
    CUDA_OK(cudaGetLastError());

    const int total_rows = static_cast<int>(host_rows.size());
    gather_kernel<<<(total_rows + block - 1) / block, block>>>(
        d_rows, spec.n_heads, n_tokens, d_row_ids, d_row_data,
        static_cast<int>(unique_rows.size()), spec.row_stride, d_payload);
    CUDA_OK(cudaGetLastError());

    fnv_kernel<<<1, 1>>>(d_payload, host_payload.size(), d_fnv);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    // 3) 对拍
    std::vector<uint64_t> device_rows(host_rows.size());
    CUDA_OK(cudaMemcpy(device_rows.data(), d_rows, host_rows.size() * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
    size_t row_mismatch = 0;
    for (size_t i = 0; i < host_rows.size(); ++i) {
        if (device_rows[i] != host_rows[i]) {
            if (row_mismatch < 4) {
                std::printf("  row mismatch @%zu: host=%llu device=%llu\n", i,
                            static_cast<unsigned long long>(host_rows[i]),
                            static_cast<unsigned long long>(device_rows[i]));
            }
            ++row_mismatch;
        }
    }
    std::vector<unsigned char> device_payload(host_payload.size());
    CUDA_OK(cudaMemcpy(device_payload.data(), d_payload, host_payload.size(),
                       cudaMemcpyDeviceToHost));
    size_t byte_mismatch = 0;
    for (size_t i = 0; i < host_payload.size(); ++i) {
        if (device_payload[i] != host_payload[i]) ++byte_mismatch;
    }
    uint64_t device_fnv = 0;
    CUDA_OK(cudaMemcpy(&device_fnv, d_fnv, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    std::printf("device: rows=%zu payload=%zuB fnv1a64=0x%016llx\n",
                device_rows.size(), device_payload.size(),
                static_cast<unsigned long long>(device_fnv));
    std::printf("compare: row_mismatch=%zu byte_mismatch=%zu\n", row_mismatch, byte_mismatch);

    bool ok = row_mismatch == 0 && byte_mismatch == 0 && device_fnv == host_fnv;
    if (!expect_fnv.empty()) {
        const uint64_t expected = std::strtoull(expect_fnv.c_str(), nullptr, 0);
        const bool match = expected == host_fnv;
        std::printf("expect-fnv: %s (host=0x%016llx expected=0x%016llx)\n",
                    match ? "MATCH" : "MISMATCH",
                    static_cast<unsigned long long>(host_fnv),
                    static_cast<unsigned long long>(expected));
        ok = ok && match;
    }

    cudaFree(d_tokens); cudaFree(d_prevs); cudaFree(d_mult); cudaFree(d_sizes);
    cudaFree(d_offsets); cudaFree(d_rows); cudaFree(d_row_ids); cudaFree(d_row_data);
    cudaFree(d_payload); cudaFree(d_fnv);

    std::printf("%s\n", ok ? "PLE_GATHER_TEST PASS" : "PLE_GATHER_TEST FAIL");
    return ok ? 0 : 1;
}

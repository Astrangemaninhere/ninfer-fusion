// N3 / S28 round 1: env-gated sidecar apply for the KV row-scale constant.
// NINFER_KV_ROWSCALE=<path> (unset = feature off, nothing happens). Any
// validation failure HARD-FAILS (throws) -- no silent fallback to the baked
// table, because a silently-ignored sidecar is indistinguishable from success.
#include "ops/kernel/gqa_isoquant_row_scale_loader.h"
#include "ops/kernel/gqa_isoquant_row_scale.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

bool kv_rowscale_sidecar_apply_from_env(std::uint32_t model_layers,
                                        std::uint32_t model_kv_heads,
                                        std::uint32_t model_head_dim,
                                        std::uint64_t model_hash) {
    const char* path = std::getenv("NINFER_KV_ROWSCALE");
    if (path == nullptr || *path == '\0') { return false; }
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        throw std::runtime_error(std::string("KVRS open: ") + path);
    }
    std::string blob;
    char buf[65536];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) { blob.append(buf, n); }
    std::fclose(f);
    KvRowScaleSidecar sc;
    std::string err;
    if (!kv_rowscale_sidecar_parse(blob, sc, err) ||
        !kv_rowscale_sidecar_check(sc, model_layers, model_kv_heads,
                                   model_head_dim, model_hash, err)) {
        throw std::runtime_error("KVRS " + err + " (" + path + ")");
    }
    // Both capacities are the same object viewed from host and device; this is
    // the only translation unit that sees both headers, so bind them here.
    static_assert(kKvRowScalePoolCapacity ==
                      static_cast<std::uint32_t>(kKvRowScalePoolWords),
                  "KVRS pool capacity differs between loader.h and "
                  "gqa_isoquant_row_scale.cuh");
    // Payload first, descriptor last: the descriptor is the commit point, so a
    // half-applied update can only ever be observed as the OLD geometry.
    const cudaError_t e = cudaMemcpyToSymbol(kGqaKvRowScalePool, sc.words.data(),
                                             sc.words.size() * sizeof(unsigned short));
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("KVRS upload: ") + cudaGetErrorString(e));
    }
    const int geom[4] = {static_cast<int>(sc.layers), static_cast<int>(sc.kv_heads),
                         static_cast<int>(sc.head_dim),
                         static_cast<int>(sc.words.size())};
    const cudaError_t eg = cudaMemcpyToSymbol(kGqaKvRowScaleGeom, geom, sizeof geom);
    if (eg != cudaSuccess) {
        throw std::runtime_error(std::string("KVRS geom upload: ") +
                                 cudaGetErrorString(eg));
    }
    std::fprintf(stderr,
                 "[kvrs] applied %s: layers=%u kv_heads=%u head_dim=%u words=%u "
                 "pool=%u identity=%u crc=%08x hash=%016llx tag=%s\n",
                 path, sc.layers, sc.kv_heads, sc.head_dim,
                 static_cast<unsigned>(sc.words.size()), kKvRowScalePoolCapacity,
                 (sc.flags & kKvRowScaleFlagIdentity) ? 1u : 0u, sc.crc,
                 static_cast<unsigned long long>(sc.model_hash), sc.tag.c_str());
    return true;
}

}  // namespace ninfer::ops

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
    const char* spec = std::getenv("NINFER_KV_ROWSCALE");
    if (spec == nullptr) { return false; }
    return kv_rowscale_sidecar_apply_spec(std::string(spec), model_layers,
                                          model_kv_heads, model_head_dim,
                                          model_hash);
}

bool kv_rowscale_sidecar_apply_spec(const std::string& spec,
                                    std::uint32_t model_layers,
                                    std::uint32_t model_kv_heads,
                                    std::uint32_t model_head_dim,
                                    std::uint64_t model_hash) {
    KvRowScaleMode mode = KvRowScaleMode::Auto;
    std::string path;
    std::string mode_err;
    if (!kv_rowscale_mode_from_spec(spec, mode, path, mode_err)) {
        throw std::runtime_error("KVRS " + mode_err);
    }
    if (mode == KvRowScaleMode::Auto) {
        // Back to the baked calibration geometry. This is a RESTORE, not a
        // no-op, and it is the whole reason Auto is explicit: the descriptor is
        // process-global device state, so an engine built with "off" and a
        // later engine built with "auto" in the same process must not inherit
        // the zeroed descriptor. The bytes written here are the ones the baked
        // initializer holds (kKvRowScaleBakedGeom), so the on-path device
        // behaviour is unchanged; the pool itself is never touched, so a baked
        // payload that a previous sidecar overwrote is NOT restored (documented
        // hazard: build at most one row-scale mode per process, or set the mode
        // before the first engine).
        const int geom[4] = {kKvRowScaleBakedGeom[0], kKvRowScaleBakedGeom[1],
                             kKvRowScaleBakedGeom[2], kKvRowScalePoolWords};
        const cudaError_t ea = cudaMemcpyToSymbol(kGqaKvRowScaleGeom, geom, sizeof geom);
        if (ea != cudaSuccess) {
            throw std::runtime_error(std::string("KVRS auto: ") + cudaGetErrorString(ea));
        }
        return false;
    }
    if (mode == KvRowScaleMode::Off) {
        // Off is the accessor's OWN identity path: zero the geometry so every
        // range check in gqa_kv_row_scale() fails and it answers 1.0 for every
        // (layer, kv_head, d). No kernel change, no instruction added, and no
        // all-ones sidecar file to generate. The descriptor is the commit
        // point, exactly like the sidecar path below.
        const int geom[4] = {0, 0, 0, 0};
        const cudaError_t eg = cudaMemcpyToSymbol(kGqaKvRowScaleGeom, geom, sizeof geom);
        if (eg != cudaSuccess) {
            throw std::runtime_error(std::string("KVRS off: ") + cudaGetErrorString(eg));
        }
        std::fprintf(stderr,
                     "[kvrs] row scale OFF: geometry zeroed, gqa_kv_row_scale() "
                     "answers 1.0 for every channel (K*s == K, Q/s == Q)\n");
        return true;
    }
    const char* cpath = path.c_str();
    std::FILE* f = std::fopen(cpath, "rb");
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
                 path.c_str(), sc.layers, sc.kv_heads, sc.head_dim,
                 static_cast<unsigned>(sc.words.size()), kKvRowScalePoolCapacity,
                 (sc.flags & kKvRowScaleFlagIdentity) ? 1u : 0u, sc.crc,
                 static_cast<unsigned long long>(sc.model_hash), sc.tag.c_str());
    return true;
}

}  // namespace ninfer::ops

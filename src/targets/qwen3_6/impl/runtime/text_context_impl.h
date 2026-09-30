#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/nvtx.h"
#include "ops/stream_capture.h"   // F881: the one capture-predicate reader
// F738 injectchan: the ingress surface (src/spec/inject_channel.h + inject_ingress.h).
#include "targets/qwen3_6/impl/runtime/inject_ingress.h"
#include "targets/qwen3_6/impl/runtime/visual_scatter.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/mtp_proposal_topk.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "targets/qwen3_6/impl/runtime/kv_calibration.h"
#include <mutex>
namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {
// ---------------------------------------------------------------------------------------------
// THE TWO DOUBLE-NORM LEAVES, SELECTED BY THE ARCH'S OWN DECLARATION (accfix F910).
//
// PRE, MEASURED AND NOT ASSUMED: `ModelConfig::attn_out_double_norm()` and
// `ModelConfig::mlp_out_double_norm()` were DEFINED (text_context.h :285/:288) and CALLED FROM
// NOWHERE; `MlpW::post_ff_norm` and `FullLayerW::post_attn_out_norm` were declared and NEVER
// ASSIGNED; `Variant::post_mixer_double_norm` was declared in every variant.h and CALLED FROM
// NOWHERE. muse_glimmer_30b, whose loader binds `post_attention_layernorm` and
// `post_feedforward_layernorm` (impl/load/bindings.cpp, the "Double-norm layer graph" blocks) and
// whose config.h declares `post_norm_eps = 1e-08F`, therefore took the single-norm leaf and the
// TWO BOUND TENSORS WERE NEVER APPLIED -- rc=0 and a wrong answer, with nothing printed. That is
// the same class as the gate's silent default: a declaration the tree carries and no reader.
//
// WHY THESE ARE TEMPLATES. The mode is a compile-time property of the arch, but in NON-templated
// code the discarded branch of an `if constexpr` is still fully checked, so naming
// `Variant::attention_output_projection_double_norm` here would be a name-lookup error for every
// arch whose variant.h does not declare it. As templates the branch is instantiated ONLY where the
// arch turns the mode on -- and an arch that turns it on WITHOUT declaring the leaf gets a COMPILE
// ERROR BY NAME, which is the point: the stub cannot come back.
//
// AND THE DEREFERENCE IS GUARDED: an arch that declares the mode while its loader bound no tensor
// is REFUSED BY NAME, not handed an empty Tensor.
template <class VariantT>
void attn_output_projection_leaf(const Tensor& attention, const Weight& weight,
                                 const Tensor* post_attn_out_norm, Tensor& residual,
                                 qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                                 cudaStream_t stream) {
    if constexpr (ModelConfig::attn_out_double_norm()) {
        if (post_attn_out_norm == nullptr || post_attn_out_norm->data == nullptr) {
            throw std::logic_error(
                "attn output projection: this arch declares attn_out_post_norm() but its loader "
                "bound no post-attention output norm (FullAttentionWeights::post_attn_out_norm is "
                "empty), so the double-norm layer graph cannot be executed. Binding it and "
                "declaring it are the same fact stated twice; make them agree rather than running "
                "the single-norm leaf on a double-norm model.");
        }
        VariantT::attention_output_projection_double_norm(attention, weight, *post_attn_out_norm,
                                                          residual, phase, workspace, stream);
    } else {
        VariantT::attention_output_projection(attention, weight, residual, phase, workspace, stream);
    }
}

template <class VariantT>
void post_mixer_leaf(const Tensor& hidden, const MlpW& weights, Tensor& residual,
                     qwen3_6::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream) {
    if constexpr (ModelConfig::mlp_out_double_norm()) {
        if (weights.post_ff_norm == nullptr || weights.post_ff_norm->data == nullptr) {
            throw std::logic_error(
                "post mixer: this arch declares mlp_out_post_norm() but its loader bound no MLP "
                "output norm (MlpW::post_ff_norm is empty), so the double-norm layer graph cannot "
                "be executed. Binding it and declaring it are the same fact stated twice; make "
                "them agree rather than running the single-norm leaf on a double-norm model.");
        }
        VariantT::post_mixer_double_norm(hidden, *weights.payload, *weights.post_ff_norm, residual,
                                         phase, workspace, stream);
    } else {
        VariantT::post_mixer(hidden, *weights.payload, residual, phase, workspace, stream);
    }
}


// NINFER_HEADDBG=1 dumps the first BF16 values of a decode-head tensor so a degenerate (all-zero)
// hidden or logits is visible without a debugger (_TODO.md 102: Muse decode emits token 0).
inline bool head_debug_enabled() {
    static const bool enabled = std::getenv("NINFER_HEADDBG") != nullptr;
    return enabled;
}

// ⚠️ PATCH A2 (diagnostic, scratch/PATCHSET/A2_headdbg_columns.diff).
// `column` selects WHICH column of a [.., columns] tensor is sampled. It MUST be
// applied with the tensor's own byte stride `nb[1]`, never as `column * ne[0] * 2`:
// `x` is an arena sub-view, so the byte distance between two columns is a property
// of the view and only `nb[1]` carries it.
//
// WHY THIS CHANGES BEHAVIOUR: the pre-patch probe read the FIRST 8 BF16 values
// contiguously, which is COLUMN 0 ONLY of a [hidden, columns] tensor. The measured
// argmax flip is in column 1 and later columns (round 0 column 1: 19670->436;
// round 2 column 3: 387->279), so the old probe was BLIND to the very phenomenon
// it was being used to explain.
inline void debug_head_probe(cudaStream_t stream, const Tensor& tensor, const char* label,
                             std::int32_t column = 0) {
    if (!head_debug_enabled() || tensor.data == nullptr || tensor.dtype != DType::BF16) { return; }
    if (column < 0 || tensor.ne[1] <= column || tensor.nb[1] <= 0) { return; }
    const std::size_t count = std::min<std::size_t>(8, tensor.bytes() / sizeof(__nv_bfloat16));
    if (count == 0) { return; }
    const char* source = static_cast<const char*>(tensor.data) +
                         static_cast<std::int64_t>(column) * tensor.nb[1];
    // F881 -- GUARD (same class as the `[accmask]` readback). This probe is a D2H plus a sync on
    // the round's own stream, gated only by NINFER_HEAD_DEBUG, so with CUDA graphs on it would
    // invalidate the capture and abort the process. Skipping it is named, once.
    if (ninfer::ops::stream_is_capturing(stream)) {
        static bool headdbg_capture_warned = false;
        if (!headdbg_capture_warned) {
            headdbg_capture_warned = true;
            std::fprintf(stderr, "[headdbg] probe skipped: stream capture in flight\n");
        }
        return;
    }
    __nv_bfloat16 host[8] = {};
    if (cudaMemcpyAsync(host, source, count * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        (void)cudaGetLastError();
        return;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        (void)cudaGetLastError();
        return;
    }
    // Widened from %-14s: the label now carries layer/stage/phase/width/batch/column/pos.
    std::fprintf(stderr, "[headdbg] %-34s", label);
    bool any_nan = false;
    for (std::size_t i = 0; i < count; ++i) {
        const float value = __bfloat162float(host[i]);
        if (value != value) { any_nan = true; }
        std::fprintf(stderr, " %.6g", static_cast<double>(value));
    }
    std::fprintf(stderr, "%s\n", any_nan ? "  <NAN>" : "");
}

// KV forensics (NINFER_KVDUMP_DIR): dump raw K/V planes and the exact BF16 K/V
// the append path writes, so host tools can compare "written" against "stored"
// without a debugger (_TODO.md 104: Muse layer 16 early-position KV reads NaN).
inline const char* kvdump_dir() {
    static const char* dir = std::getenv("NINFER_KVDUMP_DIR");
    return (dir != nullptr && *dir != '\0') ? dir : nullptr;
}

inline bool kvdump_layer_enabled(const char* env_name, int layer) {
    static const char* spec = std::getenv(env_name);
    if (spec == nullptr || *spec == '\0') { return false; }
    if (std::strcmp(spec, "all") == 0) { return true; }
    const std::string list(spec);
    std::size_t pos = 0;
    while (pos <= list.size()) {
        const std::size_t comma = list.find(',', pos);
        const std::string token =
            list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!token.empty() && std::atoi(token.c_str()) == layer) { return true; }
        if (comma == std::string::npos) { break; }
        pos = comma + 1;
    }
    return false;
}

// N3/S38 round 2: offline KV calibration capture wiring.
// KvCalibrationCapture was dead code: the EngineOptions field it documented
// has since been removed, so the env var IS the gate. Env-gated like the S28
// loader: NINFER_KV_CALIB_DIR
// unset = one cached getenv = zero cost. The capture COPIES (ordered
// async D2H on the producing stream) and never writes the tensors, so
// numerics are untouched; per-layer token cap is enforced inside the
// capture (NINFER_KV_CALIB_MAX_TOKENS, default 4096).
inline bool kvcalib_enabled() {
    static const char* dir = std::getenv("NINFER_KV_CALIB_DIR");
    return dir != nullptr && *dir != '\0';
}

inline void kvcalib_capture(std::uint32_t full_layer, const Tensor& k,
                             const Tensor& v, const Tensor& positions,
                             cudaStream_t stream) {
    static KvCalibrationCapture* capture = [] {
        const char* dir = std::getenv("NINFER_KV_CALIB_DIR");
        return dir != nullptr && *dir != '\0'
                   ? new KvCalibrationCapture(dir)
                   : nullptr;
    }();
    static std::mutex mutex;
    if (capture == nullptr) { return; }
    std::lock_guard<std::mutex> lock(mutex);
    capture->capture(full_layer, k, v, positions, stream);
}

// =====================================================================================
// layerrec probe -- GDN-branch intermediate-tensor dump (READ-ONLY).
//
// WHY IT EXISTS: the attention branch of `attn_mix` has `mix_probe` and the GDN branch of
// `gdn_mix` has NO probe at all.  That asymmetry is why "the artifact side is already
// consistent / the engine side still disagrees" could not be split on the GDN path
// (registered by dl/gdntrace/REPORT.md section 7 item 2).
//
// GATES (each one a cached getenv, so an unset environment costs one branch):
//   NINFER_GDNDUMP_DIR=<dir>          destination directory; unset == probe completely off
//   NINFER_GDNDUMP_LAYERS=<i,j|all>   full-layer filter; default all
//   NINFER_GDNDUMP_MAX_CALLS=<n>      stop after n gdn_mix calls; default 0 == no cap
//   NINFER_GDNDUMP_MAX_ELEMS=<n>      per-tensor element cap; default 2097152
//
// READ-ONLY, by construction: the probe only ever (a) reads member fields and the integer
// position selectors, and (b) issues an ORDERED async D2H copy on the producing stream and
// writes those bytes to a file.  It never writes a Tensor, never launches a kernel, and
// never throws: every failure mode (absent data, non-contiguous view, D2H error, fopen
// error, over the element cap) becomes a `note` line in the meta file and the run goes on.
// The paired arms `a1_red_nograph` (gates off) and `a2_red_dump` (gates on) are the same
// binary under the same flags, so their token streams must be identical.
//
// It must be run with `--no-cuda-graph`: the copy is issued from inside the decoded layer
// loop, and an illegal D2H during stream capture is why the tree's own offline
// instruments (NINFER_KVDUMP_DIR / NINFER_KV_CALIB_DIR) document that flag.
// =====================================================================================
inline const char* gdndump_dir() {
    static const char* dir = std::getenv("NINFER_GDNDUMP_DIR");
    return (dir != nullptr && *dir != '\0') ? dir : nullptr;
}

inline bool gdndump_layer_enabled(int full_layer) {
    static const char* spec = std::getenv("NINFER_GDNDUMP_LAYERS");
    if (spec == nullptr || *spec == '\0' || std::strcmp(spec, "all") == 0) { return true; }
    const std::string list(spec);
    std::size_t pos = 0;
    while (pos <= list.size()) {
        const std::size_t comma = list.find(',', pos);
        const std::string token =
            list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!token.empty() && std::atoi(token.c_str()) == full_layer) { return true; }
        if (comma == std::string::npos) { break; }
        pos = comma + 1;
    }
    return false;
}

// Writes raw little-endian bytes: the meta line carries ne/nb/dtype so the reader never
// has to guess.  Never throws -- see the READ-ONLY note above.
inline void gdndump_one(cudaStream_t stream, const Tensor& tensor, const std::string& path,
                        std::FILE* meta, const char* name) {
    if (meta == nullptr) { return; }
    if (tensor.data == nullptr || tensor.numel() <= 0) {
        std::fprintf(meta, "note %s absent\n", name);
        return;
    }
    static const char* cap_env = std::getenv("NINFER_GDNDUMP_MAX_ELEMS");
    static const std::size_t cap =
        (cap_env != nullptr && *cap_env != '\0')
            ? static_cast<std::size_t>(std::strtoull(cap_env, nullptr, 10))
            : static_cast<std::size_t>(2097152);
    const std::size_t elems = static_cast<std::size_t>(tensor.numel());
    if (elems > cap) {
        std::fprintf(meta, "note %s over cap elems=%zu cap=%zu\n", name, elems, cap);
        return;
    }
    if (!tensor.is_contiguous()) {
        std::fprintf(meta, "note %s not contiguous\n", name);
        return;
    }
    const std::size_t bytes = static_cast<std::size_t>(tensor.bytes());
    std::fprintf(meta,
                 "tensor %s file=%s ne=%d,%d,%d,%d nb=%lld,%lld,%lld,%lld dtype=%d bytes=%zu\n",
                 name, path.c_str(), tensor.ne[0], tensor.ne[1], tensor.ne[2], tensor.ne[3],
                 static_cast<long long>(tensor.nb[0]), static_cast<long long>(tensor.nb[1]),
                 static_cast<long long>(tensor.nb[2]), static_cast<long long>(tensor.nb[3]),
                 static_cast<int>(tensor.dtype), bytes);
    std::fflush(meta);
    // F881 -- GUARD. The gdndump writer's D2H + sync, on the round's own stream; the note goes
    // into the dump's own meta file so the MISSING dump is recorded where the dump would have been.
    if (ninfer::ops::stream_is_capturing(stream)) {
        static bool gdndump_capture_warned = false;
        if (!gdndump_capture_warned) {
            gdndump_capture_warned = true;
            std::fprintf(meta, "note %s dump skipped: stream capture in flight\n", name);
            std::fflush(meta);
        }
        return;
    }
    std::vector<std::byte> host(bytes);
    if (cudaMemcpyAsync(host.data(), tensor.data, bytes, cudaMemcpyDeviceToHost, stream) !=
        cudaSuccess) {
        (void)cudaGetLastError();
        std::fprintf(meta, "note %s d2h failed\n", name);
        std::fflush(meta);
        return;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        (void)cudaGetLastError();
        std::fprintf(meta, "note %s sync failed\n", name);
        std::fflush(meta);
        return;
    }
    std::FILE* out = std::fopen(path.c_str(), "wb");
    if (out == nullptr) {
        std::fprintf(meta, "note %s fopen failed\n", name);
        std::fflush(meta);
        return;
    }
    std::fwrite(host.data(), 1, bytes, out);
    std::fclose(out);
}

// One instance per gdn_mix call.  The full-layer index is recovered with the same
// predicate `run_layers` uses (ModelConfig::is_full / gdn_idx), so a probe row can be
// joined to the layer numbers in every other report on this box.
struct GdnDump {
    std::FILE* meta       = nullptr;
    std::uint32_t call    = 0;
    int layer             = -1;
    int gidx              = -1;
    cudaStream_t stream   = nullptr;

    GdnDump(int gdn_index, bool verify_phase, cudaStream_t s) : gidx(gdn_index), stream(s) {
        if (!verify_phase || gdndump_dir() == nullptr) { return; }
        for (int L = 0; L < kCfg.n_layers; ++L) {
            if (!ModelConfig::is_full(L) && ModelConfig::gdn_idx(L) == gdn_index) {
                layer = L;
                break;
            }
        }
        if (layer < 0 || !gdndump_layer_enabled(layer)) { return; }
        static std::atomic<std::uint32_t> counter{0};
        static const char* max_env = std::getenv("NINFER_GDNDUMP_MAX_CALLS");
        static const std::uint32_t max_calls =
            (max_env != nullptr && *max_env != '\0')
                ? static_cast<std::uint32_t>(std::strtoul(max_env, nullptr, 10))
                : 0u;
        call = counter.fetch_add(1);
        if (max_calls != 0 && call >= max_calls) { return; }
        meta = std::fopen((std::string(gdndump_dir()) + "/gdn_c" + std::to_string(call) + "_L" +
                           std::to_string(layer) + "_meta.txt")
                              .c_str(),
                          "w");
        if (meta != nullptr) {
            std::fprintf(meta, "call=%u full_layer=%d gidx=%d phase=verify threads=%d\n", call,
                         layer, gidx, 1);
            std::fflush(meta);
        }
    }

    ~GdnDump() {
        if (meta != nullptr) { std::fclose(meta); }
    }

    GdnDump(const GdnDump&)            = delete;
    GdnDump& operator=(const GdnDump&) = delete;

    void dump(const char* stage, const Tensor& tensor) {
        if (meta == nullptr) { return; }
        gdndump_one(stream, tensor,
                    std::string(gdndump_dir()) + "/gdn_c" + std::to_string(call) + "_L" +
                        std::to_string(layer) + "_" + stage + ".bin",
                    meta, stage);
    }
};

inline void kvdump_write_file(const std::string& path, const void* data, std::size_t bytes) {
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) { throw std::runtime_error("kvdump cannot open " + path); }
    std::fwrite(data, 1, bytes, file);
    std::fclose(file);
}

// max_dim3 > 0 caps the physical-page extent (planes are {lead, 64, heads, pages}).
inline void kvdump_dump_tensor(cudaStream_t stream, const Tensor& tensor, int max_dim3,
                               const std::string& path) {
    if (tensor.data == nullptr || tensor.numel() == 0) { return; }
    Tensor view = tensor;
    if (max_dim3 > 0 && tensor.ne[3] > max_dim3) { view = tensor.slice(3, 0, max_dim3); }
    if (!view.is_contiguous()) { throw std::runtime_error("kvdump tensor is not contiguous: " + path); }
    // F881 -- GUARD. `kvdump_dump_tensor` is a pure dump helper; under capture it returns and
    // says so rather than invalidating the capture and aborting at the next CUDA_CHECK.
    if (ninfer::ops::stream_is_capturing(stream)) {
        static bool kvdump_capture_warned = false;
        if (!kvdump_capture_warned) {
            kvdump_capture_warned = true;
            std::fprintf(stderr, "[kvdump] %s skipped: stream capture in flight\n", path.c_str());
        }
        return;
    }
    std::vector<std::byte> host(view.bytes());
    CUDA_CHECK(cudaMemcpyAsync(host.data(), view.data, host.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    kvdump_write_file(path, host.data(), host.size());
}

inline void kvdump_describe(std::FILE* file, const char* name, const Tensor& tensor) {
    std::fprintf(file, "%s ne=%d,%d,%d,%d nb=%lld,%lld,%lld,%lld dtype=%d\n", name, tensor.ne[0],
                 tensor.ne[1], tensor.ne[2], tensor.ne[3],
                 static_cast<long long>(tensor.nb[0]), static_cast<long long>(tensor.nb[1]),
                 static_cast<long long>(tensor.nb[2]), static_cast<long long>(tensor.nb[3]),
                 static_cast<int>(tensor.dtype));
}

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }

    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;

    ~ScopedPositions() { slot_ = nullptr; }

private:
    const Tensor*& slot_;
};

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::GqaExecutionEnvelope*& slot,
                   const ops::GqaExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }

    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;

    ~ScopedEnvelope() { slot_ = nullptr; }

private:
    const ops::GqaExecutionEnvelope*& slot_;
};

template <class T>
class ScopedValue {
public:
    ScopedValue(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

} // namespace

void DFlashFeatureSink::begin(const Tensor& value) {
    const bool prefill = features != nullptr && positions != nullptr && batch_features == nullptr;
    const bool batch   = batch_features != nullptr && batch_lanes != nullptr &&
                       batch_valid_columns != nullptr && batch_width > 0 && batch_size > 0;
    if ((!prefill && !batch) || layers.empty()) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination     = batch_features != nullptr ? batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features != nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features->slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > features->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(features->nb[1]);
    auto* target                    = static_cast<std::byte*>(features->data) + index * width_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    const std::uint32_t complete_mask = layers.size() == 32 ? ~0U : ((1U << layers.size()) - 1U);
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (batch_features != nullptr) {
        if (source.dtype != DType::I32 || source.ne[0] != batch_width ||
            source.ne[1] != batch_size) {
            throw std::logic_error("DFlash batch feature positions are invalid");
        }
        return;
    }
    if (active_tokens <= 0 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
        positions == nullptr || active_tokens > positions->ne[0]) {
        throw std::logic_error("DFlash feature positions are invalid");
    }
    CUDA_CHECK(cudaMemcpyAsync(positions->data, source.data,
                               static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void DFlashFeatureSink::consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (!consume_prefill || tokens != active_tokens) {
        throw std::logic_error("DFlash prefill feature consumer is unavailable");
    }
    Tensor feature_window  = features->slice(1, 0, tokens);
    Tensor position_window = positions->slice(0, 0, tokens);
    consume_prefill(feature_window, position_window, rewrite_checkpoint);
}

TextContext::TextContext(DeviceContext& ctx, const LoadedModelData& weights, WorkspaceArena& work,
                         qwen3_6::PagedKVCacheView kv, LinearAttentionStatePool& state,
                         qwen3_6::RoundState& io, Tensor& prefill_hidden,
                         std::uint32_t prefill_chunk, std::uint32_t text_kv_base,
                         qwen3_6::PagedKVCacheView mtp_kv,
                         const qwen3_6::PagedKVCache* batch_text_kv,
                         const qwen3_6::PagedKVCache* batch_mtp_kv)
    : ctx_(ctx), weights_(weights), work_(work), kv_(kv), mtp_kv_(mtp_kv), state_(state), io_(io),
      prefill_hidden_(prefill_hidden), prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base),
      batch_text_kv_(batch_text_kv), batch_mtp_kv_(batch_mtp_kv) {
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    if (mtp_enabled() && !io_.mtp_decode && !io_.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    set_linear_state_slots(0, 0);
    bind();
}

TextContext::~TextContext() = default;

void TextContext::set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot) {
    if (source_slot < 0 || source_slot >= state_.slot_count() || destination_slot < 0 ||
        destination_slot >= state_.slot_count()) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    linear_state_source_slot_      = source_slot;
    linear_state_destination_slot_ = destination_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action_ = action;
    replay_records_   = replay_records;
}

void TextContext::bind() {
    using TargetBindings = LoadedModelData;
    using TargetMlp      = MlpWeights;
    // F738 injectchan: read the ingress declaration ONCE, here, because this is the
    // first point at which both numbers the admission needs are known -- the model's
    // hidden width (a property of the config) and the context capacity (a property of
    // the KV plan). A refused declaration throws HERE, before a token is consumed.
    inject_ingress::configure({ModelConfig::hidden,
                               static_cast<std::int32_t>(kv_.max_context()),
                               spec::inject::ElementType::Bf16});
    const auto bind_mlp  = [](const TargetMlp& source) { return MlpW{&source}; };

    embed_      = &weights_.token_embedding;
    final_norm_ = &weights_.final_norm;
    lm_head_    = &weights_.output_head;
    if (weights_.optimized_proposal) {
        const auto& proposal = *weights_.optimized_proposal;
        set_proposal_head(&proposal.head, static_cast<const std::int32_t*>(proposal.token_ids.data),
                          proposal.head.n);
    }

    if (mtp_enabled()) {
        if (!weights_.mtp) {
            throw std::invalid_argument("MTP state was enabled without materialized MTP weights");
        }
        const auto& source = *weights_.mtp;
        mtp_               = MtpW{&source,
                    &source.input_projection,
                    &source.embedding_norm,
                    &source.hidden_norm,
                    &source.input_norm,
                    &source.query_norm,
                    &source.key_norm,
                    &source.output,
                    &source.post_attention_norm,
                    &source.final_norm};
    }

    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            FullLayerW& out = full_[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            const auto& source =
                weights_.full_layers[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            out.input_norm     = &source.input_norm;
            out.projection     = &source.projection;
            out.o_proj         = &source.output;
            out.q_norm         = &source.query_norm;
            out.k_norm         = &source.key_norm;
            out.post_attn_norm = &source.post_attention_norm;
            // F910: THE TWO TENSORS THE ARCH'S LOADER BINDS AND NOTHING ASSIGNED.
            // `FullAttentionWeights` (the SHARED export template, targets/qwen3_6/export/...
            // model_view.h) carries BOTH `post_attn_out_norm` and `post_mlp_out_norm`, and this
            // function -- the one place a FullLayerW is filled -- assigned neither pointer, so the
            // double-norm leaf could never have received them even if it had been called. Both
            // assignments are UNCONDITIONAL on purpose: the fields exist for every arch, and the
            // READER is gated by the arch's own declaration, so a single-norm arch that binds
            // neither tensor is unaffected (its pointers point at empty Tensors and
            // `post_mixer_leaf` never dereferences them). The refusal for "declared but not bound"
            // lives in the leaf selector, by name.
            out.post_attn_out_norm = &source.post_attn_out_norm;
            out.mlp            = bind_mlp(source.post_mixer);
            out.mlp.post_ff_norm = &source.post_mlp_out_norm;
        } else {
            const std::size_t gidx = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
            GdnLayerW& out         = gdn_[gidx];
            const auto& source     = weights_.gdn_layers[gidx];
            out.input_norm         = &source.input_norm;
            out.projection         = &source.projection;
            out.conv1d             = &source.convolution;
            out.gdn_norm           = &source.norm;
            out.out_proj           = &source.output;
            out.post_attn_norm     = &source.post_attention_norm;
            out.mlp                = bind_mlp(source.post_mixer);
        }
    }
}

const MtpW& TextContext::mtp_weights() const {
    if (!mtp_enabled()) { throw std::runtime_error("MTP draft weights are not enabled"); }
    return mtp_;
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({kCfg.hidden, T});

    auto roots = workspace_recipe::mtp_stem<TextConfig>(work_, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 || input_embeddings->ne[0] != kCfg.hidden ||
            input_embeddings->numel() != static_cast<std::int64_t>(kCfg.hidden) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({kCfg.hidden, T});
    } else {
        emb = roots.embedding;
        ops::embedding(flat_ids, *embed_, emb, s);
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, *mtp_.pre_fc_norm_embedding, kCfg.rms_eps, true, e, s);
    ops::rmsnorm(flat_hidden, *mtp_.pre_fc_norm_hidden, kCfg.rms_eps, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    ops::linear(fc_in, *mtp_.fc, x, s);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, *mtp_.input_norm, kCfg.rms_eps, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace_recipe::mtp_attention_projection<TextConfig>(work_, T);
    Tensor q              = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k              = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor gate           = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor v              = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat         = q.view({kCfg.q_size, T});
    Tensor gate_flat      = gate.view({kCfg.q_size, T});
    Tensor k_flat         = k.view({kCfg.kv_size, T});
    Tensor v_flat         = v.view({kCfg.kv_size, T});
    Variant::mtp_attention_projection(ah, mtp_.payload->attention, q_flat, gate_flat, k_flat,
                                      v_flat, work_, s);

    const auto results = workspace_recipe::mtp_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    if (ctx_.yarn_enabled) {
        ops::rope_yarn4(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);
    } else {
        ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);
    }

    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        ops::gqa_attention(
            q_batch, k_batch, v_batch, position_batch, *active_valid_columns_, Tensor{},
            *active_backend_kv_table_rows_, kAttnScale,
            batch_mtp_kv_->batch_layer_view(0), envelope, work_, a_batch, s);
    } else {
        ops::gqa_attention(qn, kn, v, positions, Tensor{}, Tensor{}, io_.backend_kv_table_row,
                                      kAttnScale,
                                      batch_mtp_kv_->batch_layer_view(0), envelope, work_, a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    const auto post = workspace_recipe::mtp_post_attention<TextConfig>(work_, T);
    Tensor o        = post.output;
    ops::linear(a.view({kCfg.q_size, T}), *mtp_.o_proj, o, s);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x, work_, s);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({kCfg.hidden, T});
    ops::rmsnorm(x, *mtp_.norm, kCfg.rms_eps, true, flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    nvtx::ScopedRange forward_range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                                    static_cast<std::uint64_t>(ids.numel()));
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::GqaExecutionEnvelope envelope,
                                    bool final_chunk, Tensor* final_hidden, Tensor* logits,
                                    Tensor* draft_token) {
    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {kCfg.hidden, 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ah_last = work_.alloc(DType::BF16, {kCfg.hidden, 1});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Tensor v_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Variant::mtp_kv_projection(ah, mtp_.payload->attention, k_flat, v_flat, work_, s);
        Tensor k  = k_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor v  = v_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
        ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
        if (ctx_.yarn_enabled) {
        ops::rope_yarn4(rope_positions, kCfg.rotary_dim, kCfg.rope_theta, kn, s);
    } else {
        ops::rope(rope_positions, kCfg.rotary_dim, kCfg.rope_theta, kn, s);
    }
        ops::gqa_kv_append(kn, v, positions, mtp_kv_.layer_view(0), s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(kCfg.hidden) * dtype_size(DType::BF16);
            const auto* x_src = static_cast<const unsigned char*>(x.data) +
                                static_cast<std::size_t>(T - 1) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            CUDA_CHECK(
                cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
        }
    }

    if (final_chunk) {
        Tensor q_flat    = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        Tensor gate_flat = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        Variant::mtp_q_gate_projection(ah_last, mtp_.payload->attention, q_flat, gate_flat, work_,
                                       s);
        Tensor q    = q_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor gate = gate_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor qn   = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, T - 1, 1);
        } else {
            last_rope_position = work_.alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions.data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst = static_cast<std::int32_t*>(last_rope_position.data) + axis;
                CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
        }
        if (ctx_.yarn_enabled) {
        ops::rope_yarn4(last_rope_position, kCfg.rotary_dim, kCfg.rope_theta, qn, s);
    } else {
        ops::rope(last_rope_position, kCfg.rotary_dim, kCfg.rope_theta, qn, s);
    }

        Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::gqa_attention_cached(qn, last_position,
                                             kAttnScale,
                                             mtp_kv_.layer_view(0), envelope, work_, a, s);
        ops::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::linear(a.view({kCfg.q_size, 1}), *mtp_.o_proj, o, s);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::rmsnorm(x_last, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);
        {
            auto post_mixer_scope = work_.scope();
            Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x_last, work_, s);
        }
        ops::rmsnorm(x_last, *mtp_.norm, kCfg.rms_eps, true, *final_hidden, s);
        proposal_argmax(*final_hidden, *logits, *draft_token);
    }
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens) {
    const int T = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, kCfg.vocab, T, "proposal logits");
    nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                     static_cast<std::uint64_t>(T));
    if (proposal_head_ != nullptr) {
        Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, T});
        ops::linear(hidden, *proposal_head_, proposal_logits, ctx_.stream);
        ops::argmax(proposal_logits, proposal_tokens, proposal_head_n_, ctx_.stream);
        ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                      ctx_.stream);
    } else {
        Tensor output_logits = matrix_window(logits, T);
        ops::linear(hidden, *lm_head_, output_logits, ctx_.stream);
        ops::argmax(output_logits, proposal_tokens, kCfg.token_domain, ctx_.stream);
    }
}

// --draft-tree L,d (L > 1): the per-depth top-L candidate rows of the proposal row, published
// beside the draft token that SAME row's argmax produced (proposal_argmax() just above). The two
// are one device source read two ways, and the host reader re-runs the check
// (detail::mtp_proposal_ready_steps compares lattice_ids[depth * kMtpTreeProposalDepthStride]
// against next_drafts[depth]) so a depth this function never wrote is refused instead of
// publishing a lattice no device output backs.
//
// It mirrors the decode loop's extraction exactly (mtp_impl.h, mtp_decode_batch_body): same op,
// same `rows` (kCfg.token_domain -- the very window proposal_argmax()'s argmax reads), same
// one-lane spelling (`tokens == 1`, because one prefill proposal proposes for one lane), same
// `top_l`, same per-depth block stride. `logits` is the contiguous [kCfg.vocab, 1] window
// matrix_window() handed the proposal, so rank i of the written block and the draft token both
// come out of one and the same logits row.
void TextContext::extract_mtp_proposal_lattice(Tensor& logits, std::uint32_t depth) {
    if (mtp_tree_paths_ <= 1 || depth >= mtp_tree_depth_ || !io_.mtp.has_value()) { return; }
    if (io_.mtp->lattice_ids.data == nullptr) {
        throw std::logic_error("MTP prefill lattice is not bound");
    }
    ops::mtp_proposal_topk(logits.data, kCfg.token_domain, 1,
                           static_cast<std::int32_t>(mtp_tree_paths_),
                           static_cast<std::int32_t*>(io_.mtp->lattice_ids.data) +
                               static_cast<std::size_t>(depth) * kMtpTreeProposalDepthStride,
                           ctx_.stream);
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions,
                                    ops::GqaExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, T}, "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position,
                                      ops::GqaExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                     nullptr);
    auto logits_scope = work_.scope();
    proposal_argmax(mtp_hidden, logits, draft_token);
}

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_source_slots,
                                        const Tensor& linear_state_destination_slots,
                                        ops::GqaExecutionEnvelope envelope,
                                        Tensor& hidden, Tensor& logits) {
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,16]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention source slots");
    require_tensor_shape(linear_state_destination_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention destination slots");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "ordinary decode logits");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> source_binding(active_linear_state_source_slots_,
                                                  &linear_state_source_slots);
        ScopedValue<const Tensor*> destination_binding(active_linear_state_destination_slots_,
                                                       &linear_state_destination_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);

        Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, batch});
        ops::embedding(ids, *embed_, x, stream);
        debug_head_probe(stream, x, "post_embed");
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        debug_head_probe(stream, x, "post_layers_x");
        ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, hidden, stream);
        debug_head_probe(stream, hidden, "final_hidden");
        ops::linear(hidden, *lm_head_, logits, stream);
        debug_head_probe(stream, logits, "logits");
    }
    work_.reset();
}

template <class Tap>
void TextContext::target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& column_masks,
                                           const Tensor& kv_table_rows,
                                           const Tensor& linear_state_source_slots,
                                           ops::GqaExecutionEnvelope envelope,
                                           Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                           Tap& tap) {
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verify batch ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verify batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verify batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "target verify batch valid columns");
    if (column_masks.data != nullptr) {
        require_tensor_shape(column_masks, DType::I64, {width, batch},
                             "target verify batch column masks");
    }
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "target verify batch KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "target verify batch Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, width, batch},
                         "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_source_slots_,
                                                 &linear_state_source_slots);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<const Tensor*> mask_binding(active_column_masks_, &column_masks);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);

        Tensor x        = work_.alloc(DType::BF16, {kCfg.hidden, columns});
        Tensor flat_ids = ids.view({columns});
        ops::embedding(flat_ids, *embed_, x, stream);
        if constexpr (Tap::enabled) { tap.begin(x); }
        run_layers(x, Phase::Verify, tap);
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({kCfg.hidden, columns});
        Tensor flat_logits = logits.view({kCfg.vocab, columns});
        Tensor flat_tokens = target_tokens.view({columns});
        ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, flat_hidden, stream);
        ops::linear(flat_hidden, *lm_head_, flat_logits, stream);
        // Contract (TextContext::apply_final_logit_policy): every lm_head logits
        // production site applies the architecture's final-logit policy. Verify was
        // the one site that did not: a compile-time no-op for the qwen family
        // (softcap 0, multiplier 1) but REQUIRED for Muse (softcap 20, multiplier
        // 0.196), where omitting it made the verifier's argmax differ from plain.
        kCfg.apply_final_logit_policy(flat_logits, stream);
        ops::argmax(flat_logits, flat_tokens, kCfg.token_domain, stream);
    }
    work_.reset();
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& column_masks, const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::GqaExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens) {
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, column_masks,
                             kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             tap);
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& column_masks, const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::GqaExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink& sink) {
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, column_masks,
                             kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             sink);
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::GqaExecutionEnvelope envelope,
                                           Tensor& mtp_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "MTP decode batch target hidden");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr);
}

#include <cstdio>
#include <cstdlib>

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);

}

void TextContext::attn_mix(const FullLayerW& w, Tensor& x, int fidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    if (active_causal_attention_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    const auto projection = workspace_recipe::text_attention_projection<TextConfig>(work_, T);
    Tensor h              = projection.hidden;
    ops::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, h, s);

    Tensor q         = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    // ---- (C) the attention-output gate: TWO legal shapes, chosen by the arch ----
    // A per-channel gate is query_size rows wide and keeps the {head_dim, n_q, T}
    // view it has always had, so ops::sigmoid_mul sees the same ne[] it always saw.
    // A HEADWISE gate must reach the op 2-D: sigmoid_mul routes headwise BY SHAPE
    // (src/ops/wrapper/sigmoid_mul.cpp:34-37, dispatched at :48) and the contract is
    // stated in src/ops/launcher/sigmoid_gate_mul.h:16-19 -- x is [head_dim, H, T],
    // gate is [H, T], one sigmoid per (head, token) over the head_dim axis.  A
    // {head_dim, n_q, T} view of a 16-row gate makes that test `256 == 16` == false,
    // so the pair falls to the per-element route, which ACCEPTS it and reads gate
    // rows 16..4095 -- numbers spark_x2_5_4b/impl/variant.cpp never writes.  The
    // shape agreement between this arch's two gate declarations is asserted at
    // namespace scope in text_context.h (see the static_assert after kCfg).
    Tensor gate      = ModelConfig::headwise_gate()
                           ? projection.gate.view({ModelConfig::gate_rows(), T})
                           : projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k         = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor v         = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat    = q.view({kCfg.q_size, T});
    Tensor gate_flat = gate.view({ModelConfig::gate_rows(), T});
    Tensor k_flat    = k.view({kCfg.kv_size, T});
    Tensor v_flat    = v.view({kCfg.kv_size, T});
    Variant::attention_projection(h, *w.projection, q_flat, gate_flat, k_flat, v_flat, ph, work_,
                                  s);
    // NINFER_HEADDBG=1: stage probes for the layers around Muse's first divergence (_TODO.md 103).
    const auto mix_probe = [&](const char* stage, const Tensor& tensor) {
        if (!head_debug_enabled() || (fidx != 15 && fidx != 16)) { return; }
        char label[32];
        std::snprintf(label, sizeof(label), "L%02d_%s", fidx, stage);
        debug_head_probe(s, tensor, label);
    };
    mix_probe("in_x", x);
    mix_probe("q", q);
    mix_probe("k", k);
    mix_probe("v", v);

    const auto results = workspace_recipe::text_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    // ---- (A) q/k rmsnorm: the arch decides whether its checkpoint has one ----
    // NOT an all-ones stand-in when it does not: rmsnorm divides by the RMS whatever
    // the weight is, and a checkpoint that normalises NEITHER would have a
    // normalisation applied to it.  On the skip branch the tensors the rope/attention
    // path consumes ARE q and k -- ops::rope is in-place on its q/k arguments
    // (src/ops/wrapper/rope.cpp:120-125), so aliasing them is well-defined -- and
    // `w.q_norm` / `w.k_norm`, whose addresses FullLayerW holds unconditionally at
    // text_context_impl.h:593-594, are never dereferenced.
    if constexpr (ModelConfig::qk_norm()) {
        ops::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, qn, s);
        ops::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, kn, s);
    } else {
        qn = q;
        kn = k;
    }
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    // ---- (B) one width and one base PER LAYER, not for the whole stack ----
    // For an arch that has not opted in, these two resolve to kCfg.rotary_dim and
    // kCfg.rope_theta -- the exact constants this line passed before -- through a
    // compile-time NO-OP.  The width comes from the arch's own rotary_dim_at and the
    // base from its own rope_theta_at; it is deliberately NOT taken from a sibling
    // target.  The yarn branch keeps its full-shape requirement
    // (src/ops/wrapper/rope.cpp:150-153 demands theta == 1e7 and D256/R64), so an
    // arch whose per-layer thetas are not 1e7 still fails there BY NAME rather than
    // silently rotating.
    const int   rope_dim   = ModelConfig::rope_dim_for(fidx);
    const float rope_theta = ModelConfig::rope_theta_for(fidx);
    // F910: THE NoPE LEAF. A layer whose DECLARED theta is 0.0F gets NO rotation. The PRE tree
    // rotated it by the global TextConfig::rope_theta instead, because the per-layer trigger
    // tested `rotary_dim_at` (which muse does not declare) and muse's 13 NoPE layers fell to
    // `TextConfig::rope_theta`. The skip is compiled ONLY for an arch that declares a per-layer
    // theta, so every other arch emits exactly the call sequence it emitted before.
    if constexpr (ModelConfig::rope_declares_theta()) {
        if (ModelConfig::rope_is_nope(fidx)) {
            // AN OBSERVATION REMOVED MUST BE NAMED, never silently dropped (the same rule
            // dl/allons F881 applied to its suppressed D2H).
            static std::atomic<int> nope_reported{0};
            if (nope_reported.fetch_add(1) < 8) {
                std::fprintf(stderr,
                             "[nope] layer=%d declares rope_theta_at==0 -> RoPE SKIPPED (NoPE); "
                             "before this seam the layer was rotated by the GLOBAL "
                             "TextConfig::rope_theta=%g, a value this arch never declared for "
                             "it\n",
                             fidx, static_cast<double>(TextConfig::rope_theta));
            }
        } else if (ctx_.yarn_enabled) {
            ops::rope_yarn4(rope_for_op, rope_dim, rope_theta, qn, kn, s);
        } else {
            ops::rope(rope_for_op, rope_dim, rope_theta, qn, kn, s);
        }
    } else if (ctx_.yarn_enabled) {
        ops::rope_yarn4(rope_for_op, rope_dim, rope_theta, qn, kn, s);
    } else {
        ops::rope(rope_for_op, rope_dim, rope_theta, qn, kn, s);
    }

    mix_probe("qn", qn);
    mix_probe("kn", kn);
    // NINFER_KVDUMP_DIR + NINFER_KVDUMP_KV=<layers|all>: dump the exact BF16 K/V
    // (post rmsnorm+rope) and positions the append path consumes.
    if (kvdump_dir() != nullptr && (ph == Phase::Prefill || ph == Phase::Verify) &&
        kvdump_layer_enabled("NINFER_KVDUMP_KV", fidx)) {
        static std::atomic<std::uint32_t> kv_source_counter{0};
        const std::uint32_t dump_id = kv_source_counter.fetch_add(1);
        const std::string base = std::string(kvdump_dir()) + "/kvsrc_" + std::to_string(dump_id) +
                                 "_L" + std::to_string(fidx);
        kvdump_dump_tensor(s, kn, 0, base + "_kn.bin");
        kvdump_dump_tensor(s, v, 0, base + "_v.bin");
        kvdump_dump_tensor(s, cache_positions, 0, base + "_pos.bin");
        std::FILE* meta = std::fopen((base + "_meta.txt").c_str(), "w");
        if (meta != nullptr) {
            std::fprintf(meta, "layer=%d phase=prefill T=%d head_dim=%d n_q=%d n_kv=%d\n", fidx, T,
                         kCfg.head_dim, kCfg.n_q, kCfg.n_kv);
            kvdump_describe(meta, "kn", kn);
            kvdump_describe(meta, "v", v);
            kvdump_describe(meta, "pos", cache_positions);
            std::fclose(meta);
        }
    }

    // N3/S38: calibration record -- the exact tensors the KVDUMP_KV block
    // above dumps are the ones the append path quantizes; capture them as
    // framed .kvc records for offline baking (kv_rowscale_sidecar bake).
    // Batched-sequence prefill is REFUSED loudly: positions arrive as
    // [width, batch] there and would produce plausible-looking misaligned
    // records (enforced, not warned).
    if (ph == Phase::Prefill && kvcalib_enabled()) {
        if (active_sequence_batch_ != 0) {
            throw std::logic_error(
                "[kvcalib] batched-sequence prefill cannot be captured "
                "(positions are [width, batch]); run calibration in "
                "single-sequence mode");
        }
        kvcalib_capture(static_cast<std::uint32_t>(fidx), kn, v,
                        cache_positions, s);
    }

    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    const Tensor& kv_table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch        = qn.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor position_batch = cache_positions.view({width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        // M1: the per-column ancestor masks of a tree verify round (empty for a chain round).
        const Tensor masks = active_column_masks_ != nullptr ? *active_column_masks_ : Tensor{};
        ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, valid, masks,
                                      kv_table_rows, 
                                      kAttnScale, batch_text_kv_->batch_layer_view(fidx),
                                      *active_causal_attention_envelope_, work_, a_batch, s);
    } else {
        ops::gqa_attention(qn, kn, v, cache_positions, Tensor{}, Tensor{}, kv_table_rows,
                                      kAttnScale,
                                      batch_text_kv_->batch_layer_view(fidx),
                                      *active_causal_attention_envelope_, work_, a, s);
    }
    ops::sigmoid_mul(gate, a, s);
    mix_probe("attn_out", a);

    // F910: the arch's own mode selects the leaf; the attention-output norm belongs between
    // o_proj and the residual add, which only the variant leaf can place.
    attn_output_projection_leaf<Variant>(a.view({kCfg.q_size, T}), *w.o_proj,
                                         w.post_attn_out_norm, x, ph, work_, s);
    mix_probe("out_x", x);
}

void TextContext::gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    // layerrec probe session (READ-ONLY; see the gdndump_* block above for the gates and
    // the reason this cannot change a number).  `gidx` is the GDN index; GdnDump recovers
    // the full layer index the same way `run_layers` does.
    GdnDump gdn_probe(gidx, ph == Phase::Verify, s);
    const auto gdn_dump = [&](const char* stage, const Tensor& t) { gdn_probe.dump(stage, t); };
    // The layer's INPUT residual, before the control projection touches anything.
    gdn_dump("in_x", x);

    const auto control = workspace_recipe::gdn_control<TextConfig>(work_, T);
    Tensor h           = control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    Variant::gdn_norm_control_projection(x, *w.input_norm, kCfg.rms_eps, *w.projection, h, g, beta,
                                         work_, s);
    // Control-projection outputs: `h` is the projection input, `g`/`beta` the GDN gate.
    gdn_dump("h", h);
    gdn_dump("g", g);
    gdn_dump("beta", beta);

    const auto projection = workspace_recipe::gdn_projection<TextConfig>(work_, T);
    Tensor z              = projection.output_gate.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor qc             = projection.query;
    Tensor kc             = projection.key;
    Tensor vc             = projection.value;
    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_source_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        if (gdn_state_action_ == GdnStateAction::UpdateInPlace && width != 1) {
            throw std::logic_error("In-place batched GDN update requires width one");
        }
        Tensor projection_input = h.view({kCfg.hidden, width, active_sequence_batch_});
        Tensor query_output     = qc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor key_output       = kc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor value_output     = vc.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor gate_output      = z.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor conv_states      = state_.layer_view(static_cast<std::uint32_t>(gidx)).conv;
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            if (replay_records_ == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            Variant::gdn_input_projection_record(
                projection_input, *w.projection, *w.conv1d, conv_states, valid,
                *active_linear_state_source_slots_, records.conv, query_output, key_output,
                value_output, gate_output, ph, work_, s);
        } else {
            Variant::gdn_input_projection_snapshot(
                projection_input, *w.projection, *w.conv1d, conv_states, valid,
                *active_linear_state_source_slots_, *active_linear_state_destination_slots_,
                query_output, key_output, value_output, gate_output, ph, work_, s);
        }
    } else {
        Tensor qkv = workspace_recipe::gdn_prefill_conv<TextConfig>(work_, T);
        Variant::gdn_input_projection(h, *w.projection, qkv, z, ph, work_, s);
        Tensor conv_state_in =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor conv_state_out =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::causal_conv1d_silu_split(qkv, *w.conv1d, conv_state_in, conv_state_out, qc, kc, vc, s);
    }

    // The four projection outputs the GDN recurrence consumes, exactly as written by
    // whichever registered route the dispatch picked (snapshot / record / materialized).
    gdn_dump("qc", qc);
    gdn_dump("kc", kc);
    gdn_dump("vc", vc);
    gdn_dump("z", z);
    if (ph == Phase::Verify && active_linear_state_source_slots_ != nullptr) {
        // The width-three convolution history the snapshot op reads, plus the selector
        // that says WHICH slot of it: without both, qc/kc/vc cannot be recomputed.
        gdn_dump("conv_pool", state_.layer_view(static_cast<std::uint32_t>(gidx)).conv);
        gdn_dump("src_slots", *active_linear_state_source_slots_);
    }

    Tensor q_recurrent = qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});
    Tensor k_recurrent = kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});

    Tensor vv = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor o  = workspace_recipe::gdn_recurrent_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    if (ph == Phase::Verify) {
        Tensor recurrent_states  = state_.layer_view(static_cast<std::uint32_t>(gidx)).recurrent;
        const std::int32_t width = active_sequence_width_;
        Tensor q_batch =
            q_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor k_batch =
            k_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor v_batch = vv.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor g_batch = g.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor beta_batch = beta.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor out_batch =
            o.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            ops::gated_delta_net_replay_record(q_batch, k_batch, v_batch, g_batch, beta_batch,
                                               kGdnScale, recurrent_states, valid,
                                               *active_linear_state_source_slots_, records.key,
                                               records.value, records.gate, out_batch, s);
        } else {
            ops::gated_delta_net_batch_update(
                q_batch, k_batch, v_batch, g_batch, beta_batch, kGdnScale,
                /*normalize_qk=*/true, recurrent_states, *active_linear_state_source_slots_,
                *active_linear_state_destination_slots_, out_batch, s);
        }
    } else {
        Tensor recurrent_state_in =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor recurrent_state_out =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::gated_delta_net(q_recurrent, k_recurrent, vv, g, beta, kGdnScale,
                             /*normalize_qk=*/true, work_, recurrent_state_in, recurrent_state_out,
                             o, s);
    }

    Tensor on = workspace_recipe::gdn_normalized_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    ops::gated_rmsnorm(o, *w.gdn_norm, z, kCfg.rms_eps, on, s);

    // The recurrence output and the gated normalisation, before the output projection.
    gdn_dump("o", o);
    gdn_dump("on", on);

    Variant::gdn_output_projection(on.view({kCfg.value_dim, T}), *w.out_proj, x, ph, work_, s);
    // The layer's OUTPUT residual (x after the mixer), for the next layer's `in_x` join.
    gdn_dump("out_x", x);
}

void TextContext::mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    Tensor h       = workspace_recipe::post_mixer_hidden<TextConfig>(work_, T);
    ops::rmsnorm(x, *post_norm, kCfg.rms_eps, true, h, s);

    // F910: the leaf is chosen by the arch's OWN declaration, in one place, and the double-norm
    // leaf finally has a caller (`Variant::post_mixer_double_norm` was called from nowhere).
    post_mixer_leaf<Variant>(h, m, x, ph, work_, s);
}


// ---------------------------------------------------------------------------
// --stage-layers: the boundary payload
// ---------------------------------------------------------------------------
// 64-bit magic, so a file this engine did not write is refused rather than read as a payload.
// The layout is fixed and little-endian, and it is written by stage k and read by stage k+1 --
// one producer, one consumer, no third reader.
namespace {
constexpr std::uint64_t kStageHandoffMagic = 0x50575057'43474E48ULL; // "PWPWCGNH"
struct StageHandoffHeader {
    std::uint64_t magic = 0;
    std::int32_t layer = 0;      // the layer the producing stage's LAST step ran
    std::int32_t stage = 0;      // which stage wrote it
    std::uint64_t bytes = 0;
    std::uint64_t fnv1a = 0;
    std::uint32_t phase = 0;     // 0 = prefill, 1 = verify/decode; a payload from the wrong
                                 // phase is a different round and must not be consumed silently
    std::uint32_t reserved = 0;
};
[[nodiscard]] std::uint64_t stage_fnv1a(const void* data, std::size_t bytes) {
    const auto* bytes_in = static_cast<const unsigned char*>(data);
    std::uint64_t hash   = 1469598103934665603ULL;
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes_in[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}
[[nodiscard]] std::string stage_handoff_path(const std::string& dir, std::uint32_t stage) {
    return dir + "/stage_" + std::to_string(stage) + ".bin";
}
} // namespace

void TextContext::set_stage_layers_spec(std::string_view spec, std::string handoff_dir,
                                        bool handoff_cut) {
    stage_plan_        = multi::StagePlan{};
    stage_handoff_dir_ = std::move(handoff_dir);
    stage_handoff_cut_ = handoff_cut;
    if (spec.empty()) {
        // The flag was absent: no parse, no validation, and the layer walk is exactly what it
        // was before core/stage_plan.h existed. The only thing that can still be wrong is a
        // --stage-handoff with no stage set, which options.cpp already refuses by name; it is
        // re-checked here because the runtime is reachable without going through that front
        // door (ninfer-serve builds EngineOptions itself).
        if (!stage_handoff_dir_.empty()) {
            throw std::invalid_argument(
                std::string(multi::kStageLayersHandoffRefusal) + ": " +
                std::string(multi::kStageHandoffFlag) + " was given with no " +
                std::string(multi::kStageLayersFlag) + ", so there is no stage boundary to carry");
        }
        return;
    }
    multi::StagePlan plan;
    if (const std::string refusal = multi::parse_stage_layers(spec, plan); !refusal.empty()) {
        throw std::invalid_argument(refusal);
    }
    // THE GEOMETRY IS THE ARTIFACT'S OWN. This is the whole reason the axis check lives HERE
    // and not in the front door: kCfg is the loaded model's config, so plan_shards() is asked
    // about the real layer count -- and `weight_columns` is left 0 because the pp arm of
    // plan_shards does not read it (the weights are whole on every stage; only tp splits N, and
    // no spec this flag accepts can ask for tp).
    const multi::ModelGeometry geometry{
        .text_layers    = static_cast<std::uint32_t>(kCfg.n_layers),
        .q_heads        = static_cast<std::uint32_t>(kCfg.n_q),
        .kv_heads       = static_cast<std::uint32_t>(kCfg.n_kv),
        .head_dim       = static_cast<std::uint32_t>(kCfg.head_dim),
        .weight_columns = 0U,
    };
    if (const std::string refusal =
            multi::stage_layers_partition_refusal(plan, geometry.text_layers);
        !refusal.empty()) {
        throw std::invalid_argument(refusal);
    }
    if (const std::string refusal = multi::stage_plan_axis_refusal(plan, geometry);
        !refusal.empty()) {
        throw std::invalid_argument(refusal);
    }
    if (const std::string refusal = multi::stage_handoff_refusal(plan, stage_handoff_dir_);
        !refusal.empty()) {
        throw std::invalid_argument(refusal);
    }
    stage_plan_ = std::move(plan);
    // The one reading this line adds at startup: which world was actually accepted, printed
    // BEFORE the first token so a run's stages are a fact in the log and not an inference from
    // the flags. A single line, so an identity run and a pp run differ by one line and nothing
    // else -- which is what arm T2 measures.
    std::fprintf(stderr, "[stage] accepted %s: %u stage(s) over %u text layers%s\n",
                 std::string(spec).c_str(), stage_plan_.world_size(), geometry.text_layers,
                 stage_plan_.uniform() ? "" : " (uneven)");
    for (std::uint32_t i = 0; i < stage_plan_.world_size(); ++i) {
        std::fprintf(stderr, "[stage]   stage %u: layers [%u,%u] (%u layers)\n", i,
                     stage_plan_.stages[i].first, stage_plan_.stages[i].last,
                     stage_plan_.stages[i].count());
    }
    if (!stage_handoff_dir_.empty()) {
        std::fprintf(stderr, "[stage]   handoff dir %s%s\n", stage_handoff_dir_.c_str(),
                     stage_handoff_cut_ ? " (PRODUCER SILENCED -- negative control)" : "");
    }
}

void TextContext::stage_handoff_write(Tensor& x, int layer_last) {
    if (stage_handoff_dir_.empty()) {
        throw std::logic_error("stage boundary reached with no --stage-handoff directory: a "
                               "multi-stage world must name where its hidden state crosses");
    }
    // --stage-handoff-cut: the producer is deliberately silent. The consumer is NOT told --
    // that is the entire point of the control, and it is why the ids must move. The payload
    // that is already in the file (from a previous run, or all zeros) is consumed as if it were
    // this step's.
    if (stage_handoff_cut_) { return; }
    const std::size_t bytes = static_cast<std::size_t>(x.bytes());
    if (bytes == 0 || x.data == nullptr) {
        throw std::logic_error("stage boundary payload is empty");
    }
    std::vector<unsigned char> host(bytes);
    if (cudaMemcpy(host.data(), x.data, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void)cudaGetLastError();
        throw std::runtime_error("stage boundary payload D2H failed");
    }
    StageHandoffHeader header;
    header.magic = kStageHandoffMagic;
    header.layer = layer_last;
    header.stage = 0;
    header.bytes = bytes;
    header.fnv1a = stage_fnv1a(host.data(), bytes);
    header.phase = 0;
    const std::string path = stage_handoff_path(stage_handoff_dir_, 0U);
    std::FILE* file        = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        throw std::runtime_error("stage boundary payload: cannot open " + path +
                                 " for writing");
    }
    const bool wrote = std::fwrite(&header, sizeof(header), 1, file) == 1 &&
                       (bytes == 0 || std::fwrite(host.data(), bytes, 1, file) == 1);
    std::fclose(file);
    if (!wrote) {
        throw std::runtime_error("stage boundary payload: short write to " + path);
    }
    // The producer's FNV, on the producer's stderr. The consumer prints the FNV of what it
    // READ, so the two numbers being equal is a reading about the payload rather than a promise
    // about the code path -- dl/ppaxis (F-739) built this instrument and this line re-runs it.
    std::fprintf(stderr, "[stage] producer wrote %s: layer %d, %zu B, fnv=%016llx\n", path.c_str(),
                 layer_last, bytes, static_cast<unsigned long long>(header.fnv1a));
}

void TextContext::stage_handoff_read(Tensor& x) {
    if (stage_handoff_dir_.empty()) {
        throw std::logic_error("stage boundary reached with no --stage-handoff directory");
    }
    const std::string path = stage_handoff_path(stage_handoff_dir_, 0U);
    std::FILE* file        = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        throw std::runtime_error(
            "stage boundary payload " + path +
            " does not exist. A consuming stage refuses rather than continuing from a hidden "
            "state no stage produced -- the same loud-contradiction rule the cold tier uses.");
    }
    StageHandoffHeader header;
    if (std::fread(&header, sizeof(header), 1, file) != 1) {
        std::fclose(file);
        throw std::runtime_error("stage boundary payload " + path + " is shorter than its header");
    }
    if (header.magic != kStageHandoffMagic) {
        std::fclose(file);
        throw std::runtime_error("stage boundary payload " + path +
                                 " does not begin with this engine's magic: it is not a payload "
                                 "this build wrote, and it is refused rather than consumed");
    }
    const std::size_t bytes = static_cast<std::size_t>(x.bytes());
    // A SIZE MISMATCH IS A STALE PAYLOAD, NOT A CORRUPT ONE, and this is the one place where
    // that distinction is load-bearing. The payload's size is a function of the ROUND (a prefill
    // chunk's token count, a decode step's batch), so a producer and a consumer that disagree
    // about which round they are in are exactly the state --stage-handoff-cut produces: the
    // producer is silent and the consumer consumes whatever the file still holds. The reader
    // therefore consumes what is there, zero-fills the remainder, and SAYS SO on every call --
    // the tolerance is not silent, and the FNV line below is the tripwire that reports whether
    // what was consumed is what was produced. In a correct world (a producer that writes the
    // round's own shape) this path is unreachable, which is why it can afford to be tolerant and
    // must afford to be loud.
    const std::size_t available = header.bytes;
    std::vector<unsigned char> host(bytes, 0U);
    std::size_t read_back = 0;
    if (available != 0) {
        const std::size_t take = available < bytes ? available : bytes;
        read_back              = std::fread(host.data(), 1, take, file);
    }
    std::fclose(file);
    if (available != bytes) {
        std::fprintf(stderr,
                     "[stage] consumer read %s: STALE SHAPE -- the payload holds %zu B and this "
                     "round's hidden state is %zu B, so %zu B were %s. This is the state a cut "
                     "handoff produces; the ids are the evidence.\n",
                     path.c_str(), available, bytes,
                     available < bytes ? bytes - available : available - bytes,
                     available < bytes ? "zero-filled" : "truncated");
    }
    const std::uint64_t read_fnv = stage_fnv1a(host.data(), bytes);
    std::fprintf(stderr,
                 "[stage] consumer read %s: produced at layer %d, %zu B read of %zu B, "
                 "fnv=%016llx%s\n",
                 path.c_str(), header.layer, read_back, bytes,
                 static_cast<unsigned long long>(read_fnv),
                 read_fnv == header.fnv1a ? " (== producer)" : " (STALE: != producer FNV)");
    if (cudaMemcpy(x.data, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        (void)cudaGetLastError();
        throw std::runtime_error("stage boundary payload H2D failed");
    }
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    const bool prefill = ph == Phase::Prefill;
    // ⚠️ PATCH A2 cap: never dump more than this many columns per (layer, stage). Keeps a
    // prefill chunk (thousands of columns) from turning one probe into thousands of lines.
    static constexpr std::int32_t kHeadDebugProbeColumns = 16;
    // NINFER_HEADDBG=1: per-stage probe on every layer so the first NaN inside the decode stack
    // is visible (see _TODO.md 102/103).
    // NINFER_HEADDBG=1: per-stage probe over the FIRST kHeadDebugProbeColumns columns of
    // `x`, one LINE per column, labelled (layer, stage, phase, width, batch, column, pos)
    // so hd_diff.py can JOIN BY THAT TUPLE instead of by output line index.
    //
    // WHERE w / b / c COME FROM: `active_sequence_width_` and `active_sequence_batch_` are
    // MEMBERS of TextContext, so a member function already reads them here
    // (target_verify_batch_impl:899-900 and ordinary_decode_batch:837 bind them; prefill
    // binds neither, so both stay 0 and the fallback 1/1 reproduces the old single-column
    // dump for prefill). `cache_positions[0]` is read from `active_cache_positions_` D2H.
    //
    // ⚠️ `column_begin` IS NOT IN THIS SCOPE AND IS NOT INVENTED HERE. It is the attention
    // invocation's chunk offset (ops/launcher/gqa_attention.h:39). On this path it is
    // PROVABLY 0 for the round under study: ops/wrapper/gqa_attention.cpp:597-598 passes a
    // literal 0, and the only non-zero value on the chunked route is a multiple of
    // kSmallTChunkTokens (6) from launch_chunked_small_t (:414-424), where chunk 2 writes
    // only columns >= 6. To move this window, thread a `std::int32_t column_begin` through
    // text_context.h:376/378 and the three run_layers call sites -- NOT done here, to keep
    // A2 inside ONE translation unit.
    const auto probe = [&](int layer, const char* stage) {
        if (!head_debug_enabled()) { return; }
        const std::int32_t width = active_sequence_width_ != 0 ? active_sequence_width_ : 1;
        const std::int32_t batch = active_sequence_batch_ != 0 ? active_sequence_batch_ : 1;
        const std::int32_t columns = width * batch;
        const std::int32_t dump =
            columns < kHeadDebugProbeColumns ? columns : kHeadDebugProbeColumns;
        std::int32_t cache_pos = -1;
        if (active_cache_positions_ != nullptr && active_cache_positions_->data != nullptr &&
            active_cache_positions_->dtype == DType::I32 &&
            active_cache_positions_->bytes() >= sizeof(std::int32_t)) {
            if (cudaMemcpyAsync(&cache_pos, active_cache_positions_->data, sizeof(std::int32_t),
                                cudaMemcpyDeviceToHost, ctx_.stream) != cudaSuccess) {
                (void)cudaGetLastError();
                cache_pos = -1;
            } else if (cudaStreamSynchronize(ctx_.stream) != cudaSuccess) {
                (void)cudaGetLastError();
                cache_pos = -1;
            }
        }
        for (std::int32_t j = 0; j < dump; ++j) {
            char label[64];
            std::snprintf(label, sizeof(label), "L%02d_%s_%s_w%d_b%d_c%d_p%d", layer, stage,
                          prefill ? "prefill" : "verify", static_cast<int>(width),
                          static_cast<int>(batch), static_cast<int>(j),
                          static_cast<int>(cache_pos));
            debug_head_probe(ctx_.stream, x, label, j);
        }
    };
    // --stage-layers: THE STAGE WALK. The whole point of this line's change is these four
    // lines: the loop below used to be `for (int layer = 0; layer < kCfg.n_layers; ++layer)`,
    // i.e. every layer of the model, always. It is now bounded by the stage the run was ASKED
    // for, and between two stages the hidden state crosses a boundary -- through the file
    // --stage-handoff names when one was named. With no flag, stage_count is 1 and the bounds
    // are [0, kCfg.n_layers - 1], so the walk is the identical single walk; that equivalence is
    // arm T2 of this line's landq entry and it is asserted rather than intended.
    // W13 (notehook) -- resolved ONCE per walk, before the stage loop, because the answer is a
    // property of the STREAM and not of the layer: is a capture running on the stream this walk
    // is about to run on? `ops::stream_is_capturing` is the tree's single reader for that
    // question (src/ops/stream_capture.h, F881) and it is asked at RUNTIME rather than threaded
    // in, because a capture body is the graph DEFINITION and runs once while every later round
    // of the same width is only a cudaGraphLaunch -- the caller cannot tell the two apart, and
    // the stream can be asked.
    product::WeightResidencyRuntime* const w13_residency = weights_.backing.weight_residency();
    // F1059 (streamfix): THE HOOK NOW RUNS UNDER CAPTURE TOO, and it can because the fetch no
    // longer has to be a host-side call outside the graph.
    //
    // notehook gated this on `!ops::stream_is_capturing(ctx_.stream)` for a real reason: the H2D
    // rode the engine's own transfer_stream, unjoined to the capture, so issuing it from the
    // capturing thread was the cudaErrorStreamCapture* fault (src/ops/stream_capture.h:6-11).
    // The fetch now rides a stream the offload OWNS, and `fetch_enroll()` forks that stream into
    // the capture the moment a capture is detected on the consuming stream, so the copies become
    // graph NODES and each layer's record/wait pair becomes a graph EDGE -- the protocol
    // src/core/decode_graph_peer.h:60-83 names and :326-346 implements. Both capture entry points
    // this tree uses are on this same stream (decode_impl.h:85 and graph_impl.h:26, both
    // `state.execution.device.stream`), which is why the predicate below is the capture query and
    // not a threaded-in flag.
    const bool w13_from_hook = w13_residency != nullptr;
    if (w13_from_hook && ops::stream_is_capturing(ctx_.stream)) {
        // NOTHING IS SKIPPED HERE ANY MORE, and this line says so once. Before F1059 this arm
        // meant "the hook is being skipped, so a rotating arena is being read as of capture
        // time"; now the hook runs and the fetch is enrolled, so the arena is re-filled by the
        // graph on every replay. It is kept because a reader of this file has to be able to tell
        // "capture, and the fetch is inside it" from "capture, and the fetch is not" without a
        // debugger -- the rule in ops/stream_capture.h:36-39.
        static bool w13_capture_note_printed = false;
        if (!w13_capture_note_printed) {
            w13_capture_note_printed = true;
            std::fprintf(stderr,
                         "[weight-offload] F1059: CUDA-graph capture is in flight on this stream; "
                         "note_layer() RUNS and each layer's H2D is enrolled into the capture on "
                         "the offload's own stream, so the arena is re-filled by graph nodes on "
                         "every replay rather than read as of capture time. The engine's "
                         "transfer_stream is not used by the offload at all.\n");
        }
    }
    const int stage_count = stage_plan_.requested
                                ? static_cast<int>(stage_plan_.world_size())
                                : 1;
    for (int stage = 0; stage < stage_count; ++stage) {
        const int layer_first = stage_plan_.requested
                                    ? static_cast<int>(stage_plan_.stages[stage].first)
                                    : 0;
        const int layer_last = stage_plan_.requested
                                   ? static_cast<int>(stage_plan_.stages[stage].last)
                                   : kCfg.n_layers - 1;
        // A stage above the first does NOT compute its input: it is handed the hidden state the
        // stage below produced, so this stage's own layer stack starts from that, not from an
        // embedding it has no business computing.
        if (stage != 0) { stage_handoff_read(x); }
        for (int layer = layer_first; layer <= layer_last; ++layer) {
        // W13: the residency hook, at the layer boundary. The H2D for layer
        // L + arena_layers - 1 is issued here while layer L is being computed, so
        // the arena slot a live layer occupies is never the one being refilled.
        //
        // PREFILL ONLY, deliberately. Prefill is not CUDA-graph captured and its
        // transfer is amortized over a whole chunk of tokens, which is the regime
        // where a weight offload is net-positive; decode captures these nodes into
        // a graph, where a host-side fetch would be replayed with stale data.
        // Landing W13 on decode needs the H2D lifted into the graph on a second
        // stream -- specified in scratch/w13a/DESIGN.md section 5, NOT done.
        // W13 (notehook): THE HOOK NOW RUNS IN EVERY PHASE THAT IS NOT CUDA-GRAPH CAPTURED,
        // not in Prefill alone. The old gate was `ph == Phase::Prefill`, and it was the whole
        // reason --weight-host-bytes was refused by name: a decode pass never entered an
        // offloaded layer, so the arena's other strips were read stale (see the refusal in
        // weight_residency.h, quoted verbatim in the F1048 packet). The predicate is the capture
        // query resolved above and NOT the phase, because the phase is not what makes the fetch
        // illegal -- the capture is.
        //
        // WHY THE FETCH IS ILLEGAL UNDER CAPTURE, stated where it is decided: enqueue_h2d and
        // the event record/wait are CUDA calls issued from the capturing thread onto streams
        // that are not part of the capture, which invalidates it (cudaErrorStreamCapture*), and
        // a replay would re-run the node without re-running the host decision that put it there.
        if (w13_from_hook) {
            w13_residency->note_layer(static_cast<std::uint32_t>(layer));
        }
        if (ModelConfig::is_full(layer)) {
            const int fidx         = ModelConfig::full_idx(layer);
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            nvtx::ScopedRange layer_range(
                prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull,
                nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention,
                    nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                // L26 prototype: a dropped layer has no KV planes at all, so its
                // attention is never enqueued -- no KV append, no QK/PV. The layer
                // keeps only its MLP tail, which is exactly the discard semantics
                // this instrument measures. fidx is the full-attention index, the
                // same index plan_cache() uses for the plane geometry.
                if (!batch_text_kv_->layer_is_dropped(static_cast<std::uint32_t>(fidx))) {
                    attn_mix(full, x, fidx, ph);
                }
            }
            probe(layer, "attn");
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(full.post_attn_norm, full.mlp, x, ph);
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, ctx_.stream); }
            }
            probe(layer, "mlp");
        } else {
            const int gidx       = ModelConfig::gdn_idx(layer);
            const GdnLayerW& gdn = gdn_.at(static_cast<std::size_t>(gidx));
            nvtx::ScopedRange layer_range(prefill ? nvtx::Name::PrefillLayerGdn
                                                  : nvtx::Name::VerifyLayerGdn,
                                          nvtx::Category::Gdn, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn, nvtx::Category::Gdn,
                    static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                gdn_mix(gdn, x, gidx, ph);
            }
            probe(layer, "gdn");
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(gdn.post_attn_norm, gdn.mlp, x, ph);
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, ctx_.stream); }
            }
            probe(layer, "mlp");
        }
        }
        // Every stage but the last hands its output up. The LAST stage is the one that keeps
        // the hidden state for the head, which is why the sampler lives there -- see
        // set_stage_layers_spec's note on the rank-0-owns-sampling rule.
        if (stage + 1 < stage_count) { stage_handoff_write(x, layer_last); }
    }
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64    = static_cast<std::int64_t>(base);
    const std::int64_t split_abs = prefill_split_frontier_;
    const bool has_split = split_abs > base64 && split_abs <= base64 + static_cast<std::int64_t>(T);
    const int split_rel  = has_split ? static_cast<int>(split_abs - base64) : -1;
    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }
    int t0 = 0;
    for (; t0 < T;) {
        int len = std::min(chunk, T - t0);
        if (split_rel > 0 && t0 < split_rel && t0 + len > split_rel) { len = split_rel - t0; }
        work_.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        const bool is_last = finalize_at_end && (t0 + len == T);
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        {
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0);
                const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] =
                        begin[i] - static_cast<std::int32_t>(prompt_t0);
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
            const auto roots             = workspace_recipe::text_prefill_roots<TextConfig>(
                work_, len, rope_axes, static_cast<std::int32_t>(local_scatter_indices.size()));
            Tensor ids_device = roots.ids;
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len);
            // FIX-A: the third field is the pinned split reference. Left out, this two-field
            // aggregate value-initialises split_reference_keys to 0, so the MTP bridge and the
            // MTP prompt chunks reduced their keys on a 32-key grid derived from the LIVE window
            // (`[splitdbg] ... pin=0 split_reference=47 ... split_units=32`) while the batch-1
            // decode of the same row used the capacity-pinned grid. The pin is the compile-time
            // maximum visible-key count, a constant of the produced graph, which is what the
            // contract asks for; the planner's own capacity is not in scope here.
            const ops::GqaExecutionEnvelope chunk_envelope{
                visible, visible, ops::kGqaAttentionMaximumVisibleKeys};
            ScopedEnvelope scoped_envelope(active_causal_attention_envelope_, chunk_envelope);

            Tensor x = roots.residual;
            ops::embedding(ids_device, *embed_, x, s);
            // F738 injectchan: THE CONSUMPTION POINT. The chosen tensor is written into
            // the column of the input-embedding matrix that the declared absolute position
            // owns -- the same tensor, the same column, the same instant a gathered
            // token's embedding arrives at. That is what sum_dir.h:98 requires, and it is
            // the only point at which "looks to the model like ordinary context" is a
            // statement about arithmetic rather than about intent. Before the vision
            // scatter on purpose: a declared column that is also a vision destination is
            // refused by name rather than resolved by write order. A no-op, one branch
            // deep, unless NINFER_INJECT_SPEC named a spec.
            inject_ingress::apply(x, static_cast<std::int32_t>(base_i + t0),
                                  local_scatter_indices, s);
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            if constexpr (Tap::enabled) { tap.begin(x); }
            run_layers(x, Phase::Prefill, tap);
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {kCfg.hidden, len});
            ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, xf, s);

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                ops::linear(last_xf, *lm_head_, logits, s);
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token, kCfg.token_domain, sampling_config_, io_.pos,
                                ops::kSamplePurposePrefill, work_, s);
                } else {
                    ops::argmax(logits, io_.token, kCfg.token_domain, s);
                }
            }

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const qwen3_6::MtpAlignmentWindow mtp_window = qwen3_6::plan_mtp_alignment_window(
                    alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                if (prompt_columns != 0) {
                    Tensor prompt_mtp_ids = mtp_ids.slice(0, 0, prompt_columns);
                    copy_i32(alignment_ids.data() + mtp_window.shifted_embedding_begin,
                             prompt_mtp_ids, s);
                }
                if (mtp_window.final_column_uses_generated_token) {
                    Tensor generated_mtp_id = mtp_ids.slice(0, len - 1, 1);
                    CUDA_CHECK(cudaMemcpyAsync(generated_mtp_id.data, io_.token.data,
                                               sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
                }

                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings = work_.alloc(DType::BF16, {kCfg.hidden, len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const qwen3_6::MtpVisualOverlap overlap = qwen3_6::shifted_visual_overlap(
                            vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace_recipe::visual_scatter_indices(
                                work_, static_cast<std::int32_t>(overlap.size()));
                            qwen3_6::detail::scatter_shifted_visual_embeddings(
                                mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                shifted_indices, s);
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent_ != 0) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp->ar_hidden,
                                      &logits, &draft0);
                    // --draft-tree L,d (L > 1): the FIRST verify round is a tree round and the
                    // round-1 gate reads the published tree FIELD, so the depth-0 lattice has to be
                    // produced by the proposal that just ran. THIS block -- not
                    // mtp_bridge_and_propose() -- is the proposal loop an ordinary fresh prompt
                    // takes, and a fresh prompt is the common case; leaving only the bridge wired
                    // publishes no lattice for it at all and the round is refused at depth 0.
                    extract_mtp_proposal_lattice(logits, 0);

                    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                    ops::set_i32_scalar(ar_position, base_i + T, s);
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                        Tensor prev_token     = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token     = io_.mtp->draft_tokens.slice(0, i, 1);
                        Tensor next_hidden    = work_.alloc(DType::BF16, {kCfg.hidden, 1});
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                        // FIX-A: same pin, same reason. This is the prompt-time MTP AR step:
                        // k-1 of them per run, one per proposal token, and each used to
                        // partition its keys from the live window.
                        const ops::GqaExecutionEnvelope ar_envelope{
                            ar_visible, ar_visible, ops::kGqaAttentionMaximumVisibleKeys};
                        mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                            ar_envelope, next_hidden, logits, next_token);
                        extract_mtp_proposal_lattice(logits, static_cast<std::uint32_t>(i));
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                                   io_.mtp->ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (split_rel > 0 && t0 + len == split_rel &&
                rewrite_checkpoint_hidden_output_ != nullptr) {
                require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                                     {kCfg.hidden, 1}, "rewrite checkpoint hidden output");
                const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                           checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
            }
        }

        if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
            work_.reset();
            tap.consume_prefill_chunk(len, split_rel > 0 && t0 + len == split_rel);
        }

        t0 += len;
        break;
    }

    prefill_split_frontier_ = -1;
    // F738 injectchan: an admitted declaration that this run's prefill did not fully
    // cover is refused by name rather than left as a partial injection.
    if (finalize_at_end) { inject_ingress::finish(); }

    timing.begin_wait();
    ctx_.synchronize();
    timing.end_wait();
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T,
                              .timing           = timing.finish()};
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    NullTap tap;
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, sink,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_6::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    NullTap tap;
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, tap,
                        finalize_at_end);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule


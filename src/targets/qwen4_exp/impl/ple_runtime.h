#pragma once

// S33: PLE runtime seam for the qwen4_exp (FlashNext) target.
//
// `src/ops/ple/ple_table.{h,cu}` implements the SSD-backed gather end to end
// (PleLayout row derivation, pinned LRU, ple_gather_rows_kernel) but nothing
// instantiates it. This header is the single place the future qwen4_exp
// runtime plugs into: it owns the PleTable, resolves the sidecar root, and
// exposes one batched gather that produces the per-token PLE embedding
// [n_heads * row_dim, n_tokens] = [2560, T] BF16 (16 heads x 160 = hidden).
//
// DEFAULT-OFF CONTRACT (S33 requirement 2):
//   * No sidecar root configured and none found next to the artifact
//     -> attach() returns nullptr; the text-only path runs unchanged and
//        never touches this header's code paths.
//   * A root IS configured/discovered but ple-manifest.json is missing or
//     unreadable -> attach() THROWS std::runtime_error naming the exact path
//        and the way to disable (--ple-sidecar= / leaving the directory
//        absent). Loud failure, never silent zeros: PleTable::gather has no
//        zero-fill fallback and we do not add one.
//
// INTEGRATION CONTRACT for the future runtime (file:line citations in
// _collab/B_s33_ple_wiring.md; these symbols do not exist yet -- that is the
// recorded finding, not something this header invents):
//   * Construction: alongside LoadedQwen4Exp construction, mirroring how
//     registry.cpp builds Loaded/Instance pairs (construct_registered,
//     src/targets/registry.cpp:112). One PleRuntime per engine instance;
//     PleTable is non-copyable and owns 4 fds + pinned cache (ple_table.h:35).
//   * Per-token ids: the current step's committed token batch. ctx0 = the
//     token itself; prevs = its 2 predecessors, OLDEST FIRST, token-major
//     ((ngram_size-1) * n_tokens entries); missing predecessors and anything
//     at/after an EOS cut pass eos (PleLayout::derive_rows, ple_layout.h:90).
//   * Entry into the layer computation: the gathered [2560, T] BF16 tensor is
//     the PLE embedding that layer 1's PLE stack consumes (checkpoint keys
//     model.layers.1.ple.* -- S26 contract) before ple.key_proj/value_proj.
//     The layer-1 residual stack itself is future runtime work.
//   * EOS id: comes from the model's tokenizer/config (eos_token_id); the
//     qwen4_exp stub carries no frontend yet, so the value is passed in by
//     the caller until one exists.

#include "ops/ple/ple_layout.h"
#include "ops/ple/ple_table.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen4_exp {

inline constexpr std::string_view kPleManifestName = "ple-manifest.json";

// Resolves the sidecar root: explicit override wins; otherwise the directory
// adjacent to the artifact (artifact.parent_path() / "ple-root") is used when
// it holds ple-manifest.json (the sidecar builder writes <out>/ple-manifest.json
// plus <out>/ple/ple-bf16-*.bin, tools/convert/ple_sidecar_build.py:55-75).
// Returns an empty path when PLE is off (no override and no adjacent root).
inline std::filesystem::path resolve_ple_sidecar_root(
    const std::filesystem::path& artifact_path,
    const std::filesystem::path& explicit_root) {
    if (!explicit_root.empty()) { return explicit_root; }
    if (artifact_path.empty()) { return {}; }
    std::filesystem::path adjacent = artifact_path.parent_path() / "ple-root";
    std::error_code ec;
    if (std::filesystem::exists(adjacent / kPleManifestName, ec)) { return adjacent; }
    return {};
}

class PleRuntime {
public:
    struct Options {
        std::size_t cache_bytes = 512ULL << 20; // pinned LRU budget (ple_table.h:30)
        // 0 only: PleTable refuses a non-zero value by name because no prefetch pool exists
        // (ple_table.h:31 records the same conclusion from the op's side).
        unsigned prefetch_workers = 0;
    };

    // Returns nullptr when PLE is off (empty resolved root). Throws loudly
    // when a root is configured but the manifest cannot be loaded.
    //
    // The two-argument form is an explicit overload, not a defaulted parameter,
    // for a compiler reason: `Options` carries default member initializers and a
    // default argument that needs them is rejected while the enclosing class is
    // still incomplete ("error: default member initializer for
    // 'PleRuntime::Options::cache_bytes' required before the end of its enclosing
    // class"). Passing the default from a function *body* -- a complete-class
    // context -- is well-formed, and both call forms (`attach(a, b)` and
    // `attach(a, b, opts)`) keep working unchanged.
    static std::unique_ptr<PleRuntime> attach(const std::filesystem::path& artifact_path,
                                              const std::filesystem::path& explicit_root) {
        return attach(artifact_path, explicit_root, Options{});
    }

    static std::unique_ptr<PleRuntime> attach(const std::filesystem::path& artifact_path,
                                              const std::filesystem::path& explicit_root,
                                              Options options) {
        const std::filesystem::path root =
            resolve_ple_sidecar_root(artifact_path, explicit_root);
        if (root.empty()) { return nullptr; }
        ops::ple::PleTableOptions table_options;
        table_options.sidecar_root    = root;
        table_options.cache_bytes     = options.cache_bytes;
        table_options.prefetch_workers = options.prefetch_workers;
        // PleTable's constructor loads + validates the manifest
        // (PleLayout::from_manifest throws std::runtime_error on structural
        // mismatch); a missing manifest surfaces as the open failure of
        // <root>/ple-manifest.json -- rethrown with the disable hint.
        try {
            return std::unique_ptr<PleRuntime>(
                new PleRuntime(std::move(table_options)));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                "qwen4_exp PLE sidecar unusable at '" + root.string() +
                "': " + error.what() +
                " (disable PLE by removing --ple-sidecar / the adjacent "
                "ple-root directory)");
        }
    }

    const ops::ple::PleLayout& layout() const noexcept { return table_.layout(); }

    // The phase-aware gather. `ple_phase_window` (ops/ple/ple_stage.h:70) is the
    // only place the phase -> column mapping lives and callers must not re-derive
    // it, so a Draft/Verify path has to come through here rather than through
    // gather_batch(): a Verify pass carries K+1 columns but keeps only the last,
    // and gathering all of them would fault K+1 columns of SSD traffic for one
    // column of useful data (ple_stage.h:12-17).
    void gather_phase(std::span<const std::int32_t> tokens,
                      std::span<const std::int32_t> prevs, std::int32_t eos,
                      ops::ple::PlePhase phase, void* dst, cudaStream_t stream) const {
        if (prevs.size() !=
            static_cast<std::size_t>(table_.layout().ngram_size - 1) * tokens.size()) {
            throw std::invalid_argument(
                "PLE gather_phase: prevs must hold (ngram_size-1)*n_tokens ids");
        }
        table_.gather_phase(tokens, prevs, eos, phase, dst, stream);
    }

    // The acceptance observable (ops/ple/ple_stage.h:110-115): a wired stage that
    // never gathers is indistinguishable from an unwired one from the outside.
    [[nodiscard]] const ops::ple::PleForensics& forensics() const noexcept {
        return table_.forensics();
    }

    // Geometry cross-check against the target config (impl/config.h carries
    // the knobs as comments today; the check pins the values that matter).
    void validate_for_target(std::uint32_t hidden, std::uint32_t ngram_size,
                             std::uint32_t heads_per_ngram) const {
        const auto& lay = table_.layout();
        if (lay.n_heads * lay.embedding_row_dimension != hidden) {
            throw std::runtime_error(
                "PLE sidecar geometry n_heads*row_dim != hidden (" +
                std::to_string(lay.n_heads * lay.embedding_row_dimension) +
                " != " + std::to_string(hidden) + ")");
        }
        if (lay.ngram_size != ngram_size ||
            lay.heads_per_ngram != heads_per_ngram) {
            throw std::runtime_error("PLE sidecar ngram geometry mismatch");
        }
    }

    // Gathers the per-token PLE embedding into dst (device memory, BF16,
    // [n_heads * row_dim, n_tokens], token-major -- ple_table.h:50-56).
    // tokens: n_tokens ids (ctx0). prevs: (ngram_size-1)*n_tokens ids,
    // oldest-first per token. Blocks until the rows are resident, then
    // launches ple_gather_rows_kernel on `stream`.
    void gather_batch(std::span<const std::int32_t> tokens,
                      std::span<const std::int32_t> prevs, std::int32_t eos,
                      void* dst, cudaStream_t stream) const {
        if (prevs.size() !=
            static_cast<std::size_t>(table_.layout().ngram_size - 1) *
                tokens.size()) {
            throw std::invalid_argument(
                "PLE gather: prevs must hold (ngram_size-1)*n_tokens ids");
        }
        rows_.resize(tokens.size() * table_.layout().n_heads);
        table_.derive_rows(tokens, prevs, eos, rows_.data());
        table_.gather(rows_.data(), tokens.size(), dst, stream);
    }

private:
    // PleTable declares a destructor and deletes its copy constructor, so it has
    // neither a copy nor a move constructor: it cannot be received by value and
    // re-`std::move`d. It is therefore built in place from its options.
    explicit PleRuntime(ops::ple::PleTableOptions options) : table_(std::move(options)) {}

    // `mutable` because PleTable::gather is non-const (it bumps the pinned
    // cache's gather epoch and the forensics counters). The seam keeps const
    // gather entry points: towards the model a gather is logically read-only.
    mutable ops::ple::PleTable table_;
    mutable std::vector<std::int32_t> rows_; // host scratch, 16 ids/token
};

} // namespace ninfer::targets::qwen4_exp

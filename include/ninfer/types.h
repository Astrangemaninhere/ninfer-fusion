#pragma once

#include <chrono>
// After <chrono>, not first alphabetically: <chrono> already includes
// <array> here, so this line expands to nothing. Moving it earlier would
// reorder the preprocessed stream; MSVC does not reach <array> transitively.
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency               = 8;
inline constexpr std::size_t kMaximumContextCacheSessionKeyBytes = 256;
// Aggregate encoded image/video payload retained by one prompt, independent of item count.
inline constexpr std::size_t kMaximumPromptMediaBytes    = 256ULL << 20;
inline constexpr std::size_t kDefaultMediaCacheBytes     = 1ULL << 30;
inline constexpr std::size_t kDefaultMediaLiveBytes      = 2ULL << 30;
inline constexpr std::uint32_t kDefaultHostStateSlots    = 8;
inline constexpr std::size_t kDefaultHostKvCapacityBytes = 8ULL << 30;

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    Fp8E4M3Row256,
    Nvfp4Group16,
    Fp8Group16,
    Iso3Group16,
    // Packed 4-bit E8-lattice K + i4 V, per-64 FP16 scales (int8-kernel path).
    E8Group64,
    // L26 instrument: this layer's KV planes were DISCARDED (NINFER_KV_DROP_LAYERS),
    // so the pool holds no storage for it at all. Published where the resolved
    // per-layer table is reported, instead of the BF16 a default-constructed view
    // would otherwise show (a missing layer is not bf16). Appended last so every
    // existing enumerator keeps its numeric encoding.
    Dropped,
    // e8 family, narrower K-plane code widths: E8Group64's K code plane packed at 3 / 2
    // bits per element over the SAME per-64 FP16 scale plane. Appended AFTER Dropped for
    // the same reason Dropped was appended last -- every existing enumerator keeps its
    // numeric encoding, because E8Group64 == 6 and Dropped == 7 are what serialized state
    // and the per-layer table encode. Geometry: product/kv_e8_width.h.
    E8K3Group64,
    E8K2Group64,
    // INTEGRATE4 (landing order row 3): landed from sergiuszm/ninfer-4090 @ rtx4090-port,
    // whose include/ninfer/types.h:39 carries exactly this enumerator. It is the K/V codec
    // PAIR, not a new geometry: its resolved planes in src/core/paged_kv_storage.h:92-97 are
    // {DType::FP8_E4M3FN, 256, DType::FP16, 1} for K -- byte-for-byte what Fp8E4M3Row256
    // resolves to -- and {DType::U8, 128, DType::U8, 16} for V, byte-for-byte what
    // Nvfp4Group16 resolves to. This tree could not name that pair, and now it can.
    //
    // Appended LAST, deliberately, and NOT at the fork's position (which is between
    // Nvfp4Group16 and Fp8Group16). The convention is the one this enum already states twice
    // above: every existing enumerator keeps its numeric encoding. The fork's position would
    // renumber Fp8Group16(4), Iso3Group16(5), E8Group64(6), Dropped(7), E8K3Group64(8) and
    // E8K2Group64(9), and E8Group64 == 6 and Dropped == 7 are what serialized state and the
    // per-layer table encode. The enumerator NAME is what the consumer switches on; its
    // number is free, and only one of the two choices is free of consequence.
    Fp8KeyNvfp4Value,
    // INTEGRATE4 (landing order row 3, completing it): the remaining four enumerators
    // src/core/paged_kv_storage.h switches on. Measured, not read: landing only
    // Fp8KeyNvfp4Value took the consumer from 13 errors to 12, and the 12 name exactly these
    // four, at four switch sites (:103/:111/:112/:115, :141-:144, :153-:163). A single
    // enumerator would have been a half-landing INSIDE one consumer -- the same defect as a
    // half-landed name, one level in, and the compiler is what found it.
    //
    // Spelled as the fork spells them (RotatedInt8KeyInt4ValueGroup64, RotatedInt4KeyInt4ValueGroup64,
    // RK4V4E8, RK2V4E8). The last two are the ones this file's own provenance block warns about:
    // src/kvcfg/kv_formats.h carries Rk4v4 / Rk3v4 / Rk2v4 as LIVE TIERS with 13 tracked users,
    // so RK4V4E8-as-storage and Rk4v4-as-tier are two names two characters apart meaning two
    // different things. That pair is a danger-column entry (SPLIT-7 in the INTEGRATE4 report),
    // NOT a reason to re-spell: MERGE flag C2 was instantiated once already by renaming e8 to
    // rk4v4 in this family, and INTEGRATE2 reverted it on three measured grounds. Reproducing the
    // fork's spelling is the only choice that does not create a THIRD name for the same codec.
    // Appended last, in the fork's relative order, for the reason stated above.
    RotatedInt8KeyInt4ValueGroup64,
    RotatedInt4KeyInt4ValueGroup64,
    RK4V4E8,
    RK2V4E8,
};

// GDN recurrent SSM state storage precision.  Requested by the Flash-Next target only (its
// LoadPlan reads options.gdn_state_storage); every other target's LoadPlan ignores the field.
// Same spelling and same two enumerators as the donor tree.
enum class GdnStateStorage : std::uint8_t {
    FP32,
    BF16,
};

// Per-layer table width shared by targets that publish per-layer KV storage.
// Entries are indexed by full-attention layer order, not physical model layer.
// Family capacity: the smallest member has 16 full-attention layers; larger
// members index beyond that, so the table must cover the family maximum.
inline constexpr std::size_t kKvLayerStorageSlots = 64;

// SEPARATION: codec of the V plane of the NVFP4 KV tier. K is always E2M1
// there; V has always been ISO3 sign-magnitude INT3 (decoder_state.cpp
// kv_layer_v_dtype + the ISO3 V decode in gqa_attention_decode_nvfp4.cuh).
// Both codecs share the SAME plane geometry -- two codes per byte with one
// E4M3FN group scale per 16 channels -- so the choice is a pure decoder
// switch: no extra plane, no extra allocation, and no new kernel
// instantiation (the V=E2M1 shapes are already compiled for the prefill and
// partial-decode paths).
//
// E2M1 is offered as an ablation knob only. It is REFUSED when a mechanism
// that hardcodes ISO3 V is active, because those paths do not read v_dtype:
//   * a V residual plane on any NVFP4 layer
//     (gqa_attention_prefill_fill_nvfp4k_iso3v_kernel writes ISO3, and both
//      the prefill and decode producers decode the residual only on the
//      VVDType == ISO3 / Iso3V branch);
//   * the entropy cold pool for NVFP4 layers
//     (program_impl.h requantizes V with EntropyColdRequantMode::Iso4eVG16
//      unconditionally).
enum class KvVCodec : std::uint8_t {
    // Default: NVFP4 layers store V as ISO3 (byte-for-byte the pre-separation
    // engine). Plain Nvfp4Group16 layers already store V as E2M1 and are
    // unaffected by this knob.
    Iso3 = 0,
    // Ablation: NVFP4 layers store V as E2M1 (no rotation, /6 max).
    E2M1 = 1,
};

// Three readings of the K/V bit-width request (product/kv_kv_bits.h). There are TWO
// entry points and this names which one runs:
//   Joint   -- ONE overall ceiling for the whole KV stack ("合起来整体定").
//   Split   -- K and V each get a ceiling and EACH PLANE'S PER-LAYER LAYERING IS
//              SOLVED ON ITS OWN, then reconciled per layer against the (K format,
//              V format) pairs the engine can build ("分开定，内部分层"). Default.
//   Ceiling -- the split reading is undeliverable for some layer, so deploy
//              min(k,v) on both planes and report the headroom that could not be
//              spent. Opt-in BY NAME; never the default, because it merges two
//              requests into one number.
enum class KvBitsMode : std::uint8_t {
    Joint   = 0,
    Split   = 1,
    Ceiling = 2,
};

// How the prefill unit is chosen. `EngineOptions::prefill_chunk` is the CEILING either way: the
// workspace and the persistent buffers are sized for it at startup, so anything in
// [prefill_chunk_alignment, prefill_chunk] is safe to install while the engine runs. The mode
// decides who picks the value inside that range, which is the difference between the two operator
// surfaces ("prefill chunk 要能动态调整或者手动调整两种模式"):
//   Dynamic -- the bandwidth governor owns the unit. The engine installs
//              BandwidthGovernor::prefill_chunk_for(prefill_chunk) at every worker-loop boundary,
//              so the unit starts at the ceiling, shrinks while decode latency sits above its
//              measured noise floor, and recovers when it falls back. Default, and the behaviour
//              of every release before the mode had a name.
//   Manual  -- the unit is `prefill_chunk` ITSELF for the whole run. The engine installs nothing,
//              so nothing can move it: the governor is constructed disabled, reads no counters,
//              and the trace reports installed=0 with chunk == the value the caller asked for.
//              This is the mode for a caller who has measured a chunk and does not want it
//              re-derived behind their back.
// Both modes report the unit the engine installed (NINFER_FT_BW_TRACE=1), because a mode that
// cannot be told apart from the other from outside the process is not a mode.
enum class PrefillChunkMode : std::uint8_t {
    Dynamic = 0,
    Manual  = 1,
};

// Desensitized note about the most recent request-domain (lane-level) failure
// (docs/maintainer/engine-failure-recovery.md 3.2/3.5). Carries no exception
// text, so unauthenticated consumers (/health is auth-exempt) may surface it.
struct EngineRequestFailureNote {
    bool occurred            = false;
    std::uint64_t request_id = 0;
    std::string phase;  // "prefill" | "decode" | "control"
    std::chrono::system_clock::time_point at{};
};

// Engine health snapshot (W16). After the worker loop hits an engine-domain
// exception the engine rejects work with RequestErrorKind::Unavailable (serve
// returns 503) until recover() (3.3) respawns the worker. `reason` is the raw
// exception text for diagnostics and authenticated callers ONLY; `category` is
// the 3.1 class label ("resource" | "invariant") that is safe to expose on the
// auth-exempt /health endpoint. `last_request_error` records the latest
// request-domain failure the engine survived (3.2).
struct EngineFailureState {
    bool failed = false;
    std::string reason;
    std::chrono::system_clock::time_point since{};
    std::string category;
    EngineRequestFailureNote last_request_error;
};

enum class EnginePurpose : std::uint8_t {
    Generation,
    CausalScoring,
};

// Entropy-coded cold KV pool. Window compresses fully-written pages that
// are at least cold_keep_tokens behind the decode frontier; attention
// producers decode those pages inline from fixed-size slots.
enum class ColdPolicy : std::uint8_t {
    None,
    Window,
    Host,
    // Cold pages spill to per-layer disk files (compressed slots, one fixed
    // stride per slot); the device slot pool acts as a working set. The
    // engine keeps the file open for the process lifetime.
    Disk,
    // Layered cold tier -- "bound the offload memory, put the excess on SSD".
    // BOTH byte budgets are live at once: the pinned Host tier
    // (--cold-host-bytes) takes the READ-FREE pages while it has room and the
    // per-layer disk spill (--cold-disk-bytes) takes the overflow. The device
    // cold-slot pool is still reserved as the spill rung's working set, sized
    // exactly like Window/Disk (--max-cold-pages, else the keep-tokens
    // derivation). Which rung a page lands on is the admission ladder in
    // product/kv_cold_tier_budget.h; a page neither rung can take keeps its
    // device replica and is counted.
    //
    // Appended after Disk on purpose: this enum is carried as uint8_t, so every
    // pre-existing policy keeps its numeric value and any serialized or compared
    // form of it stays valid.
    HostThenDisk,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 1024ULL * 1024ULL * 1024ULL;

struct KvCapacityPolicy {
    KvCapacityMode mode                  = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens        = 2048;
    std::size_t automatic_headroom_bytes = 0;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens, 0};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy
    automatic(std::size_t headroom_bytes = kDefaultKvCapacityHeadroomBytes) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0, headroom_bytes};
    }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
    // Auto: resolved by profile in resolved_auto_speculative (a DFlash2 artifact
    // carries text/draft_head, so it maps to Optimized; other profiles stay Full).
    // It must be resolved there: that function's result feeds the planner, the load
    // plan and the program, otherwise the frozen-startup-features check rejects the
    // loaded weights (see the note in registry.cpp).
    Auto,
};

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
    DFlash2,
    Auto,
};

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    std::uint32_t draft_tokens = 0;
    // --draft-tree L,d: the MTP TREE verify shape. L = the number of rank-paths the draft side
    // keeps per depth, d = the tree depth in draft steps. One verify column is spent per tree
    // node on top of the anchor, and the tree's node budget L*d IS the round's draft extent, so
    // L*d <= kMtpDecodeMaximumDrafts. {0,0} = no tree: the scalar --draft-tokens path is
    // byte-identical to what it was before this field existed.
    std::uint32_t draft_tree_paths = 0;
    std::uint32_t draft_tree_depth = 0;
    ProposalHead proposal_head = ProposalHead::Auto;
};

struct LoadProgress {
    std::function<void(std::string_view phase, std::uint64_t done, std::uint64_t total)> callback;
};

struct ContextCacheOptions {
    // Engine resolves every optional once at construction. With C=max_concurrency, the enabled
    // defaults are H=C, R=8, Host KV=8 GiB, P=2C, S=C and L=2; Engine::options() returns those
    // effective values.
    bool enabled = true;
    // Extra Device checkpoint StateImage slots H. Total Device StateImage capacity is C + H.
    std::optional<std::uint32_t> device_state_slots;
    // Host StateImages and Host KV bytes are independently configured pinned-memory capacities.
    std::uint32_t host_state_slots     = kDefaultHostStateSlots;
    std::size_t host_kv_capacity_bytes = kDefaultHostKvCapacityBytes;
    // Bounded private/shared logical catalogs and per-continuation long-anchor count.
    std::optional<std::uint32_t> max_private_continuations;
    std::optional<std::uint32_t> max_shared_prefixes;
    std::optional<std::uint32_t> max_long_anchors_per_continuation;
};

struct ContextCostOptions {
    // Empty selects generic defaults plus any matching values compiled into the binary. A
    // nonempty runtime preset independently overrides its matching machine transfer and
    // artifact-prefill components; absent entries retain the preceding numerical layer.
    std::filesystem::path preset_path;
};

// ---------------------------------------------------------------------------------------------
// Flash-Next (qwen3.8-flash-next) port block -- startup observer and structured-output options.
//
// Both are ADDITIVE: this tree has no startup-progress publishing and no structured-output feature
// of its own, and nothing in the tree outside the copied target names either type.  They are taken
// verbatim from the donor so that the copied target and the donor's own tests compile unchanged.
// Donor: igorls/ninfer @ 5e4a66d include/ninfer/types.h:88-134, :270-274.
// ---------------------------------------------------------------------------------------------

enum class StartupPhase : std::uint8_t {
    EngineStartup,
    CudaInitialize,
    ArtifactInspect,
    TargetPlan,
    WeightsMaterialize,
    WeightsStagingPin,
    TargetFinalize,
    FrontendInitialize,
    ProgramInitialize,
    HostStatePin,
    HostKvPin,
    CudaGraphPrepare,
    EngineFinalize,
};

enum class StartupStatus : std::uint8_t {
    Begin,
    Progress,
    Complete,
    Failed,
};

enum class StartupProgressUnit : std::uint8_t {
    None,
    Bytes,
};

struct StartupEvent {
    StartupPhase phase                = StartupPhase::EngineStartup;
    StartupStatus status              = StartupStatus::Begin;
    StartupProgressUnit progress_unit = StartupProgressUnit::None;
    std::uint64_t current             = 0;
    std::uint64_t total               = 0;
    std::uint64_t elapsed_ns          = 0;
};

struct StartupObserver {
    // Startup diagnostics never participate in Engine control flow. Callback exceptions are
    // ignored by the publishing boundary so a logging failure cannot invalidate model startup.
    std::function<void(const StartupEvent& event)> callback;
};

enum class StructuredOutputKind : std::uint8_t { Text, JsonObject, JsonSchema };

struct StructuredOutputOptions {
    StructuredOutputKind kind = StructuredOutputKind::Text;
    // Owning serialized JSON Schema; empty for Text/JsonObject.
    std::string schema;
};

// `EngineOptions::unload_watermark_pages` sentinel: derive the reserve from the
// plan's own prefill chunk (see the field comment for why the chunk is the unit).
inline constexpr std::uint32_t kUnloadWatermarkDerive = 0xFFFFFFFFU;

struct EngineOptions {
    std::filesystem::path artifact_path;
    EnginePurpose purpose              = EnginePurpose::Generation;
    int device                         = 0;
    std::uint32_t max_context          = 2048; // Logical ceiling of one request or score window.
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 3072;
    // Which of the two modes above owns the prefill unit. nullopt == this caller named no mode,
    // which is not the same as naming Dynamic: normalize_engine_options() then defers to the
    // switch that predates the flag (NINFER_FT_BW_GOV, unset -> Dynamic) and writes the answer
    // back, so `Engine::options().prefill_chunk_mode` is never nullopt after construction and
    // describes the RUN rather than the request -- the same contract `prefill_chunk` itself has,
    // where the engine adopts the memory ladder's settled value into its own options.
    std::optional<PrefillChunkMode> prefill_chunk_mode;
    KvCacheStorage kv_cache            = KvCacheStorage::BFloat16;
    // True when kv_cache came from an explicit --kv-dtype. Without a per-layer table this makes
    // the global dtype win over the target's registered per-layer default table, so e.g.
    // --kv-dtype bf16 really selects a BF16 KV pool (see _TODO.md 97).
    bool kv_cache_explicit = false;
    // Per-layer KV storage override, indexed by full-attention layer order.
    // BFloat16 entries inherit kv_cache. Any non-BFloat16 entry replaces the
    // target's registered per-layer default table wholesale.
    // The table is FAMILY-WIDE (kKvLayerStorageSlots = 64, which covers the largest
    // member: muse_glimmer_30b has 52 full-attention layers, qwen3.6-27b has 16).
    // Only the slots BELOW the active target's full-attention layer count reach the
    // pool geometry; a slot at or past it parses, is accepted, and has no effect on
    // that model, which layouts_impl.h now reports as an inert-slot diagnostic.
    // CORRECTION (l26c): this comment used to claim such entries "are rejected".
    // They never were -- only an index >= kKvLayerStorageSlots is refused, in
    // product/kv_options.h, and that bound exists so the larger family members work.
    std::array<KvCacheStorage, kKvLayerStorageSlots> kv_layer_storage{};
    bool kv_layer_storage_explicit = false;
    // --kv-bit-budget <bits>: fractional bits/element target for the full-attention KV.
    // Resolved into kv_layer_storage by the parity-proven DP (product/kv_bit_budget.h,
    // _collab/A_kvbit_cold_cpp.md) at sequence-planner time, where the target's
    // full-attention layer count is known; parse time stores only the raw number
    // because no model knowledge exists there. Mutually exclusive with an explicit
    // kv_layer_storage (checked in serve_options.cpp).
    double kv_bit_budget_bits = 0.0;
    // Separable form of the same budget ("0-7:8,8-63:4.5"): distinct ceilings per layer
    // range, minimised per range (globally optimal because the objective is additive and
    // the constraints are per-range). Empty means the single-ceiling form above.
    std::string kv_bit_budget_ranges;
    // Two-score selection (speed vs quality). kv_quality_weight < 0 keeps the shipped
    // single-penalty ladder; >= 0 selects the weighted ladder, where the per-tier penalty is
    // w*quality + (1-w)*speed from the measured score table (see kv_bit_budget.h).
    double kv_quality_weight = -1.0;
    // Inline score table ("<tier> <quality_x100> <speed_x100>" lines) or a file path.
    std::string kv_tier_scores;
    bool kv_bit_budget_explicit = false;
    // ---- K/V bit widths: the TWO entry points of product/kv_kv_bits.h ----------
    // --kv-bits is the JOINT form ("合起来整体定": ONE overall ceiling for the whole
    // KV stack). --kv-k-bits/--kv-v-bits are the SPLIT form ("分开定，内部分层": each
    // plane's per-layer layering is solved in its own budget and then reconciled per
    // layer). 0.0 == not named. Mutually exclusive with the kv_bit_budget_* scalar /
    // range form above (checked in both parsers and in the planner).
    double kv_joint_bits      = 0.0;
    double kv_k_bits          = 0.0;
    double kv_v_bits          = 0.0;
    bool kv_kv_bits_explicit  = false;
    KvBitsMode kv_bits_mode      = KvBitsMode::Split;
    bool kv_bits_mode_explicit   = false;
    // Per-plane score tables for the split entry (same "<tier> <quality_x100>
    // <speed_x100>" grammar, or a file path). Empty == same table as kv_tier_scores.
    std::string kv_k_tier_scores;
    std::string kv_v_tier_scores;
    // --kv-codec-preference CODEC[,CODEC...] (SLIDERWIRE) -> the candidate preference the
    // JOINT K/V fit uses (product/kv_bit_budget.h KvGearSolveRequest::candidate_order):
    // WHICH codec the solver picks among candidates that cost the SAME bits, which is what
    // the slider is for -- "同 bit 分配不同种类的量化", not fewer bits. Ladder SLOTS, not
    // names, because the solver speaks slots; apps/cli/options.cpp validates the grammar by
    // name before anything reaches here, and product/kv_kv_bits.h refuses a preference the
    // fit could not honour. Empty (the default) is the shipped pack order, so every existing
    // path is byte-identical.
    std::vector<std::int32_t> kv_codec_preference;
    // --kv-score-table show|emit=<path>: the penalty table's OWN entry point. "show"
    // prints the table the planner would use plus its real provenance; "emit=<path>"
    // writes it in the grammar kv_bit_budget_parse_scores reads. Handled before any
    // device work, so it runs with no model at all.
    std::string kv_score_table_spec;
    bool kv_score_table_explicit = false;
    // Per-layer two-stage residual planes for the NVFP4 tier (second-stage
    // E2M1 K / ISO3 V over the first-stage error). Each enabled NVFP4 layer
    // doubles its plane count — volume parity with Int8Group64; other
    // storages ignore the table. Empty (default) keeps single-plane KV.
    std::array<bool, kKvLayerStorageSlots> kv_residual_layers{};
    bool kv_residual_explicit = false;
    // --kv-tier-formats SPEC: the KV tier vocabulary (hot/tail/cold + nvfp4 mode,
    // src/kvcfg/kv_formats.h). Stored raw because the landing is per-layer and the
    // layer count is target knowledge: make_sequence_planner_impl resolves it into
    // the same per-layer dtype table the knobs above feed (product/kv_tier_formats.h
    // documents which tiers the engine can and cannot express). Unset keeps every
    // existing path byte-identical.
    std::string kv_tier_formats_spec;
    bool kv_tier_formats_explicit = false;
    // --nvfp4-mode pure: stay off the fusion tiers (nvfp4/iso3/e8) for attribution.
    bool kv_nvfp4_pure = false;
    // --kv-v-codec iso3|e2m1: codec of the V plane on the NVFP4 tier (see the
    // KvVCodec comment above). Iso3 is the engine default and the only mode a
    // residual-plane or cold-pool NVFP4 layer accepts; e2m1 is an ablation.
    KvVCodec kv_v_codec      = KvVCodec::Iso3;
    bool kv_v_codec_explicit = false;
    // --kv-rotation on|off: SO(4) IsoQuant rotation of K (on cache write) and Q
    // (before quantization) on the NVFP4/FP8/ISO3 tiers. off is an ablation
    // that takes the identity map on BOTH sides, so QK^T stays exact and only
    // the quantization domain changes.
    bool kv_rotation_off = false;
    bool kv_rotation_explicit = false;
    // --kv-row-scale auto|off|<path>: Sinkhorn row scale applied on K write and
    // inverted on Q read. auto keeps the baked calibration table, off takes the
    // identity map (in the kernel, via the geometry descriptor's own guard),
    // <path> loads a NINFERKVRS1 sidecar.
    std::string kv_row_scale_spec;
    bool kv_row_scale_explicit = false;
    SpeculativeOptions speculative;
    std::size_t media_cache_bytes = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes  = kDefaultMediaLiveBytes;
    // Zero selects a bounded worker count from the detected host concurrency.
    std::uint32_t media_preprocess_threads = 0;
    bool enable_vision                     = false;
    bool use_cuda_graph                    = true;
    bool yarn_enabled                      = false;
    // On-demand graph capture: 0 (default) captures the full decode ladder at
    // startup; a positive value captures only segments fully below it and the
    // runtime extends the family on demand as the decode frontier grows past
    // each captured segment (one capture per growth crossing).
    std::uint32_t graph_capture_ceiling = 0;
    // Flash-Next only. GDN recurrent SSM state storage precision (FP32 default, BF16 for reduced
    // memory); ignored by every other target's LoadPlan.
    GdnStateStorage gdn_state_storage = GdnStateStorage::FP32;
    // Flash-Next only. Default off keeps the BF16 output head and byte-identical serving.
    bool quantize_output_head_fp8 = false;
    // Flash-Next only. Default off keeps the BF16 token embedding and byte-identical serving.
    bool quantize_token_embedding_fp8 = false;
    // Flash-Next only. Default off uses the scalar reduction QSA prefill path.
    bool use_qsa_prefill_mma = true;

    ContextCacheOptions context_cache;
    ContextCostOptions context_cost;
    ColdPolicy cold_policy            = ColdPolicy::None;
    std::uint32_t cold_keep_tokens    = 128;
    // Explicit cold-pool cap in pages (--max-cold-pages). 0 = derive from the
    // policy (Window/Disk/HostThenDisk: cold_keep_tokens/kPagedKVPageSize + 16;
    // others 0). ColdPolicy::None always wins over an explicit cap; Host keeps 0
    // because the Host tier's medium is pinned host memory and its pages are
    // read-free, so no device cold slot is involved at all (see layouts_impl.h
    // derivation + the cold_host_tier.h consumer). HostThenDisk keeps the pool
    // too: there it is the disk rung's working set, so an explicit cap is
    // meaningful and is NOT rejected (the contradiction check is Host-only).
    std::uint32_t max_cold_pages       = 0;
    // THE UNLOAD WATERMARK (--kv-unload-watermark-pages, overridable with
    // NINFER_KV_UNLOAD_WATERMARK_PAGES). Free text-KV pool pages at or below
    // which the Engine PROACTIVELY unloads the blocks its semantic directory
    // judges unloadable -- at the round boundary, with no request driving it and
    // with no dependence on the pool actually overflowing.
    //
    // This is the user's design order of 2026-09-15, verbatim: "达到显存剩下多少
    // 的时候主动开始总结压缩而不是等到溢出了". So the trigger is a WATERMARK and
    // not a spill: the old shape (pool full -> evict -> spill) is the passive one
    // this replaces, and it is why the default is non-zero rather than "as before".
    //
    // 0 = OFF, and OFF is byte-for-byte the pre-watermark behaviour (the only
    // other consumer of the pool, enqueue_cold_compressions, still runs exactly
    // when --cold-policy says it does). kUnloadWatermarkDerive = "derive the
    // reserve from plan.prefill_chunk", which is the largest unit of work the
    // Engine must be able to PLACE without first freeing anything: a chunk's KV
    // cannot be written half-way, so a reserve below one chunk would make the
    // watermark fire too late to be proactive.
    std::uint32_t unload_watermark_pages = kUnloadWatermarkDerive;
    // Pinned host-memory budget for the host rung of the cold ladder (tier 1:
    // host first, then the disk tier). Default 7 GiB, not 4: the 1M band needs
    // F - D = 5,634 pages = 6.2004 GiB at 1,181,745 B/page (6.9513 GiB on the
    // rANS record), so 4 GiB leaves cold_host_window_band EMPTY and 7 GiB is
    // the safe round-up. This is the cap the tier's
    // admission enforces, in whole pages (host_bytes / page_stride); it is a
    // separate pool from context_cache.host_kv_capacity_bytes. ColdPolicy::Host
    // stops at this rung; ColdPolicy::HostThenDisk is the policy that makes the
    // second rung reachable, so here the cap is a hard ceiling on pinned memory
    // and everything past it spills instead of staying hot.
    std::uint64_t cold_host_bytes     = 7ULL << 30;
    // ColdPolicy::Disk | HostThenDisk: directory for per-layer cold spill files
    // (created on demand) and the total spill budget. Empty path uses the system
    // temp dir. Under HostThenDisk this bounds only the overflow rung.
    std::string cold_disk_path;
    std::uint64_t cold_disk_bytes     = 32ULL << 30;
    // W13 weight host-offload budget (bytes of weight payload that may leave the
    // device arena). DELIBERATELY a separate budget from cold_host_bytes (KV cold
    // tier): different lifetimes and failure modes. 0 = off. See
    // product/weight_residency.h and scratch/w13a/DESIGN.md.
    std::uint64_t weight_host_offload_bytes = 0;
    // W13 device side: the working set kept on device for the OFFLOADED layers.
    // 0 = derive from weight_prefetch_layers. What W13 actually frees is
    // `offloaded weight bytes - this`, so this is the number that buys context.
    std::uint64_t weight_device_arena_bytes = 0;
    // W13 look-ahead, in layer strips. Below 2 the arena slot of the layer being
    // computed could be overwritten by its own prefetch, so it is rejected.
    std::uint32_t weight_prefetch_layers = 2;
    // W13 span floor in bytes; 0 = derive from the artifact (1 MiB, or 1/4096 of
    // the device weight arena, whichever is larger).
    std::uint64_t weight_span_floor_bytes = 0;
    LoadProgress load_progress;
    // WHICH slots the per-layer spec actually WROTE, for kv_layer_storage above.
    // BFloat16 doubles as the "unset" sentinel in that table, so without this mask
    // an explicit `--kv-layer-storage 0-11:bf16` is indistinguishable from an empty
    // spec and layers 0..11 keep INHERITING the global --kv-dtype. That was the
    // last silent downgrade of this class: the operator asked for BF16 on those
    // layers and got a quantized tier.
    //
    // Semantics, and they are load-bearing:
    //   set[L] == true  -> layer L's storage is the entry in kv_layer_storage[L],
    //                      INCLUDING BFloat16 (a real per-layer BF16 request);
    //   set[L] == false -> layer L is resolved exactly as before (a BFloat16
    //                      entry inherits the global --kv-dtype). Every slot
    //                      defaults to false, so a caller that says nothing
    //                      behaves bit-for-bit as it did before this mask existed.
    // The rule is applied at the one place that commits the per-layer table to the
    // page pool: ninfer::product::kv_resolve_slot_dtype()
    // (product/kv_component_switch.h), called from plan_cache()
    // (targets/qwen3_6/impl/state/decoder_state.cpp).
    //
    // LAST MEMBER ON PURPOSE: a field appended at the end cannot move any existing
    // field's offset, so an object compiled before this change still reads every
    // field it knows about where it expects it. That keeps the "default all-false
    // means nothing changed" claim checkable against binaries built from the
    // previous revision (objdump-able, and it survives the incremental-rebuild
    // hazard the acceptance notes warn about).
    std::array<bool, kKvLayerStorageSlots> kv_layer_storage_set{};
};

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 20;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ThinkingControlOptions {
    // Positive maximum accepted model-origin tokens while the Qwen thinking phase remains open.
    // Omitted means unlimited. Injected target-control tokens consume the total output budget but
    // not this model-origin budget.
    std::optional<std::uint32_t> budget;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
    // Presentation constraint supplied by the protocol adapter. It bounds only Qwen's emitted
    // function-name grammar; it does not require the name to match a currently declared tool.
    std::uint32_t tool_name_max_length = 128;
};

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

enum class ImageResizePolicy : std::uint8_t {
    Downsize,
    RejectOversized,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
    ImageResizePolicy image_resize_policy = ImageResizePolicy::Downsize;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Model-origin structured output. Protocol adapters own any wire-level call identifier.
struct GeneratedToolCall {
    std::string name;
    std::string arguments_json;
};

// Wire-independent conversation authority. Protocol adapters preserve these roles and their
// ordering; a target frontend owns any model-specific role lowering.
enum class ChatRole : std::uint8_t {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    Low,
    Medium,
    XHigh,
};

struct ReasoningEffortCapabilities {
    bool low    = false;
    bool medium = false;
    bool xhigh  = false;
    std::optional<ReasoningEffort> default_effort;

    [[nodiscard]] constexpr bool supports(ReasoningEffort effort) const noexcept {
        switch (effort) {
        case ReasoningEffort::Low:
            return low;
        case ReasoningEffort::Medium:
            return medium;
        case ReasoningEffort::XHigh:
            return xhigh;
        }
        return false;
    }
};

struct PromptCapabilities {
    bool enable_thinking = false;
    ReasoningEffortCapabilities reasoning_effort;
};

enum class PromptContinuationMode : std::uint8_t {
    NewAssistantTurn,
    ContinueFinalAssistant,
};

struct PromptOptions {
    PromptContinuationMode continuation = PromptContinuationMode::NewAssistantTurn;
    bool enable_thinking                = true;
    std::optional<ReasoningEffort> reasoning_effort;
    bool preserve_thinking = false;
    bool add_vision_id     = false;
    std::vector<std::string> tool_jsons;
};

enum class CacheRetentionHint : std::uint8_t {
    Default,
    LiveSession,
    Disposable,
};

enum class PromptCacheMarkerKind : std::uint8_t {
    SharedStablePrefix,
    PrivateLongAnchor,
};

enum class SharedCandidateEvidence : std::uint8_t {
    None               = 0,
    ExplicitBoundary   = 1U << 0U,
    RequestedAutomatic = 1U << 1U,
    DefaultAutomatic   = 1U << 2U,
    EngineStructural   = 1U << 3U,
    EngineObserved     = 1U << 4U,
};

[[nodiscard]] constexpr SharedCandidateEvidence operator|(SharedCandidateEvidence left,
                                                          SharedCandidateEvidence right) noexcept {
    return static_cast<SharedCandidateEvidence>(static_cast<std::uint8_t>(left) |
                                                static_cast<std::uint8_t>(right));
}

constexpr SharedCandidateEvidence& operator|=(SharedCandidateEvidence& left,
                                              SharedCandidateEvidence right) noexcept {
    left = left | right;
    return left;
}

[[nodiscard]] constexpr bool has_shared_candidate_evidence(SharedCandidateEvidence value,
                                                           SharedCandidateEvidence evidence) {
    return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(evidence)) != 0;
}

enum class PromptCacheMarkerLocation : std::uint8_t {
    MessageBoundary,
    MessagePartBoundary,
    LeadingInstructionBoundary,
    ToolBoundary,
};

struct PromptCacheMarker {
    std::uint32_t after_message_count  = 0;
    PromptCacheMarkerKind kind         = PromptCacheMarkerKind::SharedStablePrefix;
    SharedCandidateEvidence evidence   = SharedCandidateEvidence::ExplicitBoundary;
    PromptCacheMarkerLocation location = PromptCacheMarkerLocation::MessageBoundary;
    // Byte count within the untrimmed leading System/Developer message.
    std::uint32_t leading_instruction_bytes = 0;
    std::uint32_t after_tool_count          = 0;
    // For MessagePartBoundary, after_message_count identifies the containing message using a
    // one-based count and this value identifies the number of serialized parts within it.
    std::uint32_t after_message_part_count = 0;

    [[nodiscard]] friend constexpr bool operator==(PromptCacheMarker,
                                                   PromptCacheMarker) noexcept = default;
};

struct ContextCacheHints {
    std::optional<std::string> session_key;
    CacheRetentionHint retention = CacheRetentionHint::Default;
    std::vector<PromptCacheMarker> markers;
    // Protocols with their own automatic/explicit write policy disable the Engine's structural
    // candidates. Exact reads from already-published shared prefixes remain enabled.
    bool allow_engine_automatic_shared_prefixes = true;
    // Advance the named session lineage when session_key is present. This does not require an
    // anonymous content-matched source to be retained.
    bool update_session_index = true;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
    ContextCacheHints context_cache;
};

enum class RequestErrorKind : std::uint8_t {
    ContextLengthExceeded,
    ThinkingBudgetCapacityInsufficient,
    MediaBudgetExceeded,
    InvalidMedia,
    Overloaded,
    QueueTimeout,
    Cancelled,
    Unavailable,
    // S35 (3.4): request-shaped prompt/sampling validations raised by the planner.
    InvalidPrompt,
};

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

private:
    RequestErrorKind kind_;
};

struct PromptSummary {
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
};

struct PromptPreparationStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
};

struct MediaCacheSummary {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

// Exact prompt accounting selected at admission. Streaming consumers receive this once before any
// OutputDelta, after the prefix choice and materialization reservation are committed and before
// transfer/prefill execution.
struct GenerationStart {
    PromptSummary prompt;
    std::uint32_t reused_prompt_tokens = 0;
};

class OutputSink {
public:
    virtual ~OutputSink()                     = default;
    virtual void start(GenerationStart start) = 0;
    virtual void publish(OutputDelta delta)   = 0;
};

enum class OutputConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

// Deadline and cancellation apply to all host-side prompt preparation work. Empty values mean
// unbounded preparation.
struct PreparationControl {
    std::chrono::steady_clock::time_point deadline;
    CancellationView cancellation;
};

// Request-stage wall timings retained for end-to-end latency/rate reporting. Prefill/decode are
// Program execution elapsed time and include Device completion waits; total also includes queueing
// and other request lifetime. They are not Host-work phases. GenerationEngineTiming below is the
// direct, mutually-exclusive Host observation contract.
struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    double decode_seconds      = 0.0;
    double total_seconds       = 0.0;
};

// Wall elapsed time directly observed in Engine-owned regions. "Exposed" values are latency
// exposure: every active request delayed by one compact-batch unit observes that unit's full
// elapsed time, so values from concurrent requests must not be summed. Device wait is reported
// separately from Host-active work.
struct GenerationEngineTiming {
    double queue_wait_seconds                   = 0.0;
    double engine_boundary_exposed_seconds      = 0.0;
    double program_submit_exposed_seconds       = 0.0;
    double program_post_exposed_seconds         = 0.0;
    double engine_commit_output_exposed_seconds = 0.0;
    double engine_maintenance_exposed_seconds   = 0.0;
    double device_wait_exposed_seconds          = 0.0;
    double decode_host_exposed_seconds          = 0.0;
    double decode_device_wait_exposed_seconds   = 0.0;
    std::uint64_t prefill_units                 = 0;
    std::uint64_t decode_rounds                 = 0;
    std::uint64_t control_units                 = 0;
};

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::vector<std::uint64_t> accepted_per_position;
    // True when the draft window was chosen adaptively (the MTP capture ladder plus the
    // survival/cost criterion) instead of being pinned by --draft-tokens. `draft_window` then
    // reports the ladder TOP (the widest captured rung), not a fixed k.
    bool adaptive_window = false;
    // Realized mean width the criterion actually ran at, over the rounds this request drafted
    // (0 when not adaptive). The open +-1 window discrepancy in mtp_window_cut.h needs this
    // number next to the acceptance rate to be judged.
    double mean_window = 0.0;
};

struct ThinkingBudgetStats {
    std::optional<std::uint32_t> configured_budget;
    // Model-origin tokens accepted while capped thinking remained open.
    std::uint32_t model_thinking_tokens = 0;
    // Complete tokenizer-derived target-control suffix committed by Engine.
    std::uint32_t injected_tokens = 0;
    bool applied                  = false;
};

enum class PrefixReusePath : std::uint8_t {
    Root,
    PrivateEndpoint,
    PrivateTurnClosure,
    PrivateResponseReplay,
    PrivateLongAnchor,
    SharedStablePrefix,
};

// Why pressure planning stopped for the materialization decision committed to one request.
// "ModelOptimal" is relative to the configured target graph, canonical transaction order, and
// numerical cost model; it is not a claim about globally optimal observed TTFT.
enum class MaterializationStopReason : std::uint8_t {
    NoPressure,
    ModelOptimal,
    QueueExhausted,
    TargetBudget,
    ExpansionCapacity,
    TimeBudget,
    ValueOfNextExpansion,
};

[[nodiscard]] inline constexpr const char*
materialization_stop_reason_name(MaterializationStopReason reason) noexcept {
    switch (reason) {
    case MaterializationStopReason::NoPressure:
        return "no_pressure";
    case MaterializationStopReason::ModelOptimal:
        return "model_optimal";
    case MaterializationStopReason::QueueExhausted:
        return "queue_exhausted";
    case MaterializationStopReason::TargetBudget:
        return "target_budget";
    case MaterializationStopReason::ExpansionCapacity:
        return "expansion_capacity";
    case MaterializationStopReason::TimeBudget:
        return "time_budget";
    case MaterializationStopReason::ValueOfNextExpansion:
        return "value_of_next_expansion";
    }
    return "no_pressure";
}

struct MaterializationDiagnostics {
    std::uint64_t predicted_now_ns              = 0;
    std::uint64_t predicted_future_loss_ns      = 0;
    std::uint64_t predicted_total_ns            = 0;
    std::uint32_t targets_evaluated             = 0;
    std::uint64_t projection_work               = 0;
    std::uint64_t planning_elapsed_ns           = 0;
    std::uint64_t search_elapsed_ns             = 0;
    MaterializationStopReason stop_reason       = MaterializationStopReason::NoPressure;
    bool model_optimal                          = true;
    bool budget_exhausted                       = false;
    std::uint64_t best_remaining_lower_bound_ns = 0;
    std::uint64_t absolute_bound_gap_ns         = 0;
    double relative_bound_gap                   = 0.0;
    std::uint32_t selected_degradation_units    = 0;
    bool selected_maximal_fallback              = false;

    [[nodiscard]] friend constexpr bool
    operator==(const MaterializationDiagnostics&,
               const MaterializationDiagnostics&) noexcept = default;
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::vector<GeneratedToolCall> tool_calls;
    std::uint32_t reasoning_tokens = 0;
    FinishReason finish_reason     = FinishReason::None;
    std::optional<std::string> matched_stop_string;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;
    MaterializationDiagnostics materialization;
    GenerationTimings timings;
    GenerationEngineTiming engine_timing;
    SpeculativeStats speculative;
    ThinkingBudgetStats thinking;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

// Logical regions within the one physical workspace allocation. These byte values describe
// layout and live extents and must not be added to workspace.capacity_bytes.
struct VisionWorkspaceMemorySummary {
    std::uint32_t aggregate_prompt_tokens = 0;
    std::uint32_t max_item_tokens         = 0;
    std::size_t general_capacity_bytes    = 0;
    std::size_t encode_peak_bytes         = 0;
    std::size_t handoff_offset_bytes      = 0;
    std::size_t handoff_capacity_bytes    = 0;
    std::size_t handoff_active_bytes      = 0;
    std::size_t handoff_peak_bytes        = 0;
};

struct MemorySummary {
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    // The RESOLVED per-layer KV storage, indexed by full-attention layer order, as the
    // page pool was actually built from it. `kv_cache` above is only the global dtype a
    // BF16 table entry inherits: qwen3.6-27b's registered default table puts E8 on
    // layers {0,1,3,4,6,7} and NVFP4 on the rest, and reading `kv_cache` alone then
    // describes a store the run is not using
    // (targets/qwen3_6_27b/impl/variant.cpp default_layer_kv_dtypes).
    std::array<KvCacheStorage, kKvLayerStorageSlots> kv_layer_storage{};
    std::uint32_t kv_full_attention_layers = 0;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    std::optional<VisionWorkspaceMemorySummary> vision_workspace;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t kv_capacity_headroom_bytes        = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    std::size_t kv_payload_bytes                  = 0;
    std::uint32_t host_state_capacity_slots       = 0;
    std::uint32_t host_state_occupied_slots       = 0;
    std::size_t host_kv_capacity_bytes            = 0;
    std::size_t host_kv_occupied_bytes            = 0;
};

// Worker-owned monotonic nanosecond counters. Top-level Host phases are mutually exclusive;
// device_wait_ns is blocked wall time and is intentionally excluded from their sum. Detail values
// are subsets of a top-level phase and must not be added to Host-active time again.
struct RuntimeHostWorkStats {
    std::uint64_t engine_boundary_ns      = 0;
    std::uint64_t program_submit_ns       = 0;
    std::uint64_t program_post_ns         = 0;
    std::uint64_t engine_commit_output_ns = 0;
    std::uint64_t engine_maintenance_ns   = 0;
    std::uint64_t device_wait_ns          = 0;

    std::uint64_t decode_host_ns         = 0;
    std::uint64_t decode_device_wait_ns  = 0;
    std::uint64_t prefill_host_ns        = 0;
    std::uint64_t prefill_device_wait_ns = 0;
    std::uint64_t control_host_ns        = 0;
    std::uint64_t control_device_wait_ns = 0;
    std::uint64_t prefill_units          = 0;
    std::uint64_t control_units          = 0;

    std::uint64_t admission_policy_ns           = 0;
    std::uint64_t context_progress_ns           = 0;
    std::uint64_t stats_publication_ns          = 0;
    std::uint64_t admission_policy_invocations  = 0;
    std::uint64_t context_progress_invocations  = 0;
    std::uint64_t stats_publication_invocations = 0;
};

// Monotonic execution counters, boundary-consistent current gauges, and explicitly named last
// decision observations. Consumers derive interval counters by subtracting two snapshots.
struct RuntimeStats {
    RuntimeHostWorkStats host_work;
    // Actual prompt tokens evaluated by prefill; reused checkpoint-prefix tokens are excluded.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds             = 0;
    std::uint64_t decode_row_rounds         = 0;
    std::uint32_t running_requests          = 0;
    std::uint32_t prefilling_requests       = 0;
    std::uint32_t decode_ready_requests     = 0;
    std::uint32_t waiting_requests          = 0;
    std::uint32_t materializing_requests    = 0;
    std::uint32_t capture_pending_requests  = 0;
    std::uint32_t terminal_pending_requests = 0;
    std::uint64_t active_captures_completed = 0;
    std::uint64_t active_captures_aborted   = 0;

    std::uint64_t root_selections                    = 0;
    std::uint64_t private_endpoint_selections        = 0;
    std::uint64_t private_turn_closure_selections    = 0;
    std::uint64_t private_response_replay_selections = 0;
    std::uint64_t private_long_anchor_selections     = 0;
    std::uint64_t shared_stable_prefix_selections    = 0;
    std::uint64_t reused_prompt_tokens               = 0;
    std::uint32_t last_selected_frontier_tokens      = 0;

    std::uint64_t state_moves     = 0;
    std::uint64_t state_forks     = 0;
    std::uint64_t state_restores  = 0;
    std::uint64_t state_d2h_count = 0;
    std::uint64_t state_h2d_count = 0;
    std::uint64_t state_d2d_count = 0;
    std::uint64_t state_d2h_bytes = 0;
    std::uint64_t state_h2d_bytes = 0;
    std::uint64_t state_d2d_bytes = 0;
    double state_d2h_seconds      = 0.0;
    double state_h2d_seconds      = 0.0;
    double state_d2d_seconds      = 0.0;

    std::uint64_t main_kv_d2h_pages    = 0;
    std::uint64_t main_kv_h2d_pages    = 0;
    std::uint64_t main_kv_d2d_pages    = 0;
    std::uint64_t main_kv_d2h_bytes    = 0;
    std::uint64_t main_kv_h2d_bytes    = 0;
    std::uint64_t main_kv_d2d_bytes    = 0;
    double main_kv_d2h_seconds         = 0.0;
    double main_kv_h2d_seconds         = 0.0;
    double main_kv_d2d_seconds         = 0.0;
    std::uint64_t backend_kv_d2h_pages = 0;
    std::uint64_t backend_kv_h2d_pages = 0;
    std::uint64_t backend_kv_d2d_pages = 0;
    std::uint64_t backend_kv_d2h_bytes = 0;
    std::uint64_t backend_kv_h2d_bytes = 0;
    std::uint64_t backend_kv_d2d_bytes = 0;
    double backend_kv_d2h_seconds      = 0.0;
    double backend_kv_h2d_seconds      = 0.0;
    double backend_kv_d2d_seconds      = 0.0;

    std::uint64_t pressure_spill_pages                 = 0;
    std::uint64_t partial_tail_cow_pages               = 0;
    std::uint32_t device_state_occupied_slots          = 0;
    std::uint32_t host_state_occupied_slots            = 0;
    std::uint32_t device_main_kv_occupied_pages        = 0;
    std::uint32_t device_backend_kv_occupied_pages     = 0;
    std::size_t host_kv_occupied_bytes                 = 0;
    std::uint64_t pressure_private_owners_degraded     = 0;
    std::uint64_t pressure_private_owners_evicted      = 0;
    std::uint64_t pressure_shared_owners_degraded      = 0;
    std::uint64_t pressure_shared_owners_evicted       = 0;
    std::uint64_t pressure_checkpoints_dropped         = 0;
    std::uint64_t pressure_searches                    = 0;
    std::uint64_t pressure_search_budget_exhaustions   = 0;
    std::uint64_t pressure_maximal_fallback_selections = 0;
    std::uint32_t shared_active_references             = 0;
    std::uint64_t historical_fork_hits                 = 0;
    double actual_context_transfer_seconds             = 0.0;
    // mtplogx: cumulative observation of on-demand MTP rung capture. extend_mtp_graphs() is a
    // sub-second event, so a 1 Hz gauge can sit either side of it and report nothing; a monotone
    // pair can be differenced across any two later samples. No value that reaches graph capture
    // reads these.
    std::uint64_t mtp_graph_extension_calls       = 0;
    std::uint64_t mtp_graph_extension_nanoseconds = 0;
};

enum class ContextCostPresetSource : std::uint8_t {
    GenericDefault,
    CompiledDefault,
    // Compiled coefficients measured for another model in the same weight format, not for this
    // one. Reported apart from CompiledDefault so a reader can tell a borrowed profile from a
    // measured one.
    CompiledWeightsFallback,
    External,
};

[[nodiscard]] inline constexpr const char*
context_cost_preset_source_name(ContextCostPresetSource source) noexcept {
    switch (source) {
    case ContextCostPresetSource::GenericDefault:
        return "generic-default";
    case ContextCostPresetSource::CompiledDefault:
        return "compiled-default";
    case ContextCostPresetSource::CompiledWeightsFallback:
        return "compiled-weights-fallback";
    case ContextCostPresetSource::External:
        return "external";
    }
    return "unknown";
}

struct ContextCostSummary {
    ContextCostPresetSource transfer_source = ContextCostPresetSource::GenericDefault;
    ContextCostPresetSource prefill_source  = ContextCostPresetSource::GenericDefault;
    std::string hardware_class;
    std::string model_id;
    std::string weights_id;
    std::filesystem::path preset_path;
};

struct LoadSummary {
    std::string target;
    std::string model_id;
    std::string weights_id;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::size_t tensor_count           = 0;
    std::size_t resource_count         = 0;
    ContextCostSummary context_cost;
};

} // namespace ninfer

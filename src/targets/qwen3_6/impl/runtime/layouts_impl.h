#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"
#include "targets/qwen3_6/impl/runtime/mtp_tree_proposal_fill.h"

#include "core/device.h"
#include "core/device_capabilities.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_round.h"
#include "product/kv_bit_budget.h"
#include "product/kv_cold_tier_budget.h"
#include "product/kv_paging_preallocation.h"   // RECALLSPAN: the paging-set guard
#include <cstdlib>
#include <cstring>
#include <stdexcept>
// product::kv_resolve_slot_dtype() -- "a slot the spec WROTE as bf16 is a real BF16
// layer, an unwritten one inherits the global dtype" -- is used by the tier-class loop
// below. It MUST be included here and not only in decoder_state.cpp: layouts_impl.h is
// a template instantiated once per target variant (src/targets/*/impl/variant.cpp), and
// those TUs do not include kv_component_switch.h themselves. Without this include the
// qwen3_6 variant still compiles but the qwen3_6_27b / qwen3_6_35b_a3b /
// muse_glimmer_30b / qwen3_5_9b instantiations fail with
// "'kv_resolve_slot_dtype' is not a member of 'ninfer::product'".
#include "product/kv_component_switch.h"
#include "product/kv_kv_bits.h"
#include "product/kv_options.h"
#include "product/kv_storage_dtype.h"
#include "product/kv_tier_formats.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;

enum class GdnWorkspacePath : std::uint8_t {
    Prefill,
    Snapshot,
    ReplayRecord,
};

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

struct TargetKVCacheProfile {
    DType dtype;
    std::int32_t quant_group;
};

TargetKVCacheProfile target_kv_cache_profile(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return {DType::BF16, 0};
    case KvCacheStorage::Int8Group64:
        return {DType::I8, qwen3_6::kKvInt8QuantGroup};
    case KvCacheStorage::Fp8E4M3Row256:
    // BOTH fp8 spellings name the SAME target tier, and the tree already says so in
    // three places: src/product/kv_storage_dtype.h:48-54, this file's own note at
    // :1512-1518 ("target_kv_cache_profile() maps both to ..."), and
    // core/device_capabilities.h:203-204, which groups them under one capability set.
    // product::parse_kv_storage emits Fp8Group16 for a per-layer `fp8` spec
    // (src/product/kv_options.h:46).  Only Fp8E4M3Row256 was named here, so the
    // Fp8Group16 spelling fell past the switch into the trailing "unknown KV-cache
    // storage profile" throw -- the comments claimed a mapping the code lacked.
    case KvCacheStorage::Fp8Group16:
        return {DType::FP8_E4M3FN, qwen3_6::kKvFp8QuantGroup};
    case KvCacheStorage::Nvfp4Group16:
        return {DType::NVFP4, qwen3_6::kNvfp4KvQuantGroup};
    // ISO4E is a tier of its own: identical planes to NVFP4 but a distinct
    // sign-magnitude nibble codec (4 bit: bit3 sign + magnitude 0..7, scale
    // amax/7 -- NOT the 3-bit reference codec, see
    // src/ops/kernel/gqa_iso3_codec.cuh), so the resolved dtype must stay ISO4E
    // and the K/V decode kernel is the ISO4E one (src/ops/kernel/gqa_attention_decode_iso3.cuh).
    // An earlier revision of these two lines named gqa_iso4e_codec.cuh and
    // gqa_attention_decode_iso4e.cuh; NEITHER of those files is in this tree -- `gqa_iso4e_*` is
    // the deferred target spelling, and the files that exist are the two iso3 paths above.
    case KvCacheStorage::Iso3Group16:
        return {DType::ISO3, qwen3_6::kNvfp4KvQuantGroup};
    case KvCacheStorage::E8Group64:
        return {DType::E8Kv, qwen3_6::kKvInt8QuantGroup};
    case KvCacheStorage::E8K3Group64:
    case KvCacheStorage::E8K2Group64:
        // A defined rk4v4 tier with no codec in this tree: same refusal as Dropped below,
        // and for a reason this function can state from its own arguments -- it returns
        // a (DType, quant group) PAIR, and there is no quant group that makes a 3-bit or
        // 2-bit K plate readable by the nibble reader the rk4v4 profile selects
        // (qwen3_6::kKvInt8QuantGroup above, whose code_extent is head_dim/2).
        throw std::invalid_argument(
            "an rk3v4/rk2v4 layer has no storage profile: the tier's K plane is 3-bit or "
            "2-bit and no reader for it exists, so it can be neither planned nor built. "
            "See product/kv_e8_width.h; use rk4v4 for a 4-bit K plane.");
    case KvCacheStorage::Dropped:
        // L26 instrument: a discarded layer has no storage profile to build. Refuse
        // rather than fold it into BF16, which is what a default case would do.
        throw std::invalid_argument(
            "a discarded KV layer has no storage profile (NINFER_KV_DROP_LAYERS)");
    }
    throw std::invalid_argument("unknown KV-cache storage profile");
}

template <class ProfileAllowance>
std::size_t graph_topology_allowance(const std::vector<GraphExecutionProfile>& profiles,
                                     ProfileAllowance&& profile_allowance, const char* label) {
    std::vector<std::pair<std::uint32_t, std::size_t>> classes;
    for (const GraphExecutionProfile profile : profiles) {
        const std::size_t allowance = profile_allowance(profile);
        const auto existing = std::find_if(classes.begin(), classes.end(), [&](const auto& entry) {
            return entry.first == profile.topology_class;
        });
        if (existing == classes.end()) {
            classes.emplace_back(profile.topology_class, allowance);
        } else {
            existing->second = std::max(existing->second, allowance);
        }
    }

    std::size_t total = 0;
    for (const auto& [topology_class, allowance] : classes) {
        (void)topology_class;
        total = checked_add(total, allowance, label);
    }
    return total;
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

// S30: single source of truth for the effective cold-pool size (S25 precedence).
// ColdPolicy::None always forces 0 (no cold path; slots would be dead device
// memory). ColdPolicy::Host keeps 0 as well, and for a better reason than "the
// consumer is missing": the Host tier's medium is pinned host memory, its pages
// are READ-FREE (no kernel ever reads them again), so it needs no device cold
// slot at all -- see cold_host_tier.h. Window/Disk take the explicit
// --max-cold-pages cap when non-zero, else the keep-tokens derivation.
// HostThenDisk takes the same pool: its host rung is read-free (no slot), but its
// disk rung IS a slot tier -- the spill holds the compressed slot and decode reads
// it back from the device cold slot -- so the pool is that rung's working set and
// must be reserved. Used by BOTH the layout reservation and the --kv-bit-budget DP
// so the DP's cold capacity always equals the reserved pool.
[[nodiscard]] inline std::uint32_t effective_cold_pages(ColdPolicy policy,
                                                        std::uint32_t keep_tokens,
                                                        std::uint32_t explicit_pages) {
    // The device cold-slot pool is the *slot* tiers' resource. Under the Host
    // policy a non-zero --max-cold-pages would reserve device slots the Host tier
    // never uses (it may not: its pages are the ones nothing reads, so there is no
    // slot to read them from), and --cold-host-bytes is the knob that bounds that
    // tier. Asking for both is a contradiction; make it loud instead of quietly
    // reserving a pool that stays empty.
    if (policy == ColdPolicy::Host && explicit_pages != 0) {
        throw std::invalid_argument(
            "cold policy 'host' uses pinned host memory, not the device cold-slot pool, "
            "so --max-cold-pages cannot be honoured; bound the tier with --cold-host-bytes "
            "or use --cold-policy window|disk");
    }
    switch (policy) {
    case ColdPolicy::Window:
    case ColdPolicy::Disk:
    // HostThenDisk sizes its pool exactly like the two above. Only its BACKING
    // store is layered: the first rung releases the device page into pinned host
    // memory, the second keeps the compressed slot on disk with the device pool as
    // the working set. The ladder itself (which rung a page takes) lives in
    // product/kv_cold_tier_budget.h and needs no arithmetic from here.
    case ColdPolicy::HostThenDisk:
        return explicit_pages != 0
                   ? explicit_pages
                   : static_cast<std::uint32_t>(keep_tokens / kPagedKVPageSize + 16);
    default:
        return 0;
    }
}

PersistentLayout persistent_layout(const SequencePlanImpl& plan) {
    if (!plan.context_cache.device_state_slots) {
        throw std::logic_error("Qwen3.6 context cache options are not normalized");
    }
    const std::int32_t state_image_slots = checked_i32(
        static_cast<std::uint64_t>(plan.max_concurrency) + *plan.context_cache.device_state_slots,
        "Qwen3.6 StateImage slot count exceeds int32");
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    const std::uint32_t logical_pages  = page_count(plan.capacity);
    const std::uint32_t physical_pages = plan.main_page_groups;
    const std::uint64_t mtp_extra_pages =
        plan.features.mtp()
            ? static_cast<std::uint64_t>(plan.max_concurrency) *
                  ((static_cast<std::uint64_t>(plan.draft_window - 1U) + kPagedKVPageSize - 1U) /
                   static_cast<std::uint32_t>(kPagedKVPageSize))
            : 0ULL;
    const std::uint32_t mtp_physical_pages = static_cast<std::uint32_t>(
        checked_i32(static_cast<std::uint64_t>(physical_pages) + mtp_extra_pages,
                    "MTP Paged KV physical pages exceed int32"));
    LayoutBuilder builder;
    PersistentLayout out;
        // E3/S24: per-layer SWA windows from the target config. Muse declares
        // sliding_window (2048) + is_swa_attention() (39 of its 52 layers); the qwen
        // family declares 0 + false, i.e. every layer keeps window 0 = full attention,
        // byte-identical to the behaviour before this wiring (kernels guard on > 0).
        // ================= RUNTIME WINDOW OVERRIDE (winknob) =================
        // THE OWNER'S KNOB. `TextConfig::sliding_window` is a COMPILE-TIME constant,
        // so before this the only way to move W was a rebuild of every TU that sees
        // the target config -- which is why moving it was called "a one-way door"
        // and why the speed/correctness trade could not be traded at all.
        // NINFER_KV_WINDOW_TOKENS makes W a RUNTIME input.
        //
        // UNIT: **TOKENS**, and THIS COMMENT IS THE UNIT BOUNDARY. The compile-time
        // value is in TOKENS (4096 = 64 pages of kPagedKVPageSize = 64) and every
        // consumer downstream (PagedKVLayerView::sliding_window_tokens -> the
        // kernels' plain `int sliding_window` argument) is in TOKENS. The OTHER
        // window predicate that lives next to this one in cold_host_tier.h,
        // `cold_host_window_band`, is in PAGES. A 64x unit error here is INVISIBLE
        // at every context <= W, because gqa_attention_decode_nvfp4.cuh computes
        // `window_full = last_pos + 1` and then `token_begin = window_full -
        // sliding_window`, which clamps to a no-op. So "bit-identical at short
        // context" does NOT prove the unit is right; the print below names it.
        //
        // WHICH LAYERS: the target's policy decides WHERE, the override decides HOW
        // WIDE. It never flips a full-attention layer (window 0, "attend the whole
        // cache") into a windowed one. THE ANSWER IS PER-VARIANT -- this comment used
        // to claim one answer for the whole family, which was true of ONE variant:
        //
        //   qwen3_6_27b       is_swa_attention == true  -> 16/16 non-zero, REACHES all
        //   qwen3_6_35b_a3b   is_swa_attention == false -> all-zero, INERT
        //   qwen3_5_9b        is_swa_attention == false -> all-zero, INERT
        //   muse_glimmer_30b  layer_kind==1 -> 39 of 52, 13 ZEROS -> INERT
        //
        // windowed_layers == 0 below means the override is INERT AND the Cold Host
        // tier can never admit a page (cold_host_tier.h:81-91 returns false on the
        // first zero), which makes the 1M band unreachable at any host budget. On a
        // MIXED stack (muse: 39 sliding + 13 full) the same is true even though the
        // model genuinely declares a window: `cold_host_layers_are_windowed` needs
        // all_of (cold_host_tier.h:96-101), so one full-attention layer closes the
        // tier. The 1M mechanism therefore exists where the model has NO window and
        // is dead where it HAS one -- see dl/winpolicy/REPORT.md TASK 2a.
        //
        // MAX vs MIN: `cold_host_page_is_read_free` (cold_host_tier.h) returns false
        // if ANY layer's window has not passed the page, so the layer that BINDS
        // read-free admission is the **MAX** window. On this family the table is
        // uniform (one scalar for every paged layer) so max == min HERE and only
        // here; on a per-layer model (Muse: 39 of 52) a single scalar would be
        // wrong -- which is why the print reports every layer separately rather than
        // one number.
        //
        // PLAN TIME: this is consumed INSIDE persistent_layout(), i.e. while the
        // arena and the cold-pool reservation are computed, so the override MUST be
        // set before the engine is planned. It is read ONCE per process (the static
        // below), so two plans evaluated in the same process can never disagree, and
        // a late setenv cannot silently disagree with a plan that already saw the
        // old value: the plan keeps the resolved table and the kernels read the
        // plan, so the print below and the kernels are the same number.
        struct KvWindowResolution {
            std::uint32_t tokens;
            const char*   source;
        };
        static const KvWindowResolution kv_window = []() -> KvWindowResolution {
            const char* const raw_value = std::getenv("NINFER_KV_WINDOW_TOKENS");
            if (raw_value == nullptr || *raw_value == '\0') {
                return KvWindowResolution{
                    static_cast<std::uint32_t>(TextConfig::sliding_window),
                    "compile-time TextConfig::sliding_window (no override set)"};
            }
            std::uint64_t parsed = 0;
            for (const char* p = raw_value; *p != '\0'; ++p) {
                if (*p < '0' || *p > '9') {
                    throw std::invalid_argument(
                        std::string("NINFER_KV_WINDOW_TOKENS must be a plain "
                                    "non-negative integer number of TOKENS (digits "
                                    "only: no sign, no whitespace, no suffix); got '") +
                        raw_value + "'");
                }
                parsed = parsed * 10ULL + static_cast<std::uint64_t>(*p - '0');
                if (parsed > 0xFFFFFFFFULL) {
                    throw std::invalid_argument(
                        std::string("NINFER_KV_WINDOW_TOKENS is out of range for "
                                    "uint32 tokens: '") + raw_value + "'");
                }
            }
            if (parsed == 0ULL) {
                throw std::invalid_argument(
                    "NINFER_KV_WINDOW_TOKENS=0 is refused BY NAME: at this boundary 0 "
                    "would mean 'window 0 = full attention' on every paged layer, "
                    "which is the OPPOSITE of a window and would silently remove the "
                    "Cold Host tier's only bound while the caller still believes a "
                    "window is in force. Unset NINFER_KV_WINDOW_TOKENS to get the "
                    "compile-time value; there is no override spelling for full "
                    "attention because this family declares a window on every paged "
                    "layer");
            }
            return KvWindowResolution{static_cast<std::uint32_t>(parsed),
                                      "NINFER_KV_WINDOW_TOKENS (RUNTIME OVERRIDE)"};
        }();
        std::array<std::uint32_t, 64> layer_windows{};
        std::size_t windowed_layers = 0;
        // ================= WINDOWDOMAIN (dl/tierbug, F742) =============================
        // THE DECLARED WINDOW IS DECLARED ONLY WHERE THE TIER READS IT.
        //
        // WHAT WAS WRONG. `is_swa_attention(i)` is the TARGET's policy, and on
        // qwen3_6_27b it is true on all 16 paged layers (impl/config.h:54). The
        // window table was therefore 16/16 non-zero whatever KV tier the plan ran,
        // and the domain refusal below (kv_sliding_window_domain_error) then fired
        // on the PLAN's tiers. The engine's OWN registered per-layer table puts
        // DType::E8Kv (rk4v4) on {0,1,3,4,6,7} (impl/variant.cpp:53-58), and rk4v4's
        // decode path does NOT read the field: gqa_attention_decode_e8.cu's own
        // copy of launch_tc_partial_i8_e8 omits the argument, so the kernel takes
        // its defaulted `sliding_window = 0` (gqa_attention_decode_i8.cuh:83-86).
        // Measured on the pin: `ninfer` with NO KV FLAGS AT ALL answered rc=1 and
        // graded 0/27, twice, refused by this guard at layer 0 (dl/kvcomb, F-725).
        // A default configuration that cannot start is a defect, not a slow one.
        //
        // WHY THIS FIX AND NOT THE OTHER TWO, in the engine's own terms. The
        // refusal's own remedy sentence offers three: --kv-dtype nvfp4|iso4e, a
        // per-layer SPEC naming one of them, or "clear the variant's
        // sliding_window". The first two change the KV TIER, i.e. they change
        // QUALITY: the registered table is the measured-best mix at short context
        // (13.3k zh, ctx 4096: ppl 1.020 against 1.706 for all-nvfp4, variant.cpp)
        // and accuracy comes first. Making rk4v4's decode read the field is a
        // numerics change on the compute path, and it would repair only the DECODE
        // half of the census while the PREFILL half stays full attention
        // (kv_component_switch.h:300-330) -- which is the silent corruption the
        // two-phase census exists to prevent.
        //
        // WHY IT IS NOT A CAPABILITY LOSS. The declaration has exactly one
        // consumer, the Cold Host tier, whose admission predicate is
        // `cold_host_layers_are_windowed` (all_of, cold_host_tier.h:96-101). On a
        // plan that mixes an honouring tier with a non-honouring one that predicate
        // is FALSE in both readings: before, the refusal aborted the process;
        // after, the window table has a zero, the tier admits nothing, the program
        // already says so by name ("[cold] --cold-policy host admits nothing: no
        // layer has a sliding window ..."), and the S3 release census
        // (kv_window_tier_release_safe, program_impl.h:13076) still refuses to
        // retire a single page. The plan loses nothing it could do: the Cold Host
        // tier was inert on it either way.
        //
        // AND IT IS BIT-IDENTICAL ON THE COMPUTE PATH **ONLY FOR THE TIERS WHOSE DECODE
        // DOES NOT READ THE FIELD** (F1227 CORRECTION). `sliding_window == 0` is the
        // kernels' contract for "the field is not read", byte-identical to
        // `window = last_pos + 1` (gqa_attention_decode_i8.cuh:83-86 is that default).
        // That sentence was written when the honoured set was the decode census; three
        // of those tiers (ISO3, BF16, I8) DO read the field in decode now
        // (gqa_attention_decode_iso3.cuh:250, gqa_attention_decode_bf16.cuh:166,
        // gqa_attention_decode_i8.cuh:215), so for them "not installed" is NOT
        // bit-identical: their decode becomes full attention, which is the same
        // behaviour as the PREFILL that never applied the window. The plan is
        // self-consistent after this change and it was not before -- and the change is
        // visible only above the window length, where the two phases used to disagree.
        // No kernel, no plane geometry and no allocation moves.
        //
        // KEPT LOUD: the layers that lose the declaration are named, with the tier that
        // caused it, the PHASE that refuses it, and the flag that re-arms the tier.
        std::string window_undeclared;
        std::string window_undeclared_tiers;
        std::size_t window_undeclared_layers = 0;
        for (std::int32_t i = 0;
             i < static_cast<std::int32_t>(TextConfig::full_attention_layers()); ++i) {
            const std::size_t slot = static_cast<std::size_t>(i);
            const bool slot_explicit = plan.layer_kv_dtypes_set[slot];
            const DType slot_dtype   = product::kv_resolve_slot_dtype(
                plan.kv_dtype, plan.layer_kv_dtypes[slot], slot_explicit);
            const bool declared_here = TextConfig::is_swa_attention(i);
            const bool honoured_here = product::kv_window_tier_honoured(slot_dtype);
            layer_windows[slot] = (declared_here && honoured_here) ? kv_window.tokens : 0U;
            if (declared_here && !honoured_here) {
                if (!window_undeclared.empty()) {
                    window_undeclared += ",";
                    window_undeclared_tiers += ",";
                }
                window_undeclared += std::to_string(i);
                window_undeclared_tiers += std::string(ninfer::dtype_name(slot_dtype));
                ++window_undeclared_layers;
            }
            if (layer_windows[slot] != 0U) { ++windowed_layers; }
        }
        if (window_undeclared_layers != 0U) {
            // THE SENTENCE THIS PRINTS IS THE WHOLE POINT OF THE TWO-PHASE PREDICATE, and
            // it is FALSE for the tiers that reach it if it is not written per phase: the
            // decode kernel of ISO3/BF16/I8 DOES read `sliding_window_tokens` now. What
            // does not read it is their PREFILL. So the window is not installed at all
            // (a bound applied by one phase and not the other is silent above the window
            // length), the affected layers keep window 0 = full attention in BOTH phases,
            // and the arming advice names the one tier whose prompt kernel reads the field
            // -- NVFP4 (gqa_attention_prefill_nvfp4.cuh:1097). Spark-x2.5-4b is the case
            // this reads on today: 27 of 36 layers at 512 tokens, bf16, decode honoured,
            // prefill absent (dl/musesparkfix, F1227).
            std::fprintf(stderr,
                         "[kv-window] THE DECLARED WINDOW IS NOT INSTALLED on %zu of %d paged "
                         "layer(s) [%s], whose KV tier(s) are [%s]: this tier's DECODE kernel "
                         "applies sliding_window_tokens and its PREFILL (prompt) kernel does "
                         "not, so a window installed here would bound one phase and not the "
                         "other. Those layers keep window 0 = FULL ATTENTION in BOTH phases "
                         "(the tier's own contract for the field, and self-consistent). The "
                         "Cold Host tier admits nothing on this plan (it requires EVERY entry "
                         "non-zero: cold_host_page_is_read_free) and retires no page. TO "
                         "INSTALL IT, put those layers on a tier whose PREFILL reads the "
                         "field -- nvfp4 today (gqa_attention_prefill_nvfp4.cuh:1097 is "
                         "`(sliding_window > 0 && KVDType == DType::NVFP4)`; the ISO4E "
                         "instance of the same template gets window 0, and the bf16/i8 prompt "
                         "kernels have no such parameter): --kv-dtype nvfp4, or a per-layer "
                         "--kv-layer-storage SPEC naming it. The PREFILL side is the missing "
                         "half; product/kv_component_switch.h carries the per-tier census.\n",
                         window_undeclared_layers,
                         static_cast<int>(TextConfig::full_attention_layers()),
                         window_undeclared.c_str(), window_undeclared_tiers.c_str());
        }
        // OBSERVABILITY: the effective window, PER PAGED LAYER, WITH ITS UNIT, and
        // with the source it came from. Unconditional on purpose -- an instrument
        // that cannot report the thing it is doing is the same defect as no
        // instrument.
        std::fprintf(stderr,
                     "[kv-window] source=%s  unit=TOKENS  paged_layers=%d  "
                     "windowed=%zu  window_tokens=%u  = %u pages of %d\n",
                     kv_window.source,
                     static_cast<int>(TextConfig::full_attention_layers()),
                     windowed_layers, kv_window.tokens,
                     kv_window.tokens / static_cast<std::uint32_t>(kPagedKVPageSize),
                     static_cast<int>(kPagedKVPageSize));
        if (windowed_layers == 0U) {
            std::fprintf(stderr,
                         "[kv-window] WARNING the window table is ALL ZERO: the "
                         "override is INERT on this target and every paged layer "
                         "keeps full attention\n");
            // NOT a second warning about the same thing: this one names the
            // CONSEQUENCE for the 1M branch requirement, which the line above does
            // not. `cold_host_page_is_read_free` returns false on the first zero
            // entry, so the Cold Host tier admits nothing and the band
            // max(0, F-K-H) <= W <= D-K is EMPTY at every --cold-host-bytes: the 1M
            // frontier is unreachable on this variant. NINFER_KV_WINDOW_TOKENS cannot
            // change it, because the ternary above keys on is_swa_attention(i). The
            // fix is a variant edit (see qwen3_6_27b/impl/config.h), not a flag.
            //
            // DELIBERATELY UNGATED ON --cold-policy: `EngineOptions` is not in scope
            // at this site (this function takes `plan`), and gating adds nothing --
            // the tier is unreachable whether or not anyone requested it, and a
            // `--cold-policy host` run on this variant is exactly the case that used
            // to be accepted, sized to 0 device pages, reported by NEITHER branch of
            // the arming report below, and dead. This line is what makes that loud.
            std::fprintf(stderr,
                         "[kv-window] CONSEQUENCE: the Cold Host tier can admit no "
                         "page on this variant (cold_host_page_is_read_free returns "
                         "false on the first zero window), so the 1M band "
                         "max(0, F-K-H) <= W <= D-K is EMPTY at every "
                         "--cold-host-bytes and a 1M frontier that needs an evicted "
                         "page is UNREACHABLE. NINFER_KV_WINDOW_TOKENS cannot fix "
                         "this: the window table keys on is_swa_attention(), which "
                         "this variant returns false for. --cold-policy host on this "
                         "variant therefore changes nothing; use "
                         "--cold-policy window|disk for the slot tiers, which do not "
                         "depend on the window.\n");
        }
        for (std::size_t i = 0;
             i < static_cast<std::size_t>(TextConfig::full_attention_layers()); ++i) {
            std::fprintf(stderr,
                         "[kv-window]   paged layer %2zu: %u TOKENS = %u pages of %d  "
                         "[%s]\n",
                         i, layer_windows[i],
                         layer_windows[i] / static_cast<std::uint32_t>(kPagedKVPageSize),
                         static_cast<int>(kPagedKVPageSize),
                         layer_windows[i] == 0U
                             ? "0 = FULL ATTENTION (reads whole cache)"
                             : "windowed");
        }
        // ============ THE BINDING'S PRECONDITION, AS A NUMBER (F1227 / F1207) ============
        // THE COMMENT THIS REPLACES. targets/muse_glimmer_30b/impl/load/bindings.cpp binds
        // all 52 of muse's layers to the full-attention leaf and states its reason in words:
        // "the 39 sliding layers run as full for the acceptance phase (short contexts:
        // window is not clipped)". That sentence is a statement about a NUMBER -- how long a
        // sequence may grow before the two mechanisms stop agreeing -- and a sentence in a
        // loader is not a check. This is the one site in the engine where the number meets
        // the plan, so it is reported here, unconditionally, in the same block and the same
        // unit as the window table itself.
        //
        // WHICH LAYERS COUNT: declared sliding (is_swa_attention) AND bound to the
        // full-attention leaf (is_full_attention) AND with NO window installed by THIS plan
        // (layer_windows == 0, i.e. no tier in this plan makes the PREFILL read the field)
        // AND with a non-zero declared window. Those layers attend the whole cache in BOTH
        // phases; the model's own topology would have clipped at the declared window, so the
        // two agree exactly while every sequence stays at or below it. A layer whose plan
        // window is NON-ZERO is deliberately NOT counted: there both KV phases clip at that
        // window, which is the model's own semantics at every length.
        {
            const std::uint32_t declared_window =
                static_cast<std::uint32_t>(TextConfig::sliding_window);
            std::uint32_t regime_tokens = 0U;
            std::string   regime_layers;
            std::size_t   regime_count = 0U;
            for (std::size_t i = 0;
                 i < static_cast<std::size_t>(TextConfig::full_attention_layers()); ++i) {
                const auto layer = static_cast<std::int32_t>(i);
                if (!TextConfig::is_swa_attention(layer) ||
                    !TextConfig::is_full_attention(layer) || layer_windows[i] != 0U ||
                    declared_window == 0U) {
                    continue;
                }
                if (regime_tokens == 0U || declared_window < regime_tokens) {
                    regime_tokens = declared_window;
                }
                if (!regime_layers.empty()) { regime_layers += ","; }
                regime_layers += std::to_string(i);
                ++regime_count;
            }
            if (regime_count != 0U) {
                std::fprintf(stderr,
                             "[kv-regime] %zu of %d paged layer(s) [%s] DECLARE a sliding "
                             "window of %u TOKENS and are BOUND to the full-attention leaf "
                             "(is_swa_attention && is_full_attention), and THIS PLAN "
                             "installs NO window on them (see the [kv-window] report above). "
                             "They therefore attend the whole cache in BOTH phases, which "
                             "matches the model's own topology only while every sequence "
                             "stays at or below %u TOKENS -- above it the model would have "
                             "clipped and this plan does not. Keep sequences <= %u tokens, "
                             "or put those layers on a tier whose PREFILL reads the field; "
                             "any reading taken outside this regime belongs to a topology "
                             "the model does not have (F1207 / F1227).\n",
                             regime_count,
                             static_cast<int>(TextConfig::full_attention_layers()),
                             regime_layers.c_str(), regime_tokens, regime_tokens,
                             regime_tokens);
            }
        }
        // ================= end runtime window override ======================
    // SEPARATION (window2), RESTATED FOR THE TWO-PHASE PREDICATE (F1227): a window table
    // that could let the Cold Host tier release a page which a later phase reads anyway is
    // REFUSED BY NAME here, before the spec below commits it to the device: silent
    // corruption, not a quality loss. The refusal keys on the same conjunction the install
    // site uses (kv_sliding_window_domain_error -> kv_window_tier_honoured), so a plan whose
    // tiers cannot honour the field in BOTH phases no longer arrives here with a non-zero
    // table -- the install already zeroed those layers, the all-zero branch in the function
    // below makes the refusal inert, and the [kv-window] report above names every layer and
    // tier that lost the declaration. This call is therefore the SECOND gate, and it is
    // kept deliberately: the window table is an input to this function as well as its
    // output, and a producer that fills it any other way still meets this refusal.
    // product/kv_component_switch.h holds the domain and the evidence.
    if (const std::string window_refusal = product::kv_sliding_window_domain_error(
            std::span<const std::uint32_t>(
                layer_windows.data(),
                static_cast<std::size_t>(TextConfig::full_attention_layers())),
            plan.kv_dtype,
            std::span<const DType>(
                plan.layer_kv_dtypes.data(),
                static_cast<std::size_t>(TextConfig::full_attention_layers())),
            std::span<const bool>(
                plan.layer_kv_dtypes_set.data(),
                static_cast<std::size_t>(TextConfig::full_attention_layers())));
        !window_refusal.empty()) {
        throw std::invalid_argument(window_refusal);
    }
    out.decoder = qwen3_6::plan_decoder_state(
        builder, qwen3_6::DecoderStateSpec{
                     .full_attention_layers     = TextConfig::full_attention_layers(),
                     .mtp_layers                = TextConfig::mtp_layers,
                     .capacity                  = plan.capacity,
                     .kv_heads                  = TextConfig::kv_heads,
                     .attention_head_dim        = TextConfig::head_dim,
                     .kv_dtype                  = plan.kv_dtype,
                     .kv_quant_group            = plan.kv_quant_group,
                     .layer_kv_dtypes           = plan.layer_kv_dtypes,
                     .layer_residual            = plan.kv_residual_layers,
                     .layer_sliding_windows    = layer_windows,
                     // SEPARATION: the three KV component switches reach the
                     // decoder plan through here; plan_decoder_state() is the
                     // single place that commits them to the device (rotation
                     // gate, row-scale gate, V codec + its hard-fail checks).
                     .kv_v_codec                = plan.kv_v_codec,
                     .kv_rotation_off           = plan.kv_rotation_off,
                     .kv_row_scale_spec         = plan.kv_row_scale_spec,
                     .enable_mtp                = plan.features.mtp(),
                     .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency + 1),
                     .text_physical_page_groups = physical_pages,
                     .mtp_physical_page_groups  = mtp_physical_pages,
                     // --max-cold-pages via effective_cold_pages (same S25
                     // precedence, now shared with the --kv-bit-budget DP so the
                     // DP's cold capacity always equals the pool reserved here:
                     // None->0 always; Host->0 because the Host tier lives in
                     // pinned host memory and holds read-free pages only (no
                     // device slot is involved, --cold-host-bytes is its cap);
                     // Window/Disk/HostThenDisk: explicit cap else keep-tokens
                     // derivation -- under HostThenDisk the pool is the disk
                     // rung's working set, not the host rung's).
                     .max_cold_pages            = effective_cold_pages(
                         plan.cold_policy, plan.cold_keep_tokens, plan.max_cold_pages),
                     // The mask travels with the table: without it plan_cache()
                     // would treat every BF16 slot as "inherit the global dtype"
                     // and the per-layer BF16 request the operator wrote would be
                     // honoured nowhere. Declared last in DecoderStateSpec, so the
                     // designator has to come last too.
                     .layer_kv_dtypes_set       = plan.layer_kv_dtypes_set,
                 });
    qwen3_6::StateImageSpec state_image_spec{
        .linear =
            {
                // 纯 softmax 族 (gdn==0) 无线性状态: 池规划要求非零几何,
                // 用最小假层占位 (运行时永不触碰; qwen 家族不受影响).
                .layers         = std::max(1, TextConfig::gdn_layers()),
                .conv_channels  = std::max(1, TextConfig::convolution_dim),
                .conv_width     = std::max(1, TextConfig::gdn_conv_state_width),
                .value_heads    = std::max(1, TextConfig::gdn_value_heads),
                .value_head_dim = std::max(1, TextConfig::gdn_value_head_dim),
                .key_head_dim   = std::max(1, TextConfig::gdn_key_head_dim),
                .slot_count     = state_image_slots,
                .conv_dtype     = DType::BF16,
            },
        .hidden = TextConfig::hidden,
    };
    if constexpr (Variant::supports_dflash) {
        if (plan.features.dflash()) {
            state_image_spec.dflash_local = qwen3_6::DFlashLocalStateSpec{
                .layers   = DFlashConfig::local_layers,
                .capacity = DFlashConfig::local_capacity,
                .kv_heads = DFlashConfig::kv_heads,
                .head_dim = DFlashConfig::head_dim,
            };
        }
    }
    out.state_images = qwen3_6::plan_state_image_device_pool(builder, state_image_spec);
    if (plan.speculative_backend != SpeculativeBackend::None) {
        out.replay_records = plan_gdn_replay_records(
            builder, GdnReplayRecordSpec{
                         .layers          = TextConfig::gdn_layers(),
                         .record_capacity = static_cast<std::int32_t>(plan.max_concurrency),
                         .width           = static_cast<std::int32_t>(plan.draft_window + 1U),
                         .conv_channels   = TextConfig::convolution_dim,
                         .qk_heads        = TextConfig::gdn_key_heads,
                         .value_heads     = TextConfig::gdn_value_heads,
                         .key_dim         = TextConfig::gdn_key_head_dim,
                         .value_dim       = TextConfig::gdn_value_head_dim,
                     });
    }
    if constexpr (Variant::supports_dflash) {
        if (plan.features.dflash()) {
            DFlashPersistentLayout& dflash = out.dflash.emplace();
            // Every DSpark layer keeps its own BF16 context-K/V plane pair in
            // one shared page pool; attention selects the layer plane base.
            KVPageGeometry full_geometry{
                .page_tokens        = kPagedKVPageSize,
                .device_plane_order = PagedKVPlaneOrder::HeadMajor,
                .planes             = {},
            };
            for (int layer = 0; layer < DFlashConfig::full_layers; ++layer) {
                full_geometry.planes.push_back(
                    {DType::BF16, DFlashConfig::head_dim, DFlashConfig::kv_heads, 256});
                full_geometry.planes.push_back(
                    {DType::BF16, DFlashConfig::head_dim, DFlashConfig::kv_heads, 256});
            }
            qwen3_6::PagedKVCacheLayout full_layout{
                .pages = plan_device_kv_page_pool(
                    builder, DeviceKVPagePoolSpec{.page_group_count = physical_pages,
                                                  .geometry         = std::move(full_geometry)}),
                .execution_tables = plan_kv_execution_tables(
                    builder,
                    KVExecutionTableSpec{
                        .logical_page_capacity = logical_pages,
                        .table_rows            = static_cast<std::int32_t>(plan.max_concurrency + 1),
                    }),
                .layers      = static_cast<std::uint32_t>(DFlashConfig::full_layers),
                .max_context = plan.capacity,
                .kv_heads    = DFlashConfig::kv_heads,
                .head_dim    = DFlashConfig::head_dim,
                .dtype       = DType::BF16,
                .quant_group = 0,
            };
            for (int layer = 0; layer < DFlashConfig::full_layers; ++layer) {
                full_layout.layer_plane_base[static_cast<std::size_t>(layer)] =
                    static_cast<std::uint32_t>(2 * layer);
            }
            dflash.full = full_layout;
            dflash.prefill_features = add_tensor(
                builder, DType::BF16, {DFlashConfig::feature_rows, effective_prefill_chunk},
                "DFlash prefill target features");
            dflash.prefill_positions = add_tensor(builder, DType::I32, {effective_prefill_chunk},
                                                  "DFlash prefill target positions");
            dflash.pending_features  = add_tensor(builder, DType::BF16,
                                                  {DFlashConfig::feature_rows,
                                                   static_cast<std::int32_t>(plan.draft_window + 1U),
                                                   static_cast<std::int32_t>(plan.max_concurrency)},
                                                  "DFlash pending target features");
        }
    }
    if constexpr (Variant::supports_dflash2) {
        if (plan.features.dflash2()) {
            DFlash2PersistentLayout& dflash2 = out.dflash2.emplace();
            dflash2.local = plan_cyclic_kv_cache(builder, DFlash2Config::local_layers,
                                                 DFlash2Config::local_capacity,
                                                 DFlash2Config::kv_heads, DFlash2Config::head_dim,
                                                 static_cast<std::int32_t>(plan.max_concurrency));
            dflash2.rewrite_checkpoint_local = plan_cyclic_kv_cache(
                builder, DFlash2Config::local_layers, DFlash2Config::local_capacity,
                DFlash2Config::kv_heads, DFlash2Config::head_dim,
                static_cast<std::int32_t>(plan.max_concurrency));
            dflash2.prefill_features = add_tensor(
                builder, DType::BF16, {DFlash2Config::feature_rows, effective_prefill_chunk},
                "DFlash2 prefill target features");
            dflash2.prefill_positions = add_tensor(builder, DType::I32, {effective_prefill_chunk},
                                                   "DFlash2 prefill target positions");
            dflash2.pending_features = add_tensor(
                builder, DType::BF16,
                {DFlash2Config::feature_rows,
                 static_cast<std::int32_t>(plan.draft_window + 1U),
                 static_cast<std::int32_t>(plan.max_concurrency)},
                "DFlash2 pending target features");
        }
    }

    out.round = qwen3_6::begin_round_state_layout(
        builder,
        qwen3_6::RoundStateSpec{.hidden           = TextConfig::hidden,
                                .output_rows      = TextConfig::output_rows,
                                .batch_capacity   = plan.max_concurrency,
                                .draft_window     = plan.draft_window,
                                .draft_tree_paths = plan.draft_tree_paths,
                                .draft_tree_depth = plan.draft_tree_depth,
                                .enable_mtp       = plan.features.mtp(),
                                .enable_dflash    = plan.features.dflash_like()});
    out.prefill_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, effective_prefill_chunk}, "step prefill hidden");
    if (plan.causal_scoring) {
        out.score_hidden = add_tensor(
            builder, DType::BF16, {TextConfig::hidden, static_cast<std::int32_t>(kCausalScoreTile)},
            "causal score hidden staging");
    }
    qwen3_6::complete_round_state_layout(builder, out.round);
    const auto i32 = [&](std::size_t n, const char* label) {
        return add_tensor(builder, DType::I32, {static_cast<std::int32_t>(n)}, label);
    };
    out.token_counts =
        add_tensor(builder, DType::I32,
                   {TextConfig::token_domain, static_cast<std::int32_t>(plan.max_concurrency)},
                   "sampling token counts");
    const auto config_words = static_cast<std::int32_t>(
        (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
    out.sampling_config = add_tensor(
        builder, DType::I32, {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
        "sampling config");
    out.bytes = builder.finish(kArenaAlign, "persistent layout");
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
    return out;
}

WorkspacePlan build_workspace_plan(const SequencePlanImpl& plan) {
    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        plan.draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    const auto verify = drafts + 1;
    // FIX-A: the third field is the pinned split reference. Leaving it out made this a
    // two-field aggregate, so split_reference_keys read 0 and every small-T split-KV launch
    // this envelope sized fell back to the LIVE window as its split reference -- a split
    // grid that follows how far the sequence has advanced, which is what
    // include/ninfer/ops/gqa_attention.h:18-25 forbids. The workspace is sized from the
    // split capacity this envelope produces, so the pin has to be here for the allocation
    // to match what the launch grid will actually be dispatched with.
    const ops::GqaExecutionEnvelope text_envelope{1, plan.capacity, plan.capacity};

    const auto matrix  = [](WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                           std::int32_t tokens) { (void)layout.alloc(dtype, {rows, tokens}); };
    const auto scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        if (bytes == 0) { return; }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(bytes);
    };
    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };

    const auto text_common_root = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        (void)workspace_recipe::text_prefill_roots<TextConfig>(
            layout, tokens, plan.features.vision ? 3 : 0, plan.features.vision ? tokens : 0);
    };
    const auto attention_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                     std::int32_t last, qwen3_6::TextPhase phase,
                                     std::int32_t batch_size, std::int32_t min_width,
                                     std::int32_t max_width,
                                     ops::GqaExecutionEnvelope envelope) {
        auto stage = layout.scope();
        (void)workspace_recipe::text_attention_projection<TextConfig>(layout, last);
        scratch(layout, Variant::attention_projection_workspace_capacity_bytes(plan.weights_profile,
                                                                               phase, first, last));
        (void)workspace_recipe::text_attention_results<TextConfig>(layout, last);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(TextConfig::head_dim, 
                            TextConfig::query_heads, plan.kv_dtype, envelope, batch_size,
                            min_width, max_width));
        scratch(layout, Variant::attention_output_projection_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };
    const auto gdn_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                               std::int32_t last, qwen3_6::TextPhase phase, GdnWorkspacePath path,
                               std::int32_t batch_size, std::int32_t min_width,
                               std::int32_t max_width) {
        auto stage = layout.scope();
        (void)workspace_recipe::gdn_control<TextConfig>(layout, last);
        scratch(layout, Variant::gdn_norm_control_projection_workspace_capacity_bytes(first, last));
        (void)workspace_recipe::gdn_projection<TextConfig>(layout, last);
        if (path == GdnWorkspacePath::Snapshot) {
            scratch(layout, Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
                                plan.weights_profile, phase, batch_size, min_width, max_width));
        } else if (path == GdnWorkspacePath::ReplayRecord) {
            scratch(layout, Variant::gdn_input_projection_record_workspace_capacity_bytes(
                                plan.weights_profile, phase, batch_size, min_width, max_width));
        } else {
            (void)workspace_recipe::gdn_prefill_conv<TextConfig>(layout, last);
            scratch(layout, Variant::gdn_input_projection_workspace_capacity_bytes(
                                plan.weights_profile, phase, first, last));
        }
        (void)workspace_recipe::gdn_recurrent_output<TextConfig>(layout, last);
        if (path == GdnWorkspacePath::Prefill) {
            scratch(layout,
                    ops::gated_delta_net_workspace_capacity_bytes(
                        TextConfig::gdn_key_heads, TextConfig::gdn_value_heads, true, first, last));
        }
        (void)workspace_recipe::gdn_normalized_output<TextConfig>(layout, last);
        scratch(layout, Variant::gdn_output_projection_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };
    const auto post_mixer_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                      std::int32_t last, qwen3_6::TextPhase phase) {
        auto stage = layout.scope();
        (void)workspace_recipe::post_mixer_hidden<TextConfig>(layout, last);
        scratch(layout, Variant::post_mixer_workspace_capacity_bytes(plan.weights_profile, phase,
                                                                     first, last));
    };
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, qwen3_6::TextPhase phase, GdnWorkspacePath path,
                                 std::int32_t batch_size, std::int32_t min_width,
                                 std::int32_t max_width,
                                 ops::GqaExecutionEnvelope envelope) {
        attention_stage(layout, first, last, phase, batch_size, min_width, max_width, envelope);
        gdn_stage(layout, first, last, phase, path, batch_size, min_width, max_width);
        post_mixer_stage(layout, first, last, phase);
    };
    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, Variant::draft_head_rows, columns);
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        (void)workspace_recipe::mtp_stem<TextConfig>(layout, tokens, !preembedded);
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::GqaExecutionEnvelope envelope) {
        auto core = layout.scope();
        mtp_stem(layout, tokens, false);
        (void)workspace_recipe::mtp_attention_projection<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_attention_projection_workspace_capacity_bytes(tokens, tokens));
        (void)workspace_recipe::mtp_attention_results<TextConfig>(layout, tokens);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(TextConfig::head_dim, 
                            TextConfig::query_heads, plan.kv_dtype, envelope, 1, tokens, tokens));
        (void)workspace_recipe::mtp_post_attention<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(tokens, tokens));
    };
    const auto mtp_full_call = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::GqaExecutionEnvelope envelope,
                                   bool build_proposal) {
        auto call = layout.scope();
        matrix(layout, DType::I32, 1, tokens);
        mtp_full_core(layout, tokens, envelope);
        if (build_proposal) {
            auto proposal = layout.scope();
            proposal_scratch(layout, 1);
        }
    };
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                       std::int32_t last, bool preembedded) {
        auto call = layout.scope();
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        {
            auto bulk = layout.scope();
            mtp_stem(layout, last, preembedded);
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
            scratch(layout, Variant::mtp_kv_projection_workspace_capacity_bytes(first, last));
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
        }
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        scratch(layout, Variant::mtp_q_gate_projection_workspace_capacity_bytes(1, 1));
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::I32, 3, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(TextConfig::head_dim, 
                            TextConfig::query_heads, plan.kv_dtype, text_envelope, 1, 1, 1));
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(1, 1));
        proposal_scratch(layout, 1);
    };

    WorkspacePlan out;
    WorkspaceLayoutBuilder text_prefill;
    text_common_root(text_prefill, chunk);
    target_body(text_prefill, 1, chunk, qwen3_6::TextPhase::Prefill, GdnWorkspacePath::Prefill, 1,
                1, chunk, text_envelope);
    scratch(text_prefill, ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, 1, 1));
    out.text_prefill = finish(text_prefill);

    if (plan.causal_scoring) {
        WorkspaceLayoutBuilder causal_score;
        matrix(causal_score, DType::BF16, TextConfig::output_rows,
               static_cast<std::int32_t>(kCausalScoreTile));
        matrix(causal_score, DType::I32, 1, static_cast<std::int32_t>(kCausalScoreTile));
        matrix(causal_score, DType::FP32, 1, static_cast<std::int32_t>(kCausalScoreTile));
        out.causal_score = finish(causal_score);
    }

    for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
         ++batch) {
        WorkspaceLayoutBuilder ordinary;
        matrix(ordinary, DType::BF16, TextConfig::hidden, batch);
        target_body(ordinary, batch, batch, qwen3_6::TextPhase::Verify, GdnWorkspacePath::Snapshot,
                    batch, 1, 1, text_envelope);
        scratch(ordinary,
                ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, batch, batch));
        out.ordinary_round = std::max(out.ordinary_round, finish(ordinary));
    }

    if (plan.features.mtp()) {
        WorkspaceLayoutBuilder mtp_prefill;
        text_common_root(mtp_prefill, chunk);
        target_body(mtp_prefill, 1, chunk, qwen3_6::TextPhase::Prefill, GdnWorkspacePath::Prefill,
                    1, 1, chunk, text_envelope);
        matrix(mtp_prefill, DType::I32, 1, chunk);
        if (plan.features.vision) {
            matrix(mtp_prefill, DType::BF16, TextConfig::hidden, chunk);
            (void)workspace_recipe::visual_scatter_indices(mtp_prefill, chunk);
        }
        mtp_prefill_chunk(mtp_prefill, 1, chunk, plan.features.vision);
        for (std::int32_t i = 1; i < drafts; ++i) {
            matrix(mtp_prefill, DType::BF16, TextConfig::hidden, 1);
            mtp_full_call(mtp_prefill, 1, text_envelope, true);
        }
        out.mtp_prefill = finish(mtp_prefill);

        WorkspaceLayoutBuilder mtp_batch;
        mtp_full_call(mtp_batch, verify, text_envelope, false);
        WorkspaceLayoutBuilder mtp_ar;
        mtp_full_call(mtp_ar, 1, text_envelope, true);
        WorkspaceLayoutBuilder mtp_align;
        mtp_full_call(mtp_align, 1, text_envelope, false);
        WorkspaceLayoutBuilder mtp_proposal;
        proposal_scratch(mtp_proposal, 1);
        const std::size_t accept = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            TextConfig::token_domain, drafts, drafts, 1, 1);
        out.mtp_round = std::max({accept, finish(mtp_batch), finish(mtp_ar), finish(mtp_proposal)});
        out.ordinary_round = std::max(out.ordinary_round, finish(mtp_align));

        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = batch * verify;
            WorkspaceLayoutBuilder target;
            matrix(target, DType::BF16, TextConfig::hidden, aggregate);
            target_body(target, aggregate, aggregate, qwen3_6::TextPhase::Verify,
                        GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);

            const auto mtp_decode_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t width) {
                const std::int32_t tokens = batch * width;
                auto core                 = layout.scope();
                mtp_stem(layout, tokens, false);
                (void)workspace_recipe::mtp_attention_projection<TextConfig>(layout, tokens);
                scratch(layout,
                        Variant::mtp_attention_projection_workspace_capacity_bytes(tokens, tokens));
                (void)workspace_recipe::mtp_attention_results<TextConfig>(layout, tokens);
                scratch(layout,
                        ops::gqa_attention_workspace_capacity_bytes(TextConfig::head_dim, 
                            TextConfig::query_heads, plan.kv_dtype, text_envelope, batch, width, width));
                (void)workspace_recipe::mtp_post_attention<TextConfig>(layout, tokens);
                scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(tokens, tokens));
            };

            WorkspaceLayoutBuilder alignment;
            mtp_decode_core(alignment, verify);
            WorkspaceLayoutBuilder ar;
            mtp_decode_core(ar, 1);
            WorkspaceLayoutBuilder proposal;
            proposal_scratch(proposal, batch);
            const std::size_t batch_accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    TextConfig::token_domain, drafts, drafts, batch, batch);
            out.mtp_round = std::max({out.mtp_round, finish(target), finish(alignment), finish(ar),
                                      finish(proposal), batch_accept});
        }
    }

    if (plan.features.dflash()) {
        if constexpr (!Variant::supports_dflash) {
            throw std::logic_error("unsupported target reached DFlash scratch planning");
        } else {
            const auto dflash_context_capacity = [&](std::int32_t tokens, bool compact_input) {
                WorkspaceLayoutBuilder layout;
                if (compact_input) {
                    matrix(layout, DType::BF16, DFlashConfig::feature_rows, tokens);
                }
                (void)workspace_recipe::dflash_context<DFlashConfig>(layout, tokens);
                (void)ops::linear_workspace_capacity_bytes(
                    QType::BF16_CTRL, DFlashConfig::hidden, DFlashConfig::feature_rows,
                    ops::LinearPolicy::A16Only, tokens, tokens);
                {
                    auto layer = layout.scope();
                    (void)workspace_recipe::dflash_context_layer<DFlashConfig>(layout, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlashConfig::kv_size, DFlashConfig::hidden,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                }
                return finish(layout);
            };
            const auto dflash_proposal_capacity = [&](std::int32_t width, std::int32_t batch) {
                WorkspaceLayoutBuilder layout;
                const std::int32_t tokens = width * batch;
                matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                {
                    auto attention = layout.scope();
                    (void)workspace_recipe::dflash_attention<DFlashConfig>(layout, tokens);
                    scratch(layout,
                            std::max(ops::swa_workspace_capacity_bytes(
                                         {0, plan.capacity}, width, width, batch,
                                         DFlashConfig::local_window),
                                     ops::bidirectional_gqa_attention_workspace_capacity_bytes(
                                         {0, plan.capacity}, width, width, batch)));
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlashConfig::query_size + 2 * DFlashConfig::kv_size,
                        DFlashConfig::hidden, ops::LinearPolicy::A16Only, tokens, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlashConfig::hidden, DFlashConfig::query_size,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                }
                {
                    auto mlp = layout.scope();
                    (void)workspace_recipe::dflash_mlp<DFlashConfig>(layout, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, 2 * DFlashConfig::intermediate, DFlashConfig::hidden,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlashConfig::hidden, DFlashConfig::intermediate,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                }
                matrix(layout, DType::BF16, DFlashConfig::hidden, drafts * batch);
                matrix(layout, DType::BF16, DFlashConfig::hidden, drafts * batch);
                if (plan.proposal_head == ProposalHead::Optimized) {
                    matrix(layout, DType::BF16, Variant::draft_head_rows, drafts * batch);
                } else {
                    matrix(layout, DType::BF16, TextConfig::output_rows, drafts * batch);
                }
                // DSpark Markov-argmax/SVIP per-row scratch.
                matrix(layout, DType::I32, batch, 1);
                matrix(layout, DType::I32, batch, 1);
                matrix(layout, DType::I32, batch, 1);
                return finish(layout);
            };

            out.dflash_context = dflash_context_capacity(chunk, false);
            for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
                 ++batch) {
                const std::int32_t aggregate = verify * batch;
                WorkspaceLayoutBuilder target;
                matrix(target, DType::BF16, TextConfig::hidden, aggregate);
                target_body(target, aggregate, aggregate, qwen3_6::TextPhase::Verify,
                            GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);
                const std::size_t accept =
                    ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                        TextConfig::token_domain, drafts, drafts, batch, batch);
                const std::size_t proposal = dflash_proposal_capacity(verify, batch);
                out.dflash_round           = std::max({out.dflash_round, finish(target), accept,
                                                       dflash_context_capacity(aggregate, true), proposal});
            }
        }
    }

    if (plan.features.dflash2()) {
        if constexpr (!Variant::supports_dflash2) {
            throw std::logic_error("unsupported target reached DFlash2 scratch planning");
        } else {
            const auto dflash2_context_capacity = [&](std::int32_t tokens, bool compact_input) {
                WorkspaceLayoutBuilder layout;
                if (compact_input) {
                    matrix(layout, DType::BF16, DFlash2Config::feature_rows, tokens);
                }
                (void)workspace_recipe::dflash_context<DFlash2Config>(layout, tokens);
                (void)ops::linear_workspace_capacity_bytes(
                    QType::BF16_CTRL, DFlash2Config::hidden, DFlash2Config::feature_rows,
                    ops::LinearPolicy::A16Only, tokens, tokens);
                {
                    auto layer = layout.scope();
                    (void)workspace_recipe::dflash_context_layer<DFlash2Config>(layout, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlash2Config::kv_size, DFlash2Config::hidden,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                }
                return finish(layout);
            };
            const auto dflash2_proposal_capacity = [&](std::int32_t width, std::int32_t batch) {
                WorkspaceLayoutBuilder layout;
                const std::int32_t tokens = width * batch;
                matrix(layout, DType::BF16, DFlash2Config::hidden, tokens);
                {
                    auto attention = layout.scope();
                    (void)workspace_recipe::dflash_attention<DFlash2Config>(layout, tokens);
                    matrix(layout, DType::BF16, 1280, tokens);
                    matrix(layout, DType::BF16, DFlash2Config::hidden, tokens);
                    matrix(layout, DType::BF16, DFlash2Config::hidden, tokens);
                    scratch(layout, ops::swa_workspace_capacity_bytes(
                                       {0, plan.capacity}, width, width, batch,
                                       DFlash2Config::local_window));
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlash2Config::query_size + 2 * DFlash2Config::kv_size,
                        DFlash2Config::hidden, ops::LinearPolicy::A16Only, tokens, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlash2Config::hidden, DFlash2Config::query_size,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, 1280, DFlash2Config::hidden, ops::LinearPolicy::A16Only,
                        tokens, tokens);
                }
                {
                    auto mlp = layout.scope();
                    (void)workspace_recipe::dflash_mlp<DFlash2Config>(layout, tokens);
                    matrix(layout, DType::BF16, 1280, tokens);
                    matrix(layout, DType::BF16, DFlash2Config::hidden, tokens);
                    matrix(layout, DType::BF16, DFlash2Config::hidden, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, 2 * DFlash2Config::intermediate, DFlash2Config::hidden,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlash2Config::hidden, DFlash2Config::intermediate,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, 1280, DFlash2Config::hidden, ops::LinearPolicy::A16Only,
                        tokens, tokens);
                }
                matrix(layout, DType::BF16, DFlash2Config::hidden, drafts * batch);
                matrix(layout, DType::BF16, DFlash2Config::hidden, drafts * batch);
                matrix(layout, DType::BF16, TextConfig::output_rows, drafts * batch);
                matrix(layout, DType::BF16, DFlash2Config::selector_rank, drafts * batch);
                (void)ops::linear_workspace_capacity_bytes(
                    QType::BF16_CTRL, DFlash2Config::selector_rank, DFlash2Config::hidden,
                    ops::LinearPolicy::A16Only, drafts * batch, drafts * batch);
                // Scratch shapes must match the proposal step count the schedule
                // actually runs (dflash2_impl.h), i.e. the runtime draft window.
                matrix(layout, DType::I32, batch * drafts * DFlash2Config::selector_top_k,
                       1);
                matrix(layout, DType::FP32, batch * drafts * DFlash2Config::selector_top_k,
                       1);
                matrix(layout, DType::FP32,
                       batch * drafts * DFlash2Config::selector_top_k *
                           DFlash2Config::selector_top_k,
                       1);
                return finish(layout);
            };

            out.dflash2_context = dflash2_context_capacity(chunk, false);
            for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
                 ++batch) {
                const std::int32_t aggregate = verify * batch;
                WorkspaceLayoutBuilder target;
                matrix(target, DType::BF16, TextConfig::hidden, aggregate);
                target_body(target, aggregate, aggregate, qwen3_6::TextPhase::Verify,
                            GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);
                const std::size_t accept =
                    ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                        TextConfig::token_domain, drafts, drafts, batch, batch);
                const std::size_t proposal = dflash2_proposal_capacity(verify, batch);
                out.dflash2_round          = std::max({out.dflash2_round, finish(target), accept,
                                                       dflash2_context_capacity(aggregate, true),
                                                       proposal});
            }
        }
    }

    out.general_capacity =
        std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                  out.dflash_context, out.dflash_round, out.dflash2_context, out.dflash2_round,
                  out.causal_score});
    // Stopgap for the Muse multi-chunk prefill bad_alloc (see _TODO.md 89): the text_prefill
    // recipe under-estimates the per-call peak (34.5 MiB planned vs >=55 MiB used at
    // prefill_chunk=512), so the bump arena ran out. The recipe still needs reconciling; until
    // then the arena gets headroom, which is safe because every buffer keeps its own size.
    // NINFER_WS_HEADROOM_PCT overrides the default 100% (i.e. 2x) for experiments.
    {
        static const std::uint32_t headroom_pct = [] {
            const char* value = std::getenv("NINFER_WS_HEADROOM_PCT");
            const long parsed = value == nullptr ? 100 : std::atol(value);
            return parsed >= 0 && parsed <= 1000 ? static_cast<std::uint32_t>(parsed) : 100U;
        }();
        out.general_capacity += out.general_capacity * headroom_pct / 100U;
    }
    out.capacity = out.general_capacity;
    if (std::getenv("NINFER_WS_DUMP") != nullptr) {
        std::fprintf(stderr,
                     "[ws] chunk=%u text_prefill=%zu ordinary=%zu mtp_prefill=%zu mtp_round=%zu "
                     "general=%zu\n",
                     chunk_u32, out.text_prefill, out.ordinary_round, out.mtp_prefill,
                     out.mtp_round, out.general_capacity);
    }
    if (plan.features.vision) {
        const std::uint32_t merged = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(plan.capacity, kMaximumVisionItemTokens));
        out.vision   = schedule::VisionContext::plan_workspace(merged, out.general_capacity);
        out.capacity = std::max(out.capacity, out.vision->capacity_bytes);
    }
    return out;
}

// 这次运行**真的会用到的格式** -> 能力需求 (core/device_capabilities.h 的表)。
// 依据是 options 里已经落定的东西, 不是任何核心编号:
//   * --kv-dtype / 注册的逐层默认表 -> KvCacheStorage (layouts_impl.h 自己后面
//     在 layer_overrides 上用的就是这一份, 见 build_sequence_candidate);
//   * --kv-tier-formats / --nvfp4-mode pure -> 融合档 (nvfp4/iso4e/rk4v4) 的 V 面 codec。
// 每条需求都带上"是谁要的", 失败消息直接引用它。
struct CapabilityRequirement {
    // 满足这条需求**必须**有位的能力之一 (消息用它开题); needs 是完整需求。
    DeviceCapability capability;
    std::string source;
    // 完整需求: 合取 + 可选析取 (core/device_capabilities.h 的 CapabilityNeeds)。
    // 这里存整份而不是只存 capability, 是因为 bf16 槽位有两条可选路线: 只带着一位走完这个循环
    // 会把一条析取需求压成合取, 也就是又回到"两条都要"那个错误答案上。
    CapabilityNeeds needs = {};
};

void require(std::vector<CapabilityRequirement>& out, DeviceCapability capability,
             std::string source, CapabilityNeeds needs = {}) {
    for (const CapabilityRequirement& existing : out) {
        if (existing.capability == capability) { return; }
    }
    if (needs.all_of == kNoCapabilities) { needs.all_of = capability_bit(capability); }
    out.push_back(CapabilityRequirement{capability, std::move(source), needs});
}

std::vector<CapabilityRequirement> required_capabilities(const EngineOptions& options) {
    std::vector<CapabilityRequirement> required;
    // 基线永远要: 本 build 的 cubin 在这台设备上到底起不起得来 kernel。
    require(required, DeviceCapability::KernelImage, "the build's kernel image on this device");

    const auto from_storage = [&](KvCacheStorage storage, const std::string& source) {
        const CapabilityNeeds needs = requirements_for_kv_storage(storage);
        // 合取部分的每一位都要记; 完整需求 (含析取) 只挂在第一位上记一次 -- 判定结果一样, 但同一条
        // 需求被判两次会在日志和异常里各出现两次。
        bool needs_attached = false;
        for (std::size_t i = 0; i < kDeviceCapabilityCount; ++i) {
            const auto capability = static_cast<DeviceCapability>(i);
            if (!set_contains(needs.all_of, capability)) { continue; }
            if (!needs_attached && needs_has_disjunction(needs)) {
                require(required, capability, source, needs);
                needs_attached = true;
            } else {
                require(required, capability, source);
            }
        }
    };

    if (options.kv_cache_explicit) {
        from_storage(options.kv_cache, std::string("--kv-dtype ") +
                                          std::string(kv_storage_name(options.kv_cache)));
    }
    if (options.kv_layer_storage_explicit) {
        for (std::size_t layer = 0; layer < options.kv_layer_storage.size(); ++layer) {
            const KvCacheStorage storage = options.kv_layer_storage[layer];
            // 表里的 BF16 槽位继承全局 dtype, 不是一次选择 (见 types.h:221-225) -- UNLESS the
            // spec WROTE that slot (EngineOptions::kv_layer_storage_set). A written bf16
            // slot IS a choice: it builds a real BF16 layer, so it needs the same
            // Bf16Mma the global --kv-dtype bf16 path already requires
            // (core/device_capabilities.h). Without this the capability preflight
            // would describe a different stack than plan_cache() builds -- the two
            // halves of one guard disagreeing, again. The mask is all-false for every
            // caller that predates it, so this adds no requirement to any existing
            // configuration.
            const bool written =
                layer < options.kv_layer_storage_set.size() && options.kv_layer_storage_set[layer];
            if (storage == KvCacheStorage::BFloat16 && !written) { continue; }
            from_storage(storage, "--kv-layer-storage[" + std::to_string(layer) + "]=" +
                                      std::string(kv_storage_name(storage)));
        }
    }
    if (options.kv_nvfp4_pure) {
        from_storage(KvCacheStorage::Nvfp4Group16,
                     "--nvfp4-mode pure (the NVFP4 fusion tier is the only tier left)");
    }
    if (options.kv_v_codec_explicit) {
        // --kv-v-codec iso4e|e2m1 只改 NVFP4 档 V 面的解码分支 (types.h:46-53), 不改架构;
        // 记一条来源, 让失败消息能说清是哪个开关要的。
        require(required, DeviceCapability::Nvfp4MmaBlockScale,
                std::string("--kv-v-codec ") +
                    (options.kv_v_codec == KvVCodec::E2M1 ? "e2m1" : "iso4e") +
                    " (NVFP4 tier V-plane codec)");
    }
    return required;
}

void validate_target_options(DeviceContext& device, const EngineOptions& options) {
    // Static YaRN factor-4 extends the rope domain to 4x the native context.
    const std::uint32_t context_limit =
        options.yarn_enabled ? 4ULL * Variant::maximum_context : Variant::maximum_context;
    if (options.max_context == 0 || options.max_context > context_limit) {
        throw std::invalid_argument(
            options.yarn_enabled
                ? "max_context exceeds the variant YaRN-extended context capacity"
                : "max_context exceeds the variant native context capacity");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("max_concurrency must be in [1,16]");
    }
    const std::uint32_t logical_pages = page_count(options.max_context);
    // An explicit --kv-capacity may floor the device page pool below max_context
    // (YaRN extends the rope domain beyond the pool; the cold pool recycles pages).
    std::uint32_t minimum_pages = std::max(logical_pages, options.max_concurrency);
    if (options.kv_capacity.mode == KvCapacityMode::Explicit) {
        minimum_pages = std::max(page_count(options.kv_capacity.explicit_tokens),
                                 options.max_concurrency);
    }
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(options.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit: {
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("kv_capacity must be positive");
        }
        const std::uint32_t requested_pages = page_count(options.kv_capacity.explicit_tokens);
        if (requested_pages < minimum_pages || requested_pages > maximum_pages64) {
            throw std::invalid_argument(
                "kv_capacity is outside the usable range for max_context and max_concurrency");
        }
        break;
    }
    case KvCapacityMode::Automatic:
        break;
    default:
        throw std::invalid_argument("unknown kv_capacity policy");
    }
    switch (options.speculative.backend) {
    case SpeculativeBackend::None:
        if (options.speculative.draft_tokens != 0 ||
            options.speculative.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "disabled speculative decoding requires draft_tokens=0 and the full proposal head");
        }
        break;
    case SpeculativeBackend::Mtp:
        // 0 = adaptive (the capture ladder below); >0 = one pinned captured width. The ladder's
        // top rung has to fit this target's MTP draft domain, and it is the planner that clamps
        // the ladder to kMaximumMtpDraftTokens, so only the explicit width is bounded here.
        if (options.speculative.draft_tokens > kMaximumMtpDraftTokens) {
            throw std::invalid_argument("MTP draft window must be 0 (adaptive) or in [1,15]");
        }
        // The tree's node budget IS the draft width (one live verify column per node over the
        // anchor), so it is bounded by the same domain. The CLI already refuses a malformed
        // shape; the planner repeats the check because the server also reaches it.
        if (options.speculative.draft_tree_paths * options.speculative.draft_tree_depth >
            kMaximumMtpDraftTokens) {
            throw std::invalid_argument(
                "MTP draft tree L,d must satisfy L*d <= 15 nodes (one verify column per node "
                "over the anchor)");
        }
        // --draft-tree L,d (L > 1) is published from the proposal head's own top-L rows, and those
        // rows are TOKEN IDS only under the full head. The shortlist (Optimized) head never writes
        // the round's proposal-logits frame at all -- TextContext::proposal_argmax runs its own
        // {proposal_head_n_, T} scratch and remaps the argmax through proposal_head_ids_
        // (text_context_impl.h:697-703) -- so its top-L indices would be published as token ids
        // that are really shortlist positions, read out of rows nothing wrote. include/
        // ninfer/ops/mtp_proposal_topk.h:27-28 CLAIMS this refusal already existed here; it did
        // not, and this is it. L <= 1 is the chain spelling, so the guard must not fire for it.
        {
            const std::string tree_head_defect = qwen3_6::detail::mtp_proposal_head_defect(
                options.speculative.proposal_head != ProposalHead::Full,
                options.speculative.draft_tree_paths, options.speculative.draft_tree_depth);
            if (!tree_head_defect.empty()) { throw std::invalid_argument(tree_head_defect); }
        }
        break;
    case SpeculativeBackend::DFlash2:
        // --draft-tokens picks the startup-fixed draft window K; 0 keeps the
        // historical 7-draft default, normalized by the planner below. The bound
        // is enforced here too so the planner never sees an out-of-frame draft
        // window (empty graph profiles / over-wide layouts).
        if (options.speculative.draft_tokens > qwen3_6::kDFlashDecodeMaximumDrafts) {
            throw std::invalid_argument(
                "DFlash2 draft window must be 0 (default 7) or in [1,15]");
        }
        // [F-1019] F-1006's COMPLEMENTARITY, CLOSED. The DFlash arm below refuses
        // `options.enable_vision` by name; this arm -- the SIBLING SPELLING of the same unvalidated
        // pair -- had no vision check at all. With the silent substitution in the qwen3_6_27b
        // package removed (P3b there), `--spec auto --vision` on a DFlash2 artifact now RESOLVES to
        // DFlash2, so this arm is the one that must refuse it. The reason category is the same one
        // the DFlash arm states: a POLICY refusal, not an artifact limit -- 9 of the 91 readable
        // `.ninfer` artifacts carry a complete DFlash2 head and a complete vision tower at once,
        // and what does not exist is a validation of the two together.
        if (options.enable_vision) {
            throw std::invalid_argument(
                "DFlash2 and Vision are not co-validated as a pair: refusing to run an "
                "unvalidated combination. This is a policy refusal, not an artifact limit -- "
                "one artifact does carry the DFlash2 draft head and the vision tower at once "
                "(dl/unlock/logs/20_artifact_census.txt).");
        }
        break;
    case SpeculativeBackend::DFlash:
        if (kMaximumDFlashDraftTokens == 0) {
            throw std::invalid_argument("DFlash is not supported by this target");
        }
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumDFlashDraftTokens) {
            throw std::invalid_argument("DFlash draft window must be in [1,15]");
        }
        if (options.enable_vision) {
            // landq/unlock -- a POLICY refusal whose old wording ("cannot be enabled together")
            // read like a physical law. The draft head and the vision tower do coexist in one
            // artifact (byte caliber: 9 of the 91 readable .ninfer artifacts on this box carry
            // the four dflash2/* entry objects and all five vision/* entry objects), so what is
            // missing is a VALIDATED PAIR, not a possibility. The refusal stays; the category
            // is now stated.
            throw std::invalid_argument(
                "DFlash and Vision are not co-validated as a pair: refusing to run an "
                "unvalidated combination. This is a policy refusal, not an artifact limit -- "
                "one artifact does carry the draft head and the vision tower at once.");
        }
        break;
    case SpeculativeBackend::Auto:
        // TOLERATED ON PURPOSE, and named rather than defaulted.  Auto is resolved
        // before planning by construction: construct_registered() calls
        // Target::resolved_auto_speculative() up front "so the planner, the load plan
        // and the program all see the same concrete backend" (src/targets/registry.cpp
        // :105-110, and again before re-planning at :361), and each package resolves it
        // the same way (src/targets/qwen3_6_27b/impl/package.cpp:84-100).  So this arm
        // is not a hole in the validation: it states that Auto has nothing left to
        // validate here.  It does NOT throw, because nothing in this line establishes
        // that Auto is unreachable on EVERY path into the planner, and converting a
        // skip into a startup failure on an unproven reachability claim is exactly the
        // sort of change this line must not make silently.  (A stricter throw is
        // reported as a candidate, not landed.)
        break;
    }
    // Capability gate (core/device_capabilities.h). 这里原本是
    //   `if (device.sm() != 120) throw ...`   (2026-09-13 删除)
    // 它按核心编号推断"这台卡能不能跑", 而 device.sm() = major*10+minor
    // (src/core/device.cu:122) 连 'a' 后缀都看不见 —— sm_120 与 sm_120a 在运行期
    // 是同一个数字, 而引擎真正依赖的是后者 (kind::mxf4nvf4 只在 120a/121a 可汇编,
    // 实测见 tools/archkit/_GPU_MATRIX.md 的指令底线表)。
    //
    // 现在改成实测: 把这次运行会用到的格式翻译成能力需求 (required_capabilities),
    // 每个能力跑一次极小探针 (真起 kernel + 与参考值比数), 只有探针没过的能力才拒绝,
    // 拒绝时把"跑了什么/期望什么/实测到什么/这条能力来自哪个 kernel"一并给出。
    // 探针每设备只跑一次, 结论缓存在 core/device_probe.cu 里。
    {
        // 探针按"当前线程绑定的 CUDA 设备"跑, 而 DeviceContext 构造时绑的就是
        // device.device (device.cu:294-299)。这里只做一致性检查 —— 门禁不再读任何
        // 设备属性 (major/minor) 来判断能力, 只看探针结论。
        int bound = -1;
        if (cudaGetDevice(&bound) == cudaSuccess && bound != device.device) {
            throw std::runtime_error(
                "capability probe device mismatch: DeviceContext holds device " +
                std::to_string(device.device) + " but this thread is bound to " +
                std::to_string(bound));
        }
        const CapabilityReport& capabilities = probe_device_capabilities();
        for (const CapabilityRequirement& needed : required_capabilities(options)) {
            if (needs_has_disjunction(needed.needs)) {
                // 有可选路线的需求: "满足了吗" 和 "由谁满足" 是两个问题, 第二个才是"哪个 kernel
                // 会上"。两条都没了才拒绝, 而拒绝时 capability_needs_failure_message() 会把**每一条
                // 可选路线**分别点名。
                const NeedsResolution resolution = resolve_needs(needed.needs, capabilities);
                if (!resolution.satisfied) {
                    throw std::invalid_argument(capability_needs_failure_message(
                        needed.needs, capabilities, current_device_facts(), needed.source));
                }
                if (resolution.served_by.has_value()) {
                    std::fprintf(stderr, "[caps] %s: requirement met by %s\n",
                                 needed.source.c_str(),
                                 capability_arm_text(*resolution.served_by).c_str());
                }
                continue;
            }
            if (capabilities.supported(needed.capability)) { continue; }
            throw std::invalid_argument(capability_failure_message(
                needed.capability, capabilities, current_device_facts(), needed.source));
        }
        if (std::getenv("NINFER_PROBE_REPORT") != nullptr) {
            std::fprintf(stderr, "[caps] device %s\n%s",
                         device_facts_line(current_device_facts()).c_str(),
                         capabilities.table().c_str());
        }
    }
}

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups) {
    if (main_page_groups == 0) {
        throw std::invalid_argument("Main KV physical page count must be positive");
    }
    auto impl                 = std::make_unique<SequencePlanImpl>();
    impl->weights_profile     = inputs.weights_profile;
    impl->capacity            = inputs.capacity;
    impl->main_page_groups    = main_page_groups;
    impl->kv_capacity         = static_cast<std::uint32_t>(checked_i32(
        static_cast<std::uint64_t>(main_page_groups) * static_cast<std::uint32_t>(kPagedKVPageSize),
        "resolved Paged KV capacity exceeds int32"));
    impl->max_concurrency     = inputs.max_concurrency;
    impl->prefill_chunk       = inputs.prefill_chunk;
    impl->draft_window        = inputs.draft_window;
    impl->mtp_ladder          = inputs.mtp_ladder;
    impl->draft_tree_paths    = inputs.draft_tree_paths;
    impl->draft_tree_depth    = inputs.draft_tree_depth;
    impl->speculative_backend = inputs.speculative_backend;
    impl->proposal_head       = inputs.proposal_head;
    impl->features            = inputs.features;
    impl->use_cuda_graph      = inputs.use_cuda_graph;
    impl->cold_policy      = inputs.cold_policy;
    impl->cold_keep_tokens = inputs.cold_keep_tokens;
    impl->unload_watermark_pages = inputs.unload_watermark_pages;
    impl->max_cold_pages   = inputs.max_cold_pages;
    impl->cold_host_bytes  = inputs.cold_host_bytes;
    impl->cold_disk_path   = inputs.cold_disk_path;
    impl->cold_disk_bytes  = inputs.cold_disk_bytes;
    // The two cold byte budgets are config, not runtime state: a tier the
    // operator explicitly selected whose cap can never admit one page is
    // rejected here, before anything is laid out (the same loud-contradiction
    // rule effective_cold_pages applies to --max-cold-pages + host). The
    // capacities in pages are derived later, where the strides are known.
    product::validate_cold_tier_budget(product::cold_tier_budget_from(
        impl->cold_policy, impl->cold_host_bytes, impl->cold_disk_bytes));
    impl->causal_scoring      = inputs.causal_scoring;
    impl->graph_capture_ceiling = inputs.graph_capture_ceiling;
    impl->device              = inputs.device;
    impl->context_cache       = inputs.context_cache;
    impl->kv_dtype            = inputs.kv_dtype;
    impl->kv_quant_group      = inputs.kv_quant_group;
    impl->layer_kv_dtypes     = inputs.layer_kv_dtypes;
    // The "was this slot written?" mask travels with the table it belongs to, or
    // the per-layer BF16 request would be dropped exactly here.
    impl->layer_kv_dtypes_set = inputs.layer_kv_dtypes_set;
    impl->kv_residual_layers  = inputs.kv_residual_layers;
    impl->kv_v_codec          = inputs.kv_v_codec;
    impl->kv_rotation_off     = inputs.kv_rotation_off;
    impl->kv_row_scale_spec   = inputs.kv_row_scale_spec;
    // --stage-layers: the spec and its boundary travel from the inputs to the plan unchanged,
    // in the same block as the per-layer row-scale spec because both are operator text that
    // downstream code parses rather than interprets.
    impl->stage_layers_spec   = inputs.stage_layers_spec;
    impl->stage_handoff_dir   = inputs.stage_handoff_dir;
    impl->stage_handoff_cut   = inputs.stage_handoff_cut;
    impl->persistent          = persistent_layout(*impl);
    impl->workspace           = build_workspace_plan(*impl);
    if (impl->use_cuda_graph) {
        // Definitions remain per execution profile, but only one executable is instantiated for
        // each reachable node-topology class. These bounds cover the largest profile installed in
        // each class and the driver/module state materialized while qualifying all definitions.
        if (impl->speculative_backend == SpeculativeBackend::None) {
            impl->graph_allowance_bytes = checked_mul(12ULL * kMiB, impl->max_concurrency,
                                                      "ordinary exact-b graph allowance");
        } else if (impl->speculative_backend == SpeculativeBackend::Mtp) {
            // The ladder costs one executable per rung per batch size (each rung is its own
            // topology class, because a width switch must not ask cudaGraphExecUpdate to change
            // a node's kernel function). graph_topology_allowance sums the per-class maxima, so
            // the reservation grows |ladder|-fold: at a 262144 context the MTP allowance goes
            // from 82 MiB to |ladder| * 82 MiB per batch size. That is the honest price of the
            // ladder; a fixed-k run (empty ladder) is unchanged.
            const auto profiles =
                mtp_graph_ladder_profiles(impl->capacity, impl->draft_window, impl->mtp_ladder);
            const std::size_t per_batch_allowance = graph_topology_allowance(
                profiles,
                [&](GraphExecutionProfile profile) {
                    const std::uint32_t width =
                        profile.draft_width != 0 ? profile.draft_width : impl->draft_window;
                    const std::uint64_t final_visible = std::min<std::uint64_t>(
                        impl->capacity, static_cast<std::uint64_t>(profile.max) + 2ULL * width);
                    return (final_visible <= 4096 ? 12ULL : 82ULL) * kMiB;
                },
                "MTP graph allowance");
            impl->graph_allowance_bytes = checked_mul(per_batch_allowance, impl->max_concurrency,
                                                      "MTP exact-b graph allowance");
        } else {
            const auto class_allowance = [&](std::uint32_t batch_size) {
                const auto profiles =
                    dflash_graph_profiles(impl->capacity, impl->draft_window, batch_size);
                return graph_topology_allowance(
                    profiles,
                    [&](GraphExecutionProfile profile) {
                        const std::uint64_t final_visible = std::min<std::uint64_t>(
                            impl->capacity,
                            static_cast<std::uint64_t>(profile.max) + impl->draft_window + 1ULL);
                        return (final_visible <= 4096 ? 64ULL : 96ULL) * kMiB;
                    },
                    "DFlash graph allowance");
            };
            for (std::uint32_t batch_size = 1; batch_size <= impl->max_concurrency; ++batch_size) {
                impl->graph_allowance_bytes =
                    checked_add(impl->graph_allowance_bytes, class_allowance(batch_size),
                                "DFlash exact-b graph allowance");
            }
        }
    }

    impl->device_reservation_bytes = checked_add(
        checked_add(impl->persistent.bytes, impl->workspace.capacity, "sequence memory plan"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    return impl;
}

} // namespace

std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile) {
    validate_target_options(device, options);
    const TargetKVCacheProfile kv_profile = target_kv_cache_profile(options.kv_cache);

    // N1/S30: --kv-bit-budget deferred resolution. The CLI parse site has no
    // model knowledge; here TextConfig::full_attention_layers() is available.
    // The DP (product/kv_bit_budget.h, parity-proven against
    // tools/archkit/kv_bit_budget.py, _collab/A_kvbit_cold_cpp.md) resolves the
    // budget TOGETHER with the cold capacity: cold_cap = effective_cold_pages,
    // the same helper (and precedence) the layout reservation uses, so the DP
    // can never plan more cold layers than the pool that will exist. With no
    // cold capacity (cold_cap = 0: no policy, ColdPolicy::None/Host, or a zero
    // cap) the DP runs hot-only and the resolved table is byte-identical to
    // the N1 path (S12/S14 goldens).
    std::array<KvCacheStorage, 64> budget_storage_table = options.kv_layer_storage;
    bool storage_explicit = options.kv_layer_storage_explicit;
    // ... and the mask those two came with. Every one of the three sources below
    // (options, the --kv-tier-formats `hot` override, the bit-budget deployment)
    // has to move the table AND the mask together, or a source that names a slot
    // stops being distinguishable from one that left it unset.
    std::array<bool, 64> budget_storage_set = options.kv_layer_storage_set;

    // ===================================================================================
    // F903 / kvcompose -- THE PIN SET, FROZEN BEFORE ANY RESOLUTION TOUCHES THE TABLE.
    // ===================================================================================
    // THE CONSTRUCTION THAT IS BEING REPLACED, and why IT could not compose. The old schema
    // had ONE boolean for TWO different facts:
    //     storage_explicit == "a table is present"  AND  "the table is COMPLETE, so no
    //                          ceiling may run over it".
    // A ceiling resolves into the WHOLE table (deploy_kv_budget_spec replaces table and mask
    // for all 64 slots), so once `storage_explicit` was true the only way to keep the ceiling
    // from being "accepted and read by nothing" was to REFUSE the pair. That is what the CLI
    // did (`--kv-bit-budget and --kv-layer-storage are mutually exclusive`) and it is why the
    // pair read as non-composable -- not because a ceiling and a table cannot describe one
    // stack, but because this function had one slot for two facts.
    // THE ENGINE ALREADY HAD THE VOCABULARY: EngineOptions::kv_layer_storage_set is exactly
    // "WHICH slots the per-layer spec actually WROTE" (include/ninfer/types.h:621-646), and
    // the file already relies on that distinction for the extra capabilities a WRITTEN bf16
    // slot requires. The two facts were never separated HERE.
    // THE PIN SET separates them: kv_pin_set is the mask the OPERATOR'S --kv-layer-storage
    // arrived with, frozen at this point, before --kv-tier-formats may fill it (it fills all
    // 64) and before either ceiling may replace the table. Its consequence, stated exactly:
    //   * a ceiling now runs over the WHOLE stack, exactly as it does when no table was given;
    //   * the pins are applied OVER that result, on the slots they name;
    //   * so both knobs are READ, on different slots, in ONE run. That is composition.
    const std::array<KvCacheStorage, 64> kv_pin_table = budget_storage_table;
    const std::array<bool, 64> kv_pin_set = budget_storage_set;

    // --kv-tier-formats / --nvfp4-mode: the KV tier vocabulary (hot/tail/cold +
    // nvfp4 mode). Resolved HERE, not at the parse site, for the same reason as
    // --kv-bit-budget below: `hot` is the resident format of every full-attention
    // layer, so the landing needs this target's layer count, and the mode/cold
    // gates must judge the table this plan will really build. Stage 1 (here)
    // applies `hot` to the same storage table the other knobs produce, so the
    // engine keeps exactly one resolution path into the per-layer dtypes; stage 2
    // (after layer_overrides) judges mode/cold on the effective per-layer dtype.
    // Whatever the engine cannot express -- the `tail` tier, the unreachable cold
    // codecs -- is refused with the missing mechanism named, never approximated.
    std::optional<product::KvTierPlan> tier_plan;
    // Set by the ARMING GATE below when a requested cold pool cannot be filled by the
    // table this plan will really build: the two designators the pool reads are then
    // cleared before they are handed on.
    bool cold_tier_closed = false;
    if (options.kv_tier_formats_explicit) {
        tier_plan = product::kv_tier_formats_plan(options.kv_tier_formats_spec,
                                                  options.kv_nvfp4_pure, budget_storage_table);
        if (tier_plan->override_hot &&
            (storage_explicit || options.kv_bit_budget_explicit || options.kv_cache_explicit)) {
            throw std::invalid_argument(
                "--kv-tier-formats with an explicit hot format cannot be combined with "
                "--kv-layer-storage / --kv-bit-budget / --kv-dtype: hot already names the "
                "resident format of every full-attention layer, so the two describe the same "
                "table");
        }
        if (tier_plan->override_hot) {
            budget_storage_table = tier_plan->table;
            storage_explicit     = true;
            // `hot` is the resident format of EVERY full-attention layer, so the
            // plan wrote every slot: the mask is all-writes, not all-unset. With
            // the mask left clear, a `hot=bf16` under a quantized --kv-dtype would
            // keep inheriting it and the flag would do nothing.
            budget_storage_set.fill(true);
        }
    }

    // The target's full-attention layer count and the cold capacity the allocator may
    // spend, resolved ONCE for every budget entry below (--kv-bit-budget, and the two K/V
    // entries of product/kv_kv_bits.h). cold_cap = effective_cold_pages is the same helper
    // (and precedence) the layout reservation uses, so no entry can plan more cold layers
    // than the pool that will exist.
    const std::int32_t full_layers =
        static_cast<std::int32_t>(TextConfig::full_attention_layers());
    const std::uint32_t cold_pages = effective_cold_pages(
        options.cold_policy, options.cold_keep_tokens, options.max_cold_pages);

    // F903: how many pins can reach the pool at all. A slot at or past this target's
    // full-attention count parses, is accepted, and changes nothing on this model -- the
    // inert-slot report far below says so, and counting it here would make the composition
    // condition lie about how much of the stack the operator actually pinned.
    std::size_t pin_count = 0;
    for (std::int32_t l = 0; l < full_layers; ++l) {
        if (kv_pin_set[static_cast<std::size_t>(l)]) { ++pin_count; }
    }
    // THE WHOLE COMPOSITION CONDITION, in one line: a proper, NON-EMPTY subset of the stack
    // was named. Both ends matter -- empty means there are no pins (and the ceiling's own
    // plan is the table, the pre-F903 behaviour), and the FULL stack means there is nothing
    // left for the ceiling to decide, which is refused below by name rather than accepted
    // and ignored.
    const bool pin_and_solve = pin_count != 0 &&
                               pin_count < static_cast<std::size_t>(full_layers);
    // The pins, applied OVER whatever a ceiling just resolved. ONE implementation, called by
    // BOTH ceiling entries after their own deploy, so neither can forget it and the two
    // entries cannot drift.
    const auto apply_kv_pins = [&]() {
        for (std::size_t i = 0; i < kv_pin_set.size(); ++i) {
            if (!kv_pin_set[i]) { continue; }
            budget_storage_table[i] = kv_pin_table[i];
            budget_storage_set[i]   = true;
        }
        return pin_count;
    };
    // The pinned runs, as the operator would spell them, for the report line. A run is
    // maximal and ascending, so `0,2-3` is one pin set and cannot be confused with `0-3`.
    const auto kv_pin_runs_text = [&]() {
        std::string runs;
        for (std::int32_t l = 0; l < full_layers; ++l) {
            if (!kv_pin_set[static_cast<std::size_t>(l)]) { continue; }
            std::int32_t last = l;
            while (last + 1 < full_layers && kv_pin_set[static_cast<std::size_t>(last + 1)]) {
                ++last;
            }
            if (!runs.empty()) { runs += ','; }
            runs += std::to_string(l);
            if (last != l) { runs += '-'; runs += std::to_string(last); }
            l = last;
        }
        return runs;
    };
    // THE COMPOSED SPEC, one entry per layer, so the `hot=` field of a ceiling's own report
    // line reports the table that will be BUILT rather than the ceiling's plan with the pins
    // left out -- which would be a FALSE READING produced by this very line. The digits are
    // scanned by hand (no <cstdlib>) and the result is RE-PARSED with the tree's own parser
    // and compared slot by slot against the final table, so a name that does not round-trip is
    // REPORTED as roundtrip=false instead of being printed as a fact.
    const auto kv_pins_composed_spec = [&](const std::string& budget_spec) {
        std::string out;
        std::size_t cursor = 0;
        while (cursor < budget_spec.size()) {
            const std::size_t comma = budget_spec.find(',', cursor);
            const std::string item =
                budget_spec.substr(cursor, comma == std::string::npos
                                              ? std::string::npos
                                              : comma - cursor);
            const std::size_t colon = item.rfind(':');
            if (colon != std::string::npos) {
                const std::string ranges = item.substr(0, colon);
                const std::size_t dash = ranges.find('-');
                const auto number = [](const std::string& text, std::size_t from,
                                       std::size_t to) {
                    std::int32_t value = 0;
                    for (std::size_t i = from; i < to && i < text.size(); ++i) {
                        const char c = text[i];
                        if (c < '0' || c > '9') { break; }
                        value = value * 10 + (c - '0');
                    }
                    return value;
                };
                const std::int32_t first = number(ranges, 0, ranges.size());
                const std::int32_t last =
                    dash == std::string::npos
                        ? first
                        : number(ranges, dash + 1, ranges.size());
                const std::string tier = item.substr(colon + 1);
                for (std::int32_t l = first; l <= last && l < full_layers; ++l) {
                    if (!out.empty()) { out += ','; }
                    out += std::to_string(l);
                    out += ':';
                    out += kv_pin_set[static_cast<std::size_t>(l)]
                               ? std::string(kv_storage_name(
                                     kv_pin_table[static_cast<std::size_t>(l)]))
                               : tier;
                }
            }
            if (comma == std::string::npos) { break; }
            cursor = comma + 1;
        }
        return out;
    };
    // The round-trip, run on the FINAL table, so the printed spec cannot be a second spelling
    // of something else. Returns false when the parser disagrees with the table it names.
    const auto kv_pins_spec_roundtrips = [&](const std::string& text) {
        const auto parsed = product::parse_kv_layer_storage_spec(text);
        for (std::int32_t l = 0; l < full_layers; ++l) {
            const std::size_t i = static_cast<std::size_t>(l);
            if (parsed.set[i] != budget_storage_set[i]) { return false; }
            if (parsed.set[i] && parsed.table[i] != budget_storage_table[i]) { return false; }
        }
        return true;
    };

    // Two spellings of the same ceiling cannot be merged: they would describe different
    // stacks, and the allocation would depend on which one happened to be applied last.
    if (options.kv_kv_bits_explicit && options.kv_bit_budget_explicit) {
        throw std::invalid_argument(
            "--kv-bit-budget and --kv-bits/--kv-k-bits/--kv-v-bits are two spellings of the "
            "same ceiling set: --kv-bit-budget is the plane-agnostic form, --kv-bits (or "
            "--kv-k-bits + --kv-v-bits) the K/V form. Give one of them");
    }
    // The speed/quality slider is read ONLY inside a bit-budget resolution. Without a
    // ceiling it used to be accepted and then ignored -- this project's worst outcome -- so
    // it is refused here, where both front ends meet, with the flags that would make it act
    // named.
    if (options.kv_quality_weight >= 0.0 && !options.kv_bit_budget_explicit &&
        !options.kv_kv_bits_explicit) {
        throw std::invalid_argument(
            "--kv-quality-weight / --kv-tier-scores need a ceiling to act on: both are read "
            "only inside the bit-budget fit. Add --kv-bits, --kv-k-bits/--kv-v-bits or "
            "--kv-bit-budget, or drop the flag");
    }

    // res-kv5 FIX (--kv-dtype dropped by a ceiling), RE-STATED by F903 now that the
    // construction has changed -- the rule and the measurement behind it are UNCHANGED.
    // The reason is --kv-dtype's SCOPE and not "a ceiling cannot coexist with a table":
    // --kv-dtype `layer_overrides.fill(kv_profile.dtype)` names the GLOBAL tier for every
    // layer, and a ceiling TILES every full-attention layer, so there is no slot left for the
    // global tier to be the default of and the dtype would be read by nothing. Measured on
    // the pre-fix arm: `--kv-bit-budget 4.5 --kv-dtype int8` came out identical in dtype AND
    // payload to `--kv-bit-budget 4.5` alone (per-layer 0-9:rk4v4-group64 10-15:nvfp4-group16,
    // 278.00 MiB), while `--kv-dtype int8` alone is int8-group64 / 528.00 MiB.
    // WHAT F903 ADDS IS THE OTHER HALF OF THE SENTENCE, which the old text did not carry and
    // which cost the operator a working configuration: --kv-layer-storage names a SUBSET of
    // the stack, and a subset DOES compose with either ceiling (the ceiling solves the whole
    // stack and the named slots override it, see the pin set above). The composition is not
    // unavailable; it is spelled with the knob that can name a subset.
    if (options.kv_cache_explicit &&
        (options.kv_bit_budget_explicit || options.kv_kv_bits_explicit)) {
        throw std::invalid_argument(
            "--kv-dtype and --kv-bit-budget/--kv-bits are mutually exclusive: --kv-dtype "
            "pins the GLOBAL KV tier (every layer) and a ceiling tiles every layer, so the "
            "dtype would be read by nothing. Measured: --kv-bit-budget 4.5 --kv-dtype int8 "
            "deployed the same per-layer table as --kv-bit-budget 4.5 alone (278.00 MiB), "
            "while --kv-dtype int8 alone is 528.00 MiB. Use --kv-layer-storage to name the "
            "layers you mean -- a SUBSET composes with the ceiling -- or give one of them.");
    }
    // ONE deployment path for BOTH budget entries: the allocator's spec becomes the
    // per-layer storage table, "cold" is NOT a --kv-layer-storage tier, so a cold-planned
    // layer keeps a HOT NVFP4 window (cold slots hold requantized E2M1-family data) and its
    // ranges are routed to the cold pool. Shared so the two entries cannot drift.
    struct KvDeployedSpec {
        std::string hot_spec;
        std::string cold_ranges;
    };
    const auto deploy_kv_budget_spec = [&](const std::string& spec) -> KvDeployedSpec {
        KvDeployedSpec deployed;
        std::size_t cursor = 0;
        while (cursor < spec.size()) {
            const std::size_t comma = spec.find(',', cursor);
            const std::string item =
                spec.substr(cursor, comma == std::string::npos ? spec.size() - cursor
                                                               : comma - cursor);
            const std::size_t colon = item.rfind(':');
            const std::string ranges = item.substr(0, colon);
            const std::string tier = item.substr(colon + 1);
            if (!deployed.hot_spec.empty()) { deployed.hot_spec += ","; }
            if (tier == "cold") {
                if (!deployed.cold_ranges.empty()) { deployed.cold_ranges += ","; }
                deployed.cold_ranges += ranges;
                deployed.hot_spec += ranges + ":nvfp4";
            } else {
                deployed.hot_spec += item;
            }
            if (comma == std::string::npos) { break; }
            cursor = comma + 1;
        }
        // Table AND mask from the SAME parse. This entry REPLACES the whole
        // per-layer table, so every slot the deployed hot spec did not name is
        // now unset -- taking the table without the mask would silently mark the
        // named slots as "unset bf16" and hand them back to the global dtype.
        const auto deployed_spec = product::parse_kv_layer_storage_spec(deployed.hot_spec);
        budget_storage_table = deployed_spec.table;
        budget_storage_set   = deployed_spec.set;
        storage_explicit = true;
        return deployed;
    };

    // F903: this guard used to read `&& !storage_explicit`, i.e. "a table was given, so the
    // ceiling is skipped" -- and THAT is what made the CLI refuse the pair: an accepted flag
    // that is read by nothing is this project's worst outcome, so the door was closed rather
    // than the seam opened. With a PIN SET the ceiling has work to do even when a table is
    // present, so it runs, and the pins are applied over its plan. The ONE case that still
    // cannot compose is a table that names EVERY layer; it is refused below, by name.
    if (options.kv_bit_budget_explicit && (!storage_explicit || pin_and_solve)) {
        if (storage_explicit && !pin_and_solve && pin_count != 0) {
            throw std::invalid_argument(
                "--kv-bit-budget with a --kv-layer-storage table that names EVERY "
                "full-attention layer: the ceiling would resolve into a table that already "
                "exists, i.e. be accepted and read by nothing. Name FEWER layers with "
                "--kv-layer-storage (the named slots then OVERRIDE the ceiling and the rest "
                "is solved by it), or drop the ceiling.");
        }
        // F911 / SLIDERWIRE-COMPOSE-COMPLETE: THE REFUSAL THAT STOOD HERE IS GONE, and it is
        // gone because the plumbing it named was added rather than because the check was
        // removed. F903 left this sentence and the one above it as the last named residual of
        // the composition -- "kv_bit_budget_scored_ranges takes no candidate_order, and the fix
        // is one parameter in that header". The parameter now exists (defaulted, empty == the
        // shipped order, so every other call site is byte-identical) and the caller below hands
        // it the operator's own --kv-codec-preference. The preference is now READ on both
        // spellings of the ceiling instead of being accepted-and-ignored on one of them, and
        // `accepted and read by nothing` is the outcome this project refuses, so removing the
        // guard without the parameter would have been a fault rather than a repair.
        // Two forms of the same knob: a single ceiling for every full-attention layer, or
        // separable per-range ceilings ("0-7:8,8-15:4.5", 16-layer examples) which the DP minimises per range
        // (globally optimal: additive objective, per-range constraints).
        // Two-score path: when a quality weight is given, the per-tier penalty becomes the
        // weighted sum of the measured quality and speed columns, and the same DP resolves it.
        // The two knobs are orthogonal - one describes the constraint structure, the other the
        // ladder that breaks ties - so all four combinations must exist. The ranges branch
        // comes first for a concrete reason: the range form leaves options.kv_bit_budget_bits
        // at 0, so a scored run that ignored the ranges would resolve every layer against a
        // zero-bit ceiling instead of the ceiling the operator actually gave.
        // THE ABSENT WEIGHT IS THE QUALITY END (product/kv_bit_budget.h
        // kKvQualityWeightQualityEnd) -- the same one-word change product/kv_kv_bits.h
        // kv_bits_solve_plane makes for the K/V spelling of the ceiling, applied to THIS (the
        // --kv-bit-budget) spelling. The scored table therefore runs by default, and because the
        // default table's quality column IS the shipped penalty column row for row (static_assert
        // in product/kv_bit_budget.h) the default plan is unchanged. The explicit-flag refusal at
        // :1528 keeps reading the RAW sentinel, so an absent flag still does not demand a ceiling.
        const double kv_weight = product::kv_quality_weight_resolved(options.kv_quality_weight);
        const bool scored_active = true;
        const bool ranges_given = !options.kv_bit_budget_ranges.empty();
        const std::int32_t rk4v4_limit = product::kKvBitBudgetE8LayerLimit;
        const std::int32_t cold_cap = static_cast<std::int32_t>(cold_pages);
        product::KvTierScoreTable scores = product::kv_bit_budget_default_scores();
        product::KvBitBudgetSolution scored;
        if (scored_active) {
            if (!options.kv_tier_scores.empty()) {
                scores = options.kv_tier_scores.find('\n') != std::string::npos
                             ? product::kv_bit_budget_parse_scores(options.kv_tier_scores)
                             : product::kv_bit_budget_parse_scores([&] {
                                   std::ifstream in(options.kv_tier_scores);
                                   if (!in) {
                                       throw std::invalid_argument(
                                           "kv-tier-scores: cannot open " +
                                           options.kv_tier_scores);
                                   }
                                   return std::string(std::istreambuf_iterator<char>(in),
                                                      std::istreambuf_iterator<char>());
                               }());
            }
            if (!ranges_given) {
                // F903 / SLIDERWIRE-COMPOSE: the operator's candidate order travels INTO
                // the fit. Empty is the shipped order and byte-identical to the pre-F903
                // call, which is what keeps every existing run where it was.
                scored = product::kv_bit_budget_solve_scored(
                    full_layers, options.kv_bit_budget_bits, scores, kv_weight,
                    rk4v4_limit, cold_cap, options.kv_codec_preference);
                // Gap D: the REAL provenance of the two columns, printed from the one
                // place that decides it (product/kv_kv_bits.h kv_scores_provenance). The old
                // text called BOTH columns "measured" while the table it prints is described
                // as provisional at product/kv_bit_budget.h [text: "Provisional default score table"; = :1270 on 2026-09-26], so the report line was
                // the false half of a contradiction in the tree.
                std::fputs(product::kv_scores_provenance(options.kv_tier_scores).line.c_str(),
                           stderr);
                std::fputc('\n', stderr);
                std::fprintf(stderr,
                             "[kv-score] quality_weight=%.2f requested_bits=%.2f "
                             "achieved_bits=%.2f shortfall=%.2f penalty=%.2f spec=%s "
                             "(the weighted ladder picks the best plan that FILLS the "
                             "ceiling from below; it can no longer trade bits away for a "
                             "lower penalty)\n",
                             kv_weight, scored.requested_bits,
                             scored.achieved_bits, scored.shortfall_bits, scored.penalty,
                             scored.spec.c_str());
            }
        }
        std::string spec;
        // BUDGETSAT saturation accounting for the [kv-bit-budget] line below: filled by
        // whichever branch resolved the ceiling. Ranges leave these at 0, because each range
        // has its own ceiling and there is no single requested value to subtract from.
        double budget_achieved  = 0.0;
        double budget_requested = 0.0;
        double budget_shortfall = 0.0;
        if (ranges_given) {
            const auto ranges =
                product::kv_bit_budget_parse_ranges(options.kv_bit_budget_ranges, full_layers);
            spec = scored_active
                       ? product::kv_bit_budget_scored_ranges(
                             full_layers, ranges, scores, kv_weight, rk4v4_limit,
                             cold_cap, options.kv_codec_preference)
                       : product::kv_bit_budget_spec_ranges(full_layers, ranges, rk4v4_limit,
                                                            cold_cap);
            if (scored_active) {
                std::fprintf(stderr,
                             "[kv-score] quality_weight=%.2f ranges=%s spec=%s (the weighted "
                             "speed/quality ladder ran inside each range's own ceiling)\n",
                             kv_weight, options.kv_bit_budget_ranges.c_str(),
                             spec.c_str());
            }
        } else if (scored_active) {
            spec             = scored.spec;
            budget_achieved  = scored.achieved_bits;
            budget_requested = scored.requested_bits;
            budget_shortfall = scored.shortfall_bits;
        } else {
            // The WHOLE solution, not just its spec: the report below has to be able to
            // state the shortfall (product/kv_bit_budget.h: requested_bits/shortfall_bits)
            // instead of leaving the operator to infer it from the spec line.
            const product::KvBitBudgetSolution solved =
                product::kv_bit_budget_solve(full_layers, options.kv_bit_budget_bits, rk4v4_limit,
                                             cold_cap);
            spec             = solved.spec;
            budget_achieved  = solved.achieved_bits;
            budget_requested = solved.requested_bits;
            budget_shortfall = solved.shortfall_bits;
        }
        // One shared deployment path (see deploy_kv_budget_spec above): the spec becomes
        // the per-layer table, "cold" keeps a HOT NVFP4 window and is routed to the cold
        // pool, and this line reports both against the pool that will really exist.
        const KvDeployedSpec deployed = deploy_kv_budget_spec(spec);
        // F903 THE COMPOSITION, AT ITS ONE POINT. deploy_kv_budget_spec just replaced the
        // table AND the mask with the ceiling's own plan for every slot; the pins are applied
        // OVER it here. The ORDER is the semantics and is deliberate: solve first, pin
        // second, so a pinned slot can never be re-solved away by the objective.
        const std::size_t pinned_applied = apply_kv_pins();
        // THE COMPOSED SPEC is what the `hot=` field must show, or the line would report the
        // ceiling's plan with the pins left out -- a false reading produced by this fix.
        const std::string composed_hot = kv_pins_composed_spec(deployed.hot_spec);
        const bool composed_ok = kv_pins_spec_roundtrips(composed_hot);
        // The report line the composition needs, printed BEFORE the ceiling's own line so a
        // reader cannot take `achieved` as the deployed table's average: it is the CEILING's
        // fit over all layers, and the pinned slots are not that. Stated, not implied.
        // PRINTED ONLY WHEN SOMETHING WAS PINNED: a run with no pins must keep the byte-exact
        // stderr every existing gate was calibrated on, so the no-pin path is untouched.
        if (pinned_applied != 0) {
            std::fprintf(stderr,
                         "[kv-pin] --kv-layer-storage pinned=%zu of %d full-attention layer(s) runs=%s; those slots OVERRIDE the ceiling's plan and the remaining %d are the ceiling's own. `achieved`/`shortfall` on the line below describe the CEILING'S FIT OVER ALL %d, not the deployed table, whose average includes the pinned slots. `hot=` below IS the deployed table (pins merged in) and roundtrip=%s.\n",
                         pinned_applied, static_cast<int>(full_layers),
                         kv_pin_runs_text().c_str(),
                         static_cast<int>(full_layers) - static_cast<int>(pinned_applied),
                         static_cast<int>(full_layers), composed_ok ? "true" : "false");
        }
        std::fprintf(stderr,
                     "[kv-bit-budget] full_attention_layers=%d bits=%.2f ranges=%s "
                     "cold_pages=%u cold_placed=%s hot=%s achieved=%.2f shortfall=%.2f "
                     "(the ceiling is FILLED FROM BELOW: max bits under it, then min "
                     "penalty; a nonzero shortfall is the ladder's step here, never a plan "
                     "over the ceiling) (cold residency needs --cold-policy window|disk; "
                     "pool sized by --max-cold-pages >= %u)\n",
                     static_cast<int>(full_layers), options.kv_bit_budget_bits,
                     options.kv_bit_budget_ranges.empty() ? "-"
                                                          : options.kv_bit_budget_ranges.c_str(),
                     static_cast<unsigned>(cold_pages),
                     deployed.cold_ranges.empty() ? "-" : deployed.cold_ranges.c_str(),
                     composed_hot.c_str(), budget_achieved, budget_shortfall,
                     static_cast<unsigned>(cold_pages));
    }


    // --kv-bits / --kv-k-bits / --kv-v-bits / --kv-bits-mode: THE TWO K/V BIT-WIDTH
    // ENTRIES (product/kv_kv_bits.h). Resolved HERE for the same reason as
    // --kv-bit-budget: the target's full-attention layer count is model knowledge and the
    // parse sites have none.
    //   * joint ("合起来整体定")      ONE overall ceiling for the whole KV stack;
    //   * split ("分开定，内部分层")  each plane's per-layer layering solved in its OWN
    //                               budget, then reconciled per layer against the (K format,
    //                               V format) pairs the engine can build.
    // A layer whose K and V requirements meet no single tier is refused BY INDEX with the
    // missing cell named, and the refusal carries the deployable joint plan, so the operator
    // is never left without a runnable spec. Nothing is substituted silently.
    // F903: the same guard, the same reason, the other ceiling. --kv-bits/--kv-k-bits/
    // --kv-v-bits composes with a --kv-layer-storage PIN SET exactly as --kv-bit-budget does.
    if (options.kv_kv_bits_explicit && (!storage_explicit || pin_and_solve)) {
        if (storage_explicit && !pin_and_solve && pin_count != 0) {
            throw std::invalid_argument(
                "--kv-bits/--kv-k-bits/--kv-v-bits with a --kv-layer-storage table that names "
                "EVERY full-attention layer: the ceiling would resolve into a table that "
                "already exists, i.e. be accepted and read by nothing. Name FEWER layers with "
                "--kv-layer-storage, or drop the ceiling.");
        }
        const std::int32_t rk4v4_limit = product::kKvBitBudgetE8LayerLimit;
        const std::int32_t cold_cap = static_cast<std::int32_t>(cold_pages);
        // The score tables. The joint entry fits against ONE table; the split entry may be
        // given one table PER PLANE, which is the only place the two planes' objectives can
        // legitimately differ (the tier ladder itself is symmetric).
        const auto load_scores = [](const std::string& spec, const char* flag) {
            if (spec.empty()) { return product::kv_bit_budget_default_scores(); }
            if (spec.find('\n') != std::string::npos) {
                return product::kv_bit_budget_parse_scores(spec);
            }
            std::ifstream in(spec);
            if (!in) {
                throw std::invalid_argument(std::string(flag) + ": cannot open " + spec);
            }
            return product::kv_bit_budget_parse_scores(
                std::string(std::istreambuf_iterator<char>(in),
                            std::istreambuf_iterator<char>()));
        };
        const product::KvTierScoreTable scores =
            load_scores(options.kv_tier_scores, "--kv-tier-scores");
        const product::KvTierScoreTable k_scores =
            load_scores(options.kv_k_tier_scores, "--kv-k-tier-scores");
        const product::KvTierScoreTable v_scores =
            load_scores(options.kv_v_tier_scores, "--kv-v-tier-scores");
        // Gap D, at the second usage point: the real provenance of the two columns.
        // kv4 P4: one provenance line PER NAMED TABLE. This block used to print only
        // `primary` (kv_tier_scores ?: k_tier_scores ?: v_tier_scores), so in the split
        // reading -- the only reading in which a per-plane table is read at all -- naming
        // BOTH --kv-k-tier-scores and --kv-v-tier-scores printed ONE provenance line and
        // silently omitted the other, with nothing saying which table it described. An
        // unnamed table gets no line: there is no table to describe.
        {
            const auto print_provenance = [](const char* flag, const std::string& spec) {
                if (spec.empty()) { return; }
                std::fprintf(stderr, "[kv-bit-score] table %s: ", flag);
                std::fputs(product::kv_scores_provenance(spec).line.c_str(), stderr);
                std::fputc('\n', stderr);
            };
            print_provenance("--kv-tier-scores", options.kv_tier_scores);
            print_provenance("--kv-k-tier-scores", options.kv_k_tier_scores);
            print_provenance("--kv-v-tier-scores", options.kv_v_tier_scores);
        }
        product::KvBitsRequest request;
        request.k_bits         = options.kv_k_bits;
        request.v_bits         = options.kv_v_bits;
        request.joint_bits     = options.kv_joint_bits;
        request.quality_weight = options.kv_quality_weight;
        request.mode           = options.kv_bits_mode;
        request.mode_explicit  = options.kv_bits_mode_explicit;
        request.joint_explicit = options.kv_joint_bits > 0.0;
        // "Naming a table" cannot be recovered from the value (the loader substitutes the
        // built-in default for an unset spec, and an operator file may equal it), so pass
        // the fact in: a named table that is read by nothing is the failure this whole
        // entry exists to avoid.
        request.scores_explicit   = !options.kv_tier_scores.empty();
        request.k_scores_explicit = !options.kv_k_tier_scores.empty();
        request.v_scores_explicit = !options.kv_v_tier_scores.empty();
        // SLIDERWIRE: --kv-codec-preference travels INSIDE the request, so the ONE decision
        // point (kv_kv_bits_resolve) is where it is validated against the candidate grammar
        // and where a preference the fit could not honour is refused by name instead of
        // being dropped in silence. Empty is the shipped order and byte-identical.
        request.candidate_order   = options.kv_codec_preference;
        // CLI > environment > default, the same contract as every other KV knob.
        const product::KvBitsRequest resolved_request =
            product::kv_bits_request_from_env(request);
        const product::KvBitsPlan kv_plan = product::kv_kv_bits_resolve(
            resolved_request, full_layers, scores, k_scores, v_scores, rk4v4_limit, cold_cap,
            options.kv_v_codec);
        std::fputs(kv_plan.report.c_str(), stderr);
        if (kv_plan.refused) {
            // Loud, per layer, with the missing (K,V) cell and the deployable plan named.
            throw std::invalid_argument(kv_plan.refusal);
        }
        const KvDeployedSpec deployed = deploy_kv_budget_spec(kv_plan.spec);
        // F903: the SAME pin application as the --kv-bit-budget entry, through the SAME
        // lambda, so the two ceiling spellings cannot drift on what a pin means.
        const std::size_t pinned_applied = apply_kv_pins();
        const std::string composed_hot = kv_pins_composed_spec(deployed.hot_spec);
        const bool composed_ok = kv_pins_spec_roundtrips(composed_hot);
        if (pinned_applied != 0) {
            std::fprintf(stderr,
                         "[kv-pin] --kv-layer-storage pinned=%zu of %d full-attention layer(s) runs=%s; those slots OVERRIDE this ceiling's plan and the remaining %d are the ceiling's own. `hot=` below IS the deployed table (pins merged in) and roundtrip=%s.\n",
                         pinned_applied, static_cast<int>(full_layers),
                         kv_pin_runs_text().c_str(),
                         static_cast<int>(full_layers) - static_cast<int>(pinned_applied),
                         composed_ok ? "true" : "false");
        }
        std::fprintf(stderr,
                     "[kv-bits] deployed mode=%s full_attention_layers=%d cold_pages=%u "
                     "cold_placed=%s hot=%s (cold residency needs --cold-policy window|disk; "
                     "pool sized by --max-cold-pages >= %u)\n",
                     product::kv_bits_mode_name(kv_plan.mode), static_cast<int>(full_layers),
                     static_cast<unsigned>(cold_pages),
                     deployed.cold_ranges.empty() ? "-" : deployed.cold_ranges.c_str(),
                     composed_hot.c_str(), static_cast<unsigned>(cold_pages));
    }

    // Both fp8 spellings name the SAME target tier: KvCacheStorage carries an old
    // Fp8E4M3Row256 (the standalone ops/kv_cache row-scaled codec's name, kept for
    // --kv-dtype / request-log naming) and Fp8Group16 (what product::parse_kv_storage
    // emits for the per-layer spec), and target_kv_cache_profile() maps both to
    // (DType::FP8_E4M3FN, kKvFp8QuantGroup). Only Fp8Group16 used to be handled here, so
    // a table written with the other spelling resolved to DType::BF16 - a silent downgrade
    // that drops the fp8 request instead of building it.
    //
    // The chain is now one call to product::kv_dtype_for_storage(), whose LAST arm is a
    // throw, not `: DType::BF16`. The old final arm was worse than a downgrade: DType::BF16
    // is also the "inherit the global --kv-dtype" sentinel (decoder_state.cpp layer_dtype()),
    // so an unmapped storage value silently expressed "ignore the operator's dtype for this
    // layer" AND plan_cache()'s own validator could not fire, because the value was laundered
    // one function earlier. Every storage code now gets a codec or an invalid_argument.
    std::array<DType, 64> layer_overrides{};
    // The mask that belongs to layer_overrides, resolved in the SAME branch so the
    // two can never describe different tables.
    std::array<bool, 64> layer_overrides_set{};
    const bool has_override = storage_explicit;
    if (has_override) {
        for (std::size_t i = 0; i < layer_overrides.size(); ++i) {
            layer_overrides[i] = product::kv_dtype_for_storage(
                budget_storage_table[i], "--kv-layer-storage[" + std::to_string(i) + "]");
            layer_overrides_set[i] = budget_storage_set[i];
        }
    } else if (options.kv_cache_explicit) {
        // An explicit global --kv-dtype replaces the target's registered per-layer default table
        // for every layer (the table is only consulted when the user pinned layers). Without
        // this the global dtype never reached the KV page geometry (_TODO.md 97).
        // The mask stays all-false on purpose: this is the GLOBAL dtype, not a per-layer
        // write. kv_profile.dtype IS plan.kv_dtype here, so resolving an unset BF16 slot
        // against it is the identity -- a mask would change nothing and only claim the
        // operator named 64 layers they never named.
        layer_overrides.fill(kv_profile.dtype);
    } else if constexpr (Variant::supports_per_layer_kv_defaults) {
        // The target's REGISTERED default table is not a per-layer write either: its
        // BFloat16 entries mean "inherit the global dtype", which is exactly what the
        // all-false mask says. Marking them written would change the meaning of every
        // existing default table.
        layer_overrides = Variant::default_layer_kv_dtypes(
            weights_profile);
    }
    if (has_override) {
        // Item 5 (l26c): the per-layer table is family-wide, but only the slots
        // below THIS target's full-attention count reach the pool geometry. A slot
        // past that count parses, is accepted, and changes nothing on this model --
        // which used to be completely silent ("--kv-layer-storage 20:rk4v4" on 27b).
        // Report the inert slots; the engine report still prints 16 layers.
        const std::size_t active_layers =
            static_cast<std::size_t>(TextConfig::full_attention_layers());
        std::string inert_slots;
        for (std::size_t slot = active_layers; slot < budget_storage_table.size(); ++slot) {
            if (budget_storage_table[slot] == KvCacheStorage::BFloat16) { continue; }
            if (!inert_slots.empty()) { inert_slots += ','; }
            inert_slots += std::to_string(slot);
        }
        if (!inert_slots.empty()) {
            std::fprintf(stderr,
                         "[kv] --kv-layer-storage names slot(s) %s at or past this target's %zu "
                         "full-attention layers: accepted, but they have NO effect on the pool "
                         "this model builds (the table covers the whole family, %zu slots).\n",
                         inert_slots.c_str(), active_layers, budget_storage_table.size());
        }
    }
    // Stage 2: mode and cold, judged on the EFFECTIVE dtype of every layer. A BF16
    // override slot inherits the global dtype UNLESS the spec wrote it, exactly as
    // PagedKVCache::plan_cache resolves it (decoder_state.cpp layer_dtype() ->
    // kv_resolve_slot_dtype), so resolving it here is what makes the check describe the
    // built plan and not just the option text. With the mask omitted,
    // `--kv-layer-storage 0-11:bf16` under `--kv-dtype nvfp4` was judged as an
    // all-NVFP4 stack -- the gate would have been checking a plan nobody was going to
    // build. The per-layer class carries the two facts the gates need: whether the
    // layer sits on a fusion tier (mode=pure) and which cold-slot codec it can feed
    // (cold=), mirroring the two branches of enqueue_cold_compressions (program_impl.h)
    // and the all-INT8 gate above them.
    //
    // The classification and the arming gate sit OUTSIDE the `--kv-tier-formats`
    // branch on purpose. How reachable the cold pool is is a property of the table this
    // plan really builds (the target's own default mix, --kv-layer-storage, --kv-dtype),
    // NOT of the format vocabulary that happens to be in play; asking the question only
    // when that flag was given left the DEFAULT path never asking it at all -- pool
    // armed, predicate admitting its pages, zero bytes, no line printed.
    const std::int32_t tier_layers =
        static_cast<std::int32_t>(TextConfig::full_attention_layers());
    std::array<product::KvLayerClass, kKvLayerStorageSlots> tier_classes{};
    for (std::size_t i = 0; i < tier_classes.size(); ++i) {
        const DType selected = product::kv_resolve_slot_dtype(
            kv_profile.dtype, layer_overrides[i], layer_overrides_set[i]);
        tier_classes[i] = product::kv_layer_class_of(selected);
    }
    const bool tier_cold_pool =
        effective_cold_pages(options.cold_policy, options.cold_keep_tokens,
                             options.max_cold_pages) > 0;
    // THE ARMING GATE. A requested pool whose resolved stack feeds no slot codec is NOT
    // built: it would reserve --max-cold-pages of device memory, be handed every
    // retirable page by the predicate, and transfer zero bytes -- and the compressor's
    // only report of it is per PAGE ("REFUSED at layer ...", program_impl.h), which the
    // operator has to read as "the whole stack is unpackable". Every offender is named
    // here instead, once, before anything is allocated.
    //
    // CLOSED, not a refusal to run: the residency knob is a DIFFERENT axis from the
    // format one, and the established contract for a stack with no codec is "one such
    // layer anywhere in the stack leaves the pool idle" (kv_tier_formats.h) -- the
    // engine does not refuse to start. So the run proceeds, every page stays hot, and
    // the difference from the silent case is that the reason is on stderr before the
    // first token. What is NOT done here: the offender is never SKIPPED per page (the
    // physical page would be handed back and its next read would land on another
    // page's bytes), and the default layer table is not touched (its rk4v4 layers buy the
    // measured K lattice gain -- src/targets/qwen3_6_27b/impl/variant.cpp).
    const product::KvColdPool tier_cold_state =
        product::kv_cold_pool_state(tier_classes, tier_layers, tier_cold_pool);
    if (tier_cold_state == product::KvColdPool::Closed) {
        cold_tier_closed = true;
        std::fputs(product::kv_cold_pool_close_notice(tier_classes, tier_layers).c_str(),
                   stderr);
    } else if (!tier_plan.has_value() &&
               tier_cold_state != product::KvColdPool::NotRequested) {
        // The residency axis reports on its own account: `cold_residency=` used to be
        // reachable only through kv_tier_formats_check(), which this file calls only
        // when --kv-tier-formats was given, so a --cold-policy/--max-cold-pages sweep
        // printed it 0 times in 18 cells.
        std::fprintf(stderr, "[kv-cold-tier] %s\n",
                     product::kv_cold_residency_text(tier_classes, tier_layers,
                                                     tier_cold_state)
                         .c_str());
    }
    if (tier_plan.has_value()) {
        const std::string tier_report = product::kv_tier_formats_check(
            *tier_plan, tier_classes, tier_layers, tier_cold_pool);
        std::fprintf(stderr, "%s\n", tier_report.c_str());
    }
    // MTP adaptive draft window: `--spec mtp --draft-tokens 0`, and the backend `auto` (which
    // the target packages resolve to MTP with draft_tokens left at 0). The planner captures one
    // decode graph per ladder rung and the criterion picks the rung per round.
    // A tree is a FIXED shape: it must not also engage the adaptive capture ladder, because the
    // criterion would then pick rungs whose capture width no longer matches the node budget.
    const bool mtp_adaptive = options.speculative.backend == SpeculativeBackend::Mtp &&
                              options.speculative.draft_tokens == 0 &&
                              options.speculative.draft_tree_paths == 0;
    // mtpadapt (dl/_orch/landq/mtpadapt/01-laddertop): THE WIDEST RUNG AN ADAPTIVE CAPTURE
    // LADDER MAY CARRY. Not this artifact's draft maximum -- the rung the MEASURED width model
    // selects. See the effective_mtp_ladder note below for the measurement behind the value.
    //
    // 2 is the throughput argmax of Phi(k) = AL(k)*1000/(a + b*k) with the cost constants the
    // criterion itself is calibrated on (mtp_window_cut.h: kMtpRoundCostBaseMs /
    // kMtpRoundCostPerColumnMs, measured for this artifact class), and it is also where this
    // record's own fixed-width sweep peaked (111.5 tok/s at k=2 against 107.8 at 3, 108-110 at 4,
    // 59 at 9-10, 44.5 at 15).
    constexpr std::uint32_t kMtpAdaptiveLadderTopRung = 2;
    // Clamped to this target's MTP draft domain (kMaximumMtpDraftTokens is a per-target
    // constant; 27b raised it to kMtpDecodeMaximumDrafts). A target too narrow for the first
    // rung degrades to a single rung at its own maximum, i.e. the widest legal fixed width.
    const auto effective_mtp_ladder = [&] {
        std::vector<std::uint32_t> ladder;
        if (!mtp_adaptive) { return ladder; }
        for (const std::uint32_t width : kMtpWindowLadder) {
            if (width <= kMaximumMtpDraftTokens) { ladder.push_back(width); }
        }
        if (ladder.empty()) {
            ladder.push_back(kMaximumMtpDraftTokens != 0 ? kMaximumMtpDraftTokens : 1U);
        }
        // mtpadapt / 01-laddertop: SIZE THE CAPTURE LADDER BY THE COST MODEL, NOT BY THE ARTIFACT.
        //
        // WHY. `draft_window` for an adaptive run is `effective_mtp_ladder.back()`, resolved a few
        // lines below, and the MTP decode FRAME is `draft_window + 1` columns wide. The round's
        // target-verify work follows the FRAME, not the rung the criterion picks. Measured on this
        // artifact (SC_004096, ctx 20480, --max-new 64, greedy, --kv-dtype nvfp4, bin
        // 42b7855bae90b35d, --spec mtp, NINFER_MTP_ROUND_LOG=1): the SAME implemented width of 2
        // columns costs 35.3 ms/round of device wait plus 5-10 ms of host when the frame is 16
        // wide, against 15.9 + 1.4 ms/round when the frame is 3 wide, and a 4-point fit over the
        // fixed and adaptive arms gives
        //     round(ms) = 10.963 + 1.868*(draft_window + 1) + 1.45*(launch_width - 1)
        // with a maximum residual of 0.9 ms. A ladder top of 15 therefore charges ~24 ms/round
        // for five rungs -- and pays the widest one on every round -- to express a decision that
        // lands on the narrowest one.
        //
        // WHAT THIS KEEPS AND WHAT IT REMOVES. Every rung up to kMtpAdaptiveLadderTopRung stays,
        // so the criterion is still evaluated per round and the width is still SELECTED: mtp_ladder
        // is still non-empty, the profile width key is still live, and `adaptive_window` still
        // reports true. Only the rungs the cost model never selects are dropped. A single-rung
        // ladder additionally takes the |ladder| > 1 deferral off the capture path (program_impl.h
        // gates it on `graph_capture_ceiling != 0 && rungs.size() > 1`), so the rung is captured up
        // front exactly as a fixed-k run captures its own single rung, and no `[graphs] extended
        // MTP ladder rung` print and no per-crossing cudaStreamSynchronize can occur at all.
        //
        // REVERSIBILITY. Removing this block restores the artifact-maximum ladder byte for byte
        // (see revert.py, which restores the whole pre-image).
        while (ladder.size() > 1 && ladder.back() > kMtpAdaptiveLadderTopRung) {
            ladder.pop_back();
        }
        // ladderrung (dl/_orch/landq/ladderrung/01-ladderfloor): THE TOP IS ALSO THE FLOOR THE
        // ACTIVE PROPOSAL HEAD DEMANDS.
        //
        // WHY. When `draft_tokens == 0` the package's own rewrite feeds kMtpShortlistMinimumDrafts
        // (5) to resolved_proposal_head (qwen3_6_27b/impl/package.cpp), so an adaptive run on an
        // artifact that DECLARES the shortlist head resolves `proposal_head = Optimized` --
        // measured on the shipping pin:
        //   "proposal head auto-resolved head=optimized reason=mtp-at-or-above-minimum (profile=3,
        //    backend=1, window=0, declared shortlist head=present)"
        // program_impl.h:11258-11260 then sets head_floor = kMtpShortlistMinimumDrafts and floors
        // EVERY chosen rung at it, and its single write point (:11277-11280) refuses a rung that is
        // outside the captured ladder -- the ladder THIS lambda returns.
        //
        // So the clamp above can strip the ladder below the floor the same run will demand: a top
        // of 2 leaves {2} while the floor asks for 5, and the run stops mid-decode with
        //     error: MTP target width is not a rung of the captured ladder
        // Measured (dl/ladderrung/REPORT.md): bin 4771a95999ce90df, `--spec mtp` with NO
        // --draft-tokens, rc=1 in 3/3 runs -- and 4/4 with the recall axis stripped -- against rc=0
        // for the frozen pre-image 42b7855bae90b35d on the same argv. The narrowed ladder is what
        // makes widths below the floor POSSIBLE, so the two decisions are not independent: a ladder
        // whose top is under the floor has to carry the floor's own rung.
        //
        // WHAT THIS DOES NOT DO. It does not move the floor -- the Optimized head's drafts leak into
        // the emitted tokens below kMtpShortlistMinimumDrafts (startup_features.h:74, 88/160
        // positions at k=3), so lowering it would change tokens, not only speed -- and it does not
        // change the head. It only keeps the rung the floor will demand, taken from the same
        // decision ladder and the same target domain the clamp above already reads.
        //
        // THE PRICE, NAMED. A floor-bearing ladder ends at 5, so `draft_window` and the decode
        // frame (draft_window + 1 = 6) are that much wider than the k=2 ladder's 3: the round cost
        // folows the frame, and dl/_orch/landq/mtpadapt/01-laddertop priced 1.868 ms per frame
        // column. Rungs above the floor are still dropped, so the clamp's own saving survives on
        // every rung it removed.
        if (!ladder.empty() && options.speculative.proposal_head == ProposalHead::Optimized) {
            const std::uint32_t floor_rung = qwen3_6::kMtpShortlistMinimumDrafts;
            const std::uint32_t clamped_top = ladder.back();
            if (clamped_top < floor_rung) {
                for (const std::uint32_t width : kMtpWindowLadder) {
                    if (width > clamped_top && width <= floor_rung &&
                        width <= kMaximumMtpDraftTokens) {
                        ladder.push_back(width);
                    }
                }
            }
        }
        return ladder;
    }();
    // --draft-tree L,d: the round's scalar draft extent becomes the tree's NODE BUDGET. That is
    // the one number a tree changes here: the frame is still one column per draft plus the
    // anchor, and program_impl.h:13196-13198 clamps the live extent to that frame, so the frame
    // must already BE the node budget (L*d columns are live in a tree round).
    const std::uint32_t mtp_tree_nodes =
        options.speculative.draft_tree_paths * options.speculative.draft_tree_depth;
    const std::uint32_t resolved_mtp_window =
        mtp_tree_nodes != 0 ? mtp_tree_nodes
                            : (mtp_adaptive ? effective_mtp_ladder.back()
                                            : options.speculative.draft_tokens);
    SequencePlanningInputs inputs{
        .weights_profile     = weights_profile,
        .capacity            = options.max_context,
        .kv_capacity_tokens  = options.kv_capacity.mode == KvCapacityMode::Explicit
                                   ? std::optional<std::uint32_t>(options.kv_capacity.explicit_tokens)
                                   : std::nullopt,
        .max_concurrency     = options.max_concurrency,
        .prefill_chunk       = std::min(options.prefill_chunk, options.max_context),
        .draft_window = options.speculative.backend == SpeculativeBackend::DFlash2 &&
 options.speculative.draft_tokens == 0
 ? 7U
 : resolved_mtp_window,
        // Non-empty = adaptive: capture one MTP graph per rung and let the criterion pick.
        // (DFlash/DFlash2 keep their own startup-fixed block width; making those adaptive is a
        // separate mechanism and is deliberately not done here.)
        .mtp_ladder = effective_mtp_ladder,
        .draft_tree_paths = options.speculative.draft_tree_paths,
        .draft_tree_depth = options.speculative.draft_tree_depth,
        .speculative_backend = options.speculative.backend,
        .kv_dtype            = kv_profile.dtype,
        .kv_quant_group      = kv_profile.quant_group,
        .layer_kv_dtypes     = layer_overrides,
        .kv_residual_layers  = options.kv_residual_explicit
                                   ? options.kv_residual_layers
                                   : std::array<bool, 64>{},
        .kv_v_codec          = options.kv_v_codec,
        .kv_rotation_off     = options.kv_rotation_explicit && options.kv_rotation_off,
        .kv_row_scale_spec   = options.kv_row_scale_explicit ? options.kv_row_scale_spec
                                                             : std::string{},
        .stage_layers_spec   = options.stage_layers_spec,
        .stage_handoff_dir   = options.stage_handoff_dir,
        .stage_handoff_cut   = options.stage_handoff_cut,
        .proposal_head       = options.speculative.proposal_head,
        .features            = qwen3_6::startup_features(options),
        .use_cuda_graph      = options.use_cuda_graph,
        .graph_capture_ceiling = options.graph_capture_ceiling,
        .causal_scoring      = options.purpose == EnginePurpose::CausalScoring,
        .cold_policy         = options.cold_policy,
        // Designators must follow declaration order (layouts.h): max_cold_pages
        // is declared before cold_keep_tokens.
        .max_cold_pages      = options.max_cold_pages,
        .cold_keep_tokens    = options.cold_keep_tokens,
        .unload_watermark_pages = options.unload_watermark_pages,
        .cold_host_bytes     = options.cold_host_bytes,
        .cold_disk_path      = options.cold_disk_path,
        .cold_disk_bytes     = options.cold_disk_bytes,
        .device              = options.device,
        .context_cache       = options.context_cache,
        // Declared last in SequencePlanningInputs, so the designator comes last.
        .layer_kv_dtypes_set = layer_overrides_set,
    };
    // The arming gate closed the cold tier: the two designators the pool really reads
    // (impl->cold_policy, which is what gates enqueue_cold_compressions at all, and
    // spec.max_cold_pages, which is what sizes the slot tensors) are cleared here, so
    // nothing downstream reserves a slot, sizes a staging buffer or takes a spill-file
    // slot for a pool that could never hold a page. ColdPolicy::None is exactly the
    // state of a run that never passed --cold-policy, which is the state this stack can
    // honour.
    if (cold_tier_closed) {
        inputs.cold_policy    = ColdPolicy::None;
        inputs.max_cold_pages = 0;
    }
    // and the one honest caveat this gate cannot state for itself: the bit-budget
    // entries above derive their cold capacity from --cold-policy/--max-cold-pages
    // BEFORE the effective per-layer table is known, so their `cold_pages=` reading was
    // taken on the REQUEST and not on the STACK. Named here rather than left to be
    // discovered.
    if (cold_tier_closed && (options.kv_bit_budget_explicit || options.kv_kv_bits_explicit)) {
        std::fprintf(stderr,
                     "[kv-cold-tier] NOTE: the bit-budget entry above sized its cold "
                     "capacity from the request (cold_pages=%u), because it runs before the "
                     "effective per-layer table exists; with the cold tier closed those "
                     "layers stay hot.\n",
                     static_cast<unsigned>(cold_pages));
    }
    const std::uint32_t logical_pages = page_count(inputs.capacity);
    // The device page pool normally covers the full max_context; an explicit
    // --kv-capacity below max_context instead floors the pool at that size so
    // the rope domain (4x under YaRN) can exceed what the pool can hold.
    std::uint32_t minimum_pages = std::max(logical_pages, inputs.max_concurrency);
    if (inputs.kv_capacity_tokens) {
        minimum_pages = std::max(page_count(*inputs.kv_capacity_tokens), inputs.max_concurrency);
    }

    // -------------------------------------------------------------------------------
    // RECALLSPAN: WIRE PAGEPREALLOC'S REFUSAL. The contract landed by PAGEPREALLOC
    // (product/kv_paging_preallocation.h) exists to stop "a page with neither replica
    // nor tier being silently counted as stored", and until this block it had ZERO
    // non-test consumers, so it could not fire on anything. It is wired HERE because
    // this is where a pool is named, and an explicit pool is the guard's own domain.
    //
    // OPT-IN: unset (or "0") leaves the two lines above as the whole answer, so a run
    // that does not ask for the check is bit-identical to before.
    //
    // THE ONE INPUT THIS SITE CANNOT SUPPLY, NAMED RATHER THAN GUESSED.
    // `cold_page_record_bytes` is the BYTES PER COLD PAGE ACROSS ALL SLOT-BEARING
    // LAYERS -- the sum over layers of (stride x kv_heads x 2) = turn_recall_page_bytes()
    // (program_impl.h:13335 ProgramImplCore::turn_recall_page_bytes), which is what the
    // engine prints as `bytes_per_record` on
    // its [textcargo] line (1,232,896 B when all 16 layers are nvfp4 rANS slots
    // = 128 x 9,632; 1,213,696 B on the factory 6 x rk4v4 + 10 x nvfp4 table, whose
    // rk4v4 layers take the 9,232 B raw record; 1,181,696 B when all 16 are raw.
    // The record is PER LAYER (decoder_state.cpp cold_slot_stride_for), so this
    // page total is a function of the stack's dtype table and is not a constant:
    // the retired 1,220,608 was 128 x 9,536, the 4-bit rANS record at
    // kv_tier_formats.h:523. dl/coldagree/REPORT.md TASK C.1.)
    // It is NOT reachable here: the cold-slot tensors whose nb[3] carries it are declared
    // later, by the engine (decoder_state.cpp:931-933), not by this layout. ONE LAYER'S
    // STRIDE (9,536 B) IS NOT A PAGE -- passing it would under-count by 128x, which is
    // the exact unit defect kv_paging_preallocation.h:173-179 warns the next reader
    // about. So it is passed as 0 = "not known", never as a guess.
    //
    // CONSEQUENCE, STATED: with the per-page record unknown the Disk term of
    // paging_backing_capacity_pages() reads 0 pages, i.e. the capacity is a LOWER BOUND
    // and the guard can only refuse MORE than the truth. To keep that from becoming a
    // false refusal, the guard is consulted only when its verdict cannot depend on the
    // unknown: under DeviceWindow/Disk/HostThenDisk the device cold-slot pool is the
    // backing store's hard ceiling, so when that ceiling alone is already below the
    // requirement the refusal holds for EVERY possible byte-per-page. When it is not,
    // the check is SKIPPED and says so -- it never refuses on a stand-in.
    if (const char* paging_gate = std::getenv("NINFER_KV_PAGING_PREALLOC");
        paging_gate != nullptr && paging_gate[0] != '\0' && std::strcmp(paging_gate, "0") != 0 &&
        inputs.kv_capacity_tokens) {
        ninfer::product::PagingPreallocationInputs paging;
        paging.max_context_tokens     = inputs.capacity;
        paging.page_tokens            = static_cast<std::uint32_t>(kPagedKVPageSize);
        paging.max_concurrency        = inputs.max_concurrency;
        paging.cold_keep_tokens       = inputs.cold_keep_tokens;
        paging.prefill_chunk          = inputs.prefill_chunk;
        paging.unload_watermark_pages = inputs.unload_watermark_pages;
        switch (inputs.cold_policy) {
        case ColdPolicy::None:         paging.medium = ninfer::product::PagingColdMedium::None;         break;
        case ColdPolicy::Window:       paging.medium = ninfer::product::PagingColdMedium::DeviceWindow; break;
        case ColdPolicy::Host:         paging.medium = ninfer::product::PagingColdMedium::PinnedHost;   break;
        case ColdPolicy::Disk:         paging.medium = ninfer::product::PagingColdMedium::Disk;         break;
        case ColdPolicy::HostThenDisk: paging.medium = ninfer::product::PagingColdMedium::HostThenDisk; break;
        }
        paging.max_cold_pages         = inputs.max_cold_pages;
        paging.cold_disk_bytes        = inputs.cold_disk_bytes;
        paging.cold_host_bytes        = inputs.cold_host_bytes;
        paging.cold_page_record_bytes = 0; // see above: unknown here, never guessed
        paging.host_page_bytes        = 0; // HostKVPageLayout::page_stride, likewise later
        const std::uint32_t paging_requested =
            std::max(page_count(*inputs.kv_capacity_tokens), inputs.max_concurrency);
        const std::uint32_t paging_required =
            ninfer::product::paging_required_backing_pages(paging);
        const std::uint32_t paging_ceiling = ninfer::product::paging_device_cold_slot_pages(
            paging.cold_keep_tokens, paging.page_tokens, paging.medium, paging.max_cold_pages);
        const bool paging_decidable =
            paging.medium == ninfer::product::PagingColdMedium::None ||
            (paging.medium != ninfer::product::PagingColdMedium::PinnedHost &&
             paging_ceiling < paging_required);
        if (paging_decidable) {
            const std::string paging_refusal =
                ninfer::product::paging_explicit_capacity_refusal(paging, paging_requested);
            if (!paging_refusal.empty()) { throw std::invalid_argument(paging_refusal); }
        } else {
            std::fprintf(stderr,
                         "[paging-prealloc] NOT EVALUATED: the backing store's page ceiling (%u) "
                         "is not below the requirement (%u), so the verdict would depend on the "
                         "byte-per-cold-page, which this site cannot supply; the pool stays at %u "
                         "page(s) exactly as before\n",
                         static_cast<unsigned>(paging_ceiling),
                         static_cast<unsigned>(paging_required),
                         static_cast<unsigned>(minimum_pages));
        }
    }
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(inputs.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    const auto maximum_pages = static_cast<std::uint32_t>(maximum_pages64);

    auto planner     = std::make_unique<qwen3_6::detail::SequencePlannerImpl<Variant>>();
    planner->inputs  = inputs;
    planner->minimum = build_sequence_candidate(inputs, minimum_pages);
    planner->curve   = runtime::SequenceCapacityCurve{
          .main_page_tokens                     = static_cast<std::uint32_t>(kPagedKVPageSize),
          .minimum_main_page_groups             = minimum_pages,
          .maximum_main_page_groups             = maximum_pages,
          .minimum_device_reservation_bytes     = planner->minimum->device_reservation_bytes,
          .bytes_per_additional_main_page_group = 0,
    };
    if (minimum_pages < maximum_pages) {
        auto adjacent = build_sequence_candidate(inputs, minimum_pages + 1U);
        if (adjacent->device_reservation_bytes <= planner->minimum->device_reservation_bytes) {
            throw std::logic_error("Qwen3.6 sequence layout has a nonpositive KV capacity stride");
        }
        planner->curve.bytes_per_additional_main_page_group =
            adjacent->device_reservation_bytes - planner->minimum->device_reservation_bytes;
    }
    return planner;
}

std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>> planner,
                            std::uint32_t main_page_groups) {
    if (planner == nullptr || planner->minimum == nullptr) {
        throw std::invalid_argument("Qwen3.6 sequence planner is empty");
    }
    const std::size_t expected = planner->curve.reservation_bytes(main_page_groups);
    std::unique_ptr<SequencePlanImpl> plan;
    if (main_page_groups == planner->curve.minimum_main_page_groups) {
        plan = std::move(planner->minimum);
    } else {
        plan = build_sequence_candidate(planner->inputs, main_page_groups);
    }
    if (plan->device_reservation_bytes != expected) {
        throw std::logic_error(
            "Qwen3.6 physical sequence layout is not affine in Main KV page capacity");
    }
    return plan;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

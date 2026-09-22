#include "runtime/engine/context_cost.h"

#include "targets/declared_capabilities.h"

#include <array>
#include <string>
#include <vector>

namespace ninfer::runtime {

const std::array<ContextTransferCost, 3>& generic_context_transfer_cost() {
    // Conservative but useful on an unmeasured machine. These values preserve numerical ranking;
    // they are not a switch that disables the cost model.
    static constexpr std::array<ContextTransferCost, 3> value{
        ContextTransferCost{
            .batch_ns = 100'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 50'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 25'000, .operation_ns = 12'000, .ns_per_byte_q32 = 42'949'673},
    };
    return value;
}

const ContextPrefillCost& generic_context_prefill_cost() {
    // The slower measured 27B profile is the conservative generic estimate. An unknown model still
    // receives continuous recomputation costs instead of falling back to a discrete tuple.
    static constexpr ContextPrefillCost value{
        .chunk_ns              = 40'813'570,
        .token_ns_q32          = 1'012'273'154'411'951,
        .attention_pair_ns_q32 = 30'497'396'515,
        .vision_item_ns        = 5'986'585,
        .vision_patch_ns_q32   = 23'710'212'854'694,
    };
    return value;
}

// Accepted project defaults live only in this table and are compiled into the binary. Runtime JSON
// presets are independent local-machine overrides and never become a build dependency.
const std::vector<ContextCostMachinePreset>& compiled_context_cost_defaults() {
    // One measured prefill value, two declared artifact identities.
    constexpr ContextPrefillCost kQwen3_8Nvfp4Prefill{
        .chunk_ns              = 14'672'989,
        .token_ns_q32          = 375'800'765'711'778,
        .attention_pair_ns_q32 = 8'200'474'657,
        .vision_item_ns        = 5'860'255,
        .vision_patch_ns_q32   = 23'832'529'381'413,
    };
    static const std::vector<ContextCostMachinePreset> defaults{
        ContextCostMachinePreset{
            .hardware_class = "nvidia-geforce-rtx-5090-sm120",
            // Unified from the measured 2026-08-24 transfer corpus. A fresh machine-only transfer
            // calibration can replace this whole array without loading a model.
            .transfer =
                std::array{
                    ContextTransferCost{
                        .batch_ns = 48'655, .operation_ns = 7'433, .ns_per_byte_q32 = 91'692'315},
                    ContextTransferCost{
                        .batch_ns = 0, .operation_ns = 8'457, .ns_per_byte_q32 = 83'354'284},
                    ContextTransferCost{
                        .batch_ns = 3'343, .operation_ns = 9'520, .ns_per_byte_q32 = 2'658'314},
                },
            .prefill =
                {
                    ContextPrefillPreset{
                        .model_id   = "qwen3.6-27b",
                        .weights_id = "groupwise-int",
                        .cost =
                            {
                                .chunk_ns              = 40'813'570,
                                .token_ns_q32          = 1'012'273'154'411'951,
                                .attention_pair_ns_q32 = 30'497'396'515,
                                .vision_item_ns        = 5'986'585,
                                .vision_patch_ns_q32   = 23'710'212'854'694,
                            },
                    },
                    ContextPrefillPreset{
                        .model_id   = "qwen3.8-27b",
                        .weights_id = "nvfp4",
                        .cost       = kQwen3_8Nvfp4Prefill,
                    },
                    // The ModelOpt source of the same export declares a different weights flavour
                    // for the same tensors (docs/maintainer/modelopt-nvfp4-import.md section 3), so
                    // it is a second declared identity for one measured value: not a second copy of
                    // the number, and not a fallback to the generic estimate.
                    ContextPrefillPreset{
                        .model_id   = "qwen3.8-27b",
                        .weights_id = "nvfp4-modelopt",
                        .cost       = kQwen3_8Nvfp4Prefill,
                    },
                },
        },
    };
    return defaults;
}

std::string context_prefill_identities(std::string_view hardware_class) {
    std::string rendered;
    for (const ContextCostMachinePreset& machine : compiled_context_cost_defaults()) {
        if (machine.hardware_class != hardware_class) { continue; }
        for (const ContextPrefillPreset& preset : machine.prefill) {
            if (!rendered.empty()) { rendered += ", "; }
            rendered += preset.model_id;
            rendered += '/';
            rendered += preset.weights_id;
        }
    }
    return rendered.empty() ? std::string{"(none declared)"} : rendered;
}

std::string context_prefill_preset_miss_note(const ContextCostIdentity& identity) {
    if (identity.hardware_class.empty() || identity.model_id.empty() ||
        identity.weights_id.empty()) {
        return {};
    }
    // The same key `context_cost.cpp` matches presets on, so this note and `find_prefill` cannot
    // disagree about what "a preset for this identity" is.
    const targets::WeightsIdentityKey wanted{.model_id   = identity.model_id,
                                             .weights_id = identity.weights_id};
    const std::string declared = context_prefill_identities(identity.hardware_class);
    for (const ContextCostMachinePreset& machine : compiled_context_cost_defaults()) {
        if (machine.hardware_class != identity.hardware_class) { continue; }
        for (const ContextPrefillPreset& preset : machine.prefill) {
            const targets::WeightsIdentityKey stored{.model_id   = preset.model_id,
                                                     .weights_id = preset.weights_id};
            if (stored == wanted) { return {}; }
        }
    }
    return "context-cost prefill: no preset declared for '" + identity.model_id + "/" +
           identity.weights_id + "' on '" + identity.hardware_class +
           "'; the compiled defaults declare {" + declared +
           "}, so the generic prefill model is used";
}

} // namespace ninfer::runtime

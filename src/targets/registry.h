#pragma once

#include "ninfer/types.h"
#include "runtime/engine/context_cost.h"
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6_35b_a3b/package.h>
#include <ninfer/targets/muse_glimmer_30b/package.h>

#include <memory>
#include <variant>

namespace ninfer {

struct DeviceContext;

namespace targets {

using Qwen3_6_27B    = qwen3_6_27b::Package;
using Qwen3_6_35BA3B = qwen3_6_35b_a3b::Package;
using MuseGlimmer30B = muse_glimmer_30b::Package;

struct LoadedQwen3_6_27B {
    std::unique_ptr<Qwen3_6_27B::LoadedModel> model;
    Qwen3_6_27B::Frontend frontend;

    LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                      const EngineOptions& options);
    ~LoadedQwen3_6_27B();

    LoadedQwen3_6_27B(const LoadedQwen3_6_27B&)            = delete;
    LoadedQwen3_6_27B& operator=(const LoadedQwen3_6_27B&) = delete;
};

struct Qwen3_6_27BInstance {
    using Package = Qwen3_6_27B;

    std::unique_ptr<LoadedQwen3_6_27B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_27B::Program> program;
    Qwen3_6_27B::WeightsProfile weights_profile;

    Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                        runtime::KvCapacityResolution resolution,
                        Qwen3_6_27B::SequencePlan sequence_plan,
                        Qwen3_6_27B::WeightsProfile weights_profile_in, DeviceContext& device);
    ~Qwen3_6_27BInstance();

    Qwen3_6_27BInstance(const Qwen3_6_27BInstance&)            = delete;
    Qwen3_6_27BInstance& operator=(const Qwen3_6_27BInstance&) = delete;
};

struct LoadedQwen3_6_35BA3B {
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> model;
    Qwen3_6_35BA3B::Frontend frontend;

    LoadedQwen3_6_35BA3B(std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model,
                         const EngineOptions& options);
    ~LoadedQwen3_6_35BA3B();

    LoadedQwen3_6_35BA3B(const LoadedQwen3_6_35BA3B&)            = delete;
    LoadedQwen3_6_35BA3B& operator=(const LoadedQwen3_6_35BA3B&) = delete;
};

struct Qwen3_6_35BA3BInstance {
    using Package = Qwen3_6_35BA3B;

    std::unique_ptr<LoadedQwen3_6_35BA3B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_35BA3B::Program> program;
    Qwen3_6_35BA3B::WeightsProfile weights_profile;

    Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                           runtime::KvCapacityResolution resolution,
                           Qwen3_6_35BA3B::SequencePlan sequence_plan,
                           Qwen3_6_35BA3B::WeightsProfile weights_profile_in,
                           DeviceContext& device);
    ~Qwen3_6_35BA3BInstance();

    Qwen3_6_35BA3BInstance(const Qwen3_6_35BA3BInstance&)            = delete;
    Qwen3_6_35BA3BInstance& operator=(const Qwen3_6_35BA3BInstance&) = delete;
};


struct LoadedMuseGlimmer30B {
    std::unique_ptr<MuseGlimmer30B::LoadedModel> model;
    MuseGlimmer30B::Frontend frontend;

    LoadedMuseGlimmer30B(std::unique_ptr<MuseGlimmer30B::LoadedModel> stable_model,
                         const EngineOptions& options);
    ~LoadedMuseGlimmer30B();

    LoadedMuseGlimmer30B(const LoadedMuseGlimmer30B&)            = delete;
    LoadedMuseGlimmer30B& operator=(const LoadedMuseGlimmer30B&) = delete;
};

struct MuseGlimmer30BInstance {
    using Package = MuseGlimmer30B;

    std::unique_ptr<LoadedMuseGlimmer30B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<MuseGlimmer30B::Program> program;
    MuseGlimmer30B::WeightsProfile weights_profile;

    MuseGlimmer30BInstance(std::unique_ptr<LoadedMuseGlimmer30B> stable_loaded,
                           runtime::KvCapacityResolution resolution,
                           MuseGlimmer30B::SequencePlan sequence_plan,
                           MuseGlimmer30B::WeightsProfile weights_profile_in,
                           DeviceContext& device);
    ~MuseGlimmer30BInstance();

    MuseGlimmer30BInstance(const MuseGlimmer30BInstance&)            = delete;
    MuseGlimmer30BInstance& operator=(const MuseGlimmer30BInstance&) = delete;
};

using ActiveTarget = std::variant<std::unique_ptr<Qwen3_6_27BInstance>,
                                              std::unique_ptr<Qwen3_6_35BA3BInstance>,
                                              std::unique_ptr<MuseGlimmer30BInstance>>;

struct ConstructedTarget {
    ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    runtime::ContextMachineCostModel context_cost;
};

[[nodiscard]] ConstructedTarget construct_target(const EngineOptions& options,
                                                 DeviceContext& device);

// Re-runs the sequence plan (and only the sequence plan) for the loaded target with
// updated EngineOptions — kv_layer_storage / kv_residual_layers / kv_capacity. The old
// Program is destroyed first so the new plan's capacity resolution sees the freed KV
// budget; a failure past that point leaves the instance without a Program (retry or
// restart needed). Caller contract: no generation may be in flight and none may start
// until this returns (the serve layer drains before calling).
void replan_target_kv(ActiveTarget& target, const EngineOptions& options, DeviceContext& device);

} // namespace targets
} // namespace ninfer

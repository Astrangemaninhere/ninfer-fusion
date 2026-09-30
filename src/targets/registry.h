#pragma once

#include "ninfer/types.h"
#include "runtime/engine/context_cost.h"
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6_35b_a3b/package.h>
#include <ninfer/targets/muse_glimmer_30b/package.h>
#include <ninfer/targets/qwen3_5_9b/package.h>
#include <ninfer/targets/spark_x2_5_4b/package.h>

#include <memory>
#include <variant>

namespace ninfer {

struct DeviceContext;

namespace targets {

using Qwen3_6_27B    = qwen3_6_27b::Package;
using Qwen3_6_35BA3B = qwen3_6_35b_a3b::Package;
using MuseGlimmer30B = muse_glimmer_30b::Package;
using Qwen3_5_9B    = qwen3_5_9b::Package;
// Spark-X2.5-4B.  A NEW FAMILY, not a new checkout of an existing one: its
// geometry is 16 query / 4 KV heads at head_dim 256 with a per-head sigmoid
// output gate and a gated-GELU MLP, and dl/sparktarget measured that no existing
// package declares its model id.  The package header is
// src/targets/spark_x2_5_4b/export/ninfer/targets/spark_x2_5_4b/package.h and its
// `declares_model("spark-x2.5-4b")` is the routing predicate the row below uses.
using SparkX2_5_4B  = spark_x2_5_4b::Package;

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

struct LoadedQwen3_5_9B {
    std::unique_ptr<Qwen3_5_9B::LoadedModel> model;
    Qwen3_5_9B::Frontend frontend;

    LoadedQwen3_5_9B(std::unique_ptr<Qwen3_5_9B::LoadedModel> stable_model,
                     const EngineOptions& options);
    ~LoadedQwen3_5_9B();

    LoadedQwen3_5_9B(const LoadedQwen3_5_9B&)            = delete;
    LoadedQwen3_5_9B& operator=(const LoadedQwen3_5_9B&) = delete;
};

struct Qwen3_5_9BInstance {
    using Package = Qwen3_5_9B;

    std::unique_ptr<LoadedQwen3_5_9B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_5_9B::Program> program;
    Qwen3_5_9B::WeightsProfile weights_profile;

    Qwen3_5_9BInstance(std::unique_ptr<LoadedQwen3_5_9B> stable_loaded,
                       runtime::KvCapacityResolution resolution,
                       Qwen3_5_9B::SequencePlan sequence_plan,
                       Qwen3_5_9B::WeightsProfile weights_profile_in,
                       DeviceContext& device);
    ~Qwen3_5_9BInstance();

    Qwen3_5_9BInstance(const Qwen3_5_9BInstance&)            = delete;
    Qwen3_5_9BInstance& operator=(const Qwen3_5_9BInstance&) = delete;
};

struct LoadedSparkX2_5_4B {
    std::unique_ptr<SparkX2_5_4B::LoadedModel> model;
    SparkX2_5_4B::Frontend frontend;

    LoadedSparkX2_5_4B(std::unique_ptr<SparkX2_5_4B::LoadedModel> stable_model,
                       const EngineOptions& options);
    ~LoadedSparkX2_5_4B();

    LoadedSparkX2_5_4B(const LoadedSparkX2_5_4B&)            = delete;
    LoadedSparkX2_5_4B& operator=(const LoadedSparkX2_5_4B&) = delete;
};

struct SparkX2_5_4BInstance {
    using Package = SparkX2_5_4B;

    std::unique_ptr<LoadedSparkX2_5_4B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<SparkX2_5_4B::Program> program;
    SparkX2_5_4B::WeightsProfile weights_profile;

    SparkX2_5_4BInstance(std::unique_ptr<LoadedSparkX2_5_4B> stable_loaded,
                         runtime::KvCapacityResolution resolution,
                         SparkX2_5_4B::SequencePlan sequence_plan,
                         SparkX2_5_4B::WeightsProfile weights_profile_in,
                         DeviceContext& device);
    ~SparkX2_5_4BInstance();

    SparkX2_5_4BInstance(const SparkX2_5_4BInstance&)            = delete;
    SparkX2_5_4BInstance& operator=(const SparkX2_5_4BInstance&) = delete;
};

// Spark-X2.5-4B is a NEW FAMILY, so this list grows by one alternative (the
// rule is quoted three paragraphs up in this file).  `replan_target_kv` in
// registry.cpp visits this variant with a generic lambda and
// `InstanceType::Package` is taken from the alternative itself
// (registry.cpp:419-426), so a new alternative needs no second switch.
using ActiveTarget = std::variant<std::unique_ptr<Qwen3_6_27BInstance>,
                                              std::unique_ptr<Qwen3_6_35BA3BInstance>,
                                              std::unique_ptr<MuseGlimmer30BInstance>,
                                              std::unique_ptr<Qwen3_5_9BInstance>,
                                              std::unique_ptr<SparkX2_5_4BInstance>>;

struct ConstructedTarget {
    ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    runtime::ContextMachineCostModel context_cost;
    // W6: the prefill unit the finalized sequence plan was actually built with. The memory ladder
    // inside construct_registered() can lower the requested chunk before the plan is finalized (it
    // retries the plan at successively smaller chunks until the device runtime budget fits), so
    // EngineOptions::prefill_chunk on the way IN is a request, not the value the plan and the
    // prefill loop use. It rides this struct because construct_target() is a free function whose
    // only output is this value, and Engine::Impl adopts it back into its own options so that
    // options() -- which callers read to report what ran (apps/perplexity/main.cpp:376) -- cannot
    // name a chunk no run used. 0 means the target reported nothing; the request then stands.
    std::uint32_t effective_prefill_chunk = 0;
    // ---- the PLE n-gram sidecar, when this load was given one ---------------------------
    // Type-erased on purpose. The owning type is targets/qwen4_exp's PleRuntime, whose header
    // pulls in <cuda_runtime.h>, and this header is included by front ends; the engine's whole
    // relationship with the sidecar is "keep it alive and read its root", so
    // shared_ptr<const void> is the honest width. null + empty root == PLE off, which is what
    // every build before this field existed did and the arm every existing run takes.
    // It rides this struct for the same reason effective_prefill_chunk does: it is a fact
    // about THIS load that no caller can recompute, and registry.cpp's attach would otherwise
    // be a local destroyed before the function returned.
    std::shared_ptr<const void> ple_sidecar;
    std::string ple_sidecar_root;
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

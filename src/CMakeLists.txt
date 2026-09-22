function(ninfer_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
endfunction()

function(ninfer_cuda_archive target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

# Device/tensor/layout/graph primitives and the model-independent request runtime.
add_library(ninfer_core STATIC
  core/arena.cu
  core/device.cu
  core/device_probe.cu
  core/dtype.cpp
  core/host_worker_pool.cpp
  core/gdn_replay_records.cpp
  core/cyclic_kv_cache.cpp
  core/host_kv_arena.cpp
  core/paged_kv_cache.cpp
  core/linear_attention_state.cpp
  core/layout.cpp
  core/tensor.cpp
  ops/ple/ple_layout.cpp
  core/decode_graph.cpp
  runtime/engine/admission_policy.cpp
  runtime/engine/context_cost.cpp
  runtime/engine/context_cost_defaults.cpp
  runtime/engine/kv_capacity.cpp
  runtime/engine/public_types.cpp
  core/build_arch_list.cpp)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
# NOTE: NINFER_HAVE_QPN, the fp16-fallback build fact, is NOT published here. It is published
# by the ninfer_qpn_build_fact INTERFACE target, created beside the source line that makes it
# true, in the ninfer_ops block below. Not published on ninfer_core because of measured blast
# radius: a definition here changes the command line of every TU that links ninfer_core --
# nearly all of ninfer_ops -- to add one -D. MEASURED (grep over src/ and tests/): the TUs that
# actually read the fact are src/targets/registry.cpp plus five host test binaries, and all of
# them link ninfer_artifact.
# The arch list is published to the ONE translation unit that owns it for the whole program,
# core/build_arch_list.cpp, which defines ninfer::caps::kBuildArchList. Every reader -- and the
# test that checks them -- reads that object, so there is exactly one value in the binary and
# exactly one body of build_arch_list(). The probe keeps its own copy for its own failure text.
#
# WHAT THE PREVIOUS SENTENCE CLAIMED, AND WHY IT WAS FALSE. It said the macro "is scoped to this
# one TU so no other compilation unit depends on it". MEASURED (2026-09-19): 1 of the 573
# recorded compile commands defines the macro (core/device_probe.cu), and that TU does not
# include core/arch_caps.h. So the FUNCTION that depended on it -- build_arch_list() -- could
# never receive it and returned the empty view in all 13 TUs that include the header. Scoping a
# macro to a TU that does not read it is not a narrow blast radius; it is a silent zero.
string(REPLACE ";" "," NINFER_CUDA_ARCHS_TEXT "${CMAKE_CUDA_ARCHITECTURES}")
set_source_files_properties(core/device_probe.cu PROPERTIES
  COMPILE_DEFINITIONS "NINFER_BUILD_CUDA_ARCHS=\"${NINFER_CUDA_ARCHS_TEXT}\"")
if(NINFER_CUDA_ARCHS_TEXT STREQUAL "")
  message(FATAL_ERROR
    "CMAKE_CUDA_ARCHITECTURES is empty, so the binary's arch list would be the empty string and "
    "ninfer::caps::build_arch_list() would report <unreported> for a binary that WAS built for "
    "some arch. Set -DCMAKE_CUDA_ARCHITECTURES (default 120a).")
endif()
set_source_files_properties(core/build_arch_list.cpp PROPERTIES
  COMPILE_DEFINITIONS "NINFER_BUILD_CUDA_ARCHS=\"${NINFER_CUDA_ARCHS_TEXT}\"")
target_link_libraries(ninfer_core PUBLIC CUDA::cudart CUDA::nvtx3 Threads::Threads)

# Generic .ninfer framing, binding, and direct final materialization.
add_library(ninfer_artifact STATIC
  artifact/reader.cpp
  artifact/storage_layouts.cpp
  artifact/binder.cpp
  artifact/materializer.cpp
  artifact/typed_binding.cpp)
ninfer_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core)

# The warp-specialized SM120 kernel uses setmaxnreg to transfer registers from its
# producer warpgroup to its consumer warpgroups. Relocatable device code makes ptxas
# discard that contract, so this self-contained kernel is intentionally compiled in a
# non-RDC archive and called through a host launcher from ninfer_ops.
add_library(ninfer_nvfp4_tma STATIC
  ops/linear/nvfp4/nvfp4_w4a4_tma.cu
  ops/linear/nvfp4/nvfp4_w4a4_tma_attn.cu
  ops/linear/nvfp4/nvfp4_w4a4_tma_gdn.cu
  ops/linear/nvfp4/nvfp4_w4a4_tma_mlp.cu
  ops/linear/nvfp4/nvfp4_w4a4_tma_residual.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu)
ninfer_internal_includes(ninfer_nvfp4_tma)
set_target_properties(ninfer_nvfp4_tma PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
target_compile_options(ninfer_nvfp4_tma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(ninfer_nvfp4_tma PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)

# ---------------------------------------------------------------------------
# THE fp16 FALLBACK BUILD FACT -- created here, four lines above the source line that makes
# it true, and nowhere else.
# ---------------------------------------------------------------------------
# ninfer_qpn_build_fact publishes NINFER_HAVE_QPN, which src/core/arch_caps.h reads as
# caps::kQpnInBuild and src/core/kernel_route.h turns into the fp16 fallback route's entry
# condition. The definition and the add_library entry below are the two halves of ONE
# statement: creating this target anywhere else in the file would let the claim and the thing
# that makes it true drift apart.
#
# REACH: linked into ninfer_artifact (below) PUBLICLY, which is what src/targets/registry.cpp
# -- the engine's only sm() gate call site -- and every host test binary that reads the two
# tables links. See the note on ninfer_core above for why it is not published there.
#
# PUBLIC, NOT PRIVATE, AND THAT IS A MEASURED DEFECT FIX. An INTERFACE target's
# INTERFACE_COMPILE_DEFINITIONS do not propagate through a PRIVATE link, so with PRIVATE the
# five ninfer_artifact TUs got -DNINFER_HAVE_QPN=1 while src/targets/registry.cpp (target
# ninfer_engine) got -DNDEBUG as its only -D. MEASURED 2026-09-18 against
# build/compile_commands.json (SIM-RUN section 5.1): the engine's gate therefore answered as
# if the fallback kernel were ABSENT while the kernel was linked into the same binary -- the
# exact "two tables answering one question drift" hazard the paragraph above says this
# arrangement exists to prevent, and the phantom-route class it names. The comment promised a
# reach the word PRIVATE does not deliver; PUBLIC is the word that delivers it.
#
# IT IS NOT A SWITCH A CALLER MAY FLIP. A hand-written -DNINFER_HAVE_QPN=1 with the source
# below removed yields a route selection naming a kernel the binary does not contain -- the
# phantom route this tree has already shipped once. The tie that makes it impossible is in
# tests/test_arch_generic_fallback.cpp: it links ninfer_ops AND takes the ADDRESS of
# ops::qpn::gemm_qpn, so the definition and the object file have to arrive together or the test
# does not link.
add_library(ninfer_qpn_build_fact INTERFACE)
target_compile_definitions(ninfer_qpn_build_fact INTERFACE NINFER_HAVE_QPN=1)
target_link_libraries(ninfer_artifact PUBLIC ninfer_qpn_build_fact)

# Shared mathematical Ops. The list is intentionally explicit: adding a
# source is a build-boundary decision, not an accidental recursive-glob side effect.
add_library(ninfer_ops STATIC
  ops/launcher/add_bias.cu
  ops/launcher/argmax.cu
  ops/launcher/suffix_lookup.cu
  ops/launcher/vocab_topk16.cu
  ops/launcher/mtp_proposal_topk.cu
  ops/split_attention.cu
  ops/launcher/bidirectional_gqa_attention.cu
  ops/launcher/dspark_markov_argmax.cu
  ops/launcher/dspark_markov_argmax.cu
  ops/launcher/cast.cu
  ops/launcher/causal_conv1d.cu
  ops/launcher/dflash2_grouped_conv.cu
  ops/launcher/gqa_attention_decode.cu
  ops/launcher/gqa_attention_decode_smallt.cu
  ops/launcher/gqa_attention_decode_e8.cu
  ops/launcher/gqa_attention_decode_i8.cu
  ops/launcher/gqa_attention_decode_nvfp4.cu
  ops/launcher/gqa_attention_decode_fp8.cu
  ops/launcher/gqa_attention_decode_iso3.cu
  ops/launcher/gqa_attention_decode_bf16.cu
  ops/launcher/gqa_attention_prefill.cu
  ops/launcher/gqa_attention_prefill_e8.cu
  # ---------------------------------------------------------------------------
  # THE W3/W2 E8 LATTICE K-PLANE READER'S COMPILE FLOOR -- the third e8 entry in this
  # file, and the first that is not 4-bit.
  # ---------------------------------------------------------------------------
  # src/ops/kernel/e8_lattice_kv_plane.cuh is the DEVICE reader for this tree's 3-bit and
  # 2-bit E8 lattice K plate (the writer, the reconstruction reader, and the integer reader
  # the s8 MMA path needs, at both widths). Before this line it was compiled by NOTHING:
  # `e8_lattice_kv_plane` had 0 hits in src/CMakeLists.txt, tests/CMakeLists.txt and the root
  # CMakeLists.txt, and 0 occurrences in ninfer_ops's own
  # build/src/CMakeFiles/ninfer_ops.dir/compiler_depend.internal -- MEASURED 2026-09-18, with
  # control `cuda_runtime.h` = 247 in that same file, so the 0 is a real absence and not a
  # broken needle. Flipping the rk3v4/rk2v4 `selectable` gate against that state builds a
  # correct 3-bit/2-bit K plate and hands it to a reader no target compiles. NO READER, NO FLIP.
  #
  # WHY FORCED-INSTANTIATION, AND NOT A LAUNCHER: every function in the reader is
  # `__device__ __forceinline__`, so a TU that only INCLUDES the header emits no device code
  # at all (MEASURED: a bare `nvcc -c` of an include-only TU exits 0, writes a 6,656-byte
  # object, and nm shows no reader symbol). That check reads as a pass while testing nothing.
  # src/ops/kernel/e8_lattice_kv_plane_inst.cu therefore calls all six entry points from ONE
  # __global__, and that is its only job.
  #
  # WHY NOT A FEW MORE LINES IN gqa_attention_decode_e8.cu: that TU's own header says it was
  # split out "so each translation unit's ptxas stage fits the host memory budget (the E8
  # template flag doubles the kernel instantiation set)". It is the 4-bit tier launcher; this
  # reader is not 4-bit code and must not be re-merged into the one TU documented as
  # memory-tight. Its closure is three headers under src/ops/kernel/ only.
  #
  # THE FALSIFIER is in dl/e8reader/REPORT.md section 3. Remove the reader and the compile
  # goes RED (rc=2, 4,220 stderr bytes, no object). Keep the include and gut the body and the
  # compile is STILL GREEN with the SAME device-entry name and the same nm symbol count -- so
  # the gate must demand the reader's ARITHMETIC (8,584 SASS instructions, 26x the round-half-up
  # 0x1000), never a symbol and never the file.
  ops/kernel/e8_lattice_kv_plane_inst.cu
  # ---------------------------------------------------------------------------
  # THE E8 ACTIVATION-SIDE ROTATION'S COMPILE FLOOR -- the fourth e8 entry in this file, and
  # the second that is a forced-instantiation TU rather than a launcher.
  # ---------------------------------------------------------------------------
  # src/ops/linear/e8act/ is the activation-side half of the weight-side E8 recipe: the
  # rotation that makes W' = W*H equivalent to W on a rotated activation. Its arithmetic is
  # this tree's OWN 64-point Sylvester butterfly, src/ops/kernel/e8_lattice_codec.cuh's
  # e8_lattice_hadamard64_t<float> -- CALLED, not copied, for the reason that header gives at
  # its own line 107-111 (two hand-written copies drift).
  #
  # WHY A LINE IS NEEDED, for exactly the reason the reader above needed one: every function
  # in e8_activation_rotation.cuh is `__device__ __forceinline__`, so a TU that only INCLUDES
  # it emits no device code and an include-only compile check reads as a pass while testing
  # nothing. e8_activation_rotation.cu carries the ONE __global__ that forces instantiation.
  #
  # IT ADMITS NOTHING. This is not a linear dispatch arm, not a NumericFormat, it consumes no
  # Weight, opens no gate and widens no admission: it adds objects to ninfer_ops and nothing
  # else. Registering it does NOT make E8 weights runnable (dl/e8model/REPORT.md section 4.5).
  #
  # THE FALSIFIER: dl/e8model/REPORT.md sections 4.2 and 4.4 -- the tree's own command line
  # gives rc=0 on ten archs (sm_52/53/61/70/75/80/86/89/90/120) and the sm_52 cubin holds
  # ZERO fp64-class instructions, zero LDL, zero STL, zero spill. dl/e8reg/REPORT.md section 1
  # re-measures the same compile from THIS build system, with rc recorded separately from bytes.
  ops/linear/e8act/e8_activation_rotation.cu
  ops/launcher/dflash2_selector.cu
  ops/launcher/dflash2_tree_walk.cu
  ops/launcher/embed_gather.cu
  ops/launcher/gdn_gating.cu
  ops/launcher/gelu.cu
  ops/launcher/gelu_and_mul.cu
  ops/launcher/cold_i8.cu
  ops/launcher/entropy_cold_requant.cu
  ops/launcher/entropy_nvfp4_slot.cu
  ops/launcher/l2norm.cu
  ops/launcher/layer_norm.cu
  ops/launcher/mtp_pack.cu
  ops/launcher/mtp_round.cu
  ops/launcher/speculative_round.cu
  ops/launcher/position.cu
  ops/launcher/prepare_ragged_prefix.cu
  ops/launcher/prepare_masked_block.cu
  ops/launcher/residual_add.cu
  ops/launcher/rmsnorm.cu
  ops/launcher/rope.cu
  ops/launcher/sampling.cu
  ops/launcher/scalar.cu
  ops/launcher/swa.cu
  ops/kernel/gqa_isoquant_rot.cu
  ops/kernel/gqa_isoquant_row_scale.cu
  ops/kernel/gqa_isoquant_row_scale_loader.cu
  ops/launcher/scatter.cu
  ops/launcher/sigmoid_gate_mul.cu
  ops/launcher/logit_policy.cu
  ops/launcher/silu_and_mul.cu
  ops/launcher/target_logprobs.cu
  ops/launcher/vision_pos_embed.cu
  ops/ple/ple_table.cu
  # Softmax Attention and KV-cache state transitions.
  ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp
  ops/softmax_attention/dense/causal_cache/small_t.cu
  ops/softmax_attention/dense/causal_cache/small_t_fp8.cu
  ops/softmax_attention/dense/causal_cache/prompt.cu
  ops/softmax_attention/dense/causal_cache/prompt_fp8.cu
  ops/softmax_attention/dense/packed/packed_softmax_attention.cpp
  ops/softmax_attention/dense/packed/launch.cu
  ops/softmax_attention/dense/context/context_softmax_attention.cpp
  ops/softmax_attention/dense/context/launch.cu
  ops/softmax_attention/sliding_window/sliding_window_attention.cpp
  ops/softmax_attention/sliding_window/launch.cu
  ops/softmax_attention/selected_block/selected_block_attention.cu
  ops/kv_cache/append/kv_cache_append.cpp
  ops/kv_cache/append/launch.cu
  # Gated DeltaNet Linear Attention core.
  ops/linear_attention/gated_delta_net/gated_delta_net.cpp
  ops/linear_attention/gated_delta_net/replay.cpp
  ops/linear_attention/gated_delta_net/recurrent.cu
  ops/linear_attention/gated_delta_net/chunked/launch.cu
  ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cu
  ops/linear_attention/gated_delta_net/chunked/state_passing.cu
  ops/linear_attention/gated_delta_net/chunked/output.cu
  ops/attn_input_proj/bf16/bf16_attn_input_decode.cu
  ops/attn_input_proj/bf16/bf16_attn_input_gemm_mma.cu
  ops/attn_input_proj/bf16/bf16_attn_input_small_t.cu
  ops/attn_input_proj/bf16/bf16_attn_input_plan.cpp
  ops/attn_input_proj/fp8/fp8_attn_input_decode.cu
  ops/attn_input_proj/fp8/fp8_attn_input_small_t.cu
  ops/attn_input_proj/fp8/fp8_attn_input_a8.cu
  ops/attn_input_proj/fp8/fp8_attn_input_plan.cpp
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_small_t.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_w4a4.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_gemm_mma.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp
  ops/attn_input_proj/w8/w8_attn_input_decode.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_mma.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_simt.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_splitk.cu
  ops/attn_input_proj/w8/w8_attn_input_plan.cpp
  ops/gdn_input_proj/fp8/fp8_gdn_input_decode.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_small_t.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_plan.cpp
  ops/gdn_input_proj/fp8/fp8_gdn_conv_fused.cu
  ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.cpp
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_decode.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_small_t.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_decode.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_small_t.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_post.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_gemm_mma.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp
  ops/gdn_input_proj/gdn_projected_conv.cu
  ops/gdn_input_proj/w8/w8_gdn_input_decode.cu
  ops/gdn_input_proj/w8/w8_gdn_input_gemm_mma.cu
  ops/gdn_input_proj/w8/w8_gdn_input_gemm_splitk.cu
  ops/gdn_input_proj/w8/w8_gdn_input_plan.cpp
  ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu
  ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp
  ops/linear/bf16/bf16_gemv.cu
  ops/linear/bf16/bf16_gemm_mma.cu
  ops/linear/bf16/bf16_small_t.cu
  ops/linear/bf16/bf16_dispatch.cpp
  ops/linear/fp8/fp8_format.cpp
  ops/linear/fp8/fp8_gemv.cu
  ops/linear/fp8/fp8_small_t.cu
  ops/linear/fp8/fp8_a16_small_t_mma.cu
  ops/linear/fp8/fp8_a16_gemm_mma.cu
  ops/linear/fp8/fp8_a8.cu
  ops/linear/fp8/fp8_dispatch.cpp
  ops/linear/nvfp4/nvfp4_format.cpp
  ops/linear/nvfp4/nvfp4_gemv.cu
  ops/linear/nvfp4/nvfp4_small_t.cu
  ops/linear/nvfp4/nvfp4_w4a4.cu
  ops/linear/nvfp4/nvfp4_dispatch.cpp
  # ---------------------------------------------------------------------------
  # THE ARCH-GENERIC fp16 FALLBACK GEMM (the QPN W4A16 family).
  # ---------------------------------------------------------------------------
  # This is the ONE source line that decides whether the fp16 fallback route exists in a
  # build, and it is deliberately adjacent to the definition that publishes the fact
  # (NINFER_HAVE_QPN, set on ninfer_core above) so that the two cannot be read apart.
  #
  # WHY IT WAS MISSING, AND WHAT IT COST. The QPN family existed in the tree and was
  # compiled by NO target: `qpn` had 0 hits in this file, this file's source lists are
  # explicit (no GLOB, deliberately -- see the ninfer_ops comment), and no archive under
  # build/src/ carried a qpn object. So src/core/kernel_route.h refused NVFP4 and
  # FP8_E4M3FN_ROW_BF16S on sm_70/sm_75 with a reason that named a kernel the build could
  # not launch, and the two capability bits that describe this path (Cap::Fp16Mma,
  # Cap::Int8Mma) were required by NO kFormatRequirements row and decided nothing.
  #
  # WHY IT IS ARCH-GENERIC AND NOT AN sm_70 SPECIAL CASE. The family reaches the tensor
  # core through mma.sync.aligned.m8n8k4, which is measured to assemble on EVERY rung from
  # sm_70 to sm_120a, and one TU compiles for all of them: MEASURED rc=0 -cubin for
  # sm_70/sm_75/sm_80/sm_86/sm_89/sm_90/sm_100a/sm_120a with this file's own flags, and the
  # sm_70/sm_75 cubins contain HMMA.884.F32.F32.STEP0..3 with ZERO calls into ptxas's
  # software FFMA emulation routine. It needs no ldmatrix (sm_75+) and no cp.async (sm_80+):
  # both counts are 0 in src/ops/linear/qpn/qpn_kernels.cuh, which is why it is the one
  # fp16 GEMM in this tree that reaches Volta at all.
  #
  # WHAT IT DOES NOT DO: it does not make the route selectable on a rung where ptxas
  # answers that mma with a software routine. From sm_80 up the m8n8k4 channel is EMULATED
  # (measured per rung, kQpnMmaRungs in src/core/arch_caps.h), so
  # caps::fp16_fallback_executable() refuses there and the route stays out of reach. The
  # build fact and the arch fact are two different questions and are answered in two
  # different places on purpose.
  #
  # THE TIE: tests/test_qpn_build_fact.cpp links this target and references
  # ops::qpn::gemm_qpn, so removing this source line while leaving NINFER_HAVE_QPN defined
  # is a LINK ERROR rather than a phantom route.
  ops/linear/qpn/qpn_host.cu
  # THE ENGINE'S CALLER OF THE ROUTE TABLES, added beside the kernel it can dispatch. Without
  # this line the route layer stays host-only (MEASURED 2026-09-18, SIM-RUN section 1.2: zero
  # callers in src/ and apps/, `qpn-w4a16` absent from the binary, route_log_line() called
  # from nowhere) and a simulated rung cannot be observed in an engine run at all.
  ops/linear/qpn/qpn_arch_route.cpp
  ops/linear/q4/q4_small_t_mma.cu
  ops/linear/q4/q4_rowsplit_gemm_mma.cu
  ops/linear/q4/q4_rowsplit_gemm_simt.cu
  ops/linear/q4/q4_rowsplit_gemv.cu
  ops/linear/q4/q4_dispatch.cpp
  ops/linear/q5/q5_dispatch.cpp
  ops/linear/q5/q5_rowsplit_gemm_mma.cu
  ops/linear/q5/q5_rowsplit_gemm_simt.cu
  ops/linear/q5/q5_rowsplit_gemv.cu
  ops/linear_add/bf16/bf16_linear_add_decode.cu
  ops/linear_add/bf16/bf16_linear_add_gemm_mma.cu
  ops/linear_add/bf16/bf16_linear_add_small_t.cu
  ops/linear_add/bf16/bf16_linear_add_plan.cpp
  ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_small_t.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_w4a4.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp
  ops/linear_add/fp8/fp8_linear_add_decode.cu
  ops/linear_add/fp8/fp8_linear_add_small_t.cu
  ops/linear_add/fp8/fp8_linear_add_a8.cu
  ops/linear_add/fp8/fp8_linear_add_plan.cpp
  ops/linear_add/q5/q5_linear_add_gemm_mma.cu
  ops/linear_add/q5/q5_linear_add_gemm_simt.cu
  ops/linear_add/q5/q5_linear_add_gemv.cu
  ops/linear_add/q5/q5_linear_add_plan.cpp
  ops/linear_add/w8/w8_linear_add_gemm_mma.cu
  ops/linear_add/w8/w8_linear_add_gemm_simt.cu
  ops/linear_add/w8/w8_linear_add_gemm_splitk.cu
  ops/linear_add/w8/w8_linear_add_plan.cpp
  ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cu
  ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu
  ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp
  ops/linear_swiglu/fp8/fp8_linear_swiglu_decode.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_small_t.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp
  ops/linear_swiglu/w8/w8_linear_swiglu_decode.cu
  ops/linear_swiglu/w8/w8_linear_swiglu_gemm_mma.cu
  ops/linear_swiglu/w8/w8_linear_swiglu_gemm_splitk.cu
  ops/linear_swiglu/w8/w8_linear_swiglu_plan.cpp
  ops/linear/q6/q6_dispatch.cpp
  ops/linear/q6/q6_rowsplit_gemm_mma.cu
  ops/linear/q6/q6_rowsplit_gemm_simt.cu
  ops/linear/w8/w8_rowsplit_gemm_mma.cu
  ops/linear/w8/w8_rowsplit_gemm_simt.cu
  ops/linear/w8/w8_rowsplit_gemm_decode.cu
  ops/linear/w8/w8_small_t.cu
  ops/linear/w8/w8_rowsplit_gemm_splitk.cu
  ops/linear/w8/w8_dispatch.cpp
  ops/linear_pair/w8/w8_pair_decode.cu
  ops/linear_pair/w8/w8_pair_gemm_concat.cu
  ops/linear_pair/w8/w8_pair_gemm_mma.cu
  ops/linear_pair/w8/w8_pair_gemm_splitk.cu
  ops/linear_pair/w8/w8_pair_plan.cpp
  ops/sparse_moe/decode/sparse_moe_decode_kernels.cu
  ops/sparse_moe/decode/sparse_moe_decode_plan.cpp
  ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu
  ops/sparse_moe/prefill/sparse_moe_prefill_plan.cpp
  ops/sparse_moe/small_t/sparse_moe_small_t_kernels.cu
  ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp
  ops/linear/linear.cpp
  # Unregistered-shape fallback for the k128-v1 row-split families. The registered kernels are
  # exact (N,K) registries -- every entry in them came from some model's geometry -- so a target
  # whose geometry no entry carries has no execution path at all. This module decodes the same
  # payload with the same bit conventions and is correct before it is fast.
  ops/generic/rowsplit_generic.cu
  ops/wrapper/add_bias.cpp
  ops/wrapper/argmax.cpp
  ops/wrapper/suffix_lookup.cpp
  ops/wrapper/bidirectional_gqa_attention.cpp
  ops/wrapper/dspark_markov_argmax.cpp
  ops/wrapper/cast.cpp
  ops/wrapper/causal_conv1d_silu.cpp
  ops/wrapper/dflash2_grouped_conv.cpp
  ops/wrapper/gqa_attention.cpp
  ops/wrapper/dflash2_selector.cpp
  ops/wrapper/dflash2_tree_walk.cpp
  ops/wrapper/embedding.cpp
  ops/wrapper/cold_i8.cpp
  ops/wrapper/entropy_cold_requant.cpp
  ops/wrapper/entropy_nvfp4_slot.cpp
  ops/wrapper/gdn_gating.cpp
  ops/wrapper/gdn_gating_proj.cpp
  ops/wrapper/gdn_input_proj.cpp
  ops/wrapper/gelu.cpp
  ops/wrapper/gelu_mul.cpp
  ops/wrapper/attn_input_proj.cpp
  ops/wrapper/l2norm.cpp
  ops/wrapper/layer_norm.cpp
  ops/wrapper/linear_add.cpp
  ops/wrapper/linear_pair.cpp
  ops/wrapper/linear_swiglu.cpp
  ops/wrapper/mtp_pack.cpp
  ops/wrapper/mtp_round.cpp
  ops/wrapper/speculative_round.cpp
  ops/wrapper/position.cpp
  ops/wrapper/prepare_ragged_prefix.cpp
  ops/wrapper/prepare_masked_block.cpp
  ops/wrapper/residual_add.cpp
  ops/wrapper/rmsnorm.cpp
  ops/wrapper/rope.cpp
  ops/wrapper/sampling.cpp
  ops/wrapper/swa.cpp
  ops/wrapper/scalar.cpp
  ops/wrapper/scatter.cpp
  ops/wrapper/sigmoid_mul.cpp
  ops/wrapper/logit_policy.cpp
  ops/wrapper/silu_mul.cpp
  ops/wrapper/sparse_moe.cpp
  ops/wrapper/target_logprobs.cpp
  ops/wrapper/vision_pos_embed.cpp)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_archive(ninfer_ops)
# ---------------------------------------------------------------------------
# ninfer_artifact IS PART OF ninfer_ops'S INTERFACE, BECAUSE AN OPS HEADER NAMES ITS SYMBOL.
# ---------------------------------------------------------------------------
# src/core/kernel_route.h renders the route table's `why` strings through
# artifact::format_name() (declared artifact/reader.h:65, DEFINED only in
# artifact/storage_layouts.cpp:67). Those call sites are inline, so the reference belongs to every
# TU that instantiates them -- and the member that carries it is NOT the member that needs it:
# ninfer_ops's own ops/linear/nvfp4/nvfp4_dispatch.cpp.o takes ops::qpn::select_qpn_arch_route(),
# which pulls ops/linear/qpn/qpn_arch_route.cpp.o out of THIS archive, and THAT object undefines
# `ninfer::artifact::format_name(ninfer::artifact::NumericFormat)`. A grep of qpn_arch_route.cpp
# finds zero occurrences because the call sits ONE INLINE LEVEL DOWN, in the header.
# MEASURED 2026-09-18 (nm over build/src/libninfer_ops.a, then the member closure):
# 19 T/W symbols in qpn_arch_route.cpp.o, 6 members reachable from it
# (qpn_arch_route, nvfp4_dispatch, linear, linear_swiglu, nvfp4_linear_swiglu_plan,
# q4_linear_swiglu_plan), and 13 final links that carry ninfer_ops without ninfer_artifact:
# ninfer_linear_{bf16,fp8_a16,fp8_a8,nvfp4_a16,nvfp4_a4,q4,q5,q6,w8}_a16_test and
# ninfer_linear_swiglu_{fp8,nvfp4,q4_a16,w8_a16}_test. Verbatim from one of them
# (scratch/PATCHSET/PROBE2ROUTE/arm_A1b_relink.log, rc=2):
#   ../src/libninfer_ops.a(qpn_arch_route.cpp.o): in function
#   `ninfer::caps::route_log_line(int, ninfer::artifact::NumericFormat, ...)`:
#   undefined reference to `ninfer::artifact::format_name(ninfer::artifact::NumericFormat)'
# THE FAILURE SHAPE IS THE REASON THIS IS THE RIGHT PLACE: GNU make DELETES the output of a failed
# link, so the symptom is an absent binary -- "not built" -- and a stale binary hides it until
# something forces the relink.
#
# PUBLIC, NOT PRIVATE. The requirement is in the INTERFACE and not in the implementation: the
# symbol is named by a header that every consumer of ninfer_ops compiles, and the four consumers
# that already call it -- tests/test_arch_caps.cpp, tests/test_arch_generic_fallback.cpp,
# tests/test_kernel_route.cpp, tests/test_multidev_wiring.cpp -- each link ninfer_artifact by hand
# today for exactly this reason. PRIVATE would reach a consumer's link line anyway -- CMake records
# a static archive's private dependencies as $<LINK_ONLY:> in INTERFACE_LINK_LIBRARIES -- but it
# would state the contract as "an implementation detail of ninfer_ops's TUs", which is the
# opposite of what is true. (MEASURE it rather than argue it: a throwaway 5-line project shows
# PUBLIC/PRIVATE/$<LINK_ONLY:> all add the archive to the consumer's link line, and differ only in
# whether the compile definition rides along to the consumer.)
#
# AND YES, THE ARCHIVE'S OWN DEPENDENCIES COME WITH IT -- one of them deliberately does something
# extra, so it is named here rather than discovered later. ninfer_artifact's interface carries
# ninfer_core (already PUBLIC on ninfer_ops: no change) AND ninfer_qpn_build_fact, an INTERFACE
# target whose only content is the compile definition NINFER_HAVE_QPN=1. Linking this archive
# therefore adds that -D to the command line of every ninfer_ops TU and of every consumer of
# ninfer_ops. That is NOT a side effect to be hidden: ninfer_ops is the archive that CONTAINS
# ops/linear/qpn/qpn_host.cu, so its TUs answering kQpnInBuild == true is the truthful answer
# (src/core/arch_caps.h:755-759 turns the macro into that constant), and the previously documented
# hazard is the false one -- src/core/kernel_route.h:218-224 records the measured defect that this
# very TU saw kQpnInBuild == false while the QPN kernel was in the binary it was linked into, which
# is why qpn_arch_route.cpp answers the build fact from the LINKER through the explicit
# `qpn_in_build` overload instead of from the macro.
#
# WHAT IT DOES NOT DO: it does not move format_name, and that would be wrong. The name is the
# ARTIFACT layer's own spelling of the artifact layer's own key -- artifact/reader.h says the wire
# format carries formats and layouts AS THESE NAMES, so format_name() is the inverse of
# reader.cpp's parse_format() -- and a second name table under src/core/ would be a second answer
# to one question, whose failure mode is a WRONG NAME IN A DIAGNOSTIC rather than a link error.
# The core layer already depends on this layer's header by design (src/core/arch_caps.h includes
# artifact/reader.h and keys kFormatRequirements on artifact::NumericFormat), so the name is a
# consumed export, not a layering inversion.
target_link_libraries(ninfer_ops PUBLIC ninfer_core ninfer_artifact PRIVATE ninfer_nvfp4_tma)

# Product-independent text and media mechanisms.
add_library(ninfer_text STATIC
  text/unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c)
ninfer_internal_includes(ninfer_text)

add_library(ninfer_media_decode STATIC
  media/decode/decode.cpp)
ninfer_internal_includes(ninfer_media_decode)
target_link_libraries(ninfer_media_decode PRIVATE PkgConfig::FFMPEG)

if(NINFER_BUILD_MEDIA_ACQUIRE)
  # Product-only path/data/HTTP acquisition. No target package links this library.
  add_library(ninfer_media_acquire STATIC
    product/media_acquire/acquire.cpp)
  ninfer_internal_includes(ninfer_media_acquire)
  target_link_libraries(ninfer_media_acquire PRIVATE PkgConfig::LIBCURL)
endif()

if(NINFER_BUILD_PROMPT_INPUT)
  # Owning command-line message input shared by product entry points and target diagnostics.
  add_library(ninfer_product_prompt_input STATIC
    product/prompt_input/prompt_input.cpp)
  ninfer_internal_includes(ninfer_product_prompt_input)
  target_link_libraries(ninfer_product_prompt_input PRIVATE ninfer_media_acquire)
endif()

if(NINFER_BUILD_APPS OR BUILD_TESTING)
  add_library(ninfer_product_load_progress STATIC
    product/load_progress/load_progress.cpp)
  ninfer_internal_includes(ninfer_product_load_progress)
endif()

# One cold closed registry plus the opaque public product Engine. Family and exact-target
# directories contribute their sources directly to this single compile owner below.
add_library(ninfer_engine STATIC
  targets/registry.cpp
  runtime/contract/sampling.cpp
  # Flash-Next port: the copied target calls runtime::OutputConstraintState,
  # OutputConstraintCompiler, CompiledOutputConstraint and validate_structured_output, and the
  # donor's implementation of those was never taken with the copy-in, so they were undefined at
  # link time.  The donor's file needs <xgrammar/xgrammar.h> and the donor vendors the library as
  # third_party/xgrammar; THIS TREE DOES NOT HAVE IT.  The file registered here is therefore a
  # REFUSING STUB, not the donor implementation: it accepts StructuredOutputKind::Text and throws
  # for JsonObject/JsonSchema.  Read its header before relying on structured output.
  runtime/contract/structured_output.cpp
  runtime/engine/engine.cpp
  # N6: the qwen4_exp PLE seam (impl/ple_runtime.h) had no compilation owner at
  # all before this line; the header accumulated six compile errors no build
  # could see. A dedicated src/targets/qwen4_exp/CMakeLists.txt is the natural
  # home once the target has more sources (a sibling work stream is adding one);
  # registering the single TU here keeps this patch independent of that.
  targets/qwen4_exp/impl/ple_runtime.cpp)
ninfer_internal_includes(ninfer_engine)
target_include_directories(ninfer_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(targets/qwen3_6)
add_subdirectory(targets/qwen3_6_27b)
add_subdirectory(targets/qwen3_6_35b_a3b)
add_subdirectory(targets/muse_glimmer_30b)
add_subdirectory(targets/qwen3_5_9b)
# S37 stage (a): identity-only registration. The export header (identity + geometry
# cross-check) and impl/config.h join the engine include path so the registry
# recognition branch compiles; no sources are added until stage (b).
add_subdirectory(targets/qwen4_exp)
add_subdirectory(targets/gemma4_31b)

# FlashNext (Qwen3.8-Flash-Next / qwen4_exp): the imported qwen3_8_flash_next target and its
# vision encoder dependency. BOUNDARY NOTE: this file is the FALLBACK agent's region; that
# agent finished and released the boundary (scratch/PATCHSET/FALLBACK/REPORT.md), so these two
# lines are added here rather than only reported. Nothing else in this file is touched --
# ninfer_qpn_build_fact and the source lists below are deliberate and left alone.
#
# EXCLUDE_FROM_ALL is deliberate, but its REASON has changed. The compile side is done: a host
# -fsyntax-only sweep of the target's 24 host TUs over the tree's own compile_commands.json is
# now 24 OK / 0 FAIL (it was 10 OK / 14 FAIL on 2026-09-17, then 16/8 after the first six
# symbols landed). Everything the earlier comment listed -- the four artefact/dtype variant
# spellings and the absent shared types -- has landed.
#
# What still keeps these two lines EXCLUDE_FROM_ALL is REGISTRATION, not compilation: the copied
# tree is a target library only. It exports Package/plan_load/create_program, but it has no
# engine-facing Instance adapter, which is what construct_registered<Target, Loaded, Instance>
# needs -- compare src/targets/qwen3_5_9b, which carries ~1080 lines of package/variant/load
# glue for that role. Without an adapter a kTargetRegistrations row cannot be written, and
# without the row, building the archive by default would cost every other line on this box a
# build with no reachable benefit. Measured 2026-09-18.
# See scratch/PATCHSET/FNRUN/REPORT.md.
add_subdirectory(targets/qwen3_vision EXCLUDE_FROM_ALL)
add_subdirectory(targets/qwen3_8_flash_next EXCLUDE_FROM_ALL)

target_link_libraries(ninfer_engine
  PRIVATE
    ninfer_artifact
    ninfer_core
    ninfer_ops
    ninfer_text
    ninfer_media_decode)
add_library(ninfer::engine ALIAS ninfer_engine)

if(NINFER_BUILD_SERVE)
  # HTTP protocol/product adapter. No serving header is part of the public API.
  add_library(ninfer_serve STATIC
    serve/http_transport.cpp
    serve/request_validation.cpp
    serve/openai_chat_request.cpp
    serve/openai_chat_response.cpp
    serve/openai_chat_http.cpp
    serve/openai_common.cpp
    serve/openai_responses_request.cpp
    serve/openai_responses_state.cpp
    serve/openai_responses_response.cpp
    serve/anthropic_thinking_signature.cpp
    serve/anthropic_messages_request.cpp
    serve/anthropic_messages_response.cpp
    serve/anthropic_messages_http.cpp
    serve/openai_responses_store.cpp
    serve/translate.cpp
    serve/console_log.cpp
    serve/generation_service.cpp
    serve/request_log.cpp
    serve/http_server.cpp
    serve/openai_responses_http.cpp
    serve/serve_options.cpp
    serve/kv_auto_relayout.cpp
    # Gap 1 (free-VRAM axis): the only source in the decision path that names
    # cudaMemGetInfo. Its own TU so kv_auto_relayout.{h,cpp} stay host-only and
    # linkable without libcudart (serve/kv_vram_probe.h); CUDA::cudart is
    # already a PRIVATE dependency of this target (below).
    serve/kv_vram_probe.cpp)
  ninfer_internal_includes(ninfer_serve)
  target_include_directories(ninfer_serve PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  target_link_libraries(ninfer_serve
    PUBLIC ninfer_engine Threads::Threads
    PRIVATE ninfer_media_acquire CUDA::cudart)
endif()

# ---------------------------------------------------------------------------
# THE TEST-ONLY ARCHITECTURE SIMULATOR BUILD KEY (option NINFER_ENABLE_ARCH_SIM, root
# CMakeLists.txt, DEFAULT OFF)
# ---------------------------------------------------------------------------
# NINFER_ARCH_SIM_ENABLED is the FIRST of the three keys src/core/arch_sim.h requires (the
# other two are the NINFER_SIM_ARCH / NINFER_SIM_ARCH_ACK environment pair). It is defined
# ONLY on the targets whose TUs actually include arch_sim.h, because the key's only possible
# effect is on a TU that includes the header -- MEASURED 2026-09-18 (SIM-RUN section 1.3):
# compiling src/targets/registry.cpp with and without the key produced BYTE-IDENTICAL objects
# (sha256 0955fdc801aea2a4...) because no engine TU included arch_sim.h at all. Those two
# facts together are why this block is a list of targets and not a global add_compile_definitions.
#
# It is a separate if() rather than a default-ON because the shipping default must be
# PROVABLE by inspection of a default build: `strings <binary> | grep -c NINFER_SIM_ARCH` = 0.
# A family-wide define would make that check pass by accident and make the OFF default
# unobservable.
if(NINFER_ENABLE_ARCH_SIM)
  message(STATUS "NInfer: the TEST-ONLY architecture simulator is BUILT IN "
                 "(NINFER_ENABLE_ARCH_SIM=ON). A binary from this tree can be told to "
                 "report a lower compute capability to the route and capability tables by "
                 "NINFER_SIM_ARCH + NINFER_SIM_ARCH_ACK. The kernels that execute are still "
                 "this binary's own; only the route decision is simulated.")
  foreach(_sim_target ninfer_engine ninfer_ops ninfer_nvfp4_tma)
    if(TARGET ${_sim_target})
      target_compile_definitions(${_sim_target} PUBLIC NINFER_ARCH_SIM_ENABLED=1)
    endif()
  endforeach()
endif()

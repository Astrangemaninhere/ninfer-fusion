# Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
add_library(ninfer_qwen3_8_flash_next STATIC
  impl/expert_bank.cpp
  impl/frontend.cpp
  impl/gdn.cpp
  impl/gdn_kernels.cu
  impl/hyper_connection.cpp
  impl/hyper_connection_kernels.cu
  impl/lane_ledger.cpp
  impl/load/bindings.cpp
  impl/load/loader.cpp
  impl/load/materialized.cpp
  impl/load/quantize_nvfp4_expert_bank.cu
  impl/load/quantize_output_head.cu
  impl/moe.cpp
  impl/moe_kernels.cu
  impl/moe_route.cu
  impl/moe_shared_kernels.cu
  impl/mtp_forward.cpp
  impl/mtp_forward_kernels.cu
  impl/package.cpp
  impl/program.cpp
  impl/ple_decode.cpp
  impl/ple_decode_kernels.cu
  impl/ple_index.cpp
  impl/ple_pipeline.cpp
  impl/ple_table.cpp
  impl/qsa_attention.cpp
  impl/qsa_attention_kernels.cu
  impl/qsa_indexer.cpp
  impl/qsa_indexer_kernels.cu
  impl/runtime_plan.cpp
  impl/runtime_state.cpp
  impl/stage_ledger.cpp
  impl/text_decode.cpp
  impl/text_decode_kernels.cu
  impl/text_executor.cpp
  impl/vision_execute.cpp)
ninfer_internal_includes(ninfer_qwen3_8_flash_next)
target_include_directories(ninfer_qwen3_8_flash_next PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
target_link_libraries(ninfer_qwen3_8_flash_next PUBLIC ninfer_artifact ninfer_ops ninfer_qwen3_vision ninfer_core)
ninfer_cuda_archive(ninfer_qwen3_8_flash_next)

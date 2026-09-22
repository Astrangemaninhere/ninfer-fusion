// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
add_executable(ninfer_qwen3_8_flash_next_reference
  main.cpp
  options.cpp)
ninfer_internal_includes(ninfer_qwen3_8_flash_next_reference)
target_include_directories(ninfer_qwen3_8_flash_next_reference PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_8_flash_next
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_vision/export)
target_link_libraries(ninfer_qwen3_8_flash_next_reference PRIVATE
  ninfer_qwen3_8_flash_next
  ninfer_engine
  ninfer_qwen3_vision
  ninfer_artifact
  ninfer_core
  ninfer_ops
  CUDA::cudart)

add_executable(ninfer_quantize_mtp
  quantize_mtp.cpp)
ninfer_internal_includes(ninfer_quantize_mtp)
target_include_directories(ninfer_quantize_mtp PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_8_flash_next)
target_link_libraries(ninfer_quantize_mtp PRIVATE
  ninfer_qwen3_8_flash_next
  ninfer_artifact
  ninfer_core
  CUDA::cudart)


# src/targets/qwen4_exp/stage_b.cmake -- STAGE (b) COMPLETENESS, DERIVED FROM THE TREE.
#
# include()d by CMakeLists.txt in this directory, and deliberately SELF-CONTAINED so its
# verdict can be re-run on its own against either file-set shape:
#
#     cmake -P src/targets/qwen4_exp/stage_b.cmake
#
# That is how both arms below are rehearsed here without a build: the predicate is a pure
# function of the file set, so the two arms can be exercised by adding and removing files.
#
# WHY AN EXISTENCE TEST AND NOT A COMMENT. src/targets/registry.cpp refused this family with an
# UNCONDITIONAL throw -- taken whether or not the runtime existed -- so the day stage (b)
# landed, the gate would have stayed shut until somebody remembered to edit the registry.
# Inverted here: the registry's refusal is CONDITIONAL on NINFER_QWEN4_EXP_STAGE_B_COMPLETE,
# that definition is computed from the filesystem, and the absent-but-declared sources are
# named in the refusal message by the SAME test. The open arm is then the arm that has to
# exist, and it cannot be reached by accident.
#
# FAIL-SAFE DIRECTION: a translation unit that does not see these definitions reads
# kStageBImplemented == false (export/ninfer/targets/qwen4_exp/package.h) and refuses BY NAME.
# An unknown state must refuse, never load.
#
# Inputs : Q4_STAGE_B_ROOT -- this directory (defaulted from this file's own location)
# Outputs: Q4_STAGE_B_REQUIRED_SOURCES -- the declared set, in one place
#          Q4_STAGE_B_MISSING          -- the declared-but-absent subset, space separated
#          Q4_STAGE_B_COMPLETE         -- TRUE exactly when that subset is empty

if(NOT DEFINED Q4_STAGE_B_ROOT)
  get_filename_component(Q4_STAGE_B_ROOT "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
endif()

# The three sources CMakeLists.txt's own header names as stage (b). ONE list: the existence
# test, the build's source list, the published definition and the refusal message all read it.
set(Q4_STAGE_B_REQUIRED_SOURCES
  impl/package.cpp
  impl/variant.cpp
  impl/load/bindings.cpp)

set(Q4_STAGE_B_MISSING "")
foreach(_q4_stage_b_source IN LISTS Q4_STAGE_B_REQUIRED_SOURCES)
  if(NOT EXISTS "${Q4_STAGE_B_ROOT}/${_q4_stage_b_source}")
    string(APPEND Q4_STAGE_B_MISSING "${_q4_stage_b_source} ")
  endif()
endforeach()

# ONE FLAG PER SOURCE, because the consumer needs the list as a C STRING and a `-D` whose value
# is a list cannot supply one (see package.h: the message that names the missing files would
# then not compile). These integers are what CMakeLists.txt publishes.
if(EXISTS "${Q4_STAGE_B_ROOT}/impl/package.cpp")
  set(Q4_STAGE_B_HAS_IMPL_PACKAGE 1)
else()
  set(Q4_STAGE_B_HAS_IMPL_PACKAGE 0)
endif()
if(EXISTS "${Q4_STAGE_B_ROOT}/impl/variant.cpp")
  set(Q4_STAGE_B_HAS_IMPL_VARIANT 1)
else()
  set(Q4_STAGE_B_HAS_IMPL_VARIANT 0)
endif()
if(EXISTS "${Q4_STAGE_B_ROOT}/impl/load/bindings.cpp")
  set(Q4_STAGE_B_HAS_IMPL_BINDINGS 1)
else()
  set(Q4_STAGE_B_HAS_IMPL_BINDINGS 0)
endif()

if(Q4_STAGE_B_MISSING STREQUAL "")
  set(Q4_STAGE_B_COMPLETE TRUE)
else()
  set(Q4_STAGE_B_COMPLETE FALSE)
endif()

if(CMAKE_SCRIPT_MODE_FILE)
  # `cmake -P` rehearsal: report the verdict and leave the caller's variables alone.
  message(STATUS "Q4_STAGE_B_ROOT=${Q4_STAGE_B_ROOT}")
  message(STATUS "Q4_STAGE_B_MISSING=${Q4_STAGE_B_MISSING}")
  message(STATUS "Q4_STAGE_B_HAS=P${Q4_STAGE_B_HAS_IMPL_PACKAGE}V${Q4_STAGE_B_HAS_IMPL_VARIANT}B${Q4_STAGE_B_HAS_IMPL_BINDINGS}")
  message(STATUS "Q4_STAGE_B_COMPLETE=${Q4_STAGE_B_COMPLETE}")
  return()
endif()

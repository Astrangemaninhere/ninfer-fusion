# src/targets/spark_x2_5_4b/CMakeLists.txt
#
# The first of the eight files tools/archkit/archkit_target.py:121-132 enumerates for a
# new target.  This is the BUILD half of this line's landq entry; the registration half
# lives in the file that already owns it on the live tree:
#
#   src/CMakeLists.txt:689-691 -- the self-activating guard cmakebatch landed
#   (dl/_orch/landq/cmakebatch/items.tsv item 0001).  It asks the filesystem the one
#   question that decides whether this directory is wired:
#       if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/targets/spark_x2_5_4b/CMakeLists.txt")
#   so the moment THIS file exists the subdirectory is added, and NO further edit to
#   src/CMakeLists.txt is needed or wanted.  Adding one anyway fails CONFIGURE with
#   "The binary directory ... is already used to build a source directory" -- reproduced
#   as arm RED-1 in dl/_orch/landq/sparktarget/arms.sh.
#
# WHY THIS FILE IS NOT A TARGET ALIAS.  Two things it deliberately does not do:
#
#   * it does not define a second compilation of its own sources.  Adding
#     `add_library(spark_x2_5_4b ...)` over the same TUs would compile each of them
#     twice and put two copies of every symbol in the link.  The precedent every sibling
#     target follows (src/targets/muse_glimmer_30b/CMakeLists.txt:1-9) is that the
#     target directory CONTRIBUTES its sources to the single compile owner
#     `ninfer_engine` and defines no library of its own.
#   * it does not add EXCLUDE_FROM_ALL.  EXCLUDE_FROM_ALL is what src/CMakeLists.txt's
#     two imported-target lines use for targets whose compile side is done but which
#     carry no engine-facing Instance adapter.  Spark is in a comparable registration
#     state today, but the honest place to record that is src/targets/registry.{h,cpp}
#     -- which has no row for this target yet and is NOT in this entry's scope -- and
#     not a CMake exclusion that would also hide errors in the three TUs.

# Three translation units, one per role.  Listed one per line on purpose: this project's
# CMakeLists carry no GLOB by design (src/CMakeLists.txt's own comment says so), because
# a new source is a build-boundary decision.
#   impl/package.cpp          the Package statics + the draft-identity static_assert
#   impl/load/bindings.cpp    artifact object plan -> BindingPlan -> RuntimeModelView
#   impl/variant.cpp          the execution leaves + the family macro instantiation
target_sources(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/variant.cpp)

# `impl` carries config.h / variant.h / load/bindings.h; `export` carries the identity
# header <ninfer/targets/spark_x2_5_4b/package.h>, which this target's own TUs and the
# future registry row both include by that path.
target_include_directories(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
  ${CMAKE_CURRENT_SOURCE_DIR}/export)

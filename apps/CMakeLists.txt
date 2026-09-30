add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
target_include_directories(ninfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_load_progress
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_load_progress)
if(UNIX)
  target_link_options(ninfer-serve PRIVATE -rdynamic)
endif()

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/digest.cpp
  perplexity/evaluation.cpp)
target_include_directories(ninfer-perplexity PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/perplexity)
target_link_libraries(ninfer-perplexity PRIVATE ninfer_engine)

# --- ninfer-cufree: THE FRONT DOOR THAT CAN START WITH NO CUDA STACK ---------------
#
# LANDED BY dl/amdland2 on top of dl/cufree's prepared patches (its P2 + P3), because the
# user-visible goal -- do not hand the operator a raw loader sentence -- is met only once the
# binary that CAN print the honest report is built by the tree's own build system.
#
# WHY THIS TARGET LINKS NOTHING. MEASURED (dl/cufree section 1): the three front doors above all
# link libcudart.so.13 and libcuda.so.1 DYNAMICALLY, and an ELF stays CUDA-free only when CUDA is
# nowhere on its link line. The 59 CUDA-free test binaries that keep -lcudart on the line are
# CUDA-free by a LINKER DEFAULT (--as-needed), which dl/cufree measured to be one flag deep
# (-Wl,--no-as-needed brings the NEEDED entry back). This target therefore names no library at
# all, and that property is the reason the file exists at all.
#
# NOT ADDED TO install(TARGETS ...), AND THAT IS A DECISION TAKEN HERE ON PURPOSE, NOT AN
# OVERSIGHT: the install list below is the deploy contract ("ONLY THE THREE EXECUTABLES ARE
# NAMED, AND THAT IS THE WHOLE SET"), and changing what `cmake --install` ships is the owner's
# call rather than a side effect of this landing. Until that call is made, the binary is reached
# with: cmake --build build --target ninfer-cufree
add_executable(ninfer-cufree cufree/main.cpp)
target_include_directories(ninfer-cufree PRIVATE
  ${PROJECT_SOURCE_DIR}/src)
# DELIBERATELY NO target_link_libraries() CALL HERE. Adding one is the edit that turns this
# binary back into one that cannot start on the machine it is for.

# --- Install rules (deployment) ----------------------------------------------
#
# MEASURED 2026-09-20 (dl/winstage2): before this block the tree had ZERO `install(...)`
# commands across all 13 tracked CMakeLists.txt files, and README.md:40-41 recorded the
# consequence in words -- "There is no install target or packaged binary distribution".
# These are the rules that make `cmake --install` the single deploy step.
#
# ONLY THE THREE EXECUTABLES ARE NAMED, AND THAT IS THE WHOLE SET. They are the only
# link-complete products in the tree: it builds no shared library and exports no headers,
# so an installed `ninfer_engine.a` would arrive without the transitive link line that
# makes it usable. A new app is added to this list on purpose, never by a glob.
#
# NO PATH IS WRITTEN DOWN. ${CMAKE_INSTALL_BINDIR} comes from GNUInstallDirs, which the
# root CMakeLists.txt includes ahead of this subdirectory, so -DCMAKE_INSTALL_PREFIX=...,
# `cmake --install --prefix ...` and DESTDIR=... all work and none of them is special-cased.
install(TARGETS ninfer ninfer-serve ninfer-perplexity
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT runtime)

# --- ninfer-hostpath: THE FRONT DOOR THAT OPENS AN ARTIFACT WITH NO CUDA STACK ------
#
# LANDED BY dl/hostsim (F854). ADDITIVE: one new executable target, no existing target's
# sources, link line or options are touched, and nothing is added to install(TARGETS ...).
#
# WHY IT EXISTS. The tree's two front doors each do half of what a machine with no usable GPU
# needs: `ninfer` adjudicates artifacts and cannot start there (MEASURED 2026-09-26: with
# CUDA_VISIBLE_DEVICES="" it throws at src/core/device.cu:52 -- "cudaGetDeviceCount failed:
# cudaErrorNoDevice" -- rc=1, stdout 0 bytes, and it does so even for an artifact path that
# does not exist, so none of its gates is reached); `ninfer-cufree` starts there and takes no
# artifact by its own documented decision. This target is the conjunction.
#
# WHY IT NAMES NO LIBRARY. Same discipline as ninfer-cufree above, and for the same measured
# reason: an ELF stays CUDA-free only when CUDA is nowhere on its link line. It calls nothing
# from ninfer_core, which is the target that PUBLIC-links CUDA::cudart
# (src/CMakeLists.txt:69).
#
# WHY THE TWO TUs ARE COMPILED HERE RATHER THAN LINKED FROM ninfer_artifact -- AND THIS IS THE
# FINDING, NOT A SHORTCUT. The engine's own artifact reader is already CUDA-free: neither
# artifact/reader.cpp nor artifact/storage_layouts.cpp includes a CUDA header, and MEASURED
# 2026-09-26 they compile and link under plain g++ 15.2.0 with no CUDA at all (the resulting
# binary's DT_NEEDED is libstdc++/libgcc_s/libc only). What is NOT CUDA-free is the tree's
# packaging of them: src/CMakeLists.txt:79 is `target_link_libraries(ninfer_artifact PUBLIC
# ninfer_core)`. So `ninfer_artifact` cannot be consumed on a host without a CUDA toolchain
# even though the reader inside it needs none. Until that PUBLIC is revisited -- which changes
# the link line of every consumer and is therefore the owner's call, not a side effect of this
# landing -- a host-only reader is built by naming the two sources directly. Because that
# choice is load-bearing it is visible HERE, next to its only consumer, instead of inside a
# helper function that another target could inherit by accident.
add_library(ninfer_artifact_host STATIC
  ${PROJECT_SOURCE_DIR}/src/artifact/reader.cpp
  ${PROJECT_SOURCE_DIR}/src/artifact/storage_layouts.cpp)
target_include_directories(ninfer_artifact_host PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
# DELIBERATELY NO target_link_libraries() ON ninfer_artifact_host: a link line here is the
# edit that welds the host half back onto CUDA, which is the defect this target exists around.

add_executable(ninfer-hostpath hostpath/main.cpp)
target_include_directories(ninfer-hostpath PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_compile_definitions(ninfer-hostpath PRIVATE
  NINFER_HOSTPATH_TREE_ROOT="${PROJECT_SOURCE_DIR}")
target_link_libraries(ninfer-hostpath PRIVATE ninfer_artifact_host)
# DELIBERATELY NO CUDA-BEARING LIBRARY HERE. Adding ninfer_core, ninfer_artifact or
# ninfer_engine to this line is the edit that makes this binary one that cannot start on the
# machine it is for.

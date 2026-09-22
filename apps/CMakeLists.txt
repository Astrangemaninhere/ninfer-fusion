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
if(UNIX)
  target_link_options(ninfer-serve PRIVATE -rdynamic)
endif()
if(UNIX)
  target_link_options(ninfer-serve PRIVATE -rdynamic)
endif()
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

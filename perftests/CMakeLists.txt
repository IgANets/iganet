########################################################################
# CMakeLists.txt
#
# Author: Matthias Moller
# Copyright (C) 2021-2025 by the IgANet authors
#
# This file is part of the IgANet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
########################################################################

# Set a default coefficient numeric types if not specified
if(NOT IGANET_PERFTESTS_COEFF_TYPE)
  set (IGANET_PERFTESTS_COEFF_TYPE "double" CACHE STRING
    "Coefficient type(float, double)" FORCE)
endif()
set_property(CACHE IGANET_PERFTESTS_COEFF_TYPE PROPERTY STRINGS
  "float" "double")

message("PerfTests options:")
message("IGANET_PERFTESTS_COEFF_TYPE........: ${IGANET_PERFTESTS_COEFF_TYPE}")
message("")

########################################################################
# Config file
########################################################################
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/perftest_config.hpp.in
  "${CMAKE_CURRENT_BINARY_DIR}/perftest_config.hpp" @ONLY)

########################################################################
# Include directories
########################################################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

########################################################################
# Define remote repository
########################################################################
include(FetchContent)

########################################################################
# GoogleTest
########################################################################
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
  )
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

########################################################################
# BSplineLib
########################################################################
FetchContent_Declare(
  bsplinelib
  URL https://github.com/tataratat/BSplineLib/archive/refs/heads/main.zip
  )
set(BSPLINELIB_SHARED ON CACHE BOOL "" FORCE)
set(BSPLINELIB_BUILD_PUGIXML OFF CACHE BOOL "" FORCE)
set(pugixml_DIR ${pugixml_BINARY_DIR})
FetchContent_MakeAvailable(googletest bsplinelib)
FetchContent_GetProperties(bsplinelib)
include_directories(${bsplinelib_SOURCE_DIR}/Sources)

########################################################################
# Collect perftest sources
########################################################################

add_custom_target(iganet_perftests)

file(GLOB PERFTESTS *.cxx)

foreach(perftest ${PERFTESTS})

  if (NOT IGANET_BUILD_CPUONLY AND CMAKE_CUDA_COMPILER)
    set_source_files_properties(${perftest} PROPERTIES LANGUAGE CUDA)
  endif()

  get_filename_component(name ${perftest} NAME_WE)
  add_executable(${name} ${perftest})
  add_dependencies(iganet_perftests ${name})

  if(IGANET_BUILD_PCH AND NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS "3.16.0")
    target_precompile_headers(${name} REUSE_FROM iganet_pch)
  endif()

  target_compile_options(${name} PUBLIC "-DIGANET_DATA_DIR=\"${PROJECT_SOURCE_DIR}/filedata/\"")

  # LibTorch
  target_link_libraries(${name} PUBLIC "${TORCH_LIBRARIES}")
  target_link_libraries(${name} PUBLIC pugixml)

  # Intel Extension for PyTorch
  if (NOT IGANET_BUILD_CPUONLY AND IPEX_FOUND)
    target_link_libraries(${name} PUBLIC ${TORCH_IPEX_LIBRARIES})
  endif()

  # BSplineLib
  target_link_libraries(${name} PUBLIC splines)

  # Google test
  target_link_libraries(${name} PUBLIC gtest_main)

  if (IGANET_WITH_GISMO)
    target_link_libraries(${name} PUBLIC gismo_static)
  endif()

  if (IGANET_WITH_MATPLOT)
    target_link_libraries(${name} PUBLIC Matplot++::matplot)
  endif()

  if (IGANET_WITH_MPI)
    target_link_libraries(${name} PUBLIC MPI::MPI_CXX)
  endif()

  if (IGANET_WITH_OPENMP)
    target_link_libraries(${name} PUBLIC OpenMP::OpenMP_CXX)
  endif()

  target_compile_definitions(${name} PUBLIC UNITTEST_DISABLE_SHORT_MACROS)
  add_test(${name} ${name})
endforeach()

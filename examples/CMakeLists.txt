########################################################################
# CMakeLists.txt
#
# Author: Matthias Moller
# Copyright (C) 2021-2023 by the IgANet authors
#
# This file is part of the IgANet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
########################################################################

file(GLOB EXAMPLES *.cxx)

foreach(example ${EXAMPLES})

  if (NOT IGANET_BUILD_CPUONLY AND CMAKE_CUDA_COMPILER)
    set_source_files_properties(${example} PROPERTIES LANGUAGE CUDA)
  endif()

  get_filename_component(name ${example} NAME_WE)
  add_executable(${name} ${example})

  if(IGANET_BUILD_PCH AND NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS "3.16.0")
    target_precompile_headers(${name} REUSE_FROM iganet_pch)
  endif()

  target_compile_options(${name} PUBLIC "-DIGANET_DATA_DIR=\"${PROJECT_SOURCE_DIR}/filedata/\"")

  # LibTorch
  target_link_libraries(${name} PUBLIC ${TORCH_LIBRARIES})
  target_link_libraries(${name} PUBLIC pugixml)

  # Intel Extension for PyTorch
  if (NOT IGANET_BUILD_CPUONLY AND IPEX_FOUND)
    target_link_libraries(${name} PUBLIC ${TORCH_IPEX_LIBRARIES})
  endif()

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
endforeach()

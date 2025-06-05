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

########################################################################
# Models
########################################################################

file(GLOB MODELS *.cxx *.cpp)
foreach(model ${MODELS})

  if (NOT IGANET_BUILD_CPUONLY AND CMAKE_CUDA_COMPILER)
      set_source_files_properties(${model} PROPERTIES LANGUAGE CUDA)
  endif()

  get_filename_component(name ${model} NAME_WE)
  add_library(${name} SHARED ${model})

  if(IGANET_BUILD_PCH AND NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS "3.16.0")
    target_precompile_headers(${name} REUSE_FROM iganet_pch)
  endif()

  target_link_libraries(${name} PUBLIC ${TORCH_LIBRARIES})
  target_link_libraries(${name} PUBLIC ${ZLIB_LIBRARIES})
  target_link_libraries(${name} PUBLIC ${CMAKE_DL_LIBS})
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

  if (IGANET_WITH_OPENMP)
    target_link_libraries(${name} PUBLIC OpenMP::OpenMP_CXX)
  endif()

  install(TARGETS ${name} DESTINATION models)
endforeach()

file(GLOB SOURCES *.hpp)
install(FILES ${SOURCES} DESTINATION include/iganet/webapp/models)

########################################################################
# Include directories
########################################################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

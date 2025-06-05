########################################################################
# CMakeLists.txt
#
# Author: Matthias Moller
# Copyright (C) 2021-2025 by the IgaNet authors
#
# This file is part of the IgaNet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
########################################################################

find_package( Matlab COMPONENTS MX_LIBRARY )

if (Matlab_FOUND)

  add_custom_target(iganet_mexs)

  file(GLOB MEXFILES *.cpp)

  foreach(mexfile ${MEXFILES})

    get_filename_component(mexname ${mexfile} NAME_WE)
    string(REGEX REPLACE "^mex" "" name "${mexname}")

    matlab_add_mex(NAME ${name} SRC ${mexfile})
    add_dependencies(iganet_mexs ${name})

    target_compile_options(${name} PUBLIC "-DIGANET_DATA_DIR=\"${PROJECT_SOURCE_DIR}/filedata/\"")

    # LibTorch
    target_link_libraries(${name} ${TORCH_LIBRARIES})
    target_link_libraries(${name} pugixml)

    # Intel Extension for PyTorch
    if (NOT IGANET_BUILD_CPUONLY AND IPEX_FOUND)
      target_link_libraries(${name} ${TORCH_IPEX_LIBRARIES})
    endif()

    if (IGANET_WITH_GISMO)
      target_link_libraries(${name} gismo_static)
    endif()

    if (IGANET_WITH_MATPLOT)
      target_link_libraries(${name} Matplot++::matplot)
    endif()

    if (IGANET_WITH_MPI)
      target_link_libraries(${name} MPI::MPI_CXX)
    endif()

    if (IGANET_WITH_OPENMP)
      target_link_libraries(${name} OpenMP::OpenMP_CXX)
    endif()

  endforeach()
endif()

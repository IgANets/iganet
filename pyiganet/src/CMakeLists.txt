########################################################################
# CMakeLists.txt
#
# Author: Matthias Moller
# Copyright (C) 2021-2023 by the IgaNet authors
#
# This file is part of the IgaNet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# CMakeLists.txt accepts the following command line parameters
#
# PYIGANET_COEFF_TYPE
#
########################################################################

include(boost_preprocessor)
include(pybind11)

########################################################################
# Options
########################################################################

if(NOT PYIGANET_COEFF_TYPE)
  set(PYIGANET_COEFF_TYPE double CACHE STRING
    "Coefficient type(float, double)" FORCE)
endif()
set_property(CACHE PYIGANET_COEFF_TYPE PROPERTY STRINGS "float" "double")

message("Pyiganet options:")
message("PYIGANET_COEFF_TYPE................: ${PYIGANET_COEFF_TYPE}")
message("")

########################################################################
# Auto-generate source files
########################################################################

# Config
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pyconfig.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/pyconfig.hpp @ONLY)

# Create list of PyBind11 init subroutines
list(APPEND init_core "init_options")

# Explicit instantiation of templated blocktensor classes
foreach(ROWS RANGE 1 4)
  foreach(COLS RANGE 1 4)

    set(SUFFIX "_${ROWS}_${COLS}")
    list(APPEND init_core "init_BlockTensor${SUFFIX}")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pyblocktensor.cxx.in
      ${CMAKE_CURRENT_BINARY_DIR}/pyblocktensor${SUFFIX}.cxx @ONLY)
  endforeach()
endforeach()

# Create list of PyBind11 init subroutines
list(APPEND init_splines "init_bspline")

# Explicit instantiation of templated spline classes
foreach(GEODIM RANGE 1 4)
  foreach(DEGREES IN ITEMS
      "1" "1,1" "1,1,1" "1,1,1,1")
#      "2" "2,2" "2,2,2" "2,2,2,2"
#      "3" "3,3" "3,3,3" "3,3,3,3"
#      "4" "4,4" "4,4,4" "4,4,4,4"
#      "5" "5,5" "5,5,5" "5,5,5,5")

    set(SUFFIX "${GEODIM}d_${DEGREES}")
    string(REPLACE "," "_" SUFFIX ${SUFFIX})
    list(APPEND init_splines "init_UniformBSpline${SUFFIX}")

    string(REPLACE "," ";" PARDIM ${DEGREES})
    list(LENGTH PARDIM PARDIM)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pybspline.cxx.in
      ${CMAKE_CURRENT_BINARY_DIR}/pybspline${SUFFIX}.cxx @ONLY)
  endforeach()
endforeach()

# Create main PyIgANet source file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/pyiganet.cxx
  "/**
   @file pyiganet/pyiganet.cxx

   @brief PyIgANet

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <pybind11/pybind11.h>

#include <pyconfig.hpp>

namespace py = pybind11;\n\n")

foreach(init IN LISTS init_core init_splines)
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/pyiganet.cxx
    "void ${init}(py::module_ &);\n")
endforeach()

file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/pyiganet.cxx
  "\n\n/**
   @brief Creates IgANet Python module
*/

PYBIND11_MODULE(pyiganet_core, m) {\n

m.attr(\"__name__\") = \"pyiganet_core\";
m.attr(\"__version__\") = \"${IGANET_VERSION}\";
m.doc() = \"IgANet (Physics-informed isogeometric analysis neural network)\";

py::module core = m.def_submodule(\"core\");

core.attr(\"__name__\") = \"pyiganet_core.core\";
core.attr(\"__version__\") = \"${IGANET_VERSION}\";
core.doc() = \"IgANet (Physics-informed isogeometric analysis neural network): Core module\";

")

foreach(init IN LISTS init_core)
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/pyiganet.cxx
    "${init}(core);\n")
endforeach()

file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/pyiganet.cxx
  "\n

py::module splines = m.def_submodule(\"splines\");

splines.attr(\"__name__\") = \"pyiganet_core.splines\";
splines.attr(\"__version__\") = \"${IGANET_VERSION}\";
splines.doc() = \"IgANet (Physics-informed isogeometric analysis neural network): Spline module\";

")

foreach(init IN LISTS init_splines)
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/pyiganet.cxx
    "${init}(splines);\n")
endforeach()

file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/pyiganet.cxx
    "}\n")

# Include directories
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx ${CMAKE_CURRENT_BINARY_DIR}/*.cxx)

pybind11_add_module(pyiganet_core MODULE ${SRCS})

target_include_directories(pyiganet_core PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(pyiganet_core PUBLIC pugixml)

# Intel Extension for PyTorch
if (IPEX_FOUND)
  target_link_libraries(${name} PUBLIC ${TORCH_IPEX_LIBRARIES})
endif()

if (IGANET_WITH_GISMO)
  target_link_libraries(pyiganet_core PUBLIC gismo)
endif()

if (IGANET_WITH_MATPLOT)
  target_link_libraries(pyiganet_core PUBLIC Matplot++::matplot)
endif()

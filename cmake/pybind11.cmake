########################################################################
# pybind11.cmake
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

########################################################################
# PyBind11
########################################################################

include(FetchContent)
FetchContent_Declare(pybind11
  URL https://github.com/pybind/pybind11/archive/refs/heads/master.zip
  )
FetchContent_MakeAvailable(pybind11)

########################################################################
# PyBind11_json
########################################################################

FetchContent_Declare(pybind11_json
  URL https://github.com/pybind/pybind11_json/archive/refs/heads/master.zip
  FIND_PACKAGE_ARGS
  )
FetchContent_Populate(pybind11_json)
FetchContent_GetProperties(pybind11_json)
include_directories(${pybind11_json_SOURCE_DIR}/include)

########################################################################
# popl.cmake
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
# popl
########################################################################

include(FetchContent)
FetchContent_Declare(popl
  URL https://github.com/badaix/popl/archive/refs/tags/v1.3.0.zip
  )

set(BUILD_EXAMPLE 0 CACHE INTERNAL "")
set(BUILD_TESTS   0 CACHE INTERNAL "")
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE INTERNAL "")
FetchContent_MakeAvailable(popl)
FetchContent_GetProperties(popl)
include_directories(${popl_SOURCE_DIR}/include)

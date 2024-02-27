########################################################################
# boost_preprocessor.cmake
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
########################################################################

########################################################################
# Boost.Preprocessor
########################################################################

include(FetchContent)
FetchContent_Declare(boost_preprocessor
  URL https://github.com/boostorg/preprocessor/archive/refs/tags/boost-1.84.0.zip
  )

FetchContent_MakeAvailable(boost_preprocessor)
FetchContent_GetProperties(boost_preprocessor)
include_directories(${boost_preprocessor_SOURCE_DIR}/include)

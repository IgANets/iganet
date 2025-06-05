########################################################################
# gismo.cmake
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
# G+Smo
########################################################################

include(FetchContent)
FetchContent_Declare(gismo
  URL https://github.com/gismo/gismo/archive/refs/heads/stable.zip
)

set(BUILD_TESTING        OFF CACHE INTERNAL "")
set(GISMO_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(GISMO_OPTIONAL       "gsHLBFGS;gsKLShell;gsElasticity" CACHE INTERNAL "")
set(GISMO_SHORT_TYPE     "int" CACHE INTERNAL "")
set(GISMO_WITH_OPENMP    ${IGANET_WITH_OPENMP} CACHE INTERNAL "")
set(GISMO_WITH_MPI       ${IGANET_WITH_MPI} CACHE INTERNAL "")
set(NOSNIPPETS           ON  CACHE INTERNAL "")
FetchContent_MakeAvailable(gismo)
include_directories(${GISMO_INCLUDE_DIRS})

########################################################################
# pugixml.cmake
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
# pugixml
########################################################################

include(FetchContent)
FetchContent_Declare(pugixml
  URL https://github.com/zeux/pugixml/releases/download/v1.15/pugixml-1.15.zip
  )

FetchContent_MakeAvailable(pugixml)
FetchContent_GetProperties(pugixml)
include_directories(${pugixml_SOURCE_DIR}/src)

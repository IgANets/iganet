########################################################################
# websockets.cmake
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
# uSockets
########################################################################

include(FetchContent)
FetchContent_Declare(usockets
  URL https://github.com/uNetworking/uSockets/archive/refs/tags/v0.8.8.zip
  )

FetchContent_MakeAvailable(usockets)
FetchContent_GetProperties(usockets)
include_directories(${usockets_SOURCE_DIR}/src)

file(GLOB usockets_SOURCE_FILES
  ${usockets_SOURCE_DIR}/src/*.c
  ${usockets_SOURCE_DIR}/src/crypto/*.c*
  ${usockets_SOURCE_DIR}/src/eventing/*.c)
add_library(usockets ${usockets_SOURCE_FILES})

# Compile with OpenSSL support
if (IGANET_WITH_OPENSSL)
  find_package(OpenSSL REQUIRED)
  target_compile_definitions(usockets PUBLIC LIBUS_USE_OPENSSL)
  target_include_directories(usockets PUBLIC ${OPENSSL_INCLUDE_DIR})
  target_link_libraries(usockets ${OPENSSL_LIBRARIES})
else()
  target_compile_definitions(usockets PUBLIC LIBUS_NO_SSL)
endif()

# Compile with LibUV support
if (IGANET_WITH_LIBUV)
  find_package(LibUV REQUIRED)
  target_compile_definitions(usockets PUBLIC LIBUS_USE_LIBUV)
  target_include_directories(usockets ${LibUV_INCLUDE_DIR})
  target_link_libraries(usockets ${LibUV_LIBRARIES})
endif()

########################################################################
# uWebSockets
########################################################################

include(FetchContent)
FetchContent_Declare(uwebsockets
  URL https://github.com/uNetworking/uWebSockets/archive/refs/tags/v20.60.0.zip
  #PATCH_COMMAND     patch -p1 -N -d ${PROJECT_BINARY_DIR}/_deps/uwebsockets-src < ${PROJECT_SOURCE_DIR}/cmake/uwebsockets.patch
  )

FetchContent_MakeAvailable(uwebsockets)
FetchContent_GetProperties(uwebsockets)
include_directories(${uwebsockets_SOURCE_DIR}/src)

########################################################################
# CMakeLists.txt
#
# Author: Matthias Moller
# Copyright (C) 2021-2024 by the IgANet authors
#
# This file is part of the IgANet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# CMakeLists.txt accepts the following command line parameters
#
# IGANET_BUILD_DOCS
# IGANET_BUILD_EXAMPLES
# IGANET_BUILD_MEX
# IGANET_BUILD_PCH
# IGANET_BUILD_PERFTESTS
# IGANET_BUILD_PYTHON
# IGANET_BUILD_UNITTESTS
# IGANET_BUILD_WEBAPPS
#
# IGANET_WITH_GISMO
# IGANET_WITH_MATPLOT
# IGANET_WITH_MPI
# IGANET_WITH_OPENMP
#
########################################################################

########################################################################
# IgANet version number
########################################################################
set(iganet_VERSION_MAJOR 24) #year
set(iganet_VERSION_MINOR 04) #month
set(iganet_VERSION_PATCH 0 ) #patch
set(iganet_VERSION
  "${iganet_VERSION_MAJOR}.${iganet_VERSION_MINOR}.${iganet_VERSION_PATCH}" CACHE INTERNAL "iganet version number")
set(IGANET_VERSION "${iganet_VERSION}" CACHE INTERNAL "IgANet version number")

########################################################################
# Force CMake version 3.24 or above
########################################################################
cmake_minimum_required (VERSION 3.24)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

########################################################################
# Set RPATH on MacOSX
########################################################################
if (APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  SET(CMAKE_SKIP_BUILD_RPATH FALSE)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  IF ("${isSystemDir}" STREQUAL "-1")
    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  ENDIF ()
endif()

########################################################################
# Use solution folders for Visual Studio
########################################################################
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # Set default build type to RelWithDebInfo
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Type of build (None Debug Release RelWithDebInfo MinSizeRel)" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "RelWithDebInfo" "MinSizeRel")
endif()

########################################################################
# We do not support in-source build
########################################################################
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds are not permitted.\nPlease make a separate folder for building, otherwise type \nmake\nthat will create a ./build folder and will compile in that folder. Cmake has created garbage files/dirs (you may manually remove them):\nCMakeCache.txt CMakeFiles")
endif()

########################################################################
#
# Project: IgANet
#
########################################################################
project(IgANet)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.24)
  cmake_policy(SET CMP0135 NEW)
endif()

########################################################################
# Append path to additional modules
########################################################################
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

########################################################################
# Option list
########################################################################
option(IGANET_BUILD_CPUONLY     "Build CPU-only code"       OFF)
option(IGANET_BUILD_DOCS        "Build documentation"       OFF)
option(IGANET_BUILD_EXAMPLES    "Build examples"            ON )
option(IGANET_BUILD_MEX         "Build Matlab bindings"     OFF)
option(IGANET_BUILD_PCH         "Build precompiled headers" ON )
option(IGANET_BUILD_PERFTESTS   "Build perftests"           OFF)
option(IGANET_BUILD_PYTHON      "Build Python bindings"     OFF)
option(IGANET_BUILD_UNITTESTS   "Build unittests"           OFF)
option(IGANET_BUILD_WEBAPPS     "Build WebApps"             OFF)
option(IGANET_WITH_GISMO        "With G+Smo support"        OFF)
option(IGANET_WITH_MATPLOT      "With Matplot support"      OFF)
option(IGANET_WITH_MPI          "With MPI support"          OFF)
option(IGANET_WITH_OPENMP       "With OpenMP support"       ON )

########################################################################
# Prerequisites
########################################################################
find_package(Torch REQUIRED)
list(APPEND CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})

if(DEFINED HIP_COMPILER AND IGANET_BUILD_PCH)
  message(WARNING "HIP compiler does not (yet) support precompiled headers")
  set(IGANET_BUILD_PCH OFF CACHE BOOL "Build precompiled headers" FORCE)
endif()

find_package(IPEX)

if(NOT WIN32)
find_package(ZLIB REQUIRED)
endif()

include(boost_preprocessor)
include(nlohmann_json)
include(pugixml)

if (IGANET_WITH_GISMO)
  include(gismo)
endif()

if (IGANET_WITH_MATPLOT)
  include(Matplot++)
endif()

########################################################################
# Enable MPI support
########################################################################
if (IGANET_WITH_MPI)
  find_package(MPI REQUIRED)
endif()

########################################################################
# Enable OpenMP support
########################################################################
if (IGANET_WITH_OPENMP)
  # Apple explicitly disabled OpenMP support in their compilers that
  # are shipped with XCode but there is an easy workaround as
  # described at https://mac.r-project.org/openmp/
  if (CMAKE_C_COMPILER_ID STREQUAL "AppleClang" OR
      CMAKE_C_COMPILER_ID STREQUAL "Clang" AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" OR
      CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
      CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    find_path(OpenMP_C_INCLUDE_DIR
      NAMES "omp.h"
      PATHS /usr/local /usr/local/opt/libomp /opt /opt/local /opt/homebrew/opt/libomp
      PATH_SUFFIXES include)
    find_path(OpenMP_CXX_INCLUDE_DIR
      NAMES "omp.h"
      PATHS /usr/local /usr/local/opt/libomp /opt /opt/local /opt/homebrew/opt/libomp
      PATH_SUFFIXES include)
    find_library(OpenMP_libomp_LIBRARY
      NAMES "omp"
      PATHS /usr/local /usr/local/opt/libomp /opt /opt/local /opt/homebrew/opt/libomp
      PATH_SUFFIXES lib)
  endif()
  find_package(OpenMP REQUIRED)
endif()

########################################################################
# Doxygen
########################################################################
if(IGANET_BUILD_DOCS)
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIR ${PROJECT_BINARY_DIR}/docs/doxygen)
    set(DOXYGEN_TAGFILE ${PROJECT_BINARY_DIR}/docs/doxygen/iganet-doxygen.tag.xml)
    configure_file(${PROJECT_SOURCE_DIR}/docs/Doxyfile.in ${PROJECT_BINARY_DIR}/docs/doxygen/Doxyfile @ONLY)
    add_custom_target(Doxygen
      COMMAND
      ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/docs/doxygen/Doxyfile
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      COMMENT "Generating documentation with Doxygen"
      )

  else()
    add_custom_target(Doxygen
      ${CMAKE_COMMAND} -E cmake_echo_color --cyan "Please install doxygen, and run cmake . \\&\\& make Doxygen to generate the Doxygen documentation."
      )
  endif()
endif()

########################################################################
# Sphinx
########################################################################
if(IGANET_BUILD_DOCS)
  find_package(Sphinx)
  if(Sphinx_FOUND)
    set(SPHINX_SOURCE ${PROJECT_SOURCE_DIR}/docs)
    set(SPHINX_BUILD ${PROJECT_BINARY_DIR}/docs/sphinx)
    set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)
    add_custom_target(Sphinx
      COMMAND
      ${SPHINX_EXECUTABLE} -b html
      -Dbreathe_projects.LibKet=${PROJECT_BINARY_DIR}/docs/doxygen/xml
      ${SPHINX_SOURCE} ${SPHINX_BUILD}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      COMMENT "Generating documentation with Sphinx"
      )
    add_dependencies(Sphinx Doxygen)
  else()
    add_custom_target(Sphinx
      ${CMAKE_COMMAND} -E cmake_echo_color --cyan "Please install sphinx, and run cmake . \\&\\& make Sphinx to generate the Sphinx documentation."
      )
  endif()
endif()

########################################################################
# Documentation
########################################################################
if(IGANET_BUILD_DOCS)
  add_custom_target(docs
    DEPENDS Doxygen
    DEPENDS Sphinx)
endif()

########################################################################
# Summary
########################################################################
message("Configuration:")
message("Build type.........................: ${CMAKE_BUILD_TYPE}")
message("Build shared libraries.............: ${BUILD_SHARED_LIBS}")
message("Build directory....................: ${PROJECT_BINARY_DIR}")
message("Source directory...................: ${PROJECT_SOURCE_DIR}")
message("Install directory..................: ${CMAKE_INSTALL_PREFIX}")

message("")
message("AR command.........................: ${CMAKE_AR}")
message("RANLIB command.....................: ${CMAKE_RANLIB}")

if(CMAKE_C_COMPILER)
  message("")
  message("C compiler.........................: ${CMAKE_C_COMPILER}")
  message("C compiler flags ..................: ${CMAKE_C_FLAGS}")
  message("C compiler flags (debug)...........: ${CMAKE_C_FLAGS_DEBUG}")
  message("C compiler flags (release).........: ${CMAKE_C_FLAGS_RELEASE}")
  message("C compiler flags (release+debug)...: ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
endif()

if(CMAKE_CXX_COMPILER)
  message("")
  message("CXX compiler.......................: ${CMAKE_CXX_COMPILER}")
  message("CXX standard.......................: ${CMAKE_CXX_STANDARD}")
  message("CXX compiler flags ................: ${CMAKE_CXX_FLAGS}")
  message("CXX compiler flags (debug).........: ${CMAKE_CXX_FLAGS_DEBUG}")
  message("CXX compiler flags (release).......: ${CMAKE_CXX_FLAGS_RELEASE}")
  message("CXX compiler flags (release+debug).: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

message("")
message("EXE linker flags...................: ${CMAKE_EXE_LINKER_FLAGS}")
message("EXE linker flags (debug)...........: ${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
message("EXE linker flags (release).........: ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
message("EXE linker flags (release+debug)...: ${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO}")

if(DOXYGEN_FOUND)
  message("")
  message("Doxygen............................: ${DOXYGEN_EXECUTABLE}")

  if(DOXYGEN_DOT_FOUND)
    message("Dot................................: ${DOXYGEN_DOT_EXECUTABLE}")
  endif()
endif()

if(SPHINX_FOUND)
  message("")
  message("Sphinx.............................: ${SPHINX_EXECUTABLE}")
endif()

message("")
message("Options:")
message("IGANET_BUILD_CPUONLY...............: ${IGANET_BUILD_CPUONLY}")
message("IGANET_BUILD_DOCS..................: ${IGANET_BUILD_DOCS}")
message("IGANET_BUILD_EXAMPLES..............: ${IGANET_BUILD_EXAMPLES}")
message("IGANET_BUILD_MEX...................: ${IGANET_BUILD_MEX}")
message("IGANET_BUILD_PCH...................: ${IGANET_BUILD_PCH}")
message("IGANET_BUILD_PERFTESTS.............: ${IGANET_BUILD_PERFTESTS}")
message("IGANET_BUILD_PYTHON................: ${IGANET_BUILD_PYTHON}")
message("IGANET_BUILD_UNITTESTS.............: ${IGANET_BUILD_UNITTESTS}")
message("IGANET_BUILD_WEBAPPS...............: ${IGANET_BUILD_WEBAPPS}")

message("")
message("Features:")
message("IGANET_WITH_GISMO..................: ${IGANET_WITH_GISMO}")
message("IGANET_WITH_MATPLOT................: ${IGANET_WITH_MATPLOT}")
message("IGANET_WITH_MPI....................: ${IGANET_WITH_MPI}")
message("IGANET_WITH_OPENMP.................: ${IGANET_WITH_OPENMP}")
message("")

########################################################################
# Add kernel directory
########################################################################
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/include)

########################################################################
# Add documentation (if enabled)
########################################################################
if (IGANET_BUILD_DOCS)
  add_subdirectory(${PROJECT_SOURCE_DIR}/docs)
endif()

########################################################################
# Add examples (if enabled)
########################################################################
if (IGANET_BUILD_EXAMPLES)
  add_subdirectory(${PROJECT_SOURCE_DIR}/examples)
endif()

########################################################################
# Add Matlab bindings (if enabled)
########################################################################
if (IGANET_BUILD_MEX)
  add_subdirectory(${PROJECT_SOURCE_DIR}/mex/src)
endif()

########################################################################
# Add perftests (if enabled)
########################################################################
if (IGANET_BUILD_PERFTESTS)
  enable_testing()
  add_subdirectory(${PROJECT_SOURCE_DIR}/perftests)
endif()

########################################################################
# Add Python bindings (if enabled)
########################################################################
if (IGANET_BUILD_PYTHON)
  add_subdirectory(${PROJECT_SOURCE_DIR}/python/src)
endif()

########################################################################
# Add unittests (if enabled)
########################################################################
if (IGANET_BUILD_UNITTESTS)
  enable_testing()
  add_subdirectory(${PROJECT_SOURCE_DIR}/unittests)
endif()

########################################################################
# Add WebApps (if enabled)
########################################################################
if (IGANET_BUILD_WEBAPPS)
  add_subdirectory(${PROJECT_SOURCE_DIR}/webapps)
endif()

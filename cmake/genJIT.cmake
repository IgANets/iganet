########################################################################
# genJIT.cmake
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

#
# CMake function: generate just-in-time compiler configuration
#
# Remark: The source files must be given with relative paths
#
function(genJITCompiler SOURCE_FILES SOURCE_TARGET)

  #
  # Set JIT compiler command
  #
  set(JIT_CXX_COMPILER ${CMAKE_CXX_COMPILER})

  # ====================================================================

  #
  # Set JIT compiler input/output flag
  #
  if (MSVC)
    set(JIT_CXX_INCLUDE_FLAG       "/I")
    set(JIT_CXX_LINKER_FLAG        "/l")
    set(JIT_CXX_LINKER_SEARCH_FLAG "/L")
    set(JIT_CXX_OUTPUT_FLAG        "/Fo")
  else()
    set(JIT_CXX_INCLUDE_FLAG       "-I")
    set(JIT_CXX_LINKER_FLAG        "-l")
    set(JIT_CXX_LINKER_SEARCH_FLAG "-L")
    set(JIT_CXX_OUTPUT_FLAG        "-o ")
  endif()

  # ====================================================================

  # Get build-type as upper-case string
  string(TOUPPER ${CMAKE_BUILD_TYPE} JIT_BUILD_TYPE)

  # Set JIT compiler flags (build-type dependent)
  set(JIT_CXX_FLAGS ${CMAKE_CXX_FLAGS_${JIT_BUILD_TYPE}})

  # Set additional global compile definitions
  get_directory_property(JIT_COMPILE_DEFINITIONS COMPILE_DEFINITIONS)
  if (JIT_COMPILE_DEFINITIONS)
    foreach (flag ${JIT_COMPILE_DEFINITIONS})
      set (JIT_CXX_FLAGS "${JIT_CXX_FLAGS} -D${flag}")
    endforeach()
  endif()

  # Set additional global compile options
  get_directory_property(JIT_COMPILE_OPTIONS COMPILE_OPTIONS)
  if (JIT_COMPILE_OPTIONS)
    foreach (flag ${JIT_COMPILE_OPTIONS})
      set (JIT_CXX_FLAGS "${JIT_CXX_FLAGS} -D${flag}")
    endforeach()
  endif()

  # Set additional target-specific compile definitions and options (if available)
  if (TARGET ${SOURCE_TARGET})
    get_target_property(JIT_COMPILE_DEFINITIONS ${SOURCE_TARGET} COMPILE_DEFINITIONS)
    if (JIT_COMPILE_DEFINITIONS)
      foreach (flag ${JIT_COMPILE_DEFINITIONS})
        set (JIT_CXX_FLAGS "${JIT_CXX_FLAGS} -D${flag}")
      endforeach()
    endif()

    get_target_property(JIT_COMPILE_OPTIONS ${SOURCE_TARGET} COMPILE_OPTIONS)
    if (JIT_COMPILE_OPTIONS)
      foreach (flag ${JIT_COMPILE_OPTIONS})
        set (JIT_CXX_FLAGS "${JIT_CXX_FLAGS} ${flag}")
      endforeach()
    endif()
  endif()

  # Set Torch-specific compile flags
  if (TORCH_CXX_FLAGS)
    foreach (flag ${TORCH_CXX_FLAGS})
      set (JIT_CXX_FLAGS "${JIT_CXX_FLAGS} ${flag}")
    endforeach()
  endif()

  # Set OpenMP-specific compiler flags
  if (OpenMP_CXX_FLAGS)
    foreach (flag ${OpenMP_CXX_FLAGS})
      set (JIT_CXX_FLAGS "${JIT_CXX_FLAGS} ${flag}")
    endforeach()
  endif()

  # ====================================================================

  # Set SYSROOT on MacOS
  if (APPLE)
    set(JIT_CXX_FLAGS "${JIT_CXX_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
  endif()

  # ====================================================================

  # Create a set of shared library variable specific to C++
  # For 90% of the systems, these are the same flags as the C versions
  # so if these are not set just copy the flags from the C version
  if(NOT DEFINED CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS)
    set(JIT_CXX_FLAGS "${JIT_CXX_FLAGS} ${CMAKE_CXX_FLAGS} ${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS}")
  else()
    set(JIT_CXX_FLAGS "${JIT_CXX_FLAGS} ${CMAKE_CXX_FLAGS} ${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS}")
  endif()

  # Add C++ standard and PIC (position independent code)
  if(NOT DEFINED CMAKE_CXX_EXTENSIONS OR NOT CMAKE_CXX_EXTENSIONS)
    set(JIT_CXX_FLAGS "${JIT_CXX_FLAGS} ${CMAKE_CXX${CMAKE_CXX_STANDARD}_STANDARD_COMPILE_OPTION} ${CMAKE_CXX_COMPILE_OPTIONS_PIC}")
  else()
    set(JIT_CXX_FLAGS "${JIT_CXX_FLAGS} ${CMAKE_CXX${CMAKE_CXX_STANDARD}_EXTENSION_COMPILE_OPTION} ${CMAKE_CXX_COMPILE_OPTIONS_PIC}")
  endif()

  # Fix visibility
  string(REPLACE "-fvisibility=hidden"         "" JIT_CXX_FLAGS ${JIT_CXX_FLAGS})
  string(REPLACE "-fvisibility-inlines-hidden" "" JIT_CXX_FLAGS ${JIT_CXX_FLAGS})

  # ====================================================================

  # Generate list of global include directories
  get_property(IGANET_INCLUDE_DIRECTORIES DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
  if(IGANET_INCLUDE_DIRECTORIES)
    string(REPLACE ";" " ${JIT_CXX_INCLUDE_FLAG}"
      JIT_INCLUDE_DIRECTORIES
      "${JIT_CXX_INCLUDE_FLAG}${IGANET_INCLUDE_DIRECTORIES}")
  endif()

  # Generate list of target-specific include directories (if available)
  if (TARGET ${SOURCE_TARGET})
    get_target_property(IGANET_INCLUDE_DIRECTORIES ${SOURCE_TARGET} INCLUDE_DIRECTORIES)
    if (IGANET_INCLUDE_DIRECTORIES)
      foreach (dir ${IGANET_INCLUDE_DIRECTORIES})
        set (JIT_INCLUDE_DIRECTORIES
          "${JIT_INCLUDE_DIRECTORIES} ${JIT_CXX_INCLUDE_FLAG}${dir}")
      endforeach()
    endif()
  endif()

  # Add Torch-specific include directories
  if (TORCH_INCLUDE_DIRS)
    foreach (dir ${TORCH_INCLUDE_DIRS})
      set (JIT_INCLUDE_DIRECTORIES
        "${JIT_INCLUDE_DIRECTORIES} ${JIT_CXX_INCLUDE_FLAG}${dir}")
    endforeach()
  endif()

  # ====================================================================

  # Generate list of global external libraries
  get_property(IGANET_LINK_DIRECTORIES DIRECTORY PROPERTY LINK_DIRECTORIES)
  if(IGANET_LINK_DIRECTORIES)
    string(REPLACE ";" " ${JIT_CXX_LINKER_SEARCH_FLAG}"
      JIT_LIBRARIES
      "${JIT_CXX_LINKER_SEARCH_FLAG}${IGANET_LINK_DIRECTORIES}")
  endif()

  # Generate list of target-specific external libraries
  if (TARGET ${SOURCE_TARGET})
    get_target_property(IGANET_LINK_LIBRARIES ${SOURCE_TARGET} LINK_DIRECTORIES)
    if (IGANET_LINK_LIBRARIES)
      foreach (lib ${IGANET_LINK_LIBRARIES})
        set (JIT_LIBRARIES
          "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${lib}")
      endforeach()
    endif()
  endif()

  # Generate list of target-specific external libraries
  if (TARGET ${SOURCE_TARGET})
    get_target_property(IGANET_LINK_LIBRARIES ${SOURCE_TARGET} LINK_LIBRARIES)
    if (IGANET_LINK_LIBRARIES)

      # Generate include and link directories
      foreach (lib ${IGANET_LINK_LIBRARIES})

        if(lib STREQUAL "gismo_static")
          set(JIT_LIBRARIES
            "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${PROJECT_BINARY_DIR}/lib")

          list(APPEND LIBS gismo)

        elseif(lib STREQUAL "pugixml")
          set(JIT_LIBRARIES
            "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${pugixml_BINARY_DIR}")

          list(APPEND LIBS pugixml)

        elseif (lib STREQUAL "torch")
          if (WIN32)
            set(JIT_LIBRARIES
              "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${Torch_DIR}\\..\\..\\..\\lib")
          else()
            set(JIT_LIBRARIES
              "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${Torch_DIR}/../../../lib")
          endif()

          list(APPEND LIBS torch)

        elseif(lib STREQUAL "torch_library")
          if (Torch_CUDA_FOUND)

            if (WIN32)
              set(JIT_LIBRARIES
                "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${Torch_DIR}\\..\\..\\..\\lib")
            else()
              set(JIT_LIBRARIES
                "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${Torch_DIR}/../../../lib")
            endif()

            list(APPEND LIBS torch_cuda)

          else()

            if (WIN32)
              set(JIT_LIBRARIES
                "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${Torch_DIR}\\..\\..\\..\\lib")
            else()
              set(JIT_LIBRARIES
                "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${Torch_DIR}/../../../lib")
            endif()

            list(APPEND LIBS torch_cpu)
          endif()

        elseif(lib STREQUAL "Matplot++::matplot")
          set(JIT_LIBRARIES
            "${JIT_LIBRARIES} ${JIT_CXX_LINKER_SEARCH_FLAG}${matplotplusplus_BINARY_DIR}/source/matplot")

          list(APPEND LIBS matplot)

        elseif(lib STREQUAL "OpenMP::OpenMP_CXX")

          if (OpenMP_CXX_INCLUDE_DIR)
            set(JIT_LIBRARIES
              "${JIT_LIBRARIES} ${JIT_CXX_INCLUDE_FLAG}${OpenMP_CXX_INCLUDE_DIR}")
          endif()

          foreach (libname ${OpenMP_CXX_LIB_NAMES})
            list(APPEND LIBS ${OpenMP_${libname}_LIBRARY})
          endforeach()

        else()

          list(APPEND LIBS ${lib})

        endif()

      endforeach()

      # Generate linking directives
      foreach(lib ${LIBS})

        if (IS_ABSOLUTE ${lib})

          if (lib STREQUAL "pugixml" AND UNIX AND NOT APPLE)
            set(JIT_LIBRARIES
              "${JIT_LIBRARIES} -Wl,--whole-archive ${lib} -Wl,--no-whole-archive")
          else()
            set(JIT_LIBRARIES
              "${JIT_LIBRARIES} ${lib}")
          endif()

        else()

          if (lib STREQUAL "pugixml" AND UNIX AND NOT APPLE)
            set(JIT_LIBRARIES
              "${JIT_LIBRARIES} -Wl,--whole-archive ${JIT_CXX_LINKER_FLAG}${lib} -Wl,--no-whole-archive")
          else()
            set(JIT_LIBRARIES
              "${JIT_LIBRARIES} ${JIT_CXX_LINKER_FLAG}${lib}")
          endif()
        endif()

      endforeach()
    endif()
  endif()

  # ====================================================================

  # Generate source files
  foreach (input_file ${SOURCE_FILES})
    get_filename_component(output_file ${input_file} NAME_WLE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${input_file}
      ${CMAKE_CURRENT_BINARY_DIR}/${output_file} @ONLY)
  endforeach()

endfunction()

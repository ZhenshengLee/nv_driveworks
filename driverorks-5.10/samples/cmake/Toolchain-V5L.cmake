################################################################################
#
# Notice
# ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
# NVIDIA MAKES NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR
# OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED
# WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
# PURPOSE.
#
# NVIDIA CORPORATION & AFFILIATES assumes no responsibility for the consequences
# of use of such information or for any infringement of patents or other rights
# of third parties that may result from its use. No license is granted by
# implication or otherwise under any patent or patent rights of NVIDIA
# CORPORATION & AFFILIATES. No third party distribution is allowed unless
# expressly authorized by NVIDIA. Details are subject to change without notice.
# This code supersedes and replaces all information previously supplied. NVIDIA
# CORPORATION & AFFILIATES products are not authorized for use as critical
# components in life support devices or systems without express written approval
# of NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-FileCopyrightText: Copyright (c) 2016-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.
#
################################################################################

set(CMAKE_SYSTEM_NAME "Linux")
set(CMAKE_SYSTEM_VERSION 1)
set(VIBRANTE_BUILD ON)       #flags for the CMakeList.txt
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_LIBRARY_ARCHITECTURE aarch64-linux-gnu)

# need that one here, because this is a toolchain file and hence executed before
# default cmake settings are set
set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".so")

# check that Vibrante PDK must be set
if(NOT DEFINED VIBRANTE_PDK)
    if(DEFINED ENV{VIBRANTE_PDK})
        message(STATUS "VIBRANTE_PDK = ENV : $ENV{VIBRANTE_PDK}")
        set(VIBRANTE_PDK $ENV{VIBRANTE_PDK} CACHE STRING "Path to the vibrante-XXX-linux path for cross-compilation" FORCE)
    endif()
else()
     message(STATUS "VIBRANTE_PDK = ${VIBRANTE_PDK}")
endif()

if(NOT VIBRANTE_PDK)
    message(FATAL_ERROR
        "Path to PDK must be specified using cache variable VIBRANTE_PDK"
    )
endif()

list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES VIBRANTE_PDK)

set(ARCH "aarch64")
set(VIBRANTE TRUE)
set(VIBRANTE_V5L TRUE)
add_definitions(-DVIBRANTE -DVIBRANTE_V5L -DNVMEDIA_NVSCI_ENABLE)

if(EXISTS "${VIBRANTE_PDK}/lib-target/version-nv-pdk.txt")
    set(_VIBRANTE_PDK_FILE "${VIBRANTE_PDK}/lib-target/version-nv-pdk.txt")
elseif(EXISTS "${VIBRANTE_PDK}/lib-target/version-nv-sdk.txt")
    set(_VIBRANTE_PDK_FILE "${VIBRANTE_PDK}/lib-target/version-nv-sdk.txt")
else()
    message(FATAL_ERROR
        "Cauld NOT open file \"${VIBRANTE_PDK}/lib-target/version-nv-(pdk/sdk).txt\" for PDK branch detection"
    )
endif()
file(READ "${_VIBRANTE_PDK_FILE}" _VERSION_NV_PDK)
unset(_VIBRANTE_PDK_FILE)

if(_VERSION_NV_PDK MATCHES "^(.+)-[0123456789]+")
    set(_VIBRANTE_PDK_BRANCH "${CMAKE_MATCH_1}")
else()
    message(FATAL_ERROR
        "Could NOT determine PDK branch for PDK \"${VIBRANTE_PDK}\""
    )
endif()
unset(_VERSION_NV_PDK)

string(REPLACE "." ";" _VIBRANTE_PDK_VERSION_LIST "${_VIBRANTE_PDK_BRANCH}")
unset(_VIBRANTE_PDK_BRANCH)

list(LENGTH _VIBRANTE_PDK_VERSION_LIST _VIBRANTE_PDK_VERSION_LIST_LENGTH)
while(_VIBRANTE_PDK_VERSION_LIST_LENGTH LESS 4)
    list(APPEND VIBRANTE_PDK_VERSION_LIST 0)
    list(LENGTH _VIBRANTE_PDK_VERSION_LIST _VIBRANTE_PDK_VERSION_LIST_LENGTH)
endwhile()
unset(_VIBRANTE_PDK_VERSION_LIST_LENGTH)

list(GET _VIBRANTE_PDK_VERSION_LIST 0 _VIBRANTE_PDK_MAJOR)
list(GET _VIBRANTE_PDK_VERSION_LIST 1 _VIBRANTE_PDK_MINOR)
list(GET _VIBRANTE_PDK_VERSION_LIST 2 _VIBRANTE_PDK_PATCH)
list(GET _VIBRANTE_PDK_VERSION_LIST 3 _VIBRANTE_PDK_BUILD)
unset(_VIBRANTE_PDK_VERSION_LIST)

if(_VIBRANTE_PDK_MAJOR MATCHES "^([0-9]+)")
    set(VIBRANTE_PDK_MAJOR "${CMAKE_MATCH_1}")
else()
    message(FATAL_ERROR "Invalid PDK major version \"${_VIBRANTE_PDK_MAJOR}\"")
endif()
unset(_VIBRANTE_PDK_MAJOR)

if(_VIBRANTE_PDK_MINOR MATCHES "^([0-9]+)")
    set(VIBRANTE_PDK_MINOR "${CMAKE_MATCH_1}")
else()
    message(FATAL_ERROR "Invalid PDK minor version \"${_VIBRANTE_PDK_MINOR}\"")
endif()
unset(_VIBRANTE_PDK_MINOR)

if(_VIBRANTE_PDK_PATCH MATCHES "^([0-9]+)")
    set(VIBRANTE_PDK_PATCH "${CMAKE_MATCH_1}")
else()
    message(FATAL_ERROR "Invalid PDK patch version \"${_VIBRANTE_PDK_PATCH}\"")
endif()
unset(_VIBRANTE_PDK_PATCH)

if(_VIBRANTE_PDK_BUILD MATCHES "^([0-9]+)")
    set(VIBRANTE_PDK_BUILD "${CMAKE_MATCH_1}")
else()
    message(FATAL_ERROR "Invalid PDK build version \"${_VIBRANTE_PDK_BUILD}\"")
endif()
unset(_VIBRANTE_PDK_PATCH)

set(VIBRANTE_PDK_VERSION
    "${VIBRANTE_PDK_MAJOR}.${VIBRANTE_PDK_MINOR}.${VIBRANTE_PDK_PATCH}.${VIBRANTE_PDK_BUILD}"
)

math(EXPR VIBRANTE_PDK_DECIMAL
    "${VIBRANTE_PDK_MAJOR} * 1000000 + ${VIBRANTE_PDK_MINOR} * 10000 + ${VIBRANTE_PDK_PATCH} * 100 + ${VIBRANTE_PDK_BUILD}"
)

message(STATUS "Found PDK version ${VIBRANTE_PDK_VERSION} (${VIBRANTE_PDK_DECIMAL})")

# VIBRANTE_PDK_VERSION requires escaping so that it is treated as a string and
# not as an invalid floating point with too many decimal points.
add_compile_definitions(
    VIBRANTE_PDK_MAJOR=${VIBRANTE_PDK_MAJOR}
    VIBRANTE_PDK_MINOR=${VIBRANTE_PDK_MINOR}
    VIBRANTE_PDK_PATCH=${VIBRANTE_PDK_PATCH}
    VIBRANTE_PDK_BUILD=${VIBRANTE_PDK_BUILD}
    VIBRANTE_PDK_VERSION=\"${VIBRANTE_PDK_VERSION}\"
    VIBRANTE_PDK_DECIMAL=${VIBRANTE_PDK_DECIMAL}
)

# If VIBRANTE_C_COMPILER and VIBRANTE_CXX_COMPILER are defined, they will be used.
# if not the PDK-internal compiler will be used (default behavior)
if(DEFINED VIBRANTE_C_COMPILER AND DEFINED VIBRANTE_CXX_COMPILER)
  # Determine C and CXX compiler versions
  exec_program(${VIBRANTE_C_COMPILER} ARGS -dumpversion OUTPUT_VARIABLE C_COMPILER_VERSION RETURN_VALUE C_COMPILER_VERSION_ERROR)
  exec_program(${VIBRANTE_CXX_COMPILER} ARGS -dumpversion OUTPUT_VARIABLE CXX_COMPILER_VERSION RETURN_VALUE CXX_COMPILER_VERSION_ERROR)

  # Make sure C and CXX compiler versions match
  if(${C_COMPILER_VERSION_ERROR})
    message(FATAL_ERROR "Received error ${C_COMPILER_VERSION_ERROR} when determining compiler version for ${VIBRANTE_C_COMPILER}")
  elseif(${CXX_COMPILER_VERSION_ERROR})
    message(FATAL_ERROR
    "Received error ${CXX_COMPILER_VERSION_ERROR} when determining compiler version for ${VIBRANTE_CXX_COMPILER}")
  elseif(NOT ${C_COMPILER_VERSION} VERSION_EQUAL ${CXX_COMPILER_VERSION})
    message(FATAL_ERROR
    "C and CXX compiler versions must match.\n"
    "Found C Compiler Version = ${C_COMPILER_VERSION}\n"
    "Found CXX Compiler Version = ${CXX_COMPILER_VERSION}\n")
  endif()
  set(CMAKE_C_COMPILER ${VIBRANTE_C_COMPILER})
  set(CMAKE_CXX_COMPILER ${VIBRANTE_CXX_COMPILER})
  set(GCC_COMPILER_VERSION "${C_COMPILER_VERSION}" CACHE STRING "GCC Compiler version")
else()
  set(TOOLCHAIN "${VIBRANTE_PDK}/../toolchains/aarch64--glibc--stable-2022.03-1")
  set(CMAKE_CXX_COMPILER "${TOOLCHAIN}/bin/aarch64-linux-g++")
  set(CMAKE_C_COMPILER "${TOOLCHAIN}/bin/aarch64-linux-gcc")
  set(GCC_COMPILER_VERSION "9.3.0" CACHE STRING "GCC Compiler version")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Toolchain-V5_private.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Toolchain-V5_private.cmake)
endif()

set(LD_PATH ${VIBRANTE_PDK}/lib-target)

# Please, be careful looks like "-Wl,-unresolved-symbols=ignore-in-shared-libs" can lead to silent "ld" problems
set(CMAKE_SHARED_LINKER_FLAGS   "-L${LD_PATH} -Wl,--stats -Wl,-rpath,${LD_PATH} -Wl,-rpath-link,${LD_PATH} ${CMAKE_SHARED_LINKER_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS   "-L${LD_PATH} -Wl,-rpath,${LD_PATH} -Wl,-rpath-link,${LD_PATH} ${CMAKE_MODULE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS      "-L${LD_PATH} -Wl,-rpath,${LD_PATH} -Wl,-rpath-link,${LD_PATH} ${CMAKE_EXE_LINKER_FLAGS}")

# Set default library search path
set(CMAKE_LIBRARY_PATH ${LD_PATH})

# Set cmake root path. If there is no "/usr/local" in CMAKE_FIND_ROOT_PATH then FinCUDA.cmake doesn't work
set(CMAKE_FIND_ROOT_PATH ${VIBRANTE_PDK} ${VIBRANTE_PDK}/filesystem/targetfs/usr/local/ ${VIBRANTE_PDK}/filesystem/targetfs/ /usr/local)

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)


# set system default include dir
#
# NOTE: Set CUDA_DIR BEFORE PDK include, so that PDK include will actually come first.
# This is important for ensuring that PDK and CUDA directories are searched in a predictable
# deterministic manner. In the case of include files that exist in both places, this ensures
# That we always check PDK first, and THEN CUDA.
include_directories(BEFORE SYSTEM ${CUDA_DIR}/targets/aarch64-linux/include)
include_directories(BEFORE SYSTEM ${VIBRANTE_PDK}/include)

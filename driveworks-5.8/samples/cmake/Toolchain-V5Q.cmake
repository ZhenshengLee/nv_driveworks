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

option(DW_USE_QCC "Use QNX QCC compiler" ON)

set(CMAKE_SYSTEM_NAME "QNX")
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(VIBRANTE_BUILD ON)       #flags for the CMakeList.txt
set(ARCH gcc_ntoaarch64le)
set(CMAKE_LIBRARY_ARCHITECTURE aarch64-unknown-nto-qnx)

# need that one here, because this is a toolchain file and hence executed before
# default cmake settings are set
set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".so")

# QNX-required environment variables
if(NOT DEFINED ENV{QNX_HOST} OR NOT DEFINED ENV{QNX_TARGET})
    message(FATAL_ERROR "Need to define QNX_HOST and QNX_TARGET")
endif()

set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CUDA_DIR)
if(NOT DEFINED CUDA_DIR)
    message(FATAL_ERROR "Need to define CUDA_DIR")
endif()

set(QNX_HOST "$ENV{QNX_HOST}")
set(QNX_TARGET "$ENV{QNX_TARGET}")

# QNX-required definitions & compiler flags
set(QNX_TOOLCHAIN_VERSION "8.3.0")
set(QNX_TOOLCHAIN_PATH "${QNX_HOST}/usr/bin")
set(QNX_TOOLCHAIN_TRIPLE "aarch64-unknown-nto-qnx7.1.0")
set(QNX_TOOLCHAIN_PREFIX "${QNX_TOOLCHAIN_PATH}/${QNX_TOOLCHAIN_TRIPLE}")

include_directories(SYSTEM
    ${QNX_HOST}/usr/lib/gcc/${QNX_TOOLCHAIN_TRIPLE}/${QNX_TOOLCHAIN_VERSION}/include
    ${QNX_TARGET}/usr/include/io-pkt
    $<$<COMPILE_LANGUAGE:CXX>:${QNX_TARGET}/usr/include/c++/v1>
    $<$<COMPILE_LANGUAGE:CUDA>:${QNX_TARGET}/usr/include/c++/v1>
    ${QNX_TARGET}/usr/include)
add_compile_definitions(_POSIX_C_SOURCE=200112L _QNX_SOURCE WIN_INTERFACE_CUSTOM _FILE_OFFSET_BITS=64)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-tree-vectorize>)

# check that Vibrante PDK must be set
if(NOT DEFINED VIBRANTE_PDK)
    if(DEFINED ENV{VIBRANTE_PDK})
        message(STATUS "VIBRANTE_PDK = ENV : $ENV{VIBRANTE_PDK}")
        set(VIBRANTE_PDK $ENV{VIBRANTE_PDK} CACHE STRING "Path to the vibrante-XXX-qnx path for cross-compilation" FORCE)
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

set(VIBRANTE TRUE)
set(VIBRANTE_V5Q TRUE)
add_definitions(-DVIBRANTE -DVIBRANTE_V5Q -DNVMEDIA_NVSCI_ENABLE)

if(EXISTS "${VIBRANTE_PDK}/lib-target/version-pdk.txt")
    set(_VIBRANTE_PDK_FILE "${VIBRANTE_PDK}/lib-target/version-pdk.txt")
elseif(EXISTS "${VIBRANTE_PDK}/lib-target/version-sdk.txt")
    set(_VIBRANTE_PDK_FILE "${VIBRANTE_PDK}/lib-target/version-sdk.txt")
else()
    message(FATAL_ERROR
        "Cauld NOT open file \"${VIBRANTE_PDK}/lib-target/version-(pdk/sdk).txt\" for PDK branch detection"
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

set(GCC_COMPILER_VERSION ${QNX_TOOLCHAIN_TRIPLE} CACHE STRING "GCC Compiler version")


set(QNX_LINKER_FLAGS "")

if(DW_USE_QCC)
  message(STATUS "Enabling QNX QOS Compiler (qcc/q++)")

  if (NOT EXISTS $ENV{HOME}/.qnx/license/licenses)
     message(FATAL_ERROR "QNX Software Development Platform Developer License could NOT be found at ~/.qnx/license/licenses.")
  endif()

  set(CMAKE_C_COMPILER      "${QNX_TOOLCHAIN_PATH}/qcc")
  set(CMAKE_CXX_COMPILER    "${QNX_TOOLCHAIN_PATH}/q++")
  set(CMAKE_C_FLAGS         "${CMAKE_C_FLAGS} -V${ARCH}")
  set(CMAKE_CXX_FLAGS       "${CMAKE_CXX_FLAGS} -V${ARCH}")
  set(CMAKE_CUDA_FLAGS      "--qpp-config ${ARCH}")
  set(QNX_LINKER_FLAGS      "-V${ARCH}")
else()
  set(CMAKE_CXX_COMPILER "${QNX_TOOLCHAIN_PREFIX}-g++")
  set(CMAKE_C_COMPILER "${QNX_TOOLCHAIN_PREFIX}-gcc")
endif()

set(CMAKE_AR      "${QNX_TOOLCHAIN_PREFIX}-ar")
set(CMAKE_LINKER  "${QNX_TOOLCHAIN_PREFIX}-ld")
set(CMAKE_NM      "${QNX_TOOLCHAIN_PREFIX}-nm")
set(CMAKE_OBJCOPY "${QNX_TOOLCHAIN_PREFIX}-objcopy")
set(CMAKE_OBJDUMP "${QNX_TOOLCHAIN_PREFIX}-objdump")

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Toolchain-V5_private.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Toolchain-V5_private.cmake)
endif()

set(LD_PATH ${VIBRANTE_PDK}/lib-target:${VIBRANTE_PDK}/nvidia-bsp/aarch64le/usr/lib)
set(QNX_LIBRARY_DIRS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE})
set(CUDA_LIBRARY_DIRS ${CUDA_DIR}/targets/aarch64-qnx/lib)
# Please, be careful looks like "-Wl,-unresolved-symbols=ignore-in-shared-libs" can lead to silent "ld" problems
set(CMAKE_SHARED_LINKER_FLAGS   "${QNX_LINKER_FLAGS} -L${LD_PATH} -Wl,-rpath-link,${LD_PATH} ${CMAKE_SHARED_LINKER_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS   "${QNX_LINKER_FLAGS} -L${LD_PATH} -Wl,-rpath-link,${LD_PATH} ${CMAKE_MODULE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS      "${QNX_LINKER_FLAGS} -L${CUDA_LIBRARY_DIRS} -Wl,-rpath-link,${CUDA_LIBRARY_DIRS} -L${LD_PATH} -Wl,-rpath-link,${LD_PATH}  -L${QNX_LIBRARY_DIRS} -Wl,-rpath-link,${QNX_LIBRARY_DIRS} ${CMAKE_EXE_LINKER_FLAGS}")
if(VIBRANTE_PDK_VERSION STREQUAL 5.1.12.4)
    set(BSP_PATH ${VIBRANTE_PDK}/nvidia-bsp/aarch64le/usr/lib/)
    set(CMAKE_SHARED_LINKER_FLAGS   "${CMAKE_SHARED_LINKER_FLAGS} -L${BSP_PATH}")
    set(CMAKE_MODULE_LINKER_FLAGS   "${CMAKE_MODULE_LINKER_FLAGS} -L${BSP_PATH}")
    set(CMAKE_EXE_LINKER_FLAGS      "${CMAKE_EXE_LINKER_FLAGS} -L${BSP_PATH}")
endif()

# Set cmake root path.
SET(CMAKE_FIND_ROOT_PATH ${VIBRANTE_PDK} /usr/local/)

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
include_directories(BEFORE SYSTEM ${CUDA_DIR}/targets/aarch64-qnx/include)
include_directories(BEFORE SYSTEM ${VIBRANTE_PDK}/include)

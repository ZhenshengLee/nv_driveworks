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
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(CMAKE_SYSTEM_NAME QNX)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_SYSTEM_VERSION 7.1.0)

string(TOLOWER "${CMAKE_SYSTEM_NAME}" _SYSTEM_NAME_LOWER)

set(CMAKE_LIBRARY_ARCHITECTURE
    "${CMAKE_SYSTEM_PROCESSOR}-unknown-nto-${_SYSTEM_NAME_LOWER}-safety"
)

if(NOT VIBRANTE_PDK)
    if(ENV{VIBRANTE_PDK})
        set(VIBRANTE_PDK "$ENV{VIBRANTE_PDK}")
    else()
        set(_PDK_TOP_DIRS)
        if(ENV{NV_WORKSPACE})
            list(APPEND _PDK_TOP_DIRS "$ENV{NV_WORKSPACE}")
        endif()
        if(ENV{PDK_TOP})
            list(APPEND _PDK_TOP_DIRS "$ENV{PDK_TOP}")
        endif()
        list(APPEND _PDK_TOP_DIRS
            /drive
            /rfs
            /usr/local/driveos-sdk
        )
        set(VIBRANTE_PDK)
        foreach(_PDK_TOP_DIR ${_PDK_TOP_DIRS})
            if(EXISTS "${_PDK_TOP_DIR}/drive-${_SYSTEM_NAME_LOWER}-safety")
                set(VIBRANTE_PDK
                    "${_PDK_TOP_DIR}/drive-${_SYSTEM_NAME_LOWER}-safety"
                )
                break()
            endif()
            if(EXISTS "${_PDK_TOP_DIR}/drive-t234ref-${_SYSTEM_NAME_LOWER}-safety")
                set(VIBRANTE_PDK
                    "${_PDK_TOP_DIR}/drive-t234ref-${_SYSTEM_NAME_LOWER}-safety"
                )
                break()
            endif()
        endforeach()
        unset(_PDK_TOP_DIRS)
    endif()
    set(VIBRANTE_PDK "${VIBRANTE_PDK}" CACHE PATH
        "Path to ${CMAKE_SYSTEM_NAME} for Safety PDK for cross-compilation." FORCE
    )
endif()

if(NOT VIBRANTE_PDK)
    message(FATAL_ERROR
        "Could NOT find ${CMAKE_SYSTEM_NAME} for Safety PDK for "
        "cross-compilation. Path must be specified using the cache variable "
        "VIBRANTE_PDK"
    )
endif()

list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES VIBRANTE_PDK)

if(NOT VIBRANTE_PDK_VERSION
    AND NOT VIBRANTE_PDK_MAJOR
    AND NOT VIBRANTE_PDK_MINOR
    AND NOT VIBRANTE_PDK_PATCH
    AND NOT VIBRANTE_PDK_BUILD
    AND NOT VIBRANTE_PDK_DECIMAL
)
    set(_VIBRANTE_PDK_VERSION_FILES
        "${VIBRANTE_PDK}/lib-target/version-pdk.txt"
        "${VIBRANTE_PDK}/lib-target/version-sdk.txt"
    )
    set(_VERSION_NV_PDK)
    foreach(_VIBRANTE_PDK_VERSION_FILE ${_VIBRANTE_PDK_VERSION_FILES})
        if(EXISTS "${_VIBRANTE_PDK_VERSION_FILE}")
            file(READ "${_VIBRANTE_PDK_VERSION_FILE}" _VERSION_NV_PDK)
            break()
        endif()
    endforeach()
    unset(_VIBRANTE_PDK_VERSION_FILES)

    if(NOT _VERSION_NV_PDK)
        message(FATAL_ERROR
            "Could NOT open file \"${VIBRANTE_PDK}/lib-target/version-*.txt\ "
            "for ${CMAKE_SYSTEM_NAME} for Safety PDK version detection."
        )
    endif()
    if(_VERSION_NV_PDK MATCHES "^(.+)-[0123456789]+")
        set(_VIBRANTE_PDK_BRANCH "${CMAKE_MATCH_1}")
    else()
        message(FATAL_ERROR
            "Could NOT determine PDK version for ${CMAKE_SYSTEM_NAME} for "
            "Safety PDK \"${VIBRANTE_PDK}\"."
        )
    endif()
    unset(_VERSION_NV_PDK)

    string(REPLACE "." ";" _VIBRANTE_PDK_VERSION_LIST
        "${_VIBRANTE_PDK_BRANCH}"
    )
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
endif()

list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
    VIBRANTE_PDK_VERSION
    VIBRANTE_PDK_MAJOR
    VIBRANTE_PDK_MINOR
    VIBRANTE_PDK_PATCH
    VIBRANTE_PDK_BUILD
    VIBRANTE_PDK_DECIMAL
)

if(NOT _VIBRANTE_PDK_FOUND_MESSAGE)
    message(STATUS
        "Found ${CMAKE_SYSTEM_NAME} for Safety PDK for cross-compilation: "
        "${VIBRANTE_PDK} (found version: \"${VIBRANTE_PDK_VERSION}\")")
    set(_VIBRANTE_PDK_FOUND_MESSAGE ON)
endif()

set(DW_IS_SAFETY ON)
set(VIBRANTE ON)
set(VIBRANTE_V5L OFF)
set(VIBRANTE_V5Q ON)

list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES _VIBRANTE_PDK_FOUND_MESSAGE)

set(QNX_HOST "$ENV{QNX_HOST}")

if(NOT QNX_HOST)
    message(FATAL_ERROR "QNX_HOST environment variable is NOT set")
endif()

if(NOT EXISTS "${QNX_HOST}")
    message(FATAL_ERROR
        "Directory specified by QNX_HOST environment variable does NOT exist"
    )
endif()

set(QNX_TARGET "$ENV{QNX_TARGET}")

if(NOT QNX_TARGET)
    message(FATAL_ERROR "QNX_TARGET environment variable is NOT set")
endif()

if(NOT EXISTS "${QNX_TARGET}")
    message(FATAL_ERROR
        "Directory \"${QNX_TARGET}\" specified by the QNX_TARGET environment "
        "variable does NOT exist"
    )
endif()

set(QNX_CONFIGURATION_EXCLUSIVE "$ENV{QNX_CONFIGURATION_EXCLUSIVE}")

if(NOT QNX_CONFIGURATION_EXCLUSIVE)
    set(QNX_CONFIGURATION_EXCLUSIVE "$ENV{HOME}/.qnx")
endif()

if(NOT EXISTS "${QNX_CONFIGURATION_EXCLUSIVE}/license/licenses")
    message(FATAL_ERROR
        "QNX Software Development Platform Developer License could NOT be "
        "found at \"${QNX_CONFIGURATION_EXCLUSIVE}/license/licenses\""
    )
endif()

set(_TOOLCHAIN_DIR "$ENV{QNX_HOST}/usr/bin")

set(CMAKE_C_COMPILER "${_TOOLCHAIN_DIR}/qcc" CACHE FILEPATH
    "Path to the C compiler."
)
set(CMAKE_C_COMPILER_TARGET 8.3.0,gcc_ntoaarch64le)

set(CMAKE_CXX_COMPILER "${_TOOLCHAIN_DIR}/q++" CACHE FILEPATH
    "Path to the CXX compiler."
)
set(CMAKE_CXX_COMPILER_TARGET "${CMAKE_C_COMPILER_TARGET}")

set(CMAKE_CUDA_COMPILER_FORCED ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-safe-11.4/bin/nvcc CACHE FILEPATH
    "Path to the CUDA compiler."
)
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "${CMAKE_CXX_COMPILER}")

set(_TOOLCHAIN_PREFIX
    "${_TOOLCHAIN_DIR}/${CMAKE_SYSTEM_PROCESSOR}-unknown-nto-${_SYSTEM_NAME_LOWER}${CMAKE_SYSTEM_VERSION}-"
)

set(CMAKE_ADDR2LINE "${_TOOLCHAIN_PREFIX}addr2line" CACHE FILEPATH
    "Path to the tool to convert addresses into file names and line numbers."
)

set(CMAKE_AR "${_TOOLCHAIN_PREFIX}ar" CACHE FILEPATH
    "Path to the tool to  create, modify, and extract from archives."
)

set(CMAKE_C_COMPILER_AR "${CMAKE_AR}" CACHE FILEPATH
    "Path to the tool to  create, modify, and extract from archives."
)

set(CMAKE_CXX_COMPILER_AR "${CMAKE_AR}" CACHE FILEPATH
    "Path to the tool to  create, modify, and extract from archives."
)

set(CMAKE_LINKER "${_TOOLCHAIN_PREFIX}ld" CACHE FILEPATH
    "Path to the linker."
)

set(CMAKE_NM "${_TOOLCHAIN_PREFIX}nm" CACHE FILEPATH
    "Path to the tool to list symbols from object files."
)

set(CMAKE_OBJCOPY "${_TOOLCHAIN_PREFIX}objcopy" CACHE FILEPATH
    "Path to the tool to copy and translate object files."
)

set(CMAKE_RANLIB "${_TOOLCHAIN_PREFIX}ranlib" CACHE FILEPATH
    "Path to the tool to generate an index to an archive."
)

set(CMAKE_C_COMPILER_RANLIB "${CMAKE_RANLIB}" CACHE FILEPATH
    "Path to the tool to generate an index to an archive."
)

set(CMAKE_CXX_COMPILER_RANLIB "${CMAKE_RANLIB}" CACHE FILEPATH
    "Path to the tool to generate an index to an archive."
)

set(CMAKE_READELF "${_TOOLCHAIN_PREFIX}readelf" CACHE FILEPATH
    "Path to the tool to display information about ELF files."
)

set(CMAKE_STRIP "${_TOOLCHAIN_PREFIX}strip" CACHE FILEPATH
    "Path to the tool to discard symbols and other data from object files."
)

unset(_TOOLCHAIN_PREFIX)
unset(_TOOLCHAIN_DIR)
unset(_SYSTEM_NAME_LOWER)

set(_COMPILE_DEFINITIONS
    -DDW_IS_SAFETY
    -DNV_IS_SAFETY=1
    -DVIBRANTE
    -DVIBRANTE_V5Q
    -DVIBRANTE_PDK_DECIMAL=${VIBRANTE_PDK_DECIMAL}
    -DVIBRANTE_PDK_MAJOR=${VIBRANTE_PDK_MAJOR}
    -DVIBRANTE_PDK_MINOR=${VIBRANTE_PDK_MINOR}
    -DVIBRANTE_PDK_PATCH=${VIBRANTE_PDK_PATCH}
    -DVIBRANTE_PDK_BUILD=${VIBRANTE_PDK_BUILD}
    -DVIBRANTE_PDK_VERSION=\"${VIBRANTE_PDK_VERSION}\"
)

set(_C_COMPILE_OPTIONS
  -D_POSIX_C_SOURCE=200112L
  -D_QNX_SOURCE
  -DFD_SETSIZE=1024
  ${_COMPILE_DEFINITIONS}
  -N8M
)
string(REPLACE ";" " " _C_FLAGS "${_C_COMPILE_OPTIONS}")

if(CMAKE_C_FLAGS)
    string(APPEND CMAKE_C_FLAGS " ${_C_FLAGS}")
else()
    set(CMAKE_C_FLAGS "${_C_FLAGS}")
endif()

if(CMAKE_CXX_FLAGS)
    string(APPEND CMAKE_CXX_FLAGS " ${_C_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "${_C_FLAGS}")
endif()

string(REPLACE " " "," _C_FLAGS_FOR_CUDA "${_C_FLAGS}")
set(_CUDA_COMPILE_OPTIONS
    ${_COMPILE_DEFINITIONS}
    -Xcompiler ${_C_FLAGS_FOR_CUDA}
)
string(REPLACE ";" " " _CUDA_FLAGS "${_CUDA_COMPILE_OPTIONS}")

if(CMAKE_CUDA_FLAGS)
    string(PREPEND CMAKE_CUDA_FLAGS
        "-qpp-config=${CMAKE_C_COMPILER_TARGET} --target-directory=aarch64-qnx-safe "
    )
    string(APPEND CMAKE_CUDA_FLAGS " ${_CUDA_FLAGS}")
else()
    set(CMAKE_CUDA_FLAGS
        "-qpp-config=${CMAKE_C_COMPILER_TARGET} --target-directory=aarch64-qnx-safe ${_CUDA_FLAGS}"
    )
endif()

unset(_C_COMPILE_OPTIONS)
unset(_C_FLAGS_FOR_CUDA)
unset(_C_FLAGS)
unset(_COMPILE_DEFINITIONS)
unset(_CUDA_COMPILE_OPTIONS)
unset(_CUDA_FLAGS)

if(CMAKE_EXE_LINKER_FLAGS)
    string(PREPEND CMAKE_EXE_LINKER_FLAGS "-V${CMAKE_C_COMPILER_TARGET} ")
else()
    set(CMAKE_EXE_LINKER_FLAGS "-V${CMAKE_C_COMPILER_TARGET}")
endif()

if(CMAKE_MODULE_LINKER_FLAGS)
    string(PREPEND CMAKE_MODULE_LINKER_FLAGS "-V${CMAKE_C_COMPILER_TARGET} ")
else()
    set(CMAKE_MODULE_LINKER_FLAGS "-V${CMAKE_C_COMPILER_TARGET}")
endif()

if(CMAKE_SHARED_LINKER_FLAGS)
    string(PREPEND CMAKE_SHARED_LINKER_FLAGS "-V${CMAKE_C_COMPILER_TARGET} ")
else()
    set(CMAKE_SHARED_LINKER_FLAGS "-V${CMAKE_C_COMPILER_TARGET}")
endif()

set(CMAKE_C_FLAGS_DEBUG_INIT -g)
set(CMAKE_C_FLAGS_MINSIZEREL_INIT "-DNDEBUG -Os")
set(CMAKE_C_FLAGS_RELEASE_INIT "-DNDEBUG -O2")
set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "-DNDEBUG -g -O2")

set(CMAKE_CXX_FLAGS_DEBUG_INIT "${CMAKE_C_FLAGS_DEBUG_INIT}")
set(CMAKE_CXX_FLAGS_MINSIZEREL_INIT "${CMAKE_C_FLAGS_MINSIZEREL_INIT}")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "${CMAKE_C_FLAGS_RELEASE_INIT}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "${CMAKE_C_FLAGS_RELWITHDEBINFO_INIT}")

set(CMAKE_CUDA_FLAGS_DEBUG_INIT -g)
set(CMAKE_CUDA_FLAGS_MINSIZEREL "-DNDEBUG -O1")
set(CMAKE_CUDA_FLAGS_RELEASE_INIT "-DNDEBUG -O2")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO_INIT "-DNDEBUG -g -O2")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES
    /usr/local/cuda-safe-11.4/targets/aarch64-qnx-safe/include
)

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES
    /usr/local/cuda-safe-11.4/targets/cuda-safe-11.4/lib/stubs
    /usr/local/cuda-safe-11.4/targets/cuda-safe-11.4/lib
)

set(CMAKE_FIND_ROOT_PATH "${VIBRANTE_PDK}")

set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

set(CMAKE_SYSTEM_INCLUDE_PATH
    "${VIBRANTE_PDK}/include"
    "/usr/include/${CMAKE_LIBRARY_ARCHITECTURE}"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(CMAKE_SYSTEM_LIBRARY_PATH
    "${VIBRANTE_PDK}/lib-target"
    "${VIBRANTE_PDK}/nvidia-bsp/aarch64le/usr/lib"
    "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
    ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES}
)

set(_LINKER_OPTIONS)
foreach(_LIBRARY_PATH ${CMAKE_SYSTEM_LIBRARY_PATH})
    list(APPEND _LINKER_OPTIONS
        "-L${_LIBRARY_PATH}"
        "-Wl,-rpath-link,${_LIBRARY_PATH}"
    )
endforeach()

string(REPLACE ";" " " _LINKER_FLAGS "${_LINKER_OPTIONS}")

string(APPEND CMAKE_EXE_LINKER_FLAGS " ${_LINKER_FLAGS}")
string(APPEND CMAKE_MODULE_LINKER_FLAGS " ${_LINKER_FLAGS}")
string(APPEND CMAKE_SHARED_LINKER_FLAGS " ${_LINKER_FLAGS}")

unset(_LINKER_OPTIONS)
unset(_LINKER_FLAGS)

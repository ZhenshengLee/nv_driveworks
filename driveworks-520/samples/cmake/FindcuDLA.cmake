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
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# FindcuDLA
# ---------
#
# Finds the NVIDIA (R) CUDA (R) Deep Learning Accelerator (cuDLA) libraries.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module provides the following imported targets, if found:
#
# ``cuDLA::cuDLA``
#   The cuDLA shared library, if found, else the cuDLA static library.
# ``cuDLA::cudla``
#   The cuDLA shared library.
# ``cuDLA::cudla_static``
#   The cuDLA static library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This will define the following variables:
#
# ``cuDLA_FOUND``
#   True if the system has the cuDLA library.
# ``cuDLA_VERSION``
#   The version of the cuDLA library that was found.
# ``cuDLA_INCLUDE_DIRS``
#   Include directories needed to use the cuDLA library.
# ``cuDLA_LIBRARIES``
#   Libraries needed to link to the cuDLA library.
#
# Cache Variables
# ^^^^^^^^^^^^^^^
#
# The following cache variables may also be set:
#
# ``cuDLA_INCLUDE_DIR``
#   The directory containing ``cudla.h``.
# ``cuDLA_LIBRARY``
#   The path to ``libcudla.so``.
# ``cuDLA_STATIC_LIBRARY``
#   The path to ``libcudla_static.a``.

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    get_filename_component(_CUDLA_PATHS
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" DIRECTORY
    )
else()
    set(_CUDLA_PATHS)
endif()

if(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION)
    string(REPLACE "." ";" _CUDLA_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR
        "${CMAKE_CUDA_COMPILER_TOOLKIT_VERSION}"
    )
    list(GET _CUDLA_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR 0 1
        _CUDLA_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR
    )
    list(JOIN _CUDLA_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR .
        _CUDLA_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR
    )

    if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL aarch64-linux-gnu)
        set(_CUDLA_LIBRARY_ARCHITECTURE aarch64-linux)
    elseif(CMAKE_LIBRARY_ARCHITECTURE STREQUAL aarch64-unknown-nto-qnx)
        set(_CUDLA_LIBRARY_ARCHITECTURE aarch64-qnx)
    elseif(CMAKE_LIBRARY_ARCHITECTURE STREQUAL aarch64-unknown-nto-qnx-safety)
        set(_CUDLA_LIBRARY_ARCHITECTURE aarch64-qnx-safe)
    else()
        set(_CUDLA_LIBRARY_ARCHITECTURE x86_64-linux)
    endif()

    if(CMAKE_LIBRARY_ARCHITECTURE MATCHES "^aarch64-unknown-nto-qnx")
        list(APPEND _CUDLA_PATHS
            "/usr/local/cuda-safe-${_CUDLA_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR}/targets/${_CUDLA_LIBRARY_ARCHITECTURE}"
        )
    endif()

    list(APPEND _CUDLA_PATHS
        "/usr/local/cuda-${_CUDLA_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR}/targets/${_CUDLA_LIBRARY_ARCHITECTURE}"
    )
    unset(_CUDLA_LIBRARY_ARCHITECTURE)
    unset(_CUDLA_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR)

    set(cuDLA_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")
else()
    set(cuDLA_VERSION)
endif()

find_path(cuDLA_INCLUDE_DIR
    NAMES cudla.h
    PATHS ${_CUDLA_PATHS}
    PATH_SUFFIXES include
    DOC "Path to the directory containing the header file cudla.h."
)

set(_CUDLA_SONAME
    "${CMAKE_SHARED_LIBRARY_PREFIX}cudla${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
find_library(cuDLA_LIBRARY
    NAMES "${_CUDLA_SONAME}"
    PATHS ${_CUDLA_PATHS}
    PATH_SUFFIXES lib64 lib
    DOC "Path to the shared library file ${_CUDLA_SONAME}."
)

if(cuDLA_LIBRARY)
    get_filename_component(_CUDLA_HINTS "${cuDLA_LIBRARY}" DIRECTORY)
else()
    set(_CUDLA_HINTS)
endif()

if(NOT cuDLA_LIBRARY)
    find_library(cuDLA_LIBRARY
        NAMES "${_CUDLA_SONAME}"
        PATHS ${_CUDLA_PATHS}
        PATH_SUFFIXES lib64/stubs lib/stubs
        DOC "Path to the shared library file ${_CUDLA_SONAME}."
    )

    if(cuDLA_LIBRARY)
        get_filename_component(_CUDLA_HINTS "${cuDLA_LIBRARY}"
            DIRECTORY
        )
        get_filename_component(_CUDLA_HINTS "${_CUDLA_HINTS}" DIRECTORY)
    endif()
endif()

set(_CUDLA_STATIC_LIBRARY_NAME
    "${CMAKE_STATIC_LIBRARY_PREFIX}cudla_static${CMAKE_STATIC_LIBRARY_SUFFIX}"
)
find_library(cuDLA_STATIC_LIBRARY
    NAMES "${_CUDLA_STATIC_LIBRARY_NAME}"
    HINTS ${_CUDLA_HINTS}
    PATHS ${_CUDLA_PATHS}
    PATH_SUFFIXES lib64 lib
    DOC "Path to the static library file ${_CUDLA_STATIC_LIBRARY_NAME}."
)

unset(_CUDLA_STATIC_LIBRARY_NAME)
unset(_CUDLA_PATHS)
unset(_CUDLA_HINTS)

include(FindPackageHandleStandardArgs)

set(_CUDLA_REQUIRED_VARS cuDLA_INCLUDE_DIR)
if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL aarch64-unknown-nto-qnx-safety)
    list(APPEND _CUDLA_REQUIRED_VARS cuDLA_STATIC_LIBRARY)
else()
    list(APPEND _CUDLA_REQUIRED_VARS cuDLA_LIBRARY)
endif()

find_package_handle_standard_args(cuDLA
    FOUND_VAR cuDLA_FOUND
    REQUIRED_VARS ${_CUDLA_REQUIRED_VARS}
    VERSION_VAR cuDLA_VERSION
)

unset(_CUDLA_REQUIRED_VARS)

if(cuDLA_FOUND)
    set(cuDLA_INCLUDE_DIRS "${cuDLA_INCLUDE_DIR}")
    mark_as_advanced(cuDLA_INCLUDE_DIR)

    if(cuDLA_LIBRARY)
        set(cuDLA_LIBRARIES "${cuDLA_LIBRARY}")
    else()
        set(cuDLA_LIBRARIES "${cuDLA_STATIC_LIBRARY}")
    endif()

    if(cuDLA_LIBRARY)
        mark_as_advanced(cuDLA_LIBRARY)

        if(NOT TARGET cuDLA::cudla)
            add_library(cuDLA::cudla SHARED IMPORTED)
            set_target_properties(cuDLA::cudla PROPERTIES
                IMPORTED_LOCATION "${cuDLA_LIBRARY}"
                IMPORTED_SONAME "${_CUDLA_LIBRARY_SONAME}"
            )
            target_include_directories(cuDLA::cudla SYSTEM INTERFACE
                "${cuDLA_INCLUDE_DIR}"
            )
        endif()
    endif()

    if(cuDLA_STATIC_LIBRARY)
        mark_as_advanced(cuDLA_STATIC_LIBRARY)

        if(NOT TARGET cuDLA::cudla_static)
            add_library(cuDLA::cudla_static STATIC IMPORTED)
            set_target_properties(cuDLA::cudla_static PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES CXX
                IMPORTED_LOCATION "${cuDLA_STATIC_LIBRARY}"
            )
            target_include_directories(cuDLA::cudla_static SYSTEM INTERFACE
                "${cuDLA_INCLUDE_DIR}"
            )
        endif()
    endif()

    if(NOT TARGET cuDLA::cuDLA)
        add_library(cuDLA::cuDLA INTERFACE IMPORTED)
        if(TARGET cuDLA::cudla)
            target_link_libraries(cuDLA::cuDLA INTERFACE cuDLA::cudla)
        else()
            target_link_libraries(cuDLA::cuDLA INTERFACE cuDLA::cudla_static)
        endif()
    endif()
endif()

unset(_CUDLA_SONAME)

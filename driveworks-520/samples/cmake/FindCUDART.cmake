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

# FindCUDART
# ----------
#
# Finds the NVIDIA (R) CUDA (R) driver and runtime (CUDART) libraries.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module provides the following imported targets, if found:
#
# ``CUDART::CUDART``
#   The CUDART libraries.
# ``CUDART::cuda``
#   The CUDA shared library.
# ``CUDART::cudart``
#   The CUDART shared library.
# ``CUDART::cudart_static``
#   The CUDART static library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This will define the following variables:
#
# ``CUDART_FOUND``
#   True if the system has the CUDART libraries.
# ``CUDART_VERSION``
#   The version of the CUDART libraries that was found.
# ``CUDART_INCLUDE_DIRS``
#   Include directories needed to use the CUDART libraries.
# ``CUDART_LIBRARIES``
#   Libraries needed to link to the CUDART libraries.
#
# Cache Variables
# ^^^^^^^^^^^^^^^
#
# The following cache variables may also be set:
#
# ``CUDART_INCLUDE_DIR``
#   The directory containing ``cuda_runtime_api.h``.
# ``CUDART_CUDA_LIBRARY``
#   The path to ``libcuda.so``.
# ``CUDART_CUDART_LIBRARY``
#   The path to ``libcudart.so``.
# ``CUDART_CUDART_STATIC_LIBRARY``
#   The path to ``libcudart_static.a``.

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    get_filename_component(_CUDART_PATHS
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" DIRECTORY
    )
else()
    set(_CUDART_PATHS)
endif()

if(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION)
    string(REPLACE "." ";" _CUDART_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR
        "${CMAKE_CUDA_COMPILER_TOOLKIT_VERSION}"
    )
    list(GET _CUDART_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR 0 1
        _CUDART_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR
    )
    list(JOIN _CUDART_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR .
        _CUDART_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR
    )

    if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL aarch64-linux-gnu)
        set(_CUDART_LIBRARY_ARCHITECTURE aarch64-linux)
    elseif(CMAKE_LIBRARY_ARCHITECTURE STREQUAL aarch64-unknown-nto-qnx)
        set(_CUDART_LIBRARY_ARCHITECTURE aarch64-qnx)
    elseif(CMAKE_LIBRARY_ARCHITECTURE STREQUAL aarch64-unknown-nto-qnx-safety)
        set(_CUDART_LIBRARY_ARCHITECTURE aarch64-qnx-safe)
    else()
        set(_CUDART_LIBRARY_ARCHITECTURE x86_64-linux)
    endif()

    if(CMAKE_LIBRARY_ARCHITECTURE MATCHES "^aarch64-unknown-nto-qnx")
        list(APPEND _CUDART_PATHS
            "/usr/local/cuda-safe-${_CUDART_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR}/targets/${_CUDART_LIBRARY_ARCHITECTURE}"
        )
    endif()

    list(APPEND _CUDART_PATHS
        "/usr/local/cuda-${_CUDART_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR}/targets/${_CUDART_LIBRARY_ARCHITECTURE}"
    )
    unset(_CUDART_LIBRARY_ARCHITECTURE)
    unset(_CUDART_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR)

    set(CUDART_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")
else()
    set(CUDART_VERSION)
endif()

find_path(CUDART_INCLUDE_DIR
    NAMES cuda_runtime_api.h
    PATHS ${_CUDART_PATHS}
    PATH_SUFFIXES include
    DOC "Path to the directory containing the header file cuda_runtime_api.h."
)

set(_CUDART_CUDART_SONAME
    "${CMAKE_SHARED_LIBRARY_PREFIX}cudart${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
find_library(CUDART_CUDART_LIBRARY
    NAMES "${_CUDART_CUDART_SONAME}"
    PATHS ${_CUDART_PATHS}
    PATH_SUFFIXES lib64 lib
    DOC "Path to the shared library file ${_CUDART_CUDART_SONAME}."
)

if(CUDART_CUDART_LIBRARY)
    get_filename_component(_CUDART_HINTS "${CUDART_CUDART_LIBRARY}" DIRECTORY)
else()
    set(_CUDART_HINTS)
endif()

if(NOT CUDART_CUDART_LIBRARY)
    find_library(CUDART_CUDART_LIBRARY
        NAMES "${_CUDART_CUDART_SONAME}"
        PATHS ${_CUDART_PATHS}
        PATH_SUFFIXES lib64/stubs lib/stubs
        DOC "Path to the shared library file ${_CUDART_CUDART_SONAME}."
    )

    if(CUDART_CUDART_LIBRARY)
        get_filename_component(_CUDART_HINTS "${CUDART_CUDART_LIBRARY}"
            DIRECTORY
        )
        get_filename_component(_CUDART_HINTS "${_CUDART_HINTS}" DIRECTORY)
    endif()
endif()

set(_CUDART_CUDART_STATIC_LIBRARY_NAME
    "${CMAKE_STATIC_LIBRARY_PREFIX}cudart_static${CMAKE_STATIC_LIBRARY_SUFFIX}"
)
find_library(CUDART_CUDART_STATIC_LIBRARY
    NAMES "${_CUDART_CUDART_STATIC_LIBRARY_NAME}"
    HINTS ${_CUDART_HINTS}
    PATHS ${_CUDART_PATHS}
    PATH_SUFFIXES lib64 lib
    DOC "Path to the static library file ${_CUDART_CUDART_STATIC_LIBRARY_NAME}."
)

set(_CUDART_CUDA_SONAME
    "${CMAKE_SHARED_LIBRARY_PREFIX}cuda${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
find_library(CUDART_CUDA_LIBRARY
    NAMES "${_CUDART_CUDA_SONAME}"
    HINTS ${_CUDART_HINTS}
    PATHS ${_CUDART_PATHS}
    PATH_SUFFIXES lib64 lib
    DOC "Path to the shared library file ${_CUDART_CUDA_SONAME}."
)

if(NOT CUDART_CUDA_LIBRARY)
    find_library(CUDART_CUDA_LIBRARY
        NAMES "${_CUDART_CUDA_SONAME}"
        HINTS ${_CUDART_HINTS}
        PATHS ${_CUDART_PATHS}
        PATH_SUFFIXES lib64/stubs lib/stubs
        DOC "Path to the shared library file ${_CUDART_CUDA_SONAME}."
    )
endif()

unset(_CUDART_CUDART_STATIC_LIBRARY_NAME)
unset(_CUDART_PATHS)
unset(_CUDART_HINTS)

find_package(Threads QUIET MODULE)

include(FindPackageHandleStandardArgs)

set(_CUDART_REQUIRED_VARS Threads_FOUND CUDART_INCLUDE_DIR CUDART_CUDA_LIBRARY)
if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL aarch64-unknown-nto-qnx-safety)
    list(APPEND _CUDART_REQUIRED_VARS CUDART_CUDART_STATIC_LIBRARY)
else()
    list(APPEND _CUDART_REQUIRED_VARS CUDART_CUDART_LIBRARY)
endif()

find_package_handle_standard_args(CUDART
    FOUND_VAR CUDART_FOUND
    REQUIRED_VARS ${_CUDART_REQUIRED_VARS}
    VERSION_VAR CUDART_VERSION
)

unset(_CUDART_REQUIRED_VARS)

if(CUDART_FOUND)
    set(CUDART_INCLUDE_DIRS "${CUDART_INCLUDE_DIR}")
    set(CUDART_LIBRARIES
        "${CUDART_CUDA_LIBRARY}"
        ${CMAKE_THREAD_LIBS_INIT}
    )
    mark_as_advanced(
        CUDART_INCLUDE_DIR
        CUDART_CUDA_LIBRARY
    )

    if(CUDART_CUDART_LIBRARY)
        list(APPEND CUDART_LIBRARIES "${CUDART_CUDART_LIBRARY}")
    else()
        list(APPEND CUDART_LIBRARIES "${CUDART_CUDART_STATIC_LIBRARY}")
    endif()

    if(NOT TARGET CUDART::cuda)
        add_library(CUDART::cuda SHARED IMPORTED)
        set_target_properties(CUDART::cuda PROPERTIES
            IMPORTED_LOCATION "${CUDART_CUDA_LIBRARY}"
            IMPORTED_SONAME "${_CUDART_CUDA_LIBRARY_SONAME}"
        )
    endif()

    if(CUDART_CUDART_LIBRARY)
        mark_as_advanced(CUDART_CUDART_LIBRARY)

        if(NOT TARGET CUDART::cudart)
            add_library(CUDART::cudart SHARED IMPORTED)
            set_target_properties(CUDART::cudart PROPERTIES
                IMPORTED_LOCATION "${CUDART_CUDART_LIBRARY}"
                IMPORTED_SONAME "${_CUDART_CUDART_SONAME}"
            )
            target_include_directories(CUDART::cudart SYSTEM INTERFACE
                "${CUDART_INCLUDE_DIR}"
            )
            target_link_libraries(CUDART::cudart INTERFACE Threads::Threads)
        endif()
    endif()

    if(CUDART_CUDART_STATIC_LIBRARY)
        mark_as_advanced(CUDART_CUDART_STATIC_LIBRARY)

        if(NOT TARGET CUDART::cudart_static)
            add_library(CUDART::cudart_static STATIC IMPORTED)
            set_target_properties(CUDART::cudart_static PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES CXX
                IMPORTED_LOCATION "${CUDART_CUDART_STATIC_LIBRARY}"
            )
            target_include_directories(CUDART::cudart_static SYSTEM INTERFACE
                "${CUDART_INCLUDE_DIR}"
            )
            target_link_libraries(CUDART::cudart_static INTERFACE Threads::Threads)
        endif()
    endif()

    if(NOT TARGET CUDART::CUDART)
        add_library(CUDART::CUDART INTERFACE IMPORTED)
        target_link_libraries(CUDART::CUDART INTERFACE CUDART::cuda)
        if(TARGET CUDART::cudart)
            target_link_libraries(CUDART::CUDART INTERFACE CUDART::cudart)
        else()
            target_link_libraries(CUDART::CUDART INTERFACE CUDART::cudart_static)
        endif()
    endif()
endif()

unset(_CUDART_CUDA_SONAME)
unset(_CUDART_CUDART_SONAME)

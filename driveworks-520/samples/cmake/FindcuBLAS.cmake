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

# FindcuBLAS
# ----------
#
# Finds the NVIDIA (R) CUDA (R) Basic Linear Algebra Subroutine (cuBLAS)
# libraries.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module provides the following imported targets, if found:
#
# ``cuBLAS::cuBLAS``
#   The cuBLAS libraries.
# ``cuBLAS::cublas``
#   The cuBLAS shared library.
# ``cuBLAS::cublas_static``
#   The cuBLAS static library.
# ``cuBLAS::cublasLt``
#   The cuBLAS lightweight shared library.
# ``cuBLAS::cublasLt_static``
#   The cuBLAS lightweight static library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This will define the following variables:
#
# ``cuBLAS_FOUND``
#   True if the system has the cuBLAS libraries.
# ``cuBLAS_VERSION``
#   The version of the cuBLAS libraries that was found.
# ``cuBLAS_INCLUDE_DIRS``
#   Include directories needed to use the cuBLAS libraries.
# ``cuBLAS_LIBRARIES``
#   Libraries needed to link to the cuBLAS libraries.
#
# Cache Variables
# ^^^^^^^^^^^^^^^
#
# The following cache variables may also be set:
#
# ``cuBLAS_CUBLAS_INCLUDE_DIR``
#   The directory containing ``cublas_v2.h``.
# ``cuBLAS_CUBLASLT_INCLUDE_DIR``
#   The directory containing ``cublasLt.h``.
# ``cuBLAS_CUBLAS_LIBRARY``
#   The path to ``libcublas.so``
# ``cuBLAS_CUBLAS_STATIC_LIBRARY``
#   The path to ``libcublas_static.a``.
# ``cuBLAS_CUBLASLT_LIBRARY``
#   The path to ``libcublasLt.so``.
# ``cuBLAS_CUBLASLT_STATIC_LIBRARY``
#   The path to ``libcublasLt_static.a``.

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    get_filename_component(_CUBLAS_PATHS
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" DIRECTORY
    )
else()
    set(_CUBLAS_PATHS)
endif()

if(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION)
    string(REPLACE "." ";" _CUBLAS_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR
        "${CMAKE_CUDA_COMPILER_TOOLKIT_VERSION}"
    )
    list(GET _CUBLAS_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR 0 1
        _CUBLAS_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR
    )
    list(JOIN _CUBLAS_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR .
        _CUBLAS_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR
    )

    if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL aarch64-linux-gnu)
        set(_CUBLAS_LIBRARY_ARCHITECTURE aarch64-linux)
    elseif(CMAKE_LIBRARY_ARCHITECTURE STREQUAL aarch64-unknown-nto-qnx)
        set(_CUBLAS_LIBRARY_ARCHITECTURE aarch64-qnx)
    elseif(CMAKE_LIBRARY_ARCHITECTURE STREQUAL aarch64-unknown-nto-qnx-safety)
        set(_CUBLAS_LIBRARY_ARCHITECTURE aarch64-qnx-safe)
    else()
        set(_CUBLAS_LIBRARY_ARCHITECTURE x86_64-linux)
    endif()

    if(CMAKE_LIBRARY_ARCHITECTURE MATCHES "^aarch64-unknown-nto-qnx")
        list(APPEND _CUBLAS_PATHS
            "/usr/local/cuda-safe-${_CUBLAS_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR}/targets/${_CUBLAS_LIBRARY_ARCHITECTURE}"
        )
    endif()

    list(APPEND _CUBLAS_PATHS
        "/usr/local/cuda-${_CUBLAS_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR}/targets/${_CUBLAS_LIBRARY_ARCHITECTURE}"
    )
    unset(_CUBLAS_LIBRARY_ARCHITECTURE)
    unset(_CUBLAS_CUDA_COMPILER_TOOLKIT_VERSION_MAJOR_MINOR)

    set(cuBLAS_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")
else()
    set(cuBLAS_VERSION)
endif()

find_path(cuBLAS_CUBLAS_INCLUDE_DIR
    NAMES cublas_v2.h
    PATHS ${_CUBLAS_PATHS}
    PATH_SUFFIXES include
    DOC "Path to the directory containing the header file cublas_v2.h."
)

find_path(cuBLAS_CUBLASLT_INCLUDE_DIR
    NAMES cublasLt.h
    PATHS ${_CUBLAS_PATHS}
    PATH_SUFFIXES include
    DOC "Path to the directory containing the header file cublasLt.h."
)

set(_CUBLAS_CUBLAS_SONAME
    "${CMAKE_SHARED_LIBRARY_PREFIX}cublas${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
find_library(cuBLAS_CUBLAS_LIBRARY
    NAMES "${_CUBLAS_CUBLAS_SONAME}"
    PATHS ${_CUBLAS_PATHS}
    PATH_SUFFIXES lib64 lib
    DOC "Path to the shared library file ${_CUBLAS_CUBLAS_SONAME}."
)

if(cuBLAS_CUBLAS_LIBRARY)
    get_filename_component(_CUBLAS_HINTS "${cuBLAS_CUBLAS_LIBRARY}" DIRECTORY)
else()
    set(_CUBLAS_HINTS)
endif()

if(NOT cuBLAS_CUBLAS_LIBRARY)
    find_library(cuBLAS_CUBLAS_LIBRARY
        NAMES "${_CUBLAS_CUBLAS_SONAME}"
        PATHS ${_CUBLAS_PATHS}
        PATH_SUFFIXES lib64/stubs lib/stubs
        DOC "Path to the shared library file ${_CUBLAS_CUBLAS_SONAME}."
    )

    if(cuBLAS_CUBLAS_LIBRARY)
        get_filename_component(_CUBLAS_HINTS "${cuBLAS_CUBLAS_LIBRARY}"
            DIRECTORY
        )
        get_filename_component(_CUBLAS_HINTS "${_CUBLAS_HINTS}" DIRECTORY)
    endif()
endif()

set(_CUBLAS_CUBLAS_STATIC_LIBRARY_NAME
    "${CMAKE_STATIC_LIBRARY_PREFIX}cublas_static${CMAKE_STATIC_LIBRARY_SUFFIX}"
)
find_library(cuBLAS_CUBLAS_STATIC_LIBRARY
    NAMES "${_CUBLAS_CUBLAS_STATIC_LIBRARY_NAME}"
    HINTS ${_CUBLAS_HINTS}
    PATHS ${_CUBLAS_PATHS}
    PATH_SUFFIXES lib64 lib
    DOC "Path to the static library file ${_CUBLAS_CUBLAS_STATIC_LIBRARY_NAME}."
)

set(_CUBLAS_CUBLASLT_SONAME
    "${CMAKE_SHARED_LIBRARY_PREFIX}cublasLt${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
find_library(cuBLAS_CUBLASLT_LIBRARY
    NAMES "${_CUBLAS_CUBLASLT_SONAME}"
    HINTS ${_CUBLAS_HINTS}
    PATHS ${_CUBLAS_PATHS}
    PATH_SUFFIXES lib64 lib
    DOC "Path to the shared library file ${_CUBLAS_CUBLASLT_SONAME}."
)

if(NOT cuBLAS_CUBLASLT_LIBRARY)
    find_library(cuBLAS_CUBLASLT_LIBRARY
        NAMES "${_CUBLAS_CUBLASLT_SONAME}"
        HINTS ${_CUBLAS_HINTS}
        PATHS ${_CUBLAS_PATHS}
        PATH_SUFFIXES lib64/stubs lib/stubs
        DOC "Path to the shared library file ${_CUBLAS_CUBLASLT_SONAME}."
    )
endif()

set(_CUBLAS_CUBLASLY_STATIC_LIBRARY_NAME
    "${CMAKE_STATIC_LIBRARY_PREFIX}cublasLt_static${CMAKE_STATIC_LIBRARY_SUFFIX}"
)
find_library(cuBLAS_CUBLASLT_STATIC_LIBRARY
    NAMES "${_CUBLAS_CUBLASLT_STATIC_LIBRARY_NAME}"
    HINTS ${_CUBLAS_HINTS}
    PATHS ${_CUBLAS_PATHS}
    PATH_SUFFIXES lib64 lib
    DOC "Path to the static library file ${_CUBLAS_CUBLASLT_STATIC_LIBRARY_NAME}."
)

unset(_CUBLAS_CUBLAS_STATIC_LIBRARY_NAME)
unset(_CUBLAS_CUBLASLT_STATIC_LIBRARY_NAME)
unset(_CUBLAS_PATHS)
unset(_CUBLAS_HINTS)

find_package(Threads QUIET MODULE)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(cuBLAS
    FOUND_VAR cuBLAS_FOUND
    REQUIRED_VARS
        cuBLAS_CUBLAS_INCLUDE_DIR
        cuBLAS_CUBLASLT_INCLUDE_DIR
        cuBLAS_CUBLAS_LIBRARY
        cuBLAS_CUBLASLT_LIBRARY
        Threads_FOUND
    VERSION_VAR cuBLAS_VERSION
)

if(cuBLAS_FOUND)
    set(cuBLAS_INCLUDE_DIRS
        "${cuBLAS_CUBLAS_INCLUDE_DIR}"
        "${cuBLAS_CUBLASLT_INCLUDE_DIR}"
    )
    set(cuBLAS_LIBRARIES
        "${cuBLAS_CUBLAS_LIBRARY}"
        "${cuBLAS_CUBLASLT_LIBRARY}"
        ${CMAKE_DL_LIBS}
        ${CMAKE_THREAD_LIBS_INIT}
    )
    mark_as_advanced(
        cuBLAS_CUBLAS_INCLUDE_DIR
        cuBLAS_CUBLASLT_INCLUDE_DIR
        cuBLAS_CUBLAS_LIBRARY
        cuBLAS_CUBLASLT_LIBRARY
    )

    if(NOT TARGET cuBLAS::cublas)
        add_library(cuBLAS::cublas SHARED IMPORTED)
        set_target_properties(cuBLAS::cublas PROPERTIES
            IMPORTED_LOCATION "${cuBLAS_CUBLAS_LIBRARY}"
            IMPORTED_SONAME "${_CUBLAS_CUBLAS_SONAME}"
        )
        target_include_directories(cuBLAS::cublas SYSTEM INTERFACE
            "${cuBLAS_CUBLAS_INCLUDE_DIR}"
        )
        target_link_libraries(cuBLAS::cublas INTERFACE
            Threads::Threads
            ${CMAKE_DL_LIBS}
        )
    endif()

    if(cuBLAS_CUBLAS_STATIC_LIBRARY)
        mark_as_advanced(cuBLAS_CUBLAS_STATIC_LIBRARY)

        if(NOT TARGET cuBLAS::cublas_static)
            add_library(cuBLAS::cublas_static STATIC IMPORTED)
            set_target_properties(cuBLAS::cublas_static PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES CXX
                IMPORTED_LOCATION "${cuBLAS_CUBLAS_STATIC_LIBRARY}"
            )
            target_include_directories(cuBLAS::cublas_static SYSTEM INTERFACE
                "${cuBLAS_CUBLAS_INCLUDE_DIR}"
            )
            target_link_libraries(cuBLAS::cublas_static INTERFACE
                Threads::Threads
                ${CMAKE_DL_LIBS}
            )
        endif()
    endif()

    if(NOT TARGET cuBLAS::cublasLt)
        add_library(cuBLAS::cublasLt SHARED IMPORTED)
        set_target_properties(cuBLAS::cublasLt PROPERTIES
            IMPORTED_LOCATION "${cuBLAS_CUBLASLT_LIBRARY}"
            IMPORTED_SONAME "${_CUBLAS_CUBLASLT_LIBRARY_SONAME}"
        )
        target_include_directories(cuBLAS::cublasLt SYSTEM INTERFACE
            "${cuBLAS_CUBLASLT_INCLUDE_DIR}"
        )
        target_link_libraries(cuBLAS::cublasLt INTERFACE
            Threads::Threads
            ${CMAKE_DL_LIBS}
        )
    endif()

    if(cuBLAS_CUBLASLT_STATIC_LIBRARY)
        mark_as_advanced(cuBLAS_CUBLASLT_STATIC_LIBRARY)

        if(NOT TARGET cuBLAS::cublasLt_static)
            add_library(cuBLAS::cublasLt_static STATIC IMPORTED)
            set_target_properties(cuBLAS::cublasLt_static PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES CXX
                IMPORTED_LOCATION "${cuBLAS_CUBLASLT_STATIC_LIBRARY}"
            )
            target_include_directories(cuBLAS::cublasLt_static SYSTEM INTERFACE
                "${cuBLAS_CUBLASLT_INCLUDE_DIR}"
            )
            target_link_libraries(cuBLAS::cublasLt_static INTERFACE
                Threads::Threads
                ${CMAKE_DL_LIBS}
            )
        endif()
    endif()

    if(NOT TARGET cuBLAS::cuBLAS)
        add_library(cuBLAS::cuBLAS INTERFACE IMPORTED)
        target_link_libraries(cuBLAS::cuBLAS INTERFACE
            cuBLAS::cublas
            cuBLAS::cublasLt
        )
    endif()
endif()

unset(_CUBLAS_CUBLASLT_SONAME)
unset(_CUBLAS_CUBLAS_SONAME)

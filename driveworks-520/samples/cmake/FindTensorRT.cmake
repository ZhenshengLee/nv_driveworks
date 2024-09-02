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

# FindcuDNN
# ---------
#
# Finds the NVIDIA (R) TensorRT (TM) SDK for high-performance deep learning
# inference.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module provides the following imported targets, if found:
#
# ``TensorRT::TensorRT``
#   The TensorRT libraries.
# ``TensorRT::nvinfer``
#   The nvinfer shared library.
# ``TensorRT::nvinfer_static``
#   The nvinfer static library.
# ``TensorRT::nvinfer_safe``
#   The nvinfer_safe shared library.
# ``TensorRT::nvinfer_safe_static``
#   The nvinfer_safe static library.
# ``TensorRT::nvonnxparser``
#   The nvonnxparser shared library.
# ``TensorRT::nvonnxparser_static``
#   The nvonnxparser static library.
# ``TensorRT::nvparsers``
#   The nvparsers shared library.
# ``TensorRT::nvparsers_static``
#   The nvparsers static library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This will define the following variables:
#
# ``TensorRT_FOUND``
#   True if the system has the TensorRT libraries.
# ``TensorRT_VERSION``
#   The version of the TensorRT libraries that were found.
# ``TensorRT_INCLUDE_DIRS``
#   Include directories needed to use TensorRT.
# ``TensorRT_LIBRARIES``
#   Libraries needed to link to TensorRT.
#
# Cache Variables
# ^^^^^^^^^^^^^^^
#
# The following cache variables may also be set:
#
# ``TensorRT_INCLUDE_DIR``
#   The directory containing ``NvInferVersion.h``.
# ``TensorRT_nvinfer_LIBRARY``
#   The path to ``libnvinfer.so``.
# ``TensorRT_nvinfer_STATIC_LIBRARY``
#   The path to ``libnvinfer_static.a``.
# ``TensorRT_nvinfer_safe_LIBRARY``
#   The path to ``libnvinfer_safe.so``.
# ``TensorRT_nvinfer_safe_STATIC_LIBRARY``
#   The path to ``libnvinfer_safe_static.a``.
# ``TensorRT_nvonnxparser_LIBRARY``
#   The path to ``libnvonnxparser.so``.
# ``TensorRT_nvonnxparser_STATIC_LIBRARY``
#   The path to ``libnvonnxparser_static.a``
# ``TensorRT_nvparsers_LIBRARY``
#   The path to ``libnvparsers.so``.
# ``TensorRT_nvonnxparser_STATIC_LIBRARY``
#   The path to ``libnvparsers_static.a``

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInferVersion.h
    PATH_SUFFIXES "${CMAKE_LIBRARY_ARCHITECTURE}"
    DOC "Path to the directory containing the header file NvInferVersion.h."
)

set(_TensorRT_VERSION_FILEPATH "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
if(TensorRT_INCLUDE_DIR AND EXISTS "${_TensorRT_VERSION_FILEPATH}")
    file(STRINGS "${_TensorRT_VERSION_FILEPATH}" _TensorRT_VERSION_STRINGS
        REGEX "^#define[ ]+NV_TENSORRT_(MAJOR|MINOR|PATCH|BUILD)[ ]+[0-9]+.*$"
    )
    if(_TensorRT_VERSION_STRINGS)
        string(REGEX REPLACE ".*#define[ ]+NV_TENSORRT_MAJOR[ ]+([0-9]+).*" "\\1"
            TensorRT_VERSION_MAJOR "${_TensorRT_VERSION_STRINGS}"
        )
        string(REGEX REPLACE ".*#define[ ]+NV_TENSORRT_MINOR[ ]+([0-9]+).*" "\\1"
            TensorRT_VERSION_MINOR "${_TensorRT_VERSION_STRINGS}"
        )
        string(REGEX REPLACE ".*#define[ ]+NV_TENSORRT_PATCH[ ]+([0-9]+).*" "\\1"
            TensorRT_VERSION_PATCH "${_TensorRT_VERSION_STRINGS}"
        )
        string(REGEX REPLACE ".*#define[ ]+NV_TENSORRT_BUILD[ ]+([0-9]+).*" "\\1"
            TensorRT_VERSION_BUILD "${_TensorRT_VERSION_STRINGS}"
        )
        if(TensorRT_VERSION_MAJOR
            AND TensorRT_VERSION_MINOR
            AND TensorRT_VERSION_PATCH
            AND TensorRT_VERSION_BUILD
        )
            set(TensorRT_VERSION
                "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}.${TensorRT_VERSION_BUILD}"
            )
        else()
            set(TensorRT_VERSION)
        endif()
    else()
        set(TensorRT_VERSION)
    endif()
    unset(_TensorRT_VERSION_STRINGS)
else()
    set(TensorRT_VERSION)
endif()
unset(_TensorRT_VERSION_FILEPATH)

set(_TensorRT_nvinfer_SONAME
    "${CMAKE_SHARED_LIBRARY_PREFIX}nvinfer${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
find_library(TensorRT_nvinfer_LIBRARY
    NAMES "${_TensorRT_nvinfer_SONAME}"
    PATH_SUFFIXES
        "${CMAKE_LIBRARY_ARCHITECTURE}/stubs"
        "${CMAKE_LIBRARY_ARCHITECTURE}"
    DOC "Path to the shared library file ${_TensorRT_nvinfer_SONAME}."
)
get_filename_component(_TensorRT_LIBRARY_DIR "${TensorRT_nvinfer_LIBRARY}" DIRECTORY)

set(_TensorRT_nvinfer_STATIC_LIBRARY_NAME
    "${CMAKE_STATIC_LIBRARY_PREFIX}nvinfer{CMAKE_STATIC_LIBRARY_SUFFIX}"
)
find_library(TensorRT_nvinfer_STATIC_LIBRARY
    NAMES "${_TensorRT_nvinfer_STATIC_LIBRARY_NAME}"
    PATH_SUFFIXES "${CMAKE_LIBRARY_ARCHITECTURE}"
    DOC "Path to the static library file ${_TensorRT_nvinfer_STATIC_LIBRARY_NAME}."
)
unset(_TensorRT_nvinfer_STATIC_LIBRARY_NAME)

set(_TensorRT_nvinfer_safe_SONAME
    "${CMAKE_SHARED_LIBRARY_PREFIX}nvinfer_safe${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
find_library(TensorRT_nvinfer_safe_LIBRARY
    NAMES "${_TensorRT_nvinfer_safe_SONAME}"
    PATH_SUFFIXES
        "${CMAKE_LIBRARY_ARCHITECTURE}/stubs"
        "${CMAKE_LIBRARY_ARCHITECTURE}"
    DOC "Path to the shared library file ${_TensorRT_nvinfer_safe_SONAME}."
)

set(_TensorRT_nvinfer_safe_STATIC_LIBRARY_NAME
    "${CMAKE_STATIC_LIBRARY_PREFIX}nvinfer_safe${CMAKE_STATIC_LIBRARY_SUFFIX}"
)
find_library(TensorRT_nvinfer_safe_STATIC_LIBRARY
    NAMES "${_TensorRT_nvinfer_safe_STATIC_LIBRARY_NAME}"
    PATH_SUFFIXES "${CMAKE_LIBRARY_ARCHITECTURE}"
    DOC "Path to the static library file ${_TensorRT_nvinfer_safe_STATIC_LIBRARY_NAME}."
)
unset(_TensorRT_nvinfer_safe_STATIC_LIBRARY_NAME)

set(_TensorRT_nvonnxparser_SONAME
    "${CMAKE_SHARED_LIBRARY_PREFIX}nvonnxparser${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
find_library(TensorRT_nvonnxparser_LIBRARY
    NAMES "${_TensorRT_nvonnxparser_SONAME}"
    PATH_SUFFIXES
        "${CMAKE_LIBRARY_ARCHITECTURE}/stubs"
        "${CMAKE_LIBRARY_ARCHITECTURE}"
    HINTS "${_TensorRT_LIBRARY_DIR}"
    DOC "Path to the shared library file ${_TensorRT_nvonnxparser_SONAME}."
)

set(_TensorRT_nvonnxparser_STATIC_LIBRARY_NAME
    "${CMAKE_STATIC_LIBRARY_PREFIX}nvonnxparser${CMAKE_STATIC_LIBRARY_SUFFIX}"
)
find_library(TensorRT_nvonnxparser_STATIC_LIBRARY
    NAMES "${_TensorRT_nvonnxparser_STATIC_LIBRARY_NAME}"
    PATH_SUFFIXES "${CMAKE_LIBRARY_ARCHITECTURE}"
    DOC "Path to the static library file ${_TensorRT_nvonnxparser_STATIC_LIBRARY_NAME}."
)
unset(_TensorRT_nvonnxparser_STATIC_LIBRARY_NAME)

set(_TensorRT_nvparsers_SONAME
    "${CMAKE_SHARED_LIBRARY_PREFIX}nvparsers${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
find_library(TensorRT_nvparsers_LIBRARY
    NAMES "${_TensorRT_nvparsers_SONAME}"
    PATH_SUFFIXES
        "${CMAKE_LIBRARY_ARCHITECTURE}/stubs"
        "${CMAKE_LIBRARY_ARCHITECTURE}"
    HINTS "${_TensorRT_LIBRARY_DIR}"
    DOC "Path to the shared library file ${_TensorRT_nvparsers_SONAME}."
)

set(_TensorRT_nvparsers_STATIC_LIBRARY_NAME
    "${CMAKE_STATIC_LIBRARY_PREFIX}nvparsers${CMAKE_STATIC_LIBRARY_SUFFIX}"
)
find_library(TensorRT_nvparsers_STATIC_LIBRARY
    NAMES "${_TensorRT_nvparsers_STATIC_LIBRARY_NAME}"
    PATH_SUFFIXES "${CMAKE_LIBRARY_ARCHITECTURE}"
    DOC "Path to the static library file ${TensorRT_nvparsers_STATIC_LIBRARY}."
)
unset(_TensorRT_nvparsers_STATIC_LIBRARY_NAME)

find_package(Threads QUIET MODULE)

include(FindPackageHandleStandardArgs)

set(_TensorRT_REQUIRED_VARS Threads_FOUND TensorRT_INCLUDE_DIR)
if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL aarch64-unknown-nto-qnx-safety)
    list(APPEND _TensorRT_REQUIRED_VARS TensorRT_nvinfer_safe_LIBRARY)
else()
    list(APPEND _TensorRT_REQUIRED_VARS
        TensorRT_nvinfer_LIBRARY
        TensorRT_nvonnxparser_LIBRARY
        TensorRT_nvparsers_LIBRARY
    )
endif()

find_package_handle_standard_args(TensorRT
    FOUND_VAR TensorRT_FOUND
    REQUIRED_VARS ${_TensorRT_REQUIRED_VARS}
    VERSION_VAR TensorRT_VERSION
)

if(TensorRT_FOUND)
    set(TensorRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}")
    set(TensorRT_LIBRARIES ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT})
    mark_as_advanced(TensorRT_INCLUDE_DIR)

    if(TensorRT_nvinfer_LIBRARY)
        mark_as_advanced(TensorRT_nvinfer_LIBRARY)
        list(APPEND TensorRT_LIBRARIES "${TensorRT_nvinfer_LIBRARY}")
        if(NOT TARGET TensorRT::nvinfer)
            add_library(TensorRT::nvinfer SHARED IMPORTED)
            target_include_directories(TensorRT::nvinfer SYSTEM INTERFACE
                "${TensorRT_INCLUDE_DIR}"
            )
            target_link_libraries(TensorRT::nvinfer INTERFACE
                ${CMAKE_DL_LIBS}
            )
            set_target_properties(TensorRT::nvinfer PROPERTIES
                IMPORTED_LOCATION "${TensorRT_nvinfer_LIBRARY}"
                IMPORTED_SONAME "${_TensorRT_nvinfer_SONAME}"
            )
            if(TensorRT_VERSION AND TensorRT_VERSION)
                set_target_properties(TensorRT::nvinfer PROPERTIES
                    SOVERSION "${TensorRT_VERSION_MAJOR}"
                    VERSION "${TensorRT_VERSION}"
                )
            endif()
        endif()
    endif()

    if(TensorRT_nvinfer_STATIC_LIBRARY)
        mark_as_advanced(TensorRT_nvinfer_STATIC_LIBRARY)
        if(NOT TARGET TensorRT::nvinfer_static)
            add_library(TensorRT::nvinfer_static STATIC IMPORTED)
            target_include_directories(TensorRT::nvinfer_static SYSTEM
                INTERFACE "${TensorRT_INCLUDE_DIR}"
            )
            target_link_libraries(TensorRT::nvinfer_static INTERFACE
                ${CMAKE_DL_LIBS}
            )
            set_target_properties(TensorRT::nvinfer_static PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES CXX
                IMPORTED_LOCATION "${TensorRT_nvinfer_STATIC_LIBRARY}"
            )
        endif()
    endif()

    if(TensorRT_nvinfer_safe_LIBRARY)
        mark_as_advanced(TensorRT_nvinfer_safe_LIBRARY)
        list(APPEND TensorRT_LIBRARIES "${TensorRT_nvinfer_safe_LIBRARY}")
        if(NOT TARGET TensorRT::nvinfer_safe)
            add_library(TensorRT::nvinfer_safe SHARED IMPORTED)
            target_include_directories(TensorRT::nvinfer_safe SYSTEM INTERFACE
                "${TensorRT_INCLUDE_DIR}"
            )
            set_target_properties(TensorRT::nvinfer_safe PROPERTIES
                IMPORTED_LOCATION "${TensorRT_nvinfer_safe_LIBRARY}"
                IMPORTED_SONAME "${_TensorRT_nvinfer_safe_SONAME}"
            )
            if(TensorRT_VERSION AND TensorRT_VERSION)
                set_target_properties(TensorRT::nvinfer_safe PROPERTIES
                    SOVERSION "${TensorRT_VERSION_MAJOR}"
                    VERSION "${TensorRT_VERSION}"
                )
            endif()
        endif()
    endif()

    if(TensorRT_nvinfer_safe_STATIC_LIBRARY)
        mark_as_advanced(TensorRT_nvinfer_safe_STATIC_LIBRARY)
        if(NOT TARGET TensorRT::nvinfer_safe_static)
            add_library(TensorRT::nvinfer_safe_static STATIC IMPORTED)
            target_include_directories(TensorRT::nvinfer_safe_static SYSTEM
                INTERFACE "${TensorRT_INCLUDE_DIR}"
            )
            set_target_properties(TensorRT::nvinfer_safe_static PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES CXX
                IMPORTED_LOCATION "${TensorRT_nvinfer_safe_STATIC_LIBRARY}"
            )
        endif()
    endif()

    if(TensorRT_nvonnxparser_LIBRARY)
        mark_as_advanced(TensorRT_nvonnxparser_LIBRARY)
        list(APPEND TensorRT_LIBRARIES "${TensorRT_nvonnxparser_LIBRARY}")
        if(NOT TARGET TensorRT::nvonnxparser)
            add_library(TensorRT::nvonnxparser SHARED IMPORTED)
            target_include_directories(TensorRT::nvonnxparser SYSTEM INTERFACE
                "${TensorRT_INCLUDE_DIR}"
            )
            target_link_libraries(TensorRT::nvonnxparser INTERFACE
                TensorRT::nvinfer
            )
            set_target_properties(TensorRT::nvonnxparser PROPERTIES
                IMPORTED_LOCATION "${TensorRT_nvonnxparser_LIBRARY}"
                IMPORTED_SONAME "${_TensorRT_nvonnxparser_SONAME}"
            )
            if(TensorRT_VERSION AND TensorRT_VERSION)
                set_target_properties(TensorRT::nvonnxparser PROPERTIES
                    SOVERSION "${TensorRT_VERSION_MAJOR}"
                    VERSION "${TensorRT_VERSION}"
                )
            endif()
        endif()
    endif()

    if(TensorRT_nvonnxparser_STATIC_LIBRARY)
        mark_as_advanced(TensorRT_nvonnxparser_STATIC_LIBRARY)
        if(NOT TARGET TensorRT::nvonnxparser_static)
            add_library(TensorRT::nvonnxparser_static STATIC IMPORTED)
            target_include_directories(TensorRT::nvonnxparser_static SYSTEM
                INTERFACE "${TensorRT_INCLUDE_DIR}"
            )
            target_link_libraries(TensorRT::nvonnxparser_static INTERFACE
                TensorRT::nvinfer
            )
            set_target_properties(TensorRT::nvonnxparser_static PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES CXX
                IMPORTED_LOCATION "${TensorRT_nvonnxparser_STATIC_LIBRARY}"
            )
        endif()
    endif()

    if(TensorRT_nvparsers_LIBRARY)
        mark_as_advanced(TensorRT_nvparsers_LIBRARY)
        if(NOT TARGET TensorRT::nvparsers)
            add_library(TensorRT::nvparsers SHARED IMPORTED)
            target_include_directories(TensorRT::nvparsers SYSTEM INTERFACE
                "${TensorRT_INCLUDE_DIR}"
            )
            target_link_libraries(TensorRT::nvparsers INTERFACE
                TensorRT::nvinfer
            )
            set_target_properties(TensorRT::nvparsers PROPERTIES
                IMPORTED_LOCATION "${TensorRT_nvparsers_LIBRARY}"
                IMPORTED_SONAME "${_TensorRT_nvparsers_SONAME}"
            )
            if(TensorRT_VERSION AND TensorRT_VERSION)
                set_target_properties(TensorRT::nvparsers PROPERTIES
                    SOVERSION "${TensorRT_VERSION_MAJOR}"
                    VERSION "${TensorRT_VERSION}"
                )
            endif()
        endif()
    endif()

    if(TensorRT_nvparsers_STATIC_LIBRARY)
        mark_as_advanced(TensorRT_nvparsers_STATIC_LIBRARY)
        if(NOT TARGET TensorRT::nvparsers_static)
            add_library(TensorRT::nvparsers_static STATIC IMPORTED)
            target_include_directories(TensorRT::nvparsers_static SYSTEM
                INTERFACE "${TensorRT_INCLUDE_DIR}"
            )
            target_link_libraries(TensorRT::nvparsers_static INTERFACE
                TensorRT::nvinfer
            )
            set_target_properties(TensorRT::nvparsers_static PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES CXX
                IMPORTED_LOCATION "${TensorRT_nvparsers_STATIC_LIBRARY}"
            )
        endif()
    endif()

    if(NOT TARGET TensorRT::TensorRT)
        add_library(TensorRT::TensorRT INTERFACE IMPORTED)
        target_include_directories(TensorRT::TensorRT SYSTEM INTERFACE
            "${TensorRT_INCLUDE_DIR}"
        )
        if(TARGET TensorRT::nvinfer)
            target_link_libraries(TensorRT::TensorRT INTERFACE
                TensorRT::nvinfer
            )
        endif()
        if(TARGET TensorRT::nvinfer_safe)
            target_link_libraries(TensorRT::TensorRT INTERFACE
                TensorRT::nvinfer_safe
            )
        endif()
        if(TARGET TensorRT::nvonnxparser)
            target_link_libraries(TensorRT::TensorRT INTERFACE
                TensorRT::nvonnxparser
            )
        endif()
        if(TARGET TensorRT::nvparsers)
            target_link_libraries(TensorRT::TensorRT INTERFACE
                TensorRT::nvparsers
            )
        endif()
    endif()
endif()

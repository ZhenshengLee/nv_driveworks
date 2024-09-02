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
# SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# FindGLES
# --------
#
# Finds the OpenGL ES library.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module provides the following imported targets, if found:
#
# ``GLES::GLES``
#   The GLES library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This will define the following variables:
#
# ``GLES_FOUND``
#   True if the system has the GLES library.
# ``GLES_VERSION``
#   The version of the GLES library which was found.
# ``GLES_INCLUDE_DIRS``
#   Include directories needed to use GLES.
# ``GLES_LIBRARIES``
#   Libraries needed to link to GLES.
#
# Cache Variables
# ^^^^^^^^^^^^^^^
#
# The following cache variables may also be set:
#
# ``GLES_GLES2_INCLUDE_DIR``
#   The directory containing ``GLES2/gl2ext.h``.
# ``GLES_GLES3_INCLUDE_DIR``
#   The directory containing ``GLES3/gl31.h``.
# ``GLES_LIBRARY``
#   The path to the GLESv2 library.

set(GLES_VERSION)

set(_GLES_INCLUDE_DIRS)
set(_GLES_LIBRARY_DIRS)

if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL x86_64-linux-gnu)
    find_package(PkgConfig MODULE QUIET)
    pkg_check_modules(_PC_GLES QUIET glesv2)

    if(_PC_GLES_FOUND)
        set(GLES_VERSION "${_PC_GLES_VERSION}")

        set(_GLES_INCLUDE_DIRS "${_PC_GLES_INCLUDE_DIRS}")
        set(_GLES_LIBRARY_DIRS "${_PC_GLES_LIBRARY_DIRS}")
    endif()
endif()

find_path(GLES_GLES2_INCLUDE_DIR
    NAMES GLES2/gl2ext.h
    PATHS "${_GLES_INCLUDE_DIRS}"
    DOC "Path to the directory containing the header file GLES2/gl2ext.h."
)

find_path(GLES_GLES3_INCLUDE_DIR
    NAMES GLES3/gl31.h
    PATHS "${_GLES_INCLUDE_DIRS}"
    DOC "Path to the directory containing the header file GLES3/gl31.h."
)

unset(_GLES_INCLUDE_DIRS)

set(_GLES_LIBRARY_SONAME
    "${CMAKE_SHARED_LIBRARY_PREFIX}GLESv2${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

find_library(GLES_LIBRARY
    NAMES "${_GLES_LIBRARY_SONAME}"
    PATHS "${_GLES_LIBRARY_DIRS}"
    DOC "Path to the shared library file ${_GLES_LIBRARY_SONAME}."
)

unset(_GLES_LIBRARY_DIRS)

find_package(Threads QUIET MODULE)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(GLES
    FOUND_VAR GLES_FOUND
    REQUIRED_VARS
        GLES_GLES2_INCLUDE_DIR
        GLES_GLES3_INCLUDE_DIR
        GLES_LIBRARY
        Threads_FOUND
    VERSION_VAR GLES_VERSION
)

if(GLES_FOUND)
    set(GLES_INCLUDE_DIRS
        "${GLES_GLES2_INCLUDE_DIR}"
        "${GLES_GLES3_INCLUDE_DIR}"
    )
    set(GLES_LIBRARIES "${GLES_LIBRARY}" ${CMAKE_THREAD_LIBS_INIT})
    mark_as_advanced(
        GLES_GLES2_INCLUDE_DIR
        GLES_GLES3_INCLUDE_DIR
        GLES_LIBRARY
    )

    if(NOT TARGET GLES::GLES)
        add_library(GLES::GLES SHARED IMPORTED)
        set_target_properties(GLES::GLES PROPERTIES
            IMPORTED_LOCATION "${GLES_LIBRARY}"
            IMPORTED_SONAME "${_GLES_LIBRARY_SONAME}"
        )
        target_include_directories(GLES::GLES SYSTEM INTERFACE
            "${GLES_INCLUDE_DIRS}"
        )
        target_link_libraries(GLES::GLES INTERFACE Threads::Threads)
    endif()
endif()

unset(_GLES_LIBRARY_SONAME)

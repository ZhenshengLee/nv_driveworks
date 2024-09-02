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
# SPDX-FileCopyrightText: Copyright (c) 2016-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# FindEGL
# -------
#
# Finds the EGL library.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module provides the following imported targets, if found:
#
# ``EGL::EGL``
#   The EGL library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This will define the following variables:
#
# ``EGL_FOUND``
#   True if the system has the EGL library.
# ``EGL_VERSION``
#   The version of the EGL library which was found.
# ``EGL_INCLUDE_DIRS``
#   Include directories needed to use EGL.
# ``EGL_LIBRARIES``
#   Libraries needed to link to EGL.
#
# Cache Variables
# ^^^^^^^^^^^^^^^
#
# The following cache variables may also be set:
#
# ``EGL_INCLUDE_DIR``
#   The directory containing ``EGL/egl.h``.
# ``EGL_LIBRARY``
#   The path to the EGL library.

set(EGL_VERSION)

set(_EGL_INCLUDE_DIRS)
set(_EGL_LIBRARY_DIRS)

if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL x86_64-linux-gnu)
    find_package(PkgConfig MODULE QUIET)
    pkg_check_modules(_PC_EGL QUIET egl)

    if(_PC_EGL_FOUND)
        set(EGL_VERSION "${_PC_EGL_VERSION}")

        set(_EGL_INCLUDE_DIRS "${_PC_EGL_INCLUDE_DIRS}")
        set(_EGL_LIBRARY_DIRS "${_PC_EGL_LIBRARY_DIRS}")
    endif()
endif()

find_path(EGL_INCLUDE_DIR
    NAMES EGL/egl.h
    PATHS "${_EGL_INCLUDE_DIRS}"
    DOC "Path to the directory containing the header file EGL/egl.h."
)

unset(_EGL_INCLUDE_DIRS)

set(_EGL_LIBRARY_SONAME
    "${CMAKE_SHARED_LIBRARY_PREFIX}EGL${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

find_library(EGL_LIBRARY
    NAMES "${_EGL_LIBRARY_SONAME}"
    PATHS "${_EGL_LIBRARY_DIRS}"
    DOC "Path to the shared library file ${_EGL_LIBRARY_SONAME}."
)

unset(_EGL_LIBRARY_DIRS)

find_package(Threads QUIET MODULE)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(EGL
    FOUND_VAR EGL_FOUND
    REQUIRED_VARS EGL_INCLUDE_DIR EGL_LIBRARY Threads_FOUND
    VERSION_VAR EGL_VERSION
)

if(EGL_FOUND)
    set(EGL_DEFINITIONS -DDW_USE_EGL)
    set(EGL_INCLUDE_DIRS "${EGL_INCLUDE_DIR}")
    set(EGL_LIBRARIES "${EGL_LIBRARY}" ${CMAKE_THREAD_LIBS_INIT})
    mark_as_advanced(EGL_INCLUDE_DIR EGL_LIBRARY)

    if(NOT TARGET EGL::EGL)
        add_library(EGL::EGL SHARED IMPORTED)
        set_target_properties(EGL::EGL PROPERTIES
            IMPORTED_LOCATION "${EGL_LIBRARY}"
            IMPORTED_SONAME "${_EGL_LIBRARY_SONAME}"
        )
        target_compile_definitions(EGL::EGL INTERFACE DW_USE_EGL)
        target_include_directories(EGL::EGL SYSTEM INTERFACE
            "${EGL_INCLUDE_DIRS}"
        )
        target_link_libraries(EGL::EGL INTERFACE Threads::Threads)
    endif()
endif()

unset(_EGL_LIBRARY_SONAME)

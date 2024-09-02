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

# FindDRM
# -------
#
# Finds the libdrm userspace interface to kernel DRM services.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module provides the following imported targets, if found:
#
# ``DRM::DRM``
#   The DRM library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This will define the following variables:
#
# ``DRM_FOUND``
#   True if the system has the DRM library.
# ``DRM_VERSION``
#   The version of the DRM library which was found.
# ``DRM_INCLUDE_DIRS``
#   Include directories needed to use DRM.
# ``DRM_LIBRARIES``
#   Libraries needed to link to DRM.
#
# Cache Variables
# ^^^^^^^^^^^^^^^
#
# The following cache variables may also be set:
#
# ``DRM_DRM_INCLUDE_DIR``
#   The directory containing ``drm.h``.
# ``DRM_XF86DRM_INCLUDE_DIR``
#   The directory containing ``xf86drm.h``.
# ``DRM_LIBRARY``
#   The path to the drm library.

set(DRM_VERSION)

set(_DRM_INCLUDE_DIRS)
set(_DRM_LIBRARY_DIRS)

if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL x86_64-linux-gnu)
    find_package(PkgConfig MODULE QUIET)
    pkg_check_modules(_PC_DRM QUIET libdrm)

    if(_PC_DRM_FOUND)
        set(DRM_VERSION "${_PC_DRM_VERSION}")

        set(_DRM_INCLUDE_DIRS "${_PC_DRM_INCLUDE_DIRS}")
        set(_DRM_LIBRARY_DIRS "${_PC_DRM_LIBRARY_DIRS}")
    endif()
endif()

find_path(DRM_DRM_INCLUDE_DIR
    NAMES drm.h
    PATHS "${_DRM_INCLUDE_DIRS}"
    PATH_SUFFIXES libdrm
    DOC "Path to the directory containing the header file drm.h."
)

find_path(DRM_XF86DRM_INCLUDE_DIR
    NAMES xf86drm.h
    PATHS "${_DRM_INCLUDE_DIRS}"
    DOC "Path to the directory containing the header file xf86drm.h."
)

unset(_DRM_INCLUDE_DIRS)

set(_DRM_LIBRARY_SONAME
    "${CMAKE_SHARED_LIBRARY_PREFIX}drm${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

find_library(DRM_LIBRARY
    NAMES "${_DRM_LIBRARY_SONAME}"
    PATHS "${_DRM_LIBRARY_DIRS}"
    DOC "Path to the shared library file ${_DRM_LIBRARY_SONAME}."
)

unset(_DRM_LIBRARY_DIRS)

find_package(Threads QUIET MODULE)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(DRM
    FOUND_VAR DRM_FOUND
    REQUIRED_VARS
        DRM_DRM_INCLUDE_DIR
        DRM_XF86DRM_INCLUDE_DIR
        DRM_LIBRARY
        Threads_FOUND
    VERSION_VAR DRM_VERSION
)

if(DRM_FOUND)
    set(DRM_INCLUDE_DIRS "${DRM_DRM_INCLUDE_DIR}" "${DRM_XF86DRM_INCLUDE_DIR}")
    set(DRM_LIBRARIES "${DRM_LIBRARY}" ${CMAKE_THREAD_LIBS_INIT})
    mark_as_advanced(
        DRM_DRM_INCLUDE_DIR
        DRM_XF86DRM_INCLUDE_DIR
        DRM_LIBRARY
    )

    if(NOT TARGET DRM::DRM)
        add_library(DRM::DRM SHARED IMPORTED)
        set_target_properties(DRM::DRM PROPERTIES
            IMPORTED_LOCATION "${DRM_LIBRARY}"
            IMPORTED_SONAME "${_DRM_LIBRARY_SONAME}"
        )
        target_include_directories(DRM::DRM SYSTEM INTERFACE
            "${DRM_INCLUDE_DIRS}"
        )
        target_link_libraries(DRM::DRM INTERFACE Threads::Threads)
    endif()
endif()

unset(_DRM_LIBRARY_SONAME)

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

# FindNvTime
# ----------
#
# Finds the NVIDIA (R) NvTime library.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# This module provides the following imported targets, if found:
#
# ``NvTime::NvTime``
#   The NvTime library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This will define the following variables:
#
# ``NvTime_FOUND``
#   True if the system has the NvTime library.
# ``NvTime_INCLUDE_DIRS``
#   Include directories needed to use NvTime.
# ``NvTime_LIBRARIES``
#   Libraries needed to link to NvTime.
#
# Cache Variables
# ^^^^^^^^^^^^^^^
#
# The following cache variables may also be set:
#
# ``NvTime_INCLUDE_DIR``
#   The directory containing ``nvtime2.h``.
# ``NvTime_NVTIME_LIBRARY``
#   The path to ``libnvtime2.a``.
# ``NvTime_NVOS_LIBRARY``
#   The path to ``libnvos.so``.

find_path(NvTime_INCLUDE_DIR
    NAMES nvtime2.h
    DOC "Path to the directory containing the header file nvtime2.h."
)

set(_NvTime_NVTIME_LIBRARY_NAME
    "${CMAKE_STATIC_LIBRARY_PREFIX}nvtime2${CMAKE_STATIC_LIBRARY_SUFFIX}"
)

find_library(NvTime_NVTIME_LIBRARY
    NAMES "${_NvTime_NVTIME_LIBRARY_NAME}"
    DOC "Path to the static library file ${_NvTime_NVTIME_LIBRARY_NAME}."
)

unset(_NvTime_NVTIME_LIBRARY_NAME)

set(_NvTime_NVOS_SONAME
    "${CMAKE_SHARED_LIBRARY_PREFIX}nvos${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

find_library(NvTime_NVOS_LIBRARY
    NAMES "${_NvTime_NVOS_SONAME}"
    DOC "Path to the shared library file ${_NvTime_NVOS_SONAME}."
)

find_package(Threads QUIET MODULE)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(NvTime
    FOUND_VAR NvTime_FOUND
    REQUIRED_VARS
        NvTime_INCLUDE_DIR
        NvTime_NVOS_LIBRARY
        NvTime_NVTIME_LIBRARY
        Threads_FOUND
)

if(NvTime_FOUND)
    set(NvTime_INCLUDE_DIRS "${NvTime_INCLUDE_DIR}")
    set(NvTime_LIBRARIES
        "${NvTime_NVOS_LIBRARY}"
        "${NvTime_NVTIME_LIBRARY}"
        ${CMAKE_DL_LIBS}
        ${CMAKE_THREAD_LIBS_INIT}
    )
    mark_as_advanced(
        NvTime_INCLUDE_DIR
        NvTime_NVOS_LIBRARY
        NvTime_NVTIME_LIBRARY
    )

    if(NOT TARGET NvTime::nvos)
        add_library(NvTime::nvos SHARED IMPORTED)
        set_target_properties(NvTime::nvos PROPERTIES
            IMPORTED_LOCATION "${NvTime_NVOS_LIBRARY}"
            IMPORTED_SONAME "${_NvTime_NVOS_SONAME}"
        )
        target_link_libraries(NvTime::nvos INTERFACE
            Threads::Threads
            ${CMAKE_DL_LIBS}
        )
    endif()

    if(NOT TARGET NvTime::nvtime)
        add_library(NvTime::nvtime SHARED IMPORTED)
        set_target_properties(NvTime::nvtime PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES C
            IMPORTED_LOCATION "${NvTime_NVTIME_LIBRARY}"
        )
        target_include_directories(NvTime::nvtime SYSTEM INTERFACE
            "${NvTime_INCLUDE_DIRS}"
        )
    endif()

    if(NOT TARGET NvTime::NvTime)
        add_library(NvTime::NvTime INTERFACE IMPORTED)
        target_link_libraries(NvTime::NvTime INTERFACE
            NvTime::nvos
            NvTime::nvtime
        )
    endif()
endif()

unset(_NvTime_NVOS_SONAME)

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
# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if(CMAKE_VERSION VERSION_LESS 3.9)
    message(FATAL_ERROR "CMake >= 3.9 required")
endif()

cmake_policy(PUSH)
cmake_policy(VERSION 3.9)

include(CMakeFindDependencyMacro)

find_dependency(Threads MODULE)

find_dependency(driveworks CONFIG PATHS "${CMAKE_CURRENT_LIST_DIR}")
find_dependency(driveworks-shared CONFIG PATHS "${CMAKE_CURRENT_LIST_DIR}")
find_dependency(dwdynamicmemory CONFIG PATHS "${CMAKE_CURRENT_LIST_DIR}")

find_dependency(modern-json CONFIG PATHS "${CMAKE_CURRENT_LIST_DIR}")

set(CMAKE_IMPORT_FILE_VERSION 1)

if(NOT TARGET dwframework::dwframework)
    get_filename_component(_DWFRAMEWORK_IMPORT_PREFIX
        "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY
    )
    if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL aarch64-unknown-nto-qnx-safety)
        set(_DWFRAMEWORK_IMPORT_PREFIX
            "${_DWFRAMEWORK_IMPORT_PREFIX}/targets/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}-safety"
        )
    else()
        set(_DWFRAMEWORK_IMPORT_PREFIX
            "${_DWFRAMEWORK_IMPORT_PREFIX}/targets/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}"
        )
    endif()

    add_library(dwframework::dwframework INTERFACE IMPORTED)
    target_compile_features(dwframework::dwframework INTERFACE cxx_std_14)
    target_include_directories(dwframework::dwframework INTERFACE
        "${_DWFRAMEWORK_IMPORT_PREFIX}/include"
    )
    target_link_libraries(dwframework::dwframework INTERFACE
        driveworks::driveworks
        driveworks-shared::driveworks-shared
        dwdynamicmemory::dwdynamicmemory
        modern-json::modern-json
        Threads::Threads
    )

    set(_DWFRAMEWORK_LIBRARY_NAMES
        dwpbwire dwcgf dwframework_dwnodes dwframework_dwnodes_channel_packets dwroadcast-frontend dw_roadcast_v2_1
    )

    foreach(_DWFRAMEWORK_LIBRARY_NAME ${_DWFRAMEWORK_LIBRARY_NAMES})
        set(_DWFRAMEWORK_IMPORTED_LOCATION
            "${_DWFRAMEWORK_IMPORT_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${_DWFRAMEWORK_LIBRARY_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}"
        )

        if(NOT TARGET dwframework::_${_DWFRAMEWORK_LIBRARY_NAME})
            add_library(dwframework::_${_DWFRAMEWORK_LIBRARY_NAME} SHARED
                IMPORTED
            )
            set_target_properties(dwframework::_${_DWFRAMEWORK_LIBRARY_NAME}
                PROPERTIES
                    IMPORTED_LOCATION "${_DWFRAMEWORK_IMPORTED_LOCATION}"
            )
        endif()

        target_link_libraries(dwframework::dwframework INTERFACE
            dwframework::_${_DWFRAMEWORK_LIBRARY_NAME}
        )

        if(NOT EXISTS "${_DWFRAMEWORK_IMPORTED_LOCATION}")
            message(FATAL_ERROR
                "The imported target \"dwframework::_${_DWFRAMEWORK_LIBRARY_NAME}\" references the file \"${_DWFRAMEWORK_IMPORTED_LOCATION}\" but this file does not exist."
            )
        endif()

        unset(_DWFRAMEWORK_IMPORTED_LOCATION)
        unset(_DWFRAMEWORK_LIBRARY_NAME)
    endforeach()

    unset(_DWFRAMEWORK_LIBRARY_NAMES)
    unset(_DWFRAMEWORK_IMPORT_PREFIX)
endif()

if(NOT TARGET dwframework)
    add_library(dwframework INTERFACE IMPORTED)
    target_link_libraries(dwframework INTERFACE dwframework::dwframework)

    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
        set_target_properties(dwframework PROPERTIES DEPRECATION
            "The imported target \"dwframework\" is deprecated, please use the imported target \"dwframework::dwframework\" instead."
        )
    endif()
endif()

unset(CMAKE_IMPORT_FILE_VERSION)

cmake_policy(POP)

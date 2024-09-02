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

if(CMAKE_VERSION VERSION_LESS 3.9)
    message(FATAL_ERROR "CMake >= 3.9 required")
endif()

cmake_policy(PUSH)
cmake_policy(VERSION 3.9)

set(CMAKE_IMPORT_FILE_VERSION 1)

get_filename_component(_GLFW_IMPORT_PREFIX
    "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY
)
if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL aarch64-unknown-nto-qnx-safety)
    set(_GLFW_IMPORT_PREFIX
       "${_GLFW_IMPORT_PREFIX}/targets/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}-safety"
    )
else()
    set(_GLFW_IMPORT_PREFIX
        "${_GLFW_IMPORT_PREFIX}/targets/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}"
    )
endif()

if(NOT TARGET glfw::glfw)
    set(_GLFW_IMPORTED_SONAME "${CMAKE_SHARED_LIBRARY_PREFIX}glfw${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(_GLFW_IMPORTED_LOCATION "${_GLFW_IMPORT_PREFIX}/lib/${_GLFW_IMPORTED_SONAME}")

    add_library(glfw::glfw SHARED IMPORTED)
    set_target_properties(glfw::glfw PROPERTIES
        IMPORTED_LOCATION "${_GLFW_IMPORTED_LOCATION}"
        IMPORTED_SONAME "${_GLFW_IMPORTED_SONAME}"
    )
    target_compile_features(glfw::glfw INTERFACE c_std_11)
    target_include_directories(glfw::glfw INTERFACE
        "${_GLFW_IMPORT_PREFIX}/include/3rdparty"
    )

    if(NOT EXISTS "${_GLFW_IMPORTED_LOCATION}")
        message(FATAL_ERROR
           "The imported target \"glfw::glfw\" references the file \"${_GLFW_IMPORTED_LOCATION}\" but this file does not exist."
        )
    endif()

    unset(_GLFW_IMPORTED_LOCATION)
    unset(_GLFW_IMPORTED_SONAME)
endif()

unset(_GLFW_IMPORT_PREFIX)
unset(CMAKE_IMPORT_FILE_VERSION)

cmake_policy(POP)

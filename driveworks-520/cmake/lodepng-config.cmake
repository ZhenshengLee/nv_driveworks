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

get_filename_component(_LODEPNG_IMPORT_PREFIX
    "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY
)
if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL aarch64-unknown-nto-qnx-safety)
    set(_LODEPNG_IMPORT_PREFIX
       "${_LODEPNG_IMPORT_PREFIX}/targets/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}-safety"
    )
else()
    set(_LODEPNG_IMPORT_PREFIX
        "${_LODEPNG_IMPORT_PREFIX}/targets/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}"
    )
endif()

if(NOT TARGET lodepng::lodepng)
    set(_LODEPNG_IMPORTED_LOCATION
        "${_LODEPNG_IMPORT_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}lodepng${CMAKE_STATIC_LIBRARY_SUFFIX}"
    )

    add_library(lodepng::lodepng STATIC IMPORTED)
    set_target_properties(lodepng::lodepng PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES CXX
        IMPORTED_LOCATION "${_LODEPNG_IMPORTED_LOCATION}"
    )
    target_compile_features(lodepng::lodepng INTERFACE cxx_std_14)
    target_include_directories(lodepng::lodepng INTERFACE
        "${_LODEPNG_IMPORT_PREFIX}/include/3rdparty"
    )

    if(NOT EXISTS "${_LODEPNG_IMPORTED_LOCATION}")
        message(FATAL_ERROR
           "The imported target \"lodepng::lodepng\" references the file \"${_LODEPNG_IMPORTED_LOCATION}\" but this file does not exist."
        )
    endif()

    unset(_LODEPNG_IMPORTED_LOCATION)
endif()

unset(_LODEPNG_IMPORT_PREFIX)
unset(CMAKE_IMPORT_FILE_VERSION)

cmake_policy(POP)
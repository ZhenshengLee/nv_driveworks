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
# SPDX-FileCopyrightText: Copyright (c) 2017-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#-------------------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------------------
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/src/lodepng)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/src/glfw)

if(VIBRANTE)
    set(vibrante_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/${SDK_ARCH_DIR}/vibrante" CACHE PATH '' FORCE)
    find_package(vibrante REQUIRED CONFIG)
    if(NOT VIBRANTE_V5Q)
        set(vibrante_Xlibs_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/${SDK_ARCH_DIR}/vibrante_Xlibs" CACHE PATH '' FORCE)
        find_package(vibrante_Xlibs CONFIG REQUIRED)

        set(zlib_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/${SDK_ARCH_DIR}/zlib" CACHE PATH '' FORCE)
        find_package(zlib REQUIRED)
    endif()
    set(DW_USE_NVMEDIA_DRIVE ON)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
else()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/src/glew)
    # this is for parts of the code needed only on LINUX because of special care needed for nvmedia x86
    set(DW_USE_NVMEDIA_X86 ON)
endif()

# Hide settings in default cmake view
mark_as_advanced(vibrante_DIR vibrante_Xlibs_DIR)

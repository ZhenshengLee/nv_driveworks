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
# SPDX-FileCopyrightText: Copyright (c) 2016-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Platform selection
#-------------------------------------------------------------------------------

if(VIBRANTE)
    message(STATUS "Cross Compiling for Vibrante")
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(WINDOWS TRUE)
    add_definitions(-DWINDOWS)
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(LINUX TRUE)
        add_definitions(-DLINUX)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7l" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        message(FATAL_ERROR "Direct compilation not supported for ${CMAKE_SYSTEM_PROCESSOR}, use cross compilation.")
    else()
        message(FATAL_ERROR "Unsupported Linux CPU architecture ${CMAKE_SYSTEM_PROCESSOR}.")
    endif()
else()
    message(FATAL_ERROR "Cannot identify OS")
endif()

#-------------------------------------------------------------------------------
# Architecture selection
#-------------------------------------------------------------------------------
if(VIBRANTE)
    if(VIBRANTE_V5Q)
        # Qnx arm64
        set(ARCH_DIR     "qnx-aarch64")
        set(SYS_ARCH_DIR "aarch64-qnx-gnu")
        # NOTE: V5Q is already C++11ABI exclusively
    else()
        # Linux arm64
        set(ARCH_DIR     "linux-aarch64")
        set(SYS_ARCH_DIR "aarch64-linux-gnu")
    endif()
else()
    # Linux x86_64
    set(ARCH_DIR     "linux-x86")
    set(SYS_ARCH_DIR "x86_64-linux-gnu")
endif()
set(C_ARCH_DIR   "${ARCH_DIR}/abiC")
set(CPP_ARCH_DIR "${ARCH_DIR}/abi11")

set(CPP_ABI ABI11) # supporting only gcc >= 5.0

# Dependencies that are C++ abi dependent are stored under SDK_CPP_ARCH_DIR
unset(SDK_CPP_ARCH_DIR CACHE)
set(SDK_CPP_ARCH_DIR ${CPP_ARCH_DIR} CACHE INTERNAL "")

# Dependencies that are C only and don't care about abi are stored under SDK_C_ARCH_DIR
unset(SDK_C_ARCH_DIR CACHE)
set(SDK_C_ARCH_DIR ${C_ARCH_DIR} CACHE INTERNAL "")

# Generic top-level architecture subfolder name SDK_ARCH_DIR
unset(SDK_ARCH_DIR CACHE)
set(SDK_ARCH_DIR ${ARCH_DIR} CACHE INTERNAL "")

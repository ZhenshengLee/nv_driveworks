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
# SPDX-FileCopyrightText: Copyright (c) 2016-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

#-------------------------------------------------------------------------------
# Set basic configurations
#-------------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(SamplesSetBuildType) # Set the build type before project is created

set(SDK_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SDK_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
#-------------------------------------------------------------------------------
# Root project
#-------------------------------------------------------------------------------
project(DriveworksSDK-Samples C CXX)

#-------------------------------------------------------------------------------
# CUDA host compiler must be set before CUDA is enabled as a language
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Set CUDA_DIR
#-------------------------------------------------------------------------------
if (DEFINED CUDA_DIR)
    if((DEFINED CUDA_TOOLKIT_ROOT_DIR) AND (NOT CUDA_TOOLKIT_ROOT_DIR STREQUAL CUDA_DIR))
        message(FATAL_ERROR "Cannot set both CUDA_DIR and (legacy) CUDA_TOOLKIT_ROOT_DIR")
    endif()
elseif (DEFINED CUDA_TOOLKIT_ROOT_DIR)
    message(WARNING "Please set CUDA_DIR instead of (legacy) CUDA_TOOLKIT_ROOT_DIR")
    set(CUDA_DIR  ${CUDA_TOOLKIT_ROOT_DIR} CACHE PATH "CUDA Toolkit location.")
else()
    set(CUDA_DIR  "/usr/local/cuda/" CACHE PATH "CUDA Toolkit location.")
endif()
if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${CUDA_DIR}/bin/nvcc")
endif()
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_C_STANDARD 99)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#-------------------------------------------------------------------------------
# System packages
#-------------------------------------------------------------------------------
find_package(EGL)
find_package(Threads REQUIRED)

#-------------------------------------------------------------------------------
# Basic configuration
#-------------------------------------------------------------------------------
include(ArchConfiguration)
include(SamplesConfiguration)
include(ResourcesConfiguration)
include(Samples3rdparty)
if (NOT SAMPLES_DWVAL_ONLY)
    
endif()
include(SamplesInstallConfiguration)
include(UploadVibrante)
set(SDK_BINARY_DIR ${CMAKE_BINARY_DIR})

#-------------------------------------------------------------------------------
# DriveWorks SDK
#-------------------------------------------------------------------------------
find_package(CUDART MODULE REQUIRED)
find_package(cuBLAS MODULE REQUIRED)
find_package(driveworks REQUIRED CONFIG HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
find_package(driveworks-visualization REQUIRED CONFIG HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
find_package(driveworks-shared REQUIRED CONFIG HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
find_package(dwdynamicmemory REQUIRED CONFIG HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

set(Driveworks_LIBRARIES
    CUDART::CUDART
    cuBLAS::cuBLAS
    driveworks
    driveworks-shared
    driveworks-visualization
    samples_allocator)

if(VIBRANTE)
    find_package(cuPVA MODULE REQUIRED)
    list(APPEND Driveworks_LIBRARIES ${vibrante_LIBRARIES} ${vibrante_Xlibs_LIBRARIES} cuPVA::cuPVA)
    find_package(cuDLA MODULE)
    if(cuDLA_FOUND)
        list(APPEND Driveworks_LIBRARIES cuDLA::cuDLA)
    endif()
    find_package(NvMedia MODULE)
    if(NvMedia_FOUND)
        list(APPEND Driveworks_LIBRARIES NvMedia::NvMedia)
    endif()   
endif()

if (VIBRANTE OR DW_EXPERIMENTAL_FORCE_EGL)
    list(APPEND Driveworks_LIBRARIES ${EGL_LIBRARIES})
endif()

#-------------------------------------------------------------------------------
# Samples
#-------------------------------------------------------------------------------
set(SAMPLES framework;egomotion;sensors;imageprocessing;rig;visualization;dnn;comms;hello_world;image;template;icp;calibration;vehicleio;vehicleio_plugin;pointcloudprocessing;dwchannel;camera_interprocess;cgf_nodes;dataspeedBridge;)
file(GLOB ADDITIONAL_SAMPLES "CMakeListsSamples*.txt")
foreach(ADDITIONAL_SAMPLE ${ADDITIONAL_SAMPLES})
  include(${ADDITIONAL_SAMPLE})
endforeach()

if (SAMPLES_DWVAL_ONLY)
    set (SAMPLES framework;dwval)
endif()

foreach(SAMPLE ${SAMPLES})
    if(NOT VIBRANTE AND ${SAMPLE} STREQUAL "camera_interprocess")
        message(STATUS "Skipping camera_interprocess sample on X86")
    else()
        add_subdirectory(src/${SAMPLE})
    endif()
endforeach()

# Create a symlink '_data' to '<DW-install>/data' in the build folder's root, so that `DataPath`
# can lookup the required path independent of the build folder's location relative to the
# repository's root
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/../data/DATA_ROOT")
    message(WARNING "Expected data root '${CMAKE_SOURCE_DIR}/../data/DATA_ROOT' not available, "
                    "likely because DW sample code was moved from install location or installation is incomplete. "
                    "Samples will likely not run correctly")
endif()

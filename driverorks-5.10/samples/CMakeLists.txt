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

cmake_minimum_required(VERSION 3.16)

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
project(DriveWorksSDK-Samples
    DESCRIPTION "NVIDIA DriveWorks SDK samples"
    HOMEPAGE_URL "https://developer.nvidia.com/drive/driveworks"
)

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
include(SamplesInstallConfiguration)
include(UploadVibrante)
set(SDK_BINARY_DIR ${CMAKE_BINARY_DIR})

#-------------------------------------------------------------------------------
# DriveWorks SDK
#-------------------------------------------------------------------------------
find_package(CUDART MODULE REQUIRED)
find_package(cuBLAS MODULE REQUIRED)

get_filename_component(DRIVEWORKS_DIR ${CMAKE_CURRENT_LIST_DIR} REALPATH)
get_filename_component(DRIVEWORKS_DIR ${DRIVEWORKS_DIR} DIRECTORY)

find_package(driveworks REQUIRED CONFIG PATHS "${DRIVEWORKS_DIR}/cmake" NO_DEFAULT_PATH)
find_package(driveworks-visualization REQUIRED CONFIG PATHS "${DRIVEWORKS_DIR}/cmake" NO_DEFAULT_PATH)
find_package(driveworks-shared REQUIRED CONFIG PATHS "${DRIVEWORKS_DIR}/cmake" NO_DEFAULT_PATH)
find_package(dwdynamicmemory REQUIRED CONFIG PATHS "${DRIVEWORKS_DIR}/cmake" NO_DEFAULT_PATH)

set(Driveworks_LIBRARIES
    CUDART::CUDART
    cuBLAS::cuBLAS
    driveworks
    driveworks-shared
    driveworks-visualization
    samples_allocator)

if(VIBRANTE)
    list(APPEND Driveworks_LIBRARIES ${vibrante_LIBRARIES} ${vibrante_Xlibs_LIBRARIES})
    find_package(cuDLA MODULE)
    if(cuDLA_FOUND)
        list(APPEND Driveworks_LIBRARIES cuDLA::cuDLA)
    endif()
    find_package(cuPVA MODULE)
    if(cuPVA_FOUND)
        list(APPEND Driveworks_LIBRARIES cuPVA::cuPVA)
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
add_subdirectory(src/framework)

if(SAMPLES_DWVAL_ONLY)
    add_subdirectory(src/dwval)
else()
    add_subdirectory(src/egomotion)
    add_subdirectory(src/sensors)
    add_subdirectory(src/imageprocessing)
    add_subdirectory(src/rig)
    add_subdirectory(src/visualization)
    add_subdirectory(src/dnn)
    add_subdirectory(src/comms)
    add_subdirectory(src/hello_world)
    add_subdirectory(src/image)
    add_subdirectory(src/icp)
    add_subdirectory(src/calibration)
    add_subdirectory(src/vehicleio)
    add_subdirectory(src/vehicleio_plugin)
    add_subdirectory(src/pointcloudprocessing)
    add_subdirectory(src/dwchannel)
    if(VIBRANTE)
        add_subdirectory(src/camera_interprocess)
    endif()
    add_subdirectory(src/cgf_nodes)
    if(NOT VIBRANTE_V5Q)
        add_subdirectory(src/dataspeedBridge)
    endif()
endif()

if(NOT SAMPLES_DWVAL_ONLY)
    set(SAMPLES)
    file(GLOB ADDITIONAL_SAMPLES "CMakeListsSamples*.txt")
    foreach(ADDITIONAL_SAMPLE ${ADDITIONAL_SAMPLES})
        include(${ADDITIONAL_SAMPLE})
    endforeach()

    foreach(SAMPLE ${SAMPLES})
        add_subdirectory(src/${SAMPLE})
    endforeach()
endif()

# Create a symlink '_data' to '<DW-install>/data' in the build folder's root, so that `DataPath`
# can lookup the required path independent of the build folder's location relative to the
# repository's root
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/../data/DATA_ROOT")
    message(WARNING "Expected data root '${CMAKE_SOURCE_DIR}/../data/DATA_ROOT' not available, "
                    "likely because DW sample code was moved from install location or installation is incomplete. "
                    "Samples will likely not run correctly")
endif()

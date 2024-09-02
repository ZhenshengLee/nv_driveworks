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
# Root project
#-------------------------------------------------------------------------------
project(DriveWorksSDK-Samples
    DESCRIPTION "NVIDIA DriveWorks SDK samples"
    HOMEPAGE_URL "https://developer.nvidia.com/drive/driveworks"
)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SamplesSetBuildType)

set(SDK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SDK_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

#-------------------------------------------------------------------------------
# CUDA host compiler must be set before CUDA is enabled as a language
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Set CUDA_DIR
#-------------------------------------------------------------------------------
if(NOT CUDA_DIR)
    set(CUDA_DIR "/usr/local/cuda" CACHE PATH "CUDA Toolkit location.")
endif()
if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${CUDA_DIR}/bin/nvcc")
endif()
if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()
enable_language(CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#-------------------------------------------------------------------------------
# System packages
#-------------------------------------------------------------------------------
find_package(Threads MODULE REQUIRED)

if(NOT VIBRANTE)
    option(DW_EXPERIMENTAL_FORCE_EGL
        "Force EGL support on host x86_64-linux-gnu" OFF
    )
endif()

if(VIBRANTE OR DW_EXPERIMENTAL_FORCE_EGL)
    if(NOT DW_IS_SAFETY)
        if(NOT VIBRANTE_V5Q)
            find_package(DRM MODULE REQUIRED)
        endif()

        find_package(EGL MODULE REQUIRED)
        find_package(GLES MODULE REQUIRED)
    endif()
endif()

#-------------------------------------------------------------------------------
# Basic configuration
#-------------------------------------------------------------------------------
include(ArchConfiguration)
include(SamplesConfiguration)
if(NOT SAMPLES_DWVAL_ONLY)
    include(ResourcesConfiguration)
endif()
include(Samples3rdparty)
include(SamplesInstallConfiguration)

get_filename_component(DRIVEWORKS_DIR ${CMAKE_CURRENT_LIST_DIR} REALPATH)
get_filename_component(DRIVEWORKS_DIR ${DRIVEWORKS_DIR} DIRECTORY)

#-------------------------------------------------------------------------------
# DRIVE OS
#-------------------------------------------------------------------------------
find_package(CUDART MODULE REQUIRED)
find_package(NvSCI MODULE REQUIRED)

if(NOT DW_IS_SAFETY)
    find_package(cuBLAS MODULE REQUIRED)

    if(VIBRANTE)
        find_package(cupva CONFIG REQUIRED PATHS "${DRIVEWORKS_DIR}/cmake"
            NO_DEFAULT_PATH
    )
    endif()

    if(NOT SAMPLES_DWVAL_ONLY)
        find_package(cuDNN ${CUDNN_VERSION} MODULE REQUIRED)
    endif()
endif()

if(NOT SAMPLES_DWVAL_ONLY)
    find_package(TensorRT ${TRT_VERSION} MODULE REQUIRED)
endif()

if(VIBRANTE)
    find_package(cuDLA MODULE REQUIRED)
endif()

#-------------------------------------------------------------------------------
# DriveWorks SDK
#-------------------------------------------------------------------------------
find_package(dwdynamicmemory CONFIG REQUIRED PATHS "${DRIVEWORKS_DIR}/cmake"
    NO_DEFAULT_PATH
)

find_package(driveworks REQUIRED CONFIG PATHS "${DRIVEWORKS_DIR}/cmake"
    NO_DEFAULT_PATH
)

if(NOT DW_IS_SAFETY)
    find_package(driveworks-visualization CONFIG REQUIRED
        PATHS "${DRIVEWORKS_DIR}/cmake" NO_DEFAULT_PATH
    )
endif()

find_package(driveworks-shared CONFIG PATHS "${DRIVEWORKS_DIR}/cmake"
    NO_DEFAULT_PATH
)

find_package(dwframework CONFIG REQUIRED PATHS "${DRIVEWORKS_DIR}/cmake"
    NO_DEFAULT_PATH
)

set(Driveworks_LIBRARIES
    CUDART::CUDART
    NvSCI::NvSciBuf
    NvSCI::NvSciSync
    driveworks::driveworks
    driveworks-shared::driveworks-shared
    samples_allocator
)

if(NOT DW_IS_SAFETY)
    list(APPEND Driveworks_LIBRARIES
        cuBLAS::cuBLAS
        driveworks-visualization::driveworks-visualization
        lodepng-src
    )

    if(VIBRANTE)
        list(APPEND Driveworks_LIBRARIES cupva::cupva_wrapper)
    else()
        list(APPEND Driveworks_LIBRARIES glew-src)
    endif()

    if(VIBRANTE OR DW_EXPERIMENTAL_FORCE_EGL)
        if(NOT VIBRANTE_V5Q)
            list(APPEND Driveworks_LIBRARIES DRM::DRM)
        endif()
        list(APPEND Driveworks_LIBRARIES EGL::EGL GLES::GLES)
    endif()
endif()

if(VIBRANTE)
    list(APPEND Driveworks_LIBRARIES cuDLA::cuDLA)

    if(VIBRANTE_V5L)
        list(APPEND Driveworks_LIBRARIES vibrante vibrante_Xlibs)
    endif()
endif()

#-------------------------------------------------------------------------------
# Samples
#-------------------------------------------------------------------------------
add_subdirectory(src/framework)

if(SAMPLES_DWVAL_ONLY)
    add_subdirectory(src/dwval)
else()
    add_subdirectory(src/hello_world)
    if(NOT DW_IS_SAFETY)
        add_subdirectory(src/egomotion)
        add_subdirectory(src/sensors)
        add_subdirectory(src/imageprocessing)
        add_subdirectory(src/rig)
        add_subdirectory(src/visualization)
        add_subdirectory(src/dnn)
        add_subdirectory(src/comms)
        add_subdirectory(src/image)
        add_subdirectory(src/icp)
        add_subdirectory(src/calibration)
        add_subdirectory(src/vehicleio)
        add_subdirectory(src/vehicleio_plugin)
        add_subdirectory(src/pointcloudprocessing)
        if(NOT VIBRANTE_V5Q)
            add_subdirectory(src/dataspeedBridge)
        endif()
        if(dwframework_FOUND AND driveworks-shared_FOUND)
            add_subdirectory(src/dwchannel)
            add_subdirectory(src/cgf_nodes)
            add_subdirectory(src/minipipeline)
            if(VIBRANTE)
                add_subdirectory(src/camera_interprocess)
            endif()
        endif()
    endif()
endif()

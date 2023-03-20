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

project(samples_framework C CXX)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES
    Checks.hpp
    SampleFramework.hpp
    SampleFramework.cpp

    DriveWorksSample.hpp
    DriveWorksSample.cpp

    DriveWorksSampleSeek.hpp
    DriveWorksSampleSeek.cpp

    ProgramArguments.hpp
    ProgramArguments.cpp

    Window.hpp
    Window.cpp

    Grid.hpp
    Grid.cpp

    WindowGLFW.hpp
    WindowGLFW.cpp

    MathUtils.hpp
    MathUtils.cpp
    MouseView3D.cpp
    MouseView3D.hpp

    Mat3.hpp
    Mat4.hpp

    Log.hpp
    Log.cpp

    StatsCounter.hpp
    CudaTimer.hpp

    ProfilerCUDA.hpp
    ProfilerCUDA.cpp

    SimpleStreamer.hpp
    CameraFramePipeline.hpp
    CameraFramePipeline.cpp
    SimpleCamera.hpp
    SimpleCamera.cpp

    SimpleRenderer.cpp
    SimpleRenderer.hpp
    SimpleRecordingPlayer.hpp
    SimpleRecordingPlayer.cpp

    ScreenshotHelper.hpp
    ScreenshotHelper.cpp
    RenderUtils.hpp
    RenderUtils.cpp

    SamplesDataPath.hpp
    SamplesDataPath.cpp
)

set(LIBRARIES
    glfw-src
    lodepng-src
    drm
    ${Driveworks_LIBRARIES}
)

if(DW_USE_EGL)
    list(APPEND SOURCES
         WindowEGL.hpp
         WindowEGL.cpp
    )

    if(VIBRANTE_V5Q)
        list(APPEND SOURCES
            WindowQNXEGL.hpp
            WindowQNXEGL.cpp
        )
        list(APPEND LIBRARIES
            screen
        )
    elseif(VIBRANTE_V5L OR LINUX)
        list(APPEND SOURCES
            WindowLinuxEGL.hpp
            WindowLinuxEGL.cpp
        )
    endif()
endif()

if(LINUX)
    list(APPEND LIBRARIES
        glew-src
    )
endif()

if(VIBRANTE)
    list(APPEND LIBRARIES
        vibrante
        gles
    )

    if(VIBRANTE_V5L)
        list(APPEND LIBRARIES
            vibrante_Xlibs
        )
    endif()
endif()

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBRARIES})
target_include_directories(${PROJECT_NAME}
                           PUBLIC
                           ${CMAKE_CURRENT_SOURCE_DIR}/..
                           ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Samples")

#-------------------------------------------------------------------------------
# Special aligned allocator library
#-------------------------------------------------------------------------------
add_library(samples_allocator STATIC Allocator.cpp)
target_link_libraries(samples_allocator PUBLIC dwdynamicmemory)

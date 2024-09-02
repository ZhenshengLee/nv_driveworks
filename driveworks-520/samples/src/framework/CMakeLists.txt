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

project(samples_framework)

if(NOT DW_IS_SAFETY)
    #-------------------------------------------------------------------------------
    # Project files
    #-------------------------------------------------------------------------------
    set(SOURCES
        DriveWorksSample.cpp
        DriveWorksSample.hpp
        Window.cpp
        Window.hpp
        WindowGLFW.cpp
        WindowGLFW.hpp
    )

    set(LIBRARIES
        ${Driveworks_LIBRARIES}
        glfw-src
        samples_framework_camera_frame_pipeline
        samples_framework_checks
        samples_framework_data_path
        samples_framework_grid
        samples_framework_log
        samples_framework_math_utils
        samples_framework_mouse_view
        samples_framework_profiler_cuda
        samples_framework_program_arguments
        samples_framework_render_utils
        samples_framework_screenshot_helper
        samples_framework_simple_camera
        samples_framework_simple_recording_player
        samples_framework_simple_renderer
        samples_framework_simple_streamer
    )

    if(VIBRANTE OR DW_EXPERIMENTAL_FORCE_EGL)
        list(APPEND SOURCES
            WindowEGL.cpp
            WindowEGL.hpp
        )

        if(VIBRANTE_V5Q)
            list(APPEND SOURCES
                WindowQNXEGL.cpp
                WindowQNXEGL.hpp
            )
            list(APPEND LIBRARIES screen)
        else()
            list(APPEND SOURCES
                WindowLinuxEGL.cpp
                WindowLinuxEGL.hpp
            )
        endif()

        list(APPEND LIBRARIES EGL::EGL)
    endif()

    if(VIBRANTE)
        list(APPEND LIBRARIES GLES::GLES)

        if(VIBRANTE_V5L)
            list(APPEND LIBRARIES
                vibrante
                vibrante_Xlibs
            )
        endif()
    else()
        list(APPEND LIBRARIES glew-src)
    endif()

    #-------------------------------------------------------------------------------
    # Final target
    #-------------------------------------------------------------------------------
    add_library(${PROJECT_NAME} STATIC ${SOURCES})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBRARIES})
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )

    #-------------------------------------------------------------------------------
    # Samples Framework - Camera Frame Pipeline Static Library
    #-------------------------------------------------------------------------------
    add_library(samples_framework_camera_frame_pipeline STATIC
        CameraFramePipeline.cpp
        CameraFramePipeline.hpp
    )
    target_include_directories(samples_framework_camera_frame_pipeline PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(samples_framework_camera_frame_pipeline PUBLIC
        ${Driveworks_LIBRARIES}
        driveworks-visualization
        samples_framework_checks
        samples_framework_log
        samples_framework_simple_streamer
    )

    #-------------------------------------------------------------------------------
    # Samples Framework - Grid Static Library
    #-------------------------------------------------------------------------------
    add_library(samples_framework_grid STATIC
        Grid.cpp
        Grid.hpp
    )
    target_include_directories(samples_framework_grid PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(samples_framework_grid PUBLIC
        ${Driveworks_LIBRARIES}
        driveworks-visualization
    )

    #-------------------------------------------------------------------------------
    # Samples Framework - Render Utils Static Library
    #-------------------------------------------------------------------------------
    add_library(samples_framework_render_utils STATIC
        RenderUtils.cpp
        RenderUtils.hpp
    )
    target_include_directories(samples_framework_render_utils PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(samples_framework_render_utils PUBLIC
        ${Driveworks_LIBRARIES}
        driveworks-visualization
        samples_framework_checks
    )

    #-------------------------------------------------------------------------------
    # Samples Framework - Screenshot Helper Static Library
    #-------------------------------------------------------------------------------
    add_library(samples_framework_screenshot_helper STATIC
        ScreenshotHelper.cpp
        ScreenshotHelper.hpp
    )
    target_include_directories(samples_framework_screenshot_helper PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(samples_framework_screenshot_helper PUBLIC
        ${Driveworks_LIBRARIES}
        driveworks-visualization
        lodepng-src
        samples_framework_checks
    )

    #-------------------------------------------------------------------------------
    # Samples Framework - Simple Camera Static Library
    #-------------------------------------------------------------------------------
    add_library(samples_framework_simple_camera STATIC
        SimpleCamera.cpp
        SimpleCamera.hpp
    )
    target_include_directories(samples_framework_simple_camera PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(samples_framework_simple_camera PUBLIC
        ${Driveworks_LIBRARIES}
        driveworks-visualization
        samples_framework_camera_frame_pipeline
        samples_framework_checks
        samples_framework_simple_streamer
    )

    #-------------------------------------------------------------------------------
    # Samples Framework - Simple Recording Player Static Library
    #-------------------------------------------------------------------------------
    add_library(samples_framework_simple_recording_player STATIC
        SimpleRecordingPlayer.cpp
        SimpleRecordingPlayer.hpp
    )
    target_include_directories(samples_framework_simple_recording_player PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(samples_framework_simple_recording_player PUBLIC
        ${Driveworks_LIBRARIES}
        samples_framework_simple_camera
    )

    #-------------------------------------------------------------------------------
    # Samples Framework - Simple Renderer Static Library
    #-------------------------------------------------------------------------------
    add_library(samples_framework_simple_renderer STATIC
        SimpleRenderer.cpp
        SimpleRenderer.hpp
    )
    target_include_directories(samples_framework_simple_renderer PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(samples_framework_simple_renderer PUBLIC
        ${Driveworks_LIBRARIES}
        driveworks-visualization
        samples_framework_checks
    )

    #-------------------------------------------------------------------------------
    # Samples Framework - Simple Streamer Interface Library
    #-------------------------------------------------------------------------------
    add_library(samples_framework_simple_streamer INTERFACE)
    target_include_directories(samples_framework_simple_streamer INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    target_link_libraries(samples_framework_simple_streamer INTERFACE
        ${Driveworks_LIBRARIES}
        driveworks-visualization
        samples_framework_checks
    )
    target_sources(samples_framework_simple_streamer INTERFACE SimpleStreamer.hpp)
endif()

#-------------------------------------------------------------------------------
# Special aligned allocator library
#-------------------------------------------------------------------------------
add_library(samples_allocator STATIC Allocator.cpp)
target_link_libraries(samples_allocator PUBLIC dwdynamicmemory)

#-------------------------------------------------------------------------------
# Samples Framework - Checks Interface Library
#-------------------------------------------------------------------------------
add_library(samples_framework_checks INTERFACE)
target_include_directories(samples_framework_checks INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(samples_framework_checks INTERFACE CUDART::CUDART)
target_sources(samples_framework_checks INTERFACE Checks.hpp)

#-------------------------------------------------------------------------------
# Samples Framework - Data Path Static Library
#-------------------------------------------------------------------------------
add_library(samples_framework_data_path STATIC
    SamplesDataPath.cpp
    SamplesDataPath.hpp
)
target_include_directories(samples_framework_data_path PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(samples_framework_data_path PUBLIC
    ${Driveworks_LIBRARIES}
)

#-------------------------------------------------------------------------------
# Samples Framework - Log Static Library
#-------------------------------------------------------------------------------
add_library(samples_framework_log STATIC
    Log.cpp
    Log.hpp
)
target_include_directories(samples_framework_log PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(samples_framework_log PUBLIC ${Driveworks_LIBRARIES})

#-------------------------------------------------------------------------------
# Samples Framework - Mouse View Library
#-------------------------------------------------------------------------------
add_library(samples_framework_mouse_view STATIC
    MouseView3D.cpp
    MouseView3D.hpp
)
target_include_directories(samples_framework_mouse_view PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(samples_framework_mouse_view PUBLIC
    ${Driveworks_LIBRARIES}
    samples_framework_math_utils
)

#-------------------------------------------------------------------------------
# Samples Framework - Math Utils Static Library
#-------------------------------------------------------------------------------
add_library(samples_framework_math_utils STATIC
    Mat3.hpp
    Mat4.hpp
    MathUtils.cpp
    MathUtils.hpp
)
target_include_directories(samples_framework_math_utils PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(samples_framework_math_utils PUBLIC ${Driveworks_LIBRARIES})

#-------------------------------------------------------------------------------
# Samples Framework - Profiler CUDA Static Library
#-------------------------------------------------------------------------------
add_library(samples_framework_profiler_cuda STATIC
    CudaTimer.hpp
    ProfilerCUDA.cpp
    ProfilerCUDA.hpp
    StatsCounter.hpp
)
target_include_directories(samples_framework_profiler_cuda PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(samples_framework_profiler_cuda PUBLIC
    CUDART::CUDART
    ${Driveworks_LIBRARIES}
)

#-------------------------------------------------------------------------------
# Samples Framework - Program Arguments Static Library
#-------------------------------------------------------------------------------
add_library(samples_framework_program_arguments STATIC
    ProgramArguments.cpp
    ProgramArguments.hpp
)
target_include_directories(samples_framework_program_arguments PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(samples_framework_program_arguments PUBLIC
    ${Driveworks_LIBRARIES}
)

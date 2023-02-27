#Copyright © 2021, NVIDIA CORPORATION. All rights reserved. 
set(CODEOWNERS @tborusiak)

project(sample_nv_lidar_plugin C CXX)

include_directories(${SAMPLE_PLUGINS_COMMON_DIR})

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------

set(PUBLIC_DOCS
    README-NV_lidar_plugin.md
)

set(SOURCES
    main.cpp
    NVLidar.cpp
    NVLidar.hpp
    NVLidar_Properties.hpp
)

set(LIBRARIES
    ${Driveworks_LIBRARIES}
    sample_sensors_plugin_common
)

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Samples")

#-------------------------------------------------------------------------------
# Install target
#-------------------------------------------------------------------------------
sdk_install_sample_library(${PROJECT_NAME})

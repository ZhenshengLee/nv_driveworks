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

project(cgf_custom_nodes)

set(DRIVEWORKS_PATH "/usr/local/driveworks")
set(NSIGHT_SYS_PATH "/opt/nvidia/nsight-systems/2023.1.3/")
set(DWCGF_RUNNER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(DWCGF_NODES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes)
set(DWCGF_CHANNEL_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/channel)
set(DWCGF_GRAPHS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/graphs)
set(DWCGF_APPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/graphs/apps)
set(DWCGF_NVSCI_RST ${CMAKE_CURRENT_SOURCE_DIR}/bin/nvsciipc_reset)
set(TCP_NODELAY_PRELOAD_SO ${CMAKE_CURRENT_SOURCE_DIR}/bin/libnodelay.so)

set(DWCGF_SCHE_DIR ${SDK_BINARY_DIR}/schedule/${PROJECT_NAME})
file(MAKE_DIRECTORY ${DWCGF_SCHE_DIR})
set(DWCGF_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)

file(GLOB_RECURSE DWCGF_NODES_FILES
    ${DWCGF_NODES_DIR}/*.hpp
    ${DWCGF_NODES_DIR}/*.cpp
)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES
    HelloWorldNode.cpp
    HelloWorldNode.hpp
    HelloWorldNodeImpl.cpp
    HelloWorldNodeImpl.hpp
    SumNode.cpp
    SumNode.hpp
    SumNodeImpl.cpp
    SumNodeImpl.hpp
)

if(NOT TARGET driveworks-shared)
    find_package(driveworks-shared REQUIRED CONFIG PATHS "${DRIVEWORKS_DIR}/cmake" NO_DEFAULT_PATH)
endif()

if(NOT TARGET dwframework)
    find_package(dwframework REQUIRED CONFIG PATHS "${DRIVEWORKS_DIR}/cmake" NO_DEFAULT_PATH)
endif()

set(LIBRARIES
    samples_framework
    ${Driveworks_LIBRARIES}
    driveworks-shared
    dwframework
)

file(GLOB_RECURSE DWCGF_GRAPHS_FILES
    ${DWCGF_GRAPHS_DIR}/*.yaml
    ${DWCGF_GRAPHS_DIR}/*.json
)
# json-validater execute during configuring time.
execute_process(
    COMMAND ${DRIVEWORKS_PATH}/tools/schema/validate_cgfdescriptors.py --ignore-order --ignore-array-order --ignore-indentation ${DWCGF_GRAPHS_DIR}
    WORKING_DIRECTORY ${DWCGF_GRAPHS_DIR}
    TIMEOUT 3
    RESULT_VARIABLE _EXIT_CODE
    OUTPUT_VARIABLE _OUTPUT_CODE
    COMMAND_ECHO STDOUT
    # OUTPUT_QUIET
    # ERROR_QUIET
    # ECHO_OUTPUT_VARIABLE
    # https://cmake.org/cmake/help/v3.16/command/execute_process.html
)
message(STATUS "${_OUTPUT_CODE}")
if(NOT _EXIT_CODE EQUAL 0)
    message(FATAL_ERROR "validate_cgfdescriptors.py failed: please check the json files.")
endif()

#-------------------------------------------------------------------------------
# Custom nodes shared library
#-------------------------------------------------------------------------------
# zs: bug: must be compiled with channel src files
file(GLOB_RECURSE DWCGF_CHANNEL_FILES
    ${DWCGF_CHANNEL_SRC_DIR}/*.cpp
)
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${DWCGF_NODES_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})
target_include_directories(${PROJECT_NAME}
                           PUBLIC
                           include
                           src
                           ${CUDA_DIR}/include
                           ${DW_ROOT}/include
                           ${DW_ROOT}/include/3rdparty
                           ${DW_ROOT}/include/dwshared/dwfoundation
                           ${DW_ROOT}/include/dwshared/dwtrace
                           ${NSIGHT_SYS_PATH}/target-linux-tegra-armv8/nvtx/include)

#-------------------------------------------------------------------------------
# tools
#-------------------------------------------------------------------------------
add_executable(tcp_nodelay_preload_test
    ./tools/tcp_nodelay_preload_test.cpp
)
sdk_add_sample(tcp_nodelay_preload_test)

# cgf generation
set(GW_MODULE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(GW_MODULE_BINARY_DESTINATION  ${CMAKE_INSTALL_PREFIX}/bin/${PROJECT_NAME}/)
set(GW_MODULE_LIBRARY_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}/)
set(GW_MODULE_ARCHIVE_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}/)
set(GW_MODULE_DATA_DESTINATION ${CMAKE_INSTALL_PREFIX}/data/${PROJECT_NAME})
set(GW_MODULE_GRAPHS_DESTINATION ${CMAKE_INSTALL_PREFIX}/graphs/${PROJECT_NAME})

set(DWCGF_APP_LIST
    example/appHelloworld;
)
foreach(DWCGF_APP ${DWCGF_APP_LIST})
    get_filename_component(DWCGF_APP_DIRNAME ${DWCGF_APPS_DIR}/${DWCGF_APP} NAME)
    get_filename_component(DWCGF_APP_FOLDERPATH ${DWCGF_APPS_DIR}/${DWCGF_APP} PATH)
    get_filename_component(DWCGF_APP_FOLDERNAME ${DWCGF_APP_FOLDERPATH} NAME)
    string(REGEX REPLACE "^app" "" DWCGF_APP_TYPE ${DWCGF_APP_DIRNAME})
    set(DWCGF_FILE_PREFIX DWCGF${DWCGF_APP_TYPE})
    set(DWCGF_RUNNER ${DWCGF_RUNNER_DIR}/${DWCGF_APP_FOLDERNAME}/run${DWCGF_APP_TYPE}.sh)
    set(DWCGF_YAML ${DWCGF_SCHE_DIR}/${DWCGF_FILE_PREFIX}__standardSchedule.yaml)
    set(DWCGF_YAML_LOG ${DWCGF_SCHE_DIR}/${DWCGF_FILE_PREFIX}__standardSchedule.yaml.log)
    set(DWCGF_STM ${DWCGF_SCHE_DIR}/${DWCGF_FILE_PREFIX}__standardSchedule.stm)
    set(DWCGF_STM_LOG ${DWCGF_SCHE_DIR}/${DWCGF_FILE_PREFIX}__standardSchedule.stm.log)
    set(DWCGF_SVG ${DWCGF_SCHE_DIR}/${DWCGF_FILE_PREFIX}__standardSchedule.svg)
    set(DWCGF_SVG_LOG ${DWCGF_SCHE_DIR}/${DWCGF_FILE_PREFIX}__standardSchedule.svg.log)
    set(DWCGF_HTML ${DWCGF_SCHE_DIR}/${DWCGF_FILE_PREFIX}__standardSchedule.html)
    set(DWCGF_HTML_LOG ${DWCGF_SCHE_DIR}/${DWCGF_FILE_PREFIX}__standardSchedule.html.log)
    # app2yaml
    add_custom_command(
        OUTPUT ${DWCGF_YAML}
        COMMAND ${DRIVEWORKS_PATH}/tools/descriptionScheduleYamlGenerator/descriptionScheduleYamlGenerator.py --app ${DWCGF_APPS_DIR}/${DWCGF_APP}/${DWCGF_FILE_PREFIX}.app.json --output ${DWCGF_YAML} 2>&1 | tee ${DWCGF_YAML_LOG}
        WORKING_DIRECTORY ${DWCGF_SCHE_DIR}
        MAIN_DEPENDENCY ${DWCGF_APPS_DIR}/${DWCGF_APP}/${DWCGF_FILE_PREFIX}.app.json
        DEPENDS ${DWCGF_GRAPHS_FILES}
        COMMENT "app2yaml[${DWCGF_APP_TYPE}]...!"
        VERBATIM
    )
    add_custom_target(${PROJECT_NAME}_yaml${DWCGF_APP_TYPE} ALL DEPENDS ${DWCGF_YAML})
    # yaml2stm
    add_custom_command(
        OUTPUT ${DWCGF_STM}
        COMMAND ${DRIVEWORKS_PATH}/tools/stmcompiler -i ${DWCGF_YAML} -k 2>&1 | tee ${DWCGF_STM_LOG}
        WORKING_DIRECTORY ${DWCGF_SCHE_DIR}
        MAIN_DEPENDENCY ${DWCGF_YAML}
        DEPENDS ${PROJECT_NAME}_yaml${DWCGF_APP_TYPE}
        COMMENT "yaml2stm[${DWCGF_APP_TYPE}]...!"
        VERBATIM
    )
    add_custom_target(${PROJECT_NAME}_stm${DWCGF_APP_TYPE} ALL DEPENDS ${DWCGF_STM})
    # yaml2svg
    # add_custom_command(
    #     OUTPUT ${DWCGF_SVG}
    #     COMMAND ${DRIVEWORKS_PATH}/tools/stmvizgraph -i ${DWCGF_YAML} -o ${DWCGF_SVG} 2>&1 | tee ${DWCGF_SVG_LOG}
    #     WORKING_DIRECTORY ${DWCGF_SCHE_DIR}
    #     MAIN_DEPENDENCY ${DWCGF_YAML}
    #     DEPENDS ${PROJECT_NAME}_yaml${DWCGF_APP_TYPE}
    #     COMMENT "yaml2svg[${DWCGF_APP_TYPE}]...!"
    #     VERBATIM
    # )
    # add_custom_target(${PROJECT_NAME}_svg${DWCGF_APP_TYPE} ALL DEPENDS ${DWCGF_SVG})
    # stm2html
    add_custom_command(
        OUTPUT ${DWCGF_HTML}
        COMMAND ${DRIVEWORKS_PATH}/tools/stmcompiler -i ${DWCGF_YAML} -o ${DWCGF_STM} -k 2>&1 | tee ${DWCGF_STM_LOG}
        COMMAND ${DRIVEWORKS_PATH}/tools/stmvizschedule -i ${DWCGF_STM} -o ${DWCGF_HTML} 2>&1 | tee ${DWCGF_HTML_LOG}
        WORKING_DIRECTORY ${DWCGF_SCHE_DIR}
        MAIN_DEPENDENCY ${DWCGF_STM}
        DEPENDS ${PROJECT_NAME}_stm${DWCGF_APP_TYPE}
        COMMENT "stm2html[${DWCGF_APP_TYPE}]...!"
        VERBATIM
    )
    add_custom_target(${PROJECT_NAME}_html${DWCGF_APP_TYPE} ALL DEPENDS ${DWCGF_HTML})
    install(FILES ${DWCGF_YAML} DESTINATION "${GW_MODULE_BINARY_DESTINATION}")
    # install(FILES ${DWCGF_SVG} DESTINATION "${GW_MODULE_BINARY_DESTINATION}")
    install(FILES ${DWCGF_STM} DESTINATION "${GW_MODULE_BINARY_DESTINATION}")
    install(FILES ${DWCGF_HTML} DESTINATION "${GW_MODULE_BINARY_DESTINATION}")
    install(PROGRAMS ${DWCGF_RUNNER} DESTINATION ${GW_MODULE_BINARY_DESTINATION}/${DWCGF_APP_FOLDERNAME})
endforeach()

# ------------------------------------------------------------------------------
# Install target
# ------------------------------------------------------------------------------
sdk_add_sample(${PROJECT_NAME})

install(DIRECTORY ${DWCGF_GRAPHS_DIR}/ DESTINATION ${GW_MODULE_GRAPHS_DESTINATION})
install(DIRECTORY ${DWCGF_DATA_DIR}/ DESTINATION ${GW_MODULE_DATA_DESTINATION})
install(PROGRAMS ${DWCGF_NVSCI_RST} DESTINATION ${GW_MODULE_BINARY_DESTINATION})
install(PROGRAMS ${TCP_NODELAY_PRELOAD_SO} DESTINATION ${GW_MODULE_BINARY_DESTINATION})

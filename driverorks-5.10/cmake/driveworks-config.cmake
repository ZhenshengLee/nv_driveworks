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
# SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(NAME driveworks)
set(LIBS dw_base dw_calibration dw_egomotion dw_imageprocessing dw_pointcloudprocessing dw_sensors dw_vehicleio dw_dnn_base)

# Interface target that depends on all real so's
add_library(${NAME} INTERFACE)

# search for the library and for the include path
foreach(LIB ${LIBS})

    # Include dir
    get_filename_component(_DRIVEWORKS_DIR ${CMAKE_CURRENT_LIST_DIR} REALPATH)
    get_filename_component(_DRIVEWORKS_DIR ${_DRIVEWORKS_DIR} DIRECTORY)   
    set(PATH "${_DRIVEWORKS_DIR}/targets/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}/include"
             "${_DRIVEWORKS_DIR}/include")
    find_path(${LIB}_INCLUDE_DIR
      NAMES dw/core/base/Version.h
      PATHS ${PATH}
      NO_DEFAULT_PATH
    )

    if (NOT ${LIB}_INCLUDE_DIR)
        message(FATAL_ERROR "Cannot find 'dw/core/base/Version.h' for ${LIB} at ${PATH}")
    else()
        message(STATUS "Found 'dw/core/base/Version.h' in ${${LIB}_INCLUDE_DIR}")
    endif()

    target_include_directories(${NAME} INTERFACE "${${LIB}_INCLUDE_DIR}")

    # library
    set(PATH "${_DRIVEWORKS_DIR}/targets/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}/lib"
             "${_DRIVEWORKS_DIR}/lib")
    find_library(${LIB}_LIBRARY
        NAMES ${LIB}
        PATHS ${PATH}
        NO_DEFAULT_PATH
    )

    if (NOT ${LIB}_LIBRARY)
        message(FATAL_ERROR "Cannot find ${LIB} library at ${PATH}")
    else()
        message(STATUS "Found ${LIB} library in ${${LIB}_LIBRARY}")
    endif()

    # create imported library from the found one
    add_library(driveworks::${LIB}-lib SHARED IMPORTED)

    set_target_properties(driveworks::${LIB}-lib PROPERTIES
      IMPORTED_LOCATION "${${LIB}_LIBRARY}"
    )

    target_link_libraries(${NAME} INTERFACE driveworks::${LIB}-lib)
    unset(_DRIVEWORKS_DIR)
endforeach()

# nvupload (if linked) currently brings this dependency
if (TARGET dw_base-lib AND TARGET zlib)
    target_link_libraries(dw_base-lib INTERFACE zlib)
endif()

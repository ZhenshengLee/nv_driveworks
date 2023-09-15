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

set(NAME dwcgf)
set(LIBS dwcgf)

# Interface target that depends on all real so's
add_library(${NAME} INTERFACE)

# search for the library and for the include path
foreach(LIB ${LIBS})

    # Include dir
    get_filename_component(_DRIVEWORKS_DIR ${CMAKE_CURRENT_LIST_DIR} REALPATH)
    get_filename_component(_DRIVEWORKS_DIR ${_DRIVEWORKS_DIR} DIRECTORY)   
    set(PATH "${_DRIVEWORKS_DIR}/targets/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}/include"
             "${_DRIVEWORKS_DIR}/include")
    find_path(${LIB}_CGF_INCLUDE_DIR
      NAMES dwcgf/channel/ChannelFactory.hpp
      PATHS ${PATH}
      NO_DEFAULT_PATH
    )

    if (NOT ${LIB}_CGF_INCLUDE_DIR)
        message(FATAL_ERROR "Cannot find 'dwcgf/channel/ChannelFactory.hpp' for ${LIB} at ${PATH}")
    else()
        message(STATUS "Found 'dwcgf/channel/ChannelFactory.hpp' in ${${LIB}_CGF_INCLUDE_DIR}")
    endif()

    target_include_directories(${NAME} INTERFACE "${${LIB}_CGF_INCLUDE_DIR}")

    set(3RDPARTY_PATH "${_DRIVEWORKS_DIR}/targets/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}/include/3rdparty"
                      "${_DRIVEWORKS_DIR}/include/3rdparty")
    find_path(${LIB}_MODERN_JSON_INCLUDE_DIR
      NAMES modern-json/json.hpp
      PATHS ${3RDPARTY_PATH}
      NO_DEFAULT_PATH
    )

    if (NOT ${LIB}_MODERN_JSON_INCLUDE_DIR)
        message(FATAL_ERROR "Cannot find 'modern-json/json.hpp' for ${LIB} at ${3RDPARTY_PATH}")
    else()
        message(STATUS "Found 'modern-json/json.hpp' in ${${LIB}_MODERN_JSON_INCLUDE_DIR}")
    endif()

    target_include_directories(${NAME} INTERFACE "${${LIB}_MODERN_JSON_INCLUDE_DIR}")

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
    add_library(dwcgf::${LIB}-lib SHARED IMPORTED)

    set_target_properties(dwcgf::${LIB}-lib PROPERTIES
      IMPORTED_LOCATION "${${LIB}_LIBRARY}"
    )

    target_link_libraries(${NAME} INTERFACE dwcgf::${LIB}-lib)
    unset(_DRIVEWORKS_DIR)
endforeach()

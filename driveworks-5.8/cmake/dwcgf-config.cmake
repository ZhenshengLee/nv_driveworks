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
    set(PATH ${CMAKE_CURRENT_LIST_DIR}/../targets/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}/include
             ${CMAKE_CURRENT_LIST_DIR}/../include)
    find_path(${LIB}_INCLUDE_DIR
      NAMES dw/core/base/Version.h
      PATHS ${PATH}
      NO_DEFAULT_PATH
      NO_CMAKE_PATH
      NO_CMAKE_FIND_ROOT_PATH
    )

    if (NOT ${LIB}_INCLUDE_DIR)
        message(FATAL_ERROR "Cannot find 'dw/core/base/Version.h' for ${LIB} at ${PATH}")
    else()
        message(STATUS "Found 'dw/core/base/Version.h' in ${${LIB}_INCLUDE_DIR}")
    endif()

    target_include_directories(${NAME} INTERFACE "${${LIB}_INCLUDE_DIR}/dwcgf")

    # library
    set(PATH ${CMAKE_CURRENT_LIST_DIR}/../targets/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}/lib
             ${CMAKE_CURRENT_LIST_DIR}/../lib)
    find_library(${LIB}_LIBRARY
        NAMES ${LIB}
        PATHS ${PATH}
        NO_DEFAULT_PATH
        NO_CMAKE_PATH
        NO_CMAKE_FIND_ROOT_PATH
    )

    if (NOT ${LIB}_LIBRARY)
        message(FATAL_ERROR "Cannot find ${LIB} library at ${PATH}")
    else()
        message(STATUS "Found ${LIB} library in ${${LIB}_LIBRARY}")
    endif()

    # create imported library from the found one
    add_library(${LIB}-lib SHARED IMPORTED)

    set_target_properties(${LIB}-lib PROPERTIES
      IMPORTED_LOCATION_RELEASE     "${${LIB}_LIBRARY}"
      IMPORTED_CONFIGURATIONS       RELEASE
    )

    target_link_libraries(${NAME} INTERFACE ${LIB}-lib)
endforeach()

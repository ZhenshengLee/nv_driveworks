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
# SPDX-FileCopyrightText: Copyright (c) 2016-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#-------------------------------------------------------------------------------
# Upload to Vibrante board target
#-------------------------------------------------------------------------------
if(VIBRANTE_BUILD)
    set(VIBRANTE_INSTALL_PATH "/home/nvidia/driveworks" CACHE STRING "Directory on the target board where to install the SDK")
    set(VIBRANTE_USER "nvidia" CACHE STRING "User used for ssh to upload files over to the board")
    set(VIBRANTE_PASSWORD "nvidia" CACHE STRING "Password of the specified user")
    set(VIBRANTE_HOST "192.168.10.10" CACHE STRING "Hostname or IP address of the tegra board")
    set(VIBRANTE_PORT "22" CACHE STRING "SSH port of the tegra board")

    if(SDK_DEPLOY_BUILD OR ${PROJECT_NAME} MATCHES "DriveworksSDK-Samples")
        if(NOT TARGET upload)
                add_custom_target(upload
                    # create installation
                    COMMAND "${CMAKE_COMMAND}" --build ${SDK_BINARY_DIR} --target install
                    # create installation folder on target
                    COMMAND sshpass -p "${VIBRANTE_PASSWORD}" ssh -o StrictHostKeyChecking=no -p ${VIBRANTE_PORT} ${VIBRANTE_USER}@${VIBRANTE_HOST} "mkdir -p ${VIBRANTE_INSTALL_PATH}"
                    # upload installation
                    COMMAND sshpass -p "${VIBRANTE_PASSWORD}" rsync --progress -rltgDz -e "ssh -p ${VIBRANTE_PORT}" ${CMAKE_INSTALL_PREFIX}/ ${VIBRANTE_USER}@${VIBRANTE_HOST}:${VIBRANTE_INSTALL_PATH}/
                    )
        endif()
    endif()
endif()

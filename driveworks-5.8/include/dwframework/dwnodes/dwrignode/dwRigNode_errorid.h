/////////////////////////////////////////////////////////////////////////////////////////
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA CORPORATION & AFFILIATES assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA CORPORATION & AFFILIATES. No third party distribution is allowed unless
// expressly authorized by NVIDIA. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA CORPORATION & AFFILIATES products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA CORPORATION & AFFILIATES.
//
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_FRAMEWORK_NODES_RIGNODE_RIGNODEERRORID_H_
#define DW_FRAMEWORK_NODES_RIGNODE_RIGNODEERRORID_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Rig node specific errors.
 * 
 * NOMINALS_MISSING indicates that the sensor has identity set as the nominal transform.
 */
typedef enum dwRigNodeErrorID {
    DW_RIGNODE_ERROR_ID_INVALID = 0, ///< invalid error id

    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_0 = 100, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_1 = 101, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_2 = 102, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_3 = 103, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_4 = 104, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_5 = 105, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_6 = 106, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_7 = 107, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_8 = 108, ///< nominal transform is identity

    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_RADAR_0 = 200, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_RADAR_1 = 201, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_RADAR_2 = 202, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_RADAR_3 = 203, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_RADAR_4 = 204, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_RADAR_5 = 205, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_RADAR_6 = 206, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_RADAR_7 = 207, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_RADAR_8 = 208, ///< nominal transform is identity

    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_LIDAR_0 = 300, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_LIDAR_1 = 301, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_LIDAR_2 = 302, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_LIDAR_3 = 303, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_LIDAR_4 = 304, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_LIDAR_5 = 305, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_LIDAR_6 = 306, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_LIDAR_7 = 307, ///< nominal transform is identity

    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_IMU_0 = 400, ///< nominal transform is identity
} dwRigNodeErrorID;

#ifdef __cplusplus
} // extern C
#endif

#endif // DW_FRAMEWORK_NODES_RIGNODE_RIGNODEERRORID_H_
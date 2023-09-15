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
 * CORRECTION_MISSING indicates that the sensor has no correction values from extrinsic calibration.
 */
typedef enum dwRigNodeErrorID {
    DW_RIGNODE_ERROR_ID_INVALID = 0, ///< invalid error id

    // NOMINALS_MISSING
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_0  = 100, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_1  = 101, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_2  = 102, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_3  = 103, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_4  = 104, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_5  = 105, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_6  = 106, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_7  = 107, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_8  = 108, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_9  = 109, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_10 = 110, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_11 = 111, ///< nominal transform is identity
    DW_RIGNODE_ERROR_ID_NOMINALS_MISSING_CAMERA_12 = 112, ///< nominal transform is identity

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

    // EXTRINSICS_MISSING
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_CAMERA_0  = 1100, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_CAMERA_1  = 1101, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_CAMERA_2  = 1102, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_CAMERA_3  = 1103, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_CAMERA_4  = 1104, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_CAMERA_5  = 1105, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_CAMERA_6  = 1106, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_CAMERA_7  = 1107, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_CAMERA_8  = 1108, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_CAMERA_9  = 1109, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_CAMERA_10 = 1110, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_CAMERA_11 = 1111, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_CAMERA_12 = 1112, ///< extrinsics correction missing

    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_RADAR_0 = 1200, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_RADAR_1 = 1201, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_RADAR_2 = 1202, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_RADAR_3 = 1203, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_RADAR_4 = 1204, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_RADAR_5 = 1205, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_RADAR_6 = 1206, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_RADAR_7 = 1207, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_RADAR_8 = 1208, ///< extrinsics correction missing

    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_LIDAR_0 = 1300, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_LIDAR_1 = 1301, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_LIDAR_2 = 1302, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_LIDAR_3 = 1303, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_LIDAR_4 = 1304, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_LIDAR_5 = 1305, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_LIDAR_6 = 1306, ///< extrinsics correction missing
    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_LIDAR_7 = 1307, ///< extrinsics correction missing

    DW_RIGNODE_ERROR_ID_EXTRINSICS_MISSING_IMU_0 = 1400, ///< extrinsics correction missing
} dwRigNodeErrorID;

#ifdef __cplusplus
} // extern C
#endif

#endif // DW_FRAMEWORK_NODES_RIGNODE_RIGNODEERRORID_H_
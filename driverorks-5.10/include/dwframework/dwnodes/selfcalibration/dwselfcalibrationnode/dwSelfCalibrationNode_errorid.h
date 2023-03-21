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

#ifndef DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONNODE_DWSELFCALIBRATIONNODE_ERRORID_H_
#define DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONNODE_DWSELFCALIBRATIONNODE_ERRORID_H_

#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * error-ids reported by the self-calibration node
 */
typedef enum dwSelfCalibrationErrorID {
    DW_SELFCALIBRATION_ERROR_ID_INVALID = 0,          //!<  0 - invalid error id
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_0_FAILED,      //!<  1 - camera 0 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_1_FAILED,      //!<  2 - camera 1 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_2_FAILED,      //!<  3 - camera 2 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_3_FAILED,      //!<  4 - camera 3 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_4_FAILED,      //!<  5 - camera 4 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_5_FAILED,      //!<  6 - camera 5 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_6_FAILED,      //!<  7 - camera 6 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_7_FAILED,      //!<  8 - camera 7 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_8_FAILED,      //!<  9 - camera 8 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_9_FAILED,      //!<  10 - camera 9 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_10_FAILED,     //!<  11 - camera 10 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_11_FAILED,     //!<  12 - camera 11 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_12_FAILED,     //!<  13 - camera 12 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_0_FAILED,       //!< 12 - radar 0 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_1_FAILED,       //!< 13 - radar 1 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_2_FAILED,       //!< 14 - radar 2 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_3_FAILED,       //!< 15 - radar 3 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_4_FAILED,       //!< 16 - radar 4 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_5_FAILED,       //!< 17 - radar 5 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_6_FAILED,       //!< 18 - radar 6 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_7_FAILED,       //!< 19 - radar 7 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_8_FAILED,       //!< 20 - radar 8 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_0_FAILED,       //!< 21 - lidar 0 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_1_FAILED,       //!< 22 - lidar 1 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_2_FAILED,       //!< 23 - lidar 2 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_3_FAILED,       //!< 24 - lidar 3 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_4_FAILED,       //!< 25 - lidar 4 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_5_FAILED,       //!< 26 - lidar 5 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_6_FAILED,       //!< 27 - lidar 6 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_7_FAILED,       //!< 28 - lidar 7 self-calibration failed error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_0_DEADJUSTED,  //!<  29 - camera 0 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_1_DEADJUSTED,  //!<  30 - camera 1 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_2_DEADJUSTED,  //!<  31 - camera 2 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_3_DEADJUSTED,  //!<  32 - camera 3 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_4_DEADJUSTED,  //!<  33 - camera 4 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_5_DEADJUSTED,  //!<  34 - camera 5 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_6_DEADJUSTED,  //!<  35 - camera 6 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_7_DEADJUSTED,  //!<  36 - camera 7 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_8_DEADJUSTED,  //!<  37 - camera 8 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_9_DEADJUSTED,  //!<  38 - camera 9 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_10_DEADJUSTED, //!<  39 - camera 10 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_11_DEADJUSTED, //!<  40 - camera 11 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_12_DEADJUSTED, //!<  41 - camera 12 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_0_DEADJUSTED,   //!< 42 - radar 0 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_1_DEADJUSTED,   //!< 43 - radar 1 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_2_DEADJUSTED,   //!< 44 - radar 2 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_3_DEADJUSTED,   //!< 45 - radar 3 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_4_DEADJUSTED,   //!< 46 - radar 4 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_5_DEADJUSTED,   //!< 47 - radar 5 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_6_DEADJUSTED,   //!< 48 - radar 6 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_7_DEADJUSTED,   //!< 49 - radar 7 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_RADAR_8_DEADJUSTED,   //!< 50 - radar 8 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_0_DEADJUSTED,   //!< 51 - lidar 0 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_1_DEADJUSTED,   //!< 52 - lidar 1 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_2_DEADJUSTED,   //!< 53 - lidar 2 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_3_DEADJUSTED,   //!< 54 - lidar 3 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_4_DEADJUSTED,   //!< 55 - lidar 4 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_5_DEADJUSTED,   //!< 56 - lidar 5 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_6_DEADJUSTED,   //!< 57 - lidar 6 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_7_DEADJUSTED,   //!< 58 - lidar 7 sensor deadjusted error
    DW_SELFCALIBRATION_ERROR_ID_NUM                   //!< 59 - number of self-calibration error ids
} dwSelfCalibrationErrorID;

#ifdef __cplusplus
} // extern C
#endif

#endif // DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONNODE_DWSELFCALIBRATIONNODE_ERRORID_H_
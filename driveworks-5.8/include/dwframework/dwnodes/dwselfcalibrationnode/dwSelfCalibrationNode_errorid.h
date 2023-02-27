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

#ifndef DW_FRAMEWORK_NODES_SELFCALIBRATIONNODE_SELFCALIBRATIONERRORID_H_
#define DW_FRAMEWORK_NODES_SELFCALIBRATIONNODE_SELFCALIBRATIONERRORID_H_

#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * error-ids reported by the self-calibration node
 */
typedef enum dwSelfCalibrationErrorID {
    DW_SELFCALIBRATION_ERROR_ID_INVALID = 0, //!<  0 - invalid error id
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_0,    //!<  1 - camera 0 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_1,    //!<  2 - camera 1 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_2,    //!<  3 - camera 2 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_3,    //!<  4 - camera 3 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_4,    //!<  5 - camera 4 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_5,    //!<  6 - camera 5 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_6,    //!<  7 - camera 6 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_7,    //!<  8 - camera 7 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_CAMERA_8,    //!<  9 - camera 8 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_RADAR_0,     //!< 10 - radar 0 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_RADAR_1,     //!< 11 - radar 1 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_RADAR_2,     //!< 12 - radar 2 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_RADAR_3,     //!< 13 - radar 3 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_RADAR_4,     //!< 14 - radar 4 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_RADAR_5,     //!< 15 - radar 5 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_RADAR_6,     //!< 16 - radar 6 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_RADAR_7,     //!< 17 - radar 7 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_RADAR_8,     //!< 18 - radar 8 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_0,     //!< 19 - lidar 0 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_1,     //!< 20 - lidar 1 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_2,     //!< 21 - lidar 2 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_3,     //!< 22 - lidar 3 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_4,     //!< 23 - lidar 4 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_5,     //!< 24 - lidar 5 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_6,     //!< 25 - lidar 6 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_LIDAR_7,     //!< 26 - lidar 7 self-calibration failed
    DW_SELFCALIBRATION_ERROR_ID_NUM          //!< 27 - number of self-calibration error ids
} dwSelfCalibrationErrorID;

#ifdef __cplusplus
} // extern C
#endif

static_assert(dw::framework::SELF_CALIBRATION_NODE_MAX_CAMERAS == (DW_SELFCALIBRATION_ERROR_ID_RADAR_0 - DW_SELFCALIBRATION_ERROR_ID_CAMERA_0), "number of cameras in dwSelfCalibrationErrorID doesn't match SELF_CALIBRATION_NODE_MAX_CAMERAS");
static_assert(dw::framework::SELF_CALIBRATION_NODE_MAX_RADARS == (DW_SELFCALIBRATION_ERROR_ID_LIDAR_0 - DW_SELFCALIBRATION_ERROR_ID_RADAR_0), "number of radars in dwSelfCalibrationErrorID doesn't match SELF_CALIBRATION_NODE_MAX_CAMERAS");
static_assert(dw::framework::SELF_CALIBRATION_NODE_MAX_LIDARS == (DW_SELFCALIBRATION_ERROR_ID_NUM - DW_SELFCALIBRATION_ERROR_ID_LIDAR_0), "number of lidars in dwSelfCalibrationErrorID doesn't match SELF_CALIBRATION_NODE_MAX_CAMERAS");

#endif // DW_FRAMEWORK_NODES_SELFCALIBRATIONNODE_SELFCALIBRATIONERRORID_H_
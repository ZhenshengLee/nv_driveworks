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
// SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_ROADCAST_COMMON_TYPES_HPP_
#define DW_ROADCAST_COMMON_TYPES_HPP_

#include <dw/core/base/Types.h>
#include <dw/rig/Rig.h>
#include <dw/calibration/engine/common/CalibrationTypes.h>
#include <dw/calibration/engine/common/CalibrationTypesExtra.h>

namespace dw
{
namespace framework
{

#define MAX_CALIBRATION_SENSORS 32

using dwRoadCastNodeCalibrationData = struct dwRoadCastNodeCalibrationData
{
    char sensorName[DW_MAX_RIG_SENSOR_NAME_SIZE];
    char extrinsicProfileName[DW_MAX_EXTRINSIC_PROFILE_NAME_SIZE];
    dwCalibrationSensorPositionState positionState;
    bool positionStateChanged;
    dwTransformation3f sensorToRig;
    dwCalibrationStatus status;
    bool calibrationStateChanged;
    dwTime_t timestamp;
    dwCalibrationManeuverArray maneuvers;
    dwCalibrationProperties properties;
};

using dwRoadCastNodeCalibrationDataArray = struct dwRoadCastNodeCalibrationDataArray
{
    dwRoadCastNodeCalibrationData calibrationData[MAX_CALIBRATION_SENSORS];
    uint32_t count;
};

} // namespace framework
} // namespace dw

#endif // DW_ROADCAST_COMMON_TYPES_HPP_

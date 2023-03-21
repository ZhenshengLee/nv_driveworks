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

#ifndef DWFRAMEWORK_DWNODES_COMMON_SELFCALIBRATIONTYPES_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_SELFCALIBRATIONTYPES_HPP_

#include <dw/calibration/engine/common/CalibrationTypes.h>
#include <dw/calibration/engine/common/CalibrationTypesExtra.h>
#include <dw/calibration/engine/vehicle/VehicleParams.h>
#include <dw/core/health/HealthSignals.h>
#include <dw/rig/Vehicle.h>
#include <dw/rig/Rig.h>
#include <dw/core/base/Types.h>
#include <dw/core/container/BaseString.hpp>

namespace dw
{
namespace framework
{

static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_IMUS          = 3;
static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_VEHICLEIO     = 1;
static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_CAMERAS       = 13;
static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_RADARS        = 9;
static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_LIDARS        = 8;
static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_ROUTINE_COUNT = SELF_CALIBRATION_NODE_MAX_CAMERAS * MAX_EXTRINSIC_PROFILE_COUNT +
                                                                   SELF_CALIBRATION_NODE_MAX_RADARS +
                                                                   SELF_CALIBRATION_NODE_MAX_LIDARS +
                                                                   SELF_CALIBRATION_NODE_MAX_IMUS +
                                                                   SELF_CALIBRATION_NODE_MAX_VEHICLEIO;
static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_WHEELS = 4;

// For every estimation, a property will be added to the rig file by the dwRigNode
// to indicating results of previous calibration:
//
// "properties": {
//     "self-calibration": "accepted"
// },
static constexpr char SELF_CALIBRATION_TAG[]                  = "self-calibration";
static constexpr char CALIBRATION_ACCEPTED_STATE_STRING[]     = "accepted";
static constexpr char CALIBRATION_NOT_ACCEPTED_STATE_STRING[] = "not-accepted";
static constexpr char CALIBRATION_FAILED_STATE_STRING[]       = "failed";
static constexpr char CALIBRATION_INVALID_STATE_STRING[]      = "invalid";
// Additionally, the property "supports-eeprom-intrinsics" will be set to "true" if the intrinsics written to rig
// are from EEPROM; otherwise the property will be propagated from the input rig file.
static constexpr char SUPPORTS_EEPROM_INTRINSICS_TAG[]         = "supports-eeprom-intrinsics";
static constexpr char SUPPORTS_EEPROM_INTRINSICS_TRUE_STRING[] = "true";

using CalibrationResultsString        = dw::core::FixedString<512>;
using CalibrationIntrinsicsString     = dw::core::FixedString<64>;
using CalibrationExtrinsicProfileName = dw::core::FixedString<DW_MAX_EXTRINSIC_PROFILE_NAME_SIZE>;

struct dwCalibratedExtrinsics
{
    CalibrationExtrinsicProfileName extrinsicProfileName;
    dwCalibrationSensorPositionState positionState;
    bool positionStateChanged;
    dwTransformation3f currentSensor2Rig;
    dwCalibrationStatus status;
    bool calibrationStateChanged;
    dwCalibrationProperties properties;
    dwCalibrationManeuverArray maneuvers;
    // Timestamp of the latest input data contributing to the calibration result
    dwTime_t timestamp;
};

struct dwCalibratedWheelRadii
{
    float32_t currentWheelRadius[DW_VEHICLE_NUM_WHEELS];
    dwCalibrationStatus status;
    dwCalibrationProperties properties;
    dwCalibrationManeuverArray maneuvers;
    // Timestamp of the latest input data contributing to the calibration result
    dwTime_t timestamp;
};

struct dwCalibratedSteeringProperties
{
    dwVehicleSteeringProperties steeringProperties;
    dwCalibrationStatus status;
    dwCalibrationProperties properties;
    dwCalibrationManeuverArray maneuvers;
};

struct dwCalibratedIMUIntrinsics
{
    dwVector3f gyroscopeBias;
    dwVector3f accelerometerBias;
    /// Calibration status for accelerometer bias calibration.
    dwCalibrationStatus status;
    /// Calibration properties for accelerometer bias calibration.
    dwCalibrationProperties properties;
    /// Required calibration maneuvers for accelerometer bias calibration.
    dwCalibrationManeuverArray maneuvers;
    uint32_t sensorID;
    bool validGyroscopeBias;
    bool validAccelerometerBias;
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_SELFCALIBRATIONTYPES_HPP_

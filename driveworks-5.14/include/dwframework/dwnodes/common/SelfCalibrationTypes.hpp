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
// SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <dwshared/dwfoundation/dw/core/container/BaseString.hpp>

namespace dw
{
namespace framework
{

static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_IMUS{3U};
static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_VEHICLEIO{1U};
static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_CAMERAS{13U};
static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_RADARS{9U};
static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_LIDARS{8U};
static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_ROUTINE_COUNT{SELF_CALIBRATION_NODE_MAX_CAMERAS * DW_MAX_EXTRINSIC_PROFILE_COUNT +
                                                                 SELF_CALIBRATION_NODE_MAX_RADARS +
                                                                 SELF_CALIBRATION_NODE_MAX_LIDARS +
                                                                 SELF_CALIBRATION_NODE_MAX_IMUS +
                                                                 SELF_CALIBRATION_NODE_MAX_VEHICLEIO};
static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_WHEELS{4U};
static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_NUM_SIGNALS{8U};

// For every estimation, a property will be added to the rig file by the dwRigNode
// to indicating results of previous calibration:
//
// "properties": {
//     "self-calibration": "accepted"
// },
static constexpr char8_t SELF_CALIBRATION_TAG[]{"self-calibration"};
static constexpr char8_t CALIBRATION_ACCEPTED_STATE_STRING[]{"accepted"};
static constexpr char8_t CALIBRATION_NOT_ACCEPTED_STATE_STRING[]{"not-accepted"};
static constexpr char8_t CALIBRATION_FAILED_STATE_STRING[]{"failed"};
static constexpr char8_t CALIBRATION_INVALID_STATE_STRING[]{"invalid"};
// Additionally, the property "supports-eeprom-intrinsics" will be set to "true" if the intrinsics written to rig
// are from EEPROM; otherwise the property will be propagated from the input rig file.
static constexpr char8_t SUPPORTS_EEPROM_INTRINSICS_TAG[]{"supports-eeprom-intrinsics"};
static constexpr char8_t SUPPORTS_EEPROM_INTRINSICS_TRUE_STRING[]{"true"};

using CalibrationResultsString        = dw::core::FixedString<512>;
using CalibrationIntrinsicsString     = dw::core::FixedString<64>;
using CalibrationExtrinsicProfileName = dw::core::FixedString<DW_MAX_EXTRINSIC_PROFILE_NAME_SIZE>;

struct dwCalibratedExtrinsics
{
    CalibrationExtrinsicProfileName extrinsicProfileName; //!< name of the currently selected profile
    dwCalibrationSensorPositionState positionState;       //!< current position status of the sensor
    bool positionStateChanged;                            //!< set to true for one cycle when `positionState` changed
    dwTransformation3f currentSensor2Rig;                 //!< currently estimated sensor2rig
    dwCalibrationStatus status;                           //!< overall status of the whole routine, i.e. representing senor2Rig pose
    bool calibrationStateChanged;                         //!< set to true for one cycle when `status` changed
    dwCalibrationProperties properties;                   //!< description of where the value is coming from
    dwCalibrationManeuverArray maneuvers;                 //!< maneuvers left to achieve full calibration
    dwTime_t timestamp;                                   //!< timestamp of the latest input data contributin to the result

    dwCalibrationStatus signalStatus[SELF_CALIBRATION_NODE_MAX_NUM_SIGNALS]; //!< status of each individual calibrated signal (i.e. roll, pitch, ...)
    dwCalibrationSignal signalType[SELF_CALIBRATION_NODE_MAX_NUM_SIGNALS];   //!< type of the signal status of which is populated in signalStatus
    bool signalStatusChanged[SELF_CALIBRATION_NODE_MAX_NUM_SIGNALS];         //!< set to true for one cycle when `signalStatus` changed
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

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_SELFCALIBRATIONTYPES_HPP_

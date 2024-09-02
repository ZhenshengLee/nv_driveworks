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
// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <dw/calibration/engine/common/VehicleParamsTypes.h>
#include <dw/core/health/HealthSignals.h>
#include <dw/rig/Vehicle.h>
#include <dw/rig/Rig.h>
#include <dw/core/base/Types.h>
#include <dwshared/dwfoundation/dw/core/container/BaseString.hpp>

namespace dw
{
namespace framework
{

// TODO(ahempel): These values need to be harmonized with DW_MAX_INSTANCE_CAMERA et al. in SehDwInternal.h to ensure correct SEH error reporting when more sensors are activated
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

static constexpr uint8_t SELF_CALIBRATION_NODE_MAX_CAMERAS_FOR_SUSPENSION_STATE{4U};
static constexpr uint8_t SELF_CALIBRATION_NUM_SUSPENSION_LEVEL_SIGNALS{4U};

// For every estimation, a property will be added to the rig file by the dwRigNode
// to indicating results of previous calibration:
//
// "properties": {
//     "self-calibration": "accepted"
// },

static constexpr char8_t CALIBRATION_STATE_TAG[]{"self-calibration"}; // TODO(hlanker): can we rename this to more generic "calibration"?
static constexpr char8_t CALIBRATION_ACCEPTED_STATE_STRING[]{"accepted"};
static constexpr char8_t CALIBRATION_NOT_ACCEPTED_STATE_STRING[]{"not-accepted"};
static constexpr char8_t CALIBRATION_FAILED_STATE_STRING[]{"failed"};
static constexpr char8_t CALIBRATION_INVALID_STATE_STRING[]{"invalid"};

static constexpr char8_t CALIBRATION_TYPE_TAG[]{"calibration-type"};
static constexpr char8_t CALIBRATION_TYPE_SELF_CALIBRATION_STRING[]{"self-calibration"};
static constexpr char8_t CALIBRATION_TYPE_SERVICE_CALIBRATION_STRING[]{"service-calibration"};
static constexpr char8_t CALIBRATION_TYPE_EOL_CALIBRATION_STRING[]{"eol-calibration"};

static constexpr char8_t SERVICE_CALIBRATION_DONE_TAG[]{"service-calibration-done"};
static constexpr char8_t EOL_CALIBRATION_DONE_TAG[]{"eol-calibration-done"};

// Additionally, the property "supports-eeprom-intrinsics" will be set to "true" if the intrinsics written to rig
// are from EEPROM; otherwise the property will be propagated from the input rig file.
static constexpr char8_t SUPPORTS_EEPROM_INTRINSICS_TAG[]{"supports-eeprom-intrinsics"};
static constexpr char8_t TRUE_STRING[]{"true"};

using CalibrationPropertyString       = dw::core::FixedString<512>;
using CalibrationIntrinsicsString     = dw::core::FixedString<64>;
using CalibrationExtrinsicProfileName = dw::core::FixedString<DW_MAX_EXTRINSIC_PROFILE_NAME_SIZE>;

struct dwCalibratedExtrinsics
{
    uint32_t sensorRigIndex;                              //!< sensor rig index of the calibrated sensor
    CalibrationExtrinsicProfileName extrinsicProfileName; //!< name of the currently selected profile
    dwCalibrationSensorPositionState positionState;       //!< current position status of the sensor
    dwTransformation3f currentSensor2Rig;                 //!< currently estimated sensor2rig
    dwCalibrationStatus status;                           //!< overall status of the whole routine, i.e. representing senor2Rig pose
    dwCalibrationProperties properties;                   //!< description of where the value is coming from
    dwCalibrationManeuverArray maneuvers;                 //!< maneuvers left to achieve full calibration
    dwTime_t timestamp;                                   //!< timestamp of the latest input data contributing to the result

    dwCalibrationStatus signalStatus[SELF_CALIBRATION_NODE_MAX_NUM_SIGNALS]; //!< status of each individual calibrated signal (i.e. roll, pitch, ...)
    dwCalibrationSignal signalType[SELF_CALIBRATION_NODE_MAX_NUM_SIGNALS];   //!< type of the signal status of which is populated in signalStatus
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

/**
 * @struct dwCameraTwoViewTransformation
 * @brief a struct defining the result of two view estimation
 *
 * dwCameraTwoViewTransformation struct allow to retrieve the results of two view estimation performed inside camera self calibration node.
 *
 * It contains
 * - currentTimestamp and previousTimestamp: the timestamps of the two analyzed frames.
 * - rigCurrentToPrevious: the reference motion in rig reference frame.
 * - cameraCurrentToPrevious: the estimated motion in camera coordinate reference frame.
 * - rigCurrentToPreviousValid and cameraCurrentToPreviousValid: two boolean flags indicating the validity of the previous two fields.
 * - groundPlaneInCameraCurrent: the estimated ground plane in camera current reference frame.
 * - groundPlaneInCameraCurrentValid: boolean flags indicating the validity of the previous field.
 *
 * Naming convention:
 * - current frame is the most recent frame used in two view estimation
 * - previous frame is a the reference frame, less recent, used in two view estimation
 * - currentTimestamp is the timestamp of current frame
 * - previousTimestamp is the timestamp of previous frame
 *
 * Note: frames are not necessarily consecutive: do not expect to have constant difference between timestamps.
 * The two view estimation process has several different policies to choose frame pairs
 *
 * Assumptions:
 * - if both currentTimestamp and previousTimestamp are valid (!= DW_TIME_INVALID): currentTimestamp > previousTimestamp
 * - if rigCurrentToPreviousValid is false, rigCurrentToPrevious should be ignored by consumer
 * - if cameraCurrentToPreviousValid is false, cameraCurrentToPrevious should be ignored by consumer
 * - if groundPlaneInCameraCurrentValid is false, groundPlaneInCameraCurrent should be ignored by consumer
 *
 * A full valid struct means that two view estimation and plane estimation have been successfully performed
 *
 * A partial valid struct means that some early exit condition has been met in the processing or that the estimation failed.
 *
 * For example:
 * - valid currentTimestamp and invalid previousTimestamp: means that the two view estimation could not find a previous frame to perform two view estimation.
 * - valid currentTimestamp and previousTimestamp with an invalid rigCurrentToPreviousValid: means that it has been impossible to retrieve the rig motion between the two timestamps.
 * - valid currentTimestamp, previousTimestamp and rigCurrentToPreviousValid with invalid cameraCurrentToPreviousValid: means that some condition has prevented to perform two view estimation, or that two view solution has not been accepted.
 *
 * TODO(sceriani) Jira AVC-3378 move this struct to protobuf
 */
typedef struct dwCameraTwoViewTransformation
{
    uint32_t sensorRigIndex; //!< index of the camera in the rig that produced the two view  estimation

    dwTime_t currentTimestamp;  //!< timestamp of the current frame
    dwTime_t previousTimestamp; //!< timestamp of the previous frame

    bool rigCurrentToPreviousValid;          //!< validity flag for rigCurrentToPrevious
    dwTransformation3f rigCurrentToPrevious; //!< reference motion between rig reference system at previousTimestamp and currentTimestamp

    bool cameraCurrentToPreviousValid;          //!< validity flag for cameraCurrentToPrevious
    dwTransformation3f cameraCurrentToPrevious; //!< observed motion between camera reference system at previousTimestamp and currentTimestamp

    bool groundPlaneInCameraCurrentValid;  //!< validity flag for groundPlaneInCameraCurrent
    dwVector4f groundPlaneInCameraCurrent; //!< homogeneous ground plane coefficients. The first three coefficients (x,y,z) are plane normal, i.e., a unit vector, the last (w) is the plane distance (meters). Plane is expressed in cameraCurrent reference system.
} dwCameraTwoViewTransformation;

/**
 * @struct dwSuspensionStateProperties
 * @brief a struct containing the current values of suspension state estimation
 *
 * It contains
 * - currentRigToReferenceRig: a transformation that allows to transform geometrical entities
 *   (e.g., points) expressed in the current rig to a fixed reference rig
 *
*/
typedef struct dwSuspensionStateProperties
{
    dwTransformation3f currentRigToReferenceRig; //!< transformation between current rig (where egomotion produces estimations) and reference rig (body aligned rig)
} dwSuspensionStateProperties;

/**
 * @struct dwCalibratedSuspensionStateProperties
 * @brief a struct containing the current status of suspension state estimation
 *
 * It contains
 * - suspensionStateProperties: the current estimated suspension state, see dwSuspensionStateProperties for further details
 * - timestamp: timestamp of the latest input data contributing to the result
 * - status: the current calibration status, see dwCalibrationStatus for further details
 *
*/
struct dwCalibratedSuspensionStateProperties
{
    dwSuspensionStateProperties suspensionStateProperties; //!< current estimated suspension state
    dwTime_t timestamp;                                    //!< timestamp of the latest input data contributing to the result
    dwCalibrationStatus status;                            //!< the current calibration status
};

/// empty struct, placeholder for suspension state diagnostics.
typedef struct dwSelfCalibrationSuspensionStateDiagnostics
{
    /// TODO(sceriani): AVC-2686 Extend this stub to contain actual diagnostics
} dwSelfCalibrationSuspensionStateDiagnostics;

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_SELFCALIBRATIONTYPES_HPP_

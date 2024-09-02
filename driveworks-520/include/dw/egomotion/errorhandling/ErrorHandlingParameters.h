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
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Copyright (c) 2022-2024 Mercedes-Benz AG. All rights reserved.
//
// Mercedes-Benz AG as copyright owner and NVIDIA Corporation as licensor retain
// all intellectual property and proprietary rights in and to this software
// and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement is strictly prohibited.
//
// This code contains Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE ARE MADE.
//
// No responsibility is assumed for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights. No third party distribution is allowed unless
// expressly authorized.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// Products are not authorized for use as critical
// components in life support devices or systems without express written approval.
//
///////////////////////////////////////////////////////////////////////////////////////

// WARNING!!!
// Please don't use any type definition in this file.
// All of data types in this file are going to be modified and will not
// follow Nvidia deprecation policy.

#ifndef DW_EGOMOTION_ERRORHANDLING_ERRORHANDLINGPARAMETERS_H_
#define DW_EGOMOTION_ERRORHANDLING_ERRORHANDLINGPARAMETERS_H_

#include <dw/core/logger/Logger.h>

#define DW_EGOMOTION_ERROR_ID_COUNT (559)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Limits for uint8_t signal ranges.
 */
typedef struct dwEgomotionSignalLimitsUInt8
{
    uint8_t lower; //!< Lower limit.
    uint8_t upper; //!< Upper limit.
} dwEgomotionSignalLimitsUInt8;

/**
 * \brief Limits for int16_t signal ranges.
 */
typedef struct dwEgomotionSignalLimitsInt16
{
    int16_t lower; //!< Lower limit.
    int16_t upper; //!< Upper limit.
} dwEgomotionSignalLimitsInt16;

/**
 * \brief Limits for float32_t signal ranges.
 */
typedef struct dwEgomotionSignalLimitsFloat32
{
    float32_t lower; //!< Lower limit.
    float32_t upper; //!< Upper limit.
} dwEgomotionSignalLimitsFloat32;

/**
 * \brief Limits for float64_t signal ranges.
 */
typedef struct dwEgomotionSignalLimitsFloat64
{
    float64_t lower; //!< Lower limit.
    float64_t upper; //!< Upper limit.
} dwEgomotionSignalLimitsFloat64;

/**
 * \brief Limits for dwTime_t signal ranges.
 */
typedef struct dwEgomotionSignalLimitsDwTime
{
    dwTime_t lower; //!< Lower limit.
    dwTime_t upper; //!< Upper limit.
} dwEgomotionSignalLimitsDwTime;

/**
 * \brief Parameters for error handling IMU L2 Monitor
 */
typedef struct dwEgomotionIMUMonitorParameters
{
    // Validity conditions
    /// Range for the wheel acceleration hysteresis in m/s².
    dwEgomotionSignalLimitsFloat32 wheelAccelHyst;

    /// Range for the wheel acceleration difference hysteresis in m/s².
    dwEgomotionSignalLimitsFloat32 wheelAccelDiffHyst;

    /// Range for the normalised wheel velocity hysteresis.
    dwEgomotionSignalLimitsFloat32 normalisedWheelVelocityDiffHyst;

    /// Range for the yaw acceleration hysteresis in rad/s².
    dwEgomotionSignalLimitsFloat32 yawAccelerationHyst;

    /// Range for the yaw rate hysteresis in rad/s.
    dwEgomotionSignalLimitsFloat32 yawrateHyst;

    /// Range for the lateral acceleration hysteresis in m/s².
    dwEgomotionSignalLimitsFloat32 lateralAcceleration;

    /// Range for the front wheel angle hysteresis in rad.
    dwEgomotionSignalLimitsFloat32 frontWheelAngleHyst;

    /// Range for the reference velocity hysteresis in m/s.
    dwEgomotionSignalLimitsFloat32 referenceVelocity;

    // Activation thresholds
    // Minimum longitudinal velocity for evaluation of acceleration based yaw rate.
    float32_t minLongitudinalVelocityForYawRateFromAcc;

    /// Range of the reference yaw rate (IMU) hystersis in rad/s.
    dwEgomotionSignalLimitsFloat32 yawRateReferenceHyst;

    /// Range for the yaw rate from acceleration hysteresis in rad/s.
    dwEgomotionSignalLimitsFloat32 yawRateFromAccelHyst;

    /// Range for the yaw rate from the one track model hysteresis in rad/s.
    dwEgomotionSignalLimitsFloat32 yawRateFromModelHyst;

    /// Range for the yaw rate from the wheel speeds hysteresis in rad/s.
    dwEgomotionSignalLimitsFloat32 yawRateFromWheelsHyst;

    /// Maximum absolute steering angle for validity of front wheel based yaw rate in rad.
    float32_t maxAbsSteeringForYawRateFromFrontWheels;

    /// Maximum absolute median yawrate for validity of monitor in rad/s.
    float32_t maxAbsYawrateForValidityOfMonitor;
} dwEgomotionIMUMonitorParameters;

/**
 * \brief Parameters for error handling related logic.
 */
typedef struct dwEgomotionErrorHandlingParameters
{
    /// Whether errors should be notified to SEH.
    /// Error monitoring always runs in the background but only notifies the
    /// system error handler if this parameter is set to true. Set the parameter
    /// to false if the vehicle platform does not comply with egomotion technical
    /// safety concept or if it is not yet stable enough.
    bool notifySEH;

    /// Whether errors shall lead egomotion to degrade or go unavailable.
    /// Error monitoring always runs in the background but is only acted upon
    /// if this parameter is set to true. Set the parameter to false if the
    /// vehicle platform does not comply with egomotion technical safety concept
    /// or if it is not yet stable enough. This might result in egomotion providing
    /// erroneous estimation.
    bool enableDegradations;

    /// Verbosity level of logging.
    dwLoggerVerbosity logLevel;

    /// Cycles between repeated logging of egomotion error states.
    /// A cycle corresponds to an execution pass of the error monitoring logic, running on each
    /// addition of input data (VIO, IMU). A value of zero will result in no logging. A value of one
    /// will log errors every time they are detected. A higher value will result in subsampling of
    /// log messages.
    uint32_t logCycles;

    // Enable initialisation checks
    bool initializationChecksActive;

    // Enable shutdown detection
    bool shutdownDetectionActive;

    /// Informs if the vehicle provides rear wheel angle signals
    bool sigPresenceRearWheelAngle;
    bool sigPresenceRearWheelAngleQuality;
    bool sigPresenceRearWheelAngleTimestamp;
    bool sigPresenceRearWheelAngleTimestampQuality;

    /// Informs if the vehicle provides suspension level signals
    bool sigPresenceSuspensionLevel;
    bool sigPresenceSuspensionLevelQuality;
    bool sigPresenceSuspensionLevelTimestamp;
    bool sigPresenceSuspensionLevelTimestampQuality;
    bool sigPresenceSuspensionLevelCalibrationState;

    /// Informs if the vehicle provides redundant (High) signals
    bool sigPresenceWheelSpeedRedundant;
    bool sigPresenceWheelSpeedQualityRedundant;
    bool sigPresenceWheelTicksRedundant;
    bool sigPresenceWheelTicksDirectionRedundant;
    bool sigPresenceWheelTicksTimestampRedundant;
    bool sigPresenceWheelTicksTimestampQualityRedundant;
    bool sigPresenceFrontSteeringAngleHigh;
    bool sigPresenceFrontSteeringAngleControlQualityHigh;
    bool sigPresenceFrontSteeringTimestampHigh;

    // Informs if the vehilc provides other optional signals
    bool sigPresenceIMUTimestampQuality;
    bool sigPresenceIMUAccelerometerOffsetZ;
    bool sigPresenceBrakeTorqueWheelsQuality;
    bool sigPresenceIMUStatus;
    bool sigPresenceIMUSequenceCounter;
    bool sigPresenceIMUTurnrateOffsetQualityStatus;

    /// Mask of disabled errors. @see getErrorIdDisableMask for how to generate this mask.
    bool disabledErrorIdMask[DW_EGOMOTION_ERROR_ID_COUNT];

    // input-monitor-generation:start id:parameters-declarations # GENERATED CODE - DO NOT CHANGE MANUALLY
    /// Wheel Speed valid range (rad/s)
    dwEgomotionSignalLimitsFloat32 vioWheelSpeedRange_radps;

    /// Front Steering Angle valid range (rad)
    dwEgomotionSignalLimitsFloat32 vioFrontSteeringAngleRange_rad;

    /// Rear Wheel Angle valid range (rad)
    dwEgomotionSignalLimitsFloat32 vioRearWheelAngleRange_rad;

    /// Front Steering Timestamp valid range (us)
    dwEgomotionSignalLimitsDwTime vioFrontSteeringTimestampRange_us;

    /// Front Steering Angle Offset valid range (rad)
    dwEgomotionSignalLimitsFloat32 vioFrontSteeringAngleOffsetRange_rad;

    /// Rear Wheel Angle Timestamp valid range (us)
    dwEgomotionSignalLimitsDwTime vioRearWheelAngleTimestampRange_us;

    /// Wheel Ticks valid range (unitless)
    dwEgomotionSignalLimitsInt16 vioWheelTicksRange;

    /// Wheel Ticks Timestamp valid range (us)
    dwEgomotionSignalLimitsDwTime vioWheelTicksTimestampRange_us;

    /// Wheel Torque valid range (N*m)
    dwEgomotionSignalLimitsFloat32 vioWheelTorqueRange_Nxm;

    /// Speed Min valid range (m/s)
    dwEgomotionSignalLimitsFloat32 vioSpeedMinRange_mps;

    /// Speed Max valid range (m/s)
    dwEgomotionSignalLimitsFloat32 vioSpeedMaxRange_mps;

    /// Brake Torque Wheels valid range (N*m)
    dwEgomotionSignalLimitsFloat32 vioBrakeTorqueWheelsRange_Nxm;

    /// Suspension Level valid range (m)
    dwEgomotionSignalLimitsFloat32 vioSuspensionLevelRange_m;

    /// Suspension Level Timestamp valid range (us)
    dwEgomotionSignalLimitsDwTime vioSuspensionLevelTimestampRange_us;

    /// Host Timestamp valid range (us)
    dwEgomotionSignalLimitsDwTime imuHostTimestampRange_us;

    /// Sensor Timestamp valid range (us)
    dwEgomotionSignalLimitsDwTime imuSensorTimestampRange_us;

    /// Sequence Counter valid range (unitless)
    dwEgomotionSignalLimitsUInt8 imuSequenceCounterRange;

    /// Turnrate valid range (rad/s)
    dwEgomotionSignalLimitsFloat64 imuTurnrateRange_radps;

    /// Turnrate Offset valid range (rad/s)
    dwEgomotionSignalLimitsFloat64 imuTurnrateOffsetRange_radps;

    /// Imu Turnrate Offset Quality valid range (unitless)
    dwEgomotionSignalLimitsUInt8 imuImuTurnrateOffsetQualityRange;

    /// Turnrate Accel valid range (rad/s²)
    dwEgomotionSignalLimitsFloat64 imuTurnrateAccelRange_radps2;

    /// Acceleration valid range (m/s²)
    dwEgomotionSignalLimitsFloat64 imuAccelerationRange_mps2;

    /// Acceleration Offset valid range (m/s²)
    dwEgomotionSignalLimitsFloat64 imuAccelerationOffsetRange_mps2;

    /// Temperature valid range (Celsius)
    dwEgomotionSignalLimitsFloat32 imuTemperatureRange_c;

    /// Imu Turnrate Offset Quality Status valid range (unitless)
    dwEgomotionSignalLimitsUInt8 imuImuTurnrateOffsetQualityStatusRange;
    // input-monitor-generation:end id:parameters-declarations # END OF GENERATED CODE

    /// Front Steering Timestamp cycle time range (us)
    dwEgomotionSignalLimitsDwTime frontSteeringTimestampCycleTimeRange_us;

    /// Wheel Tick Timestamp cycle time range (us)
    dwEgomotionSignalLimitsDwTime wheelTickTimestampCycleTimeRange_us;

    /// Rear Wheel Angle Timestamp cycle time range (us)
    dwEgomotionSignalLimitsDwTime rearWheelAngleTimestampCycleTimeRange_us;

    /// Suspension Level Timestamp cycle time range (us)
    dwEgomotionSignalLimitsDwTime suspensionLevelTimestampCycleTimeRange_us;

    /// IMU Timestamp cycle time range (us)
    dwEgomotionSignalLimitsDwTime imuTimestampCycleTimeRange_us;

    /// Front Steering Timestamp time offset limit (us)
    dwEgomotionSignalLimitsDwTime frontSteeringTimestampTimeOffsetLimit_us;

    /// Wheel Tick Timestamp time offset limit (us)
    dwEgomotionSignalLimitsDwTime wheelTickTimestampTimeOffsetLimit_us;

    /// Rear Wheel Angle Timestamp time offset limit (us)
    dwEgomotionSignalLimitsDwTime rearWheelAngleTimestampTimeOffsetLimit_us;

    /// Suspension Level Timestamp time offset limit (us)
    dwEgomotionSignalLimitsDwTime suspensionLevelTimestampTimeOffsetLimit_us;

    /// IMU Timestamp time offset limit (us)
    dwEgomotionSignalLimitsDwTime imuTimestampTimeOffsetLimit_us;

    /// Time interval until absence of new VIO input results in timeout declared (us)
    dwTime_t vioTimeout_us;

    /// Time interval until absence of new IMU input results in timeout declared (us)
    dwTime_t imuTimeout_us;

    /// Nominal wheel radius in meters
    float32_t wheelRadius_m;

    /// Valid range for wheel speeds in rad/s.
    dwEgomotionSignalLimitsFloat32 wheelSpeedRange_radps;

    /// Maximum allowable deviation between fastest and slowest linear wheel speed at ground (m/s)
    float32_t maxAllowableGroundSpeedDeviation_mps;

    /// Maximum allowable deviation between fastest and slowest linear wheel speed at ground (% of mean speed)
    float32_t maxAllowableGroundSpeedDeviation_percent;

    /// Minimum mean body speed at which ground speed monitor activates (m/s)
    float32_t minGroundSpeedMonitorSpeed_mps;

    /// IMU Monitor parameters
    dwEgomotionIMUMonitorParameters imuMonitorParameters;
} dwEgomotionErrorHandlingParameters;

/**
 * @brief Convert a list of ErrorId names to a mask representing their status
 *
 * @param disabledErrorIdMask Output bools indicating for each ErrorId whether the ErrorId should be enabled. Set by this function.
 * @param disabledErrorIds Array of string representations of ErrorIds to disable.
 * @return dwStatus Will indicate DW_INVALID_ARGUMENT if unknown error id is specified in disabledErrorIds.
 * @note Does not clear the output array before writing. You can consider this function as OR'ing the disabling of the specified IDs.
 */
dwStatus dwEgomotion2_getErrorIdDisableMask(bool disabledErrorIdMask[DW_EGOMOTION_ERROR_ID_COUNT], char8_t const* const disabledErrorIds[DW_EGOMOTION_ERROR_ID_COUNT]);

#ifdef __cplusplus
}
#endif
#endif // DW_EGOMOTION_ERRORHANDLING_ERRORHANDLINGPARAMETERS_H_

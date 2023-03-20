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
// SPDX-FileCopyrightText: Copyright (c) 2016-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file
 * <b>NVIDIA DriveWorks API: Calibration</b>
 *
 * @b Description: Contains fundamental types associated with Calibration
 */

/**
 * @defgroup calibration_types_group Calibration Types
 * @ingroup calibration_group
 *
 * @brief Fundamental types associated with Calibration
 *
 * @{
 *
 */

#ifndef DW_CALIBRATION_ENGINE_CALIBRATIONTYPES_H_
#define DW_CALIBRATION_ENGINE_CALIBRATIONTYPES_H_

#include <dw/core/base/Types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief This defines the number of anticipated concurrent calibrations
 */
#define DW_CALIBRATION_MAXROUTINES 64

/**
 * @brief Handles to calibration module objects.
 */
typedef struct dwCalibrationRoutineObject* dwCalibrationRoutineHandle_t;
typedef struct dwCalibrationEngineObject* dwCalibrationEngineHandle_t;

/**
 * @brief Defines the current state of an individual calibration
 */
typedef enum dwCalibrationState {
    /// The routine hasn't accepted an estimate
    DW_CALIBRATION_STATE_NOT_ACCEPTED = 0,

    /// The routine has accepted an estimate and calibration continues
    DW_CALIBRATION_STATE_ACCEPTED = 1,

    /// The routine has failed calibration
    DW_CALIBRATION_STATE_FAILED = 2,

    /// The calibration state is invalid (e.g. when a door with a sensor is open or a mirror is moving)
    DW_CALIBRATION_STATE_INVALID = 3,
} dwCalibrationState;

/**
 * @brief Converts a calibration state enum to a human-readable string representation.
 * @param str the returned string
 * @param state the state to translate
 * @return DW_INVALID_ARGUMENT - if the arguments in the parameters are invalid <br>
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwCalibrationState_toString(const char** str, dwCalibrationState state);

/**
 * @brief Defines the current status of an individual calibration
 */
typedef struct dwCalibrationStatus
{
    /// Flag indicating whether a calibration routine is running
    bool started;

    /// The current state of a calibration routine
    dwCalibrationState state;

    /// The current calibration percentage complete status. Valid percentages are in the range [0,1]
    float32_t percentageComplete;
} dwCalibrationStatus;

/**
 * @brief Defines a callback function that is called when calibration routine
 * has changed its internal status
 */
typedef void (*dwCalibrationStatusChanged)(dwCalibrationRoutineHandle_t routine,
                                           dwCalibrationStatus status,
                                           void* userData);

/**
 * @brief Defines signal types supported by a calibration routine.
 */
typedef enum dwCalibrationSignal {
    /// Pose-related signals and pose components
    DW_CALIBRATION_SIGNAL_POSE_SENSOR_TO_RIG    = 1 << 1, /// estimating a "sensor to rig" pose
    DW_CALIBRATION_SIGNAL_POSE_SENSOR_TO_SENSOR = 1 << 2, /// estimating "sensor to sensor" poses
    DW_CALIBRATION_SIGNAL_POSE_ROLL             = 1 << 3, /// estimating the roll component of an orientation
    DW_CALIBRATION_SIGNAL_POSE_PITCH            = 1 << 4, /// estimating the pitch component of an orientation
    DW_CALIBRATION_SIGNAL_POSE_YAW              = 1 << 5, /// estimating the yaw component of an orientation
    DW_CALIBRATION_SIGNAL_POSE_X                = 1 << 6, /// estimating the x component of a translation / direction
    DW_CALIBRATION_SIGNAL_POSE_Y                = 1 << 7, /// estimating the y component of a translation / direction
    DW_CALIBRATION_SIGNAL_POSE_Z                = 1 << 8, /// estimating the z component of a translation / direction

    /// Vehicle-related signals
    DW_CALIBRATION_SIGNAL_VEHICLE_SPEED_FACTOR          = 1 << 9,  /// estimating speed correction factor for CAN odometry
    DW_CALIBRATION_SIGNAL_VEHICLE_WHEEL_RADII           = 1 << 10, /// estimating wheel radii
    DW_CALIBRATION_SIGNAL_VEHICLE_FRONT_STEERING_OFFSET = 1 << 11, /// estimating front steering offset

    /// Camera intrinsics-related signals
    DW_CALIBRATION_SIGNAL_INTRINSICS_SCALE = 1 << 12 /// scale factor applied to angle in f-theta camera model
} dwCalibrationSignal;

/**
 * @brief Fast-acceptance options to configure calibration routines with.
 *
 * If previously accepted estimates are available, fast-acceptance is a method to reduce re-calibration times
 * in case the previous estimates can be validated with latest measurements. This option allows to configure
 * the fast-acceptance behaviour of calibration routines supporting fast-acceptance
 */
typedef enum dwCalibrationFastAcceptanceOption {
    /// Let the calibration engine decide if fast-acceptance should be used
    DW_CALIBRATION_FAST_ACCEPTANCE_DEFAULT = 0,

    /// Unconditionally enable fast-acceptance (previously accepted estimates need to be available)
    DW_CALIBRATION_FAST_ACCEPTANCE_ENABLED = 1,

    /// Unconditionally disable fast-acceptance (previously accepted estimates will not be used)
    DW_CALIBRATION_FAST_ACCEPTANCE_DISABLED = 2,
} dwCalibrationFastAcceptanceOption;

#ifdef __cplusplus
} // extern C
#endif
/** @} */

#endif // DW_CALIBRATION_ENGINE_CALIBRATIONTYPES_H_

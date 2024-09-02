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
// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_SENSORS_COMMON_SENSORSVEHICLESTATETYPES_H_
#define DW_SENSORS_COMMON_SENSORSVEHICLESTATETYPES_H_

#include <dw/core/base/Types.h>
#include <dw/core/signal/SignalStatus.h>

/**
 * @brief Driving Direction options.
 */
typedef enum dwSensorDrivingDirection {
    /** Driving direction is unknown or no moving */
    DW_SENSOR_DRIVING_DIRECTION_UNKNOWN = 0,
    /** Driving direction is fowarding */
    DW_SENSOR_DRIVING_DIRECTION_FORWARD = 1,
    /** Driving direction is backwarding */
    DW_SENSOR_DRIVING_DIRECTION_BACKWARD = 2,
    /** Enum count */
    DW_SENSOR_DRIVING_DIRECTION_COUNT = 3
} dwSensorDrivingDirection;

/**
 * @brief Current vehicle drive position: Parking, Rear, Neutral, Driving (PRND).
 */
typedef enum dwSensorDrivePositionStatus {
    /** D */
    DW_SENSOR_DRIVE_POSITION_STATUS_D = 0,

    /** N */
    DW_SENSOR_DRIVE_POSITION_STATUS_N = 1,

    /** R */
    DW_SENSOR_DRIVE_POSITION_STATUS_R = 2,

    /** P */
    DW_SENSOR_DRIVE_POSITION_STATUS_P = 3,

    /** Unsupported mapping */
    DW_SENSOR_DRIVE_POSITION_STATUS_UNSUPPORTED_MAPPING = 4,

    DW_SENSOR_DRIVE_POSITION_STATUS_FORCE32 = 0x7fffffff,
} dwSensorDrivePositionStatus;

/**
 * @brief Door lock (latch) state.
 */
typedef enum dwSensorDoorLockState {
    /** undefined value */
    DW_SENSOR_DOOR_LOCK_STATE_UNKNOWN = 0,

    /** closed fully */
    DW_SENSOR_DOOR_LOCK_STATE_SECURE_CLOSED = 1,

    /** closed but not secure (two stage lock) */
    DW_SENSOR_DOOR_LOCK_STATE_UNKNOWN_CLOSED = 2,

    /** open */
    DW_SENSOR_DOOR_LOCK_STATE_OPEN = 3,

    /** Unsupported mapping */
    DW_SENSOR_DOOR_LOCK_STATE_UNSUPPORTED_MAPPING = 4,

    DW_SENSOR_DOOR_LOCK_STATE_FORCE32 = 0x7fffffff,
} dwSensorDoorLockState;

/**
 * @brief Whether a trailer hitch is extended, installed but not extended, or not present.
 */
typedef enum dwSensorTrailerHitchStatus {
    /** unknown status */
    DW_SENSOR_TRAILER_HITCH_STATUS_HITCH_UNKNOWN = 0,

    /** error */
    DW_SENSOR_TRAILER_HITCH_STATUS_HITCH_ERROR = 1,

    /** not fitted / installed */
    DW_SENSOR_TRAILER_HITCH_STATUS_HITCH_NOT_FITTED = 2,

    /** Retracted position */
    DW_SENSOR_TRAILER_HITCH_STATUS_HITCH_RETRACTED = 3,

    /** Working position with nothing plugged in */
    DW_SENSOR_TRAILER_HITCH_STATUS_HITCH_EXTENDED = 4,

    /** Working position with device plugged in */
    DW_SENSOR_TRAILER_HITCH_STATUS_HITCH_EXTENDED_ATTACHED = 5,

    /** Unsupported mapping */
    DW_SENSOR_TRAILER_HITCH_STATUS_HITCH_UNSUPPORTED_MAPPING = 6,
    DW_SENSOR_TRAILER_HITCH_STATUS_FORCE32                   = 0x7fffffff,
} dwSensorTrailerHitchStatus;

/**
 * @brief Whether a trailer is connected to the vehicle or not.
 */
typedef enum dwSensorTrailerConnected {
    /** Unknown status */
    DW_SENSOR_TRAILER_CONNECTED_TRAILER_CONNECTED_UNKNOWN = 0,
    /** No trailer connected */
    DW_SENSOR_TRAILER_CONNECTED_TRAILER_CONNECTED_NO = 1,
    /** Trailer connected */
    DW_SENSOR_TRAILER_CONNECTED_TRAILER_CONNECTED_YES = 2,
    /** Error */
    DW_SENSOR_TRAILER_CONNECTED_TRAILER_CONNECTED_ERROR = 3,
    /** Unsupported mapping */
    DW_SENSOR_TRAILER_CONNECTED_TRAILER_CONNECTED_UNSUPPORTED_MAPPING = 4,
    DW_SENSOR_TRAILER_CONNECTED_FORCE32                               = 0x7fffffff,
} dwSensorTrailerConnected;

/**
 * @brief Vehicle State inputs.
 */
typedef struct dwSensorVehicleState
{
    /** Structure for storing the signal validity of speed, temperature and direction and other Rest of Vehicle signals */
    struct
    {
        /** the signal validity of speed*/
        dwSignalValidity speed;
        /** the signal validity of temperature*/
        dwSignalValidity temperature;
        /** the signal validity of direction*/
        dwSignalValidity direction;
        /** the signal validity of the drive gear position status*/
        dwSignalValidity drivePositionStatus;
        /** the signal validity of the rear trunk state*/
        dwSignalValidity rearTrunkState;
        /** the signal validity of the trailer hitch status*/
        dwSignalValidity trailerHitchStatus;
        /** the signal validity of the trailer connected status*/
        dwSignalValidity trailerConnected;
        /** the signal validity of the roll angle*/
        dwSignalValidity longCtrlActiveFunction;
        /** the signal validity of the pitch angle*/
        dwSignalValidity latCtrlModeStatus;
        /** the signal validity of the suspension level*/
        dwSignalValidity suspensionLevel[4];
        /** the signal validity of the door status*/
        dwSignalValidity doorStates[4];
    } validityInfo;

    /** Timestamp for the current message. Indicates when it's first received */
    dwTime_t timestamp_us;

    /** vehicle velocity (m/s) */
    float32_t speed;

    /** ambient temperature (C) */
    float32_t temperature;

    /** driving direction */
    dwSensorDrivingDirection direction;

    /** Current vehicle drive position: Parking, Rear, Neutral, Driving (PRND). */
    dwSensorDrivePositionStatus drivePositionStatus;

    /** Trunk lock state. */
    dwSensorDoorLockState rearTrunkState;

    /** Status of Trailer Hitch. */
    dwSensorTrailerHitchStatus trailerHitchStatus;

    /** Status whether a trailer is connected */
    dwSensorTrailerConnected trailerConnected;

    /** Height of the suspension. Order is FL, FR, RL, RR. */
    float32_t suspensionLevel[4];

    /** Status whether the doors are open or closed in the order FL, FR, RL, RR. */
    dwSensorDoorLockState doorStates[4];
} dwSensorVehicleState;

/**
 * @brief Motion state derived from wheeltick, to assess whether ego is stationary or moving
 */
typedef enum dwSensorMotionState {
    /** Motion state is unknown */
    DW_SENSOR_MOTION_STATE_UNKNOWN = 0,
    /** Motion state is standstill */
    DW_SENSOR_MOTION_STATE_STANDSTILL = 1,
    /** Motion state is moving */
    DW_SENSOR_MOTION_STATE_MOVING = 2,
    /** Enum count */
    DW_SENSOR_MOTION_STATE_COUNT = 3
} dwSensorMotionState;

/**
 * @brief Egomotion State inputs.
 */
typedef struct dwSensorEgomotionState
{
    /** Structure for storing the signal validity of egomotion signals */
    struct
    {
        /** the signal validity of linear velocity*/
        dwSignalValidity linearVelocity[3];
        /** the signal validity of angular velocity*/
        dwSignalValidity angularVelocity[3];
        /** the signal validity of linear acceleration*/
        dwSignalValidity linearAcceleration[3];
        /** the signal validity of angular acceleration*/
        dwSignalValidity angularAcceleration[3];
        /** the signal validity of position*/
        dwSignalValidity position_localCoordinates[3];
        /** the signal validity of linear velocity in local coordinates*/
        dwSignalValidity linearVelocity_localCoordinates[3];
        /** the signal validity of roll angle*/
        dwSignalValidity rollAngle;
        /** the signal validity of pitch angle*/
        dwSignalValidity pitchAngle;
        /** the signal validity of yaw angle*/
        dwSignalValidity yawAngle;
        /** the signal validity of driving direction*/
        dwSignalValidity drivingDirection;
        /** the signal validity of motion state*/
        dwSignalValidity motionState;
    } validityInfo;

    /** Egomotion timestamp */
    dwTime_t timestamp_us;
    /** Whether the estimation provided is valid */
    bool isEstimationValid;
    /** Linear velocity in NDAS reference frame, so with respect to the projection to the ground of
    the center of the rear axle of the rigid body */
    float32_t linearVelocity[3];
    /** Angular velocity in NDAS reference frame */
    float32_t angularVelocity[3];
    /** Linear acceleration in NDAS reference frame */
    float32_t linearAcceleration[3];
    /** Angular acceleration in NDAS reference frame */
    float32_t angularAcceleration[3];
    /** Position in world coordinate frame, so with respect to the beginning of the drive. Note that
    yaw error accumulates overtime, and this position makes sense only in a relative 
    sense locally */
    float32_t position_localCoordinates[3];
    /** Linear velocity in world coordinate frame with local validity */
    float32_t linearVelocity_localCoordinates[3];
    /** Roll angle in the rigid body frame */
    float32_t rollAngle;
    /** Pitch angle in the rigid body frame */
    float32_t pitchAngle;
    /** Yaw angle in the world/local coordinate frame */
    float32_t yawAngle;
    /** Driving direction estimated with wheel ticks */
    dwSensorDrivingDirection drivingDirection;
    /** Egomotion motion state */
    dwSensorMotionState motionState;
} dwSensorEgomotionState;

#endif // DW_SENSORS_COMMON_SENSORSVEHICLESTATETYPES_H_
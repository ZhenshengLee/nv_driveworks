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
 * <b>NVIDIA DriveWorks API: IMU</b>
 *
 * @b Description: This file defines methods to access the IMU sensor.
 */

/**
 * @defgroup imu_group IMU Sensor
 * @ingroup sensors_group
 *
 * @brief Defines the IMU sensor.
 *
 * @{
 */

#ifndef DW_SENSORS_IMU_IMU_H_
#define DW_SENSORS_IMU_IMU_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>

#include <dw/sensors/Sensors.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Each flag shows if that value is valid in this IMU frame
typedef enum dwIMUFlags {

    DW_IMU_HEADING DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.heading instead") = 1U << 1, //!< Value of dwIMUFrame.heading is valid.

    DW_IMU_ROLL DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.orientation[0] instead")  = 1U << 2, //!< Value of dwIMUFrame.orientation[0] is valid.
    DW_IMU_PITCH DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.orientation[1] instead") = 1U << 3, //!< Value of dwIMUFrame.orientation[1] is valid.
    DW_IMU_YAW DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.orientation[2] instead")   = 1U << 4, //!< Value of dwIMUFrame.orientation[2] is valid.

    DW_IMU_QUATERNION_X DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.orientationQuaternion instead") = 1U << 5, //!< Value of dwIMUFrame.orientationQuaternion.x is valid.
    DW_IMU_QUATERNION_Y DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.orientationQuaternion instead") = 1U << 6, //!< Value of dwIMUFrame.orientationQuaternion.y is valid.
    DW_IMU_QUATERNION_Z DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.orientationQuaternion instead") = 1U << 7, //!< Value of dwIMUFrame.orientationQuaternion.z is valid.
    DW_IMU_QUATERNION_W DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.orientationQuaternion instead") = 1U << 8, //!< Value of dwIMUFrame.orientationQuaternion.w is valid.

    DW_IMU_ROLL_RATE DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.turnrate[0] instead")  = 1U << 9,  //!< Value of dwIMUFrame.turnrate[0] is valid.
    DW_IMU_PITCH_RATE DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.turnrate[1] instead") = 1U << 10, //!< Value of dwIMUFrame.turnrate[1] is valid.
    DW_IMU_YAW_RATE DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.turnrate[2] instead")   = 1U << 11, //!< Value of dwIMUFrame.turnrate[2] is valid.

    DW_IMU_ACCELERATION_X DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.acceleration[0] instead") = 1U << 12, //!< Value of dwIMUFrame.acceleration[0] is valid.
    DW_IMU_ACCELERATION_Y DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.acceleration[1] instead") = 1U << 13, //!< Value of dwIMUFrame.acceleration[1] is valid.
    DW_IMU_ACCELERATION_Z DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.acceleration[2] instead") = 1U << 14, //!< Value of dwIMUFrame.acceleration[2] is valid.

    DW_IMU_MAGNETOMETER_X DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.magnetometer[0] instead") = 1U << 15, //!< Value of dwIMUFrame.magnetometer[0] is valid.
    DW_IMU_MAGNETOMETER_Y DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.magnetometer[1] instead") = 1U << 16, //!< Value of dwIMUFrame.magnetometer[1] is valid.
    DW_IMU_MAGNETOMETER_Z DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.magnetometer[2] instead") = 1U << 17, //!< Value of dwIMUFrame.magnetometer[2] is valid.

    DW_IMU_ALIGNMENTSTATUS DW_DEPRECATED_ENUM("Use dwIMUFrame.validityInfo.alignmentStatus instead") = 1U << 18 //!< Value of dwIMUFrame.alignmentStatus is valid.

} dwIMUFlags
    DW_DEPRECATED("dwIMUFlags is deprecated and will be removed in the future, use dwIMUFrame.validityInfo and APIs from IMUGetterSetter.h instead");

/// Types of the heading degree
typedef enum dwIMUHeadingType {

    /// True heading
    DW_IMU_HEADING_TRUE DW_DEPRECATED_ENUM("dwIMUHeadingType will be removed") = 0, //!< 'dwIMUFrame.heading' points towards true north.

    /// Magnetic heading
    DW_IMU_HEADING_MAGNETIC DW_DEPRECATED_ENUM("dwIMUHeadingType will be removed") = 1, //!< 'dwIMUFrame.heading' points towards magnetic north.

    /// Guard
    DW_IMU_HEADING_FORCE32 = 0x7FFFFFFF
} dwIMUHeadingType
    DW_DEPRECATED("dwIMUHeadingType is deprecated and will be removed in the future");

/// High rate data output from GNSS-IMU device requires fusion of GNSS and IMU. The data
/// can only be accurate if the GNSS/IMU device has gone through an internal alignment
/// initialization, which allows the device to know the orientation of the IMU. This status
/// indicates the quality of IMU alignment.
typedef enum dwIMUAlignmentStatus {

    /// Unknown status means the device does not or has not yet provided this information.
    DW_IMU_ALIGNMENT_STATUS_UNKNOWN = 0,
    /// Invalid means the IMU alignment is not yet valid, and the output data is not accurate.
    DW_IMU_ALIGNMENT_STATUS_INVALID = 1,
    /// Coarse means the IMU is roughly aligned, so the data is useful, but not of the highest quality.
    DW_IMU_ALIGNMENT_STATUS_COARSE = 2,
    /// Fine means the IMU alignment is complete, and the device can output high quality data.
    DW_IMU_ALIGNMENT_STATUS_FINE    = 3,
    DW_IMU_ALIGNMENT_STATUS_FORCE32 = 0x7FFFFFFF
} dwIMUAlignmentStatus;

typedef enum dwIMUImuTempQuality {
    /// Signal initializing
    DW_IMU_IMU_TEMP_QUALITY_INIT = 0,
    /// Sensor uncalibrated
    DW_IMU_IMU_TEMP_QUALITY_UNCALIB = 1,
    /// Signal in specification
    DW_IMU_IMU_TEMP_QUALITY_OK = 2,
    /// Signal temporary failure
    DW_IMU_IMU_TEMP_QUALITY_TMP_FAIL = 3,
    /// Signal permanent failure
    DW_IMU_IMU_TEMP_QUALITY_PRMNT_FAIL = 4,
    /// Sensor not installed
    DW_IMU_IMU_TEMP_QUALITY_SENS_NOT_INST = 5,

    DW_IMU_IMU_TEMP_QUALITY_FORCE32 = 0x7FFFFFFF
} dwIMUImuTempQuality;

typedef enum dwIMUImuAccelerationQuality {
    /// Signal Unknown
    DW_IMU_IMU_ACCELERATION_QUALITY_UNKNOWN = 0,
    /// Signal initializing
    DW_IMU_IMU_ACCELERATION_QUALITY_INIT = 1,
    /// Sensor uncalibrated
    DW_IMU_IMU_ACCELERATION_QUALITY_UNCALIB = 2,
    /// Signal in specification
    DW_IMU_IMU_ACCELERATION_QUALITY_OK = 3,
    /// Signal temporary failure
    DW_IMU_IMU_ACCELERATION_QUALITY_TMP_FAIL = 4,
    /// Signal permanent failure
    DW_IMU_IMU_ACCELERATION_QUALITY_PRMNT_FAIL = 5,
    /// Sensor not installed
    DW_IMU_IMU_ACCELERATION_QUALITY_SENS_NOT_INST = 6,
    /// Sensor overloaded
    DW_IMU_IMU_ACCELERATION_QUALITY_OVERLOAD = 7,
    /// Sensor out of operating temperature
    DW_IMU_IMU_ACCELERATION_QUALITY_TEMPERATURE = 8,
    DW_IMU_IMU_ACCELERATION_QUALITY_FORCE32     = 0x7FFFFFFF
} dwIMUImuAccelerationQuality;

typedef enum dwIMUImuTurnrateQuality {
    /// Signal Unknown
    DW_IMU_IMU_TURNRATE_QUALITY_UNKNOWN = 0,
    /// Signal initializing
    DW_IMU_IMU_TURNRATE_QUALITY_INIT = 1,
    /// Sensor uncalibrated
    DW_IMU_IMU_TURNRATE_QUALITY_UNCALIB = 2,
    /// Signal in specification
    DW_IMU_IMU_TURNRATE_QUALITY_OK = 3,
    /// Signal temporary failure
    DW_IMU_IMU_TURNRATE_QUALITY_TMP_FAIL = 4,
    /// Signal permanent failure
    DW_IMU_IMU_TURNRATE_QUALITY_PRMNT_FAIL = 5,
    /// Sensor not installed
    DW_IMU_IMU_TURNRATE_QUALITY_SENS_NOT_INST = 6,
    /// Sensor overloaded
    DW_IMU_IMU_TURNRATE_QUALITY_OVERLOAD = 7,
    /// Sensor out of operating temperature
    DW_IMU_IMU_TURNRATE_QUALITY_TEMPERATURE = 8,
    DW_IMU_IMU_TURNRATE_QUALITY_FORCE32     = 0x7FFFFFFF
} dwIMUImuTurnrateQuality;

typedef enum dwIMUImuTurnrateAccelQuality {
    /// Signal Unknown
    DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_UNKNOWN = 0,
    /// Signal initializing
    DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_INIT = 1,
    /// Sensor uncalibrated
    DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_UNCALIB = 2,
    /// Signal in specification
    DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_OK = 3,
    /// Signal temporary failure
    DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_TMP_FAIL = 4,
    /// Signal permanent failure
    DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_PRMNT_FAIL = 5,
    /// Sensor not installed
    DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_SENS_NOT_INST = 6,
    /// Sensor overloaded
    DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_OVERLOAD = 7,
    /// Sensor out of operating temperature
    DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_TEMPERATURE = 8,
    DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_FORCE32     = 0x7FFFFFFF
} dwIMUImuTurnrateAccelQuality;

typedef enum dwIMUImuTimestampQuality {
    /// Not Initialized. still initializing
    DW_IMU_IMU_TIMESTAMP_QUALITY_NOT_INIT = 0,
    /// Normal Operation. Functional and Electrical Checks Passed
    DW_IMU_IMU_TIMESTAMP_QUALITY_OK = 2,
    /// Synchronization lost.
    DW_IMU_IMU_TIMESTAMP_QUALITY_SYNC_LOST = 3,

    DW_IMU_IMU_TIMESTAMP_QUALITY_FORCE32 = 0x7FFFFFFF
} dwIMUImuTimestampQuality;

typedef enum dwIMUImuStatus {
    /// Signal Unknown
    DW_IMU_IMU_STATUS_UNKNOWN = 0,
    /// Signal initializing
    DW_IMU_IMU_STATUS_INIT = 1,
    /// Sensor uncalibrated
    DW_IMU_IMU_STATUS_UNCALIB = 2,
    /// Signal in specification
    DW_IMU_IMU_STATUS_OK = 3,
    /// Signal temporary failure
    DW_IMU_IMU_STATUS_TMP_FAIL = 4,
    /// Signal permanent failure
    DW_IMU_IMU_STATUS_PRMNT_FAIL = 5,
    /// Sensor not installed
    DW_IMU_IMU_STATUS_SENS_NOT_INST = 6,
    DW_IMU_IMU_STATUS_FORCE32       = 0x7FFFFFFF
} dwIMUImuStatus;

/**
 * This structure contains validity of each signal provided by IMU sensors.
 * Access fields in this structure with APIs provided by SignalStatus.h
 */
typedef struct dwIMUSignalValidityInfo
{
    /// Validity of timestamp signal
    dwSignalValidity timestamp_us;
    /// Validity of measurements of orientations in roll, pitch and yaw angles
    dwSignalValidity orientation[3];
    /// Validity of orientation values represented in quaternions
    dwSignalValidity orientationQuaternion;
    /// Validity of gyroscope in roll, pitch and yaw angles
    dwSignalValidity turnrate[3];
    /// Validity of acceleration signals in X, Y and Z axis
    dwSignalValidity acceleration[3];
    /// Validity of measurement of magnetometer signals in X, Y and Z axis
    dwSignalValidity magnetometer[3];
    /// Validity of heading signal
    dwSignalValidity heading;
    /// Validity of measured temperature
    dwSignalValidity temperature;
    /// Validity of measured acceleration offsets in X, Y and Z axis
    dwSignalValidity accelerationOffset[3];
    /// Validity of gyroscope offsets in roll, pitch and yaw angles
    dwSignalValidity turnrateOffset[3];
    /// Validity of gyroscope accelerations in roll, pitch and yaw angles
    dwSignalValidity turnrateAccel[3];
    /// Validity of quality of measured temperature
    dwSignalValidity imuTempQuality;
    /// Validity of status of acceleration values for X, Y and Z axis
    dwSignalValidity imuAccelerationQuality[3];
    /// Validity of status of gyroscope values for roll, pitch and yaw angles
    dwSignalValidity imuTurnrateQuality[3];
    /// Validity of status of gyroscope offsets for roll, pitch and yaw angles
    dwSignalValidity imuTurnrateOffsetQuality[3];
    /// Validity of status of gyroscope accelerations for roll, pitch and yaw angles
    dwSignalValidity imuTurnrateAccelQuality[3];
    /// Validity of status of timestamp signal
    dwSignalValidity imuTimestampQuality;
    /// Validity of overall IMU status signal
    dwSignalValidity imuStatus;
    /// Validity of alignment status signal
    dwSignalValidity alignmentStatus;
    /// Validity of IMU frame sequence counter value
    dwSignalValidity sequenceCounter;
    /// Validity of status of IMU gyroscope offset quality values for roll, pitch and yaw angles
    dwSignalValidity imuTurnrateOffsetQualityStatus[3];
    /// Validity of status of time sync
    dwSignalValidity timeSyncStatus;
    /// Reserved space
    dwSignalValidity reserved[58];
} dwIMUSignalValidityInfo;

/**
* This structure contains one frame of data from a IMU sensor.
*/
typedef struct dwIMUFrame
{
    /// Signal validity info
    dwIMUSignalValidityInfo validityInfo;

    /// @note ID VS-90010
    /// @note description Timestamp for the current message. Indicates when it's first received.
    /// @note min nan    max nan
    /// @note freq 100    unit us
    dwTime_t timestamp_us;

    /// @note ID VS-90020
    /// @note description Roll, pitch, and yaw angle of the orientation returned by the IMU. This is a signal that can be provided by certain types of IMUs as part of their internal state estimation. It is not used by egomotion.
    /// @note min nan    max nan
    /// @note freq 100    unit deg
    float64_t orientation[3];

    /// @note ID VS-90030
    /// @note description Quaternion representation (x, y, z, w) of the orientation returned by the IMU. This is a signal that can be provided by certain types of IMUs as part of their internal state estimation. It is not used by egomotion..
    /// @note min nan    max nan
    /// @note freq 100    unit unitless
    dwQuaterniond orientationQuaternion;

    /// @note ID VS-90040
    /// @note description Roll, pitch, and yaw turn rate (i.e., gyroscope). Angular velocities measured by the IMU sensor, they are given in the coordinate system of the sensor..
    /// @note min nan    max nan
    /// @note freq 100    unit rad/s
    float64_t turnrate[3];

    /// @note ID VS-90050
    /// @note description Acceleration in X, Y, and Z directions. Linear acceleration measured by the IMU sensor, it does include the gravity component as the sensor measures the reaction due to gravity. This is sometimes referred to as “proper acceleration”. This is an IMU sensor signal, as such it reflects all the motions of the sensor itself, including suspension flexing, gravity, vehicle turning, rolling and pitching. Coordinate system is that of the sensor itself, refer to manufacturer datasheet..
    /// @note min nan    max nan
    /// @note freq 100    unit m/s²
    float64_t acceleration[3];

    /// @note ID VS-90060
    /// @note description Measurement of the magnetometer unit in X, Y, and Z directions.
    /// @note min nan    max nan
    /// @note freq 100    unit utesla
    float64_t magnetometer[3];

    /// @note ID VS-90070
    /// @note description Heading of the IMU measured in respect to the ENU system, i.e., compass. This is a signal that can be provided by certain IMU+GNSS sensor solutions as part of the state estimation. It is not used by egomotion. Heading is defined with respect to ENU coordinate system, whereas yaw is in an arbitrary local coordinate system..
    /// @note min nan    max nan
    /// @note freq 100    unit deg
    float64_t heading;

    /// @note ID VS-90090
    /// @note description IMU temperature.
    /// @note min nan    max nan
    /// @note freq 100    unit C
    float32_t temperature;

    /// @note ID VS-90100
    /// @note description IMU acceleration offset values.
    /// @note min nan    max nan
    /// @note freq 100    unit m/s²
    float64_t accelerationOffset[3];

    /// @note ID VS-90110
    /// @note description IMU gyroscope offset values.
    /// @note min nan    max nan
    /// @note freq 100    unit rad/s
    float64_t turnrateOffset[3];

    /// @note ID VS-90120
    /// @note description IMU gyroscope acceleration values.
    /// @note min nan    max nan
    /// @note freq 100    unit rad/sec²
    float64_t turnrateAccel[3];

    /// @note ID VS-90140
    /// @note description Vehicle IMU temperature status.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 100    unit  -
    /// @note *** valid values**: {
    ///                DW_IMU_IMU_TEMP_QUALITY_INIT,
    ///                DW_IMU_IMU_TEMP_QUALITY_UNCALIB,
    ///                DW_IMU_IMU_TEMP_QUALITY_OK,
    ///                DW_IMU_IMU_TEMP_QUALITY_TMP_FAIL,
    ///                DW_IMU_IMU_TEMP_QUALITY_PRMNT_FAIL,
    ///                DW_IMU_IMU_TEMP_QUALITY_SENS_NOT_INST
    ///        }
    dwIMUImuTempQuality imuTempQuality;

    /// @note ID VS-90150
    /// @note description Vehicle IMU acceleration values status.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 100    unit  -
    /// @note *** valid values**: {
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_UNKNOWN
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_INIT,
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_UNCALIB,
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_OK,
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_TMP_FAIL,
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_PRMNT_FAIL,
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_SENS_NOT_INST,
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_OVERLOAD,
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_TEMPERATURE
    ///        }
    dwIMUImuAccelerationQuality imuAccelerationQuality[3];

    /// @note ID VS-90160
    /// @note description Vehicle IMU gyroscope values quality.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 100    unit  -
    /// @note *** valid values**: {
    ///                DW_IMU_IMU_TURNRATE_QUALITY_UNKNOWN
    ///                DW_IMU_IMU_TURNRATE_QUALITY_INIT,
    ///                DW_IMU_IMU_TURNRATE_QUALITY_UNCALIB,
    ///                DW_IMU_IMU_TURNRATE_QUALITY_OK,
    ///                DW_IMU_IMU_TURNRATE_QUALITY_TMP_FAIL,
    ///                DW_IMU_IMU_TURNRATE_QUALITY_PRMNT_FAIL,
    ///                DW_IMU_IMU_TURNRATE_QUALITY_SENS_NOT_INST,
    ///                DW_IMU_IMU_TURNRATE_QUALITY_OVERLOAD,
    ///                DW_IMU_IMU_TURNRATE_QUALITY_TEMPERATURE
    ///        }
    dwIMUImuTurnrateQuality imuTurnrateQuality[3];

    /// @note ID VS-90170
    /// @note description Vehicle IMU gyroscope offset values quality on a scale of 0...62.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 100    unit unitless
    uint8_t imuTurnrateOffsetQuality[3];

    /// @note ID VS-90180
    /// @note description Vehicle IMU gyroscope acceleration values quality.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 100    unit  -
    /// @note *** valid values**: {
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_UNKNOWN
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_INIT,
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_UNCALIB,
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_OK,
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_TMP_FAIL,
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_PRMNT_FAIL,
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_SENS_NOT_INST,
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_OVERLOAD,
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_TEMPERATURE
    ///        }
    dwIMUImuTurnrateAccelQuality imuTurnrateAccelQuality[3];

    /// @note ID VS-90190
    /// @note description Vehicle IMU timestamp quality.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 100    unit  -
    /// @note *** valid values**: {
    ///                DW_IMU_IMU_TIMESTAMP_QUALITY_NOT_INIT,
    ///                DW_IMU_IMU_TIMESTAMP_QUALITY_OK,
    ///                DW_IMU_IMU_TIMESTAMP_QUALITY_SYNC_LOST,
    ///        }
    dwIMUImuTimestampQuality imuTimestampQuality;

    /// @note ID VS-90200
    /// @note description Vehicle IMU overall status.
    /// @note min nan    max nan
    /// @note freq 100    unit  -
    /// @note *** valid values**: {
    ///                DW_IMU_IMU_STATUS_UNKNOWN
    ///                DW_IMU_IMU_STATUS_INIT,
    ///                DW_IMU_IMU_STATUS_UNCALIB,
    ///                DW_IMU_IMU_STATUS_OK,
    ///                DW_IMU_IMU_STATUS_TMP_FAIL,
    ///                DW_IMU_IMU_STATUS_PRMNT_FAIL,
    ///                DW_IMU_IMU_STATUS_SENS_NOT_INST
    ///        }
    dwIMUImuStatus imuStatus;

    /// Alignment status
    dwIMUAlignmentStatus alignmentStatus;

    /// @note description Sequence counter
    /// @note TODO: Backend not yet implemented
    uint8_t sequenceCounter;

    /// @note ID VS-90171
    /// @note description Vehicle IMU gyroscope offset values quality is free of errors.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 100    unit unitless
    uint8_t imuTurnrateOffsetQualityStatus[3];

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    /// Type of the heading information.
    dwIMUHeadingType headingType
        DW_DEPRECATED("headingType is deprecated and will be removed in the next major release. Please don't use this field.");
#pragma GCC diagnostic pop

    /// The flags to show which values are valid in this IMU frame. Definition of each bit is in enum @b dwIMUFlags
    uint32_t flags
        DW_DEPRECATED("flags is deprecated and will be removed in next major release. Please use APIs provided in IMUGetterSetter.h to access and check validity of signals");

    /// time sync status
    uint8_t timeSyncStatus;

    /// Reserved
    uint8_t reserved[495];
} dwIMUFrame;

/**
* This structure contains one frame of data from a IMU sensor.
 */
typedef struct dwIMUFrameNew
{
    /// Signal validity information
    dwIMUSignalValidityInfo validityInfo;

    /// @note ID VS-90010
    /// @note description Timestamp for the current message. Indicates when it's first received.
    /// @note min nan    max nan
    /// @note freq 100    unit us
    dwTime_t timestamp_us;

    /// @note ID VS-90020
    /// @note description Roll, pitch, and yaw angle of the orientation returned by the IMU. This is a signal that can be provided by certain types of IMUs as part of their internal state estimation. It is not used by egomotion.
    /// @note min nan    max nan
    /// @note freq 100    unit deg
    float64_t orientation[3];

    /// @note ID VS-90030
    /// @note description Quaternion representation (x, y, z, w) of the orientation returned by the IMU. This is a signal that can be provided by certain types of IMUs as part of their internal state estimation. It is not used by egomotion..
    /// @note min nan    max nan
    /// @note freq 100    unit unitless
    dwQuaterniond orientationQuaternion;

    /// @note ID VS-90040
    /// @note description Roll, pitch, and yaw turn rate (i.e., gyroscope). Angular velocities measured by the IMU sensor, they are given in the coordinate system of the sensor..
    /// @note min nan    max nan
    /// @note freq 100    unit rad/s
    float64_t turnrate[3];

    /// @note ID VS-90050
    /// @note description Acceleration in X, Y, and Z directions. Linear acceleration measured by the IMU sensor, it does include the gravity component as the sensor measures the reaction due to gravity. This is sometimes referred to as “proper acceleration”. This is an IMU sensor signal, as such it reflects all the motions of the sensor itself, including suspension flexing, gravity, vehicle turning, rolling and pitching. Coordinate system is that of the sensor itself, refer to manufacturer datasheet..
    /// @note min nan    max nan
    /// @note freq 100    unit m/s²
    float64_t acceleration[3];

    /// @note ID VS-90060
    /// @note description Measurement of the magnetometer unit in X, Y, and Z directions.
    /// @note min nan    max nan
    /// @note freq 100    unit utesla
    float64_t magnetometer[3];

    /// @note ID VS-90070
    /// @note description Heading of the IMU measured in respect to the ENU system, i.e., compass. This is a signal that can be provided by certain IMU+GNSS sensor solutions as part of the state estimation. It is not used by egomotion. Heading is defined with respect to ENU coordinate system, whereas yaw is in an arbitrary local coordinate system..
    /// @note min nan    max nan
    /// @note freq 100    unit deg
    float64_t heading;

    /// @note ID VS-90090
    /// @note description IMU temperature.
    /// @note min nan    max nan
    /// @note freq 100    unit C
    float32_t temperature;

    /// @note ID VS-90100
    /// @note description IMU acceleration offset values.
    /// @note min nan    max nan
    /// @note freq 100    unit m/s²
    float64_t accelerationOffset[3];

    /// @note ID VS-90110
    /// @note description IMU gyroscope offset values.
    /// @note min nan    max nan
    /// @note freq 100    unit rad/s
    float64_t turnrateOffset[3];

    /// @note ID VS-90120
    /// @note description IMU gyroscope acceleration values.
    /// @note min nan    max nan
    /// @note freq 100    unit rad/sec²
    float64_t turnrateAccel[3];

    /// @note ID VS-90140
    /// @note description Vehicle IMU temperature status.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 100    unit  -
    /// @note *** valid values**: {
    ///                DW_IMU_IMU_TEMP_QUALITY_INIT,
    ///                DW_IMU_IMU_TEMP_QUALITY_UNCALIB,
    ///                DW_IMU_IMU_TEMP_QUALITY_OK,
    ///                DW_IMU_IMU_TEMP_QUALITY_TMP_FAIL,
    ///                DW_IMU_IMU_TEMP_QUALITY_PRMNT_FAIL,
    ///                DW_IMU_IMU_TEMP_QUALITY_SENS_NOT_INST
    ///        }
    dwIMUImuTempQuality imuTempQuality;

    /// @note ID VS-90150
    /// @note description Vehicle IMU acceleration values status.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 100    unit  -
    /// @note *** valid values**: {
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_UNKNOWN
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_INIT,
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_UNCALIB,
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_OK,
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_TMP_FAIL,
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_PRMNT_FAIL,
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_SENS_NOT_INST,
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_OVERLOAD,
    ///                DW_IMU_IMU_ACCELERATION_QUALITY_TEMPERATURE
    ///        }
    dwIMUImuAccelerationQuality imuAccelerationQuality[3];

    /// @note ID VS-90160
    /// @note description Vehicle IMU gyroscope values quality.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 100    unit  -
    /// @note *** valid values**: {
    ///                DW_IMU_IMU_TURNRATE_QUALITY_UNKNOWN
    ///                DW_IMU_IMU_TURNRATE_QUALITY_INIT,
    ///                DW_IMU_IMU_TURNRATE_QUALITY_UNCALIB,
    ///                DW_IMU_IMU_TURNRATE_QUALITY_OK,
    ///                DW_IMU_IMU_TURNRATE_QUALITY_TMP_FAIL,
    ///                DW_IMU_IMU_TURNRATE_QUALITY_PRMNT_FAIL,
    ///                DW_IMU_IMU_TURNRATE_QUALITY_SENS_NOT_INST,
    ///                DW_IMU_IMU_TURNRATE_QUALITY_OVERLOAD,
    ///                DW_IMU_IMU_TURNRATE_QUALITY_TEMPERATURE
    ///        }
    dwIMUImuTurnrateQuality imuTurnrateQuality[3];

    /// @note ID VS-90170
    /// @note description Vehicle IMU gyroscope offset values quality on a scale of 0...62.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 100    unit unitless
    uint8_t imuTurnrateOffsetQuality[3];

    /// @note ID VS-90180
    /// @note description Vehicle IMU gyroscope acceleration values quality.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 100    unit  -
    /// @note *** valid values**: {
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_UNKNOWN
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_INIT,
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_UNCALIB,
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_OK,
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_TMP_FAIL,
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_PRMNT_FAIL,
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_SENS_NOT_INST,
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_OVERLOAD,
    ///                DW_IMU_IMU_TURNRATE_ACCEL_QUALITY_TEMPERATURE
    ///        }
    dwIMUImuTurnrateAccelQuality imuTurnrateAccelQuality[3];

    /// @note ID VS-90190
    /// @note description Vehicle IMU timestamp quality.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 100    unit  -
    /// @note *** valid values**: {
    ///                DW_IMU_IMU_TIMESTAMP_QUALITY_NOT_INIT,
    ///                DW_IMU_IMU_TIMESTAMP_QUALITY_OK,
    ///                DW_IMU_IMU_TIMESTAMP_QUALITY_SYNC_LOST,
    ///        }
    dwIMUImuTimestampQuality imuTimestampQuality;

    /// @note ID VS-90200
    /// @note description Vehicle IMU overall status.
    /// @note min nan    max nan
    /// @note freq 100    unit  -
    /// @note *** valid values**: {
    ///                DW_IMU_IMU_STATUS_UNKNOWN
    ///                DW_IMU_IMU_STATUS_INIT,
    ///                DW_IMU_IMU_STATUS_UNCALIB,
    ///                DW_IMU_IMU_STATUS_OK,
    ///                DW_IMU_IMU_STATUS_TMP_FAIL,
    ///                DW_IMU_IMU_STATUS_PRMNT_FAIL,
    ///                DW_IMU_IMU_STATUS_SENS_NOT_INST
    ///        }
    dwIMUImuStatus imuStatus;

    /// Alignment status
    dwIMUAlignmentStatus alignmentStatus;

    /// @note description Sequence counter
    /// @note TODO: Backend not yet implemented
    uint8_t sequenceCounter;

    /// @note ID VS-90171
    /// @note description Vehicle IMU gyroscope offset values quality is free of errors.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 100    unit unitless
    uint8_t imuTurnrateOffsetQualityStatus[3];

    /// Reserved
    uint8_t reserved[504];
} dwIMUFrameNew
    DW_DEPRECATED("dwIMUFrameNew is deprecated and will be removed soon. Please use dwIMUFrame instead.");

/**
 * Reads the next IMU frame from the sensor within a given timeout. The method blocks until
 * either a new valid frame is received from the sensor or the given timeout is exceeded.
 *
 * @param[out] frame A pointer to an IMU frame structure to be filled with new data.
 * @param[in] timeoutUs Timeout, in us, to wait for a new message. Special values: DW_TIMEOUT_INFINITE - to wait infinitely.  Zero - means polling of internal queue.
 * @param[in] sensor Sensor handle of the IMU sensor previously created with dwSAL_createSensor().
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid <br>
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid <br>
 *         DW_TIME_OUT - if operation has timeout <br>
 *         DW_NOT_AVAILABLE    - if sensor has not been started or data is not available in polling mode. <br>
 *         DW_END_OF_STREAM    - if end of stream reached (virtual sensor only). <br>
 *         DW_SAL_SENSOR_ERROR - if there was an i/o or bus error. <br>
 *         DW_SUCCESS
**/
DW_API_PUBLIC
dwStatus dwSensorIMU_readFrame(dwIMUFrame* const frame, dwTime_t const timeoutUs, dwSensorHandle_t const sensor);

/**
 * Reads the IMU frame from raw data. Any processed messages can be picked up using
 * the dwSensorIMU_readFrame() method. This happens on the CPU thread where the function is called,
 * incurring an additional load on that thread.
 *
 * @param[in] data Undecoded imu data.
 * @param[in] size Size in bytes of the raw data.
 * @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid, <br>
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid, <br>
 *         DW_NOT_READY - if more data needs to be passed in (loop while it returns 'DW_NOT_READY'). <br>
 *         DW_SUCCESS
**/
DW_API_PUBLIC
dwStatus dwSensorIMU_processRawData(uint8_t const* const data, size_t const size, dwSensorHandle_t const sensor);

/**
 * Returns any IMU data previously processed through the raw data stream.
 * This happens on the CPU thread where the function is called, incurring an additional load on that thread.
 *
 * @param[out] frame Pointer to an IMU frame structure to be filled with new data.
 * @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid. <br>
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid. <br>
 *         DW_NOT_AVAILABLE - if no more data is available <br>
 *         DW_SUCCESS
**/
DW_API_PUBLIC
dwStatus dwSensorIMU_popFrame(dwIMUFrame* const frame, dwSensorHandle_t const sensor);

/**
 * Reads the next IMU frame New from the sensor within a given timeout. The method blocks until
 * either a new valid frame is received from the sensor or the given timeout is exceeded.
 *
 * @param[out] frame A pointer to an IMU frame New structure to be filled with new data.
 * @param[in] timeoutUs Timeout, in us, to wait for a new message. Special values: DW_TIMEOUT_INFINITE - to wait infinitely.  Zero - means polling of internal queue.
 * @param[in] sensor Sensor handle of the IMU sensor previously created with dwSAL_createSensor().
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid <br>
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid <br>
 *         DW_TIME_OUT - if operation has timeout <br>
 *         DW_NOT_AVAILABLE    - if sensor has not been started or data is not available in polling mode. <br>
 *         DW_END_OF_STREAM    - if end of stream reached (virtual sensor only). <br>
 *         DW_SAL_SENSOR_ERROR - if there was an i/o or bus error. <br>
 *         DW_SUCCESS
**/
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
DW_API_PUBLIC
dwStatus dwSensorIMU_readFrameNew(dwIMUFrameNew* const frame, dwTime_t const timeoutUs, dwSensorHandle_t const sensor)
    DW_DEPRECATED("dwSensorIMU_readFrameNew() is deprecated and will be removed in next major release. Please use dwSensorIMU_readFrame() instead");
#pragma GCC diagnostic pop

/**
 * Reads the IMU frame New from raw data. Any processed messages can be picked up using
 * the dwSensorIMU_readFrameNew() method. This happens on the CPU thread where the function is called,
 * incurring an additional load on that thread.
 *
 * @param[in] data Undecoded imu data.
 * @param[in] size Size in bytes of the raw data.
 * @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid, <br>
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid, <br>
 *         DW_NOT_READY - if more data needs to be passed in (loop while it returns 'DW_NOT_READY'). <br>
 *         DW_SUCCESS
**/
DW_API_PUBLIC
dwStatus dwSensorIMU_processRawDataNew(uint8_t const* const data, size_t const size, dwSensorHandle_t const sensor)
    DW_DEPRECATED("dwSensorIMU_processRawDataNew() is deprecated and will be removed in next major release. Please use dwSensorIMU_processRawData() instead");

/**
 * Returns any IMU Frame New previously processed through the raw data stream.
 * This happens on the CPU thread where the function is called, incurring an additional load on that thread.
 *
 * @param[out] frame Pointer to an IMU frame New structure to be filled with new data.
 * @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid. <br>
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid. <br>
 *         DW_NOT_AVAILABLE - if no more data is available <br>
 *         DW_SUCCESS
**/
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
DW_API_PUBLIC
dwStatus dwSensorIMU_popFrameNew(dwIMUFrameNew* const frame, dwSensorHandle_t const sensor)
    DW_DEPRECATED("dwSensorIMU_popFrameNew() is deprecated and will be removed in next major release. Please use dwSensorIMU_popFrame() instead");
#pragma GCC diagnostic pop

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_SENSORS_IMU_IMU_H_
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

/**
 * @file
 * <b>NVIDIA DriveWorks API: SensorTypes</b>
 *
 * @b Description: This file defines the sensor base types.
 */

/**
 * @defgroup sensors_common_group Common
 *
 * @brief Defines base types common to all sensors.
 *
 * @ingroup sensors_group
 * @{
 */

#ifndef DW_SENSORS_COMMON_SENSORTYPES_H_
#define DW_SENSORS_COMMON_SENSORTYPES_H_

#include <dw/core/base/Types.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Defines the operating system in use.
typedef enum dwPlatformOS {
    /// Default Linux based platform.
    DW_PLATFORM_OS_LINUX = 0,

    /// aarch64 based Drive 5 Linux.
    DW_PLATFORM_OS_V5L = 1,

    /// aarch64 based Drive 5 QNX.
    DW_PLATFORM_OS_V5Q = 2,

    /// Number of available platforms.
    DW_PLATFORM_NUM = 3
} dwPlatformOS;

/// Handle representing the Sensor Abstraction Layer interface.
typedef struct dwSALObject* dwSALHandle_t;

/// Handle representing a sensor.
typedef struct dwSensorObject* dwSensorHandle_t;

/// Maximal length of the protocol name of a sensor.
#define DW_SENSOR_MAX_PROTOCOL_NAME_LENGTH 32
/// Maximal length of the parameter string of a sensor.
#define DW_SENSOR_MAX_PARAMETER_STRING_LENGTH 512

/// Holds sets of parameters for sensor creation.
typedef struct dwSensorParams
{
    /** Name of the protocol. Only first DW_SENSOR_MAX_PROTOCOL_NAME_LENGTH
     * characters are used.
     */
    const char8_t* protocol;

    /** Array of additional parameters provided to sensor creation.
     * In general, this array has a form of key/value pairs separated by commas,
     * i.e., key1=value1,key2=value2,key3=value3.
     * Only first DW_SENSOR_MAX_PARAMETER_STRING_LENGTH characters are used.
     */
    const char8_t* parameters;

    /**
     * Additional data to pass to sensor creation function. This data can be also used
     * for data which cannot be passed as a string, but is required during sensor creation.
     * Only specific subset of sensors might expect any data here, hence in general this should be NULL.
     **/
    const void* auxiliarydata;

} dwSensorParams;

/// Defines the type of sensors that are available in DriveWorks
typedef enum dwSensorType {
    /// CAMERA
    DW_SENSOR_CAMERA = 0,
    /// Lidar
    DW_SENSOR_LIDAR,
    /// GPS
    DW_SENSOR_GPS,
    /// IMU
    DW_SENSOR_IMU,
    /// CAN
    DW_SENSOR_CAN,
    /// RADAR
    DW_SENSOR_RADAR,
    /// TIME
    DW_SENSOR_TIME,
    /// DATA
    DW_SENSOR_DATA,
    /// Ultrasonic
    DW_SENSOR_ULTRASONIC,
    /// Sensor count which the sensor type value is less than
    DW_SENSOR_COUNT
} dwSensorType;

/// A seek structure is made of memory offset, eventcount and timestamp
typedef struct dwSensorSeekTableEntry
{
    /// timestamp of this entry [us]
    dwTime_t timestamp;

    /// count of this event
    uint64_t event;

    /// offset into a file of this event
    uint64_t offset;

    /// size of the event present in the virtual data file, in bytes
    uint64_t size;

} dwSensorSeekTableEntry;

/** bit-shift of dwSensorErrorID when reported via module health service */
#define DW_SENSOR_ERROR_ID_OFFSET_BIT 24U
/** offset of dwSensorErrorID when reported via module health service */
#define DW_SENSOR_ERROR_ID_OFFSET (1U << DW_SENSOR_ERROR_ID_OFFSET_BIT)

/**
 * @brief Sensor Error ID to be used in, e.g., dwErrorSignal.errorID[31:24]
 * when DW_SENSOR_ERROR_CODE_OFFSET_BIT equals 24
 */
typedef enum dwSensorErrorID {
    /** sensor ok */
    DW_SENSOR_ERROR_CODE_INVALID = 0,
    /** no new data received */
    /** Error number is defined so that health service filters the error and not report to SEH */
    DW_SENSOR_ERROR_CODE_NO_NEW_DATA = 120001,
    /** No data is received from publisher*/
    DW_SENSOR_ERROR_NO_DATA_FROM_PUBLISHER,
    /** Not enough dats from publisher since initialization */
    DW_SENSOR_ERROR_NOT_ENOUGH_DATA,
    /** E2E failures*/
    DW_SENSOR_ERROR_E2E_FAILURE,
    /** State machine is disabled*/
    DW_SENSOR_ERROR_STATE_MACHINE_DISABLED,
    /** Non monotonic timestamp received from the sensor*/
    /** Error number is defined so that health service filters the error and not report to SEH */
    DW_SENSOR_ERROR_CODE_NON_MONOTONIC_DATA,
    /** Radar specific errors: azimuth is out of range */
    DW_RADAR_ERROR_CODE_AZIMUTH_OUT_OF_RANGE = 121000,
    /** Radar specific errors: return num is out of range */
    DW_RADAR_ERROR_CODE_NUM_RETURNS_OUT_OF_RANGE = 121001,
    /** Radar specific errors: zero return */
    DW_RADAR_ERROR_CODE_ZERO_RETURNS = 121002,
    /** Radar specific errors: end index */
    DW_RADAR_ERROR_CODE_END_INDEX = 121199,
    /* Place holder error IDs - use SEH official error IDs later */
    /** GPS sensor working mode error */
    DW_SENSORS_ERROR_CODE_GPS_MODE = 130001,
    /** GPS sensor accuracy warning */
    DW_SENSORS_ERROR_CODE_GPS_ACCURACY = 130002,
    /** IMU sensor bad alignment status */
    DW_SENSORS_ERROR_CODE_IMU_ALIGNMENT_STATUS = 140001,
} dwSensorErrorID;

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_SENSORS_COMMON_SENSORTYPES_H_

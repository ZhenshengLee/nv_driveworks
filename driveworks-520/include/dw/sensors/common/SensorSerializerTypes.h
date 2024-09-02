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
 * <b>NVIDIA DriveWorks API: Sensor Serializer Types</b>
 */

/**
 * @defgroup sensor_serializer_group Sensor Serializer
 * @ingroup sensors_group
 *
 * @brief Defines sensor serializer base types.
 * @{
 *
 */

#ifndef DW_SENSORS_COMMON_SENSORSERIALIZERTYPES_H_
#define DW_SENSORS_COMMON_SENSORSERIALIZERTYPES_H_

#include <dw/core/base/Types.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Defines the various delta used in statistics.
typedef enum dwSerializerStatTimeDifference {
    /// Stat to hold the disk write time delta
    DW_SERIALIZER_STAT_DISK_WRITE_TIME = 0,

    /// Stat to hold the encode time delta
    DW_SERIALIZER_STAT_ENCODE_TIME = 1,

    /// Stat to hold the stage1 time delta
    DW_SERIALIZER_STAT_STAGE1_TIME = 2,

    /// Stat to hold the stage2 time delta
    DW_SERIALIZER_STAT_STAGE2_TIME = 3,

    /// Stat to hold the stage3 time delta
    DW_SERIALIZER_STAT_STAGE3_TIME = 4,

    /// Count which the type value is less than
    DW_SERIALIZER_STAT_COUNT = 5,
} dwSerializerStatTimeDifference;

/// Holds the available statistics for a serializer.
typedef struct dwSerializerStats
{
    /// Array of current latencies of all the time deltas between various stages of serialization
    /// as per the dwSerializerStatTimeDifference [us]
    dwTime_t currentDeltaUs[DW_SERIALIZER_STAT_COUNT];
    /// Array of min latencies of all the time deltas between various stages of serialization
    /// as per the dwSerializerStatTimeDifference [us]
    dwTime_t minDeltaUs[DW_SERIALIZER_STAT_COUNT];
    /// Array of max latencies of all the time deltas between various stages of serialization
    /// as per the dwSerializerStatTimeDifference [us]
    dwTime_t maxDeltaUs[DW_SERIALIZER_STAT_COUNT];
    /// Array of Variance of all the time deltas between various stages of serialization
    /// as per the dwSerializerStatTimeDifference
    float64_t standardDeviationDelta[DW_SERIALIZER_STAT_COUNT];
    /// Array of Mean of all the time deltas between various stages of serialization
    /// as per the dwSerializerStatTimeDifference
    float64_t meanDelta[DW_SERIALIZER_STAT_COUNT];
} dwSerializerStats;

/**
 * Callback type for getting data from sensor serializer.
 * @param[in] data A pointer to the byte array of serialized data.
 * @param[in] size A pointer to the size of the byte array.
 * @param[in] userData User Data
 */
typedef void (*dwSensorSerializerOnDataFunc_t)(const uint8_t* data, size_t size, void* userData);

/** Handle representing a sensor serializer. */
typedef struct dwSensorSerializerObject* dwSensorSerializerHandle_t;

/** Holds the parameters for sensor serializer creation.
 */
typedef struct dwSerializerParams
{
    /** Array for additional parameters provided to sensor serializer creation.
      * The @a parameters argument is an array in the form of key-value pairs separated by commas,
      * i.e., key1=value1,key2=value2,key3=value3.
      *
      * Supported 'keys' are:
      *
      * - 'type' - Required. Specifies data-sink settings.
      *   - If the value of 'type' is 'disk', the serializer
      *     streams data to the file specified in the 'file' key.
      *     For an example, see dwSensorSerializer_initialize().
      *   - If the value of 'type' is 'user', the serializer uses the provided
      *     callback to stream data. When new data is available,
      *     the serializer calls the function provided
      *     in @a onData and puts the data in the buffer provided by
      *     @a userData.<br>
      * - 'file' - See description for 'type'.
      * - 'file-buffer-size' - Size of output buffer to use for file operations.
      * - 'async-record' - Optional. Interpreted as uint16_t.
      *                    Specify whether the serializer works in 'sync' mode (with a value of zero) or 'async' mode (with a non-zero value).
      *                    The default mode is 'sync'. <br>
      * Following key-value pairs only applied for video data. <br>
      * - 'format' - Optional. Specifies the video format. Supported values are 'h264', 'mp4' and 'raw'.
      * - 'bitrate' - Optional. Interpreted as uint32_t. Specify average frame bitrate. Default value is zero.
      * - 'framerate' - Optional. Interpreted as uint32_t. Specify frame proccessing rate. Default value is zero.
      * - 'quality' - Optional. String value. If non-empty, encoder qualitywill be set to max value.
      * - 'gop-length' - Optional. String value. If non-empty, set group of pictures's length to 16, default length is 1.
      * - 'image-type' - Optional. Interpreted as uint32_t. Specify image output type, see 'dwCameraOutputType', default output type is 'DW_CAMERA_OUTPUT_NATIVE_PROCESSED'.
      *
      * For a code snippet, see dwSensorSerializer_initialize().
      */
    const char8_t* parameters;

    /// Callback executed by the serializer on new data available.
    dwSensorSerializerOnDataFunc_t onData;

    /// Context used by callback @ref onData
    void* userData;
} dwSerializerParams;

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_SENSORS_COMMON_SENSORSERIALIZERTYPES_H_

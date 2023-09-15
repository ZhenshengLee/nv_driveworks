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
 * <b>NVIDIA DriveWorks API: Sensor Serializer</b>
 *
 * @b Description: This file defines sensor serializer methods.
 */

/**
 * @defgroup sensor_serializer_group Sensor Serializer
 * @ingroup sensors_group
 *
 * @brief Defines sensor serializer.
 * @{
 *
 */

#ifndef DW_SENSORS_SENSORSERIALIZER_H_
#define DW_SENSORS_SENSORSERIALIZER_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>
#include <dw/core/context/Context.h>

#include <dw/sensors/Sensors.h>
#include <dw/sensors/camera/Camera.h>

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
      * - 'format' - Required. Specifies the video format. Supported values are 'h264' and 'raw'.
      * - 'bitrate' - Required if 'format' is 'h264'; optional if it is 'raw'.
      * - 'framerate' - Optional.
      *
      * For a code snippet, see dwSensorSerializer_initialize().
      */
    const char8_t* parameters;

    /// Callback executed by the serializer on new data available.
    dwSensorSerializerOnDataFunc_t onData;

    /// User data to be passed to the callback.
    void* userData;
} dwSerializerParams;

/**
 * Initializes a sensor serializer with the parameters provided.
 *
 *
 * @param[out] serializer A pointer to the sensor serializer handle.
 * @param[in] params A pointer to the sensor serializer parameters.
 * @param[in] sensor Specifies the sensor used to create the serializer. This is
 *                   necessary because each sensor has its own unique
 *                   serializer. For example, camera provides a serializer
 *                   that can encode the data, while GPS serializes RAW
 *                   data.
 *
 * @note Creating serializer from @b virtual sensors will perform a copy of the data
 *
 * @return DW_INVALID_HANDLE - if provided inputs are invalid <br>
 *         DW_INVALID_ARGUMENT <br>
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwSensorSerializer_initialize(dwSensorSerializerHandle_t* const serializer,
                                       dwSerializerParams const* const params,
                                       dwSensorHandle_t const sensor);

/**
 * Releases a sensor serializer. If the serializer has been
 * attached to a master serializer via dwSensorSerializer_attachTo(),
 * this method fails with DW_NOT_AVAILABLE. If this happens,
 * dwSensorSerializer_detachFrom() must be called, and then this method
 * succeeds. If the serializer is a master serializer
 * that has other serializers attached, it releases as normal
 * and the slave serializers are no longer be attached.
 *
 * @param[in] serializer The sensor serializer handle.
 *
 * @return DW_INVALID_HANDLE - if provided inputs are invalid. <br>
 *         DW_INVALID_ARGUMENT <br>
 *         DW_NOT_AVAILABLE - if provided serializer is attached to another.
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwSensorSerializer_release(dwSensorSerializerHandle_t const serializer);

/**
 * Starts and stops serialization of a sensor with a master serializer.
 * This method attaches the serializer to the same thread of the master
 * serializer. The slave serializer handle is still used to
 * call dwSensorSerializer_serializeDataAsync().
 *
 * @note To ensure you don't miss any data, activate serialization before calling
 * dwSensorSerializer_startSensor().
 *
 * @param[in] serializer Specifies the sensor serializer handle.
 * @param[in] masterSerializer Specifies the sensor serializer handle that
 * is the main thread.
 *
 * @return DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_SUCCESS
 *
 */
DW_API_PUBLIC
DW_DEPRECATED("This API is deprecated and will be removed in the next major release")
dwStatus dwSensorSerializer_attachTo(dwSensorSerializerHandle_t const serializer,
                                     dwSensorSerializerHandle_t const masterSerializer);

/**
 * Query method to check whether the serializer is attached to another.
 * @see dwSensorSerializer_attachTo()
 *
 * @param[out] isAttached Specifies whether the sensor serializer handle is attached to another.
 * @param[in] serializer The sensor serializer handle.
 *
 * @return DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_SUCCESS
 *
 */
DW_API_PUBLIC
DW_DEPRECATED("This API is deprecated and will be removed in the next major release")
dwStatus dwSensorSerializer_isAttached(bool* const isAttached, dwSensorSerializerHandle_t const serializer);

/**
 * This method detaches the serializer previously attached with
 * dwSensorSerializer_attachTo().
 * Due to the asyncronous nature of this call,
 * the serializer may still be attached to the serializing thread
 * immediately following this call. dwSensorSerializer_isAttached()
 * must be polled for status.
 *
 * @note To ensure you don't miss any data, activate serialization before calling
 * dwSensorSerializer_startSensor().
 *
 * @param[in] serializer Specifies the sensor serializer handle.
 * @param[in] masterSerializer Specifies the sensor serializer handle that
 * takes ownership of the serializer.
 *
 * @return DW_NOT_AVAILABLE - serializer is not attached to another serializer. <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_SUCCESS
 *
 */
DW_API_PUBLIC
DW_DEPRECATED("This API is deprecated and will be removed in the next major release")
dwStatus dwSensorSerializer_detachFrom(dwSensorSerializerHandle_t const serializer,
                                       dwSensorSerializerHandle_t const masterSerializer);

/**
 * Starts serialization of sensor. This method creates a new thread and
 * begins the serialization loop.
 *
 * @note To ensure you don't miss any data, activate serialization
 *       before calling dwSensorSerializer_startSensor().
 *
 * @param[in] serializer Specifies the sensor serializer handle.
 *
 * @return DW_NOT_AVAILABLE - serialization is not available at this moment. <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_SUCCESS
 *
 */
DW_API_PUBLIC
dwStatus dwSensorSerializer_start(dwSensorSerializerHandle_t const serializer);

/**
 * Starts serialization of sensor. This method stops the thread and
 * the serialization loop.
 *
 * @param[in] serializer Specifies the sensor serializer handle.
 *
 * @return DW_NOT_AVAILABLE - serialization is not available at this moment. <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_SUCCESS
 *
 */
DW_API_PUBLIC
dwStatus dwSensorSerializer_stop(dwSensorSerializerHandle_t const serializer);

/**
 * Pushes data to the serializer. This method is thread-safe and thus
 * can be used on the capture thread (or any other thread).
 *
 * @param[in] data A pointer to the byte array of data.
 * @param[in] size Specifies the size of the byte array.
 * @param[in] serializer Specifies the handle to the sensor serializer.
 *
 * @return DW_NOT_AVAILABLE - serialization is not available at this moment. <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_SUCCESS
 *
 */
DW_API_PUBLIC
dwStatus dwSensorSerializer_serializeData(uint8_t const* const data, size_t const size,
                                          dwSensorSerializerHandle_t const serializer);

/**
 * Pushes data to the serializer. This method is thread-safe and thus
 * can be used on the capture thread (or any other thread). Use this method
 * in conjunction with 'dwSensorSerializer_start'/'dwSensorSerializer_stop'.
 *
 * @param[in] data A pointer to the byte array of data.
 * @param[in] size Specifies the size of the byte array.
 * @param[in] serializer Specifies the handle to the sensor serializer.
 *
 * @return DW_NOT_AVAILABLE - serialization is not available at this moment. <br>
 *         DW_BUFFER_FULL - serializer buffer is full, data was not pushed to serializer. <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_SUCCESS
 *
 * \ingroup sensors
 */
DW_API_PUBLIC
dwStatus dwSensorSerializer_serializeDataAsync(uint8_t const* const data, size_t const size,
                                               dwSensorSerializerHandle_t const serializer);

/**
 * Pushes a camera frame to the serializer.This method must only be used if 'dwSensorSerializer_start'
 * is not called. This pushes the serialized image directly to the sink.
 *
 * @param[in] frame Handle to the camera frame.
 * @param[in] serializer Specifies the sensor serializer handle.
 *
 * @return DW_NOT_AVAILABLE - serialization is not available at this moment, <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_SUCCESS
 *
 */
DW_API_PUBLIC
dwStatus
dwSensorSerializer_serializeCameraFrame(dwCameraFrameHandle_t const frame,
                                        dwSensorSerializerHandle_t const serializer);

/**
 * Pushes a camera frame to the serializer. This method is thread-safe.
 *
 * @param[in] frame Handle to the camera frame.
 * @param[in] serializer Specifies the sensor serializer handle.
 *
 * @return DW_NOT_AVAILABLE - serialization is not available at this moment, <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_SUCCESS
 *
 */
DW_API_PUBLIC
dwStatus dwSensorSerializer_serializeCameraFrameAsync(dwCameraFrameHandle_t const frame,
                                                      dwSensorSerializerHandle_t const serializer);

/**
 * Gets serializer statistics (if available).
 *
 * @param[out] outStats A pointer to a structure containing the statistics.
 * @param[in] serializer Handle to the serializer
 *
 * @return DW_INVALID_HANDLE - if provided serializer handle is invalid.<br>
 *         DW_NOT_SUPPORTED - if serializer does not implement this functionality<br>
 *         DW_SUCCESS - if call is successful.
 **/
DW_API_PUBLIC
dwStatus dwSensorSerializer_getStats(dwSerializerStats* const outStats,
                                     dwSensorSerializerHandle_t const serializer);
#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_SENSORS_SENSORSERIALIZER_H_

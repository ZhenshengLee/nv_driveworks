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

#ifndef DW_SENSORS_CODECS_SENSORSERIALIZER_SENSORSERIALIZER_H_
#define DW_SENSORS_CODECS_SENSORSERIALIZER_SENSORSERIALIZER_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>
#include <dw/core/context/Context.h>

#include <dw/sensors/Sensors.h>
#include <dw/sensors/camera/Camera.h>

#include <dw/sensors/CodecHeader.h>
#include <dw/sensors/containers/Container.h>

#include <dw/sensors/SensorSerializer.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Handle representing a sensor serializer. */
typedef struct dwSensorSerializerNewObject* dwSensorSerializerNewHandle_t;

/**
 * Create and initialize a sensor serializer with the parameters provided.
 * Output C handle of the sensor serializer created.
 *
 * @param[out] serializer A pointer to the sensor serializer handle
 * @param[in] codecHeader A pointer to the codec header for this serializer
 * @param[in] serializerConfig Sensor serializer config parameter
 * @param[in] context The DW context
 *
 * @retval DW_INVALID_HANDLE if provided inputs are invalid.
 * @retval DW_INVALID_ARGUMENT if any input parameters is NULL.
 * @retval DW_SUCCESS deal successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorSerializerNew_initialize(dwSensorSerializerNewHandle_t* const serializer,
                                          dwCodecHeaderHandle_t const codecHeader,
                                          dwSerializerParams const* const serializerConfig,
                                          dwContextHandle_t const context);

/**
 * Releases the resources of a sensor serializer.
 *
 * @param[in] serializer The sensor serializer handle.
 *
 * @retval DW_INVALID_HANDLE if provided inputs are invalid.
 * @retval DW_INVALID_ARGUMENT the input handle is NULL.
 * @retval DW_SUCCESS deal successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorSerializerNew_release(dwSensorSerializerNewHandle_t const serializer);

/**
 * Pushes data to the serializer and save it in the output assigned file after calling.
 *
 * @param[in] data A pointer to the byte array of data.
 * @param[in] size Specifies the size of the byte array.
 * @param[in] timestamp Specifies the timestamp for the data.
 * @param[in] serializer Specifies the handle to the sensor serializer.
 *
 * @retval DW_NOT_AVAILABLE serialization is not available at this moment. <br>
 * @retval DW_CALL_NOT_ALLOWED if calling this function in async mode or the media type of serializer is DW_MEDIA_TYPE_VIDEO.
 * @retval DW_SUCCESS deal successfully.
 * @retval DW_INVALID_ARGUMENT if the input parameters 'data'/'serializer' is NULL or size is zero.
 * @note Other return value will depend on the internal implementations of the serialization.
 */
DW_API_PUBLIC
dwStatus dwSensorSerializerNew_serializeData(uint8_t const* const data, size_t const size, dwTime_t const timestamp,
                                             dwSensorSerializerNewHandle_t const serializer);

/**
 * Pushes data to the serializer. This method is thread-safe and thus
 * can be used on the capture thread (or any other thread). Use this method
 * in conjunction with 'dwSensorSerializer_start'/'dwSensorSerializer_stop'.
 *
 * @param[in] data A pointer to the byte array of data.
 * @param[in] size Specifies the size of the byte array.
 * @param[in] timestamp Specifies the timestamp for the data.
 * @param[in] serializer Specifies the handle to the sensor serializer.
 *
 * @retval DW_NOT_AVAILABLE serialization is not available at this moment. <br>
 * @retval DW_BUFFER_FULL serializer buffer is full, data was not pushed to serializer. <br>
 * @retval DW_CALL_NOT_ALLOWED if calling this function in sync mode or the media type of serializer is DW_MEDIA_TYPE_VIDEO or dwSensorSerializer_start have not been called before calling this function.
 * @retval DW_INVALID_ARGUMENT if the input parameters 'data'/'serializer' is NULL or size is zero.
 * @retval DW_SUCCESS deal successfully.
 *
 * \ingroup sensors
 */
DW_API_PUBLIC
dwStatus dwSensorSerializerNew_serializeDataAsync(uint8_t const* const data, size_t const size, dwTime_t const timestamp,
                                                  dwSensorSerializerNewHandle_t const serializer);

/**
 * Pushes a camera frame to the serializer and save it in the output assigned file after calling.
 *
 * @param[in] frame Handle to the camera frame.
 * @param[in] serializer Specifies the sensor serializer handle.
 *
 * @retval DW_NOT_AVAILABLE serialization is not available at this moment.
 * @retval DW_INVALID_ARGUMENT if the input frame or serializer is NULL.
 * @retval DW_SUCCESS deal successfully.
 * @retval DW_INTERNAL_ERROR some internal error(eg: internal timestamp queue is empty).
 *
 */
DW_API_PUBLIC
dwStatus dwSensorSerializerNew_serializeCameraFrame(dwCameraFrameHandle_t const frame,
                                                    dwSensorSerializerNewHandle_t const serializer);

/**
 * Pushes a camera frame to the serializer and working thread will deal with it later.
 * This method is thread-safe.
 * And this can only be working in asyn mode. dwSensorSerializer_start should be
 * called before this funtion is called.
 *
 * @param[in] frame Handle to the camera frame.
 * @param[in] serializer Specifies the sensor serializer handle.
 *
 * @retval DW_CALL_NOT_ALLOWED if calling this function in sync mode or the media type of serializer is not DW_MEDIA_TYPE_VIDEO or dwSensorSerializer_start have not been called before calling this function.
 * @retval DW_NOT_AVAILABLE serialization is not available at this moment.
 * @retval DW_INVALID_ARGUMENT if the input frame or serializer is NULL.
 * @retval DW_SUCCESS deal successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorSerializerNew_serializeCameraFrameAsync(dwCameraFrameHandle_t const frame,
                                                         dwSensorSerializerNewHandle_t const serializer);

/**
 * Starts serialization of sensor. This method creates a new thread and
 * begins the serialization loop.
 *
 * @note To ensure you don't miss any data, activate serialization
 *       before calling dwSensorSerializer_startSensor().
 *
 * @param[in] serializer Specifies the sensor serializer handle.
 * @retval DW_CALL_NOT_ALLOWED if calling this function in sync mode or dwSensorSerializer_start have been called before.
 * @retval DW_SUCCESS deal successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorSerializerNew_start(dwSensorSerializerNewHandle_t const serializer);

/**
 * Stop serialization of sensor. This method stops the thread and
 * the serialization loop.
 *
 * @param[in] serializer Specifies the sensor serializer handle.
 *
 * @retval DW_CALL_NOT_ALLOWED dwSensorSerializer_start have not been called before.
 * @retval DW_SUCCESS deal successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorSerializerNew_stop(dwSensorSerializerNewHandle_t const serializer);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_SENSORS_CODECS_SENSORSERIALIZER_SENSORSERIALIZER_H_

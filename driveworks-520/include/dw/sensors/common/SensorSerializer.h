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
// SPDX-FileCopyrightText: Copyright (c) 2016-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_SENSORS_COMMON_SENSORSERIALIZER_H_
#define DW_SENSORS_COMMON_SENSORSERIALIZER_H_

#include "SensorSerializerTypes.h"
#include "Sensors.h"

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/context/Context.h>
#include <dw/sensors/camera/Camera.h>

#ifdef __cplusplus
extern "C" {
#endif

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
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
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
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
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
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
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
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
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
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
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
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
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
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
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
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
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
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
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
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
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
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
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
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 **/
DW_API_PUBLIC
dwStatus dwSensorSerializer_getStats(dwSerializerStats* const outStats,
                                     dwSensorSerializerHandle_t const serializer);

/**
 * Append the allocation attribute such that images allocated by the application and given to the serializer
 * This API is used to append the underlying driver's allocation attributes to the image properties.
 * @param[inout] imgProps Image properties
 * @param[in] serializer Handle to the serializer
 * 
 * @note The given imgProps should be compatible with serialzier
 * @note The imgProps are read and used to generate the allocation attributes
 *       needed by the driver. The allocation attributes are stored back into
 *       imgProps.meta.allocAttrs. Applications do not need to free or alter the
 *       imgProps.meta.allocAttrs in any way. The imgProps.meta.allocAttrs are only used
 *       by DriveWorks as needed when the given imgProps are used to allocate dwImages.
 *       If the application alters the imgProps after calling this API, the
 *       imgProps.meta.allocAttrs may no longer be applicable to the imgProps and calls related
 *       to allocating images will fail.
 * @note if imgProps.meta.allocAttrs does not have allocated Memory, this would be allocated by
 *       DW and will be owned by DW context until context is destroyed
 *       and should be used wisely as it the space is limited.
 * @note Must be called after dwSensorSerializer_initialize().
 *
 * @return DW_NVMEDIA_ERROR - if underlying driver had an NvMedia error. <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_NOT_IMPLEMENTED - if the method for this image type is not implemented by given camera. <br>
 *         DW_SUCCESS - if call is successful.
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwSensorSerializer_appendAllocationAttributes(dwImageProperties* const imgProps, dwSensorSerializerHandle_t const serializer);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_SENSORS_COMMON_SENSORSERIALIZER_H_

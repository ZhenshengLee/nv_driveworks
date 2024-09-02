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

#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/camera/Camera.h>

#include <dw/sensors/codecs/CodecHeader.h>
#include <dw/sensors/containers/Container.h>

#include <dw/sensors/common/SensorSerializer.h>

#ifdef __cplusplus
extern "C" {
#endif

//! Handle representing a sensor serializer.
typedef struct dwSensorSerializerNewObject* dwSensorSerializerNewHandle_t;

/**
 * @brief Create and initialize a sensor serializer based on the driveworks context parameter, CodecHeader parameter and dwSerializerParams parameter.
 * - Set the serializer working in sync-mode or async-mode.
 * - Create an output stream based on the file path where serialized data will be written to.
 * - Check the media type
 *   - video type. Check the codec type, only support h26x and raw format video data. Different codec types need create different internal muxer objects to serialize the video data. The output stream is bound to the muxer object.
 *                 The serializer creates an internal encoder object to encode the video data.
 *   - non-video type. Create a muxer object to serialize the data packet, the output stream is bound to the muxer object.
 *                     If the serializer works in async-mode, it creates internal buffer pool and queue to cache the data packet to be serialized asynchronously.
 * 
 * @param[out] serializer A pointer to the created sensor serializer handle.
 * @param[in] codecHeader CodecHeader handle. It determines the media type and codec type of the Serializer.
 * @param[in] serializerConfig A Parameter to Configure the Serializer, such as work mode, file path to save the serialized data. Refer to comments of dwSerializerParams to see more details.
 * @param[in] context The DW context
 *
 * @return DW_INVALID_HANDLE Only for video data, failed to allocate CPU wait context from @c context.
 * @return DW_INVALID_ARGUMENT If any input parameter is NULL or has invalid configuration.
 * @return DW_SUCCESS Initialize successfully, the Serializer is correctly configured.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwSensorSerializerNew_initialize(dwSensorSerializerNewHandle_t* const serializer,
                                          dwCodecHeaderHandle_t const codecHeader,
                                          dwSerializerParams const* const serializerConfig,
                                          dwContextHandle_t const context);

/**
 * @brief Release the sensor serializer.
 *        Stop the worker thread if the serializer works in async-mode.
 *        All packets cached in the serializer will be processed. The output stream is flushed and closed.
 *        Destroy the underlying Sensor Serializer object.
 *
 * @param[in] serializer sensor serializer handle.
 *
 * @return DW_INVALID_ARGUMENT If serializer handle is NULL.
 * @return DW_SUCCESS Release successfully. All packets that are successfully serialized are persisted to the storage. Allocated memory in the serializer object is returned to the allocator.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSensorSerializerNew_release(dwSensorSerializerNewHandle_t const serializer);

/**
 * @brief Serialize the data synchronously. Once this API returns successfully, the data is serialized into an output stream. After serialization, the serialized data is flushed to storage, guaranteeing that it is written and saved.
 *
 * @param[in] data First byte of the data packet to be serialized.
 * @param[in] size Total count of bytes of the data packet to be serialized.
 * @param[in] timestamp The timestamp of the data packet to be serialized, will be recorded into the serialized data.
 * @param[in] serializer Sensor serializer handle.
 *
 * @return DW_NOT_AVAILABLE Serialization is not available at this moment. <br>
 * @return DW_CALL_NOT_ALLOWED If calling this function in async mode or the media type of serializer is DW_MEDIA_TYPE_VIDEO.
 * @return DW_INVALID_ARGUMENT If the input parameters 'data'/'serializer' is NULL or size is zero.
 * @return DW_SUCCESS The data packet is successfully serialized into the output stream.
 *
 * @note Other return value will depend on the internal implementations of the serialization.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
// coverity[misra_c_2012_rule_5_1_violation] RFD Accepted: TID-1658
dwStatus dwSensorSerializerNew_serializeData(uint8_t const* const data, size_t const size, dwTime_t const timestamp,
                                             dwSensorSerializerNewHandle_t const serializer);

/**
 * @brief Serialize the data asynchronously. Once this API returns successfully, the data packet is added to an internal buffer or queue. An internal worker thread periodically serializes the packets from this buffer or queue. Finally, the serialized data is flushed into storage, ensuring it is written and saved. No additional action or option is required for this particular data packet.
 *
 * @param[in] data First byte of the data packet to be serialized.
 * @param[in] size Total count of bytes of the data packet to be serialized.
 * @param[in] timestamp The timestamp of the data packet to be serialized, will be recorded into the serialized data.
 * @param[in] serializer Sensor serializer handle.
 *
 * @return DW_NOT_AVAILABLE serialization is not available at this moment. <br>
 * @return DW_BUFFER_FULL serializer buffer is full, data was not pushed to serializer. <br>
 * @return DW_CALL_NOT_ALLOWED if calling this function in sync mode or the media type of serializer is DW_MEDIA_TYPE_VIDEO or dwSensorSerializer_start have not been called before calling this function.
 * @return DW_INVALID_ARGUMENT if the input parameters 'data'/'serializer' is NULL or size is zero.
 * @return DW_SUCCESS the data packet is successfully placed into the internal buffer or queue, there is no guarantee that it will be written into the file finally.
 *
 * \ingroup sensors
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
// coverity[misra_c_2012_rule_5_1_violation] RFD Accepted: TID-1658
dwStatus dwSensorSerializerNew_serializeDataAsync(uint8_t const* const data, size_t const size, dwTime_t const timestamp,
                                                  dwSensorSerializerNewHandle_t const serializer);

/**
 * @brief Serialize the camera frame synchronously. Once this API returns successfully, the camera frame is serialized into an output stream. After serialization, the serialized data is flushed to storage, guaranteeing that it is written and saved.
 *
 * @param[in] frame The camera frame data to be serialized.
 * @param[in] serializer Sensor serializer handle.
 *
 * @return DW_NOT_AVAILABLE serialization is not available at this moment.
 * @return DW_INVALID_ARGUMENT if the input frame or serializer is NULL.
 * @return DW_INTERNAL_ERROR some internal error(eg: internal timestamp queue is empty).
 * @return DW_SUCCESS the camera frame is successfully serialized into the output stream.
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSensorSerializerNew_serializeCameraFrame(dwCameraFrameHandle_t const frame,
                                                    dwSensorSerializerNewHandle_t const serializer);

/**
 * @brief Serialize the camera frame asynchronously. Once this API returns successfully, the camera frame packet is added to an internal buffer or queue. An internal worker thread periodically serializes the packets from this buffer or queue. Finally, the serialized data is flushed into storage, ensuring it is written and saved. No additional action or option is required for this particular camera frame packet.
 *
 * @param[in] frame The camera frame data to be serialized.
 * @param[in] serializer Sensor serializer handle.
 *
 * @return DW_CALL_NOT_ALLOWED if calling this function in sync mode or the media type of serializer is not DW_MEDIA_TYPE_VIDEO or dwSensorSerializer_start have not been called before calling this function.
 * @return DW_NOT_AVAILABLE serialization is not available at this moment.
 * @return DW_INVALID_ARGUMENT if the input frame or serializer is NULL.
 * @return DW_SUCCESS the camera frame is successfully placed into the internal buffer or queue, there is no guarantee that it will be written into the file finally.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
// coverity[misra_c_2012_rule_5_1_violation] RFD Accepted: TID-1658
dwStatus dwSensorSerializerNew_serializeCameraFrameAsync(dwCameraFrameHandle_t const frame,
                                                         dwSensorSerializerNewHandle_t const serializer);

/**
 *
 * @brief Create and launch a worker thread to do serialization task.
 *        This action is applicable only when the Serializer is operating in async-mode.
 *
 * @param[in] serializer Sensor serializer handle.
 *
 * @return DW_CALL_NOT_ALLOWED if calling this function in sync-mode or @ref dwSensorSerializer_start has been called before.
 * @return DW_SUCCESS dedicated worker thread is created and launched successfully.
 *
 * @note To ensure that no data is missed, it is recommended to call this API before feeding any data packet to the Serializer.
 * 
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 *
 */
DW_API_PUBLIC
dwStatus dwSensorSerializerNew_start(dwSensorSerializerNewHandle_t const serializer);

/**
 *
 * @brief Stop the internal worker thread. all packets cached in the serializer are processed before the dedicated thread is halted.
 *        This action is applicable only when the Serializer is operating in async-mode.
 *
 * @param[in] serializer Sensor serializer handle.
 *
 * @return DW_CALL_NOT_ALLOWED dwSensorSerializer_start have not been called before.
 * @return DW_SUCCESS All packets that are cached in the serializer are successfully serialized. However, an additional flush or sync operation is required to ensure that these serialized packets are flushed and written into the storage. 
 * 
 * @note To ensure that no data is missed, avoid feeding any data packet to the Serializer after invoking this API.
 * 
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSensorSerializerNew_stop(dwSensorSerializerNewHandle_t const serializer);

/**
 * @brief The internal encoder's allocation attributes are appended to the image properties @c imgProps of the allocated images. These images can be set as the image pool for camera frames. By doing so, the camera frames fed to the serializer can be properly encoded by the internal encoder, utilizing the allocated images with the appropriate attributes.
 *
 * @param[inout] imgProps Image properties to have allocation attributes appended from the serializer(internal encoder).
 * @param[in] serializer Sensor serializer handle.
 *
 * @return DW_NVMEDIA_ERROR - If underlying driver had an NvMedia error. <br>
 *         DW_INVALID_HANDLE - If given handle is not valid. <br>
 *         DW_NOT_IMPLEMENTED - If the method for this image type is not implemented by given camera. <br>
 *         DW_SUCCESS - The appending operation is successful.
 *
 * @note The given imgProps should be compatible with the serializer(internal encoder).
 * @note The imgProps are read and used to generate the allocation attributes
 *       needed by the driver. The allocation attributes are stored back into
 *       imgProps.meta.allocAttrs. Applications do not need to free or alter the
 *       imgProps.meta.allocAttrs in any way. The imgProps.meta.allocAttrs are only used
 *       by DriveWorks as needed when the given imgProps are used to allocate dwImages.
 *       If the application alters the imgProps after calling this API, the
 *       imgProps.meta.allocAttrs may no longer be applicable to the imgProps and calls related
 *       to allocating images will fail.
 * @note If imgProps.meta.allocAttrs does not have allocated memory, this would be allocated by
 *       DW and will be owned by DW context until context is destroyed
 *       and should be used wisely as the space is limited.
 * @note Must be called after @ref dwSensorSerializer_initialize and only for serializing video data.
 * 
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 *
 */
DW_API_PUBLIC
dwStatus dwSensorSerializerNew_appendAllocationAttributes(dwImageProperties* const imgProps, dwSensorSerializerNewHandle_t const serializer);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_SENSORS_CODECS_SENSORSERIALIZER_SENSORSERIALIZER_H_

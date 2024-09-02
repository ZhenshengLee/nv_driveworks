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

#include "IMUTypes.h"

#include <dw/core/base/Config.h>
#include <dw/sensors/common/Sensors.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Reads the next IMU frame from the sensor within a given timeout.
 *
 * This API function will achieve two functionalities of this software unit:
 * - Read an IMU sensor packet from physical IMU sensor
 * - Decode this IMU sensor packet and add cache decoded IMU frames
 *
 * This API function will firstly check if there are already decoded IMU frames available, if there are decoded IMU frames then this API function
 * returns the first decoded IMU frame then return; otherwise it will read IMU sensor packet then decode it and return first decoded IMU frames, or
 * return DW_TIME_OUT if no IMU frames could be decoded till timeout is detected.
 *
 * This function could only be called after an IMU sensor is instantiated, started and is running.
 *
 * @param[out] frame A pointer to an IMU frame structure to be filled with new data.
 * @param[in] timeoutUs Timeout, in us, to wait for a new message. Range: [0, INT_MAX] <br>
 *                      Special values: <br>
 *                      - DW_TIMEOUT_INFINITE: to wait infinitely.
 *                      - Zero: means polling mode - if there are already decoded IMU frames, return first frame immediately; otherwise return DW_TIME_OUT directly.
 * @param[in] sensor Sensor handle of the IMU sensor previously created with dwSAL_createSensor().
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid. Only handles returned by dwSAL_createSensor with IMU sensor protocols are allowed. <br>
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid. Here it means frame pointer is empty <br>
 *         DW_TIME_OUT - If no valid IMU frames are decoded within specified timeout @b timeoutUs (note that @b timeoutUs may be zero) <br>
 *         DW_NOT_AVAILABLE    - if sensor has not been started or raw IMU sensor packet is not available in polling mode. <br>
 *         DW_END_OF_STREAM    - if end of stream reached (virtual sensor only). <br>
 *         DW_SUCCESS - successfully returned a decoded IMU frame in @b frame within specified timeout @b timeoutUs from specified sensor @b sensor
 * @note This function will block until a frame is decoded or timeout is detected, if @b timeoutUs is specified as non-zero values.
 * @note If no valid IMU packets could be returned from physical IMU sensors(for Vehicle IMU sensors, this means either all received IMU packets have wrong packet size
 *       or no IMU packets are received at all) till timeout, no IMU frames could be decoded and this API will return DW_TIME_OUT.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwSensorIMU_readFrame(dwIMUFrame* const frame, dwTime_t const timeoutUs, dwSensorHandle_t const sensor);

/**
 * @brief Decodes IMU frames from received IMU sensor raw data/packets.
 * Any processed messages can be picked up using dwSensorIMU_readFrame() or dwSensorIMU_popFrame() function.
 *
 * This API function will achieve functionality of this software unit:
 * - Decode IMU sensor packets into IMU frames and cache decoded IMU frames
 *
 * This API function will push specified IMU sensor packet into its internal IMU decoders. Since internal IMU decoder will return once an IMU frame is decoded, or no IMU frames
 * could be decoded from remaining bytes from specified IMU sensor packet, thus this API function will keep driving the decoder to return as many IMU frames as possible, until
 * decoder could return no new IMU frames. All decoded IMU frames are cached internally and could be accessed by dwSensorIMU_readFrame() or dwSensorIMU_popFrame().
 * Based on sensor configuration value 'output-timestamp=', decoded IMU frames would be modified in their hostTimestamp field values accordingly.
 *
 * This function should only be called after sensor is started, and with raw IMU sensor packet(s) returned from dwSensor_readRawData. After
 * this raw sensor packet is processed by this function, user should also call dwSensor_returnRawData to return the memory block to SAL.
 *
 * This function is a synchronous function. Customers have to wait till the raw IMU sensor data is processed.
 *
 * @param[in] data Undecoded imu data. Non-null pointer
 * @param[in] size Size in bytes of the raw data. Range: (0, INT_MAX]. Note 0 will result in DW_NOT_READY return value.
 * @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid. Only handles returned by dwSAL_createSensor with IMU sensor protocols are allowed. <br>
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid, Here it means input pointer 'data' is empty <br>
 *         DW_NOT_READY - if more data needs to be passed in (loop while it returns 'DW_NOT_READY'). <br>
 *         DW_SUCCESS - specified raw IMU sensor packet was decoded into one or more IMU frames. Please read decoded IMU frames via @ref dwSensorIMU_readFrame or @ref dwSensorIMU_popFrame
 * @note It's possible that raw IMU data block @b data will be decoded into zero, one or more IMU frames. All these decoded IMU frames will be stored into internal IMU frame queue and
 *       could be accessed via @ref dwSensorIMU_readFrame() or @ref dwSensorIMU_popFrame().
 *       Scenarios that no IMU frames could be decoded: The IMU packet itself is invalid(for example, an IMU packet with incorrect packet size)
 *       Scenarios that one IMU frame could be decoded: The IMU packet itself is valid(for example, an IMU packets with expected size and could be recognized and parsed successfully by decoder)
 *       Scenarios that more than one IMU frames could be decoded: IMU packet contains multiple sensor messages(for example, an IMU packet that contains multiple IMU messages that are defined by IMU vendors, and each such IMU message could lead to a separate IMU frame to be decoded).
 * @note It's also possible that part of a raw IMU data block @b data could not be recognized and decoded by IMU sensor's internal decoder, because these
 *       decoders are usually designed to recognize and parse a selected part of available types of raw sensor packets. This function returns DW_SUCCESS
 *       if part of this raw IMU sensor packet is decoded into one or more IMU frames.
 * @note If internal cache of IMU frames is full, decoded IMU frames won't be cached anymore and they'll be discarded. In this case a warning message 'IMU: sensor generates data faster, then reader 
 *       consumes. Losing messages' is printed, but API function won't return any error codes in this case - users will still see DW_SUCCESS if at least one decoded IMU frame(s) are cached.
 *       This could happen if customers called this API for too many times w/o calling @ref dwSensorIMU_readFrame or @ref dwSensorIMU_popFrame to retrieve decoded IMU frames which is a typical buffer overflow scenario.
 * @sa https://developer.nvidia.com/docs/drive/drive-os/6.0.8.1/public/driveworks-nvsdk/nvsdk_dw_html/sensors_usecase4.html for description of available option values for 'output-timestamp=' sensor parameter.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwSensorIMU_processRawData(uint8_t const* const data, size_t const size, dwSensorHandle_t const sensor);

/**
 * @brief Returns the earliest IMU frame previously processed from previous IMU sensor packets.
 *
 * This function returns only the first single decoded IMU frame from cached IMU frames(and removes it from cache). If no frames are available in queue,
 * DW_NOT_AVAILABLE is returned and no IMU frame is returned.
 *
 * This function should only be called after sensor is started. From above descriptions, this function requires previous callings to function
 * @ref dwSensorIMU_readFrame or @ref dwSensorIMU_processRawData returns success otherwise this function will return DW_NOT_AVAILABLE.
 *
 * This function is a synchronous function. Customers have to wait till the polling of IMU frame queue is finished.
 *
 * @param[out] frame Pointer to an IMU frame structure to be filled with new data.
 * @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid. Only handles returned by dwSAL_createSensor with IMU sensor protocols are allowed. <br>
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid. Here it means input pointer 'frame' is empty<br>
 *         DW_NOT_AVAILABLE - if no more data is available <br>
 *         DW_SUCCESS - An IMU frame is dequeued from internal IMU frame queue into @b frame
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
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
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
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
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
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
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
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

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
 * <b>NVIDIA DriveWorks API: CAN</b>
 *
 * @b Description: This file defines the CAN sensor methods.
 */

/**
 * @defgroup can_group CAN Sensor
 * @ingroup sensors_group
 *
 * @brief Defines the CAN sensor methods.
 *
 * @{
 */

#ifndef DW_SENSORS_CANBUS_CAN_H_
#define DW_SENSORS_CANBUS_CAN_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>

#include <dw/sensors/canbus/CANTypes.h>
#include <dw/sensors/common/Sensors.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Enables or disables hardware timestamp of the CAN messages. Hardware timestamps are used per default when
 * supported by the sensor. If HW timestamps are not supported, SW timestamps are used per default and no operation shall be performed.
 * HW timestamps can be turned off if the default behavior is not working properly with the current hardware stack.
 * If any result different from DW_SUCCESS is returned, no operation shall be performed by this method.
 *
 * @note The effect takes place on the next (re)start of the sensor.
 *
 * @param[in] flag Specifies either 'true' or 'false' to enable or disable hardware timestamping.
 * @param[in] sensor Specifies the sensor handle of the CAN sensor previously created with 'dwSAL_createSensor()'.
 *
 * @attention If the same physical CAN device/interface is shared by multiple sensors,
 *            setting HW timestamping affects timestamping of all sensors sharing the same device.
 *
 * @note If using AurixCAN as a driver, hardware timestamps are used as timestamped by Aurix.
 *
 * @return DW_NOT_SUPPORTED - if the underlying sensor does not support hardware timestamps. <br>
 *         DW_INVALID_HANDLE - if given sensor handle is invalid <br>
 *         DW_SUCCESS - if operation succeeds.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSensorCAN_setUseHwTimestamps(bool const flag, dwSensorHandle_t const sensor);

/**
 * Specifes a set of CAN IDs to be filtered.
 * The filter is active if it contains at least one sender ID.
 *
 * @param[in] ids A pointer to an array of CAN IDs to be filtered. Any matching CAN ID is used together
 *                with the mask to filter.
 * @param[in] masks A pointer to an array of filter masks to be applied. A mask is applied together with the
 *                 ID as 'id & mask'. If mask is set to NULL, a default mask of '1FFFFFFF' is used.
 * @param[in] num Specifies the number of elements passed in the array. To remove the filter, pass 0.
 * @param[in] sensor Specifies the sensor handle of the CAN sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_NOT_SUPPORTED - if the underlying sensor does not support filter operation. <br>
 *         DW_INVALID_HANDLE - if given sensor handle is invalid.<br>
 *         DW_INVALID_ARGUMENT - if ids or masks pointers are null.<br>
 *         DW_SAL_SENSOR_ERROR - if there was a sensor error, i.e., filter cannot be set. <br>
 *         DW_SUCCESS - if operation succeeds.
 *
 * @note Some CAN implementations, such as AurixCAN, only support a few messages for a filter.
 * @note The effect takes place on the next (re)start of the sensor.
 * @note The general implementation is to pass a message if 'received_can_id & mask == can_id & mask'.
 *       If multiple filters are provided, a message is passed if at least one of the filters match.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwSensorCAN_setMessageFilter(const uint32_t* ids, const uint32_t* masks,
                                      uint16_t num, dwSensorHandle_t sensor);

/**
 * Reads a CAN packet (raw data and process it) within a given timeout from the CAN bus.
 * The method blocks until either a new valid message is received on the bus or the given timeout exceeds.
 *
 * @param[out] msg A pointer to a CAN message structure to be filled with new data.
 * @param[in] timeoutUs Specifies a timeout, in us, to wait for a new message. Special values: DW_TIMEOUT_INFINITE - to wait infinitly.  Zero - means polling of internal queue.
 * @param[in] sensor Specifies a sensor handle of the CAN sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_SUCCESS - A new message has been successfully read<br>
 *         DW_INVALID_HANDLE - if given sensor handle is invalid.<br>
 *         DW_INVALID_ARGUMENT - if msg pointer is null.<br>
 *         DW_TIME_OUT - If a timeout occurred during read of a new raw CAN packet from the CAN bus<br>
 *         DW_END_OF_STREAM - End of raw data stream reached<br>
 *         DW_FILE_INVALID - Raw data stream content is invalid<br>
 *         DW_SAL_SENSOR_ERROR - I/O bus error<br>
 *         DW_NOT_AVAILABLE - if the sensor has not been initialized and started.<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSensorCAN_readMessage(dwCANMessage* const msg, dwTime_t const timeoutUs, dwSensorHandle_t const sensor);

/**
 * Sends a message over the CAN bus within a specified timeout. A message is guaranteed
 * to have been sent when this method returns 'DW_SUCCESS'. The method can block
 * for up-to the specified amount of time if a bus is blocked or previous dwSensorCAN_readMessage() operation has not yet finished.
 * The way the message is packed and sent to the sensor depends on the specific CAN class implementation used.
 *
 * @param[in] msg A pointer to the message to be sent.
 * @param[in] timeoutUs Specifies the timeout, in microseconds from system epoch, to wait at most before giving up.
 * @param[in] sensor Specifies a CAN bus sensor to send the message over.
 *
 * @return DW_SUCCESS - The message has been sent successfully on the CAN bus<br>
 *         DW_NOT_IMPLEMENTED - if the concrecte CAN class used does not implement the send operation<br>
 *         DW_INVALID_HANDLE - if given sensor handle is invalid<br>
 *         DW_INVALID_ARGUMENT - if msg pointer is null<br>
 *         DW_TIME_OUT - If a timeout occurred while sending the message<br>
 *         DW_SAL_SENSOR_ERROR - I/O bus error<br>
 *         DW_NOT_AVAILABLE - if the sensor has not been initialized and started<br>
 *
 * @note Send operation using can.virtual is a no-op and returns always DW_SUCCESS
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSensorCAN_sendMessage(const dwCANMessage* const msg,
                                 dwTime_t const timeoutUs, dwSensorHandle_t const sensor);

/**
* Decodes CAN data previously read as a RAW data stream into internal queue.
*
* @param[in] data Undecoded CAN data.
* @param[in] size Size in bytes of the raw data.
* @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
*
* @return DW_INVALID_HANDLE - if given sensor handle is invalid. <br>
*         DW_INVALID_ARGUMENT - if data pointer is null. <br>
*         DW_FAILURE - if given raw stream data is not valid <br>
*         DW_NOT_READY - if given raw stream did not have any data at all <br>
*         DW_SUCCESS - if operation succeeds.
*
* @see dwSensorCAN_popMessage()
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwSensorCAN_processRawData(const uint8_t* data, size_t size,
                                    dwSensorHandle_t sensor);

/**
* Returns any CAN data previously processed through a RAW data stream.
* This happens on the CPU thread where the function is called, incurring on additional load on that thread.
*
* @param[out] msg Container for the decoded message.
* @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
*
* @return DW_INVALID_HANDLE - if given sensor handle is invalid. <br>
*         DW_INVALID_ARGUMENT - if msg pointer is null.<br>
*         DW_NOT_AVAILABLE - if no more data is available <br>
*         DW_SUCCESS - if operation succeeds.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwSensorCAN_popMessage(dwCANMessage* msg,
                                dwSensorHandle_t sensor);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_SENSORS_CANBUS_CAN_H_

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

#include <dw/sensors/Sensors.h>

// clang-format off

/**
*
* This module provides access to the CAN bus typically used for communication
* between different ECUs in a vehicle. The implementation provides an abstracted layer over any
* implemented sensor drivers, supporting receive, send and filtering routines.
*
* CAN data packet as of ISO 11898-1:
* ~~~~~~~~~~~~~~~
* Byte:|           0           |         1             |           2           |           3           |
* Bit: |07-06-05-04-03-02-01-00|15-14-13-12-11-10-09-08|23-22-21-20-19-18-17-16|31-30-29-28-27-26-25-24|
*
* Byte:|           4           |           5           |           6           |           7           |
* Bit: |39-38-37-36-35-34-33-32|47-46-45-44-43-42-41-40|55-54-53-52-51-50-49-48|63-62-61-60-59-58-57-56|
* ~~~~~~~~~~~~~~~
*
* @note Currently supported frame message format is ISO 11898-1.
*
* #### Testing (Linux only) ####
*
* If you have a real CAN device, activate it with this command: <br>
* ~~~~~~~~~~~~~~~
* :> sudo ip link set can0 up type can bitrate 500000
* ~~~~~~~~~~~~~~~
*
* A virtual device is created using following commands:
* ~~~~~~~~~~~~~~~
* :> sudo modprobe vcan
* :> sudo ip link add dev vcan0 type vcan
* :> sudo ip link set up vcan0
* ~~~~~~~~~~~~~~~
* In order to send data from console to the virtual CAN bus, the cansend tool (from the
* can-utils package) can be used.
* ~~~~~~~~~~~~~~~
* :> cansend vcan0 30B#1122334455667788
* ~~~~~~~~~~~~~~~
*/
// clang-format on

/// Maximal length of the supported CAN message id [bits].
#define DW_SENSORS_CAN_MAX_ID_LEN 29

/// Maximal length of the supported CAN payload [bytes].
#define DW_SENSORS_CAN_MAX_MESSAGE_LEN 64

/// Maximum number of filter that can be specified
#define DW_SENSORS_CAN_MAX_FILTERS 255

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(push, 1) // Makes sure you have consistent structure packings.

/// Holds a CAN package.
typedef struct dwCANMessage
{
    /// Timestamp of the message in microseconds (using clock of the context).
    dwTime_t timestamp_us;

    /// CAN ID of the message sender.
    uint32_t id;

    /// Number of bytes of the payload.
    uint16_t size;

    /// Payload.
    uint8_t data[DW_SENSORS_CAN_MAX_MESSAGE_LEN];
} dwCANMessage;

#pragma pack(pop)

/**
 * Enables or disables hardware timestamp of the CAN messages. Hardware timestamps are used per default when
 * supported by the sensor. If HW timestamps are not supported, SW timestamps are used per default.
 * HW timestamps can be turned off if the default behavior is not working properly with the current hardware stack.
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
 *         DW_INVALID_ARGUMENT - if given arguments are invalid <br>
 *         DW_SUCCESS
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
 *         DW_INVALID_HANDLE - if given sensor handle is invalid. <br>
 *         DW_INVALID_ARGUMENT - if given arguments are invalid. <br>
 *         DW_SAL_SENSOR_ERROR - if there was a sensor error, i.e., filter cannot be set. <br>
 *         DW_SUCCESS
 *
 * @note Some CAN implementations, such as AurixCAN, only support a few messages for a filter.
 * @note The effect takes place on the next (re)start of the sensor.
 * @note The general implementation is to pass a message if 'received_can_id & mask == can_id & mask'.
 *       If multiple filters are provided, a message is passed if at least one of the filters match.
**/
DW_API_PUBLIC
dwStatus dwSensorCAN_setMessageFilter(const uint32_t* ids, const uint32_t* masks,
                                      uint16_t num, dwSensorHandle_t sensor);

/**
 * Reads a CAN packet with a given timeout from the CAN bus. The method blocks until
 * either a new valid message is received on the bus or the given timeout exceeds.
 *
 * @param[out] msg A pointer to a CAN message structure to be filled with new data.
 * @param[in] timeoutUs Specifies a timeout, in us, to wait for a new message. Special values: DW_TIMEOUT_INFINITE - to wait infinitly.  Zero - means polling of internal queue.
 * @param[in] sensor Specifies a sensor handle of the CAN sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid. <br>
 *         DW_INVALID_ARGUMENT - if given arguments are invalid. <br>
 *         DW_TIME_OUT - if operation has timeout. <br>
 *         DW_NOT_AVAILABLE - if sensor has not been started or data is not available in polling mode. <br>
 *         DW_SAL_SENSOR_ERROR - if there was an i/o or bus error. <br>
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwSensorCAN_readMessage(dwCANMessage* const msg, dwTime_t const timeoutUs, dwSensorHandle_t const sensor);

/**
 * Sends a message over the CAN bus within a specified timeout. A message is guaranteed
 * to have been sent when this method returns 'DW_SUCCESS'. The method can block
 * for up-to the specified amount of time if a bus is blocked or previous @see dwSensorCAN_readMessage()
 * operation has not yet finished.
 *
 * @param[in] msg A pointer to the message to be sent.
 * @param[in] timeoutUs Specifies the timeout, in microseconds, to wait at most before giving up.
 * @param[in] sensor Specifies a CAN bus sensor to send the message over.
 *
 * @return DW_NOT_SUPPORTED - if the underlying sensor does not support send operation. <br>
 *         DW_INVALID_HANDLE - if given sensor handle is invalid. <br>
 *         DW_INVALID_ARGUMENT - if given arguments are invalid. <br>
 *         DW_TIME_OUT - if operation has timed out. <br>
 *         DW_SAL_SENSOR_ERROR - if there was an i/o or bus error. <br>
 *         DW_NOT_AVAILABLE - if sensor has not been started. <br>
 *         DW_SUCCESS
 *
 * @note Send operation using can.virtual is a no-op and returns always DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwSensorCAN_sendMessage(const dwCANMessage* const msg,
                                 dwTime_t const timeoutUs, dwSensorHandle_t const sensor);

/**
* Decodes CAN data previously read as a RAW data stream into internal queue.
* Any processed messages can be picked up using the dwSensorCAN_popMessage() method.
*
* @param[in] data Undecoded CAN data.
* @param[in] size Size in bytes of the raw data.
* @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
*
* @return DW_INVALID_HANDLE - if given sensor handle is invalid. <br>
*         DW_INVALID_ARGUMENT - if given arguments are invalid. <br>
*         DW_FAILURE - if given raw stream data is not valid <br>
*         DW_NOT_READY - if given raw stream did not have any data at all <br>
*         DW_SUCCESS
*
* @see dwSensorCAN_popMessage()
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
*         DW_INVALID_ARGUMENT - if given arguments are invalid. <br>
*         DW_NOT_AVAILABLE - if no more data is available <br>
*         DW_SUCCESS
**/
DW_API_PUBLIC
dwStatus dwSensorCAN_popMessage(dwCANMessage* msg,
                                dwSensorHandle_t sensor);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_SENSORS_CANBUS_CAN_H_

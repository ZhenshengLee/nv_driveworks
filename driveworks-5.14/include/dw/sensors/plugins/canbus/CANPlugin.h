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
// SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks: CAN Sensor Plugin Interface</b>
 *
 * @b Description: This file defines the interfaces to be implemented for CAN sensor plugins.
 */

#ifndef DW_SENSORS_CAN_PLUGIN_H
#define DW_SENSORS_CAN_PLUGIN_H

#include <dw/sensors/plugins/SensorCommonPlugin.h>
#include <dw/sensors/canbus/CAN.h>

/**
 * @defgroup sensor_plugins_ext_canbus_group CAN Plugin
 * Provides an interface for non-standard CAN sensors.
 * @ingroup sensor_plugins_ext_group
 *
 * @{
 */

#ifdef __cplusplus
extern "C" {
#endif

//readRawData should return always a dwCANMessage

/**
 * Reset the filter set by 'dwSensorCANPlugin_setFilter' interface.
 *
 * @param[in] sensor Specifies the sensor.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorCANPlugin_clearFilter)(dwSensorPluginSensorHandle_t sensor);

/**
 * Specifes a set of CAN IDs to be filtered.
 * The filter is active if it contains at least one sender ID.
 *
 * @param[in] canIDs A pointer to an array of CAN IDs to be filtered. Any matching CAN ID is used together
 *                with the mask to filter.
 * @param[in] masks A pointer to an array of filter masks to be applied. A mask is applied together with the
 *                 ID as 'id & mask'. If mask is set to NULL, a default mask of '1FFFFFFF' is used.
 * @param[in] numCanIDs Specifies the number of elements passed in the array. To remove the filter, pass 0.
 * @param[in] sensor Specifies the sensor handle.
 *
 * @return DW_NOT_SUPPORTED - if the underlying sensor does not support filter operation. <br>
 *         DW_INVALID_HANDLE - if given sensor handle is invalid. <br>
 *         DW_INVALID_ARGUMENT - if given arguments are invalid. <br>
 *         DW_SAL_SENSOR_ERROR - if there was a sensor error, i.e., filter cannot be set. <br>
 *         DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorCANPlugin_setFilter)(const uint32_t* canIDs, const uint32_t* masks, uint16_t numCanIDs, dwSensorPluginSensorHandle_t sensor);

/**
 * Enables or disables hardware timestamp of the CAN messages. Hardware timestamps are used per default when
 * supported by the sensor. If HW timestamps are not supported, SW timestamps are used per default.
 * HW timestamps can be turned off if the default behavior is not working properly with the current hardware stack.
 *
 * @note The effect takes place on the next (re)start of the sensor.
 *
 * @param[in] use Specifies either 'true' or 'false' to enable or disable hardware timestamping.
 * @param[in] sensor Specifies the sensor handle.
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
typedef dwStatus (*dwSensorCANPlugin_setUseHwTimestamps)(bool use, dwSensorPluginSensorHandle_t sensor);

/**
 * Sends a message over the CAN bus within a specified timeout. A message is guaranteed
 * to have been sent when this method returns 'DW_SUCCESS'. The method can block
 * for up-to the specified amount of time if a bus is blocked or previous @see dwSensorCAN_readMessage()
 * operation has not yet finished.
 *
 * @param[in] msg A pointer to the message to be sent.
 * @param[in] timeout_us Specifies the timeout, in microseconds, to wait at most before giving up.
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
typedef dwStatus (*dwSensorCANPlugin_send)(const dwCANMessage* msg, dwTime_t timeout_us, dwSensorPluginSensorHandle_t sensor);

/**
 * Processes the data previously passed via the 'dwSensorPlugin_pushData' interface.
 *
 * The interpreted memory buffer outputted from this API is owned by the plugin.
 * The plugin shall support multiple buffers in flight via this API.
 *
 * @param[out] output Pointer to decoded CAN message
 * @param[in] sensor Specifies the sensor the data came from.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_NOT_AVAILABLE - if no frame is ready for consumption
 *         DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorCANPlugin_parseDataBuffer)(dwCANMessage* output,
                                                      dwSensorPluginSensorHandle_t sensor);

/// Function Table exposing CAN plugin functions
typedef struct
{
    dwSensorCommonPluginFunctions common;
    dwSensorCANPlugin_clearFilter clearFilter;
    dwSensorCANPlugin_setFilter setFilter;
    dwSensorCANPlugin_setUseHwTimestamps setUseHwTimestamps;
    dwSensorCANPlugin_send send;
    dwSensorCANPlugin_parseDataBuffer parseDataBuffer;
} dwSensorCANPluginFunctionTable;

/**
 * Gets the handle to functions defined in 'dwSensorCANPluginFunctionTable' structure.
 *
 * @param[out] functions A pointer to the function table
 *
 * @return DW_INVALID_ARGUMENT - if pointer to the function table is NULL. <br>
 *         DW_SUCCESS
 *
 */
dwStatus dwSensorCANPlugin_getFunctionTable(dwSensorCANPluginFunctionTable* functions);

/** @} */

#ifdef __cplusplus
}
#endif

#endif

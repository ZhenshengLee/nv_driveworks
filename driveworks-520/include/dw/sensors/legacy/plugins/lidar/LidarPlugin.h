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
// SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks: Lidar Sensor Plugin Interface</b>
 *
 * @b Description: This file defines the interfaces to be implemented for Lidar sensor plugins.
 */

#ifndef DW_SENSORS_LIDAR_PLUGIN_H
#define DW_SENSORS_LIDAR_PLUGIN_H

#include <dw/sensors/legacy/plugins/SensorCommonPlugin.h>
#include <dw/sensors/lidar/LidarTypes.h>

/**
 * @defgroup sensor_plugins_ext_lidar_group Lidar Plugin
 * Provides an interface for non-standard Lidar sensors.
 * @ingroup sensor_plugins_ext_group
 *
 * @{
 */

#ifdef __cplusplus
extern "C" {
#endif

/// Holds constants for a given lidar.
typedef struct
{
    /// Lidar properties.
    dwLidarProperties properties;

    /// Packet max payload size, in bytes.
    size_t maxPayloadSize;
} _dwSensorLidarDecoder_constants;

/**
 * Processes the data previously passed via the 'dwSensorPlugin_pushData' interface.
 *
 * The interpreted memory buffer outputted from this API is owned by the plugin.
 * The plugin shall support multiple buffers in flight via this API.
 *
 * @param[out] output Pointer to decoded lidar point cloud
 * @param[in] hostTimeStamp Specifies the host timeStamp when raw data was received.
 * @param[in] sensor Specifies the sensor the data came from.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_NOT_AVAILABLE - if no frame is ready for consumption
 *         DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorLidarPlugin_parseDataBuffer)(
    dwLidarDecodedPacket* output, const dwTime_t hostTimeStamp,
    dwSensorPluginSensorHandle_t sensor);

/**
 * Gets constants associated with this lidar sensor.
 *
 * @param[out] constants Constant parameters for this sensor are written here.
 * @param[in] sensor Specifies the sensor to which the constants are related.
 *
 * @return DW_FAILURE, DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorLidarPlugin_getDecoderConstants)(_dwSensorLidarDecoder_constants* constants, dwSensorPluginSensorHandle_t sensor);

/**
 * Send message to lidar sensor.
 *
 * @param[in] cmd Identification of the message.
 * @param[in] data Raw payload of the message.
 * @param[in] size Raw payload size of the message.
 * @param[in] sensor Specifies the sensor to send the message data.
 *
 * @return DW_INVALID_ARGUMENT - if the `data` is invalid<br>
 *         DW_NOT_SUPPORTED - if the sensor handle is NULL or invalid, or specified sensor
 *                            doesn't support `cmd` or the output connection is not set<br>
 *         DW_FAILURE - if not able to send data out successfully<br>
 *         DW_SUCCESS - if send the message out to the sensor successfully
 */
typedef dwStatus (*dwSensorLidarPlugin_sendMessage)(
    uint32_t const cmd, uint8_t const* const data, size_t const size,
    dwSensorPluginSensorHandle_t sensor);

/**
 * Holds the list of exported functions implemented by the vendor-provided
 * library.
 */
typedef struct
{
    dwSensorCommonPluginFunctions common;
    dwSensorLidarPlugin_parseDataBuffer parseDataBuffer;
    dwSensorLidarPlugin_getDecoderConstants getDecoderConstants;
    dwSensorLidarPlugin_sendMessage sendMessage;
} dwSensorLidarPluginFunctionTable;

/**
 * Returns the table of functions that are provided by the vendor-provided library for the sensor.
 *
 * @param[out] functions Function table exported by the library
 *
 * @return DW_FAILURE - unspecified failure while getting the function table. <br>
 *         DW_SUCCESS
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
dwStatus dwSensorLidarPlugin_getFunctionTable(dwSensorLidarPluginFunctionTable* functions);

/** @} */

#ifdef __cplusplus
}
#endif

#endif

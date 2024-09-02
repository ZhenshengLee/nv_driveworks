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
 * <b>NVIDIA DriveWorks: Radar Sensor Plugin Interface</b>
 *
 * @b Description: This file defines the interfaces to be implemented for Radar sensor plugins.
 */

#ifndef DW_SENSORS_RADAR_PLUGIN_H
#define DW_SENSORS_RADAR_PLUGIN_H

#include <dw/sensors/legacy/plugins/SensorCommonPlugin.h>
#include <dw/sensors/radar/RadarFullTypes.h>

/**
 * @defgroup sensor_plugins_ext_radar_group Radar Plugin
 * Provides an interface for non-standard radar sensors.
 * @ingroup sensor_plugins_ext_group
 *
 * @{
 */

#ifdef __cplusplus
extern "C" {
#endif

/// Holds constants for a given radar.
typedef struct
{
    /// Radar properties.
    dwRadarProperties properties;

    /// Packet max payload size, in bytes.
    size_t maxPayloadSize;

    /// Maximum points per packet associate to each radar return type and range.
    size_t maxPointsPerPacket[DW_RADAR_RETURN_TYPE_COUNT][DW_RADAR_RANGE_COUNT];

    /// Maximum packets per scan.
    size_t maxPacketsPerScan;

    /// Packets per scan associate to each radar return type and range.
    size_t packetsPerScan[DW_RADAR_RETURN_TYPE_COUNT][DW_RADAR_RANGE_COUNT];

    /// Dynamics Size, in bytes.
    size_t dynamicsSizeInBytes;
} _dwSensorRadarDecoder_constants;

/**
 * Processes the data previously passed via the 'dwSensorPlugin_pushData' interface.
 *
 * The interpreted memory buffer outputted from this API is owned by the plugin.
 * The plugin shall support multiple buffers in flight via this API.
 *
 * @param[out] output Pointer to decoded radar scan.
 * @param[in] scanType Specifies the scantype of the previously pushed raw data via the 'dwSensorPlugin_pushData' interface.
 * @param[in] sensor Specifies the sensor the data came from.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_NOT_AVAILABLE - if no scan is ready for consumption
 *         DW_NOT_READY - more raw data may needed to get a complete decoded packet
 *         DW_INVALID_ARGUMENT - if invalid argument
 *         DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorRadarPlugin_parseDataBuffer)(dwRadarScan* output, const dwRadarScanType scanType,
                                                        dwSensorPluginSensorHandle_t sensor);

/**
 * Gets constants associated with this radar sensor.
 *
 * @param[out] constants pointer to constants struct, parameters for this sensor are written here.
 * @param[in] sensor Specifies the sensor to which the constants are related.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_FAILURE, DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorRadarPlugin_getDecoderConstants)(_dwSensorRadarDecoder_constants* constants, dwSensorPluginSensorHandle_t sensor);

/**
 * Validates the raw data packet
 *
 * @param[in] rawData pointer to raw data
 * @param[in] size size of raw data
 * @param[out] messageType specifies the radarScanType
 * @param[in] sensor Specifies the sensor the data came from.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_FAILURE - if invalid packet
 *         DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorRadarPlugin_validatePacket)(const char* rawData, size_t size, dwRadarScanType* messageType,
                                                       dwSensorPluginSensorHandle_t sensor);

/**
 * Sends vehicle dynamics information to the radar.
 *
 * @param[in] state A pointer to the struct containing the vehicle dynamics information to send
 * @param[in] sensor Specifies the sensor the data came from.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_INVALID_ARGUMENT - if invalid argument
 *         DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorRadarPlugin_setVehicleState)(const dwRadarVehicleState* state, dwSensorPluginSensorHandle_t sensor);

/**
 * Holds the list of exported functions implemented by the vendor-provided
 * library.
 */
typedef struct
{
    dwSensorCommonPluginFunctions common;
    dwSensorRadarPlugin_parseDataBuffer parseDataBuffer;
    dwSensorRadarPlugin_getDecoderConstants getDecoderConstants;
    dwSensorRadarPlugin_validatePacket validatePacket;
    dwSensorRadarPlugin_setVehicleState setVehicleState;
} dwSensorRadarPluginFunctionTable;

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
dwStatus dwSensorRadarPlugin_getFunctionTable(dwSensorRadarPluginFunctionTable* functions);

/** @} */

#ifdef __cplusplus
}
#endif

#endif

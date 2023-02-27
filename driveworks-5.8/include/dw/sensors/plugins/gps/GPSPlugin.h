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
 * <b>NVIDIA DriveWorks: GPS Sensor Plugin Interface</b>
 *
 * @b Description: This file defines the interfaces to be implemented for GPS sensor plugins.
 */

#ifndef DW_SENSORS_GPS_PLUGIN_H
#define DW_SENSORS_GPS_PLUGIN_H

#include <dw/sensors/plugins/SensorCommonPlugin.h>
#include <dw/sensors/gps/GPS.h>

/**
 * @defgroup sensor_plugins_ext_gps_group GPS Plugin
 * Provides an interface for non-standard GPS sensors.
 * @ingroup sensor_plugins_ext_group
 *
 * @{
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Processes the data previously passed via the 'dwSensorPlugin_pushData' interface.
 *
 * @param[out] frame Interpreted GPS frame
 * @param[out] consumed Number of raw bytes (including header) consumed to successfully parse the frame
 * @param[in] sensor Specifies the sensor the data came from.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_FAILURE - if no frame is ready for consumption
 *         DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorGPSPlugin_parseDataBuffer)(dwGPSFrame* frame, size_t* consumed, dwSensorPluginSensorHandle_t sensor);

/// Function Table exposing GPS plugin functions
typedef struct
{
    dwSensorCommonPluginFunctions common;
    dwSensorGPSPlugin_parseDataBuffer parseDataBuffer;
} dwSensorGPSPluginFunctionTable;

/**
 * Gets the handle to functions defined in 'dwSensorGPSPluginFunctionTable' structure.
 *
 * @param[out] functions A pointer to the function table
 *
 * @return DW_INVALID_ARGUMENT - if pointer to the function table is NULL. <br>
 *         DW_SUCCESS
 *
 */
dwStatus dwSensorGPSPlugin_getFunctionTable(dwSensorGPSPluginFunctionTable* functions);

/** @} */

#ifdef __cplusplus
}
#endif

#endif

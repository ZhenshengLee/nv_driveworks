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
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks: Data Sensor Plugin Interface</b>
 *
 * @b Description: This file defines the interfaces to be implemented for data sensor plugins.
 */

#ifndef DW_SENSORS_DATA_PLUGIN_H
#define DW_SENSORS_DATA_PLUGIN_H

#include <dw/sensors/plugins/SensorCommonPlugin.h>
#include <dw/sensors/data/Data.h>

/**
 * @defgroup sensor_plugins_ext_data_group Data Plugin
 * Provides an interface for non-standard data sensors.
 * @ingroup sensor_plugins_ext_group
 *
 * @{
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Send a packet via data sensor plugin.
 *
 * @param[in] packet A pointer to the packet to be sent.
 * @param[in] sensor Specifies the data sensor to send the message over.
 *
 * @return DW_NOT_SUPPORTED - if the underlying sensor does not support send operation. <br>
 *         DW_INVALID_HANDLE - if given sensor handle is invalid. <br>
 *         DW_INVALID_ARGUMENT - if given arguments are invalid. <br>
 *         DW_NOT_AVAILABLE - if sensor has not been started. <br>
 *         DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorDataPlugin_sendPacket)(dwDataPacket* packet, dwSensorPluginSensorHandle_t sensor);

/// Function Table exposing data plugin functions
typedef struct
{
    dwSensorCommonPluginFunctions common;
    dwSensorDataPlugin_sendPacket sendPacket;
} dwSensorDataPluginFunctionTable;

/**
 * Gets the handle to functions defined in 'dwSensorDataPluginFunctionTable' structure.
 *
 * @param[out] functions A pointer to the function table
 *
 * @return DW_INVALID_ARGUMENT - if pointer to the function table is NULL. <br>
 *         DW_SUCCESS
 *
 */
dwStatus dwSensorDataPlugin_getFunctionTable(dwSensorDataPluginFunctionTable* functions);

/** @} */

#ifdef __cplusplus
}
#endif

#endif

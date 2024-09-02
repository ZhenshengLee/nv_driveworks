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
// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks: Sensor Plugin Interface to interact with live sensors</b>
 *
 * @b Description: This file defines the interfaces to be implemented for live sensor plugins.
 */

#ifndef DW_SENSORS_SENSORLIVE_SENSORPLUGIN_H
#define DW_SENSORS_SENSORLIVE_SENSORPLUGIN_H

#include <dw/core/base/Types.h>
#include <dw/core/context/Context.h>

#include <dw/sensors/common/SensorTypes.h>
#include <dw/sensors/legacy/plugins/SensorCommonPlugin.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize a new handle to the sensor managed by the plugin module.
 *
 * @param[out] handle A pointer to sensor handle.
 * @param[in] params Specifies the parameters for the sensor.
 * @param[in] ctx context handle.
 *
 * @return DW_INVALID_ARGUMENT - if pointer to the sensor handle is NULL. <br>
 *         DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorPlugin_initializeHandle)(dwSensorPluginSensorHandle_t* handle,
                                                    char const* params, dwContextHandle_t ctx);

#define SENSOR_PLUGIN_COMMON_FUNCTIONS                \
    dwSensorPlugin_initializeHandle initializeHandle; \
    dwSensorPlugin_release release;                   \
    dwSensorPlugin_start start;                       \
    dwSensorPlugin_stop stop;                         \
    dwSensorPlugin_reset reset;                       \
    dwSensorPlugin_readRawData readRawData;           \
    dwSensorPlugin_returnRawData returnRawData

/// Function Table exposing sensor plugin functions
typedef struct dwSensorPluginFunctions
{
    SENSOR_PLUGIN_COMMON_FUNCTIONS;
} dwSensorPluginFunctions;

/**
* Register sensor plugin which works with live sensors
*
* @param[in] sensorType the sensor type of registered sensor plugin
* @param[in] codecMimeType the codec type of registered CodecHeader plugin
* @param[in] funcTable pointer to sensor plugin function pointer table
* @param[in] sal Specifies the SAL handle to register sensor plugin.
*
* @return DW_INVALID_ARGUMENT if codecMimeType, funcTable or sal is nullptr. <br>
*         DW_BUFFER_FULL if no available entry in registration factory
*         DW_SUCCESS
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: No
*/
DW_API_PUBLIC
dwStatus dwSAL_registerSensorPlugin(dwSensorType sensorType, char const* codecMimeType,
                                    void const* funcTable, dwSALHandle_t const sal);

#ifdef __cplusplus
}
#endif

#endif // DW_SENSORS_SENSORLIVE_SENSORPLUGIN_H

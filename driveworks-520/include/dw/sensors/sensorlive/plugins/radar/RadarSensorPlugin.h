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
 * <b>NVIDIA DriveWorks: Radar Sensor Plugin Interface to interact with live sensors</b>
 *
 * @b Description: This file defines the interfaces to be implemented for radar sensor plugins.
 */

#ifndef DW_SENSORS_SENSORLIVE_PLUGINS_RADAR_RADARSENSORPLUGIN_H
#define DW_SENSORS_SENSORLIVE_PLUGINS_RADAR_RADARSENSORPLUGIN_H

#include <dw/sensors/sensorlive/plugins/SensorPlugin.h>
#include <dw/sensors/legacy/plugins/radar/RadarPlugin.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Function Table exposing radar sensor plugin functions
typedef struct dwRadarSensorPluginFunction
{
    SENSOR_PLUGIN_COMMON_FUNCTIONS;
    dwSensorRadarPlugin_setVehicleState setVehicleState;
} dwRadarSensorPluginFunction;

#ifdef __cplusplus
}
#endif

#endif // DW_SENSORS_SENSORLIVE_PLUGINS_RADAR_RADARSENSORPLUGIN_H

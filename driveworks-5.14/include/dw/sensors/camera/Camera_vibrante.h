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
// SPDX-FileCopyrightText: Copyright (c) 2016-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Vibrante Cameras</b>
 *
 * @b Description: This file defines Vibrante camera methods.
 */

/**
 * @defgroup vib_camera_group Vibrante Cameras Interface
 *
 * @brief Defines the Vibrante camera module.
 * @ingroup camera_group
 * @{
 */

#ifndef DW_SENSORS_CAMERA_VIBRANTE_H_
#define DW_SENSORS_CAMERA_VIBRANTE_H_

#include <dw/core/system/NvMedia.h>
#include <dw/core/base/Status.h>
#include <dw/image/Image.h>
#include <dw/sensors/Sensors.h>
#include <dw/sensors/camera/Camera.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
* Parses the data lines to extract information like exposure time and gain using NvMedia provided parser.
*
* @param[in] parsedData Contains information like exposure time and gain.
* @param[in] dataLines Additional data lines that were sent from the sensor.
* @param[in] sensor Sensor handle of the camera sensor previously created with 'dwHAL_createSensor()'.
*
* \ingroup Camera
**/
DW_API_PUBLIC
dwStatus dwSensorCamera_parseDataNvMedia(NvMediaISCEmbeddedDataInfo* parsedData,
                                         const dwImageDataLines* dataLines, dwSensorHandle_t sensor);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_SENSORS_CAMERA_VIBRANTE_H_

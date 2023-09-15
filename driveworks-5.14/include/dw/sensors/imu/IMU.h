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
// SPDX-FileCopyrightText: Copyright (c) 2016-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: IMU</b>
 *
 * @b Description: This file defines methods to access the IMU sensor.
 */

/**
 * @defgroup imu_group IMU Sensor
 * @ingroup sensors_group
 *
 * @brief Defines the IMU sensor.
 *
 * @{
 */

#ifndef DW_SENSORS_IMU_IMU_H_
#define DW_SENSORS_IMU_IMU_H_

#include "IMUTypes.h"

#include <dw/core/base/Config.h>
#include <dw/sensors/Sensors.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Reads the next IMU frame from the sensor within a given timeout. The method blocks until
 * either a new valid frame is received from the sensor or the given timeout is exceeded.
 *
 * @param[out] frame A pointer to an IMU frame structure to be filled with new data.
 * @param[in] timeoutUs Timeout, in us, to wait for a new message. Special values: DW_TIMEOUT_INFINITE - to wait infinitely.  Zero - means polling of internal queue.
 * @param[in] sensor Sensor handle of the IMU sensor previously created with dwSAL_createSensor().
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid <br>
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid <br>
 *         DW_TIME_OUT - if operation has timeout <br>
 *         DW_NOT_AVAILABLE    - if sensor has not been started or data is not available in polling mode. <br>
 *         DW_END_OF_STREAM    - if end of stream reached (virtual sensor only). <br>
 *         DW_SAL_SENSOR_ERROR - if there was an i/o or bus error. <br>
 *         DW_SUCCESS
**/
DW_API_PUBLIC
dwStatus dwSensorIMU_readFrame(dwIMUFrame* const frame, dwTime_t const timeoutUs, dwSensorHandle_t const sensor);

/**
 * Reads the IMU frame from raw data. Any processed messages can be picked up using
 * the dwSensorIMU_readFrame() method. This happens on the CPU thread where the function is called,
 * incurring an additional load on that thread.
 *
 * @param[in] data Undecoded imu data.
 * @param[in] size Size in bytes of the raw data.
 * @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid, <br>
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid, <br>
 *         DW_NOT_READY - if more data needs to be passed in (loop while it returns 'DW_NOT_READY'). <br>
 *         DW_SUCCESS
**/
DW_API_PUBLIC
dwStatus dwSensorIMU_processRawData(uint8_t const* const data, size_t const size, dwSensorHandle_t const sensor);

/**
 * Returns any IMU data previously processed through the raw data stream.
 * This happens on the CPU thread where the function is called, incurring an additional load on that thread.
 *
 * @param[out] frame Pointer to an IMU frame structure to be filled with new data.
 * @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid. <br>
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid. <br>
 *         DW_NOT_AVAILABLE - if no more data is available <br>
 *         DW_SUCCESS
**/
DW_API_PUBLIC
dwStatus dwSensorIMU_popFrame(dwIMUFrame* const frame, dwSensorHandle_t const sensor);

/**
 * Reads the next IMU frame New from the sensor within a given timeout. The method blocks until
 * either a new valid frame is received from the sensor or the given timeout is exceeded.
 *
 * @param[out] frame A pointer to an IMU frame New structure to be filled with new data.
 * @param[in] timeoutUs Timeout, in us, to wait for a new message. Special values: DW_TIMEOUT_INFINITE - to wait infinitely.  Zero - means polling of internal queue.
 * @param[in] sensor Sensor handle of the IMU sensor previously created with dwSAL_createSensor().
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid <br>
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid <br>
 *         DW_TIME_OUT - if operation has timeout <br>
 *         DW_NOT_AVAILABLE    - if sensor has not been started or data is not available in polling mode. <br>
 *         DW_END_OF_STREAM    - if end of stream reached (virtual sensor only). <br>
 *         DW_SAL_SENSOR_ERROR - if there was an i/o or bus error. <br>
 *         DW_SUCCESS
**/
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
DW_API_PUBLIC
dwStatus dwSensorIMU_readFrameNew(dwIMUFrameNew* const frame, dwTime_t const timeoutUs, dwSensorHandle_t const sensor)
    DW_DEPRECATED("dwSensorIMU_readFrameNew() is deprecated and will be removed in next major release. Please use dwSensorIMU_readFrame() instead");
#pragma GCC diagnostic pop

/**
 * Reads the IMU frame New from raw data. Any processed messages can be picked up using
 * the dwSensorIMU_readFrameNew() method. This happens on the CPU thread where the function is called,
 * incurring an additional load on that thread.
 *
 * @param[in] data Undecoded imu data.
 * @param[in] size Size in bytes of the raw data.
 * @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid, <br>
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid, <br>
 *         DW_NOT_READY - if more data needs to be passed in (loop while it returns 'DW_NOT_READY'). <br>
 *         DW_SUCCESS
**/
DW_API_PUBLIC
dwStatus dwSensorIMU_processRawDataNew(uint8_t const* const data, size_t const size, dwSensorHandle_t const sensor)
    DW_DEPRECATED("dwSensorIMU_processRawDataNew() is deprecated and will be removed in next major release. Please use dwSensorIMU_processRawData() instead");

/**
 * Returns any IMU Frame New previously processed through the raw data stream.
 * This happens on the CPU thread where the function is called, incurring an additional load on that thread.
 *
 * @param[out] frame Pointer to an IMU frame New structure to be filled with new data.
 * @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid. <br>
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid. <br>
 *         DW_NOT_AVAILABLE - if no more data is available <br>
 *         DW_SUCCESS
**/
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
DW_API_PUBLIC
dwStatus dwSensorIMU_popFrameNew(dwIMUFrameNew* const frame, dwSensorHandle_t const sensor)
    DW_DEPRECATED("dwSensorIMU_popFrameNew() is deprecated and will be removed in next major release. Please use dwSensorIMU_popFrame() instead");
#pragma GCC diagnostic pop

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_SENSORS_IMU_IMU_H_

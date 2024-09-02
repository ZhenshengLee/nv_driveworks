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
// SPDX-FileCopyrightText: Copyright (c) 2016-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Lidar</b>
 *
 * @b Description: This file defines the Lidar sensor.
 */

/**
 * @defgroup lidar_group Lidar Sensor
 * @ingroup sensors_group
 *
 * @brief Defines the Lidar sensor methods.
 *
 * @{
 */

#ifndef DW_SENSORS_LIDAR_LIDAR_H_
#define DW_SENSORS_LIDAR_LIDAR_H_

#include "LidarTypes.h"

#include <dw/sensors/common/Sensors.h>
#include <stdalign.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns size of auxiliary data element in bytes
 *
 * @param[out] sizeBytes element size
 * @param[in] auxType auxiliary data type
 * @retval DW_INVALID_ARGUMENT: The input parameter is invalid.
 * @retval DW_SUCCESS: Successful deal.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSensorLidar_getAuxElementSize(uint32_t* const sizeBytes, dwLidarAuxDataType const auxType);

/**
* Enables the decoding of the Lidar packets, which incurs an additional CPU load.
* Method fails if the sensor has been started and is capturing data. Stop the sensor first.
* The default state is to have decoding on. If on, dwSensor_readRawData(see reference [15]) returns DW_CALL_NOT_ALLOWED.
*
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_INVALID_HANDLE: The input handle is not a lidar handle.
* @retval DW_CALL_NOT_ALLOWED: The sensor is not stopped.
* @retval DW_SUCCESS: Successful deal.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_enableDecoding(dwSensorHandle_t const sensor);

/**
* Disable the decoding of the Lidar packets, which frees additional CPU load.
* Method fails if the sensor has been started and is capturing data. Stop the sensor first.
* The default state is to have decoding on. If on, dwSensor_readRawData(see reference [15]) returns DW_CALL_NOT_ALLOWED.
*
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_INVALID_HANDLE: The input handle is not a lidar handle.
* @retval DW_CALL_NOT_ALLOWED: The sensor is not stopped.
* @retval DW_SUCCESS: Successful deal.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_disableDecoding(dwSensorHandle_t const sensor);

/**
* Retrieves the state of packet decoding.
*
* @param[out] enable Contains the result of the query, which is true when decoding. False if RAW data.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_SUCCESS: Successful deal.
* @retval DW_INVALID_HANDLE: The input handle is not a lidar handle.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_isDecodingEnabled(bool* const enable, dwSensorHandle_t const sensor);

/**
* Reads one scan packet. The pointer returned is to the internal data pool. DW guarantees that the data
* remains constant until returned by the application. The data must be explicitly returned by the
* application.
*
* @param[out] data A pointer to a pointer that can read data from the sensor. The struct contains the
*                  numbers of points read, which depends on the sensor used.
* @param[in] timeoutUs Specifies the timeout in microseconds. Special values:
*                  DW_TIMEOUT_INFINITE - to wait infinitly.  Zero - means polling of internal queue.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_INVALID_HANDLE: the input handle is not a lidar handle.
* @retval DW_CALL_NOT_ALLOWED: the decoder is not working.
* @retval DW_INVALID_ARGUMENT: the input argument is invalid.
* @retval DW_NOT_AVAILABLE: device is disconneted or sensor is not working.
* @retval DW_TIME_OUT: timeout.
* @retval DW_SUCCESS: successful deal.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_readPacket(dwLidarDecodedPacket const** const data, dwTime_t const timeoutUs,
                                  dwSensorHandle_t const sensor);

/**
* Returns the data read to the internal pool. At this point the pointer is still be valid, but data is
* change based on newer readings of the sensor.
*
* @param[in] data A pointer to the scan data previously read from the Lidar to be returned to the pool.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_INVALID_HANDLE: the input handle is not a lidar handle.
* @retval DW_CALL_NOT_ALLOWED: the decoder is not working.
* @retval DW_SUCCESS: successful deal.
* @note Other return value will depend on the lidar type due to the reason that different type lidar will have different internal implements.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_returnPacket(dwLidarDecodedPacket const* const data, dwSensorHandle_t const sensor);

/**
* Decodes RAW data previously read and returns a pointer to it. This happens on the CPU thread where
* the function is called, incurring on additional load on that thread. The data is valid until the
* application calls dwSensor_returnRawData.
*
* @param[out] data A pointer to the memory pool owned by the sensor.
* @param[in] rawData A pointer for the non-decoded Lidar packet, returned by 'dwSensor_readRawData(see reference [15])'.
* @param[in] size Specifies the size in bytes of the raw data.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_INVALID_HANDLE: the input handle is not a lidar handle.
* @retval DW_CALL_NOT_ALLOWED: the decoder is not working.
* @retval DW_SUCCESS: successful deal.
* @note Other return value will depend on the lidar type due to the reason that different type lidar will have different internal implements.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_processRawData(dwLidarDecodedPacket const** const data, uint8_t const* const rawData, size_t const size,
                                      dwSensorHandle_t const sensor);

/**
* Gets information about the Lidar sensor.
*
* @param[out] lidarProperties A pointer to the struct containing the properties of the Lidar.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_INVALID_HANDLE: the input handle is not a lidar handle.
* @retval DW_CALL_NOT_ALLOWED: config is not allowed to get in passive mode.
* @retval DW_SUCCESS: successful deal.
* @note Other return value will depend on the lidar type due to the reason that different type lidar will have different internal implements.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_getProperties(dwLidarProperties* const lidarProperties, dwSensorHandle_t const sensor);

/**
* Sends a message to Lidar sensor.
*
* @param[in] cmd Command identifier associated to the given message data.
* @param[in] data A pointer to the message data.
* @param[in] size Size in bytes of the \p data.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_INVALID_HANDLE: the input handle is not a lidar handle.
* @note Other return value will depend on the lidar type due to the reason that different type lidar will have different internal implements.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_sendMessage(uint32_t const cmd, uint8_t const* const data,
                                   size_t const size, dwSensorHandle_t const sensor);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_SENSORS_LIDAR_LIDAR_H_

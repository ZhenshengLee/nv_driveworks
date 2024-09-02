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
// SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Radar</b>
 *
 * @b Description: This file defines the Radar sensor.
 */

/**
 * @defgroup radar_group Radar Sensor
 * @ingroup sensors_group
 *
 * @brief Defines the Radar sensor methods.
 *
 * @{
 */

#ifndef DW_SENSORS_RADAR_RADAR_H_
#define DW_SENSORS_RADAR_RADAR_H_

#include "RadarFullTypes.h"

#include <dw/core/base/Config.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/radar/RadarScan.h>
#include <dw/sensors/radar/RadarTypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
* Enables/Disables a scan type for the radar sensor. Method fails if the sensor does not support the
* specified scan type OR the sensor has been started and is capturing data.
*
* @param[in] enable Specifies if the scan should be enabled or disabled, true for enable.
* @param[in] scanType Specifies the scan type to enable/disable. The range of this parameters should be in struct value of dwRadarReturnType and dwRadarRange.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with dwSAL_createSensor().
*
* @note The scanType.returnType should be less than DW_RADAR_RETURN_TYPE_COUNT. The scanType.range should be less than DW_RADAR_RANGE_COUNT.
* @retval DW_INVALID_HANDLE: sensor handle is invalid, which means sensor handle is nullptr or sensor is not a radar
* @retval DW_INVALID_ARGUMENT: input arguments invalid. (scanType.returnType < 0 or scanType.range < 0)
* @retval DW_CALL_NOT_ALLOWED: operation is not allowed now. Radar started and running.
* @retval DW_SUCCESS: successful deal.
* @retval DW_NOT_SUPPORTED: The scan type in @a scanType is not supported.
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_toggleScanType(bool const enable, dwRadarScanType const scanType, dwSensorHandle_t const sensor);

/**
* Enables/disables the decoding of the Radar packets, which incurs in additional CPU load.
* Method fails if the sensor has been started and is capturing data. Stop the sensor first.
*
* @param[in] enable Specifies TRUE when decoding, false if RAW data.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with dwSAL_createSensor()
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid, which means sensor handle is nullptr or sensor is not a radar.
* @retval DW_CALL_NOT_ALLOWED: operation is not allowed now. Radar started and running.
* @retval DW_SUCCESS: successful deal.
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_setDataDecoding(bool const enable, dwSensorHandle_t const sensor);

/**
* Reads a single packet, which might be an entire scan or a fraction of a scan, which is sensor dependent.
* The data return to users is decoded from rawdata by DW, which is relative to 'dwSensor_readRawData'.
* The pointer returned is to the internal data pool. DW guarantees that the data
* remains constant until returned by the application. The data must be explicitly returned by the application.
* @note dwSensorRadar_returnData should be called if user doesn't need the @a data anymore.
* @param[out] data A pointer to a pointer to the decoded data read from the sensor. The struct contains the
*                  numbers of points read, which depends on the sensor used.
* @param[in] type Type of scan requested
* @param[in] timeoutUs Specifies the timeout in microseconds. Special values:
                        DW_TIMEOUT_INFINITE - to wait infinitely.  Zero - means polling of internal queue.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with dwSAL_createSensor().
*
* @note The scanType.returnType should be less than DW_RADAR_RETURN_TYPE_COUNT. The scanType.range should be less than DW_RADAR_RANGE_COUNT.
* @retval DW_INVALID_HANDLE: sensor handle is invalid, which means sensor handle is nullptr or sensor is not a radar.
* @retval DW_CALL_NOT_ALLOWED: Sensor is not decoding.
* @retval DW_INVALID_ARGUMENT: input arguments invalid.(type.returnType < 0, type.range < 0, @a data is nullptr)
* @retval DW_SUCCESS: successful deal.
* @retval DW_TIME_OUT: time out.
* @retval DW_NOT_AVAILABLE Sensor stops or scan type in @a type is not supported
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_readData(const dwRadarScan** const data, const dwRadarScanType type,
                                const dwTime_t timeoutUs, dwSensorHandle_t const sensor);
/**
* Returns the data read to the internal pool. At this point the pointer is still valid, but data is
* changed based on newer readings of the sensor.
*
* @param[in] scan A pointer to the scan data previously read from the Radar using 'dwSensorRadar_readData()' to be returned to the pool.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with dwSAL_createSensor().
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid, which means sensor handle is nullptr or sensor is not a radar.
* @retval DW_INVALID_ARGUMENT: input arguments invalid. @a data is nullptr.
* @retval DW_CALL_NOT_ALLOWED: Sensor is not decoding.
* @retval DW_SUCCESS: successful deal.
* @retval DW_TIME_OUT: time out.
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_returnData(const dwRadarScan* const scan, dwSensorHandle_t const sensor);

/**
* Reads one scan chunk. The pointer returned is to the internal data pool. DW guarantees that the data
* remains constant until returned by the application. The data must be explicitly returned by the
* application.
*
* @note This method returns the oldest scan contained in the internal pool.
*
* @param[out] data A pointer to a pointer to a decoded scan from the sensor. The struct contains the
*                  numbers of points read, which depends on the sensor used.
* @param[in] timeoutUs Specifies the timeout in microseconds. Special values:
                        DW_TIMEOUT_INFINITE - to wait infinitely.  Zero - means polling of internal queue.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with dwSAL_createSensor().
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid, which means sensor handle is nullptr or sensor is not a radar.
* @retval DW_INVALID_ARGUMENT: input arguments invalid. @a data is nullptr.
* @retval DW_CALL_NOT_ALLOWED: Sensor is not decoding.
* @retval DW_SUCCESS: successful deal.
* @retval DW_TIME_OUT: time out.
* @retval DW_NOT_AVAILABLE Sensor stops.
* 
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_readScan(const dwRadarScan** const data,
                                const dwTime_t timeoutUs, dwSensorHandle_t const sensor);

/**
* Returns the data covering an entire scan read to the internal pool. At this point the pointer is still valid,
* but data is changed based on newer readings of the sensor.
*
* @param[in] scan A pointer to an entire scan's data previously read from the Radar using 'dwSensorRadar_readScan()' to be returned to the pool.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with dwSAL_createSensor().
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid, which means sensor handle is nullptr or sensor is not a radar.
* @retval DW_INVALID_ARGUMENT: input arguments invalid. @a data is nullptr.
* @retval DW_CALL_NOT_ALLOWED: Sensor is not decoding.
* @retval DW_SUCCESS: successful deal.
* @retval DW_TIME_OUT: time out.
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_returnScan(const dwRadarScan* const scan, dwSensorHandle_t const sensor);

/**
* Decodes raw data previously read onto the application side structure. This happens on the CPU thread where
* the function is called, incurring on additional load on that thread.
*
* @param[out] data A pointer to a container for the decoded data.
* @param[in] rawData A pointer for the non-decoded Radar packet, as returned from 'dwSensor_readRawData()'.
* @param[in] size Specifies the size in bytes of the raw data.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid, which means sensor handle is nullptr or sensor is not a radar.
* @retval DW_INVALID_ARGUMENT: input arguments invalid. @a data is nullptr or @a rawData is nullptr or dataSize in @a rawData is not equal to @a size.
* @retval DW_CALL_NOT_ALLOWED: Sensor is decoding.
* @retval DW_SUCCESS: successful deal.
* @retval DW_TIME_OUT: time out.
*
* @note The returned dwRadarScan is only valid till the next 'dwSensorRadar_processRawData()' call.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_processRawData(const dwRadarScan** const data,
                                      const uint8_t* const rawData, size_t const size, dwSensorHandle_t const sensor);

/**
* Gets information about the radar sensor.
*
* @param[out] radarProperties A pointer to the struct containing the properties of the radar.
* @param[in] sensor Sensor handle created with dwSAL_createSensor().
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid, which means sensor handle is nullptr or sensor is not a radar.
* @retval DW_INVALID_ARGUMENT: input arguments invalid. @a data is nullptr.
* @retval DW_SUCCESS: successful deal.
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_getProperties(dwRadarProperties* const radarProperties, dwSensorHandle_t const sensor);

/**
* Sends vehicle dynamics information to the radar.
*
* @param[in] data A pointer to the struct containing the vehicle dynamics information to send
* @param[in] sensor Sensor handle created with dwSAL_createSensor().
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid, which means sensor handle is nullptr or sensor is not a radar.
* @retval DW_INVALID_ARGUMENT: input arguments invalid. @a data is nullptr.
* @retval DW_SUCCESS: successful deal.
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_setVehicleState(dwRadarVehicleState* const data, dwSensorHandle_t const sensor);

/**
* Sends the radar mount position information to the radar.
*
* @param[in] data A pointer to the struct containing the radar mount position information to send.
* @param[in] sensor Sensor handle created with dwSAL_createSensor().
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid, which means sensor handle is nullptr or sensor is not a radar.
* @retval DW_INVALID_ARGUMENT: input argument invalid. @a data is nullptr.
* @retval DW_SUCCESS: successful deal.
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_setMountPosition(dwRadarMountPosition* const data, dwSensorHandle_t const sensor);

/**
* Deep-copy a radar scan
*
* Will not allocate memory, the buffers in dst must be pre-allocated. The pointers in dst will not be
* modified, but the memory they are pointing to will be overwritten. All pointers except for radarSSI
* must be non-nullptr.
*
* Copying of radarSSI is optional: If src->radarSSI is non-nullptr, dst->radarSSI must also be non-nullptr.
* The buffer will then be copied. Use dwSensorRadar_getProperties() to get the buffer size
* dwRadarProperties::radarSSISizeInBytes for allocating the dst buffer.
*
* @param[out] dst Pointer to destination radar scan object (will be overwritten)
* @param[in] src Pointer to source radar scan object
*
* @note the pointer member in @a dst and @a src should not be NULL.
* @retval DW_INVALID_ARGUMENT: invalid pointer. @a dst is NULL or @a src is NULL.
* @retval DW_SUCCESS: copy successfully.
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_copyScan(dwRadarScan* dst, dwRadarScan const* src);

/**
* Get size of the data buffer of a radar scan
*
* @param[out] size Size of the data section, in bytes
* @param[in] scan Pointer to radar scan object
*
* @retval DW_INVALID_ARGUMENT: @a size is NULL or @a scan is NULL.
* @retval DW_SUCCESS: get successfully.
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_getScanDataSize(size_t* size, dwRadarScan const* scan);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_SENSORS_RADAR_RADAR_H_

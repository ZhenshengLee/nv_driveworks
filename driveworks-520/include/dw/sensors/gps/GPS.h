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
 * <b>NVIDIA DriveWorks API: GPS</b>
 *
 * @b Description: This file defines methods to access GPS sensor.
 */

/**
 * @defgroup gps_group GPS Sensor
 * @ingroup sensors_group
 *
 * @brief Defines the GPS sensor.
 *
 * @{
 */

#ifndef DW_SENSORS_GPS_GPS_H_
#define DW_SENSORS_GPS_GPS_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>

#include <dw/sensors/common/Sensors.h>

#include "GPSFrame.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
* This structure contains one frame of data from an GPS/GNSS sensor.
*/
typedef struct dwGPSFrameNew
{
    /// Signal validity information
    dwGPSSignalValidityInfo validityInfo;

    /// @note ID VS-91010
    /// @note description Timestamp of the message when first received.
    /// @note min nan    max nan
    /// @note freq 4    unit us
    dwTime_t timestamp_us;

    /// @note ID VS-91020
    /// @note description Latitude in WGS-84.
    /// @note min nan    max nan
    /// @note freq 4    unit deg
    float64_t latitude;

    /// @note ID VS-91030
    /// @note description Longitude in WGS-84.
    /// @note min nan    max nan
    /// @note freq 4    unit deg
    float64_t longitude;

    /// @note ID VS-91040
    /// @note description Altitude over WGS-84 ellipsoid.
    /// @note min nan    max nan
    /// @note freq 4    unit m
    float64_t altitude;

    /// @note ID VS-91050
    /// @note description Course relative to true north.
    /// @note min nan    max nan
    /// @note freq 4    unit deg
    float64_t course;

    /// @note ID VS-91060
    /// @note description Horizontal speed.
    /// @note min nan    max nan
    /// @note freq 4    unit m/s
    float64_t speed;

    /// @note ID VS-91070
    /// @note description Vertical speed.
    /// @note min nan    max nan
    /// @note freq 4    unit m/s
    float64_t climb;

    /// @note ID VS-91080
    /// @note description Horizontal Dilution of Precision.
    /// @note min nan    max nan
    /// @note freq 4    unit unitless
    float64_t hdop;

    /// @note ID VS-91090
    /// @note description Vertical Dilution of Precision.
    /// @note min nan    max nan
    /// @note freq 4    unit unitless
    float64_t vdop;

    /// @note ID VS-91100
    /// @note description Position Dilution of Precision.
    /// @note min nan    max nan
    /// @note freq 4    unit unitless
    float64_t pdop;

    /// @note ID VS-91110
    /// @note description Horizontal Accuracy of Solution.
    /// @note min nan    max nan
    /// @note freq 4    unit m
    float64_t hacc;

    /// @note ID VS-91120
    /// @note description Vertical Accuracy of Solution.
    /// @note min nan    max nan
    /// @note freq 4    unit m
    float64_t vacc;

    /// @note ID VS-91130
    /// @note description UTC time.
    /// @note min nan    max nan
    /// @note freq 4    unit us
    dwTime_t utcTime;

    /// @note ID VS-91140
    /// @note description Number of satellites in view.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 4    unit unitless
    uint8_t satelliteCount;

    /// @note ID VS-91150
    /// @note description 2D or 3D fix.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 4    unit unitless
    /// @note *** valid values**: {
    ///                DW_GPS_FIX_STATUS_NO_FIX,
    ///                DW_GPS_FIX_STATUS_2D_FIX,
    ///                DW_GPS_FIX_STATUS_3D_FIX
    ///        }
    dwGPSFixStatus fixStatus;

    /// @note ID VS-91170
    /// @note description GNSS Timestamp Quality.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 4    unit unitless
    /// @note *** valid values**: {
    ///                DW_GPS_TIMESTAMP_QUALITY_NOT_INIT,
    ///                DW_GPS_TIMESTAMP_QUALITY_OK,
    ///                DW_GPS_TIMESTAMP_QUALITY_SYNC_LOST,
    ///        }
    dwGPSTimestampQuality timestampQuality;

    /// @note ID VS-91175
    /// @note description GPS mode.
    /// @note TODO: Backend not yet implemented
    /// @note min nan    max nan
    /// @note freq 4    unit unitless
    /// @note *** valid values**: {
    ///                DW_GPS_MODE_UNKNOWN,
    ///                DW_GPS_MODE_INVALID_GPS,
    ///                DW_GPS_MODE_DEAD_RECKONING,
    ///                DW_GPS_MODE_STAND_ALONE,
    ///                DW_GPS_MODE_PRECISE_POINT,
    ///                DW_GPS_MODE_CODE_DIFFERENTIAL,
    ///                DW_GPS_MODE_RTK_FLOAT,
    ///                DW_GPS_MODE_RTK_FIXED,
    ///                DW_GPS_MODE_PPP_CONVERGING,
    ///                DW_GPS_MODE_PPP_CONVERGED
    ///        }
    dwGPSMode mode;

    /// Reserved space
    uint8_t reserved[508];
} dwGPSFrameNew
    DW_DEPRECATED("dwGPSFrameNew is deprecated and will be removed soon. Please use dwGPSFrame instead");

/**
 * @brief Reads the next available GPS data frame with a given timeout. The method blocks until
 *        either a new valid frame is received from the sensor or the given timeout exceeds.
 *        Frames read from the sensor do not have to be returned because the messages are copied.
 * 
 * This API function will achieve two functionalities of this software unit:
 * - Read a GPS sensor packet from physical GPS sensor
 * - Decode this GPS sensor packet and add cache decoded GPS frames
 *
 * This API function will firstly check if there are already decoded GPS frames available, if there are decoded GPS frames then this API function
 * returns the first decoded GPS frame then return; otherwise it will read GPS sensor packet then decode it and return first decoded GPS frames, or
 * return DW_TIME_OUT if no GPS frames could be decoded till timeout is detected.
 *
 * This function is only used after the GPS sensor is initialized and started successfully.
 *
 * @param[out] frame A pointer to a GPS frame structure to be filled with new localization data.
 * @param[in] timeoutUs Specify the timeout in us to wait for a new message. 
 *                      Special values: <br>
 *                      - DW_TIMEOUT_INFINITE - to wait infinitly.  
 *                      - Zero - mean polling of internal queue. - if there are already decoded GPS frames, return first frame immediately; otherwise return DW_TIME_OUT directly.
 * @param[in] sensor Specify the sensor handle of the GPS sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid. <br>
 *         DW_INVALID_ARGUMENTS - if the internal member m_decoder of GPS object is equal to nullptr. <br>
 *         DW_TIME_OUT - if operation has timeout. <br>
 *         DW_NOT_AVAILABLE    - if sensor has not been started or data is not available in polling mode. <br>
 *         DW_END_OF_STREAM    - if end of stream reached (virtual sensor only). <br>
 *         DW_SAL_SENSOR_ERROR - if there was an i/o or bus error. <br>
 *         DW_SUCCESS - if read frame successful.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 * @note This function will block until a frame is decoded or timeout is detected, if @b timeoutUs is specified as non-zero values.
 * @note If no valid GPS packets could be returned from physical GPS sensors(for Vehicle GPS sensors, this means either all received GPS packets have wrong packet size. or no GPS packets are received at all) till timeout, no GPS frames could be decoded.
 *       
 */
DW_API_PUBLIC
dwStatus dwSensorGPS_readFrame(dwGPSFrame* const frame, dwTime_t const timeoutUs, dwSensorHandle_t const sensor);

/**
* @brief Decodes RAW data previously read. Any processed messages can be picked up using
*        the dwSensorGPS_popMessage() method. This happens on the CPU thread where the function is called,
*        incurring on additional load on that thread.
*
* This API function will achieve functionality of this software unit:
* - Decode GPS sensor packets into GPS frames and cache decoded GPS frames
*
* This API function will push specified GPS sensor packet into its internal GPS decoders. Since internal GPS decoder will return once an GPS frame is decoded, or no GPS frames
* could be decoded from remaining bytes from specified GPS sensor packet, thus this API function will keep driving the decoder to return as many GPS frames as possible, until
* decoder could return no new GPS frames. All decoded GPS frames are cached internally and could be accessed by dwSensorGPS_readFrame() or dwSensorGPS_popFrame().
* Based on sensor configuration value 'output-timestamp=', decoded GPS frames would be modified in their hostTimestamp field values accordingly.
*
* @sa https://developer.nvidia.com/docs/drive/drive-os/6.0.8.1/public/driveworks-nvsdk/nvsdk_dw_html/sensors_usecase4.html for description of available option values for 'output-timestamp=' sensor parameter.
*
* This function should only be called after sensor is started, and with raw GPS sensor packet(s) returned from dwSensor_readRawData. After
* this raw sensor packet is processed by this function, user should also call dwSensor_returnRawData to return the memory block to SAL.
*
* This function is a synchronous function. Customers have to wait till the raw GPS sensor data is processed.
*

* @param[in] data Undecoded gps data.
* @param[in] size Size in bytes of the raw data.
* @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
*
* @return DW_INVALID_HANDLE - if given sensor handle is invalid. Only handles returned by dwSAL_createSensor with GPS sensor protocols are allowed. <br>
*         DW_INVALID_ARGUMENTS - if given arguments are invalid, Here it means input pointer 'data' is empty <br>
*         DW_NOT_READY - if more data needs to be passed in (loop while it returns 'DW_NOT_READY'). <br>
*         DW_SUCCESS - specified raw GPS sensor packet was decoded into one or more GPS frames. Please read decoded GPS frames via @ref dwSensorGPS_readFrame or @ref dwSensorGPS_popFrame
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwSensorGPS_processRawData(uint8_t const* const data, size_t const size, dwSensorHandle_t const sensor);

/**
* @brief Returns any GPS frame previously processed through RAW data stream.
*        This happens on the CPU thread where the function is called, incurring an additional load on that thread.
*
* This function returns only the first single decoded GPS frame from cached GPS frames(and removes it from cache). If no frames are available in queue,
* DW_NOT_AVAILABLE is returned and no GPS frame is returned.
*
* This function should only be called after sensor is started. From above descriptions, this function requires previous callings to function
* @ref dwSensorGPS_readFrame or @ref dwSensorGPS_processRawData returns success otherwise this function will return DW_NOT_AVAILABLE.
*
* This function is a synchronous function. Customers have to wait till the polling of GPS frame queue is finished.
*
*
* @param[out] frame Pointer to an GPS frame structure to be filled with new data.
* @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
*
* @return DW_INVALID_HANDLE - if given sensor handle is invalid. Only handles returned by dwSAL_createSensor with GPS sensor protocols are allowed. <br>
*         DW_INVALID_ARGUMENTS - if given arguments are invalid. Here it means input pointer 'frame' is empty<br>
*         DW_NOT_AVAILABLE - if no more data is available <br>
*         DW_SUCCESS - An GPS frame is dequeued from internal GPS frame queue into frame
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwSensorGPS_popFrame(dwGPSFrame* const frame, dwSensorHandle_t const sensor);

/**
 * @brief Reads the next GPS packet with a given timeout. The method blocks until
 * either a new valid frame is received from the sensor or the given timeout exceeds.
 *
 * @param[out] frame A pointer to a GPS frame New structure to be filled with new localization data.
 * @param[in] timeoutUs Specify the timeout in us to wait for a new message. Special values: DW_TIMEOUT_INFINITE - to wait infinitly.  Zero - means polling of internal queue.
 * @param[in] sensor Specify the sensor handle of the GPS sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid. <br>
 *         DW_INVALID_ARGUMENTS - if the member m_decoder of GPS object is equal to nullptr. <br>
 *         DW_TIME_OUT - if operation has timeout. <br>
 *         DW_NOT_AVAILABLE    - if sensor has not been started or data is not available in polling mode. <br>
 *         DW_END_OF_STREAM    - if end of stream reached(virtual sensor only). <br>
 *         DW_SAL_SENSOR_ERROR - if there was an i/o or bus error. <br>
 *         DW_SUCCESS - the procedure is done successfully.
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
#pragma GCC diagnostic push // TODO(pshu): Remove in next major release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
DW_API_PUBLIC
dwStatus dwSensorGPS_readFrameNew(dwGPSFrameNew* const frame, dwTime_t const timeoutUs, dwSensorHandle_t const sensor)
    DW_DEPRECATED("dwSensorGPS_readFrameNew() is deprecated and will be removed soon. Please use dwSensorGPS_readFrame() instead");
#pragma GCC diagnostic pop

/**
* Decodes RAW data previously read. Any processed messages can be picked up using
* the dwSensorGPS_popMessageNew() method. This happens on the CPU thread where the function is called,
* incurring on additional load on that thread.
*
* @param[in] data Undecoded gps data.
* @param[in] size Size in bytes of the raw data.
* @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
*
* @return DW_INVALID_HANDLE - if given sensor handle is invalid, <br>
*         DW_INVALID_ARGUMENTS - if the member m_decoder of GPS object is not initialized correctly or equal to nullptr, <br>
*         DW_NOT_READY - if more data needs to be passed in (loop while it returns 'DW_NOT_READY'). <br>
*         DW_SUCCESS - if the procedure is done successfully.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwSensorGPS_processRawDataNew(uint8_t const* const data, size_t const size, dwSensorHandle_t const sensor)
    DW_DEPRECATED("dwSensorGPS_processRawDataNew() is deprecated and will be removed soon. Please use dwSensorGPS_processRawData() instead");

/**
* Returns any GPS frame New previously processed through RAW data stream.
* This happens on the CPU thread where the function is called, incurring an additional load on that thread.
*
* @param[out] frame Pointer to an GPS frame New structure to be filled with new data.
* @param[in] sensor Sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
*
* @return DW_INVALID_HANDLE - if given sensor handle is invalid. <br>
*         DW_INVALID_ARGUMENTS - if given arguments are invalid. Here it means input pointer 'frame' is empty<br>
*         DW_NOT_AVAILABLE - if no more data is available <br>
*         DW_SUCCESS - if the procedure is done successfully.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
#pragma GCC diagnostic push // TODO(pshu): Remove in next major release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
DW_API_PUBLIC
dwStatus dwSensorGPS_popFrameNew(dwGPSFrameNew* const frame, dwSensorHandle_t const sensor)
    DW_DEPRECATED("dwSensorGPS_popFrameNew() is deprecated and will be removed soon. Please use dwSensorGPS_popFrame() instead");
#pragma GCC diagnostic pop

#ifdef __cplusplus
}
#endif
#endif // DW_SENSORS_GPS_GPS_H_
/** @} */

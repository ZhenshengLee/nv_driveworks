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
// SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Data</b>
 *
 * @b Description: This file defines the Data sensor structure.
 */

/**
 * @defgroup data_group Data Sensor
 * @ingroup sensors_group
 *
 * @brief Defines the Data sensor structure.
 * @{
 */

#ifndef DW_SENSORS_DATA_H_
#define DW_SENSORS_DATA_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>

#include <dw/sensors/Sensors.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Holds a data packet
typedef struct dwDataPacket
{
    /// Number of bytes of the payload.
    size_t size;

    /// Timestamp of the message in microseconds (using clock of the context).
    dwTime_t hostTimestamp;

    /// Payload.
    uint8_t* raw;
} dwDataPacket;

/**
* Reads the next packet. The pointer returned is to the internal data pool. The
* data must be explicitly returned by the application. The method blocks until
* either a new valid frame is received from the sensor or the given timeout
* exceeds.
*
* @param[out] packet A pointer to a pointer to a data packet read from the sensor.
* @param[in] timeoutUs Specifies the timeout in microseconds. Special values:
*                       DW_TIMEOUT_INFINITE - to wait infinitly.  Zero - means
*                       polling of internal queue.
* @param[in] sensor Specifies the sensor handle of the sensor previously created
*                   with 'dwSAL_createSensor()'.
*
* @return DW_INVALID_HANDLE, DW_INVALID_ARGUMENT, DW_NOT_AVAILABLE, DW_TIME_OUT,
* DW_SUCCESS
*
*/
DW_API_PUBLIC
dwStatus dwSensorData_readPacket(const dwDataPacket** const packet, dwTime_t const timeoutUs, dwSensorHandle_t const sensor);

/**
* Returns the data read to the internal pool.
*
* @param[in] scan A pointer to the data previously read from the data sensor to
*                 be returned to the pool.
* @param[in] sensor Specifies the sensor handle of the sensor previously created
*                   with 'dwSAL_createSensor()'.
*
* @return DW_INVALID_HANDLE, DW_INVALID_ARGUMENT, DW_NOT_AVAILABLE, DW_TIME_OUT,
* DW_SUCCESS
*
*/
DW_API_PUBLIC
dwStatus dwSensorData_returnPacket(dwDataPacket const* const scan, dwSensorHandle_t const sensor);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_SENSORS_DATA_H_

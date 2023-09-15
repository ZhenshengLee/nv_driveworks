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
// SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks: Common Sensor Plugin Interface</b>
 *
 * @b Description: This file defines the interfaces to be implemented for all sensor plugins.
 */

#ifndef DW_SENSORS_COMMON_PLUGIN_H
#define DW_SENSORS_COMMON_PLUGIN_H

#include <dw/core/base/Types.h>
#include <dw/core/context/Context.h>

#include <dw/sensors/Sensors.h>

/**
 * @defgroup sensor_plugins_group Plugins
 * @ingroup sensors_group
 * @brief Defines plugin interfaces for non-standard sensors.
 *
 */

/**
 * @defgroup sensor_plugins_ext_group Plugins (Full)
 * @ingroup sensors_group
 * @brief Defines full-sensor plugin interfaces for non-standard sensors.
 *
 */

/**
 * @defgroup sensor_plugins_ext_common_group Common Interface
 * Provides an interface for non-standard sensors.
 * @ingroup sensor_plugins_ext_group
 * @{
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef void* dwSensorPluginSensorHandle_t;

// Enum to indicate the relationship between raw and decoded data for a custom sensor
typedef enum {
    /// 1-1 map between raw data and decoded packet
    DW_SENSORS_RAW_DEC_ONE_TO_ONE = 0,
    /// N-1(N>1) map between raw data and decoded packet
    DW_SENSORS_RAW_DEC_MANY_TO_ONE = 1,
    /// Not supported
    DW_SENSORS_RAW_DEC_NOT_SUPPORTED = 2
} dwSensorPlugin_rawToDecMap;

// Enum to define Time Domain
typedef enum {
    DW_SENSORS_PLUGIN_TIME_DOMAIN_HOST   = 0, //!< Host Clock, as given by DW TimeSource.
    DW_SENSORS_PLUGIN_TIME_DOMAIN_TSC    = 1, //!< Tegra Timestamp System Counter
    DW_SENSORS_PLUGIN_TIME_DOMAIN_PTP    = 2, //!< Precision Time Protocol
    DW_SENSORS_PLUGIN_TIME_DOMAIN_UTC    = 3, //!< Coordinated Universal Time
    DW_SENSORS_PLUGIN_TIME_DOMAIN_UTCTOH = 4, //!< Top of Hour in Coordinated Universal Time
    DW_SENSORS_PLUGIN_TIME_DOMAIN_MISC   = 5, //!< Any other timestamp
    DW_SENSORS_PLUGIN_TIME_DOMAIN_COUNT  = 6  //!< Number of time domains in this enum (leave at end)
} dwSensorPlugin_timeDomain;

/// Structure for generic constants returned by the plugin
typedef struct
{
    /// Packet size for each raw data message
    size_t packetSize;
    /// Indicate the relationship between raw data and decoded packet
    dwSensorPlugin_rawToDecMap rawToDec;
} dwSensorPluginProperties;

// Holds the sensor firmware version information
typedef struct
{
    // Separate the fwVersion string into three pieces.
    uint64_t versionMajor; // Firmware version major number
    uint64_t versionMinor; // Firmware version minor number
    uint64_t versionPatch; // Firmware version patch number
    char* versionString;   // Firmware version string
} dwSensorPlugin_firmwareVersion;

// Holds the sensor information
typedef struct
{
    dwSensorPlugin_firmwareVersion firmware;
} dwSensorPlugin_information;

// Enum to indicate which level the raw data to read from
typedef enum {
    DW_SENSORS_RAW_DATA_LEVEL_ZERO  = 0, // Raw data same as from readRawData, TP segment in SOME/IP use case
    DW_SENSORS_RAW_DATA_LEVEL_ONE   = 1, // Raw data combined from LEVEL_ZERO raw Data, reassembled pdu(from segments) in SOME/IP use case
    DW_SENSORS_RAW_DATA_LEVEL_TWO   = 2, // Raw data combined from LEVEL_ONE raw Data, combined pdu(from split pdus)
    DW_SENSORS_RAW_DATA_LEVEL_COUNT = 3  // Count for validating user passed level
} dwSensorPlugin_rawDataLevel;

/**
 * Creates a new handle to the sensor managed by the plugin module.
 *
 * @param[out] handle A pointer to sensor handle.
 * @param[out] properties Sensor-specific properties & constants returned by the plugin.
 * @param[in] params Specifies the parameters for the sensor.
 * @param[in] ctx context handle.
 *
 * @return DW_INVALID_ARGUMENT - if pointer to the sensor handle is NULL. <br>
 *         DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorPlugin_createHandle)(dwSensorPluginSensorHandle_t* handle,
                                                dwSensorPluginProperties* properties,
                                                char const* params, dwContextHandle_t ctx);

/**
 * Releases a sensor managed by the plugin module.
 *
 * @note This method renders the sensor handle unusable.
 *
 * @param[in] handle The handle to a sensor created previously with the 'dwSensorPlugin_createHandle' interface.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorPlugin_release)(dwSensorPluginSensorHandle_t handle);

/**
 * Creates and initializes a new sensor managed by the plugin.
 * The created sensor will be released using the 'dwSensorPlugin_releaseSensor' interface.
 *
 * @param[in] params Specifies the parameters for sensor creation.
 * @param[in] sal SAL handle.
 * @param[in] handle A sensor handle previously created w/ the 'dwSensorPlugin_createHandle' interface.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_SAL_SENSOR_ERROR - if a non recoverable error happens during sensor creation. <br>
 *         DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorPlugin_createSensor)(char const* params, dwSALHandle_t sal, dwSensorPluginSensorHandle_t handle);

/**
 * Starts the sensor previously successfully created with 'dwSensorPlugin_createSensor' interface.
 * Sensor data should ready to be received using the '_dwSensorPlugin_readRawData()' API after
 * the execution of this call.
 *
 * @param[in] handle A sensor handle previously created w/ the 'dwSensorPlugin_createHandle' interface.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_SAL_SENSOR_ERROR - if a non recoverable error happens during sensor start. <br>
 *         DW_SUCCESS
 *
**/
typedef dwStatus (*dwSensorPlugin_start)(dwSensorPluginSensorHandle_t handle);

/**
 * Stops the sensor. This method shall block while the sensor is stopped.
 *
 * @param[in] handle A sensor handle previously created w/ the 'dwSensorPlugin_createHandle' interface.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_SAL_SENSOR_ERROR - if a non recoverable error happens during sensor stop. <br>
 *         DW_SUCCESS
**/
typedef dwStatus (*dwSensorPlugin_stop)(dwSensorPluginSensorHandle_t handle);

/**
 * Resets the sensor. The method shall block while the sensor is reset.
 *
 * @note It is guarunteed that all outstanding references to sensor data will be returned prior to this call.
 *
 * @param[in] handle A sensor handle previously created w/ the 'dwSensorPlugin_createHandle' interface.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_SAL_SENSOR_ERROR - if a non recoverable error happens during sensor reset. <br>
 *         DW_SUCCESS
**/
typedef dwStatus (*dwSensorPlugin_reset)(dwSensorPluginSensorHandle_t handle);

/**
 * Reads RAW data for one single message from the sensor as byte array. This should
 * be the raw unprocessed data received from the sensor. Generally this is the entrypoint to
 * perform a read operation on the sensor.
 *
 * For each raw "message" from the sensor, data must be packed in the following memory layout:
 *
 * <b>Payload Size (uint32_t) | Timestamp (dwTime_t) | Payload</b>
 *
 * It is the responsiblility of the plugin author to do perform the memory allocation
 * for the memory buffer that will be given out via this API.
 *
 * Please note the following considerations on the behavior of this API:
 *
 *     1. This API may be called several times before a call to 'dwSensorPlugin_returnRawData',
 *        which means your plugin implementation must support multiple "raw data" buffers in flight.
 *     2. The buffer given out by this API will be returned by a call to '_dwSensorPlugin_returnRawData()'
 *     3. The size reported as the output of this API shall include the header shown above
 *
 * @note To support the multiple buffers in flight behavior described again, a reference BufferPool data
 *       structure implementation is released with the plugin samples.
 *
 *
 * @param[out] data A pointer to the pointer to data that is populated with the RAW data.
 * @param[out] size A pointer to the size of the data array.
 * @param[out] timestamp Specifies the host timestamp of raw data message.
 * @param[in] timeout_us Specifies the timeout in us to wait before unblocking.
 * @param[in] handle Specifies the sensor handle to read from.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality. <br>
 *         DW_CALL_NOT_ALLOWED - if sensor cannot execute the call, for example, due to decoding data. <br>
 *         DW_INVALID_ARGUMENT - if one of the given arguments is invalid.<br>
 *         DW_TIME_OUT - if the requested timed out.<br>
 *         DW_SAL_SENSOR_ERROR - if there was an unrecoverable i/o error.<br>
 *         DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorPlugin_readRawData)(uint8_t const** data, size_t* size, dwTime_t* timestamp,
                                               dwTime_t timeout_us, dwSensorPluginSensorHandle_t handle);

/**
 * Returns RAW data to sensor as a byte array. The returned pointer must have been
 * previously obtained by a call to the 'dwSensorPlugin_readRawData' interface.
 *
 * @param[in] data A pointer to data that was populated with the RAW data.
 * @param[in] handle Specifies the sensor the data came from.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_CALL_NOT_ALLOWED - if sensor cannot execute the call, for example, due to decoding data. <br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality. <br>
 *         DW_INVALID_ARGUMENT - if given data pointer is invalid.<br>
 *         DW_SUCCESS
 */
typedef dwStatus (*dwSensorPlugin_returnRawData)(uint8_t const* data, dwSensorPluginSensorHandle_t handle);

/**
 * Pushes raw data obtained from a previous 'dwSensorPlugin_readRawData' call for decoding.
 *
 * Depending on the sensor implementation, actual decoding may happen synchronously on this call, or on a subsequently
 * 'dwSensorPlugin_pushData' call when enough raw data has been received.
 *
 * @param[out] lenPushed A pointer to the amount of data that was successfully pushed to the plugin
 * @param[in] data A pointer to data that was populated with the RAW data.
 * @param[in] size Size of the data to be pushed
 * @param[in] handle Specifies the sensor the data came from.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_CALL_NOT_ALLOWED - if sensor cannot execute the call, for example, due to decoding data. <br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality. <br>
 *         DW_INVALID_ARGUMENT - if given data pointer is invalid.<br>
 *         DW_SUCCESS
 *
 * @note CAN sensor plugins do not need to provide this method, as decoding happens by DriveWorks sensor layer. The corresponding
 *       entry in 'dwSensorCommonPluginFunctions' can be null.
 */
typedef dwStatus (*dwSensorPlugin_pushData)(size_t* lenPushed, uint8_t const* data, size_t const size, dwSensorPluginSensorHandle_t handle);

/**
 * Gets information of this sensor.
 *
 * @param[out] information Information struct for this sensor.
 * @param[in] sensor Specifies the sensor to which the constants are related.
 *
 * @return DW_FAILURE, DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorPlugin_getSensorInformation)(dwSensorPlugin_information* information, dwSensorPluginSensorHandle_t sensor);

/**
 * Reads RAW data from different level
 *
 *
 * @param[in] rawData A pointer to the pointer to data that is populated with the RAW data.
 * @param[out] size A pointer to the size of the data array.
 * @param[in] level Specify which raw data level to be read from.
 * @param[in] groupNum Additonal param to specify the group to read from since there may be different groups of data per level,
 *                  for example, we can pass service ID(soda/ssi) to specify the service group to read from for a LEVEL_TWO raw data,
 *                  if -1 passed, the data from all the group will be returned
 * @param[out] data A pointer to the size of the data array.
 * @param[in] handle Specifies the sensor handle to read from.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality. <br>
 *         DW_NOT_READY - if sensor cannot generate a combine raw packet for a sepcific level. <br>
 *         DW_CALL_NOT_ALLOWED - if sensor cannot execute the call, for example, due to decoding data. <br>
 *         DW_INVALID_ARGUMENT - if one of the given arguments is invalid.<br>
 *         DW_TIME_OUT - if the requested timed out.<br>
 *         DW_SAL_SENSOR_ERROR - if there was an unrecoverable i/o error.<br>
 *         DW_SUCCESS
 *
 */
DW_DEPRECATED("dwSensorPlugin_getRawPackets is deprecated and will be removed in next major release. Please use dwSensorPlugin_getRawPackets instead")
typedef dwStatus (*dwSensorPlugin_getRawPackets)(uint8_t const* const rawData, size_t* const size,
                                                 dwSensorPlugin_rawDataLevel level, int32_t groupNum, uint8_t const** const data, dwSensorPluginSensorHandle_t handle);

/**
 * Reads RAW data from different level
 *
 *
 * @param[in] rawData A pointer to the pointer to data that is populated with the RAW data.
 * @param[out] size A pointer to the size of the data array.
 * @param[in] level Specify which raw data level to be read from.
 * @param[in] groupNum Additonal param to specify the group to read from since there may be different groups of data per level,
 *                  for example, we can pass service ID(soda/ssi) to specify the service group to read from for a LEVEL_TWO raw data,
 *                  if -1 passed, the data from all the group will be returned
 * @param[out] data A pointer to the size of the data array.
 * @param[out] extra1 used to return extra info to client, for example soda/ssi ID.
 * @param[out] extra2 used to return extra info to client, for example someip intance ID which used to sort the split PDU.
 * @param[in] handle Specifies the sensor handle to read from.
 *
 * @return DW_INVALID_HANDLE - if the sensor handle is NULL or invalid <br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality. <br>
 *         DW_NOT_READY - if sensor cannot generate a combine raw packet for a sepcific level. <br>
 *         DW_CALL_NOT_ALLOWED - if sensor cannot execute the call, for example, due to decoding data. <br>
 *         DW_INVALID_ARGUMENT - if one of the given arguments is invalid.<br>
 *         DW_TIME_OUT - if the requested timed out.<br>
 *         DW_SAL_SENSOR_ERROR - if there was an unrecoverable i/o error.<br>
 *         DW_SUCCESS
 *
 */

typedef dwStatus (*dwSensorPlugin_getRawPacketsNew)(uint8_t const* const rawData, size_t* const size,
                                                    dwSensorPlugin_rawDataLevel level, int32_t groupNum, uint8_t const** const data, int32_t* extra1, int32_t* extra2, dwSensorPluginSensorHandle_t handle);
/**
 * API for sensor that raw data and decoded packet are many to one relationship.
 * Push raw data segments to sensor plugin and check if the raw data is ready for decode.
 *
 * @param[out] offset for a packet
 * @param[out] size for a complete packet
 * @param[in] data raw TP segment data.
 * @param[in] ctx context handle.
 *
 * @return DW_INVALID_ARGUMENT - if pointer to the sensor handle is NULL. <br>
 *         DW_NOT_READY - if sensor cannot generate a combine raw packet. <br>
 *         DW_SUCCESS
 *
 */
typedef dwStatus (*dwSensorPlugin_rawDataReadyForDecode)(size_t* offset, size_t* size, uint8_t const* const data, dwSensorPluginSensorHandle_t ctx);

/// Function Table exposing common plugin functions
typedef struct
{
    dwSensorPlugin_createHandle createHandle;
    dwSensorPlugin_createSensor createSensor;
    dwSensorPlugin_release release;
    dwSensorPlugin_start start;
    dwSensorPlugin_stop stop;
    dwSensorPlugin_reset reset;
    dwSensorPlugin_readRawData readRawData;
    dwSensorPlugin_returnRawData returnRawData;
    dwSensorPlugin_pushData pushData;
    dwSensorPlugin_getSensorInformation getSensorInformation;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    dwSensorPlugin_getRawPackets getRawPackets;
#pragma GCC diagnostic pop
    dwSensorPlugin_getRawPacketsNew getRawPacketsNew;
    /// Funciton pointer used to check if raw data is ready for decode
    dwSensorPlugin_rawDataReadyForDecode rawDataReadyForDecode;
} dwSensorCommonPluginFunctions;

/** @} */

#ifdef __cplusplus
}
#endif

#endif

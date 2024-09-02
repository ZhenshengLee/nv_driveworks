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
 * <b>NVIDIA DriveWorks API: Sensors</b>
 *
 * @b Description: This file defines the sensor abstraction layer (SAL).
 */

/**
 * @defgroup sensors_group Sensors Interface
 *
 * @brief Defines sensor access available on the given hardware.
 *
 */

/**
 * @defgroup sensors_common_group Common
 *
 * @brief Defines methods common to all sensors.
 *
 * @ingroup sensors_group
 * @{
 */

#ifndef DW_SENSORS_COMMON_SENSORS_H_
#define DW_SENSORS_COMMON_SENSORS_H_

#include <dw/core/base/Exports.h>
#include <dw/core/context/Context.h>
#include <dw/core/health/HealthSignals.h>
#include <dw/core/signal/SignalStatus.h>

#include "SensorTypes.h"
#include "SensorStats.h"
#include "SensorsVehicleStateTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates and initializes a SAL (sensor abstraction layer) module.
 *
 * @param[out] sal A pointer to the SAL handle will be returned here.
 * @param[in] context Specifies the handle to the context under which the SAL module is created.
 *
 * @return DW_INVALID_ARGUMENT - if pointer to the SAL handle is NULL. <br>
 *         DW_INVALID_HANDLE - if provided context handle is invalid. <br>
 *         DW_SUCCESS - if call is successful.
 */
DW_API_PUBLIC
dwStatus dwSAL_initialize(dwSALHandle_t* const sal, dwContextHandle_t const context);

/**
 * @brief Resets the SAL (sensor abstraction layer) module. This method resets all sensors managed by this module.
 *
 * @param[in] sal Specifies the SAL handle to reset.
 *
 * @return DW_INVALID_HANDLE - if provided SAL handle is invalid. <br>
 *         DW_ERROR - error of a sensor that could not be reset. <br>
 *         DW_SUCCESS - if call is successful. <br>
 */
DW_API_PUBLIC
dwStatus dwSAL_reset(dwSALHandle_t const sal);

/**
 * @brief Releases the SAL (sensor abstraction layer) module. This method releases all created sensors and unloads all drivers.
 *
 * @note This method renders the SAL handle unusable.
 *
 * @param[in] sal The SAL handle to be released.
 *
 * @return DW_INVALID_HANDLE - if provided SAL handle is invalid. <br>
 *         DW_SUCCESS - if call is successful.
 *
 */
DW_API_PUBLIC
dwStatus dwSAL_release(dwSALHandle_t const sal);

/**
 * @brief Creates a new sensor managed by the SAL module with the given parameters.
 *
 * @param[out] sensor A pointer to sensor handle that became valid after creation.
 * @param[in] params Specifies the parameters for sensor creation.
 * @param[in] sal Specifies the handle to the SAL module that will manage the sensor.
 *
 * @return DW_INVALID_ARGUMENT - if pointer to the sensor is NULL or protocol name is NULL <br>
 *         DW_INVALID_HANDLE - if provided SAL handle is invalid. <br>
 *         DW_SAL_NO_DRIVER_FOUND - if provided protocol name is unknown. <br>
 *         DW_SAL_SENSOR_ERROR - if a non recoverable error happens during sensor creation. For a virtual
 *                               sensor which is reading from a file it could be file I/O error. <br>
 *         DW_BUFFER_FULL - if internal buffer has insufficient capacity to hold parsed value from params. <br>
 *         DW_SUCCESS - if call is successful.
 * @note This returned sensor handle must be released using the 'dwSAL_releaseSensor()' method.
 */
DW_API_PUBLIC
dwStatus dwSAL_createSensor(dwSensorHandle_t* const sensor, dwSensorParams const params, dwSALHandle_t const sal);

/**
 * @brief Releases a sensor managed by the SAL module.
 *
 * @note This method renders the sensor handle unusable.
 *
 * @param[in] sensor The handle to a sensor created previously with 'dwSAL_createSensor()'.
 *
 * @return DW_INVALID_HANDLE - if provided SAL or sensor handle is invalid. <br>
 *         DW_SAL_SENSOR_ERROR - if provided sensor is not managed by this SAL. <br>
 *         DW_SUCCESS - if call is successful.
 * @note In cases where a specific sensor needs to be released, this method can be used.
 *       In all other cases, the automatic sensor release of the 'dwSAL_release()' method
 *       is sufficient.
 */
DW_API_PUBLIC
dwStatus dwSAL_releaseSensor(dwSensorHandle_t const sensor);

/**
 * @brief Bootstraps all sensors managed by the SAL module.
 * 
 * @param[in] sal Specifies the SAL handle created with 'dwSAL_initialize()'.
 *
 * @return DW_INVALID_HANDLE - if provided SAL is invalid. <br>
 *         DW_NOT_INITIALIZED if the sensor has no initialized component. <br>
 *         DW_SUCCESS if call is successful.
 * @note Sensor streaming will not start until dwSensor_start. Calling this API after sensor start will be a NO-OP,
 *       as sensor start of any sensor will implicitly bootstrap all sensors.
 */
DW_API_PUBLIC
dwStatus dwSAL_start(dwSALHandle_t const sal);

/**
 * @brief Gets current platform the SDK is running on.
 *
 * @param[out] os A pointer to the identifier of the platform the SDK is running on.
 * @param[in] sal Specifies the SAL handle created with 'dwSAL_initialize()'.
 *
 * @return DW_INVALID_HANDLE - if provided SAL or sensor handle is invalid. <br>
 *         DW_SUCCESS - if call is successful.
 *
 */
DW_API_PUBLIC
dwStatus dwSAL_getPlatform(dwPlatformOS* const os, dwSALHandle_t sal);

/**
 * @brief Gets detailed information about the running hardware platform and operating system.
 *
 * @param[out] osName A pointer to the pointer of the return location; short string of the name of the current platform.
 * @param[in] os Specifies the identifier of the platform you are querying for.
 * @param[in] sal Specifies the SAL handle created with 'dwSAL_initialize()'.
 *
 * @return DW_INVALID_HANDLE - if provided SAL or sensor handle is invalid <br>
 *         DW_SUCCESS - if call is successful.
 *
 */
DW_API_PUBLIC
dwStatus dwSAL_getPlatformInfo(char8_t const** const osName, dwPlatformOS const os, dwSALHandle_t const sal);

/**
 * @brief Gets number of available sensors for a platform.
 *
 * @note Only sensor queried for the current platform can be created by using dwSAL_createSensor().
 *
 * @param[out] num A pointer to return number of sensor in the SAL.
 * @param[in] os Specifies the identifier of the platform you are querying for.
 * @param[in] sal Specifies the SAL handle created with 'dwSAL_initialize()'.
 *
 * @return DW_INVALID_HANDLE - if provided SAL or sensor handle is invalid. <br>
 *         DW_INVALID_ARGUMENT - if provided pointer is invalid. <br>
 *         DW_SUCCESS - if call is successful.
 *
 */
DW_API_PUBLIC
dwStatus dwSAL_getNumSensors(uint32_t* const num, dwPlatformOS const os, dwSALHandle_t const sal);

/**
 * @brief Gets protocol name of a sensor for a given index, e.g., 'camera.gmsl' or 'can.file'.
 *
 * @param[out] name A pointer to the pointer of the null terminated string will be returned here.
 * @param[in] idx Specifies the index of a sensor; index must be between 0 and 'dwSAL_getNumSensors()-1'.
 * @param[in] os Specifies the identifier of the platform you query.
 * @param[in] sal Specifies the SAL handle created with 'dwSAL_initialize()'.
 *
 * @return DW_INVALID_HANDLE - if provided SAL or sensor handle is invalid. <br>
 *         DW_INVALID_ARGUMENT - if provided pointer is invalid or the index outside of range. <br>
 *         DW_SUCCESS - if call is successful.
 */
DW_API_PUBLIC
dwStatus dwSAL_getSensorProtocol(const char** const name, uint32_t const idx, dwPlatformOS const os, dwSALHandle_t const sal);

/**
 * @brief Gets the parameter string acceptable by a sensor whose type is determined by the index.
 * A sensor generates a human readable string with parameters it accepts.
 *
 * @param[out] parameters A pointer to the pointer to the null terminated string will be returned here.
 * @param[in] idx Specifies the index of a sensor; index must be between 0 and 'dwSAL_getNumSensors()-1'.
 * @param[in] os Specifies the identifier of the platform you query.
 * @param[in] sal Specifies the SAL handle created with 'dwSAL_initialize()'.
 *
 * @return DW_INVALID_HANDLE - if provided SAL or sensor handle is invalid. <br>
 *         DW_INVALID_ARGUMENT - if provided pointer is invalid or index outside of range. <br>
 *         DW_SUCCESS - if call is successful.
 */
DW_API_PUBLIC
dwStatus dwSAL_getSensorParameterString(const char** const parameters, uint32_t const idx,
                                        dwPlatformOS const os, dwSALHandle_t const sal);

/**
 * @brief Pass vehicle state to all sensors.
 *
 * @param[in] vehicleState Vehicle state that needs to be passed in the structure of dwSensorVehicleState .
 * @param[in] sal Specifies the SAL handle created with 'dwSAL_initialize()'
 *
 * @return DW_INVALID_ARGUMENT - no function with that pointer exists<br>
 *         DW_INVALID_HANDLE - if provided SM handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality. <br>
 *         DW_SUCCESS - if call is successful.
 */
DW_API_PUBLIC
dwStatus dwSAL_setVehicleState(dwSensorVehicleState* const vehicleState, dwSALHandle_t const sal);

/**
 * @brief Pass egomotion state to all sensors.
 *
 * @param[in] egomotionState Egomotion state that needs to be passed in the structure of dwEgomotionVehicleState .
 * @param[in] sal Specifies the SAL handle created with 'dwSAL_initialize()'
 *
 * @return DW_INVALID_ARGUMENT - no function with that pointer exists<br>
 *         DW_INVALID_HANDLE - if provided SM handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality. <br>
 *         DW_SUCCESS - if call is successful.
 */
DW_API_PUBLIC
dwStatus dwSAL_setEgomotionState(dwSensorEgomotionState* const egomotionState, dwSALHandle_t const sal);

/**
 * @brief Starts the sensor previously successfully created with 'dwSAL_createSensor()'.
 * After a sensor is successfully started, new sensor data can be acquired using
 * corresponding sensor methods.
 *
 * @note This method might spawn a thread, depending on the sensor implementation.
 *       It is, however, guaranteed that the data returned by the sensor is valid
 *       in the calling thread. For example, a CUDA camera image is created in the same
 *       CUDA context as the callee.
 *
 * @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_CUDA_ERROR - if the underlying sensor driver had a CUDA error. <br>
 *         DW_NVMEDIA_ERROR - if underlying sensor driver had an NvMedia error. <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_END_OF_STREAM - if end of stream reached. <br>
 *         DW_SUCCESS - if call is successful.
 *
 **/
DW_API_PUBLIC
dwStatus dwSensor_start(dwSensorHandle_t const sensor);

/**
 * @brief Stops the sensor. The method blocks while the sensor is stopped.
 *
 * @param[in] sensor Specifies the sensor handle of a sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_CUDA_ERROR - if the underlying sensor driver had a CUDA error. <br>
 *         DW_NVMEDIA_ERROR - if underlying sensor driver had an NvMedia error. <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_SUCCESS - if call is successful.
 **/
DW_API_PUBLIC
dwStatus dwSensor_stop(dwSensorHandle_t const sensor);

/**
 * @brief Resets the sensor. The method blocks while the sensor is reset.
 * 
 * @param[in] sensor Specifies the sensor handle of a sensor previously created with 'dwSAL_createSensor()'.
 *
 * @return DW_CUDA_ERROR - if the underlying sensor driver had a CUDA error. <br>
 *         DW_NVMEDIA_ERROR - if underlying sensor driver had an NvMedia error. <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_SUCCESS - if call is successful.
 **/
DW_API_PUBLIC
dwStatus dwSensor_reset(dwSensorHandle_t const sensor);

/**
 * @brief Reads RAW data from sensor as byte array at a given time (TimeoutUs).
 *
 * @param[out] data A pointer to the pointer to data that is populated with the RAW data.
 * @param[out] size A pointer to the size of the data array.
 * @param[in] timeoutUs Specifies the timeout in US to wait before unblocking.
 * @param[in] sensor Specifies the sensor to read from.
 *
 * @return DW_INVALID_HANDLE - if provided sensor handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality. <br>
 *         DW_END_OF_STREAM - if currently no new RAW data is available. <br>
 *         DW_CALL_NOT_ALLOWED - if sensor cannot execute the call, for example, due to decoding data. <br>
 *         DW_INVALID_ARGUMENT - if one of the given arguments is invalid.<br>
 *         DW_TIME_OUT - if the requested timed out.<br>
 *         DW_SAL_SENSOR_ERROR - if there was an unrecoverable i/o error.<br>
 *         DW_SUCCESS - if call is successful.
 *
 * @note On an I/O error a new read attempt will start before the failed location. Hence on file systems,
 *       such as network based file system, a new attempt to read data can be taken. This call must be followed by 'dwSensor_returnRawData'. A sensor might provide access to raw data (i.e., unprocessed as received
 *       from hardware) data for an extended usage. Such data can for example be passed to serializer.
 */
DW_API_PUBLIC
dwStatus dwSensor_readRawData(const uint8_t** const data, size_t* const size,
                              dwTime_t const timeoutUs,
                              dwSensorHandle_t const sensor);

/**
 * @brief Returns RAW data to sensor as a byte array. This call must be preceded by
 * 'dwSensor_readRawData'.
 *
 * @param[in] data A pointer to data that was populated with the RAW data.
 * @param[in] sensor Specifies the sensor the data came from.
 *
 * @return DW_INVALID_HANDLE - if provided sensor handle is invalid. <br>
 *         DW_CALL_NOT_ALLOWED - if sensor cannot execute the call, for example, due to decoding data. <br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality. <br>
 *         DW_INVALID_ARGUMENT - if given data pointer is invalid.<br>
 *         DW_SUCCESS - if call is successful.
 */
DW_API_PUBLIC
dwStatus dwSensor_returnRawData(const uint8_t* const data, dwSensorHandle_t const sensor);

/**
 * @brief Retrieves the valid range of seek parameters, for sensors that support seeking.
 *
 * @param[out] eventCount Number of events available to be sought to. Each event represents one data unit
 *                        returned by the sensor from its read() method, e.g. one camera frame for camera,
 *                        one CAN message, for a CAN sensor, etc.
 * @param[out] startTimestampUs Timestamp of the very first event in usec
 * @param[out] endTimestampUs Timestamp of the very last event in usec
 * @param[in] sensor Handle to the sensor which seek table to write out
 *
 * @return DW_INVALID_HANDLE - if provided sensor handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality, i.e. no seeking <br>
 *         DW_NOT_AVAILABLE - if sensor supports seeking, but no seeking information is available <br>
 *         DW_SUCCESS - if call is successful.
 **/
DW_API_PUBLIC
dwStatus dwSensor_getSeekRange(size_t* const eventCount,
                               dwTime_t* const startTimestampUs, dwTime_t* const endTimestampUs,
                               dwSensorHandle_t const sensor);

/**
 * @brief Gets the current seek position of the sensor. The event index corresponding to the next
 * sensor event which will be read with sensor's read methods.
 *
 * @param[out] event Event number which in the range [0;eventCount) of the next event.
 * @param[in] sensor Handle to the sensor which seek table to write out
 *
 * @return DW_INVALID_HANDLE - if provided sensor handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality, i.e. no seeking <br>
 *         DW_NOT_AVAILABLE - if sensor supports seeking, but no seeking information is available <br>
 *         DW_END_OF_STREAM - if next read position is end of stream <br>
 *         DW_SUCCESS - if call is successful.
 **/
DW_API_PUBLIC
dwStatus dwSensor_getCurrentSeekPosition(size_t* const event, dwSensorHandle_t const sensor);

/**
 * @brief Returns a pointer to the shared buffer with the contents of the sensor header and bufferSize
 * will contain the number of valid bytes in the buffer.
 *
 * @param[out] buffer A pointer to the buffer owned by sensor which holds the header information.
 * @param[out] bufferSize A pointer to variable indicating number of valid bytes in the buffer
 * @param[in]  sensor Handle to the sensor to which the header info is requested
 *
 * @return DW_INVALID_HANDLE - if given sensor handle is invalid
 *         DW_INVALID_ARGUMENTS - if given arguments are invalid
 *         DW_SUCCESS - if call is successful.
 */
dwStatus dwSensor_getHeader(uint8_t const** const buffer, size_t* bufferSize, dwSensorHandle_t const sensor);

/**
 * @brief Seeks the sensor to a specific timestamp, for sensors that support seeking.
 * Next readFrame(), readMessage(), etc. method would start to read from an event which timestamp is greater
 * or equal to the seek timestamp.
 *
 * @param[in] timestampUs Timestamp in usec to seek sensor to. Must be in the range [startTimestamp_us; endTimestamp_us]
 * @param[in] sensor Handle to the sensor which seek table to write out
 *
 * @return DW_INVALID_HANDLE - if provided sensor handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality, i.e. no seeking <br>
 *         DW_NOT_AVAILABLE - if sensor supports seeking, but no seeking information is available <br>
 *         DW_INVALID_ARGUMENT - if given timestamp is outside the supported range<br>
 *         DW_SUCCESS - if call is successful.
 * @see dwSensor_getSeekRange()
 **/
DW_API_PUBLIC
dwStatus dwSensor_seekToTime(dwTime_t const timestampUs, dwSensorHandle_t const sensor);

/**
 * @brief Seeks the sensor to an event, for sensors that support seeking.
 * Next readFrame(), readMessage(), etc. method would start to read from this event.
 *
 * @param[in] event Number of the event to seek to. Must be in the range [0; eventCount)
 * @param[in] sensor Handle to the sensor which seek table to write out
 *
 * @return DW_INVALID_HANDLE - if provided sensor handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality, i.e. no seeking <br>
 *         DW_NOT_AVAILABLE - if sensor supports seeking, but no seeking information is available <br>
 *         DW_INVALID_ARGUMENT - if given event number is above available number of events<br>
 *         DW_SUCCESS - if call is successful.
 * @see dwSensor_getSeekRange()
 **/
DW_API_PUBLIC
dwStatus dwSensor_seekToEvent(size_t const event, dwSensorHandle_t const sensor);

/**
 * @brief Forces recreation of the seek table, for sensors that support seeking.
 *
 * @param[in] sensor Handle to the sensor which seek table to write out
 *
 * @return DW_INVALID_HANDLE - if provided sensor handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality, i.e. no seeking <br>
 *         DW_INVALID_ARGUMENT - if given fileName is invalid <br>
 *         DW_NOT_AVAILABLE - if sensor data contains non-monotonic timestamps <br>
 *         DW_SAL_SENSOR_ERROR - if a non recoverable error happens during table creation,
 *                               for example file I/O error. <br>
 *         DW_SUCCESS - if call is successful.
 * @note This method will perform dynamic memory allocations, hence should not be used during run-time.
 *       A sensor will be reset to the very first event after table creation.
 **/

DW_API_PUBLIC
dwStatus dwSensor_createSeekTable(dwSensorHandle_t const sensor);

/**
 * @brief Get the number of entries in the seek Table.
 *
 * @param[out] size A pointer to the size of the seek table
 * @param[in] hsensor Handle to the sensor to get size of seek table entry
 *
 * @return DW_INVALID_HANDLE - if provided sensor handle is invalid. <br>
 *         DW_SUCCESS - if call is successful.
 *
 * @note Use this method after calling dwSensor_createSeekTable
 **/
DW_API_PUBLIC
dwStatus dwSensor_getNumSeekTableEntries(size_t* const size, dwSensorHandle_t const hsensor);

/**
 * @brief Fill in the pre-allocated dwSensorSeekTableEntry array
 *
 * @param[out] entries Filled in array with seekTable entries
 * @param[in] numEntries The number of entries allocated in the entries array
 * @param[in] sensor Handle to the sensor which seek table to write out
 *
 * @return  DW_INVALID_HANDLE - if provided sensor handle is invalid. <br>
 *          DW_NOT_SUPPORTED - if sensor does not implement this functionality, i.e. no seeking <br>
 *          DW_INVALID_ARGUMENT - if the table provided is not large enough to fit the entire table <br>
 *          DW_SUCCESS - if call is successful.
 *
 **/
DW_API_PUBLIC
dwStatus dwSensor_getSeekTableEntries(dwSensorSeekTableEntry* const entries, size_t const numEntries, dwSensorHandle_t const sensor);

/**
 * @brief Saves the seek table for the sensor to a file, for sensors that support seek tables.
 * A seek table is available if a virtual sensor has been requested to create/load one
 * during creation. Refer to each individual sensor to see what parameters need to be passed to
 * instantiate or load seek table.
 *
 * It is guaranteed that any seek table written out by this method can be parsed by
 * the corresponding virtual sensor.
 *
 * @param[in] fileName Name of the file under a reachable path to write seek table to.
 * @param[in] sensor Handle to the sensor which seek table to write out
 *
 * @return DW_INVALID_HANDLE - if provided sensor handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality, i.e. no seeking <br>
 *         DW_FILE_NOT_FOUND - if cannot open the file for writing <br>
 *         DW_END_OF_STREAM - if there was a problem while writing out the file, for example disk full <br>
 *         DW_INVALID_ARGUMENT - if given fileName is invalid <br>
 *         DW_NOT_AVAILABLE - if no seek table is currently available for the sensor<br>
 *         DW_SUCCESS - if call is successful.
 * @note Most of the virtual sensors expect their seek table to be stored next to the data file, having
 *       same name as the data file with an additional extension .seek, e.g. '/path/canbus.can.seek'
 **/
DW_API_PUBLIC
dwStatus dwSensor_saveSeekTable(const char* const fileName, dwSensorHandle_t const sensor);

/**
 * @brief Sets the priority of the internal thread, for sensors that use
 * an internal thread to communicate to the OS drivers. This method
 * sets the priority of the internal thread, if the sensor implementation supports it.
 *
 * @param[in] priority Priority of the thread to set. In general the priority is in range [1;99]
 * @param[in] sensor Handle to the sensor to set thread priority
 *
 * @return DW_INVALID_HANDLE - if provided sensor handle is invalid.<br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality, i.e. sensor does not has an internal thread
 *                            or priority of the internal thread cannot be changed<br>
 *         DW_INVALID_ARGUMENT - if given priority number is outside of the valid range<br>
 *         DW_NOT_AVAILABLE - if sensor is not running yet, i.e. sensor must be started before change can happen<br>
 *         DW_CALL_NOT_ALLOWED - if setting thread priority is not allowed now, for example due to wrong privileges<br>
 *         DW_INTERNAL_ERROR - an internal error indicating unrecoverable error, in general this should not happen<br>
 *         DW_SUCCESS - if call is successful.
 * @note If multiple sensors have been created from same physical sensor, the thread priority will be applied
 *       to the thread communicating with the phyiscal device, hence affecting all these sensors.
 **/
DW_API_PUBLIC
dwStatus dwSensor_setThreadPriority(int32_t const priority, dwSensorHandle_t const sensor);

/**
 * @brief Sets the affinity of the internal thread, for sensors that use
 * an internal thread to communicate to the OS drivers.
 *
 * @param[in] affinityMask Bit mask setting 1 to CPU which shall execute sensor thread
 * @param[in] sensor Handle to the sensor to set thread priority
 *
 * @return DW_INVALID_HANDLE - if provided sensor handle is invalid.<br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality, i.e. sensor does not has an internal thread
 *                            or priority of the internal thread cannot be changed<br>
 *         DW_INVALID_ARGUMENT - if given affinity mask is outside of the valid range, i.e. points to invalid CPU<br>
 *         DW_NOT_AVAILABLE - if sensor is not running yet, i.e. sensor must be started before change can happen<br>
 *         DW_CALL_NOT_ALLOWED - if setting thread affinity is not allowed now, for example due to wrong privileges<br>
 *         DW_INTERNAL_ERROR - an internal error indicating unrecoverable error, in general this should not happen<br>
 *         DW_SUCCESS - if call is successful.
 * @note If multiple sensors have been created from same physical sensor, the thread affinity will be applied
 *       to the thread communicating with the phyiscal device, hence affecting all these sensors.
 **/

DW_API_PUBLIC
dwStatus dwSensor_setThreadAffinity(uint32_t const affinityMask, dwSensorHandle_t const sensor);

/**
 * @brief Gets sensor statistics (if available).
 *
 * @param[out] stats A pointer to a structure containing the statistics.
 * @param[in] sensor Handle to the sensor
 *
 * @return DW_INVALID_HANDLE - if provided sensor handle is invalid.<br>
 *         DW_NOT_SUPPORTED - if sensor does not implement this functionality<br>
 *         DW_NOT_AVAILABLE - if attempted to get stats from a running sensor<br>
 *         DW_SUCCESS - if call is successful.
 **/
DW_API_PUBLIC
dwStatus dwSensor_getStats(dwSensorStats* const stats, dwSensorHandle_t const sensor);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_SENSORS_COMMON_SENSORS_H_

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
// SPDX-FileCopyrightText: Copyright (c) 2017-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: SensorManager</b>
 *
 * @b Description: This file defines the SensorManager interface.
 */

/**
 * @defgroup sensormanager_group SensorManager
 * @ingroup sensors_group
 *
 * @brief Defines sensor management interface layer
 * @{
 */
#ifndef DW_SENSORS_SENSORMANAGER_H_
#define DW_SENSORS_SENSORMANAGER_H_

#include <dw/rig/Rig.h>
#include <dw/sensors/Sensors.h>
#include <dw/sensors/camera/Camera.h>
#include <dw/sensors/imu/IMU.h>
#include <dw/sensors/lidar/Lidar.h>
#include <dw/sensors/gps/GPS.h>
#include <dw/sensors/canbus/CAN.h>
#include <dw/sensors/radar/Radar.h>
#include <dw/sensors/lidar/Lidar.h>
#include <dw/sensors/data/Data.h>

#include "SensorManagerConstants.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwSensorManagerObject* dwSensorManagerHandle_t; //!< Handle of sensor manager.

/// Not available as of current release. Will be added in future releases.
typedef struct dwUltrasonicEnvelope dwUltrasonicEnvelope;

/// Structure for returning data upon any sensor event
typedef struct
{
    /// Type of sensor providing data for this event.
    dwSensorType type;

    /// Index of the given sensor as defined by the order in which it was created
    /// and the type of sensor it is (ie Camera and Lidar can have sensorTypeIndex 0
    /// because they are different sensor types.
    uint32_t sensorTypeIndex;

    /// Timestamp (us)
    dwTime_t timestamp_us;

    /// The index of the sensor as defined by the order in which it was created.  If the
    /// sensor manager was created via the rig configuration file this will match that sensor index.
    /// If it was not created via the rig configuration will be the number of sensors already created.
    uint32_t sensorIndex;

    /// If the event is a multisensor event then the sensor indices of all sensors involved will be stored here
    uint32_t sensorIndices[DW_SENSORMANGER_MAX_CAMERAS];

    /// Data accessor for camera. Special case as can be multisensor event. Frames from cameras related
    /// to single group will be stored here.
    uint32_t numCamFrames;                                        //!< The number of camera frames.
    dwCameraFrameHandle_t camFrames[DW_SENSORMANGER_MAX_CAMERAS]; //!< An array saving all created camera's frame in this event. Users get the frame by 'sensorTypeIndex'.

    // Data accessor for all other sensors
    dwGPSFrame gpsFrame;                         //!< GPS frame in this event.
    dwIMUFrame imuFrame;                         //!< IMU frame in this event.
    dwCANMessage canFrame;                       //!< CAN message frame in this event.
    const dwRadarScan* radFrame;                 //!< Radar scan in this event.
    const dwLidarDecodedPacket* lidFrame;        //!< Lidar decoded packet in this event.
    const dwDataPacket* dataFrame;               //!< dwDataPacket(see reference 15) in this event.
    const uint8_t* rawData;                      //!< Raw data in this event.
    size_t rawDataSize;                          //!< Raw data size in this event.
    const dwUltrasonicEnvelope* ultrasonicFrame; // Ultrasonic envelope in this event.
} dwSensorEvent;

/**
 * @brief Data mode in this unit, which determines whether read raw data.
 *
 */
enum dwSensorManagerDataMode
{
    DW_SENSOR_MANAGER_DATA_MODE_NON_RAW = 0, //!< Sensor manager doesn't read raw data, but just offer decoded packet for users.
    DW_SENSOR_MANAGER_DATA_MODE_RAW     = 1  //!< Sensor manager will read only raw data.
};

/**
 * @brief Parameters for dispatcher, used to define some behaviors of dispatcher.
 *
 */
typedef struct
{
    /// Whether Dispatcher shall accumulated frames from all cameras into a single event
    bool accumCamFrames;
    /// Up to how much can timestamps of camera frames accumulated in a single event differ, in microsecond.
    dwTime_t camFramesTimeDiffLimit;
    /// Timeout value to be used in dispatcher mode for virtual files, in microsecond.
    dwTime_t timeout;
} dwDispatcherParams;

/**
 * @brief Parameters for sensor manager, used to create a sensor manager.
 *
 */
typedef struct
{
    /// Parameters to configure dispatcher mode
    dwDispatcherParams dispatcherParams;

    /// List of sensors indices to be enabled during initialization (i.e. whitelist)
    uint32_t enableSensors[DW_SENSORMANGER_MAX_NUM_SENSORS];

    /// Number of entries in the 'enableSensors' list
    /// @note if this number is 0 all sensors will be enabled
    uint32_t numEnableSensors;

    /// Whether to associate virtual cameras to individual 'camera-group's (default), or
    /// to a single 'camera-group' (in case no dedicated per sensor 'camera-group' is specified for
    /// a given camera sensor in its sensor parameters).
    /// Distinct groups for each virtual camera are beneficial if sensors were recorded
    /// within different groups / on different devices (as otherwise potential
    /// time-offsets can be introduced if all virtual cameras are treated as a single group),
    /// but some applications might expect virtual cameras to be associated to a single group
    bool singleVirtualCameraGroup;
} dwSensorManagerParams;

typedef void (*dwSensorManagerDispatcher_t)(const dwSensorEvent*, void*, dwStatus);

/**
 * @brief Creates an instance of SensorManager module.
 *
 * @param[out] sm A pointer to the sm handle will be returned here.
 * @param[in] poolSize Size of the event pool to be allocated. Has to be greater than 0
 * @param[in] sal SAL handle instantiated by the caller
 *
 * @retval DW_INVALID_ARGUMENT if pointer to the SM handle is NULL. <br>
 * @retval DW_INVALID_HANDLE if provided contex/SAL handle is invalid. <br>
 * @retval DW_SUCCESS init successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_initialize(dwSensorManagerHandle_t* sm,
                                    size_t poolSize, dwSALHandle_t const sal);

/**
 * @brief Creates and initializes a SensorManager module, adding all
 *        sensors in the provided Rig Configuration.
 *
 * @param[out] sm A pointer to the sm handle will be returned here.
 * @param[in] rc Rig Configuration handle
 * @param[in] poolSize Size of the event pool to be allocated. Has to be greater than 0
 * @param[in] sal SAL handle instantiated by the caller
 *
 *
 * @retval DW_INVALID_ARGUMENT if pointer to the SM handle or Rig Configuration handle is NULL. <br>
 * @retval DW_INVALID_HANDLE if provided contex/SAL handle is invalid. <br>
 * @retval DW_SAL_NO_DRIVER_FOUND if provided protocol name is unknown. <br>
 * @retval DW_SAL_SENSOR_ERROR if a non recoverable error happens during <br>
 *                               sensor creation. For a virtual sensor <br>
 *                               which is reading from a file it could be <br>
 *                               file I/O error. <br>
 * @retval DW_SUCCESS init successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_initializeFromRig(dwSensorManagerHandle_t* sm,
                                           dwRigHandle_t rc,
                                           size_t poolSize, dwSALHandle_t sal);

/**
 * @brief Creates and initializes a SensorManager module, adding enabled
 *        sensors in the provided Rig Configuration, and configuring
 *        SensorManager according to params provided.
 *
 * @param[out] sm A pointer to the sm handle will be returned here.
 * @param[in] rc Rig Configuration handle
 * @param[in] params Params to configure SensorManager
 * @param[in] poolSize Size of the event pool to be allocated.  Has to be greater than 0
 * @param[in] sal SAL handle instantiated by the caller
 *
 *
 * @retval DW_INVALID_ARGUMENT if pointer to the SM handle is NULL. <br>
 * @retval DW_INVALID_HANDLE if provided contex/SAL handle is invalid. <br>
 * @retval DW_SAL_NO_DRIVER_FOUND if provided protocol name is unknown. <br>
 * @retval DW_SAL_SENSOR_ERROR if a non recoverable error happens during <br>
 *                               sensor creation. For a virtual sensor <br>
 *                               which is reading from a file it could be <br>
 *                               file I/O error. <br>
 * @retval DW_SUCCESS init successfully.
 * @note This method will perform memory allocations.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_initializeFromRigWithParams(dwSensorManagerHandle_t* sm,
                                                     dwRigHandle_t const rc,
                                                     const dwSensorManagerParams* params,
                                                     size_t poolSize, dwSALHandle_t const sal);

/**
 * @brief Releases the SensorManager module by deleting the handle.
 *
 * @param[out] sm The sensor manager handle that needs to be released
 *
 * @retval DW_INVALID_HANDLE if provided sm handle is invalid. <br>
 * @retval DW_SUCCESS release successfully.
 * @note This API implements the function by calling the deleteUniqueCHandle.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_release(dwSensorManagerHandle_t sm);

/**
 * @brief Adds a sensor to the SAL instance. All addSensor()
 *        calls must be completed before the start() call.
 *
 * @param[in] params Specifies the parameters for sensor creation.
 * @param[in] clientData Client data to be added for this sensor
 * @param[in] sm SM handle instantiated by the caller
 *
 * @retval DW_INVALID_HANDLE if provided SM handle is invalid. <br>
 * @retval DW_SAL_NO_DRIVER_FOUND if provided protocol name is unknown. <br>
 * @retval DW_SAL_SENSOR_ERROR if a non recoverable error happens during sensor
 *                               creation. For a virtual sensor which is reading
 *                               from a file it could be file I/O error. <br>
 * @retval DW_SUCCESS add successfully.
 *
 * @note This method will perform memory allocations.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_addSensor(dwSensorParams params, uint64_t clientData,
                                   dwSensorManagerHandle_t sm);

/**
 * @brief Adds a camera sensor to the SAL instance. All addCameraSensor()
 *        calls must be completed before the start() call.
 *
 * @param[in] groupName Specifies the group name for this camera
 * @param[in] siblingIndex Specifies the sibling id for this camera  (GMSL only)
 * @param[in] params Specifies the parameters for sensor creation.
 * @param[in] clientData Client data to be added for this sensor
 * @param[in] sm SM handle instantiated by the caller
 *
 * @retval DW_INVALID_HANDLE if provided SM handle is invalid. <br>
 * @retval DW_SAL_NO_DRIVER_FOUND if provided protocol name is unknown. <br>
 * @retval DW_SAL_SENSOR_ERROR if a non recoverable error happens during sensor
 *                               creation. For a virtual sensor which is reading
 *                               from a file it could be file I/O error. <br>
 * @retval DW_INVALID_ARGUMENT if the sensor parameters are invalid or incomplete. <br>
 * @retval DW_FILE_NOT_FOUND if a specified video or other file parameter cannot be found. <br>
 * @retval DW_SUCCESS add successfully.
 *
 * @note This method will perform memory allocations.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_addCameraSensor(const char8_t* groupName, uint32_t siblingIndex,
                                         dwSensorParams params, uint64_t clientData,
                                         dwSensorManagerHandle_t sm);

/**
 * @brief Starts all sensors. Sensor manager will begin to read data and decode them after calling.
 *
 * @param[in] sm SM handle instantiated by the caller
 *
 * @retval DW_INVALID_HANDLE if provided SM handle is invalid. <br>
 * @retval DW_FAILURE if the sensor already running <br>
 * @retval DW_SUCCESS start successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_start(dwSensorManagerHandle_t sm);

/**
 * @brief Stops all sensors. Sensor manager will stop to read data and decode them after calling.
 *
 * @param[in] sm SM handle instantiated by the caller
 *
 * @retval DW_INVALID_HANDLE if provided SM handle is invalid. <br>
 * @retval DW_FAILURE if the sensor is not running and is tried to be stopped <br>
 * @retval DW_SUCCESS stop successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_stop(dwSensorManagerHandle_t sm);

/**
 * @brief Resets all sensors. All sensors's reset function will be called by the sensor manager.
 *
 * @param[in] sm SM handle instantiated by the caller
 *
 * @retval DW_INVALID_HANDLE if provided SM handle is invalid. <br>
 * @retval DW_FAILURE if the sensor is tried to be reset while it is still running <br>
 * @retval DW_SUCCESS reset successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_reset(dwSensorManagerHandle_t sm);

/**
 * @brief Called by the application to consume the next available sensor
 *        event ready for consumption. The method will return the oldest
 *        available event among all sensors.
 *
 * @param[out] ev A pointer to a pointer of the event to be acquired
 * @param[in] timeoutMicroSeconds time threshold to bail out if no new event is available
 * @param[in] sm SM handle instantiated by the caller
 *
 * @retval DW_INVALID_ARGUMENT pointer to the dwSensorEvent point is null <br>
 * @retval DW_INVALID_HANDLE if provided SM handle is invalid. <br>
 * @retval DW_BUFFER_FULL all event buffers are currently being used <br>
 * @retval DW_TIME_OUT no new event was available within timeout period <br>
 * @retval DW_END_OF_STREAM reached end of all sensor streams <br>
 * @retval DW_SUCCESS acquire successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_acquireNextEvent(const dwSensorEvent** ev, dwTime_t timeoutMicroSeconds,
                                          dwSensorManagerHandle_t sm);

/**
 * @brief Releases a previously acquired event back to the pool. For certain
 *        sensors, this call will also trigger returning the backing buffer
 *        back to the DW sensor bufferpools.
 *
 * @param[in] ev A pointer to the event to be returned
 * @param[in] sm SM handle instantiated by the caller
 *
 * @retval DW_INVALID_ARGUMENT pointer to the event is null <br>
 * @retval DW_INVALID_HANDLE if provided SM handle is invalid. <br>
 * @retval DW_SUCCESS release successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_releaseAcquiredEvent(const dwSensorEvent* ev,
                                              dwSensorManagerHandle_t sm);

/**
 * @brief Gets the number of sensors instantiated for a given sensor type
 *
 * @param[out] count A pointer to return the number of sensors
 * @param[in] type Type of sensor to return the count for
 * @param[in] sm SM handle instantiated by the caller
 *
 * @retval DW_INVALID_ARGUMENT pointer to the count is null <br>
 * @retval DW_INVALID_HANDLE if provided SM handle is invalid. <br>
 * @retval DW_SUCCESS get successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_getNumSensors(uint32_t* count, dwSensorType type, dwSensorManagerHandle_t sm);

/**
 * @brief Gets the sensor handle to the specified sensor
 *
 * @param[in] handle Pointer to location where handle shall be returned
 * @param[in] sensorIndex Index of the sensor
 * @param[in] sm SM handle instantiated by the caller
 *
 * @retval DW_INVALID_ARGUMENT pointer to the handle is null <br>
 * @retval DW_INVALID_HANDLE if provided SM handle is invalid. <br>
 * @retval DW_SUCCESS get successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_getSensorHandle(dwSensorHandle_t* handle, uint32_t sensorIndex, dwSensorManagerHandle_t sm);

/**
 * @brief Gets sensor's clientData according to the assigned sensorIndex.
 *
 * @param[out] cd Pointer to location where clientData shall be returned
 * @param[in] sensorIndex Index of the sensor
 * @param[in] sm SM handle instantiated by the caller
 *
 * @retval DW_INVALID_ARGUMENT pointer to the cd is null <br>
 * @retval DW_INVALID_HANDLE if provided SM handle is invalid. <br>
 * @retval DW_SUCCESS get successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_getSensorClientData(uint64_t* cd, uint32_t sensorIndex, dwSensorManagerHandle_t sm);

/**
 * @brief Gets sensor's index according to the dwSensorType and the sensorTypeIndex.
 *
 * @param[out] sensorIndex Pointer to location where the sensor index should be returned
 * @param[in] type Type of sensor being requested
 * @param[in] sensorTypeIndex Index of the sensor as defined by the dwSensorType
 * @param[in] sm SM handle instantiated by the caller
 *
 * @retval DW_INVALID_ARGUMENT pointer to the sensorIndex is null or no sensor of that type and index exists <br>
 * @retval DW_INVALID_HANDLE if provided SM handle is invalid. <br>
 * @retval DW_SUCCESS get successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_getSensorIndex(uint32_t* sensorIndex, dwSensorType type,
                                        uint32_t sensorTypeIndex, dwSensorManagerHandle_t sm);

/**
 * @brief Gets sensor's relative index and type based upon its sensor index
 *
 * @param[out] sensorTypeIndex Pointer to location where the index of the sensor as defined by
 * the dwSensorType should be returned
 * @param[out] type Pointer to location where the type of sensor should be returned
 * @param[in] sensorIndex Index of the sensor
 * @param[in] sm SM handle instantiated by the caller
 *
 * @retval DW_INVALID_ARGUMENT pointer to the sensorTypeIndex or type is null or no sensorIndex exists<br>
 * @retval DW_INVALID_HANDLE if provided SM handle is invalid. <br>
 * @retval DW_SUCCESS get successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_getSensorTypeIndex(uint32_t* sensorTypeIndex, dwSensorType* type,
                                            uint32_t sensorIndex, dwSensorManagerHandle_t sm);

/**
 * @brief Sets sensor's dispatcher function when the feature is turned on
 *
 * @param[in] dispatchPtr dispatcher function pointer
 * @param[in] cookie pointer of the the class object in which the dispatcher member function is defined
 * @param[in] sm SM handle instantiated by the caller
 *
 * @retval DW_INVALID_ARGUMENT pointer to the dispatchPtr is null or no function with that pointer exists<br>
 * @retval DW_INVALID_HANDLE if provided SM handle is invalid. <br>
 * @retval DW_SUCCESS set successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_setDispatcher(dwSensorManagerDispatcher_t dispatchPtr, void* const cookie, dwSensorManagerHandle_t sm);

/**
 * @brief Sets the timeout to wait for a new frame across all sensors.
 * Should be called before dwSensorManager_start.
 *
 * @param[in] timeout Timeout in microseconds to wait for a new frame.
 * @param[in] sm SM handle instantiated by the caller
 *
 * @retval DW_INVALID_ARGUMENT no function with that pointer exists<br>
 * @retval DW_INVALID_HANDLE if provided SM handle is invalid. <br>
 * @retval DW_SUCCESS set successfully.
 */
DW_API_PUBLIC
dwStatus dwSensorManager_setTimeout(dwTime_t timeout, dwSensorManagerHandle_t sm);
#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_SENSORS_SENSORMANAGER_H_

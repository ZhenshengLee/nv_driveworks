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
// SPDX-FileCopyrightText: Copyright (c) 2015-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Rig Configuration</b>
 *
 * @b Description: This file defines vehicle rig configuration methods.
 */

/**
 * @defgroup rig_configuration_group Rig Configuration Interface
 *
 * @brief Defines rig configurations for the vehicle.
 *
 * @{
 */

#ifndef DW_RIG_RIG_H_
#define DW_RIG_RIG_H_

#include "RigTypes.h"

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/rig/Vehicle.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup rigconfiguration Rig Configuration
 * @brief Defines vehicle rig configuration.
 *
 * This module manages the rig configuration of the car including vehicle properties, mounted sensors,
 * and their calibration information.
 */

/// Handle representing the Rig interface.
typedef struct dwRigObject* dwRigHandle_t;
/// Handle representing the const Rig interface.
typedef struct dwRigObject const* dwConstRigHandle_t;

/**
* Initializes the Rig Configuration module from a file.
*
* @note: Any relative file-system reference will be relative to the rig file location.
*
* @param[out] obj A pointer to the Rig Configuration handle for the created module.
* @param[in] ctx Specifies the handler to the context under which the Rigconfiguration module is created.
* @param[in] configurationFile The path of a rig file that contains the rig configuration.
                               Typically produced by the DriveWorks calibration tool.
*
* @retval DW_INVALID_ARGUMENT when the rig configuration handle is NULL or if the json file has no extension
* @retval DW_INVALID_HANDLE when the context handle is invalid, i.e null or wrong type
* @retval DW_FILE_INVALID when the json file is invalid
* @retval DW_FILE_NOT_FOUND when the json file cannot be found
* @retval DW_INTERNAL_ERROR when internal error happens
* @retval DW_BUFFER_FULL when too many extrinsic profiles are available (> 3)
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: No
*/
DW_API_PUBLIC
dwStatus dwRig_initializeFromFile(dwRigHandle_t* const obj,
                                  dwContextHandle_t const ctx,
                                  char8_t const* const configurationFile);

/**
* Initializes the Rig Configuration module from a string.
*
* @param[out] obj A pointer to the Rig Configuration handle for the created module.
* @param[in] ctx Specifies the handler to the context under which the Rigconfiguration module is created.
* @param[in] configurationString A pointer to a JSON string that contains the rig configuration.
*                                Typically produced by the DriveWorks calibration tool.
* @param[in] relativeBasePath A base path all relative file references in the rig will be resolved with respect to.
*                             If NULL, then the current working directory of the process will be used implicitly.
*
* @retval DW_INVALID_ARGUMENT when the rig configuration handle is NULL or if the json file has no extension
* @retval DW_INVALID_HANDLE when the context handle is invalid, i.e null or wrong type
* @retval DW_INTERNAL_ERROR when internal error happens
* @retval DW_BUFFER_FULL when too many extrinsic profiles are available (> 3)
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: No
*/
DW_API_PUBLIC
dwStatus dwRig_initializeFromString(dwRigHandle_t* const obj,
                                    dwContextHandle_t const ctx,
                                    char8_t const* const configurationString,
                                    char8_t const* const relativeBasePath);

/**
* Resets the Rig Configuration module.
*
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_HANDLE when the rig handle is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwRig_reset(dwRigHandle_t const obj);

/**
* Releases the Rig Configuration module.
*
* @param[in] obj The Rig Configuration module handle.
*
* @retval DW_INVALID_HANDLE when the configuration handle is invalid , i.e NULL or wrong type
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_release(dwRigHandle_t const obj);

/**
* DEPRECATED: Gets the properties of a passenger car vehicle.
* @deprecated Use dwRig_getGenericVehicle.
*
* @param[out] vehicle A pointer to the struct holding vehicle properties. The returned pointer is valid
* until module reset or release is called.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the rig configuration handle is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_NOT_AVAILABLE when no vehicle in configuration is available
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwRig_getVehicle(dwVehicle const** const vehicle, dwConstRigHandle_t const obj);

/**
* Gets the properties of a generic vehicle (car or truck).
*
* @param[out] vehicle A pointer to the struct to be filled with vehicle properties.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the rig configuration handle is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_NOT_AVAILABLE when no generic vehicle in configuration is available
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwRig_getGenericVehicle(dwGenericVehicle* const vehicle, dwConstRigHandle_t const obj);

/**
* DEPRECATED: Sets the properties of a passenger car vehicle.
* @deprecated Use dwRig_setGenericVehicle.
*
* @param[in] vehicle A pointer to the struct holding vehicle properties.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the rig configuration handle is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_NOT_AVAILABLE when no vehicle in configuration is available
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwRig_setVehicle(dwVehicle const* const vehicle, dwRigHandle_t const obj);

/**
* Sets the properties of a generic vehicle (car or truck).
*
* @param[in] vehicle A pointer to the struct holding vehicle properties.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the rig configuration handle is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_NOT_AVAILABLE when no generic vehicle in configuration is available
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwRig_setGenericVehicle(dwGenericVehicle const* const vehicle, dwRigHandle_t const obj);

/**
* Gets the number of vehicle IO sensors.
*
* @param[out] vioConfigCount A pointer to the number of vehicle IO sensors in the Rig Configuration.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the rig configuration handle is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwRig_getVehicleIOConfigCount(uint32_t* const vioConfigCount,
                                       dwConstRigHandle_t const obj);

/**
* Gets the number of all available sensors.
*
* @param[out] sensorCount A pointer to the number of sensors in the rig configuration.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwRig_getSensorCount(uint32_t* const sensorCount,
                              dwConstRigHandle_t const obj);

/**
* Find number of sensors of a given type.
*
* @param[out] sensorCount Return number of sensors available of the given type
* @param[in] sensorType Type of the sensor to query
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT `given pointer is null
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorCountOfType(uint32_t* const sensorCount,
                                    dwSensorType const sensorType,
                                    dwConstRigHandle_t const obj);

/**
* Gets the protocol string of a sensor. This string can be used in sensor creation or to identify
* the type of a sensor.
*
* @param[out] sensorProtocol A pointer to the pointer to the protocol of the sensor, for example, camera.gmsl. The returned pointer is valid
* until module reset or release is called.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the pointer to the pointer of sensor protocol  is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorProtocol(char8_t const** const sensorProtocol,
                                 uint32_t const sensorId,
                                 dwConstRigHandle_t const obj);

/**
* Gets the parameter string for a sensor. This string can be used in sensor creation.
*
* @param[out] sensorParameter A pointer to the pointer to the parameters of the sensor, for example camera driver and csi port. The returned
* pointer is valid until module reset or release is called.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the pointer to the pointer of sensor parameters is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC dwStatus dwRig_getSensorParameter(char8_t const** const sensorParameter,
                                                uint32_t const sensorId,
                                                dwConstRigHandle_t const obj);

/**
* Sets the parameter string for a sensor. This string can be used in sensor creation.
*
* @param[in] sensorParameter string representing sensor parameters, for example camera driver and csi port.
* Maximal length is limited to 512.
* @param[in] sensorId Specifies the index of the sensor of which to set sensor parameter.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the sensor parameter string is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the sensor to be updated is more than MAX_SENSOR_COUNT
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC dwStatus dwRig_setSensorParameter(char8_t const* const sensorParameter,
                                                uint32_t const sensorId,
                                                dwRigHandle_t const obj);

/**
* Gets the parameter string for a sensor with any path described by file=,video=,timestamp= property modified
* to be in respect to the current rig file's directory (if initializing a rig from file), or in respect to the
* relativeBasePath (when initializing a rig from string). For example, given a rig.json file stored at
* this/is/rig.json with a virtual sensor pointing to file=video.lraw, the call to this function will
* return sensor properties modified as file=this/is/video.lraw.
*
* @param[out] sensorParameter Sensor parameters with modified path inside of file=,video=,timestamp= returned
* here.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the pointer to the pointer of sensor parameters is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC dwStatus dwRig_getSensorParameterUpdatedPath(char8_t const** const sensorParameter,
                                                           uint32_t const sensorId,
                                                           dwConstRigHandle_t const obj);

/**
* Gets the sensor to rig transformation for a sensor. This transformation relates the sensor and
* the rig coordinate system to each other. For example, the origin in sensor coordinate system is
* the position of the sensor in rig coordinates. Also, if the sensor's type doesn't support extrinsics,
* the identity transformation will be returned.
*
* @param[out] transformation A pointer to the transformation from sensor to rig coordinate system.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the transformation pointer is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorToRigTransformation(dwTransformation3f* const transformation,
                                            uint32_t const sensorId,
                                            dwConstRigHandle_t const obj);

/**
* Gets the sensor FLU to rig transformation for a sensor. This transformation relates the sensor
* FLU and the rig coordinate system to each other. For example, the origin in sensor coordinate
* system is the position of the sensor in rig coordinates.
*
* @param[out] transformation A pointer to the transformation from sensor to rig coordinate system.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the transformation pointer is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorFLUToRigTransformation(dwTransformation3f* const transformation,
                                               uint32_t const sensorId,
                                               dwConstRigHandle_t const obj);

/**
* Gets the nominal sensor to rig transformation for a sensor.  This transform differs from transform T
* provided by getSensorToRigTransformation() in that it represents a static reference transformation
* from factory calibration and/or mechanical drawings, whereas T can change over time. Also, if the sensor's
* type doesn't support extrinsics, the identity transformation will be returned.
*
* @param[out] transformation A pointer to the nominal transformation from sensor to rig coordinate system.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the transformation pointer is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getNominalSensorToRigTransformation(dwTransformation3f* const transformation,
                                                   uint32_t const sensorId,
                                                   dwConstRigHandle_t const obj);

/**
* Gets the sensor to sensor transformation for a pair of sensors. This transformation relates the first and
* second sensor coordinate systems to each other. Identity transformations are used for sensors that don't
* support a native extrinsic frame.
*
* @param[out] transformation A pointer to the transformation from sensor to sensor coordinate system.
* @param[in] sensorIdFrom Specifies the index of the source sensor.
* @param[in] sensorIdTo Specifies the index of the destination sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the transformation pointer is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorToSensorTransformation(dwTransformation3f* const transformation,
                                               uint32_t const sensorIdFrom,
                                               uint32_t const sensorIdTo,
                                               dwConstRigHandle_t const obj);

/**
* Gets the nominal sensor to sensor transformation for a pair of sensors.  This transform differs from transform T
* provided by getSensorToSensorTransformation() in that it represents a static reference transformation
* from factory calibration and/or mechanical drawings, whereas T can change over time. Identity transformations
* are used for sensors that don't support a native extrinsic frame.
*
* @param[out] transformation A pointer to the nominal transformation from sensor to sensor coordinate system.
* @param[in] sensorIdFrom Specifies the index of the source sensor.
* @param[in] sensorIdTo Specifies the index of the destination sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the transformation pointer is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getNominalSensorToSensorTransformation(dwTransformation3f* const transformation,
                                                      uint32_t const sensorIdFrom,
                                                      uint32_t const sensorIdTo,
                                                      dwConstRigHandle_t const obj);

/**
* Sets the sensor to rig transformation for a sensor.
* @see dwRig_getSensorToRigTransformation.
*
* @param[in] transformation A pointer to the transformation from sensor to rig coordinate system.
* @param[in] sensorId Specifies the index of the updates sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the transformation pointer is NULL
* @retval DW_INVALID_HANDLE when the transformation pointer is NULL
* @retval DW_CALL_NOT_ALLOWED when the sensor's type doesn't support extrinsics
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_setSensorToRigTransformation(dwTransformation3f const* const transformation,
                                            uint32_t const sensorId,
                                            dwRigHandle_t const obj);

/**
* Gets the name of a sensor as given in the configuration. For example, "Front Camera".
*
* @param[out] sensorName A pointer to the name of the sensor. The pointer is valid until module reset or release is
* called.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the sensor pointer is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorName(char8_t const** const sensorName,
                             uint32_t const sensorId,
                             dwConstRigHandle_t const obj);

/**
* Gets path to sensor recording. The call is only valid for virtual sensors.
*
* @param[out] dataPath A pointer to the path with sensor data. The pointer is valid until module reset or release is
* called.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null
* @retval DW_NOT_AVAILABLE when data path for the given sensor is not available
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorDataPath(char8_t const** const dataPath,
                                 uint32_t const sensorId,
                                 dwConstRigHandle_t const obj);

/**
* Gets path to camera timestamp file. The call is only relevant for virtual h264/h265 cameras.
* Otherwise returned value is always nullptr.
*
* @param[out] timestampPath A pointer to the path containing timestamp data.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT  when given pointer is null
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getCameraTimestampPath(char8_t const** const timestampPath,
                                      uint32_t const sensorId,
                                      dwConstRigHandle_t const obj);

/**
* Returns property stored inside of a sensor. Properties are stored in name=value pairs and implement
* properties which are specific for a certain sensor in a generic way.
* For example a camera might store calibration data there, an IMU might store bias values there, etc.
*
* @param[out] propertyValue A pointer to return the value of a certain property
* @param[in] propertyName Name of the property to retrieve value from
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null or sensorId doesn't exist
* @retval DW_NOT_AVAILABLE when a certain property is not available in the rig configuration
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorPropertyByName(char8_t const** const propertyValue,
                                       char8_t const* const propertyName,
                                       uint32_t const sensorId,
                                       dwConstRigHandle_t const obj);

/**
* Overwrite content of an existing sensor property. If property does not exists, it will be added.
* Properties are stored as name=value pairs.
*
* @param[in] propertyValue Value of the property to be changed to. Maximal length limited to 512 characters.
* @param[in] propertyName Name of the property to change
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null or sensorId doesn't exist
* @retval DW_BUFFER_FULL when there are no more space for new properties, max 32
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_addOrSetSensorPropertyByName(char8_t const* const propertyValue,
                                            char8_t const* const propertyName,
                                            uint32_t const sensorId,
                                            dwRigHandle_t const obj);
/**
* Returns property stored inside of rig. Properties are stored in name=value pairs and implement
* properties which are specific for the rig in a generic way.
* For example a particular sensor layout or configuration
*
* @param[out] propertyValue A pointer to return the value of a certain property
* @param[in] propertyName Name of the property to retrieve value from
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null
* @retval DW_NOT_AVAILABLE when a certain property is not available in the rig configration
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getPropertyByName(char8_t const** const propertyValue,
                                 char8_t const* const propertyName,
                                 dwConstRigHandle_t const obj);

/**
* Overwrite content of an existing rig property. If property does not exists, it will be added.
* Properties are stored as name=value pairs.
*
* @param[in] propertyValue Value of the property to be changed to. Maximal length limited to 256 characters.
* @param[in] propertyName Name of the property to change
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null
* @retval DW_BUFFER_FULL when there are no more space for new properties, max 32
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_addOrSetPropertyByName(char8_t const* const propertyValue,
                                      char8_t const* const propertyName,
                                      dwRigHandle_t const obj);

/**
* Finds the sensor with the given name and returns its index.
*
* @param[out] sensorId The index of the matching sensor (unchanged if the function fails).
* @param[in] sensorName The sensor name to search for. If the character '*' is found, only the characters before are compared for a match.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null
* @retval DW_NOT_AVAILABLE when no sensor matches the name
* @retval DW_INVALID_HANDLE when the rig configuration module handle is invalid, i.e NULL or wrong type
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_findSensorByName(uint32_t* const sensorId,
                                char8_t const* const sensorName,
                                dwConstRigHandle_t const obj);
/**
* Finds a sensor with the given vehicleIO ID and returns the index.
*
* @param[out] sensorId The Specifies the index of the matching sensor. Undefined if the function fails.
* @param[in] vehicleIOId The vehicleIO ID to search for.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null
* @retval DW_NOT_AVAILABLE when no sensor matches the vehicle IO ID
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_findSensorIdFromVehicleIOId(uint32_t* const sensorId,
                                           uint32_t const vehicleIOId,
                                           dwConstRigHandle_t const obj);

/**
* Finds the absolute sensor index of the Nth sensor of a given type.
*
* @param[out] sensorId The index of the matching sensor (unchanged if the function fails).
* @param[in] sensorType The type of the sensor to search for.
* @param[in] sensorTypeIndex The idx of the sensor within that type.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null
* @retval DW_NOT_AVAILABLE when no sensor matches the type
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_findSensorByTypeIndex(uint32_t* const sensorId,
                                     dwSensorType const sensorType,
                                     uint32_t const sensorTypeIndex,
                                     dwConstRigHandle_t const obj);

/**
* Returns the type of sensor based upon the sensorID sent into the method
*
* @param[out] sensorType A pointer to return the type of sensor
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null or sensorId doesn't exist
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorType(dwSensorType* const sensorType,
                             uint32_t const sensorId,
                             dwConstRigHandle_t const obj);

/**
* Gets the model type of the camera intrinsics. The supported models are OCam, Pinhole, and FTheta.
*
* @param[out] cameraModel A pointer to the model type for the camera intrinsics.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the pointer to the model type is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_NOT_AVAILABLE when the sensor has no camera model
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getCameraModel(dwCameraModel* const cameraModel,
                              uint32_t const sensorId,
                              dwConstRigHandle_t const obj);

/**
* Gets the parameters of the Pinhole camera model.
*
* @param[out] config A pointer to the configuration of the camera intrinsics.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the config pointer is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_NOT_AVAILABLE when the sensor has no camera model
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getPinholeCameraConfig(dwPinholeCameraConfig* const config,
                                      uint32_t const sensorId,
                                      dwConstRigHandle_t const obj);

/**
* Gets the parameters of the FTheta camera model.
*
* @note This method clears the data passed in config in order to check if data was set.
*
* @param[out] config A pointer to the configuration of the camera intrinsics.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the config pointer is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_NOT_AVAILABLE when the sensor has no camera model
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_getFThetaCameraConfig(dwFThetaCameraConfig* const config,
                                     uint32_t const sensorId,
                                     dwConstRigHandle_t const obj);

/**
* Gets the parameters of the FTheta camera model.
*
* @note This method clears the data passed in config in order to check if data was set.
*
* @param[out] config A pointer to the configuration of the camera intrinsics.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the config pointer is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_NOT_AVAILABLE when the sensor has no camera model
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
DW_DEPRECATED("dwRig_getFThetaCameraConfigNew is replaced by dwRig_getFThetaCameraConfig.")
dwStatus dwRig_getFThetaCameraConfigNew(dwFThetaCameraConfig* const config,
                                        uint32_t const sensorId,
                                        dwConstRigHandle_t const obj);

/**
* Sets the parameters of the pinhole camera model.
*
* @param[in] config A pointer to the configuration of the camera intrinsics.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the config pointer is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_NOT_AVAILABLE when the sensor has no camera model
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_setPinholeCameraConfig(dwPinholeCameraConfig const* const config,
                                      uint32_t const sensorId,
                                      dwRigHandle_t const obj);

/**
* Sets the parameters of the FTheta camera model.
*
* @param[in] config A pointer to the configuration of the camera intrinsics.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the config pointer is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_CANNOT_CREATE_OBJECT when the sensor has no camera model
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwRig_setFThetaCameraConfig(dwFThetaCameraConfig const* const config,
                                     uint32_t const sensorId,
                                     dwRigHandle_t const obj);

/**
* This method serializes the rig-configuration object to a human-readable rig-configuration file.
* The output file contains the full state of the rig-configuration and can again be loaded with
* dwRig_initializeFromFile().
*
* The serialization format is selected based on the file name extension; currently supported extensions are json.
*
* @param[in] configurationFile The name of the file to serialize to. It's extension is used to
*                              select the serialization format. This method will overwrite the file if it exists.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the configurationFile pointer is invalid,
*                               or if the serialization format is not supported
* @retval DW_INVALID_HANDLE when provided RigConfigurationHandle handle is invalid.
* @retval DW_FILE_INVALID in case of error during serialization.
* @retval DW_SUCCESS when operation succeeded
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwRig_serializeToFile(char8_t const* const configurationFile,
                               dwConstRigHandle_t const obj);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_RIG_RIG_H_

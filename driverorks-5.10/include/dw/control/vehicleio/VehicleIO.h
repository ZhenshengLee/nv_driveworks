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
// SPDX-FileCopyrightText: Copyright (c) 2015-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: VehicleIO car controller</b>
 *
 * @b Description: API to access car controller box
 */

/**
 * @defgroup VehicleIO_actuators_group VehicleIO Actuators Interface
 *
 * @brief Defines the APIs to access the VehicleIO car controller box.
 *
 * @{
 */

#ifndef DW_VEHICLEIO_H_
#define DW_VEHICLEIO_H_

#include <dw/control/vehicleio/VehicleIOLegacyStructures.h>
#include <dw/control/vehicleio/VehicleIOValStructures.h>
#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>
#include <dw/sensors/Sensors.h>
#include <dw/sensors/canbus/CAN.h>
#include <dw/sensors/data/Data.h>
#include <dw/rig/Rig.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwVehicleIOObject* dwVehicleIOHandle_t;

/**
* Initialize VehicleIO and prepare all internal structures.
*
* @param[out] obj A pointer to the car controller handle for the created module.
* @param[in] type Specifies the VehicleIO controller type.
* @param[in] properties Vehicle properties as described by dwRig module.
* @param[in] ctx Specifies the handler to the DriveWorks context.
*
* @return DW_INVALID_ARGUMENT - if any of the given arguments are not valid <br>
*         DW_NOT_IMPLEMENTED - if given type is not implemented <br>
*         DW_SUCCESS - if the initialization is successful
*/
DW_API_PUBLIC
dwStatus dwVehicleIO_initialize(dwVehicleIOHandle_t* const obj, dwVehicleIOType const type, const dwVehicle* const properties,
                                dwContextHandle_t const ctx);
/**
* Initialize VehicleIO and prepare all internal structures from DBC File.
*
* @param[out] obj A pointer to the car controller handle for the created module.
* @param[in] type Specifies the VehicleIO controller type.
* @param[in] properties Specified Vehicle properties (from rig.json)
* @param[in] dbcFilePath Speficifes path to the dbc file for initializing a DBC-based canbus interpreter
* @param[in] ctx Specifies the handler to the DriveWorks context.
*
* @return DW_INVALID_ARGUMENT - if any of the given arguments are not valid <br>
*         DW_SUCCESS - if the initialization is successful
*/
DW_API_PUBLIC
dwStatus dwVehicleIO_initializeFromDBC(dwVehicleIOHandle_t* const obj, dwVehicleIOType const type, const dwVehicle* const properties,
                                       const char* const dbcFilePath, dwContextHandle_t const ctx);

/**
* Initialize VehicleIO and prepare all internal structures from Rig
* Configuration.
*
* @param[out] obj A pointer to the car controller handle for the created module.
* @param[in] rig Specifies the handler to the Rig Configuration.
* @param[in] ctx Specifies the handler to the DriveWorks context.
*
* @return DW_INVALID_ARGUMENT - if any of the given arguments are not valid <br>
*         DW_NOT_IMPLEMENTED - if given type is not implemented <br>
*         DW_SUCCESS - if the initialization is successful
*/
DW_API_PUBLIC
dwStatus dwVehicleIO_initializeFromRig(dwVehicleIOHandle_t* const obj,
                                       dwConstRigHandle_t const rig,
                                       dwContextHandle_t const ctx);

/**
* Reset VehicleIO to default state.
* @param[in] obj Specifies the car controller module handle.
*
* @return DW_INVALID_HANDLE - if given obj handle is invalid <br>
*         DW_SUCCESS - if the initialization is successful
*
* @note This would not perform any changes regarding current vehicle state.
**/
DW_API_PUBLIC
dwStatus dwVehicleIO_reset(dwVehicleIOHandle_t const obj);

/**
* Release used memory and close all modules. Connection to VehicleIO will be closed.
* No more car commands can be accepted when module is released.
*
* @param[in] obj The car controller module handle.
*
* @return DW_INVALID_HANDLE - if given obj handle is invalid <br>
*         DW_SUCCESS - if the initialization is successful
*/
DW_API_PUBLIC
dwStatus dwVehicleIO_release(dwVehicleIOHandle_t const obj);

/**
* Select the overrides that the driver can use to disable vehicle control.
*
* @param[in] throttleOverride Enables driver override by throttle application
* @param[in] steeringOverride Enables driver override by steering application
* @param[in] brakeOverride Enables driver override by brake application
* @param[in] gearOverride Enables driver override by brake application
* @param[in] obj A pointer to the car controller module handle.
*
* @return DW_INVALID_HANDLE - if given obj handle is invalid <br>
*         DW_SUCCESS - if the initialization is successful
*/
DW_API_PUBLIC
dwStatus dwVehicleIO_selectDriverOverrides(bool const throttleOverride, bool const steeringOverride,
                                           bool const brakeOverride, bool const gearOverride,
                                           dwVehicleIOHandle_t const obj);

/**
* Parse a received event. A parsed messages will generate certain reports, which
* can be gathered using the according callbacks. A sensor ID not corresponding to
* the incoming message may cause incorrect behavior. For manual VehicleIO
* initialization with single can BUS, a sensor ID of 0 is expected.
*
* @param[in] msg CAN message to be parsed by the controller.
* @param[in] sensorId Specifies index of CAN sensor that message came from.
* @param[in] obj Specifies the car controller module handle.
*
* @return DW_INVALID_HANDLE - if given obj handle is invalid <br>
*         DW_INVALID_ARGUMENT - if given msg is null <br>
*         DW_SUCCESS - if the initialization is successful
**/
DW_API_PUBLIC
dwStatus dwVehicleIO_consumeCANFrame(dwCANMessage const* const msg, uint32_t const sensorId, dwVehicleIOHandle_t const obj);

/**
* Similar to dwVehicleIO_consumeCANFrame. Expects a data packet generated by
* a data sensor. Used for socket based vehicle communication.
*
* @param[in] pkt data packet to be parsed by the controller.
* @param[in] sensorId Specifies index of data sensor that message came from.
* @param[in] obj Specifies the car controller module handle.
*
* @return DW_INVALID_HANDLE - if given obj handle is invalid <br>
*         DW_INVALID_ARGUMENT - if given pkt is null <br>
*         DW_SUCCESS - if the initialization is successful
**/
DW_API_PUBLIC
dwStatus dwVehicleIO_consumeDataPacket(const dwDataPacket* const pkt, uint32_t const sensorId, dwVehicleIOHandle_t const obj);

/**
* Send a vehicle command to the VehicleIO.
*
* @param[in] cmd Command to be sent to the controller.
* @param[in] obj Specifies the car controller module handle.
*
* @deprecated This will be removed in the next major release. Use dwVehicleIO_sendNonSafetyCommand or dwVehicleIO_sendSafetyCommand instead.
*
* @return DW_INVALID_HANDLE - if given obj or sensor handle are invalid <br>
*         DW_INVALID_ARGUMENT - if the command contains an invalid element given IO state
*         DW_SUCCESS - if the message was sent successfully
**/
DW_API_PUBLIC
DW_DEPRECATED("dwVehicleIO_sendCommand is deprecated and will be removed in the next major release. Use either dwVehicleIO_sendNonSafetyCommand or dwVehicleIO_sendSafetyCommand instead.")
dwStatus dwVehicleIO_sendCommand(const dwVehicleIOCommand* const cmd,
                                 dwVehicleIOHandle_t const obj);

/**
* Send a vehicle command to the VehicleIO.
*
* @param[in] cmd Command to be sent to the controller.
* @param[in] obj Specifies the car controller module handle.
*
* @deprecated This will be removed in the next major release. Use dwVehicleIO_sendNonSafetyCommand or dwVehicleIO_sendSafetyCommand instead.
*
* @return DW_INVALID_HANDLE - if given obj or sensor handle are invalid <br>
*         DW_INVALID_ARGUMENT - if the command contains an invalid element given IO state
*         DW_SUCCESS - if the initialization is successful
**/
DW_API_PUBLIC
DW_DEPRECATED("dwVehicleIO_sendMiscCommand is deprecated and will be removed in the next major release. Use either dwVehicleIO_sendNonSafetyCommand or dwVehicleIO_sendSafetyCommand instead.")
dwStatus dwVehicleIO_sendMiscCommand(const dwVehicleIOMiscCommand* const cmd,
                                     dwVehicleIOHandle_t const obj);

/**
* Retrieve current vehicle state. Note that if called immediately after sending a new command it might
* not be reflected in the status, as the command needs to be executed and reported back by the vehicle.
*
* @param[out] state returned vehicle state.
* @param[in] obj Specifies the car controller module handle.
*
* @deprecated This will be removed in the next major release. Use dwVehicleIO_getNonSafetyState, dwVehicleIO_getSafetyState, or dwVehicleIO_getActuationFeedback instead.
*
* @return DW_INVALID_HANDLE - if given obj handle is invalid <br>
*         DW_SUCCESS - if the initialization is successful
**/
DW_API_PUBLIC

DW_DEPRECATED("dwVehicleIO_getVehicleState is deprecated and will be removed in the next major release. Use either dwVehicleIO_getVehicleSafetyState, dwVehicleIO_getVehicleNonSafetyState, or dwVehicleIO_getVehicleActuationFeedback instead.")
dwStatus dwVehicleIO_getVehicleState(dwVehicleIOState* const state, dwVehicleIOHandle_t const obj);

/**
* Retrieve current VehicleIO capabilities.
*
* @param[out] caps returned VehicleIO capabilities.
* @param[in] obj Specifies the car controller module handle.
*
* @return DW_INVALID_HANDLE - if given obj handle is invalid <br>
*         DW_SUCCESS - if the call was successful
**/
DW_API_PUBLIC
dwStatus dwVehicleIO_getCapabilities(dwVehicleIOCapabilities* const caps, dwVehicleIOHandle_t const obj);

/**
* Send a vehicle safety command to the VehicleIO.
*
* @param[in] safeCmd Safety command to be sent to the controller.
* @param[in] obj Specifies the car controller module handle.
*
* @return DW_INVALID_HANDLE - if given obj or sensor handle are invalid <br>
*         DW_INVALID_ARGUMENT - if the command contains an invalid element given IO state
*         DW_SUCCESS - if the message was sent successfully
**/
DW_API_PUBLIC
dwStatus dwVehicleIO_sendSafetyCommand(const dwVehicleIOSafetyCommand* const safeCmd,
                                       dwVehicleIOHandle_t const obj);

/**
* Send a vehicle non-safety command to the VehicleIO.
*
* @param[in] nonSafeCmd Non-safety command to be sent to the controller.
* @param[in] obj Specifies the car controller module handle.
*
* @return DW_INVALID_HANDLE - if given obj or sensor handle are invalid <br>
*         DW_INVALID_ARGUMENT - if the command contains an invalid element given IO state
*         DW_SUCCESS - if the message was sent successfully
**/
DW_API_PUBLIC
dwStatus dwVehicleIO_sendNonSafetyCommand(const dwVehicleIONonSafetyCommand* const nonSafeCmd,
                                          dwVehicleIOHandle_t const obj);

/**
* Retrieve current vehicle safety state. Note that if called immediately after sending a new command it might
* not be reflected in the status, as the command needs to be executed and reported back by the vehicle.
*
* @param[out] safeState returned vehicle safety state.
* @param[in] obj Specifies the car controller module handle.
*
* @return DW_INVALID_HANDLE - if given obj handle is invalid <br>
*         DW_SUCCESS - if the initialization is successful
**/
DW_API_PUBLIC
dwStatus dwVehicleIO_getVehicleSafetyState(dwVehicleIOSafetyState* const safeState,
                                           dwVehicleIOHandle_t const obj);

/**
* Retrieve current vehicle non-safety state. Note that if called immediately after sending a new command it might
* not be reflected in the status, as the command needs to be executed and reported back by the vehicle.
*
* @param[out] nonSafeState returned vehicle non-safety RoV state.
* @param[in] obj Specifies the car controller module handle.
*
* @return DW_INVALID_HANDLE - if given obj handle is invalid <br>
*         DW_SUCCESS - if the initialization is successful
**/
DW_API_PUBLIC
dwStatus dwVehicleIO_getVehicleNonSafetyState(dwVehicleIONonSafetyState* const nonSafeState,
                                              dwVehicleIOHandle_t const obj);

/**
* Retrieve current vehicle actuation feedback. Note that if called immediately after sending a new command it might
* not be reflected in the status, as the command needs to be executed and reported back by the vehicle.
*
* @param[out] actuationFeedback returned actuation feedback.
* @param[in] obj Specifies the car controller module handle.
*
* @return DW_INVALID_HANDLE - if given obj handle is invalid <br>
*         DW_SUCCESS - if the initialization is successful
**/
DW_API_PUBLIC
dwStatus dwVehicleIO_getVehicleActuationFeedback(dwVehicleIOActuationFeedback* const actuationFeedback,
                                                 dwVehicleIOHandle_t const obj);

/**
* Setting driving mode allows to control the behaviour of VehicleIO module with
* regards to the permitted commands and number of safety checks performed.
*
* NOTE: Whether certain driving mode is actually supported and when/if it is
* taken into effect is highly dependent on the type of the actuation interface.
*
* Use `dwVehicleIO_getVehicleState` call to determine which driving mode is in
* effect.
*
* Use `dwVehicleIO_getCapabilities` call to determine which limits are supported
* when in DW_VEHICLEIO_DRIVING_LIMITED or DW_VEHICLEIO_DRIVING_LIMITED_ND modes.
*
* @param[in] mode specifies the mode of driving.
* @param[in] obj Specifies the car controller module handle.
*
* @return DW_INVALID_HANDLE - if given obj handle is invalid
*         DW_NOT_SUPPORTED - if the mode is not supported
*         DW_SUCCESS - if the call was successful
*
* @see dwVehicleIODrivingMode
**/
DW_API_PUBLIC
dwStatus dwVehicleIO_setDrivingMode(dwVehicleIODrivingMode const mode, dwVehicleIOHandle_t const obj);

/**
* Add CAN sensor handle and corresponding VehicleIO configuration ID. Allows sending commands to
* multiple sensors. For manual VehicleIO initialization with single CAN bus, a vehicleIO ID
* of 0 is expected.
*
* @param[in] vehicleIOId Specifies ID of vehicle IO configuration.
* @param[in] sensorHandle Specifies the underlying VehicleIO CAN sensor.
* @param[in] obj Specifies the car controller module handle.
*
* @return DW_INVALID_HANDLE - if given obj handle or sensorHandle are invalid
*         DW_SUCCESS - if the call was successful
**/
DW_API_PUBLIC
dwStatus dwVehicleIO_addCANSensor(uint32_t const vehicleIOId, dwSensorHandle_t const sensorHandle, dwVehicleIOHandle_t const obj);

/**
* Add data sensor handle and corresponding VehicleIO configuration ID. Allows sending commands to
* multiple sensors. For manual VehicleIO initialization with single data sensor, a vehicleIO ID
* of 0 is expected.
*
* @param[in] vehicleIOId Specifies ID of vehicle IO configuration.
* @param[in] sensorHandle Specifies the underlying VehicleIO data sensor.
* @param[in] obj Specifies the car controller module handle.
*
* @return DW_INVALID_HANDLE - if given obj handle or sensorHandle are invalid
*         DW_SUCCESS - if the call was successful
**/
DW_API_PUBLIC
dwStatus dwVehicleIO_addDataSensor(uint32_t const vehicleIOId, dwSensorHandle_t const sensorHandle, dwVehicleIOHandle_t const obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_VEHICLEIO_H_

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
// SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: VehicleIO Driver</b>
 *
 * @b Description: This file defines the VehicleIO custom plugin interface layer.
 */

#ifndef DW_CONTROL_PLUGINS_VEHICLEIO_DRIVER_H_
#define DW_CONTROL_PLUGINS_VEHICLEIO_DRIVER_H_

/**
 * @defgroup vehicleiodriver_group VehicleIO Driver Interface
 *
 * @brief Defines the VehicleIO Driver module for accessing a custom VehicleIO backend.
 *
 * @ingroup VehicleIO_actuators_group
 * @{
 */

#include <dw/core/base/Types.h>
#include <dw/control/vehicleio/VehicleIO.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initializes the VehicleIO Driver.
 *
 * @return DW_FAILURE, DW_SUCCESS
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_DEPRECATED("_dwVehicleIODriver_initialize is deprecated and will be removed in the next major release. Use either _dwVehicleIODriver_initialize_V2 or _dwVehicleIODriver_initialize_V3 instead.")
dwStatus _dwVehicleIODriver_initialize();

/**
 * Releases the VehicleIO Driver.
 *
 * @return DW_FAILURE, DW_SUCCESS
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
dwStatus _dwVehicleIODriver_release();

/**
* Consume a received CAN message and update the vehicle state.
*
* @param[in] msg CAN message to be consumed.
* @param[in] state Structure updated with data from CAN message.
*
* @return DW_FAILURE, DW_SUCCESS
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_DEPRECATED("_dwVehicleIODriver_consume is deprecated and will be removed in the next major release. Use _dwVehicleIODriver_consumeExt instead.")
dwStatus _dwVehicleIODriver_consume(const dwCANMessage* msg, dwVehicleIOState* state);

/**
* Send a vehicle command to the given CAN sensor.
*
* @param[in] cmd Command with parameters to be sent.
* @param[in] sensor CAN sensor to send CAN messages.
*
* @return DW_FAILURE, DW_SUCCESS
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_DEPRECATED("_dwVehicleIODriver_sendCommand is deprecated and will be removed in the next major release.")
dwStatus _dwVehicleIODriver_sendCommand(const dwVehicleIOCommand* cmd,
                                        dwSensorHandle_t sensor);

/**
* Send misc vehicle command to the given CAN sensor.
*
* @param[in] cmd Command with parameters to be sent.
* @param[in] sensor CAN sensor to send CAN messages.
*
* @return DW_FAILURE, DW_SUCCESS
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_DEPRECATED("_dwVehicleIODriver_sendMiscCommand is deprecated and will be removed in the next major release.")
dwStatus _dwVehicleIODriver_sendMiscCommand(const dwVehicleIOMiscCommand* cmd,
                                            dwSensorHandle_t sensor);

/**
* Clear faults in current vehicle state.
*
* @param[in] sensor CAN sensor to send CAN messages.
* @param[in] state Structure specifying overrides or faults.
*
* @return DW_FAILURE, DW_SUCCESS
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
DW_DEPRECATED("_dwVehicleIODriver_clearFaults is deprecated and will be removed in the next major release.")
dwStatus _dwVehicleIODriver_clearFaults(dwSensorHandle_t sensor, const dwVehicleIOState* state);

/**
* Set driving mode.
*
* @param[in] mode specifies the mode of driving.
*
* @return DW_NOT_SUPPORTED - if the mode is not supported.
*         DW_SUCCESS - if the new mode has been accepted.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
dwStatus _dwVehicleIODriver_setDrivingMode(const dwVehicleIODrivingMode mode);

/**
* Reset driver to default state.
*
* @return DW_FAILURE, DW_SUCCESS
*
* @note This would not perform any changes regarding current vehicle state.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
**/
dwStatus _dwVehicleIODriver_reset();

/**
 * @brief Initialize Plugin VIO Driver
 *
 * NOTE: This function will be deprecated in a future release. Please use use _dwVehicleIODriver_initialize_V3 instead.
 *
 * @param[in] context Driveworks context
 * @param[in] vehicleTypeString The string to specify the type of partner VehicleIO driver to create
 * @param[in] vehicleProperties Specified Vehicle properties (from rig.json).
 * @param[in] vehicleIOCapabilities Specifies VehicleIO capabilities.
 * @param[in] dbcFilepath Speficifes path to the dbc file for initializing a DBC-based canbus interpreter
 * @param[in] vioState Vehicle IO State.
 * @param[in] vioSafetyState Vehicle IO Safety State.
 * @param[in] vioNonSafetyState Vehicle IO Non-Safety State.
 * @param[in] vioActuationFeedback Vehicle IO Actuation Feedback State.
 * @return DW_FAILURE, DW_SUCCESS
 */
dwStatus _dwVehicleIODriver_initialize_V2(dwContextHandle_t context, char8_t const* vehicleTypeString, dwVehicle const* vehicleProperties, dwVehicleIOCapabilities* vehicleIOCapabilities, char8_t const* dbcFilepath, dwVehicleIOState* vioState, dwVehicleIOSafetyState* vioSafetyState, dwVehicleIONonSafetyState* vioNonSafetyState, dwVehicleIOActuationFeedback* vioActuationFeedback);

/**
 * @brief Extract dwVehicleIOSafetyState from incoming CAN message
 *
 * @param[in] canMessage CAN message
 * @param[out] safetyState Extracted dwVehicleIOSafetyState
 * @return dwStatus DW_FAILURE, DW_SUCCESS
 */
DW_DEPRECATED("_dwVehicleIODriver_consumeForSafeState is deprecated and will be removed in the next major release. Use _dwVehicleIODriver_consumeExt instead")
dwStatus _dwVehicleIODriver_consumeForSafeState(dwCANMessage const* canMessage, dwVehicleIOSafetyState* safetyState);

/**
 * @brief Extract dwVehicleIONonSafetyState from incoming CAN message
 *
 * @param[in] canMessage CAN message
 * @param[out] nonSafetyState Extracted dwVehicleIONonSafetyState
 * @return DW_FAILURE, DW_SUCCESS
 */
DW_DEPRECATED("_dwVehicleIODriver_consumeForNonSafeState is deprecated and will be removed in the next major release. Use _dwVehicleIODriver_consumeExt instead")
dwStatus _dwVehicleIODriver_consumeForNonSafeState(dwCANMessage const* canMessage, dwVehicleIONonSafetyState* nonSafetyState);

/**
 * @brief Extract dwVehicleIOActuationFeedback from incoming CAN message
 *
 * @param[in] canMessage CAN message
 * @param[out] actuationFeedback Extracted dwVehicleIOActuationFeedback
 * @return DW_FAILURE, DW_SUCCESS
 */
DW_DEPRECATED("_dwVehicleIODriver_consumeForActuationFeedback is deprecated and will be removed in the next major release. Use _dwVehicleIODriver_consumeExt instead")
dwStatus _dwVehicleIODriver_consumeForActuationFeedback(dwCANMessage const* canMessage, dwVehicleIOActuationFeedback* actuationFeedback);

/**
 * @brief Extract dwVehicleIOSafetyState from incoming data packet.
 *
 * @param[in] dataPacket Incoming data-packet.
 * @param[out] safetyState Extracted dwVehicleIOSafetyState.
 * @return DW_FAILURE, DW_SUCCESS
 */
DW_DEPRECATED("_dwVehicleIODriver_consumeDataForSafeState is deprecated and will be removed in the next major release. Use _dwVehicleIODriver_consumeDataExt instead")
dwStatus _dwVehicleIODriver_consumeDataForSafeState(dwDataPacket const* dataPacket, dwVehicleIOSafetyState* safetyState);

/**
 * @brief Extract dwVehicleIONonSafetyState from incoming data packet.
 *
 * @param[in] dataPacket Incoming data-packet.
 * @param[out] nonSafetyState Extracted dwVehicleIONonSafetyState.
 * @return DW_FAILURE, DW_SUCCESS
 */
DW_DEPRECATED("_dwVehicleIODriver_consumeDataForNonSafeState is deprecated and will be removed in the next major release. Use _dwVehicleIODriver_consumeDataExt instead")
dwStatus _dwVehicleIODriver_consumeDataForNonSafeState(dwDataPacket const* dataPacket, dwVehicleIONonSafetyState* nonSafetyState);

/**
 * @brief Extract dwVehicleIOActuationFeedback from incoming data packet.
 *
 * @param[in] dataPacket Incoming data-packet.
 * @param[out] actuationFeedback Extracted dwVehicleIOActuationFeedback.
 * @return DW_FAILURE, DW_SUCCESS
 */
DW_DEPRECATED("_dwVehicleIODriver_consumeDataForActuationFeedback is deprecated and will be removed in the next major release. Use _dwVehicleIODriver_consumeDataExt instead")
dwStatus _dwVehicleIODriver_consumeDataForActuationFeedback(dwDataPacket const* dataPacket, dwVehicleIOActuationFeedback* actuationFeedback);

/**
 * @brief Send dwVehicleIOSafetyCommand over specified sensor
 *
 * NOTE: This function will be deprecated in a future release. Please use use _dwVehicleIODriver_sendASILCommand, _dwVehicleIODriver_sendQMCommand, _dwVehicleIODriver_sendEgomotion or _dwVehicleIODriver_sendSensorCalibration instead.
 *
 * @param[in] safetyCommand dwVehicleIOSafetyCommand to send
 * @param[in] sensorHandle Sensor handle
 * @return DW_FAILURE, DW_SUCCESS
 */
dwStatus _dwVehicleIODriver_sendSafetyCommand(dwVehicleIOSafetyCommand const* safetyCommand, dwSensorHandle_t sensorHandle);

/**
 * @brief Send dwVehicleIONonSafetyCommand over specified sensor
 *
 * NOTE: This function will be deprecated in a future release. Please use use _dwVehicleIODriver_sendASILCommand, _dwVehicleIODriver_sendQMCommand, _dwVehicleIODriver_sendEgomotion or _dwVehicleIODriver_sendSensorCalibration instead.
 *
 * @param[in] nonSafetyCommand dwVehicleIONonSafetyCommand to send
 * @param[in] sensorHandle Sensor handle
 * @return DW_FAILURE, DW_SUCCESS
 */
dwStatus _dwVehicleIODriver_sendNonSafetyCommand(dwVehicleIONonSafetyCommand const* nonSafetyCommand, dwSensorHandle_t sensorHandle);

/**
 * @brief Clear any outstanding faults based on incoming VIO state structures
 *
 * NOTE: This function will be deprecated in a future release.
 *
 * @param[in] sensorHandle Sensor handle
 * @param[in] vioSafetyState dwVehicleIOSafetyState
 * @param[in] vioNonSafetyState dwVehicleIONonSafetyState
 * @param[in] vioActuationFeedback dwVehicleIOActuationFeedback
 * @return DW_FAILURE, DW_SUCCESS
 */
dwStatus _dwVehicleIODriver_clearFaults_new(dwSensorHandle_t const sensorHandle, dwVehicleIOSafetyState const* vioSafetyState, dwVehicleIONonSafetyState const* vioNonSafetyState, dwVehicleIOActuationFeedback const* vioActuationFeedback);

/**
 * @brief Initialize Plugin VIO Driver
 *
 * @param[in] context Driveworks context
 * @param[in] vehicleProperties Specified Vehicle properties (from rig.json).
 * @param[in] vehicleIOCapabilities Specifies VehicleIO capabilities.
 * @param[in] dbcFilepath Speficifes path to the dbc file for initializing a DBC-based canbus interpreter
 * @param[in] vioAsilState Vehicle IO ASIL State.
 * @param[in] vioQmState Vehicle IO QM State.
 * @return DW_FAILURE, DW_SUCCESS
 */
dwStatus _dwVehicleIODriver_initialize_V3(dwContextHandle_t context, dwVehicle const* vehicleProperties, dwVehicleIOCapabilities* vehicleIOCapabilities, char8_t const* dbcFilepath, dwVehicleIOASILStateE2EWrapper* vioAsilState, dwVehicleIOQMState* vioQmState);

/**
 * @brief Extract VehicleIO signals from incoming CAN message
 *
 * @param[in] canMessage CAN message
 * @return dwStatus DW_FAILURE, DW_SUCCESS
 */
dwStatus _dwVehicleIODriver_consumeExt(dwCANMessage const* canMessage);

/**
 * @brief Extract VehicleIO signals from incoming data packet.
 *
 * @param[in] dataPacket Incoming data-packet.
 * @return DW_FAILURE, DW_SUCCESS
 */
dwStatus _dwVehicleIODriver_consumeDataExt(dwDataPacket const* dataPacket);

/**
 * @brief Send dwVehicleIOASILCommandE2EWrapper over specified sensor
 *
 * @param[in] asilCommand dwVehicleIOASILCommandE2EWrapper to send
 * @param[in] sensorHandle Sensor handle
 * @return DW_FAILURE, DW_SUCCESS
 */
dwStatus _dwVehicleIODriver_sendASILCommand(dwVehicleIOASILCommandE2EWrapper const* asilCommand, dwSensorHandle_t sensorHandle);

/**
 * @brief Send dwVehicleIOQMCommand over specified sensor
 *
 * @param[in] qmCommand dwVehicleIOQMCommand to send
 * @param[in] sensorHandle Sensor handle
 * @return DW_FAILURE, DW_SUCCESS
 */
dwStatus _dwVehicleIODriver_sendQMCommand(dwVehicleIOQMCommand const* qmCommand, dwSensorHandle_t sensorHandle);

/**
 * @brief Send dwValEgomotion over specified sensor
 *
 * @param[in] egomotion dwValEgomotion to send
 * @param[in] sensorHandle Sensor handle
 * @return DW_FAILURE, DW_SUCCESS
 */
dwStatus _dwVehicleIODriver_sendEgomotion(dwValEgomotion const* egomotion, dwSensorHandle_t sensorHandle);

/**
 * @brief Send dwValSensorCalibration over specified sensor
 *
 * @param[in] calibration dwValSensorCalibration to send
 * @param[in] sensorHandle Sensor handle
 * @return DW_FAILURE, DW_SUCCESS
 */
dwStatus _dwVehicleIODriver_sendSensorCalibration(dwValSensorCalibration const* calibration, dwSensorHandle_t sensorHandle);

/** @} */

#ifdef __cplusplus
}
#endif

#endif

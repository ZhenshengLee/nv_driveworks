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
// SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 */
dwStatus _dwVehicleIODriver_initialize();

/**
 * Releases the VehicleIO Driver.
 *
 * @return DW_FAILURE, DW_SUCCESS
 *
 */
dwStatus _dwVehicleIODriver_release();

/**
* Consume a received CAN message and update the vehicle state.
*
* @param[in] msg CAN message to be consumed.
* @param[in] state Structure updated with data from CAN message.
*
* @return DW_FAILURE, DW_SUCCESS
**/
dwStatus _dwVehicleIODriver_consume(const dwCANMessage* msg, dwVehicleIOState* state);

/**
* Send a vehicle command to the given CAN sensor.
*
* @param[in] cmd Command with parameters to be sent.
* @param[in] sensor CAN sensor to send CAN messages.
*
* @return DW_FAILURE, DW_SUCCESS
**/
dwStatus _dwVehicleIODriver_sendCommand(const dwVehicleIOCommand* cmd,
                                        dwSensorHandle_t sensor);

/**
* Send misc vehicle command to the given CAN sensor.
*
* @param[in] cmd Command with parameters to be sent.
* @param[in] sensor CAN sensor to send CAN messages.
*
* @return DW_FAILURE, DW_SUCCESS
**/
dwStatus _dwVehicleIODriver_sendMiscCommand(const dwVehicleIOMiscCommand* cmd,
                                            dwSensorHandle_t sensor);

/**
* Clear faults in current vehicle state.
*
* @param[in] sensor CAN sensor to send CAN messages.
* @param[in] state Structure specifying overrides or faults.
*
* @return DW_FAILURE, DW_SUCCESS
**/
dwStatus _dwVehicleIODriver_clearFaults(dwSensorHandle_t sensor, const dwVehicleIOState* state);

/**
* Set driving mode.
*
* @param[in] mode specifies the mode of driving.
*
* @return DW_NOT_SUPPORTED - if the mode is not supported.
*         DW_SUCCESS - if the new mode has been accepted.
**/
dwStatus _dwVehicleIODriver_setDrivingMode(const dwVehicleIODrivingMode mode);

/**
* Reset driver to default state.
*
* @return DW_FAILURE, DW_SUCCESS
*
* @note This would not perform any changes regarding current vehicle state.
**/
dwStatus _dwVehicleIODriver_reset();

/** @} */

#ifdef __cplusplus
}
#endif

#endif

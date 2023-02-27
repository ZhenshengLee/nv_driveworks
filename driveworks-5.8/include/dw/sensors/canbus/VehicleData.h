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
// SPDX-FileCopyrightText: Copyright (c) 2016-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Vehicle Data</b>
 *
 * @b Description: This file defines vehicle data methods.
 */

/**
 * @defgroup vehicle_group Vehicle Data Interface
 * @ingroup can_group
 * @brief Defines the methods used to obtain and store vehicle data.
 *
 * @{
 */

#ifndef DW_SENSORS_CANBUS_INTERPRETER_VEHICLEDATA_H_
#define DW_SENSORS_CANBUS_INTERPRETER_VEHICLEDATA_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A set of enumeration constants representing the default data signals.
 * The signals can be remapped to actual string based signal representation, but for sanity and
 * to be able to let SDK modules query predefined signals, these defintions are provided here.
 *
 *
 * @note There are two angles: steering wheel and steering angle. In general, there is a mapping
 *       between the angle of the steering wheel and the actual steering radius of the car.
 *       The rule of thumb for some car manufacturers is that the angle of the steering wheel
 *       has to be divided by 16 to get an approximation of the actual steering angle. In general,
 *       however, that is a non-linear mapping reflecting the current status of the whole car
 *       steering geometry.
 **/
typedef enum dwCANVehicleData {
    DW_CAN_STEERING_ANGLE       = 0x00000100, /*!< rad - steering angle of the car.
                                                         ( neg -> right, pos -> left) */
    DW_CAN_STEERING_WHEEL_ANGLE = 0x00000101, /*!< rad - angle of the steering wheel.
                                                         ( neg -> right, pos -> left) */

    DW_CAN_STEERING_RATE = 0x00000102, /*!< rad/s    - rotation speed of the steering wheel */
    DW_CAN_YAW_RATE      = 0x00000103, /*!< rad/s    - rate of the yaw rotation of the car */

    // speed
    DW_CAN_CAR_SPEED     = 0x00000200, /*!< m/s  - speed of the car */
    DW_CAN_WHEEL_SPEED_0 = 0x00000201, /*!< m/s  - speed of the wheel 0 */
    DW_CAN_WHEEL_SPEED_1 = 0x00000202, /*!< m/s  - speed of the wheel 1 */
    DW_CAN_WHEEL_SPEED_2 = 0x00000203, /*!< m/s  - speed of the wheel 2 */
    DW_CAN_WHEEL_SPEED_3 = 0x00000204, /*!< m/s  - speed of the wheel 3 */

    // acceleration
    DW_CAN_LONG_ACCEL = 0x00000301, /*!< m/ss - acceleration along the car axis */
    DW_CAN_LAT_ACCEL  = 0x00000302, /*!< m/ss - acceleration perpendicular to car axis */
    DW_CAN_Z_ACCEL    = 0x00000303, /*!< m/ss - acceleration along the z-axis (height) */
    DW_CAN_X_ACCEL    = DW_CAN_LONG_ACCEL,
    DW_CAN_Y_ACCEL    = DW_CAN_LAT_ACCEL,

    // lights

    // undefined
    DW_CAN_NOT_DEFINED = 0xFFFFFFFF /*!< undefined or unknown type  */

} dwCANVehicleData;

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_SENSORS_CANBUS_INTERPRETER_VEHICLEDATA_H_

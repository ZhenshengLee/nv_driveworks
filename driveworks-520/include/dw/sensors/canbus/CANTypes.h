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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: CAN types</b>
 *
 * @b Description: This file defines the CAN sensor types.
 */

/**
 * @ingroup can_group
 *
 * @brief Defines the CAN sensor types.
 *
 * @{
 */

#ifndef DW_SENSORS_CANBUS_CANTYPES_H_
#define DW_SENSORS_CANBUS_CANTYPES_H_

#include <dw/sensors/common/SensorTypes.h>

// clang-format off

/**
*
* This module provides access to the CAN bus typically used for communication
* between different ECUs in a vehicle. The implementation provides an abstracted layer over any
* implemented sensor drivers, supporting receive, send and filtering routines.
*
* CAN data packet as of ISO 11898-1:
* ~~~~~~~~~~~~~~~
* Byte:|           0           |         1             |           2           |           3           |
* Bit: |07-06-05-04-03-02-01-00|15-14-13-12-11-10-09-08|23-22-21-20-19-18-17-16|31-30-29-28-27-26-25-24|
*
* Byte:|           4           |           5           |           6           |           7           |
* Bit: |39-38-37-36-35-34-33-32|47-46-45-44-43-42-41-40|55-54-53-52-51-50-49-48|63-62-61-60-59-58-57-56|
* ~~~~~~~~~~~~~~~
*
* @note Currently supported frame message format is ISO 11898-1.
*
* #### Testing (Linux only) ####
*
* If you have a real CAN device, activate it with this command: <br>
* ~~~~~~~~~~~~~~~
* :> sudo ip link set can0 up type can bitrate 500000
* ~~~~~~~~~~~~~~~
*
* A virtual device is created using following commands:
* ~~~~~~~~~~~~~~~
* :> sudo modprobe vcan
* :> sudo ip link add dev vcan0 type vcan
* :> sudo ip link set up vcan0
* ~~~~~~~~~~~~~~~
* In order to send data from console to the virtual CAN bus, the cansend tool (from the
* can-utils package) can be used.
* ~~~~~~~~~~~~~~~
* :> cansend vcan0 30B#1122334455667788
* ~~~~~~~~~~~~~~~
*/
// clang-format on

/// Maximal length of the supported CAN message id [bits].
#define DW_SENSORS_CAN_MAX_ID_LEN 29

/// Maximal length of the supported CAN payload [bytes].
#define DW_SENSORS_CAN_MAX_MESSAGE_LEN 64

/// Maximum number of filter that can be specified
#define DW_SENSORS_CAN_MAX_FILTERS 255

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(push, 1) // Makes sure you have consistent structure packings.

/// Holds a CAN package.
typedef struct dwCANMessage
{
    /// Timestamp of the message in microseconds from system epoch (using clock of the context).
    dwTime_t timestamp_us;

    /// CAN ID of the message sender.
    uint32_t id;

    /// Number of bytes of the payload.
    uint16_t size;

    /// Payload.
    uint8_t data[DW_SENSORS_CAN_MAX_MESSAGE_LEN];
} dwCANMessage;

#pragma pack(pop)

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_SENSORS_CANBUS_CANTYPES_H_

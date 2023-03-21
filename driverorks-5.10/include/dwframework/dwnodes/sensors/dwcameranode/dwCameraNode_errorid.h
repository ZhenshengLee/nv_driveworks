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
// SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_NODES_SENSORS_CAMERANODE_CAMERANODEERRORID_H_
#define DW_FRAMEWORK_NODES_SENSORS_CAMERANODE_CAMERANODEERRORID_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Offset of camera node specific error ids.
 */
#define DW_CAMERANODE_ERROR_ID_OFFSET (1U << 16U)

/**
 * Camera node specific errors.
 *
 * The camera node will report error ids from three sources
 *
 * - SIPL errors in range [0, 2^16)
 * - node errors in range [2^16, 2^24)
 * - common errors in range [2^24, 2^32)
 *
 * given DW_CAMERANODE_ERROR_ID_OFFSET = 2^16 and DW_SENSOR_ERROR_ID_OFFSET = 2^24.
 *
 * This enum specifies node errors only.
 */
typedef enum dwCameraNodeErrorID {
    /** 0 - invalid error id */
    DW_CAMERANODE_ERROR_ID_INVALID = 0,
    /** 65536 - camera intrinsics missing */
    DW_CAMERANODE_ERROR_ID_INTRINSICS_MISSING = DW_CAMERANODE_ERROR_ID_OFFSET,
    /** 65537 - intrinsics read from eeprom and from rig file don't match */
    DW_CAMERANODE_ERROR_ID_INTRINSICS_EEPROM_RIG_MISMATCH = DW_CAMERANODE_ERROR_ID_OFFSET + 1U,
    /** 65538 - cannot read intrinsics from eeprom */
    DW_CAMERANODE_ERROR_ID_INTRINSICS_EEPROM_READING_FAILED = DW_CAMERANODE_ERROR_ID_OFFSET + 2U,
} dwCameraNodeErrorID;

/**
 * Maximum error id of camera node specific error ids.
 */
#define DW_CAMERANODE_ERROR_ID_MAX DW_CAMERANODE_ERROR_ID_INTRINSICS_EEPROM_READING_FAILED

#ifdef __cplusplus
} // extern C
#endif

#endif // DW_FRAMEWORK_NODES_SENSORS_CAMERANODE_CAMERANODEERRORID_H_

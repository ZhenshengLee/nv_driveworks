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
// SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Health Signal Methods</b>
 *
 * @b Description: This file defines the health signal methods of the SDK.
 **/

/**
 * @defgroup core_group Core Interface
 *
 * @brief Defines the health signals methods of the SDK.
 *
 * Unless explicitly specified, all errors returned by DriveWorks APIs are non recoverable and the user application should transition to fail safe mode.
 * In addition, any error code not described in this documentation should be consider as fatal and the user application should also transition to fail safe mode.
 **/

#ifndef DW_CORE_HEALTH_SIGNALS_H_
#define DW_CORE_HEALTH_SIGNALS_H_

#include <dw/core/base/Types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup core_health_signal_group Health Signal
 * @ingroup core_group
 * Defines the core Health Signal methods of the SDK.
 *
 * @{
 */

#define DW_MAX_ERROR_SIGNAL_ERRORS_COUNT 32U

/**
 * @brief Basic error signal that gets reported only when there is an error
 **/
typedef struct dwErrorSignal
{
    /// timestamp at which the error occured, filled by module
    dwTime_t timestamp;
    /// module id, automatically filled
    uint16_t sourceID;
    /// the number of errors in @a errorIds
    size_t count;
    /// module defined error
    uint32_t errorIDs[DW_MAX_ERROR_SIGNAL_ERRORS_COUNT];
} dwErrorSignal;

/// The size of the data field in @a dwHealthSignal
#define DW_MAX_HEALTH_BYTEARRAY_SIZE 862U

/**
 * @brief Basic health signal that describes the health status of a particular software element
 * @note each health signal is defined to be 1024 bytes
 **/
typedef struct dwHealthSignal
{
    /// timestamp at which the health status was last updated, filled by module
    dwTime_t timestamp;
    /// module id, automatically filled
    uint16_t sourceID;
    /// the number of errors in @a errorIds
    size_t count;
    /// module defined error
    uint32_t errorIDs[DW_MAX_ERROR_SIGNAL_ERRORS_COUNT];

    /// bytes used in the optional byte array
    size_t dataSize;
    /// optional byte array for additional information
    uint8_t data[DW_MAX_HEALTH_BYTEARRAY_SIZE];
} dwHealthSignal;

/// The maximum number of individual @a dwHealthSignal that can be stored in a @a dwHealthSignalArray
#define DW_MAX_HEALTH_SIGNAL_ARRAY_SIZE 64

/**
 * @brief Represents an array of health signals
 **/
typedef struct dwHealthSignalArray
{
    /// The individual signals
    dwHealthSignal signal[DW_MAX_HEALTH_SIGNAL_ARRAY_SIZE];
    /// Number of health signals present in the array
    uint32_t count;
} dwHealthSignalArray;

#ifdef __cplusplus
}
#endif
/** @} */
#endif

/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2021-2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

/**
 * @file
 * <b>DriveWorks API: Signal Status Library </b>
 *
 * @b Description: This file defines signal status types and encoder/decoder functions
 */

/**
 * @defgroup core_signal_status_group Signal Status
 * @brief Defines of signal status types, and encoder/decorder functions
 *
 * @{
 * @ingroup core_group
 */

#ifndef DWV_CORE_SIGNAL_STATUS_H_
#define DWV_CORE_SIGNAL_STATUS_H_

#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>

#ifdef __cplusplus
extern "C" {
#endif

/// The type of the encoded status field
typedef uint8_t dwSignalValidity;

/**
* Overall status of the signal
*/
typedef enum {
    /// Initial value. Means that the signal has never had an assigned value.
    DW_SIGNAL_STATUS_INIT = 0,
    /// Signal contains the last valid value that was set
    DW_SIGNAL_STATUS_LAST_VALID = 1,
    /// Signal value is in error
    DW_SIGNAL_STATUS_ERROR = 2,
    /// Signal value is outside acceptable bounds
    DW_SIGNAL_STATUS_OUT_OF_BOUNDS_ERROR = 3,
    /// Signal enum max value
    DW_SIGNAL_STATUS_MAX_ENUM_VALUE = 3
} dwSignalStatus;

/**
* Timeout related status of the signal
*/
typedef enum {
    /// This signal has never been received
    DW_SIGNAL_TIMEOUT_NEVER_RECEIVED = 0,
    /// No timeout error
    DW_SIGNAL_TIMEOUT_NONE = 1,
    /// Signal is overdue
    DW_SIGNAL_TIMEOUT_OVERDUE = 2,
    /// No timeout information
    DW_SIGNAL_TIMEOUT_NO_INFORMATION = 3,
    /// Signal received but not for more than twice the specified cycle
    DW_SIGNAL_TIMEOUT_DELAYED = 4,
    /// Signal enum max value
    DW_SIGNAL_TIMEOUT_MAX_ENUM_VALUE = 4
} dwSignalTimeoutStatus;

/**
* End-2-End related status of the signal
*/
typedef enum {
    /// No E2E error
    DW_SIGNAL_E2E_NO_ERROR = 0,
    /// E2E Sequence error. Signal arrived out of sequence
    /// or having skipped values
    DW_SIGNAL_E2E_SEQ_ERROR = 1,
    /// Hash error. Signal did not verify against hash properly
    DW_SIGNAL_E2E_HASH_ERROR = 2,
    /// No E2E information
    DW_SIGNAL_E2E_NO_INFORMATION = 3,
    /// Signal enum max value
    DW_SIGNAL_E2E_MAX_ENUM_VALUE = 3
} dwSignalE2EStatus;

/**
 * @brief Encode dwSignal*Status values into a dwSignalValidity value
 *
 * @param[out] validity Encoded signal validity data
 * @param[in] status Signal status to encode
 * @param[in] timeoutStatus Signal timeout status to encode
 * @param[in] e2eStatus Signal E2E status to encode
 * @return DW_SUCCESS <br>
 *         DW_INVALID_ARGUMENT - If one of the arguments has an invalid value or if validity is null <br>
 */
DW_API_PUBLIC
dwStatus dwSignal_encodeSignalValidity(dwSignalValidity* validity,
                                       dwSignalStatus const status,
                                       dwSignalTimeoutStatus const timeoutStatus,
                                       dwSignalE2EStatus const e2eStatus);

/**
 * @brief Decode dwSignal*Status values from a dwSignalValidity value
 *
 * @param[out] status Signal status after decoding
 * @param[out] timeoutStatus Signal timeout status after decoding
 * @param[out] e2eStatus Signal E2E status after decoding
 * @param[in] validity Encoded signal validity data to decode
 * @return DW_SUCCESS <br>
 *         DW_INVALID_ARGUMENT - If validity contains invalid data or if one of the statuses is null <br>
 */
DW_API_PUBLIC
dwStatus dwSignal_decodeSignalValidity(dwSignalStatus* status,
                                       dwSignalTimeoutStatus* timeoutStatus,
                                       dwSignalE2EStatus* e2eStatus,
                                       dwSignalValidity const validity);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DWV_CORE_SIGNAL_STATUS_H_

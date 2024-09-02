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
// Copyright (c) 2021-2024 NVIDIA Corporation. All rights reserved.
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

/**
 * @brief Signal validity status. This is an encoded 8-bit value containing below information:
 *        1. Signal value validity status. Definition and meaning see @ref dwSignalStatus
 *        2. Signal timeout status. Definition and meaning see @ref dwSignalTimeoutStatus
 *        3. Signal end to end transportation status. Definition and meaning see @ref dwSignalE2EStatus
 *        API function @ref dwSignal_encodeSignalValidity returns such an encoded @ref dwSignalValidity value from its signal value validity status, signal timeout status and its end to end status<br>
 *        API function @ref dwSignal_decodeSignalValidity accepts an encoded signal validity status value and outputs its signal value validity status, signal timeout status and its end to end status<br>
 *        The final value is a bitwise concatenation of individual enums.<br>
 *        The following notation of (a,b,c) defines an error as `(a) | (b << 2) | (c << 5)`<br>
 *        Valid ranges for component `a`, `b` and `c` for above notations are:
 *        - a: Defined by @ref dwSignalStatus. Valid values are 0, 1, 2, 3
 *        - b: Defined by @ref dwSignalTimeoutStatus. Valid values are 0, 1, 2, 4
 *        - c: Defined by @ref dwSignalE2EStatus. Valid values are 0, 1, 2
 */
typedef uint8_t dwSignalValidity;

/**
* Overall status of the signal
*/
typedef enum {
    /// Initial value. Means that the signal has never had an assigned value.
    /// Not Valid
    DW_SIGNAL_STATUS_INIT = 0,
    /// Signal contains the last valid value that was set.
    /// Valid
    DW_SIGNAL_STATUS_LAST_VALID = 1,
    /// Signal value is in error.
    /// Not Valid
    DW_SIGNAL_STATUS_ERROR = 2,
    /// Signal value is outside acceptable bounds.
    /// Not Valid
    DW_SIGNAL_STATUS_OUT_OF_BOUNDS_ERROR = 3,
    /// Signal enum max value
    DW_SIGNAL_STATUS_MAX_ENUM_VALUE = 3
} dwSignalStatus;

/**
* Timeout related status of the signal
*/
typedef enum {
    /// This signal has never been received.
    /// Not Valid
    DW_SIGNAL_TIMEOUT_NEVER_RECEIVED = 0,
    /// No timeout error.
    /// Valid
    DW_SIGNAL_TIMEOUT_NONE = 1,
    /// Signal is overdue.
    /// This means that the signal is late outside of an acceptable time window.
    /// The acceptable time window is twice the specified signal time, within this window, the signal is DW_SIGNAL_TIMEOUT_DELAYED
    /// Not Valid
    DW_SIGNAL_TIMEOUT_OVERDUE = 2,
    /// No timeout information.
    /// Not Valid
    DW_SIGNAL_TIMEOUT_NO_INFORMATION DW_DEPRECATED_ENUM("REL_23_09") = 3,
    /// Signal was received before, but is not received for more than twice the specified cycle time.
    /// Valid
    DW_SIGNAL_TIMEOUT_DELAYED = 4,
    /// Signal enum max value
    DW_SIGNAL_TIMEOUT_MAX_ENUM_VALUE = 4
} dwSignalTimeoutStatus;

/**
* End-2-End related status of the signal
*/
typedef enum {
    /// No E2E error.
    /// Valid
    DW_SIGNAL_E2E_NO_ERROR = 0,
    /// E2E Sequence error. Signal arrived out of sequence
    /// or having skipped values.
    /// Not Valid
    DW_SIGNAL_E2E_SEQ_ERROR = 1,
    /// Hash error. Signal did not verify against hash properly.
    /// Not Valid
    DW_SIGNAL_E2E_HASH_ERROR = 2,
    /// No E2E information.
    /// Not Valid
    DW_SIGNAL_E2E_NO_INFORMATION DW_DEPRECATED_ENUM("REL_23_09") = 3,
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
 * @return DW_SUCCESS All three input statuses have been encoded into @c validity successfully. <br>
 *         DW_INVALID_ARGUMENT - If one of the arguments has an invalid value or if validity is null <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
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
 * @return DW_SUCCESS All three output statuses have been decoded from @c validity successfully. <br>
 *         DW_INVALID_ARGUMENT - If validity contains invalid data or if one of the statuses is null <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSignal_decodeSignalValidity(dwSignalStatus* status,
                                       dwSignalTimeoutStatus* timeoutStatus,
                                       dwSignalE2EStatus* e2eStatus,
                                       dwSignalValidity const validity);

/**
 * @brief Checks whether dwSignal*Status values contains a valid signal
 *
 * @param[in] validity Encoded signal validity data to check
 * @return DW_SUCCESS - If input signal is a valid signal <br>
 *         DW_NOT_AVAILABLE - If input signal is an invalid signal <br>
 *         DW_INVALID_ARGUMENT - If parameter @b validity contains invalid values <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwSignal_checkSignalValidity(dwSignalValidity const validity);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DWV_CORE_SIGNAL_STATUS_H_

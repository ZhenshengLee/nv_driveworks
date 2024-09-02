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
// SPDX-FileCopyrightText: Copyright (c) 2016-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Core Warp Primitives</b>
 *
 * @b Description: This file defines DW core warp primitives.
 */

/**
 * @defgroup core_logger_group Core Logger
 * @brief Processes wide logger API.
 *
 * Unless explicitly specified, all errors returned by DriveWorks APIs are non recoverable and the user application should transition to fail safe mode.
 * In addition, any error code not described in this documentation should be consider as fatal and the user application should also transition to fail safe mode.
 *
 *
 * @{
 * @ingroup core_group
 */

#ifndef DW_CORE_LOGGER_H_
#define DW_CORE_LOGGER_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>
#include <dw/core/base/Status.h>
#include <dw/core/logger/LoggerDefs.h>

#ifdef __cplusplus
extern "C" {
#endif

/// The Driveworks context handle.
// Forward declaration of dwContextHandle_t to avoid circular dependency.
// @note: The API using the dwContextHandle_t is deprecated
typedef struct dwContextObject* dwContextHandle_t;
typedef struct dwContextObject const* dwConstContextHandle_t;

typedef struct dwLoggerMessage
{
    char8_t const* msg;          //!< message to log
    dwLoggerVerbosity verbosity; //!< how severe is the log
    char8_t const* tag;          //!< name used to group related logs
    char8_t const* fileName;     //!< file name from where message was logged
    int32_t lineNum;             //!< line number where log originated from
    char8_t const* threadId;     //!< identifier for thread
} dwLoggerMessage;

/** Defines a user callback method called by the SDK to log the output.
 *
 * @param[in] context Deprecated context pointer, set to DW_NULL_HANDLE. Should not be used
 * @param[in] type Specifies the type of message being logged.
 * @param[in] msg A pointer to the message.
 *
 */
typedef void (*dwLogCallback)(dwContextHandle_t context, dwLoggerVerbosity type, char8_t const* msg);

/** Defines a user callback method called by the SDK to log the output with meta-data
 *
 * @param[in] msg A struct containing message and meta-data
 *
 */
typedef void (*dwLoggerCallback)(dwLoggerMessage const* msg);

/**
 * Creates a new logger instance.
 *
 * @details The initialization behavior of the logger is as follows. If the logger is initialized with dwLogger_initializeExtended,
 * then the dwLoggerCallback provided to that function will be used. If the logger is initialized with dwLogger_initialize,
 * then the dwLogCallback provided to that function will be used. If the logger is initialized with both functions,
 * then the function provided to dwLogger_initializeExtended will take precedence
 *
 * @param[in] msgCallback Specifies the callback method used by the SDK to pass log messages. It must
 * be thread-safe.
 *
 * @retval DW_INVALID_ARGUMENT if msgCallback is NULL. Provide a valid input parameter.
 * @retval DW_SUCCESS if operation succeeds.
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 **/
DW_API_PUBLIC dwStatus dwLogger_initialize(dwLogCallback msgCallback);

/**
 * Creates a new logger instance that provides meta-data with the message.
 *
 * @details The initialization behavior of the logger is as follows. If the logger is initialized with dwLogger_initializeExtended,
 * then the dwLoggerCallback provided to that function will be used. If the logger is initialized with dwLogger_initialize
 * then the dwLogCallback provided to that function will be used. If the logger is initialized with both functions, then
 * the function provided to dwLogger_initializeExtended will take precedence
 *
 * @param[in] msgCallback Specifies the callback method used by the SDK to pass log messages. It must
 * be thread-safe.
 *
 * @retval DW_INVALID_ARGUMENT if msgCallback is NULL. Provide a valid input parameter.
 * @retval DW_SUCCESS if operation succeeds.
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 **/
DW_API_PUBLIC dwStatus dwLogger_initializeExtended(dwLoggerCallback msgCallback);

/**
 * Sets the verbosity level of the logger instance. Any messages with higher
 * or equal verbosity level is forwarded to the logger callback.
 *
 * @param[in] verbosity Specifies the verbosity level to use.
 *
 * @retval DW_INVALID_ARGUMENT if verbosity parameter is invalid, not inside the dwLoggerVerbosity enum.
 * @retval DW_SUCCESS if operation succeeds.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC dwStatus dwLogger_setLogLevel(dwLoggerVerbosity const verbosity);

/**
 * Gets the verbosity level of the logger instance. 
 * 
 * @param[out] verbosity Specifies a pointer to the verbosity object to fill.
 * 
 * @retval DW_INVALID_ARGUMENT if verbosity pointer is invalid.
 * @retval DW_SUCCESS if operation succeeds.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC dwStatus dwLogger_getLogLevel(dwLoggerVerbosity* verbosity);

/**
 * Enable or disable logging of timestamps before each message. dwLogger is initialized with
 * timestamps enabled.
 *
 * @param[in] enabled Whether to log timestamps (true) or not (false).
 *
 * @retval DW_SUCCESS always.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC dwStatus dwLogger_enableTimestamps(bool const enabled);

/**
 * Logs message.
 *
 * @details if an extended callback has been initialized then it will return default values for parameters not provided@n
 * tag will be "NO_TAG"@n
 * fileName will be ""@n
 * lineNum will be 0@n
 * threadId will be ""
 *
 * @param[in] context Specifies the DriveWorks context that generated this message.
 * @param[in] verbosity Specifies the verbosity level to use.
 * @param[in] msg Specifies message which is to be logged.
 *
 * @retval DW_INVALID_ARGUMENT if verbosity parameter is not inside the dwLoggerVerbosity enum OR if msg is a null pointer.<br>
 * @retval DW_SUCCESS if no issues are encountered.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC dwStatus dwLogger_log(dwConstContextHandle_t const context, dwLoggerVerbosity const verbosity, char8_t const* const msg);

/**
 * Logs message
 *
 * @param[in] msg Specifies message to be logged with additional meta-data
 *
 * @retval DW_INVALID_ARGUMENT if msg is a null pointer.
 * @retval DW_SUCCESS if no issues are encountered.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC dwStatus dwLogger_logExtended(const dwLoggerMessage* msg);

/**
 * Release logger instance and free up used memory.
 *
 * @return DW_SUCCESS always.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC dwStatus dwLogger_release(void);

/**
 * Set the name of the current thread that will be returned when using extended logger
 *
 * @details Thread id should be set prior to calling any other DW function to avoid confusion@n
 * If this function is not called the logger uses an incrementing counter by default@n
 * Truncates after 64 characters
 *
 * @param[in] threadId Specifies name of the current thread
 *
 * @retval DW_SUCCESS always.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC dwStatus dwLogger_setThreadId(char8_t const* threadId);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_CORE_LOGGER_H_

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
 * <b>NVIDIA DriveWorks API: Core Status Methods</b>
 *
 * @b Description: This file defines status and error messages.
 */

/**
 * @defgroup core_status_group Core Status
 * Defines status and error messages.
 *
 * @{
 * @ingroup core_group
 */

#ifndef DW_CORE_STATUS_H_
#define DW_CORE_STATUS_H_

#include "Exports.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DW_CORE_GENERATE_ENUM(e) e,
// clang-format off
#define DW_CORE_ERROR_LIST(s)                                                                                                                    \
    s(DW_SUCCESS)                       /*!<  0 - No error */                                                                                    \
    s(DW_INVALID_VERSION)               /*!<  1 - Invalid version between headers and runtime library */                                         \
    s(DW_INVALID_ARGUMENT)              /*!<  2 - One of the given arguments cannot be processed by the function */                              \
    s(DW_BAD_ALLOC)                     /*!<  3 - Cannot allocate required memory */                                                             \
    s(DW_BAD_ALIGNMENT)                 /*!<  4 - Incorrect alignment for a pointer */                                                           \
    s(DW_BAD_CAST)                      /*!<  5 - Cannot cast given handle to an expected type */                                                \
    s(DW_NOT_IMPLEMENTED)               /*!<  6 - Requested feature/method is not yet implemented */                                             \
    s(DW_END_OF_STREAM)                 /*!<  7 - End of stream reached */                                                                       \
    s(DW_INVALID_HANDLE)                /*!<  8 - Given handle is invalid, i.e., not a valid handle. */                                          \
                                                                                                                                                 \
    s(DW_CALL_NOT_ALLOWED)              /*!<  9 - The call to a method is not allowed at this state */                                           \
    s(DW_NOT_AVAILABLE)                 /*!< 10 - Requested method or object instance is not yet available */                                    \
    s(DW_NOT_RELEASED)                  /*!< 11 - Given handle has not been released yet */                                                      \
    s(DW_NOT_SUPPORTED)                 /*!< 12 - Desired set of parameters or functionality is not suported */                                  \
    s(DW_NOT_INITIALIZED)               /*!< 13 - The object requires initialization first */                                                    \
    s(DW_INTERNAL_ERROR)                /*!< 14 - Internal error indicating a non recoverable situation */                                       \
    s(DW_FILE_NOT_FOUND)                /*!< 15 - Requested file was not found */                                                                \
    s(DW_FILE_INVALID)                  /*!< 16 - File content is invalid. */                                                                    \
    s(DW_CANNOT_CREATE_OBJECT)          /*!< 17 - Failed to create a DW Object*/                                                                 \
    s(DW_BUFFER_FULL)                   /*!< 18 - An internal buffer has reached its capacity. */                                                \
    s(DW_NOT_READY)                     /*!< 19 - The processing is not finished or ready */                                                     \
    s(DW_TIME_OUT)                      /*!< 20 - The request has timed out */                                                                   \
    s(DW_BUSY_WAITING)                  /*!< 21 - No response since too busy, not an error but please retry */                                   \
                                                                                                                                                 \
    s(DW_LOG_CANNOT_WRITE)              /*!< 22 - Logger is unable to write output */                                                            \
    s(DW_LOG_CANNOT_FLUSH)              /*!< 23 - Logger is unable to flush output*/                                                             \
                                                                                                                                                 \
    s(DW_SAL_CANNOT_INITIALIZE)         /*!< 24 - A sensor cannot be initialized, e.g., sensor might not be responding. */                       \
    s(DW_SAL_CANNOT_CREATE_SENSOR)      /*!< 25 - A sensor cannot be created, e.g. possible issues are wrong parameters */                       \
    s(DW_SAL_NO_DRIVER_FOUND)           /*!< 26 - Requested sensor driver cannot be found */                                                     \
    s(DW_SAL_SENSOR_UNSUPPORTED)        /*!< 27 - Requested sensor is unsupported */                                                             \
    s(DW_SAL_SENSOR_ERROR)              /*!< 28 - Internal non-recoverable sensor error */                                                       \
                                                                                                                                                 \
    s(DW_CUDA_ERROR)                    /*!< 29 - There was an error from the CUDA API */                                                        \
    s(DW_GL_ERROR)                      /*!< 30 - Last call to OpenGL API resulted in an error */                                                \
    s(DW_NVMEDIA_ERROR)                 /*!< 31 - NvMedia API resulted in an error */                                                            \
                                                                                                                                                 \
    s(DW_DNN_INVALID_MODEL)             /*!< 32 - Given network model is not valid */                                                            \
                                                                                                                                                 \
    s(DW_FAILURE)                       /*!< 33 - Unknown, non-recoverable error */                                                              \
    s(DW_DNN_INVALID_TYPE)              /*!< 34 - DNN model type is invalid */                                                                   \
                                                                                                                                                 \
    s(DW_HAL_CANNOT_OPEN_CHANNEL)       /*!< 35 - HAL cannot open a channel. */                                                                  \
                                                                                                                                                 \
    s(DW_OUT_OF_BOUNDS)                 /*!< 36 - Out of bounds exception (e.g. in array access). */                                             \
                                                                                                                                                 \
    s(DW_UNEXPECTED_IPC_EVENT)          /*!< 37 - Unexpected/Invalid Event received for IPC. */                                                  \
                                                                                                                                                 \
    s(DW_UNEXPECTED_EVENT)              /*!< 38 - Unexpected/Invalid Event received from a block. */                                             \
    s(DW_CUDA_CONTEXT_ERROR)            /*!< 39 - The CUDA context is either not present or cannot be initialized. */                            \
    s(DW_CUDA_DEVICE_ERROR)             /*!< 40 - The CUDA device currently selected is either incompatible with the operation or has failed. */ \
    s(DW_CANNOT_SYNCHRONIZE)            /*!< 41 - The synchronization call has failed. */                                                        \
    s(DW_NUM_ERROR_CODES)               /*!< Total number of DW error codes. */

// clang-format on

/**
* Status definition.
*/
//# sergen(generate)
typedef enum dwStatus {
    DW_CORE_ERROR_LIST(DW_CORE_GENERATE_ENUM)
} dwStatus;

/**
 * Converts dwStatus enum value to a string value.
 * Returns a NULL terminated character string of the status value.
 *
 */
DW_API_PUBLIC const char* dwGetStatusName(dwStatus const s);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_CORE_STATUS_H_

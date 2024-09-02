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
// SPDX-FileCopyrightText: Copyright (c) 2016-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: IO Streams</b>
 *
 * @b Description: This file defines objects that wrap file and memory streams.
 */

/**
 * @defgroup iostream_group IO Streams
 * @brief IO Stream API
 *
 */

#ifndef DW_CORE_IOSTREAM_H_
#define DW_CORE_IOSTREAM_H_

#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>
#include <dw/core/base/Status.h>
#include <dw/core/context/Context.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Enumerated type for IO Stream flags (read / write / append).
 */
typedef enum dwIOStreamFlags {
    DW_IOSTREAM_READ   = 1 << 0,
    DW_IOSTREAM_WRITE  = 1 << 1,
    DW_IOSTREAM_APPEND = 1 << 2,
} dwIOStreamFlags;

/**
 * Stream handle.
 */
typedef struct dwIOStreamObject* dwIOStreamHandle_t;

/**
 * Open a file stream with the given path and read / write parameters.
 * 
 * @param[out] stream A dwIOStreamHandle_t that will be updated with the newly opened IO Stream.
 * @param[in] path The path to the stream to be opened.
 * @param[in] flags The create flags used to open the stream.
 * @param[in] context The DriveWorks context.
 * 
 * @return DW_SUCCESS If the file is successfully opened and the stream is updated.<br>
 *         DW_INVALID_ARGUMENT If the stream is NULL.<br>
 *         DW_INVALID_HANDLE If the context is a nullptr.<br>
 *         DW_FILE_NOT_FOUND If the file could not be opened.<br>
 *         DW_FILE_INVALID If the requested flags are invalid for the file.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwIOStream_openFile(dwIOStreamHandle_t* stream, const char8_t* path, uint64_t flags, dwContextHandle_t context);

/**
 * Adopt a currently open POSIX file descriptor.
 * 
 * @param[out] stream A dwIOStreamHandle_t that will be updated with the newly adopted IO Stream.
 * @param[in] posixFd A POSIX file descriptor to a currently open file.
 * 
 * @return DW_SUCCESS If the POSIX file descriptor is successfully adopted.<br>
 *         DW_INVALID_ARGUMENT If the stream is NULL or the POSIX file descriptor is invalid.<br>
 *         DW_INVALID_HANDLE If the newly opened file stream is a nullptr and the stream output parameter cannot be updated.<br>
 *         DW_FAILURE If the stream fails to open the backing file.
 * 
 * @note This function will allocate memory to hold the stream backed by the adopted file.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwIOStream_wrapFileDescriptor(dwIOStreamHandle_t* stream, int posixFd);

// Reading, seeking and other interfaces intentionally omitted, as the goal of this API is not
// to provide a replacement for stdio to users, but to serve as a way to communicate how data is
// to be read to other DW APIs.

/**
 * Close a file stream that was opened with @c dwIOStream_openFile
 * 
 * @param stream A dwIOStreamHandle_t to a currently open IO Stream handle.
 * 
 * @return DW_SUCCESS If the stream is successfully closed.<br>
 *         DW_INVALID_HANDLE If the stream is NULL.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwIOStream_close(dwIOStreamHandle_t stream);

#ifdef __cplusplus
}
#endif

#endif

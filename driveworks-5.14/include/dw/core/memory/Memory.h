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
// SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_MEMORY_H_
#define DW_CORE_MEMORY_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup core_group Core Interface
 *
 * @brief Defines the core methods of the SDK.
 *
 * Unless explicitly specified, all errors returned by DriveWorks APIs are non recoverable and the user application should transition to fail safe mode.
 * In addition, any error code not described in this documentation should be consider as fatal and the user application should also transition to fail safe mode.
 *
 */

/**
 * Memory copy of virtual process memory. This subroutine call std::memcpy() for x86_64, and custom optimized implementation for aarch64.
 * As in standard memcpy call destination and source memory blocks should not overlap.
 *
 * NOTE: only use this function if the source address belongs to a CUDA Pinned Host memory region.
 * If either destination or source is an invalid or null pointer, the behavior is undefined, even if count is zero.
 * Please check std::memcpy() documentation for details.
 *
 *
 * @param[out] destination pointer to destination memory into which num bytes will be copied from source
 * @param[in] source pointer to source memory which will be copied destination
 * @param[in] num number of bytes to copy
 * @return pointer to destination
 */
DW_API_PUBLIC
void* memcpy_opt(void* __restrict const destination, const void* __restrict const source, size_t const num);

#ifdef __cplusplus
}
#endif

#endif // DW_CORE_MEMORY_HPP_

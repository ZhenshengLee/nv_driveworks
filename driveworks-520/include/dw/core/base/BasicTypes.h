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
 * <b>NVIDIA DriveWorks API: Core BasicTypes</b>
 *
 * @b Description: This file defines DW core basic types.
 */

/**
 * @defgroup core_basictypes_group Core BasicTypes
 * @ingroup core_group
 *
 * Defines DW core basic types.
 * @{
 */

#ifndef DW_CORE_BASICTYPES_H_
#define DW_CORE_BASICTYPES_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Specifies POD types.
 */
typedef float float32_t;
typedef double float64_t;
typedef char char8_t;

/** Specifies a timestamp unit, in microseconds.
 */
typedef int64_t dwTime_t;

/** Special value of timestamp which means infinitely long duration, in microseconds
 */
static const dwTime_t DW_TIMEOUT_INFINITE = 0x0123456789ABCDEF;

/** Special value indicating invalid time, in microseconds
 */
static const dwTime_t DW_TIME_INVALID = INT64_MIN;

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_CORE_BASICTYPES_H_

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

#ifndef DW_CORE_GEOMETRICDEFS_H_
#define DW_CORE_GEOMETRICDEFS_H_

#include "GeometricTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Invalid dwRect.
 * An invalid dwRect is not able to draw.
 * Cannot use an invalid dwRect in any calculation.
 **/
static const dwRect DW_INVALID_RECT = {.x = 0, .y = 0, .width = -1, .height = -1};

/** @brief Invalid dwRectf.
 * An invalid dwRectf is not able to draw.
 * Cannot use an invalid dwRectf in any calculation.
 */
static const dwRectf DW_INVALID_RECTF = {.x = (float32_t)0, .y = (float32_t)0, .width = (float32_t)-1, .height = (float32_t)-1};

typedef dwRect dwBox2D;
typedef dwRectf dwBox2Df;

/**
 * Identity for dwQuaternionf
 */
static const dwQuaternionf DW_IDENTITY_QUATERNIONF = {0.f, 0.f, 0.f, 1.f};

/**
 * Identity for dwQuaterniond
 */
static const dwQuaterniond DW_IDENTITY_QUATERNIOND = {0., 0., 0., 1.};

#ifdef __cplusplus
}
#endif

#endif // DW_CORE_GEOMETRICDEFS_H_

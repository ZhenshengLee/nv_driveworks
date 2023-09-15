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
 * <b>NVIDIA DriveWorks API: Core Types</b>
 *
 * @b Description: This file defines POD types, timestamps, and trivial data types.
 */

/**
 * @defgroup core_types_group Core Types
 * @brief Defines of POD types, timestamps, and trivial data types.
 *
 * @{
 * @ingroup core_group
 */

#ifndef DW_CORE_TYPES_H_
#define DW_CORE_TYPES_H_

#include "BasicTypes.h"
#include "Exports.h"
#include "GeoPoints.h"
#include "GeometricDefs.h"
#include "GeometricTypes.h"
#include "MatrixDefs.h"
#include "MatrixTypes.h"
#include "Status.h"
#include "TypesExtra.h"

#include <cuda_runtime_api.h>
// RFD - MISRAC 2012 Rule 1.4: Using emergent head file (stdalign.h)
// coverity[misra_c_2012_rule_1_4_violation]
#include <stdalign.h>

#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__cplusplus)
typedef __half dwFloat16_t;
#else
typedef __half_raw dwFloat16_t;
#endif

#if defined(__cplusplus)
#define DW_NULL_HANDLE nullptr
#define DW_NO_CALLBACK nullptr
#else
#define DW_NULL_HANDLE NULL
#define DW_NO_CALLBACK NULL
#endif

/** Get the size of dwTrivialDataType type
 */
DW_API_PUBLIC uint8_t dwSizeOf(dwTrivialDataType const type);

/** Defines a single-precision 2D polyline. */
typedef struct dwPolyline2f
{
    /// A point is a vertex of two connected line segments in a polyline.
    /// Points point to the first point in the container.
    const dwVector2f* points;
    /// number of points.
    uint32_t pointCount;
} dwPolyline2f;

/** Defines a double-precision 2D polyline. */
typedef struct dwPolyline2d
{
    /// A point is a vertex of two connected line segments in a polyline.
    /// Points point to the first point in the container.
    const dwVector2d* points;
    /// number of points.
    uint32_t pointCount;
} dwPolyline2d;

/** Defines a single-precision 3D polyline. */
typedef struct dwPolyline3f
{
    /// A point is a vertex of two connected line segments in a polyline.
    /// Points point to the first point in the container.
    const dwVector3f* points;
    /// number of points.
    uint32_t pointCount;
} dwPolyline3f;

/** Defines a double-precision 3D polyline. */
typedef struct dwPolyline3d
{
    /// A point is a vertex of two connected line segments in a polyline.
    /// Points point to the first point in the container.
    const dwVector3d* points;
    /// number of points.
    uint32_t pointCount;
} dwPolyline3d;

/// @brief Macro to place validity status in DW C struct with standardized
/// name such that we have a unified way to represent data validity.
///
/// For example, with an extra line of DEFINE_DW_VALIDITY_STATUS,
/// the validity status is included into the DW C struct and ready
/// to be consumed.
///
/// typedef dwFoo {
///    uint32_t x;
///    float32_t y;
///    char8_t z[64];
///
///    DEFINE_DW_VALIDITY_STATUS;
/// } dwFoo;
///
#define DEFINE_DW_VALIDITY_STATUS dwValidityStatus validityStatus

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_CORE_TYPES_H_

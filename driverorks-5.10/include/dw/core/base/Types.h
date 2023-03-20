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

#include <cuda_runtime_api.h>
// RFD - MISRAC 2012 Rule 1.4: Using emergent head file (stdalign.h)
// coverity[misra_c_2012_rule_1_4_violation]
#include <stdalign.h>

#include <cuda_fp16.h>

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

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

/** Specifies a type indicator of the underlying trivial data type.
 */

typedef enum dwTrivialDataType {

    /// The type of the data is unknown.
    DW_TYPE_UNKNOWN = 0,

    /// The underlying data type is bool.
    DW_TYPE_BOOL = 1 << 1,

    /// 8-bit signed integer.
    DW_TYPE_INT8 = 1 << 2,
    /// 16-bit signed integer.
    DW_TYPE_INT16 = 1 << 3,
    /// 32-bit signed integer.
    DW_TYPE_INT32 = 1 << 4,
    /// 64-bit signed integer.
    DW_TYPE_INT64 = 1 << 5,

    /// 8-bit unsigned integer.
    DW_TYPE_UINT8 = 1 << 6,
    /// 16-bit unsigned integer.
    DW_TYPE_UINT16 = 1 << 7,
    /// 32-bit unsigned integer.
    DW_TYPE_UINT32 = 1 << 8,
    /// 64-bit unsigned integer.
    DW_TYPE_UINT64 = 1 << 9,

    /// 32-bit float number.
    DW_TYPE_FLOAT32 = 1 << 10,
    /// 64-bit float number, i.e., double.
    DW_TYPE_FLOAT64 = 1 << 11,
    /// 16-bit float number.
    DW_TYPE_FLOAT16 = 1 << 12,

    /// chat8_t
    DW_TYPE_CHAR8 = 1 << 13,
} dwTrivialDataType;

/** Get the size of dwTrivialDataType type
 */
DW_API_PUBLIC uint8_t dwSizeOf(dwTrivialDataType const type);

/** Precision type definitions
 */
typedef enum dwPrecision {
    /// INT8 precision.
    DW_PRECISION_INT8 = 0,
    /// FP16 precision.
    DW_PRECISION_FP16 = 1,
    /// FP32 precision.
    DW_PRECISION_FP32 = 2,
    /// Combination of multiple precisions.
    DW_PRECISION_MIXED = 3
} dwPrecision;

/** GPU device type definitions
 * Only applicable on Drive platforms.
 * On x86 platforms, the GPU is considered to be of discrete type always.
 */
typedef enum dwGPUDeviceType {
    DW_GPU_DEVICE_DISCRETE   = 0,
    DW_GPU_DEVICE_INTEGRATED = 1
} dwGPUDeviceType;

/** Processor type definitions.
 */
typedef enum dwProcessorType {
    DW_PROCESSOR_TYPE_CPU     = 0,
    DW_PROCESSOR_TYPE_GPU     = 1,
    DW_PROCESSOR_TYPE_DLA_0   = 2,
    DW_PROCESSOR_TYPE_DLA_1   = 3,
    DW_PROCESSOR_TYPE_PVA_0   = 4,
    DW_PROCESSOR_TYPE_PVA_1   = 5,
    DW_PROCESSOR_TYPE_NVENC_0 = 6,
    DW_PROCESSOR_TYPE_NVENC_1 = 7,
    DW_PROCESSOR_TYPE_CUDLA   = 8,
} dwProcessorType;

/** Process type definitions.
 */
typedef enum dwProcessType {
    DW_PROCESS_TYPE_ASYNC = 0,
    DW_PROCESS_TYPE_SYNC  = 1
} dwProcessType;

/** Memory type definitions.
 */
typedef enum dwMemoryType {
    /// CUDA memory
    DW_MEMORY_TYPE_CUDA = 0,

    /// pageable CPU memory
    DW_MEMORY_TYPE_CPU = 1,

    /// pinned memory
    DW_MEMORY_TYPE_PINNED = 2,
} dwMemoryType;

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

/**
 * @brief The slot enum used when an application wants a dw module to bind some particular input data to an internal slot
 * for future processing and unbinding.
 * Particularly the module expects an array of instances of such data structure hence maintaining an internal container for them.
 * For example, dwObjectArray instances inputting to dwObjectInPathAnalyzer module
 */
typedef enum {
    DW_BIND_SLOT_INVALID = 0,
    DW_BIND_SLOT_1,
    DW_BIND_SLOT_2,
    DW_BIND_SLOT_3,
    DW_BIND_SLOT_4,
    DW_BIND_SLOT_5,
    DW_BIND_SLOT_6,
    DW_BIND_SLOT_7,
    DW_BIND_SLOT_8,
    DW_BIND_SLOT_9,
    DW_BIND_SLOT_10,
    DW_BIND_SLOT_11,
    DW_BIND_SLOT_12,
    DW_BIND_SLOT_13,
    DW_BIND_SLOT_14,
    DW_BIND_SLOT_15,
    DW_BIND_SLOT_16,
    DW_BIND_SLOT_MAX_COUNT
} dwBindSlot;

/** Holds blob dimensions.
 */
typedef struct
{
    /// Batch size (n).
    uint32_t batchsize;
    /// Number of channels (c).
    uint32_t channels;
    /// Height (h).
    uint32_t height;
    /// Width (w).
    uint32_t width;
} dwBlobSize;

/// @brief Defines the validity of DW struct.
typedef enum dwValidity {
    DW_VALIDITY_INVALID = 0,
    DW_VALIDITY_VALID   = 1,
    DW_VALIDITY_FORCE32 = 0x7FFFFFFF
} dwValidity;

/// @brief A light weighted 16 Btyes status to be carried over along with
/// each DW C struct instance that can indicate the data validity status.
typedef struct dwValidityStatus
{
    /// Validity of the whole data entity.
    dwValidity validity;
    /// Reserved 12 bytes which can be extended later.
    uint8_t reserved[12];
} dwValidityStatus;

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

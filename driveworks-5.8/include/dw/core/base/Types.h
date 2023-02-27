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

#include "Exports.h"
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

/** Specifies POD types.
 */

typedef float float32_t;
typedef double float64_t;
typedef char char8_t;

#if defined(__cplusplus)
typedef __half dwFloat16_t;
#else
typedef __half_raw dwFloat16_t;
#endif

/** Specifies a timestamp unit, in microseconds.
 */
typedef int64_t dwTime_t;

/** Special value of timestamp which means infinitely long duration, in microseconds
 */
static const dwTime_t DW_TIMEOUT_INFINITE = 0x0123456789ABCDEF;

/** Special value indicating invalid time, in microseconds
 */
static const dwTime_t DW_TIME_INVALID = INT64_MIN;

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

/** Defines a rectangle.
 */
typedef struct dwRect
{
    //! x coordinate.
    alignas(8) int32_t x;
    //! y coordinate.
    int32_t y;
    //! Rectangle width.
    int32_t width;
    //! Rectangle height.
    int32_t height;
} dwRect;

/** @brief Invalid dwRect.
 * An invalid dwRect is not able to draw.
 * Cannot use an invalid dwRect in any calculation.
 **/
static const dwRect DW_INVALID_RECT = {.x = 0, .y = 0, .width = -1, .height = -1};

/** Defines a rectangle with floating point numbers.
 */
//# sergen(generate)
typedef struct dwRectf
{
    //! Specifies the x coordinate.
    alignas(8) float32_t x;
    //! Specifies the y coordinate.
    float32_t y;
    //! Rectangle width.
    float32_t width;
    //! Rectangle height.
    float32_t height;
} dwRectf;

/** @brief Invalid dwRectf.
 * An invalid dwRectf is not able to draw.
 * Cannot use an invalid dwRectf in any calculation.
 */
static const dwRectf DW_INVALID_RECTF = {.x = (float32_t)0, .y = (float32_t)0, .width = (float32_t)-1, .height = (float32_t)-1};

typedef dwRect dwBox2D;
typedef dwRectf dwBox2Df;

/** Defines a 2x2 matrix of floating point numbers by using only one array.
 * To access an element of the matrix: item(row,col) = _array[row + col*2].
 */
typedef struct dwMatrix2f
{
    alignas(16) float32_t array[2 * 2];
} dwMatrix2f;

/** Identity for dwMatrix2f */
static const dwMatrix2f DW_IDENTITY_MATRIX2F = {{(float32_t)1, (float32_t)0,
                                                 (float32_t)0, (float32_t)1}};

/** Defines a 3x3 matrix of floating point numbers by using only one array.
 * To access an element of the matrix: item(row,col) = _array[row + col*3].
 */
//# sergen(generate)
typedef struct dwMatrix3f
{
    float32_t array[3 * 3];
} dwMatrix3f;

/** Identity for dwMatrix3f */
static const dwMatrix3f DW_IDENTITY_MATRIX3F = {{(float32_t)1, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)1, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)1}};

/** Defines a 3x3 matrix of double floating point numbers by using only one array.
 * To access an element of the matrix: item(row,col) = _array[row + col*3].
 */
typedef struct dwMatrix3d
{
    float64_t array[3 * 3];
} dwMatrix3d;

/** Identity for dwMatrix3d */
static const dwMatrix3d DW_IDENTITY_MATRIX3D = {{(float32_t)1, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)1, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)1}};

/** Defines a 3x4 matrix of floating point numbers (column major) by using only one array.
 * To access an element of the matrix: item(row,col) = _array[row + col*4].
 */
typedef struct dwMatrix34f
{
    alignas(16) float32_t array[3 * 4];
} dwMatrix34f;

/** Defines a 4x4 matrix of floating point numbers (column major) by using only one array.
 * To access an element of the matrix: item(row,col) = _array[row + col*4].
 */
typedef struct dwMatrix4f
{
    alignas(16) float32_t array[4 * 4];
} dwMatrix4f;

/** Identity for dwMatrix4f */
static const dwMatrix4f DW_IDENTITY_MATRIX4F = {{(float32_t)1, (float32_t)0, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)1, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)1, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)1}};

/** Defines a 6x6 matrix of floating point numbers (column major) by using only one array.
 * To access an element of the matrix: item(row,col) = _array[row + col*6].
 */
//# sergen(generate)
typedef struct dwMatrix6f
{
    alignas(16) float32_t array[6 * 6];
} dwMatrix6f;

/** Identity for dwMatrix6f */
static const dwMatrix6f DW_IDENTITY_MATRIX6F = {{(float32_t)1, (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)1, (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)1, (float32_t)0, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)1, (float32_t)0, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)1, (float32_t)0,
                                                 (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)0, (float32_t)1}};

/** Defines a two-element single-precision floating-point vector. */
//# sergen(generate)
typedef struct dwVector2f
{
    alignas(8) float32_t x;
    float32_t y;
} dwVector2f;

/** Defines a two-element double-precision floating-point vector. */
//# sergen(generate)
typedef struct dwVector2d
{
    alignas(16) float64_t x;
    float64_t y;
} dwVector2d;

/** Defines a two-element integer vector. */
typedef struct dwVector2i
{
    alignas(8) int32_t x;
    int32_t y;
} dwVector2i;

/** Defines a two-element unsigned-integer vector. */
typedef struct dwVector2ui
{
    alignas(8) uint32_t x;
    uint32_t y;
} dwVector2ui;

/** Defines a three-element floating-point vector. */
//# sergen(generate)
typedef struct dwVector3f
{
    float32_t x;
    float32_t y;
    float32_t z;
} dwVector3f;

/** Defines a three-element double-precision floating point vector. */
//# sergen(generate)
typedef struct dwVector3d
{
    float64_t x;
    float64_t y;
    float64_t z;
} dwVector3d;

/** Defines a four-element single-precision floating point vector. */
typedef struct dwVector4f
{
    alignas(16) float32_t x;
    float32_t y;
    float32_t z;
    float32_t w;
} dwVector4f;

/** Defines a four-element double-precision floating point vector. */
typedef struct dwVector4d
{
    alignas(16) float64_t x;
    float64_t y;
    float64_t z;
    float64_t w;
} dwVector4d;

/** Defines a three-element unsigned-integer vector. */
typedef struct dwVector3ui
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
} dwVector3ui;

/** Defines a four-element unsigned-integer vector. */
typedef struct dwVector4ui
{
    alignas(16) uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
} dwVector4ui;

/**
 * @brief Defines an AABB bounding box 3D.
 * An AABB bounding box as known as axis-aligned bounding box is simply a rectangular parallelepiped
 * whose faces are each perpendicular to one of the basis vectors.
 * Use 2 opposite vertices(AA and BB) to define the AABB bounding box since it's axis-aligned.
 */
typedef struct dwBbox3Df
{
    dwVector3f aa; //!< Point AA, one of vertices of a AABB bounding box.
    dwVector3f bb; //!< Point BB, opposite vertex of AA.
} dwBbox3Df;

/**
 * @brief Defines an AABB bounding box 2D.
 * An AABB bounding box as known as axis-aligned bounding box is simply a rectangle
 * whose lines are each perpendicular to one of the basis vectors.
 * Use 2 opposite vertices(AA and BB) to define the AABB bounding box since it's axis-aligned.
 */
typedef struct dwBbox2Df
{
    dwVector2f aa; //!< Point AA, one of vertices of a AABB bounding box.
    dwVector2f bb; //!< Point BB, opposite vertex of AA.
} dwBbox2Df;

/** Defines a single-precision line segment. */
typedef struct dwLine3f
{
    //! p[0] start, p[1] end
    dwVector3f p[2];
} dwLine3f;

/** Defines a double-precision line segment. */
typedef struct dwLine3d
{
    //! p[0] start, p[1] end
    dwVector3d p[2];
} dwLine3d;

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

/** Defines a single-precision quaternion. */
//# sergen(generate)
typedef struct dwQuaternionf
{
    alignas(16) float32_t x;
    float32_t y;
    float32_t z;
    float32_t w;
} dwQuaternionf;

/**
 * Identity for dwQuaternionf
 */
static const dwQuaternionf DW_IDENTITY_QUATERNIONF = {0.f, 0.f, 0.f, 1.f};

/** Defines a double-precision quaternion. */
//# sergen(generate)
typedef struct dwQuaterniond
{
    alignas(16) float64_t x;
    float64_t y;
    float64_t z;
    float64_t w;
} dwQuaterniond;

/**
 * Identity for dwQuaterniond
 */
static const dwQuaterniond DW_IDENTITY_QUATERNIOND = {0., 0., 0., 1.};

/** Specifies a 2D transformation as a 3 x 3 matrix in column-major order.
 * The top left 2 x 2 represents rotation and scaling, the right column is the translation.
 * The bottom row is expected to be [0 0 1]
 * To access an element of the matrix: item(row,col) = _array[row + col*3].
 */
typedef struct dwTransformation2f
{
    float32_t array[3 * 3];
} dwTransformation2f;

/** Identity for dwTransformation2f */
static const dwTransformation2f DW_IDENTITY_TRANSFORMATION2F = {{1.f, 0.f, 0.f,
                                                                 0.f, 1.f, 0.f,
                                                                 0.f, 0.f, 1.f}};
/**
 * \brief Specifies a 3D rigid transformation.
 * The transformation is a 4x4 matrix in column-major order.
 * The top left 3x3 represents rotation, and the right column is the translation.
 * The bottom row is expected to be [0 0 0 1]
 * To access an element of the matrix: item(row,col) = _array[row + col*4].
 */
//# sergen(generate)
typedef struct dwTransformation3f
{
    alignas(16) float32_t array[4 * 4]; //!< 3D rigid transformation array
} dwTransformation3f;

/** Identity for dwTransformation3f */
static const dwTransformation3f DW_IDENTITY_TRANSFORMATION3F = {{1.f, 0.f, 0.f, 0.f,
                                                                 0.f, 1.f, 0.f, 0.f,
                                                                 0.f, 0.f, 1.f, 0.f,
                                                                 0.f, 0.f, 0.f, 1.f}};

/**
 * A generic side enum definition to improve consistency of objects with a 'side' concept
 * Usage: define enum entity e.g. DW_MAPS_SIDE_RIGHT = DW_SIDE_RIGHT
 */
typedef enum dwSide {
    DW_SIDE_LEFT   = 0,
    DW_SIDE_RIGHT  = 1,
    DW_SIDE_CENTER = 2
} dwSide;

/** Data structure representing an oriented bounding box in the local object coordinate frame
 * The box is defined using the center 3D point, the XYZ half axis lengths and a rotation matrix
 */
//# sergen(generate)
typedef struct dwOrientedBoundingBox3f
{
    /// Coordinate of the position of the center of the bounding box in the local frame
    dwVector3f center;
    /// Half of the width, height and depth of the box in the local frame
    dwVector3f halfAxisXYZ;
    /// Rotation matrix defining the orientation in the local frame
    dwMatrix3f rotation;
} dwOrientedBoundingBox3f;

/** Data structure representing an oriented bounding box in the local object coordinate frame
 * The box is defined using the center 2D point, the XY half axis lengths and a rotation matrix
 */
typedef struct dwOrientedBoundingBox2f
{
    /// Coordinate of the position of the center of the bounding box in the local frame
    dwVector2f center;
    /// Half of the width, and height of the box in the local frame
    dwVector2f halfAxisXY;
    /// Rotation matrix defining the orientation in the local frame
    dwMatrix2f rotation;
} dwOrientedBoundingBox2f;

/**
 * @brief Confidence structure with variance of inliers.
 *
 * Our strategy for uncertainty representation is to give classification confidence scalars for
 * classifications and confidence intervals and covariance for coordinate estimates.
 * The classification confidence scalars are straightforward, a scalar in the [0,1] interval.
 * For obstacles, lanes and lane edges we give the main classification confidence and also a sub-classification confidence.
 * This is to provide access to good information for major classes that matter and also for detailed sub-classification.
 *
 * For coordinates and other values we provide confidence and covariance which allow us to handle both outliers
 * (the small but non-zero number of cases where estimates are very wrong) and the properties of the inlier distribution.
 * We essentially provide the covariance matrix of the inlier distribution and the confidence and corresponding scaling of
 * the confidence ellipsoid. This allows to easily test whether any new value belong to the confidence interval
 * for example by checking that x' (covariance)^-1 x <= threshold'. The confidence (inlier ratio) in this case represents
 * the amount of inliers within this threshold.
 */
//# sergen(generate)
typedef struct dwConfidence1f
{
    float32_t confidence; //!< Inlier ratio [0,1]
    float32_t threshold;  //!< Inlier threshold
    float32_t variance;   //!< Variance of inliers
} dwConfidence1f;

/**
 * @brief Confidence structure with 2x2 covariance matrix.
 *
 * Refer to dwConfidence1f for more details
 */
typedef struct dwConfidence2f
{
    float32_t confidence;  //!< Inlier ratio [0,1]
    float32_t threshold;   //!< Inlier threshold
    dwMatrix2f covariance; //!< 2x2 covariance matrix
} dwConfidence2f;

/**
 * @brief Confidence structure with 3x3 covariance matrix.
 *
 * Refer to dwConfidence1f for more details
 */
//# sergen(generate)
typedef struct dwConfidence3f
{
    float32_t confidence;  //!< Inlier ratio [0,1]
    float32_t threshold;   //!< Inlier threshold
    dwMatrix3f covariance; //!< 3x3 covariance matrix
} dwConfidence3f;

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

/**
 * Location point defined by WGS84 coordinates.
 */
typedef struct dwGeoPointWGS84
{
    float64_t lon;    //!< longitude. Degree is the unit of measurement of longitude.
    float64_t lat;    //!< latitude. Degree is the unit of measurement of latitude.
    float64_t height; //!< height above WGS84 earth spheroid. Meter is the unit of measurement of height.
} dwGeoPointWGS84;

/**
* Location point defined by WGS84 coordinates without elevation
*/
typedef struct dwGeoPoint2dWGS84
{
    float64_t lon; //!< longitude
    float64_t lat; //!< latitude
} dwGeoPoint2dWGS84;

/**
 * @brief Geographic coordinate bounds
 */
typedef struct dwGeoBounds
{
    float64_t minLon; /*!< minimum longitude, west - east [-180.0:180.0) */
    float64_t minLat; /*!< minimum latitude,  south - north [-90.0:90.0) */
    float64_t maxLon; /*!< maximum longitude, west - east [-180.0:180.0) */
    float64_t maxLat; /*!< maximum latitude,  south - north [-90.0:90.0) */
} dwGeoBounds;

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_CORE_TYPES_H_

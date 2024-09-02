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
// SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Point Cloud Box Filter</b>
 *
 * @b Description: This file defines API of point cloud box filtering module.
 */

/**
 * @defgroup pointcloudfilter_group Point Cloud Filter
 * @ingroup pointcloudprocessing_group
 *
 * @brief Defines module to perform point cloud box filtering.
 *
 * @{
 */

#ifndef DW_POINTCLOUDPROCESSING_POINTCLOUDFILTER_H_
#define DW_POINTCLOUDPROCESSING_POINTCLOUDFILTER_H_

#include <dw/core/base/Types.h>
#include <dw/core/context/Context.h>
#include <dw/pointcloudprocessing/pointcloud/PointCloud.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwPointCloudBoxFilterObject* dwPointCloudBoxFilterHandle_t;

/**
 * @brief Defines box filter type
 */
typedef enum {
    DW_POINT_CLOUD_BOX_FILTER_TYPE_INNER = 0, //!< pick points inside the box
    DW_POINT_CLOUD_BOX_FILTER_TYPE_OUTER = 1  //!< pick points outside the box
} dwPointCloudBoxFilterType;

/**
 * @brief Defines parameters for point cloud box filter
 */
typedef struct
{
    bool enableCuda;                    //!< if true filtering is performed on GPU, otherwise on CPU
    bool allowDownSampling;             //!< if set, the amount of output points will automatically fit output point cloud capacity
    uint32_t maxInputPointCount;        //!< maximum size of input point cloud, only relevant for GPU version
    const dwOrientedBoundingBox3f* box; //!< oriented bounding box
    dwPointCloudBoxFilterType type;     //!< type of a filter
} dwPointCloudBoxFilterParams;

/**
 * @brief Gets default point cloud box filter parameters
 * @param[in] params Pointer to point cloud box filter parameter
 * @return DW_INVALID_ARGUMENT - if `params` is nullptr<br>
 *         DW_SUCCESS<br>
 */

/**
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudBoxFilter_getDefaultParameters(dwPointCloudBoxFilterParams* const params);

/**
 * @brief Initializes point cloud box filter
 * @param[out] obj     Handle to point cloud box filter
 * @param[in]  params  Pointer to point could filter parameter
 * @param[in]  ctx     Handle to the context
 * @return DW_INVALID_HANDLE   - if provided `ctx` is null<br>
 *         DW_INVALID_ARGUMENT - if one of `obj` or `params` is null<br>
 *         DW_SUCCESS<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwPointCloudBoxFilter_initialize(dwPointCloudBoxFilterHandle_t* const obj,
                                          dwPointCloudBoxFilterParams const* const params,
                                          dwContextHandle_t const ctx);

/**
 * @brief Resets point cloud box filter
 * @param[in] obj Handle to point cloud box filter
 * @return DW_INVALID_HANDLE - if provided `obj` is null
 *         DW_SUCCESS<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudBoxFilter_reset(dwPointCloudBoxFilterHandle_t const obj);

/**
 * @brief Releases point cloud box filter
 * @param[in] obj Handle to point cloud box filter
 * @return DW_INVALID_HANDLE - if `obj` is null
 *         DW_SUCCESS<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudBoxFilter_release(dwPointCloudBoxFilterHandle_t const obj);

/**
 * @brief Binds input point cloud buffer to filter
 * @param[in] pointCloud  Pointer to point cloud buffer
 * @param[in] obj         Handle to point cloud box filter
 * @return DW_INVALID_ARGUMENT - if provided point cloud is `nullptr` or it`s characteristics<br>
 *                               does not match thouse specified at initialization<br>
 *         DW_NOT_SUPPORTED    - if point cloud format is other than `DW_POINTCLOUD_FORMAT_XYZI`
 *         DW_INVALID_HANDLE   - if `obj` is null
 *         DW_SUCCESS<br>
 *
 * @note If using GPU implementation i.e. `enableCuda` flag has been set, memory type of
 *       input point cloud must be DW_MEMORY_TYPE_CUDA. Otherwise - `DW_MEMORY_TYPE_CPU`.
 */

/**
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudBoxFilter_bindInput(dwPointCloud const* const pointCloud,
                                         dwPointCloudBoxFilterHandle_t const obj);
/**
 * @brief Gets CUDA stream of point cloud box filter
 * @param[out] stream  Pointer to CUDA stream handle
 * @param[in]  obj     Handle to point cloud box filter
 * @return DW_INVALID_ARGUMENT - if `stream` pointer is nullptr
 *         DW_INVALID_HANDLE   - if `obj` is null
 *         DW_SUCCESS<br>
 */

/**
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudBoxFilter_getCUDAStream(cudaStream_t* const stream,
                                             dwPointCloudBoxFilterHandle_t const obj);

/**
 * @brief Sets CUDA stream of point cloud box filter
 * @param[in] stream  Handle to CUDA stream
 * @param[in] obj     Handle to point cloud box filter
 * @return DW_INVALID_HANDLE - if `obj` is null
 *         DW_SUCCESS<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudBoxFilter_setCUDAStream(cudaStream_t const stream,
                                             dwPointCloudBoxFilterHandle_t const obj);

/**
 * @brief Binds output buffers to point cloud box filter
 * @param[out] pointCloud   Pointer to output point cloud holding filtered points
 * @param[in]  obj          Handle to point cloud box filter
 * @return DW_INVALID_ARGUMENT - if provided point cloud is nullptr or does not match
 *                               characteristics specified at initialization.<br>
 *         DW_NOT_SUPPORTED    - if point cloud format is other than `DW_POINTCLOUD_FORMAT_XYZI`<br>
 *         DW_INVALID_HANDLE   - if `obj` is null<br>
 *         DW_SUCCESS<br>
 *
 * @note Memory type of input point cloud must match `enableCuda` flag specified at initialization time.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudBoxFilter_bindOutput(dwPointCloud* const pointCloud,
                                          dwPointCloudBoxFilterHandle_t const obj);

/**
 * @brief Sets bounding box for point cloud box filter
 * @param[out] boundingBox  A pointer to oriented bounding box
 * @param[in]  obj          Handle to point cloud box filter
 * @return DW_INVALID_ARGUMENT - if provided `boundingBox` is nullptr
 *         DW_INVALID_HANDLE   - if `obj` is null
 *         DW_SUCCESS<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudBoxFilter_setBoundingBox(dwOrientedBoundingBox3f* const boundingBox,
                                              dwPointCloudBoxFilterHandle_t const obj);

/**
 * @brief Sets type for point cloud box filter
 * @param[out] type Type of a filter
 * @param[in]  obj  Handle to point cloud box filter
 * @return DW_INVALID_HANDLE - if `obj` is null
 *         DW_SUCCESS<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudBoxFilter_setFilterType(dwPointCloudBoxFilterType const type,
                                             dwPointCloudBoxFilterHandle_t const obj);

/**
 * @brief Turns on/off down sampling
 * @param[in] enabled Flag to enable/disable down sampling
 * @param[in] obj  Handle to point cloud box filter
 * @return DW_INVALID_HANDLE - if `obj` is null
 *         DW_SUCCESS<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudBoxFilter_setDownSamplingEnabled(bool const enabled, dwPointCloudBoxFilterHandle_t const obj);

/**
 * @brief Applies point cloud box filter to previously bound point cloud
 * @param[in] obj Handle to point cloud box filter
 * @return DW_CALL_NOT_ALLOWED - input or output buffer has not been not bound
 *         DW_INVALID_HANDLE   - if `obj` is null
 *         DW_SUCCESS<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudBoxFilter_process(dwPointCloudBoxFilterHandle_t const obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_POINTCLOUDPROCESSING_POINTCLOUDFILTER_H_

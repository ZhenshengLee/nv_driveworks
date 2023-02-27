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
// SPDX-FileCopyrightText: Copyright (c) 2018-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Point Cloud Stitcher</b>
 *
 * @b Description: This file defines API of point cloud stitcher module.
 */

/**
 * @defgroup pointcloudstitcher_group Point Cloud Stitcher
 * @ingroup pointcloudprocessing_group
 *
 * @brief Defines module to register/stitch multiple sets of point clouds.
 *
 * @{
 */

#ifndef DW_POINTCLOUDPROCESSING_POINTCLOUDSTITCHER_H_
#define DW_POINTCLOUDPROCESSING_POINTCLOUDSTITCHER_H_

#include <dw/core/base/Types.h>
#include <dw/egomotion/Egomotion.h>
#include <dw/pointcloudprocessing/pointcloud/PointCloud.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief This defines the maximum number of point clouds anticipated to be stitched
 */
#define DW_POINTCLOUD_STITCHER_MAX_POINT_CLOUDS DW_BIND_SLOT_MAX_COUNT

typedef struct dwPointCloudStitcherObject* dwPointCloudStitcherHandle_t;

/**
 * @brief Initializes point cloud stitcher
 * @param[out] obj  Pointer to point cloud stitcher handle
 * @param[in]  ctx  Handle to the context
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPointCloudStitcher_initialize(dwPointCloudStitcherHandle_t* obj,
                                         dwContextHandle_t ctx);

/**
 * @brief Resets point cloud stitcher
 * @param[in] obj Handle to point cloud stitcher
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPointCloudStitcher_reset(dwPointCloudStitcherHandle_t obj);

/**
 * @brief Releases point cloud stitcher
 * @param[in] obj Handle to point cloud stitcher
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPointCloudStitcher_release(dwPointCloudStitcherHandle_t obj);

/**
 * @brief Gets CUDA stream of point cloud stitcher
 * @param[out] stream Pointer to CUDA stream handle
 * @param[in]  obj    Handle to point cloud stitcher
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPointCloudStitcher_getCUDAStream(cudaStream_t* stream,
                                            dwPointCloudStitcherHandle_t obj);

/**
 * @brief Sets CUDA stream of point cloud stitcher
 * @param[in] stream  Handle to CUDA stream
 * @param[in] obj     Handle to point cloud stitcher
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPointCloudStitcher_setCUDAStream(const cudaStream_t stream,
                                            dwPointCloudStitcherHandle_t obj);

/**
 * @brief Enables motion compensation for the stitched point cloud
 * @param[in] timestamp   The reference timestamp to align the point clouds with
 * @param[in] egomotion   Handle to egomotion
 * @param[in] obj         Handle to point cloud stitcher
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT - if provided egomotion or stitcher handle is null
 */
DW_API_PUBLIC
dwStatus dwPointCloudStitcher_enableMotionCompensation(dwTime_t timestamp,
                                                       dwEgomotionConstHandle_t egomotion,
                                                       dwPointCloudStitcherHandle_t obj);

/**
 * @brief Disables motion compensation for the stitched point cloud
 * @param[in] obj Handle to point cloud stitcher
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPointCloudStitcher_disableMotionCompensation(dwPointCloudStitcherHandle_t obj);

/**
 * @brief Sets global transformation for the stitched point cloud
 * @param[in] tx          Pointer to the transformation that transforms the stitched point <br>
 *                        cloud to one common coordinate system. Parameter is optional, can <br>
 *                        be nullptr. In this case identity transformation will be used.
 * @param[in] obj         Handle to point cloud stitcher
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPointCloudStitcher_setGlobalTransformation(const dwTransformation3f* tx,
                                                      dwPointCloudStitcherHandle_t obj);

/**
 * @brief Binds input point cloud to the point cloud stitcher
 * @param[in] slot destination slot
 * @param[in] pointCloud  Pointer to the input point cloud
 * @param[in] transform  Pointer to point cloud transformation
 * @param[in] obj Handle to the point cloud stitcher
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE - provided handle to point clou stitcher is null<br>
 *         DW_OUT_OF_BOUNDS  - slot index is out of bounds
 */
DW_API_PUBLIC
dwStatus dwPointCloudStitcher_bindInput(dwBindSlot slot,
                                        const dwPointCloud* pointCloud,
                                        const dwTransformation3f* transform,
                                        dwPointCloudStitcherHandle_t obj);

/**
 * @brief Binds output buffer to the point cloud stitcher
 * @param[out] points  Pointer to output buffer which stores the stitched/motion compensated point cloud
 * @param[in]  obj     Handle to the point cloud stitcher
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPointCloudStitcher_bindOutput(dwPointCloud* points,
                                         dwPointCloudStitcherHandle_t obj);

/**
 * @brief Transforms all the input point clouds to a common coordinate
 * @param[in] obj Handle to the point cloud sitcher
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT    - the memory type of input/output does not match <br>
 *         DW_CALL_NOT_ALLOWED    - input list is not bound <br>
 *         DW_OUT_OF_BOUNDS       - the number of input points exceeds the output buffer capacity
 */
DW_API_PUBLIC
dwStatus dwPointCloudStitcher_process(dwPointCloudStitcherHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_POINTCLOUDPROCESSING_POINTCLOUDSTITCHER_H_

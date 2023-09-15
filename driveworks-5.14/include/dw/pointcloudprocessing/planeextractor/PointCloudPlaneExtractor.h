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
// SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Point Cloud Plane Extractor</b>
 *
 * @b Description: This file defines API of point cloud plane extractor module.
 */

/**
 * @defgroup pointcloudplaneextractor_group Point Cloud Plane Extractor
 * @ingroup pointcloudprocessing_group
 *
 * @brief Defines module to extract one 3D plane nearby the sensor.
 *
 * @{
 */

#ifndef DW_POINTCLOUDPROCESSING_POINTCLOUDPLANEEXTRACTOR_H_
#define DW_POINTCLOUDPROCESSING_POINTCLOUDPLANEEXTRACTOR_H_

#include <dw/core/base/Types.h>
#include <dw/core/context/Context.h>
#include <dw/pointcloudprocessing/pointcloud/PointCloud.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwPointCloudPlaneExtractorObject* dwPointCloudPlaneExtractorHandle_t;

/**
 * @brief Defines parameters for point cloud box filter
 * @note A filtering is performed on point cloud before running
 *       ransac algorithm.
 */
typedef struct
{
    uint32_t maxPointCount;      //!< maximum number of accepted points
    dwOrientedBoundingBox3f box; //!< oriented bounding box
} dwPlaneExtractorBoxFilterParams;

/**
 * @brief Defines plane extraction parameter
 * @note This module first applies ransac estimation to fit a 3D ground plane.
 *       It also checks the validility of the plane fitting via non-linear optimization.
 *       It will indicate a invalid plane fitting if the optimization failed to converge.
 */
typedef struct
{
    uint32_t maxInputPointCount;                     //!< maximum number of points in input point cloud
    uint32_t ransacIterationCount;                   //!< ransac iteration number
    uint32_t optimizerIterationCount;                //!< optimization iteration number
    float32_t minInlierFraction;                     //!< minimum inlier percentage for ransac plane fitting
    float32_t maxInlierDistance;                     //!< maximum inlier distance to the estimated plane
    dwMatrix3f rotation;                             //!< rotation that aligns the point cloud with ground plane
    bool cudaPipelineEnabled;                        //!< Setting to true will process with CUDA pipeline
    dwPlaneExtractorBoxFilterParams boxFilterParams; //!< box filter parameters
} dwPointCloudPlaneExtractorParams;

/**
 * @brief Defines extracted 3D plane
 */
typedef struct
{
    dwTransformation3f transformation; //!< rotation and translation of the plane given the estimated normal vector and plane offset
    dwVector3f normal;                 //!< normal vector of the ground plane
    float32_t offset;                  //!< offset distance of the ground plane to the coordinate origin
    bool valid;                        //!< If this is false, it indicates the ransac plane fitting and
                                       //!< optimization failed to produce a 3D ground plane, user should
                                       //!< not use the estimated normal and offset
} dwPointCloudExtractedPlane;

/**
 * @brief Initializes point cloud plane extractor
 * @param[out] obj     Handle to point cloud plane extractor
 * @param[in]  params  Pointer to point could plane extractor parameter
 * @param[in]  ctx     Handle to the context
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPCPlaneExtractor_initialize(dwPointCloudPlaneExtractorHandle_t* const obj,
                                       dwPointCloudPlaneExtractorParams const* const params,
                                       dwContextHandle_t const ctx);

/**
 * @brief Resets point cloud plane extractor
 * @param[in] obj Handle to point cloud plane extractor
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPCPlaneExtractor_reset(dwPointCloudPlaneExtractorHandle_t const obj);

/**
 * @brief Releases point cloud plane extractor
 * @param[in] obj Handle to point cloud plane extractor
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPCPlaneExtractor_release(dwPointCloudPlaneExtractorHandle_t const obj);

/**
 * @brief Gets default point cloud plane extractor parameters
 * @param[in] params Pointer to point cloud plane extractor parameter
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPCPlaneExtractor_getDefaultParameters(dwPointCloudPlaneExtractorParams* const params);

/**
 * @brief Binds point cloud buffer to plane extractor
 * @param[in] pointCloud  Pointer to point cloud buffer
 * @param[in] obj         Handle to point cloud plane extractor
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
*/
DW_API_PUBLIC
dwStatus dwPCPlaneExtractor_bindInput(dwPointCloud const* const pointCloud,
                                      dwPointCloudPlaneExtractorHandle_t const obj);
/**
 * @brief Gets CUDA stream of point cloud plane extractor
 * @param[out] stream  Pointer to CUDA stream handle
 * @param[in]  obj     Handle to point cloud plane extractor
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPCPlaneExtractor_getCUDAStream(cudaStream_t* const stream,
                                          dwPointCloudPlaneExtractorHandle_t const obj);

/**
 * @brief Sets CUDA stream of point cloud plane extractor
 * @param[in] stream  Handle to CUDA stream
 * @param[in] obj     Handle to point cloud plane extractor
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPCPlaneExtractor_setCUDAStream(cudaStream_t const stream,
                                          dwPointCloudPlaneExtractorHandle_t const obj);

/**
 * @brief Binds output buffers to point cloud plane extractor
 * @param[out] inliers      Pointer to output buffer stores inlier points (optional, can be null)
 * @param[out] outliers     Pointer to output buffer stores outlier points (optional, can be null)
 * @param[out] outputPlane  Pointer to extracted 3D plane
 * @param[in]  obj          Handle to point cloud plane extractor
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPCPlaneExtractor_bindOutput(dwPointCloud* const inliers,
                                       dwPointCloud* const outliers,
                                       dwPointCloudExtractedPlane* const outputPlane,
                                       dwPointCloudPlaneExtractorHandle_t const obj);

/**
 * @brief Extracts 3D ground plane and stores the results to output buffer
 * @param[in] obj Handle to point cloud plane extractor
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT    - input/output buffer memory type does not match with
 *                                the variable `cudaPipelineEnabled` in `dwPointCloudPlaneExtractorParams` <br>
 *         DW_CALL_NOT_ALLOWED    - user did not bind input buffer and output plane
 */
DW_API_PUBLIC
dwStatus dwPCPlaneExtractor_process(dwPointCloudPlaneExtractorHandle_t const obj);

/////////////////////////////////////////////////////////////////////////////////////////
// DEPRECATED API FUNCTIONS

/**
 * @brief Initializes point cloud plane extractor
 * @param[out] obj     Handle to point cloud plane extractor
 * @param[in]  params  Pointer to point could plane extractor parameter
 * @param[in]  ctx     Handle to the context
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudPlaneExtractor_initialize() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCPlaneExtractor_initialize() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudPlaneExtractor_initialize(dwPointCloudPlaneExtractorHandle_t* const obj,
                                               dwPointCloudPlaneExtractorParams const* const params,
                                               dwContextHandle_t const ctx);

/**
 * @brief Resets point cloud plane extractor
 * @param[in] obj Handle to point cloud plane extractor
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudPlaneExtractor_reset() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCPlaneExtractor_reset() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudPlaneExtractor_reset(dwPointCloudPlaneExtractorHandle_t const obj);

/**
 * @brief Releases point cloud plane extractor
 * @param[in] obj Handle to point cloud plane extractor
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudPlaneExtractor_release() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCPlaneExtractor_release() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudPlaneExtractor_release(dwPointCloudPlaneExtractorHandle_t const obj);

/**
 * @brief Gets default point cloud plane extractor parameters
 * @param[in] params Pointer to point cloud plane extractor parameter
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudPlaneExtractor_getDefaultParameters() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCPlaneExtractor_getDefaultParameters() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudPlaneExtractor_getDefaultParameters(dwPointCloudPlaneExtractorParams* const params);

/**
 * @brief Binds point cloud buffer to plane extractor
 * @param[in] pointCloud  Pointer to point cloud buffer
 * @param[in] obj         Handle to point cloud plane extractor
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
*/
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudPlaneExtractor_bindInput() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCPlaneExtractor_bindInput() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudPlaneExtractor_bindInput(dwPointCloud const* const pointCloud,
                                              dwPointCloudPlaneExtractorHandle_t const obj);
/**
 * @brief Gets CUDA stream of point cloud plane extractor
 * @param[out] stream  Pointer to CUDA stream handle
 * @param[in]  obj     Handle to point cloud plane extractor
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudPlaneExtractor_getCUDAStream() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCPlaneExtractor_getCUDAStream() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudPlaneExtractor_getCUDAStream(cudaStream_t* const stream,
                                                  dwPointCloudPlaneExtractorHandle_t const obj);

/**
 * @brief Sets CUDA stream of point cloud plane extractor
 * @param[in] stream  Handle to CUDA stream
 * @param[in] obj     Handle to point cloud plane extractor
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudPlaneExtractor_setCUDAStream() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCPlaneExtractor_setCUDAStream() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudPlaneExtractor_setCUDAStream(cudaStream_t const stream,
                                                  dwPointCloudPlaneExtractorHandle_t const obj);

/**
 * @brief Binds output buffers to point cloud plane extractor
 * @param[out] inliers      Pointer to output buffer stores inlier points (optional, can be null)
 * @param[out] outliers     Pointer to output buffer stores outlier points (optional, can be null)
 * @param[out] outputPlane  Pointer to extracted 3D plane
 * @param[in]  obj          Handle to point cloud plane extractor
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudPlaneExtractor_bindOutput() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCPlaneExtractor_bindOutput() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudPlaneExtractor_bindOutput(dwPointCloud* const inliers,
                                               dwPointCloud* const outliers,
                                               dwPointCloudExtractedPlane* const outputPlane,
                                               dwPointCloudPlaneExtractorHandle_t const obj);

/**
 * @brief Extracts 3D ground plane and stores the results to output buffer
 * @param[in] obj Handle to point cloud plane extractor
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT    - input/output buffer memory type does not match with
 *                                the variable `cudaPipelineEnabled` in `dwPointCloudPlaneExtractorParams` <br>
 *         DW_CALL_NOT_ALLOWED    - user did not bind input buffer and output plane
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudPlaneExtractor_process() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCPlaneExtractor_process() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudPlaneExtractor_process(dwPointCloudPlaneExtractorHandle_t const obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_POINTCLOUDPROCESSING_POINTCLOUDPLANEEXTRACTOR_H_

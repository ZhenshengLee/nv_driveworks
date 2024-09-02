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
// SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Point Cloud ICP</b>
 *
 * @b Description: This file defines API of Point Cloud ICP module.
 */

/**
 * @defgroup pointcloudicp_group Point Cloud ICP
 * @ingroup pointcloudprocessing_group
 *
 * @brief Defines Point Cloud ICP module to align point clouds using iterative closest point algorithms.
 *
 * @{
 */

#ifndef DW_POINTCLOUDPROCESSING_POINTCLOUDICP_H_
#define DW_POINTCLOUDPROCESSING_POINTCLOUDICP_H_
#include <dw/core/context/Context.h>
#include <dw/pointcloudprocessing/pointcloud/PointCloud.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwPointCloudICPObject* dwPointCloudICPHandle_t;

/**
 * @brief Defines a type of the Iterative Closest Point (ICP) algorithm
 */
typedef enum dwPointCloudICPType {
    //! Grid based depthmap representation for the lidar point cloud.
    //! It is assumed that the structured point cloud passed for alignment was generated from a spinning lidar
    DW_POINT_CLOUD_ICP_TYPE_DEPTH_MAP

} dwPointCloudICPType;

/**
 * @brief Defines point cloud icp parameter
 */
typedef struct dwPointCloudICPParams
{
    //! Type of the ICP implementation to be used
    dwPointCloudICPType icpType;

    //! Maximum number of points that will be pushed to ICP optimization
    uint32_t maxPoints;

    //! If icpType is DW_POINT_CLOUD_ICP_TYPE_DEPTH_MAP, this defines the size of depthmap
    dwVector2ui depthmapSize;

    //! Angle convergence tolerance, change in radians between two consecutive iteration steps
    float32_t angleConvergenceTol;

    //! Distance convergence tolerance, change in units between two consecutive iteration steps
    float32_t distanceConvergenceTol;

    //! Maximum number of iterations which need to be executed
    uint16_t maxIterations;

    //! Controls whether or not ICP uses the initialization pose as a prior in the optimization
    bool usePriors;
} dwPointCloudICPParams;

//! Callback function to be executed by ICP module allowing user to overwrite default convergence criteria method.
//! The method will receive original and new transformations as computed by the ICP module
typedef bool (*dwPointCloudICPConvergenceCheck)(const dwTransformation3f* prevSrc2Tgt, const dwTransformation3f* newSrc2Tgt, void* userData);

/**
 * @brief Resulting statistics about the latest ICP run
 *
 * This includes statistics about how well ICP worked (residual errors, number of iterations, inlier fraction).
 */
typedef struct dwPointCloudICPResultStats
{
    //! How many iterations were actually performed
    uint16_t actualNumIterations;
    //! Weighted root mean square (RMS) cost after last ICP iteration
    float32_t rmsCost;
    //! Fraction of points which are inliers to the final ICP pose
    float32_t inlierFraction;
    //! Number of 3D points which qualify as valid correspondences
    uint32_t numCorrespondences;

} dwPointCloudICPResultStats;

/**
 * @brief Initializes point cloud icp
 * @param[out] obj     Handle to point cloud icp
 * @param[in]  params  Pointer to point cloud icp parameters
 * @param[in]  ctx     Handle to the context 
 * @retval DW_INVALID_ARGUMENT when at least one of input arguments is not valid or an ICP type is not valid
 * @retval DW_SUCCESS when operation succeeded
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwPointCloudICP_initialize(dwPointCloudICPHandle_t* obj,
                                    const dwPointCloudICPParams* params,
                                    dwContextHandle_t ctx);
/**
 * @brief Resets pointers to the source, target point clouds and an output pose to a null pointer value
 * @param[in] obj Handle to point cloud icp
 * @retval DW_INVALID_HANDLE when handle to point cloud icp is not valid
 * @retval DW_SUCCESS when operation succeeded.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudICP_reset(dwPointCloudICPHandle_t obj);

/**
 * @brief Releases a handle of a point cloud icp created using 'dwPointCloudICP_initialize'
 * @param[in] obj Handle to point cloud icp
 * @retval DW_INVALID_HANDLE when handle to point cloud icp is not valid
 * @retval DW_SUCCESS when operation succeeded
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudICP_release(dwPointCloudICPHandle_t obj);

/**
 * @brief Gets default values of dwPointCloudICPParams object 
 * @param[out] params Pointer to point cloud icp parameters 
 * @retval DW_INVALID_ARGUMENT when handle to icp parameters is not valid
 * @retval DW_SUCCESS when operation succeeded
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudICP_getDefaultParams(dwPointCloudICPParams* params);

/**
 * @brief Gets CUDA stream of point cloud icp
 * @param[out] stream  Pointer to CUDA stream handle
 * @param[in]  obj     Handle to point cloud icp 
 * @retval DW_INVALID_HANDLE when handle to point cloud icp is not valid
 * @retval DW_INVALID_ARGUMENT when handle a CUDA stream is not valid
 * @retval DW_SUCCESS when operation succeeded
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudICP_getCUDAStream(cudaStream_t* stream,
                                       dwPointCloudICPHandle_t obj);

/**
 * @brief Sets CUDA stream of point cloud icp
 * @param[in] stream  Handle to CUDA stream
 * @param[in] obj     Handle to point cloud icp 
 * @retval DW_INVALID_HANDLE when handle to point cloud icp or a CUDA stream is not valid 
 * @retval DW_SUCCESS when operation succeeded
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudICP_setCUDAStream(cudaStream_t const stream,
                                       dwPointCloudICPHandle_t obj);

/**
 * @brief Binds input buffers of a source and a target point clouds as well as a buffer with an initial transformation
 * @param[in] sourcePCD       Pointer to source point cloud input buffer. Must be in cartesian coordinate space.
 * @param[in] targetPCD       Pointer to target point cloud input buffer. Must be in cartesian coordinate space.
 * @param[in] sourceToTarget  Pointer to the initial transformation <br>
 *                            that transforms from source to target point cloud
 * @param[in] obj             Handle to point cloud icp 
 * @retval DW_INVALID_ARGUMENT when at least one of input arguments is not valid
 * @retval DW_INVALID_HANDLE when handle to point cloud icp is not valid 
 * @retval DW_SUCCESS when operation succeeded
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudICP_bindInput(const dwPointCloud* sourcePCD,
                                   const dwPointCloud* targetPCD,
                                   const dwTransformation3f* sourceToTarget,
                                   dwPointCloudICPHandle_t obj);

/**
 * @brief Binds an output buffer to the resulting transformation between a source and a target point clouds
 * @param[out] pose  Pointer to the transformation that aligns <br>
 *                   the source point cloud with target point cloud
 * @param[in]  obj   Handle to point cloud icp module 
 * @retval DW_INVALID_ARGUMENT when handle to a transformation is not valid 
 * @retval DW_INVALID_HANDLE when handle to point cloud icp is not valid 
 * @retval DW_SUCCESS when operation succeeded
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudICP_bindOutput(dwTransformation3f* pose,
                                    dwPointCloudICPHandle_t obj);

/**
 * @brief Dryrun the point cloud estimation on GPU to record CUDAGraph.
 * PointCloudICP uses GPU for the estimation. The GPU implementation 
 * uses CUDA graph based pipeline which requires a recording step to avoid runtime allocations 
 * when running. This call prepares this fixed function pipeline. It has to be executed during 
 * initialization phase in safety platform otherwise error will be reported.
 * 
 * @param[in] obj Handle to point cloud icp
 * 
 * @retval DW_INVALID_HANDLE when provided point cloud icp handle is invalid, i.e., null or of wrong type
 * @retval DW_CUDA_ERROR recording failed
 * @retval DW_SUCCESS when operation succeeded
 * 
 * @note safety platform MUST call this API before dwPointCloudICP_process,
 *       Non-safety platform can call dwPointCloudICP_process directly
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudICP_dryrunEstimation(dwPointCloudICPHandle_t obj);

/**
 * @brief Estimates the transformation aligns two PointClouds
 * @param[in] obj Handle to point cloud icp
 * @retval DW_INVALID_HANDLE when handle to point cloud icp is not valid 
 * @retval DW_CALL_NOT_ALLOWED - when input or output buffers are not bound
 * @retval DW_SUCCESS when operation succeeded
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloudICP_process(dwPointCloudICPHandle_t obj);

/**
* Allows to set a user-defined callback to be executed for ICP convergence test. If nullptr, the default test will be used.
*
* @param[in] callback Method to execute to test for convergence
* @param[in] userData User data to be passed to the convergence callback
* @param[in]  obj Handle to point cloud icp
*
* @retval DW_INVALID_HANDLE when handle to point cloud icp is not valid 
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwPointCloudICP_setConvergenceCriteriaCallback(dwPointCloudICPConvergenceCheck callback, void* userData, dwPointCloudICPHandle_t obj);

/**
* Set tolerances used by the default ICP convergence criteria method.
*
* @param[in] angleTol Change in angular parameters between two consecutive iteration steps, radians
* @param[in] distanceTol Change in translation between two consecutive iteration steps, the same units as a source point cloud
* @param[in] obj The initialized pointcloud ICP Module.
*
* @retval DW_INVALID_HANDLE when handle to point cloud icp is not valid 
* @retval DW_SUCCESS when operation succeeded
*
* @note both tolerances has to be fulfilled, to declare convergence (i.e. it is a binary AND).
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwPointCloudICP_setConvergenceTolerance(float32_t angleTol, float32_t distanceTol, dwPointCloudICPHandle_t obj);

/**
* Set maximum number of iterations which need to be executed. Note that ICP might converge earlier, due to
* the tolerances set in `dwPointCloudICP_setConvergenceTolerance()`.
*
* @param[in] maxIterations Maximal number of iterations to execute.
* @param[in]  obj The initialized pointcloud ICP Module.
*
* @retval DW_INVALID_HANDLE when handle to point cloud icp is not valid 
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwPointCloudICP_setMaxIterations(uint16_t maxIterations, dwPointCloudICPHandle_t obj);

/**
* Get statistics about latest point cloud ICP run, returns the costs for last pose (see a description of 'dwPointCloudICPResultStats' for details)
*
* @param[out]  resultStats Struct with stats about latest ICP run
* @param[in]   obj Handle to point cloud icp
*
* @retval DW_INVALID_HANDLE when handle to point cloud icp is not valid 
* @retval DW_INVALID_ARGUMENT when handle to statistic results is not valid
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwPointCloudICP_getLastResultStats(dwPointCloudICPResultStats* resultStats, dwPointCloudICPHandle_t obj);

/**
* Get the maximum allowed size of the depth map in number of points supported by the ICP implementation.
* Width x height of the depth map has to be less than this number.
* @param[out]  maxDepthMapSize - return maximal size in number of points supported by the ICP implementation.
*
* @retval DW_INVALID_HANDLE when handle to point cloud icp is not valid 
* @retval DW_SUCCESS when operation succeeded
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwPointCloudICP_getMaximumDepthMapSize(uint32_t* maxDepthMapSize);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_POINTCLOUDPROCESSING_POINTCLOUDICP_H_

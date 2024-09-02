/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

/**
 * @file
 * <b>NVIDIA DriveWorks API: Point Cloud Processing</b>
 *
 * @b Description: This file defines API of point cloud processing module
 */

/**
 * @defgroup pointcloudprocessing_group Point Cloud Processing Interface
 *
 * @brief Defines point cloud motion compensation module
 *
 * @{
 */

#ifndef DW_POINTCLOUDPROCESSING_MOTIONCOMPENSATOR_MOTIONCOMPENSATOR_H_
#define DW_POINTCLOUDPROCESSING_MOTIONCOMPENSATOR_MOTIONCOMPENSATOR_H_

#include <dw/pointcloudprocessing/lidarpointcloud/LidarPointCloud.h>
#include <dw/egomotion/base/EgomotionState.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwMotionCompensatorObject* dwMotionCompensatorHandle_t;
typedef struct dwMotionCompensatorObject const* dwMotionCompensatorConstHandle_t;

/**
 * @brief Defines transform interpolation strategy
 */
typedef enum {
    //! Default interpolation strategy applying linear interpolation
    DW_PC_MOTION_COMPENSATOR_INTERPOLATION_LINEAR = 0,

} dwMotionCompensatorInterpolationStrategy;

/**
 * @brief Defines point cloud motion compensator parameters
 */
typedef struct
{
    bool enableCuda; //!< If set the module operate on GPU point clouds

    uint32_t maxPointCloudSize; //!< Maximum number of points in input/output point cloud

    dwMotionCompensatorInterpolationStrategy interpolationStrategy; //!< Interpolation strategy

    uint32_t motionModelResolution; //!< Number of transfomations calculated at equal time intervals
                                    //!< between start and the end of a spin

    bool outputInRigCoordinates; //!< If `true` output points are transformed to rig coordinate system

    dwTransformation3f pointCloudToRig; //!< Transformation aligns the lidar sensor with the platform that produces the egomotion

} dwMotionCompensatorParams;

/**
 * @brief Get default parameters for motion compensator
 *
 * @param[out] params   Pointer to parameters structure
 * @param[in] lidarProps Lidar properties the module is supposed to be used with
 *
 * @return DW_INVALID_ARGUMENT - if given pointer is NULL<br>
 *         DW_SUCCESS - in all other cases
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwMotionCompensator_getDefaultParams(dwMotionCompensatorParams* params, dwLidarProperties const* lidarProps);

/**
 * @brief Initialize motion compensator module
 *
 * @param[out] obj                 Pointer to motion compensator handle
 * @param[in] params               Initialization parameters
 * @param[in]  ctx                 Handle to the context
 *
 * @return DW_INVALID_ARGUMENT - if given pointer is NULL<br>
 *         DW_INVALID_HANDLE - if given handle is invalid<br>
 *         DW_SUCCESS - in all other cases
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwMotionCompensator_initialize(dwMotionCompensatorHandle_t* obj,
                                        dwMotionCompensatorParams const* params,
                                        dwContextHandle_t ctx);

/**
 * @brief Reset motion compensator
 *
 * @param[in] obj  Motion compensator handle
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid<br>
 *         DW_SUCCESS - in all other cases
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwMotionCompensator_reset(dwMotionCompensatorHandle_t obj);

/**
 * @brief Release motion compensator
 *
 * @param[in] obj Motion compensator handle
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid<br>
 *         DW_SUCCESS - in all other cases
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwMotionCompensator_release(dwMotionCompensatorHandle_t obj);

/**
 * @brief Set CUDA stream of motion compensator
 *
 * @param[in] stream  Handle to CUDA stream
 * @param[in] obj     Motion compensator handle
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid<br>
 *         DW_CALL_NOT_ALLOWED   - motion compensator is initialized to process host memory <br>
 *         DW_SUCCESS - in all other cases
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwMotionCompensator_setCUDAStream(const cudaStream_t stream,
                                           dwMotionCompensatorHandle_t obj);
/**
 * @brief Get CUDA stream of motion compensator
 *
 * @param[out] stream  Pointer to CUDA stream handle
 * @param[in]  obj     Motion compensator handle
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid<br>
 *         DW_CALL_NOT_ALLOWED   - motion compensator is initialized to process host memory<br>
 *         DW_SUCCESS - in all other cases
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwMotionCompensator_getCUDAStream(cudaStream_t* stream,
                                           dwMotionCompensatorConstHandle_t obj);

/**
 * @brief Bind egomotion state handle
 *
 * @param[out] motionState  A handle to egomotion state
 * @param[in]  obj         Motion compensator handle
 * @return DW_INVALID_HANDLE - if one of given handles is invalid<br>
 *         DW_SUCCESS - in all other cases
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwMotionCompensator_bindEgomotionState(dwConstEgomotionStateHandle_t motionState,
                                                dwMotionCompensatorHandle_t obj);

/**
 * @brief Update point cloud transformation to egomotion coordinate frame
 *
 * @param[in] transform  Transformation aligning the lidar sensor with the platform that produces the egomotion
 * @param[in]  obj       Motion compensator handle
 * @return DW_INVALID_HANDLE     - if one of given handles is invalid<br>
 *         DW_SUCCESS - in all other cases
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwMotionCompensator_setTransform(dwTransformation3f const* transform,
                                          dwMotionCompensatorHandle_t obj);

/**
 * @brief Set reference time for motion compensation
 *
 * @param[out] timestamp  Timestamp for motion compensation
 * @param[in]  obj        Motion compensator handle
 * @return DW_INVALID_HANDLE - if one of given handles is invalid<br>
 *         DW_SUCCESS - in all other cases
 *
 * @note By default ending point cloud timestamp is used as a reference
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwMotionCompensator_setReferenceTimestamp(dwTime_t const timestamp,
                                                   dwMotionCompensatorHandle_t obj);

/**
 * @brief Bind input point cloud
 *
 * @param[out] pointCloud               Input point cloud
 * @param[in]  timestampChannelIndex    Index of aux channel holding timestamp information
 * @param[in]  obj                      Motion compensator handle
 * @return DW_INVALID_HANDLE     - if one of given handles is invalid<br>
 *         DW_SUCCESS - in all other cases
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwMotionCompensator_bindInput(dwPointCloud const* pointCloud,
                                       uint32_t const timestampChannelIndex,
                                       dwMotionCompensatorHandle_t obj);

/**
 * @brief Bind input lidar point cloud
 *
 * @param[out] pointCloud               Input lidar point cloud
 * @param[in]  obj                      Motion compensator handle
 * @return DW_INVALID_HANDLE     - if one of given handles is invalid<br>
 *         DW_SUCCESS - in all other cases
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwMotionCompensator_bindInputLidarPointCloud(dwLidarPointCloud const* pointCloud,
                                                      dwMotionCompensatorHandle_t obj);

/**
 * @brief Bind output point cloud
 *
 * @param[out] pointCloud  Output point cloud
 * @param[in]  obj         Motion compensator handle
 * @return DW_INVALID_HANDLE - if one of given handles is invalid<br>
 *         DW_SUCCESS - in all other cases
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwMotionCompensator_bindOutput(dwPointCloud* pointCloud,
                                        dwMotionCompensatorHandle_t obj);

/**
 * @brief Bind output lidar point cloud
 *
 * @param[out] pointCloud  Output lidar point cloud
 * @param[in]  obj         Motion compensator handle
 * @return DW_INVALID_HANDLE - if one of given handles is invalid<br>
 *         DW_SUCCESS - in all other cases
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwMotionCompensator_bindOutputLidarPointCloud(dwLidarPointCloud* pointCloud,
                                                       dwMotionCompensatorHandle_t obj);

/**
 * @brief Perform motion compensation
 *
 * @param[in] obj Module handle
 *
 * @return DW_INVALID_HANDLE   - if given handle is invalid<br>
 *         DW_CALL_NOT_ALLOWED - if input or output is not bound or their memory type is different
 *         DW_SUCCESS - in all other cases
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwMotionCompensator_process(dwMotionCompensatorHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_POINTCLOUDPROCESSING_MOTIONCOMPENSATOR_MOTIONCOMPENSATOR_H_

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
// SPDX-FileCopyrightText: Copyright (c) 2015-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Structure from Motion Methods</b>
 *
 * @b Description: This file defines structure from motion methods.
 */

/**
 * @defgroup sfm_group Structure from Motion Interface
 *
 * @brief Recovers camera pose and scene structure from visual matches.
 *
 * @{
 */

#ifndef DW_IMAGEPROCESSING_SFM_SFM_H_
#define DW_IMAGEPROCESSING_SFM_SFM_H_

#include <dw/core/base/Config.h>
#include <dw/rig/Rig.h>
#include <dw/calibration/cameramodel/CameraModel.h>
#include <dw/core/base/Exports.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>

#include <dw/imageprocessing/features/FeatureList.h>

#include <cuda_runtime_api.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup reconstructor_group Reconstructor
 *
 * @brief Performs pose estimation and feature triangulation from visual matches.
 * @{
 */

///////////////////////////////////////////////////////////////////////
// dwReconstructor

/** Handle representing a reconstructor object.
 * This object performs pose estimation and feature triangulation from visual matches.
 */
typedef struct dwReconstructorObject* dwReconstructorHandle_t;

/** Handle representing a const reconstructor object. */
typedef struct dwReconstructorObject const* dwConstReconstructorHandle_t;

/**
* Configuration parameters for a reconstructor.
*/
typedef struct dwReconstructorConfig
{
    ///Specifies the rig to use for reconstruction.
    dwConstRigHandle_t rig;

    ///Specifies the maximum number of features for each camera.
    uint32_t maxFeatureCount;

    ///Specifies the maximum size of the history.
    uint32_t maxPoseHistoryLength;

    ///Specifies the minimum cosine of the angle between two optical rays to add a new one to the
    ///feature history.
    float32_t minNewObservationAngleRad;

    ///Specifies the minimum distance between vehicle poses to add a new one to the list.
    /// \note Not implemented
    float32_t minRigDistance;

    ///Specifies the minimum number of entries in the feature history needed for triangulation.
    uint8_t minTriangulationEntries;

    ///Specifies the max angle of the reprojection error (angle between tracked optical ray and triangulated
    ///optical ray) to consider the feature an outlier during triangulation. Observations with a reprojection
    ///error higher than this after triangulation are discarded.
    float32_t maxReprojectionErrorAngleRad;

    ///Specifies the max angle of the reprojection error (angle between tracked optical ray and triangulated
    ///optical ray) to consider the feature an outlier during pose estimation. Observations with a reprojection
    ///error higher than this will have less influence in the pose estimation.
    float32_t poseEstimationOutlierThresholdRad;
} dwReconstructorConfig;

/**
 * Initializes the reconstructor config with default values.
 *
 * @param[out] config Config to initialize.
 *
 * @return DW_SUCCESS - Operation completed successfully<br>
 *         DW_INVALID_ARGUMENT - One of the given arguments cannot be processed by the function<br>
 */
DW_API_PUBLIC
dwStatus dwReconstructor_initConfig(dwReconstructorConfig* config);

/**
 * Creates and initializes a reconstructor.
 *
 * @param[out] obj A pointer to the reconstructor handle is returned here.
 * @param[in] config Spcifies the configuration parameters for the reconstructor.
 * @param[in] stream Specifies the CUDA stream to use for all reconstructor operations.
 * @param[in] context Specifies the handle to the context under which it is created.
 *
 * @return DW_SUCCESS - Operation completed successfully<br>
 *         DW_INVALID_HANDLE - If the given context handle is invalid,i.e. null or of wrong type<br>
 *         DW_INVALID_ARGUMENT - One of the given arguments cannot be processed by the function<br>
 */
DW_API_PUBLIC
dwStatus dwReconstructor_initialize(dwReconstructorHandle_t* obj,
                                    const dwReconstructorConfig* config,
                                    cudaStream_t stream,
                                    dwContextHandle_t context);

/**
 * Resets a reconstructor.
 *
 * @param[in] obj Specifies the reconstructor handle to reset.
 *
 * @return DW_SUCCESS - Operation completed successfully<br>
 *         DW_INVALID_HANDLE - If the given context handle is invalid,i.e. null or of wrong type<br>
 *         DW_INVALID_ARGUMENT - One of the given arguments cannot be processed by the function<br>
 */
DW_API_PUBLIC
dwStatus dwReconstructor_reset(dwReconstructorHandle_t obj);

/**
 * Releases a reconstructor.
 * This method releases all resources associated with a reconstructor.
 *
 *
 * @param[in] obj Specifies the object handle to release.
 *
 * @return DW_SUCCESS - Operation completed successfully<br>
 *         DW_INVALID_HANDLE - If the given context handle is invalid,i.e. null or of wrong type<br>
 *
 */
// @note This method renders the handle unusable.
DW_API_PUBLIC
dwStatus dwReconstructor_release(dwReconstructorHandle_t obj);

/**
* Marks the cameras to use for pose estimation. Cameras with their flag set to false
* are ignored and the value of the feature list and pointers are ignored for that
* camera during pose estimation.
*
* @param[in] enabled Array of flags to indicate that a camera is enabled. Camera i is enabled if
*               enabled[i] != 0. Size of the buffer must be the same as the rig's camera count.
* @param[in] obj The reconstructor object handle.
*
* @return DW_SUCCESS - Operation completed successfully<br>
*         DW_INVALID_HANDLE - If the given context handle is invalid,i.e. null or of wrong type  <br>
*         DW_INVALID_ARGUMENT - One of the given arguments cannot be processed by the function<br>
*
* \ingroup sfm
*/
DW_API_PUBLIC
dwStatus dwReconstructor_enableCamerasForPoseEstimation(uint8_t const enabled[], dwReconstructorHandle_t obj);

/**
* Uses all tracked features from all cameras to estimate the current rig pose.
*
* Processes all cameras simultaneously.
*
* @param[in] previousRig2World A pointer to the position of the rig in the previous frame.
* @param[in] predictedRig2World A pointer to the initial estimate of the rig's pose. The reconstructor
* refines this initial estimate based on the tracked features.
* @param[in] listCount The number of feature lists provided in the following parameters.
* This must match the number of cameras in the rig.
* @param[in] d_statuses A pointer to the status of the tracked features. Features with an invalid status are
* ignored. Pointers to GPU memory.
* @param[in] d_featureCounts A pointer to the number of tracked features. There is one feature count per
* feature list. Pointers to GPU memory.
* @param[in] d_trackedLocations A pointer to the 2D position of the tracked features in the images.
* One list per camera in the rig. List i has listCount[i] elements. Pointer to GPU memory.
* @param[in] d_worldPoints A pointer to the 3D position of the tracked features in the images.
* One list per camera in the rig. List i has listCount[i] elements. Pointer to GPU memory.
* @param[in] obj The reconstructor object handle.
*
* @return DW_SUCCESS - Operation completed successfully<br>
*         DW_INVALID_HANDLE - If the given context handle is invalid,i.e. null or of wrong type  <br>
*         DW_INVALID_ARGUMENT - One of the given arguments cannot be processed by the function<br>
*
* \ingroup sfm
*/
DW_API_PUBLIC
dwStatus dwReconstructor_estimatePoseAsync(const dwTransformation3f* previousRig2World,
                                           const dwTransformation3f* predictedRig2World,
                                           const uint32_t listCount,
                                           const uint32_t* const d_featureCounts[],
                                           const dwFeature2DStatus* const d_statuses[],
                                           const dwVector2f* const d_trackedLocations[],
                                           const dwVector4f* const d_worldPoints[],
                                           dwReconstructorHandle_t obj);

/**
* Returns the estimated pose from a previous call to dwReconstructor_estimatePoseAsync
*
* @param[out] correctedRig2World The corrected rig to world pose.
* @param[in] obj The reconstructor object handle.
*
* @return DW_SUCCESS - Operation completed successfully<br>
*         DW_INVALID_HANDLE - If the given context handle is invalid,i.e. null or of wrong type  <br>
*         DW_INVALID_ARGUMENT - One of the given arguments cannot be processed by the function<br>
*
* \ingroup sfm
*/
DW_API_PUBLIC
dwStatus dwReconstructor_getEstimatedPose(dwTransformation3f* correctedRig2World,
                                          dwReconstructorHandle_t obj);

/**
* Updates the feature and pose history. Pose history is updated if the vehicle moves or
* rotates enough. Feature history is updated if the tracked location provides new information for
* triangulation.
*
* This method receives the features tracked by each camera in the rig. There must be one feature list
* per camera in the rig. Thus, listCount must be equal to the number of cameras in the rig.
*
* Processes all cameras simultaneously.
*
* @param[out] rig2WorldHistoryIdx A pointer to the index of the rig's pose in the internal pose history list. -1 if
* it was not added to the history. You can pass NULL.
* @param[in] rig2World A pointer to the current position of the rig (vehicle).
* @param[in] listCount Specifies the number of feature lists provided in the following parameters.
* This must match the number of cameras in the rig.
* @param[in] d_featureCounts A pointer to the number of tracked features.
* There is one feature count per feature list. Pointers to GPU memory.
* @param[in] d_trackedLocations A pointer to the 2D position of the tracked features in the images.
* One list per camera in the rig. List i has listCount[i] elements. Pointer to GPU memory.
* @param[in] obj Specifies the reconstructor object handle.
*
* @return DW_SUCCESS - Operation completed successfully<br>
*         DW_INVALID_HANDLE - If the given context handle is invalid,i.e. null or of wrong type<br>
*         DW_INVALID_ARGUMENT - One of the given arguments cannot be processed by the function<br>
* \ingroup sfm
*/
DW_API_PUBLIC
dwStatus dwReconstructor_updateHistory(int32_t* rig2WorldHistoryIdx,
                                       const dwTransformation3f* rig2World,
                                       const uint32_t listCount,
                                       const uint32_t* const d_featureCounts[],
                                       const dwVector2f* const d_trackedLocations[],
                                       dwReconstructorHandle_t obj);

/**
* Triangulates the features of a camera from the internal feature and pose history.
*
* Processes only a single camera.
*
* @param[out] d_worldPoints A pointer to the list of triangulated points. This list must hold at
* least maxFeatureCount items. Pointer must be 16-byte aligned.
* @param[out] d_statuses A pointer to the list of feature statuses. If the reprojection error is too high,
* a feature is marked here as invalid. Otherwise, the status is unchanged.
* @param[in] d_featureCount A pointer to the number of tracked features. Pointer to GPU memory.
* @param[in] cameraIdx Specifies the index of the camera in the rig that this call should triangulate for.
* @param[in] obj Specifies the reconstructor object handle.
*
* @return DW_SUCCESS - Operation completed successfully<br>
*         DW_INVALID_HANDLE - If the given context handle is invalid,i.e. null or of wrong type  <br>
*         DW_INVALID_ARGUMENT - One of the given arguments cannot be processed by the function<br>
* \ingroup sfm
*/
DW_API_PUBLIC
dwStatus dwReconstructor_triangulateFeatures(dwVector4f* d_worldPoints,
                                             dwFeature2DStatus* d_statuses,
                                             const uint32_t* d_featureCount,
                                             const uint32_t cameraIdx,
                                             dwReconstructorHandle_t obj);

/**
* Projects triangulated features back to the image. This can be used to predict locations for the
* feature tracker. Processes all cameras simultaneously.
*
* @param[out] d_locations A GPU pointer to the projection results.
* @param[in] rig2World A pointer to the current position of the rig (vehicle).
* @param[in] d_pointCount A GPU pointer to the number of points in each feature buffer.
* @param[in] d_worldPoints A GPU pointer to the list of triangulated points.
* @param[in] obj Specifies the reconstructor object handle.
*
* @return DW_SUCCESS - Operation completed successfully<br>
*         DW_INVALID_HANDLE - the given context handle is invalid, i.e. null or of wrong type<br>
*         DW_INVALID_ARGUMENT - One of the given arguments cannot be processed by the function<br>
* 
* \ingroup sfm
*/
DW_API_PUBLIC
dwStatus dwReconstructor_project(dwVector2f* d_locations[],
                                 const dwTransformation3f* rig2World,
                                 const uint32_t* const d_pointCount[],
                                 const dwVector4f* const d_worldPoints[],
                                 dwReconstructorHandle_t obj);

/**
* Predicts the positions of features based on the predicted car motion.
* If the feature has been triangulated, it reprojects it using the predicted pose.
* Else, if the feature is below the horizon, it uses the homography induced by the ground plane to
* predict the feature movement. If it is above the horizon and it has not been triangulated, the
* current location is copied as the prediction.
*
* @param[out] d_predictedLocations GPU buffer of dwVector2f. This is where the predicted locations are returned. The buffer must be at least d_featureCount long.
* @param[in] cameraIdx Index of the camera.
* @param[in] previousRigToWorld The position of the rig corresponding to the observations in d_featureLocations.
* @param[in] predictedRigToWorld The predicted position of the rig corresponding to d_predictedLocations.
* @param[in] d_featureCount GPU pointer to the number of features to predict.
* @param[in] d_featureStatuses GPU array. Status of each feature. Non-active features are ignored.
* @param[in] d_featureLocations GPU array of dwVector2f. The last observed feature locations. Must have d_featureCount valid entries.
* @param[in] d_worldPoints GPU array of dwVector4f. The triangulated 3D positions of the features. If a triangulation is not available the 4th component should be zero. Must have d_featureCount valid entries.
* @param[in] obj The reconstructor object handle.
*
* @return DW_SUCCESS - Operation completed successfully<br>
*         DW_INVALID_HANDLE - the given context handle is invalid, i.e. null or of wrong type<br>
*         DW_INVALID_ARGUMENT - One of the given arguments cannot be processed by the function<br>
*
* \ingroup sfm
*/
DW_API_PUBLIC
dwStatus dwReconstructor_predictFeaturePosition(dwVector2f d_predictedLocations[],
                                                uint32_t cameraIdx,
                                                const dwTransformation3f* previousRigToWorld,
                                                const dwTransformation3f* predictedRigToWorld,
                                                const uint32_t* d_featureCount,
                                                const dwFeature2DStatus d_featureStatuses[],
                                                const dwVector2f d_featureLocations[],
                                                const dwVector4f d_worldPoints[],
                                                dwReconstructorHandle_t obj);

/**
* Compacts the internal feature history by keeping only selected features.
*
* @param[in] cameraIdx Index of the camera to compact for.
* @param[in] d_validIndexCount The number of valid indexes in the list. GPU pointer.
* @param[in] d_newToOldMap Mapping array of compacted data to uncompacted sparse data. GPU pointer.
* @param[in] obj The reconstructor object handle.
*
* @return DW_SUCCESS - Operation completed successfully<br>
*         DW_INVALID_HANDLE - the given context handle is invalid, i.e. null or of wrong type<br>
*         DW_INVALID_ARGUMENT - One of the given arguments cannot be processed by the function<br>
*
* \ingroup sfm
**/
// @note `d_newToOldMap` must be got by `dwFeature2DTracker_getNewToOldMap`
// @note This method must be called after `dwFeature2DTracker_compact()` or
// `dwFeature2DTracker_process(DW_FEATURE2D_TRACKER_STAGE_COMPACT_GPU_ASYNC)` to keep the
// feature lists synchronized.
DW_API_PUBLIC
dwStatus dwReconstructor_compactFeatureHistory(const uint32_t cameraIdx,
                                               const uint32_t* d_validIndexCount,
                                               const uint32_t* d_newToOldMap,
                                               dwReconstructorHandle_t obj);

/**
* Compacts the world point array by keeping only selected features.
* Note that this method must be called after `dwFeature2DTracker_compact()` or
* `dwFeature2DTracker_process(DW_FEATURE2D_TRACKER_STAGE_COMPACT_GPU_ASYNC)` to keep the
* feature lists synchronized.
*
* @param[inout] d_worldPoints GPU array of 4-floats.
* @param[in] d_validIndexCount The number of valid indexes in the list. GPU pointer.
* @param[in] d_newToOldMap Mapping array of compacted data to uncompacted sparse data. GPU pointer.
* @param[in] obj The reconstructor object handle.
*
* @return DW_SUCCESS - Operation completed successfully<br>
*         DW_INVALID_HANDLE - the given context handle is invalid, i.e. null or of wrong type<br>
*         DW_INVALID_ARGUMENT - One of the given arguments cannot be processed by the function<br>
*
* \ingroup sfm
**/
DW_API_PUBLIC
dwStatus dwReconstructor_compactWorldPoints(dwVector4f* d_worldPoints,
                                            const uint32_t* d_validIndexCount,
                                            const uint32_t* d_newToOldMap,
                                            dwReconstructorHandle_t obj);

/**
 * Sets the CUDA stream for CUDA related operations.
 *
 *
 * @param[in] stream The CUDA stream to be used. Default is the one passed during dwReconstructor_initialize.
 * @param[in] obj A handle to the reconstructor module to set CUDA stream for.
 *
 * @return DW_SUCCESS - Operation completed successfully<br>
 *         DW_INVALID_HANDLE - the given context handle is invalid, i.e. null or of wrong type<br>
 *         DW_INVALID_ARGUMENT - One of the given arguments cannot be processed by the function<br>
 */
// @note The ownership of the stream remains by the callee.
DW_API_PUBLIC
dwStatus dwReconstructor_setCUDAStream(cudaStream_t stream, dwReconstructorHandle_t obj);

/**
 * Gets CUDA stream used by the reconstructor.
 *
 * @param[out] stream The CUDA stream currently used
 * @param[in] obj A handle to the feature list module.
 *
 * @return DW_SUCCESS - Operation completed successfully<br>
 *         DW_INVALID_HANDLE - the given context handle is invalid, i.e. null or of wrong type<br>
 *         DW_INVALID_ARGUMENT - One of the given arguments cannot be processed by the function<br>
 */
DW_API_PUBLIC
dwStatus dwReconstructor_getCUDAStream(cudaStream_t* stream, dwReconstructorHandle_t obj);

#ifdef __cplusplus
}
#endif

/** @} */
/** @} */
#endif // DW_IMAGEPROCESSING_SFM_SFM_H_

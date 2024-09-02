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

#ifndef DW_IMAGEPROCESSING_TRACKING_BOXTRACKER2D_BOXTRACKER2D_H_
#define DW_IMAGEPROCESSING_TRACKING_BOXTRACKER2D_BOXTRACKER2D_H_

// C api
#include <dw/core/base/Config.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>
#include <dw/imageprocessing/features/FeatureList.h>

/**
 * @file
 * <b>NVIDIA DriveWorks API: 2D Box Tracker</b>
 *
 * @b Description: This file defines methods for tracking 2D boxes.
 */

/**
 * @defgroup boxtracker_group BoxTracker
 *
 * @brief Defines the 2D box tracker API.
 *
 * @{
 * @ingroup tracker2d_group
 */

#ifdef __cplusplus
extern "C" {
#endif

/////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Handle to a 2D object tracker.
 */
typedef struct dwBoxTracker2DObject* dwBoxTracker2DHandle_t;

/**
 * @brief Holds a tracked 2D bounding box.
 * The structure includes the bounding box location, size, box ID, tracked features,
 * and confidence.
 */
typedef struct
{
    //! Bounding box location and size.
    dwBox2D box;
    //! Bounding box ID.
    int32_t id;
    //! Confidence in tracking results.
    float32_t confidence;
    //! Total number of tracked frames.
    int32_t trackedFrameCount;
    //! 2d feature coordinates inside the bounding box.
    const float32_t* featureLocations;
    //! Total number of tracked 2D features.
    size_t nFeatures;
} dwTrackedBox2D;

/**
 * @brief Holds 2D object-tracker parameters.
 */
typedef struct
{
    //! Maximum 2D boxes to track.
    uint32_t maxBoxCount;
    //! Maximum features to track for each 2D bounding box.
    uint32_t maxFeatureCountPerBox;
    /**
     * Maximum box scale in the image to track. It multiplies
     * with image width and height to get the maximum box size.
     */
    float64_t maxBoxImageScale;
    /**
     * Minimum box scale in the image to track. It multiplies
     * with image width and height to get the minimum box size.
     */
    float64_t minBoxImageScale;
    /**
     * The threshold to define the location and size similarity of the bounding boxes.
     * It does not perform clustering when similarity = 0. All the boxes are in one cluster
     * when similarity = +inf.
     */
    float64_t similarityThreshold;
    /**
     * Minimum possible number of boxes minus 1. groupThreshold = 0 means every input box is
     * considered as a valid initialization. Setting a larger value is more robust to outliers.
     * It merges close-by input boxes and uses the average instead.
     */
    uint32_t groupThreshold;
    /**
     * Maximum distance around the closest tracked box to search for a candidate matching box.  Distance here is
     * defined as delta 1 - IOU.  Within this margin, the box with the longest track history is preferred
     * and is selected as the candidate matching box.  The candidate still has to pass the minMatchOverlap
     * test to be considered a positive match for the new box.
     */
    float32_t maxMatchDistance;
    /**
     * Minimum amount of overlap between a tracked box and an added box such that the 2 boxes
     * can be considered the same object.  Overlap is defined as the intersection over the union (IOU).
     */
    float32_t minMatchOverlap;
    /**
     * Rate at which to combine confidence values of new boxes to existing tracked boxes when
     * a new box is found to match an existing box, i.e., conf = conf + conf_new * rate.
     * This also applies to the initial confidence value for a new box, i.e., conf = conf_new * rate.
     */
    float32_t confRateDetect;
    /**
     * Rate at which  confidence values of tracked boxes changes from frame to frame,
     * i.e., conf = conf - rate.
     */
    float32_t confRateTrack;
    /**
     * Threshold on confidence below which tracker no longer report box location.  Box continues to be
     * tracked until confidence falls below discard threshold.
     */
    float32_t confThreshConfirm;
    /**
     * Threshold on confidence below which tracker no longer track box location.
     */
    float32_t confThreshDiscard;
} dwBoxTracker2DParams;

/////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Initializes 2D tracker parameters with default values.
 *
 * @param[out] parameters 2D tracker parameters.
 *
 * @return DW_INVALID_ARGUMENT if parameters are NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwBoxTracker2D_initParams(dwBoxTracker2DParams* parameters);

/////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Initializes 2D bounding box tracker.
 *
 * @param[out] obj Handle to 2D bounding box tracker.
 * @param[in] parameters The 2D tracker parameters.
 * @param[in] imageWidth The width of the image.
 * @param[in] imageHeight The height of the image.
 * @param[in] context Handle to the context under which it is created.
 *
 * @return DW_INVALID_ARGUMENT if box tracker2D handle or parameters are NULL or
 *         imageWidht or imageHeight are equal or less than zero.<br>
 *         DW_SUCCESS otherwise.<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwBoxTracker2D_initialize(dwBoxTracker2DHandle_t* obj, const dwBoxTracker2DParams* parameters,
                                   int32_t imageWidth, int32_t imageHeight, dwContextHandle_t context);

/**
 * @brief Resets the 2D bounding box tracker.
 *
 * @param[in] obj Handle to reset.
 *
 * @return DW_INVALID_ARGUMENT if box tracker2D handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwBoxTracker2D_reset(dwBoxTracker2DHandle_t obj);

/**
 * @brief Performs a shallow reset on the 2D bounding box tracker.
 * This function does not initialize all variables. It just sets the
 * count to zero.
 *
 * @param[in] obj Handle to reset.
 *
 * @return DW_INVALID_ARGUMENT if box tracker2D handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwBoxTracker2D_shallowReset(dwBoxTracker2DHandle_t obj);

/**
 * @brief Releases the 2D bounding box tracker.
 *
 * @param[in] obj Handle to release.
 *
 * @return DW_INVALID_HANDLE if box tracker2D handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwBoxTracker2D_release(dwBoxTracker2DHandle_t obj);

/**
 * @brief Adds bounding boxes to the tracker. The tracker first performs clustering to the group
 * close-by redundant bounding boxes. In each cluster, the average box is computed and added
 * to the tracker.
 *
 * @param[in] boxes The list of 2D bounding boxes to added.
 * @param[in] num Number of boxes passed in the given list.
 * @param[in] obj Handle to the 2D bounding box tracker.
 *
 * @return DW_INVALID_ARGUMENT if box tracker2D handle is NULL or num is greater than 0 and boxes is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwBoxTracker2D_add(const dwBox2D* boxes, size_t num, dwBoxTracker2DHandle_t obj);

/**
 * @brief Adds pre-clustered bounding boxes to the tracker. The tracker assumes input boxes are pre-clustered
 * and all boxes are added to the tracker.  If the input box matches an internal tracked box, the ID of the
 * internal tracked box is preferred.
 *
 * @param[in] boxes The list of pre-clustered boxes.
 * @param[in] num Number of boxes passed in the given list.
 * @param[in] obj Handle to the 2D bounding box tracker.
 *
 * @return DW_INVALID_ARGUMENT if box tracker2D handle is NULL or num is greater than 0 and boxes is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwBoxTracker2D_addPreClustered(const dwTrackedBox2D* boxes, size_t num, dwBoxTracker2DHandle_t obj);

/**
 * @brief Tracks the bounding boxes.
 *
 * @param[in] curFeatureLocations The feature locations computed in the current image frame.
 *                                It has 2*featureCount number of items.
 * @param[in] curFeatureStatuses The feature statues of the current image frame.
 *                               It has featureCount number of items.
 * @param[in] preFeatureLocations The feature locations computed in the previous image frame.
 *                                It has 2*featureCount number of items.
 * @param[in] obj Handle to the 2D bounding box tracker.
 *
 * @return DW_INVALID_ARGUMENT if any parameter is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 *
 * @note Ensure the @a curFeatureLocations argument has the same size as @a preFeatureLocations.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwBoxTracker2D_track(const float32_t* curFeatureLocations,
                              const dwFeature2DStatus* curFeatureStatuses,
                              const float32_t* preFeatureLocations, dwBoxTracker2DHandle_t obj);

/**
 * @brief Updates the feature locations of the 2D bounding boxes.
 * This function call must occur between
 * dwBoxTracker2D_track() and dwBoxTracker2D_get().
 *
 * @param[in] featureLocations The feature locations with 2*nFeatures items.
 * @param[in] statuses The feature statuses with nFeatures items.
 * @param[in] nFeatures The number of 2D features, 0 < nFeatures < 8000.
 * @param[in] obj Handle to the 2D bounding box tracker.
 *
 * @return DW_INVALID_ARGUMENT if box the tracker2D handle, @a featureLocations, or
 *         statuses are NULL; or if @a nFeatures is greater than 8000.<br>
 *         DW_SUCCESS otherwise.<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwBoxTracker2D_updateFeatures(const float32_t* featureLocations,
                                       const dwFeature2DStatus* statuses,
                                       size_t nFeatures, dwBoxTracker2DHandle_t obj);

/**
 * @brief Gets tracked bounding boxes and IDs.
 *
 * @param[out] boxList The list of tracked bounding boxes.
 * @param[out] num Number of bounding box returned in boxList.
 * @param[in] obj Handle to the 2D bounding box tracker.
 *
 * @return DW_INVALID_ARGUMENT if any parameter is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 *
 * @note `boxList` is pointing to an internally allocated memory.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwBoxTracker2D_get(const dwTrackedBox2D** boxList, size_t* num, dwBoxTracker2DHandle_t obj);

/**
 * Enables priority tracking of a boundary box.
 * The priority of the bounding boxes can be used to control the association between 2D features and
 * bounding boxes in case of overlap. With priority tracking enabled, the box with the
 * higher priority gets the feature.
 *
 * @param[in] enable Enables (true)/disable(false) priority tracking.
 * @param[in] obj Handle to the 2D bounding box tracker.
 *
 * @return DW_INVALID_ARGUMENT if box tracker2D handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwBoxTracker2D_enablePriorityTracking(bool enable, dwBoxTracker2DHandle_t obj);

/**
 * Sets the priority of a bounding box.
 *
 * @param[in] idx Index of the object.
 * @param[in] priority Tracking priority (smaller value == higher priority).
 * @param[in] obj Handle to the 2D bounding box tracker.
 *
 * @return DW_INVALID_ARGUMENT if box tracker2D handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwBoxTracker2D_setTrackingPriority(uint32_t idx, float32_t priority, dwBoxTracker2DHandle_t obj);

/**
 * Returns the priority of a bounding box.
 *
 * @param[out] priority Tracking priority (smaller value == higher priority).
 * @param[in] idx Index of the object.
 * @param[in] obj Handle to the 2D bounding box tracker.
 *
 * @return DW_INVALID_ARGUMENT if box tracker2D handle or priority is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwBoxTracker2D_getTrackingPriority(float32_t* priority, uint32_t idx, dwBoxTracker2DHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_IMAGEPROCESSING_TRACKING_BOXTRACKER2D_BOXTRACKER2D_H_

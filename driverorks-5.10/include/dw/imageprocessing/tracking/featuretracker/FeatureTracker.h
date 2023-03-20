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
 * <b>NVIDIA DriveWorks API: 2D Tracker</b>
 *
 * @b Description: This file defines 2D tracking methods.
 */

/**
 * @defgroup feature2dtracker_group Feature 2D Tracker Interface
 *
 * @brief Defines 2D-based feature detection and tracking.
 *
 * @{
 */

#ifndef DW_IMAGEPROCESSING_TRACKING_FEATURETRACKER_FEATURETRACKER_H_
#define DW_IMAGEPROCESSING_TRACKING_FEATURETRACKER_FEATURETRACKER_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>
#include <dw/image/Image.h>
#include <dw/imageprocessing/featuredetector/FeatureDetector.h>
#include <dw/core/system/PVA.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum tracking window size */
#define DW_FEATURE2D_TRACKER_MAX_WINDOW_SIZE 16

/** Handle representing a feature tracker. */
typedef struct dwFeature2DTrackerObject* dwFeature2DTrackerHandle_t;

/** Handle representing a const feature tracker. */
typedef struct dwFeature2DTrackerObject const* dwConstFeature2DTrackerHandle_t;

/**
 * Defines different KLT tracking algorithms
 */
typedef enum dwFeature2DTrackerAlgorithm {
    /** 3-DOF (dx, dy, dscale) standard KLT tracking. */
    DW_FEATURE2D_TRACKER_ALGORITHM_STD = 0,

    /** 3-DOF (dx, dy, dscale) extended KLT tracking. */
    DW_FEATURE2D_TRACKER_ALGORITHM_EX,

    /** 3-DOF (dx, dy, dscale) fast extended KLT tracking. */
    DW_FEATURE2D_TRACKER_ALGORITHM_EX_FAST,

    DW_FEATURE2D_TRACKER_ALGORITHM_COUNT
} dwFeature2DTrackerAlgorithm;

/**
 * Holds configuration parameters for a feature tracker.
 */
typedef struct dwFeature2DTrackerConfig
{
    /** Tracking Algorithm defined by `dwFeature2DTrackerAlgorithm`. */
    dwFeature2DTrackerAlgorithm algorithm;

    /** Type of detector that connects to the tracker, should be the same value as
     *  dwFeature2DDetectorConfig::type during detector initialization */
    dwFeature2DDetectorType detectorType;

    /** Upper bound on number of features handled. */
    uint32_t maxFeatureCount;

    /** Upper bound of history in feature history array. */
    uint32_t historyCapacity;

    /** Levels of pyramid to track
     *  Valid only when algorithm = DW_FEATURE2D_TRACKER_ALGORITHM_EX or 
     *  DW_FEATURE2D_TRACKER_ALGORITHM_EX_FAST
     */
    uint32_t pyramidLevelCount;

    /** Width of the images that the tracker runs on. */
    uint32_t imageWidth;

    /** Height of the images that the tracker runs on. */
    uint32_t imageHeight;

    /** Window size used in the KLT tracker. Supported sizes are
     *   ** DW_FEATURE2D_TRACKER_ALGORITHM_STD: 6, 8, 10, 12, 14.
     *   ** DW_FEATURE2D_TRACKER_ALGORITHM_EX: 10, 12.
     *   ** DW_FEATURE2D_TRACKER_ALGORITHM_EX_FAST: 10, 12.
     *
     *  If set to zero the default value will be used.
     *  Larger window size provides better tracking results but costs more time.
     */
    uint32_t windowSizeLK;

    /** Enable adaptive window size
     *  If enabled, the tracker will use windowSizeLK to track only at the first and the last
     *  pyramid levels, and use smaller window size at the rest pyramid levels.
     *
     *  Valid only when algorithm = DW_FEATURE2D_TRACKER_ALGORITHM_EX or 
     *  DW_FEATURE2D_TRACKER_ALGORITHM_EX_FAST
     */
    bool enableAdaptiveWindowSizeLK;

    /** Upper bound on number of 2-DOF translation-only KLT iterations per level.
     *  If set to zero the default value will be used.
     *  More iterations helps improve tracking results but cost more time
     */
    uint32_t numIterTranslationOnly;

    /** Upper bound on number of 3-DOF translation+scaling KLT iterations per level.
     *  Valid only when algorithm = DW_FEATURE2D_TRACKER_ALGORITHM_EX or 
     *  DW_FEATURE2D_TRACKER_ALGORITHM_EX_FAST
     */
    uint32_t numIterScaling;

    /** Number of levels in pyramid that will use translation-only KLT tracking,
     *  level [maxPyramidLevel - 1, maxPyramidLevel - numTranslationOnlyLevel] will use translation-only track
     *  level [maxPyramidLevel - numTranslationOnlyLevel - 1, 0] will use translation-scaling track
     *  if numTranslationOnlyLevel = 0, translation-scaling track will be applied to all levels in pyramid
     *
     *  Valid only when algorithm = DW_FEATURE2D_TRACKER_ALGORITHM_EX or 
     *  DW_FEATURE2D_TRACKER_ALGORITHM_EX_FAST
     */
    uint32_t numLevelTranslationOnly;

    /**
     * Set it to true to use half float as intermediate results during tracking
     * It saves register usage and will be faster than full 32-bit float,
     * but will lose a bit of precision
     *
     * Valid only when algorithm = DW_FEATURE2D_TRACKER_ALGORITHM_EX or 
     *  DW_FEATURE2D_TRACKER_ALGORITHM_EX_FAST
     */
    bool useHalf;

    /**
    * Enable sparse output if set to 1.
    * If enabled, the output feature array will contain invalid features, which
    * can be removed by explicitly calling `dwFeature2DTracker_compact`
    */
    uint32_t enableSparseOutput;

    /** The maximum allowed scale change for the tracked points across consecutive frames.
     *  Valid only when algorithm = DW_FEATURE2D_TRACKER_ALGORITHM_EX or 
     *  DW_FEATURE2D_TRACKER_ALGORITHM_EX_FAST
     */
    float32_t maxScaleChange;

    /** Features will be killed if the Cross Correlation Score is less than this threshold during tracking.
     *  The value should be between -1.0 and 1.0.
     *  Valid only when algorithm = DW_FEATURE2D_TRACKER_ALGORITHM_EX or 
     *  DW_FEATURE2D_TRACKER_ALGORITHM_EX_FAST
     */
    float32_t nccKillThreshold;

    /** Feature template will be updated if the Cross Correlation Score is less than this threshold during tracking.
     *  The value should be between -1.0 and 1.0.
     *  The value should be no less than nccKillThreshold.
     *  Valid only when algorithm = DW_FEATURE2D_TRACKER_ALGORITHM_EX
    */
    float32_t nccUpdateThreshold;

    /**
     * Features will be killed if the motion is larger than the template size times the large motion killing threshold during tracking.
     * Valid only when algorithm = DW_FEATURE2D_TRACKER_ALGORITHM_EX or 
     *  DW_FEATURE2D_TRACKER_ALGORITHM_EX_FAST
     */
    float32_t largeMotionKillRatio;

    /**
     * If difference of translation prediction between 2 adjacent KLT iteration is less than this value,
     * it is thought the result is converged and will abort tracking iteration.
     * Default value = 0.1F, larger value causes faster convergence, but will lose the precision.
     */
    float32_t displacementThreshold;

    /** Processor type which determines on which processor the algorithm should be executed on.
     * Supported options are: DW_PROCESSOR_TYPE_GPU, DW_PROCESSOR_TYPE_PVA_0, DW_PROCESSOR_TYPE_PVA_1.
     * @note DW_PROCESSOR_TYPE_GPU supports DW_FEATURE2D_TRACKER_ALGORITHM_STD, and DW_FEATURE2D_TRACKER_ALGORITHM_EX
     * @note DW_PROCESSOR_TYPE_PVA_0/1 supports DW_FEATURE2D_TRACKER_ALGORITHM_STD, DW_FEATURE2D_TRACKER_ALGORITHM_EX
     */
    dwProcessorType processorType;

} dwFeature2DTrackerConfig;

/**
 * Initializes dwFeature2DTracker parameters with default values.
 * @param[inout] params dwFeature2DTracker parameters; user can optionally set imageWidth/imageHeight
 *                      before the API call to obtain the default parameters for the resolution
 * @return DW_INVALID_ARGUMENT if params is NULL. <br>
 *         DW_SUCCESS otherwise. <br>
 */
DW_API_PUBLIC
dwStatus dwFeature2DTracker_initDefaultParams(dwFeature2DTrackerConfig* params);

/**
 * Initializes dwFeature2DTracker parameters with values best suited for the given camera using
 * camera extrinsic (dwTransformation3f* cameraToRig) and camera intrinsic (dwConstCameraModelHandle_t cameraHandle)
 *
 * @param[out] params dwFeature2DTracker parameters
 * @param[in] cameraToRig transformed camera extrinsic parameters with respect to rig
 * @param[in] cameraHandle holds camera intrinsic information
 *
 * @return DW_INVALID_ARGUMENT if any parameter is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
// TODO(dwplc): RFD coverity tool claims that first 31 characters of identifiers is same with dwFeature2DTracker_initDefaultParams, which causes ambiguity
// coverity[misra_c_2012_rule_5_1_violation]
DW_API_PUBLIC
dwStatus dwFeature2DTracker_initDefaultParamsForCamera(dwFeature2DTrackerConfig* params,
                                                       const dwTransformation3f* cameraToRig,
                                                       dwConstCameraModelHandle_t cameraHandle);

/**
 * Creates and initializes a feature tracker.
 *
 * @param[out] obj A pointer to the feature tracker handle is returned here.
 * @param[in] config the configuration parameters for tracker.
 * @param[in] cudaStream the CUDA stream to use for tracker operations.
 * @param[in] context the handle to the context under which it is created.
 *
 * @return DW_INVALID_ARGUMENT if feature tracker handle, featureHistoryArray or context are NULL.<br>
 *                             if config is invalid. <br>
 *                             if featureHistoryArray is not allocated on GPU
 *                             if featureHistoryArray contains invalid pointers
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeature2DTracker_initialize(dwFeature2DTrackerHandle_t* obj,
                                       const dwFeature2DTrackerConfig* config,
                                       cudaStream_t cudaStream, dwContextHandle_t context);

/**
 * Resets a feature tracker.
 *
 * @param[in] obj Specifies the feature tracker handle to be reset.
 *
 * @return DW_INVALID_ARGUMENT if feature tracker handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeature2DTracker_reset(dwFeature2DTrackerHandle_t obj);

/**
 * Releases the feature tracker.
 * This method releases all resources associated with a feature tracker.
 *
 * @note This method renders the handle unusable.
 *
 * @param[in] obj The object handle to be released.
 *
 * @return DW_INVALID_ARGUMENT if feature tracker handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 *
 */
DW_API_PUBLIC
dwStatus dwFeature2DTracker_release(dwFeature2DTrackerHandle_t obj);

/**
 * Tracks features and store the tracked results to `predictedFeatures` between the previous and current images.
 * Features to track are defined by `dwFeatureHistoryArray` bound to the tracker during initialization
 *
 * @param[out] featureHistoryArray output feature history array, must be on GPU memory.
 * @param[out] predictedFeatures list of predicted features, it's also the top slice of `dwFeatureHistoryArray`
 * @param[out] d_normalizedCrossCorrelation Device pointer to nccScore of tracked features, will be ignored if it's NULL.
 * @param[in] featuresToTrack list of features to be tracked, usually the output of feature detector.
 * @param[in] d_predictedPositions A GPU pointer to a list of expected positions of the features to be tracked.
 *            The indexes of this list must match the indexes of the internal feature list.
 *            If this is NULL(== 0), then use the locations in `featuresToTrack`
 * @param[in] previousPyramid pyramid constructed from the last image.
 * @param[in] currentPyramid pyramid constructed from the current image.
 * @param[in] obj Specifies the feature tracker handle.
 *
 * @return DW_INVALID_ARGUMENT if any parameter except d_normalizedCrossCorrelation and d_predictedPositions is NULL,
 *                             or previous and current pyramids have a different number of levels.<br>
 *                             if previousFeatures or detectedFeatures are not on GPU. <br>
 *         A specific CUDA error in case of an underlying cuda failure.<br>
 *         DW_SUCCESS otherwise.<br>
 *
 * @note Output `predictedFeatures` array contains a list of compacted features, invalid trackings are removed automatically,
 * the orders between input `featuresToTrack` and output `predictedFeatures` can be queried by
 * `dwFeatureHistoryArray::newToOldMap` which maps new->old index: newToOldMap[newIdx] == oldIdx.
 * newToOldMap[i] = j means the i-th feature in output `predictedFeatures` is the j-th item in input `featuresToTrack` array.
 *       i.e. predictedFeatures[i] = featuresToTrack[newToOldMap[i]], below is an example:
 * data:              [1 2 3 4 5 6 7 8 9]
 * tracked/untracked: [O O X X O X O X O]
 * compacted output (overwrite 3 and 4 by 9 and 7):
 * data:              [1 2 9 7 5]
 * newToOldMap:       [0 1 8 6 4]
 */
DW_API_PUBLIC
dwStatus dwFeature2DTracker_trackFeatures(dwFeatureHistoryArray* featureHistoryArray,
                                          dwFeatureArray* predictedFeatures,
                                          float32_t* d_normalizedCrossCorrelation,
                                          const dwFeatureArray* featuresToTrack,
                                          const dwVector2f* d_predictedPositions,
                                          const dwPyramidImage* previousPyramid,
                                          const dwPyramidImage* currentPyramid,
                                          dwFeature2DTrackerHandle_t obj);

/**
 * Remove invalid features.
 * @param[inout] featureHistoryArray Sparse feature history array, the features whose status = DW_FEATURE2D_STATUS_INVALID
 * will be removed after calling
 * @param[in] obj Specifies the feature tracker handle.
 *
 * @return DW_INVALID_ARGUMENT if featureHistoryArray or obj is NULL. <br>
 *         A specific CUDA error in case of an underlying cuda failure.<br>
 *         DW_SUCCESS otherwise.<br>
 *
 * @note this API does real work only when `dwFeature2DTrackerConfig::enableSparseOutput = 1`,
 * It will also update the `dwFeatureHistoryArray::newToOldMap`.
 */
DW_API_PUBLIC
dwStatus dwFeature2DTracker_compact(dwFeatureHistoryArray* featureHistoryArray,
                                    dwFeature2DTrackerHandle_t obj);

/**
 * Sets the CUDA stream for CUDA related operations.
 *
 * @note The ownership of the stream remains by the callee.
 *
 * @param[in] stream The CUDA stream to be used. Default is the one passed during dwFeature2DTracker_initialize.
 * @param[in] obj A handle to the feature tracker module to set CUDA stream for.
 *
 * @return DW_INVALID_ARGUMENT if feature tracker handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeature2DTracker_setCUDAStream(cudaStream_t stream, dwFeature2DTrackerHandle_t obj);

/**
 * Gets the CUDA stream used by the feature tracker.
 *
 * @param[out] stream The CUDA stream currently used.
 * @param[in] obj A handle to the feature tracker module.
 *
 * @return DW_INVALID_ARGUMENT if feature tracker handle or stream are NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeature2DTracker_getCUDAStream(cudaStream_t* stream, dwFeature2DTrackerHandle_t obj);

/**
 * Sets the cuPVA stream for PVA related operations.
 *
 * @note The ownership of the stream remains by the callee.
 *
 * @param[in] stream The cuPVA stream to be used.
 * @param[in] obj A handle to the feature tracker module to set cuPVA stream for.
 *
 * @return DW_NOT_AVAILABLE if PVA is not available on the platform.<br>
           DW_INVALID_ARGUMENT if feature tracker handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeature2DTracker_setPVAStream(cupvaStream_t stream, dwFeature2DTrackerHandle_t obj);

/**
 * Gets the cuPVA stream used by the feature tracker.
 *
 * @param[out] stream The cuPVA stream currently used.
 * @param[in] obj A handle to the feature tracker module.
 *
 * @return DW_NOT_AVAILABLE if PVA is not available on the platform.<br>
           DW_INVALID_ARGUMENT if feature tracker handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeature2DTracker_getPVAStream(cupvaStream_t* stream, dwFeature2DTrackerHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_IMAGEPROCESSING_TRACKING_FEATURETRACKER_FEATURETRACKER_H_

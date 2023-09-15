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
// SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: TemplateTracker Methods</b> *
 * @b Description: This file defines scaling-extend KLT tracking methods.
 */

/**
 * @defgroup template_tracker_group Template Tracker Interface
 *
 * @brief Defines 2D-based template tracking.
 *
 * @{
 */

#ifndef DW_IMAGEPROCESSING_TRACKING_TEMPLATETRACKER_TEMPLATETRACKER_H_
#define DW_IMAGEPROCESSING_TRACKING_TEMPLATETRACKER_TEMPLATETRACKER_H_

// C api
#include <dw/core/base/Config.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>
#include <dw/image/Image.h>
#include <dw/imageprocessing/features/FeatureList.h>
#include <dw/imageprocessing/pyramid/Pyramid.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * *   Template <br>
 *     It's a subimage defined by ROI (regions of interest) that is going to be
 *     tracked in the new frame.<br>
 *
 * *   Tracker workflow<br> *
 *     1. Add new templates to the list. This module does not provide detector. App decides what new templates to add.
 *     2. Do tracking.
 *
 * *   Template updating strategy<br>
 *     The template is not updated every frame. When KLT iteration is finished and the
 * tracker gets the new predicted template in the new frame, it will calculate the
 * ZNCC (zero mean normalized cross-correlation) between the warped template and
 * the predicted one, only when ZNCC is less than a predefined updateThreshold,
 * the stored template and its location/size will be updated, elsewise the tracker will
 * use the old template and its size as an estimation and moves its location to
 * the new one. There is another killing threshold (smaller than updateThreshold), if
 * ZNCC is even less than killing threshold, it will be considered that the tracked result
 * differs so much to the stored template that the tracking is failed and the current template
 * entry will be killed.
 *
 * Template tracking is usually applied on video sequence, so the scale between frame N-1 and N
 * should be smooth. maxScaleChange is a parameter to kill significant size changes. Once the
 * ratio of old and new template size is greater than it, the current template entry will be killed.
 * <pre><code>
 * Pseudo code:
 *
 * if (ZNCC < killThreshold)
 *     tracking is lost, kill the current template
 *     return
 *
 * if (N-1 to N scaler is not within [1/maxScaleChang, maxScaleChang])
 *     Too significant size change, kill the current template
 *     return
 *
 * if (ZNCC < updateThreshold)
 *     template image <-- new image
 *     template location <-- new location
 *     template size <-- new size
 * else
 *     if ((templateCenter - newCenter) > 1px)
 *         template location <-- new location
 *     else
 *         template image <-- new image
 *         template location <-- new location
 *         template size <-- new size
 *
 * </code></pre>
 */

/** Handle representing a TemplateTracker tracker */
typedef struct dwTemplateTrackerObject* dwTemplateTrackerHandle_t;

/** Handle representing a const TemplateTracker tracker */
typedef struct dwTemplateTrackerObject const* dwConstTemplateTrackerHandle_t;

/** Different versions of the template tracker
  * Both can be found in paper: "Lucas-Kanade 20 Years On: A Unifying Framework" by
  * SIMON BAKER AND IAIN MATTHEWS */
typedef enum dwTemplateTrackerAlgorithm {
    DW_TEMPLATE_TRACKER_ALGORITHM_IA = 0, //!< inverse additive KLT
    DW_TEMPLATE_TRACKER_ALGORITHM_IC      //!< inverse compositional KLT.
} dwTemplateTrackerAlgorithm;

/** Configuration parameters for a dwTemplateTrackerIA */
typedef struct dwTemplateTrackerParameters
{
    /** Tracking Algorithm defined by `dwTemplateTrackerAlgorithm`. */
    dwTemplateTrackerAlgorithm algorithm;

    /** Upper bound on number of templates handled. */
    uint32_t maxTemplateCount;

    /** Max pyramid level to track. */
    uint32_t maxPyramidLevel;

    /** Maximum size of templates to track, if ROI > maxTemplateSize * maxTemplateSize,
     *  template tracker will clamp the center maxTemplateSize * maxTemplateSize region to do tracking */
    uint32_t maxTemplateSize;

    /** Minimum size of templates to track, if ROI < minTemplateSize * minTemplateSize,
     *  template tracker will extend the contour to include minTemplateSize * minTemplateSize region to do tracking */
    uint32_t minTemplateSize;

    /** Width of the images that the tracker runs on. */
    uint32_t imageWidth;

    /** Height of the images that the tracker runs on. */
    uint32_t imageHeight;

    /** Iteration number to apply the KLT tracker. */
    uint32_t numIterationsFine;

    /** for DW_TEMPLATE_TRACKER_ALGO_IA only
     * Iteration number to apply the coarse KLT for robustness.
     * Coarse tracking converges with smaller iteration number,
     * but is not as accurate as fine tracking.
     * If setting as 0, the algorithm will skip the coarse pass, use fine only iteration */
    uint32_t numIterationsCoarse;

    /** Updating threshold in [0, 1]. If ZNCC between 2 frames is less than this value,
     * tracking template will be updated. Otherwise only updates its location but keep
     * the original size and image data unchanged. Larger value means updating the template more frequently.
     * If thresholdUpdate = 1, template will be updated every frame
     * If thresholdUpdate = 0, template won't be updated unless the size of stored template and new one
     * differs too much. */
    float32_t thresholdUpdate;

    /** Killing threshold in [0, 1]. If ZNCC between 2 frames is less than this value,
     * tracking will be killed. Smaller value means less possible to be killed.
     * If thresholdKill = 0 or -1, the tracking will not be marked as failure until it's out of time
     * or move out of boundary. If thresholdKill = 1, the tracking will always be killed. */
    float32_t thresholdKill;

    /** for DW_TEMPLATE_TRACKER_ALGO_IC only
     * Stop threshold in [-1, 1]. If ZNCC between 2 frames is larger than this value,
     * tracking will have early stop even before it reaches the maximal number of
     * allowed iterations. Larger value means less possible to be considered as converged track.
     * If nccThresholdStop = 1, the tracking will always run the maximal number of allowed iterations.
     * If nccThresholdStop = -1, the tracking will run one iteration. */
    float32_t thresholdStop;

    /** If scalingFactor between frame N to N-1 is outside range [1/maxScaleChange, maxScaleChange]
     * tracking will be killed */
    float32_t maxScaleChange;

    /** Maximum valid template width, any templates with bbox.width > validWidth will
     * be killed after tracking.
     * validWidth = -1.F means there's no width limitation */
    float32_t validWidth;

    /** Maximum valid template height, any templates with bbox.height > validHeight will
     * be killed after tracking.
     * validHeight = -1.F means there's no height limitation */
    float32_t validHeight;

    /** Processor type which determines on which processor the algorithm should be executed on.
     * Supported options are: DW_PROCESSOR_TYPE_GPU, DW_PROCESSOR_TYPE_PVA_0, DW_PROCESSOR_TYPE_PVA_1.
     * @note DW_PROCESSOR_TYPE_GPU supports both DW_TEMPLATE_TRACKER_ALGORITHM_IA and DW_TEMPLATE_TRACKER_ALGORITHM_IC
     * @note DW_PROCESSOR_TYPE_PVA_0/1 supports only DW_TEMPLATE_TRACKER_ALGORITHM_IC
     */
    dwProcessorType processorType;
} dwTemplateTrackerParameters;

typedef struct dwTemplateArray
{
    dwFeature2DStatus* statuses; /**< Status of each template. 1D array of size maxTemplates.  */
    dwRectf* bboxes;             /**< bounding box of each template. 1D array of size maxTemplates. */
    uint32_t* ids;               /**< Id of each template. 1D array of size maxTemplates */
    uint32_t* ages;              /**< Age of each template. 1D array of size maxTemplates. */
    float32_t* scaleFactors;     /**< scaleFactor from frame N to N-1 of each template. 1D array of size maxTemplates. */
    uint32_t* newToOldMap;       /**< New to old index map, 1D array of size maxTemplates. See more details in `dwTemplateTracker_trackImage`. and `dwTemplateTracker_trackPyramid` */

    uint32_t* templateCount; /**< Total number of templates. Single value. */
    uint32_t maxTemplates;   /**< Max number of templates in template array. */

    uint8_t* data; /**< Pointer to the raw data address*/
    size_t bytes;  /**< Bytes of raw data*/

    dwMemoryType memoryType; /**< Whether the template array is located on CPU or GPU. */

} dwTemplateArray;

///////////////////////////////////////////////////////////////////////
/**
 * Creates and initializes a template array.
 *
 * @param[out] templateArray pointer to the dwTemplateArray is returned here.
 * @param[in] maxTemplateCount maximum number of templates that the template array can have.
 * @param[in] memoryType DW_MEMORY_TYPE_CUDA for CUDA array, <br>
 *                       DW_MEMORY_TYPE_CPU for CPU array, <br>
 *                       DW_MEMORY_TYPE_PINNED for pinned memory
 * @param[in] context handle to the context under which it is created.
 *
 * @return DW_INVALID_ARGUMENT if template arry or context are NULL. <br>
 *                             if maxTemplateCount is 0. <br>
 *                             if memoryType is not listed in dwMemoryType. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
DW_DEPRECATED("WARNING: will be removed in the next major release, use dwTemplateArray_createNew instead")
dwStatus dwTemplateArray_create(dwTemplateArray* templateArray,
                                const uint32_t maxTemplateCount,
                                const dwMemoryType memoryType,
                                dwContextHandle_t context);

/**
 * Creates and initializes a template array.
 *
 * @param[out] templateArray pointer to the dwTemplateArray is returned here.
 * @param[in] maxTemplateCount maximum number of templates that the template array can have.
 * @param[in] memoryType DW_MEMORY_TYPE_CUDA for CUDA array, <br>
 *                       DW_MEMORY_TYPE_CPU for CPU array, <br>
 *                       DW_MEMORY_TYPE_PINNED for pinned memory
 * @param[in] stream  Working CUDA stream
 * @param[in] context handle to the context under which it is created.
 *
 * @return DW_INVALID_ARGUMENT if template arry or context are NULL. <br>
 *                             if maxTemplateCount is 0. <br>
 *                             if memoryType is not listed in dwMemoryType. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwTemplateArray_createNew(dwTemplateArray* templateArray,
                                   const uint32_t maxTemplateCount,
                                   const dwMemoryType memoryType,
                                   cudaStream_t stream,
                                   dwContextHandle_t context);

/**
 * Destroys the template array and frees any memory created by dwTemplateArray_createNew().
 *
 * @param[in] templateArray template array to be destroyed.
 *
 * @return DW_INVALID_ARGUMENT if templateArray contains invalid pointers. <br>
 *                             if templateArray.memoryType is not listed in dwMemoryType. <br>
 *         DW_SUCCESS otherwise. <br>
 */
DW_API_PUBLIC
dwStatus dwTemplateArray_destroy(dwTemplateArray templateArray);

/**
 * Resets the template array.
 * Sets the template count back to zero.
 *
 * @param[in] templateArray template array to be reset.
 * @param[in] stream CUDA stream used to reset the template array.
 *
 * @return DW_INVALID_ARGUMENT if templateArray.templateCount is NULL. <br>
 *                             if templateArray.memoryType is not listed in dwMemoryType. <br>
 *         DW_SUCCESS otherwise. <br>
 */
DW_API_PUBLIC
dwStatus dwTemplateArray_reset(dwTemplateArray* templateArray, cudaStream_t stream);

/**
 * Deep copy all contents from `srcTemplateArray` to `dstTemplateArray`
 * @param[out] dstTemplateArray `dwTemplateArray` to copy to
 * @param[in] srcTemplateArray `dwTemplateArray` to copy from
 * @param[in] stream Working cuda stream
 * @return DW_INVALID_ARGUMENT if `dstTemplateArray` or `srcTemplateArray` is NULL. <br>
 *                             if `dstTemplateArray.bytes != srcTemplateArray.bytes`. <br>
 *         A specific CUDA error in case of an underlying cuda failure.<br>
 *         DW_SUCCESS otherwise. <br>
 *
 * @note If the copy function is run on a non-zero stream, it's asynchronous calling, need do
 * stream sync or event sync to make sure the copy is done.
 */
DW_API_PUBLIC
dwStatus dwTemplateArray_copyAsync(dwTemplateArray* dstTemplateArray,
                                   const dwTemplateArray* srcTemplateArray,
                                   cudaStream_t stream);
/**
 * @brief Initializes TemplateTracker parameters with default values.
 * @param[out] params TemplateTracker parameters.
 * @return DW_INVALID_ARGUMENT if params is NULL. <br>
 *         DW_SUCCESS otherwise. <br>
 */
DW_API_PUBLIC
dwStatus dwTemplateTracker_initDefaultParams(dwTemplateTrackerParameters* params);

/**
 * @brief Initialize the TemplateTracker module
 * @param[out] obj A pointer to TemplateTracker handle that is initialized from parameters.
 * @param[in] params TemplateTracker parameters.
 * @param[in] stream Specifies the cuda stream to use
 * @param[in] context Specifies the handle to the context.
 * @return DW_INVALID_ARGUMENT if TemplateTracker handle, params or context are NULL, or
 *         the params has invalid values. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwTemplateTracker_initialize(dwTemplateTrackerHandle_t* obj,
                                      const dwTemplateTrackerParameters* params,
                                      cudaStream_t stream, dwContextHandle_t context);

/**
 * @brief Resets the TemplateTracker.
 * @param[in] obj Handle to reset.
 * @return DW_INVALID_ARGUMENT if TemplateTracker handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwTemplateTracker_reset(dwTemplateTrackerHandle_t obj);

/**
 * @brief Releases the TemplateTracker module.
 * @param[in] obj The object handle to release.
 * @return DW_INVALID_ARGUMENT if TemplateTracker handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 *
 * @note This method renders the handle unusable.
 */
DW_API_PUBLIC
dwStatus dwTemplateTracker_release(dwTemplateTrackerHandle_t obj);

/**
 * @brief Sets the CUDA stream used.
 * @param[in] cudaStream The CUDA stream used.
 * @param[in] obj A pointer to the TemplateTracker handle that is updated.
 * @return DW_INVALID_ARGUMENT if TemplateTracker handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwTemplateTracker_setCUDAStream(cudaStream_t cudaStream,
                                         dwTemplateTrackerHandle_t obj);

/**
 * @brief Gets the CUDA stream used.
 * @param[out] cudaStream The CUDA stream used.
 * @param[in] obj A pointer to the TemplateTracker handle that is updated.
 * @return DW_INVALID_ARGUMENT if TemplateTracker handle or cudaStream are NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwTemplateTracker_getCUDAStream(cudaStream_t* cudaStream,
                                         dwTemplateTrackerHandle_t obj);

/**
 * @brief Track the templates in currentImage <br>.
 * The computation takes place asynchronously on the device (GPU).
 * @param[inout] templateArray template to be tracked, in-place tracking
 * @param[in] currentImage Current image data to track to.
 * @param[in] previousImage Previous image data to track from.
 * @param[in] obj Specifies the TemplateTracker handle.
 * @return DW_INVALID_ARGUMENT if any parameter is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 *
 * @note Output `templateArray` array contains a list of compacted templates,
 * `dwTemplateArray::newToOldMap` which maps new->old index: newToOldMap[newIdx] == oldIdx.
 * newToOldMap[i] = j means the i-th template in prediction is the j-th item in input array.
 *       i.e. predictedTemplate[i] = oldTemplateToTrack[newToOldMap[i]], below is an example:
 * data:              [1 2 3 4 5 6 7 8 9]
 * tracked/untracked: [O O X X O X O X O]
 * compacted output (overwrite 3 and 4 by 9 and 7):
 * data:              [1 2 9 7 5]
 * newToOldMap:       [0 1 8 6 4]
 */
DW_API_PUBLIC
dwStatus dwTemplateTracker_trackImage(dwTemplateArray* templateArray,
                                      const dwImageCUDA* currentImage,
                                      const dwImageCUDA* previousImage,
                                      dwTemplateTrackerHandle_t obj);

/**
 * @brief Track the templates in currentPyramid <br>.
 * The computation takes place asynchronously on the device (GPU).
 * @param[inout] templateArray template to be tracked, in-place tracking
 * @param[in] currentPyramid Current pyramid data to track to.
 * @param[in] previousPyramid Previous pyramid data to track from.
 * @param[in] obj Specifies the TemplateTracker handle.
 * @return DW_INVALID_ARGUMENT if any parameter is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 *
 * @note Output `templateArray` array contains a list of compacted templates,
 * `dwTemplateArray::newToOldMap` which maps new->old index: newToOldMap[newIdx] == oldIdx.
 * newToOldMap[i] = j means the i-th template in prediction is the j-th item in input array.
 *       i.e. predictedTemplate[i] = oldTemplateToTrack[newToOldMap[i]], below is an example:
 * data:              [1 2 3 4 5 6 7 8 9]
 * tracked/untracked: [O O X X O X O X O]
 * compacted output (overwrite 3 and 4 by 9 and 7):
 * data:              [1 2 9 7 5]
 * newToOldMap:       [0 1 8 6 4]
 */
DW_API_PUBLIC
dwStatus dwTemplateTracker_trackPyramid(dwTemplateArray* templateArray,
                                        const dwPyramidImage* currentPyramid,
                                        const dwPyramidImage* previousPyramid,
                                        dwTemplateTrackerHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_IMAGEPROCESSING_TRACKING_TEMPLATETRACKER_TEMPLATETRACKER_H_

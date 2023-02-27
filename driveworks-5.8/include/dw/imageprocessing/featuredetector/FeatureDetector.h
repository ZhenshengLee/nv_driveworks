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
 * <b>NVIDIA DriveWorks API: 2D Detector</b>
 *
 * @b Description: This file defines 2D detection
 */

/**
 * @defgroup featureDetector_group Feature 2D Detector Interface
 *
 * @brief Defines 2D-based feature detection.
 *
 * @{
 */

#ifndef DW_IMAGEPROCESSING_FEATUREDETECTOR_FEATUREDETECTOR_H_
#define DW_IMAGEPROCESSING_FEATUREDETECTOR_FEATUREDETECTOR_H_

#include <dw/core/base/Exports.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>
#include <dw/image/Image.h>
#include <dw/imageprocessing/features/FeatureList.h>
#include <dw/imageprocessing/filtering/Pyramid.h>
#include <dw/calibration/cameramodel/CameraModel.h>
#include <dw/core/system/PVA.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Handle representing a feature detector. */
typedef struct dwFeature2DDetectorObject* dwFeature2DDetectorHandle_t;

/** Handle representing a const feature detector. */
typedef struct dwFeature2DDetectorObject const* dwConstFeature2DDetectorHandle_t;

/**
 * Defines different KLT tracking algorithms
 */
typedef enum dwFeature2DDetectorType {
    /** Standard Harris Corner detector with fixed parameters, quicker */
    DW_FEATURE2D_DETECTOR_TYPE_STD = 0,

    /** Extended Harris Corner detector with more configurable parameters,
     *  more flexible, better quality but slower */
    DW_FEATURE2D_DETECTOR_TYPE_EX,

    /** Fast corner detector, quicker. This algorithm was introduced by Edward Rosten and
     *  Tom Drummond in their paper "Machine learning for high-speed corner detection" in 2006 */
    DW_FEATURE2D_DETECTOR_TYPE_FAST9,

    DW_FEATURE2D_DETECTOR_TYPE_COUNT
} dwFeature2DDetectorType;

/** Feature distribution mask for extended detector */
typedef enum {
    /** output feature in uniform distribution */
    DW_FEATURE2D_SELECTION_MASK_TYPE_UNIFORM = 0,

    /** output feature in 2D gaussian distribution which has more features in
        center area and less ones in boundary */
    DW_FEATURE2D_SELECTION_MASK_TYPE_GAUSSIAN,

    DW_FEATURE2D_SELECTION_MASK_TYPE_COUNT
} dwFeature2DSelectionMaskType;

/**
 * Holds configuration parameters for a feature detector.
 */
typedef struct dwFeature2DDetectorConfig
{
    /** Detecting algorithm defined by `dwFeature2DDetectorType` */
    dwFeature2DDetectorType type;

    /** Width of the images that the Detector runs on.
     *  When PVA fast9 detector is used, the width should be in range of [65, 3264] */
    uint32_t imageWidth;

    /** Height of the images that the Detector runs on.
     *  When PVA fast9 detector is used, the height should be in range of [65, 2448] */
    uint32_t imageHeight;

    /** Upper bound on number of features handled.
     *  When using EX detector, maxFeatureCount must be no less than the amount of cells, i.e.,
     *  maxFeatureCount >= ceil(imageWidth / cellSize) * ceil(imageHeight / cellSize). */
    uint32_t maxFeatureCount;

    /** Weigting K of the harris corner score defined as
     *  det(M) - K * trace(M) * trace(M), where M is the structural matrix
     *  0.04 - 0.06 is used typically
     *  If set to zero the default value (5e-2) will be used */
    float32_t harrisK;

    /** Cell size in pixel to split the image into cells. Must be an integer multiplicaiton of 4 */
    uint32_t cellSize;

    /** for DW_PROCESSOR_TYPE_PVA_0 or DW_PROCESSOR_TYPE_PVA_1 only
     *  Gradient window size. Must be 3, 5 or 7. */
    uint32_t gradientSize;

    /** for DW_PROCESSOR_TYPE_PVA_0 or DW_PROCESSOR_TYPE_PVA_1 only
     *  Block window size used to compute the Harris Corner score. Must be 3, 5 or 7.
     */
    uint32_t blockSize;

    /** Threshold to filter out low latency points. All points whose scores are
     *  less than detectorScoreThreshold will be considered as a non-feature point.
     *  Lower value will output more features, higher value will only keep the
     *  high frequency points.
     *  If set to zero the default value will be used. */
    float32_t scoreThreshold;

    /** for DW_FEATURE2D_DETECTOR_TYPE_STD only
     *  features in the cell that have scores higher than this value will
     *  be considered as high frequency point and will always be outputed.
     *  If set it <= detectorScoreThreshold, all features detected are high
     *  frequency and the next numEvenDistributionPerCell will be invalid */
    float32_t detailThreshold;

    /** for DW_FEATURE2D_DETECTOR_TYPE_STD only
     *  Number of features to be appended after high frequency points, they have
     *  second highest scores but are less than detectorDetailThreshold.
     *  During detection, the input image will be splitted into cellSize x cellSize
     *  small tiles, there'll be at most numEvenDistributionPerCell second-highest-score
     *  features in each cell.
     *
     *  Larger numEvenDistributionPerCell will output an evener distributed feature map
     *  Smaller numEvenDistributionPerCell will have more features in high frequence points
     *  (score >= detectorDetailThreshold) and a less evener distributed feature map.
     *  Invalid if = 0, in that case, only detectorDetailThreshold takes effect. */
    uint32_t numEvenDistributionPerCell;

    /** for DW_FEATURE2D_DETECTOR_TYPE_EX only
     *  STD detector always detects on level 0 pyramid image.
     *
     *  Level at which features are detected
     *  Default to one (half of original image resolution) */
    uint32_t detectionLevel;

    /** for DW_FEATURE2D_DETECTOR_TYPE_EX only
     *  STD detector use fixed harrisRadius = 1.
     *
     *  Radius of detector harris averaging window
     *  If set to zero the default value (1) will be used
     */
    uint32_t harrisRadius;

    /** for DW_FEATURE2D_DETECTOR_TYPE_EX and DW_FEATURE2D_DETECTOR_TYPE_FAST9
     *  STD detector use fixed NMSRadius = 3.
     *
     *  Radius of detector non-maximum suppression
     *  For EX detector, if it is set to zero the default value (1) will be used
     *  As for FAST9 detector, nonmax suppression will be enabled for non-zero value
     *  and will be disabled for zero */
    uint32_t NMSRadius;

    /** for DW_FEATURE2D_DETECTOR_TYPE_EX only
     *  STD detector always output UNIFORM distribution.
     *
     *  The type of feature selection mask */
    dwFeature2DSelectionMaskType maskType;

    /** for DW_FEATURE2D_DETECTOR_TYPE_EX only
     *  Ignored when not using DW_FEATURE2D_SELECTION_MASK_TYPE_GAUSSIAN
     *
     *  The center X of the Gaussian mask, range: [0.0f, 1.0f] */
    float32_t gaussianMaskCenterX;

    /** for DW_FEATURE2D_DETECTOR_TYPE_EX only
     *  Ignored when not using DW_FEATURE2D_SELECTION_MASK_TYPE_GAUSSIAN
     *
     *  The center Y of the Gaussian mask, range: [0.0f, 1.0f] */
    float32_t gaussianMaskCenterY;

    /** for DW_FEATURE2D_DETECTOR_TYPE_EX only
     *  Ignored when not using DW_FEATURE2D_SELECTION_MASK_TYPE_GAUSSIAN
     *
     *  The standard deviation in X of the Gaussian mask
     *  1.0 means the standard deviation equals to the width of the image */
    float32_t gaussianMaskStDevX;

    /** for DW_FEATURE2D_DETECTOR_TYPE_EX only
     *  Ignored when not using DW_FEATURE2D_SELECTION_MASK_TYPE_GAUSSIAN
     *
     *  The standard deviation in Y of the Gaussian mask
     *  1.0 means the standard deviation equals to the height of the image */
    float32_t gaussianMaskStDevY;

    /** for DW_FEATURE2D_DETECTOR_TYPE_EX only
     *  use half-precision floating point to calculate harris corner score */
    bool useHalf;

    /** for DW_FEATURE2D_DETECTOR_TYPE_EX only
     *  Switch to use mask adjusment */
    bool isMaskAdjustmentEnabled;

    /** Determines whether the detector uses `dwFeatureArray::newToOldMap` directly from
     *  input `preTrackedFeatures` in `dwFeature2DDetector_detectFromImage` and `dwFeature2DDetector_detectFromPyramid`.
     *  If set to true, the output `outputDetections::newToOldMap` represents the mapping of output detections to frame(N-1) tracking
     *  If set to false, the output `outputDetections::newToOldMap` represents the mapping of output detections to frame(N) tracking
     *  Default is true. */
    DW_DEPRECATED("WARNING: will be removed in the next major release")
    bool useNewToOldMapFromPreTracked;

    /** Determines whether the detector generates unique feature ids for new detections automatically.
     *  If set to true, each new detected features will be assigned with a unique feature id.
     *  If set to false, dwFeatureArray::ids will be kept.
     *  Default is true. */
    DW_DEPRECATED("WARNING: will be removed in the next major release")
    bool autoGenerateFeatureID;

    /** for DW_FEATURE2D_DETECTOR_TYPE_STD only
     *  set to DW_PROCESSOR_TYPE_PVA_0 or DW_PROCESSOR_TYPE_PVA_1 to call PVA harris detector.
     */
    dwProcessorType processorType;
} dwFeature2DDetectorConfig;

const uint32_t DW_FEATURES2D_DETECTOR_MAX_CELL_SIZE = 128;

/**
 * Initializes dwFeature2DDetector parameters with default values.
 * @param[inout] params dwFeature2DDetector parameters; user can optionally set imageWidth/imageHeight
 *                      before the API call to obtain the default parameters for the resolution
 * @return DW_INVALID_ARGUMENT if params is NULL. <br>
 *         DW_SUCCESS otherwise. <br>
 */
DW_API_PUBLIC
dwStatus dwFeature2DDetector_initDefaultParams(dwFeature2DDetectorConfig* params);

/**
 * Initializes dwFeature2DDetector parameters with values best suited for the given camera using
 * camera extrinsic (dwTransformation3f* cameraToRig) and camera intrinsic (dwConstCameraModelHandle_t cameraHandle)
 *
 * @param[out] params dwFeature2DDetector parameters
 * @param[in] cameraToRig transformed camera extrinsic parameters with respect to rig
 * @param[in] cameraHandle holds camera intrinsic information
 *
 * @return DW_INVALID_ARGUMENT if any parameter is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeature2DDetector_initDefaultParamsForCamera(dwFeature2DDetectorConfig* params,
                                                        const dwTransformation3f* cameraToRig,
                                                        dwConstCameraModelHandle_t cameraHandle);

/**
 * Creates and initializes a feature Detector.
 *
 * @param[out] obj A pointer to the feature Detector handle is returned here.
 * @param[in] config Specifies the configuration parameters.
 * @param[in] cudaStream Specifies the CUDA stream to use for Detector operations.
 * @param[in] context Specifies the handle to the context under which it is created.
 *
 * @return DW_INVALID_ARGUMENT if feature Detector handle or context are NULL or config is invalid.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeature2DDetector_initialize(dwFeature2DDetectorHandle_t* obj,
                                        const dwFeature2DDetectorConfig* config,
                                        cudaStream_t cudaStream,
                                        dwContextHandle_t context);

/**
 * Resets a feature Detector.
 *
 * @param[in] obj Specifies the feature Detector handle to be reset.
 *
 * @return DW_INVALID_ARGUMENT if feature Detector handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeature2DDetector_reset(dwFeature2DDetectorHandle_t obj);

/**
 * Releases the feature Detector.
 * This method releases all resources associated with a feature Detector.
 *
 * @note This method renders the handle unusable.
 *
 * @param[in] obj The feature detector handle to be released.
 *
 * @return DW_INVALID_ARGUMENT if feature Detector handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 *
 */
DW_API_PUBLIC
dwStatus dwFeature2DDetector_release(dwFeature2DDetectorHandle_t obj);

/**
 * Sets a mask to ignore areas of the image. Areas where mask != 0xff will not produce new features
 * when `detectNewFeatures()` is called. The size of the mask must be the same as the image.
 *
 * @note If this function is called after `dwFeature2DDetector_initialize()`, it must be called
 *       with the same input parameters after `dwFeature2DDetector_reset` because `reset()` function
 *       will clear the existing mask.
 *
 * @param[in] d_mask A GPU pointer to the mask data. The data is copied.
 * @param[in] maskStrideBytes Specifies the stride, in bytes, for each row of the mask.
 * @param[in] maskWidth Specifies the width of the mask.
 * @param[in] maskHeight Specifies the height of the mask.
 * @param[in] obj Specifies the feature detector handle.
 *
 * @return DW_INVALID_ARGUMENT if feature Detector handle or d_mask are NULL or maskWidth or maskHeight are invalid.<br>
 *        DW_SUCCESS otherwise.<br>
*/
DW_API_PUBLIC
dwStatus dwFeature2DDetector_setMask(const uint8_t* d_mask,
                                     const uint32_t maskStrideBytes,
                                     const uint32_t maskWidth,
                                     const uint32_t maskHeight,
                                     dwFeature2DDetectorHandle_t obj);

/**
 * Detects new features and append them after old tracked features. Features are added only in areas
 * where the mask == 0xff means valid area, any pixel whose mask value != 0xff will be ignored during tracking.
 *
 * @param[out] outputDetections Detected features, composed of old tracked features and new detections, new
 *             detections are appended after old features. Must be created by `dwFeatureArray_create()` with
 *             DW_PROCESSOR_TYPE_GPU flag and the same maxFeatureCount when initializing detector
 * @param[in] image Specifies the image on which feature detection takes place, must be a CUDA image.
 * @param[inout] preTrackedFeatures Previous tracked features, if detector type is `DW_FEATURE2D_DETECTOR_TYPE_EX`,
 *               some of old features will be marked as `DW_FEATURE2D_STATUS_INVALID` during detection.
 *               The `dwFeatureArray` can either be an empty array allocated by `dwFeatureArray_create` or a
 *               slice from feature history by `dwFeatureHistoryArray_getXXX`
 * @param[in] d_normalizedCrossCorrelation GPU pointer to nccScores of each feature, will be ignored if it's NULL.
 *            If the feature detector is connected to a tracker, d_normalizedCrossCorrelation can be filled by
 *            `dwFeature2DTracker_trackFeatures`.
 * @param[in] obj Specifies the feature detector handle.
 *
 * @return DW_INVALID_ARGUMENT if detector handle, preTrackedFeatures or image are NULL.<br>
 *         DW_INVALID_ARGUMENT if preTrackedFeatures are not allocated on GPU.<br>
 *         DW_INVALID_HANDLE if image is not a CUDA image.<br>
 *         DW_SUCCESS otherwise.<br>
 *
 * @note If there're N features in preTrackedFeatures,
 *       for DW_FEATURE2D_DETECTOR_TYPE_STD, the first N features in detectedFeatures are from preTrackedFeaturs,
 *       new detections are appended after old ones.
 *       for DW_FEATURE2D_DETECTOR_TYPE_EX, the first M(M <= N) features in detectedFeatures are from preTrackedFeaturs
 *       new detections are appended after old ones. The reason that M <= N is that some features in preTrackedFeatures
 *       will be marked as invalid to meet the restriction of uniform/gaussian distribution.
*/
DW_API_PUBLIC
dwStatus dwFeature2DDetector_detectFromImage(dwFeatureArray* outputDetections,
                                             dwImageHandle_t image,
                                             dwFeatureArray* preTrackedFeatures,
                                             const float32_t* d_normalizedCrossCorrelation,
                                             dwFeature2DDetectorHandle_t obj);

/**
 * Detects new features and append them after old tracked features. Features are added only in areas
 * where the mask == 0xff means valid area, any pixel whose mask value != 0xff will be ignored during tracking.
 *
 * @param[out] outputDetections Detected features, composed of old tracked features and new detections, new
 *             detections are appended after old features. Must be created by `dwFeatureArray_create()` with
 *             DW_PROCESSOR_TYPE_GPU flag and the same maxFeatureCount when initializing detector
 * @param[in] pyramid Specifies the pyramid on which feature detection takes place.
 * @param[inout] preTrackedFeatures Previous tracked features, if detector type is `DW_FEATURE2D_DETECTOR_TYPE_EX`,
 *               some of old features will be marked as `DW_FEATURE2D_STATUS_INVALID` during detection.
 *               The `dwFeatureArray` can either be an empty array allocated by `dwFeatureArray_create` or a
 *               slice from feature history by `dwFeatureHistoryArray_getXXX` *
 * @param[in] d_normalizedCrossCorrelation GPU pointer to nccScores of each feature, will be ignored if it's NULL.
 *            If the feature detector is connected to a tracker, d_normalizedCrossCorrelation can be filled by
 *            `dwFeature2DTracker_trackFeatures`.
 * @param[in] obj Specifies the feature detector handle.
 *
 * @return DW_INVALID_ARGUMENT if detector handle, preTrackedFeatures or image are NULL.<br>
 *         DW_INVALID_ARGUMENT if preTrackedFeatures are not allocated on GPU.<br>
 *         DW_SUCCESS otherwise.<br>
 *
 * @note If there're N features in preTrackedFeatures,
 *       for DW_FEATURE2D_DETECTOR_TYPE_STD, the first N features in detectedFeatures are from preTrackedFeaturs,
 *       new detections are appended after old ones.
 *       for DW_FEATURE2D_DETECTOR_TYPE_EX, the first M(M <= N) features in detectedFeatures are from preTrackedFeaturs
 *       new detections are appended after old ones. The reason that M <= N is that some features in preTrackedFeatures
 *       will be marked as invalid to meet the restriction of uniform/gaussian distribution.
*/
DW_API_PUBLIC
dwStatus dwFeature2DDetector_detectFromPyramid(dwFeatureArray* outputDetections,
                                               const dwPyramidImage* pyramid,
                                               dwFeatureArray* preTrackedFeatures,
                                               const float32_t* d_normalizedCrossCorrelation,
                                               dwFeature2DDetectorHandle_t obj);

/**
 * @brief dwFeature2DDetector_getValidTrackedCount
 * @param d_validTrackedCount
 * @param obj
 * @return
 */
DW_DEPRECATED("WARNING: will be removed in the next major release, validTrackedCount pointer can be read directly from dwFeatureArray")
DW_API_PUBLIC
dwStatus dwFeature2DDetector_getValidTrackedCount(const uint32_t** d_validTrackedCount,
                                                  dwFeature2DDetectorHandle_t obj);

/**
 * Sets the CUDA stream for CUDA related operations.
 *
 * @note The ownership of the stream remains by the callee.
 *
 * @param[in] stream The CUDA stream to be used. Default is the one passed during dwFeature2DDetector_initialize.
 * @param[in] obj A handle to the feature Detector module to set CUDA stream for.
 *
 * @return DW_INVALID_ARGUMENT if feature Detector handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeature2DDetector_setCUDAStream(cudaStream_t stream, dwFeature2DDetectorHandle_t obj);

/**
 * Gets the CUDA stream used by the feature Detector.
 *
 * @param[out] stream The CUDA stream currently used.
 * @param[in] obj A handle to the feature Detector module.
 *
 * @return DW_INVALID_ARGUMENT if feature Detector handle or stream are NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeature2DDetector_getCUDAStream(cudaStream_t* stream, dwFeature2DDetectorHandle_t obj);

/**
 * Sets the CUPVA stream for PVA related operations.
 *
 * @note The ownership of the stream remains by the callee.
 *
 * @param[in] stream The CUPVA stream to be used. Default is the one passed during dwFeature2DDetector_initialize.
 * @param[in] obj A handle to the feature Detector module to set CUPVA stream for.
 *
 * @return DW_INVALID_ARGUMENT if feature Detector handle is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeature2DDetector_setPVAStream(cupvaStream_t stream, dwFeature2DDetectorHandle_t obj);

/**
 * Gets the CUPVA stream used by the feature Detector.
 *
 * @param[out] stream The CUPVA stream currently used.
 * @param[in] obj A handle to the feature Detector module.
 *
 * @return DW_INVALID_ARGUMENT if feature Detector handle or stream are NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeature2DDetector_getPVAStream(cupvaStream_t* stream, dwFeature2DDetectorHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_IMAGEPROCESSING_FEATUREDETECTOR_FEATUREDETECTOR_H_

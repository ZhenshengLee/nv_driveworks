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
// Copyright (c) 2021-2022 NVIDIA Corporation. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: 2D Feature Descriptor</b>
 *
 * @b Description: This file defines 2D feature descriptor
 */

/**
  * @defgroup featureDescriptor_group Feature 2D Descriptor Interface
  *
  * @brief Defines 2D-based feature description.
  *
  * @{
  */

#ifndef DW_IMAGEPROCESSING_FEATURE_DESCRIPTOR_H_
#define DW_IMAGEPROCESSING_FEATURE_DESCRIPTOR_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>
#include <dw/image/Image.h>
#include <dw/imageprocessing/filtering/Pyramid.h>
#include <dw/imageprocessing/features/FeatureList.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Handle representing a feature descriptor. */
typedef struct dwFeature2DDescriptorObject* dwFeature2DDescriptorHandle_t;

/** Handle representing a const feature descriptor. */
typedef struct dwFeature2DDescriptorObject const* dwConstFeature2DDescriptorHandle_t;

/** Feature descriptor algorithm */
typedef enum {
    /** ORB descriptor */
    DW_FEATURE2D_DESCRIPTOR_ALGORITHM_ORB = 0,
    DW_FEATURE2D_DESCRIPTOR_ALGORITHM_COUNT
} dwFeature2DDescriptorAlgorithm;

/**
     * Holds configuration parameters for a feature descriptor.
     */
typedef struct dwFeature2DDescriptorConfig
{
    /** Detecting algorithm defined by `dwFeature2DDescriptorType` */
    dwFeature2DDescriptorAlgorithm algorithm;

    /** Width of the images that the Descriptor runs on. */
    uint32_t imageWidth;

    /** Height of the images that the descriptor runs on. */
    uint32_t imageHeight;

    /** Upper bound on number of features handled. */
    uint32_t maxFeatureCount;

    /** for DW_FEATURE2D_DESCRIPTOR_ALGORITHM_ORB only
     *  set to DW_PROCESSOR_TYPE_PVA_0 to call PVA orb descriptor.
     */
    dwProcessorType processorType;

} dwFeature2DDescriptorConfig;

/**
     * @brief The stage of feature descriptor
     */
typedef enum dwFeature2DDescriptorStage {
    // GPU Async stage
    DW_FEATURE2D_DESCRIPTOR_STAGE_GPU_ASYNC = 0,
    // CPU preprocess
    DW_FEATURE2D_DESCRIPTOR_STAGE_CPU_SYNC,
    // PVA Process
    DW_FEATURE2D_DESCRIPTOR_STAGE_PVA_ASYNC,
    // CPU Post process
    DW_FEATURE2D_DESCRIPTOR_STAGE_CPU_SYNC_POSTPROCESS,
    /// Process the postprocess part of the feature descriptor pipeline on CPU.
    DW_FEATURE2D_DESCRIPTOR_STAGE_GPU_ASYNC_POSTPROCESS = 100,
} dwFeature2DDescriptorStage;

/**
     * Initializes dwFeature2DDescriptor parameters with default values.
     * @param[out] params dwFeature2DDescriptor parameters
     * @return DW_INVALID_ARGUMENT if params is NULL. <br>
     *         DW_SUCCESS otherwise. <br>
     */
DW_API_PUBLIC
dwStatus dwFeature2DDescriptor_initDefaultParams(dwFeature2DDescriptorConfig* params);

/**
     * Creates and initializes a feature descriptor.
     *
     * @param[out] obj A pointer to the feature descriptor handle is returned here.
     * @param[in] config Specifies the configuration parameters.
     * @param[in] cudaStream Specifies the CUDA stream to use for descriptor operations.
     * @param[in] context Specifies the handle to the context under which it is created.
     *
     * @return DW_INVALID_ARGUMENT if feature descriptor handle or context are NULL or config is invalid.<br>
     *         DW_SUCCESS otherwise.<br>
     */
DW_API_PUBLIC
dwStatus dwFeature2DDescriptor_initialize(dwFeature2DDescriptorHandle_t* obj,
                                          const dwFeature2DDescriptorConfig* config,
                                          cudaStream_t cudaStream,
                                          dwContextHandle_t context);

/**
     * Resets a feature descriptor.
     *
     * @param[in] obj Specifies the feature descriptor handle to be reset.
     *
     * @return DW_INVALID_ARGUMENT if feature descriptor handle is NULL.<br>
     *         DW_SUCCESS otherwise.<br>
     */
DW_API_PUBLIC
dwStatus dwFeature2DDescriptor_reset(dwFeature2DDescriptorHandle_t obj);

/**
     * Releases the feature descriptor.
     * This method releases all resources associated with a feature descriptor.
     *
     * @note This method renders the handle unusable.
     *
     * @param[in] obj The feature descriptor handle to be released.
     *
     * @return DW_INVALID_ARGUMENT if feature descriptor handle is NULL.<br>
     *         DW_SUCCESS otherwise.<br>
     *
     */
DW_API_PUBLIC
dwStatus dwFeature2DDescriptor_release(dwFeature2DDescriptorHandle_t obj);

/**
     * Sets the CUDA stream for CUDA related operations.
     *
     * @note The ownership of the stream remains by the callee.
     *
     * @param[in] stream The CUDA stream to be used. Default is the one passed during dwFeature2DDescriptor_initialize.
     * @param[in] obj A handle to the feature descriptor module to set CUDA stream for.
     *
     * @return DW_INVALID_ARGUMENT if feature descriptor handle is NULL.<br>
     *         DW_SUCCESS otherwise.<br>
     */
DW_API_PUBLIC
dwStatus dwFeature2DDescriptor_setCUDAStream(cudaStream_t stream, dwFeature2DDescriptorHandle_t obj);

/**
     * Gets the CUDA stream used by the feature descriptor.
     *
     * @param[out] stream The CUDA stream currently used.
     * @param[in] obj A handle to the feature descriptor module.
     *
     * @return DW_INVALID_ARGUMENT if feature descriptor handle or stream are NULL.<br>
     *         DW_SUCCESS otherwise.<br>
     */
DW_API_PUBLIC
dwStatus dwFeature2DDescriptor_getCUDAStream(cudaStream_t* stream, dwFeature2DDescriptorHandle_t obj);

/**
     * Binds output dwFeatureDescriptorArray to descriptor object
     * @param[out] outputDescriptors Feature descriptor array containing the calculated feature
     *             descriptors features. Must be created by `dwFeatureDescriptorArray_create()` with
     *             DW_PROCESSOR_TYPE_GPU flag and the same maxFeatureCount when initializing descriptor
     * @param[in] obj Specifies the feature descriptor handle.
     *
     * @return DW_INVALID_ARGUMENT if descriptor handle or outputDescriptors are NULL.<br>
     *         DW_INVALID_ARGUMENT if outputDescriptors is not created with descriptor compatible parameters.<br>
     *         DW_SUCCESS otherwise.<br>
     *
     * @note App must call `dwFeature2DDescriptor_bindOutput` to set output feature descriptor array.
     */
DW_API_PUBLIC
dwStatus dwFeature2DDescriptor_bindOutput(dwFeatureDescriptorArray* outputDescriptors,
                                          dwFeature2DDescriptorHandle_t obj);

/**
     * Binds input parameters to descriptor object
     * @param[in] image Specifies the image on which feature detection takes place, must be a CUDA image.
     * @param[inout] features Features whose descriptor to be calculated.
     * @param[in] obj Specifies the feature descriptor handle.
     *
     * @return DW_INVALID_ARGUMENT if descriptor handle, features or image are NULL.<br>
     *         DW_INVALID_ARGUMENT if features are not allocated on GPU.<br>
     *         DW_SUCCESS otherwise.<br>
     *
     * @note `dwFeature2DDescriptor_bindInputImage` must be called at least once before `dwFeature2DDescriptor_processImage`.
     */
DW_API_PUBLIC
dwStatus dwFeature2DDescriptor_bindInputBuffers(dwImageHandle_t image,
                                                dwFeatureArray* features,
                                                dwFeature2DDescriptorHandle_t obj);

/**
     * Do detections on the image bound by `dwFeature2DDescriptor_bindInputImage`, output will be written to
     * `dwFeatureDescriptorArray` bound by `dwFeature2DDescriptor_bindOutput`.
     *
     * @param[in] stage Speicifies the detecting stage
     * @param[in] obj Specifies the feature descriptor handle.
     *
     * @return DW_CALL_NOT_ALLOWED if there's no output dwFeatureDescriptorArray bound to descriptor.<br>
     *                             if there's no input image or features bound to descriptor.<br>
     *         DW_SUCCESS otherwise.<br>
     */
DW_API_PUBLIC
dwStatus dwFeature2DDescriptor_processImage(dwFeature2DDescriptorStage stage,
                                            dwFeature2DDescriptorHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_IMAGEPROCESSING_FEATURE_DESCRIPTOR_H_
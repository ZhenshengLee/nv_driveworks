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
 * <b>NVIDIA DriveWorks API: Data Conditioner Methods</b>
 *
 * @b Description: This file defines Data Conditioner methods.
 */

/**
 * @defgroup dataconditioner_group DataConditioner Interface
 *
 * @brief Defines the DataConditioner module for performing common transformations on input images for DNN.
 *
 * @{
 */

#ifndef DW_DNN_DATACONDITIONER_H_
#define DW_DNN_DATACONDITIONER_H_

#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>
#include <dw/image/Image.h>
#include <dw/dnn/tensor/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Struct representing parameters for DataConditioner
typedef struct dwDataConditionerParams
{
    /// Mean value to be subtracted. Range [0, 255]. Default is 0.
    float32_t meanValue[DW_MAX_IMAGE_PLANES];
    /// Standard deviation with range [0, 255]. Default is 1.0.
    /// The results are computed using the following formula, where
    ///     R, G and B are from the input image and have range of [0, 255]
    ///     meanImage is optional and is 0 if it is not set
    ///     perPlaneMeanX is the mean per plane for that channel and is optional and 0 if it is not set
    /// R' = ((R - meanValue[0] - meanImage[pixelIndex] - perPlaneMeanR) / stdev[0]) * scaleCoefficient
    /// G' = ((G - meanValue[1] - meanImage[pixelIndex] - perPlaneMeanG) / stdev[1]) * scaleCoefficient
    /// B' = ((B - meanValue[2] - meanImage[pixelIndex] - perPlaneMeanB) / stdev[2]) * scaleCoefficient
    float32_t stdev[DW_MAX_IMAGE_PLANES];
    /// Mean image to be subtracted. Default is nullptr.
    /// Mean image is expected to be float16 or float32. The pixel format is required to be R or RGBA with
    /// interleaved channels. The dimensions of the mean image must meet the dimensions of network input.
    /// Pixel range: [0, 255]. Default is nullptr.
    const dwImageCUDA* meanImage;
    /// Boolean indicating whether planes should be split. Default is true.
    bool splitPlanes;
    /// Scale pixel intensities. Default is 1.0.
    float32_t scaleCoefficient;
    /// Boolean indicating whether the aspect ratio of the input image should be kept (false) or the image
    /// should be stretched to the roi specified (true). Default false
    bool ignoreAspectRatio;
    /// Boolean indicating whether to perform per-plane mean normalization. Default false
    bool doPerPlaneMeanNormalization;
    /// Index of each channel determining the channel order. channelIdx[x] = y means that the calculation on channel x will write to output channel y
    uint32_t channelIdx[DW_MAX_IMAGE_PLANES];
    /// Boolean to decide whether to convert pixels to gray before computing tensor
    bool convertToGray;
} dwDataConditionerParams;

/**
 * @brief Handle to a DataConditioner.
 */
typedef struct dwDataConditionerObject* dwDataConditionerHandle_t;

/**
 * @brief Initializes DataConditioner parameters with default values.
 *
 * @param[out] dataConditionerParams DataConditioner parameters.
 *
 * @return DW_INVALID_ARGUMENT if parameters are NULL.<br>
 *         DW_SUCCESS
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwDataConditioner_initParams(dwDataConditionerParams* const dataConditionerParams);

/**
  * Initializes a DataConditioner module.
  *
  * @param[out] obj A pointer to the DataConditioner handle for the created module.
  * @param[in] networkInputBlobSize A pointer to the size of the network input as a blob that the DataConditioner
  * expects.
  * @param[in] maxNumImages Maximum number of images that a single step can process.
  * @param[in] dataConditionerParams DataConditioner parameters.
  * @param[in] stream Specifies the CUDA stream to use for operations.
  * @param[in] ctx Specifies the handler to the context under which the DataConditioner module is created.
  *
  * @note DataConditioner parameters must be initialized using dwDataConditioner_initParams
  * before modifying.
  *
  * @return DW_INVALID_ARGUMENT if dataconditioner handle or network Input Blobsize are NULL or dataConditionerParams is invalid.<br>
  *         DW_INVALID_HANDLE if dwContext handle is NULL.<br>
  *         DW_SUCCESS
  * @par API Group
  * - Init: Yes
  * - Runtime: No
  * - De-Init: No
  */
DW_API_PUBLIC
dwStatus dwDataConditioner_initialize(dwDataConditionerHandle_t* const obj,
                                      dwBlobSize const* const networkInputBlobSize,
                                      uint32_t const maxNumImages,
                                      dwDataConditionerParams const* const dataConditionerParams,
                                      cudaStream_t const stream, dwContextHandle_t const ctx);

/**
  * Initializes a DataConditioner module.
  *
  * @param[out] obj A pointer to the DataConditioner handle for the created module.
  * @param[in] outputProperties Tensor properties of the output. This can be obtained for dwDNN.
  * @param[in] maxNumImages Maximum number of images to run DataConditioner and combine into one Tensor
  * @param[in] dataConditionerParams DataConditioner parameters.
  * @param[in] stream Specifies the CUDA stream to use for operations.
  * @param[in] ctx Specifies the handler to the context under which the DataConditioner module is created.
  *
  * @note DataConditioner parameters must be initialized using dwDataConditioner_initParams
  * before modifying.
  *
  * @note Supported output data types are DW_TYPE_FLOAT16 and DW_TYPE_FLOAT32. INT8 output is also supported, via reformatting following TRT's spec with rounding done via __float2int_rn  
  * @note Supported output layouts are DW_DNN_TENSOR_LAYOUT_NHWC and DW_DNN_TENSOR_LAYOUT_NCHW.
  * The number of dimensions must be 4.
  *
  * @return DW_INVALID_ARGUMENT if dataconditioner handle or outputProperties are NULL or dataConditionerParams is invalid.<br>
  *         DW_INVALID_HANDLE if dwContext handle is NULL.<br>
  *         DW_SUCCESS
  * @par API Group
  * - Init: Yes
  * - Runtime: No
  * - De-Init: No
  */
DW_API_PUBLIC
dwStatus dwDataConditioner_initializeFromTensorProperties(dwDataConditionerHandle_t* const obj,
                                                          dwDNNTensorProperties const* const outputProperties,
                                                          uint32_t const maxNumImages,
                                                          dwDataConditionerParams const* const dataConditionerParams,
                                                          cudaStream_t const stream, dwContextHandle_t const ctx);

/**
 * Sets the CUDA stream for CUDA related operations.
 *
 * @note The ownership of the stream remains by the callee.
 *
 * @param[in] stream The CUDA stream to be used. Default is the one passed during dwDataConditioner_initialize.
 * @param[in] obj A handle to the data conditioner module for which to set CUDA stream.
 *
 * @return DW_INVALID_ARGUMENT if the given dataconditioner handle is NULL. <br>
 *         DW_SUCCESS otherwise.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwDataConditioner_setCUDAStream(cudaStream_t const stream, dwDataConditionerHandle_t const obj);

/**
 * Gets the CUDA stream used by the data conditioner.
 *
 * @param[out] stream The CUDA stream currently used by the data conditioner.
 * @param[in] obj A handle to the data conditioner module.
 *
 * @return DW_INVALID_ARGUMENT if the given dataconditioner handle or stream are NUll. <br>
 *         DW_SUCCESS otherwise.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwDataConditioner_getCUDAStream(cudaStream_t* const stream, dwDataConditionerHandle_t const obj);

/**
 * Resets the DataConditioner module.
 *
 * @param[in] obj Specifies the DataConditioner handle to reset.
 *
 * @return DW_INVALID_ARGUMENT if dataconditioner handle is NULL.<br>
 *         DW_SUCCESS otherwise.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwDataConditioner_reset(dwDataConditionerHandle_t const obj);

/**
 * Releases the DataConditioner module.
 *
 * @param[in] obj The object handle to release.
 *
 * @return DW_INVALID_HANDLE if dataconditioner handle is NULL.<br>
 *         DW_SUCCESS otherwise.
 *
 * @note This method renders the handle unusable.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwDataConditioner_release(dwDataConditionerHandle_t const obj);

/**
 * Runs the configured transformations on an image.
 *
 * @param[out] dOutputImage A pointer to the preallocated output blob in GPU memory.
 * @param[in] inputImages Pointer to a list of pitched images to batch.
 * @param[in] numImages Number of pitched images.
 * @param[in] roi ROI to extract from input images.
 * @param[in] addressMode cudaTextureAddressMode specifies how to fill the potentially empty part of the ROI.
 * @param[in] obj Specifies the DataConditioner handle.
 *
 * @note Supported image types: R, RGB, RGBA, RCB, RCC and YUV444 (3 planes of equal size).
 * @note If the type of inputImage is RGBA, alpha channel is dropped during the operations; therefore,
 * outputImage has three channels instead of four.
 * @note If the type of inputImage is RGB, the channels may not be interleaved. Use RGBA if that is intended.
 * @note All the images in the list are required to have the same dimensions as the one given during
 * initialization.
 * @note The ROI is scaled to match the network input size. If ignoreAspectRatio is false, scaling is performed
 * by maintaining the original ROI aspect ratio. Since after scaling there might be an empty part in the
 * scaled ROI, this will be filled according to addressMode.
 * @note numImages cannot exceed the batchsize set at initialization time. If numImages is higher than 1,
 * the resultant output image is batched, and it can then be given to the corresponding DNN.
 *
 * @return DW_INVALID_ARGUMENT if outputImage or dataconditioner handle or roi are NULL, or inputImage is null or invalid.<br>
 *         DW_CUDA_ERROR in case of an underlying cuda failure.<br>
 *         DW_SUCCESS otherwise.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwDataConditioner_prepareDataRaw(float32_t* const dOutputImage, dwImageCUDA const* const* const inputImages,
                                          uint32_t const numImages, dwRect const* const roi,
                                          cudaTextureAddressMode const addressMode, dwDataConditionerHandle_t const obj);

/**
 * Runs the configured transformations on an image.
 *
 * @param[out] tensorOutput Tensor CUDA handle to store output.
 * @param[in] inputImages List of Image CUDA handles as input
 * @param[in] numImages Number of input images.
 * @param[in] rois ROIs to extract from input images.
 * @param[in] addressMode cudaTextureAddressMode specifies how to fill the potentially empty part of the ROI.
 * @param[in] obj Specifies the DataConditioner handle.
 *
 * @note Supported image types: R, RGB, RGBA, RCB and RCC.
 * @note If the type of inputImage is RGBA, alpha channel is dropped during the operations; therefore,
 * outputImage has three channels instead of four.
 * @note If the type of inputImage is RGB, the channels may not be interleaved. Use RGBA if that is intended.
 * @note All the images in the list are required to have the same dimensions as the one given during
 * initialization.
 * @note The ROIs is scaled to match the network input size. If ignoreAspectRatio is false, scaling is performed
 * by maintaining the original ROIs aspect ratio. Since after scaling there might be an empty part in the
 * scaled ROIs, this will be filled according to addressMode.
 * @note numImages cannot exceed the batchsize set at initialization time. If numImages is higher than 1,
 * the resultant output image is batched, and it can then be given to the corresponding DNN.
 *
 * @return DW_INVALID_ARGUMENT if outputImage or dataconditioner handle or rois are NULL, or inputImage is null or invalid.<br>
 *         DW_CUDA_ERROR in case of an underlying cuda failure.<br>
 *         DW_SUCCESS otherwise.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwDataConditioner_prepareData(dwDNNTensorHandle_t const tensorOutput,
                                       dwImageHandle_t const* const inputImages,
                                       uint32_t const numImages, dwRect const* const rois,
                                       cudaTextureAddressMode const addressMode, dwDataConditionerHandle_t const obj);

/**
 * Computes the output size based on the input size and the operations that have been added.
 *
 * @param[out] outputBlobSize Size of the output blob after the transformations have been applied.
 * @param[in] obj Specifies the DataConditioner handle.
 *
 * @return DW_INVALID_ARGUMENT if outputBlobSize or dataconditioner handle are NULL.<br>
 *         DW_SUCCESS otherwise.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwDataConditioner_getOutputSize(dwBlobSize* const outputBlobSize,
                                         dwDataConditionerHandle_t const obj);

/**
 * Computes the position of a point from the interpreted DNN output on the input image.
 *
 * @param[out] outputX Pointer to X coordinate on the input image.
 * @param[out] outputY Pointer to Y coordinate on the input image.
 * @param[in] inputX X coordinate from DNN output.
 * @param[in] inputY Y coordinate from DNN output.
 * @param[in] roi ROI extracted from the input image.
 * @param[in] obj Specifies the DataConditioner handle.
 *
 * @return DW_INVALID_ARGUMENT if outputX, outputY, roi or dataconditioner handle are NULL.<br>
 *         DW_SUCCESS
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwDataConditioner_outputPositionToInput(float32_t* const outputX, float32_t* const outputY,
                                                 float32_t const inputX, float32_t const inputY,
                                                 dwRect const* const roi, dwDataConditionerHandle_t const obj);
#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_DNN_DATACONDITIONER_H_

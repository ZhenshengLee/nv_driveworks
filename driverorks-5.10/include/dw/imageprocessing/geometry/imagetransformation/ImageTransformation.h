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
// SPDX-FileCopyrightText: Copyright (c) 2015-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Image Transformation Methods</b>
 *
 * @b Description: This file defines image transformation methods.
 */

/**
 * @defgroup imagetransformation_group Image Transformation Interface
 *
 * @brief Defines the image transformation module.
 *
 * @{
 */

#ifndef DW_IMAGEPROCESSING_GEOMETRY_IMAGETRANSFORMATION_H_
#define DW_IMAGEPROCESSING_GEOMETRY_IMAGETRANSFORMATION_H_

#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>
#include <dw/image/Image.h>
#include <dw/imageprocessing/common/ImageProcessingCommon.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwImageTransformationObject* dwImageTransformationHandle_t;

typedef struct dwImageTransformationParameters
{
    /// Boolean indicating whether the aspect ratio of the input image should be kept (false) or the image
    /// should be stretched to the roi specified (true). Default false
    bool ignoreAspectRatio;
} dwImageTransformationParameters;

/**
 * Initializes an Image Transformation Engine
 * @param[out] handle Pointer to the Image Transformation Engine.
 * @param[in] params parameters of image transformation.
 * @param[in] context Handle to Driveworks
 *
 * @return DW_INVALID_ARGUMENT if the images are invalid <br>
 * DW_SUCCESS if the operation is successful <br>
 * DW_CUDA_ERROR if the underlying cuda operation failed <br>
 */
DW_API_PUBLIC
dwStatus dwImageTransformation_initialize(dwImageTransformationHandle_t* handle, dwImageTransformationParameters params,
                                          dwContextHandle_t context);

/**
 * Resets an Image Transformation Engine
 * @param[in] obj Pointer to the Image Transformation Engine.
 *
 * @return DW_INVALID_ARGUMENT if the images are invalid <br>
 * DW_SUCCESS if the operation is successful <br>
 */
DW_API_PUBLIC
dwStatus dwImageTransformation_reset(dwImageTransformationHandle_t obj);

/**
 * Releases an Image Transformation Engine
 * @param[in] handle Pointer to the Image Transformation Engine.
 *
 * @return DW_INVALID_ARGUMENT if the images are invalid <br>
 * DW_SUCCESS if the operation is successful <br>
 * DW_CUDA_ERROR if the underlying cuda operation failed <br>
 */
DW_API_PUBLIC
dwStatus dwImageTransformation_release(dwImageTransformationHandle_t handle);

/**
 * Append the allocation attribute such that the images created of type DW_IMAGE_NVMEDIA can be fed to dwImageTransformation_copyXXX()
 *
 * @param[inout] imgProps Image properties
 * @param[in] obj dwImageTransformation handle
 * 
 * @note The imgProps are read and used to generate the allocation attributes
 *       needed by the driver. The allocation attributes are stored back into
 *       imgProps.meta.allocAttrs. Applications do not need to free or alter the
 *       imgProps.meta.allocAttrs in any way. The imgProps.meta.allocAttrs are only used
 *       by DriveWorks as needed when the given imgProps are used to allocate dwImages.
 *       If the application alters the imgProps after calling this API, the
 *       imgProps.meta.allocAttrs may no longer be applicable to the imgProps and calls related
 *       to allocating images will fail.
 * @note if imgProps.meta.allocAttrs does not have allocated Memory, this would be allocated by 
 *       DW and will be owned by DW context until context is destroyed 
 *       and should be used wisely as it the space is limited.
 *
 * @return DW_NVMEDIA_ERROR - if underlying camera driver had an NvMedia error. <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_SUCCESS
 *
 */
DW_API_PUBLIC
dwStatus dwImageTransformation_appendAllocationAttributes(dwImageProperties* const imgProps,
                                                          dwImageTransformationHandle_t obj);

/**
 * Sets the border mode used by the APIs of Image Transformation
 * @param[in] mode Border mode
 * @param[in] obj Handle to the Image Transformation engine
 *
 * @return DW_INVALID_ARGUMENT if the images are invalid <br>
 * DW_SUCCESS if the operation is successful <br>
 */
DW_API_PUBLIC
dwStatus dwImageTransformation_setBorderMode(dwImageProcessingBorderMode mode,
                                             dwImageTransformationHandle_t obj);

/**
 * Sets the interpolation mode used by the APIs of Image Transformation
 * @param[in] mode Interpolation mode
 * @param[in] obj Handle to the Image Transformation engine
 *
 * @return DW_INVALID_ARGUMENT if the images are invalid <br>
 * DW_SUCCESS if the operation is successful <br>
 */
DW_API_PUBLIC
dwStatus dwImageTransformation_setInterpolationMode(dwImageProcessingInterpolation mode,
                                                    dwImageTransformationHandle_t obj);

/**
 * Sets the cuda stream used by the APIs of Image Transformation
 * @param[in] stream CUDA stream
 * @param[in] obj Handle to the Image Transformation engine
 *
 * @return DW_INVALID_ARGUMENT if the images are invalid <br>
 * DW_SUCCESS if the operation is successful <br>
 * DW_CUDA_ERROR if the underlying cuda operation failed <br>
 */
DW_API_PUBLIC
dwStatus dwImageTransformation_setCUDAStream(cudaStream_t stream, dwImageTransformationHandle_t obj);

/**
 * Gets the cuda stream used by the APIs of Image Transformation
 * @param[out] stream CUDA stream
 * @param[in] obj Handle to the Image Transformation engine
 *
 * @return
 * DW_SUCCESS if the operation is successful <br>
 * DW_CUDA_ERROR if the underlying cuda operation failed <br>
 */

DW_API_PUBLIC
dwStatus dwImageTransformation_getCUDAStream(cudaStream_t* stream, dwImageTransformationHandle_t obj);

/**
 * Resizes the input image subregion and copies the result into the previously allocated output image, in a specified subregion, of the same type (CUDA or NvMedia) and format (any)
 * @param[out] outputImage Pointer to the output image.
 * @param[in] inputImage Pointer to the input image.
 * @param[in] outputROI Pointer to a ROI on the output image where to copy the result. If false, defaults to the full image
 * @param[in] inputROI Pointer to a ROI on the input image, where the source pixels are located. If null, defaults to the full image
 * @param[in] obj Handle to the Image Transformation engine
 *
 * @return DW_INVALID_ARGUMENT if the images are invalid <br>
 * DW_SUCCESS if the operation is successful <br>
 * DW_CUDA_ERROR if the underlying cuda operation failed <br>
 * DW_NVMEDIA_ERROR if the underlying NvMedia operation failed
 */
DW_API_PUBLIC
dwStatus dwImageTransformation_copy(dwImageHandle_t outputImage, const dwImageHandle_t inputImage,
                                    const dwRect* outputROI, const dwRect* inputROI,
                                    dwImageTransformationHandle_t obj);

/**
 * Resizes the input image sub region and copies the result into the previously allocated output image, of the same type (CUDA or NvMedia) and format (any)
 * @param[out] outputImage Pointer to the output image.
 * @param[in] inputImage Pointer to the input image.
 * @param[in] inputROI Pointer to a ROI on the input image, where the source pixels are located.
 * @param[in] obj Handle to the Image Transformation engine
 *
 * @return DW_INVALID_ARGUMENT if the images are invalid <br>
 * DW_SUCCESS if the operation is successful <br>
 * DW_CUDA_ERROR if the underlying cuda operation failed <br>
 * DW_NVMEDIA_ERROR if the underlying NvMedia operation failed
 */
DW_API_PUBLIC
dwStatus dwImageTransformation_copySubImage(dwImageHandle_t const outputImage, dwImageHandle_t const inputImage, dwRect const inputROI, dwImageTransformationHandle_t const obj);

/**
 * Resizes the input image and copies the result into the previously allocated output image, of the same type (CUDA or NvMedia) and format (any)
 * @param[out] outputImage Pointer to the output image.
 * @param[in] inputImage Pointer to the input image.
 * @param[in] obj Handle to the Image Transformation engine
 *
 * @return DW_INVALID_ARGUMENT if the images are invalid <br>
 * DW_SUCCESS if the operation is successful <br>
 * DW_CUDA_ERROR if the underlying cuda operation failed <br>
 * DW_NVMEDIA_ERROR if the underlying NvMedia operation failed
 */
DW_API_PUBLIC
dwStatus dwImageTransformation_copyFullImage(dwImageHandle_t const outputImage, dwImageHandle_t const inputImage, dwImageTransformationHandle_t const obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_IMAGEPROCESSING_H_

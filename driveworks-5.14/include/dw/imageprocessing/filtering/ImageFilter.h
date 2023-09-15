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
// SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: ImageFilter</b>
 *
 * @b Description: This file defines image filter.
 */

/**
 * @defgroup imagefilter_imagefilter_group Image Filter Interface
 *
 * @brief Defines the image filter structure.
 *
 * @{
 */

#ifndef DW_IMAGEPROCESSING_IMAGEFILTER_H_
#define DW_IMAGEPROCESSING_IMAGEFILTER_H_

#include <dw/core/base/Config.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>
#include <dw/image/Image.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwImageFilterObject* dwImageFilterHandle_t;

typedef enum dwImageFilterType {
    DW_IMAGEFILTER_TYPE_UNKNOWN = 0,

    /// Zero order recursive gaussian filter
    /// @note: Supporting only images with 1 or 4 channels of uint8 pixel type.
    DW_IMAGEFILTER_TYPE_RECURSIVE_GAUSSIAN_FILTER = 1,

    /// Box filter
    /// @note: Supporting only single channel uint8, float16, float32 image pixel types
    DW_IMAGEFILTER_TYPE_BOX_FILTER = 2,

    /// Convolution Filter
    /// @note: PVA based convolution filter supports only DW_TYPE_UINT(8/16) and DW_TYPE_INT(8/16).
    ///        GPU based one supports in addition DW_TYPE_FLOAT(16/32) types.
    DW_IMAGEFILTER_TYPE_CONVOLUTION_FILTER = 3,
} dwImageFilterType;

typedef struct dwImageFilterConfig
{
    /** Filtering algorithm defined by 'dwImageFilterType' */
    dwImageFilterType filterType;

    /** Processorr type
    *  set to DW_PROCESSOR_TYPE_PVA_0 or DW_PROCESSOR_TYPE_PVA_1 to run algorithm on PVA.
    *  set to DW_PROCESSOR_TYPE_GPU to run algorithm on GPU.
    */
    dwProcessorType processorType;

    /** Pixel type of the images that the ImageFilter runs on.
    * To be set to corresponding data type of input&output image, they both must match.
    */
    dwTrivialDataType pxlType;

    /** Width of the images that the ImageFilter runs on. */
    uint32_t imageWidth;

    /** Height of the images that the ImageFilter runs on. */
    uint32_t imageHeight;

    /** Width of filter kernel */
    uint32_t kernelWidth;

    /** Height of filter kernel */
    uint32_t kernelHeight;

    /** The order of GAUSSIAN filter*/
    uint32_t order;

    /** The sigma of kernel for GAUSSIAN filter*/
    float32_t sigma;

    /** Filter kernel data defined by user.
    * The kernel will be copied when intializing ImageFilter.
    * If kernelLength = kernelWidth * kernelHeight, then 2D convolution filter will be applied.
    * If kernelLength = kernelWidth + kernelHeight, then 1D separable convolution filter will be applied.
    * It is assumed that in case of separable filter first kernelWidth values defines horizontal filter,
    * and last kernelHeight entries the vertical filter.
    */
    const float32_t* kernel;

    /** Indicates the kernel data lenght defined by user */
    uint32_t kernelLength;
} dwImageFilterConfig;

/**
 * Allocates resources for an image filter of a certain type, expecting an image of a specific size.
 * @param[out] filter Pointer to an image filter handle
 * @param[in] config Specifies the configuration parameters
 * @param[in] context Specifies the handle to the active DW context
 *
 * @return DW_INVALID_ARGUMENT if filter or context are NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwImageFilter_initialize(dwImageFilterHandle_t* filter, const dwImageFilterConfig* config, dwContextHandle_t context);

/**
 * Sets the cuda stream
 * @param[in] stream Specifies the CUDA stream to use during image filtering.
 * @param[in] filter Pointer to an image filter handle
 *
 * @return DW_INVALID_ARGUMENT if filter is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwImageFilter_setCUDAStream(cudaStream_t stream, dwImageFilterHandle_t filter);

/**
 * Gets the cuda stream used by the APIs of ImageFilter
 * @param[out] stream CUDA stream
 * @param[in] filter Handle to the ImageFilter handle
 *
 * @return DW_INVALID_ARGUMENT if image filter handle or stream are NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwImageFilter_getCUDAStream(cudaStream_t* stream, dwImageFilterHandle_t filter);

/**
 * Applies the set filter to the input image.
 * @param[out] outputImage Output image of DW_IMAGE_CUDA type. Only single channel is currently supported
 * @param[in] inputImage The input image, must match the output image
 * @param[in] filter Pointer to an image filter handle
 *
 * @return DW_INVALID_ARGUMENT if filter or in/out put image are NULL.<br>
 *         DW_CUDA_ERROR if the underlying cuda operation fails.<br>
 *         DW_NVMEDIA_ERROR if the underlying VPI operation fails.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwImageFilter_applyFilter(dwImageHandle_t outputImage, dwImageHandle_t inputImage, dwImageFilterHandle_t filter);

/**
 * Releases the image filter
 * @param[in] filter Pointer to an image filter handle
 *
 * @return DW_INVALID_ARGUMENT if filter is NULL.<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwImageFilter_release(dwImageFilterHandle_t filter);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_FEATURES_FEATURES_H_

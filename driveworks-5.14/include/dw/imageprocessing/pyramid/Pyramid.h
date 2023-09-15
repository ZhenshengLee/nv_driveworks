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
 * <b>NVIDIA DriveWorks API: Pyramid</b>
 *
 * @b Description: This file defines image pyramids.
 */

/**
 * @defgroup imagefilter_pyramid_group Pyramid Interface
 *
 * @brief Defines the image pyramid structure.
 *
 * @{
 */

#ifndef DW_IMAGEPROCESSING_PYRAMID_H_
#define DW_IMAGEPROCESSING_PYRAMID_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>
#include <dw/image/Image.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DW_PYRAMID_LEVEL_MAX_COUNT 12

/// Pyramid image structure
typedef struct dwPyramidImage
{
    /// number of level images in pyramid
    uint32_t levelCount;

    /// level image data
    /// levelImages[0] is the level 0 image which has the highest resolution
    /// levelImages[N] is the level N image, it is half the size of level N-1 image
    dwImageHandle_t levelImages[DW_PYRAMID_LEVEL_MAX_COUNT];
} dwPyramidImage;

/// Pyramid image properties structure
typedef struct dwPyramidImageProperties
{
    /// number of level images in pyramid
    uint32_t levelCount;

    /// level image properties
    /// levelProps[0] refers to the level 0 image which has the highest resolution
    /// levelProps[N] refers to the level N image, it is half the size of level N-1 image
    dwImageProperties levelProps[DW_PYRAMID_LEVEL_MAX_COUNT];
} dwPyramidImageProperties;

/**
 * Fills the pyramid image properties structure
 * @param[out] props A pointer to properties structure to be filled.
 * @param[in] levelCount Number of levels in the pyramid
 * @param[in] width Width of level 0 image
 * @param[in] height Height of level 0 image
 * @param[in] pxlType Data Type of pyramid, currently only 'DW_TYPE_UINT8', 'DW_TYPE_UINT16',
 *            'DW_TYPE_FLOAT16' and 'DW_TYPE_FLOAT32' are supported
 * @return DW_INVALID_ARGUMENT if props is NULL
 *         DW_INVALID_ARGUMENT if pxlType is unsupported
 *         DW_INVALID_ARGUMENT if levelCount > DW_PYRAMID_LEVEL_MAX_COUNT
 *         DW_SUCCESS otherwise
 *
 * @note Level 0 image has the highest resolution in all pyramid level images,
 * level 0 image size should be the same as input image in 'dwImageFilter_buildPyramid'
 */
DW_API_PUBLIC
dwStatus dwPyramid_fillProperties(dwPyramidImageProperties* props, uint32_t levelCount,
                                  uint32_t width, uint32_t height, dwTrivialDataType pxlType);

/**
 * Gets the properties of a pyramid image
 * @param[out] props A pointer to properties structure to be filled.
 * @param[in] pyramid A pointer to the pyramid image
 * @param[in] context Specifies the handle to the context under which it is created.
 * @return DW_INVALID_ARGUMENT if props or pyramid or context are NULL
 *         DW_SUCCESS otherwise
 *
 * @note Level 0 image has the highest resolution in all pyramid level images,
 * level 0 image size should be the same as input image in 'dwImageFilter_buildPyramid'
 */
DW_API_PUBLIC
dwStatus dwPyramid_getProperties(dwPyramidImageProperties* props, dwPyramidImage* pyramid,
                                 dwContextHandle_t context);

/**
 * Creates and initializes an image pyramid
 * @param[out] pyramid A pointer to the pyramid image will be returned here.
 * @param[in] levelCount Number of levels in the pyramid
 * @param[in] width Width of level 0 image
 * @param[in] height Height of level 0 image
 * @param[in] pxlType Data Type of pyramid, currently only 'DW_TYPE_UINT8', 'DW_TYPE_UINT16',
 *            'DW_TYPE_FLOAT16' and 'DW_TYPE_FLOAT32' are supported
 * @param[in] context Specifies the handle to the context under which it is created.
 * @return DW_INVALID_ARGUMENT if pyramid or context are NULL<br>
 *         DW_INVALID_ARGUMENT if pxlType is unsupported<br>
 *         DW_INVALID_ARGUMENT if levelCount > DW_PYRAMID_LEVEL_MAX_COUNT<br>
 *         DW_SUCCESS otherwise<br>
 *
 * @note Level 0 image has the highest resolution in all pyramid level images,
 * level 0 image size should be the same as input image in 'dwImageFilter_buildPyramid'
 */
DW_API_PUBLIC
dwStatus dwPyramid_create(dwPyramidImage* pyramid, uint32_t levelCount,
                          uint32_t width, uint32_t height, dwTrivialDataType pxlType,
                          dwContextHandle_t context);

/**
 * Creates and initializes an image pyramid
 * @param[out] pyramid A pointer to the pyramid image will be returned here.
 * @param[in] props Properties of the pyramid image
 * @param[in] context Specifies the handle to the context under which it is created.
 * @return DW_INVALID_ARGUMENT if pyramid or context are NULL
 *         DW_INVALID_ARGUMENT if pxlType is unsupported
 *         DW_INVALID_ARGUMENT if levelCount > DW_PYRAMID_LEVEL_MAX_COUNT
 *         DW_SUCCESS otherwise
 *
 */
DW_API_PUBLIC
dwStatus dwPyramid_createFromProperties(dwPyramidImage* pyramid, const dwPyramidImageProperties* props,
                                        dwContextHandle_t context);

/**
 * Destroy pyramid images
 * @param[in] pyramid pyramid image to be destroyed
 * @return DW_INVALID_ARGUMENT if level image in pyramid contains invalid data<br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwPyramid_destroy(dwPyramidImage pyramid);

/**
 * Builds the pyramid from level 0 image.
 * @param[inout] pyramid pyramid images that will be built, must be initialized by 'dwPyramid_create' first
 * @param[in] image Specifies the level 0 image which has the highest resolution
 * @param[in] stream Specifies the CUDA stream to use during pyramid building.
 * @param[in] context Specifies the handle to the active DW context
 *
 * @return DW_INVALID_ARGUMENT if pyramid, image or context are NULL<br>
 *         DW_INVALID_ARGUMENT if width/height of input 'image' differs from the one in 'dwPyramid_create'<br>
 *         DW_INVALID_ARGUMENT if memory layout of input 'input' is not DW_IMAGE_MEMORY_TYPE_PITCH
 *         DW_BAD_CAST if level image in pyramid has invalid types<br>
 *         DW_SUCCESS otherwise<br>
 *
 * @note Before computing pyramid, 'dwPyramidImage' must be initialized by 'dwPyramid_create',
 *       input 'image' must have the same width/height as in 'dwPyramid_create'
 */
DW_API_PUBLIC
dwStatus dwImageFilter_computePyramid(dwPyramidImage* pyramid, const dwImageCUDA* image,
                                      cudaStream_t stream, dwContextHandle_t context);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_IMAGEPROCESSING_PYRAMID_H_

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
// SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks GL API: Image Conversion and Streaming Functionality</b>
 *
 * @b Description: This file defines methods for image conversion.
 */

/**
 * @defgroup gl_image_group GL Image Interface
 * @ingroup gl_group
 *
 * @brief Defines GL image abstractions, and streamer and format conversion APIs.
 *
 * @{
 */

#ifndef DWGL_IMAGE_IMAGE_H_
#define DWGL_IMAGE_IMAGE_H_

#include <dw/image/Image.h>

#include <dwvisualization/core/Exports.h>
#include <dwvisualization/gl/GL.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Defines a GL texture image.
typedef struct dwImageGL
{
    /// Specifies the properties of the image.
    /// @note prop.type is always DW_IMAGE_GL and would have to be casted accordingly
    dwImageProperties prop;
    /// Specifies the OpenGL texture handle.
    GLuint tex;
    /// Specifies the OpenGL texture target.
    GLenum target;
    /// Specifies the time, in microseconds, when the image was acquired.
    dwTime_t timestamp_us;
} dwImageGL;

/**
 * Creates and allocates resources for a dwImageHandle_t based on the properties passed as input.
 *
 * @param[out] image A handle to the image
 * @param[in] properties The image properties.
 * @param[in] ctx The DriveWorksGL context.
 *
 * @return DW_SUCCESS if the image was created, <br>
 *         DW_INVALID_ARGUMENT if the given image types are invalid or the streamer pointer is null, <br>
 *         DW_INVALID_HANDLE if the given context handle is invalid, <br>
 *
 * @note This method can only create GL images and will ignore any type set in the properties.
 */
DW_VIZ_API_PUBLIC
dwStatus dwImage_createGL(dwImageHandle_t* image,
                          dwImageProperties properties,
                          dwContextHandle_t ctx);

/**
 * Works only with DW_IMAGE_GL and DW_IMAGE_LAYOUT_BLOCK (or DEFAULT) memory layout, in which case the buffer passed is a cpu
 * memory pointer and the content is uploaded onto the GPU as a GL texture. Given that this implies a change
 * in domain and device, the ownership of the original buffer remains to the user and the image
 * allocates its own memory onto the GPU.
 *
 * @param[out] image A handle to the image
 * @param[in] properties The image properties.
 * @param[in] buffersIn An array of pointers to individual buffers.
 * @param[in] pitches An array of pitches for each buffer.
 * @param[in] bufferCount The number of buffers (maximum is DW_MAX_IMAGE_PLANES).
 * @param[in] ctx The DriveWorksGL context.
 *
 * @return DW_SUCCESS if the image was created, <br>
 *         DW_INVALID_ARGUMENT if the given image properties are invalid or the image pointer is null, <br>
 *         DW_INVALID_HANDLE if the given context handle is invalid, <br>
 */
DW_VIZ_API_PUBLIC
dwStatus dwImage_createAndBindBufferGL(dwImageHandle_t* image,
                                       dwImageProperties properties,
                                       void* buffersIn[DW_MAX_IMAGE_PLANES],
                                       size_t pitches[DW_MAX_IMAGE_PLANES], size_t bufferCount,
                                       dwContextHandle_t ctx);

/**
 * Creates a dwImageHandle_t based on the properties passed and binds a GL texture to it. Valid only for types
 * of DW_IMAGE_GL with DW_IMAGE_LAYOUT_BLOCK layout
 *
 * @param[out] image A handle to the image
 * @param[in] properties The image properties.
 * @param[in] texID The id of the gl texture
 * @param[in] target The number of buffers (maximum is DW_MAX_IMAGE_PLANES).
 *
 * @return DW_SUCCESS if the image was created, <br>
 *         DW_INVALID_ARGUMENT if the given image types is invalid, <br>
 */
DW_VIZ_API_PUBLIC
dwStatus dwImage_createAndBindTexture(dwImageHandle_t* image, dwImageProperties properties, GLenum texID,
                                      GLenum target);

/**
 * Retrieves the dwImageGL of a dwImageHandle_t. The image must have been created as a DW_IMAGE_GL type
 * Note that any modification to the image retrieved will modify the content of the original handle
 *
 * @param[out] imageGL A pointer to the dwImageGL pointer
 * @param[in] image A handle to the image
 *
 * @return DW_SUCCESS if the dwImageGL is successfully retrieved, <br>
 *         DW_INVALID_ARGUMENT if the given image pointer or image handle is null, <br>
 *         DW_INVALID_HANDLE if the given image handle is invalid, <br>
 */
DW_VIZ_API_PUBLIC
dwStatus dwImage_getGL(dwImageGL** imageGL, dwImageHandle_t image);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DWGL_IMAGE_IMAGE_H_

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
// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Video Encoder Methods</b>
 *
 * @b Description: This file defines encoder methods.
 */

/**
 * @defgroup encoder Encoder
 *
 * @brief Defines the video encoder types.
 * @{
 */
#ifndef DW_SENSORS_CODECS_VIDEO_ENCODER_H
#define DW_SENSORS_CODECS_VIDEO_ENCODER_H

#include <dw/core/base/Types.h>
#include <dw/image/Image.h>
#include <dw/sensors/codecs/Encoder.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Handle representing a video encoder
typedef struct dwCodecEncoderVideoObject* dwEncoderVideoHandle_t;

/**
 * Append the image allocation attributes required for images to be compatible with
 * this Encoder instance to the provided dwImageProperties. All images passed to this
 * Encoder instance *must* be created with the allocation attributes returned by this
 * function.
 *
 * @param[inout] imgProps Image properties
 * @param[in] encoder  Handle representing a encoder.
 * @note The given imgProps should be compatible with that returned by
 *       dwSensorCamra_getImageProperties API.
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
 * @return DW_FAILURE - if underlying imageAttributes for Encoder is not initialized. <br>
 *         DW_SUCCESS
 *
 */
dwStatus dwEncoderVideo_appendAllocationAttributes(dwImageProperties* const imgProps, dwEncoderVideoHandle_t encoder);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_SENSORS_CODECS_VIDEO_ENCODER_H

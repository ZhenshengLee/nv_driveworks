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
// SPDX-FileCopyrightText: Copyright (c) 2015-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_IMAGEPROCESSING_COMMON_H_
#define DW_IMAGEPROCESSING_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif
/// interpolation mode
typedef enum dwImageProcessingInterpolation {
    /// simplest form of interpolation
    DW_IMAGEPROCESSING_INTERPOLATION_DEFAULT,
    /// bilinear interpolation
    DW_IMAGEPROCESSING_INTERPOLATION_LINEAR
} dwImageProcessingInterpolation;

/// border mode (valid for DW_IMAGE_CUDA types)
typedef enum dwImageProcessingBorderMode {
    DW_IMAGEPROCESSING_BORDER_MODE_ZERO   = 0,
    DW_IMAGEPROCESSING_BORDER_MODE_MIRROR = 1,
    DW_IMAGEPROCESSING_BORDER_MODE_REPEAT = 2,
    DW_IMAGEPROCESSING_BORDER_MODE_WRAP   = 3,
} dwImageProcessingBorderMode;
#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_IMAGETRANSFORMATION_H_

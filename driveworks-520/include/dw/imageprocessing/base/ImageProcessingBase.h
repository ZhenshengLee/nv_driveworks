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

#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>

#ifndef DW_IMAGEPROCESSING_BASE_H_
#define DW_IMAGEPROCESSING_BASE_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Call CUDA initialization sequence for dw_imageprocessing lib
 * @note: This API should be called before any other APIs in dw_imageprocessing,
 * that call CUDA kernels in safety platform including dwFeature2DTracker, 
 * dwFeature2DDetector, dwImageFilter, etc., elsewise it will return
 * cudaErrorNotPermitted at the first time calling a CUDA kernel
 * 
 * @retval DW_CUDA_ERROR when CUDA initialization fails
 * @retval DW_SUCCESS when operation succeeded
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwImageProcessing_initCUDA();

#ifdef __cplusplus
}
#endif

#endif // DW_IMAGEPROCESSING_BASE_H_
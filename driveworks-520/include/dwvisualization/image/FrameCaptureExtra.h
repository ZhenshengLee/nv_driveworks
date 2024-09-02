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
 * <b>NVIDIA DriveWorks API: Experimental FrameCapture Methods</b>
 *
 * @b Description: This file defines extra Frame Capture methods.
 */

/**
 * @defgroup framecapture_group Frame Capture Interface
 * @ingroup gl_group
 *
 * @brief Defines FrameCapture module for performing capture of currently bound GL frame buffer.
 * @{
 *
 */

#ifndef DWGL_FRAMECAPTUREEXTRA_H_
#define DWGL_FRAMECAPTUREEXTRA_H_

#include "FrameCapture.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Starts frame capture. This method creates a new thread and
 * begins the serialization loop.
 *
 * @param[in] framecapture Specifies the frame capture handle.
 *
 * @return DW_NOT_AVAILABLE - serialization is not available at this moment. <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_SUCCESS
 *
 */
DW_VIZ_API_PUBLIC
dwStatus dwFrameCapture_start(dwFrameCaptureHandle_t framecapture);

/**
 * Stops frame capture. This method stops the thread and
 * the serialization loop.
 *
 * @param[in] framecapture Specifies the sensor serializer handle.
 *
 * @return DW_NOT_AVAILABLE - serialization is not available at this moment. <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_SUCCESS
 *
 */
DW_VIZ_API_PUBLIC
dwStatus dwFrameCapture_stop(dwFrameCaptureHandle_t framecapture);

/**
 * Asynchronously append a dwImageHandle frame to the capture and it's serialized.
 *
 * @param[in] img dwImageHandle to serialize
 * @param[in] framecapture Handle to the frame capture module being released.
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid <br/>
 *         DW_SUCCESS<br/>
 */
DW_VIZ_API_PUBLIC
dwStatus dwFrameCapture_appendFrameAsync(const dwImageHandle_t img, dwFrameCaptureHandle_t framecapture);

/**
 * Asynchronously append a dwImageCUDA frame to the capture and it's serialized.
 *
 * @param[in] img dwImageCUDA to serialize
 * @param[in] framecapture Handle to the frame capture module being released.
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid <br/>
 *         DW_SUCCESS<br/>
 */
DW_VIZ_API_PUBLIC
DW_DEPRECATED("dwFrameCapture_appendFrameCUDAAsync: this function is deprecated, replace with the dwFrameCapture_appendFrameAsync")
dwStatus dwFrameCapture_appendFrameCUDAAsync(const dwImageCUDA* img, dwFrameCaptureHandle_t framecapture);

#ifndef VIBRANTE
/**
 * Asynchronously append a dwImageCPU frame to the capture and it's serialized.
 *
 * @param[in] img dwImageCPU to serialize
 * @param[in] framecapture Handle to the frame capture module being released.
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid <br/>
 *         DW_SUCCESS<br/>
 */
DW_VIZ_API_PUBLIC
dwStatus dwFrameCapture_appendFrameCPUAsync(const dwImageCPU* img, dwFrameCaptureHandle_t framecapture);
#else
/**
 * Asynchronously append a dwImageNvMedia frame to the capture and it's serialized.
 *
 * @param[in] img dwImageNvMedia to serialize
 * @param[in] framecapture Handle to the frame capture module being released.
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid <br/>
 *         DW_SUCCESS<br/>
 */
DW_VIZ_API_PUBLIC
dwStatus dwFrameCapture_appendFrameNvMediaAsync(const dwImageNvMedia* img, dwFrameCaptureHandle_t framecapture);
#endif
/**
 * Asynchronously append a dwImageGL frame to the capture and it's serialized.
 *
 * @param[in] img dwImageGL to serialize
 * @param[in] framecapture Handle to the frame capture module being released.
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid <br/>
 *         DW_SUCCESS<br/>
 */
DW_VIZ_API_PUBLIC
#if VIBRANTE_PDK_DECIMAL >= 6000400
DW_DEPRECATED("dwFrameCapture_appendFrameGLAsync: this function is deprecated, replace with the dwFrameCapture_appendFrameAsync")
#endif
dwStatus dwFrameCapture_appendFrameGLAsync(const dwImageGL* img, dwFrameCaptureHandle_t framecapture);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DWGL_FRAMECAPTUREEXTRA_H_

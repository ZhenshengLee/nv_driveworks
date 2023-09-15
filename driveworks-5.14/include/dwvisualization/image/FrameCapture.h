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
// SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: FrameCapture Methods</b>
 *
 * @b Description: This file defines Frame Capture methods.
 */

/**
 * @defgroup framecapture_group Frame Capture Interface
 * @ingroup gl_group
 *
 * @brief Defines FrameCapture module for performing capture of currently bound GL frame buffer.
 *
 * @{
 */

#ifndef DW_FRAMECAPTURE_H_
#define DW_FRAMECAPTURE_H_

#include <dw/sensors/SensorSerializer.h>

#include "Image.h"

#include <nvscisync.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Handle to a Frame Capture module object.
 */
typedef struct dwFrameCaptureObject* dwFrameCaptureHandle_t;

/// Enables FrameCapture functionalities. These can be set as a logical OR between the two modes. Choosing a mode means that the resources needed by that feature will be allocated
typedef enum dwFrameCaptureMode {
    /// Screencapture mode allocates GL resources that enable the capture of the current rendering Window via dwFrameCapture_screenCapture. Will not work if no GL context exists.
    DW_FRAMECAPTURE_MODE_SCREENCAP = 1 << 0,
    /// Serialize enables the creation of the serializer which allows the calls to dwFrameCapture_appendFrameX. The specific resources will depend on the properties passed to the FrameCapture but they will be at least the dwSensorSerializer and usually one ImageStreamer
    DW_FRAMECAPTURE_MODE_SERIALIZE = 1 << 1,
} dwFrameCaptureMode;

/**
 * @brief Initialization parameters for the frame capture module.
 */
typedef struct
{
    //! Width of the GL window / image
    uint32_t width;

    //! Height of the GL window / image
    uint32_t height;

    //! The frameCaptured image is returned as a dwImageGL. By default (false) the resources for the dwImageGL are allocated
    //! automatically and the image received through the call dwFrameCapture_screenCapture. If true, then no resource
    //! will be allocated and the dwImageGL will have to be allocated by with dwImage_createGL()/dwImage_getGL and
    //! the frameCapture will return the image through dwFrameCapture_screenCaptureCustom()
    bool captureCustom;

    //! Logic OR of the dwFrameCaptureMode. Can be either or all. Useful for not allocating useless resources
    uint32_t mode;

    //! Boolean, if true it serializes DW_IMAGE_GL only, otherwise all other serializations (CUDA/CPU on X86, CUDA/NVMEDIA on Drive)
    bool serializeGL;

    //! SensorSerializer parameters, see SensorSerializer.h
    dwSerializerParams params;

    //! Boolean, if it is true and
    //! MUST use dwFrameCapture_appendAllocationAttributes to append nvsciBufAttrList to image's nvsciBufAttrLists,
    //! this image must be the output of camera with isp-mode=yuv420-bl
    //! then dwFrameCapture_appendFrame will append it without copyconvert overhead
    bool setupForDirectCameraOutput;
} dwFrameCaptureParams;

/**
 * Create a new frame capture module.
 *
 * @param[out] obj Handle to the frame capture module being initialized.
 * @param[in] params Frame capture initialization parameters.
 * @param[in] sal Handle to current SAL interface
 * @param[in] ctx Handle to the current driveworks context.
 *
 * @return DW_INVALID_ARGUMENT - if given arguments are invalid <br/>
 *         DW_SUCCESS<br/>
 *
 * @note SAL may internally allocate memory during initialization and will
 * be freed when the SAL is released.
 */
DW_VIZ_API_PUBLIC
dwStatus dwFrameCapture_initialize(dwFrameCaptureHandle_t* obj,
                                   const dwFrameCaptureParams* params,
                                   dwSALHandle_t sal,
                                   dwContextHandle_t ctx);

/**
 * Releases the frame capture module.
 *
 * @param[in] framecapture Handle to the frame capture module being released.
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid <br/>
 *         DW_SUCCESS<br/>
 *
 * @note To clean up memory owned by SAL during initialize, SAL needs to be
 * released as well. Please see dwFrameCapture_initialize for details.
 */
DW_VIZ_API_PUBLIC
dwStatus dwFrameCapture_release(dwFrameCaptureHandle_t framecapture);

/**
 * It grabs what is currently rendered on the current frame buffer and returns a dwImageGL out of it
 * For example, it permits to serialize additional information, such as bounding boxes, labels, etc., that are
 * rendered on top of the current GL frame. It is independent from the original source of the GL frame,
 * i.e. video or camera, and platform, i.e. Linux or Drive platforms.
 *
 * @param[out] imageGL a pointer to a dwImageGL pointer containing the captured window
 * @param[in] roi Region of interest to be captured
 * @param[in] framecapture Handle to the frame capture module being released.
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid <br/>
 *         DW_SUCCESS<br/>
 */
DW_VIZ_API_PUBLIC
dwStatus dwFrameCapture_screenCapture(const dwImageGL** imageGL, const dwRect roi,
                                      dwFrameCaptureHandle_t framecapture);

/**
 * It grabs what is currently rendered on the current frame buffer and blits onto the input dwImageGL
 *
 * @param[out] imageGL a pointer to a previously allocated dwImageGL
 * @param[in] roi Region of interest to be captured
 * @param[in] framecapture Handle to the frame capture module being released.
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid <br/>
 *         DW_SUCCESS<br/>
 */
DW_VIZ_API_PUBLIC
dwStatus dwFrameCapture_screenCaptureCustom(dwImageGL* imageGL, const dwRect roi,
                                            dwFrameCaptureHandle_t framecapture);

/**
 * Append a dwImageHandle frame to the capture and it's serialized.
 *
 * @param[in] img dwImageHandle to serialize
 * @param[in] framecapture Handle to the frame capture module being released.
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid <br/>
 *         DW_SUCCESS<br/>
 */
DW_VIZ_API_PUBLIC
dwStatus dwFrameCapture_appendFrame(const dwImageHandle_t img, dwFrameCaptureHandle_t framecapture);

/**
 * Append a dwImageCUDA frame to the capture and it's serialized.
 *
 * @param[in] img dwImageCUDA to serialize
 * @param[in] framecapture Handle to the frame capture module being released.
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid <br/>
 *         DW_SUCCESS<br/>
 */
DW_VIZ_API_PUBLIC
DW_DEPRECATED("dwFrameCapture_appendFrameCUDA: this function is deprecated, replace with the dwFrameCapture_appendFrame")
dwStatus dwFrameCapture_appendFrameCUDA(const dwImageCUDA* img, dwFrameCaptureHandle_t framecapture);

#ifndef VIBRANTE
/**
 * Append a dwImageCPU frame to the capture and it's serialized.
 *
 * @param[in] img dwImageCPU to serialize
 * @param[in] framecapture Handle to the frame capture module being released.
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid <br/>
 *         DW_SUCCESS<br/>
 */
DW_VIZ_API_PUBLIC
dwStatus dwFrameCapture_appendFrameCPU(const dwImageCPU* img, dwFrameCaptureHandle_t framecapture);
#else
/**
 * Append a dwImageNvMedia frame to the capture and it's serialized.
 *
 * @param[in] img dwImageNvMedia to serialize
 * @param[in] framecapture Handle to the frame capture module being released.
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid <br/>
 *         DW_SUCCESS<br/>
 */
DW_VIZ_API_PUBLIC
dwStatus dwFrameCapture_appendFrameNvMedia(const dwImageNvMedia* img, dwFrameCaptureHandle_t framecapture);
/**
 * Fill the sync attributes for the encoder to signal EOF fences. Note that multiple calls on the same syncAttrList will append attributes on the same
 *
 * @param[out] syncAttrList The sync attributes list to be filled
 * @param[in] framecapture The frame capture handle
 **/
DW_VIZ_API_PUBLIC
dwStatus dwFrameCapture_fillSyncAttributes(NvSciSyncAttrList syncAttrList, dwFrameCaptureHandle_t framecapture);

/**
 * Set the sync obj to which the encoder will wait on SOF fences. The sync object is not reference counted
 *
 * @param[in] syncObj The sync object
 * @param[in] framecapture The frame capture handle
 **/
DW_VIZ_API_PUBLIC
dwStatus dwFrameCapture_setSyncObject(NvSciSyncObj syncObj, dwFrameCaptureHandle_t framecapture);

/**
 * Insert fence to wait on completed operations
 *
 * @param[out] syncFence The sync fence of the frame
 * @param[in] framecapture The frame capture handle

 **/
DW_API_PUBLIC
dwStatus dwFrameCapture_insertFence(NvSciSyncFence* syncFence, dwFrameCaptureHandle_t framecapture);

#endif

/**
 * Append a dwImageGL frame to the capture and it's serialized.
 *
 * @param[in] img dwImageGL to serialize
 * @param[in] framecapture Handle to the frame capture module being released.
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid <br/>
 *         DW_SUCCESS<br/>
 */
DW_VIZ_API_PUBLIC
dwStatus dwFrameCapture_appendFrameGL(const dwImageGL* img, dwFrameCaptureHandle_t framecapture);

/**
 * Append the allocation attribute to create images that work with other NvSci based modules
 *
 * @param[inout] imgProps Image properties
 * @param[in] framecapture FrameCapture handle.
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
 *         DW_NOT_IMPLEMENTED - if the method for this image type is not implemented by given camera. <br>
 *         DW_SUCCESS
 *
 */
DW_API_PUBLIC
dwStatus dwFrameCapture_appendAllocationAttributes(dwImageProperties* imgProps,
                                                   dwFrameCaptureHandle_t framecapture);
#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_FRAMECAPTURE_H_

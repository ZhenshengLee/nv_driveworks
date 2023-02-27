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
// SPDX-FileCopyrightText: Copyright (c) 2016-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Color Correct Methods</b>
 *
 * @b Description: This file defines Color Correct methods.
 */

/**
 * @defgroup color_correct Color Correction Interface
 *
 * @brief Module providing color correction of the camera view.
 *
 * @{
 */

#ifndef DW_IMAGEPROCESSING_COLOR_CORRECTION_H_
#define DW_IMAGEPROCESSING_COLOR_CORRECTION_H_

#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>
#include <dw/core/base/Status.h>
#include <dw/core/context/Context.h>
#include <dw/image/Image.h>
#include <dw/rig/Rig.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Handles representing the Color Correction interface.
 */
typedef struct dwColorCorrectObject* dwColorCorrectHandle_t;

/**
 * Configuration parameters of the color correction module.
 **/
typedef struct dwColorCorrectParameters
{
    //! Width of the images from camera to be corrected. Must be even.
    uint32_t cameraWidth;

    //! Height of the images from camera to be corrected. Must be even.
    uint32_t cameraHeight;

    //! Width of the internally constructed projection image to perform color correction.
    //! If 0 specified, default of 256 is used.
    uint32_t projectionWidth;

    //! Height of the internally constructed projection image to perform color correction.
    //! If 0 specified, default of 128 is used.
    uint32_t projectionHeight;

} dwColorCorrectParameters;

/**
 * Creates and initializes the color correction module using dwRig.
 * All camera relationships are extracted from the provided rig. Color correction is performed by
 * reprojecting all cameras onto a common plane (i.e., groundplane) and then by extracting color information
 * based on the overlapping regions.
 *
 * @param[out] obj The module handle is returned here.
 * @param[in] parameters Configuration parameters of the camera system to setup correction algorithms.
 * @param[in] rigConfig An opening dwRig handle.
 * @param[in] ctx Handle to the context under which it is created.
 *
 * @return DW_INVALID_HANDLE - if provided color rigconfig handle is invalid, i.e null or of wrong type <br>
 *         DW_NOT_AVAILABLE - if creation of color correction handle failed. <br>
 *         DW_INVALID_ARGUMENT - if color correct handle or parameters are NULL.<br>
 *         DW_SUCCESS - color correct handle is created successfully
 *
 * @note cameraWidth/cameraHeight must have an even value because the calculation is
 * based on YUV420 (U/V plane is downsampled).
 * \ingroup color_correct
 */
DW_API_PUBLIC
dwStatus dwColorCorrect_initializeFromRig(dwColorCorrectHandle_t* obj,
                                          const dwColorCorrectParameters* parameters,
                                          dwRigHandle_t rigConfig,
                                          dwContextHandle_t ctx);

/**
 * Creates and initializes the color correction module using the existing reprojection matrix
 * that reprojects cameras onto a common plane (i.e., groundplane) and then extracts color information
 * based on the overlapping regions.
 *
 * @param[out] obj The module handle is returned here.
 * @param[in] ctx Handle to the context under which it is created.
 * @param[in] cameraCount Number of cameras to correct.
 * @param[in] pProjToGroundMap An array of pixel maps mapping from ground plane into each camera's image plane. <br>
              The given float array contains cameraCount*projectionWidth*projectionHeight*2 float numbers. <br>
              Start address of the k-th camera's data is camera[k] = pProjToGroundMap + k*projectWidth*projectHeight*2. <br>
              Each float pair at index (i,j) for a camera k represents a pixel coordinate in camera's image plane at position: <br>
              x = pProjToGroundMap[camera[k] + (j * projectionWidth + i) * 2], y = pProjToGroundMap[camera[k] + (j * projectionWidth + i) * 2 + 1] <br>
              Out(k, i, j) = CameraSpaceImage(k, x, y) is a ground plane projection image of the k-th camera.

 * @param[in] params Configuration parameters of the camera system to set up correction algorithms.
 *
 * @return DW_NOT_AVAILABLE - if creation of color correction handle failed. <br>
 *         DW_INVALID_ARGUMENT - if color correct handle is NULL.<br>
 *         DW_SUCCESS - color correct handle is created successfully
 *
 * @note cameraWidth/cameraHeight must have an even value because the calculation is
 *       based on YUV420 (U/V plane is downsampled) <br> *
 * \ingroup color_correct
 */
DW_API_PUBLIC
dwStatus dwColorCorrect_initializeFromProjectionMap(dwColorCorrectHandle_t* obj, dwContextHandle_t ctx,
                                                    const uint32_t cameraCount, const dwVector2f* pProjToGroundMap,
                                                    const dwColorCorrectParameters* params);
/**
 * This method releases all resources associated with a color_correct object.
 *
 * @param[in] obj The object handle to release.
 *
 * @return DW_SUCCESS - color correct handle is released successfully
 *
 * @note This method renders the handle unusable.
 * \ingroup color_correct
 */
DW_API_PUBLIC
dwStatus dwColorCorrect_release(dwColorCorrectHandle_t obj);

/**
 * This method resets all resources associated with a color_correct object.
 *
 * @param[in] obj The object handle to release.
 *
 * @return DW_INVALID_ARGUMENT - if provided color correct handle is NULL. <br>
 *         DW_SUCCESS - color correct handle is reset successfully
 * \ingroup color_correct
 */
DW_API_PUBLIC
dwStatus dwColorCorrect_reset(dwColorCorrectHandle_t obj);

/**
 * Sets the CUDA stream to run the required calculations of the color correction.
 *
 * @param[in] stream CUDA stream that is used during color correction.
 * @param[in] obj Handle to the color_correct class.
 *
 * @return DW_INVALID_ARGUMENT - if provided color correct handle is NULL. <br>
 *         DW_INVALID_HANDLE - if provided color correct handle is invalid, i.e null or of wrong type . <br>
 *         DW_SUCCESS - CUDA stream is set successfully
 * \ingroup color_correct
 */
DW_API_PUBLIC
dwStatus dwColorCorrect_setCUDAStream(cudaStream_t stream,
                                      dwColorCorrectHandle_t obj);

/**
 * Returns the CUDA stream on which the calculations of the color correction are executed.
 *
 * @param[out] stream Pointer to the CUDA stream to return.
 * @param[in] obj Handle to the color_correct class.
 *
 * @return DW_INVALID_ARGUMENT - if provided color correct handle or stream are NULL. <br>
 *         DW_INVALID_HANDLE - if provided color correct handle is invalid, i.e null or of wrong type . <br>
 *         DW_SUCCESS - CUDA stream is set successfully
 * \ingroup color_correct
 */
DW_API_PUBLIC
dwStatus dwColorCorrect_getCUDAStream(cudaStream_t* stream,
                                      dwColorCorrectHandle_t obj);

/**
 * This method adds reference view to color correction; the color of all the other views
 * are corrected based on this view. The reference image is not modified.
 *
 * @param[in] cameraIdx Index of the reference camera in the provided camera rig.
 * @param[in] referenceImage Image of the reference camera.
 * @param[in] obj Handle to the color_correct class.
 *
 * @return DW_INVALID_HANDLE - if provided color correct handle is invalid, i.e null or of wrong type . <br>
 *         DW_INVALID_ARGUMENT - if given color correct handle is NULL, reference
 *                               image is NULL, cameraIdx is invalid, the size
 *                               of the input image differs from the size passed
 *                               during initialization, reference image format is
 *                               not DW_IMAGE_MEMORY_TYPE_PITCH, reference image pixel
 *                               format is not DW_IMAGE_YUV420 or reference image
 *                               pixel type is not DW_TYPE_UINT8 or DW_TYPE_INT8. <br>
 *         DW_SUCCESS - reference camera view is set successfully
 * \ingroup color_correct
 */
DW_API_PUBLIC
dwStatus dwColorCorrect_setReferenceCameraView(const dwImageHandle_t referenceImage,
                                               uint32_t cameraIdx,
                                               dwColorCorrectHandle_t obj);

/**
 * Applies global color correction on the given image.
 * This function does correction based on statistic data
 * collected from the whole reprojected topview.
 *
 * @param[inout] image Image to correct. The color correction happens in-place.
 * @param[in]  curCameraIdx Camera index of the given image in the specified rig.
 * @param[in]  factor Blending weight, float number between [0.f, 1.f] <br>
 *             0.f means using the original image and no color correction <br>
 *             1.f means using full weight of reference color.
 * @param[in]  obj Handle to the color_correct class.
 * @return DW_INVALID_HANDLE - if provided color correct handle is invalid, i.e null or of wrong type . <br>
 *         DW_INVALID_ARGUMENT - if input image is NULL, color correct handle is
 *                               null, cameraIdx is invalid, the size
 *                               of the input image differs from the size passed
 *                               during initialization, reference image format is
 *                               not DW_IMAGE_MEMORY_TYPE_PITCH, reference image pixel
 *                               format is not DW_IMAGE_YUV420 or reference image
 *                               pixel type is not DW_TYPE_UINT8 or DW_TYPE_INT8. <br>
 *         DW_SUCCESS - Global color correction is applied successfully
 *
 * @note Input/Output image is YUV420p.
 *       Before calling this function, ensure that you already added the
 *       reference view by calling dwColorCorrect_addCameraView.
 * \ingroup color_correct
 */
DW_API_PUBLIC
dwStatus dwColorCorrect_correctByReferenceView(dwImageHandle_t image,
                                               uint32_t curCameraIdx,
                                               float32_t factor,
                                               dwColorCorrectHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_IMAGEPROCESSING_COLOR_CORRECTION_H_

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
// SPDX-FileCopyrightText: Copyright (c) 2015-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Rectifier Methods</b>
 *
 * @b Description: This file defines rectifier methods.
 */

/**
 * @defgroup rectifier_group Rectifier Interface
 *
 * @brief Defines the Rectifier module.
 *
 * @{
 */

#ifndef DW_IMAGEPROCESSING_GEOMETRY_RECTIFIER_H_
#define DW_IMAGEPROCESSING_GEOMETRY_RECTIFIER_H_

#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>
#include <dw/image/Image.h>
#include <dw/calibration/cameramodel/CameraModel.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A pointer to the handle representing a rectifier.
 * This object allows you to rectify an image acquired in one camera model by projecting it into another camera
 * model.
 */
typedef struct dwRectifierObject* dwRectifierHandle_t;

/**
 * Initializes a rectifier based on an input and output camera model and a homography. In particular, the rectifier
 * implements Ray2Pixel_cameraOut( H * Pixel2Ray_cameraIn(Image) ). Per default, the homography is set to identity.
 * This also implicitly sets up the VIC engine for rectificaton of dwImageNvMedia (note
 * the image must have the same resolution of the set camera and it must not be lower that 128x128)
 *
 * @param[out] obj A pointer to the rectifier handle for the created module.
 * @param[in] cameraIn Model of the input camera.
 * @param[in] cameraOut Model of the output camera.
 * @param[in] ctx Specifies the handler to the context to create the rectifier.
 *
 * @retval DW_INVALID_ARGUMENT  if the rectifier handle is NULL
 * @retval DW_INVALID_HANDLE  if the cameraIn model or cameraOut model or context handle is NULL or invalid
 * @retval DW_SUCCESS  on success
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwRectifier_initialize(dwRectifierHandle_t* obj,
                                dwCameraModelHandle_t cameraIn,
                                dwCameraModelHandle_t cameraOut,
                                dwContextHandle_t ctx);

/**
 * Resets the rectifier module.
 *
 * @param[in] obj Specifies the rectifier to reset.
 *
 * @retval DW_INVALID_HANDLE  if the given handle is invalid,i.e. null or of wrong type
 * @retval DW_BAD_CAST  if obj is not a rectifier handle
 * @retval DW_SUCCESS  on success
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwRectifier_reset(dwRectifierHandle_t obj);

/**
 * Releases the rectifier module.
 *
 * @param[in] obj The object handle to release.
 *
 * @retval DW_INVALID_HANDLE  if the given handle is invalid,i.e. null or of wrong type
 * @retval DW_BAD_CAST  if obj is not a rectifier handle
 * @retval DW_SUCCESS  on success
 *
 * @note This method renders the handle unusable.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwRectifier_release(dwRectifierHandle_t obj);

/**
 * Append the allocation attribute such that the images created of type DW_IMAGE_NVMEDIA can be fed to dwRectifier_apply()
 *
 * @param[inout] imgProps Image properties
 * @param[in] obj dwRectifier handle
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
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwRectifier_appendAllocationAttributes(dwImageProperties* const imgProps,
                                                dwRectifierHandle_t obj);

/**
* All (both input and output) images that the rectifier engine has to work with (NvMedia2D VIC mode only) must be placed in a dwImagePool and registered during initialization phase
* Note that all the images must have been created with properties with allocation attributes set 
*
* @param[in] imagePool Handle to the dwImagePool, ownership remains of the creator of the pool
 * @param[in] handle dwImageTransformation handle
*
* @return DW_NVMEDIA_ERROR - if underlying VIC engine had an NvMedia error. <br>
*         DW_INVALID_HANDLE - if given handle is not valid. <br>
*         DW_SUCCESS - if call is successful.
*
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: No
*/
DW_API_PUBLIC
dwStatus dwRectifier_registerImages(dwImagePool imagePool, dwRectifierHandle_t handle);

/**
 * Warps the image from the input camera model to the model of the output camera using either CUDA on the GPU (DW_IMAGE_CUDA) or using the Tegra VIC engine (DW_IMAGE_NVMEDIA).
 * The coordinates are calculated on the spot, unless distortion map has been enabled, in which case they will be read.
 * NOTE: the coordinates of input/output images are implicitly adapted to match the input/output camera if the
 * sizes of the images don't match the cameras. This will result in stretching of the image and possible loss of aspect ratio.
 *
 * @param[out] outputImage Handle to the output image.
 * @param[in] inputImage Handle to the input image.
 * @param[in] roi A region of interest of the output rectified area
 * @param[in] obj A pointer to the rectifier handle that performs the warping.
 *
 * @retval DW_INVALID_ARGUMENT  if the input image or output image handle is NULL
 * @retval DW_INVALID_HANDLE  if the rectifier handle is invalid or NULL
 * @retval DW_CUDA_ERROR  if underlying CUDA call fails
 * @retval DW_SUCCESS  on success
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwRectifier_apply(dwImageHandle_t outputImage, const dwConstImageHandle_t inputImage,
                           dwRect* roi, dwRectifierHandle_t obj);

/**
 * Warps the image from the input camera model to the model of the output camera using CUDA on the GPU.
 * The coordinates are calculated on the spot, unless distortion map has been enabled, in which case they will be read.
 * NOTE: the coordinates of input/output images are implicitly adapted to match the input/output camera if the
 * sizes of the images don't match the cameras. This will result in stretching of the image and possible loss of aspect ratio.
 *
 * @param[out] outputImage Pointer to the output image.
 * @param[in] inputImage Pointer to the input image.
 * @param[in] setOutsidePixelsToBlack if true, the pixels outside the original image are set to black on the undistorted image, otherwise interpolated
 * @param[in] obj A pointer to the rectifier handle that performs the warping.
 *
 * @retval DW_INVALID_ARGUMENT  if the input image or output image pointer is NULL
 * @retval DW_INVALID_HANDLE  if the rectifier handle is invalid or NULL
 * @retval DW_CUDA_ERROR  if underlying CUDA call fails
 * @retval DW_SUCCESS  on success
 */
DW_API_PUBLIC
DW_DEPRECATED("dwwRectifier_warp has been deprecated, please use dwRectifier_apply")
dwStatus dwRectifier_warp(dwImageCUDA* outputImage, const dwImageCUDA* inputImage,
                          bool setOutsidePixelsToBlack, dwRectifierHandle_t obj);

#ifdef VIBRANTE
/**
 * Warps the image from the input camera model to the model of the output camera using the Tegra VIC engine.
 * External pixels are automatically filled by replicating valid pixels
 * NOTE: the coordinates of input/output images are implicitly adapted to match the input/output camera if the
 * sizes of the images don't match the cameras. This will result in stretching of the image and possible loss of aspect ratio.
 *
 * @param[out] outputImage Pointer to the output image.
 * @param[in] inputImage Pointer to the input image.
 * @param[in] obj A pointer to the rectifier handle that performs the warping.
 *
 * @retval DW_INVALID_ARGUMENT  if the input image or output image pointer is NULL
 * @retval DW_INVALID_HANDLE  if the rectifier handle is invalid or NULL
 * @retval DW_NVMEDIA_ERROR  if underlying NvMedia call fails
 * @retval DW_SUCCESS  on success
 */
DW_API_PUBLIC
DW_DEPRECATED("dwwRectifier_warpNvMedia has been deprecated, please use dwRectifier_apply")
dwStatus dwRectifier_warpNvMedia(dwImageNvMedia* outputImage, const dwImageNvMedia* inputImage, dwRectifierHandle_t obj);
#endif
/**
 * Warps the image from the input camera model to the model of the output camera limiting the computation to a ROI.
 * The ROI is defined in output camera coordinates and the resulting image will be translated to the origin of the output
 * image. No further transformation will be performed.
 *
 * @param[out] outputImage Pointer to the output image.
 * @param[in] inputImage Pointer to the input image. NOTE: the coordinates of this image are implicitly adapted to the input camera
 * @param[in] setOutsidePixelsToBlack if true, the pixels outside the original image are set to black on the undistorted image, otherwise interpolated
 * @param[in] roi A region of interest of the output rectified area
 * @param[in] obj A pointer to the rectifier handle that performs the warping.
 *
 * @retval DW_INVALID_ARGUMENT  if the input image or output image pointer is NULL, or if roi is invalid
 * @retval DW_INVALID_HANDLE  if the rectifier handle is invalid or NULL
 * @retval DW_CUDA_ERROR  if underlying CUDA call fails
 * @retval DW_SUCCESS  on success
 */
DW_API_PUBLIC
DW_DEPRECATED("dwwRectifier_warpROI has been deprecated, please use dwRectifier_apply")
dwStatus dwRectifier_warpROI(dwImageCUDA* outputImage, const dwImageCUDA* inputImage,
                             bool setOutsidePixelsToBlack, const dwRect roi, dwRectifierHandle_t obj);

/**
 * Sets the homography matrix used.
 *
 * @param[in] homography The new homography transformation matrix.
 * @param[in] obj A pointer to the rectifier handle that is updated.
 * Initialization must not have changed.
 *
 * @retval DW_INVALID_ARGUMENT  if the matrix is NULL
 * @retval DW_INVALID_HANDLE  if the rectifier handle is invalid or NULL
 * @retval DW_SUCCESS  on success
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwRectifier_setHomography(const dwMatrix3f* homography, dwRectifierHandle_t obj);

/**
 * Calculates and sets the homography matrix used based on a value for roll, pitch and yaw. This rotation will be applied during backprojection, so from output to input
 *
 * @param[in] roll The camera roll in degrees.
 * @param[in] pitch The camera pitch in degrees.
 * @param[in] yaw The camera yaw in degrees.
 * @param[in] obj A pointer to the rectifier handle that is updated.
 * Initialization must not have changed.
 *
 * @retval DW_INVALID_HANDLE  if the rectifier handle is invalid or NULL
 * @retval DW_SUCCESS  on success 
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwRectifier_setHomographyFromRotation(float32_t roll, float32_t pitch, float32_t yaw, dwRectifierHandle_t obj);

/**
 * Warps an array of CPU dwVector2f on a preallocated output CPU buffer
 *
 * @param[out] outputPoints Array of output points.
 * @param[in] inputPoints Array of input points
 * @param[in] pointCount number of points
 * @param[in] obj A pointer to the rectifier handle that performs the warping.
 *
 * @retval DW_INVALID_ARGUMENT  if the input vector or output vector pointer is NULL
 * @retval DW_INVALID_HANDLE  if the rectifier handle is invalid or NULL
 * @retval DW_SUCCESS  on success
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwRectifier_warpPointsCPU(dwVector2f* outputPoints, const dwVector2f* inputPoints,
                                   uint32_t pointCount, dwRectifierHandle_t obj);

/**
 * Warps an array of CPU dwVector2f and writes on the same buffer
 *
 * @param[in] points Array of points to warp in place
 * @param[in] pointCount number of points
 * @param[in] obj A pointer to the rectifier handle that performs the warping.
 *
 * @retval DW_INVALID_ARGUMENT  if the vector pointer is NULL
 * @retval DW_INVALID_HANDLE  if the rectifier handle is invalid or NULL
 * @retval DW_SUCCESS  on success
 */
DW_API_PUBLIC
DW_DEPRECATED("dwRectifier_warpPointsCPUInPlace has been deprecated, please use dwRectifier_warpPointsCPU passing same input and output")
dwStatus dwRectifier_warpPointsInPlaceCPU(dwVector2f* points,
                                          uint32_t pointCount, dwRectifierHandle_t obj);

/**
 * Warps an array of dwVector2f on GPU.
 *
 * @param[out] outputPoints Array of output points.
 * @param[in] inputPoints Array of input points
 * @param[in] pointCount number of points
 * @param[in] obj A pointer to the rectifier handle that performs the warping.
 *
 * @retval DW_INVALID_ARGUMENT  if the input vector or output vector pointer is NULL
 * @retval DW_INVALID_HANDLE  if the rectifier handle is invalid or NULL
 * @retval DW_CUDA_ERROR  if underlying CUDA call fails
 * @retval DW_SUCCESS  on success 
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwRectifier_warpPointsGPU(dwVector2f* outputPoints, const dwVector2f* inputPoints,
                                   uint32_t pointCount, dwRectifierHandle_t obj);

/**
 * Warps an array of dwVector2f on the GPU and writes on the same buffer
 *
 * @param[in] points Array of points to warp in place
 * @param[in] pointCount number of points
 * @param[in] obj A pointer to the rectifier handle that performs the warping.
 *
 * @retval DW_INVALID_ARGUMENT  if the input vector or output image vector is NULL
 * @retval DW_INVALID_HANDLE  if the rectifier handle is invalid or NULL
 * @retval DW_CUDA_ERROR  if underlying CUDA call fails
 * @retval DW_SUCCESS  on success
 */
DW_API_PUBLIC
DW_DEPRECATED("dwRectifier_warpPointsGPUInPlace has been deprecated, please use dwRectifier_warpPointsGPU passing same input and output")
dwStatus dwRectifier_warpPointsInPlaceGPU(dwVector2f* points,
                                          uint32_t pointCount, dwRectifierHandle_t obj);

/**
 * Gets the homography matrix used.
 *
 * @param[out] homography The homography transformation matrix.
 * @param[in] obj A pointer to the rectifier handle.
 * Initialization must not have changed.
 *
 * @retval DW_INVALID_HANDLE  if the rectifier handle is invalid or NULL
 * @retval DW_SUCCESS  on success 
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwRectifier_getHomography(dwMatrix3f* homography, dwRectifierHandle_t obj);

/**
 * Sets the CUDA stream used.
 *
 * @param[in] stream The CUDA stream to use.
 * @param[in] obj A pointer to the rectifier handle that is updated.
 * Initialization must not have changed.
 *
 * @retval DW_INVALID_HANDLE  if the rectifier handle is invalid or NULL
 * @retval DW_SUCCESS  on success
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwRectifier_setCUDAStream(cudaStream_t stream, dwRectifierHandle_t obj);

/**
 * Gets the CUDA stream used.
 *
 * @param[in] stream Uses this CUDA stream.
 * @param[in] obj A pointer to the rectifier handle that is updated.
 * Initialization must not have changed.
 *
 * @retval DW_INVALID_HANDLE  if the rectifier handle is invalid or NULL
 * @retval DW_SUCCESS  on success
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwRectifier_getCUDAStream(cudaStream_t* stream, dwRectifierHandle_t obj);

/**
 * Gets the distortion map as a 2-channel single plane image.
 *
 * @param[out] distortionMap The distortion map.
 * @param[in] obj A pointer to the rectifier handle that is updated.
 * Initialization must not have changed.
 *
 * @retval DW_INVALID_HANDLE  if the rectifier handle is invalid or NULL
 * @retval DW_SUCCESS  on success
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwRectifier_getDistortionMap(dwImageCUDA* distortionMap, dwRectifierHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_RECTIFIER_H_

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

/**
 * @file
 * <b>NVIDIA DriveWorks API: Image Threshold Methods</b>
 *
 * @b Description: This file defines Image Threshold methods.
 */

/**
 * @defgroup Threshold_group Image Threshold Interface
 *
 * @brief Defines the Image Threshold module.
 *
 * @{
 */

#ifndef DW_IMAGEPROCESSING_SEGMENTATION_THRESHOLDING_H_
#define DW_IMAGEPROCESSING_SEGMENTATION_THRESHOLDING_H_

#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>
#include <dw/image/Image.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwThresholdObject* dwThresholdHandle_t;

/// thresholding behavior based on comparison (a > b if dwThresholdParameters.inverse is false, a < b otherwise)
typedef enum dwThresholdBehavior {
    /// if comparison between pixel and thresh yields true, pixel = maxval else 0
    DW_THRESHOLD_BEHAVIOR_BINARY = 0,
    /// if comparison between pixel and thresh yields true, pixel = thresh else pixel
    DW_THRESHOLD_BEHAVIOR_TRUNCATE = 1,
    /// if comparison between pixel and thresh yields true, pixel = pixel else 0
    DW_THRESHOLD_BEHAVIOR_TO_ZERO = 2,
} dwThresholdBehavior;

typedef enum dwThresholdMode {
    /// based on a user selected manualThresholdValue
    DW_THRESHOLD_MODE_SIMPLE = 0,
    /// automatically computes the best threshold (assuming bimodal histogram, see N. Otsu, "A Threshold Selection Method from Gray-Level Histograms", IEEE Transaction on Systems and Cybernetics, 1979)
    DW_THRESHOLD_MODE_OTSU = 1,
    /// each pixel of the input image is thresholded individually
    DW_THRESHOLD_MODE_PER_PIXEL = 2,
} dwThresholdMode;

typedef struct dwThresholdParameters
{
    /// threshold mode
    dwThresholdMode mode;
    /// threshold behavior
    dwThresholdBehavior behavior;
    /// maximum value
    uint32_t maxVal;
    /// if false the comparison is pixel > threshold, else pixel < threshold
    bool inverse;
    /// cuda stream
    cudaStream_t stream;
    /// manual value for MODE_SIMPLE
    uint32_t manualThresholdValue;
    /// thresholding image for MODE_PER_PIXEL. Each pixel (x,y) in this DW_IMAGEFORMAT_R_XXX image contains the
    /// thresholding value for pixel (x,y) in the input image. If this image is obtained by Gaussian Filtering
    /// with filtering window size S, the behavior is called "adaptive gaussian thresholding with window S"
    dwImageHandle_t thresholdingImage;
} dwThresholdParameters;

/**
 * Initializes a Threshold Handle
 * @param[out] handle Pointer to the Threshold Handle.
 * @param[in] params parameters of image Threshold.
 * @param[in] context Handle to Driveworks
 *
 * @return DW_INVALID_ARGUMENT if the handle is invalid <br>
 * DW_SUCCESS if the operation is successful <br>
 * DW_CUDA_ERROR if the underlying cuda operation failed <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwThreshold_initialize(dwThresholdHandle_t* handle, dwThresholdParameters params,
                                dwContextHandle_t context);

/**
 * Resets an Threshold Handle
 * @param[in] obj Pointer to the Threshold Handle.
 *
 * @return DW_INVALID_ARGUMENT if the handle is invalid <br>
 * DW_SUCCESS if the operation is successful <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwThreshold_reset(dwThresholdHandle_t obj);

/**
 * Releases an Threshold Handle
 * @param[in] handle Pointer to the Threshold Handle.
 *
 * @return DW_INVALID_ARGUMENT if the handle is invalid <br>
 * DW_SUCCESS if the operation is successful <br>
 * DW_CUDA_ERROR if the underlying cuda operation failed <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwThreshold_release(dwThresholdHandle_t handle);

/**
* Runs the Threshold Handle on input image using the operations set in dwThreshold_setOperations
*
* @param[out] outputImage A handle to the output image
* @param[in] inputImage A handle to the input image
* @param[in] obj Handle to the Threshold Handle
*
* @return DW_SUCCESS <br>
*        DW_INVALID_HANDLE - If the given handle is invalid,i.e. null or of wrong type  <br>
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwThreshold_applyThreshold(dwImageHandle_t outputImage, const dwImageHandle_t inputImage, dwThresholdHandle_t obj);

/**
* Changes the threshold parameters in runtime
*
* @param[in] parameters The new parameters to set
* @param[in] obj Handle to the Threshold Handle
*
* @return DW_SUCCESS <br>
*        DW_INVALID_HANDLE - If the given handle is invalid,i.e. null or of wrong type  <br>
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwThreshold_setThresholdParameters(dwThresholdParameters parameters,
                                            dwThresholdHandle_t obj);
/**
 * Sets the cuda stream used by the APIs of Image Threshold
 * @param[in] stream CUDA stream
 * @param[in] obj Handle to the Threshold Handle
 *
 * @return DW_INVALID_ARGUMENT if the handle is invalid <br>
 * DW_SUCCESS if the operation is successful <br>
 * DW_CUDA_ERROR if the underlying cuda operation failed <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwThreshold_setCUDAStream(cudaStream_t stream, dwThresholdHandle_t obj);

/**
 * Gets the cuda stream used by the APIs of Image Threshold
 * @param[out] stream CUDA stream
 * @param[in] obj Handle to the Threshold Handle
 *
 * @return
 * DW_SUCCESS if the operation is successful <br>
 * DW_CUDA_ERROR if the underlying cuda operation failed <br>
 */

/**
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwThreshold_getCUDAStream(cudaStream_t* stream, dwThresholdHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_IMAGEPROCESSING_SEGMENTATION_THRESHOLDING_H_

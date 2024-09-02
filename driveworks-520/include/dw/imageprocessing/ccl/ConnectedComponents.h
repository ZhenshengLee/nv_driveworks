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
 * <b>NVIDIA DriveWorks API: Connected Components</b>
 *
 * @b Description: This file defines 2D connected components methods.
 */

/**
 * @defgroup connected_components_group Connected Components Interface
 *
 * @brief Defines 2D Connected Components Labeling algorithm.
 *
 * @{
 */

#ifndef DW_IMAGEPROCESSING_CONNECTED_COMPONENTS_H_
#define DW_IMAGEPROCESSING_CONNECTED_COMPONENTS_H_

#include <dw/image/Image.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief A pointer to the opaque handle for Connected Components
 */
typedef struct dwConnectedComponentsObject* dwConnectedComponentsHandle_t;
/**
 * @brief A constant pointer to the opaque handle for Connected Components
 */
typedef struct dwConnectedComponentsObject* const dwConstConnectedComponentsHandle_t;

/**
* @brief Initializes connected components. Must be initialized in a thread with valid CUDA context
*
* @param[out] ccl A pointer to the opaque connected components handle
* @param[in] inputDesc Description of input image
* @param[in] context Specifies the opaque handle of a `dwContext`
* @return DW_INVALID_ARGUMENT - if any of provided arguments is NULL or input image description contain invalid image size<br>
*         DW_NOT_SUPPORTED - if input description contains attributes which are not supported<br>
*         DW_CUDA_ERROR - if module could not create some of internal CUDA resources<br>
*         DW_SUCCESS - if initialization is successful<br>
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: No
*/
DW_API_PUBLIC
dwStatus dwConnectedComponents_initialize(dwConnectedComponentsHandle_t* ccl,
                                          dwImageProperties const* inputDesc,
                                          dwContextHandle_t context);

/**
* @brief Initialize the module with limited functionality to produce connected components within local tiles
* @param[out] ccl A pointer to the opaque connected components handle.
* @param[in] inputDesc Description of input image.
* @param[in] context Specifies the opaque handle of a `dwContext`.
* @return DW_INVALID_ARGUMENT - if any of provided arguments is NULL or input image description contain invalid image size<br>
*         DW_NOT_SUPPORTED - if input description contains attributes which are not supported<br>
*         DW_CUDA_ERROR - if module could not create some of internal CUDA resources<br>
*         DW_SUCCESS<br>
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: No
*/
DW_API_PUBLIC
dwStatus dwConnectedComponents_initializeMinimal(dwConnectedComponentsHandle_t* ccl,
                                                 dwImageProperties const* inputDesc,
                                                 dwContextHandle_t context);

/**
* @brief Specifies CUDA stream where kernels are executed
*
* @param[in] stream Specifies cuda stream
* @param[in] ccl Specifies the opaque connected components handle
* @return DW_INVALID_ARGUMENT - if provided pointer to ccl handle is NULL<br>
*         DW_SUCCESS - if stream is set<br>
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwConnectedComponents_setCUDAStream(cudaStream_t stream, dwConnectedComponentsHandle_t ccl);

/**
* @brief Returns CUDA stream where kernels are executed
* @param[in] stream Specifies pointer where cuda stream will be returned. Should not be NULL
* @param[in] ccl Specifies the opaque connected components handle
* @return DW_INVALID_HANDLE - if provided ccl handle is NULL<br>
*         DW_SUCCESS - on success<br>
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwConnectedComponents_getCUDAStream(cudaStream_t* stream, dwConnectedComponentsHandle_t ccl);

/**
* @brief Specifies input image to be processed
* @param[in] image A pointer to input image
* @param[in] ccl Specifies the opaque connected components handle
* @return DW_INVALID_ARGUMENT - input image is null or does not match description provided on initialization<br>
*         DW_INVALID_HANDLE - if `ccl` handle is NULL<br>
*         DW_SUCCESS - if input is bound successfully<br>
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwConnectedComponents_bindInput(dwImageCUDA const* image,
                                         dwConnectedComponentsHandle_t ccl);

/**
* @brief Specifies threshold to binarize input image
* @param[in] threshold - 8-bit unsigned integer threshold value
* @param[in] ccl Specifies the opaque connected components handle
* @return DW_INVALID_HANDLE - if `ccl` handle is NULL<br>
*         DW_SUCCESS - if threshold is set succesfully<br>
*
* @note By default threshold is 127
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwConnectedComponents_setThreshold(uint8_t threshold,
                                            dwConnectedComponentsHandle_t ccl);

/**
* @brief Specifies output label image for ccl algorithm
* @param[in] labels A pointer to output image containing labels
* @param[in] ccl Specifies the opaque connected components handle
* @return DW_INVALID_ARGUMENT - output image does not match description provided on initialization<br>
*         DW_INVALID_HANDLE - if `ccl` handle is NULL<br>
*         DW_SUCCESS - if out label image is bound successfully<br>
*
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwConnectedComponents_bindOutputLabels(dwImageCUDA* labels, dwConnectedComponentsHandle_t ccl);

/**
* @brief Performs image labeling
* @param[in] ccl Specifies the opaque connected components handle
* @return DW_CALL_NOT_ALLOWED - if input or output is not bound<br>
*         DW_CUDA_ERROR - CUDA kernels failed to execute<br>
*         DW_FAILURE - on generic failure<br>
*         DW_INVALID_HANDLE - if `ccl` handle is NULL<br>
*         DW_SUCCESS - if labeling is successfully performed<br><br>
*
* @note The call is asynchronous. The user suppose to sync the stream to make sure all necessary work is done.
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwConnectedComponents_process(dwConnectedComponentsHandle_t ccl);

/**
* @brief Performs reset of connected components object
* @param[in] ccl Specifies the opaque connected components handle
* @return DW_INVALID_HANDLE - if `ccl` handle is NULL<br>
*         DW_SUCCESS - on success<br>
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwConnectedComponents_reset(dwConnectedComponentsHandle_t ccl);

/**
* @brief Releases connected components object
* @param[in] ccl A pointer to the opaque connected components handle
* @return DW_INVALID_HANDLE - if provided handle is NULL<br>
*         DW_SUCCESS - if the module successfully released<br>
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwConnectedComponents_release(dwConnectedComponentsHandle_t ccl);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_IMAGEPROCESSING_CONNECTED_COMPONENTS_H_

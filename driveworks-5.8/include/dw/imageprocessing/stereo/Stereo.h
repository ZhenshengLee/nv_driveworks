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
 * <b>NVIDIA DriveWorks API: Stereo Methods</b>
 *
 * @b Description: This file defines stereo disparity computation methods.
 *
 */

#ifndef DW_IMAGEPROCESSING_STEREO_STEREO_H_
#define DW_IMAGEPROCESSING_STEREO_STEREO_H_

#include <dw/image/Image.h>
#include <dw/rig/Rig.h>
#include <dw/imageprocessing/geometry/rectifier/Rectifier.h>
#include <dw/imageprocessing/filtering/Pyramid.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup stereo_group Stereo Interface
 *
 * @brief Runs the stereo pipeline and computes disparity map.
 *
 * @{
 */

///////////////////////////////////////////////////////////////////////
// dwStereo

/** A pointer to the handle representing a stereo algorithm.
 * This object allows the computation of a disparity map given two rectified stereo images
 */
typedef struct dwStereoObject* dwStereoHandle_t;

#define DW_STEREO_SIDE_COUNT 2
#define MAX_ALLOWED_DISPARITY_RANGE 1024

/**
* Side
*/
typedef enum {
    ///Left
    DW_STEREO_SIDE_LEFT = 0,
    ///Right
    DW_STEREO_SIDE_RIGHT,
    ///Both sides
    DW_STEREO_SIDE_BOTH
} dwStereoSide;

/**
* Cost types for matching
*/
typedef enum {
    ///Absolute difference
    DW_STEREO_COST_AD,
    ///Normalised cross correlation
    DW_STEREO_COST_NCC,
    ///Sum of absolute differences
    DW_STEREO_COST_SAD,
    ///Census transform
    DW_STEREO_COST_CENSUS,
    ///Absolute difference and census
    DW_STEREO_COST_ADCENSUS,
} dwStereoCostType;

/**
* Configuration parameters for a Stereo algorithm.
*/
typedef struct
{
    ///Input image width (single image).
    uint32_t width;

    ///Input image height.
    uint32_t height;

    ///Maximal displacement when searching for corresponding pixels
    uint32_t maxDisparityRange;

    ///Number of levels in the pyramid. It must be the same or less than that of the Gaussian pyramid.
    uint32_t levelCount;

    ///Level of the pyramid where disparity computation ends. It defines the resolution of the output disparity and confidence maps.
    uint32_t levelStop;

    ///Side to compute the disparity map of.
    dwStereoSide side;

    ///Specifies whether to perform a L/R occlusion test.
    bool occlusionTest;

    ///Threshold for failing the L/R consistency test (in disparity value).
    uint32_t occlusionThreshold;

    ///Specifies whether to fill occluded pixels for 100% density.
    bool occlusionFilling;

    ///Specifies threshold of invalidity
    float32_t invalidityThreshold;

    ///Specifies whether to fill invalid pixel using assumption on the scene in order to have a map with 100% density.
    bool holesFilling;

    ///Refinement level (0 no refinement, 1-3)
    uint8_t refinementLevel;

    ///Specifies the cost type used for initialization.
    dwStereoCostType initType;

} dwStereoParams;

/**
 * Initializes the stereo parameters
 *
 * @param[out] stereoParams Parameters to be initialised with default values.
 *
* @return DW_INVALID_HANDLE - if given handle is not valid, i.e. null or of wrong type . <br>
*         DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereo_initParams(dwStereoParams* stereoParams);

/**
 * Initializes the stereo algorithm with the parameters
 *
 * @param[out] obj A pointer to the stereo algorithm.
 * @param[in] width The width of one input image.
 * @param[in] height The height of one input image.
 * @param[in] stereoParams A pointer to the configuration of the stereo algorithm.
 * @param[in] ctx the handle to DW context.
 *
* @return DW_CUDA_ERROR - if the underlying stereo algorithm had a CUDA error. <br>
*         DW_INVALID_HANDLE - if given handle is not valid, i.e. null or of wrong type . <br>
*         DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereo_initialize(dwStereoHandle_t* obj, uint32_t width, uint32_t height,
                             const dwStereoParams* stereoParams, dwContextHandle_t ctx);

/**
 * Computes the disparity map given the two rectified views
 *
 * @param[in] leftPyramid The left 8 bit rectified image pyramid.
 * @param[in] rightPyramid The left 8 bit rectified image pyramid.
 * @param[in] obj The stereo algorithm handle.
 *
* @return DW_CUDA_ERROR - if the underlying stereo algorithm had a CUDA error. <br>
*         DW_INVALID_HANDLE - if given handle is not valid, i.e. null or of wrong type . <br>
*         DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereo_computeDisparity(const dwPyramidImage* leftPyramid,
                                   const dwPyramidImage* rightPyramid, dwStereoHandle_t obj);

/**
 * Returns the disparity map for a specified side. Requires dwStereo_computeDisparity() to be done first
 *
 * @param[out] disparityMap A 2D matrix containing disparity values in 8 bit.
 * @param[in] side The side, either DW_STEREO_SIDE_LEFT or DW_STEREO_SIDE_RIGHT
 * @param[in] obj The stereo algorithm handle.
 *
* @return DW_CUDA_ERROR - if the underlying stereo algorithm had a CUDA error. <br>
*         DW_INVALID_HANDLE - if given handle is not valid, i.e. null or of wrong type . <br>
*         DW_INVALID_ARGUMENT - if the input images have a mismatching attribute <br>
*         DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereo_getDisparity(const dwImageCUDA** disparityMap, dwStereoSide side, dwStereoHandle_t obj);

/**
 * Returns the confidence map for a specified side. Requires dwStereo_computeDisparity() to be done first
 *
 * @param[out] confidenceMap A 2D matrix containing confidence values in 8 bit.
 * @param[in] side The side, either DW_STEREO_SIDE_LEFT or DW_STEREO_SIDE_RIGHT
 * @param[in] obj The stereo algorithm handle.
 *
* @return DW_CUDA_ERROR - if the underlying stereo algorithm had a CUDA error. <br>
*         DW_INVALID_HANDLE - if given handle is not valid, i.e. null or of wrong type . <br>
*         DW_INVALID_ARGUMENT - if the input images have a mismatching attribute <br>
*         DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereo_getConfidence(const dwImageCUDA** confidenceMap, dwStereoSide side, dwStereoHandle_t obj);

/**
 * Set occlusion test on/off
 *
 * @param[in] doTest a bool to activate test.
 * @param[in] obj The stereo algorithm handle.
 *
* @return DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereo_setOcclusionTest(bool doTest, dwStereoHandle_t obj);

/**
 * Set occlusion infill on/off
 *
 * @param[in] doInfill a bool to activate test.
 * @param[in] obj The stereo algorithm handle.
 *
* @return DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereo_setOcclusionInfill(bool doInfill, dwStereoHandle_t obj);

/**
 * Set invalid infill on/off
 *
 * @param[in] doInfill a bool to activate test.
 * @param[in] obj The stereo algorithm handle.
 *
* @return DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereo_setInfill(bool doInfill, dwStereoHandle_t obj);

/**
 * Set invalidity threshold
 *
 * @param[in] threshold a float value for invalidity
 * @param[in] obj The stereo algorithm handle.
 *
* @return DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereo_setInvalidThreshold(float32_t threshold, dwStereoHandle_t obj);

/**
 * Sets the refinement level of the ongoing stereo algorithm
 *
 * @param[in] refinementLvl the refinement level between 0 and 6.
 * @param[in] obj The stereo algorithm handle.
 *
* @return DW_CUDA_ERROR - if the underlying stereo algorithm had a CUDA error. <br>
*         DW_INVALID_HANDLE - if given handle is not valid, i.e. null or of wrong type . <br>
*         DW_INVALID_ARGUMENT - if the input images have a mismatching attribute <br>
*         DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereo_setRefinementLevel(uint8_t refinementLvl, dwStereoHandle_t obj);

/**
 * Get size of image at a certain level
 *
 * @param[out] dispWidth with of the image.
 * @param[out] dispHeight height of the image.
 * @param[in] gLevel level of the pyramid.
 * @param[in] obj The stereo algorithm handle.
 *
* @return DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereo_getSize(uint32_t* dispWidth, uint32_t* dispHeight, uint32_t gLevel,
                          dwStereoHandle_t obj);

/**
 * Sets CUDA stream used by the stereo algorithm.
 *
 * @param[in] stream The CUDA stream.
 * @param[in] obj The stereo algorithm handle.
 *
 * @return DW_INVALID_HANDLE if the given context handle is invalid,i.e. null or of wrong type  <br>
 *         or DW_SUCCESS otherwise.
 */
DW_API_PUBLIC
dwStatus dwStereo_setCUDAStream(cudaStream_t stream, dwStereoHandle_t obj);

/**
 * Gets CUDA stream used by the stereo algorithm.
 *
 * @param[out] stream The CUDA stream currently used by the stereo algorithm.
 * @param[in] obj The stereo algorithm handle.
 *
 * @return DW_INVALID_HANDLE if the given context handle is invalid, i.e. null or of wrong type  <br>
 *         or DW_SUCCESS otherwise.
 */
DW_API_PUBLIC
dwStatus dwStereo_getCUDAStream(cudaStream_t* stream, dwStereoHandle_t obj);

/**
 * Resets the Stereo module.
 *
 * @param[in] obj Specifies the stereo handle to reset.
 *
 * @return DW_SUCCESS <br>
 *         DW_INVALID_HANDLE - If the given context handle is invalid,i.e. null or of wrong type  <br>
 *         DW_BAD_CAST <br>
 */
DW_API_PUBLIC
dwStatus dwStereo_reset(dwStereoHandle_t obj);

/**
 * Releases the stereo algorithm.
 * This method releases all resources associated with a stereo algorithm.
 *
 * @note This method renders the handle unusable.
 *
 * @param[in] obj The object handle to be released.
 *
 * @return DW_SUCCESS <br>
 *         DW_INVALID_HANDLE - If the given context handle is invalid,i.e. null or of wrong type  <br>
 *         DW_BAD_CAST <br>
 *
 */
DW_API_PUBLIC
dwStatus dwStereo_release(dwStereoHandle_t obj);

///////////////////////////////////////////////////////////////////////
// dwStereoRectifier

/** A pointer to the handle representing a stereo rectifier.
 * This object allows the rectification of two stereo images based on calibration matrices
 */
typedef struct dwStereoRectifierObject* dwStereoRectifierHandle_t;

/**
* Cropping
*/
typedef enum {
    ///No scaling, keeps output of rectifier
    DW_STEREO_RECTIFIER_UNCHANGED,
    ///Crops to inner valid rectangle
    DW_STEREO_RECTIFIER_CROP
} dwStereoRectifierCrop;

/**
 * Initializes the stereo rectifier
 *
 * @param[out] obj A pointer to the stereo rectifier.
 * @param[in] cameraLeft a handle to the calibrated camera of the left eye.
 * @param[in] cameraRight a handle to the calibrated camera of the right eye.
 * @param[in] leftToRig the extrinsic matrix from letf camera to rig centre (usually the left camera itself).
 * @param[in] rightToRig the extrinsic matrix from right camera to rig centre.
 * @param[in] ctx A handle to DW context.
 *
* @return DW_CUDA_ERROR - if the underlying rectifier had a CUDA error during initialization. <br>
*         DW_INVALID_HANDLE - if given handle is not valid, i.e. null or of wrong type . <br>
*         DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereoRectifier_initialize(dwStereoRectifierHandle_t* obj, dwCameraModelHandle_t cameraLeft,
                                      dwCameraModelHandle_t cameraRight, dwTransformation3f leftToRig,
                                      dwTransformation3f rightToRig, dwContextHandle_t ctx);

/**
 * Releases the stereo rectifier
 *
 * @param[in] obj The stereo algorithm.
 *
* @return DW_INVALID_HANDLE - if given handle is not valid, i.e. null or of wrong type . <br>
*         DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereoRectifier_release(dwStereoRectifierHandle_t obj);

/**
 * Rectifies two images acquired by a stereo rig, epipolar lines will be parallel
 *
 * @param[out] outputImageLeft Pointer to the output left image.
 * @param[out] outputImageRight Pointer to the output right image.
 * @param[in] inputImageLeft Pointer to the left input image.
 * @param[in] inputImageRight Pointer to the right input image.
 * @param[in] obj A pointer to the rectifier handle.
 *
* @return DW_CUDA_ERROR - if the underlying stereo algorithm had a CUDA error. <br>
*         DW_INVALID_HANDLE - if given handle is not valid, i.e. null or of wrong type . <br>
*         DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereoRectifier_rectify(dwImageCUDA* outputImageLeft, dwImageCUDA* outputImageRight,
                                   const dwImageCUDA* inputImageLeft, const dwImageCUDA* inputImageRight,
                                   dwStereoRectifierHandle_t obj);

/**
 * Returns a rectangle which is the roi where all valid pixels after undistortion and rectification are.
 * It is used for cropping
 *
 * @param[out] roi Pointer to a 2D box defining the roi to crop.
 * @param[in] obj A pointer to the rectifier handle.
 *
* @return DW_INVALID_HANDLE - if given handle is not valid, i.e. null or of wrong type . <br>
*         DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereoRectifier_getCropROI(dwBox2D* roi, dwStereoRectifierHandle_t obj);

/**
 * Returns a 3x4 projection matrix for the side specified
 * of the form:
 * P_left = M_rect_left*[I|0]
 * P_right = M_rict_right*[I|Tx]
 * with M the rectified intrinsics matrix and Tx the baseline
 *
 * @param[out] projectionMat Pointer to a dwMatrix3x4f.
 * @param[in] side The stereo side
 * @param[in] obj A pointer to the rectifier handle.
 *
* @return DW_INVALID_HANDLE - if given handle is not valid, i.e. null or of wrong type . <br>
*         DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereoRectifier_getProjectionMatrix(dwMatrix34f* projectionMat, dwStereoSide side,
                                               dwStereoRectifierHandle_t obj);

/**
 * Returns a 3x3 rotation matrix for the side specified. The matrix sends epipoles to infinity
 *
 * @param[out] rRectMat Pointer to a dwMatrix3f.
 * @param[in] side The stereo side
 * @param[in] obj A pointer to the rectifier handle.
 *
* @return DW_INVALID_HANDLE - if given handle is not valid, i.e. null or of wrong type . <br>
*         DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereoRectifier_getRectificationMatrix(dwMatrix3f* rRectMat, dwStereoSide side,
                                                  dwStereoRectifierHandle_t obj);

/**
 * Returns a 4x4 reprojetion matrix of the form
 *     1, 0, 0, -Cx
 * Q = 0, 1, 0, -Cy
 *     0, 0, 0, foc
 *     0, 0,-1/Tx, (Cx - C'x)/Tx
 *
 * @param[out] qMatrix Pointer to a dwMatrix4f.
 * @param[in] obj A pointer to the rectifier handle.
 *
* @return DW_INVALID_HANDLE - if given handle is not valid, i.e. null or of wrong type . <br>
*         DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwStereoRectifier_getReprojectionMatrix(dwMatrix4f* qMatrix, dwStereoRectifierHandle_t obj);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_STEREO_STEREO_H_

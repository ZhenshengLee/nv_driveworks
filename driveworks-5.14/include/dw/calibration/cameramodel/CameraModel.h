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

#ifndef DW_CALIBRATION_CAMERAMODEL_CAMERAMODEL_H_
#define DW_CALIBRATION_CAMERAMODEL_CAMERAMODEL_H_

#include <dw/core/base/Config.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>
#include <dw/rig/Rig.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file
 * <b>NVIDIA DriveWorks API: Camera Methods</b>
 *
 * @b Description: This file defines camera model and rig methods.
 */

/**
 * @defgroup cameramodel_group Camera Model
 *
 * @brief Calibrated camera model abstraction and functionality.
 *
 * @{
 */

///////////////////////////////////////////////////////////////////////
// Calibrated cameras

/**
 * A pointer to the handle representing a calibrated camera model.
 * This object allows the forward projection of 3D points onto 2D image pixels (ray2Pixel)
 * and the corresponding back-projection (`pixel2Ray`).
 */
typedef struct dwCameraModelObject* dwCameraModelHandle_t;

/**
 * A pointer to the handle representing a const calibrated camera.
 */
typedef struct dwCameraModelObject const* dwConstCameraModelHandle_t;

/**
 * Creates and initializes a calibrated pinhole camera.
 *
 * @param[out] obj A pointer to the calibrated camera handle is returned here.
 * @param[in] config A pointer to the configuration values for the camera.
 * @param[in] context Specifies the handle to the context under which it is created.
 *
 * @retval DW_INVALID_HANDLE  if given context handle is invalid
 * @retval DW_INVALID_ARGUMENT  if one of the given pointers is a nullptr
 * @retval DW_SUCCESS  no error
 */
DW_API_PUBLIC
dwStatus dwCameraModel_initializePinhole(dwCameraModelHandle_t* obj,
                                         const dwPinholeCameraConfig* config,
                                         dwContextHandle_t context);

/**
 * Creates and initializes a calibrated camera for the F-Theta distortion model.
 *
 * @param[out] obj A pointer to the calibrated camera handle is returned here.
 * @param[in] config A pointer to the configuration values for the camera.
 * @param[in] context Specifies the handle to the context under which it is created.
 *
 * @retval DW_INVALID_HANDLE  if given context handle is invalid
 * @retval DW_INVALID_ARGUMENT  if one of the given pointers is a nullptr
 * @retval DW_SUCCESS  no error
 */
DW_API_PUBLIC
dwStatus dwCameraModel_initializeFTheta(dwCameraModelHandle_t* obj,
                                        const dwFThetaCameraConfig* config,
                                        dwContextHandle_t context);

/**
 * Creates and initializes a calibrated camera for the F-Theta distortion model.
 *
 * @param[out] obj A pointer to the calibrated camera handle is returned here.
 * @param[in] config A pointer to the configuration values for the camera.
 * @param[in] context Specifies the handle to the context under which it is created.
 *
 * @retval DW_INVALID_HANDLE  if given context handle is invalid
 * @retval DW_INVALID_ARGUMENT  if one of the given pointers is a nullptr
 * @retval DW_SUCCESS  no error
 */
DW_API_PUBLIC
DW_DEPRECATED("dwCameraModel_initializeFThetaNew is renamed to dwCameraModel_initializeFTheta.")
dwStatus dwCameraModel_initializeFThetaNew(dwCameraModelHandle_t* obj,
                                           const dwFThetaCameraConfig* config,
                                           dwContextHandle_t context);

/**
 * Creates and initializes a calibrated stereographic camera.
 *
 * @param[out] obj A pointer to the calibrated camera handle is returned here.
 * @param[in] config A pointer to the configuration values for the camera.
 * @param[in] context Specifies the handle to the context under which it is created.
 *
 * @retval DW_INVALID_HANDLE  if given context handle is invalid
 * @retval DW_INVALID_ARGUMENT  if one of the given pointers is a nullptr
 * @retval DW_SUCCESS  no error
 */
DW_API_PUBLIC
dwStatus dwCameraModel_initializeStereographic(dwCameraModelHandle_t* obj,
                                               const dwStereographicCameraConfig* config,
                                               dwContextHandle_t context);

/**
 * Releases the calibrated camera.
 * This method releases all resources associated with a calibrated camera.
 *
 * @note This method renders the handle unusable.
 *
 * @param[in] obj The object handle to be released.
 *
 * @retval DW_INVALID_HANDLE  if given camera handle is invalid
 * @retval DW_SUCCESS  no error
 */
DW_API_PUBLIC
dwStatus dwCameraModel_release(dwCameraModelHandle_t obj);

/**
 * Returns the inverse polynomial used for the inverse distortion model.
 *
 * The inverse distortion model specifies
 *
 *  - the back-projection in Pinhole cameras and in FTheta cameras, and
 *  - the forward-projection in OCam cameras.
 *
 * The back-projection is the mapping of pixel coordinates in the image
 * to optical rays while the forward projection is the mapping of optical rays
 * to pixel coordinates in the image.
 *
 * Note: If the camera model handle has been instantiated with the inverse polynomial,
 *   then the returned polynomial will correspond to this polynomial. Otherwise,
 *   the returned polynomial will be a least squares fit to the inverse polynomial
 *   within the image domain of the mapping.
 *
 * @param[out] invPoly Array of coefficients, lower degrees first.
 * @param[in,out] size Input: size of the polynomial buffer.
 *                     Output: on success, the number of coefficients returned /
 *                             the effective polynomial degree.
 * @param[in] obj Camera handle.
 *
 * @retval DW_INVALID_HANDLE  if given camera handle is invalid
 * @retval DW_INVALID_ARGUMENT  if one of the given pointers is a nullptr
 * @retval DW_OUT_OF_BOUNDS  if given array is not large enough to hold the polynomial
 * @retval DW_SUCCESS  no error
 */
DW_API_PUBLIC
dwStatus dwCameraModel_getInversePolynomial(float32_t* invPoly, size_t* size,
                                            dwCameraModelHandle_t obj);

/**
 * Back-projects a 2D point in pixel coordinates to a 3D optical ray direction.
 * The ray is normalized to have a norm of 1.
 *
 * @param[out] x A pointer to the X coordinate of the ray's direction.
 * @param[out] y A pointer to the Y coordinate of the ray's direction.
 * @param[out] z A pointer to the Z coordinate of the ray's direction.
 * @param[in] u Specifies the horizontal coordinate of the pixel.
 * @param[in] v Specifies the vertical coordinate of the pixel.
 * @param[in] obj Specifies the handle to the calibrated camera model.
 *
 * @retval DW_INVALID_ARGUMENT  if one of the given pointers is a nullptr
 * @retval DW_INVALID_HANDLE  if given camera handle is invalid
 * @retval DW_SUCCESS  no error
 */
DW_API_PUBLIC
dwStatus dwCameraModel_pixel2Ray(float32_t* x, float32_t* y, float32_t* z,
                                 float32_t u, float32_t v,
                                 dwConstCameraModelHandle_t obj);

/**
 * Projects a 3D point in camera coordinates to a 2D pixel position.
 *
 * @param[out] u A pointer to the horizontal coordinate of the pixel.
 * @param[out] v A pointer to the vertical coordinate of the pixel.
 * @param[in] x Specifies the X coordinate of the point.
 * @param[in] y Specifies the Y coordinate of the point.
 * @param[in] z Specifies the Z coordinate of the point.
 * @param[in] obj Specifies the handle to the calibrated camera model.
 *
 * @retval DW_INVALID_ARGUMENT  if one of the given pointers is a nullptr
 * @retval DW_INVALID_HANDLE  if given camera handle is invalid
 * @retval DW_SUCCESS  no error
 */
DW_API_PUBLIC
dwStatus dwCameraModel_ray2Pixel(float32_t* u, float32_t* v,
                                 float32_t x, float32_t y, float32_t z,
                                 dwConstCameraModelHandle_t obj);

/**
 * Checks if the angle of a ray with the camera's optical center is
 * below the *maximum* possible angle of any ray that can be back-projected
 * from valid image domain pixels.
 *
 * Note: This is a fast but approximate check for visibility only, as rays might still
 *   be projected outside of the image bounds even if this check passes,
 *   because only the maximal possible ray angle is considered for this check.
 *   The result of `dwCameraModel_ray2Pixel` still needs to be considered for accurate
 *   inside / outside checks.
 *
 * @param[out] isInsideMaxFOV A pointer to the boolean: `true` if the ray's angle with
 *             the optical center is below or equal to the maximum possible angle of any
 *             ray back-projected from valid image domain pixel, `false` otherwise.
 * @param[in] x Specifies the X coordinate of the point.
 * @param[in] y Specifies the Y coordinate of the point.
 * @param[in] z Specifies the Z coordinate of the point.
 * @param[in] obj Specifies the handle to the calibrated camera model.
 *
 * @retval DW_INVALID_ARGUMENT  if one of the given pointers is a nullptr
 * @retval DW_INVALID_HANDLE  if given camera handle is invalid
 * @retval DW_SUCCESS  no error
 */
DW_API_PUBLIC
dwStatus dwCameraModel_isRayInsideFOV(bool* isInsideMaxFOV,
                                      float32_t x, float32_t y, float32_t z,
                                      dwConstCameraModelHandle_t obj);

/**
 * Gets the horizontal Field of View (FOV) of the calibrated camera, in radians.
 *
 * @param[out] hfov A pointer to the camera horizontal FOV in radians.
 * @param[in] obj Handle to the calibrated camera model.
 *
 * @retval DW_INVALID_ARGUMENT  if one of the given pointers is a nullptr
 * @retval DW_INVALID_HANDLE  if given camera handle is invalid
 * @retval DW_SUCCESS  no error
 */
DW_API_PUBLIC
dwStatus dwCameraModel_getHorizontalFOV(float32_t* hfov,
                                        dwConstCameraModelHandle_t obj);

/**
 * Gets the width and height of the calibrated camera, in pixels.
 *
 * @param[out] width A pointer to the camera image width, in pixels.
 * @param[out] height A pointer to the camera image height, in pixels.
 * @param[in] obj Handle to the calibrated camera model.
 *
 * @retval DW_INVALID_ARGUMENT  if one of the given pointers is a nullptr
 * @retval DW_INVALID_HANDLE  if given camera handle is invalid
 * @retval DW_SUCCESS  no error
 */
DW_API_PUBLIC
dwStatus dwCameraModel_getImageSize(uint32_t* width, uint32_t* height,
                                    dwConstCameraModelHandle_t obj);

/**
 * Sets a new origin for the image and adjusts image scales.
 *
 * Modifies the camera model so that it applies to a transformed version of the original image
 * The image0 is transformed so that pixel `p0= [u0,v0,1]^T` becomes `pt= [ut,vt,1]^T`, i.e.,
 *   `pt = transform * p0`.
 *
 * Currently, transform is limited to an affine matrix containing only scale and translation,
 * i.e., transform =
 *
 *     [s 0 tx]
 *     [0 s ty]
 *     [0 0  1]
 *
 * Examples:
 *
 * **Cropping**: if the original image is cropped by removing the first and last N rows
 * and the first and last M columns.
 *
 *     transform= [1.0, 0.0, N]
 *                [0.0, 1.0, M]
 *                [0.0, 0.0, 1.0]
 *     newsize  = `getImageSize()-dwVector2ui{2*N,2*M}`
 *
 * will produce a camera model that can be used with the cropped image.
 *
 * **Subsampling**: if the original image is subsampled by dropping every second pixel:
 *
 *     x0: 0 1 2 3 4 5
 *     xt: 0   1   2
 *     transform= [0.5, 0.0, 0.0]
 *                [0.0, 0.5, 0.0]
 *                [0.0, 0.0, 1.0]
 *     newSize  = getImageSize()*0.5f
 *
 * will produce a camera model that can be used with the subsampled image.
 *
 * **Subsampling with interpolation**: if the original image is subsampled by interpolating
 * between the pixels:
 *
 *     x0: 0 1 2 3 4 5
 *     xt: 0   1   2
 *     transform= [0.5, 0.0, 0.5]
 *                [0.0, 0.5, 0.5]
 *                [0.0, 0.0, 1.0]
 *     newSize  = getImageSize()*0.5f
 *
 * will produce a camera model that can be used with the subsampled image.
 *
 * @param[in] transform The scale+shift affine transform, in pixels.
 * @param[in] newSize The new size of the image after the transformation, in pixels.
 * @param[in] obj Handle to the calibrated camera model.
 *
 * @retval DW_INVALID_ARGUMENT  if one of the given pointers is a nullptr
 * @retval DW_INVALID_HANDLE  if given camera handle is invalid
 * @retval DW_SUCCESS  no error
 */
DW_API_PUBLIC
dwStatus dwCameraModel_applyImageTransform(const dwMatrix3f* transform,
                                           dwVector2ui newSize,
                                           dwCameraModelHandle_t obj);

/**
* Creates a calibrated camera model polymorphically for a compatible sensor
*
* @param[out] camera A pointer to a handle for the created calibrated camera object.
*                    Has to be released by the user with `dwCameraModel_release`.
* @param[in] sensorId Specifies the index of the camera sensor to create a calibrated camera for.
* @param[in] obj Specifies the rig configuration module handle containing the camera definition.
*
* @note if camera is of ftheta type and windshield parameters are part of the sensor properties of 
*       the respective sensor in the Rig, the windshield will be appended automatically.
*
* @retval DW_INVALID_ARGUMENT  if no calibrated camera can be constructed given the parameters found in the rig
*   or if the camera pointer is invalid
* @retval DW_INVALID_HANDLE  if given context handle is invalid
* @retval DW_CANNOT_CREATE_OBJECT  if given `sensorId` is not a camera sensor
* @retval DW_OUT_OF_BOUNDS  if the sensorId is larger than the number of sensors in the rig
* @retval DW_SUCCESS  no error
*/
DW_API_PUBLIC
dwStatus dwCameraModel_initialize(dwCameraModelHandle_t* camera,
                                  uint32_t sensorId,
                                  dwConstRigHandle_t obj);

/** @} */

#ifdef __cplusplus
}
#endif

#endif // DW_CALIBRATION_CAMERAMODEL_CAMERAMODEL_H_

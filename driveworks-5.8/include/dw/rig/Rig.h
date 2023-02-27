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
// SPDX-FileCopyrightText: Copyright (c) 2015-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Rig Configuration</b>
 *
 * @b Description: This file defines vehicle rig configuration methods.
 */

/**
 * @defgroup rig_configuration_group Rig Configuration Interface
 *
 * @brief Defines rig configurations for the vehicle.
 *
 * @{
 */

#ifndef DW_RIG_RIG_H_
#define DW_RIG_RIG_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>
#include <dw/sensors/Sensors.h>
#include <dw/rig/Vehicle.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup rigconfiguration Rig Configuration
 * @brief Defines vehicle rig configuration.
 *
 * This module manages the rig configuration of the car including vehicle properties, mounted sensors,
 * and their calibration information.
 */

/// Handle representing the Rig interface.
typedef struct dwRigObject* dwRigHandle_t;
typedef struct dwRigObject const* dwConstRigHandle_t;

///////////////////////////////////////////////////////////////////////
// Calibrated cameras

/** Defines the maximum number of sensors in a rig. */
#define DW_MAX_RIG_SENSOR_COUNT 128U

/** Defines the maximum number of cameras in a rig. */
#define DW_MAX_RIG_CAMERA_COUNT DW_MAX_RIG_SENSOR_COUNT

/** Defines the maximum length of a sensor name in a rig. */
#define DW_MAX_RIG_SENSOR_NAME_SIZE 64U

/** Defines the maximum length of a sensor extrinsic profile name. */
#define DW_MAX_EXTRINSIC_PROFILE_NAME_SIZE 64U

/** maximal number of extrinsic profiles per sensor. */
#define MAX_EXTRINSIC_PROFILE_COUNT 3U

/**
* Specifies the supported optical camera models. The models define the mapping between optical rays
* and pixel coordinates, e.g., the intrinsic parameters of the camera.
*/
typedef enum dwCameraModel {
    DW_CAMERA_MODEL_OCAM    = 0,
    DW_CAMERA_MODEL_PINHOLE = 1,
    DW_CAMERA_MODEL_FTHETA  = 2
} dwCameraModel;

/**
 * Defines the number of distortion coefficients for the pinhole camera model.
*/
#define DW_PINHOLE_DISTORTION_LENGTH 3U

/**
 * Configuration parameters for a calibrated pinhole camera.
 *
 * The (forward) projection of a three-dimensional ray to a two-dimensional pixel coordinate
 * is performed in three steps:
 *
 * **Step 1**: Projection of a ray `(x, y, z)` to normalized image coordinates, i.e.,
 *
 *     (xn, yn) = (x/z, y/z) .
 *
 * **Step 2**: Distortion of the normalized image coordinates by a polynomial with coefficients
 * `[k_1, k_2, k_3]` given in dwPinholeCameraConfig::distortion, i.e.,
 *
 *     xd = xn * (1 + k_1 * r^2 + k_2 * r^4 + k_3 * r^6),
 *     yd = yn * (1 + k_1 * r^2 + k_2 * r^4 + k_3 * r^6),
 *
 * whereas
 *     r^2 = (xn^2 + yn^2) .
 *
 * **Step 3**: Mapping of distorted normalized image coordinates `(xd, yd)` to
 * pixel coordinates `(u, v)`, i.e.,
 *
 *     [u] =  [focalX   0] * [xd] + [u0]
 *     [v]    [0   focalY]   [yd]   [v0] .
 */
typedef struct dwPinholeCameraConfig
{
    /** Width of the image (in pixels) */
    uint32_t width;

    /** Height of the image (in pixels) */
    uint32_t height;

    /** U coordinate for the principal point (in pixels) */
    float32_t u0;

    /** V coordinate for the principal point (in pixels) */
    float32_t v0;

    /** Focal length in the X axis (in pixels) */
    float32_t focalX;

    /** Focal length in the Y axis (in pixels) */
    float32_t focalY;

    /**
     * Polynomial coefficients `[k_1, k_2, k_3]` that allow to map undistored, normalized image coordinates
     * (xn, yn) to distorted normalized image coordinates (xd, yd).
     */
    float32_t distortion[DW_PINHOLE_DISTORTION_LENGTH];
} dwPinholeCameraConfig;

/**
 * Defines the number of distortion coefficients for the OCAM camera model.
*/
#define DW_OCAM_POLY_LENGTH 5U

/**
 * DEPRECATED: Configuration parameters for a calibrated ominidirectional (OCam) sphere camera.
 *
 * See the orignal paper for a description of the model:
 * https://sites.google.com/site/scarabotix/ocamcalib-toolbox
 * Scaramuzza, D. (2008). Omnidirectional Vision: from Calibration to Robot Motion Estimation,
 * ETH Zurich, PhD Thesis no. 17635., February 22, 2008.
 *
 * \deprecated OCam support will be removed from Driveworks in an upcoming release. Use FTheta instead.
 */
typedef struct dwOCamCameraConfig
{
    /** Width of the image (in pixels) */
    uint32_t width;

    /** Height of the image (in pixels) */
    uint32_t height;

    /** U coordinate for the principal point (in pixels) */
    float32_t u0;

    /** V coordinate for the principal point (in pixels) */
    float32_t v0;

    /** Affine matrix coefficient C */
    float32_t c;

    /** Affine matrix coefficient D */
    float32_t d;

    /** Affine matrix coefficient E */
    float32_t e;

    /** Pixel2ray polynomial coefficients */
    float32_t poly[DW_OCAM_POLY_LENGTH];
} dwOCamCameraConfig;

/**
 * Defines the number of distortion coefficients for the ftheta camera model.
*/
#define DW_FTHETA_POLY_LENGTH 6U

/**
 * DEPRECATED: Configuration parameters for a calibrated FTheta camera.
 *
 * \deprecated use dwFThetaCameraConfigNew instead, this dwFThetaCameraConfig struct will be removed in an upcoming release.
 */
typedef struct dwFThetaCameraConfig
{
    /** Width of the image (in pixels) */
    uint32_t width;

    /** Height of the image (in pixels) */
    uint32_t height;

    /** U coordinate for the principal point (in pixels) */
    float32_t u0;

    /** V coordinate for the principal point (in pixels) */
    float32_t v0;

    /** Pixel2ray backward projection polynomial coefficients */
    float32_t backwardsPoly[DW_FTHETA_POLY_LENGTH];
} dwFThetaCameraConfig;

/**
 * @brief Type of polynomial stored in FTheta.
 *
 * The FTheta model can either be defined by the
 * forward polynomial that maps a ray angle to a pixel distance
 * from the principal point, or its inverse, the backward polynomial
 * that maps a distance from the principal point to a ray angle.
 *
 * This struct defines which of those two options a polynomial represents.
 */
typedef enum dwFThetaCameraPolynomialType {
    /**
     * Backward polynomial type,
     * mapping pixel distances (offset from principal point)
     * to angles (angle between ray and forward)
     */
    DW_FTHETA_CAMERA_POLYNOMIAL_TYPE_PIXELDISTANCE_TO_ANGLE = 0,

    /**
     * Forward polynomial type,
     * mapping angles (angle between ray and forward direction)
     * to pixel distances (offset from principal point)
     */
    DW_FTHETA_CAMERA_POLYNOMIAL_TYPE_ANGLE_TO_PIXELDISTANCE,
} dwFThetaCameraPolynomialType;

/**
 * Configuration parameters for a calibrated FTheta camera.
 *
 * The FTheta camera model is able to handle both pinhole and fisheye cameras
 * by mapping sight-ray angles to pixel-distances using a polynomial. It incorporates
 * a general linear transformation of pixel offsets to pixel coordinates covering
 * a variety of geometric transformations.
 *
 * The camera model is defined by three major components
 *
 *  - the polynomial `polynomial`
 *  - the principal point `(u0, v0)`, and
 *  - the transformation matrix `A`
 *
 *
 *     A  =  [ c  d ]
 *           [ e  1 ] .
 *
 * The bottom right element of `A` is implicitly set to 1. This is general enough
 * for relevant linear transformations because any matrix can be brought into this form
 * by incorporating an absolute scale into the polynomial.
 *
 *
 * The forward projection of a three-dimensional ray to a two-dimensional pixel coordinates
 * by means of the FTheta camera model can be described in four steps:
 *
 * **Step 1**: Projection of a ray `(x, y, z)` to spherical coordinates `(direction, theta)`:
 *
 *     direction = (x, y) / norm((x, y))
 *     theta = atan2(norm((x, y)), z)
 *
 * **Step 2**: Mapping of the angle `theta` to pixel distances according to the polynomial
 * coefficients `polynomial` and polynomial type `type` :
 *
 *     distance = map_angle_to_pixeldistance( polynomial, type, theta )
 *
 * **Step 3**: Mapping of `distance` and the two-dimensional `direction` to a pixel offset:
 *
 *     offset = distance * direction
 *
 * **Step 4**: Linear transformation of the two-dimensional pixel offset to pixel coordinate `(u, v)`:
 *
 *     [u] =  [c  d] * offset^T + [u0]
 *     [v]    [e  1]              [v0]
 *
 *
 * Conversely, the backward projection of a two-dimensional pixel coordinate to
 * a three-dimensional ray is performed in four steps reversely to the forward projection:
 *
 * **Step 1**: Linear transformation of pixel coordinates `(u, v)` to pixel offset:
 *
 *     offset = inverse(A) * [u - u0]
 *                           [v - v0]
 *
 * **Step 2**: Mapping of the two-dimensional pixel offset to polar coordinates `(direction, distance)`:
 *
 *     direction = offset / norm(offset)
 *     distance = norm(offset)
 *
 * **Step 3**: Mapping of the pixel distance to the sight ray angle `theta` according to the polynomial
 * coefficients `polynomial` and the polynomial type `type` :
 *
 *     theta = map_pixel_distance_to_angle( polynomial, type, distance )
 *
 * **Step 4**: Computation of the sight ray `(x, y, z)` based on polar coordinates `(direction, angle)`:
 *
 *     (x, y) = sin(theta) * direction
 *     z = cos(theta)
 *
 *
 * The functions `map_angle_to_pixel_distance(.)` and `map_pixel_distance_to_angle(.)` depend on the
 * polynomial coefficients `polynomial` and the type of the polynomial, i.e.,
 *
 * - whenever the type of polynomial corresponds to the requested direction,
 *   the function corresponds to a simple evaluation of the polynomial:
 *
 *        map_a_to_b( polynomial, a_to_b, x ) = polynomial[0] + polynomial[1] * x + ... + polynomial[DW_FTHETA_POLY_LENGTH - 1] * x^(DW_FTHETA_POLY_LENGTH - 1)
 *
 * - whenever the type of polynomial is the inverse to the requested direction,
 *   the function is equivalent to the inverse of the polynomial:
 *
 *        map_a_to_b( polynomial, b_to_a, x ) = y
 *
 *   with `map_b_to_a( polynomial, b_to_a, y ) == x`.
 *   The solution is computed via iterative local inversion. The valid ranges are defined by the
 *   camera's resolution and field of view.
 *
 *
 * In literature, the closest description is found in [Courbon et al, 2007], TABLE II:
 *
 *   [Courbon et al, 2007]: Courbon, J., Mezouar, Y., Eckt, L., and Martinet, P. (2007, October).
 *   A generic fisheye camera model for robotic applications. In Intelligent Robots and Systems, 2007.
 *   IROS 2007, pp. 1683–1688.
 */
typedef struct dwFThetaCameraConfigNew
{
    /** Width of the image (in pixels) */
    uint32_t width;

    /** Height of the image (in pixels) */
    uint32_t height;

    /**
     *  Principal point coordinates: indicating the horizontal / vertical image
     *  coordinates of the principal point relative to the origin of the image read-out area.
     *
     *  The top-left corner of the read-out area is defined to have image coordinates [-0.5, -0.5],
     *  meaning that the center of the first pixel (with interger-indices [0, 0] and unit extend)
     *  corresponds to the point with image coordinates [0.0, 0.0].
     *
     *  U coordinate for the principal point (in pixels)
     */
    float32_t u0;

    /** V coordinate for the principal point (in pixels) */
    float32_t v0;

    /**
     *  Linear pixel transformation matrix coefficient c (top left element)
     *  If all `c`, `d`, and `e` are set to 0.0f, then the top lef element
     *  of the matrix will be set to 1.0f instead, creating identity as the
     *  linear transformation.
     */
    float32_t c;

    /** Linear pixel transformation coefficient d (top right element). */
    float32_t d;

    /** Linear pixel transformation coefficient e (bottom left element). */
    float32_t e;

    /**
     *  Polynomial describing either the mapping of angles to pixel-distances or
     *  the mapping of pixel-distances to angles, in dependence of the field `polynomialType`.
     *
     *  The polynomial function is defined
     *  as `f(x) = polynomial[0] + polynomial[1] * x + ... + polynomial[DW_FTHETA_POLY_LENGTH - 1] * x^(DW_FTHETA_POLY_LENGTH - 1)`
     */
    float32_t polynomial[DW_FTHETA_POLY_LENGTH];

    /**
     *  Defines whether the polynomial parameter
     *  either map angles to pixel-distances (called forward direction)
     *  or map pixel-distances to angles (called backward direction).
     */
    dwFThetaCameraPolynomialType polynomialType;
} dwFThetaCameraConfigNew;

/**
 * Configuration parameters for a calibrated stereographic camera.
 *
 * The stereographic camera describes a projection of a sphere onto a plane
 * located at a certain distance in viewing (=z) direction.
 *
 * The model is described by three major components:
 *
 *  - the image size `(width, height)`,
 *  - the principal point `(u0, v0)`, and
 *  - the horizontal field of view `hFOV`.
 *
 *
 * The horizontal field of view determines the radius `r` of the sphere
 * used in the projection:
 *
 *     r = 0.5 / tan(hFOV/4)
 *
 * The radius determines
 *
 *  - the distance of the image plane to the origin of the camera coordinate system and
 *  - the distance of the pole of the projection to the origin of the camera coordinate system.
 *
 *
 * The forward projection of a three-dimensional ray to a two-dimensional pixel coordinate
 * by means of the stereographic camera model can be described in two steps:
 *
 * **Step 1**: Projection of a ray `(x, y, z)` of length 1 onto a normalized coordinate `(xn, yn)`
 * on the image plane at distance r, i.e.,
 *
 *     (xn, yn) = 2 * r * (x, y) / (1 + z)
 *
 * **Step 2**: Scaling and shift to pixel coordinate `(u, v)`, i.e.,
 *
 *     u = xn * width / 2 + u0
 *     v = yn * height / 2 + v0
 *
 *
 * Conversely, the backward projection of a two-dimensional pixel coordinate to
 * a three-dimensional ray is performed in two steps reversely to the forward projection:
 *
 * **Step 1**: Scaling and shift of pixel coordinate `(u, v)` to normalized image coordinate `(xn, yn)`, i.e.,
 *
 *     xn = (u - u0) / (width / 2)
 *     yn = (v - v0) / (height / 2)
 *
 * **Step 2**: Projection of normalized image coordinates to three-dimensional unit ray `(x, y, z)`, i.e.,
 *
 *     (x, y, z) = (4 * r * xn, 4 * r * yn, - xn^2 - yn^2 + 4 * r^2)) / (xn^2 + yn^2 + 4 * r^2)
 */
typedef struct dwStereographicCameraConfig
{
    /** Width of the image (in pixels) */
    uint32_t width;

    /** Height of the image (in pixels) */
    uint32_t height;

    /** U coordinate for the principal point (in pixels) */
    float32_t u0;

    /** V coordinate for the principal point (in pixels) */
    float32_t v0;

    /** Horizontal FOV (in radians) */
    float32_t hFOV;
} dwStereographicCameraConfig;

/**
* Initializes the Rig Configuration module from a file.
*
* @note: Any relative file-system reference will be relative to the rig file location.
*
* @param[out] obj A pointer to the Rig Configuration handle for the created module.
* @param[in] ctx Specifies the handler to the context under which the Rigconfiguration module is created.
* @param[in] configurationFile The path of a rig file that contains the rig configuration.
                               Typically produced by the DriveWorks calibration tool.
*
* @retval DW_INVALID_ARGUMENT when the rig configuration handle is NULL or if the json file has no extension
* @retval DW_INVALID_HANDLE when the context handle is invalid, i.e null or wrong type
* @retval DW_FILE_INVALID when the json file is invalid
* @retval DW_FILE_NOT_FOUND when the json file cannot be found
* @retval DW_INTERNAL_ERROR when internal error happens
* @retval DW_BUFFER_FULL when too many extrinsic profiles are available (> 3)
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_initializeFromFile(dwRigHandle_t* const obj,
                                  dwContextHandle_t const ctx,
                                  char8_t const* const configurationFile);

/**
* Initializes the Rig Configuration module from a string.
*
* @param[out] obj A pointer to the Rig Configuration handle for the created module.
* @param[in] ctx Specifies the handler to the context under which the Rigconfiguration module is created.
* @param[in] configurationString A pointer to a JSON string that contains the rig configuration.
*                                Typically produced by the DriveWorks calibration tool.
* @param[in] relativeBasePath A base path all relative file references in the rig will be resolved with respect to.
*                             If NULL, then the current working directory of the process will be used implicitly.
*
* @retval DW_INVALID_ARGUMENT when the rig configuration handle is NULL or if the json file has no extension
* @retval DW_INVALID_HANDLE when the context handle is invalid, i.e null or wrong type
* @retval DW_INTERNAL_ERROR when internal error happens
* @retval DW_BUFFER_FULL when too many extrinsic profiles are available (> 3)
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_initializeFromString(dwRigHandle_t* const obj,
                                    dwContextHandle_t const ctx,
                                    char8_t const* const configurationString,
                                    char8_t const* const relativeBasePath);

/**
* Resets the Rig Configuration module.
*
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_HANDLE when the rig handle is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
**/
DW_API_PUBLIC
dwStatus dwRig_reset(dwRigHandle_t const obj);

/**
* Releases the Rig Configuration module.
*
* @param[in] obj The Rig Configuration module handle.
*
* @retval DW_INVALID_HANDLE when the configuration handle is invalid , i.e NULL or wrong type
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_release(dwRigHandle_t const obj);

/**
* DEPRECATED: Gets the properties of a passenger car vehicle.
* @deprecated Use dwRig_getGenericVehicle.
*
* @param[out] vehicle A pointer to the struct holding vehicle properties. The returned pointer is valid
* until module reset or release is called.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the rig configuration handle is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_NOT_AVAILABLE when no vehicle in configuration is available
* @retval DW_SUCCESS when operation succeeded
*
**/
DW_API_PUBLIC
dwStatus dwRig_getVehicle(dwVehicle const** const vehicle, dwConstRigHandle_t const obj);

/**
* Gets the properties of a generic vehicle (car or truck).
*
* @param[out] vehicle A pointer to the struct to be filled with vehicle properties.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the rig configuration handle is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_NOT_AVAILABLE when no generic vehicle in configuration is available
* @retval DW_SUCCESS when operation succeeded
*
**/
DW_API_PUBLIC
dwStatus dwRig_getGenericVehicle(dwGenericVehicle* const vehicle, dwConstRigHandle_t const obj);

/**
* DEPRECATED: Sets the properties of a passenger car vehicle.
* @deprecated Use dwRig_setGenericVehicle.
*
* @param[in] vehicle A pointer to the struct holding vehicle properties.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the rig configuration handle is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_NOT_AVAILABLE when no vehicle in configuration is available
* @retval DW_SUCCESS when operation succeeded
*
**/
DW_API_PUBLIC
dwStatus dwRig_setVehicle(dwVehicle const* const vehicle, dwRigHandle_t const obj);

/**
* Sets the properties of a generic vehicle (car or truck).
*
* @param[in] vehicle A pointer to the struct holding vehicle properties.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the rig configuration handle is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_NOT_AVAILABLE when no generic vehicle in configuration is available
* @retval DW_SUCCESS when operation succeeded
*
**/
DW_API_PUBLIC
dwStatus dwRig_setGenericVehicle(dwGenericVehicle const* const vehicle, dwRigHandle_t const obj);

/**
* Gets the number of vehicle IO sensors.
*
* @param[out] vioConfigCount A pointer to the number of vehicle IO sensors in the Rig Configuration.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the rig configuration handle is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
**/
DW_API_PUBLIC
dwStatus dwRig_getVehicleIOConfigCount(uint32_t* const vioConfigCount,
                                       dwConstRigHandle_t const obj);

/**
* Gets the number of all available sensors.
*
* @param[out] sensorCount A pointer to the number of sensors in the rig configuration.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
**/
DW_API_PUBLIC
dwStatus dwRig_getSensorCount(uint32_t* const sensorCount,
                              dwConstRigHandle_t const obj);

/**
* Find number of sensors of a given type.
*
* @param[out] sensorCount Return number of sensors available of the given type
* @param[in] sensorType Type of the sensor to query
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT `given pointer is null
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorCountOfType(uint32_t* const sensorCount,
                                    dwSensorType const sensorType,
                                    dwConstRigHandle_t const obj);

/**
* Gets the protocol string of a sensor. This string can be used in sensor creation or to identify
* the type of a sensor.
*
* @param[out] sensorProtocol A pointer to the pointer to the protocol of the sensor, for example, camera.gmsl. The returned pointer is valid
* until module reset or release is called.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the pointer to the pointer of sensor protocol  is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_SUCCESS when operation succeeded
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorProtocol(char8_t const** const sensorProtocol,
                                 uint32_t const sensorId,
                                 dwConstRigHandle_t const obj);

/**
* Gets the parameter string for a sensor. This string can be used in sensor creation.
*
* @param[out] sensorParameter A pointer to the pointer to the parameters of the sensor, for example camera driver and csi port. The returned
* pointer is valid until module reset or release is called.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the pointer to the pointer of sensor parameters is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_SUCCESS when operation succeeded
*/
DW_API_PUBLIC dwStatus dwRig_getSensorParameter(char8_t const** const sensorParameter,
                                                uint32_t const sensorId,
                                                dwConstRigHandle_t const obj);

/**
* Sets the parameter string for a sensor. This string can be used in sensor creation.
*
* @param[in] sensorParameter string representing sensor parameters, for example camera driver and csi port.
* Maximal length is limited to 512.
* @param[in] sensorId Specifies the index of the sensor of which to set sensor parameter.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the sensor parameter string is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the sensor to be updated is more than MAX_SENSOR_COUNT
* @retval DW_SUCCESS when operation succeeded
*/
DW_API_PUBLIC dwStatus dwRig_setSensorParameter(char8_t const* const sensorParameter,
                                                uint32_t const sensorId,
                                                dwRigHandle_t const obj);

/**
* Gets the parameter string for a sensor with any path described by file=,video=,timestamp= property modified
* to be in respect to the current rig file's directory (if initializing a rig from file), or in respect to the
* relativeBasePath (when initializing a rig from string). For example, given a rig.json file stored at
* this/is/rig.json with a virtual sensor pointing to file=video.lraw, the call to this function will
* return sensor properties modified as file=this/is/video.lraw.
*
* @param[out] sensorParameter Sensor parameters with modified path inside of file=,video=,timestamp= returned
* here.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the Rig Configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the pointer to the pointer of sensor parameters is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_SUCCESS when operation succeeded
*/
DW_API_PUBLIC dwStatus dwRig_getSensorParameterUpdatedPath(char8_t const** const sensorParameter,
                                                           uint32_t const sensorId,
                                                           dwConstRigHandle_t const obj);

/**
* Gets the sensor to rig transformation for a sensor. This transformation relates the sensor and
* the rig coordinate system to each other. For example, the origin in sensor coordinate system is
* the position of the sensor in rig coordinates. Also, if the sensor's type doesn't support extrinsics,
* the identity transformation will be returned.
*
* @param[out] transformation A pointer to the transformation from sensor to rig coordinate system.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the transformation pointer is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_SUCCESS when operation succeeded
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorToRigTransformation(dwTransformation3f* const transformation,
                                            uint32_t const sensorId,
                                            dwConstRigHandle_t const obj);

/**
* Gets the sensor FLU to rig transformation for a sensor. This transformation relates the sensor
* FLU and the rig coordinate system to each other. For example, the origin in sensor coordinate
* system is the position of the sensor in rig coordinates.
*
* @param[out] transformation A pointer to the transformation from sensor to rig coordinate system.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the transformation pointer is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_SUCCESS when operation succeeded
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorFLUToRigTransformation(dwTransformation3f* const transformation,
                                               uint32_t const sensorId,
                                               dwConstRigHandle_t const obj);

/**
* Gets the nominal sensor to rig transformation for a sensor.  This transform differs from transform T
* provided by getSensorToRigTransformation() in that it represents a static reference transformation
* from factory calibration and/or mechanical drawings, whereas T can change over time. Also, if the sensor's
* type doesn't support extrinsics, the identity transformation will be returned.
*
* @param[out] transformation A pointer to the nominal transformation from sensor to rig coordinate system.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the transformation pointer is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_getNominalSensorToRigTransformation(dwTransformation3f* const transformation,
                                                   uint32_t const sensorId,
                                                   dwConstRigHandle_t const obj);

/**
* Gets the sensor to sensor transformation for a pair of sensors. This transformation relates the first and
* second sensor coordinate systems to each other. Identity transformations are used for sensors that don't
* support a native extrinsic frame.
*
* @param[out] transformation A pointer to the transformation from sensor to sensor coordinate system.
* @param[in] sensorIdFrom Specifies the index of the source sensor.
* @param[in] sensorIdTo Specifies the index of the destination sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the transformation pointer is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorToSensorTransformation(dwTransformation3f* const transformation,
                                               uint32_t const sensorIdFrom,
                                               uint32_t const sensorIdTo,
                                               dwConstRigHandle_t const obj);

/**
* Gets the nominal sensor to sensor transformation for a pair of sensors.  This transform differs from transform T
* provided by getSensorToSensorTransformation() in that it represents a static reference transformation
* from factory calibration and/or mechanical drawings, whereas T can change over time. Identity transformations
* are used for sensors that don't support a native extrinsic frame.
*
* @param[out] transformation A pointer to the nominal transformation from sensor to sensor coordinate system.
* @param[in] sensorIdFrom Specifies the index of the source sensor.
* @param[in] sensorIdTo Specifies the index of the destination sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the transformation pointer is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_getNominalSensorToSensorTransformation(dwTransformation3f* const transformation,
                                                      uint32_t const sensorIdFrom,
                                                      uint32_t const sensorIdTo,
                                                      dwConstRigHandle_t const obj);

/**
* Sets the sensor to rig transformation for a sensor.
* @see dwRig_getSensorToRigTransformation.
*
* @param[in] transformation A pointer to the transformation from sensor to rig coordinate system.
* @param[in] sensorId Specifies the index of the updates sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the transformation pointer is NULL
* @retval DW_INVALID_HANDLE when the transformation pointer is NULL
* @retval DW_CALL_NOT_ALLOWED when the sensor's type doesn't support extrinsics
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_setSensorToRigTransformation(dwTransformation3f const* const transformation,
                                            uint32_t const sensorId,
                                            dwRigHandle_t const obj);

/**
* Gets the name of a sensor as given in the configuration. For example, "Front Camera".
*
* @param[out] sensorName A pointer to the name of the sensor. The pointer is valid until module reset or release is
* called.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the sensor pointer is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_SUCCESS when operation succeeded
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorName(char8_t const** const sensorName,
                             uint32_t const sensorId,
                             dwConstRigHandle_t const obj);

/**
* Gets path to sensor recording. The call is only valid for virtual sensors.
*
* @param[out] dataPath A pointer to the path with sensor data. The pointer is valid until module reset or release is
* called.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null
* @retval DW_NOT_AVAILABLE when data path for the given sensor is not available
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorDataPath(char8_t const** const dataPath,
                                 uint32_t const sensorId,
                                 dwConstRigHandle_t const obj);

/**
* Gets path to camera timestamp file. The call is only relevant for virtual h264/h265 cameras.
* Otherwise returned value is always nullptr.
*
* @param[out] timestampPath A pointer to the path containing timestamp data.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT  when given pointer is null
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_getCameraTimestampPath(char8_t const** const timestampPath,
                                      uint32_t const sensorId,
                                      dwConstRigHandle_t const obj);

/**
* Returns property stored inside of a sensor. Properties are stored in name=value pairs and implement
* properties which are specific for a certain sensor in a generic way.
* For example a camera might store calibration data there, an IMU might store bias values there, etc.
*
* @param[out] propertyValue A pointer to return the value of a certain property
* @param[in] propertyName Name of the property to retrieve value from
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null or sensorId doesn't exist
* @retval DW_NOT_AVAILABLE when a certain property is not available in the rig configration
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorPropertyByName(char8_t const** const propertyValue,
                                       char8_t const* const propertyName,
                                       uint32_t const sensorId,
                                       dwConstRigHandle_t const obj);

/**
* Overwrite content of an existing sensor property. If property does not exists, it will be added.
* Properties are stored as name=value pairs.
*
* @param[in] propertyValue Value of the property to be changed to. Maximal length limited to 256 characters.
* @param[in] propertyName Name of the property to change
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null or sensorId doesn't exist
* @retval DW_BUFFER_FULL when there are no more space for new properties, max 32
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*/
DW_API_PUBLIC
dwStatus dwRig_addOrSetSensorPropertyByName(char8_t const* const propertyValue,
                                            char8_t const* const propertyName,
                                            uint32_t const sensorId,
                                            dwRigHandle_t const obj);
/**
* Returns property stored inside of rig. Properties are stored in name=value pairs and implement
* properties which are specific for the rig in a generic way.
* For example a particular sensor layout or configuration
*
* @param[out] propertyValue A pointer to return the value of a certain property
* @param[in] propertyName Name of the property to retrieve value from
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null
* @retval DW_NOT_AVAILABLE when a certain property is not available in the rig configration
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_getPropertyByName(char8_t const** const propertyValue,
                                 char8_t const* const propertyName,
                                 dwConstRigHandle_t const obj);

/**
* Overwrite content of an existing rig property. If property does not exists, it will be added.
* Properties are stored as name=value pairs.
*
* @param[in] propertyValue Value of the property to be changed to. Maximal length limited to 256 characters.
* @param[in] propertyName Name of the property to change
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null
* @retval DW_BUFFER_FULL when there are no more space for new properties, max 32
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*/
DW_API_PUBLIC
dwStatus dwRig_addOrSetPropertyByName(char8_t const* const propertyValue,
                                      char8_t const* const propertyName,
                                      dwRigHandle_t const obj);

/**
* Finds the sensor with the given name and returns its index.
*
* @param[out] sensorId The index of the matching sensor (unchanged if the function fails).
* @param[in] sensorName The sensor name to search for. If the character '*' is found, only the characters before are compared for a match.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null
* @retval DW_NOT_AVAILABLE when no sensor matches the name
* @retval DW_INVALID_HANDLE when the rig configuration module handle is invalid, i.e NULL or wrong type
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_findSensorByName(uint32_t* const sensorId,
                                char8_t const* const sensorName,
                                dwConstRigHandle_t const obj);
/**
* Finds a sensor with the given vehicleIO ID and returns the index.
*
* @param[out] sensorId The Specifies the index of the matching sensor. Undefined if the function fails.
* @param[in] vehicleIOId The vehicleIO ID to search for.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null
* @retval DW_NOT_AVAILABLE when no sensor matches the vehicle IO ID
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_findSensorIdFromVehicleIOId(uint32_t* const sensorId,
                                           uint32_t const vehicleIOId,
                                           dwConstRigHandle_t const obj);

/**
* Finds the absolute sensor index of the Nth sensor of a given type.
*
* @param[out] sensorId The index of the matching sensor (unchanged if the function fails).
* @param[in] sensorType The type of the sensor to search for.
* @param[in] sensorTypeIndex The idx of the sensor within that type.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null
* @retval DW_NOT_AVAILABLE when no sensor matches the type
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_findSensorByTypeIndex(uint32_t* const sensorId,
                                     dwSensorType const sensorType,
                                     uint32_t const sensorTypeIndex,
                                     dwConstRigHandle_t const obj);

/**
* Returns the type of sensor based upon the sensorID sent into the method
*
* @param[out] sensorType A pointer to return the type of sensor
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when given pointer is null or sensorId doesn't exist
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_getSensorType(dwSensorType* const sensorType,
                             uint32_t const sensorId,
                             dwConstRigHandle_t const obj);

/**
* Gets the model type of the camera intrinsics. The supported models are OCam, Pinhole, and FTheta.
*
* @param[out] cameraModel A pointer to the model type for the camera intrinsics.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the pointer to the model type is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_NOT_AVAILABLE when the sensor has no camera model
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_getCameraModel(dwCameraModel* const cameraModel,
                              uint32_t const sensorId,
                              dwConstRigHandle_t const obj);

/**
* Gets the parameters of the Pinhole camera model.
*
* @param[out] config A pointer to the configuration of the camera intrinsics.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the config pointer is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_NOT_AVAILABLE when the sensor has no camera model
* @retval DW_SUCCESS when operation succeeded
*/
DW_API_PUBLIC
dwStatus dwRig_getPinholeCameraConfig(dwPinholeCameraConfig* const config,
                                      uint32_t const sensorId,
                                      dwConstRigHandle_t const obj);

/**
* Gets the parameters of the OCam camera model.
*
* @note This method clears the data passed in config in order to check if data was set.
*
* @param[out] config A pointer to the configuration of the camera intrinsics.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the config pointer is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_NOT_AVAILABLE
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
DW_DEPRECATED("OCam support will be removed from Driveworks in an upcmming release. Use FTheta instead.")
dwStatus dwRig_getOCamCameraConfig(dwOCamCameraConfig* const config,
                                   uint32_t const sensorId,
                                   dwConstRigHandle_t const obj);

/**
* Gets the parameters of the FTheta camera model.
*
* @note This method clears the data passed in config in order to check if data was set.
*
* @param[out] config A pointer to the configuration of the camera intrinsics.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the config pointer is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_NOT_AVAILABLE when the sensor has no camera model
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
DW_DEPRECATED("dwRig_getFThetaCameraConfig is replaced by dwRig_getFThetaCameraConfigNew.")
dwStatus dwRig_getFThetaCameraConfig(dwFThetaCameraConfig* const config,
                                     uint32_t const sensorId,
                                     dwConstRigHandle_t const obj);

/**
* Gets the parameters of the FTheta camera model.
*
* @note This method clears the data passed in config in order to check if data was set.
*
* @param[out] config A pointer to the configuration of the camera intrinsics.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the config pointer is NULL
* @retval DW_INVALID_HANDLE when the rig configuration handle is invalid, i.e null or wrong type
* @retval DW_OUT_OF_BOUNDS when the index of the queried sensor is more than MAX_SENSOR_COUNT
* @retval DW_NOT_AVAILABLE when the sensor has no camera model
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_getFThetaCameraConfigNew(dwFThetaCameraConfigNew* const config,
                                        uint32_t const sensorId,
                                        dwConstRigHandle_t const obj);

/**
* Sets the parameters of the pinhole camera model.
*
* @param[in] config A pointer to the configuration of the camera intrinsics.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the config pointer is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_NOT_AVAILABLE when the sensor has no camera model
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_setPinholeCameraConfig(dwPinholeCameraConfig const* const config,
                                      uint32_t const sensorId,
                                      dwRigHandle_t const obj);

/**
* Sets the parameters of the OCam camera model.
*
* @param[in] config A pointer to the configuration of the camera intrinsics.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the config pointer is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_CANNOT_CREATE_OBJECT when the sensor has no camera model
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
DW_DEPRECATED("OCam support will be removed from Driveworks in an upcoming release. Use FTheta instead.")
dwStatus dwRig_setOCamCameraConfig(dwOCamCameraConfig const* const config,
                                   uint32_t const sensorId,
                                   dwRigHandle_t const obj);

/**
* Sets the parameters of the FTheta camera model.
*
* @param[in] config A pointer to the configuration of the camera intrinsics.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the config pointer is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_CANNOT_CREATE_OBJECT when the sensor has no camera model
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
DW_DEPRECATED("dwRig_setFThetaCameraConfig is replaced by dwRig_setFThetaCameraConfigNew.")
dwStatus dwRig_setFThetaCameraConfig(dwFThetaCameraConfig const* const config,
                                     uint32_t const sensorId,
                                     dwRigHandle_t const obj);

/**
* Sets the parameters of the FTheta camera model.
*
* @param[in] config A pointer to the configuration of the camera intrinsics.
* @param[in] sensorId Specifies the index of the queried sensor.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the config pointer is NULL
* @retval DW_INVALID_HANDLE when at least one of the input handles is invalid, i.e null or wrong type
* @retval DW_CANNOT_CREATE_OBJECT when the sensor has no camera model
* @retval DW_SUCCESS when operation succeeded
*
*/
DW_API_PUBLIC
dwStatus dwRig_setFThetaCameraConfigNew(dwFThetaCameraConfigNew const* const config,
                                        uint32_t const sensorId,
                                        dwRigHandle_t const obj);

/**
* This method serializes the rig-configuration object to a human-readable rig-configuration file.
* The output file contains the full state of the rig-configuration and can again be loaded with
* dwRig_initializeFromFile().
*
* The serialization format is selected based on the file name extension; currently supported extensions are json.
*
* @param[in] configurationFile The name of the file to serialize to. It's extension is used to
*                              select the serialization format. This method will overwrite the file if it exists.
* @param[in] obj Specifies the rig configuration module handle.
*
* @retval DW_INVALID_ARGUMENT when the configurationFile pointer is invalid,
*                               or if the serialization format is not supported
* @retval DW_INVALID_HANDLE when provided RigConfigurationHandle handle is invalid.
* @retval DW_FILE_INVALID in case of error during serialization.
* @retval DW_SUCCESS when operation succeeded
*
**/
DW_API_PUBLIC
dwStatus dwRig_serializeToFile(char8_t const* const configurationFile,
                               dwConstRigHandle_t const obj);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_RIG_RIG_H_

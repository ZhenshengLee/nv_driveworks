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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Rig Configuration types for public</b>
 *
 * @b Description: This file defines the types for vehicle rig configuration methods.
 */

#ifndef DW_RIG_RIG_TYPES_H_
#define DW_RIG_RIG_TYPES_H_

#include <dw/core/base/Types.h>

#ifdef __cplusplus
extern "C" {
#endif

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
#define DW_MAX_EXTRINSIC_PROFILE_COUNT 3U

/** index of the default extrinsic profile */
#define DW_DEFAULT_EXTRINSIC_PROFILE_INDEX 0U

/**
* Specifies the supported optical camera models. The models define the mapping between optical rays
* and pixel coordinates, e.g., the intrinsic parameters of the camera.
*/
typedef enum dwCameraModel {

    DW_CAMERA_MODEL_PINHOLE = 1,
    DW_CAMERA_MODEL_FTHETA  = 2
} dwCameraModel;

/**
 * Defines the number of distortion coefficients for the pinhole camera model.
*/
#define DW_PINHOLE_DISTORTION_LENGTH 3U

/*
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
 * Defines the number of distortion coefficients for the ftheta camera model.
*/
#define DW_FTHETA_POLY_LENGTH 6U

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
 */
/*
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
 * **optional - step 5**: Tangential distortion
 *
 *     If `hasTangentials == true` tangential distorion of the coefficients t0 and t1 will be applied.
 *     This is only allowed for cameras with fov < Pi, i.e. z > 0.
 *     Note: Forward/backward tangential coefficients differ numerically. The direction of the tangential contribution is
 *     the same as indicated in "polynomialType"
 *
 *     [u`] = [u] + (2t0 x/z *y/z + t1/(z**2)(y**2 + 3*x**3))
 *     [v`] = [v] + (2t1 x/z *y/z + t0/(z**2)(x**2 + 3*y**3))
 *
 * Conversely, the backward projection of a two-dimensional pixel coordinate to
 * a three-dimensional ray is performed in four steps reversely to the forward projection:
 *
 * **Step 1**: Linear transformation of pixel coordinates `(u, v)` to pixel offset:
 *
 *     offset = inverse(A) * [u - u0]
 *                           [v - v0]
 * **optional - step 2**: Tangential distortion
 *
 *     If `hasTangentials == true` tangential distorion of the inverse coefficients t0 and t1 will be applied.
 *     Note: Forward/backward tangential coefficients differ numerically. The direction of the tangential contribution is
 *     the same as indicated in "polynomialType". 
 *     The contribution is applied by redefining step 1 as:
 *
 *     
 *     offset = inverse(A) * [u - u0 - (2t0 u * v + t1(v**2 + 3*u**3)]
 *                           [v - v0 - (2t1 u * v + t0(u**2 + 3*v**3)]
 *
 * **Step 3**: Mapping of the two-dimensional pixel offset to polar coordinates `(direction, distance)`:
 *
 *     direction = offset / norm(offset)
 *     distance = norm(offset)
 *
 * **Step 4**: Mapping of the pixel distance to the sight ray angle `theta` according to the polynomial
 * coefficients `polynomial` and the polynomial type `type` :
 *
 *     theta = map_pixel_distance_to_angle( polynomial, type, distance )
 *
 * **Step 5**: Computation of the sight ray `(x, y, z)` based on polar coordinates `(direction, angle)`:
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
typedef struct dwFThetaCameraConfig
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

    /** Indicates if the camera has a tangential contribution */
    bool hasTangentials;

    /** 
    * Tangential contribution describing an non radial symmetric distortion effect
    * coming from a rotational lens displacement.
    * 
    *     [u`] = [u] + (2t0 x/z *y/z + t1/(z**2)(y**2 + 3*x**3))
    *     [v`] = [v] + (2t1 x/z *y/z + t0/(z**2)(x**2 + 3*y**3))
    *
    * Oth tangential contribution
    */
    float32_t t0;

    /** 1st tangential contribution */
    float32_t t1;

} dwFThetaCameraConfig;

/**
 * Configuration parameters for a calibrated stereographic camera.
 *
 * The stereographic camera describes a projection of a sphere onto a plane
 * located at a certain distance in viewing (=z) direction.
 *
 */
/*
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
 * Defines the number of coefficients for the windshield parameters in the horizontal direction.
*/
#define DW_WINDSHIELD_HORIZONTAL_POLY_LENGTH 6U
/**
 * Defines the number of coefficients for the windshield parameters in the vertical direction.
*/
#define DW_WINDSHIELD_VERTICAL_POLY_LENGTH 15U

/**
 * @brief Type of polynomial stored in Windshield Model.
 *
 * The Windshield model can either be defined by
 * the forward polynomial that maps a ray from non-camera side to a ray on the camera side,
 * or its inverse, the backward polynomial that maps a ray from camera side to a ray on the non-camera side.
 *
 * This struct defines which of those two options a polynomial represents.
 */
typedef enum dwWindshieldPolynomialType {
    /**
     * Forward polynomial type,
     * mapping ray distortion by windshield from the non camera side to camera side
     */
    DW_WINDSHIELD_POLYNOMIAL_TYPE_FORWARD = 0,

    /**
     * Backward polynomial type,
     * mapping ray distortion by windshield from the camera side to non-camera side
     */
    DW_WINDSHIELD_POLYNOMIAL_TYPE_BACKWARD = 1,

    // polynomial type count
    DW_WINDSHIELD_POLYNOMIAL_TYPE_COUNT = 2,

    /// Force enum to be 32 bits
    DW_WINDSHIELD_POLYNOMIAL_TYPE_FORCE_32 = 0x7fffffff,
} dwWindshieldPolynomialType;

/**
 * Configuration parameters for a calibrated windshield model.
 *
 * The windshield model is able to handle the ray distortion effect
 * by mapping sight-ray to camera-ray using a set of polynomials.
 *
 */
/*
 * The windshield model is defined by two major components
 *  - the horizontal polynomial
 *  - the vertical polynomial
 *
 *
 * The forward mapping of a three-dimensional ray(x,y,z) to a distorted three-dimensional ray (x',y',z')
 * by means of the windshield model can be described as below:
 *
 * **Step 1**: conversion of a ray `(x, y, z)` to  `(phi, theta)` defined as below:
 *
 *     phi = asin(x/norm((x, y, z))
 *     theta = asin(y/norm((x, y, z))
 *
 * **Step 2**: Mapping of the angle phi to phi', theta to theta' according to the 2D polynomial
 * coefficients `polynomial` and polynomial type `type` :
 *
 *     phi' = windshield_horizontal_mapping( polynomial, type, phi )
 *     theta' = windshield_vertical_mapping( polynomial, type, theta )
 *
 * **Step 3**: Inverse conversion of the ray from (phi', theta') to a ray (x',y',z'):
 *     x' = sin(phi')
 *     y' = sin(theta')
 *     z' = sqrt(1 - x'^2 -y'^2)
 */
typedef struct dwWindshieldModelConfig
{
    /**
     *  Windshield polynomial describing the mapping of horizontal angle(phi) from ray-from-object to ray-into-camera
     *  The polynomial is 2 dimension with highest degree as DEGREE. The number of distortion coefficients equals to (DEGREE + 1) * (DEGREE + 2) / 2
     *
     *  For a ray(x, y, z), horizontal angle phi is defined as: phi = asin(x/sqrt(x^2+y^2+z^2)), vertical angle theta is defined as: theta = asin(y/sqrt(x^2+y^2+z^2))
     *
     *  The windshield horizontal polynomial function is defined
     *  as phi' = windshield_horizontal_polynomial[0] + ... + windshield_horizontal_polynomial[j * (2 * DEGREE + 3 - j) / 2 + i]* phi^(i)* theta(j) + ...
     *                + windshield_horizontal_polynomial[DW_WINDSHIELD_HORIZONTAL_POLY_LENGTH] * theta^(DEGREE)
     */
    float32_t horizontalPolynomial[DW_WINDSHIELD_HORIZONTAL_POLY_LENGTH];

    /**
     *  Windshield polynomial describing the mapping of vertical angle(theta) from ray-from-object to ray-into-camera
     *  The polynomial is 2 dimension with highest degree as DEGREE. The number of distortion coefficients equals to (DEGREE + 1) * (DEGREE + 2) / 2
     *
     *  For a ray(x, y, z), horizontal angle phi is defined as: phi = asin(x/sqrt(x^2+y^2+z^2)), vertical angle theta is defined as: theta = asin(y/sqrt(x^2+y^2+z^2))
     *
     *  The windshield vertical polynomial function is defined
     *  as theta' = windshield_vertical_polynomial[0] +  ... + windshield_vertical_polynomial[j * (2 * DEGREE + 3 - j) / 2 + i]* phi^(i)* theta(j) + ...
     *                  + windshield_vertical_polynomial[DW_WINDSHIELD_VERTICAL_POLY_LENGTH] * theta^(DEGREE)
     */
    float32_t verticalPolynomial[DW_WINDSHIELD_VERTICAL_POLY_LENGTH];

    /**
     *  Defines whether the polynomial parameter
     *  either map a ray from non-camera side to camera side (called forward direction)
     *  or map a ray from camera side to non-camera side (called backward direction).
     */
    dwWindshieldPolynomialType polynomialType;
} dwWindshieldModelConfig;

/**
 * Defines the maximum number of supports of the radar azimuth angle correction model.
*/
#define DW_RADAR_AZIMUTH_CORRECTION_TABLE_MAX_LENGTH 161U

/**
 * Configuration parameters for a radar azimuth correction model.
 *
 * The model consists of a correction function (and its standard deviation) across
 * the field of view (FOV) of the sensor, where phi is the *actual* angle of a detection and
 * phi_hat is the *measured* angle of a detection. Then,
 *
 *      phi_hat = phi + correction(phi)
 *
 * The function correction(phi) is approximated as piecewise linear function with equidistant
 * supports with range [phiMinRad,phiMaxRad] and spacing deltaPhiRad and function values
 * correctionsRad.
 */
typedef struct dwRadarAzimuthCorrectionModelConfig
{
    /** Minimum azimuth angle of FOV [rad] */
    float32_t phiMinRad;

    /** Maximum azimuth angle of FOV [rad] */
    float32_t phiMaxRad;

    /** Equidistant spacing between supports [rad] */
    float32_t deltaPhiRad;

    /** Number of correction values/supports */
    uint32_t numCorrections;

    /** Values of correction function at support i [rad]*/
    float32_t correctionsRad[DW_RADAR_AZIMUTH_CORRECTION_TABLE_MAX_LENGTH];

    /** Standard deviation of correction function at support i [rad] */
    float32_t stddevRad[DW_RADAR_AZIMUTH_CORRECTION_TABLE_MAX_LENGTH];
} dwRadarAzimuthCorrectionModelConfig;

#ifdef __cplusplus
}
#endif

#endif // DW_RIG_RIG_TYPES_H_

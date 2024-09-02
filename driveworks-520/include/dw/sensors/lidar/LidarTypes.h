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
// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Lidar types</b>
 *
 * @b Description: This file defines the Lidar sensor types.
 */

/**
 * @defgroup lidar_group Lidar Sensor
 * @ingroup sensors_group
 *
 * @brief Defines the Lidar sensor types.
 *
 * @{
 */

#ifndef DW_SENSORS_LIDAR_LIDARTYPES_H_
#define DW_SENSORS_LIDAR_LIDARTYPES_H_

#include <dw/core/base/Types.h>

#include <dw/sensors/common/SensorTypes.h>
#include <stdalign.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Enum to indicate a single return type.
 *
 * Bitmasks of this enum can be used to specify return configurations
 * of lidars, requested returns from the accumulator, or returns
 * available in a point cloud
 */
typedef enum {
    DW_LIDAR_RETURN_TYPE_ANY = 0,                /*!< API default for whatever */
                                                 /*!< returns happen to be present */
    DW_LIDAR_RETURN_TYPE_FIRST         = 1 << 0, /*!< First return */
    DW_LIDAR_RETURN_TYPE_LAST          = 1 << 1, /*!< Last return */
    DW_LIDAR_RETURN_TYPE_STRONGEST     = 1 << 2, /*!< Strongest return (vendor-specific) */
    DW_LIDAR_RETURN_TYPE_ABS_STRONGEST = 1 << 3, /*!< Derived type: if multiple */
                                                 /*!< returns are present, */
                                                 /*!< whichever is strongest */

    DW_LIDAR_RETURN_TYPE_1 = 1 << 4, /*!< Generic enum to capture lidars for which an array of returns is present */
    DW_LIDAR_RETURN_TYPE_2 = 1 << 5, /*!< Generic enum to capture lidars for which an array of returns is present */
    DW_LIDAR_RETURN_TYPE_3 = 1 << 6, /*!< Generic enum to capture lidars for which an array of returns is present */
    DW_LIDAR_RETURN_TYPE_4 = 1 << 7, /*!< Generic enum to capture lidars for which an array of returns is present */
    DW_LIDAR_RETURN_TYPE_5 = 1 << 8, /*!< Generic enum to capture lidars for which an array of returns is present */
    DW_LIDAR_RETURN_TYPE_6 = 1 << 9, /*!< Generic enum to capture lidars for which an array of returns is present */
} dwLidarReturnType;

/**
* An enum for every data element we could possibly return.
*/
typedef enum {
    DW_LIDAR_AUX_DATA_TYPE_SNR            = 0,  //!< SNR type.
    DW_LIDAR_AUX_DATA_TYPE_PULSEWIDTH_PS  = 1,  //!< PULSEWIDTH_PS type.
    DW_LIDAR_AUX_DATA_TYPE_TIME           = 2,  //!< TIME type.
    DW_LIDAR_AUX_DATA_TYPE_V_MPS          = 3,  //!< V_MPS type.
    DW_LIDAR_AUX_DATA_TYPE_EXISTENCEPROB  = 4,  //!< EXISTENCEPROB type.
    DW_LIDAR_AUX_DATA_TYPE_CROSSTALKPROB  = 5,  //!< CROSSTALKPROB type.
    DW_LIDAR_AUX_DATA_TYPE_NOISELEVEL     = 6,  //!< NOISELEVEL type.
    DW_LIDAR_AUX_DATA_TYPE_ZONEID         = 7,  //!< ZONEID type.
    DW_LIDAR_AUX_DATA_TYPE_DETECTORID     = 8,  //!< DETECTORID type.
    DW_LIDAR_AUX_DATA_TYPE_LINEID         = 9,  //!< LINEID type.
    DW_LIDAR_AUX_DATA_TYPE_DATAQUALITY    = 10, //!< DATAQUALITY type.
    DW_LIDAR_AUX_DATA_TYPE_SCANCHECKPOINT = 11, //!< SCANCHECKPOINT type.
    DW_LIDAR_AUX_DATA_TYPE_BLOCKAGEFLAG   = 12, //!< BLOCKAGEFLAG type.
    DW_LIDAR_AUX_DATA_TYPE_SENSORID       = 13, //!< SENSORID type.
    DW_LIDAR_AUX_DATA_TYPE_AZIMUTH        = 14, //!< AZIMUTH type.
    DW_LIDAR_AUX_DATA_TYPE_ELEVATION      = 15, //!< ELEVATION type.
    DW_LIDAR_AUX_DATA_TYPE_DISTANCE       = 16, //!< DISTANCE type.
    DW_LIDAR_AUX_DATA_TYPE_ROI            = 17, //!< ROI type.
    DW_LIDAR_AUX_DATA_TYPE_VALIDITY       = 18, //!< VALIDITY type.
    DW_LIDAR_AUX_DATA_TYPE_INVALIDITYFLAG = 19, //!< INVALIDITYFLAG type.
    DW_LIDAR_AUX_DATA_TYPE_COUNT          = 20, //!< COUNT type.
    DW_LIDAR_AUX_DATA_TYPE_FORCE32        = 0x7FFFFFFF
} dwLidarAuxDataType;

/**
*  An enum for specifying invalidity flags.
*/
typedef enum {
    /// No flags are set
    DW_LIDAR_INVALIDITY_NONE = 0,

    /// DriveWorks-specific validity flags
    DW_LIDAR_INVALIDITY_DW = 1 << 0,

    /// Vendor-specific validity flags
    DW_LIDAR_INVALIDITY_VEND = 1 << 1,

    /// Blockage validity flags
    DW_LIDAR_INVALIDITY_BLOCKAGE = 1 << 2,

    /// Point is not valid if any of flags are set
    DW_LIDAR_INVALIDITY_INVALID =
        DW_LIDAR_INVALIDITY_DW |
        DW_LIDAR_INVALIDITY_VEND |
        DW_LIDAR_INVALIDITY_BLOCKAGE

} dwLidarInvalidityFlag;

/** Maximum number of distinct lidar returns a point cloud can contain */
#define DW_SENSORS_LIDAR_MAX_RETURNS 10

/** Holds a Lidar point cloud XYZ and the associated intensity. */
typedef struct dwLidarPointXYZI
{
    alignas(16) float32_t x; /*!< Lidar right-handed coord system, planar direction, in meters unit. */
    float32_t y;             /*!< Lidar right-handed coord system, planar direction, in meters unit. */
    float32_t z;             /*!< Lidar right-handed coord system, vertical direction, in meters unit. */
    float32_t intensity;     /*!< Reflection intensity in range 0-X, where X can be > 1 (vendor-specific), indicating a target with (100*X)% diffuse reflectivity */
} dwLidarPointXYZI;

/** Holds a Lidar point cloud RTHI and the associated intensity. */
typedef struct dwLidarPointRTHI
{
    alignas(16) float32_t theta; /*!< Lidar right-handed polar coord system, planar direction, in rads unit. */
    float32_t phi;               /*!< Lidar right-handed polar coord system, vertical direction, in rads unit. */
    float32_t radius;            /*!< Lidar right-handed polar coord system, distance, in m unit. */
    float32_t intensity;         /*!< Reflection intensity in the 0 to 1 range. */
} dwLidarPointRTHI;

#define DW_SENSORS_LIDAR_MAX_ROWS 256 /*!< maximal number of rows in dwLidarProperties */

/** Defines the properties of the lidar */
typedef struct dwLidarProperties
{
    char8_t deviceString[256]; /*!< ASCII string identifying the device. */

    float32_t spinFrequency; /*!< Current spin frequency, in HZ. */

    uint32_t packetsPerSecond; /*!< Number of packets per second the sensor produces. */
    uint32_t packetsPerSpin;   /*!< Number of packets per sensor full spin. */

    uint32_t pointsPerSecond; /*!< Number of points per second the sensor provides. */
    uint32_t pointsPerPacket; /*!< Maximum number of points in a packet. */
    uint32_t pointsPerSpin;   /*!< Maximum number of points on a full sensor spin. */
    uint32_t pointStride;     /*!< Number of 'float32' elements between points and points which lidar send. */

    float32_t horizontalFOVStart; /*!< Lidar right-handed polar coord system,  start angle in planar direction, in rads. */
    float32_t horizontalFOVEnd;   /*!< Lidar right-handed polar coord system, end angle in planar direction, in rads. */

    uint32_t numberOfRows; /*!< Number of Rows in a spin Frame */

    float32_t verticalFOVStart; /*!< Lidar right-handed polar coord system, start angle in vertical direction, in rads. */
    float32_t verticalFOVEnd;   /*!< Lidar right-handed polar coord system, end angle in vertical direction, in rads. */

    /**
     * Lidar right-handed polar coord system, vertical angles in spin frame, in rads,
     * length of array with valid values is defined by 'numberOfRows' above
     */
    float32_t verticalAngles[DW_SENSORS_LIDAR_MAX_ROWS];
    /**
     * Lidar right-handed polar coord system, intrinsic horizontal angle offsets in spin frame, in rads,
     * length of array with valid values is defined by 'numberOfRows' above.
     */
    float32_t horizontalAngles[DW_SENSORS_LIDAR_MAX_ROWS];

    /** Bitmask of return types the lidar is configured to */
    dwLidarReturnType availableReturns;

    /** Bitmask of valid aux info fields based on enum dwLidarAuxDataType */
    uint64_t validAuxInfos;

    /** LidarSSI size in bytes, so user know the memory to be allocated for dwLidarDecodedPacket.lidarSSI */
    size_t lidarSSISizeInBytes;
} dwLidarProperties;

/** Defines the return structure for an extended decoded lidar packet. */
typedef struct dwLidarDecodedReturn
{
    dwLidarReturnType type; /*!< Type of this return */

    uint32_t maxPoints; /*!< Maximum number of points in this return */
    uint32_t numPoints; /*!< Current number of valid points in the return */

    dwLidarPointRTHI const* pointsRTHI; /*!< Pointer to the array of points (polar) */

    /** Array of pointers to auxiliary data
    * Supported aux data types are listed in lidar properties.
    * Size of auxiliary data element can be found in auxDataSize[] array below.
    * Number of elements equals to numPoints
    */
    void const* auxData[DW_LIDAR_AUX_DATA_TYPE_COUNT];

    /**
     * Data element size for each type of aux data. 0 means this
     * aux data type is not configured in lidarProperties.validAuxInfos
     */
    uint32_t auxDataSize[DW_LIDAR_AUX_DATA_TYPE_COUNT];
} dwLidarDecodedReturn;

/// Not available as of current release. Will be added in future release
typedef struct _dwLidarDecodedSSI dwLidarDecodedSSI;

/** Defines the structure for a decoded lidar packet.
 **/
typedef struct dwLidarDecodedPacket
{
    dwTime_t hostTimestamp;   /*!< Timestamp measured on the host, in microseconds. */
    dwTime_t sensorTimestamp; /*!< Timestamp of the first point in the point cloud packet, in microseconds. */
    dwTime_t duration;        /*!< Time difference between the first measurement and the last, in microseconds. */

    /** Maximum number of points in the \p pointsRTHI and \p pointsXYZI arrays in the packet. */
    uint32_t maxPoints;

    /** Current number of valid points in the \p pointsRTHI and \p pointsXYZI arrays in the packet. */
    uint32_t nPoints;

    float32_t minHorizontalAngleRad; /*!< Minimum horizontal angle in the packet, in rads */
    float32_t maxHorizontalAngleRad; /*!< Maximum horizontal angle in the packet, in rads */
    float32_t minVerticalAngleRad;   /*!< Minimum vertical angle in the packet, in rads */
    float32_t maxVerticalAngleRad;   /*!< Maximum vertical angle in the packet, in rads */

    bool scanComplete; /*!< Flag to identify if the scan is complete. */

    /** Pointer to the array of points in polar coordinates. */
    dwLidarPointRTHI const* pointsRTHI;

    /** Pointer to the array of points in cartesian coordinates. */
    dwLidarPointXYZI const* pointsXYZI;

    float32_t elevationOffsetRad; /*!< Elevation offset angle in the packet, in rads */
    float32_t azimuthOffsetRad;   /*!< Azimuth offset angle in the packet, in rads */

    uint8_t numReturns;                                            /*!< Number of returns present in this packet */
    dwLidarDecodedReturn returnData[DW_SENSORS_LIDAR_MAX_RETURNS]; /*!< An array contains data for each return in this packet */

    /// lidar supplement status info such as misc and health info
    dwLidarDecodedSSI const* lidarSSI;
} dwLidarDecodedPacket;

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_SENSORS_LIDAR_LIDARTYPES_H_

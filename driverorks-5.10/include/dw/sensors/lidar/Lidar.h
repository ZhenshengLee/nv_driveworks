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
// SPDX-FileCopyrightText: Copyright (c) 2016-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Lidar</b>
 *
 * @b Description: This file defines the Lidar sensor.
 */

/**
 * @defgroup lidar_group Lidar Sensor
 * @ingroup sensors_group
 *
 * @brief Defines the Lidar sensor methods.
 *
 * @{
 */

#ifndef DW_SENSORS_LIDAR_LIDAR_H_
#define DW_SENSORS_LIDAR_LIDAR_H_

#include <dw/core/base/Types.h>

#include <dw/sensors/Sensors.h>
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
    DW_LIDAR_AUX_DATA_TYPE_SNR            = 0,  //<! SNR type.
    DW_LIDAR_AUX_DATA_TYPE_SIGNALWIDTH    = 1,  //<! SIGNALWIDTH type.
    DW_LIDAR_AUX_DATA_TYPE_SIGNALHEIGHT   = 2,  //<! SIGNALHEIGHT type.
    DW_LIDAR_AUX_DATA_TYPE_TIME           = 3,  //<! TIME type.
    DW_LIDAR_AUX_DATA_TYPE_V_MPS          = 4,  //<! V_MPS type.
    DW_LIDAR_AUX_DATA_TYPE_EXISTENCEPROB  = 5,  //<! EXISTENCEPROB type.
    DW_LIDAR_AUX_DATA_TYPE_CROSSTALKPROB  = 6,  //<! CROSSTALKPROB type.
    DW_LIDAR_AUX_DATA_TYPE_NOISELEVEL     = 7,  //<! NOISELEVEL type.
    DW_LIDAR_AUX_DATA_TYPE_ZONEID         = 8,  //<! ZONEID type.
    DW_LIDAR_AUX_DATA_TYPE_DETECTORID     = 9,  //<! DETECTORID type.
    DW_LIDAR_AUX_DATA_TYPE_LINEID         = 10, //<! LINEID type.
    DW_LIDAR_AUX_DATA_TYPE_DATAQUALITY    = 11, //<! DATAQUALITY type.
    DW_LIDAR_AUX_DATA_TYPE_SCANCHECKPOINT = 12, //<! SCANCHECKPOINT type.
    DW_LIDAR_AUX_DATA_TYPE_BLOCKAGEFLAG   = 13, //<! BLOCKAGEFLAG type.
    DW_LIDAR_AUX_DATA_TYPE_SENSORID       = 14, //<! SENSORID type.
    DW_LIDAR_AUX_DATA_TYPE_VALIDITY       = 15, //<! VALIDITY type.
    DW_LIDAR_AUX_DATA_TYPE_INVALIDITYFLAG = 16, //<! INVALIDITYFLAG type.
    DW_LIDAR_AUX_DATA_TYPE_COUNT          = 17, //<! COUNT type.
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

    /// Point is not valid if any of flags are set
    DW_LIDAR_INVALIDITY_INVALID = DW_LIDAR_INVALIDITY_DW | DW_LIDAR_INVALIDITY_VEND

} dwLidarInvalidityFlag;

/** Maximum number of distinct lidar returns a point cloud can contain */
#define DW_SENSORS_LIDAR_MAX_RETURNS 10

/** Holds a Lidar point cloud XYZ and the associated intensity. */
typedef struct dwLidarPointXYZI
{
    alignas(16) float32_t x; /*!< Lidar right-handed coord system, planar direction, in meters unit. */
    float32_t y;             /*!< Lidar right-handed coord system, planar direction, in meters unit. */
    float32_t z;             /*!< Lidar right-handed coord system, vertical direction, in meters unit. */
    float32_t intensity;     /*!< Reflection intensity in the 0 to 1 range. */
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
    uint32_t pointStride;     /*!< Number of 'float32' elements bewteen points and points which lidar send. */

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
    * Size of auxiliary data element is known from its type,
    * see `dwSensorLidar_getAuxElementSize`.
    * Number of elements equals to numPoints
    */
    void const* auxData[DW_LIDAR_AUX_DATA_TYPE_COUNT];

} dwLidarDecodedReturn;

/** Defines the structure for a decoded lidar packet.
 **/
typedef struct dwLidarDecodedPacket
{
    dwTime_t hostTimestamp;   /*!< Timestamp measured on the host, in microseconds. */
    dwTime_t sensorTimestamp; /*!< Timestamp of the first point in the point cloud packet, in microseconds. */
    dwTime_t duration;        /*!< Time difference between the first measurement and the last, in microseconds. */

    /** Maximum number of points in the \p pointsRTHI and \p pointsXYZI arrays in the packet.
    *\deprecated This field will be removed.
    */
    uint32_t maxPoints;

    /** Current number of valid points in the \p pointsRTHI and \p pointsXYZI arrays in the packet.
    *\deprecated This field will be removed.
    */
    uint32_t nPoints;

    float32_t minHorizontalAngleRad; /*!< Minimum horizontal angle in the packet, in rads */
    float32_t maxHorizontalAngleRad; /*!< Maximum horizontal angle in the packet, in rads */
    float32_t minVerticalAngleRad;   /*!< Minimum vertical angle in the packet, in rads */
    float32_t maxVerticalAngleRad;   /*!< Maximum vertical angle in the packet, in rads */

    bool scanComplete; /*!< Flag to identify if the scan is complete. */

    /** Pointer to the array of points in polar coordinates.
    *\deprecated This field will be removed.
    */
    dwLidarPointRTHI const* pointsRTHI;

    /** Pointer to the array of points in cartesian coordinates.
    *\deprecated This field will be removed.
    */
    dwLidarPointXYZI const* pointsXYZI;

    float32_t elevationOffsetRad; /*!< Elevation offset angle in the packet, in rads */
    float32_t azimuthOffsetRad;   /*!< Azimuth offset angle in the packet, in rads */

    uint8_t numReturns;                                            /*!< Number of returns present in this packet */
    dwLidarDecodedReturn returnData[DW_SENSORS_LIDAR_MAX_RETURNS]; /*!< An array contains data for each return in this packet */
} dwLidarDecodedPacket;

/**
 * Returns size of auxiliary data element in bytes
 * 
 * @param[out] sizeBytes element size
 * @param[in] auxType auxiliary data type
 * @retval DW_INVALID_ARGUMENT: The input parameter is invalid.
 * @retval DW_SUCCESS: Successful deal.
 */
DW_API_PUBLIC
dwStatus dwSensorLidar_getAuxElementSize(uint32_t* const sizeBytes, dwLidarAuxDataType const auxType);

/**
* Enables the decoding of the Lidar packets, which incurs an additional CPU load.
* Method fails if the sensor has been started and is capturing data. Stop the sensor first.
* The default state is to have decoding on. If on, dwSensor_readRawData(see reference [15]) returns DW_CALL_NOT_ALLOWED.
*
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_INVALID_HANDLE: The input handle is not a lidar handle.
* @retval DW_CALL_NOT_ALLOWED: The sensor is not stopped.
* @retval DW_SUCCESS: Successful deal.
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_enableDecoding(dwSensorHandle_t const sensor);

/**
* Disable the decoding of the Lidar packets, which frees additional CPU load.
* Method fails if the sensor has been started and is capturing data. Stop the sensor first.
* The default state is to have decoding on. If on, dwSensor_readRawData(see reference [15]) returns DW_CALL_NOT_ALLOWED.
*
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_INVALID_HANDLE: The input handle is not a lidar handle.
* @retval DW_CALL_NOT_ALLOWED: The sensor is not stopped.
* @retval DW_SUCCESS: Successful deal.
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_disableDecoding(dwSensorHandle_t const sensor);

/**
* Retrieves the state of packet decoding.
*
* @param[out] enable Contains the result of the query, which is true when decoding. False if RAW data.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_SUCCESS: Successful deal.
* @retval DW_INVALID_HANDLE: The input handle is not a lidar handle.
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_isDecodingEnabled(bool* const enable, dwSensorHandle_t const sensor);

/**
* Reads one scan packet. The pointer returned is to the internal data pool. DW guarantees that the data
* remains constant until returned by the application. The data must be explicitly returned by the
* application.
*
* @param[out] data A pointer to a pointer that can read data from the sensor. The struct contains the
*                  numbers of points read, which depends on the sensor used.
* @param[in] timeoutUs Specifies the timeout in microseconds. Special values:
*                  DW_TIMEOUT_INFINITE - to wait infinitly.  Zero - means polling of internal queue.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_INVALID_HANDLE: the input handle is not a lidar handle.
* @retval DW_CALL_NOT_ALLOWED: the decoder is not working.
* @retval DW_INVALID_ARGUMENT: the input argument is invalid.
* @retval DW_NOT_AVAILABLE: device is disconneted or sensor is not working.
* @retval DW_TIME_OUT: timeout.
* @retval DW_SUCCESS: successful deal.
*
* \deprecated This API will be removed. Use 'dwSensorLidar_readPacketEx'
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_readPacket(dwLidarDecodedPacket const** const data, dwTime_t const timeoutUs,
                                  dwSensorHandle_t const sensor);

/**
* Returns the data read to the internal pool. At this point the pointer is still be valid, but data is
* change based on newer readings of the sensor.
*
* @param[in] data A pointer to the scan data previously read from the Lidar to be returned to the pool.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_INVALID_HANDLE: the input handle is not a lidar handle.
* @retval DW_CALL_NOT_ALLOWED: the decoder is not working.
* @retval DW_SUCCESS: successful deal.
* @note Other return value will depend on the lidar type due to the reason that different type lidar will have different internal implements.
* \deprecated This API will be removed. Use 'dwSensorLidar_returnPacketEx'
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_returnPacket(dwLidarDecodedPacket const* const data, dwSensorHandle_t const sensor);

/**
* Decodes RAW data previously read and returns a pointer to it. This happens on the CPU thread where
* the function is called, incurring on additional load on that thread. The data is valid until the
* application calls dwSensor_returnRawData.
*
* @param[out] data A pointer to the memory pool owned by the sensor.
* @param[in] rawData A pointer for the non-decoded Lidar packet, returned by 'dwSensor_readRawData(see reference [15])'.
* @param[in] size Specifies the size in bytes of the raw data.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_INVALID_HANDLE: the input handle is not a lidar handle.
* @retval DW_CALL_NOT_ALLOWED: the decoder is not working.
* @retval DW_SUCCESS: successful deal.
* @note Other return value will depend on the lidar type due to the reason that different type lidar will have different internal implements.
* \deprecated This API will be removed. Use 'dwSensorLidar_processRawDataEx'
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_processRawData(dwLidarDecodedPacket const** const data, uint8_t const* const rawData, size_t const size,
                                      dwSensorHandle_t const sensor);

/**
* Gets information about the Lidar sensor.
*
* @param[out] lidarProperties A pointer to the struct containing the properties of the Lidar.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_INVALID_HANDLE: the input handle is not a lidar handle.
* @retval DW_CALL_NOT_ALLOWED: config is not allowed to get in passive mode.
* @retval DW_SUCCESS: successful deal.
* @note Other return value will depend on the lidar type due to the reason that different type lidar will have different internal implements.
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_getProperties(dwLidarProperties* const lidarProperties, dwSensorHandle_t const sensor);

/**
* Sends a message to Lidar sensor.
*
* @param[in] cmd Command identifier associated to the given message data.
* @param[in] data A pointer to the message data.
* @param[in] size Size in bytes of the \p data.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor(see reference [15])'.
* @retval DW_INVALID_HANDLE: the input handle is not a lidar handle.
* @note Other return value will depend on the lidar type due to the reason that different type lidar will have different internal implements.
*/
DW_API_PUBLIC
dwStatus dwSensorLidar_sendMessage(uint32_t const cmd, uint8_t const* const data,
                                   size_t const size, dwSensorHandle_t const sensor);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_SENSORS_LIDAR_LIDAR_H_

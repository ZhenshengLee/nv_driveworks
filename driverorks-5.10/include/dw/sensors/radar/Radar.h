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
// SPDX-FileCopyrightText: Copyright (c) 2017-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Radar</b>
 *
 * @b Description: This file defines the Radar sensor.
 */

/**
 * @defgroup radar_group Radar Sensor
 * @ingroup sensors_group
 *
 * @brief Defines the Radar sensor methods.
 *
 * @{
 */

#ifndef DW_SENSORS_RADAR_RADAR_H_
#define DW_SENSORS_RADAR_RADAR_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Types.h>
#include <dw/sensors/radar/RadarScan.h>
#include <dw/sensors/Sensors.h>

/// The minimum payload size is larger than sizeof(dwRadarScan) to allow custom types to use a larger payload
#define DW_RADARSCAN_MINIMUM_PAYLOAD_SIZE 1744

/// Defines the range of radar return
typedef enum {
    /// Short Range
    DW_RADAR_RANGE_SHORT = 0,

    /// Medium Range
    DW_RADAR_RANGE_MEDIUM = 1,

    /// Long Range
    DW_RADAR_RANGE_LONG = 2,

    /// Unknown Range
    DW_RADAR_RANGE_UNKNOWN = 3,

    /// Count
    DW_RADAR_RANGE_COUNT = 4,
} dwRadarRange;

/// Defines the type of radar return
typedef enum {
    /// Raw detection
    DW_RADAR_RETURN_TYPE_DETECTION = 0,

    /// Processed tracker output
    DW_RADAR_RETURN_TYPE_TRACK = 1,

    /// Sensor status information
    DW_RADAR_RETURN_TYPE_STATUS = 2,

    /// Count
    DW_RADAR_RETURN_TYPE_COUNT = 3,
} dwRadarReturnType;

/// Defines the dynamic state of the radar return
//# sergen(generate)
typedef enum dwRadarDynamicState {
    /// Moving
    DW_RADAR_DYNAMIC_STATE_MOVING = 0,

    /// Stationary
    DW_RADAR_DYNAMIC_STATE_STATIONARY = 1,

    /// Oncoming
    DW_RADAR_DYNAMIC_STATE_ONCOMING = 2,

    /// Cross-traffic
    DW_RADAR_DYNAMIC_STATE_CROSS_TRAFFIC = 3,

    /// Stopped (was moving, now stationary)
    DW_RADAR_DYNAMIC_STATE_STOPPED = 4,

    /// Unknown
    DW_RADAR_DYNAMIC_STATE_UNKNOWN = 5,
} dwRadarDynamicState;

/// Defines the type of scan (combination of return type & range)
typedef struct dwRadarScanType
{
    /// Type of radar return
    dwRadarReturnType returnType;

    /// Scan range
    dwRadarRange range;
} dwRadarScanType;

// The structs below are serialized in binary and the layout is assummed to be packed
#pragma pack(push, 1)

/// Defines the return structure for a raw radar detection in sensor coordinates
//# sergen(generate)
typedef struct dwRadarDetection
{
    /// X-position (m)
    float32_t x;

    /// Y-position (m)
    float32_t y;

    /// X-component (m/s) of the velocity in the azimuth direction
    float32_t Vx;

    /// Y-component (m/s) of the velocity in the azimuth direction
    float32_t Vy;

    /// Azimuth angle (radians)
    float32_t azimuth;

    /// Radial distance (m)
    float32_t radius;

    /// Radial velocity (m/s)
    float32_t radialVelocity;

    /// Reflection amplitude (dB)
    float32_t rcs;

    /// Angle of elevation (radians)
    float32_t elevationAngle;

    /// Indicates validity of the elevation angle
    bool elevationValidity;

    /// Signal to noise ratio (dBr)
    float32_t SNR;
} dwRadarDetection;

/// Defines the track which the radar provides.
typedef struct dwRadarTrack
{
    /// Radar-provided track id
    uint32_t id;

    /// Age of tracked object (in scans)
    uint32_t age;

    /// Confidence of object existence (range: 0-1);
    float32_t confidence;

    /// Dynamic state of the object
    dwRadarDynamicState dynamicState;

    /// X-position (m)
    float32_t x;

    /// Y-position (m)
    float32_t y;

    /// Z-position (m)
    float32_t z;

    /// X-component (m/s) of the velocity
    float32_t Vx;

    /// Y-component (m/s) of the velocity
    float32_t Vy;

    /// X-component (m/s^2) of the acceleration
    float32_t Ax;

    /// Y-component (m/s^2) of the aceleration
    float32_t Ay;

    /// Azimuth angle (radians)
    float32_t azimuth;

    /// Rate of change of azimuth angle (radians/s)
    float32_t azimuthRate;

    /// Radial distance (m)
    float32_t radius;

    /// Radial velocity (m/s)
    float32_t radialVelocity;

    /// Radial acceleration (m/s^2)
    float32_t radialAcceleration;

    /// Compensated reflection amplitude (dB)
    float32_t rcs;

    /// Indicates validity of z position
    bool elevationValid;
} dwRadarTrack;

/// Defines the structure for a complete radar scan
typedef struct dwRadarScan
{
    /// Sensor-provided scan index
    uint32_t scanIndex;

    /// Sensor timestamp at which the current measurement scan was started (us)
    dwTime_t sensorTimestamp;

    /// Host timestamp at reception of first packet belonging to this scan (us)
    dwTime_t hostTimestamp;

    /// Type of scan
    dwRadarScanType scanType;

    /// Number of radar returns in this scan
    uint32_t numReturns;

    /// Doppler ambiguity free range
    float32_t dopplerAmbiguity;

    /// Radar Scan validity
    /// If the signal is unavailable or invalid, the value of the signal will be the maximum number of the data type
    dwRadarScanValidity scanValidity;

    /// Radar Scan miscellaneous fields
    dwRadarScanMisc radarScanMisc;

    /// Radar Scan ambiguity
    dwRadarScanAmbiguity radarScanAmbiguity;

    /// Pointer to the array of returns (to be casted based on return type)
    /// Size of this array is numReturns
    void* data;

    /// Pointer to the array of dwRadarDetectionMisc, size of this array is numReturns
    dwRadarDetectionMisc const* detectionMisc;

    /// Pointer to the array of dwRadarDetectionStdDev, size of this array is numReturns
    dwRadarDetectionStdDev const* detectionStdDev;

    /// Pointer to the array of dwRadarDetectionQuality, size of this array is numReturns
    dwRadarDetectionQuality const* detectionQuality;

    /// Pointer to the array of dwRadarDetectionProbability, size of this array is numReturns
    dwRadarDetectionProbability const* detectionProbability;

    /// Pointer to the array of dwRadarDetectionFFTPatch, size of this array is numReturns
    dwRadarDetectionFFTPatch const* detectionFFTPatch;
} dwRadarScan;

#pragma pack(pop)

/// Defines the return structure for sensor status messages
typedef struct dwRadarStatus
{
    /// X-position (m) of sensor mounting in AUTOSAR-coordinates from CoG
    /// Estimated radar position/orientation as reported by the radar,
    /// refer to Radar spec for the coordinate system defintion.
    /// [optional: might not be populated if radar is not supporting estimation]
    float32_t x;

    /// Y-position (m) of sensor mounting in AUTOSAR-coordinates from CoG
    /// Estimated radar position/orientation as reported by the radar,
    /// refer to Radar spec for the coordinate system defintion.
    /// [optional: might not be populated if radar is not supporting estimation]
    float32_t y;

    /// Z-position (m) of sensor mounting in AUTOSAR-coordinates from CoG
    /// Estimated radar position/orientation as reported by the radar,
    /// refer to Radar spec for the coordinate system defintion.
    /// [optional: might not be populated if radar is not supporting estimation]
    float32_t z;

    /// Yaw angle of sensor (radians)
    float32_t yaw;

    /// Pitch angle of sensor (radians)
    float32_t pitch;

    /// Roll angle of sensor (radians)
    float32_t roll;

    /*
     * Deviation of azimuth angle for returns (radians)
     * This is the deviation of the measured azimuth from what would be
     * expected based on the geometric boresight of the sensor.
     */
    float32_t azimuthDeviation[DW_RADAR_RANGE_COUNT];

    /*
     * Deviation of elevation angle for returns (radians)
     * This is the deviation of the elevation azimuth from what would be
     * expected based on the geometric boresight of the sensor.
     */
    float32_t elevationDeviation[DW_RADAR_RANGE_COUNT];

    /// Indicates if the sensor is aligned
    bool sensorAligned;

    /// Indicaes if the is OK
    bool sensorOK;

    /// Indicates if the sensor is disturbed due to interference
    bool sensorDisturbed;

    /// Indicates if the sensor is blocked
    bool sensorBlock;
} dwRadarStatus;

/// Defines the structure for reporting current vehicle dynamics state
typedef struct dwRadarVehicleState
{
    /// Longitudinal velocity (m/s)
    float32_t velocity;

    /// Longitudinal acceleration (m/s^2)
    float32_t acceleration;

    /// Lateral acceleration (m/s^2)
    float32_t lateralAcceleration;

    /// Yaw rate (radians/s)
    float32_t yawRate;
} dwRadarVehicleState;

/// Defines the structure for reporting sensor mount position
typedef struct dwRadarMountPosition
{
    /// Id of the sensor (vendor-specific)
    uint32_t sensorId;

    /// Radar position
    dwTransformation3f radarPosition;

    /// Size of wheel-base (m)
    float32_t wheelbase;

    /// Damping of radome (db)
    float32_t damping;

    /// Indicates if the sensor is reversed from its default orientation
    bool isReversed;
} dwRadarMountPosition;

/// Defines the properties of the radar
typedef struct dwRadarProperties
{
    /// Indicates whether decoding is enabled
    uint8_t isDecodingOn;

    /// Number of supported scan types
    uint32_t numScanTypes;

    /// Enumerates the types of scans supported by the radar
    uint32_t supportedScanTypes[DW_RADAR_RETURN_TYPE_COUNT][DW_RADAR_RANGE_COUNT];

    /// # of packets per scan (Note: will be deprecated soon)
    uint32_t packetsPerScan;

    /// Max # of returns in any given scan
    uint32_t maxReturnsPerScan[DW_RADAR_RETURN_TYPE_COUNT][DW_RADAR_RANGE_COUNT];

    /**
     * Number of scans (of a particular type) per second.
     * In case scan rate differ between scans, this number is the maximum
     * amongst all scan types
     */
    uint32_t scansPerSecond;

    /// Number of input odometry packets per second
    uint32_t inputPacketsPerSecond;

    /// Indicates whether the sensor is simulated
    bool isSimulation;

    /// Radar model of the current radar
    dwRadarModel radarModel;
} dwRadarProperties;

#ifdef __cplusplus
extern "C" {
#endif

/**
* Enables/Disables a scan type for the radar sensor. Method fails if the sensor does not support the
* specified scan type OR the sensor has been started and is capturing data.
*
* @param[in] enable Specifies if the scan should be enabled or disabled
* @param[in] scanType Specifies the scan type to enable
* @param[in] sensor Specifies the sensor handle of the sensor previously created with dwSAL_createSensor().
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid.
* @retval DW_INVALID_ARGUMENT: input argurment invalid.
* @retval DW_CALL_NOT_ALLOWED: operation is not allowed now.
* @retval DW_SUCCESS: successful deal.
*
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_toggleScanType(bool const enable, dwRadarScanType const scanType, dwSensorHandle_t const sensor);

/**
* Enables/disables the decoding of the Radar packets, which incurs in additional CPU load.
* Method fails if the sensor has been started and is capturing data. Stop the sensor first.
*
* @param[in] enable Specifies TRUE when decoding, false if RAW data.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with dwSAL_createSensor()
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid.
* @retval DW_CALL_NOT_ALLOWED: operation is not allowed now.
* @retval DW_SUCCESS: successful deal.
*
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_setDataDecoding(bool const enable, dwSensorHandle_t const sensor);

/**
* Reads a single packet, which might be an entire scan or a fraction of a scan, which is sensor dependent.
* The pointer returned is to the internal data pool. DW guarantees that the data
* remains constant until returned by the application. The data must be explicitly returned by the
* application.
*
* @param[out] data A pointer to a pointer to the decoded data read from the sensor. The struct contains the
*                  numbers of points read, which depends on the sensor used.
* @param[in] type Type of scan requested
* @param[in] timeoutUs Specifies the timeout in microseconds. Special values:
                        DW_TIMEOUT_INFINITE - to wait infinitly.  Zero - means polling of internal queue.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with dwSAL_createSensor().
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid.
* @retval DW_INVALID_ARGUMENT: input argurment invalid.
* @retval DW_NOT_AVAILABLE: work stopped.
* @retval DW_SUCCESS: successful deal.
* @retval DW_TIME_OUT: time out.
*
*/

DW_API_PUBLIC
dwStatus dwSensorRadar_readData(const dwRadarScan** const data, const dwRadarScanType type,
                                const dwTime_t timeoutUs, dwSensorHandle_t const sensor);
/**
* Returns the data read to the internal pool. At this point the pointer is still be valid, but data is
* changed based on newer readings of the sensor.
*
* @param[in] scan A pointer to the scan data previously read from the Radar using 'dwSensorRadar_readData()' to be returned to the pool.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with dwSAL_createSensor().
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid.
* @retval DW_INVALID_ARGUMENT: input argurment invalid.
* @retval DW_NOT_AVAILABLE: work stopped.
* @retval DW_SUCCESS: successful deal.
* @retval DW_TIME_OUT: time out.
*
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_returnData(const dwRadarScan* const scan, dwSensorHandle_t const sensor);

/**
* Reads a one scan chunk. The pointer returned is to the internal data pool. DW guarantees that the data
* remains constant until returned by the application. The data must be explicitly returned by the
* application.
*
* @note This method returns the oldest scan contained in the internal pool.
*
* @param[out] data A pointer to a pointer to a decoded scan from the sensor. The struct contains the
*                  numbers of points read, which depends on the sensor used.
* @param[in] timeoutUs Specifies the timeout in microseconds. Special values:
                        DW_TIMEOUT_INFINITE - to wait infinitly.  Zero - means polling of internal queue.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with dwSAL_createSensor().
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid.
* @retval DW_INVALID_ARGUMENT: input argurment invalid.
* @retval DW_NOT_AVAILABLE: work stopped.
* @retval DW_SUCCESS: successful deal.
* @retval DW_TIME_OUT: time out.
*
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_readScan(const dwRadarScan** const data,
                                const dwTime_t timeoutUs, dwSensorHandle_t const sensor);

/**
* Returns the data covering an entire scan read to the internal pool. At this point the pointer is still be valid,
* but data is changed based on newer readings of the sensor.
*
* @param[in] scan A pointer to an entire scan's data previously read from the Radar using 'dwSensorRadar_readScan()' to be returned to the pool.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with dwSAL_createSensor().
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid.
* @retval DW_INVALID_ARGUMENT: input argurment invalid.
* @retval DW_NOT_AVAILABLE: work stopped.
* @retval DW_SUCCESS: successful deal.
* @retval DW_TIME_OUT: time out.
*
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_returnScan(const dwRadarScan* const scan, dwSensorHandle_t const sensor);

/**
* Decodes raw data previously read onto the application side structure. This happens on the CPU thread where
* the function is called, incurring on additional load on that thread.
*
* @param[out] data A pointer to a container for the decoded data.
* @param[in] rawData A pointer for the non-decoded Radar packet, as returned from 'dwSensor_readRawData()'.
* @param[in] size Specifies the size in bytes of the raw data.
* @param[in] sensor Specifies the sensor handle of the sensor previously created with 'dwSAL_createSensor()'.
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid.
* @retval DW_INVALID_ARGUMENT: input argurment invalid.
* @retval DW_NOT_AVAILABLE: work stopped.
* @retval DW_SUCCESS: successful deal.
* @retval DW_TIME_OUT: time out.
*
* @note The returned dwRadarScan is only valid till the next 'dwSensorRadar_processRawData()' call.
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_processRawData(const dwRadarScan** const data,
                                      const uint8_t* const rawData, size_t const size, dwSensorHandle_t const sensor);

/**
* Gets information about the radar sensor.
*
* @param[out] radarProperties A pointer to the struct containing the properties of the radar.
* @param[in] sensor Sensor handle created with dwSAL_createSensor().
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid.
* @retval DW_INVALID_ARGUMENT: input argurment invalid.
* @retval DW_SUCCESS: successful deal.
*
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_getProperties(dwRadarProperties* const radarProperties, dwSensorHandle_t const sensor);

/**
* Sends vehicle dynamics information to the radar.
*
* @param[in] data A pointer to the struct containing the vehicle dynamics information to send
* @param[in] sensor Sensor handle created with dwSAL_createSensor().
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid.
* @retval DW_INVALID_ARGUMENT: input argurment invalid.
* @retval DW_SUCCESS: successful deal.
*
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_setVehicleState(dwRadarVehicleState* const data, dwSensorHandle_t const sensor);

/**
* Sends the radar mount position information to the radar.
*
* @param[in] data A pointer to the struct containing the radar mount position information to send.
* @param[in] sensor Sensor handle created with dwSAL_createSensor().
*
* @retval DW_INVALID_HANDLE: sensor handle is invalid.
* @retval DW_INVALID_ARGUMENT: input argurment invalid.
* @retval DW_SUCCESS: successful deal.
*
*/
DW_API_PUBLIC
dwStatus dwSensorRadar_setMountPosition(dwRadarMountPosition* const data, dwSensorHandle_t const sensor);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_SENSORS_RADAR_RADAR_H_

////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
// NVIDIA MAKES NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY,
// OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY
// IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of
// such information or for any infringement of patents or other rights of third
// parties that may result from its use. No license is granted by implication or
// otherwise under any patent or patent rights of NVIDIA Corporation. No third
// party distribution is allowed unless expressly authorized by NVIDIA.  Details
// are subject to change without notice. This code supersedes and replaces all
// information previously supplied. NVIDIA Corporation products are not
// authorized for use as critical components in life support devices or systems
// without express written approval of NVIDIA Corporation.
//
// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software and related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution of
// this software and related documentation without an express license agreement
// from NVIDIA Corporation is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////////
#ifndef DW_SENSORS_RADAR_RADARTYPES_H_
#define DW_SENSORS_RADAR_RADARTYPES_H_
// Generated by dwProto from radar_types.proto DO NOT EDIT BY HAND!
// See //3rdparty/shared/dwproto/README.md for more information

#include <dw/core/base/Types.h>

#include <dw/sensors/radar/RadarScan.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Defines the range of radar return
typedef enum dwRadarRange {
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
typedef enum dwRadarReturnType {
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

/// Defines the return structure for a raw radar detection in sensor coordinates.
/// The Radar Coordinate System is centered at the geometric center of the
/// radar's receptor. The x-axis points in sensing direction. The y-axis points
/// in the direction of the connector / plug and the z-axis is oriented such that
/// it completes an orthogonal right-handed coordinate system.
/// For further information, please refer to the following link:
/// https://developer.nvidia.com/docs/drive/driveworks/latest/nvsdk_dw_html/dwx_coordinate_systems.html
typedef struct dwRadarDetection
{
    /// X-position (m), with elevation assumed to be 0
    /// note: x = radius * std::cos(azimuth)
    float32_t x;

    /// Y-position (m), with elevation assumed to be 0
    /// note: y = radius * std::sin(azimuth)
    float32_t y;

    /// X-component (m/s) of the velocity in the azimuth direction, with elevation assumed to be 0
    /// note: partial velocity, vx = radial_vel * std::cos(azimuth)
    float32_t Vx;

    /// Y-component (m/s) of the velocity in the azimuth direction, with elevation assumed to be 0
    /// note: partial velocity, vy = radial_vel * std::sin(azimuth)
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

#pragma pack(pop)

// The structs below are serialized in binary and the layout is assummed to be packed
#pragma pack(push, 1)

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

    /// Deviation of azimuth angle for returns (radians)
    /// This is the deviation of the measured azimuth from what would be
    /// expected based on the geometric boresight of the sensor.
    float32_t azimuthDeviation[DW_RADAR_RANGE_COUNT];

    /// Deviation of elevation angle for returns (radians)
    /// This is the deviation of the elevation azimuth from what would be
    /// expected based on the geometric boresight of the sensor.
    float32_t elevationDeviation[DW_RADAR_RANGE_COUNT];

    /// Indicates if the sensor is aligned
    bool sensorAligned;

    /// Indicates if this is OK
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

    /// Number of scans (of a particular type) per second.
    /// In case scan rate differ between scans, this number is the maximum
    /// amongst all scan types
    uint32_t scansPerSecond;

    /// Number of input odometry packets per second
    uint32_t inputPacketsPerSecond;

    /// Indicates whether the sensor is simulated
    bool isSimulation;

    /// Radar model of the current radar
    dwRadarModel radarModel;

    /// RadarSSI size in bytes, so user know the memory to be allocated for dwRadarScan.radarSSI
    size_t radarSSISizeInBytes;
} dwRadarProperties;

#ifdef __cplusplus
}
#endif

#endif // DW_SENSORS_RADAR_RADARTYPES_H_

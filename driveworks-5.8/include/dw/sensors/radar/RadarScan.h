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
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: RadarScan</b>
 *
 * @b Description: This file defines the structures needed for the RadarScan.
 */

#ifndef DW_SENSORS_RADAR_RADARSCAN_H_
#define DW_SENSORS_RADAR_RADARSCAN_H_

#include <dw/core/base/Types.h>

/// Defines the radar sensor model
typedef enum {
    /// Unknown model
    DW_RADAR_MODEL_UNKNOWN = 0,

    /// Conti ARS430 model
    DW_RADAR_MODEL_CONTI_ARS430 = 1,

    /// Conti ARS540 model
    DW_RADAR_MODEL_CONTI_ARS540 = 2,

    /// Conti ARS620 model
    DW_RADAR_MODEL_CONTI_ARS620 = 3,

    /// Hella Gen6 model
    DW_RADAR_MODEL_HELLA_GEN6 = 4,
} dwRadarModel;

/// Defines the quality of scan
typedef enum {
    /// Quality field not available from sensor
    DW_RADAR_QUALITY_NOT_AVAILABLE = 0,

    /// Radar sensor quality is normal
    DW_RADAR_QUALITY_NORMAL = 1,

    /// Radar sensor quality has reduced coverage
    DW_RADAR_QUALITY_REDUCED_COVERAGE = 2,

    /// Radar sensor quality has reduced performance
    DW_RADAR_QUALITY_REDUCED_PERFORMANCE = 3,

    /// Radar sensor quality has reduced coverage and performance
    DW_RADAR_QUALITY_REDUCED_COVERAGE_AND_PERFORMANCE = 4,

    /// Radar sensor in test mode
    DW_RADAR_QUALITY_TEST_MODE = 5,

    /// Radar sensor quality is invalid
    DW_RADAR_QUALITY_INVALID = 6,
} dwRadarDataQuality;

/// Defines the detection status
typedef enum {
    /// Detection is invalid
    DW_RADAR_DETECTION_RECOGNITION_INVALID = 0,

    /// All detections included
    DW_RADAR_DETECTION_RECOGNITION_ALL_DETECTIONS_INCLUDED = 1,

    /// Too many detection and sorting completed
    DW_RADAR_DETECTION_RECOGNITION_TOO_MANY_DETECTIONS_SORTING_COMPLETED = 2,

    /// Too many detection and sorting failed
    DW_RADAR_DETECTION_RECOGNITION_TOO_MANY_DETECTIONS_SORTING_FAILED = 3,
} dwRadarDetectionStatus;

typedef struct dwRadarDetectionMisc
{
    /// Applied measurement model to resolve this detection. Each number corresponds to a specific state (NoUnit)
    /// Measurement model definition will be updated later
    uint8_t measurementModel;

    /// Masking angle sector in which weak targets are not detected in azimuth dimension (rad)
    float32_t maskAngleSectAzi;

    /// Masking angle sector in which weak targets are not detected in elevation dimension (rad)
    float32_t maskAngleSectElev;

    /// Detection's received signal strength. 0 dB represents the case that the full transmitted power is received. (dB)
    float32_t rxSigStrength;

    /// Detection's peak detection threshold (dB)
    float32_t peakDetectionThreshold;

    /// Index of the ambiguity domain in which the resolved Doppler velocity lies (NoUnit)
    uint8_t dopplerAmbgtIdx;
} dwRadarDetectionMisc;

typedef struct dwRadarDetectionMiscValidity
{
    bool maskAngleSectAziValidity;
    bool maskAngleSectElevValidity;
    bool rxSigStrengthValidity;
    bool peakDetectionThresholdValidity;
    bool dopplerAmbgtIdxValidity;
} dwRadarDetectionMiscValidity;

typedef struct dwRadarDetectionStdDev
{
    /// Standard deviation of the elevation angle (rad)
    float32_t elevStdDev;

    /// Standard deviation of the azimuth angle (rad)
    float32_t aziStdDev;

    /// Standard deviation of the doppler velocity (m/s)
    float32_t dopplerStdDev;

    /// Standard deviation of the range (m)
    float32_t rangeStdDev;
} dwRadarDetectionStdDev;

typedef struct dwRadarDetectionStdDevValidity
{
    bool elevStdDevValidity;
    bool aziStdDevValidity;
    bool dopplerStdDevValidity;
    bool rangeStdDevValidity;
} dwRadarDetectionStdDevValidity;

typedef struct dwRadarDetectionQuality
{
    /// Quality of azimuth measurement which ranges from 0 to 100 (NoUnit)
    /// A high value indicates a good accordance with the model. A low value does not necessarily
    /// imply a low quality of the measurement but rather increases the probability for a
    /// large measurement error
    uint8_t aziQuality;

    /// Quality of elevation measurement which ranges from 0 to 100 (NoUnit)
    /// A high value indicates a good accordance with the model. A low value does not necessarily
    /// imply a low quality of the measurement but rather increases the probability for a
    /// large measurement error
    uint8_t elevQuality;

    /// Quality of range measurement which ranges from 0 to 100 (NoUnit)
    /// A high value indicates a good accordance with the model. A low value does not necessarily
    /// imply a low quality of the measurement but rather increases the probability for a
    /// large measurement error
    uint8_t rangeQuality;

    /// Quality of doppler velocity measurement which ranges from 0 to 100 (NoUnit)
    /// A high value indicates a good accordance with the model. A low value does not necessarily
    /// imply a low quality of the measurement but rather increases the probability for a
    /// large measurement error
    uint8_t dopplerQuality;
} dwRadarDetectionQuality;

typedef struct dwRadarDetectionQualityValidity
{
    bool aziQualityValidity;
    bool elevQualityValidity;
    bool rangeQualityValidity;
    bool dopplerQualityValidity;
} dwRadarDetectionQualityValidity;

typedef struct dwRadarDetectionProbability
{
    /// Detection's existence probability (percent, e.g, 100 means 100%)
    uint8_t existProbb;

    /// The probability that this detections represents multiple unresolved
    /// detections (percent, e.g, 100 means 100%)
    uint8_t multiTrgtProbb;

    /// Ambiguous detections are assigned the same ambiguity id and unambiguous detections get the ID zero (NoUnit)
    uint16_t ambgtID;

    /// Probability that the detection represents the real reflection position
    /// among the set of all hypotheses (percent, e.g, 100 means 100%)
    uint8_t ambgtProbb;
} dwRadarDetectionProbability;

typedef struct dwRadarDetectionProbabilityValidity
{
    bool existProbbValidity;
    bool multiTrgtProbbValidity;
    bool ambgtIDValidity;
    bool ambgtProbbValidity;
} dwRadarDetectionProbabilityValidity;

typedef struct dwRadarDetectionFFTPatch
{
    /// FFT Patch value at center bin (dB)
    float32_t center;

    /// FFT Patch value at azimuth bin minus 2 (dB)
    float32_t aziM2;

    /// FFT Patch value at azimuth bin minus 1 (dB)
    float32_t aziM1;

    /// FFT Patch value at azimuth bin plus 1 (dB)
    float32_t azi1;

    /// FFT Patch value at azimuth bin plus 2 (dB)
    float32_t azi2;

    /// FFT Patch value at Doppler bin minus 2 (dB)
    float32_t dopplerM2;

    /// FFT Patch value at Doppler bin minus 1 (dB)
    float32_t dopplerM1;

    /// FFT Patch value at Doppler bin plus 1 (dB)
    float32_t doppler1;

    /// FFT Patch value at Doppler bin plus 2 (dB)
    float32_t doppler2;

    /// FFT Patch value at Range bin minus 2 (dB)
    float32_t rangeM2;

    /// FFT Patch value at Range bin minus 1 (dB)
    float32_t rangeM1;

    /// FFT Patch value at Range bin plus 1 (dB)
    float32_t range1;

    /// FFT Patch value at Range bin plus 2 (dB)
    float32_t range2;
} dwRadarDetectionFFTPatch;

typedef struct dwRadarDetectionFFTPatchValidity
{
    bool centerValidity;
    bool aziM2Validity;
    bool aziM1Validity;
    bool azi1Validity;
    bool azi2Validity;
    bool dopplerM2Validity;
    bool dopplerM1Validity;
    bool doppler1Validity;
    bool doppler2Validity;
    bool rangeM2Validity;
    bool rangeM1Validity;
    bool range1Validity;
    bool range2Validity;
} dwRadarDetectionFFTPatchValidity;

typedef struct dwRadarScanMisc
{
    /// Cycle time of sensor (radar cycle time plus the time for preparation of Ethernet packages) (us)
    dwTime_t cycleTime;

    /// Measure duration (us)
    dwTime_t duration;

    /// Data quality
    dwRadarDataQuality quality;

    /// Sensor ID
    uint8_t sensorID;

    /// Maximum number of detections, that sensor could produce
    uint32_t maxReturns;

    /// Current modulation mode of the sensor. Each number corresponds to a specific state
    /// Modulation mode definition will be updated later
    uint8_t modulationMode;

    /// Current status of the sensor. Each number corresponds to a specific state
    dwRadarDetectionStatus status;

    /// Covariance coefficient of the range and doppler dimension
    float32_t rangeDopplerCovCoeff;

    /// Probability of a low range detection (percent)
    uint8_t lowRangeInd;
} dwRadarScanMisc;

typedef struct dwRadarScanMiscValidity
{
    bool cycleTimeValidity;
    bool durationValidity;
    bool sensorIDValidity;
    bool maxReturnsValidity;
    bool modulationModeValidity;
    bool rangeDopplerCovCoeffValidity;
    bool lowRangeIndValidity;
} dwRadarScanMiscValidity;

typedef struct dwRadarScanAmbiguity
{
    /// Lower limit of the sensor's unambiguous  azimuth (rad)
    float32_t aziAnglAmbgtDLowLmt;

    /// Upper limit of the sensor's unambiguous  azimuth (rad)
    float32_t aziAnglAmbgtDUpLmt;

    /// Lower limit of the sensor's unambiguous  doppler range (m/s)
    float32_t dopplerAmbgtDLowLmt;

    /// Upper limit of the sensor's unambiguous  doppler range (m/s)
    float32_t dopplerAmbgtDUpLmt;

    /// Lower limit of the sensor's unambiguous  elevation (rad)
    float32_t elevAnglAmbgtDLowLmt;

    /// Upper limit of the sensor's unambiguous  elevation (rad)
    float32_t elevAnglAmbgtDUpLmt;

    /// Upper limit of the sensor's range ambiguity (m)
    float32_t rangeAmbgtD;
} dwRadarScanAmbiguity;

typedef struct dwRadarScanAmbiguityValidity
{
    bool aziAnglAmbgtDLowLmtValidity;
    bool aziAnglAmbgtDUpLmtValidity;
    bool dopplerAmbgtDLowLmtValidity;
    bool dopplerAmbgtDUpLmtValidity;
    bool elevAnglAmbgtDLowLmtValidity;
    bool elevAnglAmbgtDUpLmtValidity;
    bool rangeAmbgtDValidity;
} dwRadarScanAmbiguityValidity;

typedef struct dwRadarScanValidity
{
    dwRadarScanMiscValidity radarScanMiscValidity;
    dwRadarScanAmbiguityValidity radarScanAmbiguityValidity;
    dwRadarDetectionMiscValidity detectionMiscValidity;
    dwRadarDetectionStdDevValidity detectionStdDevValidity;
    dwRadarDetectionQualityValidity detectionQualityValidity;
    dwRadarDetectionProbabilityValidity detectionProbabilityValidity;
    dwRadarDetectionFFTPatchValidity detectionFFTPatchValidity;
} dwRadarScanValidity;

/// Defines the defect of radar
typedef enum {
    /// No defect
    DW_RADAR_HEALTH_DEFECT_FULL_FUNC = 0,

    /// Has defect and parital function
    DW_RADAR_HEALTH_DEFECT_NOT_FULL_FUNC = 1,

    /// Has out of order defect
    DW_RADAR_HEALTH_DEFECT_OUT_OF_ORDER = 2,

    /// Defect info is not available
    DW_RADAR_HEALTH_DEFECT_SNA = 3,
} dwRadarHealthDefect;

/// Defines the reason of defect
typedef enum {
    /// No defect
    DW_RADAR_HEALTH_DEFECT_RSN_NO_DEFECT = 0,

    /// Radar wakeup line is malfunctioning
    DW_RADAR_HEALTH_DEFECT_RSN_WAKEUP_LINE_MALFUNC = 1,

    /// Radar eth stack is malfunctioning
    DW_RADAR_HEALTH_DEFECT_RSN_ETHERNET_MALFUNC = 2,

    /// Radar has identification issue
    DW_RADAR_HEALTH_DEFECT_RSN_SENSOR_IDENTIFICATION_ISSUE = 3,

    /// Generic failure in Radar
    DW_RADAR_HEALTH_DEFECT_RSN_RADAR_FAILURE = 4,

    /// Hardware failure in Radar
    DW_RADAR_HEALTH_DEFECT_RSN_HARDWARE_FAILURE = 5,

    /// Software failure in Radar
    DW_RADAR_HEALTH_DEFECT_RSN_SOFTWARE_FAILURE = 6,

    /// Input siganls of Radar are malfunctioning
    DW_RADAR_HEALTH_DEFECT_RSN_INPUT_SIGNALS_MALFUNC = 7,
} dwRadarHealthDefectRsn;

/// Defines the severity of blockage
typedef enum {
    /// No blockage
    DW_RADAR_HEALTH_NO_BLOCKAGE = 0,

    // Full blockage
    DW_RADAR_HEALTH_FULL_BLOCKAGE = 1,

    /// High blockage
    DW_RADAR_HEALTH_PART_BLOCKAGE_HIGH = 2,

    /// Medium blockage
    DW_RADAR_HEALTH_PART_BLOCKAGE_MEDIUM = 3,

    /// Low blockage
    DW_RADAR_HEALTH_PART_BLOCKAGE_LOW = 4,

    /// Blockage has defect
    DW_RADAR_HEALTH_PART_BLOCKAGE_DEFECT = 5,
} dwRadarHealthBlockage;

/// Defines the bitmasks of errors detected by diagnostic function
typedef enum {
    /// Low voltage
    DW_RADAR_HEALTH_ERROR_VOLT_LOW = 1 << 0,

    /// High voltage
    DW_RADAR_HEALTH_ERROR_VOLT_HIGH = 1 << 1,

    /// Under temperature
    DW_RADAR_HEALTH_ERROR_TEMP_UNDER = 1 << 2,

    /// Over temperature
    DW_RADAR_HEALTH_ERROR_TEMP_OVER = 1 << 3,

    /// Interference error
    DW_RADAR_HEALTH_ERROR_INTERFERENCE = 1 << 4,

    /// Error in timesync
    DW_RADAR_HEALTH_ERROR_TIMESYNC = 1 << 5,

    /// Has unavailable data in the scan
    DW_RADAR_HEALTH_ERROR_SNA = 1 << 6,
} dwRadarHealthError;

/// Defines the structure for holding health info of radar scen
typedef struct dwRadarScanHealth
{
    /// Defect info of radar scan
    dwRadarHealthDefect defect;

    /// Defect reason of radar scan
    dwRadarHealthDefectRsn defectRsn;

    /// Blockage info of radar scan
    dwRadarHealthBlockage blockage;

    /// Bitmask of Health errors
    dwRadarHealthError errors;
} dwRadarScanHealth;

#endif // DW_SENSORS_RADAR_RADARSCAN_H_

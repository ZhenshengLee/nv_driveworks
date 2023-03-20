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
// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/// Defines the detection misc
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

/// Defines the validity of the detection misc.
typedef struct dwRadarDetectionMiscValidity
{
    /// The validity of maskAngleSectAzi in struct dwRadarDetectionMisc
    bool maskAngleSectAziValidity;

    /// The validity of maskAngleSectElev in struct dwRadarDetectionMisc
    bool maskAngleSectElevValidity;

    /// The validity of rxSigStrength in struct dwRadarDetectionMisc
    bool rxSigStrengthValidity;

    /// The validity of peakDetectionThreshold in struct dwRadarDetectionMisc
    bool peakDetectionThresholdValidity;

    /// The validity of dopplerAmbgtIdx in struct dwRadarDetectionMisc
    bool dopplerAmbgtIdxValidity;
} dwRadarDetectionMiscValidity;

/// Defines standard deviation of the detection.
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

/// Defines the validity of standard deviation of the detection.
typedef struct dwRadarDetectionStdDevValidity
{
    /// The validity of elevStdDev in struct dwRadarDetectionStdDev
    bool elevStdDevValidity;

    /// The validity of aziStdDev in struct dwRadarDetectionStdDev
    bool aziStdDevValidity;

    /// The validity of dopplerStdDev in struct dwRadarDetectionStdDev
    bool dopplerStdDevValidity;

    /// The validity of rangeStdDev in struct dwRadarDetectionStdDev
    bool rangeStdDevValidity;
} dwRadarDetectionStdDevValidity;

/// Defines the quality of the detection.
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

/// Defines the validity of the detection quality.
typedef struct dwRadarDetectionQualityValidity
{
    /// The validity of aziQuality in struct dwRadarDetectionQuality.
    bool aziQualityValidity;

    /// The validity of elevQuality in struct dwRadarDetectionQuality.
    bool elevQualityValidity;

    /// The validity of rangeQuality in struct dwRadarDetectionQuality.
    bool rangeQualityValidity;

    /// The validity of dopplerQuality in struct dwRadarDetectionQuality.
    bool dopplerQualityValidity;
} dwRadarDetectionQualityValidity;

/// Defines the probability of some items recevied in a detection.
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

    /// Probability that the detection represents the real reflection position
    /// among the set of all hypotheses (percent, e.g, 100 means 100%)
    uint8_t ambgtProbbElev;

    /// Probability that the detection represents the real reflection position
    /// among the set of all hypotheses (percent, e.g, 100 means 100%)
    uint8_t ambgtProbbAzi;
} dwRadarDetectionProbability;

/// Defines the validity flag of the probability.
typedef struct dwRadarDetectionProbabilityValidity
{
    /// The validity of existProbb in struct dwRadarDetectionProbability.
    bool existProbbValidity;

    /// The validity of multiTrgtProbb in struct dwRadarDetectionProbability.
    bool multiTrgtProbbValidity;

    /// The validity of ambgtID in struct dwRadarDetectionProbability.
    bool ambgtIDValidity;

    /// The validity of ambgtProbb in struct dwRadarDetectionProbability.
    bool ambgtProbbValidity;

    /// The validity of ambgtProbbElev in struct dwRadarDetectionProbability.
    bool ambgtProbbElevValidity;

    /// The validity of ambgtProbbAzi in struct dwRadarDetectionProbability.
    bool ambgtProbbAziValidity;
} dwRadarDetectionProbabilityValidity;

/// Defines FFT patch value of the detection.
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

/// Defines the validity of FFT patch value in the detection.
typedef struct dwRadarDetectionFFTPatchValidity
{
    /// The validity of center in struct dwRadarDetectionFFTPatch.
    bool centerValidity;

    /// The validity of aziM2 in struct dwRadarDetectionFFTPatch.
    bool aziM2Validity;

    /// The validity of aziM1 in struct dwRadarDetectionFFTPatch.
    bool aziM1Validity;

    /// The validity of azi1 in struct dwRadarDetectionFFTPatch.
    bool azi1Validity;

    /// The validity of azi2 in struct dwRadarDetectionFFTPatch.
    bool azi2Validity;

    /// The validity of dopplerM2 in struct dwRadarDetectionFFTPatch.
    bool dopplerM2Validity;

    /// The validity of dopplerM1 in struct dwRadarDetectionFFTPatch.
    bool dopplerM1Validity;

    /// The validity of doppler1 in struct dwRadarDetectionFFTPatch.
    bool doppler1Validity;

    /// The validity of doppler2 in struct dwRadarDetectionFFTPatch.
    bool doppler2Validity;

    /// The validity of rangeM2 in struct dwRadarDetectionFFTPatch.
    bool rangeM2Validity;

    /// The validity of rangeM1 in struct dwRadarDetectionFFTPatch.
    bool rangeM1Validity;

    /// The validity of range1 in struct dwRadarDetectionFFTPatch.
    bool range1Validity;

    /// The validity of range2 in struct dwRadarDetectionFFTPatch.
    bool range2Validity;
} dwRadarDetectionFFTPatchValidity;

/// Defines the radar scan misc.
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

/// Defines the validity of the radar scan misc.
typedef struct dwRadarScanMiscValidity
{
    /// The validity of cycleTime in struct dwRadarScanMisc.
    bool cycleTimeValidity;

    /// The validity of duration in struct dwRadarScanMisc.
    bool durationValidity;

    /// The validity of sensorID in struct dwRadarScanMisc.
    bool sensorIDValidity;

    /// The validity of maxReturns in struct dwRadarScanMisc.
    bool maxReturnsValidity;

    /// The validity of modulationMode in struct dwRadarScanMisc.
    bool modulationModeValidity;

    /// The validity of rangeDopplerCovCoeff in struct dwRadarScanMisc.
    bool rangeDopplerCovCoeffValidity;

    /// The validity of lowRangeInd in struct dwRadarScanMisc.
    bool lowRangeIndValidity;
} dwRadarScanMiscValidity;

/// Defines the ambiguity of a radar scan.
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

/// Defines the validity of the ambiguity.
typedef struct dwRadarScanAmbiguityValidity
{
    /// The validity of aziAnglAmbgtDLowLmt in struct dwRadarScanAmbiguity.
    bool aziAnglAmbgtDLowLmtValidity;
    /// The validity of aziAnglAmbgtDUpLmt in struct dwRadarScanAmbiguity.
    bool aziAnglAmbgtDUpLmtValidity;
    /// The validity of dopplerAmbgtDLowLmt in struct dwRadarScanAmbiguity.
    bool dopplerAmbgtDLowLmtValidity;
    /// The validity of dopplerAmbgtDUpLmt in struct dwRadarScanAmbiguity.
    bool dopplerAmbgtDUpLmtValidity;
    /// The validity of elevAnglAmbgtDLowLmt in struct dwRadarScanAmbiguity.
    bool elevAnglAmbgtDLowLmtValidity;
    /// The validity of elevAnglAmbgtDUpLmt in struct dwRadarScanAmbiguity.
    bool elevAnglAmbgtDUpLmtValidity;
    /// The validity of rangeAmbgtD in struct dwRadarScanAmbiguity.
    bool rangeAmbgtDValidity;
} dwRadarScanAmbiguityValidity;

/// Defines the validity of features in a radar scan. Include the validity structure in this page.
typedef struct dwRadarScanValidity
{
    /// Defines the validity of the radar scan misc.
    dwRadarScanMiscValidity radarScanMiscValidity;
    /// Defines the validity of the ambiguity.
    dwRadarScanAmbiguityValidity radarScanAmbiguityValidity;
    /// Defines the validity of the detection misc.
    dwRadarDetectionMiscValidity detectionMiscValidity;
    /// Defines the validity of standard deviation of the detection.
    dwRadarDetectionStdDevValidity detectionStdDevValidity;
    /// Defines the validity of the detection quality.
    dwRadarDetectionQualityValidity detectionQualityValidity;
    /// Defines the validity flag of the probability.
    dwRadarDetectionProbabilityValidity detectionProbabilityValidity;
    /// Defines the validity of FFT patch value in the detection.
    dwRadarDetectionFFTPatchValidity detectionFFTPatchValidity;
} dwRadarScanValidity;

#endif // DW_SENSORS_RADAR_RADARSCAN_H_

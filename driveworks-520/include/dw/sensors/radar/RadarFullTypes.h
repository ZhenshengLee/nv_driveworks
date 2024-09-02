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
 * <b>NVIDIA DriveWorks API: Radar</b>
 *
 * @b Description: This file defines the Radar sensor types.
 */

/**
 * @defgroup radar_group Radar Sensor
 * @ingroup sensors_group
 *
 * @brief Defines the Radar sensor types.
 *
 * @{
 */

#ifndef DW_SENSORS_RADAR_RADARFULLTYPES_H_
#define DW_SENSORS_RADAR_RADARFULLTYPES_H_

#include <dw/sensors/radar/RadarScan.h>
#include <dw/sensors/radar/RadarTypes.h>

// The structs below are serialized in binary and the layout is assummed to be packed
#pragma pack(push, 1)

// Forward declare dwRadarSSI as dwRadarScanSSI which is not public and so not available as of current release. Will be added in future release.
typedef struct dwRadarSSI dwRadarScanSSI;

/// Defines the structure for a complete radar scan
typedef struct dwRadarScan
{
    /// Sensor-provided scan index
    uint32_t scanIndex;

    /// Sensor timestamp at which the current measurement scan was started (us). Same time domain as hostTimestamp
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

    /// radar supplement status info such as calibration info, health signal, performance
    dwRadarScanSSI const* radarSSI;
} dwRadarScan;

#pragma pack(pop)

/** @} */
#endif // DW_SENSORS_RADAR_RADARFULLTYPES_H_

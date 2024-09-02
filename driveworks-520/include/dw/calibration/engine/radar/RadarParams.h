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
// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Calibration</b>
 *
 * @b Description: Contains parameters for initializing a radar calibration
 *
 */

#ifndef DW_CALIBRATION_ENGINE_RADAR_RADARPARAMS_H_
#define DW_CALIBRATION_ENGINE_RADAR_RADARPARAMS_H_

#include <dw/calibration/engine/common/CalibrationTypes.h>
#include <dw/sensors/radar/Radar.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Selection for all radar pitch calibration methods
typedef enum dwCalibrationRadarPitchMethod {
    /// pitch calibration is disabled
    DW_CALIBRATION_RADAR_PITCH_METHOD_NONE = 0,
    /// dynamic object based pitch estimation
    DW_CALIBRATION_RADAR_PITCH_METHOD_DYNAMIC_OBJECT_BASED = 1,
    /// ground plane based pitch estimation
    DW_CALIBRATION_RADAR_PITCH_METHOD_GROUND_PLANE_BASED = 2,
    /// get the calibration result from radar supplier
    /// Notes: not all radar models support this mode.
    /// Supporting radars include ARS620, ARS540 and HELLA_GEN6
    DW_CALIBRATION_RADAR_PITCH_METHOD_SUPPLIER_PROVIDED = 3,
    /// let the calibration engine decide if pitch calibration is enabled
    DW_CALIBRATION_RADAR_PITCH_METHOD_AUTOMATIC = 4,

    DW_CALIBRATION_RADAR_PITCH_METHOD_FORCE32 = 0x7FFFFFFF,
} dwCalibrationRadarPitchMethod;

/**
 * @brief Calibration parameters for calibrating a radar sensor
 * this should be added to the dwCalibrationParams params member
 */
typedef struct dwCalibrationRadarParams
{
    /// Enable radar-based estimation of wheel radii.
    bool enableWheelRadiiEstimation;

    /// pitch calibration mode
    dwCalibrationRadarPitchMethod pitchMode;

    /// Specification of fast-acceptance behaviour.
    /// By default, fast-acceptance is currently not enabled for radar calibrations (this might change in
    /// future DW versions). Enabled fast-acceptance additionally requires previously accepted estimates to be
    /// active.
    dwCalibrationFastAcceptanceOption fastAcceptance;

    /// A pointer to user data that will be passed along when a sensor calibration data has been changed
    void* userData;

    /// An optional pointer to a function that will be called when the calibration status of a routine
    /// has changed.  The function should be valid to call for as long as the sensor is being calibrated
    dwCalibrationStatusChanged onChanged;

    /// Pointer to Radar properties
    /// This pointer can't be null and needs to point to the properties of the radar to be calibrated
    dwRadarProperties const* radarProperties;
} dwCalibrationRadarParams;

#ifdef __cplusplus
} //extern C
#endif

#endif //DW_CALIBRATION_ENGINE_RADAR_RADARPARAMS_H_

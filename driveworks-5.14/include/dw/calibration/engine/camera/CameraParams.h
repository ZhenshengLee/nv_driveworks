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
// SPDX-FileCopyrightText: Copyright (c) 2016-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @b Description: Contains parameters for initializing a camera calibration
 */

/**
 * @defgroup calibration_camera_group Camera Calibration
 * @ingroup calibration_group
 *
 * @brief Parameters for initializing a camera calibration
 *
 * @{
 *
 */

#ifndef DW_CALIBRATION_ENGINE_CAMERA_CAMERAPARAMS_H_
#define DW_CALIBRATION_ENGINE_CAMERA_CAMERAPARAMS_H_

#include <dw/calibration/engine/common/CalibrationTypes.h>
#include <dw/calibration/cameramodel/CameraModel.h>

#include <dw/sensors/camera/Camera.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Camera calibration method used for estimation
typedef enum dwCalibrationCameraMethod {
    /// Feature-based calibration, supports estimation of pitch+yaw, roll, and height signals,
    /// data is provided with `dwCalibrationEngine_addFeatureDetections()`
    DW_CALIBRATION_CAMERA_METHOD_FEATURES = 0
} dwCalibrationCameraMethod;

/// Camera calibration signals to estimate
/// (either `DW_CALIBRATION_CAMERA_SIGNAL_DEFAULT` or combination of explicit signals)
typedef enum dwCalibrationCameraSignal {
    /// Activate supported signals based on properties of the camera sensor
    DW_CALIBRATION_CAMERA_SIGNAL_DEFAULT = 0,

    /// Pitch+yaw estimation, supported by feature-based calibration
    DW_CALIBRATION_CAMERA_SIGNAL_PITCHYAW = 1 << 0,

    /// Roll estimation, supported by feature-based calibration
    DW_CALIBRATION_CAMERA_SIGNAL_ROLL = 1 << 1,

    /// Height estimation, supported by feature-based calibration
    DW_CALIBRATION_CAMERA_SIGNAL_HEIGHT = 1 << 2
} dwCalibrationCameraSignal;

/**
 * @brief Calibration parameters for calibrating a camera sensor.
 *        This should be added to the dwCalibrationParams `params` member.
 */
typedef struct dwCalibrationCameraParams
{
    /// Calibration method used for estimation
    dwCalibrationCameraMethod method;

    /// Signals to be estimated (can be a bitwise "or" of `dwCalibrationCameraSignal` flags)
    dwCalibrationCameraSignal signals;

    struct
    {
        /// The maximum number of tracked features per camera
        uint32_t maxFeatureCount;

        /// The maximum number of positions in a feature's location history
        uint32_t maxFeatureHistorySize;
    } features;

    /// A handle to calibrated camera to use in the camera calibration routine.  This
    /// handle can be based off the same sensor index as passed into the calibration routine.
    /// If this parameter is DW_NULL_HANDLE then a calibrated camera will be created
    /// from the rig configuration. If a valid handle is passed,
    /// it will be used and the one from rig configuration will be ignored.
    /// Note: internally the calibration routine will clone this handle, hence the handle can be reused right after the initialization
    dwConstCameraModelHandle_t calibratedCamera;

    /// Specification of fast-acceptance behaviour.
    /// By default, fast-acceptance is currently not enabled for camera calibrations (this might change in
    /// future DW versions). Enabled fast-acceptance additionally requires previously accepted estimates to be
    /// active.
    dwCalibrationFastAcceptanceOption fastAcceptance;

    /// A pointer to user data that will be passed along when a sensor calibration data has been changed
    void* userData;

    /// An optional pointer to a function that will be called when the calibration status of a routine
    /// has changed. The function should be valid to call for as long as the sensor is being calibrated.
    dwCalibrationStatusChanged onChanged;

    /// Pointer to Camera properties
    /// This pointer can't be null and needs to point to the properties of the camera to be calibrated
    dwCameraProperties const* cameraProperties;

    /// index of the extrinsic profile to be calibrated
    uint32_t extrinsicProfileIndex;
} dwCalibrationCameraParams;

#ifdef __cplusplus
} // extern C
#endif
/** @} */

#endif // DW_CALIBRATION_ENGINE_CAMERA_CAMERAPARAMS_H_

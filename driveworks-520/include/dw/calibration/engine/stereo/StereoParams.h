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
 * @b Description: Contains parameters for initializing a stereo calibration
 *
 * @{
 */

#ifndef DW_CALIBRATION_ENGINE_STEREO_STEREOPARAMS_H_
#define DW_CALIBRATION_ENGINE_STEREO_STEREOPARAMS_H_

#include <dw/calibration/engine/common/CalibrationTypes.h>
#include <dw/calibration/cameramodel/CameraModel.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calibration parameters for calibrating a stereo sensor.
 *        This should be added to the dwCalibrationParams `params` member.
 */
typedef struct dwCalibrationStereoParams
{
    struct
    {
        /// The maximum number of matches of the camera pair
        uint32_t maxMatchesCount;
    } epipolar;

    /// A handle for each calibrated camera to use in the stereo calibration routine.  These
    /// handles can be based on the same sensor indeces passed into the calibration routine.
    /// If these parameters are DW_NULL_HANDLE then calibrated cameras will be created
    /// from the rig configuration. If valid handles are passed,
    /// they will be used and the ones from rig configuration will be ignored.
    /// Note: internally the calibration routine will clone these handles, hence the handles can be reused right after the initialization
    dwCameraModelHandle_t calibratedLeftCamera;
    dwCameraModelHandle_t calibratedRightCamera;

    /// A pointer to user data that will be passed along when a sensor calibration data has been changed
    void* userData;

    /// An optional pointer to a function that will be called when the calibration status of a routine
    /// has changed. The function should be valid to call for as long as the sensor is being calibrated.
    dwCalibrationStatusChanged onChanged;
} dwCalibrationStereoParams;

#ifdef __cplusplus
} // extern C
#endif
/** @} */

#endif // DW_CALIBRATION_STEREO_STEREOPARAMS_H_

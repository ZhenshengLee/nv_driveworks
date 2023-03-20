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
// SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @b Description: Contains parameters for initializing a IMU calibration
 *
 */

#ifndef DW_CALIBRATION_ENGINE_IMU_IMUPARAMS_H_
#define DW_CALIBRATION_ENGINE_IMU_IMUPARAMS_H_

#include <dw/calibration/engine/common/CalibrationTypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calibration parameters for calibrating a IMU sensor
 * this should be added to the dwCalibrationParams params member
 */
typedef struct dwCalibrationIMUParams
{
    //! A pointer to user data that will be passed along when a sensor calibration data has been changed
    void* userData;

    //! An optional pointer to a function that will be called when the calibration status of a routine
    //! has changed.  The function should be valid to call for as long as the sensor is being calibrated
    dwCalibrationStatusChanged onChanged;

    //! If known this entry shall indicate expected sampling rate in [Hz] of the imu sensor.
    //! A default value of 100Hz is used if no parameter passed.
    float32_t imuSamplingRateHz;

    //! If known this value shall indicate expected bias range of the gyroscope sensor. The value in [rad/s]
    //! describes the range around bias mean which bias can run to. Usually temperature
    //! controlled/calibrated gyroscopes vary around the mean by few tens of a radian. If 0 is given,
    //! it will be assumed the standard deviation around the bias mean is about +-0.2 [rad/s], ~ +- 12deg/s
    float32_t gyroBiasRange;

    //! Suspension angular compliance around X- and Y-axis.
    //! Angular rotation is given by a linear model, function of acceleration
    //! applied to the body. X-axis value is typically positive, Y-axis negative.
    dwVector2f suspensionCompliance; //!< [deg s^2 / m]

} dwCalibrationIMUParams;

#ifdef __cplusplus
} //extern C
#endif

#endif //DW_CALIBRATION_ENGINE_IMU_IMUPARAMS_H_

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
// SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @b Description: Contains parameters for initializing a vehicle calibration
 *
 */

#ifndef DW_CALIBRATION_VEHICLE_VEHICLEPARAMS_H_
#define DW_CALIBRATION_VEHICLE_VEHICLEPARAMS_H_

#include <dw/calibration/engine/common/CalibrationTypes.h>
#include <dw/calibration/engine/common/VehicleParamsTypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Vehicle steering calibration-related parameters
**/
typedef struct dwCalibrationVehicleParams
{
    /// A pointer to user data that will be passed along when a sensor calibration data has been changed
    void* userData;

    /// An optional pointer to a function that will be called when the calibration status of a routine
    /// has changed.  The function should be valid to call for as long as the sensor is being calibrated
    dwCalibrationStatusChanged onChanged;
} dwCalibrationVehicleParams;

#ifdef __cplusplus
} //extern C
#endif

#endif // DW_CALIBRATION_VEHICLE_VEHICLEPARAMS_H_

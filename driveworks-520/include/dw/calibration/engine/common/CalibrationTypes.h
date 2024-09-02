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
 * @b Description: Handles to calibration module objects, utils and callbacks associated
 *                 with Calibration
 */

/**
 * @defgroup calibration_types_group Calibration Types
 * @ingroup calibration_group
 *
 * @brief Handles to calibration module objects, converter from calibration enum
 *        to human-readable string representation, and callback function
 *        associated with Calibration
 *
 * @{
 *
 */

#ifndef DW_CALIBRATION_ENGINE_CALIBRATIONTYPES_H_
#define DW_CALIBRATION_ENGINE_CALIBRATIONTYPES_H_

#include "CalibrationBaseTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Handles to calibration module objects.
 */
typedef struct dwCalibrationRoutineObject* dwCalibrationRoutineHandle_t;
typedef struct dwCalibrationEngineObject* dwCalibrationEngineHandle_t;

/**
 * @brief Converts a calibration state enum to a human-readable string representation.
 * @param str the returned string
 * @param state the state to translate
 * @return DW_INVALID_ARGUMENT - if the arguments in the parameters are invalid <br>
 *         DW_SUCCESS
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwCalibrationState_toString(const char** str, dwCalibrationState state);

/**
 * @brief Defines a callback function that is called when calibration routine
 * has changed its internal status
 */
typedef void (*dwCalibrationStatusChanged)(dwCalibrationRoutineHandle_t routine,
                                           dwCalibrationStatus status,
                                           void* userData);

#ifdef __cplusplus
} // extern C
#endif
/** @} */

#endif // DW_CALIBRATION_ENGINE_CALIBRATIONTYPES_H_

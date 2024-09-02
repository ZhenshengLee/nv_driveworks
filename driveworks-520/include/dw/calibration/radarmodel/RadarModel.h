/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_CALIBRATION_RADARMODEL_RADARMODEL_H_
#define DW_CALIBRATION_RADARMODEL_RADARMODEL_H_

#include <dw/rig/Rig.h>
#include <dw/sensors/radar/Radar.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file
 * <b>NVIDIA DriveWorks API: Radar Methods</b>
 *
 * @b Description: This file defines radar model.
 */

/**
 * @defgroup radarmodel_group Radar Model
 *
 * @brief Calibrated radar model abstraction and functionality.
 *
 * @{
 */

/**
 * A pointer to the handle representing a calibrated radar model.
 * This object allows correcting azimuth angle bias errors in measured detections.
 */
typedef struct dwRadarModelObject* dwRadarModelHandle_t;

/**
 * A pointer to the handle representing a const calibrated radar model.
 */
typedef struct dwRadarModelObject const* dwConstRadarModelHandle_t;

/**
* Creates a calibrated radar model for a compatible sensor from rig
*
* @param[out] radarModel A pointer to a handle for the created calibrated radar model object.
*                    Has to be released by the user with `dwRadarModel_release`.
* @param[in] sensorId Specifies the index of the radar sensor to create a calibrated radar model for.
* @param[in] obj Specifies the rig configuration module handle containing the radar model definition.
*
* @retval DW_INVALID_ARGUMENT if the radar model pointer is invalid
* @retval DW_CANNOT_CREATE_OBJECT if given `sensorId` is not a radar sensor
* @retval DW_OUT_OF_BOUNDS if the sensorId is larger than the number of sensors in the rig
* @retval DW_NOT_AVAILABLE if the the sensor in the rig does not contain radar model parameters
* @retval DW_SUCCESS if successful
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: No
*/
DW_API_PUBLIC
dwStatus dwRadarModel_initialize(dwRadarModelHandle_t* radarModel, uint32_t sensorId, dwConstRigHandle_t obj);

/**
* Creates a calibrated radar model for a compatible sensor from model config
*
* @param[out] radarModel A pointer to a handle for the created calibrated radar model object.
*                    Has to be released by the user with `dwRadarModel_release`.
* @param[in] radarConfig The radar model configuration parameters.
*
* @retval DW_INVALID_ARGUMENT if the radar model pointer is invalid or if the radar model configuration is invalid
* @retval DW_SUCCESS if successful
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: No
*/
DW_API_PUBLIC
dwStatus dwRadarModel_initializeFromConfig(dwRadarModelHandle_t* radarModel, dwRadarAzimuthCorrectionModelConfig const* radarConfig);

/**
 * Releases the calibrated radar model.
 * This method releases all resources associated with a calibrated radar model.
 *
 * @note This method renders the handle unusable.
 *
 * @param[in] obj The object handle to be released.
 *
 * @retval DW_INVALID_HANDLE if given radar model handle is invalid
 * @retval DW_SUCCESS if successful
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwRadarModel_release(dwRadarModelHandle_t obj);

/**
 * Corrects a measured azimuth angle using the provided radar model.
 *
 * @param[out] correctedAzimuthRad A pointer to the corrected azimuth angle (in radians).
 * @param[in] measuredAzimuthRad The measured azimuth angle that will be corrected (in radians).
 * @param[in] obj Specifies the handle to the calibrated radar model.
 *
 * @retval DW_INVALID_ARGUMENT if the given output pointer is a nullptr
 * @retval DW_INVALID_HANDLE if given radar model handle is invalid
 * @retval DW_SUCCESS if successful
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwRadarModel_getCorrectedAzimuth(float32_t* correctedAzimuthRad,
                                          float32_t measuredAzimuthRad,
                                          dwConstRadarModelHandle_t obj);

/**
 * Apply correction to all detections in a radar scan using the provided radar model.
 *
 * @param[in,out] radarScan A pointer to the radar scan to be corrected in-place.
 * @param[in] obj Specifies the handle to the calibrated radar model.
 *
 * @retval DW_INVALID_ARGUMENT if the given radar scan pointer is a nullptr, or if it contains returns that are not of type detection
 * @retval DW_INVALID_HANDLE if given radar model handle is invalid
 * @retval DW_SUCCESS if successful
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwRadarModel_applyCorrection(dwRadarScan* radarScan,
                                      dwConstRadarModelHandle_t obj);

/** @} */

#ifdef __cplusplus
}
#endif

#endif // DW_CALIBRATION_RADARMODEL_RADARMODEL_H_

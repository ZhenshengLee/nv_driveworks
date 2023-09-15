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
 * <b>NVIDIA DriveWorks API: Radar Doppler Motion</b>
 *
 * @b Description: Defines radar Doppler motion module that is used to estimate
                   the radar motion (heading and speed) from radar scans
 */

/**
 * @defgroup RadarDopplerMotion_group RadarDopplerMotion Interface
 *
 * @brief Provides estimation of the speed and heading of the radar observing doppler based radar measurements.
 *
 * @ingroup egomotion_group
 * @{
 */

#ifndef DW_EGOMOTION_RADAR_DOPPLERMOTIONESTIMATOR_H__
#define DW_EGOMOTION_RADAR_DOPPLERMOTIONESTIMATOR_H__

#include <dw/core/context/Context.h>
#include <dw/sensors/radar/Radar.h>
#include <dw/control/vehicleio/VehicleIOValStructures.h>
#include <dw/rig/Vehicle.h>

#include "DopplerMotionEstimatorTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Handle to a radar-motion module object. */
typedef struct dwRadarDopplerMotionObject* dwRadarDopplerMotionHandle_t;

/**
 * @brief Creates and initializes a GPU-based radar motion estimation module
 *
 * @param[out] obj - A pointer to the radar Doppler motion handle.
 * @param[in] params - The parameters to initialize the radar Doppler motion module.
 * @param[in] stream - The CUDA stream to use for CUDA operations of the radar motion estimator.
 * @param[in] ctx - Specifies the handler to the context to create radar Doppler motion.
 *
 * @return DW_INVALID_HANDLE - If the provided context handle is invalid. <br>
 *         DW_INVALID_ARGUMENT - If the provided doppler motion handle is invalid. <br>
 *         DW_SUCCESS - If the estimator has been initialized successfully. <br>
 */
DW_API_PUBLIC
dwStatus dwRadarDopplerMotion_initialize(dwRadarDopplerMotionHandle_t* obj,
                                         const dwRadarDopplerMotionParams* params,
                                         cudaStream_t stream,
                                         dwContextHandle_t ctx);

/**
 * @brief Get the default Radar Doppler motion parameters.
 *
 * @param[out] params Pointer to initialization params.
 *
 * @return DW_INVALID_ARGUMENT - If params is NULL. <br>
 *         DW_SUCCESS - If the parameters have been retrieved successfully. <br>
 */
DW_API_PUBLIC
dwStatus dwRadarDopplerMotion_getDefaultParams(dwRadarDopplerMotionParams* params);

/**
 * Sets the CUDA stream for CUDA related operations.
 *
 * @note The ownership of the stream remains by the callee.
 *
 * @param[in] stream The CUDA stream to be used. Default is the one passed during initialization.
 * @param[in] obj RadarDopplerMotion handle.
 *
 * @return DW_INVALID_HANDLE - If the given object handle is invalid, null or of wrong type. <br>
 *         DW_SUCCESS - If the CUDA stream has been set successfully. <br>
 */
DW_API_PUBLIC
dwStatus dwRadarDopplerMotion_setCUDAStream(cudaStream_t stream,
                                            dwRadarDopplerMotionHandle_t obj);

/**
 * Gets CUDA stream used by the radar Doppler motion.
 *
 * @param[out] stream The CUDA stream currently used.
 * @param[in] obj RadarDopplerMotion handle.
 *
 * @return DW_INVALID_HANDLE - If the given object handle is invalid, null or of wrong type. <br>
 *         DW_INVALID_ARGUMENT - If given `stream` is invalid. <br>
 *         DW_SUCCESS - If the stream has been retrieved successfully. <br>
 */
DW_API_PUBLIC
dwStatus dwRadarDopplerMotion_getCUDAStream(cudaStream_t* stream,
                                            dwRadarDopplerMotionHandle_t obj);

/**
 * @brief Process the dwRadarScan and compute the radar motion
 *
 * @param[in] radarScan - dwRadarScan to compute motion for.
 * @param[in] obj - RadarDopplerMotion handle.
 *
 * @note Before calling this API, dwRadarDopplerMotion_addVehicleIOState must be called first.
 *
 * @return DW_INVALID_HANDLE - If the given object handle is invalid, null or of wrong type. <br>
 *         DW_INVALID_ARGUMENT - If given `radarScan` is invalid. <br>
 *         DW_NOT_READY - If radarScan doesn't contain enough information to be processed. <br>
 *         DW_SUCCESS - If the async. processing has been launched successfully. <br>
 */
DW_API_PUBLIC
dwStatus dwRadarDopplerMotion_processAsync(const dwRadarScan* radarScan,
                                           dwRadarDopplerMotionHandle_t obj);

/**
 * Notifies the radar module of a changed vehicle state. In case relevant new information is contained in this state
 * then it gets consumed, otherwise state is ignored.
 *
 * @param[in] safeState New dwVehicleIOSafetyState which contains potentially new information for radar consumption
 * @param[in] nonSafeState New dwVehicleIONonSafetyState which contains potentially new information for radar consumption
 * @param[in] actuationFeedback New dwVehicleIOActuationFeedback which contains potentially new information for radar consumption
 * @param[in] obj RadarDopplerMotion handle.
 *
 * @return DW_INVALID_HANDLE - If the given object handle is invalid, null or of wrong type. <br>
 *         DW_SUCCESS - If vehicle state has been passed to radar successfully. <br>
 *
 */
DW_API_PUBLIC
dwStatus dwRadarDopplerMotion_addVehicleIOState(dwVehicleIOSafetyState const* safeState,
                                                dwVehicleIONonSafetyState const* nonSafeState,
                                                dwVehicleIOActuationFeedback const* actuationFeedback,
                                                dwRadarDopplerMotionHandle_t obj);

/**
 * @brief Gets the available radar motion estimation result.
 *
 * @param[out] motion A pointer to the last estimated motion.
 * @param[in] obj - RadarDopplerMotion handle.
 *
 * @return DW_INVALID_HANDLE - If the given object handle is invalid, null or of wrong type. <br>
 *         DW_INVALID_ARGUMENT - If given solution is invalid. <br>
 *         DW_NOT_AVAILABLE - If currently no solution is available, i.e.
                              no call to `dwRadarDopplerMotion_processAsync()` was performed. <br>
 *         DW_SUCCESS - If the latest motion estimation result has been retrieved successfully. <br>
 */
DW_API_PUBLIC
dwStatus dwRadarDopplerMotion_getMotion(dwRadarDopplerMotion* motion,
                                        dwRadarDopplerMotionHandle_t obj);

/**
 * @brief Resets the radar Doppler motion module
 *
 * @param[in] obj A handle to the doppler motion object.
 *
 * @return DW_INVALID_HANDLE - If the given object handle is invalid, null or of wrong type. <br>
 *         DW_SUCCESS - If the module has been reset successfully. <br>
 */
DW_API_PUBLIC
dwStatus dwRadarDopplerMotion_reset(dwRadarDopplerMotionHandle_t obj);

/**
 * @brief Releases the radar Doppler motion module.
 *
 * @param[in] obj A handle to the doppler motion object.
 *
 * @return DW_INVALID_HANDLE - If the given object handle is invalid, null or of wrong type. <br>
 *         DW_SUCCESS - If the module has been released successfully. <br>
 */
DW_API_PUBLIC
dwStatus dwRadarDopplerMotion_release(dwRadarDopplerMotionHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */

#endif // DW_EGOMOTION_RADAR_DOPPLERMOTIONESTIMATOR_H__

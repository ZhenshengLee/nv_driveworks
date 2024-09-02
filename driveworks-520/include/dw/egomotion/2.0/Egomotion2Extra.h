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
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// WARNING!!!
// Please don't use any type definition in this file.
// All of data types in this file are going to be modified and will not
// follow Nvidia deprecation policy.

#ifndef DW_EGOMOTION_2_0_EGOMOTION2EXTRA_H_
#define DW_EGOMOTION_2_0_EGOMOTION2EXTRA_H_

#include <dw/egomotion/base/Egomotion.h>
#include <dw/egomotion/base/EgomotionExtra.h>
#include <dw/egomotion/base/EgomotionState.h>
#include <dw/core/base/TypesExtra.h>
#include <dw/rig/Vehicle.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get advanced accelerometer bias estimator status information.
 *
 * @param[in,out] status Pointer to the current status of the accelerometer bias calibration to be filled out.
 * @param[in,out] maneuvers Pointer to an array of maneuvers of type dwCalibrationManeuverArray to be filled out.
 * @param[in,out] properties Pointer to the properties of the accelerometer bias calibration procedure to be filled out.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_ARGUMENT - if the provided egomotion handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if the given egomotion handle does not support the request. <br>
 *         DW_SUCCESS       - if the status parameters have been queried successfully. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwEgomotion2_getAccelerometerBiasExt(dwEgomotionCalibrationStatus* status, dwEgomotionCalibrationManeuverArray* maneuvers, dwEgomotionCalibrationProperties* properties, dwEgomotionConstHandle_t obj);

/**
 * Update wheel radii.
 *
 * @param[in] wheelRadii New wheel radii with validity information.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_ARGUMENT - if the provided egomotion handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if the given egomotion handle does not support the request. <br>
 *         DW_SUCCESS       - if the wheel radii were successfully added. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwEgomotion2_updateWheelRadii(dwEgomotionCalibratedWheelRadii const* const wheelRadii, dwEgomotionHandle_t obj);

/**
 * Update IMU extrinsics.
 *
 * @param[in] imuToRig New imu extrinsics with validity information.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_ARGUMENT - if the provided egomotion handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if the given egomotion handle does not support the request. <br>
 *         DW_SUCCESS       - if the imu extrinsics were successfully added. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwEgomotion2_updateIMUExtrinsics(dwEgomotionCalibratedExtrinsics const* const imuToRig, dwEgomotionHandle_t obj);

/**
 * Filling the NvSciBufAttrList with given motion model type and pose history size.
 *
 * @param[out] outAttrs Output NvSciBufAttrList
 * @param[in] historySize State history size (maximum capacity). If 0, a default size of 1000 is used.
 *
 * @return DW_INVALID_ARGUMENT - if given outAttrs is null <br>
 *         DW_SUCCESS - if the call was successful. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwEgomotionState2_fillNvSciBufAttrs(NvSciBufAttrList* outAttrs, size_t const historySize);

/**
 * Create empty state for a given motion model type and bind the pose history buffer to a NvSciBufObj buffer.
 * The pose history buffer is pointing to a NvSciBufObj buffer.
 *
 * @param[out] state Handle to be set with pointer to created empty state.
 * @param[in] historySize State history size (maximum capacity). If 0, a default size of 1000 is used.
 * @param[in] nvSciBufObj NvSciBufObj for pose history to store
 * @param[in] ctx Handle of the context.
 *
 * @return DW_INVALID_ARGUMENT - if given state handle is null <br>
 *         DW_NOT_SUPPORTED - if given motion model is not supported by the state <br>
 *         DW_INVALID_HANDLE - if context handle is invalid <br>
 *         DW_SUCCESS - if the call was successful. <br>
 *
 * @note Ownership of the state goes back to caller. The state has to be released with @ref dwEgomotionState_release.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwEgomotionState2_createAndBindNvSciBufEmpty(dwEgomotionStateHandle_t* state, size_t const historySize, NvSciBufObj nvSciBufObj, dwContextHandle_t ctx);

/**
 * Get the raw buffer size required with given motion model type and pose history size.
 *
 * @param[out] outSize Output raw buffer size
 * @param[in] historySize State history size (maximum capacity). If 0, a default size of 1000 is used.
 *
 * @return DW_INVALID_ARGUMENT - if given outSize is null <br>
 *         DW_SUCCESS - if the call was successful. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwEgomotionState2_getRawBufferSize(size_t* outSize, size_t const historySize);

/**
 * Create empty state for a given motion model type and bind the pose history buffer to a raw external buffer
 * The pose history buffer is pointing to a raw external buffer.
 *
 * @param[out] state Handle to be set with pointer to created empty state.
 * @param[in] historySize State history size (maximum capacity). If 0, a default size of 1000 is used.
 * @param[in] dataPtr raw data pointer for pose history to store
 * @param[in] dataSize size of the raw memory region
 * @param[in] ctx Handle of the context.
 *
 * @return DW_INVALID_ARGUMENT - if given state handle is null <br>
 *         DW_NOT_SUPPORTED - if given motion model is not supported by the state <br>
 *         DW_INVALID_HANDLE - if context handle is invalid <br>
 *         DW_SUCCESS - if the call was successful. <br>
 *
 * @note Ownership of the state goes back to caller. The state has to be released with @ref dwEgomotionState_release.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwEgomotionState2_createAndBindRawBuffer(dwEgomotionStateHandle_t* state, size_t const historySize, void* dataPtr, size_t dataSize, dwContextHandle_t ctx);

#ifdef __cplusplus
}
#endif

#endif // DW_EGOMOTION_2_0_EGOMOTION2EXTRA_H_

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
// SPDX-FileCopyrightText: Copyright (c) 2015-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Egomotion Methods</b>
 *
 * @b Description: This file defines the motion models estimating the pose of the vehicle.
 *
 */

/**
 * @defgroup egomotion_group Egomotion Interface
 *
 * @brief Provides vehicle egomotion functionality.
 *
 * The egomotion module provides implementations of motion models with different sensor modalities.
 * Starting from a simple Ackerman-based odometry-only model, to a full-fledged fusion of inertial
 * sensor information.
 *
 * This module provides access to a history of motion estimates with a predefined cadence and length.
 * At any point of time an access into the history can be made to retrieve previous estimates. The
 * access into the history is timestamp-based. If an access falls between two history entries it will
 * be interpolated.
 *
 * In addition to history-based access, all motion models support prediction of the motion into the
 * future.
 *
 * @{
 */

#ifndef DW_EGOMOTION_BASE_EGOMOTION_H_
#define DW_EGOMOTION_BASE_EGOMOTION_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>

#include <dw/control/vehicleio/VehicleIO.h>
#include <dw/control/vehicleio/VehicleIOValStructures.h>

#include <dw/sensors/imu/IMU.h>

#include <dw/rig/Rig.h>
#include <dw/rig/CoordinateSystem.h>

#include "EgomotionTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Egomotion Handle. Alias for pointer to egomotion instance.
 */
typedef struct dwEgomotionObject* dwEgomotionHandle_t;

/**
 * @brief Const Egomotion Handle. Alias for pointer to const egomotion instance.
 */
typedef struct dwEgomotionObject const* dwEgomotionConstHandle_t;

/**
 * Backwards compatibility between dwEgomotionRelativeUncertainty (old) and
 * dwEgomotionTransformationQuality (new).
 */
typedef dwEgomotionTransformationQuality dwEgomotionRelativeUncertainty;

/**
 * Initialize egomotion parameters from a provided RigConfiguration. This will read out vehicle
 * as well as all relevant sensor parameters and apply them on top of default egomotion parameters.
 *
 * @param[out] params Structure to be filled out with vehicle and sensor parameters.
 * @param[in] rigConfiguration Handle to a rig configuration to retrieve parameters from.
 * @param[in] imuSensorName name of the IMU sensor to be used (optional, can be null).
 * @param[in] vehicleSensorName name of the vehicle sensor to be used (optional, can be null).
 *
 * @return DW_INVALID_ARGUMENT - if provided params pointer or rig handle are invalid<br>
 *         DW_FILE_INVALID - if provided sensor could not be found in the rig config<br>
 *         DW_SUCCESS - if parameters have been initialized successfully. <br>
 *
 * @note Clears any existing parameters set in @p params.
 *
 * @note If a sensor name is null, no sensor specific parameters will be extracted from the configuration
 *       for this sensor.
 *
 * @note Sets motionModel based on available sensor if passed in as 0; DW_EGOMOTION_IMU_ODOMETRY if IMU sensor
 *       is present, DW_EGOMOTION_ODOMETRY otherwise.
 *
 * @note Following parameters are extracted from the rig configuration:
 *       CAN sensor:
 *            - "velocity_latency" -> `dwEgomotionParameters.sensorParameters.velocityLatency`
 *            - "velocity_factor" -> `dwEgomotionParameters.sensorParameters.velocityFactor`
 *       IMU sensor:
 *            - "gyroscope_bias" -> `dwEgomotionParameters.gyroscopeBias`
 *            - Position/Orientation of the sensor ->  `dwEgomotionParameters.imu2rig`
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwEgomotion_initParamsFromRig(dwEgomotionParameters* params, dwConstRigHandle_t rigConfiguration,
                                       const char* imuSensorName, const char* vehicleSensorName);

/**
 * Same as @ref dwEgomotion_initParamsFromRig however uses sensor indices in rigConfiguration instead of their names.
 *
 * @param[out] params Structure to be filled out with vehicle and sensor parameters.
 * @param[in] rigConfiguration Handle to a rig configuration to retrieve parameters from
 * @param[in] imuSensorIdx Index of the IMU sensor to be retrieved (optional, can be (uint32_t)-1)
 * @param[in] vehicleSensorIdx Index of the vehicle sensor to be retrieved (optional, can be (uint32_t)-1)
 *
 * @note Clears any existing parameters set in @p params.
 *
 * @note Sets motionModel based on available sensor if passed in as 0; DW_EGOMOTION_IMU_ODOMETRY if IMU sensor
 *       is present, DW_EGOMOTION_ODOMETRY otherwise.
 *
 * @return DW_INVALID_ARGUMENT - if provided params pointer or rig handle are invalid<br>
 *         DW_FILE_INVALID - if provided sensor could not be found in the rig config<br>
 *         DW_SUCCESS - if parameters have been initialized successfully. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwEgomotion_initParamsFromRigByIndex(dwEgomotionParameters* params, dwConstRigHandle_t rigConfiguration,
                                              uint32_t imuSensorIdx, uint32_t vehicleSensorIdx);

/**
* Initializes the egomotion module.
*
* Configuration of the module is provided by the @p params argument. Default parameters
* can be obtained from the @ref dwEgomotion_initParamsFromRig function.
*
* @param[out] obj Handle to be set with pointer to created module.
* @param[in] params A pointer to the configuration parameters of the module.
* @param[in] ctx Specifies the handler to the context under which the Egomotion module is created.
*
* @return DW_INVALID_ARGUMENT - if provided egomotion handle or parameters are invalid. <br>
*         DW_INVALID_HANDLE - if the provided DriveWorks context handle is invalid. <br>
*         DW_SUCCESS - if module has been initialized successfully. <br>
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: No
*/
DW_API_PUBLIC
dwStatus dwEgomotion_initialize(dwEgomotionHandle_t* obj, const dwEgomotionParameters* params, dwContextHandle_t ctx);

/**
 * Resets the state estimate and all history of the egomotion module.
 * All consecutive motion estimates will be relative to the (new) origin.
 *
 * @param[in] obj Egomotion handle to be reset.
 *
 * @return DW_INVALID_HANDLE - if the provided egomotion handle is invalid. <br>
 *         DW_SUCCESS - if the state and history have been reset successfully. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwEgomotion_reset(dwEgomotionHandle_t obj);

/**
 * Releases the egomotion module.
 *
 * @note This method renders the handle unusable.
 *
 * @param[in] obj Egomotion handle to be released.
 *
 * @return DW_INVALID_HANDLE - if the provided egomotion handle is invalid. <br>
 *         DW_SUCCESS - if the egomotion module has been released successfully. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwEgomotion_release(dwEgomotionHandle_t obj);

/**
 * Notifies the egomotion module of a changed vehicle state. In case relevant new information is contained in this state
 * then it gets consumed, otherwise state is ignored.
 *
 * Signals consumed depend on the selected egomotion motion model, @see dwMotionModel.
 *
 * When using DW_EGOMOTION_ODOMETRY:
 * - steeringAngle, steeringTimestamp, wheelSpeed (1), wheelSpeedTimestamp (1),
 *   speed (2, 3), speedTimestamp (2, 3), rearWheelAngle (3)
 *
 * - (1) if dwEgomotionParameters.speedMeasurementType == DW_EGOMOTION_REAR_WHEEL_SPEED
 * - (2) if dwEgomotionParameters.speedMeasurementType == DW_EGOMOTION_FRONT_SPEED
 * - (3) if dwEgomotionParameters.speedMeasurementType == DW_EGOMOTION_REAR_SPEED
 *
 * When using DW_EGOMOTION_IMU_ODOMETRY:
 * - steeringAngle (2), steeringTimestamp (2), wheelSpeed (1), wheelSpeedTimestamp (1),
 *   speed (2, 3), speedTimestamp (2, 3), rearWheelAngle (1, 3)
 *
 * - (1) if dwEgomotionParameters.speedMeasurementType == DW_EGOMOTION_REAR_WHEEL_SPEED
 * - (2) if dwEgomotionParameters.speedMeasurementType == DW_EGOMOTION_FRONT_SPEED
 * - (3) if dwEgomotionParameters.speedMeasurementType == DW_EGOMOTION_REAR_SPEED
 *
 * @param[in] state New VehicleIOState which contains potentially new information for egomotion consumption
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_HANDLE - if provided, the egomotion handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if underlying egomotion handle does not support vehicle state. <br>
 *         DW_SUCCESS - if vehicleIOState has been consumed successfully. <br>
 *
 * @note Deprecation warning: This method has been replaced with dwEgomotion_addVehicleIOState, and will be removed in next major release.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwEgomotion_addVehicleState(const dwVehicleIOState* state, dwEgomotionHandle_t obj);

/**
 * Notifies the egomotion module of a changed vehicle state. In case relevant new information is contained in this state
 * then it gets consumed, otherwise state is ignored.
 *
 * Signals consumed depend on the selected egomotion motion model, @see dwMotionModel.
 *
 * When using DW_EGOMOTION_ODOMETRY, the following valid signals in at least one of the structs
 * dwVehicleIOSafetyState, dwVehicleIONonSafetyState or dwVehicleIOActuationFeedback are required:
 * - frontSteeringAngle, frontSteeringTimestamp, wheelSpeed (1), wheelSpeedTimestamp (1),
 *   speed (2, 3), speedTimestamp (2, 3), rearWheelAngle (3)
 *
 * - (1) if dwEgomotionParameters.speedMeasurementType == DW_EGOMOTION_REAR_WHEEL_SPEED
 * - (2) if dwEgomotionParameters.speedMeasurementType == DW_EGOMOTION_FRONT_SPEED
 * - (3) if dwEgomotionParameters.speedMeasurementType == DW_EGOMOTION_REAR_SPEED
 *
 * When using DW_EGOMOTION_IMU_ODOMETRY, the following valid signals in at least one of the structs
 * dwVehicleIOSafetyState, dwVehicleIONonSafetyState or dwVehicleIOActuationFeedback are required:
 * - frontSteeringAngle (2), frontSteeringTimestamp (2), wheelSpeed (1), wheelTickTimestamp (1),
 *   speed (2, 3), speedTimestamp (2, 3), rearWheelAngle (1, 3)
 *
 * - (1) if dwEgomotionParameters.speedMeasurementType == DW_EGOMOTION_REAR_WHEEL_SPEED
 * - (2) if dwEgomotionParameters.speedMeasurementType == DW_EGOMOTION_FRONT_SPEED
 * - (3) if dwEgomotionParameters.speedMeasurementType == DW_EGOMOTION_REAR_SPEED
 *
 * @param[in] safeState New dwVehicleIOSafetyState which contains potentially new information for egomotion consumption
 * @param[in] nonSafeState New dwVehicleIONonSafetyState which contains potentially new information for egomotion consumption
 * @param[in] actuationFeedback New dwVehicleIOActuationFeedback which contains potentially new information for egomotion consumption
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_HANDLE - if provided, the egomotion handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if underlying egomotion handle does not support vehicle state. <br>
 *         DW_SUCCESS - if vehicle state has been passed to egomotion successfully. <br>
 *
 * @note Deprecation warning: This method has been replaced with dwEgomotion_addVehicleIOState, and will be removed in next major release.
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwEgomotion_addVehicleIOState(dwVehicleIOSafetyState const* safeState,
                                       dwVehicleIONonSafetyState const* nonSafeState,
                                       dwVehicleIOActuationFeedback const* actuationFeedback,
                                       dwEgomotionHandle_t obj);

/**
 * This method updates the egomotion module with an updated vehicle.
 *
 * @param[in] vehicle Updated vehicle which may contain updated vehicle parameters.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_INVALID_ARGUMENT - if provided vehicle is nullptr. <br>
 *         DW_NOT_SUPPORTED - if given egomotion instance does not support change of this parameter. <br>
 *         DW_SUCCESS - if the vehicle has been updated successfully. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 **/
DW_API_PUBLIC
dwStatus dwEgomotion_updateVehicle(const dwVehicle* vehicle, dwEgomotionHandle_t obj);

/**
 * Adds an IMU frame to the egomotion module.
 *
 * The IMU frame shall contain either linear acceleration or angular velocity measurements for X, Y and Z
 * axes or both at once; the frame will be discarded otherwise.
 *
 * Egomotion might generate a new estimate on addition of IMU frame, see @ref dwEgomotionParameters.automaticUpdate.
 *
 * @param[in] imu IMU measurement.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_ARGUMENT - if timestamp is not greater than last measurement timestamp,
 *                               or given frame is invalid. <br/>
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if the given measurement is not supported by the egomotion model. <br>
 *         DW_SUCCESS - if the IMU data has been passed to egomotion successfully. <br/>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 **/
DW_API_PUBLIC
dwStatus dwEgomotion_addIMUMeasurement(const dwIMUFrame* imu, dwEgomotionHandle_t obj);

/**
 * This method updates the IMU extrinsics to convert from the IMU coordinate system
 * to the vehicle rig coordinate system.
 *
 * @param[in] imuToRig Transformation from the IMU coordinate system to the vehicle rig coordinate system.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_INVALID_ARGUMENT - if provided imuToRig is nullptr. <br>
 *         DW_NOT_SUPPORTED - if given egomotion instance does not support change of this parameter. <br>
 *         DW_SUCCESS - if the update was successful. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 **/
DW_API_PUBLIC
dwStatus dwEgomotion_updateIMUExtrinsics(const dwTransformation3f* imuToRig, dwEgomotionHandle_t obj);

/**
 * Runs the motion model estimation for a given timestamp. The internal state is
 * modified. The motion model advances to the given timestamp. To retrieve the
 * result of the estimation, use @ref dwEgomotion_getEstimation.
 *
 * This method allows the user to update the egomotion filter when required, for a specific timestamp, using
 * all sensor data available up to this timestamp.
 *
 * When the automatic update period is active (automaticUpdate in @ref dwEgomotionParameters is set),
 * @ref dwEgomotion_update will not update the filter state and throw a `DW_NOT_SUPPORTED` exception instead.
 *
 * @param[in] timestamp_us Timestamp for which to estimate vehicle state.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_ARGUMENT - if given timestamp is not greater than last update timestamp. <br>
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if the automatic estimation update period is active (i.e. non-zero). <br>
 *         DW_NOT_READY - if egomotion has not yet received enough data to initialize. <br>
 *         DW_SUCCESS - update was executed successfully. <br>
 *
 * @note The provided timestamp must be larger than the previous timestamp used when calling @ref dwEgomotion_estimate.
 *
 * @note A full set of new sensor measurements should be added before calling this method; otherwise the state
 *       estimate might be based on older, extrapolated sensor measurements.
 *
 * @note Updating internal state does not guarantee that a motion estimation is available, i.e.
 *       there might be not enough data to provide an estimate. Use @ref dwEgomotion_getEstimation
 *       and/or @ref dwEgomotion_computeRelativeTransformation to query estimation if it is available.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 **/
DW_API_PUBLIC
dwStatus dwEgomotion_update(dwTime_t timestamp_us, dwEgomotionHandle_t obj);

/**
 * Estimates the state for a given timestamp. This method does not modify
 * internal state and can only be used to extrapolate motion into the future
 * or interpolate motion between estimates the past.
 *
 * @param[out] pose Structure to be filled with estimate state.
 * @param[out] uncertainty Structure to be filled with uncertainties. Can be nullptr if not used.
 * @param[in] timestamp_us Timestamp to estimate state for.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_NOT_AVAILABLE - if requested timestamp is too far from available estimates. <br>
 *         DW_NOT_READY - if egomotion has not yet received enough data to initialize. <br>
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_SUCCESS - if the desired state has been provided successfully. <br>
 *
 * @note Interpolation into the past can only happen within the available history range.
 * @note Interpolation interval is limited to 5 seconds, if interval between two successive poses
 *       is larger, the function will return DW_NOT_AVAILABLE.
 * @note Extrapolation into the future is limited to 2.5 seconds and the function will return
 *       DW_NOT_AVAILABLE if the extrapolation interval is larger.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 **/
DW_API_PUBLIC
dwStatus dwEgomotion_estimate(dwEgomotionResult* pose, dwEgomotionUncertainty* uncertainty,
                              dwTime_t timestamp_us, dwEgomotionConstHandle_t obj);

/**
 * Computes the relative transformation between two timestamps and the uncertainty of this transform.
 *
 * @param[out] poseAtoB Transform to be filled with transformation mapping a point at time `a` to a point at time `b`.
 * @param[out] quality Structure to be filled with quality of transformation (optional, ignored if nullptr provided).
 * @param[in] timestamp_a Timestamp corresponding to beginning of transformation.
 * @param[in] timestamp_b Timestamp corresponding to end of transformation.
 * @param[in] obj Egomotion handle.
 *
 * @note validity of uncertainty estimates is indicated by @ref dwEgomotionTransformationQuality.valid
 * @note uncertainty estimates are currently only supported by DW_IMU_ODOMETRY motion model.
 *
 * @return DW_INVALID_ARGUMENT - if any required argument is invalid. <br>
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_NOT_AVAILABLE - if requested timestamps are outside of available range (@see dwEgomotion_estimate). <br>
 *         DW_NOT_READY - if egomotion has not yet received enough data to initialize. <br>
 *         DW_SUCCESS - if the computation was successful. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 **/
DW_API_PUBLIC
dwStatus dwEgomotion_computeRelativeTransformation(dwTransformation3f* poseAtoB,
                                                   dwEgomotionTransformationQuality* quality,
                                                   dwTime_t timestamp_a, dwTime_t timestamp_b,
                                                   dwEgomotionConstHandle_t obj);

/**
 * Compute the transformation between two coordinate systems at a specific timestamp and the
 * uncertainty of this transform.
 *
 * @param[out] transformationAToB Transformation mapping a point in coordinate system `A`
 *             to a point in coordinate system `B`.
 * @param[out] quality Structure to be filled with quality of transformation (optional, ignored if nullptr provided).
 * @param[in] timestamp Timestamp for which to get transformation.
 * @param[in] coordinateSystemA Coordinate system A.
 * @param[in] coordinateSystemB Coordinate system B.
 * @param[in] obj Egomotion handle.
 *
 * @note quality estimates are not supported at this time and are left unpopulated.
 *
 * @return DW_NOT_AVAILABLE - if the requested timestamp is too far into the past or egomotion modality
 *                            doesn't provide an estimate for the selected coordinate systems. <br>
 *         DW_NOT_READY - if egomotion has not yet received enough data to initialize. <br>
 *         DW_NOT_SUPPORTED - if any coordinate system is not supported by any egomotion. <br>
 *         DW_INVALID_ARGUMENT - if any required argument is invalid. <br>
 *         DW_INVALID_HANDLE - if the provided egomotion handle invalid. <br>
 *         DW_SUCCESS <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 **/
DW_API_PUBLIC
dwStatus dwEgomotion_computeBodyTransformation(dwTransformation3f* const transformationAToB,
                                               dwEgomotionTransformationQuality* const quality,
                                               dwTime_t const timestamp,
                                               dwCoordinateSystem const coordinateSystemA,
                                               dwCoordinateSystem const coordinateSystemB,
                                               dwEgomotionConstHandle_t const obj);

/**
 * Gets the timestamp of the latest state estimate. The timestamp will be updated
 * after each egomotion filter update.
 *
 * @param[out] timestamp Timestamp to be set with latest available state estimate timestamp.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_ARGUMENT - if the provided timestamp pointer is nullptr. <br>
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_NOT_READY - if egomotion has not yet received enough data to initialize. <br>
 *         DW_SUCCESS - if the timestamp has been provided successfully <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwEgomotion_getEstimationTimestamp(dwTime_t* timestamp, dwEgomotionConstHandle_t obj);

/**
 * Check whether estimation is available.
 *
 * @param[out] result Boolean to be set with estimation availablity.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_ARGUMENT - if the provided pointer is nullptr. <br>
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_SUCCESS - if the operation was successful. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwEgomotion_hasEstimation(bool* result, dwEgomotionConstHandle_t obj);

/**
 * Gets the latest state estimate.
 *
 * @param[out] result Structure to be filled with latest state estimate.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_ARGUMENT - if the provided pointer is nullptr. <br>
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_NOT_READY - if egomotion has not yet received enough data to initialize. <br>
 *         DW_SUCCESS - if the estimation has been provided successfully. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwEgomotion_getEstimation(dwEgomotionResult* result, dwEgomotionConstHandle_t obj);

/**
 * Gets the latest state estimate uncertainties.
 *
 * @param[out] result Structure to be filled with latest uncertainties.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_ARGUMENT - if the provided pointer is nullptr. <br>
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_NOT_READY - if egomotion has not yet received enough data to initialize. <br>
 *         DW_SUCCESS - if the uncertainty has been provided successfully. <br>
 *
 * @note Not all values do have valid uncertainties. Check dwEgomotionUncertainty.validFlags.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwEgomotion_getUncertainty(dwEgomotionUncertainty* result, dwEgomotionConstHandle_t obj);

/**
* Get estimated gyroscope bias.
*
* @param[out] gyroBias Vector to be filled with gyroscope biases.
* @param[in] obj Egomotion handle.
*
* @return DW_INVALID_ARGUMENT - if the provided egomotion handle is invalid. <br>
*         DW_NOT_SUPPORTED - if the given egomotion handle does not support the request <br>
*         DW_NOT_READY     - if the online estimation is not ready yet but an initial bias guess is available <br>
*         DW_NOT_AVAILABLE - if the online estimation is not ready yet and no initial bias guess is available <br>
*         DW_SUCCESS       - if online gyroscope bias estimation has accepted a value <br>
* @par API Group
* - Init: Yes
* - Runtime: Yes
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwEgomotion_getGyroscopeBias(dwVector3f* gyroBias, dwEgomotionConstHandle_t obj);

/**
 * Returns the number of elements currently stored in the history.
 *
 * @note the history size is always lower than or equal to the history capacity, which is set at initialization.
 *
 * @param[out] size Integer to be set with the number of elements in the history.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_ARGUMENT - if the provided pointer is nullptr. <br>
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_SUCCESS - if the call was successful. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 **/
DW_API_PUBLIC
dwStatus dwEgomotion_getHistorySize(size_t* size, dwEgomotionConstHandle_t obj);

/**
 * Returns an element from the motion history that is currently available.
 *
 * @param[out] pose Structure to be filled with state estimate at the requested index in history (can be null, in which
 *                 case it will not be filled).
 * @param[out] uncertainty Structure to be filled with uncertainty at the requested index in history (can be null, in
 *                        which case it will not be filled).
 * @param[in] index Index into the history, in the range [0; @ref dwEgomotion_getHistorySize ), with 0 being latest
 *                  estimate and last element pointing to oldest estimate.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_HANDLE - if the egomotion handle is invalid <br>
 *         DW_INVALID_ARGUMENT - if the index is outside of available history <br>
 *         DW_SUCCESS- if the call was successful. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 **/
DW_API_PUBLIC
dwStatus dwEgomotion_getHistoryElement(dwEgomotionResult* pose, dwEgomotionUncertainty* uncertainty,
                                       size_t index, dwEgomotionConstHandle_t obj);

/**
 * Returns the type of the motion model used.
 *
 * @param[out] model dwMotionModel to be set with the motion model type used by instance specified by the handle.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_ARGUMENT - if the provided pointer is nullptr. <br>
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_SUCCESS  if the call was successful. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 **/
DW_API_PUBLIC
dwStatus dwEgomotion_getMotionModel(dwMotionModel* model, dwEgomotionConstHandle_t obj);

//-----------------------------
// utility functions

/**
 * Applies the estimated relative motion as returned by @ref dwEgomotion_computeRelativeTransformation to a given
 * vehicle pose.
 *
 * @param[out] vehicleToWorldAtB Transformation to be filled with pose of the vehicle in world at time B.
 * @param[in] vehicleAToB Transformation providing motion of vehicle from time A to time B.
 * @param[in] vehicleToWorldAtA Transformation providing pose of the vehicle in world at time A.
 *
 * @return DW_INVALID_ARGUMENT - if any of the given arguments is nullptr<br>
 *         DW_SUCCESS - if the call was successful. <br>
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 **/
DW_API_PUBLIC
dwStatus dwEgomotion_applyRelativeTransformation(dwTransformation3f* vehicleToWorldAtB,
                                                 const dwTransformation3f* vehicleAToB,
                                                 const dwTransformation3f* vehicleToWorldAtA);

/**
 * Computes steering angle of the vehicle based on IMU measurement. The computation will take the yaw rate
 * measurement from the IMU and combine it with currently estimated speed and wheel base.
 * Speed used for estimation will be the reported by @ref dwEgomotion_getEstimation.
 *
 * @param[out] steeringAngle Steering angle to be set to estimation from imu yaw rate (can be null, in which case it will not be set).
 * @param[out] inverseSteeringR Inverse radius to be set to arc driven by the vehicle. (can be null, in which case it will not be filled).
 * @param[in] imuMeasurement IMU measurement with IMU measured in vehicle coordinate system.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_ARGUMENT - if any of the provided pointer arguments is nullptr
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_NOT_AVAILABLE - if there is currently no estimation available <br>
 *         DW_SUCCESS - if the call was successful. <br>
 *
 * @note Depending on the underlying model the estimation might differ. Default implementation is to output
 *       steering angle and inverse radius as derived using bicycle model.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 **/
DW_API_PUBLIC
DW_DEPRECATED("dwEgomotion_computeSteeringAngleFromIMU() is deprecated and will be removed soon. ")
dwStatus dwEgomotion_computeSteeringAngleFromIMU(float32_t* steeringAngle,
                                                 float32_t* inverseSteeringR,
                                                 const dwIMUFrame* imuMeasurement,
                                                 dwEgomotionConstHandle_t obj);

/**
 * Convert steering wheel angle to steering angle.
 *
 * @param[out] steeringAngle Pointer to steering angle to be set, in radians.
 * @param[in] steeringWheelAngle Steering wheel angle, in radians.
 * @param[in] obj Specifies the egomotion module handle.
 *
 * @return DW_INVALID_ARGUMENT - if any of the provided pointer arguments is nullptr. <br>
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if underlying egomotion handle does not support steering wheel angle to steering angle
 *                            conversion. The egomotion model has to be DW_EGOMOTION_ODOMETRY or
 *                            DW_EGOMOTION_IMU_ODOMETRY. <br>
 *         DW_SUCCESS - if the call was successful. <br>
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwEgomotion_steeringWheelAngleToSteeringAngle(float32_t* steeringAngle, float32_t steeringWheelAngle,
                                                       dwEgomotionHandle_t obj);

/**
 * Convert steering angle to steering wheel angle
 *
 * @param[out] steeringWheelAngle Pointer to steering wheel angle to be set, in radians.
 * @param[in] steeringAngle Steering angle, in radians.
 * @param[in] obj Specifies the egomotion module handle.
 *
 * @return DW_INVALID_ARGUMENT - if any of the provided pointer arguments is nullptr. <br>
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if underlying egomotion handle does not support steering angle to steering wheel angle
 *                            conversion. The egomotion model has to be DW_EGOMOTION_ODOMETRY or
 *                            DW_EGOMOTION_IMU_ODOMETRY. <br>
 *         DW_SUCCESS - if the call was successful. <br>
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
**/
DW_API_PUBLIC
dwStatus dwEgomotion_steeringAngleToSteeringWheelAngle(float32_t* steeringWheelAngle, float32_t steeringAngle,
                                                       dwEgomotionHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_EGOMOTION_BASE_EGOMOTION_H_

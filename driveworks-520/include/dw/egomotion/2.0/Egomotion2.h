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
// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Copyright (c) 2022 Mercedes-Benz AG. All rights reserved.
//
// Mercedes-Benz AG as copyright owner and NVIDIA Corporation as licensor retain
// all intellectual property and proprietary rights in and to this software
// and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement is strictly prohibited.
//
// This code contains Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE ARE MADE.
//
// No responsibility is assumed for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights. No third party distribution is allowed unless
// expressly authorized.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// Products are not authorized for use as critical
// components in life support devices or systems without express written approval.
//
///////////////////////////////////////////////////////////////////////////////////////

// WARNING!!!
// Please don't use any type definition in this file.
// All of data types in this file are going to be modified and will not
// follow Nvidia deprecation policy.

#ifndef DW_EGOMOTION_2_0_EGOMOTION2_H_
#define DW_EGOMOTION_2_0_EGOMOTION2_H_

#include <dw/egomotion/base/Egomotion.h>
#include <dw/egomotion/base/EgomotionState.h>
#include <dw/egomotion/errorhandling/ErrorHandlingParameters.h>
#include <dw/egomotion/utils/IMUBiasEstimatorParameters.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Maximum number of IMU sensors supported.
 */
#define DW_EGOMOTION_IMU_COUNT_MAX 3

/**
 * @brief Measurements used by egomotion.
 */
typedef enum dwEgomotionMeasurementTypes {
    DW_EGOMOTION_MEASUREMENT_STEERING             = 0, //!< @see dwVehicleIOState steering
    DW_EGOMOTION_MEASUREMENT_SUSPENSION           = 1, //!< @see dwVehicleIOState suspension
    DW_EGOMOTION_MEASUREMENT_ANGULAR_ACCELERATION = 2, //!< IMU angular acceleration [rad/s^2]
    DW_EGOMOTION_MEASUREMENT_PROPER_ACCELERATION  = 3, //!< @see dwIMUFrame acceleration
    DW_EGOMOTION_MEASUREMENT_ANGULAR_VELOCITY     = 4, //!< @see dwIMUFrame turnrate
    DW_EGOMOTION_MEASUREMENT_LINEAR_GROUND_SPEED  = 5, //!< @see dwVehicleIOState speeds
    DW_EGOMOTION_MEASUREMENT_LINEAR_BODY_SPEED    = 6, //!< Linear body speed [m/s]
    DW_EGOMOTION_MEASUREMENT_WHEEL_TICKS          = 7, //! @see dwVehicleIOState wheelTicks
    DW_EGOMOTION_MEASUREMENT_WHEEL_SPEEDS         = 8, //! @see dwVehicleIOState wheelSpeeds
    DW_EGOMOTION_MEASUREMENT_TRAILER_ANGLE        = 9, //! @see dwVehicleIOState articulationAngle
    DW_EGOMOTION_MEASUREMENT_TYPES_COUNT
} dwEgomotionMeasurementTypes;

/**
 * @brief Ground speed measurement types.
 *
 * Egomotion can estimate vehicle ground speed from different sources depending
 * on vehicle platform capabilities.
 */
typedef enum dwEgomotionGroundSpeedMeasurementTypes {
    //! Ground speed from wheel speeds only rear axle (Wheel observer is disabled)
    DW_EGOMOTION_GROUND_SPEED_FROM_WHEEL_SPEEDS_REAR_AXLE = 0,
    //! Ground speed from wheel speeds only both axle (Wheel observer is disabled)
    DW_EGOMOTION_GROUND_SPEED_FROM_WHEEL_SPEEDS_BOTH_AXLES = 1,
    //! Ground speed from wheel ticks and speeds rear axle (Wheel observer is enabled)
    DW_EGOMOTION_GROUND_SPEED_FROM_WHEEL_TICKS_AND_SPEEDS_REAR_AXLE = 2,
    //! Ground speed from wheel ticks and speeds both axles (Wheel observer is enabled)
    DW_EGOMOTION_GROUND_SPEED_FROM_WHEEL_TICKS_AND_SPEEDS_BOTH_AXLES = 3,
    //! Ground speed from signed vehicle linear speed at rear axle center (Wheel observer is disabled)
    DW_EGOMOTION_GROUND_SPEED_FROM_LINEAR_SPEED = 4,
    DW_EGOMOTION_GROUND_SPEED_COUNT
} dwEgomotionGroundSpeedMeasurementTypes;

/**
 * @brief Vehicle direction detection modes.
 *
 * Egomotion can estimate the vehicle moving direction from different sources depending on vehicle
 * platform capabilities.
 *
 * Select DW_EGOMOTION_DIRECTION_FROM_SPEED_SIGN if the wheel tick increments and wheel speeds are
 * signed, indicating forward and reverse driving unambiguously.
 *
 * Select DW_EGOMOTION_DIRECTION_FROM_WHEEL_DIR_AND_GEAR if the wheel tick increments and wheel
 * speeds are unsigned, such that forward and reverse driving cannot be distinguished from these
 * signals alone.
 *
 * Select DW_EGOMOTION_DIRECTION_FROM_SPEED_SIGN_AND_GEAR if the wheel tick increments and wheel
 * speeds are signed with ambiguous region around 0 to be resolved based on target gear and current
 * gear signals. @ref dwEgomotionVehicleMovingDirectionDetectorParameters.ticksAmbiguousSpeedThreshold
 */
typedef enum dwEgomotionDirectionMode {
    //! Vehicle Moving Direction based on signed wheel encoder information
    DW_EGOMOTION_DIRECTION_FROM_SPEED_SIGN = 0,
    //! Vehicle Moving Direction based on unsigned wheel encoder, wheel direction and gear information
    DW_EGOMOTION_DIRECTION_FROM_WHEEL_DIR_AND_GEAR = 1,
    //! Vehicle Moving Direction based on signed wheel encoder and gear information
    DW_EGOMOTION_DIRECTION_FROM_SPEED_SIGN_AND_GEAR = 2,
    DW_EGOMOTION_DIRECTION_COUNT
} dwEgomotionDirectionMode;

/**
 * @brief Egomotion measurement parameters.
 */
typedef struct dwEgomotionMeasurementParameters
{
    //! Delay of each measurement type, relative to its timestamp.
    //! These values are subtracted from the timestamp field in the corresponding sensor data
    //! frame to indicate the true instant of validity of a measurement.
    //! @see dwEgomotionMeasurementTypes for corresponding measurements.
    dwTime_t delay[DW_EGOMOTION_MEASUREMENT_TYPES_COUNT]; //!< [us]

    //! Expected cycle time of each measurement type.
    //! These values are used to allocate internal buffers and monitor new measurement timestamps,
    //! logging and rejecting sensor data frames that deviate strongly from expected cycle time.
    //! @see dwEgomotionMeasurementTypes for corresponding measurements.
    dwTime_t cycleTime[DW_EGOMOTION_MEASUREMENT_TYPES_COUNT]; //!< [us]
} dwEgomotionMeasurementParameters;

/**
 * @brief Vehicle motion observer parameters.
 */
typedef struct dwEgomotionVMOParameters
{
    //! Internal state history size. Recommended to leave 0 for automatic size given
    //! dwEgomotionMeasurementParameters settings.
    size_t stateHistorySize;

    //! Process covariance parameters (continuous time, coefficients on diagonal).
    //! Elements correspond approximatively to [pitch, roll, v_x, v_y, v_z] estimates.
    float32_t processCovariance[5];        //!< [rad^2, rad^2, m^2/s^2, m^2/s^2, m^2/s^2]
    float32_t initialProcessCovariance[5]; //!< [rad^2, rad^2, m^2/s^2, m^2/s^2, m^2/s^2]

    //! Measurement covariance parameters (continuous time, coefficients on diagonal).
    //! Elements correspond to [v_x, v_y, v_z] along ground and rig (body) coordinate frames, where
    //! X points forward, Y left, Z upwards and origin is at the center of the rear axle, projected
    //! on the ground.
    float32_t groundSpeedCovariance[3]; //!< [m^2/s^2, m^2/s^2, m^2/s^2]
    float32_t bodySpeedCovariance[3];   //!< [m^2/s^2, m^2/s^2, m^2/s^2]

    //! Chi-squared value used to reject outlier speed measurements.
    //! Only applies to body speed measurements at this time.
    //! A value of 7.81, corresponding to a p-value of 0.05 (for 3 DoF) is a good choice.
    float32_t chiSquaredInnovationGate; //!< [-]

    //! Variance multiplier value used to reject outlier ground speed
    //! measurements. Only applies to longitudinal ground speed measurements
    //! at this time. A value of 2.5, corresponding to a 98.7% probability of
    //! detecting outlier measurements is a decent choice. Higher values will
    //! result in fewer measurements being rejected.
    float32_t longitudinalGroundSpeedInnovationGate; //!< [-]

    //! VMO prediction step size.
    //! When predicting, steps the filter by this increment.
    //! @note see fixedStep
    dwTime_t stepSize;

    //! Whether vmo operates in fixed step or variable step operation during
    //! measurement updates. In fixed step operation, states are generated in a fixed
    //! cadence given by @param stepSize and measurements are fused into the nearest
    //! (in-time) state. If variable step operation, states are generated in a variable
    //! cadence to align with measurement timestamps. Prediction-only steps (without
    //! measurements) use @param stepSize.
    bool fixedStep;

    //! Reference point used internally, expressed in rig coordinate frame.
    //! Should be near to IMU mounting location.
    dwVector3f referencePoint; //!< [m]

    //! Maximum linear velocity along each axis, to which the state will be constrained (treated as
    //! error if error handling is active). [m/s]
    //! Must be positive.
    dwVector3f velocityMax;

    //! Minimum linear velocity along each axis, to which the state will be constrained (treated as
    //! error if error handling is active). [m/s]
    //! Must be negative.
    dwVector3f velocityMin;
} dwEgomotionVMOParameters;

/**
 * @brief Wheel observer parameters.
 */
typedef struct dwEgomotionWheelObserverParameters
{
    //! Wheel encoder parameters
    dwVehicleWheelEncoderProperties wheelEncoder;

    //! Internal state history size. Recommended to leave 0 for automatic size given
    //! dwEgomotionMeasurementParameters settings.
    size_t stateHistorySize;

    //! Wheel Observer prediction step size.
    //! When predicting, steps the filter by this increment.
    //! @note see fixedStep
    dwTime_t stepSize;

    //! Whether wheel observer operates in fixed step or variable step operation during
    //! measurement updates. In fixed step operation, states are generated in a fixed
    //! cadence given by @param stepSize and measurements are fused into the nearest
    //! (in-time) state. If variable step operation, states are generated in a variable
    //! cadence to align with measurement timestamps. Prediction-only steps (without
    //! measurements) use @param stepSize.
    bool fixedStep;

    //! Process covariance parameters (continuous time, coefficients on diagonal).
    //! Elements correspond to [position, speed, acceleration] estimates.
    float32_t processCovariance[3]; //!< [rad^2, rad^2/s^2, rad^2/s^4]

    //! Position (tick) count variance when tick increment below positionFuzzyLow.
    //! Indicates region of low trust in wheel position (tick) counters.
    float32_t positionVarianceLow; //!< [rad^2]

    //! Position (tick) count variance when tick increment between positionFuzzyLow and positionFuzzyHigh.
    //! Indicates region of high trust in wheel position (tick) counters.
    float32_t positionVarianceHigh; //!< [rad^2]

    //! Speed variance when speed below speedFuzzyLow.
    //! Indicates region of low trust in wheel speed.
    float32_t speedVarianceLow; //!< [rad^2/s^2]

    //! Speed variance when speed above speedFuzzyHigh.
    //! Indicates region of high trust in wheel speed.
    float32_t speedVarianceHigh; //!< [rad^2/s^2]

    //! Lower end of high trust in wheel position (tick) counters, inclusive (variance adaptation).
    uint32_t positionFuzzyLow; //!< [ticks] increment (in ticks) between successive measurements.

    //! Upper end of high trust in wheel position (tick) counters, inclusive (variance adaptation).
    uint32_t positionFuzzyHigh; //!< [ticks] increment (in ticks) between successive measurements.

    //! Region below which wheel speeds have low trust (variance adaptation).
    float32_t speedFuzzyLow; //!< [rad/s]

    //! Region above which wheel speeds have high trust (variance adaptation).
    float32_t speedFuzzyHigh; //!< [rad/s]

    //! Maximum wheel speed, to which the state will be constrained (treated as error if error handling is active). [rad/s]
    //! Must be positive.
    float32_t speedMax;

    //! Maximum wheel acceleration, to which the state will be constrained (treated as error if error handling is active). [rad/s^2]
    //! Must be positive.
    float32_t accelerationMax;

    //! Minimum wheel acceleration, to which the state will be constrained (treated as error if error handling is active). [rad/s^2]
    //! Must be negative.
    float32_t accelerationMin;

} dwEgomotionWheelObserverParameters;

/**
 * @brief Vehicle moving direction detection parameters.
 *
 * Used to inform egomotion on the moving direction of the vehicle.
 */
typedef struct dwEgomotionVehicleMovingDirectionDetectorParameters
{
    //! Mode of Vehicle Moving Direction Detector.
    dwEgomotionDirectionMode mode;

    //! Threshold of no wheel ticks duration to determine whether the vehicle rolling direction is void or stop.
    dwTime_t durationNoWheelTickThreshold; //!< [us]

    //! Speed below which driving direction from wheel tick increments and speeds can be ambiguous.
    //! Use this parameter if mode is DW_EGOMOTION_DIRECTION_FROM_SPEED_SIGN_AND_GEAR.
    //! If condition is met, wheel speed sign is inferred from gear.
    float32_t ticksAmbiguousSpeedThreshold; //!< [rad/s]
} dwEgomotionVehicleMovingDirectionDetectorParameters;

/**
 * @brief IMU intrinsic parameters.
 */
typedef struct dwEgomotionIMUIntrinsics
{
    //! Initial gyroscope bias (offset) values. Leave 0 if unknown.
    float32_t gyroscopeBias[3]; //!< [rad/s]

    //! Initial accelerometer bias (offset) values. Leave 0 if unknown.
    float32_t accelerometerBias[3]; //!< [m/s^2]
} dwEgomotionIMUIntrinsics;

/**
 * \brief Suspension properties.
 */
typedef struct dwEgomotionSuspensionProperties
{
    /// Suspension angular gradient around X- and Y-axis.
    dwVehicleSuspensionProperties properties;

    /// Roll (x) and pitch (y) center heights.
    dwVector2f centerHeight; //!< [m]
} dwEgomotionSuspensionProperties;

/**
 * @brief Egomotion parameters.
 * All parameters are required unless otherwise noted.
 */
typedef struct dwEgomotionParameters2
{
    //! Vehicle parameters.
    dwGenericVehicle vehicle;

    //! Measurement parameters.
    dwEgomotionMeasurementParameters measurements;

    //! IMU extrinsics, transformation from IMU coordinate frame to vehicle rig coordinate frame. Order must match the imuSensorIndices.
    dwTransformation3f imuExtrinsics[DW_EGOMOTION_IMU_COUNT_MAX];

    //! IMU intrinsics. Leave 0 initialized if unknown. Order must match the imuSensorIndices.
    dwEgomotionIMUIntrinsics imuIntrinsics[DW_EGOMOTION_IMU_COUNT_MAX];

    //! Number of IMUs used simultaneously, in range [1, DW_EGOMOTION_IMU_COUNT_MAX]
    size_t imuCount;

    //! Sensor Ids of used IMU sensors. Order must match for all IMU parameter specification arrays.
    uint32_t imuSensorIndices[DW_EGOMOTION_IMU_COUNT_MAX];

    //! Suspension parameters
    dwEgomotionSuspensionProperties suspension;

    //! Number of state estimates to keep in the history (if 0 specified default of 1000 is used).
    size_t stateHistorySize;

    //! Indicates the source for ground speed.
    dwEgomotionGroundSpeedMeasurementTypes groundSpeedType;

    //! Wheel observer parameters.
    //! Unused if @param groundSpeedType is not FROM_WHEEL_TICKS_AND_SPEEDS.
    dwEgomotionWheelObserverParameters wheelObserver;

    //! Vehicle moving direction detector parameters.
    //! Unused if @param groundSpeedType is FROM_LINEAR_SPEED.
    dwEgomotionVehicleMovingDirectionDetectorParameters vehicleMovingDirectionDetector;

    //! Vehicle motion observer parameters.
    dwEgomotionVMOParameters vehicleMotionObserver;

    //! Error handling parameters.
    dwEgomotionErrorHandlingParameters errorHandling;

    //! IMU accelerometer bias estimator parameters.
    dwEgomotionIMUAccBiasEstimatorParameters accBiasEstimator;

    //! IMU gyroscope bias estimator parameters.
    dwEgomotionIMUGyroBiasEstimatorParameters gyroBiasEstimator;

    //! Automatically update state estimation.
    //! In general to update motion estimation, a call to @ref dwEgomotion_update is required.
    //! When automaticUpdate is set, the motion estimation update is triggered by the addition of new IMU data.
    //! @note when the automatic update is active, @ref dwEgomotion_update will not update the filter state
    //! and throw a `DW_NOT_SUPPORTED` exception instead.
    bool automaticUpdate;

    //! Enable usage of suspension sensor signals
    bool enableSuspensionSensorsUsage;

    //! Parameters for linear fit of rear axle side slip angle to front steering angle in low speed maneuvers while driving forward. 0 indicates no model used.
    float32_t lowSpeedRearSideSlipGradientForward;

    //! Parameters for linear fit of rear axle side slip angle to front steering angle in low speed maneuvers while driving backward. 0 indicates no model used.
    float32_t lowSpeedRearSideSlipGradientBackward;
} dwEgomotionParameters2;

/**
 * Initializes the egomotion parameters to sane defaults given vehicle
 * configuration.
 *
 * This API is used to initialize parameters for egomotion 2.0.
 *
 * @param[in,out] params A pointer to the egomotion parameters to be filled out.
 * @param[in] rigConfiguration A pointer to the vehicle rig configuration.
 * @param[in] imu1Name Name of first IMU sensor as it appears in rig configuration.
 * @param[in] imu2Name Name of second IMU sensor as it appears in rig configuration. Optional, can be left null (1) (2).
 * @param[in] imu3Name Name of third IMU sensor as it appears in rig configuration. Optional, can be left null (1).
 *
 * @note (1) will default to egomotion parameters section in rig configuration.
 * @note (2) cannot be skipped if imu3Name is specified.
 *
 * @return DW_INVALID_ARGUMENT - if provided arguments are invalid. <br>
  *         DW_SUCCESS - if the egomotion parameters have been set successfully. <br>
  * @par API Group
  * - Init: Yes
  * - Runtime: No
  * - De-Init: No
*/
DW_API_PUBLIC
dwStatus dwEgomotion2_initParamsFromRig(dwEgomotionParameters2* params,
                                        dwConstRigHandle_t rigConfiguration,
                                        char const* imu1Name,
                                        char const* imu2Name,
                                        char const* imu3Name);

/**
 * Initializes the egomotion parameters to sane defaults applicable to most vehicles.
 *
 * This API does not set the required vehicle-specific parameters which must be provided separately.
 *
 * See also @ref dwEgomotion2_initParamsFromRig
 *
 * @param[in,out] params A pointer to the egomotion parameters to be filled out.
 *
 * @return DW_INVALID_ARGUMENT - if provided arguments are invalid. <br>
 *         DW_SUCCESS - if the egomotion parameters have been set successfully. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwEgomotion2_initDefaultParams(dwEgomotionParameters2* params);

/**
* Initializes the egomotion parameters to sane defaults given vehicle
* configuration.
*
* This API is used to initialize parameters for egomotion 2.0.
*
* @param[in,out] params A pointer to the egomotion parameters to be filled out.
* @param[in] rigConfiguration A pointer to the vehicle rig configuration.
* @param[in] imu1Index Index of first IMU sensor as it appears in rig configuration.
* @param[in] imu2Index Index of second IMU sensor as it appears in rig configuration. Optional, can be set to UINT32_MAX (1) (2).
* @param[in] imu3Index Index of third IMU sensor as it appears in rig configuration. Optional, can be set to UINT32_MAX (1).
*
* @note (1) will default to egomotion parameters section in rig configuration.
* @note (2) cannot be skipped if imu3Index is specified.
*
* @return DW_INVALID_ARGUMENT - if provided arguments are invalid. <br>
*         DW_SUCCESS - if the egomotion parameters have been set successfully. <br>
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: No
*/
DW_API_PUBLIC
dwStatus dwEgomotion2_initParamsFromRigByIndex(dwEgomotionParameters2* params,
                                               dwConstRigHandle_t rigConfiguration,
                                               uint32_t const imu1Index,
                                               uint32_t const imu2Index,
                                               uint32_t const imu3Index);

/**
* Initializes the egomotion 2.0 module.
*
* Instantiates the subfunctions composing the overall module and allocates memory for their
* operation. Configuration of the module is provided by the @p params argument. Default parameters
* can be obtained from the @ref dwEgomotion2_initParamsFromRig function.
*
* @param[in,out] obj A pointer to the egomotion handle to be set for the created module.
* @param[in] params A pointer to the configuration parameters of the module.
* @param[in] ctx Specifies the handler to the context under which the Egomotion module is created.
*
* @return DW_INVALID_ARGUMENT - if provided egomotion handle or parameters are invalid. <br>
*         DW_INVALID_HANDLE - if the provided DriveWorks context handle is invalid. <br>
*         DW_SUCCESS - if the initialization is successful. <br>
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: No
*/
DW_API_PUBLIC
dwStatus dwEgomotion2_initialize(dwEgomotionHandle_t* const obj,
                                 dwEgomotionParameters2 const* params,
                                 dwContextHandle_t const ctx);

/**
 * Initializes an instance of the egomotion 2.0 state history. This empty state history instance can be used to
 * deserialize from a binary representation of the state. After deserialization state can be queried with state API for
 * it's content. The state can contain internally a history of up-to passed amount of elements.
 *
 * @param[out] state Handle to be set with pointer to created empty state.
 * @param[in] historySize State history size (maximum capacity). If left 0, a default size of 1000 is used.
 * @param[in] ctx Handle of the context.
 *
 * @return DW_INVALID_ARGUMENT - if given state handle is null <br>
 *         DW_INVALID_HANDLE - if context handle is invalid <br>
 *         DW_SUCCESS - if the call was successful. <br>
 *
 * @note Ownership of the state goes back to caller. The state has to be released with @ref dwEgomotionState_release.
 * @note If passed `historySize` is smaller than the egomotion internal history capacity, any retrieval of the state with
 *       @ref dwEgomotion_getState will fill out this state's only with as much data as can fit, dropping oldest entries.
 *       This in turn would mean that calls to @ref dwEgomotionState_computeRelativeTransformation might not succeed if
 *       requested for timestamp outside of the covered history.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwEgomotion2_initializeState(dwEgomotionStateHandle_t* state, size_t const historySize, dwContextHandle_t ctx);

/**
 * Get estimated accelerometer bias.
 *
 * @param[in,out] accBias Pointer to dwVector3f to be filled with accelerometer biases.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_ARGUMENT - if the provided egomotion handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if the given egomotion handle does not support the request <br>
 *         DW_NOT_READY     - if the online estimation is not ready yet but an initial bias guess is available <br>
 *         DW_NOT_AVAILABLE - if the online estimation is not ready yet and no initial bias guess is available <br>
 *         DW_SUCCESS       - if online accelerometer bias estimation has accepted a value <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwEgomotion2_getAccelerometerBias(dwVector3f* accBias, dwEgomotionConstHandle_t obj);

#ifdef __cplusplus
}
#endif
#endif // DW_EGOMOTION_2_0_EGOMOTION2_H_

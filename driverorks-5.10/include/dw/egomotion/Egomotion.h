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
// SPDX-FileCopyrightText: Copyright (c) 2015-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_EGOMOTION_EGOMOTION_H_
#define DW_EGOMOTION_EGOMOTION_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>

#include <dw/control/vehicleio/VehicleIO.h>

#include <dw/sensors/imu/IMU.h>

#include <dw/rig/Rig.h>
#include <dw/rig/CoordinateSystem.h>

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

/** Defines the motion models. */
typedef enum dwMotionModel {
    /**
     * Given odometry information, estimates motion of the vehicle using a bicycle model.
     *
     * The following parameters are required for this model:
     * - dwEgomotionParameters
     *   - vehicle
     *     - wheelbase, mass, inertia3D, frontCorneringStiffness, rearCorneringStiffness,
     *       centerOfMassToRearAxle, wheelRadius (1), steeringWheelToSteeringMap (2),
     *       maxSteeringWheelAngle (2), frontSteeringOffset (2)
     *   - speedMeasurementType
     *
     * This model is capable of providing the following estimates:
     * - dwEgomotionResult
     *   - rotation (only around vehicle rig yaw axis, relative to initial orientation)
     *   - linearVelocity[0] and linearVelocity[1]
     *   - angularVelocity[2]
     *   - linearAcceleration[1]
     * - dwTransformation3f (with @ref dwEgomotion_computeRelativeTransformation)
     *   - motion only estimated in plane, i.e. x, y translation and yaw rotation.
     *
     * Uncertainty estimates are not supported at this time.
     *
     * - (1) if dwEgomotionParameters.speedMeasurementType == DW_EGOMOTION_REAR_WHEEL_SPEED
     * - (2) if using dwEgomotion_addOdometry with DW_EGOMOTION_MEASUREMENT_STEERINGWHEELANGLE
     */
    DW_EGOMOTION_ODOMETRY = 1 << 0,

    /**
     * Fuses odometry model with IMU measurements to estimate motion of the vehicle.
     *
     * The following parameters are required for this model:
     * - dwEgomotionParameters
     *   - vehicle
     *     - wheelRadius (1), steeringWheelToSteeringMap (2), maxSteeringWheelAngle (2),
     *       wheelbase (3, 4), mass (3, 4), centerOfMassHeight (3), inertia3D (3), centerOfMassToFrontAxle (4)
     *       rearCorneringStiffness (4)
     *   - imu2rig
     *   - speedMeasurementType
     *
     * This model is capable of providing the following estimates:
     * - dwEgomotionResult
     *   - rotation (in local vertical, local horizontal coordinate frame aligned with initial heading of vehicle)
     *   - linearVelocity[0] and linearVelocity[1]
     *   - angularVelocity
     *   - linearAcceleration
     * - dwTransformation3f (with @ref dwEgomotion_computeRelativeTransformation)
     *
     * Uncertainty estimates are provided for all state estimates listed above.
     *
     * - (1) if dwEgomotionParameters.speedMeasurementType == DW_EGOMOTION_REAR_WHEEL_SPEED
     * - (2) if using dwEgomotion_addOdometry with DW_EGOMOTION_MEASUREMENT_STEERINGWHEELANGLE
     * - (3) if dwEgomotionParameters.suspension.model == DW_EGOMOTION_SUSPENSION_TORSIONAL_SPRING_MODEL
     * - (4) if dwEgomotionParameters.lateralSlipCoefficient == 0
     */
    DW_EGOMOTION_IMU_ODOMETRY = 1 << 1 | DW_EGOMOTION_ODOMETRY

} dwMotionModel;

/**
 * @brief Defines motion measurements.
 */
typedef enum {
    DW_EGOMOTION_MEASUREMENT_VELOCITY           = 0, //!< Vehicle velocity [m/s].
    DW_EGOMOTION_MEASUREMENT_STEERINGANGLE      = 1, //!< Steering angle [rad].
    DW_EGOMOTION_MEASUREMENT_STEERINGWHEELANGLE = 2  //!< Steering wheel angle [rad].
} dwMotionModelMeasurement;

/**
 * @brief Defines speed measurement types.
 */
typedef enum dwEgomotionSpeedMeasurementType {
    /**
    * Indicates that speed is linear speed [m/s] measured at front wheels (along steering direction).
    * Steering angle [rad] is used internally to compute longitudinal speed.
    *
    * - Positive speed corresponds to a forward motion of the vehicle.
    * - Positive steering angle corresponds to a left turn.
    *
    * @note: estimation quality is dependent on measurement accuracy, resolution and sampling rate. We recommend the following:
    * - speed signal at >=50Hz sampling rate and resolution of 0.02 m/s or higher.
    * - steering angle signal at >=50Hz sampling rate and resolution of 0.01 deg or higher.
    *
    * Provide front speed measurement and steering angle with the @ref dwEgomotion_addOdometry API, or with
    * the @ref dwEgomotion_addVehicleState API where the `dwVehicleIOState` struct contains `speed`, `frontSteeringAngle`,
    * `speedTimestamp` and `steeringTimestamp` or with @ref dwEgomotion_addVehicleIOState API where at least one of
    * `dwVehicleIOSafetyState`, `dwVehicleIONonSafetyState` or `dwVehicleIOActuationFeedback` contain valid values of
    * `speed`, `frontSteeringAngle`, `speedTimestamp` and `frontSteeringTimestamp`.
    */
    DW_EGOMOTION_FRONT_SPEED = 0,

    /**
    * Indicates that speeds are angular speeds [rad/s] measured at rear wheels.
    *
    * - Positive angular speeds correspond to a forward motion of the vehicle.
    *
    * @note This mode requires valid `dwEgomotionParameters.dwVehicle.wheelRadius` otherwise incorrect
    * estimation of the longitudinal speed will be made.
    *
    * @note It is expected that both rear wheel speed measurements are not far apart in time, otherwise
    * degradation in estimation quality is expected.
    *
    * @note: estimation quality is dependent on measurement accuracy, resolution and sampling rate. We recommend the following:
    * - speed signal at >=50Hz sampling rate and resolution of 0.05 rad/s or higher.
    *
    * @note: Egomotion supports steered rear axles.
    *
    * Provide rear wheel speed measurements with the @ref dwEgomotion_addVehicleState API where the `dwVehicleIOState`
    * struct contains `wheelSpeed` and `wheelSpeedTimestamp` or with @ref dwEgomotion_addVehicleIOState API where at least one of
    * `dwVehicleIOSafetyState`, `dwVehicleIONonSafetyState` or `dwVehicleIOActuationFeedback` contain valid values of
    * `wheelSpeed` and `wheelTickTimestamp`.
    */
    DW_EGOMOTION_REAR_WHEEL_SPEED = 1,

    /**
    * Indicates that speed is linear speed [m/s] measured at rear axle center (along steering direction).
    * Rear steering angle [rad] given by dwVehicleIOState.rearWheelAngle is used internally to compute longitudinal speed.
    *
    * - Positive speed corresponds to a forward motion of the vehicle.
    * - Positive steering angle corresponds to a lateral motion towards the left at the rear axle.
    *
    * @note: estimation quality is dependent on measurement accuracy, resolution and sampling rate. We recommend the following:
    * - speed signal at >=50Hz sampling rate and resolution of 0.02 m/s or higher.
    *
    * Provide rear speed measurement with the @ref dwEgomotion_addVehicleState API where the `dwVehicleIOState` struct
    * contains `speed`, `rearWheelAngle` and `speedTimestamp` or with @ref dwEgomotion_addVehicleIOState API where at least one of
    * `dwVehicleIOSafetyState`, `dwVehicleIONonSafetyState` or `dwVehicleIOActuationFeedback` contain valid values for
    * `speed`, `rearWheelAngle` and `speedTimestamp`.
    */
    DW_EGOMOTION_REAR_SPEED = 2,
} dwEgomotionSpeedMeasurementType;

/**
 * @brief Defines steering measurement types.
 */
typedef enum dwEgomotionSteeringMeasurementType {
    DW_EGOMOTION_FRONT_STEERING       = 0, //!< @see dwVehicleIOState frontSteeringAngle
    DW_EGOMOTION_STEERING_WHEEL_ANGLE = 1  //!< @see dwVehicleIOState steeringWheelAngle
} dwEgomotionSteeringMeasurementType;

/**
 * @brief Defines egomotion linear acceleration filter mode.
 */
typedef enum dwEgomotionLinearAccelerationFilterMode {
    DW_EGOMOTION_ACC_FILTER_NO_FILTERING, //!< no filtering of the output linear acceleration
    DW_EGOMOTION_ACC_FILTER_SIMPLE        //!< simple low-pass filtering of the acceleration
} dwEgomotionLinearAccelerationFilterMode;

/**
 * @brief Defines egomotion suspension model.
 */
typedef enum dwEgomotionSuspensionModel {
    DW_EGOMOTION_SUSPENSION_RIGID_MODEL            = 0, //!< No suspension model. Equivalent to perfectly rigid suspension.
    DW_EGOMOTION_SUSPENSION_TORSIONAL_SPRING_MODEL = 1  //!< Models suspension as single-axis damped torsional spring.
} dwEgomotionSuspensionModel;

/**
 * @brief Defines egomotion linear acceleration filter parameters.
 */
typedef struct dwEgomotionLinearAccelFilterParams
{
    //! Linear acceleration filter mode. Default (0): no filtering.
    dwEgomotionLinearAccelerationFilterMode mode;

    // Simple filter parameters (ignored for other modes)
    float32_t accelerationFilterTimeConst;       //!< Time constant of the IMU acceleration measurements
    float32_t processNoiseStdevSpeed;            //!< Square root of continuous time process noise covariance in speed [m/s * 1/sqrt(s)]
    float32_t processNoiseStdevAcceleration;     //!< Square root of continuous time process noise covariance in acceleration [m/s^2 * 1/sqrt(s)]
    float32_t measurementNoiseStdevSpeed;        //!< Standard deviation of measurement noise in speed [m/s]
    float32_t measurementNoiseStdevAcceleration; //!< Standard deviation of measurement noise in acceleration [m/s^2]
} dwEgomotionLinearAccelerationFilterParams;

/**
 * @brief Suspension model type and parameters.
 **/
typedef struct
{
    //! Suspension model to use.
    //! If left zero-intialized, a rigid suspension system is assumed (i.e. no suspension modeling).
    dwEgomotionSuspensionModel model;

    //! Torsional spring model parameters. Used if model == DW_EGOMOTION_SUSPENSION_TORSIONAL_SPRING_MODEL.
    //! If left zero-initizialized, a default set of parameters will be used.
    //!
    //! These model parameters are suitable for a simple damped torsional spring model with external driving
    //! torque resulting from vehicle linear accelerations, described by the following ODE:
    //! \f$ I \ddot{\theta} + C \dot{\theta} + k \theta = \tau \f$
    //!
    //! where:
    //! - \f$ \theta \f$ suspension angle [rad]
    //! - I vehicle inertia around y axis [kg m^2]
    //! - C angular damping constant [J s rad^-1]
    //! - k torsion spring constant [N m rad^-1]
    //! - \f$ \tau \f$ driving torque [N m], function of linear acceleration, vehicle mass and height of center of mass.
    //!
    //! @note if selected, this model requires accurate vehicle mass, inertia and height of center of mass in the
    //! @ref dwVehicle struct provided as part of the @ref dwEgomotionParameters.
    //!
    //! Frequency at which the suspension system tends to oscillate around the pitch axis of the vehicle in the
    //! absence of an external driving force or damping [Hz].
    //! Typical passenger car suspension systems have a natural frequency in the range of [0.5, 1.5] Hz.
    //! Default value is 1.25Hz if left zero-initialized.
    float32_t torsionalSpringPitchNaturalFrequency;

    //! Level of damping relative to critical damping around the pitch axis of the vehicle [dimensionless].
    //! Typical passenger car suspension systems have a damping ratio in the range of [0.2, 0.6].
    //! Default value is 0.6 if left zero-initialized.
    float32_t torsionalSpringPitchDampingRatio;

} dwEgomotionSuspensionParameters;

/**
 * @brief Sensor measurement noise characteristics.
 **/
typedef struct dwEgomotionSensorCharacteristics
{
    //! Expected zero mean measurement noise of the gyroscope, also known as Noise Density [deg/s/sqrt(Hz)]
    //! A default value of 0.015 [deg/s/sqrt(Hz)] will be assumed if no parameter, i.e. 0 or nan, passed
    float32_t gyroNoiseDensityDeg;

    //! Expected gyroscope drift rate in [deg/s].
    //! A default value of 0.025 [deg/s] will be assumed if no parameter, i.e. 0 or nan, passed
    float32_t gyroDriftRate;

    //! If known this value in [rad/s] shall indicate standard deviation of the expected bias range of the gyroscope sensor.
    //! Usually temperature controlled/calibrated gyroscopes vary around the mean by few tens of a radian. If 0 is given,
    //! it will be assumed the standard deviation around the bias mean is about +-0.05 [rad/s], ~ +- 3deg/s
    float32_t gyroBiasRange;

    //! Expected zero mean measurement noise of the linear accelerometer, also known as Noise Density [ug/sqrt(Hz)]
    //! A default value of 100 micro-g per sqrt(Hz) will be assumed if no parameter, i.e. 0 or nan, passed
    float32_t accNoiseDensityMicroG;

    //! If known this entry shall indicate expected sampling rate in [Hz] of the IMU sensor.
    //! A default value of 100Hz is used if no parameter passed
    float32_t imuSamplingRateHz;

    //! If known this entry shall indicate expected sampling rate in [Hz] of the odometry signals.
    //! This is used for detection of delays or missing vehicle signals (valid range: [33%, 300%] of below value).
    //! A default value of 50Hz is used if no parameter passed (valid range: [16.7Hz, 150Hz])
    float32_t odometrySamplingRateHz;

    //! CAN velocity latency in microseconds which is read from can properties in rig file.
    dwTime_t velocityLatency;

    //! CAN velocity correction factor which is read from can properties in rig file.
    //! When ` dwEgomotionParameters::speedMeasurementType == DW_EGOMOTION_FRONT_SPEED or DW_EGOMOTION_REAR_SPEED`
    //! then received speed measurements are multiplied by this factor to obtain (approximately) true
    //! vehicle speed, e.g. due to non-nominal wheel diameters.
    //! @note A default value of 1 is assumed if no parameter is passed
    float32_t velocityFactor;

} dwEgomotionSensorCharacteristics;

/**
 * @brief Holds initialization parameters for the Egomotion module.
 */
typedef struct dwEgomotionParameters
{
    //! Vehicle parameters to setup the model.
    //! @note The validity of the parameters will be verified at initialization time and an error will be
    //!       returned back if vehicle parameters are found to be not plausible.
    dwVehicle vehicle;

    //! Lateral slip coefficient [rad*s^2/m].
    //! Used in linear function mapping lateral acceleration [m/s^2] to slip angle [rad], such that
    //! slipAngle = lateralSlipCoefficient * lateralAcceleration.
    //! If 0, default slip coefficient of -2.83e-3 [rad*s^2/m] is used.
    //! @note only available for DW_EGOMOTION_IMU_ODOMETRY motion model. Ignored when DW_EGOMOTION_ODOMETRY is used.
    DW_DEPRECATED("Deriving lateral slip coefficient from vehicle parameters, unless this parameter is non-zero.")
    float32_t lateralSlipCoefficient;

    //! IMU extrinsics. Transformation from the IMU coordinate system to the vehicle rig coordinate system.
    //! @note the quality of the estimated motion is highly depended on the accuracy of the extrinsics.
    dwTransformation3f imu2rig;

    //! Specifies the motion model to be used for pose estimation.
    dwMotionModel motionModel;

    //! When enabled, initial rotation will be estimated from accelerometer measurements.
    //! When disabled, initial rotation is assumed to be identity, i.e. vehicle standing on flat, horizontal ground.
    //! @note only available for DW_EGOMOTION_IMU_ODOMETRY motion model. Ignored when DW_EGOMOTION_ODOMETRY is used.
    bool estimateInitialOrientation;

    //! Automatically update state estimation.
    //! In general to update motion estimation, a call to @ref dwEgomotion_update is required.
    //! When automaticUpdate is set, the motion estimation update is triggered by the addition of new
    //! sensor measurements. The exact update timestamp is dependent on the sensor type and motion model
    //! implementation.
    //! @note when the automatic update is active, @ref dwEgomotion_update will not update the filter state
    //! and throw a `DW_NOT_SUPPORTED` exception instead.
    //! @warning @ref dwEgomotion_update is deprecated and will be removed in the next major release. The behavior
    //! will then be to only update on new sensor measurements.
    bool automaticUpdate;

    //! Number of state estimates to keep in the history (if 0 specified default of 1000 is used).
    //! Also known as the capacity of the history.
    //! Any call to @ref dwEgomotion_update, or automatic update, adds an estimate into the history.
    uint32_t historySize;

    //! Initial gyroscope biases, if known at initialization time. Gyroscope biases are estimated internally
    //! at run-time, however it can be beneficial if the filter is initialized with already known biases.
    //! If unknown, leave biases zero-initialized.
    float32_t gyroscopeBias[3];

    //! Sensor parameters, containing information about sensor characteristics.
    //! If the struct is zero initialized, default assumptions about sensor parameters are made.
    //! @see dwEgomotionSensorCharacteristics
    dwEgomotionSensorCharacteristics sensorParameters;

    //! Defines which velocity readings from `dwVehicleIOState` shall be used for egomotion estimation
    dwEgomotionSpeedMeasurementType speedMeasurementType;

    //! Defines which steering readings from `dwVehicleIOState` shall be used for egomotion estimation
    dwEgomotionSteeringMeasurementType steeringMeasurementType;

    //! Linear acceleration filter parameters
    //! @note only available for DW_EGOMOTION_IMU_ODOMETRY motion model. Ignored when other motion models are used.
    dwEgomotionLinearAccelerationFilterParams linearAccelerationFilterParameters;

    //! Suspension model parameters.
    //! The model is used internally to compensate for vehicle body rotation due to acceleration and resulting
    //! rotational suspension effects.
    //! If the struct is zero initialized, suspension will not be modeled and accounted for.
    //! @note only available for DW_EGOMOTION_IMU_ODOMETRY motion model. Ignored when other motion models are used.
    dwEgomotionSuspensionParameters suspension;

} dwEgomotionParameters;

/**
 * @brief Defines flags that indicate validity of corresponding data in @ref dwEgomotionResult and @ref dwEgomotionUncertainty.
 */
typedef enum dwEgomotionDataField {
    DW_EGOMOTION_ROTATION = 1 << 0, //!< indicates validity of rotation, @see limitations of selected @ref dwMotionModel

    DW_EGOMOTION_LIN_VEL_X = 1 << 1, //!< indicates validity of linearVelocity[0]
    DW_EGOMOTION_LIN_VEL_Y = 1 << 2, //!< indicates validity of linearVelocity[1]
    DW_EGOMOTION_LIN_VEL_Z = 1 << 3, //!< indicates validity of linearVelocity[2]

    DW_EGOMOTION_ANG_VEL_X = 1 << 4, //!< indicates validity of angularVelocity[0]
    DW_EGOMOTION_ANG_VEL_Y = 1 << 5, //!< indicates validity of angularVelocity[1]
    DW_EGOMOTION_ANG_VEL_Z = 1 << 6, //!< indicates validity of angularVelocity[2]

    DW_EGOMOTION_LIN_ACC_X = 1 << 7, //!< indicates validity of linearAcceleration[0]
    DW_EGOMOTION_LIN_ACC_Y = 1 << 8, //!< indicates validity of linearAcceleration[1]
    DW_EGOMOTION_LIN_ACC_Z = 1 << 9, //!< indicates validity of linearAcceleration[2]

    DW_EGOMOTION_ANG_ACC_X = 1 << 10, //!< indicates validity of angularAcceleration[0]
    DW_EGOMOTION_ANG_ACC_Y = 1 << 11, //!< indicates validity of angularAcceleration[1]
    DW_EGOMOTION_ANG_ACC_Z = 1 << 12, //!< indicates validity of angularAcceleration[2]

} dwEgomotionDataField;

/**
 * @brief Holds egomotion state estimate.
 * @note Validity of data fields indicated by flags in @ref dwEgomotionDataField
 **/
typedef struct dwEgomotionResult
{
    dwQuaternionf rotation; //!< Rotation represented as quaternion (x,y,z,w).
                            //!< Valid when DW_EGOMOTION_ROTATION is set.

    dwTime_t timestamp; //!< Timestamp of egomotion state estimate [us].

    float32_t linearVelocity[3]; //!< Linear velocity in body frame measured in [m/s] at the origin.
                                 //! For motion models capable of estimating slip angle imposed lateral velocity,
                                 //! y component will be populated.
                                 //! @note this represents instanteneous linear velocity

    float32_t angularVelocity[3]; //!< Rotation speed in body frame measured in [rad/s].

    float32_t linearAcceleration[3]; //!< Linear acceleration measured in body frame in [m/s^2].

    float32_t angularAcceleration[3]; //!< Angular acceleration measured in body frame in [rad/s^2].

    int32_t validFlags; //!< Bitwise combination of dwEgomotionDataField flags.
} dwEgomotionResult;

/**
 * @brief Holds egomotion uncertainty estimates.
 *
 * Data in these fields represent the uncertainties of the corresponding fields in @ref dwEgomotionResult.
 * The uncertainties are represented in covariance matrices when appropriate, or in standard deviation
 * around the mean of the estimate otherwise (assuming Gaussian distribution).
 *
 * @note Units are identical to those used in @ref dwEgomotionResult. Units in covariance matrices
 *       are squared.
 *
 * @note Rotation is represented as a quaternion; however, here a 3x3 covariance of the equivalent euler
 *       angles is given (order: roll, pitch, yaw) in [rad].
 **/
typedef struct
{
    dwMatrix3f rotation; //!< Rotation covariance represented as euler angles (order: roll, pitch, yaw) in [rad^2]

    float32_t linearVelocity[3]; //!< Linear velocity std dev in body frame measured in [m/s].

    float32_t angularVelocity[3]; //!< Rotation speed std dev in body frame measured in [rad/s].

    float32_t linearAcceleration[3]; //!< Linear acceleration std dev measured in body frame in [m/s^2].

    float32_t angularAcceleration[3];

    dwTime_t timestamp; //!< Timestamp of egomotion uncertainty estimate [us].

    int64_t validFlags; //!< Bitwise combination of dwEgomotionDataField flags.
} dwEgomotionUncertainty;

/**
 * @brief Holds egomotion uncertainty estimates for a relative motion estimate.
 **/
typedef struct
{
    dwMatrix3f rotation;    //!< a 3x3 covariance of the rotation (order: roll, pitch, yaw) [rad]
    dwMatrix3f translation; //!< a 3x3 covariance of the translation (x,y,z) [m]
    dwTime_t timeInterval;  //!< relative motion time interval [us]
    bool valid;             //!< indicates whether uncertainty estimates are valid or not
} dwEgomotionRelativeUncertainty;

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
 */
DW_API_PUBLIC
dwStatus dwEgomotion_release(dwEgomotionHandle_t obj);

/**
 * Notifies the egomotion module of a new odometry measurement.
 *
 * The following odometry measurements are required:
 *   - DW_EGOMOTION_MEASUREMENT_VELOCITY and
 *   - DW_EGOMOTION_MEASUREMENT_STEERINGANGLE or DW_EGOMOTION_MEASUREMENT_STEERINGWHEELANGLE
 *
 * @note measurement timestamp must be strictly increasing; i.e. take place after the previous measurement timestamp.
 *
 * @param[in] measuredType Type of measurement. For example: velocity, steering angle.
 * @param[in] measuredValue Value that was measured. For example: 3.6 m/s, 0.1 rad.
 * @param[in] timestamp Timestamp for the measurement.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_ARGUMENT - if @p timestamp is not greater than last measurement timestamp,  @p measuredType
 *                               is not valid, or @p measuredValue is not finite. <br>
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_NOT_SUPPORTED - if odometry measurements are not supported by the egomotion model or if egomotion speedMeasurementType
 *                            is not DW_EGOMOTION_FRONT_SPEED.<br>
 *         DW_SUCCESS - if data was added succesfully. <br>
 *
 * @note When velocity is passed to this module, it is assumed the velocity is measured on front wheels
 *       in the steering direction. See `dwEgomotionSpeedMeasurementType::DW_EGOMOTION_FRONT_SPEED`
 */
DW_API_PUBLIC
DW_DEPRECATED("Use dwEgomotion_addVehicleState instead.")
dwStatus dwEgomotion_addOdometry(dwMotionModelMeasurement measuredType,
                                 float32_t measuredValue, dwTime_t timestamp,
                                 dwEgomotionHandle_t obj);

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
 **/
DW_API_PUBLIC
DW_DEPRECATED("Deprecated, will be removed. Set dwEgomotionParameters.autoupdate to true instead.")
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
 * @return DW_NOT_AVAILABLE - if there is currently not enough data available to provide an estimate, or
 *                             requested timestamp is too far from available estimates. <br>
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_SUCCESS - if the desired state has been provided successfully. <br>
 *
 * @note Interpolation into the past can only happen within the available history range.
 * @note Interpolation interval is limited to 5 seconds, if interval between two successive poses
 *       is larger, the function will return DW_NOT_AVAILABLE.
 * @note Extrapolation into the future is limited to 2.5 seconds and the function will return
 *       DW_NOT_AVAILABLE if the extrapolation interval is larger.
 **/
DW_API_PUBLIC
dwStatus dwEgomotion_estimate(dwEgomotionResult* pose, dwEgomotionUncertainty* uncertainty,
                              dwTime_t timestamp_us, dwEgomotionConstHandle_t obj);

/**
 * Computes the relative transformation between two timestamps and the uncertainty of this transform.
 *
 * @param[out] poseAtoB Transform to be filled with transformation mapping a point at time `a` to a point at time `b`.
 * @param[out] uncertainty Structure to be filled with uncertainty of transformation (optional, ignored if nullptr provided).
 * @param[in] timestamp_a Timestamp corresponding to beginning of transformation.
 * @param[in] timestamp_b Timestamp corresponding to end of transformation.
 * @param[in] obj Egomotion handle.
 *
 * @note validity of uncertainty estimates is indicated by @ref dwEgomotionRelativeUncertainty.valid
 * @note uncertainty estimates are currently only supported by DW_IMU_ODOMETRY motion model.
 *
 * @return DW_INVALID_ARGUMENT - if any required argument is invalid. <br>
 *         DW_INVALID_HANDLE - if the given egomotion handle is invalid. <br>
 *         DW_NOT_AVAILABLE - if there is currently not enough data available to make a prediction or
 *                            requested timestamps are outside of available range (@see dwEgomotion_estimate). <br>
 *         DW_SUCCESS - if the computation was successful. <br>
 **/
DW_API_PUBLIC
dwStatus dwEgomotion_computeRelativeTransformation(dwTransformation3f* poseAtoB,
                                                   dwEgomotionRelativeUncertainty* uncertainty,
                                                   dwTime_t timestamp_a, dwTime_t timestamp_b,
                                                   dwEgomotionConstHandle_t obj);

/**
 * Compute the transformation between two coordinate systems at a specific timestamp and the
 * uncertainty of this transform.
 *
 * @param[out] transformationAToB Transformation mapping a point in coordinate system `A`
 *             to a point in coordinate system `B`.
 * @param[out] uncertainty Structure to be filled with uncertainty of transformation (optional, ignored if nullptr provided).
 * @param[in] timestamp Timestamp for which to get transformation.
 * @param[in] coordinateSystemA Coordinate system A.
 * @param[in] coordinateSystemB Coordinate system B.
 * @param[in] obj Egomotion handle.
 *
 * @note uncertainty estimates are not supported at this time are left unpopulated.
 *
 * @return DW_NOT_AVAILABLE - if there is currently not enough data available to make a prediction,
 *                            requested timestamp is too far into the past or egomotion modality
 *                            doesn't provide an estimate for the selected coordinate systems. <br>
 *         DW_NOT_SUPPORTED - if any coordinate system is not supported by any egomotion. <br>
 *         DW_INVALID_ARGUMENT - if any required argument is invalid. <br>
 *         DW_INVALID_HANDLE - if the provided egomotion handle invalid. <br>
 *         DW_SUCCESS <br>
 **/
DW_API_PUBLIC
dwStatus dwEgomotion_computeBodyTransformation(dwTransformation3f* const transformationAToB,
                                               dwEgomotionRelativeUncertainty* const uncertainty,
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
 *         DW_NOT_AVAILABLE - if there is currently no estimation available. <br>
 *         DW_SUCCESS - if the timestamp has been provided successfully <br>
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
 *         DW_NOT_AVAILABLE - if there is currently no estimation available. <br>
 *         DW_SUCCESS - if the estimation has been provided successfully. <br>
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
 *         DW_NOT_AVAILABLE - if there is currently no estimation available. <br>
 *         DW_SUCCESS - if the uncertainty has been provided successfully. <br>
 *
 * @note Not all values do have valid uncertainties. Check dwEgomotionUncertainty.validFlags.
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
**/
DW_API_PUBLIC
dwStatus dwEgomotion_steeringAngleToSteeringWheelAngle(float32_t* steeringWheelAngle, float32_t steeringAngle,
                                                       dwEgomotionHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_EGOMOTION_EGOMOTION_H_

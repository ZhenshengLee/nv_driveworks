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
 * @b Description: This file defines the calibration layer.
 */

/**
 * @defgroup calibration_group Calibration Interface
 *
 * @brief Defines the Calibration module for performing self-calibration of sensors and internal parameters.
 *
 * @{
 */

#ifndef DW_CALIBRATION_ENGINE_CALIBRATIONENGINE_H_
#define DW_CALIBRATION_ENGINE_CALIBRATIONENGINE_H_

#include <dw/core/context/Context.h>
#include <dw/rig/Rig.h>
#include <dw/egomotion/base/Egomotion.h>
#include <dw/egomotion/radar/DopplerMotionEstimator.h>
#include <dw/imageprocessing/features/FeatureList.h>

#include <dw/calibration/engine/common/CalibrationTypes.h>
#include <dw/calibration/engine/camera/CameraParams.h>
#include <dw/calibration/engine/imu/IMUParams.h>
#include <dw/calibration/engine/lidar/LidarParams.h>
#include <dw/calibration/engine/radar/RadarParams.h>
#include <dw/calibration/engine/stereo/StereoParams.h>
#include <dw/calibration/engine/vehicle/VehicleParams.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates and initializes a Calibration Engine
 *
 * @param[out] engine A pointer to the calibration handle will be returned here
 * @param[in] rig Specifies the rig module that holds the sensor information
 * @param[in] context Specifies the handle to the context under which the Calibration Engine module is created
 *
 * @note The rig handle must remain valid until the calibration module has been released
 *
 * @retval DW_INVALID_ARGUMENT when pointer to the engine handle is invalid
 * @retval DW_INVALID_HANDLE when provided context handle is invalid, i.e., null or of wrong type
 * @retval DW_SUCCESS when operation succeeded
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_initialize(dwCalibrationEngineHandle_t* engine,
                                        dwConstRigHandle_t rig,
                                        dwContextHandle_t context);

/**
 * @brief Initializes a camera calibration routine designated by the sensor provided to the method
 *
 * @param[out] routine A pointer to the new calibration routine handle associated with this engine
 * @param[in] sensorIndex The index of the sensor we are creating a calibration for. This is the same
 *            as the sensor index in the rig configuration module
 * @param[in] params Specifies the parameters that are used in calibrating the camera
 * @param[in] egomotion Specifies the egomotion module that holds the motion of the vehicle
 * @param[in] stream The CUDA stream to use for all CUDA operations of the calibration routine
 * @param[in] engine A pointer to the calibration engine handle that owns the sensor
 *
 * @par Runtime calibration dependencies (results will be inaccurate if not respected)
 *       - any height calibration: absolute scale is inferred from egomotion, which needs to use calibrated
 *                                 odometry properties (e.g., wheel radii via radar self-calibration)
 *       - side-camera roll calibration: side-camera roll is inferred from egomotion's axis of rotation, which
 *                                 needs to be based on an accurate IMU calibration
 *
 * @note The provided egomotion instance should be based on DW_EGOMOTION_IMU_ODOMETRY and the handle must
 *       remain valid until the calibration module has been released. Enabling the estimation of egomotion
 *       suspension using `dwEgomotionSuspensionModel` is beneficial for camera calibration accuracy and acceptance rates
 *
 * @retval DW_INVALID_ARGUMENT when the arguments in the parameters are invalid, if the egomotion handle is invalid or if the routine handle is invalid
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid, i.e., null or of wrong type
 * @retval DW_BUFFER_FULL when the number of sensors need to be calibrated exceeds DW_CALIBRATION_MAXROUTINES value
 * @retval DW_SUCCESS when operation succeeded
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_initializeCamera(dwCalibrationRoutineHandle_t* routine,
                                              uint32_t sensorIndex,
                                              const dwCalibrationCameraParams* params,
                                              dwEgomotionConstHandle_t egomotion,
                                              cudaStream_t stream,
                                              dwCalibrationEngineHandle_t engine);

/**
 * @brief Initializes an IMU calibration routine designated by the sensor provided to the method
 *
 * @param[out] routine A pointer to the new calibration routine handle associated with this engine
 * @param[in] imuIndex The index of the sensor we are creating a calibration for. This is the same
 *            as the sensor index in the rig module
 * @param[in] canIndex Index of the CAN bus from which speed information will be provided as dwVehicleIOState
 * @param[in] params Specifies the parameters that are used to calibrate the IMU
 * @param[in] engine A pointer to the calibration engine handle that owns the sensor
 *
 * @par Runtime calibration dependencies (results will be inaccurate if not respected)
 *       - none
 *
 * @note The calibration engine must be provided with dwVehicleIOState and dwIMUFrame data using
 *       dwCalibrationEngine_addVehicleIOState and dwCalibrationEngine_addIMUFrame, respectively.
 *
 * @retval DW_INVALID_ARGUMENT when the arguments in the parameters are invalid or if the routine handle is invalid
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid, i.e., null or of wrong type, or the given indices do point to
 *                           appropriate entries in the rig file
 * @retval DW_BUFFER_FULL when the number of sensors need to be calibrated exceeds the DW_CALIBRATION_MAXROUTINES value 
 * @retval DW_OUT_OF_BOUNDS when the given indices point outside the rig file
 * @retval DW_SUCCESS when operation succeeded
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_initializeIMU(dwCalibrationRoutineHandle_t* routine,
                                           const uint32_t imuIndex,
                                           const uint32_t canIndex,
                                           const dwCalibrationIMUParams* params,
                                           dwCalibrationEngineHandle_t engine);

/**
 * @brief Initializes a lidar calibration routine designated by the sensor provided to the method
 *
 * @param[out] routine A pointer to the new calibration routine handle associated with this engine
 * @param[in]  lidarIndex The index of the lidar to be calibrated. This is the same as the sensor index in the rig module
 * @param[in]  canIndex Index of the CAN bus from which speed information will be provided as dwVehicleIOState
 * @param[in]  params Specifies the parameters that are used to in calibrating the lidar
 * @param[in]  stream The CUDA stream to use for all CUDA operations of the calibration routine
 * @param[in]  engine A pointer to the calibration engine handle that owns the sensor
 *
 * @note The calibration engine must be provided with valid index of the CAN bus from which speed information will be provided as dwVehicleIOState 
 * 
 * @retval DW_INVALID_ARGUMENT when the arguments in the parameters are invalid or if the routine handle is invalid
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid,i.e. null or of wrong type
 * @retval DW_BUFFER_FULL when the number of sensors need to be calibrated exceeds DW_CALIBRATION_MAXROUTINES value.
 * @retval DW_SUCCESS when operation succeeded
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_initializeLidar(dwCalibrationRoutineHandle_t* routine,
                                             uint32_t lidarIndex,
                                             uint32_t canIndex,
                                             const dwCalibrationLidarParams* params,
                                             cudaStream_t stream,
                                             dwCalibrationEngineHandle_t engine);

/**
 * @brief Initializes a radar calibration routine designated by the sensor provided to the method.
 *        A radar calibration routine can also estimate odometry properties (speed factor / wheel radii) 
 *        by matching reported wheel speeds with radar-based speed estimates (enabled in parameters).
 *
 * @param[out] routine A pointer to the new calibration routine handle associated with this engine
 * @param[in] radarIndex The index of the radar we are creating a calibration for. This is the same as the sensor index in the rig module
 * @param[in] canIndex Index of the CAN bus from which speed information will be provided as dwVehicleIOState
 * @param[in] params Specifies the parameters that are used to in calibrating the radar
 * @param[in] engine A pointer to the calibration engine handle that owns the sensor
 *
 * @retval DW_INVALID_ARGUMENT when the arguments in the parameters are invalid or if the egomotion handle is invalid
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid, i.e., null or of wrong type
 * @retval DW_BUFFER_FULL when the number of sensors need to be calibrated exceeds DW_CALIBRATION_MAXROUTINES value
 * @retval DW_SUCCESS when operation succeeded
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_initializeRadar(dwCalibrationRoutineHandle_t* routine,
                                             uint32_t radarIndex,
                                             uint32_t canIndex,
                                             const dwCalibrationRadarParams* params,
                                             dwCalibrationEngineHandle_t engine);

/**
 * @brief This method initializes a stereo camera pose calibration routine relative to the sensor index of the
 *        left camera
 *
 * @param[out] routine A pointer to the new calibration routine handle associated with this engine
 * @param[in] vehicleSensorIndex The index of the vehicle sensor. This is the same as the sensor index in the rig configuration module
 * @param[in] leftSensorIndex The index of the left camera sensor. This is the same as the sensor index in the rig configuration module
 * @param[in] rightSensorIndex The index of the right camera sensor. This is the same as the sensor index in the rig configuration module
 * @param[in] params Specifies the parameters that are used in calibrating the stereo camera rig
 * @param[in] stream The CUDA stream to use for all CUDA operations of the calibration routine
 * @param[in] engine A pointer to the calibration engine handle that owns the sensor
 *
 * @par Runtime calibration dependencies (results will be inaccurate if not respected)
 *       - none
 *
 * @retval DW_INVALID_ARGUMENT when the arguments in the parameters are invalid
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid
 * @retval DW_BUFFER_FULL when the number of sensors need to be calibrated exceeds DW_CALIBRATION_MAXROUTINES value
 * @retval DW_SUCCESS when operation succeeded
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_initializeStereo(dwCalibrationRoutineHandle_t* routine,
                                              uint32_t vehicleSensorIndex,
                                              uint32_t leftSensorIndex,
                                              uint32_t rightSensorIndex,
                                              const dwCalibrationStereoParams* params,
                                              cudaStream_t stream,
                                              dwCalibrationEngineHandle_t engine);

/**
 * Initialize vehicle parameter calibration
 *
 * @param[out] routine A pointer to the new calibration routine handle associated with this engine.
 * @param[in] sensorIndex The index of the sensor that provides vehicle i/o data. This should be the same as the number in the rig configuration module
 * @param[in] params Specifies the parameters that are used to in calibrating the vehicle parameters
 * @param[in] egoMotion Specifies the ego motion module that holds the motion of the vehicle, needs to be of DW_EGOMOTION_IMU_ODOMETRY motion type
 * @param[in] vehicle A pointer to a vehicle parameter data.
 * @param[in] engine A pointer to the calibration engine handle that owns the sensor.
 *
 * @note Runtime calibration dependencies (results will be inaccurate if not respected):
 *        - IMU-based egomotion needs to be based on accurate IMU calibration and odometry properties
 *
 * @retval DW_INVALID_ARGUMENT when the arguments in the parameters are invalid or wrong egomotion type is passed
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid
 * @retval DW_BUFFER_FULL when the number of sensors need to be calibrated exceeds DW_CALIBRATION_MAXROUTINES value
 * @retval DW_SUCCESS when operation succeeded
 **/
DW_API_PUBLIC
dwStatus dwCalibrationEngine_initializeVehicle(dwCalibrationRoutineHandle_t* routine,
                                               uint32_t sensorIndex,
                                               const dwCalibrationVehicleParams* params,
                                               dwEgomotionConstHandle_t egoMotion,
                                               const dwVehicle* vehicle,
                                               dwCalibrationEngineHandle_t engine);

/**
 * @brief Resets the Calibration Engine module.
 *
 * This method resets all calibrations that are being calculated by the module. The method does not remove all
 * calibration routines associated with the engine, but rather resets each individual one.
 *
 * @param[in] engine Specifies the calibration engine handle to reset
 *
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid, i.e., null or of wrong type
 * @retval DW_SUCCESS when operation succeeded
 *
 * @note The call is equivalent in calling dwCalibrationEngine_resetCalibration() for all created routines
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_reset(dwCalibrationEngineHandle_t engine);

/**
 * @brief Releases the Calibration Engine module.
 *
 * This method stops all calibrations that are in process and invalidates all
 * calibration routines associated with the engine.
 *
 * @note This method renders the engine handle unusable
 *
 * @param[in] engine The calibration engine handle to be released
 *
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid, i.e., null or of wrong type
 * @retval DW_SUCCESS when operation succeeded
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_release(dwCalibrationEngineHandle_t engine);

/**
 * @brief Starts a calibration routine associated with a calibration engine.
 * @note Calibrations should only be started once runtime calibration dependencies are met.
 * @param[in] routine Specifies the handle of the calibration routine to start
 * @param[in] engine Specifies the handle of the calibration engine that is managing the calibration
 *
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid, i.e., null or of wrong type or if the calibration routine is not managed by the calibration engine
 * @retval DW_SUCCESS when operation succeeded
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_startCalibration(dwCalibrationRoutineHandle_t routine,
                                              dwCalibrationEngineHandle_t engine);

/**
 * @brief Stops a calibration routine associated with a calibration engine
 * 
 * @note A valid calibration engine handle has to be provided.
 * 
 * @param[in] routine Specifies the handle of the calibration routine to stop
 * @param[in] engine Specifies the handle of the calibration engine that is managing the calibration
 *
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid , i.e., null or of wrong type or if the calibration routine is not managed by the calibration engine
 * @retval DW_SUCCESS when operation succeeded
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_stopCalibration(dwCalibrationRoutineHandle_t routine,
                                             dwCalibrationEngineHandle_t engine);

/**
 * @brief Resets the calibration of a specific calibration routine associated with a calibration engine
 *
 * @note Resetting a calibration routine will revert to nominal baseline calibration
 *
 * @param[in] routine Specifies the handle of the calibration routine to reset
 * @param[in] engine Specifies the handle to the calibration engine that is managing the calibration
 *
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid, i.e., null or of wrong type or if the calibration routine is not managed by the calibration engine
 * @retval DW_SUCCESS when operation succeeded
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_resetCalibration(dwCalibrationRoutineHandle_t routine,
                                              dwCalibrationEngineHandle_t engine);

/**
 * @brief Returns the current status of a calibration routine. 
 *      
 * @note A valid calibration engine handle has to be provided.
 * 
 * @param[out] status A pointer to the returned status
 * @param[in] routine Specifies the handle of the calibration routine to query
 * @param[in] engine Specifies the calibration engine module we are checking against
 * 
 * @retval DW_INVALID_ARGUMENT when the calibration routine is not managed by the calibration engine
 * @retval DW_INVALID_HANDLE when any of the provided handles is invalid, i.e., null or of wrong type
 * @retval DW_SUCCESS when operation succeeded
**/
DW_API_PUBLIC
dwStatus dwCalibrationEngine_getCalibrationStatus(dwCalibrationStatus* status,
                                                  dwCalibrationRoutineHandle_t routine,
                                                  dwCalibrationEngineHandle_t engine);

/**
 * @brief Query a calibration routine for the calibration type and enabled calibration signal components.
 *
 * @param[out] signals Bitflag indicating supported calibration type and enabled signal components (binary OR of dwCalibrationSignal)
 * @param[in] routine Specifies the handle of the calibration routine to query
 * @param[in] engine Specifies the calibration engine module we are checking against
 *
 * @note A valid calibration engine handle has to be provided.
 * 
 * @retval DW_INVALID_HANDLE when any of the provided handles is invalid, i.e., null or of wrong type
 * @retval DW_INVALID_ARGUMENT when the calibration routine is not managed by the calibration engine or supported is null
 * @retval DW_SUCCESS when operation succeeded
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_getSupportedSignals(dwCalibrationSignal* signals,
                                                 dwCalibrationRoutineHandle_t routine,
                                                 dwCalibrationEngineHandle_t engine);

/**
 * @brief Returns the current sensor to rig transformation of a calibration routine estimating this
 *        transformation.
 *
 * @param[out] sensorToRig A pointer to the return transform. The transform represents the
 *             extrinsic transformation from the sensor to the rig coordinate system
 * @param[in] routine Specifies the handle of the calibration routine to query
 * @param[in] engine Specifies the calibration engine module we are checking against
 *
 * @note A valid calibration engine handle has to be provided.
 * 
 * @retval DW_INVALID_HANDLE when any of the provided handles is invalid, i.e., null or of wrong type
 * @retval DW_INVALID_ARGUMENT when the calibration routine is not managed by the calibration engine
 * @retval DW_NOT_SUPPORTED when the calibration routine is not estimating a sensorToRigTransformation
 * @retval DW_SUCCESS when operation succeeded
**/
DW_API_PUBLIC
dwStatus dwCalibrationEngine_getSensorToRigTransformation(dwTransformation3f* sensorToRig,
                                                          dwCalibrationRoutineHandle_t routine,
                                                          dwCalibrationEngineHandle_t engine);

/**
 * @brief Returns the current sensor to sensor transformation of a calibration routine estimating this
 *        transformation.
 *
 * @param[out] sensorToSensor A pointer to the return transform. The transform represents the
 *             extrinsic transformation from sensor A to sensor B coordinate system
 * @param[in] routine Specifies the handle of the calibration routine to query
 * @param[in] indexA index of sensor A
 * @param[in] indexB index of sensor B
 * @param[in] engine Specifies the calibration engine module we are checking against
 *
 * @retval DW_INVALID_HANDLE when any of the provided handles is invalid
 * @retval DW_INVALID_ARGUMENT when the calibration routine is not managed by the calibration engine 
 *                               or if the calibration routine is not estimating a sensorToSensorTransformation
 * @retval DW_NOT_AVAILABLE when the sensors indices provided are not related to the calibration routine
 * @retval DW_SUCCESS when operation succeeded
**/
DW_API_PUBLIC
dwStatus dwCalibrationEngine_getSensorToSensorTransformation(dwTransformation3f* sensorToSensor,
                                                             uint32_t indexA, uint32_t indexB,
                                                             dwCalibrationRoutineHandle_t routine,
                                                             dwCalibrationEngineHandle_t engine);

/**
 * @brief Returns odometry speed factor, mapping speed as reported by odometry to actual speed.
 *
 * Odometry speed factor can be estimated using multiple data sources.
 * Following variants are currently supported:
 *                 Radar based odometry calibration
 *                 This estimation is enabled with 'dwCalibrationRadarParams::enableSpeedFactorEstimation'
 *                 for 'dwCalibrationEngine_initializeRadar()`.
 * @param[out] odometrySpeedFactor Factor to be applied on measured odometry speed to map to actual speed
 * @param[in] routine Specifies the handle of the calibration routine to query
 * @param[in] engine A pointer to the calibration engine handle that owns the sensor
 *
 * @retval DW_INVALID_ARGUMENT when given `odometrySpeedFactor` is null
 * @retval DW_INVALID_HANDLE when any of the provided handles is invalid, i.e., null or of wrong type 
 * @retval DW_NOT_SUPPORTED when calibration routine is not performing an odometry speed factor estimation
 * @retval DW_SUCCESS when operation succeeded
 *
 * @note Speed factor returned is considered invalid unless `dwCalibrationEngine_getCalibrationStatus()` returns
 *       `DW_CALIBRATION_STATE_ACCEPTED` for the passed calibration routine.
 *
 * @note The speed factor currently is only calibrated when vehicle is driving straight, i.e., steering angle of the front wheel within +-3deg.
 *       For this purpose, egomotion's steering signal is used to identify straight driving sections.
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_getOdometrySpeedFactor(float32_t* odometrySpeedFactor,
                                                    dwCalibrationRoutineHandle_t routine,
                                                    dwCalibrationEngineHandle_t engine);

/**
 * Get currently estimated wheel radius of a vehicle.
 *
 * Wheel radius can be estimated using multiple data sources.
 * Following variants are currently supported:
 *  - Radar-based: uses Doppler-based speed as observed by the radar, comparing to individual wheel speed as reported by egomotion.
 *                 This estimation is enabled with `dwCalibrationRadarParams::enableWheelRadiiEstimation`
 *                 for 'dwCalibrationEngine_initializeRadar()`.
 *
 * @param[out] radius Return calibration wheel radius in [m] for a given wheel
 * @param[in] wheel Wheel to get radius for.
 * @param[in] routine Specifies the handle of the calibration routine to query
 * @param[in] engine A pointer to the calibration engine handle that owns the sensor
 *
 * @retval DW_INVALID_ARGUMENT when given `radius` is null <br>
 * @retval DW_INVALID_HANDLE when any of the provided handles is invalid <br>
 * @retval DW_NOT_SUPPORTED when calibration routine is not estimating wheel radius <br>
 * @retval DW_SUCCESS when operation succeeded
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_getVehicleWheelRadius(float32_t* radius, dwVehicleWheels wheel,
                                                   dwCalibrationRoutineHandle_t routine,
                                                   dwCalibrationEngineHandle_t engine);

/**
 * Get vehicle parameter calibration result
 *
 * @param[out] steering A pointer to the calibration result of vehicle steering parameters
 * @param[in] routine Specifies the handle of the vehicle calibration routine to query
 * @param[in] engine A pointer to the calibration engine handle that owns the sensor.
 *
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid 
 * @retval DW_INVALID_ARGUMENT when the calibration routine is not managed by the calibration engine
 * @retval DW_NOT_SUPPORTED when calibration routine is not estimating wheel radius
 * @retval DW_SUCCESS when operation succeeded
 *
 **/
DW_API_PUBLIC
dwStatus dwCalibrationEngine_getVehicleSteeringProperties(dwVehicleSteeringProperties* steering,
                                                          dwCalibrationRoutineHandle_t routine,
                                                          dwCalibrationEngineHandle_t engine);

/**
 * @brief Adds detected visual features to the calibration engine. The calibration engine will send
 * these features to all routines that use features from this sensor.
 *
 * @param[in] featureCapacity   Max number of features that can be stored in the list
 * @param[in] historyCapacity   Max age of a feature
 * @param[in] d_featureCount    Number of valid features (GPU memory)
 * @param[in] d_ages            Age of features (GPU memory)
 * @param[in] d_locationHistory Locations of features (GPU memory), see features.h for details on memory layout
 * @param[in] d_featureStatuses Statuses of features (GPU memory), see FeatureList.h for details on possible values
 * @param[in] currentTimeIdx    The index for the current time in the locationHistory arrays
 * @param[in] timestamp         The time stamp when the detections were created
 * @param[in] sensorIndex       The index of the sensor that created the detections
 * @param[in] engine            Specifies the calibration engine module we are checking against
 *
 * @retval DW_INVALID_ARGUMENT when the d_featureCount, d_ages, or d_locationHistory pointer is invalid
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid, i.e., null or of wrong type
 * @retval DW_INTERNAL_ERROR when an internal unrecoverable error was detected
 * @retval DW_SUCCESS when operation succeeded
 **/
DW_API_PUBLIC
dwStatus dwCalibrationEngine_addFeatureDetections(uint32_t featureCapacity,
                                                  uint32_t historyCapacity,
                                                  const uint32_t* d_featureCount,
                                                  const uint32_t* d_ages,
                                                  const dwVector2f* d_locationHistory,
                                                  const dwFeature2DStatus* d_featureStatuses,
                                                  uint32_t currentTimeIdx,
                                                  dwTime_t timestamp,
                                                  uint32_t sensorIndex,
                                                  dwCalibrationEngineHandle_t engine);

/**
 * @brief Adds an IMU frame from an IMU sensor to the calibration engine. The calibration engine will send
 * these measurements to all routines that use features from this sensor.
 *
 * The IMU frame shall contain both linear acceleration and angular velocity measurements for X, Y and Z
 * axes; the frame will be discarded otherwise.
 *
 * @param[in] imuFrame The IMU data that was taken from the sensor
 * @param[in] sensorIndex The index of the sensor that created the IMU data
 * @param[in] engine Specifies the calibration engine module we are checking against
 *
 * @note only calibrations that require the IMU from this sensor as part of their calibration routine
 *       will process the imu readings
 *
 * @retval DW_INVALID_ARGUMENT when the imuFrame pointer is invalid
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid, i.e., null or of wrong type
 * @retval DW_INTERNAL_ERROR when an internal unrecoverable error was detected
 * @retval DW_SUCCESS when operation succeeded
 **/
DW_API_PUBLIC
dwStatus dwCalibrationEngine_addIMUFrame(const dwIMUFrame* imuFrame,
                                         uint32_t sensorIndex,
                                         dwCalibrationEngineHandle_t engine);

/**
 * @brief Adds a lidar sweep to the calibration engine. The calibration engine will send
 * these measurements to all routines that use features from this sensor.
 *
 * @param[in] lidarPoints The lidar data that were acquired from the sensor
 * @param[in] pointCount The number of lidar points in the lidar sweep
 * @param[in] timestamp The time stamp when the lidar sweep was created
 * @param[in] sensorIndex The index of the sensor that created the lidarPoints
 * @param[in] engine Specifies the calibration engine module we are checking against.
 * 
 * @note A valid calibration engine handle has to be provided.
 * 
 * @retval DW_INVALID_ARGUMENT when the lidarPoints pointer is invalid
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid, i.e., null or of wrong type
 * @retval DW_INTERNAL_ERROR when an internal unrecoverable error was detected
 * @retval DW_SUCCESS when operation succeeded
 **/
DW_API_PUBLIC
dwStatus dwCalibrationEngine_addLidarPointCloud(const dwVector4f* lidarPoints,
                                                uint32_t pointCount,
                                                dwTime_t timestamp,
                                                uint32_t sensorIndex,
                                                dwCalibrationEngineHandle_t engine);

/**
 * @brief Adds lidar delta-poses and ego-motion delta poses to the calibration engine. The calibration engine
 * will send these measurements to all routines that use features from this sensor.
 *
 * @param[in] deltaPoseLidarTimeAToTimeB The relative pose in lidar frame from time A to time B
 * @param[in] deltaPoseRigTimeAToTimeB The relative pose in Rig frame from time A to time B (optional)
 *            If set to NULL, deltaPoseLidarTimeAToTimeB is used to approximate deltaPoseRigTimeAToTimeB for the algorithm.
 * @param[in] timestampA The time stamp of time A
 * @param[in] timestampB The time stamp of time B
 * @param[in] sensorIndex The index of the sensor that created the lidarPoints
 * @param[in] engine Specifies the calibration engine module we are checking against.
 * 
 * @note A valid calibration engine handle has to be provided.
 * 
 * @retval DW_INVALID_ARGUMENT when the lidarPoints pointer is invalid
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid, i.e., null or of wrong type
 * @retval DW_INTERNAL_ERROR when an internal unrecoverable error was detected
 * @retval DW_SUCCESS when operation succeeded
 **/
DW_API_PUBLIC
dwStatus dwCalibrationEngine_addLidarPose(const dwTransformation3f* deltaPoseLidarTimeAToTimeB,
                                          const dwTransformation3f* deltaPoseRigTimeAToTimeB,
                                          dwTime_t timestampA,
                                          dwTime_t timestampB,
                                          uint32_t sensorIndex,
                                          dwCalibrationEngineHandle_t engine);

/**
 * Adds Radar Doppler motion to the calibration engine
 *
 * @param[in] radarMotion The radar motion that estimated by DopplerMotionEstimator
 * @param[in] sensorIndex The index of the sensor that created the radarPoints
 * @param[in] engine Specifies the calibration engine module we are checking against
 *
 * @retval DW_INVALID_ARGUMENT when the radarPoints pointer is invalid
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid, i.e., null or of wrong type
 * @retval DW_INTERNAL_ERROR when an internal unrecoverable error was detected
 * @retval DW_SUCCESS when operation succeeded
 *
 **/
DW_API_PUBLIC
dwStatus dwCalibrationEngine_addRadarDopplerMotion(dwRadarDopplerMotion const* const radarMotion,
                                                   uint32_t sensorIndex,
                                                   dwCalibrationEngineHandle_t engine);

/**
 * @brief Adds detected visual feature matches to the calibration engine. The calibration engine will send
 * these feature matches to all routines that use feature matches from this sensor.
 *
 * @param[in] matches           History of feature matches between two cameras
 * @param[in] timestamp         The time stamp when the detections were created
 * @param[in] leftSensorIndex   The index of the left sensor that created the matches
 * @param[in] rightSensorIndex  The index of the right sensor that created the matches
 * @param[in] engine            Specifies the calibration engine module
 *
 * @retval DW_INVALID_ARGUMENT when matches are invalid
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid
 * @retval DW_INTERNAL_ERROR when an internal unrecoverable error was detected
 * @retval DW_SUCCESS when operation succeeded
 **/
DW_API_PUBLIC
dwStatus dwCalibrationEngine_addMatches(const dwFeatureHistoryArray* matches,
                                        dwTime_t timestamp,
                                        uint32_t leftSensorIndex,
                                        uint32_t rightSensorIndex,
                                        dwCalibrationEngineHandle_t engine);

/**
 * Adds vehicle IO state to calibration engine
 *
 * @deprecated Use dwCalibrationEngine_addVehicleIONonSafetyState
 *             and dwCalibrationEngine_addVehicleIOActuationFeedback instead.
 *
 * @param[in] vioState    VehicleIO state
 * @param[in] sensorIndex The index of sensor corresponding to steering system
 * @param[in] engine      Specifies the calibration engine module we are checking against.
 * 
 * @note A valid calibration engine handle has to be provided.
 * 
 * @retval DW_INVALID_ARGUMENT when the vioState pointer is invalid
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid
 * @retval DW_INTERNAL_ERROR when an internal unrecoverable error was detected
 * @retval DW_SUCCESS when operation succeeded
 **/
DW_API_PUBLIC
DW_DEPRECATED("dwCalibrationEngine_addVehicleIOState() is deprecated and will be removed in the next major release,"
              " use dwCalibrationEngine_addVehicleIONonSafetyState() and dwCalibrationEngine_addVehicleIOActuationFeedback instead.")
dwStatus dwCalibrationEngine_addVehicleIOState(const dwVehicleIOState* vioState,
                                               uint32_t sensorIndex,
                                               dwCalibrationEngineHandle_t engine);

/**
 * Adds dwVehicleIONonSafetyState to calibration engine.
 *
 * @note The necessary signals might be in dwVehicleIONonSafetyState or dwVehicleIOActuationFeedback,
 *       so both structs must be passed to the calibration engine.
 * 
 * @param[in] vioNonSafetyState    dwVehicleIONonSafetyState struct from VehicleIO.
 * @param[in] sensorIndex          The index of sensor corresponding to steering system.
 * @param[in] engine               Specifies the calibration engine object to be updated.
 *
 * @retval DW_INVALID_ARGUMENT when the vioState pointer is invalid
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid
 * @retval DW_INTERNAL_ERROR when an internal unrecoverable error was detected
 * @retval DW_SUCCESS when operation succeeded
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_addVehicleIONonSafetyState(dwVehicleIONonSafetyState const* const vioNonSafetyState,
                                                        uint32_t sensorIndex,
                                                        dwCalibrationEngineHandle_t engine);

/**
 * Adds dwVehicleIOActuationFeedback to calibration engine.
 *
 * @note The necessary signals might be in dwVehicleIONonSafetyState or dwVehicleIOActuationFeedback,
 *       so both structs must be passed to the calibration engine.
 * 
 * @param[in] vioActuationFeedback dwVehicleIOActuationFeedback struct from VehicleIO.
 * @param[in] sensorIndex          The index of sensor corresponding to steering system.
 * @param[in] engine               Specifies the calibration engine object to be updated.
 *
 * @retval DW_INVALID_ARGUMENT when the vioState pointer is invalid
 * @retval DW_INVALID_HANDLE when provided engine handle is invalid
 * @retval DW_INTERNAL_ERROR when an internal unrecoverable error was detected
 * @retval DW_SUCCESS when operation succeeded
 */
DW_API_PUBLIC
dwStatus dwCalibrationEngine_addVehicleIOActuationFeedback(dwVehicleIOActuationFeedback const* const vioActuationFeedback,
                                                           uint32_t sensorIndex,
                                                           dwCalibrationEngineHandle_t engine);

#ifdef __cplusplus
}
#endif
/** @} */

#endif // DW_CALIBRATION_ENGINE_CALIBRATIONENGINE_H_

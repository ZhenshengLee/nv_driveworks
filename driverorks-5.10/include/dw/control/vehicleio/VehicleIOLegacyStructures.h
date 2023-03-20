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
// SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: VehicleIO car controller</b>
 *
 * @b Description: API to access car controller box
 */

/**
 * @defgroup VehicleIO_actuators_group VehicleIO Actuators Interface
 *
 * @brief Defines the APIs to access the VehicleIO car controller box.
 *
 * @{
 */

#ifndef DW_VEHICLEIO_LEGACY_STRUCTURES_H_
#define DW_VEHICLEIO_LEGACY_STRUCTURES_H_

#include <dw/core/base/Types.h>
#include <dw/rig/Rig.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwVehicleIOObject* dwVehicleIOHandle_t;

typedef enum dwVehicleIODrivingMode {
    /// Comfortable driving is expected (most conservative). Commands that leave
    /// the comfort zone are treated as unsafe, which immediately leads to
    /// VehicleIO being disabled.
    DW_VEHICLEIO_DRIVING_LIMITED = 0x000,

    /// Same as above, but unsafe commands are clamped to safe limits and
    /// warnings are isssued. VehicleIO stays enabled.
    DW_VEHICLEIO_DRIVING_LIMITED_ND = 0x100,

    /// Safety checks suitable for collision avoidance logic (right now same as
    /// NO_SAFETY below).
    DW_VEHICLEIO_DRIVING_COLLISION_AVOIDANCE = 0x200,

    /// VehicleIO will bypass all safety checks.
    DW_VEHICLEIO_DRIVING_NO_SAFETY = 0x300,

    /// Driving mode is not valid.
    DW_VEHICLEIO_DRIVING_MODE_INVALID = 0x400
} dwVehicleIODrivingMode;

typedef enum dwVehicleIOType {
    DW_VEHICLEIO_DATASPEED    = 0,
    DW_VEHICLEIO_GENERIC      = 1,
    DW_VEHICLEIO_CUSTOM       = 2,
    DW_VEHICLEIO_DRIVER_COUNT = 3
} dwVehicleIOType;

typedef enum dwVehicleIOFaults {
    DW_VEHICLEIO_FAULT_NONE     = 0,
    DW_VEHICLEIO_FAULT_BRAKE    = 1 << 0,
    DW_VEHICLEIO_FAULT_STEERING = 1 << 1,
    DW_VEHICLEIO_FAULT_THROTTLE = 1 << 2,
    DW_VEHICLEIO_FAULT_GEAR     = 1 << 3,
    DW_VEHICLEIO_FAULT_SAFETY   = 1 << 4,
    DW_VEHICLEIO_FAULT_MAX      = UINT32_MAX
} dwVehicleIOFaults;

typedef enum dwVehicleIOOverrides {
    DW_VEHICLEIO_OVERRIDE_NONE     = 0,
    DW_VEHICLEIO_OVERRIDE_BRAKE    = 1 << 0,
    DW_VEHICLEIO_OVERRIDE_STEERING = 1 << 1,
    DW_VEHICLEIO_OVERRIDE_THROTTLE = 1 << 2,
    DW_VEHICLEIO_OVERRIDE_GEAR     = 1 << 3,
    DW_VEHICLEIO_OVERRIDE_MAX      = UINT32_MAX
} dwVehicleIOOverrides;

//# sergen(generate)
typedef enum dwVehicleIOGear {
    DW_VEHICLEIO_GEAR_UNKNOWN = 0,

    // Automatic vehicles
    DW_VEHICLEIO_GEAR_PARK    = 1,
    DW_VEHICLEIO_GEAR_REVERSE = 2,
    DW_VEHICLEIO_GEAR_NEUTRAL = 3,
    DW_VEHICLEIO_GEAR_DRIVE   = 4,
    DW_VEHICLEIO_GEAR_LOW     = 5,

    // Stick shift vehicles
    DW_VEHICLEIO_GEAR_MANUAL_REVERSE = 100,
    DW_VEHICLEIO_GEAR_1              = 101,
    DW_VEHICLEIO_GEAR_2              = 102,
    DW_VEHICLEIO_GEAR_3              = 103,
    DW_VEHICLEIO_GEAR_4              = 104,
    DW_VEHICLEIO_GEAR_5              = 105,
    DW_VEHICLEIO_GEAR_6              = 106,
    DW_VEHICLEIO_GEAR_7              = 107,
    DW_VEHICLEIO_GEAR_8              = 108,
    DW_VEHICLEIO_GEAR_9              = 109,
} dwVehicleIOGear;

typedef enum dwVehicleIOTurnSignal {
    DW_VEHICLEIO_TURNSIGNAL_UNKNOWN   = 0,
    DW_VEHICLEIO_TURNSIGNAL_OFF       = 1,
    DW_VEHICLEIO_TURNSIGNAL_LEFT      = 2,
    DW_VEHICLEIO_TURNSIGNAL_RIGHT     = 3,
    DW_VEHICLEIO_TURNSIGNAL_EMERGENCY = 4
} dwVehicleIOTurnSignal;

typedef enum dwVehicleIOTurnSignalType {
    /// No Turn Signal requested or error if a turn signal is requested.
    DW_VEHICLEIO_TURNSIGNALTYPE_UNKNOWN = 0,
    /// Driver is the source of the cause of the turn signal request.
    DW_VEHICLEIO_TURNSIGNALTYPE_DILC = 1,
    /// System is the source of the cause of the turn signal request.
    DW_VEHICLEIO_TURNSIGNALTYPE_SILC = 2
} dwVehicleIOTurnSignalType;

typedef enum dwVehicleIODoorLock {
    DW_VEHICLEIO_DOOR_UNKNOWN = 0,
    DW_VEHICLEIO_DOOR_UNLOCK  = 1,
    DW_VEHICLEIO_DOOR_LOCK    = 2
} dwVehicleIODoorLock;

typedef enum dwVehicleIOMoonroof {
    DW_VEHICLEIO_MOONROOF_UNKNOWN = 0,
    DW_VEHICLEIO_MOONROOF_CLOSE   = 1,
    DW_VEHICLEIO_MOONROOF_OPEN    = 2
} dwVehicleIOMoonroof;

typedef enum dwVehicleIOMirror {
    DW_VEHICLEIO_MIRROR_UNKNOWN      = 0,
    DW_VEHICLEIO_MIRROR_FOLD         = 1,
    DW_VEHICLEIO_MIRROR_UNFOLD       = 2,
    DW_VEHICLEIO_MIRROR_ADJUST_LEFT  = 3,
    DW_VEHICLEIO_MIRROR_ADJUST_RIGHT = 4
} dwVehicleIOMirror;

typedef enum dwVehicleIOMirrorFoldState {
    /// Mirror/Camera is not in end position and not being moved / ERROR
    DW_VEHICLEIO_MIRROR_FOLD_STATE_UNKNOWN = 0,
    /// Mirror/Camera is in folded position
    DW_VEHICLEIO_MIRROR_FOLD_STATE_FOLDED = 1,
    /// Mirror/Camera is in unfolded position
    DW_VEHICLEIO_MIRROR_FOLD_STATE_UNFOLDED = 2,
    /// Mirror/Camera is folding in
    DW_VEHICLEIO_MIRROR_FOLD_STATE_FOLD_IN = 3,
    /// Mirror/Camera is folding out
    DW_VEHICLEIO_MIRROR_FOLD_STATE_FOLD_OUT = 4,
} dwVehicleIOMirrorFoldState;

typedef enum dwVehicleIOHeadlights {
    DW_VEHICLEIO_HEADLIGHTS_UNKNOWN   = 0,
    DW_VEHICLEIO_HEADLIGHTS_OFF       = 1,
    DW_VEHICLEIO_HEADLIGHTS_LOW_BEAM  = 2,
    DW_VEHICLEIO_HEADLIGHTS_HIGH_BEAM = 3,
    DW_VEHICLEIO_HEADLIGHTS_DRL       = 4 //DAYTIME RUNNING LIGHTS
} dwVehicleIOHeadlights;

// AEB - Automatic Emergency Braking System Status to report externally
typedef enum dwVehicleIOAEBState {
    DW_VEHICLEIO_AEB_STATE_UNKNOWN = 0, // System is in an unknown state
    DW_VEHICLEIO_AEB_STATE_OFF     = 1, // System is off
    DW_VEHICLEIO_AEB_STATE_READY   = 2  // System is operational and ready to fire if necessary
} dwVehicleIOAEBState;

// FCW - Forward Collision Warning Status to report externally
typedef enum dwVehicleIOFCWState {
    DW_VEHICLEIO_FCW_STATE_UNKNOWN = 0, // System is in an unknown state
    DW_VEHICLEIO_FCW_STATE_OFF     = 1, // System is off
    DW_VEHICLEIO_FCW_STATE_READY   = 2  // System is operational and ready to fire if necessary
} dwVehicleIOFCWState;

// CDW - Close Distance Warning Status to report externally
typedef enum dwVehicleIOCDWRequestType {
    DW_VEHICLEIO_CDW_REQUEST_NONE    = 0, // no activation
    DW_VEHICLEIO_CDW_REQUEST_LEVEL_1 = 1, // level 1 warning (short distance)
    DW_VEHICLEIO_CDW_REQUEST_LEVEL_2 = 2  // level 2 warning (extremely short distance)
} dwVehicleIOCDWRequestType;

// BSM - Blind Spot Monitoring Request
typedef enum dwVehicleIOBSMRequest {
    DW_VEHICLEIO_BSM_REQUEST_UNKNOWN    = 0, // Unknown request from Blind Spot Monitor
    DW_VEHICLEIO_BSM_REQUEST_NONE       = 1, // No request from Blind Spot Monitor
    DW_VEHICLEIO_BSM_REQUEST_CONTINUOUS = 2, // Continuous response request from Blind Spot Monitor
    DW_VEHICLEIO_BSM_REQUEST_FLASHING   = 3  // Flashing response request from Blind Spot Monitor
} dwVehicleIOBSMRequest;

typedef enum dwVehicleIOLaneChangeFeedbackRequest {
    DW_VEHICLEIO_LCF_OFF            = 0,
    DW_VEHICLEIO_LCF_AVAILABLE      = 1,
    DW_VEHICLEIO_LCF_ACTIVE_LEFT    = 2,
    DW_VEHICLEIO_LCF_ACTIVE_RIGHT   = 3,
    DW_VEHICLEIO_LCF_PROPOSED_LEFT  = 4,
    DW_VEHICLEIO_LCF_PROPOSED_RIGHT = 5,
    DW_VEHICLEIO_LCF_CANCEL_LEFT    = 6,
    DW_VEHICLEIO_LCF_CANCEL_RIGHT   = 7,
} dwVehicleIOLaneChangeFeedbackRequest;

// Lateral ADAS mode
// Note: all modes may not be supported depending on vehicle
typedef enum dwVehicleIOLatMode {
    DW_VEHICLEIO_LAT_MODE_UNKNOWN = 0,
    DW_VEHICLEIO_LAT_MODE_IDLE    = 1,
    DW_VEHICLEIO_LAT_MODE_L2      = 2,
    DW_VEHICLEIO_LAT_MODE_L2_PLUS = 3,
    DW_VEHICLEIO_LAT_MODE_L3      = 4,
    DW_VEHICLEIO_LAT_MODE_PARK    = 5,
    DW_VEHICLEIO_LAT_MODE_LSS     = 6,
    DW_VEHICLEIO_LAT_MODE_AES     = 7,
    DW_VEHICLEIO_LAT_MODE_ESS     = 8
} dwVehicleIOLatMode;

// Longitudinal ADAS mode
// Note: all modes may not be supported depending on vehicle
typedef enum dwVehicleIOLonMode {
    DW_VEHICLEIO_LON_MODE_UNKNOWN        = 0,
    DW_VEHICLEIO_LON_MODE_IDLE           = 1,
    DW_VEHICLEIO_LON_MODE_PARK           = 2,
    DW_VEHICLEIO_LON_MODE_CA             = 3, // Collision avoidance
    DW_VEHICLEIO_LON_MODE_DRIVE_L2       = 4,
    DW_VEHICLEIO_LON_MODE_DRIVE_L3       = 5,
    DW_VEHICLEIO_LON_MODE_SPEED_LIMITING = 6,
} dwVehicleIOLonMode;

// Trailer connected status
typedef enum dwVehicleIOTrailerConnectedStatus {
    DW_VEHICLEIO_TRAILER_CONNECTED_UNKNOWN = 0,
    DW_VEHICLEIO_TRAILER_CONNECTED_NO      = 1,
    DW_VEHICLEIO_TRAILER_CONNECTED_YES     = 2,
    DW_VEHICLEIO_TRAILER_CONNECTED_ERROR   = 3,
} dwVehicleIOTrailerConnectedStatus;

// Generic struct representing signal quality
typedef enum dwVehicleIOSignalQuality {
    DW_VEHICLEIO_SIGNAL_QUALITY_NOT_OK = 0,
    DW_VEHICLEIO_SIGNAL_QUALITY_OK     = 1,
} dwVehicleIOSignalQuality;

typedef enum dwVehicleIOSpeedDirectionESC {
    /// Neither forward nor backward (stop)
    DW_VEHICLEIO_SPEED_DIRECTION_E_S_C_VOID = 0,
    /// Direction forward
    DW_VEHICLEIO_SPEED_DIRECTION_E_S_C_FORWARD = 1,
    /// Direction backward
    DW_VEHICLEIO_SPEED_DIRECTION_E_S_C_BACKWARD = 2
} dwVehicleIOSpeedDirectionESC;

typedef enum dwVehicleIOTirePressureMonitoringState {
    /// TPMS active
    DW_VEHICLEIO_TIRE_PRESSURE_MONITORING_STATE_ACTIVE = 0,
    /// Waiting for pressure values
    DW_VEHICLEIO_TIRE_PRESSURE_MONITORING_STATE_WAIT = 1,
    /// TPM system error
    DW_VEHICLEIO_TIRE_PRESSURE_MONITORING_STATE_ERROR = 2,
    /// no wheel sensors
    DW_VEHICLEIO_TIRE_PRESSURE_MONITORING_STATE_NO_SENSORS = 3,
    /// TPM activation acknowledged
    DW_VEHICLEIO_TIRE_PRESSURE_MONITORING_STATE_RESET_ACK = 4
} dwVehicleIOTirePressureMonitoringState;

typedef enum dwVehicleIOIgnitionStatus {
    /// Ignition lock (0)
    DW_VEHICLEIO_IGNITION_STATUS_IGN_LOCK = 0,
    /// Ignition off (15c)
    DW_VEHICLEIO_IGNITION_STATUS_IGN_OFF = 1,
    /// Ignition accessory (15r)
    DW_VEHICLEIO_IGNITION_STATUS_IGN_ACC = 2,
    /// Ignition on (15)
    DW_VEHICLEIO_IGNITION_STATUS_IGN_ON = 3,
    /// Ignition start (50)
    DW_VEHICLEIO_IGNITION_STATUS_IGN_START = 4
} dwVehicleIOIgnitionStatus;

typedef enum dwVehicleIOESCOperationalState {
    /// Normal operation
    DW_VEHICLEIO__E_S_C_OPERATIONAL_STATE_NORM = 0,
    /// Initialization
    DW_VEHICLEIO__E_S_C_OPERATIONAL_STATE_INIT = 1,
    /// Diagnostics
    DW_VEHICLEIO__E_S_C_OPERATIONAL_STATE_DIAG = 2,
    /// Exhaust emission test
    DW_VEHICLEIO__E_S_C_OPERATIONAL_STATE_EMT = 3,
    /// Test bench cruise control mode
    DW_VEHICLEIO__E_S_C_OPERATIONAL_STATE_TBCC = 4,
    /// Temporary or continously system error
    DW_VEHICLEIO__E_S_C_OPERATIONAL_STATE_FLT = 5,
    /// ESP or ASR control active
    DW_VEHICLEIO__E_S_C_OPERATIONAL_STATE_ESP_ASR_CTRL_ACTV = 6,
    /// Sport Version
    DW_VEHICLEIO__E_S_C_OPERATIONAL_STATE_SPORT_OFF = 7,
    /// ESP off mode or Sport plus
    DW_VEHICLEIO__E_S_C_OPERATIONAL_STATE_OFF_SPORT_PLUS = 8,
    /// ABS control active
    DW_VEHICLEIO__E_S_C_OPERATIONAL_STATE_ABS_CTRL_ACTV = 9
} dwVehicleIOESCOperationalState;

typedef enum dwVehicleIOAbsIntervention {
    /// Normal operation
    DW_VEHICLEIO_ABS_INTERVENTION_NORM = 0,
    /// Initialization
    DW_VEHICLEIO_ABS_INTERVENTION_INIT = 1,
    /// Diagnostics
    DW_VEHICLEIO_ABS_INTERVENTION_DIAG = 2,
    /// Exhaust emission test
    DW_VEHICLEIO_ABS_INTERVENTION_EMT = 3,
    /// Test bench cruise control mode
    DW_VEHICLEIO_ABS_INTERVENTION_TBCC = 4,
    /// Temporary or continously system error
    DW_VEHICLEIO_ABS_INTERVENTION_FLT = 5,
    /// ESP or ASR control active
    DW_VEHICLEIO_ABS_INTERVENTION_ESP_ASR_CTRL_ACTV = 6,
    /// AMG
    DW_VEHICLEIO_ABS_INTERVENTION_SPORT_OFF = 7,
    /// ESP off mode (not AMG) or AMG
    DW_VEHICLEIO_ABS_INTERVENTION_SPORT2_OFF = 8,
    /// ABS control active
    DW_VEHICLEIO_ABS_INTERVENTION_ABS_CTRL_ACTV = 9
} dwVehicleIOAbsIntervention;

typedef enum dwVehicleIODrivePositionTarget {
    /// D
    DW_VEHICLEIO_DRIVE_POSITION_TARGET_D = 0,
    /// N
    DW_VEHICLEIO_DRIVE_POSITION_TARGET_N = 1,
    /// R
    DW_VEHICLEIO_DRIVE_POSITION_TARGET_R = 2,
    /// P
    DW_VEHICLEIO_DRIVE_POSITION_TARGET_P = 3
} dwVehicleIODrivePositionTarget;

/**
 * @brief Wheel rotation direction.
 */
typedef enum dwVehicleIOWheelTicksDirection {
    /// Neither forward nor backward (stop)
    DW_VEHICLEIO_WHEEL_TICKS_DIRECTION_VOID = 0,
    /// Direction forward
    DW_VEHICLEIO_WHEEL_TICKS_DIRECTION_FORWARD = 1,
    /// Direction backward
    DW_VEHICLEIO_WHEEL_TICKS_DIRECTION_BACKWARD = 2,
    DW_VEHICLEIO_WHEEL_TICKS_DIRECTION_FORCE32  = 0x7FFFFFFF
} dwVehicleIOWheelTicksDirection;

/**
 * \brief Generic signal structure capturing data validity and timestamp.
 */
typedef struct dwStateValueFloat
{
    float32_t value;
    dwTime_t timestamp;
    bool valid;
} dwStateValueFloat;

typedef enum dwVehicleIOEmStandStill {
    /// Vehicle moving
    DW_VEHICLEIO_EM_STAND_STILL_FALSE = 0,
    /// Vehicle not moving
    DW_VEHICLEIO_EM_STAND_STILL_TRUE = 1,
    /// Schroedingers Vehicle Motion
    DW_VEHICLEIO_EM_STAND_STILL_UNKNOWN = 2
} dwVehicleIOEmStandStill;

typedef enum dwVehicleIOLatCtrlInterventionDirectionLKA {
    DW_VEHICLEIO_LAT_CTRL_INTERVENTION_DIRECTION_L_K_A_EVENT_DIRECTION_INVALID = 0,
    DW_VEHICLEIO_LAT_CTRL_INTERVENTION_DIRECTION_L_K_A_EVENT_DIRECTION_FRONT   = 1,
    DW_VEHICLEIO_LAT_CTRL_INTERVENTION_DIRECTION_L_K_A_EVENT_DIRECTION_LEFT    = 2,
    DW_VEHICLEIO_LAT_CTRL_INTERVENTION_DIRECTION_L_K_A_EVENT_DIRECTION_RIGHT   = 3,
    DW_VEHICLEIO_LAT_CTRL_INTERVENTION_DIRECTION_L_K_A_EVENT_DIRECTION_REAR    = 4
} dwVehicleIOLatCtrlInterventionDirectionLKA;

typedef enum dwVehicleIOLatCtrlInterventionDirectionELK {
    DW_VEHICLEIO_LAT_CTRL_INTERVENTION_DIRECTION_E_L_K_EVENT_DIRECTION_INVALID = 0,
    DW_VEHICLEIO_LAT_CTRL_INTERVENTION_DIRECTION_E_L_K_EVENT_DIRECTION_FRONT   = 1,
    DW_VEHICLEIO_LAT_CTRL_INTERVENTION_DIRECTION_E_L_K_EVENT_DIRECTION_LEFT    = 2,
    DW_VEHICLEIO_LAT_CTRL_INTERVENTION_DIRECTION_E_L_K_EVENT_DIRECTION_RIGHT   = 3,
    DW_VEHICLEIO_LAT_CTRL_INTERVENTION_DIRECTION_E_L_K_EVENT_DIRECTION_REAR    = 4
} dwVehicleIOLatCtrlInterventionDirectionELK;

typedef struct dwVehicleIOEgoFeedback
{
    dwStateValueFloat linearVelocity[3];        //!< Linear velocity in body frame measured in [m/s] at the origin.
    dwStateValueFloat angularVelocity[3];       //!< Rotation speed in body frame measured in [rad/s].
    dwStateValueFloat linearAcceleration[3];    //!< Linear acceleration measured in body frame in [m/s^2].
    dwStateValueFloat emLinearVelocityStdev[3]; //!< Vehicle linear velocity standard deviation aligned to body coordinate frame.
    dwStateValueFloat emAngularAccel[3];        //!< Vehicle angular velocity aligned to body coordinate frame.
    dwStateValueFloat emOrientation[3];         //!< Vehicle orientation (integrated, as Euler angles, since t0).
    dwStateValueFloat emOrientationStdev[3];    //!< Vehicle orientation standard deviation (for each Euler angle).
    float64_t emTranslation[3];                 //!< Vehicle translation (integrated, since t0) in [m].
    dwTime_t emTimestamp;                       //!< Timestamp of egomotion signals.
    dwVehicleIOEmStandStill emStandStill;       //!< Egomotion Stnadstill detected based on wheel ticks.
} dwVehicleIOEgoFeedback;

/**
 * \brief The command data.
 */
typedef struct dwVehicleIOCommand
{
    bool enable; //!< True if we are driving by wire. Has to always be set.

    // Steering command
    float32_t steeringWheelAngle;    //!< Desired steering wheel angle (rad)
    float32_t maxSteeringWheelSpeed; //!< Maximum steering wheel speed of the turning command rad/s

    float32_t frontSteeringAngle;    //!< Desired front wheel steering angle (rad)
    float32_t maxFrontSteeringSpeed; //!< Maximum front wheel speed of the turning command (rad/s)
    float32_t steeringWheelTorque;   //!< Additional steering wheel torque request (Nm). Does not
                                     //!< affect vehicle steering, rather to be used as feedback for
                                     //!< the driver

    float32_t rearAxleCurvatureValue; //!< Path curvature request based on travelled distance (1/m)

    float32_t rearSteeringAngle; //!< Desired rear wheel steering angle (rad)

    // Throttle command - command to the accelerator pedal deflection as a value from 0 to 1
    float32_t throttleValue; //!< range 0.0 to 1.0

    // Brake command - command to the brake pedal deflection as a value from 0 to 1
    float32_t brakeValue; //!< range 0.0 to 1.0

    // Deceleration command - target deceleration rate for the vehicle. NOTE: Depending on IO driver,
    // there are potentially multiple ways to command the vehicle - directly with actuator commands
    // or with targets (deceleration for example) to downstream systems (brake controller).
    float32_t decelerationValue; //!< decleration m/s^2 - represented as a positive number

    // Acceleration command
    float32_t lonAccelerationValue; //!< longitudinal acceleration (m/s^2)
    float32_t latAccelerationValue; //!< lateral acceleration (m/s^2)

    // Other commands
    dwVehicleIOGear gear;          //!< Desired gear: 0=UNKNOWN, 1=PARK, 2=REVERSE, 3=NEUTRAL, 4=DRIVE
    dwVehicleIOTurnSignal turnSig; //!< Turn signal value

    // Clear CAN bus errors
    bool clearFaults; //!< Setting > 0 clears any canbus faults/errors

    // Booleans validating commands
    bool throttleValid;            //!< True if setting throttle
    bool brakeValid;               //!< True if setting break
    bool steeringWheelValid;       //!< True if setting steering wheel steering
    bool frontSteeringValid;       //!< True if setting front wheel steering
    bool steeringWheelTorqueValid; //!< True if setting steering torque
    bool rearAxleCurvatureValid;   //!< True if setting rear axle curvature
    bool rearSteeringValid;        //!< True if setting rear wheel steering
    bool gearValid;                //!< True if setting gear
    bool turnSigValid;             //!< True if setting turn signal
    bool decelerationValid;        //!< True if setting deceleration
    bool lonAccelerationValid;     //!< True if setting longitudinal acceleration
    bool latAccelerationValid;     //!< True if setting lateral acceleration
    bool remainingDistanceValid;   //!< True if setting remaining distance
    bool maxSpeedValid;            //!< True if setting speed request
    bool latModeValid;             //!< True if setting lateral function
    bool lonModeValid;             //!< True if setting longitudinal function

    // ABSM signals
    float32_t additionalRearAxleDeltaCurvatureValue; //!< Delta Curvature request (1/m) for additional interface
                                                     //!< executed through differential braking
    bool additionalRearAxleDeltaCurvatureValid;      //!< True if setting additional delta curvature
    bool additionalDeltaCurvatureCtrlPrefillRequest; //!< Request to activate prefill for differential braking

    // AEB signals
    bool aebRequest;              //!< Request to activate AEB
    bool dbsRequest;              //!< Dynamic brake support request
    bool holdRequest;             //!< AEB Hold request (only valid if aebRequest or dbsRequest is true)
    bool prefillRequest;          //!< Request to activate prefill
    dwVehicleIOAEBState aebState; //!< AEB system status/state

    // FCW signals
    bool fcwRequest;              //!< Request to activate FCW
    dwVehicleIOFCWState fcwState; //!< FCW system status/state

    // CDW signals
    dwVehicleIOCDWRequestType cdwRequest; //!< Request to activate CDW

    // EESF signals
    bool lonEESFRequest;  //!< True if EESF is active in the longitudinal direction
    bool latEESFRequest;  //!< True if EESF is active in the lateral direction
    bool eesfHoldRequest; //!< EESF hold request (only valid if lonEESFRequest is true)

    uint64_t heartbeatCounter; //!< VIO command heartbeat

    // High accuracy maneuver
    float32_t remainingDistance; //!< Desired longitudinal offset from current position (m)
    float32_t maxSpeed;          //!< Maximum allowed speed to support remaining distance request (m/s)

    dwVehicleIOLatMode latMode;
    dwVehicleIOLonMode lonMode;

    // Motion manager feedback
    dwVehicleIOEgoFeedback egomotionResult;

    // Collaborative Steering
    bool latCtrlReadyForCollaborativeSteering; //!< Behavior Planner ready for control

    float32_t latCtrlCrossTrackError; //!< Used for driver intervention (Lateral path tracking control error)
    float32_t latCtrlCurvReqPred;     //!< The target curvature without feedback compensation. Can be used to represent the target curvature also at some brief period in the future..

    dwTime_t timestamp_us;                                                      //!< Timestamp when dwVehicleIOCommand was assembled
    dwVehicleIOLatCtrlInterventionDirectionLKA latCtrlInterventionDirectionLKA; //!< direction of the steering intervention, opposite to side of departure.
    dwVehicleIOLatCtrlInterventionDirectionELK latCtrlInterventionDirectionELK; //!< direction of the steering intervention, opposite to side of departure.
    uint8_t latCtrlLaneChangePushingLaterally;                                  //!< This bit is set if lateral movement has started.

    dwVehicleIOTurnSignalType turnSignalType;
    bool turnSignalTypeValid;
} dwVehicleIOCommand;

/**
 * @brief Driveworks Lane Departure Warning (LDW) activation states
 */
typedef enum dwVehicleIOLaneDepartureWarningState {
    DW_VEHICLEIO_LANE_DEPARTURE_WARNING_STATE_OFF = 0, //!< Warning not active
    DW_VEHICLEIO_LANE_DEPARTURE_WARNING_STATE_ON  = 1, //!< Warning active

    DW_VEHICLEIO_LANE_DEPARTURE_WARNING_STATE_COUNT //!< Count of LDW states
} dwVehicleIOLaneDepartureWarningState;

/**
 * @brief Driveworks LDW sides with respect to the ego lane
 */
typedef enum dwVehicleIOLaneDepartureWarningSide {
    DW_VEHICLEIO_LANE_DEPARTURE_WARNING_SIDE_NONE  = 0, //!< No warning on any side
    DW_VEHICLEIO_LANE_DEPARTURE_WARNING_SIDE_LEFT  = 1, //!< Warning on the left side
    DW_VEHICLEIO_LANE_DEPARTURE_WARNING_SIDE_RIGHT = 2, //!< Warning on the right side

    DW_VEHICLEIO_LANE_DEPARTURE_WARNING_SIDE_COUNT //!< Count of LDW warning sides
} dwVehicleIOLaneDepartureWarningSide;

typedef struct dwVehicleIOMiscCommand
{

    //basic body controls
    dwVehicleIODoorLock doorLock;                  //!< basic lock or unlock
    dwVehicleIOMoonroof moonroof;                  //!< basic open or close
    dwVehicleIOMirror mirrors;                     //!< FOLD, UNFOLD, ADJUST_LEFT, ADJUST_RIGHT
    dwVehicleIOHeadlights headlights;              //!< ON/OFF, LOW_BEAM, HIGH_BEAM, DRL
    dwVehicleIOTurnSignal turnSig;                 //!< Turn signal - misc also wants this
    dwVehicleIOLaneDepartureWarningState ldwState; //!< LDW Warning state- on/off
    dwVehicleIOLaneDepartureWarningSide ldwSide;   //!< LDW Warning side of ego lane- left/right
    float32_t ldwSeverity;                         //!< LDW Warning severity
    //specifically for mirrors
    float32_t mirrorAdjustX; //!< float value 0-5.0 seconds
    float32_t mirrorAdjustY; //!< float value 0-5.0 seconds
    //specifically for center console display brightness
    uint8_t displayBrightnessValue;

    bool enable;
    bool clearFaults;
    bool doorLockValid;          //!< True if setting door locks
    bool moonroofValid;          //!< True if setting moonroof movement
    bool mirrorFoldValid;        //!< True if setting mirror un/fold
    bool mirrorAdjustValid;      //!< True if setting mirror adjustment
    bool headlightsValid;        //!< True if setting headlights
    bool displayBrightnessValid; //!< True if setting display brightness
    bool turnSigValid;           //!< True if setting turn signal

    float32_t accSetSpeed; //!< Current ACC Set Speed in m/s

    bool longitudinalTOR; //!< True if an immediate takeover is needed

    dwVehicleIOBSMRequest bsmLeftRequest;  //!< Request corresponding to the left visual warning
    dwVehicleIOBSMRequest bsmRightRequest; //!< Request corresponding to the right visual warning
    dwVehicleIOBSMRequest bsmAudioRequest; //!< Request corresponding to the audio warning

    dwVehicleIOLaneChangeFeedbackRequest lcfRequest; //!< Lane change info

    dwTime_t timestamp_us; //!< Timestamp when dwVehicleIOMiscCommand was assembled
} dwVehicleIOMiscCommand;

/**
 * \brief The capability state data. Fields may not be populated if not supported by driver.
 */
typedef struct dwVehicleIOCapabilityState
{
    float32_t longVelocityForwardMin; //!< A minimum positive longitudinal vehicle speed, when driving forward, to avoid burning the clutch.
    dwTime_t longVelocityForwardMinTimestamp;
    float32_t longVelocityForwardMax; //!< A maximum positive longitudinal vehicle speed that the vehicle is designed for
    dwTime_t longVelocityForwardMaxTimestamp;

    float32_t rearAxleCurvatureMax; //!< The tightest radius (left) that can be guaranteed
    float32_t rearAxleCurvatureMin; //!< The tightest radius (right) that can be guaranteed
    dwTime_t rearAxleCurvatureCapabilityTimestamp;

    dwStateValueFloat frontWheelAngleMax; //!< The maximum front wheel angle command that can be guaranteed
    dwStateValueFloat frontWheelAngleMin; //!< The minimum front wheel angle command that can be guaranteed
    dwStateValueFloat rearWheelAngleMax;  //!< The maximum rear wheel angle command that can be guaranteed
    dwStateValueFloat lonAccelerationMax; //!< The maximum longutidinal acceleration command that can be guaranteed
} dwVehicleIOCapabilityState;

typedef struct dwVehicleIOTrailerState
{
    dwStateValueFloat mass;              //!< Total mass of trailer unit [kg]
    dwStateValueFloat articulationAngle; //!< Yaw articulation angle between truck and trailer measured at rear axle [rad]
                                         //!< Angle measured as described by ISO8855. For example, 0 when driving in line,
                                         //!< positive when turning left.
    dwVehicleIOTrailerConnectedStatus connected;
} dwVehicleIOTrailerState;

typedef struct dwVehicleIOAxleStatus
{
    dwStateValueFloat frontLoad;   //!< Load on front axle [kg]
    dwStateValueFloat pusherLoad;  //!< Load on pusher axle [kg]
    dwStateValueFloat driverLoad;  //!< Load on driver axle [kg]
    dwStateValueFloat trailerLoad; //!< Sum of load on all trailer axles [kg]
} dwVehicleIOAxleStatus;

typedef enum dwVehicleIOLatCtrlLoopStatus {
    DW_VEHICLEIO_LAT_CTRL_LOOP_STATUS_UNKNOWN  = 0, //!< Unknown
    DW_VEHICLEIO_LAT_CTRL_LOOP_STATUS_OPEN     = 1, //!< Open_Loop
    DW_VEHICLEIO_LAT_CTRL_LOOP_STATUS_BLENDING = 2, //!< Blending
    DW_VEHICLEIO_LAT_CTRL_LOOP_STATUS_CLOSED   = 3  //!< Closed_Loop
} dwVehicleIOLatCtrlLoopStatus;

typedef enum dwVehicleIOVehicleStopped {
    /// Vehicle not moving
    DW_VEHICLEIO_VEHICLE_STOPPED_UNKNOWN = 0,
    /// Vehicle moving
    DW_VEHICLEIO_VEHICLE_STOPPED_FALSE = 1,
    /// Schroedingers Vehicle Motion
    DW_VEHICLEIO_VEHICLE_STOPPED_TRUE = 2
} dwVehicleIOVehicleStopped;

typedef enum dwVehicleIOHoldStatus {
    DW_VEHICLEIO_HOLD_STATUS_UNKNOWN    = 0, //!< Unknown
    DW_VEHICLEIO_HOLD_STATUS_NOHOLD     = 1, //!< No Hold
    DW_VEHICLEIO_HOLD_STATUS_HOLD       = 2, //!< Hold
    DW_VEHICLEIO_HOLD_STATUS_TRANSITION = 3  //!< Transition
} dwVehicleIOHoldStatus;

typedef enum dwVehicleIOFrontSteeringAngleQuality {
    /// undefined
    DW_VEHICLEIO_FRONT_STEERING_ANGLE_QUALITY_NOT_DEFINED = 0,
    /// Functional and electrical checks passed
    DW_VEHICLEIO_FRONT_STEERING_ANGLE_QUALITY_NORMAL_OPERATION = 1,
    /// reduced signal integrity
    DW_VEHICLEIO_FRONT_STEERING_ANGLE_QUALITY_REDUCED_QUALITY = 2,
    /// not initialized
    DW_VEHICLEIO_FRONT_STEERING_ANGLE_QUALITY_INIT = 3,
    /// Signal defect detected
    DW_VEHICLEIO_FRONT_STEERING_ANGLE_QUALITY_SIG_DEF = 4
} dwVehicleIOFrontSteeringAngleQuality;

typedef enum dwVehicleIOLongCtrlEcoAssistStatus {
    /// unknown
    DW_VEHICLEIO_LONG_CTRL_ECO_ASSIST_STATUS_UNKNOWN = 0,
    /// Eco Assist inactive
    DW_VEHICLEIO_LONG_CTRL_ECO_ASSIST_STATUS_INACTIVE = 1,
    /// Eco Assist active
    DW_VEHICLEIO_LONG_CTRL_ECO_ASSIST_STATUS_ACTIVE = 2,
    /// Eco Assist longitudinal speed limit control engaged
    DW_VEHICLEIO_LONG_CTRL_ECO_ASSIST_STATUS_ENGAGED = 3
} dwVehicleIOLongCtrlEcoAssistStatus;

#define DW_VEHICLEIO_NUM_LAT_CTRL_CURV_CAP 10

/**
 * \brief The vehicle IO state data. Fields only set if supported by VehicleIO driver.
 */
typedef struct dwVehicleIOState
{
    dwVector2f velocity;     //!< Vehicle velocity (longitudinal, lateral) measured in m/s at the rear axle
    float32_t speed;         //!< Signed norm of velocity vector
    dwTime_t speedTimestamp; //!< Time at which speed was updated

    dwVector2f acceleration;        //!< Actual acceleration measured in m/s^2
    dwTime_t accelerationTimestamp; //!< Time at which acceleration was updated

    dwTime_t steeringTimestamp;      //!< Time at which steering was updated
    float32_t steeringWheelAngle;    //!< Steering wheel angle (-10.0 to 10.0 +- 0.01rad)
    float32_t steeringWheelAngleCmd; //!< Last acknowledged steering wheel value from a command (-10.0 to 10.0 +- 0.01rad)
    float32_t steeringWheelTorque;   //!< Steering wheel torque (0 to 10.0 +- 0.01 Nm)

    float32_t inverseSteeringR;      //!< Inverse turning radius of the vehicle on the road.
                                     //!  The radius depends on the vehicle wheel base, steering wheel angle,
                                     //!  drivetrain properties and current speed.
    float32_t frontSteeringAngleCmd; //!< Last acknowledged front steering value from a command (-1.0 to 1.0 +- 0.01rad)
    float32_t frontSteeringAngle;    //!< Same as inverseSteeringR described as an angle instead of radius [rad]

    float32_t throttleValue;    //!< Current thottle value as requested by a driver (0..1 +- 0.01 fraction of max pedal depressed, unitless)
    float32_t throttleCmd;      //!< Last acknowledged throttle value from a command (0..1 +- 0.01 fraction of max pedal depressed, unitless)
    float32_t throttleState;    //!< Throttle value in effect (0..1 +- 0.01 fraction of max pedal depressed, unitless)
    dwTime_t throttleTimestamp; //!< Time at which throttle was updated

    float32_t brakeValue;           //!< Current brake value as requested by a driver (0..1 +- 0.01 fraction of max pedal depressed, unitless)
    float32_t brakeCmd;             //!< Last acknowledged brake value from a command (0..1 +- 0.01 fraction of max pedal depressed, unitless)
    float32_t brakeState;           //!< Brake value in effect (0..1 +- 0.01 fraction of max pedal depressed, unitless)
    float32_t brakeTorqueRequested; //!< Requested value of brake torque (Nm)
    float32_t brakeTorqueActual;    //!< Actual applied brake torque value (Nm)
    dwTime_t brakeTimestamp;        //!< Time at which brake was updated

    float32_t wheelSpeed[DW_VEHICLE_NUM_WHEELS];         //!< vehicle individual wheel speeds (rad/s)
    dwTime_t wheelSpeedTimestamp[DW_VEHICLE_NUM_WHEELS]; //!< vehicle individual timestamps of wheel speeds readings
    float32_t suspension[DW_VEHICLE_NUM_WHEELS];         //!< Vehicle Suspension data, levels relative to a calibration instant [m]
    dwTime_t suspensionTimestamp[DW_VEHICLE_NUM_WHEELS]; //!< Vehicle timestamps of Suspension data

    int16_t wheelPosition[DW_VEHICLE_NUM_WHEELS];           //!< Vehicle Wheel Position counters.
                                                            //!  The counters are subject to roll-over. Actual
                                                            //!  wheel travel distance depends on wheel radius,
                                                            //!  which requires calibration.
    dwTime_t wheelPositionTimestamp[DW_VEHICLE_NUM_WHEELS]; //!< individual timestamps of wheel position readings

    dwVehicleIOWheelTicksDirection wheelTicksDirection[DW_VEHICLE_NUM_WHEELS]; //!< Wheel rotation direction

    float32_t tirePressure[DW_VEHICLE_NUM_WHEELS]; //!< Vehicle tire pressure data

    // Vehicle Miscellanoeus data
    bool buttonCruiseControlOnOff;
    bool buttonCruiseControlReset;
    bool buttonCruiseControlCancel;
    bool buttonCruiseControlResetCancel;
    bool buttonCruiseControlIncrement;
    bool buttonCruiseControlDecrement;
    bool buttonCruiseControlGapIncrement;
    bool buttonCruiseControlGapDecrement;
    bool buttonLaneAssistOnOff;
    bool doorDriver;
    bool doorPassenger;
    bool doorRearLeft;
    bool doorRearRight;
    bool doorHood;
    bool doorTrunk;
    bool passengerDetect;
    bool passengerAirbag;
    bool buckleDriver;
    bool bucklePassenger;
    bool highBeamHeadlights;
    bool wiper;
    bool buttonLeftKeypadOk;
    bool buttonLeftKeypadUp;
    bool buttonLeftKeypadDown;
    bool buttonTimeGapCycle;
    bool handsOnWheel;

    float32_t fuelLevel;              //!< (0 to 1 +- 0.01 fraction of tank volume, unitless)
    dwVehicleIOGear gear;             //!< Vehicle gear
    dwVehicleIOGear gearCmd;          //!< Last acknowledged gear from a command
    dwVehicleIOTurnSignal turnSignal; //!< Turn signal value

    uint32_t overrides; //!< Overrides in place (0 = none). Flags defined in dwVehicleIOOverrides
    uint32_t faults;    //!< Faults detected (0 = none). Flags defined in dwVehicleIOFaults

    dwVehicleIODrivingMode drivingMode;
    dwVehicleIOLatMode latMode;
    dwVehicleIOLonMode lonMode;

    bool enabled;

    bool brakeEnabled;    //!< Brake by-wire enablement reported by vehicle
    bool throttleEnabled; //!< Throttle by-wire enablement reported by vehicle
    bool steeringEnabled; //!< Steering by-wire enablement reported by vehicle

    float32_t rearAxleCurvature;         //!< Path curvature [1/m]
    dwTime_t rearAxleCurvatureTimestamp; //!< Timestamp for all motion signals

    dwStateValueFloat rearWheelAngle; //!  Rear wheel angle on road

    dwVehicleIOCapabilityState capability;

    dwVector2f radarVelocity; //!< Reported velocity from radar unit(s)
    dwTime_t radarVelocityTimestamp;

    dwTime_t engineSpeedTimestamp; //!< engine speed timestamp[us]
    float32_t engineSpeed;         //!< engine rpm speed [RPM]

    dwStateValueFloat outsideTemperature; //!< outside temperature [degrees C]

    dwStateValueFloat mass;                //!< Total mass of vehicle [kg]
    dwVehicleIOSignalQuality speedQuality; //!< Speed and velocity signal quality (legacy, protobuf backward compatible)

    dwVehicleIOTrailerState trailer;
    dwVehicleIOAxleStatus axles;

    float32_t brakeTorqueDriver; //!< Brake torque requested by driver via physical pedal (Nm)
    bool brakeActive;            //!< True if braking system is actively applying brakes
    bool brakePedalPressed;      //!< True if the brake pedal has been pressed (note that the brake system can be active without a pedal press)

    dwVehicleIOLatCtrlLoopStatus latCtrlLoopStatus;                     //!< lateral control loop state
    dwVehicleIOSpeedDirectionESC speedDirectionESC;                     //!< ESC Longitudinal Speed Direction
    dwVehicleIOTirePressureMonitoringState tirePressureMonitoringState; //!< Tire Pressure Monitoring System Status
    dwVehicleIOIgnitionStatus ignitionStatus;                           //!< Status of vehicle ignition.
    dwVehicleIOESCOperationalState ESCOperationalState;                 //!< ESC Operational State
    dwVehicleIOAbsIntervention absIntervention;                         //!< ABS/ESP Status

    float32_t latCtrlCurvCapMax[DW_VEHICLEIO_NUM_LAT_CTRL_CURV_CAP]; //!< Maximum curvature capability the vehicle is capable of at various velocities in L2/L3 mode. Communicated as a vector.
    float32_t latCtrlCurvCapMin[DW_VEHICLEIO_NUM_LAT_CTRL_CURV_CAP]; //!< Minumum (asymetric) curvature capability the vehicle is capable of at various velocities in L2/L3 mode. Communicated as a vector.
    dwVehicleIODrivePositionTarget drivePositionTarget;              //!< Drive Position that will be shifted to (PRND).
    dwVehicleIOVehicleStopped vehicleStopped;                        //!< Vehicle in Standstill as detected by ESC.
    dwVehicleIOFrontSteeringAngleQuality frontSteeringAngleQuality;  //!< Current front axle angle status.
    dwTime_t timestamp_us;                                           //!< Timestamp when dwVehicleIOState was assembled
    dwVehicleIOHoldStatus holdStatus;                                //!< Parking brake / hold state report

    float32_t speedMin; //!< Lower bound of vehicle velocity in the longitudinal direction as measured by ESP.
    float32_t speedMax; //!< Higher bound of vehicle velocity in the longitudinal direction as measured by ESP.

    dwVehicleIOLongCtrlEcoAssistStatus ecoAssistStatus;

    dwVehicleIOMirrorFoldState mirrorFoldState[2]; //!< 0 - corresponds to the left mirror and 1 - to the right one.
} dwVehicleIOState;

#define DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES 50
#define DW_VEHICLEIO_LOW_SPEED_THRESHOLD 4

/**
 * \brief VehicleIO Capabilities. Note that this some capabilities are
 *        imposed by the VehicleIO module itself. For dynamic (vehicle-reported)
 *        capabilities, @see dwVehicleIOCapabilityState.
 */
typedef struct dwVehicleIOCapabilities
{
    float32_t reverseSpeedLimit;                                         //!< Normally a negative value (m/s)
    int32_t brakeValueLUTSize;                                           //!< Size of the corresponding lookup table.
    float32_t brakeValueLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];         //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t throttleValueLUTSize;                                        //!< Size of the corresponding lookup table.
    float32_t throttleValueLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];      //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t steeringSpeedLUTSize;                                        //!< Size of the corresponding lookup table.
    float32_t steeringSpeedLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];      //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t steeringWheelAngleLUTSize;                                   //!< Size of the corresponding lookup table.
    float32_t steeringWheelAngleLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES]; //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s

    int32_t frontSteeringSpeedLUTSize;                                           //!< Size of the corresponding lookup table.
    float32_t frontSteeringSpeedLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];         //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t frontSteeringAngleLUTSize;                                           //!< Size of the corresponding lookup table.
    float32_t frontSteeringAngleLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];         //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t curvatureRateLUTSize;                                                //!< Size of the corresponding lookup table.
    float32_t curvatureRateLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];              //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t curvatureLUTSize;                                                    //!< Size of the corresponding lookup table.
    float32_t curvatureLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];                  //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t frontSteeringSpeedLowSpeedLUTSize;                                   //!< Size of the corresponding lookup table.
    float32_t frontSteeringSpeedLowSpeedLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES]; //!< Lookup Table indexed by speed/10.0 (m/s), i.e. LUT[i] is the capability value at speed = 10.0*i m/s
    int32_t frontSteeringAngleLowSpeedLUTSize;                                   //!< Size of the corresponding lookup table.
    float32_t frontSteeringAngleLowSpeedLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES]; //!< Lookup Table indexed by speed/10.0 (m/s), i.e. LUT[i] is the capability value at speed = 10.0*i m/s
    int32_t maxAccelerationLUTSize;                                              //!< Size of the corresponding lookup table.
    float32_t maxAccelerationLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];            //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t minAccelerationLUTSize;                                              //!< Size of the corresponding lookup table.
    float32_t minAccelerationLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];            //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
} dwVehicleIOCapabilities;

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_VEHICLEIO_LEGACY_STRUCTURES_H_

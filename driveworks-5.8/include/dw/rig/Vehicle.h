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
// SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Vehicle Parameters</b>
 *
 * @b Description: This file defines vehicle parameters.
 */

/**
 * @defgroup rig_configuration_group Rig Configuration Interface
 *
 * @brief Defines vehicle parameters.
 * @{
 */

#ifndef DW_RIG_VEHICLE_H_
#define DW_RIG_VEHICLE_H_

#include <dw/core/base/Types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DW_VEHICLE_STEER_MAP_POLY_DEGREE 5U
#define DW_VEHICLE_MAX_NUM_TRAILERS 1U

/**
 * \brief Physical properties of a vehicle body.
 *
 * Coordinate system depends on body type, \see dwCoordinateSystem
 */
typedef struct dwVehicleBodyProperties
{
    /// Length of the bounding box (longitudinal dimension, along X) [m].
    float32_t length;

    /// Width of the bounding box (lateral dimension, along Y) [m].
    float32_t width;

    /// Width of the body without any side-mirrors, if applicable, otherwise
    /// same as width.
    float32_t widthWithoutMirrors;

    /// Height of the bounding box (vertical dimension, along Z) [m].
    float32_t height;

    /// Position of bounding box origin in body coordinate system [m].
    /// \note bounding box origin is midpoint of rear bottom edge.
    /// \note The bounding box includes side mirrors and wheel geometry, if
    ///       applicable. Measurements taken at nominal load.
    dwVector3f boundingBoxPosition;

    /// Position of center of mass in body coordinate system [m].
    /// \note Center of mass including unsprung mass (suspension, wheels), if
    ///       applicable. Measurements taken at nominal load.
    dwVector3f centerOfMass;

    /// Principal moments of inertia with respect to center of mass [kg m^2].
    /// \note Inertia including unsprung mass (suspension, wheels), if
    ///       applicable. Measurements taken at nominal load.
    dwVector3f inertia;

    /// Mass [kg].
    /// \note Mass including unsprung mass (suspension, wheels), if
    ///       applicable. Measurements taken at nominal load.
    float32_t mass;

} dwVehicleBodyProperties;

/**
 * \brief Properties of an axle and its wheels.
 */
typedef struct dwVehicleAxleProperties
{
    /// Position of axle midpoint along X-axis in corresponding vehicle
    /// coordinate system (DW_COORDINATE_SYSTEM_VEHICLE_BASE or
    /// DW_COORDINATE_SYSTEM_VEHICLE_TRAILER) [m].
    float32_t position;

    /// Width of the axle, measured between center line of wheels [m].
    float32_t track;

    /// Radius of left wheel, when facing towards the forward direction
    /// of the vehicle [m].
    float32_t wheelRadiusLeft;

    /// Radius of right wheel, when facing towards the forward direction
    /// of the vehicle [m].
    float32_t wheelRadiusRight;

    /// Cornering stiffness for a single tire [N/rad].
    float32_t corneringStiffness;

} dwVehicleAxleProperties;

#define DW_VEHICLE_THROTTLE_BRAKE_LUT_SIZE 15U

/**
 * \brief Throttle and brake state (input) to longitudinal force (output) lookup tables.
 */
typedef struct dwVehicleTorqueLUT
{
    /// 1-d array of range of throttle pedal values (Throttle Look-up Table Input)
    float32_t throttlePedalInput[DW_VEHICLE_THROTTLE_BRAKE_LUT_SIZE];

    /// 1-d array of range of vehicle linear speed values (Throttle Look-up Table Input) [m/s]
    float32_t throttleSpeedInput[DW_VEHICLE_THROTTLE_BRAKE_LUT_SIZE];

    /// 2-d torque table, mapping a given throttle pedal position at a given speed to a torque value (Throttle Look-up Table Output)
    /// The 2d table is represented in row-major matrix with speed varying over the columns, and pedal input over rows,
    /// i.e. `throttleTorqueOutput(pedal, speed) = throttleTorqueOutput[pedal][speed]`
    float32_t throttleTorqueOutput[DW_VEHICLE_THROTTLE_BRAKE_LUT_SIZE][DW_VEHICLE_THROTTLE_BRAKE_LUT_SIZE];

    /// 1-d array of range of brake pedal values (Brake Look-up Table Input)
    float32_t brakePedalInput[DW_VEHICLE_THROTTLE_BRAKE_LUT_SIZE];

    /// 1-d torque Table, mapping a given brake pedal position to a torque value (Brake Look-up Table Output)
    float32_t brakeTorqueOutput[DW_VEHICLE_THROTTLE_BRAKE_LUT_SIZE];

} dwVehicleTorqueLUT;

/**
 * \brief Vehicle actuation properties.
 */
typedef struct dwVehicleActuationProperties
{
    /// Torque lookup tables.
    dwVehicleTorqueLUT torqueLUT;

    /// Effective mass due to rotational inertia (wheel, engine, and other
    /// parts of the CVT drivetrain) [kg].
    float32_t effectiveMass;

    /// Time constant for first order + time delay throttle system [s].
    float32_t throttleActuatorTimeConstant;

    /// Time delay for first order + time delay throttle system [s].
    float32_t throttleActuatorTimeDelay;

    /// Time constant for first order + time delay brake system [s].
    float32_t brakeActuatorTimeConstant;

    /// Time delay for first order + time delay brake system [s].
    float32_t brakeActuatorTimeDelay;

    /// Time constant for first order + time delay drive-by-wire / steer-by-wire [s].
    float32_t driveByWireTimeConstant;

    /// Time delay for first order + time delay drive-by-wire / steer-by-wire [s].
    float32_t driveByWireTimeDelay;

    /// Natural frequency for second order + time delay drive-by-wire / steer-by-wire [hz]
    float32_t driveByWireNaturalFrequency;

    /// Damping ratio for second order + time delay drive-by-wire / steer-by-wire [unitless]
    float32_t driveByWireDampingRatio;

    /// Indicates whether the drive-by-wire / steer-by-wire is second-order or not.
    bool isDriveByWireSecondOrder;

    /// Maximum steering wheel angle [rad].
    float32_t maxSteeringWheelAngle;

    /// Polynomial relating steering wheel angle [rad] to steering angle [rad].
    /// Coefficients ordered in array by increasing power,
    /// where first element is constant term (steering offset)
    float32_t steeringWheelToSteeringMap[DW_VEHICLE_STEER_MAP_POLY_DEGREE + 1U];

} dwVehicleActuationProperties;

/**
 * \brief Properties of an articulation linking two vehicle units.
 */
typedef struct dwVehicleArticulationProperties
{
    /// Position of leading vehicle hinge attach point in leading vehicle
    /// coordinate system (DW_COORDINATE_SYSTEM_VEHICLE_BASE or, if multiple
    /// trailer units DW_COORDINATE_SYSTEM_VEHICLE_TRAILER) [m].
    dwVector3f leadingVehicleHingePosition;

    /// Position of trailing vehicle hinge attach point in trailer coordinate
    /// system (DW_COORDINATE_SYSTEM_VEHICLE_TRAILER) [m].
    dwVector3f trailingVehicleHingePosition;

} dwVehicleArticulationProperties;

/**
 * \brief Supported trailer types.
 */
typedef enum dwVehicleTrailerType {
    DW_VEHICLE_TRAILER_TYPE_FULL = 0, /// Trailer that has both front and rear axles.
    DW_VEHICLE_TRAILER_TYPE_SEMI = 1  /// Trailer that has only rear axles.
} dwVehicleTrailerType;

/**
 * \brief Vehicle cabin description.
 */
typedef struct dwVehicleCabin
{
    /// Properties of the cabin body.
    dwVehicleBodyProperties body;

} dwVehicleCabin;

/***
 * Vehicle trailer description.
 */
typedef struct dwVehicleTrailer
{
    /// Properties of the trailer body.
    dwVehicleBodyProperties body;

    /// Properties of the front (steering) axle [m].
    dwVehicleAxleProperties axleFront;

    /// Properties of the rear axle group [m].
    /// Multiple rear axles are lumped together in an equivalent virtual rear
    /// axle, in which case its properties shall be consistent with corresponding
    /// vehicle data such as wheel speed.
    dwVehicleAxleProperties axleRear;

    /// Trailer type, either full or semi, indicates presence of front axle.
    dwVehicleTrailerType type;

    /// Articulation linking trailer to leading vehicle unit.
    dwVehicleArticulationProperties articulation;
} dwVehicleTrailer;

/**
 * \brief Vehicle description.
 */
typedef struct dwGenericVehicle
{
    /// Properties of the base body (passenger car body, truck tractor chassis)
    dwVehicleBodyProperties body;

    /// Properties of the front (steering) axle [m].
    dwVehicleAxleProperties axleFront;

    /// Properties of the rear axle group [m].
    /// Multiple rear axles are lumped together in an equivalent virtual rear
    /// axle, in which case its properties shall be consistent with corresponding
    /// vehicle data such as wheel speed.
    dwVehicleAxleProperties axleRear;

    /// Vehicle actuation properties.
    dwVehicleActuationProperties actuation;

    /// Properties of an optional floating cabin attached to the base body.
    /// Applies only to vehicles with a suspended cabin (e.g. trucks).
    dwVehicleCabin cabin;

    /// Indicates presence of a cabin.
    bool hasCabin;

    /// Properties of trailer units.
    /// Applicable only to vehicles with trailer units (e.g. trucks).
    dwVehicleTrailer trailers[DW_VEHICLE_MAX_NUM_TRAILERS];

    /// Number of trailer units.
    uint32_t numTrailers;

} dwGenericVehicle;

/**
 * \brief Define index for each of the wheels on a 4 wheeled vehicle.
 *
 * These indices are to be used to point into dwVehicleIOState or dwVehicle.
 */
typedef enum {
    DW_VEHICLE_WHEEL_FRONT_LEFT  = 0,
    DW_VEHICLE_WHEEL_FRONT_RIGHT = 1,
    DW_VEHICLE_WHEEL_REAR_LEFT   = 2,
    DW_VEHICLE_WHEEL_REAR_RIGHT  = 3,

    /// Number of wheels describing the vehicle
    DW_VEHICLE_NUM_WHEELS = 4
} dwVehicleWheels;

/**
 * \brief DEPRECATED: Properties of a passenger car vehicle.
 *
 * \deprecated Use dwGenericVehicle, this dwVehicle struct will be deprecated in an upcoming release.
 */
typedef struct dwVehicle
{
    float32_t height;           /*!< Height of the vehicle. [meters]>*/
    float32_t length;           /*!< Length of the vehicle. [meters]*/
    float32_t width;            /*!< Width of the vehicle, without side mirrors. [meters]*/
    float32_t widthWithMirrors; /*!< Width of the vehicle including side mirrors. [meters] */

    float32_t wheelbase;           /*!< Distance between the centers of the front and rear wheels. [meters] */
    float32_t axlebaseFront;       /*!< Width of the front axle. [meters] */
    float32_t axlebaseRear;        /*!< Width of the rear axle.  [meters] */
    float32_t bumperRear;          /*!< Distance rear axle to rear bumper. [meters] */
    float32_t bumperFront;         /*!< Distance front axle to front bumper. [meters] */
    float32_t steeringCoefficient; /*!< Steering coefficient for trivial linear mapping between steering wheel and steering angle, i.e.
                                         `steeringAngle = steeringWheelAngle / steeringCoefficient` */
    float32_t mass;                /*!< vehicle mass [kg]. */

    dwVector3f inertia3D; /*!< vehicle inertia around each axis, w.r.t. its center of mass. [kg m^2] */

    float32_t effectiveMass; /*!< effective mass due to vehicle rotational inertia (wheel rotation, engine, and other parts of the CVT drive-train). [kg] */

    float32_t frontCorneringStiffness; /*!< front wheel cornering stiffness. */
    float32_t rearCorneringStiffness;  /*!< rear wheel cornering stiffness.  */
    float32_t centerOfMassToRearAxle;  /*!< Distance between  vehicle's CoM (center-of-mass) and center of the rear axle. [meters]  */

    float32_t driveByWireTimeDelay;    /*!< Drive-by-wire (steer-by-wire) time delay. [s]  */
    float32_t driveByWireTimeConstant; /*!< Drive-by-wire (steer-by-wire) time constant. [s] */

    DW_DEPRECATED("Will be removed, unused")
    float32_t aerodynamicDragCoeff; /*!< Aerodynamic drag coefficient.  */

    DW_DEPRECATED("Will be removed, unused")
    float32_t frontalArea; /*!< Vehicle Frontal area (m^2).  */

    float32_t centerOfMassToFrontAxle; /*!< Distance from CoM to the front axle (m).  */
    float32_t centerOfMassHeight;      /*!< Height of the CoM (m).  */

    DW_DEPRECATED("Will be removed, unused")
    float32_t aeroHeight; /*!< Equivalent height of aerodynamic force applied (m).  */

    DW_DEPRECATED("Will be removed, unused")
    float32_t rollingResistanceCoeff; /*!< Rolling resistance coefficient.  */

    DW_DEPRECATED("Will be removed, unused")
    float32_t maxEnginePower; /*!< Maximum engine power in Watts.  */

    float32_t throttleActuatorTimeConstant; /*!< Time constant for first order lp throttle system.  */
    float32_t brakeActuatorTimeConstant;    /*!< Time constant for first order lp brake system.  */

    dwVehicleTorqueLUT torqueLUT; /*!< Lookup table mapping throttle and brake pedal position to torque. */

    float32_t wheelRadius[DW_VEHICLE_NUM_WHEELS]; /*!< Radius of each individual wheel [m] */

    float32_t steeringWheelToSteeringMap[DW_VEHICLE_STEER_MAP_POLY_DEGREE + 1]; //!< polynomial coefficents of steering wheel angle to steering angle
                                                                                //! as given in c0 + c1*x + c2*x^2 + ... + cn*x^n.
                                                                                //! If not 0, then these have precedence over steeringCoefficient
    float32_t maxSteeringWheelAngle;                                            //!< maximum steering wheel [radians]
    float32_t frontSteeringOffset;                                              //!< front wheel steering offset [radians]. It is combined with the polynomial
                                                                                //! function P(steeringWheelAngle) give by steeringWheelToSteeringMap to determine
                                                                                //! the conversion from steering wheel angle to steering angle as
                                                                                //! steeringAngle = P(steeringWheelAngle) + frontSteeringOffset and its reverse.
} dwVehicle;

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_RIG_VEHICLE_H_

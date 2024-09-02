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

#ifndef DWFRAMEWORK_DWNODES_EGOMOTION_DWRELATIVEEGOMOTIONNODE_DWRELATIVEEGOMOTIONNODE_HPP_
#define DWFRAMEWORK_DWNODES_EGOMOTION_DWRELATIVEEGOMOTIONNODE_DWRELATIVEEGOMOTIONNODE_HPP_

#include <dw/core/logger/Logger.h>
#include <dw/egomotion/base/Egomotion.h>
#include <dw/egomotion/2.0/Egomotion2.h>
#include <dw/egomotion/2.0/Egomotion2Extra.h>
#include <dw/roadcast/base_types/RoadCastPacketTypes.hpp>
#include <dwcgf/channel/Channel.hpp>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/Pass.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dw/egomotion/errorhandling/ErrorHandlingParameters.h>
#include <dw/egomotion/errorhandling/Types.hpp>
#include <dwframework/dwnodes/common/channelpackets/DwframeworkTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/EgomotionState.hpp>
#include <dwframework/dwnodes/common/channelpackets/IMU.hpp>
#include <dwframework/dwnodes/common/channelpackets/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/VehicleIOValStructures.hpp>

namespace dw
{
namespace framework
{

static constexpr uint32_t DW_EGOMOTION_DISABLED_ERROR_IDS_ARRAY_SIZE = 32;
static constexpr uint32_t DW_EGOMOTION_MAX_DISABLED_ERROR_IDS        = 512;
static_assert(DW_EGOMOTION_ERROR_ID_COUNT > sizeof(uint32_t) * DW_EGOMOTION_DISABLED_ERROR_IDS_ARRAY_SIZE, "Error ID count exceeds capacity of disabled error IDs bitfield (static param).");
static_assert(DW_EGOMOTION_ERROR_ID_COUNT >= DW_EGOMOTION_MAX_DISABLED_ERROR_IDS, "Max disabled error ids must be less than or equal to error id count");

struct dwRelativeEgomotionNodeInitParams
{
    // Unnamed parameters, automatically set by CGF framework based on their type
    dwConstRigHandle_t rigHandle;
    const char* imuSensorName;

    // Named parameters, defined in Egomotion.graphlet.json. Please keep them in the same order.
    // Make sure to explictly set non-zero defaults to the same value as in graphlet.
    // GRAPHLET DEFAULT VALUES FOR BELOW PARAMETERS WILL OVERRIDE EGOMOTION'S INTERNAL AUTO-SELECTED DEFAULTS AND MUST
    // THEREFORE APPLY TO ALL CARS, BE SET TO THE SAME VALUE AS INTERNAL DEFAULT OR BE SET TO A RESERVED OR INVALID VALUE.

    dwEgomotionGroundSpeedMeasurementTypes groundSpeedType{DW_EGOMOTION_GROUND_SPEED_COUNT}; ///< Ground Speed Type (debug setting for AugResim only).
    bool sigPresenceRearWheelAngle;                                                          ///< Signals if the VIO interface signal rearWheelAngle is expected to be provided by the partner VAL client and is valid.
    bool sigPresenceRearWheelAngleQuality;                                                   ///< Signals if the VIO interface signal rearWheelAngleQuality is expected to be provided by the partner VAL client and is valid
    bool sigPresenceRearWheelAngleTimestamp;                                                 ///< Signals if the VIO interface signal rearWheelAngleTimestamp is expected to be provided by the partner VAL client and is valid.
    bool sigPresenceRearWheelAngleTimestampQuality;                                          ///< Signals if the VIO interface signal rearWheelAngleTimestampQuality is expected to be provided by the partner VAL client and is valid.
    bool sigPresenceSuspensionLevel[4];                                                      ///< Signals if the VIO interface signal suspensionLevel is expected to be provided by the partner VAL client and is valid (order: FL, FR, RL, RR).
    bool sigPresenceSuspensionLevelQuality;                                                  ///< Signals if the VIO interface signal suspensionLevelQuality is expected to be provided by the partner VAL client and is valid.
    bool sigPresenceSuspensionLevelTimestamp;                                                ///< Signals if the VIO interface signal suspensionLevelTimestamp is expected to be provided by the partner VAL client and is valid.
    bool sigPresenceSuspensionLevelTimestampQuality;                                         ///< Signals if the VIO interface signal suspensionLevelTimestampQuality is expected to be provided by the partner VAL client and is valid.
    bool sigPresenceSuspensionLevelCalibrationState;                                         ///< Signals if the VIO interface signal suspensionLevelCalibrationState is expected to be provided by the partner VAL client and is valid.
    bool sigPresenceWheelSpeedRedundant;                                                     ///< Signals if the VIO interface signal wheelSpeedRedundant expected to be provided by the partner VAL client and is valid.
    bool sigPresenceWheelSpeedQualityRedundant;                                              ///< Signals if the VIO interface signal wheelSpeedQualityRedundant expected to be provided by the partner VAL client and is valid.
    bool sigPresenceWheelTicksRedundant;                                                     ///< Signals if the VIO interface signal wheelTicksRedundant expected to be provided by the partner VAL client and is valid.
    bool sigPresenceWheelTicksDirectionRedundant;                                            ///< Signals if the VIO interface signal wheelTicksDirectionRedundant expected to be provided by the partner VAL client and is valid.
    bool sigPresenceWheelTicksTimestampRedundant;                                            ///< Signals if the VIO interface signal wheelTicksTimestampRedundant expected to be provided by the partner VAL client and is valid.
    bool sigPresenceWheelTicksTimestampQualityRedundant;                                     ///< Signals if the VIO interface signal wheelTicksTimestampQualityRedundant expected to be provided by the partner VAL client and is valid.
    bool sigPresenceFrontSteeringAngleHigh;                                                  ///< Signals if the VIO interface signal frontSteeringAngleHigh is expected to be provided by the partner VAL client and is valid.
    bool sigPresenceFrontSteeringAngleControlQualityHigh;                                    ///< Signals if the VIO interface signal frontSteeringAngleControlQualityHigh is expected to be provided by the partner VAL client and is valid.
    bool sigPresenceFrontSteeringTimestampHigh;                                              ///< Signals if the VIO interface signal frontSteeringTimestampHigh is expected to be provided by the partner VAL client and is valid.
    bool sigPresenceIMUTimestampQuality;                                                     ///< Signals if the IMU Timestamp Quality signal is provided.
    bool sigPresenceIMUAccelerometerOffsetZ;                                                 ///< Signals if the IMU Accelerometer offset for Z axis is provided.
    bool sigPresenceBrakeTorqueWheelsQuality;                                                ///< Signals if the wheel brake torques quality signal is provided.
    bool sigPresenceIMUStatus;                                                               ///< Signals if the IMU Status signal is provided.
    bool sigPresenceIMUSequenceCounter;                                                      ///< Signals if the IMU sequence counter signal is provided.
    bool sigPresenceIMUTurnrateOffsetQualityStatus;                                          ///< Signals if the IMU sturnrate offset quality status signal is provided.
    dwTime_t timeOffsetSteering;                                                             ///< Time offset of steering angle measurements. This value is subtracted from signal timestamp.
    dwTime_t timeOffsetSuspension;                                                           ///< Time offset of suspension measurements. This value is subtracted from signal timestamp.
    dwTime_t timeOffsetAngularAcceleration;                                                  ///< Time offset of angular acceleration measurements. This value is subtracted from signal timestamp.
    dwTime_t timeOffsetProperAcceleration;                                                   ///< Time offset of linear acceleration measurements. This value is subtracted from signal timestamp.
    dwTime_t timeOffsetAngularVelocity;                                                      ///< Time offset of angular velocity measurements. This value is subtracted from signal timestamp.
    dwTime_t timeOffsetWheelTicks;                                                           ///< Time offset of wheel tick measurements. This value is subtracted from signal timestamp.
    dwTime_t timeOffsetWheelSpeeds;                                                          ///< Time offset of wheel speed measurements. This value is subtracted from signal timestamp.
    dwTime_t cycleTimeSteering;                                                              ///< Indicates the interval in microseconds between two steering angle updates.
    dwTime_t cycleTimeSuspension;                                                            ///< Indicates the interval in microseconds between two suspension level updates.
    dwTime_t cycleTimeIMU;                                                                   ///< Indicates the interval in microseconds between two IMU updates.
    dwTime_t cycleTimeWheelEncoder;                                                          ///< Indicates the interval in microseconds between two wheel encoder updates (wheel speeds, wheel ticks, wheel directions).
    float32_t suspensionCenterHeight[2]{-1.F, -1.F};                                         ///< Indicates the suspension center height approximation used by egomotion, in meters, above ground. Order: Height of longitudinal X axis, Height of lateral Y axis.
    dwTime_t wheelObserverFixedStepSize{0};                                                  ///< Defines the step size when running the Wheel Observer in fixed step mode.
    bool wheelObserverEnableFixedStep;                                                       ///< When true, enables Wheel Observer fixed step mode.
    float32_t wheelObserverProcessCovariance[3];                                             ///< Process covariance parameters (continuous time, coefficients on diagonal). Elements correspond to [position, speed, acceleration] states.
    float32_t wheelObserverPositionVarianceLow;                                              ///< Position (tick) count variance when speed below positionFuzzyLow or above positionFuzzyHigh. Indicates region of low trust in wheel position (tick) counters, in rad^2.
    float32_t wheelObserverPositionVarianceHigh;                                             ///< Position (tick) count variance when speed between positionFuzzyLow and positionFuzzyHigh. Indicates region of high trust in wheel position (tick) counters, in rad^2.
    float32_t wheelObserverSpeedVarianceLow;                                                 ///< Speed variance when speed below speedFuzzyLow. Indicates region of low trust in wheel speed, in rad^2/s^2.
    float32_t wheelObserverSpeedVarianceHigh;                                                ///< Speed variance when speed above speedFuzzyHigh.  Indicates region of high trust in wheel speed, in rad^2/s^2.
    uint32_t wheelObserverPositionFuzzyLow{UINT32_MAX};                                      ///< Region below which wheel position (tick) counters have low trust (variance adaptation), in increment per successsive wheel encoder measurement cycles.
    uint32_t wheelObserverPositionFuzzyHigh{UINT32_MAX};                                     ///< Region above which wheel position (tick) counters have high trust (variance adaptation), in increment per successsive wheel encoder measurement cycles.
    float32_t wheelObserverSpeedFuzzyLow{-1.F};                                              ///< Region below which wheel speeds have low trust (variance adaptation), in rad/s.
    float32_t wheelObserverSpeedFuzzyHigh{-1.F};                                             ///< Region above which wheel speeds have high trust (variance adaptation), in rad/s.
    float32_t wheelObserverSpeedMax;                                                         ///< Maximum wheel speed, in rad/s, to which the state will be constrained (treated as error if error handling is active). Must be positive.
    float32_t wheelObserverAccelerationMax;                                                  ///< Maximum wheel acceleration, in rad/s^2, to which the state will be constrained (treated as error if error handling is active). Must be positive.
    float32_t wheelObserverAccelerationMin;                                                  ///< Minimum wheel acceleration, in rad/s^2, to which the state will be constrained (treated as error if error handling is active). Must be negative.
    dwTime_t directionDetectorDurationNoWheelTick;                                           ///< Threshold of no wheel ticks duration to determine whether the vehicle rolling direction is void or stop, in microseconds.
    dwTime_t vehicleMotionObserverFixedStepSize;                                             ///< Defines the step size when running the Vehicle Motion Observer in fixed step mode.
    bool vehicleMotionObserverEnableFixedStep;                                               ///< When true, enables Vehicle Motion Observer fixed step mode.
    float32_t vehicleMotionObserverProcessCovariance[5];                                     ///< Process covariance parameters (continuous time, coefficients on diagonal). Elements correspond (approximatively) to [pitch, roll, v_x, v_y, v_z] states.
    float32_t vehicleMotionObserverInitialProcessCovariance[5];                              ///< Initial process covariance parameters (continuous time, coefficients on diagonal). Elements correspond (approximatively) to [pitch, roll, v_x, v_y, v_z] states.
    float32_t vehicleMotionObserverGroundSpeedCovariance[3];                                 ///< Ground speed measurement covariance parameters (continuous time, coefficients on diagonal). Elements correspond to [v_x, v_y, v_z] along body coordinate frame.
    float32_t vehicleMotionObserverReferencePoint[3]{-1.F, -1.F, -1.F};                      ///< Reference point used internally for computation, expressed in body coordinate frame. Should be near to IMU mounting location. Important: does not affect reference point of egomotion output estimates.
    uint32_t drivenWheels{UINT32_MAX};                                                       ///< Indicates which traction configuration is used on this vehicle. 0=AWD, 1=FWD, 2=RWD.
    float32_t errorHandlingVIOWheelSpeedRange[2];                                            ///< Bounds on valid VIO Wheel Speed Measurements, in rad/s. Order: Low Bound, Upper Bound.
    float32_t errorHandlingVIOFrontSteeringAngleRange[2];                                    ///< Bounds on valid VIO Front Steering Angle Measurements, in rad. Order: Low Bound, Upper Bound.
    float32_t errorHandlingVIORearWheelAngleRange[2];                                        ///< Bounds on valid VIO Rear Wheel Angle Measurements, in rad. Order: Low Bound, Upper Bound.
    float32_t errorHandlingVIOFrontSteeringAngleOffsetRange[2];                              ///< Bounds on valid VIO Front Steering Angle Offset Measurements, in rad. Order: Low Bound, Upper Bound.
    float32_t errorHandlingVIOWheelTorqueRange[2];                                           ///< Bounds on valid VIO Wheel Torque Measurements, in Nm. Order: Low Bound, Upper Bound.
    float32_t errorHandlingVIOSpeedMinRange[2];                                              ///< Bounds on valid VIO Speed Min Measurements, in m/s. Order: Low Bound, Upper Bound.
    float32_t errorHandlingVIOSpeedMaxRange[2];                                              ///< Bounds on valid VIO Speed Max Measurements, in m/s. Order: Low Bound, Upper Bound.
    float32_t errorHandlingVIOBrakeTorqueWheelsRange[2];                                     ///< Bounds on valid VIO Brake Torque Wheels Measurements, in Nm. Order: Low Bound, Upper Bound.
    float32_t errorHandlingVIOSuspensionLevelRange[2];                                       ///< Bounds on valid VIO Suspension Level Measurements, in m. Order: Low Bound, Upper Bound.
    float32_t errorHandlingIMUSequenceCounterRangeDeprecated[2];                             ///< Bounds on valid IMU Sequence Counter, unitless. Order: Low Bound, Upper Bound. (deprecated)
    uint32_t errorHandlingIMUSequenceCounterRange[2];                                        ///< Bounds on valid IMU Sequence Counter, unitless. Order: Low Bound, Upper Bound.
    float32_t errorHandlingIMUTurnrateRange[2];                                              ///< Bounds on valid IMU Turnrate, in rad/s. Order: Low Bound, Upper Bound.
    float32_t errorHandlingIMUTurnrateOffsetRange[2];                                        ///< Bounds on valid IMU Turnrate Offset, in rad/s. Order: Low Bound, Upper Bound.
    uint32_t errorHandlingIMUTurnrateOffsetQualityRange[2];                                  ///< Bounds on valid IMU Turnrate Offset Quality, unitless. Order: Low Bound, Upper Bound.
    float32_t errorHandlingIMUTurnrateAccelRange[2];                                         ///< Bounds on valid IMU Turnrate Acceleration, in rad/s^2. Order: Low Bound, Upper Bound.
    float32_t errorHandlingIMUAccelerationRange[2];                                          ///< Bounds on valid IMU Acceleration, in m/s^2. Order: Low Bound, Upper Bound.
    float32_t errorHandlingIMUAccelerationOffsetRange[2];                                    ///< Bounds on valid IMU Acceleration Offset, in m/s^2. Order: Low Bound, Upper Bound.
    float32_t errorHandlingIMUTemperatureRange[2];                                           ///< Bounds on valid IMU Temperature, in deg C. Order: Low Bound, Upper Bound.
    float32_t errorHandlingInternalAccelerationOffsetRange[2];                               ///< Bounds on valid accelerometer offset for egomotion internal estimator, in m/s^2. Order: Low Bound, Upper Bound.
    float32_t errorHandlingInternalAccelerationOffsetDriftSpeed[2];                          ///< Bounds on valid accelerometer drift speed for egomotion internal estimator, in m/s^3. Order: Low Bound, Upper Bound.
    float32_t errorHandlingInternalAccelerationOffsetShortTermSpan[2];                       ///< Bounds on valid accelerometer offset short term span for egomotion internal estimator, in m/s^2. Order: Low Bound, Upper Bound.
    float32_t errorHandlingInternalGyroscopeOffsetRange[2];                                  ///< Bounds on valid gyroscope offset for egomotion internal estimator, in rad/s. Order: Low Bound, Upper Bound.
    float32_t errorHandlingInternalGyroscopeOffsetDriftSpeed[2];                             ///< Bounds on valid gyroscope drift speed for egomotion internal estimator, in rad/s^2. Order: Low Bound, Upper Bound.
    float32_t errorHandlingInternalGyroscopeOffsetShortTermSpan[2];                          ///< Bounds on valid gyroscope offset short term span for egomotion internal estimator, in rad/s. Order: Low Bound, Upper Bound.
    dwLoggerVerbosity errorHandlingLogLevel{DW_LOG_SILENT};                                  ///<
    uint32_t errorHandlingCyclesBetweenLogs;                                                 ///<
    bool strictVIOMapping;                                                                   ///< If true, signals are read only from the sources agreed upon in technical safety concept. If false, valid signals are read from any available source.
    bool notifySEH{true};                                                                    ///<
    bool enableDegradations;                                                                 ///< Whether errors shall lead egomotion to degrade or invalidate its outputs.
    uint32_t disabledErrorIdsBitfield[DW_EGOMOTION_DISABLED_ERROR_IDS_ARRAY_SIZE];           ///< When a bit is set to true, the corresponding monitor is set as not applicable (disabled). The position of the bit in the array corresponds to the error ID (starting from LSB=0).
    dw::core::FixedString<128> disabledErrorIds[DW_EGOMOTION_MAX_DISABLED_ERROR_IDS];        ///<
};

static_assert(DW_EGOMOTION_MAX_DISABLED_ERROR_IDS <= static_cast<size_t>(dw::egomotion::errorhandling::ErrorId::COUNT), "Max disabled error ids must be less than or equal to error id count");

/**
 * @brief This node computes the vehicle state and relative motion over time using signals from IMU and wheelspeed sensors.
 *
 * Input modalities
 * - IMU
 * - VehicleIOState
 *
 * Output signals
 * - Egomotion state
 *
 * @ingroup dwnodes
 **/
class dwRelativeEgomotionNode : public ExceptionSafeProcessNode, public IContainsPreShutdownAction
{
public:
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwIMUFrame, "IMU_FRAME"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOASILStateE2EWrapper, "VEHICLE_IO_ASIL_STATE"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOQMState, "VEHICLE_IO_QM_STATE"_sv),
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "IMU_EXTRINSICS"_sv),
            DW_DESCRIBE_PORT(dwCalibratedWheelRadii, "WHEEL_RADII"_sv));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwEgomotionStateHandle_t, "EGOMOTION_STATE"_sv),
            DW_DESCRIBE_PORT(dwEgomotionPosePayload, "EGOMOTION_POSE_PAYLOAD"_sv),
            DW_DESCRIBE_PORT(dwCalibratedIMUIntrinsics, "IMU_INTRINSICS"_sv),
            DW_DESCRIBE_PORT(dwValEgomotion, "VAL_EGOMOTION_DATA"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT(dwValEgomotion, "VAL_EGOMOTION_DATA_SECONDARY"_sv, PortBinding::OPTIONAL));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"ADD_IMU"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"ADD_VEHICLE_STATE"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"UPDATE_IMU_EXTRINSICS"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"UPDATE_WHEEL_RADII"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"SEND_STATE"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwRelativeEgomotionNode> create(ParameterProvider& provider);

    dwStatus getEgomotionParameters(dwEgomotionParameters2& params);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<dwRelativeEgomotionNodeInitParams>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwRelativeEgomotionNodeInitParams::rigHandle),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    const char*,
                    semantic_parameter_types::ImuName,
                    "imuIndex"_sv,
                    &dwRelativeEgomotionNodeInitParams::imuSensorName),
                DW_DESCRIBE_PARAMETER(
                    dwEgomotionGroundSpeedMeasurementTypes,
                    "groundSpeedType"_sv,
                    &dwRelativeEgomotionNodeInitParams::groundSpeedType),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceRearWheelAngle"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceRearWheelAngle),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceRearWheelAngleQuality"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceRearWheelAngleQuality),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceRearWheelAngleTimestamp"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceRearWheelAngleTimestamp),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceRearWheelAngleTimestampQuality"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceRearWheelAngleTimestampQuality),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    bool,
                    "sigPresenceSuspensionLevel"_sv, 4,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceSuspensionLevel),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceSuspensionLevelQuality"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceSuspensionLevelQuality),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceSuspensionLevelTimestamp"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceSuspensionLevelTimestamp),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceSuspensionLevelTimestampQuality"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceSuspensionLevelTimestampQuality),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceSuspensionLevelCalibrationState"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceSuspensionLevelCalibrationState),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceWheelSpeedRedundant"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceWheelSpeedRedundant),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceWheelSpeedQualityRedundant"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceWheelSpeedQualityRedundant),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceWheelTicksRedundant"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceWheelTicksRedundant),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceWheelTicksDirectionRedundant"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceWheelTicksDirectionRedundant),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceWheelTicksTimestampRedundant"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceWheelTicksTimestampRedundant),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceWheelTicksTimestampQualityRedundant"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceWheelTicksTimestampQualityRedundant),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceFrontSteeringAngleHigh"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceFrontSteeringAngleHigh),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceFrontSteeringAngleControlQualityHigh"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceFrontSteeringAngleControlQualityHigh),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceFrontSteeringTimestampHigh"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceFrontSteeringTimestampHigh),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceIMUTimestampQuality"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceIMUTimestampQuality),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceIMUAccelerometerOffsetZ"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceIMUAccelerometerOffsetZ),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceBrakeTorqueWheelsQuality"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceBrakeTorqueWheelsQuality),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceIMUStatus"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceIMUStatus),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceIMUSequenceCounter"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceIMUSequenceCounter),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceIMUTurnrateOffsetQualityStatus"_sv,
                    &dwRelativeEgomotionNodeInitParams::sigPresenceIMUTurnrateOffsetQualityStatus),
                DW_DESCRIBE_PARAMETER(
                    dwTime_t,
                    "timeOffsetSteering"_sv,
                    &dwRelativeEgomotionNodeInitParams::timeOffsetSteering),
                DW_DESCRIBE_PARAMETER(
                    dwTime_t,
                    "timeOffsetSuspension"_sv,
                    &dwRelativeEgomotionNodeInitParams::timeOffsetSuspension),
                DW_DESCRIBE_PARAMETER(
                    dwTime_t,
                    "timeOffsetAngularAcceleration"_sv,
                    &dwRelativeEgomotionNodeInitParams::timeOffsetAngularAcceleration),
                DW_DESCRIBE_PARAMETER(
                    dwTime_t,
                    "timeOffsetProperAcceleration"_sv,
                    &dwRelativeEgomotionNodeInitParams::timeOffsetProperAcceleration),
                DW_DESCRIBE_PARAMETER(
                    dwTime_t,
                    "timeOffsetAngularVelocity"_sv,
                    &dwRelativeEgomotionNodeInitParams::timeOffsetAngularVelocity),
                DW_DESCRIBE_PARAMETER(
                    dwTime_t,
                    "timeOffsetWheelTicks"_sv,
                    &dwRelativeEgomotionNodeInitParams::timeOffsetWheelTicks),
                DW_DESCRIBE_PARAMETER(
                    dwTime_t,
                    "timeOffsetWheelSpeeds"_sv,
                    &dwRelativeEgomotionNodeInitParams::timeOffsetWheelSpeeds),
                DW_DESCRIBE_PARAMETER(
                    dwTime_t,
                    "cycleTimeSteering"_sv,
                    &dwRelativeEgomotionNodeInitParams::cycleTimeSteering),
                DW_DESCRIBE_PARAMETER(
                    dwTime_t,
                    "cycleTimeSuspension"_sv,
                    &dwRelativeEgomotionNodeInitParams::cycleTimeSuspension),
                DW_DESCRIBE_PARAMETER(
                    dwTime_t,
                    "cycleTimeIMU"_sv,
                    &dwRelativeEgomotionNodeInitParams::cycleTimeIMU),
                DW_DESCRIBE_PARAMETER(
                    dwTime_t,
                    "cycleTimeWheelEncoder"_sv,
                    &dwRelativeEgomotionNodeInitParams::cycleTimeWheelEncoder),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "suspensionCenterHeight"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::suspensionCenterHeight),
                DW_DESCRIBE_PARAMETER(
                    dwTime_t,
                    "wheelObserverFixedStepSize"_sv,
                    &dwRelativeEgomotionNodeInitParams::wheelObserverFixedStepSize),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "wheelObserverEnableFixedStep"_sv,
                    &dwRelativeEgomotionNodeInitParams::wheelObserverEnableFixedStep),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "wheelObserverProcessCovariance"_sv, 3,
                    &dwRelativeEgomotionNodeInitParams::wheelObserverProcessCovariance),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "wheelObserverPositionVarianceLow"_sv,
                    &dwRelativeEgomotionNodeInitParams::wheelObserverPositionVarianceLow),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "wheelObserverPositionVarianceHigh"_sv,
                    &dwRelativeEgomotionNodeInitParams::wheelObserverPositionVarianceHigh),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "wheelObserverSpeedVarianceLow"_sv,
                    &dwRelativeEgomotionNodeInitParams::wheelObserverSpeedVarianceLow),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "wheelObserverSpeedVarianceHigh"_sv,
                    &dwRelativeEgomotionNodeInitParams::wheelObserverSpeedVarianceHigh),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "wheelObserverPositionFuzzyLow"_sv,
                    &dwRelativeEgomotionNodeInitParams::wheelObserverPositionFuzzyLow),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "wheelObserverPositionFuzzyHigh"_sv,
                    &dwRelativeEgomotionNodeInitParams::wheelObserverPositionFuzzyHigh),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "wheelObserverSpeedFuzzyLow"_sv,
                    &dwRelativeEgomotionNodeInitParams::wheelObserverSpeedFuzzyLow),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "wheelObserverSpeedFuzzyHigh"_sv,
                    &dwRelativeEgomotionNodeInitParams::wheelObserverSpeedFuzzyHigh),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "wheelObserverSpeedMax"_sv,
                    &dwRelativeEgomotionNodeInitParams::wheelObserverSpeedMax),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "wheelObserverAccelerationMax"_sv,
                    &dwRelativeEgomotionNodeInitParams::wheelObserverAccelerationMax),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "wheelObserverAccelerationMin"_sv,
                    &dwRelativeEgomotionNodeInitParams::wheelObserverAccelerationMin),
                DW_DESCRIBE_PARAMETER(
                    dwTime_t,
                    "directionDetectorDurationNoWheelTick"_sv,
                    &dwRelativeEgomotionNodeInitParams::directionDetectorDurationNoWheelTick),
                DW_DESCRIBE_PARAMETER(
                    dwTime_t,
                    "vehicleMotionObserverFixedStepSize"_sv,
                    &dwRelativeEgomotionNodeInitParams::vehicleMotionObserverFixedStepSize),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "vehicleMotionObserverEnableFixedStep"_sv,
                    &dwRelativeEgomotionNodeInitParams::vehicleMotionObserverEnableFixedStep),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "vehicleMotionObserverProcessCovariance"_sv, 5,
                    &dwRelativeEgomotionNodeInitParams::vehicleMotionObserverProcessCovariance),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "vehicleMotionObserverInitialProcessCovariance"_sv, 5,
                    &dwRelativeEgomotionNodeInitParams::vehicleMotionObserverInitialProcessCovariance),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "vehicleMotionObserverGroundSpeedCovariance"_sv, 3,
                    &dwRelativeEgomotionNodeInitParams::vehicleMotionObserverGroundSpeedCovariance),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "vehicleMotionObserverReferencePoint"_sv, 3,
                    &dwRelativeEgomotionNodeInitParams::vehicleMotionObserverReferencePoint),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "drivenWheels"_sv,
                    &dwRelativeEgomotionNodeInitParams::drivenWheels),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingVIOWheelSpeedRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingVIOWheelSpeedRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingVIOFrontSteeringAngleRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingVIOFrontSteeringAngleRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingVIORearWheelAngleRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingVIORearWheelAngleRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingVIOFrontSteeringAngleOffsetRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingVIOFrontSteeringAngleOffsetRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingVIOWheelTorqueRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingVIOWheelTorqueRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingVIOSpeedMinRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingVIOSpeedMinRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingVIOSpeedMaxRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingVIOSpeedMaxRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingVIOBrakeTorqueWheelsRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingVIOBrakeTorqueWheelsRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingVIOSuspensionLevelRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingVIOSuspensionLevelRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingIMUSequenceCounterRangeDeprecated"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingIMUSequenceCounterRangeDeprecated),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    uint32_t,
                    "errorHandlingIMUSequenceCounterRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingIMUSequenceCounterRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingIMUTurnrateRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingIMUTurnrateRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingIMUTurnrateOffsetRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingIMUTurnrateOffsetRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    uint32_t,
                    "errorHandlingIMUTurnrateOffsetQualityRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingIMUTurnrateOffsetQualityRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingIMUTurnrateAccelRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingIMUTurnrateAccelRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingIMUAccelerationRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingIMUAccelerationRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingIMUAccelerationOffsetRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingIMUAccelerationOffsetRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingIMUTemperatureRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingIMUTemperatureRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingInternalAccelerationOffsetRange"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingInternalAccelerationOffsetRange),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingInternalAccelerationOffsetDriftSpeed"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingInternalAccelerationOffsetDriftSpeed),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingInternalAccelerationOffsetShortTermSpan"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingInternalAccelerationOffsetShortTermSpan),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingInternalGyroscopeOffsetDriftSpeed"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingInternalGyroscopeOffsetDriftSpeed),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    float32_t,
                    "errorHandlingInternalGyroscopeOffsetShortTermSpan"_sv, 2,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingInternalGyroscopeOffsetShortTermSpan),
                DW_DESCRIBE_PARAMETER(
                    dwLoggerVerbosity,
                    "errorHandlingLogLevel"_sv,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingLogLevel),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "errorHandlingCyclesBetweenLogs"_sv,
                    &dwRelativeEgomotionNodeInitParams::errorHandlingCyclesBetweenLogs),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "strictVIOMapping"_sv,
                    &dwRelativeEgomotionNodeInitParams::strictVIOMapping),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "notifySEH"_sv,
                    &dwRelativeEgomotionNodeInitParams::notifySEH),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enableDegradations"_sv,
                    &dwRelativeEgomotionNodeInitParams::enableDegradations),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    uint32_t,
                    "disabledErrorIdsBitfield"_sv, DW_EGOMOTION_DISABLED_ERROR_IDS_ARRAY_SIZE,
                    &dwRelativeEgomotionNodeInitParams::disabledErrorIdsBitfield),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    dw::core::FixedString<128>,
                    "disabledErrorIds"_sv, DW_EGOMOTION_MAX_DISABLED_ERROR_IDS,
                    &dwRelativeEgomotionNodeInitParams::disabledErrorIds)),
            describeConstructorArgument<dwContextHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    dwRelativeEgomotionNode(const dwRelativeEgomotionNodeInitParams& params,
                            const dwContextHandle_t ctx);

    dwStatus preShutdown() override;
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_EGOMOTION_DWRELATIVEEGOMOTIONNODE_DWRELATIVEEGOMOTIONNODE_HPP_

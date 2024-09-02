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
// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONSUSPENSIONSTATENODE_DWSELFCALIBRATIONSUSPENSIONSTATENODE_HPP_
#define DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONSUSPENSIONSTATENODE_DWSELFCALIBRATIONSUSPENSIONSTATENODE_HPP_

#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/node/impl/ExceptionSafeNode.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>

#include <dwframework/dwnodes/common/channelpackets/EgomotionState.hpp>
#include <dwframework/dwnodes/common/channelpackets/FeatureList.hpp>
#include <dwframework/dwnodes/common/channelpackets/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/VehicleIOValStructures.hpp>
#include <dwframework/dwnodes/common/channelpackets/IMU.hpp>
#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>
#include <dw/core/context/Context.h>

namespace dw
{
namespace framework
{

/// @brief the parameters used to construct the dwSelfCalibrationSuspensionStateNode. They are assigned in describeParameters() method
struct dwSelfCalibrationSuspensionStateNodeParams
{
    // These parameters are assigned in describeParameters()
    dwConstRigHandle_t rigHandle; //!< rig handle

    int32_t imuSensorIndex; //!< Sensor type index of imu, provided as parameter. Index with value >=0 means imu has to be used, <0 do not use imu.

    int32_t cameraSensorIndices[SELF_CALIBRATION_NODE_MAX_CAMERAS_FOR_SUSPENSION_STATE]; //!< Sensor type indices for cameras (e.g. camera [0-8]), provided as parameter. Index with value >=0 means imu has to be used, <0 do not use imu.

    bool enableCalibration; //!< flag to enable or disable calibration

    uint32_t channelFifoSize;     //!< size of the fifo buffer used in vio and imu port. Need to be positive.
    cudaStream_t cudaStreamIndex; //!< the cuda stream used for processing

    bool sigPresenceSuspensionLevel[SELF_CALIBRATION_NUM_SUSPENSION_LEVEL_SIGNALS]; //!< Signals if the VIO interface signal suspensionLevel is expected to be provided by the partner VAL client and is valid (order: FL, FR, RL, RR).
    bool sigPresenceSuspensionLevelQuality;                                         //!< Signals if the VIO interface signal suspensionLevelQuality is expected to be provided by the partner VAL client and is valid.
    bool sigPresenceSuspensionLevelTimestamp;                                       //!< Signals if the VIO interface signal suspensionLevelTimestamp is expected to be provided by the partner VAL client and is valid.
    bool sigPresenceSuspensionLevelTimestampQuality;                                //!< Signals if the VIO interface signal suspensionLevelTimestampQuality is expected to be provided by the partner VAL client and is valid.
    bool sigPresenceSuspensionLevelCalibrationState;                                //!< Signals if the VIO interface signal suspensionLevelCalibrationState is expected to be provided by the partner VAL client and is valid.

    //  This is a WAR to resolve the issue seen in the development of multi-process RoadRunner.
    //  Details of the issue is available: https://nvbugs/3991262
    dwTime_t nvSciChannelWaitTimeUs{}; //!< Wait time for nvSciChannel

    //  These parameters are assigned in create()
    int32_t imuSensorRigIndex; //!< Imu sensor rig index (e.g. imu [0-127]), auto-populated by RR2 Loader. Filled in with UNUSED if the corresponding imuSensorIndex is UNUSED

    int32_t cameraSensorRigIndices[SELF_CALIBRATION_NODE_MAX_CAMERAS_FOR_SUSPENSION_STATE]; //!< Camera sensor indices rig index (e.g. camera [0-127]), auto-populated by RR2 Loader. Filled in with UNUSED if the corresponding cameraSensorIndices[i] is UNUSED
};

/**
 * @ingroup dwnodes
 *
 * @brief Node that handles the estimation of the suspension state
 *
 * This nodes observes
 * - vehicle state
 * - imu measurements
 * - cameras (up to 4)
 * and aim at the estimation of the car suspensions state.
 *
 * Construction parameters for this node are specified in `dwSelfCalibrationSuspensionStateNodeParams` struct.
 *
 * The node constructor performs the following checks on the parameters
 * - channelFifoSize has to be positive
 * - imuSensorIndex and imuSensorRigIndex have to be both valid or UNUSED. A sensor index is valid if it is >=0 
 * - cameraSensorIndices[i] and cameraSensorRigIndices[i] have to be both valid or UNUSED per each i=[0:SELF_CALIBRATION_NODE_MAX_CAMERAS_FOR_SUSPENSION_STATE-1].
 * - cameraSensorIndices cannot contains repeated valid values
 * - cameraSensorRigIndices cannot contains repeated valid values
 * If one of the requirements is not met, the constructor fails and throws a Exception with DW_INVALID_ARGUMENT. The exception message contains useful information to identify the issue.
 *
 *
 * The input ports are
 * - VEHICLE_IO_QM_STATE
 * - VEHICLE_IO_ASIL_STATE
 * - IMU_FRAME
 * - IMU_EXTRINSICS
 * - EGOMOTION_STATE_ODO_IMU
 * - CAMERA_FEATURE_DETECTION[4]
 * - CAMERA_EXTRINSICS[4]
 * - CAMERA_TWO_VIEW_TRANSFORMATION[4]
 *
 * The output ports are
 * - SUSPENSION_STATE
 * - ROADCAST_SELFCALIBRATION_SUSPENSIONSTATE_DIAGNOSTICS
 *
 * All ports are connected as optional, however 
 * - if at least one of CAMERA_*[i] input ports are bound, 
 *   the corresponding cameraSensorIndices[i] and cameraSensorRigIndices[i] parameters have to be valid
 * - if at least one of IMU_* input port are bound, the imuSensorIndex and imuSensorRigIndex parameters have to be valid
 * The above rules are ignored if the parameter enableCalibration is false.
 *
 * Checks on input ports are performed in the `validate()` method. Logs are produced in case of errors.
 *
 * Note that, at the time of writing, the `validate()` method is not called in RR2.
 * A mechanism to call `validate()` in the first execution of the first pass of this node.
 *
 * Output port are connnected as optional. Outputs are produced at every cycle.
 * If the output ports are not bound (no downstream consumers connected), a warning is reported.
 * Such warning is reported only once to not pollute the log file.
 *
 */
class dwSelfCalibrationSuspensionStateNode : public ExceptionSafeProcessNode
{
public:
    // a valid sensor has non negative a index, define -1 as the UNUSED sensor index
    static constexpr int32_t UNUSED_SENSOR_INDEX = -1;

    // TODO(sceriani): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwSelfCalibrationSuspensionStateNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(sceriani): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            // vehicle
            DW_DESCRIBE_PORT(dwVehicleIOQMState, "VEHICLE_IO_QM_STATE"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT(dwVehicleIOASILStateE2EWrapper, "VEHICLE_IO_ASIL_STATE"_sv, PortBinding::OPTIONAL),
            // imu
            DW_DESCRIBE_PORT(dwIMUFrame, "IMU_FRAME"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "IMU_EXTRINSICS"_sv, PortBinding::OPTIONAL),
            // egomotion
            DW_DESCRIBE_PORT(dwEgomotionStateHandle_t, "EGOMOTION_STATE_ODO_IMU"_sv, PortBinding::OPTIONAL),
            // add camera calibration inputs
            DW_DESCRIBE_PORT_ARRAY(dwFeatureHistoryArray, SELF_CALIBRATION_NODE_MAX_CAMERAS_FOR_SUSPENSION_STATE, "CAMERA_FEATURE_DETECTION"_sv, PortBinding::OPTIONAL),
            // add camera calibration output (final and intermediate)
            DW_DESCRIBE_PORT_ARRAY(dwCalibratedExtrinsics, SELF_CALIBRATION_NODE_MAX_CAMERAS_FOR_SUSPENSION_STATE, "CAMERA_EXTRINSICS"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT_ARRAY(dwCameraTwoViewTransformation, SELF_CALIBRATION_NODE_MAX_CAMERAS_FOR_SUSPENSION_STATE, "CAMERA_TWO_VIEW_TRANSFORMATION"_sv, PortBinding::OPTIONAL));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(sceriani): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwCalibratedSuspensionStateProperties, "SUSPENSION_STATE"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT(dwSelfCalibrationSuspensionStateDiagnostics, "ROADCAST_SELFCALIBRATION_SUSPENSIONSTATE_DIAGNOSTICS"_sv, PortBinding::OPTIONAL));
    };

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS_NONCAMERA_INPUTS"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS_SUSPENSIONSTATE_GPU_ASYNC"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"PROCESS_SUSPENSIONSTATE_CPU_SYNC"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwSelfCalibrationSuspensionStateNode> create(ParameterProvider const& provider);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeParameters()
    {
        // TODO(sceriani): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<dwSelfCalibrationSuspensionStateNodeParams>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwSelfCalibrationSuspensionStateNodeParams::rigHandle),
                DW_DESCRIBE_PARAMETER(
                    int32_t,
                    "imuSensorIndex"_sv,
                    &dwSelfCalibrationSuspensionStateNodeParams::imuSensorIndex),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    int32_t,
                    "cameraSensorIndices"_sv,
                    SELF_CALIBRATION_NODE_MAX_CAMERAS_FOR_SUSPENSION_STATE,
                    &dwSelfCalibrationSuspensionStateNodeParams::cameraSensorIndices),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enableCalibration"_sv,
                    &dwSelfCalibrationSuspensionStateNodeParams::enableCalibration),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "channelFifoSize"_sv,
                    &dwSelfCalibrationSuspensionStateNodeParams::channelFifoSize),
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "cudaStreamIndex"_sv,
                    &dwSelfCalibrationSuspensionStateNodeParams::cudaStreamIndex),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    bool,
                    "sigPresenceSuspensionLevel"_sv, SELF_CALIBRATION_NUM_SUSPENSION_LEVEL_SIGNALS,
                    &dwSelfCalibrationSuspensionStateNodeParams::sigPresenceSuspensionLevel),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceSuspensionLevelQuality"_sv,
                    &dwSelfCalibrationSuspensionStateNodeParams::sigPresenceSuspensionLevelQuality),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceSuspensionLevelTimestamp"_sv,
                    &dwSelfCalibrationSuspensionStateNodeParams::sigPresenceSuspensionLevelTimestamp),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceSuspensionLevelTimestampQuality"_sv,
                    &dwSelfCalibrationSuspensionStateNodeParams::sigPresenceSuspensionLevelTimestampQuality),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceSuspensionLevelCalibrationState"_sv,
                    &dwSelfCalibrationSuspensionStateNodeParams::sigPresenceSuspensionLevelCalibrationState),
                DW_DESCRIBE_PARAMETER_WITH_DEFAULT(
                    dwTime_t,
                    "nvSciChannelWaitTimeUs"_sv,
                    0,
                    &dwSelfCalibrationSuspensionStateNodeParams::nvSciChannelWaitTimeUs)),
            describeConstructorArgument<dwContextHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwSelfCalibrationSuspensionStateNode(dwSelfCalibrationSuspensionStateNodeParams const& param, dwContextHandle_t const ctx);
};
} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONSUSPENSIONSTATENODE_DWSELFCALIBRATIONSUSPENSIONSTATENODE_HPP_

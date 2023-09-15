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
// SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_EGOMOTION_DWRELATIVEEGOMOTIONIMUNODE_DWRELATIVEEGOMOTIONIMUNODE_HPP_
#define DWFRAMEWORK_DWNODES_EGOMOTION_DWRELATIVEEGOMOTIONIMUNODE_DWRELATIVEEGOMOTIONIMUNODE_HPP_

#include <dw/egomotion/base/Egomotion.h>
#include <dw/roadcast/base_types/RoadCastPacketTypes.hpp>
#include <dwcgf/channel/Channel.hpp>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/Pass.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
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

struct dwRelativeEgomotionIMUNodeInitParams
{
    dwConstRigHandle_t rigHandle;
    const char* imuSensorName;
    const char* vehicleSensorName;

    dwMotionModel motionModel;
    bool estimateInitialOrientation;
    bool automaticUpdate;
    bool enableSuspension;
    uint32_t historySize;
    dwEgomotionSpeedMeasurementType speedMeasurementType;
    dwEgomotionLinearAccelerationFilterParams linearAccelerationFilterParameters;
};

template <>
struct EnumDescription<dwMotionModel>
{
    static constexpr auto get()
    {
        return describeEnumeratorCollection<dwMotionModel>(
            DW_DESCRIBE_C_ENUMERATOR(DW_EGOMOTION_ODOMETRY),
            DW_DESCRIBE_C_ENUMERATOR(DW_EGOMOTION_IMU_ODOMETRY));
    }
};

template <>
struct EnumDescription<dwEgomotionSpeedMeasurementType>
{
    static constexpr auto get()
    {
        return describeEnumeratorCollection<dwEgomotionSpeedMeasurementType>(
            DW_DESCRIBE_C_ENUMERATOR(DW_EGOMOTION_FRONT_SPEED),
            DW_DESCRIBE_C_ENUMERATOR(DW_EGOMOTION_REAR_SPEED),
            DW_DESCRIBE_C_ENUMERATOR(DW_EGOMOTION_REAR_WHEEL_SPEED));
    }
};

template <>
struct EnumDescription<dwEgomotionLinearAccelerationFilterMode>
{
    static constexpr auto get()
    {
        return describeEnumeratorCollection<dwEgomotionLinearAccelerationFilterMode>(
            DW_DESCRIBE_C_ENUMERATOR(DW_EGOMOTION_ACC_FILTER_NO_FILTERING),
            DW_DESCRIBE_C_ENUMERATOR(DW_EGOMOTION_ACC_FILTER_SIMPLE));
    }
};

/**
* @brief This node computes the vehicle state and relative motion over time using signals from IMU and wheelspeed sensors.
*
* The user has the option to use a odometry-only model as well, where IMU is not used anymore.
*
* Input modalities
* - IMU
* - Wheelspeeds
* - Steering
*
* Output signals
* - Egomotion state
*
* @ingroup dwnodes
**/
class dwRelativeEgomotionIMUNode : public ExceptionSafeProcessNode, public IAsyncResetable, public IContainsPreShutdownAction
{
public:
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwIMUFrame, "IMU_FRAME"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOSafetyState, "VEHICLE_IO_SAFETY_STATE"_sv),
            DW_DESCRIBE_PORT(dwVehicleIONonSafetyState, "VEHICLE_IO_NON_SAFETY_STATE"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOActuationFeedback, "VEHICLE_IO_ACTUATION_FEEDBACK"_sv),
            DW_DESCRIBE_PORT(dwSensorNodeProperties, "IMU_EXTRINSICS"_sv),
            DW_DESCRIBE_PORT(dwCalibratedWheelRadii, "WHEEL_RADII"_sv));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwEgomotionStateHandle_t, "EGOMOTION_STATE"_sv),
            DW_DESCRIBE_PORT(void*, "MODULE_HANDLE"_sv),
            DW_DESCRIBE_PORT(dwEgomotionPosePayload, "EGOMOTION_POSE_PAYLOAD"_sv),
            DW_DESCRIBE_PORT(dwCalibratedIMUIntrinsics, "IMU_INTRINSICS"_sv));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"ADD_IMU"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"ADD_VEHICLE_STATE"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"UPDATE_IMU_EXTRINSICS"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"UPDATE_WHEEL_RADII"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"SEND_STATE"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwRelativeEgomotionIMUNode> create(ParameterProvider& provider);

    dwStatus setAsyncReset() override
    {
        return ExceptionGuard::guardWithReturn([&]() {
            // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
            auto asyncResetNode = dynamic_cast<IAsyncResetable*>(m_impl.get());
            if (asyncResetNode != nullptr)
            {
                return asyncResetNode->setAsyncReset();
            }
            return DW_FAILURE;
        },
                                               dw::core::Logger::Verbosity::DEBUG);
    }

    dwStatus executeAsyncReset() override
    {
        return ExceptionGuard::guardWithReturn([&]() {
            // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
            auto asyncResetNode = dynamic_cast<IAsyncResetable*>(m_impl.get());
            if (asyncResetNode != nullptr)
            {
                return asyncResetNode->executeAsyncReset();
            }
            return DW_FAILURE;
        },
                                               dw::core::Logger::Verbosity::DEBUG);
    }

    dwStatus preShutdown() override
    {
        // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
        auto* preShutdownNode = dynamic_cast<IContainsPreShutdownAction*>(m_impl.get());
        if (preShutdownNode)
        {
            return preShutdownNode->preShutdown();
        }
        return DW_NOT_SUPPORTED;
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments<dwRelativeEgomotionIMUNodeInitParams, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwRelativeEgomotionIMUNodeInitParams::rigHandle),
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    const char*,
                    semantic_parameter_types::ImuName,
                    &dwRelativeEgomotionIMUNodeInitParams::imuSensorName),
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    const char*,
                    semantic_parameter_types::VehicleSensorName,
                    &dwRelativeEgomotionIMUNodeInitParams::vehicleSensorName),
                DW_DESCRIBE_PARAMETER(
                    dwMotionModel,
                    "motionModel"_sv,
                    &dwRelativeEgomotionIMUNodeInitParams::motionModel),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "estimateInitialOrientation"_sv,
                    &dwRelativeEgomotionIMUNodeInitParams::estimateInitialOrientation),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "automaticUpdate"_sv,
                    &dwRelativeEgomotionIMUNodeInitParams::automaticUpdate),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enableSuspension"_sv,
                    &dwRelativeEgomotionIMUNodeInitParams::enableSuspension),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "historySize"_sv,
                    &dwRelativeEgomotionIMUNodeInitParams::historySize),
                DW_DESCRIBE_PARAMETER(
                    dwEgomotionSpeedMeasurementType,
                    "speedMeasurementType"_sv,
                    &dwRelativeEgomotionIMUNodeInitParams::speedMeasurementType),
                // when params.motionModel is DW_EGOMOTION_ODOMETRY following filter parameters take no effect.
                DW_DESCRIBE_PARAMETER(
                    dwEgomotionLinearAccelerationFilterMode,
                    "linearAccelerationFilterMode"_sv,
                    &dwRelativeEgomotionIMUNodeInitParams::linearAccelerationFilterParameters, &dwEgomotionLinearAccelerationFilterParams::mode),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "linearAccelerationFilterTimeConst"_sv,
                    &dwRelativeEgomotionIMUNodeInitParams::linearAccelerationFilterParameters, &dwEgomotionLinearAccelerationFilterParams::accelerationFilterTimeConst),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "linearAccelerationFilterProcessNoiseStdevSpeed"_sv,
                    &dwRelativeEgomotionIMUNodeInitParams::linearAccelerationFilterParameters, &dwEgomotionLinearAccelerationFilterParams::processNoiseStdevSpeed),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "linearAccelerationFilterProcessNoiseStdevAcceleration"_sv,
                    &dwRelativeEgomotionIMUNodeInitParams::linearAccelerationFilterParameters, &dwEgomotionLinearAccelerationFilterParams::processNoiseStdevAcceleration),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "linearAccelerationFilterMeasurementNoiseStdevSpeed"_sv,
                    &dwRelativeEgomotionIMUNodeInitParams::linearAccelerationFilterParameters, &dwEgomotionLinearAccelerationFilterParams::measurementNoiseStdevSpeed),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "linearAccelerationFilterMeasurementNoiseStdevAcceleration"_sv,
                    &dwRelativeEgomotionIMUNodeInitParams::linearAccelerationFilterParameters, &dwEgomotionLinearAccelerationFilterParams::measurementNoiseStdevAcceleration)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    dwRelativeEgomotionIMUNode(const dwRelativeEgomotionIMUNodeInitParams& params,
                               const dwContextHandle_t ctx);
};
} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_EGOMOTION_DWRELATIVEEGOMOTIONIMUNODE_DWRELATIVEEGOMOTIONIMUNODE_HPP_

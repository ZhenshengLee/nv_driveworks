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
// SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_SENSORS_DWIMUNODE_DWIMUCHANNELNODE_HPP_
#define DWFRAMEWORK_DWNODES_SENSORS_DWIMUNODE_DWIMUCHANNELNODE_HPP_

#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/imu/IMU.h>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/Pass.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/channelpackets/IMU.hpp>
#include <dwframework/dwnodes/common/channelpackets/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/Sensors.hpp>

namespace dw
{
namespace framework
{

struct IMUNodeChannelParams
{
    const char* sensorName;
    dwConstRigHandle_t rigHandle;
    FixedString<32> frameSkipMask;
};

/**
 * @ingroup dwnodes
 */
class dwIMUChannelNode : public ExceptionSafeSensorNode
{

public:
    // TODO(csketch): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwIMUChannelNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "SENSOR_EXTRINSICS"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT(dwTime_t, "VIRTUAL_SYNC_TIME"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwIMUFrame, "INPUT_FRAME"_sv, PortBinding::REQUIRED));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwSensorNodeProperties, "SENSOR_PROPERTIES"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "SENSOR_EXTRINSICS"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT(dwIMUFrame, "PROCESSED_DATA"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwSensorStats, "SENSOR_STATS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwSensorTsAndID, "TIMESTAMP"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwTime_t, "NEXT_TIMESTAMP"_sv, PortBinding::REQUIRED));
    };

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESSED_OUTPUT"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    };

    dwSensorType getSensorType() const
    {
        return DW_SENSOR_IMU;
    }

    static std::unique_ptr<dwIMUChannelNode> create(ParameterProvider& provider);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<const char*>(
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    const char*,
                    semantic_parameter_types::ImuName,
                    "imuIndex"_sv)),
            describeConstructorArgument<dwConstRigHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t)),
            describeConstructorArgument<dwSALHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwSALHandle_t)),
            describeConstructorArgument<dwContextHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)),
            describeConstructorArgument<FixedString<32>>(
                DW_DESCRIBE_PARAMETER(
                    dw::core::FixedString<32>,
                    "frameSkipMask"_sv)),
            describeConstructorArgument<bool>(
                DW_DESCRIBE_PARAMETER_WITH_DEFAULT(
                    bool,
                    "errorHandlingEnabled"_sv,
                    false)));
    }

    dwIMUChannelNode(const char* sensorName, dwConstRigHandle_t rigHandle,
                     dwSALHandle_t sal, dwContextHandle_t ctx, const FixedString<32>& frameSkipMask, bool errorHandlingEnabled = false);
    dwIMUChannelNode(IMUNodeChannelParams& params, const char* sensorName, dwConstRigHandle_t rigHandle,
                     dwSALHandle_t sal, dwContextHandle_t ctx);
    dwIMUChannelNode(const dwSensorParams& params,
                     dwSALHandle_t sal, dwContextHandle_t ctx);
};
}
}
#endif // DWFRAMEWORK_DWNODES_SENSORS_DWIMUNODE_DWIMUCHANNELNODE_HPP_

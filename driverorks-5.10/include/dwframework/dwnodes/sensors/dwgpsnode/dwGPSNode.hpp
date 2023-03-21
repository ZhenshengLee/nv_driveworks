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
// SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_SENSORS_DWGPSNODE_DWGPSNODE_HPP_
#define DWFRAMEWORK_DWNODES_SENSORS_DWGPSNODE_DWGPSNODE_HPP_

#include <dw/sensors/Sensors.h>
#include <dw/sensors/gps/GPS.h>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/Pass.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/channelpackets/GPS.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/Sensors.hpp>

namespace dw
{
namespace framework
{

struct dwGPSNodeParams
{
    const char* sensorName;
    dwConstRigHandle_t rigHandle;
    bool inputModeVal;
    FixedString<32> frameSkipMask;
};

/**
 * @ingroup dwnodes
 */
class dwGPSNode : public ExceptionSafeSensorNode
{
public:
    static constexpr char LOG_TAG[] = "dwGPSNode";

    static constexpr auto describeInputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwTime_t, "VIRTUAL_SYNC_TIME"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwGPSFrame, "FRAME"_sv, PortBinding::REQUIRED));
    };

    static constexpr auto describeOutputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwSensorNodeProperties, "SENSOR_PROPERTIES"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwGPSFrame, "PROCESSED_DATA"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwSensorNodeRawData, "RAW_DATA"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwSensorTsAndID, "TIMESTAMP"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwTime_t, "NEXT_TIMESTAMP"_sv, PortBinding::REQUIRED));
    };

    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("RAW_OUTPUT"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PROCESSED_OUTPUT"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    dwSensorType getSensorType() const
    {
        return DW_SENSOR_GPS;
    }

    static std::unique_ptr<dwGPSNode> create(ParameterProvider& provider);

    static constexpr auto describeParameters()
    {
        return describeConstructorArguments<dwGPSNodeParams, const char*, dwRigHandle_t, dwSALHandle_t, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_PARAMETER_WITH_DEFAULT(
                    bool,
                    "inputModeVal"_sv,
                    false,
                    &dwGPSNodeParams::inputModeVal),
                DW_DESCRIBE_PARAMETER(
                    dw::core::FixedString<32>,
                    "frameSkipMask"_sv,
                    &dwGPSNodeParams::frameSkipMask)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    const char*,
                    semantic_parameter_types::GpsName)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwRigHandle_t)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwSALHandle_t)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    };

    dwGPSNode(const char* sensorName, dwConstRigHandle_t rigHandle,
              dwSALHandle_t sal, dwContextHandle_t ctx, const FixedString<32>& frameSkipMask);
    dwGPSNode(const dwGPSNodeParams& params, const char* sensorName, dwConstRigHandle_t rigHandle,
              dwSALHandle_t sal, dwContextHandle_t ctx);
    dwGPSNode(const dwSensorParams& params,
              dwSALHandle_t sal, dwContextHandle_t ctx);
};
}
}
#endif // DWFRAMEWORK_DWNODES_SENSORS_DWGPSNODE_DWGPSNODE_HPP_

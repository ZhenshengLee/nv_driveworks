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

#ifndef DW_FRAMEWORK_GPS_CHANNEL_NODE_HPP_
#define DW_FRAMEWORK_GPS_CHANNEL_NODE_HPP_

#include <dwcgf/node/Node.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/pass/Pass.hpp>
/* Need to include the appropriate ChannelPacketTypes.hpp since port initialization requires
   the parameter_trait overrides. Otherwise, it will be considered as a packet of generic type. */
#include <dwframework/dwnodes/common/ChannelPacketTypes.hpp>

#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/SensorCommonTypes.hpp>

#include <dw/sensors/Sensors.h>
#include <dw/sensors/gps/GPS.h>

namespace dw
{
namespace framework
{

struct GPSNodeChannelParams
{
    const char* sensorName;
    dwConstRigHandle_t rigHandle;
};

/**
 * @ingroup dwnodes
 */
class dwGPSChannelNode : public ExceptionSafeSensorNode
{
public:
    static constexpr char LOG_TAG[] = "dwGPSChannelNode";

    static constexpr auto describeInputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwTime_t, "VIRTUAL_SYNC_TIME"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwGPSFrame, "INPUT_FRAME"_sv, PortBinding::REQUIRED));
    };

    static constexpr auto describeOutputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwSensorNodeProperties, "SENSOR_PROPERTIES"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwGPSFrame, "PROCESSED_DATA"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwSensorTsAndID, "TIMESTAMP"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwTime_t, "NEXT_TIMESTAMP"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwGPSFrameNew, "FRAME_NEW"_sv, PortBinding::REQUIRED));
    };

    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PROCESSED_OUTPUT"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    dwSensorType getSensorType() const
    {
        return DW_SENSOR_GPS;
    }

    static std::unique_ptr<dwGPSChannelNode> create(ParameterProvider& provider);

    static constexpr auto describeParameters()
    {
        return describeConstructorArguments<const char*, dwRigHandle_t, dwSALHandle_t, dwContextHandle_t>(
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

    dwGPSChannelNode(const char* sensorName, dwConstRigHandle_t rigHandle, dwSALHandle_t sal, dwContextHandle_t ctx);
    dwGPSChannelNode(GPSNodeChannelParams& params, const char* sensorName, dwConstRigHandle_t rigHandle, dwSALHandle_t sal,
                     dwContextHandle_t ctx);
    dwGPSChannelNode(const dwSensorParams& params, dwSALHandle_t sal, dwContextHandle_t ctx);
};
}
}
#endif // DW_FRAMEWORK_GPS_CHANNEL_NODE_HPP_

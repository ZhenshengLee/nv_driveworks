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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_RADARPROCESSING_DWRAWDETECTIONCOMPENSATORNODE_DWRAWDETECTIONCOMPENSATORNODE_HPP_
#define DWFRAMEWORK_DWNODES_RADARPROCESSING_DWRAWDETECTIONCOMPENSATORNODE_DWRAWDETECTIONCOMPENSATORNODE_HPP_

#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/node/impl/ExceptionSafeNode.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>

#include <dwframework/dwnodes/common/channelpackets/Radar.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>

#include <dw/core/context/Context.h>

namespace dw
{
namespace framework
{

struct dwRawDetectionCompensatorNodeParams
{
    // These parameters are assigned in describeParameters()
    dwConstRigHandle_t rigHandle;
    bool enableAzimuthCorrection;
    uint32_t sensorIndex;

    //  These parameters are assigned in create()
    uint32_t sensorRigIndex;
};

/**
 * @ingroup dwnodes
 */
class dwRawDetectionCompensatorNode : public ExceptionSafeProcessNode
{
public:
    // TODO(csketch): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwRawDetectionCompensatorNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwCalibratedRadarIntrinsics, "RADAR_INTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwRadarScan, "RADAR_SCAN"_sv, PortBinding::REQUIRED));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwRadarScan, "PROCESSED_RADAR_SCAN"_sv, PortBinding::REQUIRED));
    };

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwRawDetectionCompensatorNode> create(ParameterProvider& provider);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<dwRawDetectionCompensatorNodeParams>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwRawDetectionCompensatorNodeParams::rigHandle),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enableAzimuthCorrection"_sv,
                    &dwRawDetectionCompensatorNodeParams::enableAzimuthCorrection),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "sensorIndex"_sv,
                    &dwRawDetectionCompensatorNodeParams::sensorIndex)),
            describeConstructorArgument<dwContextHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwRawDetectionCompensatorNode(dwRawDetectionCompensatorNodeParams const& param, dwContextHandle_t const ctx);
};
} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_RADARPROCESSING_DWRAWDETECTIONCOMPENSATORNODE_DWRAWDETECTIONCOMPENSATORNODE_HPP_

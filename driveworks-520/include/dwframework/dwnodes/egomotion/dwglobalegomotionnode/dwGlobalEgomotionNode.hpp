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

#ifndef DWFRAMEWORK_DWNODES_EGOMOTION_DWGLOBALEGOMOTIONNODE_DWGLOBALEGOMOTIONNODE_HPP_
#define DWFRAMEWORK_DWNODES_EGOMOTION_DWGLOBALEGOMOTIONNODE_DWGLOBALEGOMOTIONNODE_HPP_

#include <dw/egomotion/base/Egomotion.h>
#include <dw/egomotion/global/GlobalEgomotion.h>
#include <dw/egomotion/global/GlobalEgomotionSerialization.h>
#include <dw/roadcast/base_types/RoadCastPacketTypes.hpp>
#include <dwcgf/channel/Channel.hpp>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/Pass.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/GlobalEgomotionCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/EgomotionState.hpp>
#include <dwframework/dwnodes/common/channelpackets/GPS.hpp>
#include <dwframework/dwnodes/common/channelpackets/GlobalEgomotionCommonTypes.hpp>

namespace dw
{
namespace framework
{

struct dwGlobalEgomotionNodeInitParams
{
    dwConstRigHandle_t rigHandle;
    const char* gpsSensorName;
    uint32_t historySize;

    /// Wait time for nvSciChannel
    dwTime_t nvSciChannelWaitTimeUs{};

    /// Signal presence parameters
    bool sigPresenceGPSCourseAccuracy{};
    bool sigPresenceGPSSpeedAccuracy{};
    bool sigPresenceGPSJamState{};
    bool sigPresenceGPSSpoofState{};
    bool sigPresenceGPSHorizontalAccuracy{};
    bool sigPresenceGPSVerticalAccuracy{};
    bool sigPresenceGPSVerticalVelocity{};
};

/**
 * @brief This node computes the global vehicle state and motion over time using signals from GPS and relative egomotion.
 *
 * Input modalities
 * - GPS
 * - Egomotion state (from any relative egomotion modality)
 *
 * Output signals
 * - Global egomotion state
 *
 * @ingroup dwnodes
 **/
class dwGlobalEgomotionNode : public ExceptionSafeProcessNode, public IContainsPreShutdownAction
{
public:
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwGPSFrame, "GPS_FRAME"_sv),
            DW_DESCRIBE_PORT(dwEgomotionStateHandle_t, "EGOMOTION_STATE"_sv));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwGlobalEgomotionState, "GLOBAL_EGOMOTION"_sv),
            DW_DESCRIBE_PORT(dwGlobalEgomotionResultPayload, "GLOBAL_EGOMOTION_RESULT_PAYLOAD"_sv));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"ADD_GPS"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"ADD_RELATIVE_EGOMOTION"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"SEND_STATE"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwGlobalEgomotionNode> create(ParameterProvider& provider);

    ///////////////////////////////////////////////////////////////////////////////////////
    dwGlobalEgomotionNode(const dwGlobalEgomotionNodeInitParams& params,
                          const dwContextHandle_t ctx);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<dwGlobalEgomotionNodeInitParams>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwGlobalEgomotionNodeInitParams::rigHandle),
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    const char*,
                    semantic_parameter_types::GpsName,
                    &dwGlobalEgomotionNodeInitParams::gpsSensorName),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "historySize"_sv,
                    &dwGlobalEgomotionNodeInitParams::historySize),
                DW_DESCRIBE_PARAMETER_WITH_DEFAULT(
                    dwTime_t,
                    "nvSciChannelWaitTimeUs"_sv,
                    0,
                    &dwGlobalEgomotionNodeInitParams::nvSciChannelWaitTimeUs),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceGPSCourseAccuracy"_sv,
                    &dwGlobalEgomotionNodeInitParams::sigPresenceGPSCourseAccuracy),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceGPSSpeedAccuracy"_sv,
                    &dwGlobalEgomotionNodeInitParams::sigPresenceGPSSpeedAccuracy),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceGPSJamState"_sv,
                    &dwGlobalEgomotionNodeInitParams::sigPresenceGPSJamState),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceGPSSpoofState"_sv,
                    &dwGlobalEgomotionNodeInitParams::sigPresenceGPSSpoofState),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceGPSHorizontalAccuracy"_sv,
                    &dwGlobalEgomotionNodeInitParams::sigPresenceGPSHorizontalAccuracy),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceGPSVerticalAccuracy"_sv,
                    &dwGlobalEgomotionNodeInitParams::sigPresenceGPSVerticalAccuracy),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "sigPresenceGPSVerticalVelocity"_sv,
                    &dwGlobalEgomotionNodeInitParams::sigPresenceGPSVerticalVelocity)),
            describeConstructorArgument<dwContextHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwStatus preShutdown() override;
};
} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_EGOMOTION_DWGLOBALEGOMOTIONNODE_DWGLOBALEGOMOTIONNODE_HPP_

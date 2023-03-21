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

#ifndef DWFRAMEWORK_DWNODES_EGOMOTION_DWGLOBALEGOMOTIONNODE_DWGLOBALEGOMOTIONNODE_HPP_
#define DWFRAMEWORK_DWNODES_EGOMOTION_DWGLOBALEGOMOTIONNODE_DWGLOBALEGOMOTIONNODE_HPP_

#include <dw/egomotion/Egomotion.h>
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
class dwGlobalEgomotionNode : public ExceptionSafeProcessNode, public IAsyncResetable, public IContainsPreShutdownAction
{
public:
    static constexpr auto describeInputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwGPSFrame, "GPS_FRAME"_sv),
            DW_DESCRIBE_PORT(dwEgomotionStateHandle_t, "EGOMOTION_STATE"_sv));
    };
    static constexpr auto describeOutputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwGlobalEgomotionState, "GLOBAL_EGOMOTION"_sv),
            DW_DESCRIBE_PORT(dwGlobalEgomotionResultPayload, "GLOBAL_EGOMOTION_RESULT_PAYLOAD"_sv));
    };
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("ADD_GPS"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("ADD_RELATIVE_EGOMOTION"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("SEND_STATE"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwGlobalEgomotionNode> create(ParameterProvider& provider);

    ///////////////////////////////////////////////////////////////////////////////////////
    dwGlobalEgomotionNode(const dwGlobalEgomotionNodeInitParams& params,
                          const dwContextHandle_t ctx);

    dwStatus setAsyncReset() override
    {
        return Exception::guardWithReturn([&]() {
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
        return Exception::guardWithReturn([&]() {
            auto asyncResetNode = dynamic_cast<IAsyncResetable*>(m_impl.get());
            if (asyncResetNode != nullptr)
            {
                return asyncResetNode->executeAsyncReset();
            }
            return DW_FAILURE;
        },
                                          dw::core::Logger::Verbosity::DEBUG);
    }

    static constexpr auto describeParameters()
    {
        return describeConstructorArguments<dwGlobalEgomotionNodeInitParams, dwContextHandle_t>(
            describeConstructorArgument(
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
                    &dwGlobalEgomotionNodeInitParams::historySize)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwStatus preShutdown() override
    {
        auto* preShutdownNode = dynamic_cast<IContainsPreShutdownAction*>(m_impl.get());
        if (preShutdownNode)
        {
            return preShutdownNode->preShutdown();
        }
        return DW_NOT_SUPPORTED;
    }
};
}
}

#endif // DWFRAMEWORK_DWNODES_EGOMOTION_DWGLOBALEGOMOTIONNODE_DWGLOBALEGOMOTIONNODE_HPP_

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
// SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_DWTRACE_TRACENODE_DWTRACENODE_HPP_
#define DWFRAMEWORK_DWNODES_DWTRACE_TRACENODE_DWTRACENODE_HPP_

#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>

namespace dw
{
namespace framework
{

using dwTraceNodeParams = struct dwTraceNodeParams
{
    // Use this flag when traces are deterministic. When there is uncontrollably large amount of trace(Ex. DW Channel traces),
    // it is good to disable this flag. After disabling this flag traces won't be dropped.
    bool stmControlTracing;
    // DWTrace supports total 32 channels. Following mask allows to enable/disable specific channels.
    // Only Enabled channels allocated buffers for their channel. This value must match channelMask provided to dwTraceCollectorNode.
    uint32_t channelMask;
    // Max trace limit that a channel can handle per epoch.
    uint32_t maxTracesPerChPerEpoch;
};

/**
 * @ingroup dwnodes
 */
class dwTraceNode : public ExceptionSafeProcessNode
{
public:
    // TODO(csketch): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwTraceNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection();
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwTraceNodeData, "TRACE"_sv, PortBinding::REQUIRED));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PASS_PROCESS"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments<dwTraceNodeParams, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "stmControlTracing"_sv,
                    &dwTraceNodeParams::stmControlTracing),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "channelMask"_sv,
                    &dwTraceNodeParams::channelMask),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "maxTracesPerChPerEpoch"_sv,
                    &dwTraceNodeParams::maxTracesPerChPerEpoch)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    static std::unique_ptr<dwTraceNode> create(ParameterProvider& provider);

    dwTraceNode(const dwTraceNodeParams& params, const dwContextHandle_t ctx); // context handle is not required in this node
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_DWTRACE_TRACENODE_DWTRACENODE_HPP_

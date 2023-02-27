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
// SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_TRACECOLLECTOR_NODE_H_
#define DW_FRAMEWORK_TRACECOLLECTOR_NODE_H_

#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
/* Need to include the appropriate ChannelPacketTypes.hpp since port initialization requires
   the parameter_trait overrides. Otherwise, it will be considered as a packet of generic type. */
#include <dwframework/dwnodes/common/ChannelPacketTypes.hpp>

namespace dw
{
namespace framework
{
static constexpr uint32_t MAX_INPUT_TRACE_PORTS = 16; // Keep in-sync with ports description of dwTraceCollectorNode

using dwTraceCollectorNodeParams = struct dwTraceCollectorNodeParams
{
    // Use this flag when traces are deterministic. When there is uncontrollably large amount of trace(Ex. DW Channel traces),
    // it is good to disable this flag. After disabling this flag traces won't be dropped.
    bool stmControlTracing;
    // Enable tracing.
    bool enabled;
    // File path where Trace*.txt will be stored if fileBackend is enabled.
    std::string filePath;
    // DWTrace supports total 32 channels. Following mask allows to enable/disable specific channels.
    // Only Enabled channels allocated buffers for their channel. This value must match channelMask provided to dwTraceNode.
    uint64_t channelMask;
    // Enable filebased backend. For this backend post processing script dwTrace.py is needed to infer results.
    bool fileBackendEnabled;
    // Enable network socket backend. For this backend post processing
    bool networkBackendEnabled;
    // customer's ipAddr
    std::string ipAddr;
    // customer's serverPort
    uint16_t serverPort;
    // Enable NVTx backend.
    bool nvtxBackendEnabled;
    // Global tracing level, any trace which has level greater than this level will be ignored.
    uint32_t tracingLevel;
    // Enable ftrace backend.
    bool ftraceBackendEnabled;
    // Enable mem trace.
    bool memTraceEnabled;
    // Enable full MemUsage.
    bool fullMemUsage;
    // Enable disk io usage(read/write bytes, io delay total) trace. Disk IO operation
    // mainly happens when running application after boot. After that data is cached(Depends upon OS and System memory)
    // For now this info will be dumped for STARTUP channels BEGIN/END, as it is expected that major IO operation happens during program init phase.
    bool diskIOStatsEnabled;
    // Max file size of generated filebackend based dwtrace. After this file limit reached log rotation will start.
    uint32_t maxFileSizeMB;
};

/**
 * @ingroup dwnodes
 */
class dwTraceCollectorNode : public ExceptionSafeProcessNode, public IAsyncResetable
{
public:
    static constexpr char LOG_TAG[] = "dwTraceCollectorNode";

    static constexpr auto describeInputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT_ARRAY(dwTraceNodeData, MAX_INPUT_TRACE_PORTS, "TRACE"_sv));
    }

    static constexpr auto describeOutputPorts()
    {
        return describePortCollection();
    }

    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PASS_PROCESS"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    }

    static constexpr auto describeParameters()
    {
        return describeConstructorArguments<dwTraceCollectorNodeParams, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "stmControlTracing"_sv,
                    &dwTraceCollectorNodeParams::stmControlTracing),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enabled"_sv,
                    &dwTraceCollectorNodeParams::enabled),
                DW_DESCRIBE_PARAMETER(
                    std::string,
                    "filePath"_sv,
                    &dwTraceCollectorNodeParams::filePath),
                DW_DESCRIBE_PARAMETER(
                    uint64_t,
                    "channelMask"_sv,
                    &dwTraceCollectorNodeParams::channelMask),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "fileBackendEnabled"_sv,
                    &dwTraceCollectorNodeParams::fileBackendEnabled),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "networkBackendEnabled"_sv,
                    &dwTraceCollectorNodeParams::networkBackendEnabled),
                DW_DESCRIBE_PARAMETER(
                    std::string,
                    "ipAddr"_sv,
                    &dwTraceCollectorNodeParams::ipAddr),
                DW_DESCRIBE_PARAMETER(
                    uint16_t,
                    "serverPort"_sv,
                    &dwTraceCollectorNodeParams::serverPort),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "nvtxBackendEnabled"_sv,
                    &dwTraceCollectorNodeParams::nvtxBackendEnabled),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "tracingLevel"_sv,
                    &dwTraceCollectorNodeParams::tracingLevel),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "ftraceBackendEnabled"_sv,
                    &dwTraceCollectorNodeParams::ftraceBackendEnabled),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "memTraceEnabled"_sv,
                    &dwTraceCollectorNodeParams::memTraceEnabled),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "fullMemUsage"_sv,
                    &dwTraceCollectorNodeParams::fullMemUsage),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "diskIOStatsEnabled"_sv,
                    &dwTraceCollectorNodeParams::diskIOStatsEnabled),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "maxFileSizeMB"_sv,
                    &dwTraceCollectorNodeParams::maxFileSizeMB)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    static std::unique_ptr<dwTraceCollectorNode> create(ParameterProvider& provider);

    dwTraceCollectorNode(const dwTraceCollectorNodeParams& params, const dwContextHandle_t ctx); // context handle is not required in this node

    dwStatus setAsyncReset() override
    {
        return Exception::guardWithReturn([&]() {
            auto asyncResetNode = dynamic_cast<IAsyncResetable*>(m_impl.get());
            if (asyncResetNode != nullptr)
            {
                return asyncResetNode->setAsyncReset();
            }
            return DW_FAILURE;
        });
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
        });
    }
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_TRACECOLLECTOR_NODE_H_

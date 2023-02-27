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

#ifndef DW_FRAMEWORK_RADAR_DOPPLER_MOTION_NODE_HPP__
#define DW_FRAMEWORK_RADAR_DOPPLER_MOTION_NODE_HPP__

#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/ParameterProvider.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dw/egomotion/radar/DopplerMotionEstimator.h>
/* Need to include the appropriate ChannelPacketTypes.hpp since port initialization requires
   the parameter_trait overrides. Otherwise, it will be considered as a packet of generic type. */
#include <dwframework/dwnodes/common/ChannelPacketTypes.hpp>

namespace dw
{
namespace framework
{

struct dwRadarDopplerMotionNodeParams
{
    bool enable;
    cudaStream_t cudaStream;
};

/**
 * @ingroup dwnodes
 */
class dwRadarDopplerMotionNode : public ExceptionSafeProcessNode
{
public:
    static constexpr char LOG_TAG[] = "dwRadarDopplerMotionNode";

    static constexpr auto describeInputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwRadarScan, "RADAR_SCAN"_sv, PortBinding::REQUIRED));
    };
    static constexpr auto describeOutputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwRadarDopplerMotion, "RADAR_DOPPLER_MOTION"_sv, PortBinding::REQUIRED));
    };

    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PROCESS_RADAR_GPU_ASYNC"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("PROCESS_RADAR_CPU_SYNC"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwRadarDopplerMotionNode> create(ParameterProvider& provider);

    static constexpr auto describeParameters()
    {
        return describeConstructorArguments<dwRadarDopplerMotionNodeParams, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_ABSTRACT_PARAMETER(
                    size_t,
                    "index"_sv),
                DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(
                    bool,
                    "enabled"_sv,
                    9),
                DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(
                    int64_t,
                    "radarDopplerMotionCudaStreamIndices"_sv,
                    9)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwRadarDopplerMotionNode(const dwRadarDopplerMotionNodeParams& param, const dwContextHandle_t ctx);
};
} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_RADAR_DOPPLER_MOTION_NODE_HPP__

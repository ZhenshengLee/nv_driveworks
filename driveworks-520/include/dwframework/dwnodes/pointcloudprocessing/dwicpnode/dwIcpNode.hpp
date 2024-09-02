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

#ifndef DWFRAMEWORK_DWNODES_POINTCLOUDPROCESSING_DWICPNODE_DWICPNODE_HPP_
#define DWFRAMEWORK_DWNODES_POINTCLOUDPROCESSING_DWICPNODE_DWICPNODE_HPP_

#include <dw/sensors/lidar/Lidar.h>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/PointCloudProcessingCommonTypes.hpp>
#include <dwframework/dwnodes/common/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/EgomotionState.hpp>
#include <dwframework/dwnodes/common/channelpackets/PointCloud.hpp>
#include <dwframework/dwnodes/common/channelpackets/PointCloudProcessingCommonTypes.hpp>

namespace dw
{
namespace framework
{

using dwIcpNodeParams = struct dwIcpNodeParams
{
    dwConstRigHandle_t rigHandle;
    bool enable;
    uint32_t sensorIndex;
    uint32_t lidarMaxICPIterations;
    uint32_t accumulatorDownsampleFactor;
    bool lidarICPUsesEgomotion;
    dwLidarProperties lidarProps;
    cudaStream_t cudaStream;
};

/**
 * @ingroup dwnodes
 */
class dwIcpNode : public ExceptionSafeProcessNode, public IChannelsConnectedListener
{
public:
    // TODO(csketch): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwIcpNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwEgomotionStateHandle_t, "EGOMOTION_STATE_ODO_IMU"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwPointCloud, "DEPTHMAP"_sv));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwLidarPose, "LIDAR_POSE"_sv));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PASS_PROCESS_ICP_GPU_ASYNC"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"PASS_PROCESS_ICP_CPU_SYNC"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<dwIcpNodeParams>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwIcpNodeParams::rigHandle),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    uint32_t,
                    semantic_parameter_types::LidarRigIdx,
                    "lidarIndex"_sv,
                    &dwIcpNodeParams::sensorIndex),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enable"_sv,
                    &dwIcpNodeParams::enable),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "lidarMaxICPIterations"_sv,
                    &dwIcpNodeParams::lidarMaxICPIterations),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "accumulatorDownsampleFactor"_sv,
                    &dwIcpNodeParams::accumulatorDownsampleFactor),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "lidarICPUsesEgomotion"_sv,
                    &dwIcpNodeParams::lidarICPUsesEgomotion),
                DW_DESCRIBE_INDEX_PARAMETER(
                    dwLidarProperties,
                    "lidarIndex"_sv,
                    &dwIcpNodeParams::lidarProps),
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "cudaStreamIndex"_sv,
                    &dwIcpNodeParams::cudaStream)),
            describeConstructorArgument<dwContextHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(dwContextHandle_t)));
    }

    static std::unique_ptr<dwIcpNode> create(ParameterProvider& provider);

    dwIcpNode(const dwIcpNodeParams& param, const dwContextHandle_t ctx);

    void onChannelsConnected() override;
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_POINTCLOUDPROCESSING_DWICPNODE_DWICPNODE_HPP_

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

#ifndef DWFRAMEWORK_DWNODES_POINTCLOUDPROCESSING_DWICPNODE_DWICPNODE_HPP_
#define DWFRAMEWORK_DWNODES_POINTCLOUDPROCESSING_DWICPNODE_DWICPNODE_HPP_

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
    dwRigHandle_t rigHandle;
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
class dwIcpNode : public ExceptionSafeProcessNode
{
public:
    static constexpr char LOG_TAG[] = "dwIcpNode";

    static constexpr auto describeInputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwEgomotionStateHandle_t, "EGOMOTION_STATE_ODO_IMU"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwPointCloud, "LIDAR_POINT_CLOUD"_sv));
    }

    static constexpr auto describeOutputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwLidarPose, "LIDAR_POSE"_sv));
    }

    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PASS_PROCESS_ICP_GPU_ASYNC"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("PASS_PROCESS_ICP_CPU_SYNC"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    }

    static constexpr auto describeParameters()
    {
        return describeConstructorArguments<dwIcpNodeParams, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwRigHandle_t,
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
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(dwContextHandle_t)));
    }

    static std::unique_ptr<dwIcpNode> create(ParameterProvider& provider);

    dwIcpNode(const dwIcpNodeParams& param, const dwContextHandle_t ctx);
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_POINTCLOUDPROCESSING_DWICPNODE_DWICPNODE_HPP_

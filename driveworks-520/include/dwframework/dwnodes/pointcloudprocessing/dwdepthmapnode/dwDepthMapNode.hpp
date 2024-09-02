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
// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_POINTCLOUDPROCESSING_DWDEPTHMAPNODE_DWDEPTHMAPNODE_HPP_
#define DWFRAMEWORK_DWNODES_POINTCLOUDPROCESSING_DWDEPTHMAPNODE_DWDEPTHMAPNODE_HPP_

#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/PointCloudProcessingCommonTypes.hpp>
#include <dwframework/dwnodes/common/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/DwframeworkTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/EgomotionState.hpp>
#include <dwframework/dwnodes/common/channelpackets/LidarPointCloud.hpp>
#include <dwframework/dwnodes/common/channelpackets/PointCloud.hpp>
#include <dwframework/dwnodes/common/channelpackets/PointCloudProcessingCommonTypes.hpp>

namespace dw
{
namespace framework
{

using dwDepthMapNodeParams = struct dwDepthMapNodeParams
{
    bool enable;
    dwLidarProperties lidarProps;
    cudaStream_t cudaStream;
};

/**
 * @ingroup dwnodes
 */
class dwDepthMapNode : public ExceptionSafeProcessNode
{
public:
    static constexpr char LOG_TAG[]{"dwDepthMapNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwLidarPointCloud, "LIDAR_POINT_CLOUD"_sv, PortBinding::REQUIRED));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwPointCloud, "DEPTHMAP"_sv),
            DW_DESCRIBE_PORT(dwVector2ui, "SWEEP_SIZE"_sv));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PASS_PROCESS_DEPTHMAP_GPU_ASYNC"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"PASS_PROCESS_DEPTHMAP_CPU_SYNC"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<dwDepthMapNodeParams>(
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enable"_sv,
                    &dwDepthMapNodeParams::enable),
                DW_DESCRIBE_INDEX_PARAMETER(
                    dwLidarProperties,
                    "lidarIndex"_sv,
                    &dwDepthMapNodeParams::lidarProps),
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "cudaStreamIndex"_sv,
                    &dwDepthMapNodeParams::cudaStream)),
            describeConstructorArgument<dwContextHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(dwContextHandle_t)));
    }

    static std::unique_ptr<dwDepthMapNode> create(ParameterProvider& provider);

    dwDepthMapNode(const dwDepthMapNodeParams& param, const dwContextHandle_t ctx);
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_POINTCLOUDPROCESSING_DWDEPTHMAPNODE_DWDEPTHMAPNODE_HPP_

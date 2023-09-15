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

#ifndef DWFRAMEWORK_DWNODES_POINTCLOUDPROCESSING_DWACCUMULATORNODE_DWPOINTCLOUDACCUMULATORNODE_HPP_
#define DWFRAMEWORK_DWNODES_POINTCLOUDPROCESSING_DWACCUMULATORNODE_DWPOINTCLOUDACCUMULATORNODE_HPP_

#include <dwshared/dwfoundation/dw/core/logger/Logger.hpp>
#include <dw/pointcloudprocessing/assembler/PointCloudAssembler.h>
#include <dw/pointcloudprocessing/motioncompensator/MotionCompensator.h>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/EgomotionState.hpp>
#include <dwframework/dwnodes/common/channelpackets/DwframeworkTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/PointCloud.hpp>
#include <dwframework/dwnodes/common/channelpackets/LidarPointCloud.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>

namespace dw
{
namespace framework
{

struct dwAccumulatorNodeParams
{
    dwPointCloudAssemblerParams assmParams;
    dwMotionCompensatorParams compParams;
    dwLidarProperties lidarProperties;
    dwEgomotionParameters egomotionParams;
    bool throttlePipeline;
    bool enableCuda;
    cudaStream_t cudaStream;
};

/**
 * @ingroup dwnodes
 */
class dwPointCloudAccumulatorNode : public ExceptionSafeProcessNode
{
public:
    // TODO(csketch): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwPointCloudAccumulatorNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwLidarPacketsArray, "LIDAR_PACKETS_ARRAY"_sv),
            DW_DESCRIBE_PORT(dwEgomotionStateHandle_t, "EGOMOTION_STATE"_sv),
            DW_DESCRIBE_PORT(dwSensorNodeProperties, "LIDAR_EXTRINSICS"_sv));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwLidarPointCloud, "LIDAR_POINT_CLOUD_RAW"_sv),
            DW_DESCRIBE_PORT(dwLidarPointCloud, "LIDAR_POINT_CLOUD_COMPENSATED"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "TIMESTAMP"_sv),
            DW_DESCRIBE_PORT(bool, "POINT_CLOUD_AVAILABLE"_sv));
    };

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS_ASSM_GPU_ASYNC"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"PROCESS_ASSM_CPU_SYNC"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS_COMP_GPU_ASYNC"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"PROCESS_COMP_CPU_SYNC"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    };

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments<dwAccumulatorNodeParams,
                                            dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enableCuda"_sv,
                    &dwAccumulatorNodeParams::enableCuda),
                DW_DESCRIBE_PARAMETER(
                    dwMotionCompensatorInterpolationStrategy,
                    "interpolationStrategy"_sv,
                    &dwAccumulatorNodeParams::compParams, &dwMotionCompensatorParams::interpolationStrategy),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "outputInRigCoordinates"_sv,
                    &dwAccumulatorNodeParams::compParams, &dwMotionCompensatorParams::outputInRigCoordinates),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    dwTransformation3f,
                    semantic_parameter_types::LidarExtrinsics,
                    "sensorId"_sv,
                    &dwAccumulatorNodeParams::compParams, &dwMotionCompensatorParams::pointCloudToRig),
                DW_DESCRIBE_INDEX_PARAMETER(
                    dwLidarProperties,
                    "sensorId"_sv,
                    &dwAccumulatorNodeParams::lidarProperties),
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    bool,
                    semantic_parameter_types::Virtual,
                    &dwAccumulatorNodeParams::throttlePipeline),
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "streamIndex"_sv,
                    &dwAccumulatorNodeParams::cudaStream)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    };

    static std::unique_ptr<dwPointCloudAccumulatorNode> create(ParameterProvider& provider);

    dwPointCloudAccumulatorNode(const dwAccumulatorNodeParams& paramsPointCloud,
                                const dwContextHandle_t ctx);
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_POINTCLOUDPROCESSING_DWACCUMULATORNODE_DWPOINTCLOUDACCUMULATORNODE_HPP_

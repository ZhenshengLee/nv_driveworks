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

#ifndef DWFRAMEWORK_DWNODES_DWLIDARPOINTCLOUDACCUMULATORNODE_DWLIDARPOINTCLOUDACCUMULATORNODE_HPP_
#define DWFRAMEWORK_DWNODES_DWLIDARPOINTCLOUDACCUMULATORNODE_DWLIDARPOINTCLOUDACCUMULATORNODE_HPP_

#include <dw/core/logger/Logger.hpp>
#include <dw/pointcloudprocessing/accumulator/PointCloudAccumulator.h>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/channelpackets/DwframeworkTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/EgomotionState.hpp>
#include <dwframework/dwnodes/common/channelpackets/PointCloud.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>

namespace dw
{
namespace framework
{

struct dwPointCloudAccumulatorNodeParams
{
    dwPointCloudAccumulatorParams pointCloudAccuParams;
    bool enableEgomotionCompensation;
    dwLidarProperties lidarProperties;
    dwEgomotionParameters egomotionParams;
    bool throttlePipeline;
    cudaStream_t cudaStream;
};

/**
 * @ingroup dwnodes
 */
class dwLidarPointCloudAccumulatorNode : public ExceptionSafeProcessNode
{
public:
    static constexpr char LOG_TAG[] = "dwLidarPointCloudAccumulatorNode";

    static constexpr auto describeInputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwLidarPacketsArray, "LIDAR_PACKETS_ARRAY"_sv),
            DW_DESCRIBE_PORT(dwEgomotionStateHandle_t, "EGOMOTION_STATE"_sv),
            DW_DESCRIBE_PORT(dwSensorNodeProperties, "LIDAR_EXTRINSICS"_sv));
    };
    static constexpr auto describeOutputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwPointCloud, "POINT_CLOUD"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(bool, "POINT_CLOUD_AVAILABLE"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwVector2ui, "LIDAR_ACCUMULATOR_SWEEP_SIZE"_sv, PortBinding::REQUIRED));
    };

    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PROCESS_GPU_ASYNC"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("PROCESS_CPU_SYNC"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    static constexpr auto describeParameters()
    {
        return describeConstructorArguments<dwPointCloudAccumulatorNodeParams,
                                            dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_PARAMETER(
                    dwMemoryType,
                    "memoryType"_sv,
                    &dwPointCloudAccumulatorNodeParams::pointCloudAccuParams, &dwPointCloudAccumulatorParams::memoryType),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "outputFormats"_sv,
                    &dwPointCloudAccumulatorNodeParams::pointCloudAccuParams, &dwPointCloudAccumulatorParams::outputFormats),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "filterWindowSize"_sv,
                    &dwPointCloudAccumulatorNodeParams::pointCloudAccuParams, &dwPointCloudAccumulatorParams::filterWindowSize),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "minAngleDegree"_sv,
                    &dwPointCloudAccumulatorNodeParams::pointCloudAccuParams, &dwPointCloudAccumulatorParams::minAngleDegree),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "maxAngleDegree"_sv,
                    &dwPointCloudAccumulatorNodeParams::pointCloudAccuParams, &dwPointCloudAccumulatorParams::maxAngleDegree),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "minDistanceMeter"_sv,
                    &dwPointCloudAccumulatorNodeParams::pointCloudAccuParams, &dwPointCloudAccumulatorParams::minDistanceMeter),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "maxDistanceMeter"_sv,
                    &dwPointCloudAccumulatorNodeParams::pointCloudAccuParams, &dwPointCloudAccumulatorParams::maxDistanceMeter),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enableMotionCompensation"_sv,
                    &dwPointCloudAccumulatorNodeParams::pointCloudAccuParams, &dwPointCloudAccumulatorParams::enableMotionCompensation),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "organized"_sv,
                    &dwPointCloudAccumulatorNodeParams::pointCloudAccuParams, &dwPointCloudAccumulatorParams::organized),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enableZeroCrossDetection"_sv,
                    &dwPointCloudAccumulatorNodeParams::pointCloudAccuParams, &dwPointCloudAccumulatorParams::enableZeroCrossDetection),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "outputInRigCoordinates"_sv,
                    &dwPointCloudAccumulatorNodeParams::pointCloudAccuParams, &dwPointCloudAccumulatorParams::outputInRigCoordinates),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    dwTransformation3f,
                    semantic_parameter_types::LidarExtrinsics,
                    "sensorId"_sv,
                    &dwPointCloudAccumulatorNodeParams::pointCloudAccuParams, &dwPointCloudAccumulatorParams::sensorTransformation),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enableEgomotionCompensation"_sv,
                    &dwPointCloudAccumulatorNodeParams::enableEgomotionCompensation),
                DW_DESCRIBE_INDEX_PARAMETER(
                    dwLidarProperties,
                    "sensorId"_sv,
                    &dwPointCloudAccumulatorNodeParams::lidarProperties),
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    bool,
                    semantic_parameter_types::Virtual,
                    &dwPointCloudAccumulatorNodeParams::throttlePipeline),
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "streamIndex"_sv,
                    &dwPointCloudAccumulatorNodeParams::cudaStream)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    };

    static std::unique_ptr<dwLidarPointCloudAccumulatorNode> create(ParameterProvider& provider);

    dwLidarPointCloudAccumulatorNode(const dwPointCloudAccumulatorNodeParams& paramsPointCloud,
                                     const dwContextHandle_t ctx);
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_DWLIDARPOINTCLOUDACCUMULATORNODE_DWLIDARPOINTCLOUDACCUMULATORNODE_HPP_

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

#ifndef DWFRAMEWORK_DWNODES_DWLIDARPOINTCLOUDSTITCHERNODE_DWLIDARPOINTCLOUDSTITCHERNODE_HPP_
#define DWFRAMEWORK_DWNODES_DWLIDARPOINTCLOUDSTITCHERNODE_DWLIDARPOINTCLOUDSTITCHERNODE_HPP_

#include <dw/pointcloudprocessing/stitcher/PointCloudStitcher.h>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/PointCloudProcessingCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/EgomotionState.hpp>
#include <dwframework/dwnodes/common/channelpackets/PointCloud.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>

namespace dw
{
namespace framework
{

struct dwLidarPointCloudStitcherNodeParams
{
    bool enabledLidar[MAX_LIDAR_STITCHER_PORT_NUM];
    dwLidarProperties lidarProperties[MAX_LIDAR_STITCHER_PORT_NUM];
    dwTransformation3f lidar2RigDefault[MAX_LIDAR_STITCHER_PORT_NUM];
    dwMemoryType memoryType;
    dwPointCloudFormat outputFormat;

    cudaStream_t stream;
};

template <>
struct EnumDescription<dwPointCloudFormat>
{
    static constexpr auto get()
    {
        return describeEnumeratorCollection<dwPointCloudFormat>(
            DW_DESCRIBE_C_ENUMERATOR(DW_POINTCLOUD_FORMAT_XYZI),
            DW_DESCRIBE_C_ENUMERATOR(DW_POINTCLOUD_FORMAT_RTHI));
    }
};

template <>
struct EnumDescription<dwMemoryType>
{
    static constexpr auto get()
    {
        return describeEnumeratorCollection<dwMemoryType>(
            DW_DESCRIBE_C_ENUMERATOR(DW_MEMORY_TYPE_CUDA),
            DW_DESCRIBE_C_ENUMERATOR(DW_MEMORY_TYPE_CPU),
            DW_DESCRIBE_C_ENUMERATOR(DW_MEMORY_TYPE_PINNED));
    }
};

/**
 * @ingroup dwnodes
 */
class dwLidarPointCloudStitcherNode : public ExceptionSafeProcessNode
{
public:
    static constexpr char LOG_TAG[] = "dwLidarPointCloudStitcherNode";

    static constexpr auto describeInputPorts()
    {
        using namespace dw::framework;
        return describePortCollection(
            DW_DESCRIBE_PORT_ARRAY(dwPointCloud, MAX_LIDAR_STITCHER_PORT_NUM, "LIDAR_PACKETS"_sv),
            DW_DESCRIBE_PORT_ARRAY(dwSensorNodeProperties, MAX_LIDAR_STITCHER_PORT_NUM, "LIDAR_EXTRINSICS"_sv),
            DW_DESCRIBE_PORT(dwEgomotionStateHandle_t, "EGOMOTION_STATE"_sv));
    };

    static constexpr auto describeOutputPorts()
    {
        using namespace dw::framework;
        return describePortCollection(
            DW_DESCRIBE_PORT(dwPointCloud, "POINT_CLOUD"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwTime_t, "TARGET_TIMESTAMP"_sv, PortBinding::OPTIONAL));
    };

    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PROCESS_GPU"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("PROCESS_CPU"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwLidarPointCloudStitcherNode> create(ParameterProvider& provider);

    static constexpr auto describeParameters()
    {
        return describeConstructorArguments<dwLidarPointCloudStitcherNodeParams, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER_WITH_SEMANTIC(
                    bool,
                    semantic_parameter_types::LidarEnabled,
                    MAX_LIDAR_STITCHER_PORT_NUM,
                    &dwLidarPointCloudStitcherNodeParams::enabledLidar),
                DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER(
                    dwLidarProperties,
                    MAX_LIDAR_STITCHER_PORT_NUM,
                    &dwLidarPointCloudStitcherNodeParams::lidarProperties),
                DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER_WITH_SEMANTIC(
                    dwTransformation3f,
                    semantic_parameter_types::LidarExtrinsics,
                    MAX_LIDAR_STITCHER_PORT_NUM,
                    &dwLidarPointCloudStitcherNodeParams::lidar2RigDefault),
                DW_DESCRIBE_PARAMETER(
                    dwMemoryType,
                    "memoryType"_sv,
                    &dwLidarPointCloudStitcherNodeParams::memoryType),
                DW_DESCRIBE_PARAMETER(
                    dwPointCloudFormat,
                    "outputFormat"_sv,
                    &dwLidarPointCloudStitcherNodeParams::outputFormat),
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "stream"_sv,
                    &dwLidarPointCloudStitcherNodeParams::stream)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwLidarPointCloudStitcherNode(const dwLidarPointCloudStitcherNodeParams&,
                                  dwContextHandle_t ctx);
};
}
}

#endif // DWFRAMEWORK_DWNODES_DWLIDARPOINTCLOUDSTITCHERNODE_DWLIDARPOINTCLOUDSTITCHERNODE_HPP_

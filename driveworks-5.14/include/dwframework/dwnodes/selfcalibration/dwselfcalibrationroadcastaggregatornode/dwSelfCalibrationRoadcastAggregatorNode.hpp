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

#ifndef DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONROADCASTAGGREGATORNODE_DWSELFCALIBRATIONROADCASTAGGREGATORNODE_HPP_
#define DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONROADCASTAGGREGATORNODE_DWSELFCALIBRATIONROADCASTAGGREGATORNODE_HPP_

#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/node/impl/ExceptionSafeNode.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwframework/dwnodes/common/DwRoadCastCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/DwRoadCastCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/selfcalibration/dwselfcalibrationroadcastaggregatornode/CalibrationSummary.h>
#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>
#include <dw/core/context/Context.h>
#include <dw/rig/Rig.h>

#include <cstdint>

namespace dw
{
namespace framework
{

struct dwSelfCalibrationRoadcastAggregatorNodeParam
{
    dwConstRigHandle_t rigHandle;

    size_t cameraSensorIndices[SELF_CALIBRATION_NODE_MAX_CAMERAS];
    size_t radarSensorIndices[SELF_CALIBRATION_NODE_MAX_RADARS];
    size_t lidarSensorIndices[SELF_CALIBRATION_NODE_MAX_LIDARS];

    std::uint32_t imuSensorRigIndices[SELF_CALIBRATION_NODE_MAX_IMUS];
    std::uint32_t cameraSensorRigIndices[SELF_CALIBRATION_NODE_MAX_CAMERAS];
    std::uint32_t radarSensorRigIndices[SELF_CALIBRATION_NODE_MAX_RADARS];
    std::uint32_t lidarSensorRigIndices[SELF_CALIBRATION_NODE_MAX_LIDARS];
};

/**
 * @ingroup dwnodes
 */
class dwSelfCalibrationRoadcastAggregatorNode : public ExceptionSafeProcessNode
{
public:
    // TODO(csketch): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwSelfCalibrationRoadcastAggregatorNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT_ARRAY(dwCalibratedExtrinsics, SELF_CALIBRATION_NODE_MAX_IMUS, "IMU_EXTRINSICS"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT_ARRAY(dwCalibratedExtrinsics, SELF_CALIBRATION_NODE_MAX_CAMERAS, "CAMERA_EXTRINSICS"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT_ARRAY(dwCalibratedExtrinsics, SELF_CALIBRATION_NODE_MAX_RADARS, "RADAR_EXTRINSICS"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT_ARRAY(dwCalibratedExtrinsics, SELF_CALIBRATION_NODE_MAX_LIDARS, "LIDAR_EXTRINSICS"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT(dwCalibratedWheelRadii, "WHEEL_RADII"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT(dwCalibratedSteeringProperties, "FRONT_STEERING_OFFSET"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT_ARRAY(dwCalibratedIMUIntrinsics, SELF_CALIBRATION_NODE_MAX_IMUS, "IMU_INTRINSICS"_sv, PortBinding::OPTIONAL));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwRoadCastNodeCalibrationDataArray, "ROADCAST_DATA"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwRoadCastNodeCalibrationWheelRadiiData, "ROADCAST_WHEEL_RADII_DATA"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwRoadCastNodeCalibrationIMUIntrinsicsArray, "ROADCAST_IMU_INTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwRoadCastNodeCalibrationSummary, "ROADCAST_CALIBRATION_SUMMARY"_sv, PortBinding::REQUIRED));
    };

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwSelfCalibrationRoadcastAggregatorNode> create(ParameterProvider& provider);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments<dwSelfCalibrationRoadcastAggregatorNodeParam, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwSelfCalibrationRoadcastAggregatorNodeParam::rigHandle),
                DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER_WITH_SEMANTIC(
                    std::uint32_t,
                    semantic_parameter_types::ImuRigIdx,
                    framework::SELF_CALIBRATION_NODE_MAX_IMUS,
                    &dwSelfCalibrationRoadcastAggregatorNodeParam::imuSensorRigIndices),
                DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER_WITH_SEMANTIC(
                    std::uint32_t,
                    semantic_parameter_types::CameraRigIdx,
                    framework::SELF_CALIBRATION_NODE_MAX_CAMERAS,
                    &dwSelfCalibrationRoadcastAggregatorNodeParam::cameraSensorRigIndices),
                DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER_WITH_SEMANTIC(
                    std::uint32_t,
                    semantic_parameter_types::RadarRigIdx,
                    framework::SELF_CALIBRATION_NODE_MAX_RADARS,
                    &dwSelfCalibrationRoadcastAggregatorNodeParam::radarSensorRigIndices),
                DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER_WITH_SEMANTIC(
                    std::uint32_t,
                    semantic_parameter_types::LidarRigIdx,
                    framework::SELF_CALIBRATION_NODE_MAX_LIDARS,
                    &dwSelfCalibrationRoadcastAggregatorNodeParam::lidarSensorRigIndices),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    size_t,
                    "cameraSensorIndices"_sv,
                    framework::SELF_CALIBRATION_NODE_MAX_CAMERAS,
                    &dwSelfCalibrationRoadcastAggregatorNodeParam::cameraSensorIndices),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    size_t,
                    "radarSensorIndices"_sv,
                    framework::SELF_CALIBRATION_NODE_MAX_RADARS,
                    &dwSelfCalibrationRoadcastAggregatorNodeParam::radarSensorIndices),
                DW_DESCRIBE_ARRAY_PARAMETER(
                    size_t,
                    "lidarSensorIndices"_sv,
                    framework::SELF_CALIBRATION_NODE_MAX_LIDARS,
                    &dwSelfCalibrationRoadcastAggregatorNodeParam::lidarSensorIndices)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwSelfCalibrationRoadcastAggregatorNode(dwSelfCalibrationRoadcastAggregatorNodeParam const& param, dwContextHandle_t const /*ctx*/);
};
} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONROADCASTAGGREGATORNODE_DWSELFCALIBRATIONROADCASTAGGREGATORNODE_HPP_

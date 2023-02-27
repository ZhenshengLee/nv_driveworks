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

#ifndef _DW_RIG_NODE_HPP_
#define _DW_RIG_NODE_HPP_

#include "dwRigNode_errorid.h"

#include <dwcgf/node/Node.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
/* Need to include the appropriate ChannelPacketTypes.hpp since port initialization requires
   the parameter_trait overrides. Otherwise, it will be considered as a packet of generic type. */
#include <dwframework/dwnodes/common/ChannelPacketTypes.hpp>

#include <dw/rig/Rig.h>
#include <dwcgf/node/impl/ExceptionSafeNode.hpp>
#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>

namespace dw
{
namespace framework
{
// TODO(dwplc): FP -- This is used in dwRigNodeImpl and is not just assigned/used once.
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr uint8_t RIG_NODE_MAX_IMUS = 1U;

using dwRigNodeParams = struct dwRigNodeParams
{
    // TODO(hongwang): the serialization will be used later
    bool serialization;

    dwRigHandle_t rigHandle;

    /** Output path where full serialized rig will be written */
    char8_t const* rigOutputFileName;

    /** Output path where calibration overlay will be written */
    char8_t const* calibrationOutputFileName;

    /** Input/Output path of persistent calibration overlay */
    char8_t const* calibrationOverlayFileName;

    // sensor type index
    bool imuSensorEnabled;
    std::uint32_t imuSensorRigIndices;
    bool cameraSensorEnabled[framework::SELF_CALIBRATION_NODE_MAX_CAMERAS];
    std::uint32_t cameraSensorRigIndices[framework::SELF_CALIBRATION_NODE_MAX_CAMERAS];
    bool radarSensorEnabled[framework::SELF_CALIBRATION_NODE_MAX_RADARS];
    std::uint32_t radarSensorRigIndices[framework::SELF_CALIBRATION_NODE_MAX_RADARS];
    bool lidarSensorEnabled[framework::SELF_CALIBRATION_NODE_MAX_LIDARS];
    std::uint32_t lidarSensorRigIndices[framework::SELF_CALIBRATION_NODE_MAX_LIDARS];
};

/**
 * @brief dwRigNode receives extrinsic calibrations from sensors in the system, and maintains that state to update the vehicle's rig file on program exit.
 *
 * @ingroup dwnodes
 */
class dwRigNode : public ExceptionSafeProcessNode, public IContainsPreShutdownAction
{
public:
    // TODO(dwplc): RFD -- This function returns auto as it is a wrapper for a template function.
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Pending: TID-1984
    static constexpr auto describeInputPorts()
    {
        // TODO(dwplc): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "IMU_EXTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT_ARRAY(dwCalibratedExtrinsics, framework::SELF_CALIBRATION_NODE_MAX_CAMERAS, "CAMERA_EXTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT_ARRAY(dwCalibratedExtrinsics, framework::SELF_CALIBRATION_NODE_MAX_RADARS, "RADAR_EXTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT_ARRAY(dwCalibratedExtrinsics, framework::SELF_CALIBRATION_NODE_MAX_LIDARS, "LIDAR_EXTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwCalibratedWheelRadii, "WHEEL_RADII"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwCalibratedSteeringProperties, "FRONT_STEERING_OFFSET"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwCalibratedIMUIntrinsics, "IMU_INTRINSICS"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT_ARRAY(dwCameraIntrinsics, framework::SELF_CALIBRATION_NODE_MAX_CAMERAS, "CAMERA_INTRINSICS"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT(bool, "STORE"_sv, PortBinding::OPTIONAL));
    };
    // TODO(dwplc): RFD -- This function returns auto as it is a wrapper for a template function.
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Pending: TID-1984
    static constexpr auto describeOutputPorts()
    {
        return describePortCollection();
    };
    // TODO(dwplc): RFD -- This function returns auto as it is a wrapper for a template function.
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Pending: TID-1984
    static constexpr auto describePasses()
    {
        // TODO(dwplc): RFD -- user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PROCESS"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwRigNode> create(ParameterProvider const& provider);

    // TODO(dwplc): RFD -- This function returns auto as it is a wrapper for a template function.
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Pending: TID-1984
    static constexpr auto describeParameters()
    {
        // TODO(dwplc): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments<dwRigNodeParams, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "serialization"_sv,
                    &dwRigNodeParams::serialization),
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwRigHandle_t,
                    &dwRigNodeParams::rigHandle),
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    char8_t const*,
                    semantic_parameter_types::RigOutputFileName,
                    &dwRigNodeParams::rigOutputFileName),
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    char8_t const*,
                    semantic_parameter_types::CalibrationOutputFileName,
                    &dwRigNodeParams::calibrationOutputFileName),
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    char8_t const*,
                    semantic_parameter_types::CalibrationOverlayFileName,
                    &dwRigNodeParams::calibrationOverlayFileName),
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    bool,
                    semantic_parameter_types::ImuEnabled,
                    &dwRigNodeParams::imuSensorEnabled),
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    uint32_t,
                    semantic_parameter_types::ImuRigIdx,
                    &dwRigNodeParams::imuSensorRigIndices),
                DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER_WITH_SEMANTIC(
                    bool,
                    semantic_parameter_types::CameraEnabled,
                    framework::SELF_CALIBRATION_NODE_MAX_CAMERAS,
                    &dwRigNodeParams::cameraSensorEnabled),
                DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER_WITH_SEMANTIC(
                    uint32_t,
                    semantic_parameter_types::CameraRigIdx,
                    framework::SELF_CALIBRATION_NODE_MAX_CAMERAS,
                    &dwRigNodeParams::cameraSensorRigIndices),
                DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER_WITH_SEMANTIC(
                    bool,
                    semantic_parameter_types::RadarEnabled,
                    framework::SELF_CALIBRATION_NODE_MAX_RADARS,
                    &dwRigNodeParams::radarSensorEnabled),
                DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER_WITH_SEMANTIC(
                    uint32_t,
                    semantic_parameter_types::RadarRigIdx,
                    framework::SELF_CALIBRATION_NODE_MAX_RADARS,
                    &dwRigNodeParams::radarSensorRigIndices),
                DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER_WITH_SEMANTIC(
                    bool,
                    semantic_parameter_types::LidarEnabled,
                    framework::SELF_CALIBRATION_NODE_MAX_LIDARS,
                    &dwRigNodeParams::lidarSensorEnabled),
                DW_DESCRIBE_UNNAMED_ARRAY_PARAMETER_WITH_SEMANTIC(
                    uint32_t,
                    semantic_parameter_types::LidarRigIdx,
                    framework::SELF_CALIBRATION_NODE_MAX_LIDARS,
                    &dwRigNodeParams::lidarSensorRigIndices)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    };

    dwRigNode(dwRigNodeParams const& params, dwContextHandle_t const ctx);
    ~dwRigNode() override             = default;
    dwRigNode(const dwRigNode& other) = default;
    dwRigNode(dwRigNode&& other)      = default;
    dwRigNode& operator=(const dwRigNode&) = default;
    dwRigNode& operator=(dwRigNode&&) = default;

    dwStatus preShutdown() override;
};

} /* namespace framework */
} /* namespace dw */
#endif /* _DW_RIG_NODE_HPP_ */

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
// SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_CAMERA_NODE_HPP_
#define DW_FRAMEWORK_CAMERA_NODE_HPP_

#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwcgf/pass/Pass.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>

#include <dwframework/dwnodes/common/SensorCommonTypes.hpp>
/* Need to include the appropriate ChannelPacketTypes.hpp since port initialization requires
   the parameter_trait overrides. Otherwise, it will be considered as a packet of generic type. */
//#include <dwframework/dwnodes/common/ChannelPacketTypes.hpp>
#include <dwcgf/channel/ChannelPacketTypes.hpp>

#include <dw/sensors/Sensors.h>
#include <dw/sensors/SensorExtras.h>
#include <dw/sensors/camera/Camera.h>

namespace dw
{
namespace framework
{

class dwCameraChannelNodeImpl;

struct dwCameraChannelNodeParams
{
    char8_t const* sensorName;
    cudaStream_t cudaStream;
    bool useEEPROMIntrinsics;
    dwConstRigHandle_t rig;
    dwSALHandle_t sal;
    dw::core::FixedString<32> frameSkipMask;
    dwImageProperties imageProperties;
};

class dwCameraChannelNode : public ExceptionSafeSensorNode
{
public:
    static constexpr char LOG_TAG[] = "dwCameraChannelNode";

    static constexpr auto describeInputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "SENSOR_EXTRINSICS"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "VIRTUAL_SYNC_TIME"_sv),
            DW_DESCRIBE_PORT(dwImageHandle_t, "IMAGE_NATIVE_PROCESSED"_sv));
    };
    static constexpr auto describeOutputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwSensorNodeProperties, "SENSOR_PROPERTIES"_sv),
            DW_DESCRIBE_PORT(dwFThetaCameraConfig, "INTRINSICS"_sv),
            DW_DESCRIBE_PORT(dwImageHandle_t, "IMAGE_NATIVE_PROCESSED"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "IMAGE_TIMESTAMP"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "NEXT_IMAGE_TIMESTAMP"_sv),
            DW_DESCRIBE_PORT(dwSensorTsAndID, "IMAGE_TIMESTAMP_AND_ID"_sv));
    };

    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PROCESSED_OUTPUT"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    dwSensorType getSensorType() const
    {
        return DW_SENSOR_CAMERA;
    }

    static std::unique_ptr<dwCameraChannelNode> create(ParameterProvider& provider);

    static constexpr auto describeParameters()
    {
        return describeConstructorArguments<dwCameraChannelNodeParams, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    const char*,
                    semantic_parameter_types::CameraName,
                    "cameraIndex"_sv,
                    &dwCameraChannelNodeParams::sensorName),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    dwImageProperties,
                    semantic_parameter_types::CameraImagePropertiesNativeProcessed,
                    "cameraIndex"_sv,
                    &dwCameraChannelNodeParams::imageProperties),
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "streamIndex"_sv,
                    &dwCameraChannelNodeParams::cudaStream),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "useEEPROMIntrinsics"_sv,
                    &dwCameraChannelNodeParams::useEEPROMIntrinsics),
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwCameraChannelNodeParams::rig),
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwSALHandle_t,
                    &dwCameraChannelNodeParams::sal)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwCameraChannelNode(dwCameraChannelNodeParams const& params,
                        const dwContextHandle_t ctx);
};
}
}
#endif // DW_FRAMEWORK_CAMERA_NODE_HPP_

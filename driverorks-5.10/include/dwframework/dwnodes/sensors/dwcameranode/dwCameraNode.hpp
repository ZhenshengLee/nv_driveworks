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
// SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_SENSORS_DWCAMERANODE_DWCAMERANODE_HPP_
#define DWFRAMEWORK_DWNODES_SENSORS_DWCAMERANODE_DWCAMERANODE_HPP_

#include <dw/sensors/Sensors.h>
#include <dw/sensors/camera/Camera.h>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/Pass.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/channelpackets/Image.hpp>
#include <dwframework/dwnodes/common/channelpackets/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/Sensors.hpp>

namespace dw
{
namespace framework
{

class dwCameraNodeImpl;

/**
 * @ingroup dwnodes
 */

/**
 * @brief Construction parameters for dwCameraNode.
 */
struct dwCameraNodeParams
{
    char8_t const* sensorName;
    cudaStream_t cudaStream;
    bool useEEPROMIntrinsics;
    dwConstRigHandle_t rig;
    dwSALHandle_t sal;
    FixedString<32> frameSkipMask;
};

class dwCameraNode : public ExceptionSafeSensorNode
{
public:
    static constexpr char LOG_TAG[] = "dwCameraNode";

    static constexpr auto describeInputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "SENSOR_EXTRINSICS"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "VIRTUAL_SYNC_TIME"_sv));
    };
    static constexpr auto describeOutputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwSensorNodeProperties, "SENSOR_PROPERTIES"_sv),
            DW_DESCRIBE_PORT(dwCameraIntrinsics, "INTRINSICS"_sv),
            DW_DESCRIBE_PORT(dwImageHandle_t, "IMAGE_NATIVE_RAW"_sv),
            DW_DESCRIBE_PORT(dwImageHandle_t, "IMAGE_NATIVE_PROCESSED"_sv),
            DW_DESCRIBE_PORT(dwImageHandle_t, "IMAGE_PROCESSED_RGBA"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "IMAGE_TIMESTAMP"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "NEXT_IMAGE_TIMESTAMP"_sv),
            DW_DESCRIBE_PORT(dwSensorTsAndID, "IMAGE_TIMESTAMP_AND_ID"_sv));
    };

    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("RAW_OUTPUT"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("PROCESSED_OUTPUT"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("PROCESSED_RGBA_OUTPUT"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    dwSensorType getSensorType() const
    {
        return DW_SENSOR_CAMERA;
    }

    dwStatus getImageProperties(dwImageProperties* prop, dwCameraOutputType type);
    dwStatus getCameraProperties(dwCameraProperties* prop);

    static std::unique_ptr<dwCameraNode> create(ParameterProvider& provider);

    static constexpr auto describeParameters()
    {
        return describeConstructorArguments<dwCameraNodeParams, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    const char*,
                    semantic_parameter_types::CameraName,
                    "cameraIndex"_sv,
                    &dwCameraNodeParams::sensorName),
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "streamIndex"_sv,
                    &dwCameraNodeParams::cudaStream),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "useEEPROMIntrinsics"_sv,
                    &dwCameraNodeParams::useEEPROMIntrinsics),
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwCameraNodeParams::rig),
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwSALHandle_t,
                    &dwCameraNodeParams::sal),
                DW_DESCRIBE_PARAMETER(
                    dw::core::FixedString<32>,
                    "frameSkipMask"_sv,
                    &dwCameraNodeParams::frameSkipMask)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwCameraNode(dwCameraNodeParams const& params,
                 const dwContextHandle_t ctx);

    // constructors with dwSensorParams
    dwCameraNode(const dwSensorParams& params,
                 cudaStream_t cudaStream,
                 dwSALHandle_t sal,
                 dwContextHandle_t ctx);
    dwCameraNode(const char* sensorName,
                 dwConstRigHandle_t rigHandle,
                 dwSALHandle_t sal,
                 dwContextHandle_t ctx,
                 const FixedString<32>& frameSkipMask);
    dwCameraNode(const dwSensorParams& params,
                 dwSALHandle_t sal,
                 dwContextHandle_t ctx);
};
}
}
#endif // DWFRAMEWORK_DWNODES_SENSORS_DWCAMERANODE_DWCAMERANODE_HPP_

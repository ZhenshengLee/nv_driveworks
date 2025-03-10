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
// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dw/sensors/common/Sensors.h>
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
    bool errorHandlingEnabled;
};

/**
 * @ingroup dwnodes
 */
class dwCameraNode : public ExceptionSafeSensorNode
{
public:
    char8_t const* LOG_TAG;

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "SENSOR_EXTRINSICS"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "VIRTUAL_SYNC_TIME"_sv));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwSensorNodeProperties, "SENSOR_PROPERTIES"_sv),
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "SENSOR_EXTRINSICS"_sv),
            DW_DESCRIBE_PORT(dwCameraIntrinsics, "INTRINSICS_EEPROM"_sv),
            DW_DESCRIBE_PORT(dwImageHandle_t, "IMAGE_NATIVE_RAW"_sv),
            DW_DESCRIBE_PORT(dwImageHandle_t, "IMAGE_NATIVE_PROCESSED"_sv),
            DW_DESCRIBE_PORT(dwImageHandle_t, "IMAGE_PROCESSED_RGBA"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "IMAGE_TIMESTAMP"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "NEXT_IMAGE_TIMESTAMP"_sv),
            DW_DESCRIBE_PORT(dwSensorStats, "SENSOR_STATS"_sv),
            DW_DESCRIBE_PORT(dwSensorTsAndID, "IMAGE_TIMESTAMP_AND_ID"_sv));
    };

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"RAW_OUTPUT"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"PROCESSED_OUTPUT"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"PROCESSED_RGBA_OUTPUT"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    };

    dwSensorType getSensorType() const
    {
        return DW_SENSOR_CAMERA;
    }

    dwStatus getImageProperties(dwImageProperties* prop, dwCameraOutputType type);
    dwStatus getCameraProperties(dwCameraProperties* prop);

    static std::unique_ptr<dwCameraNode> create(ParameterProvider& provider);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<dwCameraNodeParams>(
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
                    &dwCameraNodeParams::frameSkipMask),
                DW_DESCRIBE_PARAMETER_WITH_DEFAULT(
                    bool,
                    "errorHandlingEnabled"_sv,
                    false,
                    &dwCameraNodeParams::errorHandlingEnabled)),
            describeConstructorArgument<dwContextHandle_t>(
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

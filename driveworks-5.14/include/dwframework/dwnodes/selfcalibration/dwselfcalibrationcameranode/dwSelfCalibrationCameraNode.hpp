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

#ifndef DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONCAMERANODE_DWSELFCALIBRATIONCAMERANODE_HPP_
#define DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONCAMERANODE_DWSELFCALIBRATIONCAMERANODE_HPP_

#include <dw/calibration/cameramodel/CameraModel.h>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/channelpackets/EgomotionState.hpp>
#include <dwframework/dwnodes/common/channelpackets/FeatureList.hpp>
#include <dwframework/dwnodes/common/channelpackets/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/VehicleIOValStructures.hpp>
#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>

namespace dw
{
namespace framework
{

/// TODO(lmoltrecht): AVC-2389 Check whether there's a reason to keep two separate parameter structs
struct dwSelfCalibrationNodeCameraParams
{
    uint32_t sensorRigIndex;
    uint32_t trackerMaxFeatureCount;
    uint32_t trackerMaxHistorySize;
    uint32_t calibrationMethod;
    uint32_t calibrationSignals;         // use uint32_t cause enum bitwise-or is not supported by codegen
    float32_t rotationHistogramRangeDeg; // correction range for roll/pitch/yaw histograms in degrees
    dwCameraModelHandle_t cameraHandle;
    dwCameraProperties cameraProps;
};

struct dwSelfCalibrationCameraNodeParam
{
    dwConstRigHandle_t rigHandle;
    /// TODO(lmoltrecht): AVC-2389 Consider changing to uint32_t after node split is finished, or int32_t to enable -1 as value for unused sensors
    size_t sensorIndex;
    bool enableCalibration;
    uint32_t channelFifoSize;
    dwSelfCalibrationNodeCameraParams cameraParams;
    cudaStream_t cudaStream;
};

/**
 * @ingroup dwnodes
 */
class dwSelfCalibrationCameraNode : public ExceptionSafeProcessNode
{
public:
    // TODO(csketch): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwSelfCalibrationCameraNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwEgomotionStateHandle_t, "EGOMOTION_STATE_ODO_IMU"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwVehicleIONonSafetyState, "VEHICLE_IO_NON_SAFETY_STATE"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwFeatureHistoryArray, "CAMERA_FEATURE_DETECTION"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwTime_t, "CAMERA_TIMESTAMP"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwCameraIntrinsics, "CAMERA_INTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(bool, "SERVICE_CALIB_ENABLE"_sv, PortBinding::OPTIONAL));
    };

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "CAMERA_EXTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(bool, "SERVICE_CALIB_ACTIVE"_sv, PortBinding::OPTIONAL));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS_NONCAMERA"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS_CAMERA_GPU_ASYNC"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"PROCESS_CAMERA_CPU_SYNC"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    }

    static std::unique_ptr<dwSelfCalibrationCameraNode> create(ParameterProvider& provider);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments<dwSelfCalibrationCameraNodeParam, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwSelfCalibrationCameraNodeParam::rigHandle),
                DW_DESCRIBE_PARAMETER(
                    size_t,
                    "sensorIndex"_sv,
                    &dwSelfCalibrationCameraNodeParam::sensorIndex),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enableCalibration"_sv,
                    &dwSelfCalibrationCameraNodeParam::enableCalibration),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "cameraCalibrationMethod"_sv,
                    &dwSelfCalibrationCameraNodeParam::cameraParams, &dwSelfCalibrationNodeCameraParams::calibrationMethod),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "cameraCalibrationSignals"_sv,
                    &dwSelfCalibrationCameraNodeParam::cameraParams, &dwSelfCalibrationNodeCameraParams::calibrationSignals),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "rotationHistogramRangeDeg"_sv,
                    &dwSelfCalibrationCameraNodeParam::cameraParams, &dwSelfCalibrationNodeCameraParams::rotationHistogramRangeDeg),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "channelFifoSize"_sv,
                    &dwSelfCalibrationCameraNodeParam::channelFifoSize),
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "cameraSensorStreamIndex"_sv,
                    &dwSelfCalibrationCameraNodeParam::cudaStream)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwSelfCalibrationCameraNode(dwSelfCalibrationCameraNodeParam const& param, dwContextHandle_t const ctx);
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONCAMERANODE_DWSELFCALIBRATIONCAMERANODE_HPP_

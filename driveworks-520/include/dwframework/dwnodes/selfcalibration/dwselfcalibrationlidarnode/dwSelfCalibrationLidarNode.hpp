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

#ifndef DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONLIDARNODE_DWSELFCALIBRATIONLIDARNODE_HPP_
#define DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONLIDARNODE_DWSELFCALIBRATIONLIDARNODE_HPP_

#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/EgomotionState.hpp>
#include <dwframework/dwnodes/common/channelpackets/PointCloud.hpp>
#include <dwframework/dwnodes/common/channelpackets/LidarPointCloud.hpp>
#include <dwframework/dwnodes/common/channelpackets/PointCloudProcessingCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/VehicleIOValStructures.hpp>

namespace dw
{
namespace framework
{

struct dwSelfCalibrationNodeLidarParams
{
    uint32_t sensorRigIndex; // Sensor rig index (e.g. sensor [0-127]), auto-populated by RR2 Loader
    dwLidarProperties lidarProps;
    bool lidarHandEyeUsesEgomotion;
};

struct dwSelfCalibrationLidarNodeParam
{
    dwConstRigHandle_t rigHandle;
    /// TODO(lmoltrecht): AVC-2389 Consider changing to uint32_t after node split is finished
    size_t sensorIndex; // Sensor type index (e.g. Lidar [0-4]), provided as parameter
    bool enableCalibration;
    uint32_t channelFifoSize;
    dwSelfCalibrationNodeLidarParams lidarParams;
    cudaStream_t cudaStream;
};

/**
 * @ingroup dwnodes
 */
class dwSelfCalibrationLidarNode : public ExceptionSafeProcessNode
{
public:
    static constexpr char LOG_TAG[]{"dwSelfCalibrationLidarNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwEgomotionStateHandle_t, "EGOMOTION_STATE_ODO_IMU"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwVehicleIONonSafetyState, "VEHICLE_IO_NON_SAFETY_STATE"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwVehicleIOActuationFeedback, "VEHICLE_IO_ACTUATION_FEEDBACK"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwLidarPointCloud, "LIDAR_POINT_CLOUD_RAW"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwLidarPose, "LIDAR_POSE"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(bool, "SERVICE_CALIB_ENABLE"_sv, PortBinding::OPTIONAL));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "LIDAR_EXTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(bool, "SERVICE_CALIB_ACTIVE"_sv, PortBinding::OPTIONAL));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS_NONLIDAR"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS_LIDAR_GPU_ASYNC"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"PROCESS_LIDAR_CPU_SYNC"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    }

    static std::unique_ptr<dwSelfCalibrationLidarNode> create(ParameterProvider const& provider);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<dwSelfCalibrationLidarNodeParam>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwSelfCalibrationLidarNodeParam::rigHandle),
                DW_DESCRIBE_PARAMETER(
                    size_t,
                    "sensorIndex"_sv,
                    &dwSelfCalibrationLidarNodeParam::sensorIndex),
                DW_DESCRIBE_ABSTRACT_PARAMETER(
                    size_t,
                    "lidarSensorStreamIndex"_sv),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enableCalibration"_sv,
                    &dwSelfCalibrationLidarNodeParam::enableCalibration),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "channelFifoSize"_sv,
                    &dwSelfCalibrationLidarNodeParam::channelFifoSize),
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "lidarSensorStreamIndex"_sv,
                    &dwSelfCalibrationLidarNodeParam::cudaStream)),
            describeConstructorArgument<dwContextHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwSelfCalibrationLidarNode(dwSelfCalibrationLidarNodeParam const& param, dwContextHandle_t const ctx);
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONLIDARNODE_DWSELFCALIBRATIONLIDARNODE_HPP_

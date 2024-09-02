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
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONCAMERAIMUNODE_DWSELFCALIBRATIONCAMERAIMUNODE_HPP_
#define DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONCAMERAIMUNODE_DWSELFCALIBRATIONCAMERAIMUNODE_HPP_

#include <dw/calibration/cameramodel/CameraModel.h>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/channelpackets/EgomotionState.hpp>
#include <dwframework/dwnodes/common/channelpackets/FeatureList.hpp>
#include <dwframework/dwnodes/common/channelpackets/IMU.hpp>
#include <dwframework/dwnodes/common/channelpackets/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/VehicleIOValStructures.hpp>
#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>

namespace dw
{
namespace framework
{

/**
 * @brief Parameters for dwSelfCalibrationCameraBasedIMUNode
 */
struct dwSelfCalibrationCameraBasedIMUNodeParams
{
    dwConstRigHandle_t rigHandle; //!< Rig handle
    /// TODO(lmoltrecht): AVC-2389 Consider changing to uint32_t after node split is finished, or int32_t to enable -1 as value for unused sensors
    size_t cameraSensorIndex;                //!< Sensor type index (e.g. camera [0-12]), provided as parameter.
    uint32_t cameraSensorRigIndex;           //!< Sensor rig index (e.g. sensor [0-127]), auto-populated by RR2 Loader
    uint32_t imuSensorIndex;                 //!< Sensor type index (e.g. imu [0-2]), provided as parameter.
    uint32_t imuSensorRigIndex;              //!< Sensor rig index (e.g. sensor [0-127]), auto-populated by RR2 Loader
    bool enableCalibration;                  //!< Flag to enable and disable calibration
    uint32_t channelFifoSize;                //!< Size of the input channel FIFO queues (must be >0)
    dwConstCameraModelHandle_t cameraHandle; //!< Camera handle
    cudaStream_t cudaStream;                 //!< CUDA stream used for GPU computations
};

/**
 * @brief This node computes the IMU's extrinsic properties (rotation) with respect to the configured nominal
 *
 * @ingroup dwnodes
 */
class dwSelfCalibrationCameraBasedIMUNode : public ExceptionSafeProcessNode, public IChannelsConnectedListener
{
public:
    // TODO(csketch): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwSelfCalibrationCameraBasedIMUNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwEgomotionStateHandle_t, "EGOMOTION_STATE_ODO_IMU"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwVehicleIOASILStateE2EWrapper, "VEHICLE_IO_ASIL_STATE"_sv, PortBinding::OPTIONAL), // unused, consider removing
            DW_DESCRIBE_PORT(dwFeatureHistoryArray, "CAMERA_FEATURE_DETECTION"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwCameraTwoViewTransformation, "CAMERA_TWO_VIEW_TRANSFORMATION"_sv, PortBinding::OPTIONAL), // unused, consider removing
            DW_DESCRIBE_PORT(dwIMUFrame, "IMU_FRAME"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(bool, "SERVICE_CALIB_ENABLE"_sv, PortBinding::OPTIONAL));
    };

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "IMU_EXTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(bool, "SERVICE_CALIB_ACTIVE"_sv, PortBinding::OPTIONAL));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"GET_INPUTS"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS_GPU_ASYNC"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"PROCESS_CPU_SYNC"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    }

    static std::unique_ptr<dwSelfCalibrationCameraBasedIMUNode> create(ParameterProvider const& provider);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<dwSelfCalibrationCameraBasedIMUNodeParams>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwSelfCalibrationCameraBasedIMUNodeParams::rigHandle),
                DW_DESCRIBE_PARAMETER(
                    size_t,
                    "cameraSensorIndex"_sv,
                    &dwSelfCalibrationCameraBasedIMUNodeParams::cameraSensorIndex),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "imuSensorIndex"_sv,
                    &dwSelfCalibrationCameraBasedIMUNodeParams::imuSensorIndex),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enableCalibration"_sv,
                    &dwSelfCalibrationCameraBasedIMUNodeParams::enableCalibration),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "channelFifoSize"_sv,
                    &dwSelfCalibrationCameraBasedIMUNodeParams::channelFifoSize),
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "cameraSensorStreamIndex"_sv,
                    &dwSelfCalibrationCameraBasedIMUNodeParams::cudaStream)),
            describeConstructorArgument<dwContextHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwSelfCalibrationCameraBasedIMUNode(dwSelfCalibrationCameraBasedIMUNodeParams const& param, dwContextHandle_t const ctx);
    void onChannelsConnected() override;
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONCAMERAIMUNODE_DWSELFCALIBRATIONCAMERAIMUNODE_HPP_

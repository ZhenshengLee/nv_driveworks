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

#ifndef DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONCAMERANODE_DWSELFCALIBRATIONCAMERANODE_HPP_
#define DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONCAMERANODE_DWSELFCALIBRATIONCAMERANODE_HPP_

#include <dw/calibration/cameramodel/CameraModel.h>
#include <dw/calibration/engine/common/SelfCalibrationCameraDiagnostics.h>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/channelpackets/EgomotionState.hpp>
#include <dwframework/dwnodes/common/channelpackets/FeatureList.hpp>
#include <dwframework/dwnodes/common/channelpackets/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/VehicleIOValStructures.hpp>
#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/selfcalibration/common/NodeCommonParams.hpp>

namespace dw
{
namespace framework
{

/**
 * @brief Parameters for dwSelfCalibrationCameraNode
 */
struct dwSelfCalibrationCameraNodeParams
{
    dwConstRigHandle_t rigHandle; //!< Rig handle
    // TODO(AVC-2389): Rename to cameraIndex, consider changing to uint32_t
    size_t sensorIndex;                                  //!< Sensor type index (e.g. camera [0-12]), provided as parameter.
    uint32_t sensorRigIndex;                             //!< Sensor rig index (e.g. sensor [0-127]), auto-populated by RR2 Loader
    bool enableCalibration;                              //!< Flag to enable and disable calibration
    uint32_t channelFifoSize;                            //!< Size of the input channel FIFO queues (must be >0)
    CameraCalibrationParameters cameraParams;            //!< Camera calibration parameters
    char8_t const* calibrationOutputFileName;            //!< Output path where calibration overlay will be written, used to determine the location where calibration state is loaded/saved
    dw::core::FixedString<64> calibrationSaveFileSuffix; //!< Output file suffix
    bool loadStateOnStartup;                             //!< Flag to enable loading of stored calibration state on startup
    uint64_t stateWriteTimerPeriodInCycles;              //!< How often to serialize in node pass cycles. 0 means turn off serialization
    uint64_t stateWriteTimerOffsetInCycles;              //!< Offset, in cycles, for the write counter. The intention is to allow to delay the first write (and subsequently other writes) so that a few cycles occur before serialization is attempted must be less than stateWriteTimerPeriodInCycles for serialization to occur.
    cudaStream_t cudaStream;                             //!< CUDA stream used for GPU computations
    // This is a WAR to resolve the issue seen in the development of multi-process RoadRunner. Details of the issue is available: https://nvbugs/3991262
    dwTime_t nvSciChannelWaitTimeUs; //!< Wait time for nvSciChannel
};

/**
 * @brief This node computes the camera's extrinsic properties (rotation+translation) with respect to the configured nominals
 *
 * @ingroup dwnodes
 */
class dwSelfCalibrationCameraNode : public ExceptionSafeProcessNode, public IChannelsConnectedListener
{
public:
    // TODO(csketch): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwSelfCalibrationCameraNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwEgomotionStateHandle_t, "EGOMOTION_STATE_ODO_IMU"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwVehicleIOQMState, "VEHICLE_IO_QM_STATE"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwFeatureHistoryArray, "CAMERA_FEATURE_DETECTION"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwCameraIntrinsics, "CAMERA_INTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(bool, "SERVICE_CALIB_ENABLE"_sv, PortBinding::OPTIONAL));
    };

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "CAMERA_EXTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(bool, "SERVICE_CALIB_ACTIVE"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT(dwSelfCalibrationCameraDiagnostics, "ROADCAST_SELFCALIBRATION_CAMERA_DIAGNOSTICS"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT(dwCameraTwoViewTransformation, "CAMERA_TWO_VIEW_TRANSFORMATION"_sv, PortBinding::OPTIONAL));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS_NONCAMERA"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS_CAMERA_GPU_ASYNC"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"PROCESS_CAMERA_CPU_SYNC"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    }

    static std::unique_ptr<dwSelfCalibrationCameraNode> create(ParameterProvider const& provider);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<dwSelfCalibrationCameraNodeParams>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwSelfCalibrationCameraNodeParams::rigHandle),
                DW_DESCRIBE_PARAMETER(
                    size_t,
                    "sensorIndex"_sv,
                    &dwSelfCalibrationCameraNodeParams::sensorIndex),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enableCalibration"_sv,
                    &dwSelfCalibrationCameraNodeParams::enableCalibration),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "cameraCalibrationMethod"_sv,
                    &dwSelfCalibrationCameraNodeParams::cameraParams, &CameraCalibrationParameters::calibrationMethod),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "cameraCalibrationSignals"_sv,
                    &dwSelfCalibrationCameraNodeParams::cameraParams, &CameraCalibrationParameters::calibrationSignals),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "rotationHistogramRangeDeg"_sv,
                    &dwSelfCalibrationCameraNodeParams::cameraParams, &CameraCalibrationParameters::rotationHistogramRangeDeg),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "channelFifoSize"_sv,
                    &dwSelfCalibrationCameraNodeParams::channelFifoSize),
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    char8_t const*,
                    semantic_parameter_types::CalibrationOverlayFileName,
                    &dwSelfCalibrationCameraNodeParams::calibrationOutputFileName),
                DW_DESCRIBE_PARAMETER(
                    dw::core::FixedString<64>,
                    "calibrationSaveFileSuffix"_sv,
                    &dwSelfCalibrationCameraNodeParams::calibrationSaveFileSuffix),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "loadStateOnStartup"_sv,
                    &dwSelfCalibrationCameraNodeParams::loadStateOnStartup),
                DW_DESCRIBE_PARAMETER_WITH_DEFAULT(uint64_t, "stateWriteTimerPeriodInCycles"_sv, 0, &dwSelfCalibrationCameraNodeParams::stateWriteTimerPeriodInCycles),
                DW_DESCRIBE_PARAMETER_WITH_DEFAULT(uint64_t, "stateWriteTimerOffsetInCycles"_sv, 0, &dwSelfCalibrationCameraNodeParams::stateWriteTimerOffsetInCycles),
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "cameraSensorStreamIndex"_sv,
                    &dwSelfCalibrationCameraNodeParams::cudaStream),
                DW_DESCRIBE_PARAMETER_WITH_DEFAULT(
                    dwTime_t,
                    "nvSciChannelWaitTimeUs"_sv,
                    0,
                    &dwSelfCalibrationCameraNodeParams::nvSciChannelWaitTimeUs)),
            describeConstructorArgument<dwContextHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwSelfCalibrationCameraNode(dwSelfCalibrationCameraNodeParams const& param, dwContextHandle_t const ctx);
    void onChannelsConnected() override;
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONCAMERANODE_DWSELFCALIBRATIONCAMERANODE_HPP_

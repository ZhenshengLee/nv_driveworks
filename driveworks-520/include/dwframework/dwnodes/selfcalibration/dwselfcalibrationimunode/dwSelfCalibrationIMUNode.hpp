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

#ifndef DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONIMUNODE_DWSELFCALIBRATIONIMUNODE_HPP_
#define DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONIMUNODE_DWSELFCALIBRATIONIMUNODE_HPP_

#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/IMU.hpp>
#include <dwframework/dwnodes/common/channelpackets/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/VehicleIOValStructures.hpp>

namespace dw
{
namespace framework
{

/**
 * @brief Parameters for dwSelfCalibrationIMUNode
 */
struct dwSelfCalibrationIMUNodeParams
{
    dwConstRigHandle_t rigHandle;                        //!< Rig handle
    uint32_t sensorIndex;                                //!< Sensor type index (e.g. IMU [0-2]), provided as parameter
    uint32_t sensorRigIndex;                             //!< Sensor rig index (e.g. sensor [0-127]), auto-populated by RR2 Loader
    bool enableCalibration;                              //!< Flag to enable and disable calibration
    uint32_t histogramMaxOutlierFailureCount;            //!< Calibration will fail if histogram accumulates more outliers than configured here
    uint32_t channelFifoSize;                            //!< Size of the input channel FIFO queues (must be >0)
    char8_t const* calibrationOutputFileName;            //!< Output path where calibration overlay will be written, used to determine the location where calibration state is loaded/saved
    dw::core::FixedString<64> calibrationSaveFileSuffix; //!< Output file suffix
    bool loadStateOnStartup;                             //!< Flag to enable loading of stored calibration state on startup
    uint64_t stateWriteTimerPeriodInCycles;              //!< How often to serialize in node pass cycles. 0 means turn off serialization
    uint64_t stateWriteTimerOffsetInCycles;              //!< offset, in cycles, for the write counter. The intention is to allow to delay the first write (and subsequently other writes) so that a few cycles occur before serialization is attempted must be less than stateWriteTimerPeriodInCycles for serialization to occur.
};

/**
 * @brief This node computes the IMUs's extrinsic properties (rotation+translation) with respect to the configured nominals
 *
 * @ingroup dwnodes
 */
class dwSelfCalibrationIMUNode : public ExceptionSafeProcessNode
{
public:
    // TODO(csketch): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwSelfCalibrationIMUNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwVehicleIOASILStateE2EWrapper, "VEHICLE_IO_ASIL_STATE"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwIMUFrame, "IMU_FRAME"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(bool, "SERVICE_CALIB_ENABLE"_sv, PortBinding::OPTIONAL));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "IMU_EXTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(bool, "SERVICE_CALIB_ACTIVE"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT(dwSelfCalibrationImuDiagnostics, "ROADCAST_SELFCALIBRATION_IMU_DIAGNOSTICS"_sv, PortBinding::OPTIONAL));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    }

    static std::unique_ptr<dwSelfCalibrationIMUNode> create(ParameterProvider const& provider);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<dwSelfCalibrationIMUNodeParams>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwSelfCalibrationIMUNodeParams::rigHandle),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "sensorIndex"_sv,
                    &dwSelfCalibrationIMUNodeParams::sensorIndex),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enableCalibration"_sv,
                    &dwSelfCalibrationIMUNodeParams::enableCalibration),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "histogramMaxOutlierFailureCount"_sv,
                    &dwSelfCalibrationIMUNodeParams::histogramMaxOutlierFailureCount),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "channelFifoSize"_sv,
                    &dwSelfCalibrationIMUNodeParams::channelFifoSize),
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    char8_t const*,
                    semantic_parameter_types::CalibrationOverlayFileName,
                    &dwSelfCalibrationIMUNodeParams::calibrationOutputFileName),
                DW_DESCRIBE_PARAMETER(
                    dw::core::FixedString<64>,
                    "calibrationSaveFileSuffix"_sv,
                    &dwSelfCalibrationIMUNodeParams::calibrationSaveFileSuffix),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "loadStateOnStartup"_sv,
                    &dwSelfCalibrationIMUNodeParams::loadStateOnStartup),
                DW_DESCRIBE_PARAMETER_WITH_DEFAULT(uint64_t, "stateWriteTimerPeriodInCycles"_sv, 0, &dwSelfCalibrationIMUNodeParams::stateWriteTimerPeriodInCycles),
                DW_DESCRIBE_PARAMETER_WITH_DEFAULT(uint64_t, "stateWriteTimerOffsetInCycles"_sv, 0, &dwSelfCalibrationIMUNodeParams::stateWriteTimerOffsetInCycles)),
            describeConstructorArgument<dwContextHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwSelfCalibrationIMUNode(dwSelfCalibrationIMUNodeParams const& param, dwContextHandle_t const ctx);
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONIMUNODE_DWSELFCALIBRATIONIMUNODE_HPP_

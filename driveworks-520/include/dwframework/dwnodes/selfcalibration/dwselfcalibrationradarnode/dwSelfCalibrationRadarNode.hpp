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

#ifndef DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONRADARNODE_DWSELFCALIBRATIONRADARNODE_HPP_
#define DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONRADARNODE_DWSELFCALIBRATIONRADARNODE_HPP_

#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>

#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/DopplerMotionEstimator.hpp>
#include <dwframework/dwnodes/common/channelpackets/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/Radar.hpp>
#include <dwframework/dwnodes/common/EnumDescriptions.hpp>

#include <dw/calibration/engine/common/SelfCalibrationRadarDiagnostics.h>
#include <dw/sensors/radar/Radar.h>
#include <dw/sensors/radar/RadarTypes.h>
#include <dw/calibration/engine/radar/RadarParams.h>

namespace dw
{
namespace framework
{

/**
 * @brief Radar calibration parameters
 *
 * @todo AVC-2389 Check if necessary and consider merging with dwSelfCalibrationRadarNodeParams struct
 */
struct RadarCalibrationParameters
{
    bool sensorEnabledGlobally;   //!< Whether the radar sensor (not calibration) is enabled in the system
    uint32_t sensorRigIndex;      //!< Sensor rig index (e.g. sensor [0-127]), auto-populated by RR2 Loader
    dwRadarProperties radarProps; //!< Radar properties
};

/**
 * @brief Parameters for dwSelfCalibrationRadarNode
 */
struct dwSelfCalibrationRadarNodeParams
{
    dwConstRigHandle_t rigHandle; //!< Rig handle
    // TODO(AVC-2389): Rename to radarIndex, consider changing to uint32_t
    size_t sensorIndex; //!< Sensor type index (e.g. radar [0-8]), provided as parameter.
    // TODO(lmoltrecht): AVC-2389 Move into radarParams??
    bool calibrateWheelRadii;                               //!< Flag to enable and disable wheel radii calibration
    dwCalibrationRadarPitchMethod calibratePitch;           //!< Pitch calibration method
    bool enableCalibration;                                 //!< Flag to enable and disable overall calibration
    uint32_t channelFifoSize;                               //!< Size of the input channel FIFO queues (must be >0)
    float32_t supplierYawEstimationDeviationThresholdDeg;   //!< Required precision of yaw calibration. SEH error will be reported if difference between NDAS and supplier yaw calibration is larger than this value.
    float32_t supplierPitchEstimationDeviationThresholdDeg; //!< Required precision of pitch calibration. SEH error will be reported if difference between NDAS and supplier pitch calibration is larger than this value.
    RadarCalibrationParameters radarParams;                 //!< Radar calibration parameters

    /// Output path where calibration overlay will be written
    /// Used to determine the location where calibration state is loaded/saved
    char8_t const* calibrationOutputFileName;

    dw::core::FixedString<64> calibrationSaveFileSuffix;
    bool loadStateOnStartup;

    // periodic state writing
    // how often to serialize in node pass cycles. 0 means turn off serialization
    uint64_t stateWriteTimerPeriodInCycles;
    // offset, in cycles, for the write counter. The intention is to allow to delay the
    // first write (and subsequently other writes) so that a few cycles occur before
    // serialization is attempted
    // must be less than stateWriteTimerPeriodInCycles for serialization to occur.
    uint64_t stateWriteTimerOffsetInCycles;
};

/**
 * @brief This node computes the radar's extrinsic properties (rotation+translation) with respect to the configured nominals
 *
 * @ingroup dwnodes
 */
class dwSelfCalibrationRadarNode : public ExceptionSafeProcessNode
{
public:
    // TODO(csketch): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwSelfCalibrationRadarNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwVehicleIOASILStateE2EWrapper, "VEHICLE_IO_ASIL_STATE"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwRadarDopplerMotion, "RADAR_DOPPLER_MOTION"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwRadarScan, "RADAR_SCAN"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(bool, "SERVICE_CALIB_ENABLE"_sv, PortBinding::OPTIONAL));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "RADAR_EXTRINSICS"_sv, PortBinding::REQUIRED),
            /// TODO(lmoltrecht): AVC-2293 Make REQUIRED when splitboard node is implemented and connected
            DW_DESCRIBE_PORT(dwCalibratedWheelRadii, "WHEEL_RADII"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT(bool, "SERVICE_CALIB_ACTIVE"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT(dwSelfCalibrationRadarDiagnostics, "ROADCAST_SELFCALIBRATION_RADAR_DIAGNOSTICS"_sv, PortBinding::OPTIONAL));
    }

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    }

    static std::unique_ptr<dwSelfCalibrationRadarNode> create(ParameterProvider const& provider);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<dwSelfCalibrationRadarNodeParams>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwSelfCalibrationRadarNodeParams::rigHandle),
                DW_DESCRIBE_PARAMETER(
                    size_t,
                    "radarSensorIndex"_sv,
                    &dwSelfCalibrationRadarNodeParams::sensorIndex),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "calibrateWheelRadii"_sv,
                    &dwSelfCalibrationRadarNodeParams::calibrateWheelRadii),
                DW_DESCRIBE_PARAMETER(
                    dwCalibrationRadarPitchMethod,
                    "calibratePitch"_sv,
                    &dwSelfCalibrationRadarNodeParams::calibratePitch),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enableCalibration"_sv,
                    &dwSelfCalibrationRadarNodeParams::enableCalibration),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "channelFifoSize"_sv,
                    &dwSelfCalibrationRadarNodeParams::channelFifoSize),
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    char8_t const*,
                    semantic_parameter_types::CalibrationOverlayFileName,
                    &dwSelfCalibrationRadarNodeParams::calibrationOutputFileName),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "loadStateOnStartup"_sv,
                    &dwSelfCalibrationRadarNodeParams::loadStateOnStartup),
                DW_DESCRIBE_PARAMETER(
                    dw::core::FixedString<64>,
                    "calibrationSaveFileSuffix"_sv,
                    &dwSelfCalibrationRadarNodeParams::calibrationSaveFileSuffix),
                DW_DESCRIBE_PARAMETER_WITH_DEFAULT(uint64_t, "stateWriteTimerPeriodInCycles"_sv, 0, &dwSelfCalibrationRadarNodeParams::stateWriteTimerPeriodInCycles),
                DW_DESCRIBE_PARAMETER_WITH_DEFAULT(uint64_t, "stateWriteTimerOffsetInCycles"_sv, 0, &dwSelfCalibrationRadarNodeParams::stateWriteTimerOffsetInCycles),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "supplierYawEstimationDeviationThresholdDeg"_sv,
                    &dwSelfCalibrationRadarNodeParams::supplierYawEstimationDeviationThresholdDeg),
                DW_DESCRIBE_PARAMETER(
                    float32_t,
                    "supplierPitchEstimationDeviationThresholdDeg"_sv,
                    &dwSelfCalibrationRadarNodeParams::supplierPitchEstimationDeviationThresholdDeg)),
            describeConstructorArgument<dwContextHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwSelfCalibrationRadarNode(dwSelfCalibrationRadarNodeParams const& param, dwContextHandle_t const ctx);
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONRADARNODE_DWSELFCALIBRATIONRADARNODE_HPP_

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
// SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_DWRADARDOPPLERMOTIONNODE_DWRADARDOPPLERMOTIONNODE_HPP_
#define DWFRAMEWORK_DWNODES_DWRADARDOPPLERMOTIONNODE_DWRADARDOPPLERMOTIONNODE_HPP_

#include <dw/rig/Rig.h>
#include <dwcgf/node/impl/ExceptionSafeNode.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp> // IWYU pragma: keep
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwshared/dwfoundation/dw/core/container/StringView.hpp>
#include <dwframework/dwnodes/common/channelpackets/DopplerMotionEstimator.hpp> // IWYU pragma: keep
#include <dwframework/dwnodes/common/channelpackets/Radar.hpp>                  // IWYU pragma: keep
#include <dwframework/dwnodes/common/channelpackets/VehicleIOValStructures.hpp> // IWYU pragma: keep

namespace dw
{
namespace framework
{

// forward-declare
class ParameterProvider;
struct RadarExtrinsic;

struct dwRadarDopplerMotionNodeParams
{
    bool enable;
    cudaStream_t cudaStream;
    dwConstRigHandle_t rigHandle;
    dwTransformation3f radarExtrinsics;
    dwRadarProperties radarProperties;
    size_t sensorIndex;      // Sensor type index (e.g. radar [0-8]), provided as parameter.
    uint32_t sensorRigIndex; // The parameter is assigned in create()
};

/**
 * @ingroup dwnodes
 */
class dwRadarDopplerMotionNode : public ExceptionSafeProcessNode
{
public:
    // TODO(csketch): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwRadarDopplerMotionNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwVehicleIOASILStateE2EWrapper, "VEHICLE_IO_ASIL_STATE"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwVehicleIOQMState, "VEHICLE_IO_QM_STATE"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwRadarScan, "RADAR_SCAN"_sv, PortBinding::REQUIRED));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwRadarDopplerMotion, "RADAR_DOPPLER_MOTION"_sv, PortBinding::REQUIRED));
    };

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"ADD_VEHICLE_STATE"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS_RADAR_PROCESS_CPU_SYNC"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESS_RADAR_RETRIEVE_CPU_SYNC"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwRadarDopplerMotionNode> create(ParameterProvider& provider);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<dwRadarDopplerMotionNodeParams>(
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enabled"_sv,
                    &dwRadarDopplerMotionNodeParams::enable),
                DW_DESCRIBE_ABSTRACT_PARAMETER(
                    size_t,
                    "radarDopplerMotionCudaStreamIndex"_sv),
                DW_DESCRIBE_PARAMETER(
                    size_t,
                    "radarSensorIndex"_sv,
                    &dwRadarDopplerMotionNodeParams::sensorIndex),
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t,
                    &dwRadarDopplerMotionNodeParams::rigHandle),
                DW_DESCRIBE_INDEX_PARAMETER(
                    dwRadarProperties,
                    "radarSensorIndex"_sv,
                    &dwRadarDopplerMotionNodeParams::radarProperties),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    dwTransformation3f,
                    semantic_parameter_types::RadarExtrinsic,
                    "radarSensorIndex"_sv,
                    &dwRadarDopplerMotionNodeParams::radarExtrinsics)),
            describeConstructorArgument<dwContextHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwRadarDopplerMotionNode(const dwRadarDopplerMotionNodeParams& param, const dwContextHandle_t ctx);
};
} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_DWRADARDOPPLERMOTIONNODE_DWRADARDOPPLERMOTIONNODE_HPP_

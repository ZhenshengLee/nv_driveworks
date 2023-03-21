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
// SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_SENSORS_DWRADARNODE_DWRADARNODE_HPP_
#define DWFRAMEWORK_DWNODES_SENSORS_DWRADARNODE_DWRADARNODE_HPP_

#include <dw/sensors/Sensors.h>
#include <dw/sensors/radar/Radar.h>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/Pass.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/Radar.hpp>
#include <dwframework/dwnodes/common/channelpackets/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/Sensors.hpp>

namespace dw
{
namespace framework
{

/**
 * @ingroup dwnodes
 */
class dwRadarNode : public ExceptionSafeSensorNode
{
public:
    // TODO(dwplc): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation]
    static constexpr char8_t LOG_TAG[] = "dwRadarNode";

    // TODO(dwplc): RFD -- This function returns auto as it is a wrapper for a template function.
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Pending: TID-1984
    static constexpr auto describeInputPorts()
    {
        // TODO(dwplc): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwCalibratedExtrinsics, "SENSOR_EXTRINSICS"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "VIRTUAL_SYNC_TIME"_sv));
    };

    // TODO(dwplc): RFD -- This function returns auto as it is a wrapper for a template function.
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Pending: TID-1984
    static constexpr auto describeOutputPorts()
    {
        // TODO(dwplc): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwSensorNodeProperties, "SENSOR_PROPERTIES"_sv),
            DW_DESCRIBE_PORT(dwRadarScan, "PROCESSED_DATA"_sv),
            DW_DESCRIBE_PORT(dwSensorNodeRawData, "RAW_DATA"_sv),
            DW_DESCRIBE_PORT(dwSensorTsAndID, "TIMESTAMP"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "NEXT_TIMESTAMP"_sv));
    };

    // TODO(dwplc): RFD -- This function returns auto as it is a wrapper for a template function.
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Pending: TID-1984
    static constexpr auto describePasses()
    {
        // TODO(dwplc): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("RAW_OUTPUT"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PROCESSED_OUTPUT"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    dwSensorType getSensorType() const
    {
        return DW_SENSOR_RADAR;
    }

    static std::unique_ptr<dwRadarNode> create(ParameterProvider const& provider);

    // TODO(dwplc): RFD -- This function returns auto as it is a wrapper for a template function.
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Pending: TID-1984
    static constexpr auto describeParameters()
    {
        // TODO(dwplc): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments<char8_t const*, dwRigHandle_t, dwSALHandle_t, dwContextHandle_t, FixedString<32>>(
            describeConstructorArgument(
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    const char*,
                    semantic_parameter_types::RadarName,
                    "sensorId"_sv)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwRigHandle_t)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwSALHandle_t)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)),
            describeConstructorArgument(
                DW_DESCRIBE_PARAMETER(
                    dw::core::FixedString<32>,
                    "frameSkipMask"_sv)));
    }

    dwRadarNode(char8_t const* sensorName, dwConstRigHandle_t rigHandle,
                dwSALHandle_t sal, dwContextHandle_t ctx, const FixedString<32>& frameSkipMask);
    dwRadarNode(dwSensorParams const& params,
                dwSALHandle_t sal, dwContextHandle_t ctx);

    ~dwRadarNode() override               = default;
    dwRadarNode(dwRadarNode const& other) = default;
    dwRadarNode(dwRadarNode&& other)      = default;
    dwRadarNode& operator=(dwRadarNode const&) = default;
    dwRadarNode& operator=(dwRadarNode&&) = default;
};
} // namespace framework
} // namespace dw
#endif // DWFRAMEWORK_DWNODES_SENSORS_DWRADARNODE_DWRADARNODE_HPP_

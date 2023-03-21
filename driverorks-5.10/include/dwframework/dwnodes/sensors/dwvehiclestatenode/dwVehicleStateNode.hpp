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

#ifndef DWFRAMEWORK_DWNODES_SENSORS_DWVEHICLESTATENODE_DWVEHICLESTATENODE_HPP_
#define DWFRAMEWORK_DWNODES_SENSORS_DWVEHICLESTATENODE_DWVEHICLESTATENODE_HPP_

#include "VehicleIOTypes.hpp"
#include <dw/sensors/Sensors.h>
#include <dw/sensors/canbus/CAN.h>
#include <dw/sensors/data/Data.h>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/node/SimpleNodeT.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/Pass.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortCollectionDescriptor.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/Sensors.hpp>
#include <dwframework/dwnodes/common/channelpackets/VehicleIOLegacyStructures.hpp>
#include <dwframework/dwnodes/common/channelpackets/VehicleIOValStructures.hpp>

namespace dw
{
namespace framework
{

struct VehicleStateNodeParams
{
    // Don't technically need these two params, but seems required for codegen to work
    const char* sensorName;
    dwConstRigHandle_t rigHandle;
    FixedString<32> frameSkipMask;

    // Mutually exlusive parameters. Can be made enum if GDL can support conditional statements to enable switchboards
    bool legacyInternal;
    bool legacyExternal;
    bool external;
    bool vioStateRcEnabled;
};

/**
 * @ingroup dwnodes
 */
class dwVehicleStateNode : public ExceptionSafeSensorNode
{
public:
    static constexpr char LOG_TAG[] = "dwVehicleStateNode";

    static constexpr auto describeInputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwVehicleIOCommand, "VEHICLE_IO_COMMAND"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwVehicleIOMiscCommand, "VEHICLE_IO_MISC_COMMAND"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "VIRTUAL_SYNC_TIME"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOState, "VEHICLE_IO_LEGACY_STATE"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOSafetyState, "VEHICLE_IO_SAFETY_STATE"_sv),
            DW_DESCRIBE_PORT(dwVehicleIONonSafetyState, "VEHICLE_IO_NON_SAFETY_STATE"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOActuationFeedback, "VEHICLE_IO_ACTUATION_FEEDBACK"_sv));
    };
    static constexpr auto describeOutputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwSensorNodeRawData, "RAW_DATA"_sv),
            DW_DESCRIBE_PORT(dwSensorNodeProperties, "SENSOR_PROPERTIES"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOState, "PROCESSED_DATA"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwSensorTsAndID, "TIMESTAMP"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "NEXT_TIMESTAMP"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOSafetyState, "VEHICLE_IO_SAFETY_STATE"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwVehicleIONonSafetyState, "VEHICLE_IO_NON_SAFETY_STATE"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwVehicleIOActuationFeedback, "VEHICLE_IO_ACTUATION_FEEDBACK"_sv, PortBinding::REQUIRED));
    };

    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("RAW_OUTPUT"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PROCESSED_OUTPUT"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    dwSensorType getSensorType() const
    {
        return DW_SENSOR_CAN;
    }

    static std::unique_ptr<dwVehicleStateNode> create(ParameterProvider& provider);

    static constexpr auto describeParameters()
    {
        return describeConstructorArguments<VehicleStateNodeParams, const char*, dwRigHandle_t, dwSALHandle_t, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_PARAMETER(bool, "legacyInternal"_sv, &VehicleStateNodeParams::legacyInternal),
                DW_DESCRIBE_PARAMETER(bool, "legacyExternal"_sv, &VehicleStateNodeParams::legacyExternal),
                DW_DESCRIBE_PARAMETER(bool, "external"_sv, &VehicleStateNodeParams::external),
                DW_DESCRIBE_PARAMETER(bool, "vioStateRcEnabled"_sv, &VehicleStateNodeParams::vioStateRcEnabled),
                DW_DESCRIBE_PARAMETER(dw::core::FixedString<32>, "frameSkipMask"_sv, &VehicleStateNodeParams::frameSkipMask)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    const char*,
                    semantic_parameter_types::VehicleSensorName)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwRigHandle_t)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwSALHandle_t)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwVehicleStateNode(const char* sensorName, dwConstRigHandle_t rigHandle,
                       dwSALHandle_t sal, dwContextHandle_t ctx, const FixedString<32>& frameSkipMask);
    dwVehicleStateNode(VehicleStateNodeParams const& params, const char* sensorName, dwConstRigHandle_t rigHandle,
                       dwSALHandle_t sal, dwContextHandle_t ctx);
    dwVehicleStateNode(const dwSensorParams& params,
                       dwSALHandle_t sal, dwContextHandle_t ctx);
};
}
}
#endif // DWFRAMEWORK_DWNODES_SENSORS_DWVEHICLESTATENODE_DWVEHICLESTATENODE_HPP_

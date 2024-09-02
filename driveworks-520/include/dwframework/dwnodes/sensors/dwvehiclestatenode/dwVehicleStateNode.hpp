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
// SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dw/sensors/common/Sensors.h>
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
#include <dwframework/dwnodes/common/VehicleIOTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/Sensors.hpp>
#include <dwframework/dwnodes/common/channelpackets/VehicleIOLegacyStructures.hpp>
#include <dwframework/dwnodes/common/channelpackets/VehicleIOValStructures.hpp>
#include <dwval/base/ValCommonStructs.h>
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
    bool externalChannel;
    bool externalAQChannel;
    bool vioStateRcEnabled;
    bool errorHandlingEnabled;
};

/**
 * @ingroup dwnodes
 */
class dwVehicleStateNode : public ExceptionSafeSensorNode, public IContainsPreShutdownAction
{
public:
    // TODO(csketch): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwVehicleStateNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwVehicleIOCommand, "VEHICLE_IO_COMMAND"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwVehicleIOMiscCommand, "VEHICLE_IO_MISC_COMMAND"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "VIRTUAL_SYNC_TIME"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOState, "VEHICLE_IO_LEGACY_STATE"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOSafetyState, "VEHICLE_IO_SAFETY_STATE_CHANNEL"_sv),
            DW_DESCRIBE_PORT(dwVehicleIONonSafetyState, "VEHICLE_IO_NON_SAFETY_STATE_CHANNEL"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOActuationFeedback, "VEHICLE_IO_ACTUATION_FEEDBACK_CHANNEL"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOASILStateE2EWrapper, "VEHICLE_IO_ASIL_STATE_CHANNEL"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOQMState, "VEHICLE_IO_QM_STATE_CHANNEL"_sv));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwSensorNodeRawData, "RAW_DATA"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOState, "PROCESSED_DATA"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwSensorStats, "SENSOR_STATS"_sv),
            DW_DESCRIBE_PORT(dwSensorTsAndID, "TIMESTAMP"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "NEXT_TIMESTAMP"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOSafetyState, "VEHICLE_IO_SAFETY_STATE_OUT"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwVehicleIONonSafetyState, "VEHICLE_IO_NON_SAFETY_STATE_OUT"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwVehicleIOActuationFeedback, "VEHICLE_IO_ACTUATION_FEEDBACK_OUT"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwVehicleIOSafetyState, "VEHICLE_IO_SAFETY_STATE_EXTERNAL"_sv),
            DW_DESCRIBE_PORT(dwVehicleIONonSafetyState, "VEHICLE_IO_NON_SAFETY_STATE_EXTERNAL"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOActuationFeedback, "VEHICLE_IO_ACTUATION_FEEDBACK_EXTERNAL"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOASILStateE2EWrapper, "VEHICLE_IO_ASIL_STATE_EXTERNAL"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOQMState, "VEHICLE_IO_QM_STATE_EXTERNAL"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOASILStateE2EWrapper, "VEHICLE_IO_ASIL_STATE_OUT"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOQMState, "VEHICLE_IO_QM_STATE_OUT"_sv),
            DW_DESCRIBE_PORT(dwTime_t, "CONTEXT_TIMESTAMP"_sv),
            DW_DESCRIBE_PORT(dwValTimestampMetadata, "VIO_SAFETY_STATE_TIMESTAMP_METADATA"_sv),
            DW_DESCRIBE_PORT(dwValTimestampMetadata, "VIO_NON_SAFETY_STATE_TIMESTAMP_METADATA"_sv),
            DW_DESCRIBE_PORT(dwValTimestampMetadata, "VIO_ACTUATION_FEEDBACK_TIMESTAMP_METADATA"_sv),
            DW_DESCRIBE_PORT(dwValTimestampMetadata, "VIO_QM_STATE_TIMESTAMP_METADATA"_sv),
            DW_DESCRIBE_PORT(dwValTimestampMetadata, "VIO_ASIL_STATE_TIMESTAMP_METADATA"_sv));
    };

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"RAW_OUTPUT"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"PROCESSED_OUTPUT"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    };

    dwSensorType getSensorType() const
    {
        return DW_SENSOR_CAN;
    }

    static std::unique_ptr<dwVehicleStateNode> create(ParameterProvider& provider);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<VehicleStateNodeParams>(
                DW_DESCRIBE_PARAMETER(bool, "legacyInternal"_sv, &VehicleStateNodeParams::legacyInternal),
                DW_DESCRIBE_PARAMETER(bool, "legacyExternal"_sv, &VehicleStateNodeParams::legacyExternal),
                DW_DESCRIBE_PARAMETER(bool, "externalChannel"_sv, &VehicleStateNodeParams::externalChannel),
                DW_DESCRIBE_PARAMETER(bool, "externalAQChannel"_sv, &VehicleStateNodeParams::externalAQChannel),
                DW_DESCRIBE_PARAMETER(bool, "vioStateRcEnabled"_sv, &VehicleStateNodeParams::vioStateRcEnabled),
                DW_DESCRIBE_PARAMETER(dw::core::FixedString<32>, "frameSkipMask"_sv, &VehicleStateNodeParams::frameSkipMask),
                DW_DESCRIBE_PARAMETER_WITH_DEFAULT(bool, "errorHandlingEnabled"_sv, false, &VehicleStateNodeParams::errorHandlingEnabled)),
            describeConstructorArgument<const char*>(
                DW_DESCRIBE_UNNAMED_PARAMETER_WITH_SEMANTIC(
                    const char*,
                    semantic_parameter_types::VehicleSensorName)),
            describeConstructorArgument<dwConstRigHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwConstRigHandle_t)),
            describeConstructorArgument<dwSALHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwSALHandle_t)),
            describeConstructorArgument<dwContextHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwVehicleStateNode(const char* sensorName, dwConstRigHandle_t rigHandle,
                       dwSALHandle_t sal, dwContextHandle_t ctx, const FixedString<32>& frameSkipMask);
    dwVehicleStateNode(VehicleStateNodeParams const& params, const char* sensorName, dwConstRigHandle_t rigHandle,
                       dwSALHandle_t sal, dwContextHandle_t ctx);
    dwVehicleStateNode(const dwSensorParams& params,
                       dwSALHandle_t sal, dwContextHandle_t ctx);

    dwStatus preShutdown() override;
};
} // namespace framework
} // namespace dw
#endif // DWFRAMEWORK_DWNODES_SENSORS_DWVEHICLESTATENODE_DWVEHICLESTATENODE_HPP_

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
// SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_VEHICLE_STATE_DRAINER_HPP_
#define DW_FRAMEWORK_VEHICLE_STATE_DRAINER_HPP_

#include <dwframework/dwnodes/sensors/dwsensornode/dwSensorDrainerTemplate.hpp>
#include <dwframework/dwnodes/sensors/dwvehiclestatenode/impl/dwVehicleStateNodeImpl.hpp>
#include <dwframework/dwnodes/sensors/dwvehiclestatenode/impl/dwVehicleStateChannelNodeImpl.hpp>

#include <dw/sensors/Sensors.h>
#include <dw/core/base/Types.h>

namespace dw
{
namespace framework
{
class dwVehicleStateDrainer : public dwSensorDrainerTemplate<dwVehicleIOState, vio::ReadProcessedData>
{
public:
    dwVehicleStateDrainer(dwSensorDrainerParams params, std::unique_ptr<vio::ReadProcessedData> readProcessedDataFunc, dwSensorHandle_t sensor)
        : dwSensorDrainerTemplate<dwVehicleIOState, vio::ReadProcessedData>(params, std::move(readProcessedDataFunc), sensor)
    {
    }

    dwStatus drainProcessedData(dwVehicleIOState* processedOutput,
                                dwTime_t& timestampOutput,
                                dwTime_t& nextTimestampOutput,
                                dwTime_t virtualSyncTime)
    {
        m_readProcessedDataFunc->clearButtonsPressed();

        dwStatus status = dwSensorDrainerTemplate<dwVehicleIOState, vio::ReadProcessedData>::
            drainProcessedData(processedOutput,
                               timestampOutput,
                               nextTimestampOutput,
                               virtualSyncTime);

        m_readProcessedDataFunc->getButtonsPressed(*processedOutput);

        return status;
    }

    dwStatus replayProcessedData(dwVehicleIOState* processedOutput,
                                 dwTime_t& timestampOutput,
                                 ISensorNode::DataEventReadCallback readCb)
    {
        m_readProcessedDataFunc->clearButtonsPressed();

        dwStatus status = dwSensorDrainerTemplate<dwVehicleIOState, vio::ReadProcessedData>::
            replayProcessedData(processedOutput,
                                timestampOutput,
                                std::move(readCb));

        m_readProcessedDataFunc->getButtonsPressed(*processedOutput);

        return status;
    }
};

class dwVehicleStateChannelDrainer : public dwChannelDrainerTemplate<dwVehicleIOState, vio::ReadProcessedVehicleStateDataFromChannel>
{
    using InputPort = PortInput<dwVehicleIOState>*;

public:
    dwVehicleStateChannelDrainer(dwSensorDrainerParams params, std::unique_ptr<vio::ReadProcessedVehicleStateDataFromChannel> readProcessedDataFunc, InputPort inputPort)
        : dwChannelDrainerTemplate<dwVehicleIOState, vio::ReadProcessedVehicleStateDataFromChannel>(params, std::move(readProcessedDataFunc), inputPort)
    {
    }

    dwStatus drainProcessedData(dwVehicleIOState* processedOutput,
                                dwTime_t& timestampOutput,
                                dwTime_t& nextTimestampOutput,
                                dwTime_t virtualSyncTime)
    {
        m_readProcessedDataFunc->clearButtonsPressed();

        dwStatus status = dwChannelDrainerTemplate<dwVehicleIOState, vio::ReadProcessedVehicleStateDataFromChannel>::
            drainProcessedData(processedOutput,
                               timestampOutput,
                               nextTimestampOutput,
                               virtualSyncTime);

        m_readProcessedDataFunc->getButtonsPressed(*processedOutput);

        return status;
    }
};
}
}

#endif // DW_FRAMEWORK_VEHICLE_STATE_DRAINER_HPP_

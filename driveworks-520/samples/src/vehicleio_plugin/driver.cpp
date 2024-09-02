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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

#include <dw/control/vehicleio/VehicleIO.h>
#include <dw/control/vehicleio/VehicleIOCapabilities.h>
#include <dw/control/vehicleio/plugins/VehicleIODriver.h>
#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/core/signal/SignalStatus.h>
#include <dw/rig/Vehicle.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/canbus/CAN.h>
#include <dw/control/vehicleio/drivers/Driver.h>
#include <dw/sensors/canbus/Interpreter.h>

#include <framework/Checks.hpp>
#include <framework/SamplesDataPath.hpp>

#include "driverConf.hpp"

// exported functions
extern "C" {

static dwCANInterpreterHandle_t gIntp{};
static dwContextHandle_t gSdk = DW_NULL_HANDLE;
static dwVehicleIOASILStateE2EWrapper* gAsilState{};
static dwVehicleIOQMState* gQmState{};
static dwVehicleIOQMCommand gQmCmd{};

static void checkError(dwStatus& result, const dwStatus& status, const char* errorMessage)
{
    if (status != DW_SUCCESS)
        std::cerr << "VehicleIO Plugin Error: " << errorMessage << std::endl;

    if (result != DW_SUCCESS)
        return;
    result = status;
}

static void setSignalValid(dwSignalValidity& validity)
{
    dwSignal_encodeSignalValidity(&validity,
                                  DW_SIGNAL_STATUS_LAST_VALID,
                                  DW_SIGNAL_TIMEOUT_NONE,
                                  DW_SIGNAL_E2E_NO_ERROR);
}

static void parseSteeringReport()
{
    dwStatus status;
    uint32_t num;
    status = dwCANInterpreter_getNumberSignals(&num, gIntp);

    if (status == DW_SUCCESS && num > 0)
    {
        float32_t value    = 0;
        dwTime_t timestamp = 0;
        const char* name;

        for (uint32_t i = 0; i < num; ++i)
        {
            if (dwCANInterpreter_getSignalName(&name, i, gIntp) == DW_SUCCESS)
            {
                if (dwCANInterpreter_getf32(&value, &timestamp, i, gIntp) == DW_SUCCESS)
                {
                    if (0 == strcmp(name, GENERIC_ID_STEERING_WHEEL_ANGLE_REPORT))
                    {
                        if (gAsilState != nullptr)
                        {
                            gAsilState->payload.steeringWheelAngle = value;
                            setSignalValid(gAsilState->payload.validityInfo.steeringWheelAngle);
                        }
                    }
                    else if (0 == strcmp(name, GENERIC_ID_STEERING_WHEEL_TORQUE_REPORT))
                    {
                        if (gAsilState != nullptr)
                        {
                            gAsilState->payload.steeringWheelTorque = value;
                            setSignalValid(gAsilState->payload.validityInfo.steeringWheelTorque);
                        }
                    }
                    else if (0 == strcmp(name, GENERIC_ID_STEERING_WHEEL_SPEED_REPORT))
                    {
                        if (gAsilState != nullptr)
                        {
                            gAsilState->payload.speedESC = std::abs(value);
                            ;
                            setSignalValid(gAsilState->payload.validityInfo.speedESC);
                        }
                    }
                }
            }
        }
    }
}

static dwStatus encodeSteering(dwCANMessage& msgCAN, const dwVehicleIOASILCommand& cmd)
{
    constexpr float32_t MAX_STEERING_ANGLE = 0.459F;
    constexpr float32_t MAX_STEERING_SPEED = 4.363F;

    float32_t swa = dwSignal_checkSignalValidity(gQmCmd.validityInfo.latCtrlSteeringWheelAngleRequest) == DW_SUCCESS ? gQmCmd.latCtrlSteeringWheelAngleRequest : 0.0F;
    float32_t sws = dwSignal_checkSignalValidity(gQmCmd.validityInfo.latCtrlSteeringWheelAngleRateMax) == DW_SUCCESS ? gQmCmd.latCtrlSteeringWheelAngleRateMax : 0.0F;

    if (std::fabs(swa) > MAX_STEERING_ANGLE)
    {
        swa = std::copysign(MAX_STEERING_ANGLE, swa);
    }

    if (std::fabs(sws) > MAX_STEERING_SPEED)
    {
        sws = std::copysign(MAX_STEERING_SPEED, sws);
    }

    dwStatus status;
    dwStatus result = DW_SUCCESS;

    status = dwCANInterpreter_encodef32(swa,
                                        GENERIC_ID_STEERING_WHEEL_ANGLE_COMMAND,
                                        &msgCAN, gIntp);
    checkError(result, status, "encode steering angle failed");

    // assume using steering wheel angle interface
    status = dwCANInterpreter_encodei32(dwSignal_checkSignalValidity(gQmCmd.validityInfo.latCtrlSteeringWheelAngleRequest) == DW_SUCCESS,
                                        GENERIC_ID_STEERING_WHEEL_ANGLE_COMMAND_VALID,
                                        &msgCAN, gIntp);
    checkError(result, status, "encode steering valid failed");

    status = dwCANInterpreter_encodef32(sws,
                                        GENERIC_ID_STEERING_WHEEL_STEER_SPEED,
                                        &msgCAN, gIntp);
    checkError(result, status, "encode steering speed failed");

    bool clearFaults = false;
    if (dwSignal_checkSignalValidity(cmd.validityInfo.latCtrlModeRequest) == DW_SUCCESS &&
        dwSignal_checkSignalValidity(cmd.validityInfo.longCtrlFunctionReq) == DW_SUCCESS)
    {
        clearFaults = cmd.latCtrlModeRequest == DW_VIO_LAT_CTRL_MODE_REQUEST_IDLE &&
                      cmd.longCtrlFunctionReq == DW_VIO_LONG_CTRL_FUNCTION_REQ_IDLE;
    }

    status = dwCANInterpreter_encodei32(clearFaults ? true : false,
                                        GENERIC_ID_STEERING_WHEEL_STEER_CLEAR_FAULT,
                                        &msgCAN, gIntp);
    checkError(result, status, "encode steering clear fault failed");
    return result;
}

//################################################################################################

dwStatus _dwVehicleIODriver_initialize_V3(dwContextHandle_t, dwVehicle const*, dwVehicleIOCapabilities*, char8_t const*, dwVehicleIOASILStateE2EWrapper* asilState, dwVehicleIOQMState* qmState)
{
    dwStatus status;
    dwStatus result = DW_SUCCESS;

    gAsilState = asilState;
    gQmState   = qmState;

    dwContextParameters sdkParams = {};
    sdkParams.skipCudaInit        = true;
    CHECK_DW_ERROR(dwInitialize(&gSdk, DW_VERSION, &sdkParams));

    const std::string dbcPath = dw_samples::SamplesDataPath::get() + "/samples/vehicleio/AutonomousVehicleCANSignals.dbc";

    status = dwCANInterpreter_buildFromDBC(&gIntp, dbcPath.c_str(), gSdk);
    checkError(result, status, "cannot create DBC-based CAN message interpreter");

    return result;
}

//################################################################################################
dwStatus _dwVehicleIODriver_release()
{
    gAsilState = {};
    gQmState   = {};

    dwCANInterpreter_release(gIntp);
    dwRelease(gSdk);
    return DW_SUCCESS;
}

//################################################################################################
dwStatus _dwVehicleIODriver_consumeExt(const dwCANMessage* msg)
{
    dwStatus status;
    dwStatus result = DW_SUCCESS;

    status = dwCANInterpreter_consume(msg, gIntp);
    checkError(result, status, "consume message failed");
    parseSteeringReport();

    return result;
}

dwStatus _dwVehicleIODriver_setDrivingMode(const dwVehicleIODrivingMode mode)
{
    dwStatus ret = DW_SUCCESS;

    switch (mode)
    {
    case DW_VEHICLEIO_DRIVING_LIMITED:
        break;
    case DW_VEHICLEIO_DRIVING_LIMITED_ND:
        break;
    case DW_VEHICLEIO_DRIVING_COLLISION_AVOIDANCE:
        break;
    case DW_VEHICLEIO_DRIVING_NO_SAFETY:
        break;
    case DW_VEHICLEIO_DRIVING_MODE_INVALID:
        ret = DW_NOT_SUPPORTED;
        break;
    default:
        ret = DW_NOT_SUPPORTED;
        break;
    }

    return ret;
}

//################################################################################################
dwStatus _dwVehicleIODriver_sendASILCommand(const dwVehicleIOASILCommandE2EWrapper* cmd,
                                            dwSensorHandle_t sensor)
{
    dwCANMessage msgCAN{};
    dwStatus status;
    dwStatus result = DW_SUCCESS;

    status = dwCANInterpreter_createMessageFromName(&msgCAN, GENERIC_MSG_STEERING_CMD, gIntp);
    checkError(result, status, "create steering message failed");

    status = encodeSteering(msgCAN, cmd->payload);
    checkError(result, status, "encode steering failed");

    status = dwSensorCAN_sendMessage(&msgCAN, 1000000, sensor);
    checkError(result, status, "send steering failed");

    return result;
}

//################################################################################################
dwStatus _dwVehicleIODriver_sendQMCommand(const dwVehicleIOQMCommand* cmd,
                                          dwSensorHandle_t)
{
    gQmCmd = *cmd;
    return DW_SUCCESS;
}

//################################################################################################
dwStatus _dwVehicleIODriver_sendEgomotion(const dwValEgomotion*,
                                          dwSensorHandle_t)
{
    return DW_SUCCESS;
}

//################################################################################################
dwStatus _dwVehicleIODriver_sendSensorCalibration(const dwValSensorCalibration*,
                                                  dwSensorHandle_t)
{
    return DW_SUCCESS;
}

//################################################################################################
dwStatus _dwVehicleIODriver_reset()
{
    return DW_SUCCESS;
}

//################################################################################################
VehicleIOStructType _dwVehicleIODriver_getSupportedVIOStructType()
{
    return VehicleIOStructType::ASIL_QM_VIO;
}

} // extern "C"

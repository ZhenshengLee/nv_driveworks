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
// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <cstring>
#include <string>
#include <cmath>

#include <dw/control/vehicleio/VehicleIO.h>
#include <dw/control/vehicleio/VehicleIOCapabilities.h>
#include <dw/control/vehicleio/plugins/VehicleIODriver.h>
#include <dw/control/vehicleio/drivers/Driver.h>
#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/context/Context.h>
#include <dw/rig/Vehicle.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/canbus/CAN.h>
#include <dw/sensors/canbus/Interpreter.h>

static dwCANInterpreterHandle_t gIntp{};
static dwContextHandle_t gSdk{};
static dwVehicleIOASILStateE2EWrapper* gAsilState{};
static dwVehicleIOQMState* gQmState{};

// -----------------------------------------------------------------------------
// CAN bus codes
// A set of virtual CAN bus codes to simulate virtual car
// -----------------------------------------------------------------------------
static char8_t CAN_CAR_SPEED[]    = "M_SPEED.CAN_CAR_SPEED";
static char8_t CAN_CAR_STEERING[] = "M_STEERING.CAN_CAR_STEERING";

static char8_t DBC[] = R"DBC(
VERSION ""

NS_ :

BS_:

BU_:

BO_ 256 M_STEERING: 8 SAMPLE
 SG_ CAN_CAR_STEERING : 0|32@1- (6.25E-007,0) [-100000|100000] "rad"  SAMPLE

BO_ 512 M_SPEED: 8 SAMPLE
 SG_ CAN_CAR_SPEED : 0|32@1- (1E-005,0) [-100000|100000] "m/s"  SAMPLE

)DBC";

static void setSignalValid(dwSignalValidity& validity)
{
    dwSignal_encodeSignalValidity(&validity,
                                  DW_SIGNAL_STATUS_LAST_VALID,
                                  DW_SIGNAL_TIMEOUT_NONE,
                                  DW_SIGNAL_E2E_NO_ERROR);
}

// exported functions
extern "C" {

//################################################################################################
dwStatus _dwVehicleIODriver_initialize_V3(dwContextHandle_t, dwVehicle const*, dwVehicleIOCapabilities*, char8_t const*, dwVehicleIOASILStateE2EWrapper* asilState, dwVehicleIOQMState* qmState)
{
    // Force a heap allocation which would ensure that linker uses the sample allocator.
    // The allocator ensures that all used types are allocated respecting alignment, which is expected by DriveWorks.
    // Hence this plugin can be loaded by DriveWorks.
    const std::string dummy;

    gAsilState                    = asilState;
    gQmState                      = qmState;
    dwContextParameters sdkParams = {};
    sdkParams.skipCudaInit        = true;
    dwInitialize(&gSdk, DW_VERSION, &sdkParams);

    return dwCANInterpreter_buildFromDBCString(&gIntp, DBC, gSdk);
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
    auto status = dwCANInterpreter_consume(msg, gIntp);
    if (status != DW_SUCCESS)
    {
        return status;
    }

    uint32_t num;
    if (dwCANInterpreter_getNumberSignals(&num, gIntp) == DW_SUCCESS && num > 0)
    {
        float32_t value    = 0;
        dwTime_t timestamp = 0;
        const char8_t* name;

        for (uint32_t i = 0; i < num; ++i)
        {
            if (dwCANInterpreter_getSignalName(&name, i, gIntp) == DW_SUCCESS)
            {
                if (dwCANInterpreter_getf32(&value, &timestamp, i, gIntp) == DW_SUCCESS)
                {
                    if (0 == strcmp(name, CAN_CAR_STEERING))
                    {
                        gAsilState->payload.steeringWheelAngle     = value * 16.f; // our sample data stores angles divided by 16
                        gAsilState->payload.frontSteeringTimestamp = msg->timestamp_us;

                        setSignalValid(gAsilState->payload.validityInfo.steeringWheelAngle);
                        setSignalValid(gAsilState->payload.validityInfo.frontSteeringTimestamp);
                    }
                    else if (0 == strcmp(name, CAN_CAR_SPEED))
                    {
                        gAsilState->payload.speedESC          = std::fabs(value);
                        gAsilState->payload.speedESCTimestamp = msg->timestamp_us;

                        setSignalValid(gAsilState->payload.validityInfo.speedESC);
                        setSignalValid(gAsilState->payload.validityInfo.speedESCTimestamp);
                    }
                }
            }
        }
    }

    return DW_SUCCESS;
}

//################################################################################################
dwStatus _dwVehicleIODriver_setDrivingMode(const dwVehicleIODrivingMode)
{
    return DW_NOT_IMPLEMENTED;
}

//################################################################################################
dwStatus _dwVehicleIODriver_sendASILCommand(const dwVehicleIOASILCommandE2EWrapper*, dwSensorHandle_t)
{
    return DW_NOT_IMPLEMENTED;
}

//################################################################################################
dwStatus _dwVehicleIODriver_sendQMCommand(const dwVehicleIOQMCommand*, dwSensorHandle_t)
{
    return DW_NOT_IMPLEMENTED;
}

//################################################################################################
dwStatus _dwVehicleIODriver_sendEgomotion(const dwValEgomotion*, dwSensorHandle_t)
{
    return DW_NOT_IMPLEMENTED;
}

//################################################################################################
dwStatus _dwVehicleIODriver_sendSensorCalibration(const dwValSensorCalibration*, dwSensorHandle_t)
{
    return DW_NOT_IMPLEMENTED;
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

////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
// NVIDIA MAKES NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY,
// OR OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY
// IMPLIED WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of
// such information or for any infringement of patents or other rights of third
// parties that may result from its use. No license is granted by implication or
// otherwise under any patent or patent rights of NVIDIA Corporation. No third
// party distribution is allowed unless expressly authorized by NVIDIA.  Details
// are subject to change without notice. This code supersedes and replaces all
// information previously supplied. NVIDIA Corporation products are not
// authorized for use as critical components in life support devices or systems
// without express written approval of NVIDIA Corporation.
//
// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software and related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution of
// this software and related documentation without an express license agreement
// from NVIDIA Corporation is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////////
#ifndef DW_CONTROL_VEHICLEIO_VEHICLEIOCAPABILITIES_H_
#define DW_CONTROL_VEHICLEIO_VEHICLEIOCAPABILITIES_H_
// Generated by dwProto from vehicle_io_capabilities.proto DO NOT EDIT BY HAND!
// See //3rdparty/shared/dwproto/README.md for more information

#include <dw/core/base/Types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES 50

/// @brief VehicleIO Capabilities. Note that this some capabilities are
///        imposed by the VehicleIO module itself. For dynamic (vehicle-reported)
///        capabilities, @see dwVehicleIOCapabilityState.
typedef struct dwVehicleIOCapabilities
{
    float32_t reverseSpeedLimit;                                                 //!< Normally a negative value (m/s)
    int32_t brakeValueLUTSize;                                                   //!< Size of the corresponding lookup table.
    float32_t brakeValueLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];                 //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t throttleValueLUTSize;                                                //!< Size of the corresponding lookup table.
    float32_t throttleValueLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];              //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t steeringSpeedLUTSize;                                                //!< Size of the corresponding lookup table.
    float32_t steeringSpeedLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];              //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t steeringWheelAngleLUTSize;                                           //!< Size of the corresponding lookup table.
    float32_t steeringWheelAngleLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];         //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t frontSteeringSpeedLUTSize;                                           //!< Size of the corresponding lookup table.
    float32_t frontSteeringSpeedLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];         //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t frontSteeringAngleLUTSize;                                           //!< Size of the corresponding lookup table.
    float32_t frontSteeringAngleLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];         //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t curvatureRateLUTSize;                                                //!< Size of the corresponding lookup table.
    float32_t curvatureRateLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];              //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t curvatureLUTSize;                                                    //!< Size of the corresponding lookup table.
    float32_t curvatureLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];                  //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t frontSteeringSpeedLowSpeedLUTSize;                                   //!< Size of the corresponding lookup table.
    float32_t frontSteeringSpeedLowSpeedLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES]; //!< Lookup Table indexed by speed/10.0 (m/s), i.e. LUT[i] is the capability value at speed = 10.0*i m/s
    int32_t frontSteeringAngleLowSpeedLUTSize;                                   //!< Size of the corresponding lookup table.
    float32_t frontSteeringAngleLowSpeedLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES]; //!< Lookup Table indexed by speed/10.0 (m/s), i.e. LUT[i] is the capability value at speed = 10.0*i m/s
    int32_t maxAccelerationLUTSize;                                              //!< Size of the corresponding lookup table.
    float32_t maxAccelerationLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];            //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t minAccelerationLUTSize;                                              //!< Size of the corresponding lookup table.
    float32_t minAccelerationLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];            //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t curvatureRateL2PlusLUTSize;                                          //!< Size of the corresponding lookup table.
    float32_t curvatureRateL2PlusLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];        //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    int32_t curvatureL2PlusLUTSize;                                              //!< Size of the corresponding lookup table.
    float32_t curvatureL2PlusLUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];            //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    float32_t curvatureRateC0LUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];            //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    uint32_t curvatureRateC0LUTSize;                                             //!< Stores the occupied/valid length of curvatureRateC0LUT
    float32_t curvatureC0LUT[DW_VEHICLEIO_SPEED_LUT_MAX_ENTRIES];                //!< Lookup Table indexed by speed (m/s), i.e. LUT[i] is the capability value at speed = i m/s
    uint32_t curvatureC0LUTSize;                                                 //!< Stores the occupied/valid length of curvatureC0LUT
} dwVehicleIOCapabilities;

#ifdef __cplusplus
}
#endif

#endif // DW_CONTROL_VEHICLEIO_VEHICLEIOCAPABILITIES_H_
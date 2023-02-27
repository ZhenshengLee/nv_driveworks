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
// SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SENSOR_COMMON_TYPES_HPP_
#define SENSOR_COMMON_TYPES_HPP_

#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dw/core/base/Types.h>
#include <dw/calibration/engine/common/CalibrationTypes.h>
#include <dw/control/vehicleio/VehicleIOLegacyStructures.h>
#include <dw/control/vehicleio/VehicleIOValStructures.h>
#include <dw/rig/Rig.h>

#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>

namespace dw
{
namespace framework
{

using dwSensorNodeProperties = struct dwSensorNodeProperties
{
    uint64_t sensorId;
    CalibratedExtrinsics extrinsics;
};

enum class CameraIntrinsicsSource
{
    RIG,
    EEPROM
};

using dwCameraIntrinsics = struct CameraIntrinsics
{
    uint64_t sensorId;
    dwFThetaCameraConfigNew intrinsics;
    CameraIntrinsicsSource intrinsicsSource;
};

} // namespace framework
} // namespace dw

const uint32_t MAX_STRING_LENGTH = 1024;

#endif // SENSOR_COMMON_TYPES_HPP_

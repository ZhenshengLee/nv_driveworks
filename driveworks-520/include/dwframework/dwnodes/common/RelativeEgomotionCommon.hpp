/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2023-2024 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
///////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_EGOMOTION_RELATIVEEGOMOTIONCOMMON_HPP_
#define DW_EGOMOTION_RELATIVEEGOMOTIONCOMMON_HPP_

#include <dwcgf/enum/EnumDescriptor.hpp>
#include <dw/calibration/engine/common/CalibrationTypesExtra.h>
#include <dw/egomotion/base/EgomotionExtra.h>
#include <dw/egomotion/2.0/Egomotion2.h>
#include <dw/core/logger/Logger.h>

namespace dw
{
namespace framework
{

template <>
struct EnumDescription<dwEgomotionGroundSpeedMeasurementTypes>
{
    static constexpr EnumDescriptionReturnType<dwEgomotionGroundSpeedMeasurementTypes, 6> get()
    {
        return describeEnumeratorCollection<dwEgomotionGroundSpeedMeasurementTypes>(
            DW_DESCRIBE_C_ENUMERATOR(DW_EGOMOTION_GROUND_SPEED_FROM_WHEEL_TICKS_AND_SPEEDS_REAR_AXLE),
            DW_DESCRIBE_C_ENUMERATOR(DW_EGOMOTION_GROUND_SPEED_FROM_WHEEL_TICKS_AND_SPEEDS_BOTH_AXLES),
            DW_DESCRIBE_C_ENUMERATOR(DW_EGOMOTION_GROUND_SPEED_FROM_WHEEL_SPEEDS_REAR_AXLE),
            DW_DESCRIBE_C_ENUMERATOR(DW_EGOMOTION_GROUND_SPEED_FROM_WHEEL_SPEEDS_BOTH_AXLES),
            DW_DESCRIBE_C_ENUMERATOR(DW_EGOMOTION_GROUND_SPEED_FROM_LINEAR_SPEED),
            DW_DESCRIBE_C_ENUMERATOR(DW_EGOMOTION_GROUND_SPEED_COUNT));
    }
};

template <>
struct EnumDescription<dwLoggerVerbosity>
{
    static constexpr EnumDescriptionReturnType<dwLoggerVerbosity, 6> get()
    {
        return describeEnumeratorCollection<dwLoggerVerbosity>(
            DW_DESCRIBE_C_ENUMERATOR(DW_LOG_VERBOSE),
            DW_DESCRIBE_C_ENUMERATOR(DW_LOG_DEBUG),
            DW_DESCRIBE_C_ENUMERATOR(DW_LOG_INFO),
            DW_DESCRIBE_C_ENUMERATOR(DW_LOG_WARN),
            DW_DESCRIBE_C_ENUMERATOR(DW_LOG_ERROR),
            DW_DESCRIBE_C_ENUMERATOR(DW_LOG_SILENT));
    }
};

dwEgomotionCalibrationStatus convertCalibrationToEgomotion(dwCalibrationStatus const& status);
dwCalibrationStatus convertEgomotionToCalibration(dwEgomotionCalibrationStatus const& status);
dwCalibrationManeuverArray convertEgomotionToCalibration(dwEgomotionCalibrationManeuverArray const& maneuvers);
dwCalibrationProperties convertEgomotionToCalibration(dwEgomotionCalibrationProperties const& properties);

} // namespace framework
} // namespace dw

#endif /*DW_EGOMOTION_RELATIVEEGOMOTIONCOMMON_HPP_*/

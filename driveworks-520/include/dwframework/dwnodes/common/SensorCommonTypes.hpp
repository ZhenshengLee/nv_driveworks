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

#ifndef DWFRAMEWORK_DWNODES_COMMON_SENSORCOMMONTYPES_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_SENSORCOMMONTYPES_HPP_

#include <dw/calibration/engine/common/CalibrationTypes.h>
#include <dw/core/base/Types.h>
#include <dw/rig/Rig.h>
#include <dw/sensors/codecs/Codec.h>
#include <dw/sensors/camera/CodecHeaderVideo.h>
#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>

namespace dw
{
namespace framework
{

struct dwSensorNodeProperties
{
    uint32_t sensorId{DW_MAX_RIG_SENSOR_COUNT};
    dwCalibratedExtrinsics extrinsics;
};

enum class CameraIntrinsicsSource
{
    RIG,
    EEPROM
};

struct dwCameraIntrinsics
{
    uint32_t sensorId{DW_MAX_RIG_SENSOR_COUNT};
    dwFThetaCameraConfig intrinsics;
    CameraIntrinsicsSource intrinsicsSource;
    bool intrinsicsVerificationFailed{false};
    dwWindshieldModelConfig windshield;
    bool hasWindshield;
};

struct dwCalibratedRadarIntrinsics
{
    uint32_t sensorId{DW_MAX_RIG_SENSOR_COUNT};
    dwRadarAzimuthCorrectionModelConfig azimuthCorrection;
    dwCalibrationStatus status;
    dwCalibrationProperties properties;
    dwCalibrationManeuverArray maneuvers;
    dwTime_t timestamp;
};

struct dwFeatureNccScores
{
    float32_t* d_nccScores;
    uint32_t size;
};

struct dwSensorNodeRawData
{
    uint8_t* data;
    size_t size;
};

struct dwLatency
{
    uint64_t senderTime;
    size_t size;
    uint8_t* data;
};

// might have other field added to it, keep it separate from
// dwSensorNodeRawData for now
struct SensorServiceNodeRawData
{
    uint8_t* data;
    size_t size;
    dwTime_t timestamp;
};

struct dwTraceNodeData
{
    size_t maxDataSize;
    size_t dataSize;
    uint8_t* data;
};

struct dwCodecMetadata
{
    dwCodecMimeType codecType;
    dwCodecConfigVideo configVideo;
};

} // namespace framework
} // namespace dw

const uint32_t MAX_STRING_LENGTH{1024U};

#endif // DWFRAMEWORK_DWNODES_COMMON_SENSORCOMMONTYPES_HPP_
